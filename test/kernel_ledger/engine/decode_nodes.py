# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash decode node types (written new for the ledger).

The ported engine has matmul, MLP and collective nodes. GLM decode also needs:

  - MHCSinkhornNode / MHCCombineNode: mHC (hyper-connection, 4 residual streams)
    pre-side Sinkhorn on the 4x4 mix and post-side combine of the streams.
  - KDADecodeStepNode: one KDA linear-attention decode step (short conv over q,k,v,
    gate, recurrent fp32 state update), the unit ``kda_fused_decode`` runs.
  - DSAIndexerNode: DSA indexer scores + top-k, with the bypass regime: at
    ``ctx <= topk + kpool - 1`` (2051) selection keeps every token, so it is skipped.
  - SparseMLANode: absorbed MLA decode attention over the selected latent rows.
  - FusedRouterNode: RMSNorm + router GEMM + noaux_tc top-k in one kernel.
  - ExpertDecodeNode: routed experts of one EP rank; weight bytes follow the expected
    number of distinct local experts ``E_local x (1 - (1 - k/E)^T)``.
  - BlockQuantMLPNode: fp8 block-quant gated MLP whose per-rank intermediate is
    padded to the 128 quant block (shared expert, dense MLP).
  - VocabParallelLMHeadNode: lm_head sharded by vocabulary rows.
  - GreedySamplerNode: on-device greedy argmax over the logits row.
  - RMSNormNode: one-row RMSNorm.

Conventions follow ``node.py``: activations are ``(B, S=1, ...)`` per rank, B = tokens
in the step (one per request at decode). Every node computes FLOPs and HBM bytes per
call; ``compute_roofline`` (inherited) takes ``max(flops / PEAK, bytes / HBM, floor)``.
``shape_record()`` returns the node's shapes in the vocabulary of the matching
hardware microbenchmark JSON, so ``--emit-shapes`` output can be compared with
what the benchmarks recorded.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .node import DTYPE_BYTES, KernelName, KernelNode, OpType, Port, RooflineMode

#: Dtype names as the benchmark JSONs spell them.
TORCH_DTYPE = {"bf16": "bfloat16", "fp32": "float32", "fp8": "float8_e4m3fn", "int32": "int32"}


def expected_distinct_experts(num_experts: int, top_k: int, tokens: int) -> float:
    """Expected number of distinct experts hit by ``tokens`` tokens, uniform routing.

    ``E x (1 - (1 - k/E)^T)``: each expert is missed by one token with probability
    ``1 - k/E``. Uniform routing is assumed (``PLANNER_REPORT.md`` caveat 6).
    """
    if tokens <= 0:
        return 0.0
    return num_experts * (1.0 - (1.0 - top_k / num_experts) ** tokens)


class _DecodeNode(KernelNode):
    """KernelNode with explicit FLOPs/bytes and derived output ports.

    Subclasses implement ``_flops()``, ``_bytes()`` and ``_outputs()`` from the wired
    input ports. Roofline: MAX of compute and memory, utilisation 1.0 (the
    entitlement), floor ``min_latency_us`` (2 us, one dependent DMA).
    """

    def __init__(self, name: str, *, kernel_name: Optional[KernelName] = None,
                 layer_type: str = "", layer_count: int = 1, min_latency_us: float = 2.0):
        super().__init__(
            name,
            kernel_name=kernel_name,
            layer_type=layer_type,
            layer_count=layer_count,
            op_type=OpType.CUSTOM,
            roofline_mode=RooflineMode.MAX,
            roofline_mfu=1.0,
            roofline_mbu=1.0,
            min_latency_us=min_latency_us,
            flops=0,
        )

    # -- interface used by the engine -------------------------------------------------
    @property
    def output_ports(self) -> Dict[str, Port]:
        if not self._input_ports:
            return {}
        return self._outputs()

    def compute_flops(self) -> int:
        return self._flops() if self._input_ports else 0

    def _total_memory_bytes(self) -> int:
        return self._bytes() if self._input_ports else 0

    # -- helpers ----------------------------------------------------------------------
    def _port(self, name: str) -> Port:
        port = self._input_ports.get(name)
        if port is None:
            raise ValueError(f"Node '{self.name}': input port '{name}' is not wired")
        return port

    def _tokens(self, port_name: str) -> int:
        return self._port(port_name).shape[0]

    def shape_record(self, kernel_set: Optional[str] = None) -> Dict:
        raise NotImplementedError

    def _flops(self) -> int:
        raise NotImplementedError

    def _bytes(self) -> int:
        raise NotImplementedError

    def _outputs(self) -> Dict[str, Port]:
        raise NotImplementedError


# ============================================================================
# mHC
# ============================================================================


class MHCSinkhornNode(_DecodeNode):
    """mHC pre side: Sinkhorn-normalise the 4x4 residual mix, split pre/post gates.

    Input ``mix`` ``[B, 1, n*n + 2n]`` (the fn GEMV output). Outputs ``res``
    ``[B, 1, n, n]``, ``pre`` and ``post`` ``[B, 1, n]``. FLOPs per token: one exp over
    n^2, then ``iters`` x (row + column normalisation, 2 n^2 each).
    """

    def __init__(self, name: str, *, streams: int = 4, iters: int = 20, hidden: int = 4096, **kw):
        super().__init__(name, kernel_name=kw.pop("kernel_name", KernelName.MHC_SINKHORN_TKG), **kw)
        self.streams = streams
        self.iters = iters
        self.hidden = hidden

    def _outputs(self):
        mix = self._port("mix")
        n = self.streams
        if mix.shape[-1] != n * n + 2 * n:
            raise ValueError(
                f"Node '{self.name}': mix width {mix.shape[-1]} != n*n + 2n = {n * n + 2 * n}"
            )
        lead = mix.shape[:-1]
        return {
            "res": Port("res", lead + (n, n), "fp32"),
            "pre": Port("pre", lead + (n,), "fp32"),
            "post": Port("post", lead + (n,), "fp32"),
        }

    def _flops(self):
        n2 = self.streams * self.streams
        return self._tokens("mix") * (n2 + self.iters * 4 * n2)

    def _bytes(self):
        out = sum(p.size_bytes for p in self._outputs().values())
        return self._port("mix").size_bytes + out

    def shape_record(self, kernel_set=None):
        return {"B": self._tokens("mix"), "hidden": self.hidden, "streams": self.streams,
                "iters": self.iters}


class MHCCombineNode(_DecodeNode):
    """mHC post side (hyper_connection): ``streams' = res @ streams + post * y``.

    Inputs ``streams`` ``[B, 1, n, H]``, ``y`` ``[B, 1, H]`` (sublayer output after its
    all-reduce), ``res`` ``[B, 1, n, n]``, ``post`` ``[B, 1, n]``. FLOPs per token
    ``2 n^2 H + 2 n H``. Bytes: read the streams, y and the mix, write the streams.
    """

    def __init__(self, name: str, *, streams: int = 4, **kw):
        super().__init__(name, kernel_name=kw.pop("kernel_name", KernelName.MHC_COMBINE_TKG), **kw)
        self.streams = streams

    def _outputs(self):
        s = self._port("streams")
        return {"streams": Port("streams", s.shape, s.dtype)}

    def _flops(self):
        s = self._port("streams")
        n, h = self.streams, s.shape[-1]
        return s.shape[0] * (2 * n * n * h + 2 * n * h)

    def _bytes(self):
        read = sum(self._port(p).size_bytes for p in ("streams", "y", "res", "post"))
        return read + self._port("streams").size_bytes

    def shape_record(self, kernel_set=None):
        s = self._port("streams")
        return {"B": s.shape[0], "hidden": s.shape[-1], "streams": self.streams}


# ============================================================================
# KDA
# ============================================================================


class KDADecodeStepNode(_DecodeNode):
    """One KDA (Kimi delta attention) decode step for the rank's heads.

    Inputs ``q, k, v, g`` ``[B, 1, h, d]`` and ``beta`` ``[B, 1, h, 1]``; output ``core``
    ``[B, 1, h, d]``. Carriers (state, not ports): the short-conv carrier
    ``[B, taps-1, 3 h d]`` bf16 and the recurrent state ``[B, h, d, d]`` fp32, each read
    and written once per step.

    FLOPs per token: the depthwise conv ``2 x taps x 3hd`` plus per head ``7 d^2 + 8 d``
    (state decay d^2, ``S^T k`` 2d^2, rank-1 delta update 2d^2, ``S^T q`` 2d^2, gates 8d).
    """

    def __init__(self, name: str, *, heads_per_rank: int = 1, head_dim: int = 128,
                 conv_taps: int = 4, **kw):
        super().__init__(name, kernel_name=kw.pop("kernel_name", KernelName.KDA_DECODE_TKG), **kw)
        self.heads_per_rank = heads_per_rank
        self.head_dim = head_dim
        self.conv_taps = conv_taps

    @property
    def conv_channels(self) -> int:
        return 3 * self.heads_per_rank * self.head_dim

    def carriers(self, batch: int) -> Dict[str, Port]:
        h, d = self.heads_per_rank, self.head_dim
        return {
            "conv_carrier": Port("conv_carrier", (batch, self.conv_taps - 1, self.conv_channels), "bf16"),
            "recurrent_carrier": Port("recurrent_carrier", (batch, h, d, d), "fp32"),
        }

    def _outputs(self):
        q = self._port("q")
        return {"core": Port("core", q.shape, q.dtype)}

    def _flops(self):
        b, h, d = self._tokens("q"), self.heads_per_rank, self.head_dim
        return b * (2 * self.conv_taps * self.conv_channels + h * (7 * d * d + 8 * d))

    def _bytes(self):
        b = self._tokens("q")
        carriers = 2 * sum(p.size_bytes for p in self.carriers(b).values())
        io = sum(self._port(p).size_bytes for p in ("q", "k", "v", "g", "beta"))
        return carriers + io + self._outputs()["core"].size_bytes

    def shape_record(self, kernel_set=None):
        b = self._tokens("q")
        rec = {"B": b, "heads_per_rank": self.heads_per_rank, "head_dim": self.head_dim,
               "conv_taps": self.conv_taps, "conv_channels": self.conv_channels}
        for name, port in self.carriers(b).items():
            rec[name] = {"dtype": TORCH_DTYPE[port.dtype], "shape": list(port.shape)}
        return rec


# ============================================================================
# DSA / MLA
# ============================================================================


class DSAIndexerNode(_DecodeNode):
    """DSA indexer: score the pooled keys and pick the top ``topk`` tokens.

    Inputs ``q_idx`` ``[B, 1, n_heads, d]``, ``k_idx`` ``[B, 1, d]``, ``w_idx``
    ``[B, 1, n_heads]``; output ``indices`` ``[B, 1, selected_tokens]``.

    Regimes (``attention.md``): keys are pooled ``kpool`` tokens at a time, and top-k
    keeps ``topk / kpool`` pools plus the incomplete tail pool. While
    ``ctx <= topk + kpool - 1`` (2051) every real pool is kept, so selection is a no-op:
    ``bypass``, zero work, all ``ctx`` tokens attended. Above it: ``selected``, every one
    of ``ceil(ctx / kpool)`` pools is scored (``2 n_heads d`` per pool plus the head
    weighting) and ``topk`` tokens are attended.
    """

    def __init__(self, name: str, *, n_heads: int = 32, head_dim: int = 128, kpool: int = 4,
                 topk: int = 2048, ctx: int = 1024, **kw):
        kw.setdefault("kernel_name", None)
        super().__init__(name, **kw)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.kpool = kpool
        self.topk = topk
        self.ctx = ctx

    @property
    def bypass_max_ctx(self) -> int:
        return self.topk + self.kpool - 1

    @property
    def regime(self) -> str:
        return "bypass" if self.ctx <= self.bypass_max_ctx else "selected"

    @property
    def pools(self) -> int:
        return math.ceil(self.ctx / self.kpool)

    @property
    def selected_tokens(self) -> int:
        return self.ctx if self.regime == "bypass" else self.topk

    def _outputs(self):
        q = self._port("q_idx")
        return {"indices": Port("indices", (q.shape[0], 1, self.selected_tokens), "int32")}

    def _flops(self):
        if self.regime == "bypass":
            return 0
        b = self._tokens("q_idx")
        return b * (2 * self.n_heads * self.head_dim * self.pools + 2 * self.n_heads * self.pools)

    def _bytes(self):
        if self.regime == "bypass":
            return 0
        b = self._tokens("q_idx")
        keys = self.pools * self.head_dim * DTYPE_BYTES["bf16"]
        q = self.n_heads * self.head_dim * DTYPE_BYTES["bf16"]
        w = self.n_heads * DTYPE_BYTES["bf16"]
        return b * (keys + q + w)


class SparseMLANode(_DecodeNode):
    """Absorbed MLA decode attention over the selected latent rows (one rank's heads).

    Inputs ``q`` ``[B, 1, h, qk]``, ``latent`` ``[B, 1, kv_lora]`` (this token's
    compressed KV), ``indices`` ``[B, 1, T]``; output ``out`` ``[B, 1, h, v]``.
    Weights: ``W_UK`` ``(H_glob, qk, kv_lora)`` and ``W_UV`` ``(H_glob, kv_lora, v)``,
    bf16 (kv_b is not fp8), sharded one head group per rank.

    FLOPs per token and head: absorb q ``2 qk r``, scores ``2 r T``, P x latent ``2 r T``,
    un-absorb ``2 r v``. Bytes: the absorb weights, the ``T`` latent rows (bf16), q,
    the new latent row, and the output.
    """

    def __init__(self, name: str, *, heads_per_rank: int = 1, qk_head_dim: int = 256,
                 v_head_dim: int = 256, kv_lora_rank: int = 512, tp: int = 64, **kw):
        super().__init__(name, kernel_name=kw.pop("kernel_name", KernelName.DSA_MLA_LAYER_TKG), **kw)
        self.heads_per_rank = heads_per_rank
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.tp = tp
        self.weights = {
            "W_UK": (heads_per_rank * tp, qk_head_dim, kv_lora_rank),
            "W_UV": (heads_per_rank * tp, kv_lora_rank, v_head_dim),
        }
        self.weight_dtype = "bf16"
        #: Set by the graph builder: shape-record fields of the measured layer.
        self.layer_record: Dict = {}

    def local_weight_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {k: (self.heads_per_rank,) + v[1:] for k, v in self.weights.items()}

    def _outputs(self):
        q = self._port("q")
        return {"out": Port("out", q.shape[:-1] + (self.v_head_dim,), q.dtype)}

    def _attended(self) -> int:
        return self._port("indices").shape[-1]

    def _flops(self):
        b, h, t = self._tokens("q"), self.heads_per_rank, self._attended()
        qk, r, v = self.qk_head_dim, self.kv_lora_rank, self.v_head_dim
        return b * h * (2 * qk * r + 2 * r * t + 2 * r * t + 2 * r * v)

    def _bytes(self):
        b, t = self._tokens("q"), self._attended()
        weights = sum(math.prod(s) for s in self.local_weight_shapes().values()) * DTYPE_BYTES["bf16"]
        rows = b * t * self.kv_lora_rank * DTYPE_BYTES["bf16"]
        io = self._port("q").size_bytes + self._port("latent").size_bytes
        return weights + rows + io + self._outputs()["out"].size_bytes

    def shape_record(self, kernel_set=None):
        rec = dict(self.layer_record.get("common", {}))
        if kernel_set is not None:
            rec.update(self.layer_record.get(kernel_set, {}))
        return rec


# ============================================================================
# MoE
# ============================================================================


class FusedRouterNode(_DecodeNode):
    """RMSNorm + router GEMM + noaux_tc top-k, one kernel (router weight replicated).

    Input ``x`` ``[B, 1, H]`` (pre-norm). Outputs ``topk_ids`` (int32) and
    ``topk_weights`` (fp32) ``[B, 1, k]``. FLOPs per token: norm ``4 H``, GEMM
    ``2 H E``, sigmoid + bias + top-k + renorm ``10 E``. Bytes: router weight
    ``H x E`` bf16, the norm gain, x, the outputs.
    """

    def __init__(self, name: str, *, num_experts: int = 288, top_k: int = 8, **kw):
        super().__init__(name, kernel_name=kw.pop("kernel_name", KernelName.RMSNORM_ROUTER_TOPK_TKG), **kw)
        self.num_experts = num_experts
        self.top_k = top_k

    def _outputs(self):
        x = self._port("x")
        lead = x.shape[:-1]
        return {
            "topk_ids": Port("topk_ids", lead + (self.top_k,), "int32"),
            "topk_weights": Port("topk_weights", lead + (self.top_k,), "fp32"),
        }

    def _flops(self):
        x = self._port("x")
        h = x.shape[-1]
        return x.shape[0] * (4 * h + 2 * h * self.num_experts + 10 * self.num_experts)

    def _bytes(self):
        x = self._port("x")
        h = x.shape[-1]
        weight = (h * self.num_experts + h) * DTYPE_BYTES["bf16"]
        return weight + x.size_bytes + sum(p.size_bytes for p in self._outputs().values())

    def shape_record(self, kernel_set=None):
        x = self._port("x")
        return {"T": x.shape[0], "H": x.shape[-1], "E_global": self.num_experts, "top_k": self.top_k}


class ExpertDecodeNode(_DecodeNode):
    """Routed experts of one EP rank at decode.

    Experts are split over ``ep`` groups (``num_experts / ep`` local experts) and each
    expert's intermediate over ``tp_moe`` ranks of the group. Inputs ``x`` ``[B, 1, H]``
    (normed) and ``topk_ids`` ``[B, 1, k]``; output ``y`` ``[B, 1, H]`` (this rank's
    partial sum, reduced by the MLP all-reduce).

    Bytes: fp8 gate/up/down of the expected distinct local experts,
    ``E_local x (1 - (1 - k/E)^T)`` x ``3 H I_local``. FLOPs: ``2 x 3 H I_local`` per local
    (token, expert) pair, ``T k E_local / E`` pairs expected.
    """

    def __init__(self, name: str, *, num_experts: int = 288, top_k: int = 8,
                 intermediate_size: int = 2048, ep: int = 16, tp_moe: int = 4, **kw):
        super().__init__(name, kernel_name=kw.pop("kernel_name", KernelName.MOE_EXPERTS_TKG), **kw)
        if num_experts % ep or intermediate_size % tp_moe:
            raise ValueError(f"Node '{name}': experts/intermediate not divisible by ep/tp_moe")
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.ep = ep
        self.tp_moe = tp_moe
        self.weight_dtype = "fp8"

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.ep

    @property
    def local_intermediate(self) -> int:
        return self.intermediate_size // self.tp_moe

    @property
    def tokens(self) -> int:
        return self._tokens("x")

    @property
    def expected_distinct_local_experts(self) -> float:
        return expected_distinct_experts(self.num_experts, self.top_k, self.tokens) / self.ep

    @property
    def expected_local_pairs(self) -> float:
        return self.tokens * self.top_k * self.local_experts / self.num_experts

    def _per_expert_bytes(self) -> int:
        h = self._port("x").shape[-1]
        return 3 * h * self.local_intermediate * DTYPE_BYTES[self.weight_dtype]

    def _weight_bytes(self) -> float:
        return self.expected_distinct_local_experts * self._per_expert_bytes()

    def _outputs(self):
        x = self._port("x")
        return {"y": Port("y", x.shape, x.dtype)}

    def _flops(self):
        h = self._port("x").shape[-1]
        return 2 * 3 * h * self.local_intermediate * self.expected_local_pairs

    def _bytes(self):
        io = self._port("x").size_bytes + self._port("topk_ids").size_bytes
        return self._weight_bytes() + io + self._outputs()["y"].size_bytes

    def shape_record(self, kernel_set=None):
        x = self._port("x")
        return {
            "T": x.shape[0], "H": x.shape[-1], "E_global": self.num_experts, "top_k": self.top_k,
            "E_local": self.local_experts, "I_local": self.local_intermediate,
            "expected_distinct_local_experts": self.expected_distinct_local_experts,
        }


class BlockQuantMLPNode(_DecodeNode):
    """fp8 block-quant gated MLP (gate_up + SwiGLU + down), TP over the intermediate.

    The per-rank intermediate ``I / tp`` is padded up to the quant block (128), as
    deployed (``dense.md`` E2: "32 padded to 128", "192 padded to 256"). Weights per
    rank: ``gate_up (H, 2 I_local)``, ``down (I_local, H)``, fp8. Input/output
    ``[B, 1, H]``.
    """

    def __init__(self, name: str, *, hidden: int = 4096, intermediate_size: int = 2048,
                 tp: int = 64, block: int = 128, **kw):
        super().__init__(name, kernel_name=kw.pop("kernel_name", KernelName.SHARED_EXPERT_TKG), **kw)
        if intermediate_size % tp:
            raise ValueError(f"Node '{name}': intermediate {intermediate_size} not divisible by tp {tp}")
        self.hidden = hidden
        self.intermediate_size = intermediate_size
        self.tp = tp
        self.block = block
        self.weight_dtype = "fp8"
        self.weights = {"gate_up": (hidden, 2 * intermediate_size), "down": (intermediate_size, hidden)}

    @property
    def local_intermediate(self) -> int:
        return math.ceil(self.intermediate_size / self.tp / self.block) * self.block

    def local_weight_shapes(self) -> Dict[str, Tuple[int, ...]]:
        i = self.local_intermediate
        return {"gate_up": (self.hidden, 2 * i), "down": (i, self.hidden)}

    def _outputs(self):
        x = self._port("input")
        return {"output": Port("output", x.shape, x.dtype)}

    def _flops(self):
        x = self._port("input")
        return 2 * 3 * x.shape[0] * self.hidden * self.local_intermediate

    def _bytes(self):
        return sum(math.prod(s) for s in self.local_weight_shapes().values()) * DTYPE_BYTES[self.weight_dtype]

    def shape_record(self, kernel_set=None):
        return {"M": self._tokens("input"), "H": self.hidden, "I": self.local_intermediate}


# ============================================================================
# Tail
# ============================================================================


class VocabParallelLMHeadNode(_DecodeNode):
    """lm_head sharded by vocabulary rows: each rank computes ``vocab / tp`` logits.

    Weight per rank ``(H, vocab / tp)`` bf16; input ``[B, 1, H]``; output ``logits``
    ``[B, 1, vocab / tp]``, gathered by the following all-gather.
    """

    def __init__(self, name: str, *, hidden: int = 4096, vocab: int = 154880, tp: int = 64, **kw):
        super().__init__(name, kernel_name=kw.pop("kernel_name", KernelName.LM_HEAD), **kw)
        if vocab % tp:
            raise ValueError(f"Node '{name}': vocab {vocab} not divisible by tp {tp}")
        self.hidden = hidden
        self.vocab = vocab
        self.tp = tp
        self.weight_dtype = "bf16"
        self.weights = {"lm_head": (hidden, vocab)}

    @property
    def shard_rows(self) -> int:
        return self.vocab // self.tp

    def _outputs(self):
        x = self._port("input")
        return {"logits": Port("logits", x.shape[:-1] + (self.shard_rows,), x.dtype)}

    def _flops(self):
        return 2 * self._tokens("input") * self.hidden * self.shard_rows

    def _bytes(self):
        return self.hidden * self.shard_rows * DTYPE_BYTES[self.weight_dtype]

    def shape_record(self, kernel_set=None):
        return {"M": self._tokens("input"), "H": self.hidden, "vocab": self.vocab,
                "shard_rows": self.shard_rows}


class GreedySamplerNode(_DecodeNode):
    """On-device greedy sampler: argmax over the gathered logits row.

    Input ``logits`` ``[B, 1, V]``; output ``tokens`` ``[B, 1]`` int32. One compare per
    logit; bytes = the logits read + the token ids written.
    """

    def __init__(self, name: str, **kw):
        super().__init__(name, kernel_name=kw.pop("kernel_name", KernelName.SAMPLING), **kw)

    def _outputs(self):
        logits = self._port("logits")
        return {"tokens": Port("tokens", logits.shape[:-1], "int32")}

    def _flops(self):
        logits = self._port("logits")
        return logits.num_elements

    def _bytes(self):
        return self._port("logits").size_bytes + self._outputs()["tokens"].size_bytes

    def shape_record(self, kernel_set=None):
        logits = self._port("logits")
        return {"B": logits.shape[0], "vocab": logits.shape[-1],
                "logits_dtype": TORCH_DTYPE[logits.dtype], "mode": "all_greedy"}


class RMSNormNode(_DecodeNode):
    """One-row RMSNorm ``x / sqrt(mean(x^2) + eps) * gain``: 4 FLOPs and 3 bf16 rows."""

    def __init__(self, name: str, **kw):
        super().__init__(name, kernel_name=kw.pop("kernel_name", KernelName.RMSNORM_TKG), **kw)

    def _outputs(self):
        x = self._port("input")
        return {"output": Port("output", x.shape, x.dtype)}

    def _flops(self):
        return 4 * self._port("input").num_elements

    def _bytes(self):
        x = self._port("input")
        gain = x.shape[-1] * DTYPE_BYTES[x.dtype]
        return 2 * x.size_bytes + gain

    def shape_record(self, kernel_set=None):
        x = self._port("input")
        return {"M": x.shape[0], "H": x.shape[-1]}
