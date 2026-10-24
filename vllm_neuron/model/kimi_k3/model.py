# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 language model: hybrid KDA and MLA layers with attention residuals.

Each layer mixes the residual stream with a bank of block snapshots (AttnRes) twice,
before attention and before the MLP. The hot ops are NKI kernels behind their own
wrappers; this module holds the layer loop, the banks, the residual stream and the
call sites. A wrapper that is not in the package resolves to None at import and its
call site raises, so the module tree builds without it.

At decode each AttnRes kernel call also takes the producer's tail as its prologue:
it adds the previous attention or MLP output to the stream, so a decode layer hands
its MLP output (the tail) to the next layer's first mix instead of adding it itself.
"""

import importlib
import math

import torch
import torch.distributed as dist
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group, model_parallel_is_initialized

from vllm_neuron.functional.attention.attn_res_decode import attn_res_decode, attn_res_score_weight
from vllm_neuron.functional.attention.attn_res_prefill import attn_res_prefill
from vllm_neuron.functional.attention import o_proj as output_projection
from vllm_neuron.functional.attention.o_proj import o_proj, o_proj_tkg
from vllm_neuron.functional.embedding import embedding
from vllm_neuron.functional.kda.fused_decode import kda_fused_decode
from vllm_neuron.functional.kda.output import kda_output_prefill
from vllm_neuron.functional.moe.fused_tail import moe_fused_tail
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.cpl import ColumnParallelLinear
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

from . import weight_loaders_mxfp4
from .config import KDA_LAYER_TYPE, KimiK3Config, KimiK3TextConfig

# The checkpoint's modeling code builds both MLA latent norms at its RMSNorm
# default eps, not at the config's rms_norm_eps.
MLA_NORM_EPS = 1e-6


def _optional_kernel(module: str, name: str):
    """Return a wrapper from ``vllm_neuron.functional``, or None when it is absent."""
    path = f"vllm_neuron.functional.{module}"
    try:
        return getattr(importlib.import_module(path), name, None)
    except ModuleNotFoundError as e:
        if e.name != path:
            raise
        return None


qkv_proj_fp32acc = _optional_kernel("attention.qkv_fp32acc", "qkv_proj_fp32acc")
kda_prefill = _optional_kernel("kda.prefill", "kda_prefill")
depthwise_conv1d_silu = _optional_kernel("kda.depthwise_conv1d", "depthwise_conv1d_silu")
mla_paged_attention = _optional_kernel("attention.mla_paged", "mla_paged_attention")
routed_experts_mx = _optional_kernel("moe.routed_experts_mx", "routed_experts_mx")
router_select = _optional_kernel("moe.routed_experts_mx", "router_select")


def _require(kernel, op: str):
    """Return ``kernel``, or raise naming the op whose wrapper is not in the package."""
    if kernel is None:
        raise NotImplementedError(f"Kimi K3 {op}: its NKI kernel wrapper is not in this build")
    return kernel


def _tp_group():
    """Return vLLM's tensor-parallel group, or None outside a distributed run."""
    return get_tp_group() if model_parallel_is_initialized() else None


def _frozen(shape: tuple[int, ...], dtype: torch.dtype) -> nn.Parameter:
    """An uninitialized parameter; the loader assigns the stored tensor."""
    return nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


class _Weight(nn.Module):
    """A checkpoint module that holds one ``weight``."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        super().__init__()
        self.weight = _frozen(shape, dtype)


class KimiK3RMSNorm(_Weight):
    """RMSNorm as the checkpoint computes it: normalize in fp32, cast back, then scale."""

    def __init__(self, size: int, eps: float, dtype: torch.dtype):
        super().__init__((size,), dtype)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``weight * (x / rms(x))``, the product in the dtype of ``x``."""
        xf = x.float()
        normed = xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + self.eps)
        return self.weight * normed.to(x.dtype)


def situ_and_mul(
    gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float | None
) -> torch.Tensor:
    """SiTU of ``gate`` times ``up`` (squashed by ``linear_beta``), in fp32, cast back."""
    g, u = gate.float(), up.float()
    act = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return (act * u).to(gate.dtype)


def expert_bank_shapes(
    text_config: KimiK3TextConfig, num_programs: int
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Per-expert shape and dtype of each repacked MXFP4 bank part."""
    d, i, g = text_config.routed_expert_hidden_size, text_config.moe_intermediate_size, num_programs
    ni, nk = i // (512 * g), d // 512
    return {
        "gate_up_weight": ((g, 2 * ni, 128, nk * 512), torch.uint16),
        "gate_up_scale": ((g, 16, 2 * ni * nk * 512), torch.uint8),
        "down_weight": ((g, nk, 128, ni * 512), torch.uint16),
        "down_scale": ((g, 16, nk * ni * 512), torch.uint8),
    }


def _stack(*modules) -> nn.Parameter:
    """The modules' ``[out, in]`` weights as one ``[in, sum of out]`` weight, which replaces them.

    The fused projection reads this layout, so each original is freed once it is stacked.
    """
    stacked = torch.cat([m.weight for m in modules]).t().contiguous()
    for m in modules:
        m.weight = None
    return nn.Parameter(stacked, requires_grad=False)


def _project(x, weight, norm_weight, eps):
    """RMSNorm of ``[T, H]`` rows with gain ``norm_weight``, times a stacked ``[H, I]`` weight."""
    return _require(qkv_proj_fp32acc, "fused norm and projection")(
        x.unsqueeze(0), weight, norm_weight.to(torch.bfloat16), eps
    )[0]


def _prefill_projection(x, weight):
    """``[T, K]`` rows times ``weight`` ``[K, N]`` on the library's prefill output projection.

    The wrapper falls back to torch outside its kernel's limits, so this raises there instead.
    """
    tokens, width = x.shape
    d = max(k for k in range(1, output_projection.P_MAX + 1) if width % k == 0)
    limits = {
        "a device or the NKI simulator": can_run_kernel(x),
        f"K {width} equal to the weight's rows {weight.shape[0]}": weight.shape[0] == width,
        f"{width // d} column groups of {d}, at most {output_projection.MAX_VALIDATED_N_SIZE}":
            width // d <= output_projection.MAX_VALIDATED_N_SIZE,
        f"{tokens} tokens, at most {output_projection.MAX_VALIDATED_B_TIMES_S_SIZE}":
            tokens <= output_projection.MAX_VALIDATED_B_TIMES_S_SIZE,
        f"N {weight.shape[1]} even and at most {output_projection.MAX_VALIDATED_H_SIZE}":
            weight.shape[1] % 2 == 0 and weight.shape[1] <= output_projection.MAX_VALIDATED_H_SIZE,
    }
    failed = [name for name, ok in limits.items() if not ok]
    if failed:
        raise ValueError(
            "Kimi K3 prefill projection outside its kernel's limits: " + "; ".join(failed)
        )
    return o_proj(x.unsqueeze(0), weight)[0]


def _output_projection(x, weight):
    """Decode rows of ``x`` (``[D, T, N, 1]``) times ``weight`` (``[N*D, H]``) through the TKG kernel."""
    return o_proj_tkg(x, weight)


def _decode_layout(rows: torch.Tensor) -> torch.Tensor:
    """Copy token-major ``[T, N*D]`` rows into the decode projection's ``[D, T, N, 1]`` input.

    D is the largest divisor of the width up to 128 partitions. Element ``(d, t, n)``
    is column ``n * D + d`` of token t, so the move puts the columns on partitions.
    """
    tokens, width = rows.shape
    d = max(k for k in range(1, 129) if width % k == 0)
    return rows.reshape(tokens, width // d, d).permute(2, 0, 1).unsqueeze(-1).contiguous()


def _state_slots(meta) -> torch.Tensor:
    """Each sequence's state slot: column 0 of its KDA group's block table.

    With prefix caching off a sequence holds one block per state group, and a padding
    row holds the null block 0, which the bank has as a scratch slot.
    """
    return meta["block_table_tensor"][:, 0].long()


class KimiK3KDA(nn.Module):
    """Kimi Delta Attention: projections, short conv, gated delta rule, gated output."""

    def __init__(self, text_config: KimiK3TextConfig):
        super().__init__()
        cfg = text_config.linear_attn_config
        if not cfg.get("use_full_rank_gate", False):
            raise ValueError("Kimi K3 KDA layers take a full-rank output gate")
        hidden, dtype = text_config.hidden_size, text_config.torch_dtype
        self.num_heads, self.head_dim = int(cfg["num_heads"]), int(cfg["head_dim"])
        width = self.num_heads * self.head_dim
        taps = int(cfg["short_conv_kernel_size"])
        self.eps = text_config.rms_norm_eps
        self.lower_bound = cfg.get("gate_lower_bound")
        for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
            setattr(self, name, _Weight((width, hidden), dtype))
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            setattr(self, name, _Weight((width, 1, taps), torch.float32))
        self.f_a_proj = _Weight((self.head_dim, hidden), dtype)
        self.f_b_proj = _Weight((width, self.head_dim), dtype)
        self.b_proj = _Weight((self.num_heads, hidden), dtype)
        self.A_log = _frozen((self.num_heads,), torch.float32)
        self.dt_bias = _frozen((width,), torch.float32)
        self.o_norm = _Weight((self.head_dim,), torch.float32)
        self.o_proj = _Weight((hidden, width), dtype)
        self.register_parameter("input_weight", None)
        self.register_buffer("conv_weight", None, persistent=False)
        self.register_buffer("forget_weight", None, persistent=False)
        self.register_buffer("output_weight", None, persistent=False)
        # Per-sequence conv [slots, taps - 1, 3W] and recurrent [slots, heads, D, D]
        # state banks, bound with the KV caches.
        self.conv_state = None
        self.recurrent_state = None

    def prepare_after_load(self) -> None:
        """Lay out the weights the kernels read, once.

        The input projections stack as ``[H, I]`` in q, k, v, g, f_a, b order; the conv
        taps as fp32 ``[3W, taps]``; f_b as ``[D, W]``; and ``o_proj`` as ``[W, H]``.
        """
        if self.input_weight is None:
            self.input_weight = _stack(
                self.q_proj, self.k_proj, self.v_proj, self.g_proj, self.f_a_proj, self.b_proj
            )
        taps = [m.weight.squeeze(1) for m in (self.q_conv1d, self.k_conv1d, self.v_conv1d)]
        self.conv_weight = torch.cat(taps).float()
        self.forget_weight = self.f_b_proj.weight.t().contiguous()
        self.output_weight = self.o_proj.weight.t().contiguous()

    def _split_input(self, proj: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split the fused projection into q, k, v, g, f_a and the beta logits."""
        width = self.num_heads * self.head_dim
        return proj.split([width] * 4 + [self.head_dim, self.num_heads], dim=-1)

    def forward_prefill(self, mix, norm_weight, meta) -> torch.Tensor:
        """KDA over one sequence's prefill chunk; returns this rank's ``[T, H]`` fp32 partial.

        The chunk starts from the conv rows and recurrent state in the sequence's slot
        once an earlier chunk has run, and from zeros before. Its last conv rows and its
        final state go back to the slot for the next chunk or the first decode step.
        """
        q, k, v, gate, f_a, beta = self._split_input(
            _project(mix, self.input_weight, norm_weight, self.eps)
        )
        forget = _prefill_projection(f_a, self.forget_weight)
        slot = _state_slots(meta)[:1]
        # A fresh sequence's slot can hold an earlier owner's bytes, NaN included, so the
        # seed selects zeros instead of scaling by 0.
        carried = meta["cached_seq_len"].reshape(1, 1) > 0
        bank_rows = self.conv_state.index_select(0, slot)[0]
        history = torch.where(carried, bank_rows, torch.zeros_like(bank_rows))
        bank_state = self.recurrent_state.index_select(0, slot)[0]
        seed = torch.where(carried.unsqueeze(-1), bank_state, torch.zeros_like(bank_state))
        # Padding tokens follow the real ones, and their slot is the null block's slot 0.
        valid = meta["slot_mapping"] > 0
        # [3W, T] bf16, channel-major: the layout the recurrence reads.
        conv = _require(depthwise_conv1d_silu, "KDA short conv")(
            history, q, k, v, self.conv_weight
        )
        heads, dim, tokens = self.num_heads, self.head_dim, conv.shape[1]
        # The recurrence takes head-major gates: g [heads, T, D], beta [heads, T].
        gates = forget.float().view(tokens, heads, dim).transpose(0, 1).contiguous()
        core, state = _require(kda_prefill, "KDA recurrence")(
            *conv.split(heads * dim), gates, beta.t(), self.A_log,
            self.dt_bias.view(heads, dim), state=seed, valid=valid,
            lower_bound=self.lower_bound,
        )
        core = core.transpose(0, 1).reshape(tokens, heads * dim)
        # The next chunk's conv rows: the last taps - 1 rows up to the last real token.
        rows = torch.cat([history, torch.cat([q, k, v], dim=-1)])
        last = valid.sum() + torch.arange(history.shape[0], device=rows.device)
        self.conv_state.index_copy_(0, slot, rows.index_select(0, last).unsqueeze(0))
        self.recurrent_state.index_copy_(0, slot, state.unsqueeze(0))
        return self.output_prefill(core, gate.float())

    def output_prefill(self, core: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Gated output norm and ``o_proj`` of fp32 ``[T, W]`` core and gate: ``[T, H]`` fp32."""
        return kda_output_prefill(core, gate, self.o_norm.weight, self.o_proj.weight, self.eps)

    def forward_decode(self, mix, k_slice, norm_weight, meta) -> torch.Tensor:
        """One KDA decode step with state rows by slot; returns the ``[T, H]`` bf16 partial."""
        proj = _project(mix, self.input_weight, norm_weight, self.eps)
        # The kernel takes contiguous banks, and the bound banks are page-strided
        # views. So the step runs on the tokens' rows, which then go back by slot.
        slots = _state_slots(meta)
        rows = torch.arange(slots.shape[0], dtype=torch.int32, device=slots.device)
        # core: [128, T, heads, 1] bf16, the output projection's input layout.
        core, conv, state = kda_fused_decode(
            proj,
            rows,
            self.conv_state.index_select(0, slots),
            self.recurrent_state.index_select(0, slots),
            self.q_conv1d.weight,
            self.k_conv1d.weight,
            self.v_conv1d.weight,
            self.A_log,
            self.dt_bias,
            self.f_b_proj.weight,
            self.o_norm.weight,
            lower_bound=self.lower_bound,
            norm_eps=self.eps,
        )
        self.conv_state.index_copy_(0, slots, conv)
        self.recurrent_state.index_copy_(0, slots, state)
        return _output_projection(core, self.output_weight)


class KimiK3MLA(nn.Module):
    """Multi-head latent attention without rotary (NoPE), with a sigmoid output gate."""

    def __init__(self, text_config: KimiK3TextConfig):
        super().__init__()
        c = text_config
        if not (c.mla_use_nope and c.mla_use_output_gate):
            raise ValueError("Kimi K3 MLA layers are NoPE with an output gate")
        hidden, dtype, heads = c.hidden_size, c.torch_dtype, c.num_attention_heads
        q_head = c.qk_nope_head_dim + c.qk_rope_head_dim
        self.eps = c.rms_norm_eps
        self.q_a_proj = _Weight((c.q_lora_rank, hidden), dtype)
        self.q_a_layernorm = KimiK3RMSNorm(c.q_lora_rank, MLA_NORM_EPS, dtype)
        self.q_b_proj = _Weight((heads * q_head, c.q_lora_rank), dtype)
        self.kv_a_proj_with_mqa = _Weight((c.kv_lora_rank + c.qk_rope_head_dim, hidden), dtype)
        self.kv_a_layernorm = KimiK3RMSNorm(c.kv_lora_rank, MLA_NORM_EPS, dtype)
        kv_b_rows = heads * (c.qk_nope_head_dim + c.v_head_dim)
        self.kv_b_proj = _Weight((kv_b_rows, c.kv_lora_rank), dtype)
        self.o_proj = _Weight((hidden, heads * c.v_head_dim), dtype)
        self.g_proj = _Weight((heads * c.v_head_dim, hidden), dtype)
        self.register_parameter("input_weight", None)
        self.register_parameter("gate_weight", None)
        self.register_buffer("latent_weight", None, persistent=False)
        self.register_buffer("output_weight", None, persistent=False)
        # Paged latent cache [blocks, block_size, kv_lora_rank + rope], bound with the KV caches.
        self.kv_cache = None

    def prepare_after_load(self) -> None:
        """Lay out the weights the kernels read, once.

        q_a and kv_a go to fp32 ``[H, Q + L]`` for decode, then stack as bf16 ``[H, Q + L]``;
        g stacks as ``[H, HV]``; ``o_proj`` as ``[HV, H]``. g is projected alone at decode,
        so it is its own stack rather than a slice of q_a's.
        """
        if self.input_weight is None:
            latent = torch.cat([self.q_a_proj.weight, self.kv_a_proj_with_mqa.weight])
            self.latent_weight = latent.float().t().contiguous()
            self.input_weight = _stack(self.q_a_proj, self.kv_a_proj_with_mqa)
            self.gate_weight = _stack(self.g_proj)
        self.output_weight = self.o_proj.weight.t().contiguous()

    def _attend(self, latent, gate, meta):
        """Latent norms, q_b, absorb, paged attention and gate; stores the new latent rows."""
        attn, rows = _require(mla_paged_attention, "MLA attention")(
            latent,
            gate,
            self.q_a_layernorm.weight,
            self.kv_a_layernorm.weight,
            self.q_b_proj.weight,
            self.kv_b_proj.weight,
            cache=self.kv_cache,
            meta=meta,
            eps=MLA_NORM_EPS,
        )
        cache = self.kv_cache.view(-1, rows.shape[-1])
        cache.index_copy_(0, meta["slot_mapping"], rows.to(cache.dtype))
        return attn

    def forward_prefill(self, mix, norm_weight, meta) -> torch.Tensor:
        """MLA over a prefill chunk; returns this rank's ``[T, H]`` partial."""
        latent = _project(mix, self.input_weight, norm_weight, self.eps)
        gate = _project(mix, self.gate_weight, norm_weight, self.eps)
        return _prefill_projection(self._attend(latent, gate, meta), self.output_weight)

    def forward_decode(self, mix, k_slice, norm_weight, meta) -> torch.Tensor:
        """One MLA decode step; ``k_slice`` is the normed fp32 K slice of the mix."""
        latent = _output_projection(k_slice, self.latent_weight)
        gate = _project(mix, self.gate_weight, norm_weight, self.eps)
        return _output_projection(self._attend(latent, gate, meta), self.output_weight)


class KimiK3MLP(nn.Module):
    """Gated SiTU MLP: the dense layer and the shared experts."""

    def __init__(self, text_config: KimiK3TextConfig, intermediate_size: int):
        super().__init__()
        hidden, dtype = text_config.hidden_size, text_config.torch_dtype
        self.eps = text_config.rms_norm_eps
        self.beta = text_config.activation_situ_beta or 1.0
        self.linear_beta = text_config.activation_situ_linear_beta
        self.gate_proj = _Weight((intermediate_size, hidden), dtype)
        self.up_proj = _Weight((intermediate_size, hidden), dtype)
        self.down_proj = _Weight((hidden, intermediate_size), dtype)
        self.register_parameter("gate_up_weight", None)
        self.register_buffer("down_weight", None, persistent=False)

    def prepare_after_load(self) -> None:
        """Stack gate and up as ``[H, 2I]``, and lay out ``down_proj`` as ``[I, H]``."""
        if self.gate_up_weight is None:
            self.gate_up_weight = _stack(self.gate_proj, self.up_proj)
        self.down_weight = self.down_proj.weight.t().contiguous()

    def activation(self, mix, norm_weight) -> torch.Tensor:
        """RMSNorm of ``mix``, the gate and up projections, then SiTU-and-mul."""
        gate, up = _project(mix, self.gate_up_weight, norm_weight, self.eps).chunk(2, -1)
        return situ_and_mul(gate, up, self.beta, self.linear_beta)

    def forward_prefill(self, mix, norm_weight) -> torch.Tensor:
        """The MLP over a prefill chunk; returns this rank's ``[T, H]`` partial."""
        return _prefill_projection(self.activation(mix, norm_weight), self.down_weight)

    def forward_decode(self, mix, k_slice, norm_weight) -> torch.Tensor:
        """The MLP at decode; returns the ``[T, H]`` bf16 partial."""
        return _output_projection(_decode_layout(self.activation(mix, norm_weight)), self.down_weight)


class KimiK3Router(nn.Module):
    """Router weights: the expert score projection and the selection bias."""

    def __init__(self, text_config: KimiK3TextConfig):
        super().__init__()
        experts = text_config.num_experts
        self.weight = _frozen((experts, text_config.hidden_size), text_config.torch_dtype)
        self.e_score_correction_bias = _frozen((experts,), torch.float32)


class KimiK3Experts(nn.Module):
    """The routed experts' repacked MXFP4 banks, one leading entry per expert."""

    def __init__(self, text_config: KimiK3TextConfig, num_experts: int, num_programs: int):
        super().__init__()
        for part, (shape, dtype) in expert_bank_shapes(text_config, num_programs).items():
            self.register_parameter(part, _frozen((num_experts, *shape), dtype))

    def banks(self) -> list[torch.Tensor]:
        """The four bank parts in the loader's order."""
        return [getattr(self, part) for part in weight_loaders_mxfp4.EXPERT_PARTS]


class KimiK3MoE(nn.Module):
    """Latent MoE: down to the latent width, MXFP4 experts, normed latent up, shared experts."""

    def __init__(self, text_config: KimiK3TextConfig, num_programs: int):
        super().__init__()
        c = text_config
        if not c.latent_moe_use_norm or c.moe_router_activation_func != "sigmoid":
            raise ValueError("Kimi K3 MoE layers use a latent norm and a sigmoid router")
        hidden, latent, dtype = c.hidden_size, c.routed_expert_hidden_size, c.torch_dtype
        self.eps = c.rms_norm_eps
        self.top_k = c.num_experts_per_token
        self.renormalize = c.moe_renormalize
        self.scaling = c.routed_scaling_factor
        self.gate = KimiK3Router(c)
        self.experts = KimiK3Experts(c, c.num_experts, num_programs)
        self.shared_experts = KimiK3MLP(c, c.moe_intermediate_size * c.num_shared_experts)
        self.routed_expert_down_proj = _Weight((latent, hidden), dtype)
        self.routed_expert_up_proj = _Weight((hidden, latent), dtype)
        self.routed_expert_norm = KimiK3RMSNorm(latent, c.rms_norm_eps, dtype)
        self.register_parameter("input_weight", None)
        self.register_buffer("down_router_weight", None, persistent=False)
        self.register_buffer("latent_norm_weight", None, persistent=False)
        self.register_buffer("latent_up_weight", None, persistent=False)
        self.register_buffer("shared_down_weight", None, persistent=False)

    def prepare_after_load(self) -> None:
        """Lay out the weights the kernels read, once.

        Latent down and router go to fp32 ``[H, K + E]`` for decode, then stack as bf16
        ``[H, K + E]`` for the fused projection. The tail takes the latent norm as fp32 ``[D, 1]`` and the latent up and shared
        down weights as contiguous ``[K, H]``; the prefill projections read the same copies.
        """
        if self.input_weight is None:
            down_router = torch.cat([self.routed_expert_down_proj.weight, self.gate.weight])
            self.down_router_weight = down_router.float().t().contiguous()
            self.input_weight = _stack(self.routed_expert_down_proj, self.gate)
        self.shared_experts.prepare_after_load()
        self.latent_norm_weight = self.routed_expert_norm.weight.float().reshape(-1, 1)
        self.latent_up_weight = self.routed_expert_up_proj.weight.t().contiguous()
        self.shared_down_weight = self.shared_experts.down_weight

    def _select_and_run(self, latent, logits, ids=None, weights=None) -> torch.Tensor:
        """Routed experts on the latent rows: fp32 ``[T, D]`` combined output."""
        return _require(routed_experts_mx, "routed experts")(
            latent,
            logits,
            self.gate.e_score_correction_bias,
            *self.experts.banks(),
            ids=ids,
            weights=weights,
            top_k=self.top_k,
            renormalize=self.renormalize,
            scaling=self.scaling,
        )

    def forward_prefill(self, mix, norm_weight) -> torch.Tensor:
        """The MoE block over a prefill chunk; returns this rank's ``[T, H]`` partial."""
        latent_size = self.latent_up_weight.shape[0]
        proj = _project(mix, self.input_weight, norm_weight, self.eps)
        latent, logits = proj.split([latent_size, proj.shape[-1] - latent_size], -1)
        ids, weights = _require(router_select, "router selection")(
            logits.float(), self.gate.e_score_correction_bias, top_k=self.top_k,
            renormalize=self.renormalize, scaling=self.scaling,
        )
        combined = self._select_and_run(latent, logits, ids, weights).to(mix.dtype)
        up = _prefill_projection(self.routed_expert_norm(combined), self.latent_up_weight)
        return up + self.shared_experts.forward_prefill(mix, norm_weight)

    def forward_decode(self, mix, k_slice, norm_weight) -> torch.Tensor:
        """The MoE block at decode; returns the fp32 tail pack ``[T, 2H + 1]``."""
        latent_size = self.latent_up_weight.shape[0]
        partial = _output_projection(k_slice, self.down_router_weight)
        latent, logits = partial.split([latent_size, partial.shape[-1] - latent_size], -1)
        combined = self._select_and_run(latent, logits)
        gate_up = _project(mix, self.shared_experts.gate_up_weight, norm_weight, self.eps)
        return self.tail_decode(combined, gate_up)

    def tail_decode(self, combined, gate_up) -> torch.Tensor:
        """The fused tail of fp32 ``[T, K]`` combined rows: the fp32 pack ``[T, 2H + 1]``."""
        shared = self.shared_experts
        return moe_fused_tail(
            combined,
            gate_up,
            self.latent_norm_weight,
            self.latent_up_weight,
            self.shared_down_weight,
            beta=shared.beta,
            linear_beta=shared.linear_beta,
        )


def _tail_args(tail: torch.Tensor | None, stream: torch.Tensor, latent_size: int) -> dict:
    """The AttnRes decode arguments that fold a producer's tail into the stream.

    An MoE block's tail is the fp32 pack ``[T, 2H + 1]``, whose sum of squares covers
    the ``latent_size`` latent columns; any other tail is a bf16 ``[T, H]`` partial.
    """
    if tail is None:
        return {}
    if tail.shape[-1] == 2 * stream.shape[-1] + 1:
        return {"moe_tail": tail, "latent_size": latent_size}
    return {"residual": tail}


def _is_moe_layer(text_config: KimiK3TextConfig, layer_idx: int) -> bool:
    """True for the layers whose MLP is the MoE block."""
    c = text_config
    return (
        c.num_experts is not None
        and layer_idx >= c.first_k_dense_replace
        and layer_idx % c.moe_layer_freq == 0
    )


class KimiK3DecoderLayer(nn.Module):
    """One decoder layer: AttnRes mix, attention, AttnRes mix, then the MLP or MoE block."""

    def __init__(self, text_config: KimiK3TextConfig, layer_idx: int, num_programs: int):
        super().__init__()
        c = text_config
        hidden, dtype, eps = c.hidden_size, c.torch_dtype, c.rms_norm_eps
        self.layer_idx = layer_idx
        self.block_size = c.attn_res_block_size
        self.latent_size = c.routed_expert_hidden_size
        self.eps = eps
        self.is_kda = c.layer_types[layer_idx] == KDA_LAYER_TYPE
        self.self_attn = KimiK3KDA(c) if self.is_kda else KimiK3MLA(c)
        if _is_moe_layer(c, layer_idx):
            self.block_sparse_moe = KimiK3MoE(c, num_programs)
        else:
            self.mlp = KimiK3MLP(c, c.intermediate_size)
        self.input_layernorm = KimiK3RMSNorm(hidden, eps, dtype)
        self.post_attention_layernorm = KimiK3RMSNorm(hidden, eps, dtype)
        self.self_attention_res_norm = KimiK3RMSNorm(hidden, eps, dtype)
        self.mlp_res_norm = KimiK3RMSNorm(hidden, eps, dtype)
        self.self_attention_res_proj = _Weight((1, hidden), dtype)
        self.mlp_res_proj = _Weight((1, hidden), dtype)
        self.register_buffer("attention_score_w", None, persistent=False)
        self.register_buffer("mlp_score_w", None, persistent=False)

    @property
    def feed_forward(self) -> nn.Module:
        """The MoE block, or the dense MLP of the first layers."""
        return self.block_sparse_moe if hasattr(self, "block_sparse_moe") else self.mlp

    @property
    def block_start(self) -> bool:
        """True when this layer starts an AttnRes block and stores its input as a bank row."""
        return self.layer_idx % self.block_size == 0

    @property
    def attention_rows(self) -> int:
        """Bank rows the attention-site mix reads: the blocks started before this layer."""
        return math.ceil(self.layer_idx / self.block_size)

    @property
    def mlp_rows(self) -> int:
        """Bank rows the MLP-site mix reads: the blocks started up to this layer."""
        return self.layer_idx // self.block_size + 1

    def prepare_after_load(self) -> None:
        """Compute both sites' fp32 score weights and the sublayers' stacked weights."""
        self.attention_score_w = attn_res_score_weight(
            self.self_attention_res_norm.weight, self.self_attention_res_proj.weight
        )
        self.mlp_score_w = attn_res_score_weight(self.mlp_res_norm.weight, self.mlp_res_proj.weight)
        self.self_attn.prepare_after_load()
        self.feed_forward.prepare_after_load()

    def attention_mix(self, stream, bank) -> tuple[torch.Tensor, torch.Tensor | None]:
        """AttnRes mix at the attention site, prefill form.

        At a block start the stream is also stored as the block's bank row. Returns the
        mix and the residual base the attention output adds to: None at a block start,
        where the stream restarts from the attention output.
        """
        rows = self.attention_rows
        mix = stream
        if rows:
            mix = attn_res_prefill(bank[:, :rows], stream, self.attention_score_w, self.eps)
        if not self.block_start:
            return mix, stream
        bank[:, self.layer_idx // self.block_size] = stream
        return mix, None

    def mlp_mix(self, stream, bank) -> torch.Tensor:
        """AttnRes mix at the MLP site, prefill form."""
        return attn_res_prefill(bank[:, : self.mlp_rows], stream, self.mlp_score_w, self.eps)

    def forward_prefill(self, stream, bank, meta) -> torch.Tensor:
        """One layer over a prefill chunk; returns the new stream."""
        mix, base = self.attention_mix(stream, bank)
        attn = self.self_attn.forward_prefill(mix, self.input_layernorm.weight, meta)
        attn = attn.to(stream.dtype)
        stream = attn if base is None else base + attn
        mix = self.mlp_mix(stream, bank)
        out = self.feed_forward.forward_prefill(mix, self.post_attention_layernorm.weight)
        return stream + out.to(stream.dtype)

    def _mix_decode(self, stream, tail, bank, site: str):
        """AttnRes mix at decode with the producer's tail as its prologue.

        Returns the mix, the new stream, the bank and, at the MLA attention site and
        the MoE site, the rank's normed fp32 K slice of the mix.
        """
        attention = site == "attention"
        if attention and self.attention_rows == 0:
            mix, _ = self.attention_mix(stream, bank)
            return mix, stream, None, bank
        if not attention and self.block_start:
            # The block's stream restarts from the attention output.
            stream, tail = tail, None
        norm = self.input_layernorm if attention else self.post_attention_layernorm
        emit_k_slice = (attention and not self.is_kda) or (
            not attention and hasattr(self, "block_sparse_moe")
        )
        mix, stream, k_slice, new_bank = attn_res_decode(
            bank,
            stream,
            self.attention_score_w if attention else self.mlp_score_w,
            (self.attention_rows if attention else self.mlp_rows) + 1,
            **_tail_args(tail, stream, self.latent_size),
            bank_slot=self.layer_idx // self.block_size if attention and self.block_start else None,
            norm_weight=norm.weight if emit_k_slice else None,
            eps=self.eps,
        )
        return mix, stream, k_slice, bank if new_bank is None else new_bank

    def forward_decode(self, stream, tail, bank, meta):
        """One decode layer; returns the stream, this layer's tail for the next mix, and the bank."""
        mix, stream, k_slice, bank = self._mix_decode(stream, tail, bank, "attention")
        attn = self.self_attn.forward_decode(mix, k_slice, self.input_layernorm.weight, meta)
        mix, stream, k_slice, bank = self._mix_decode(stream, attn, bank, "mlp")
        out = self.feed_forward.forward_decode(mix, k_slice, self.post_attention_layernorm.weight)
        return stream, out, bank


class KimiK3Model(nn.Module):
    """Embedding, decoder layers, the output AttnRes mix and the final norm weight."""

    def __init__(self, text_config: KimiK3TextConfig, num_programs: int):
        super().__init__()
        c = text_config
        hidden, dtype, eps = c.hidden_size, c.torch_dtype, c.rms_norm_eps
        self.eps = eps
        self.embed_tokens = _Weight((c.vocab_size, hidden), dtype)
        self.layers = nn.ModuleList(
            [KimiK3DecoderLayer(c, i, num_programs) for i in range(c.num_hidden_layers)]
        )
        self.norm = KimiK3RMSNorm(hidden, eps, dtype)
        self.output_attn_res_norm = KimiK3RMSNorm(hidden, eps, dtype)
        self.output_attn_res_proj = _Weight((1, hidden), dtype)
        self.num_bank_rows = math.ceil(c.num_hidden_layers / c.attn_res_block_size)
        self.register_buffer("output_score_w", None, persistent=False)

    def prepare_after_load(self) -> None:
        """Compute every layer's load-time products and the output site's score weight."""
        for layer in self.layers:
            layer.prepare_after_load()
        self.output_score_w = attn_res_score_weight(
            self.output_attn_res_norm.weight, self.output_attn_res_proj.weight
        )

    def embed(self, input_ids) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed ``[T]`` token ids: the stream ``[T, H]`` and an unwritten bank ``[T, R, H]``."""
        stream = embedding(input_ids, self.embed_tokens.weight)
        bank = stream.new_empty((stream.shape[0], self.num_bank_rows, stream.shape[1]))
        return stream, bank

    def output_mix(self, stream, bank) -> torch.Tensor:
        """AttnRes mix of the last stream with every bank row, prefill form."""
        return attn_res_prefill(bank, stream, self.output_score_w, self.eps)

    def forward(self, input_ids, attn_metadata, is_prefill: bool) -> torch.Tensor:
        """Run every layer; returns the output mix ``[T, H]``, before the final norm."""
        stream, bank = self.embed(input_ids)
        metas = [attn_metadata.get(f"layers.{i}.self_attn") for i in range(len(self.layers))]
        if is_prefill:
            for layer, meta in zip(self.layers, metas):
                stream = layer.forward_prefill(stream, bank, meta)
            return self.output_mix(stream, bank)
        tail = None
        for layer, meta in zip(self.layers, metas):
            stream, tail, bank = layer.forward_decode(stream, tail, bank, meta)
        return self.output_mix_decode(stream, tail, bank)

    def output_mix_decode(self, stream, tail, bank) -> torch.Tensor:
        """AttnRes mix of the last stream, with the last layer's tail, and every bank row."""
        return attn_res_decode(
            bank, stream, self.output_score_w, self.num_bank_rows + 1,
            **_tail_args(tail, stream, self.layers[-1].latent_size), eps=self.eps,
        ).mix


class KimiK3ForConditionalGeneration(nn.Module):
    """Kimi K3 language model with its column-parallel LM head and on-device sampler."""

    def __init__(self, config: KimiK3Config):
        super().__init__()
        text = config.text_config
        if text.tie_word_embeddings:
            raise ValueError("Kimi K3 has an untied LM head")
        self.config = config
        self.text_config = text
        self.num_programs = get_logical_nc_config()
        self.model = KimiK3Model(text, self.num_programs)
        group = _tp_group()
        self.tp_size = group.world_size if group is not None else 1
        device_group = group.device_group if group is not None else None
        self.lm_head = ColumnParallelLinear(
            text.hidden_size,
            text.vocab_size,
            bias=False,
            dtype=text.torch_dtype,
            tp_group=device_group,
        )
        self.head_weights: nn.ParameterList | None = None
        sampling = text.neuron_config.on_device_sampling_config if text.neuron_config else None
        self.sampler = Sampler(sampling, process_group=device_group) if sampling else None

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None
    ) -> "KimiK3ForConditionalGeneration":
        """Build the model from the HF config and the NeuronConfig."""
        return cls(KimiK3Config.from_configs(hf_config, neuron_config))

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        """Load the checkpoint, then compute the load-time products the forward reads."""
        group = _tp_group()
        weight_loaders_mxfp4.load_weights(
            self,
            self.text_config,
            checkpoint_path,
            device,
            cache_dir,
            rank=group.rank_in_group if group is not None else 0,
            world_size=group.world_size if group is not None else 1,
            num_programs=self.num_programs,
            pipelined=dist.is_initialized(),
        )
        unloaded = [name for name, p in self.named_parameters() if p.is_meta]
        if unloaded:
            raise ValueError(
                f"{len(unloaded)} Kimi K3 parameters were not loaded, first {unloaded[0]}"
            )
        self.prepare_after_load()

    def prepare_after_load(self) -> None:
        """Compute the fp32 AttnRes score weights and the kernels' stacked weights, once.

        The LM head is stacked as two ``[H, V / 2TP]`` halves, one fused projection each.
        """
        self.model.prepare_after_load()
        if self.lm_head.weight is not None:
            halves = self.lm_head.weight.chunk(2, dim=0)
            self.head_weights = nn.ParameterList(
                [nn.Parameter(h.t().contiguous(), requires_grad=False) for h in halves]
            )
            self.lm_head.weight = None

    def get_kv_spec(self) -> KVSpec:
        """One cache spec per layer: an MLA latent cache, or a KDA layer's state per sequence."""
        text = self.text_config
        attn = text.linear_attn_config
        heads, head_dim = text.kda_heads_per_rank(self.tp_size), int(attn["head_dim"])
        # This rank's KDA state, as vLLM lays it out: the conv rows of q, k and v
        # [taps - 1, 3 * heads * head_dim], then the fp32 recurrent state.
        state_shapes = (
            (int(attn["short_conv_kernel_size"]) - 1, 3 * heads * head_dim),
            (heads, head_dim, head_dim),
        )
        state_dtypes = (text.torch_dtype, torch.float32)
        latent = text.kv_lora_rank + text.qk_rope_head_dim
        layers = []
        for i, layer in enumerate(self.model.layers):
            name = f"layers.{i}.self_attn"
            if layer.is_kda:
                spec = LayerSpec(
                    name, 0, 0, text.torch_dtype,
                    state_shapes=state_shapes, state_dtypes=state_dtypes,
                )
            else:
                spec = LayerSpec(name, 1, latent, text.torch_dtype, key_only=True)
            layers.append(spec)
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        """Bind each MLA layer's latent cache and each KDA layer's conv and recurrent state."""
        for i, layer in enumerate(self.model.layers):
            name = f"layers.{i}.self_attn"
            if name not in kv_caches:
                raise KeyError(f"KV cache for layer {name} not initialized")
            attn = layer.self_attn
            if layer.is_kda:
                attn.conv_state, attn.recurrent_state = kv_caches[name]
            else:
                # [blocks, 1, block_size, D]: one latent head.
                attn.kv_cache = kv_caches[name][0].squeeze(1)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        """Run the decoder; returns sampled tokens and no logits, or logits without a sampler."""
        if inputs_embeds is not None:
            raise NotImplementedError("Kimi K3 serves text prompts only")
        meta = next(iter(attn_metadata.values()))
        is_prefill = meta["max_query_len"] > meta["decode_token_threshold"]
        hidden = self.model(input_ids, attn_metadata, is_prefill)
        if sampling_positions is not None and (
            is_prefill or hidden.shape[0] != sampling_positions.shape[0]
        ):
            hidden = torch.index_select(hidden, 0, sampling_positions)
        logits = self.compute_logits(hidden)
        if self.sampler is None:
            return logits
        return self.sample(logits, sampling_params, logit_mask, rank), None

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Final norm and LM head on ``[R, H]`` rows: ``[R, V / TP]``, as two column halves."""
        norm = self.model.norm
        halves = [_project(hidden, w, norm.weight, norm.eps) for w in self.head_weights]
        return torch.cat(halves, dim=-1)

    def sample(self, logits, sampling_params=None, logit_mask=None, rank=None) -> torch.Tensor:
        """Sample one token per row with the on-device sampler."""
        if self.sampler is None:
            raise ValueError("Kimi K3 was built without an on-device sampling config")
        return self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
