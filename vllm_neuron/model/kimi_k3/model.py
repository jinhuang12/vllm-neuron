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
from nkilib.core.utils.common_types import OProjAttentionLayout
from torch import nn
from torch.distributed import _functional_collectives as funcol
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import (
    get_tp_group,
    get_world_group,
    model_parallel_is_initialized,
)

from vllm_neuron.functional.attention.attn_res_decode import attn_res_decode, attn_res_score_weight
from vllm_neuron.functional.attention.attn_res_prefill import attn_res_prefill
from vllm_neuron.functional.attention.mla_paged import (
    mla_paged_decode,
    mla_paged_prefill,
    prepare_mla_weights,
)
from vllm_neuron.functional.attention import o_proj as output_projection
from vllm_neuron.functional.attention.o_proj import o_proj, o_proj_tkg
from vllm_neuron.functional.kda.fused_decode import kda_fused_decode
from vllm_neuron.functional.kda.output import kda_output_prefill
from vllm_neuron.functional.moe.fused_tail import moe_fused_tail
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.cpl import ColumnParallelLinear
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.parallel.neuron_parallel_state import get_wide_ep_group
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

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
routed_experts_mx = _optional_kernel("moe.routed_experts_mx", "routed_experts_mx")
router_select = _optional_kernel("moe.routed_experts_mx", "router_select")
routed_expert_tiles_mx = _optional_kernel("moe.routed_experts_mx", "routed_expert_tiles_mx")


def _require(kernel, op: str):
    """Return ``kernel``, or raise naming the op whose wrapper is not in the package."""
    if kernel is None:
        raise NotImplementedError(f"Kimi K3 {op}: its NKI kernel wrapper is not in this build")
    return kernel


def _tp_group():
    """Return vLLM's tensor-parallel group, or None outside a distributed run."""
    return get_tp_group() if model_parallel_is_initialized() else None


def _tp_rank_and_size() -> tuple[int, int]:
    """This process's rank in the tensor-parallel group and the group's size."""
    group = _tp_group()
    return (group.rank_in_group, group.world_size) if group is not None else (0, 1)


def _all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Sum ``x`` over the tensor-parallel group, in its own dtype; ``x`` itself on one rank."""
    group = _tp_group()
    if group is None or group.world_size == 1:
        return x
    return _waited(funcol.all_reduce(x, "sum", group.device_group))


def _expert_group():
    """The group the MoE dispatch and combine span: every rank of every replica.

    On one replica that is the tensor-parallel group; across replicas, the wide expert
    group. None outside a distributed run.
    """
    group = _tp_group()
    if group is None or get_world_group().world_size == group.world_size:
        return group
    return get_wide_ep_group()


def _expert_rank(tp_rank: torch.Tensor | None) -> torch.Tensor | None:
    """This rank's ``[1]`` int32 rank in the expert group, a graph input; None on one rank."""
    group = _expert_group()
    if group is None or group.world_size == 1:
        return None
    if group.world_size == _tp_group().world_size:
        return tp_rank
    return group.rank_in_group_tensor.reshape(1).to(torch.int32)


def _replicas() -> int:
    """The replicas the expert group spans."""
    group = _expert_group()
    return 1 if group is None else group.world_size // _tp_group().world_size


def _replica_rank(tp_rank: torch.Tensor | None) -> torch.Tensor:
    """This rank's ``[1]`` int32 replica in the expert group, a graph input."""
    return torch.div(_expert_rank(tp_rank), _tp_group().world_size, rounding_mode="floor")


def _expert_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum ``x`` over the expert group in its own dtype; ``x`` itself on one rank."""
    group = _expert_group()
    if group is None or group.world_size == 1:
        return x
    return _waited(funcol.all_reduce(x, "sum", group.device_group))


def _expert_scatter(x: torch.Tensor) -> torch.Tensor:
    """Sum ``x`` over the expert group and keep this rank's part of dim 0."""
    group = _expert_group()
    if group is None or group.world_size == 1:
        return x
    return _waited(funcol.reduce_scatter_tensor(x, "sum", 0, group.device_group))


def _waited(x: torch.Tensor) -> torch.Tensor:
    """``x`` as a plain tensor once its collective is done.

    Run eagerly, a functional collective returns an async tensor, which the NKI kernels
    cannot take; a traced graph never holds one.
    """
    return x.wait() if isinstance(x, funcol.AsyncCollectiveTensor) else x


def _rank_tensor(rank: torch.Tensor | None, tp_size: int) -> torch.Tensor | None:
    """The ``[1]`` int32 tensor-parallel rank the AttnRes kernel slices by, or None on one rank.

    The rank is a graph input, so every rank traces the same graph.
    """
    if tp_size == 1:
        return None
    if rank is None:
        raise ValueError(f"Kimi K3 at tensor-parallel size {tp_size} needs the rank tensor")
    return rank.reshape(1).to(torch.int32)


def _frozen(
    shape: tuple[int, ...], dtype: torch.dtype, shard_dim: int | None = None
) -> nn.Parameter:
    """An uninitialized parameter; the loader assigns the stored tensor.

    With ``shard_dim``, ``shape`` is this rank's shard and the loader reads the rank's
    contiguous part of the stored tensor along that dim.
    """
    param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
    _, tp_size = _tp_rank_and_size()
    if shard_dim is not None and tp_size > 1:
        set_weight_loader(param, sharding_weight_loader(shard_dim, shape[shard_dim], tp_size))
    return param


class _Weight(nn.Module):
    """A checkpoint module that holds one ``weight``, whole or this rank's shard."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, shard_dim: int | None = None):
        super().__init__()
        self.weight = _frozen(shape, dtype, shard_dim)


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
        "gate_up_scale": ((g, 48, 2 * ni * nk * 512 // 3), torch.uint8),
        "down_weight": ((g, nk, 128, ni * 512), torch.uint16),
        "down_scale": ((g, 48, nk * ni * 512 // 3), torch.uint8),
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


def _prefill_projection(x, weight, heads: bool = False):
    """``[T, K]`` rows times ``weight`` ``[K, N]`` on the library's prefill output projection.

    With ``heads`` the rows go in as a free ``[1, T, K / d, d]`` view in the library's BSNd
    layout; otherwise the wrapper transposes them to BNdS. The wrapper falls back to torch
    outside its kernel's limits, so this raises there instead.
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
    if heads:
        view = x.view(1, tokens, width // d, d)
        return o_proj(view, weight, attention_layout=OProjAttentionLayout.BSNd)[0]
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
    row holds the null block 0: a scratch page that no request owns.
    """
    return meta["block_table_tensor"][:, 0].long()


def _page_layout(view: torch.Tensor, pages: torch.Tensor) -> tuple[int, torch.dtype, tuple]:
    """Where a typed cache view sits in each page of its int8 ``[blocks, page bytes]`` buffer.

    Returns the view's first byte in a page, its dtype and its shape per page.
    """
    start = view.data_ptr() - pages.data_ptr()
    width = view[0].numel() * view.element_size()
    if not (
        view.untyped_storage().data_ptr() == pages.untyped_storage().data_ptr()
        and view.shape[0] == pages.shape[0]
        and view[0].is_contiguous()
        and view.stride(0) * view.element_size() == pages.stride(0)
        and 0 <= start <= pages.shape[1] - width
    ):
        raise ValueError("a cache view must lie inside each page of its page buffer")
    return start, view.dtype, tuple(view.shape[1:])


def _typed_view(pages: torch.Tensor, layout: tuple[int, torch.dtype, tuple]) -> torch.Tensor:
    """The typed ``[blocks, ...]`` view of the page buffer that ``layout`` names."""
    start, dtype, shape = layout
    width = math.prod(shape) * dtype.itemsize
    return pages[:, start : start + width].view(dtype).view(-1, *shape)


def _with_parts(rows: torch.Tensor, parts) -> torch.Tensor:
    """Page rows ``[n, page bytes]`` with each ``(layout, values)`` part's bytes in its place."""
    for (start, dtype, shape), values in parts:
        width = math.prod(shape) * dtype.itemsize
        data = values.to(dtype).contiguous().view(torch.int8).reshape(rows.shape[0], width)
        rows = torch.cat([rows[:, :start], data, rows[:, start + width :]], dim=1)
    return rows


def _int8_pages(cache: torch.Tensor) -> torch.Tensor:
    """The int8 ``[blocks, page bytes]`` view of a cache tensor's buffer, one page per block."""
    page = cache.stride(0) * cache.element_size()
    return cache.view(torch.int8).as_strided((cache.shape[0], page), (page, 1))


def _with_positions(meta: dict | None, positions: torch.Tensor | None) -> dict | None:
    """A layer's metadata with the step's token positions added."""
    return None if meta is None else {**meta, "positions": positions}


class KimiK3KDA(nn.Module):
    """Kimi Delta Attention: projections, short conv, gated delta rule, gated output.

    Each tensor-parallel rank holds a contiguous group of heads, so ``o_proj`` gives
    this rank's partial of the output.
    """

    def __init__(self, text_config: KimiK3TextConfig):
        super().__init__()
        cfg = text_config.linear_attn_config
        if not cfg.get("use_full_rank_gate", False):
            raise ValueError("Kimi K3 KDA layers take a full-rank output gate")
        hidden, dtype = text_config.hidden_size, text_config.torch_dtype
        _, tp_size = _tp_rank_and_size()
        self.num_heads = text_config.kda_heads_per_rank(tp_size)
        self.head_dim = int(cfg["head_dim"])
        width = self.num_heads * self.head_dim
        taps = int(cfg["short_conv_kernel_size"])
        self.eps = text_config.rms_norm_eps
        self.lower_bound = cfg.get("gate_lower_bound")
        for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
            setattr(self, name, _Weight((width, hidden), dtype, shard_dim=0))
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            setattr(self, name, _Weight((width, 1, taps), torch.float32, shard_dim=0))
        self.f_a_proj = _Weight((self.head_dim, hidden), dtype)
        self.f_b_proj = _Weight((width, self.head_dim), dtype, shard_dim=0)
        self.b_proj = _Weight((self.num_heads, hidden), dtype, shard_dim=0)
        self.A_log = _frozen((self.num_heads,), torch.float32, shard_dim=0)
        self.dt_bias = _frozen((width,), torch.float32, shard_dim=0)
        self.o_norm = _Weight((self.head_dim,), torch.float32)
        self.o_proj = _Weight((hidden, width), dtype, shard_dim=1)
        self.register_parameter("input_weight", None)
        self.register_buffer("conv_weight", None, persistent=False)
        self.register_buffer("forget_weight", None, persistent=False)
        self.register_buffer("output_weight", None, persistent=False)
        # The int8 [slots, page bytes] page buffer of the KV caches, shared by the layers on
        # it, and where the conv and state views sit in a page; bound with the KV caches.
        self.pages = None
        self._conv_layout = None
        self._state_layout = None

    @property
    def conv_state(self) -> torch.Tensor:
        """Per-sequence conv rows ``[slots, taps - 1, 3W]`` bf16, a page buffer view."""
        return _typed_view(self.pages, self._conv_layout)

    @property
    def recurrent_state(self) -> torch.Tensor:
        """Per-sequence recurrent state ``[slots, heads, D, D]`` fp32, a page buffer view."""
        return _typed_view(self.pages, self._state_layout)

    def bind_pages(self, pages: torch.Tensor, conv: torch.Tensor, state: torch.Tensor) -> None:
        """Bind the page buffer and keep where the conv and state views sit in a page."""
        self.pages = pages
        self._conv_layout = _page_layout(conv, pages)
        self._state_layout = _page_layout(state, pages)

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
        self._store(slot, rows.index_select(0, last).unsqueeze(0), state.unsqueeze(0))
        return self.output_prefill(core, gate.float())

    def _store(self, slot: torch.Tensor, conv: torch.Tensor, state: torch.Tensor) -> None:
        """Write a slot's conv rows and state back as its whole page row of the buffer.

        A compiled step then writes the page buffer itself. A write through the conv or
        state view, a slice of every page, would rebuild the whole buffer.
        """
        parts = ((self._conv_layout, conv), (self._state_layout, state))
        self.pages.index_copy_(0, slot, _with_parts(self.pages.index_select(0, slot), parts))

    def output_prefill(self, core: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Gated output norm and ``o_proj`` of fp32 ``[T, W]`` core and gate: ``[T, H]`` fp32."""
        return kda_output_prefill(core, gate, self.o_norm.weight, self.o_proj.weight, self.eps)

    def forward_decode(self, mix, k_slice, norm_weight, meta) -> torch.Tensor:
        """One KDA decode step on the page buffer by slot; returns the ``[T, H]`` bf16 partial."""
        proj = _project(mix, self.input_weight, norm_weight, self.eps)
        # core: [128, T, heads, 1] bf16, the output projection's input layout. The
        # kernel writes each token's page in place, so the returned buffer is not kept.
        core, _ = kda_fused_decode(
            proj,
            _state_slots(meta).int(),
            self.pages,
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
        return _output_projection(core, self.output_weight)


class KimiK3MLA(nn.Module):
    """Multi-head latent attention without rotary (NoPE), with a sigmoid output gate.

    Each tensor-parallel rank holds a contiguous group of heads. ``q_a`` and ``kv_a`` load
    whole: prefill projects the whole normed input with them, and decode projects the
    rank's K slice with the rank's rows of them, then sums the partials over the group.
    """

    def __init__(self, text_config: KimiK3TextConfig):
        super().__init__()
        c = text_config
        if not (c.mla_use_nope and c.mla_use_output_gate):
            raise ValueError("Kimi K3 MLA layers are NoPE with an output gate")
        self.tp_rank, self.tp_size = _tp_rank_and_size()
        hidden, dtype, heads = c.hidden_size, c.torch_dtype, c.mla_heads_per_rank(self.tp_size)
        if hidden % self.tp_size:
            raise ValueError(f"hidden size {hidden} does not split over {self.tp_size} ranks")
        q_head = c.qk_nope_head_dim + c.qk_rope_head_dim
        self.eps = c.rms_norm_eps
        self.q_a_proj = _Weight((c.q_lora_rank, hidden), dtype)
        self.q_a_layernorm = KimiK3RMSNorm(c.q_lora_rank, MLA_NORM_EPS, dtype)
        self.q_b_proj = _Weight((heads * q_head, c.q_lora_rank), dtype, shard_dim=0)
        self.kv_a_proj_with_mqa = _Weight((c.kv_lora_rank + c.qk_rope_head_dim, hidden), dtype)
        self.kv_a_layernorm = KimiK3RMSNorm(c.kv_lora_rank, MLA_NORM_EPS, dtype)
        kv_b_rows = heads * (c.qk_nope_head_dim + c.v_head_dim)
        self.kv_b_proj = _Weight((kv_b_rows, c.kv_lora_rank), dtype, shard_dim=0)
        self.o_proj = _Weight((hidden, heads * c.v_head_dim), dtype, shard_dim=1)
        self.g_proj = _Weight((heads * c.v_head_dim, hidden), dtype, shard_dim=0)
        self.register_parameter("input_weight", None)
        self.register_parameter("gate_weight", None)
        self.register_buffer("latent_weight", None, persistent=False)
        self.register_buffer("output_weight", None, persistent=False)
        self.register_parameter("w_qb_t", None)
        self.register_parameter("w_uk", None)
        self.register_buffer("w_uv_t", None, persistent=False)
        # The int8 [blocks, page bytes] page buffer of the KV caches, shared by the layers on
        # it, and where the latent cache view sits in a page; bound with the KV caches.
        self.pages = None
        self._cache_layout = None

    @property
    def kv_cache(self) -> torch.Tensor:
        """Paged latent cache ``[blocks, block_size, kv_lora_rank + rope]`` bf16, a page view.

        The kernels read and write the page buffer itself; this view is for inspection only.
        """
        return _typed_view(self.pages, self._cache_layout)

    def bind_pages(self, pages: torch.Tensor, cache: torch.Tensor) -> None:
        """Bind the page buffer and keep where the latent cache view sits in a page."""
        self.pages = pages
        self._cache_layout = _page_layout(cache, pages)

    def prepare_after_load(self) -> None:
        """Lay out the weights the kernels read, once.

        The rank's K rows of q_a and kv_a go to fp32 ``[H / TP, Q + L]`` for decode, then the
        whole weights stack as bf16 ``[H, Q + L]``; g stacks as ``[H, HV]``; ``o_proj`` as
        ``[HV, H]``. g is projected alone at decode, so it is its own stack rather than a
        slice of q_a's. ``q_b`` and ``kv_b`` take the attention kernels' layouts, which
        replace them: ``q_b`` transposed, the ``k_nope`` rows of ``kv_b`` by head, and its
        value rows by head, transposed, in fp32.
        """
        if self.input_weight is None:
            latent = torch.cat([self.q_a_proj.weight, self.kv_a_proj_with_mqa.weight])
            rows = latent.shape[1] // self.tp_size
            k_rows = latent[:, self.tp_rank * rows : (self.tp_rank + 1) * rows]
            self.latent_weight = k_rows.float().t().contiguous()
            self.input_weight = _stack(self.q_a_proj, self.kv_a_proj_with_mqa)
            self.gate_weight = _stack(self.g_proj)
        if self.w_uk is None:
            w_qb_t, w_uk, self.w_uv_t = prepare_mla_weights(
                self.q_b_proj.weight, self.kv_b_proj.weight
            )
            self.w_qb_t = nn.Parameter(w_qb_t, requires_grad=False)
            self.w_uk = nn.Parameter(w_uk, requires_grad=False)
            self.q_b_proj.weight = None
            self.kv_b_proj.weight = None
        self.output_weight = self.o_proj.weight.t().contiguous()

    def _attend(self, latent, gate, meta):
        """Paged attention of the new tokens' ``q_a`` and ``kv_a`` rows; stores their latent rows.

        A prefill chunk is one sequence's tokens after its cached prefix. A decode step is one
        token per sequence, at its position: the count of tokens its sequence has cached. The
        kernels take the int8 page buffer with the latent cache's byte offset in a page, and
        write each token's latent row at its slot in place, a padding token's (-1) in the null
        block, so the returned buffer is not kept. They raise outside their limits.
        """
        start, _, (block_size, _) = self._cache_layout
        args = (
            latent.float(),
            gate,
            self.q_a_layernorm.weight,
            self.kv_a_layernorm.weight,
            self.w_qb_t,
            self.w_uk,
            self.w_uv_t,
            self.pages,
            start,
            block_size,
        )
        table, slots = meta["block_table_tensor"], meta["slot_mapping"]
        if meta["max_query_len"] > meta["decode_token_threshold"]:
            if table.shape[0] != 1:
                raise ValueError(f"Kimi K3 MLA prefill takes one sequence, got {table.shape[0]}")
            prefix = meta["cached_seq_len"].reshape(1)
            out, _ = mla_paged_prefill(*args, table[0], prefix, slots, MLA_NORM_EPS)
        else:
            out, _ = mla_paged_decode(*args, table, meta["positions"], slots, MLA_NORM_EPS)
        return out

    def forward_prefill(self, mix, norm_weight, meta) -> torch.Tensor:
        """MLA over a prefill chunk; returns this rank's ``[T, H]`` partial."""
        latent = _project(mix, self.input_weight, norm_weight, self.eps)
        gate = _project(mix, self.gate_weight, norm_weight, self.eps)
        return _prefill_projection(self._attend(latent, gate, meta), self.output_weight)

    def forward_decode(self, mix, k_slice, norm_weight, meta) -> torch.Tensor:
        """One MLA decode step; returns this rank's ``[T, H]`` bf16 partial.

        ``k_slice`` is the rank's normed fp32 K slice of the mix; the q_a and kv_a partials
        sum over the tensor-parallel group in fp32.
        """
        latent = _all_reduce(_output_projection(k_slice, self.latent_weight))
        gate = _project(mix, self.gate_weight, norm_weight, self.eps)
        return _output_projection(self._attend(latent, gate, meta), self.output_weight)


class KimiK3MLP(nn.Module):
    """Gated SiTU MLP: the dense layer and the shared experts.

    With ``split``, each tensor-parallel rank holds a contiguous part of the intermediate
    width and returns its partial of the down projection; with ``reduce`` it also sums the
    partials over the group. The shared experts split without a sum of their own: the MoE
    block's one sum adds their partial to the routed one.
    """

    def __init__(
        self,
        text_config: KimiK3TextConfig,
        intermediate_size: int,
        reduce: bool = False,
        split: bool = False,
    ):
        super().__init__()
        hidden, dtype = text_config.hidden_size, text_config.torch_dtype
        _, tp_size = _tp_rank_and_size()
        split = (split or reduce) and tp_size > 1
        self.reduce = reduce and split
        if split:
            if intermediate_size % tp_size:
                raise ValueError(
                    f"MLP width {intermediate_size} does not split over {tp_size} ranks"
                )
            intermediate_size //= tp_size
        shard = (0, 1) if split else (None, None)
        self.eps = text_config.rms_norm_eps
        self.beta = text_config.activation_situ_beta or 1.0
        self.linear_beta = text_config.activation_situ_linear_beta
        self.gate_proj = _Weight((intermediate_size, hidden), dtype, shard_dim=shard[0])
        self.up_proj = _Weight((intermediate_size, hidden), dtype, shard_dim=shard[0])
        self.down_proj = _Weight((hidden, intermediate_size), dtype, shard_dim=shard[1])
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
        """The MLP over a prefill chunk: ``[T, H]``, summed over the group with ``reduce``."""
        out = _prefill_projection(self.activation(mix, norm_weight), self.down_weight)
        return _all_reduce(out) if self.reduce else out

    def forward_decode(self, mix, k_slice, norm_weight) -> torch.Tensor:
        """The MLP at decode: ``[T, H]`` bf16, summed over the group with ``reduce``."""
        rows = _decode_layout(self.activation(mix, norm_weight))
        out = _output_projection(rows, self.down_weight)
        return _all_reduce(out) if self.reduce else out


class KimiK3Router(nn.Module):
    """Router weights: the expert score projection on this rank's columns, and the bias."""

    def __init__(self, text_config: KimiK3TextConfig):
        super().__init__()
        experts, (_, tp_size) = text_config.num_experts, _tp_rank_and_size()
        self.weight = _frozen(
            (experts, text_config.hidden_size // tp_size), text_config.torch_dtype, shard_dim=1
        )
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
    """Latent MoE over the expert group: latent down, MXFP4 experts by rank, combine, up, shared.

    Each rank of each replica holds its own range of experts and its hidden columns of the
    latent down and router weights. The dispatch all-reduce sums the ranks' K partials of
    the latent and the router logits and gathers every replica's rows; the combine
    reduce-scatter sums the experts' partials and gives each rank its replica's rows by its
    latent columns. The block's output is this rank's fp32 pack ``[up | shared | sum of
    squares]``, which one sum over the tensor-parallel group completes.
    """

    #: Rows per prefill call of the tile kernel, every local expert taking all of them.
    sub_chunk = 512

    def __init__(self, text_config: KimiK3TextConfig, num_programs: int):
        super().__init__()
        c = text_config
        if not c.latent_moe_use_norm or c.moe_router_activation_func != "sigmoid":
            raise ValueError("Kimi K3 MoE layers use a latent norm and a sigmoid router")
        if c.num_experts_per_token != 16 or not c.moe_renormalize:
            raise ValueError("Kimi K3 MoE kernels choose 16 experts per token and renormalize")
        hidden, latent, dtype = c.hidden_size, c.routed_expert_hidden_size, c.torch_dtype
        group, (_, tp_size) = _expert_group(), _tp_rank_and_size()
        ranks = group.world_size if group is not None else 1
        if c.num_experts % ranks or hidden % tp_size or latent % tp_size:
            raise ValueError(
                f"Kimi K3 MoE: {c.num_experts} experts, hidden {hidden} and latent {latent} "
                f"must split over {ranks} expert ranks and {tp_size} tensor-parallel ranks"
            )
        self.eps = c.rms_norm_eps
        self.latent_size = latent
        self.scaling = c.routed_scaling_factor
        self.local_experts = c.num_experts // ranks
        self.gate = KimiK3Router(c)
        self.experts = KimiK3Experts(c, self.local_experts, num_programs)
        self.shared_experts = KimiK3MLP(
            c, c.moe_intermediate_size * c.num_shared_experts, split=True
        )
        self.routed_expert_down_proj = _Weight((latent, hidden // tp_size), dtype, shard_dim=1)
        self.routed_expert_up_proj = _Weight((hidden, latent // tp_size), dtype, shard_dim=1)
        self.routed_expert_norm = KimiK3RMSNorm(latent, c.rms_norm_eps, dtype)
        self.register_buffer("down_router_weight", None, persistent=False)
        self.register_buffer("latent_norm_weight", None, persistent=False)
        self.register_buffer("latent_up_weight", None, persistent=False)
        self.register_buffer("shared_down_weight", None, persistent=False)

    def prepare_after_load(self) -> None:
        """Lay out the weights the kernels read, once.

        This rank's rows of the latent down and router weights go to one fp32
        ``[H / TP, K + E]`` weight, which the decode and prefill projections both read; the
        bf16 originals are freed. The tail takes this rank's latent columns of the norm
        weight as fp32 ``[K / TP, 1]`` and the latent up and shared down weights as
        contiguous ``[K / TP, H]`` and ``[I / TP, H]``.
        """
        if self.down_router_weight is None:
            down_router = torch.cat([self.routed_expert_down_proj.weight, self.gate.weight])
            self.down_router_weight = down_router.float().t().contiguous()
            self.routed_expert_down_proj.weight = None
            self.gate.weight = None
        self.shared_experts.prepare_after_load()
        tp_rank, tp_size = _tp_rank_and_size()
        norm = self.routed_expert_norm.weight.float().chunk(tp_size)[tp_rank]
        self.latent_norm_weight = norm.reshape(-1, 1).contiguous()
        self.latent_up_weight = self.routed_expert_up_proj.weight.t().contiguous()
        self.shared_down_weight = self.shared_experts.down_weight

    def _expert_base(self, tp_rank) -> torch.Tensor:
        """The ``[1, 1]`` int32 global id of this rank's first expert, a graph input."""
        rank = _expert_rank(tp_rank)
        if rank is None:
            return torch.zeros(1, 1, dtype=torch.int32, device=self.down_router_weight.device)
        return (rank * self.local_experts).reshape(1, 1).to(torch.int32)

    def _select_and_run(self, routed, base) -> torch.Tensor:
        """The local experts on fp32 ``[R T, K + E]`` decode rows: the partial in combine order."""
        shared = self.shared_experts
        return _require(routed_experts_mx, "routed experts")(
            routed,
            self.gate.e_score_correction_bias,
            base,
            *self.experts.banks(),
            beta=shared.beta,
            linear_beta=shared.linear_beta,
            scaling=self.scaling,
            replicas=_replicas(),
            chunks=_tp_rank_and_size()[1],
        )

    def _combine(self, partial) -> torch.Tensor:
        """This rank's fp32 rows ``[T, K / C]`` of the summed partials ``[R, C, T, K / C]``."""
        return _expert_scatter(partial.reshape(-1, partial.shape[-1]))

    def forward_decode(self, mix, k_slice, norm_weight, tp_rank=None) -> torch.Tensor:
        """The MoE block at decode; returns the fp32 tail pack ``[T, 2H + 1]``, summed over TP.

        ``k_slice`` holds this rank's hidden columns of every replica's rows, the other
        replicas' rows zero, so one sum over the expert group adds the K partials and
        gathers the replicas.
        """
        routed = _expert_sum(_output_projection(k_slice, self.down_router_weight))
        combined = self._combine(self._select_and_run(routed, self._expert_base(tp_rank)))
        gate_up = _project(mix, self.shared_experts.gate_up_weight, norm_weight, self.eps)
        return _all_reduce(self.tail_decode(combined, gate_up))

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

    def forward_prefill(self, mix, norm_weight, tp_rank=None) -> torch.Tensor:
        """The MoE block over a prefill chunk: this rank's fp32 pack ``[T, 2H + 1]``.

        The caller sums the packs over the tensor-parallel group and finishes the block
        with ``prefill_output``.
        """
        partial = self._down_router_prefill(mix, norm_weight, tp_rank)
        routed = _expert_sum(self._replica_rows(partial, tp_rank))
        rows = self._prefill_experts(routed, tp_rank)
        replicas, tp_size = _replicas(), _tp_rank_and_size()[1]
        order = rows.view(replicas, -1, tp_size, self.latent_size // tp_size).transpose(1, 2)
        return self._prefill_pack(self._combine(order), mix, norm_weight)

    def prefill_output(self, pack) -> torch.Tensor:
        """The fp32 block output ``[T, H]`` of the summed pack: the scaled up part plus shared."""
        hidden = (pack.shape[-1] - 1) // 2
        up, shared, squares = pack.split([hidden, hidden, 1], -1)
        return up * torch.rsqrt(squares / self.latent_size + self.eps) + shared

    def _down_router_prefill(self, mix, norm_weight, tp_rank) -> torch.Tensor:
        """This rank's fp32 K partial ``[T, K + E]`` of the latent and the router logits.

        The 1/RMS comes from the whole row in fp32. The rank's bf16 hidden columns, picked
        by the rank tensor, are scaled by it and by the gain in fp32, so the library
        projection takes fp32 rows and keeps fp32 to its store.
        """
        tokens, hidden = mix.shape
        tp_size = _tp_rank_and_size()[1]
        inv = torch.rsqrt(mix.float().pow(2).mean(-1, keepdim=True) + self.eps)
        cols, gain = mix, norm_weight.reshape(1, hidden)
        if tp_size > 1:
            width = hidden // tp_size
            cols = mix.view(tokens, tp_size, width).index_select(1, tp_rank).reshape(tokens, width)
            gain = norm_weight.view(tp_size, width).index_select(0, tp_rank).reshape(1, width)
        k_slice = cols.float() * inv * gain.float()
        return _prefill_projection(k_slice, self.down_router_weight, heads=True)

    def _replica_rows(self, partial, tp_rank) -> torch.Tensor:
        """``partial`` at this replica's rows of an fp32 ``[R T, W]`` buffer, the others zero."""
        replicas = _replicas()
        if replicas == 1:
            return partial
        mine = torch.arange(replicas, device=partial.device) == _replica_rank(tp_rank)
        return torch.where(mine.view(replicas, 1, 1), partial, 0.0).reshape(-1, partial.shape[-1])

    def _prefill_experts(self, routed, tp_rank) -> torch.Tensor:
        """The local experts' fp32 partial ``[N, K]`` of ``N`` gathered prefill rows.

        Selection runs on 128-row tiles. Each sub-chunk calls the tile kernel once with
        every local expert as an item over all its rows, weighted by each row's affinity for
        the expert and 0 where the row did not choose it, so no row can drop; the items sum
        in fp32.
        """
        latent, logits = routed.split([self.latent_size, routed.shape[-1] - self.latent_size], -1)
        select = _require(router_select, "router selection")
        bias = self.gate.e_score_correction_bias
        picks = [select(tile, bias, scaling=self.scaling) for tile in logits.split(128)]
        ids, weights = (torch.cat(part) for part in zip(*picks))
        experts = torch.arange(self.local_experts, dtype=torch.int32, device=ids.device)
        chosen = ids.unsqueeze(-1) == (self._expert_base(tp_rank).reshape(1) + experts)
        affinity = torch.where(chosen, weights.unsqueeze(-1), 0.0).sum(1)
        tiles, shared = _require(routed_expert_tiles_mx, "routed expert tiles"), self.shared_experts
        out = []
        for rows, weight in zip(latent.split(self.sub_chunk), affinity.split(self.sub_chunk)):
            items = rows.unsqueeze(0).expand(self.local_experts, -1, -1).contiguous()
            out.append(tiles(items, weight.t().contiguous(), *self.experts.banks(),
                             beta=shared.beta, linear_beta=shared.linear_beta).sum(0))
        return torch.cat(out)

    def _prefill_pack(self, combined, mix, norm_weight) -> torch.Tensor:
        """This rank's fp32 pack ``[T, 2H + 1]`` at prefill, in the decode tail's form.

        The up partial projects this rank's latent columns with the norm weight applied in
        fp32 and one bf16 rounding; the shared partial is this rank's part of the shared
        experts; the last column is the sum of squares of the combined columns.
        """
        rows = (combined * self.latent_norm_weight.t()).to(mix.dtype)
        up = _prefill_projection(rows, self.latent_up_weight)
        shared = self.shared_experts.forward_prefill(mix, norm_weight)
        squares = combined.square().sum(-1, keepdim=True)
        return torch.cat([up.float(), shared.float(), squares], -1)


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
            self.mlp = KimiK3MLP(c, c.intermediate_size, reduce=True)
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

    def forward_prefill(self, stream, bank, meta, tp_rank=None) -> torch.Tensor:
        """One layer over a prefill chunk; returns the new stream.

        ``tp_rank`` is the ``[1]`` int32 tensor-parallel rank, or None on one rank; the MoE
        block's K slice and expert range follow it.
        """
        mix, base = self.attention_mix(stream, bank)
        attn = self.self_attn.forward_prefill(mix, self.input_layernorm.weight, meta)
        attn = _all_reduce(attn).to(stream.dtype)
        stream = attn if base is None else base + attn
        mix, norm = self.mlp_mix(stream, bank), self.post_attention_layernorm.weight
        if hasattr(self, "block_sparse_moe"):
            moe = self.block_sparse_moe
            out = moe.prefill_output(_all_reduce(moe.forward_prefill(mix, norm, tp_rank)))
        else:
            out = self.mlp.forward_prefill(mix, norm)
        return stream + out.to(stream.dtype)

    def _mix_decode(self, stream, tail, bank, site: str, tp_rank=None):
        """AttnRes mix at decode with the producer's tail as its prologue.

        Returns the mix, the new stream, the bank and, at the MLA attention site and
        the MoE site, the normed fp32 K slice of the mix: the columns of tensor-parallel
        rank ``tp_rank``. At the MoE site the slice holds every replica's rows, the other
        replicas' rows zero.
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
        sliced = emit_k_slice and tp_rank is not None
        replicas = _replicas() if emit_k_slice and not attention else 1
        mix, stream, k_slice, new_bank = attn_res_decode(
            bank,
            stream,
            self.attention_score_w if attention else self.mlp_score_w,
            (self.attention_rows if attention else self.mlp_rows) + 1,
            **_tail_args(tail, stream, self.latent_size),
            bank_slot=self.layer_idx // self.block_size if attention and self.block_start else None,
            norm_weight=norm.weight if emit_k_slice else None,
            tp_rank=tp_rank if sliced else None,
            tp_size=_tp_rank_and_size()[1] if sliced else 1,
            dp_rank=_replica_rank(tp_rank) if replicas > 1 else None,
            dp_size=replicas,
            eps=self.eps,
        )
        return mix, stream, k_slice, bank if new_bank is None else new_bank

    def forward_decode(self, stream, tail, bank, meta, tp_rank=None):
        """One decode layer; returns the stream, this layer's tail for the next mix, and the bank.

        ``tp_rank`` is the ``[1]`` int32 tensor-parallel rank, or None on one rank. The
        attention output is summed over the group before the next mix folds it in.
        """
        mix, stream, k_slice, bank = self._mix_decode(stream, tail, bank, "attention", tp_rank)
        attn = self.self_attn.forward_decode(mix, k_slice, self.input_layernorm.weight, meta)
        attn = _all_reduce(attn)
        mix, stream, k_slice, bank = self._mix_decode(stream, attn, bank, "mlp", tp_rank)
        norm = self.post_attention_layernorm.weight
        if hasattr(self, "block_sparse_moe"):
            out = self.block_sparse_moe.forward_decode(mix, k_slice, norm, tp_rank)
        else:
            out = self.mlp.forward_decode(mix, k_slice, norm)
        return stream, out, bank


class KimiK3Model(nn.Module):
    """Embedding, decoder layers, the output AttnRes mix and the final norm weight."""

    def __init__(self, text_config: KimiK3TextConfig, num_programs: int):
        super().__init__()
        c = text_config
        hidden, dtype, eps = c.hidden_size, c.torch_dtype, c.rms_norm_eps
        self.eps = eps
        group = _tp_group()
        # Each rank holds a contiguous part of the vocabulary; the rows sum over the group.
        self.embed_tokens = VocabDimShardedEmbedding(
            c.vocab_size, hidden, dtype=dtype,
            tp_group=group.device_group if group is not None else None,
        )
        self.embed_tokens.weight.requires_grad_(False)
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

    def embed(self, input_ids, rank=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed ``[T]`` token ids: the stream ``[T, H]`` and an unwritten bank ``[T, R, H]``.

        ``rank`` is the tensor-parallel rank tensor that picks this rank's vocabulary rows.
        """
        stream = _waited(self.embed_tokens(input_ids, scatter_tokens=False, rank=rank))
        bank = stream.new_empty((stream.shape[0], self.num_bank_rows, stream.shape[1]))
        return stream, bank

    def output_mix(self, stream, bank) -> torch.Tensor:
        """AttnRes mix of the last stream with every bank row, prefill form."""
        return attn_res_prefill(bank, stream, self.output_score_w, self.eps)

    def forward(
        self, input_ids, attn_metadata, is_prefill: bool, positions=None, rank=None
    ) -> torch.Tensor:
        """Run every layer; returns the output mix ``[T, H]``, before the final norm.

        Each layer reads its cache group's metadata with the tokens' ``positions`` added.
        ``rank`` is the tensor-parallel rank tensor, needed when the group has more than one rank.
        """
        tp_size = _tp_rank_and_size()[1]
        stream, bank = self.embed(input_ids, rank if tp_size > 1 else None)
        metas = [
            _with_positions(attn_metadata.get(f"layers.{i}.self_attn"), positions)
            for i in range(len(self.layers))
        ]
        tail, tp_rank = None, _rank_tensor(rank, tp_size)
        if is_prefill:
            for layer, meta in zip(self.layers, metas):
                stream = layer.forward_prefill(stream, bank, meta, tp_rank)
            return self.output_mix(stream, bank)
        for layer, meta in zip(self.layers, metas):
            stream, tail, bank = layer.forward_decode(stream, tail, bank, meta, tp_rank)
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
            expert_ids=self.local_expert_ids(),
            num_programs=self.num_programs,
            pipelined=dist.is_initialized(),
        )
        unloaded = [name for name, p in self.named_parameters() if p.is_meta]
        if unloaded:
            raise ValueError(
                f"{len(unloaded)} Kimi K3 parameters were not loaded, first {unloaded[0]}"
            )
        self.prepare_after_load()

    def local_expert_ids(self) -> list[int] | None:
        """The global ids of the experts this rank holds, or None when it holds them all."""
        group = _expert_group()
        if group is None or group.world_size == 1:
            return None
        local = self.text_config.num_experts // group.world_size
        return list(range(group.rank_in_group * local, (group.rank_in_group + 1) * local))

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
        """Bind every layer to the one int8 page view of its cache buffer.

        A KDA layer gets its state views and their page buffer; an MLA layer gets its latent
        cache. The layers on one buffer share one page view and make their typed views of it
        in the graph, so a compiled step takes each buffer as one input.
        """
        names = [f"layers.{i}.self_attn" for i in range(len(self.model.layers))]
        for name in names:
            if name not in kv_caches:
                raise KeyError(f"KV cache for layer {name} not initialized")
        shared: dict[int, torch.Tensor] = {}
        layers = list(zip(names, self.model.layers))
        for name, layer in layers:
            if layer.is_kda:
                conv, state, pages = kv_caches[name]
                pages = shared.setdefault(pages.untyped_storage().data_ptr(), pages)
                layer.self_attn.bind_pages(pages, conv, state)
        for name, layer in layers:
            if not layer.is_kda:
                # [blocks, 1, block_size, D]: one latent head.
                cache = kv_caches[name][0].squeeze(1)
                key = cache.untyped_storage().data_ptr()
                pages = shared.setdefault(key, _int8_pages(cache))
                layer.self_attn.bind_pages(pages, cache)

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
        hidden = self.model(input_ids, attn_metadata, is_prefill, positions, rank)
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
