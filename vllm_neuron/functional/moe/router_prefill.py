# SPDX-License-Identifier: Apache-2.0
"""The prefill router: its RMSNorm in torch, the router GEMM and noaux_tc in one launch.

Why this exists. At 8aa22fa every MoE layer of the prefill graph hands the fused
router kernel (``router._noaux_tc_rmsnorm_router_topk_nki``) the layer's PRE-norm
activations, which are the feed-forward mHC site's collapse
``(pre_mix[..., None] * streams.float()).sum(1).to(bf16)``
(``Glm5NextHyperConnection.mhc_pre``). In the served 1k graph 240a08a9 that is HLO
``multiply.7392`` ``f32[1024,4,4096]`` -> ``reduce.7399`` (dims ``{1}``) ->
``convert.7400`` bf16 -> the router custom call ``call.7537`` (layer 3; every MoE layer
repeats it). A custom-call operand cannot fuse into its consumer, so neuronx-cc
materialises the collapse by itself and lowers the multiply + reduce as a batched dot:
for each token and each 512-column chunk of H, one fp32 matmul contracting the 4
streams (``LDWEIGHTS`` + ``MATMUL`` of 2,048 FLOP, then a cast and a DMA out), 8,192
matmuls per layer at 1024 tokens, about 8.8 ms of "compiler ops" per MoE layer.
The attention half and the dense layers run the same collapse in about 1 ms because
its only consumer there is the XLA RMSNorm, so the reduce fuses into the norm.

What this changes. The RMSNorm's scale moves out of the kernel into torch
(:func:`router_rms_scale`, ``bf16(x * rstd)``), next to the experts' own norm
(``Glm5NextModel._rms_norm``), which already reads the same pre-norm tensor. The
collapse then feeds only XLA elementwise consumers, the reduce fuses, and the router
kernel (:func:`noaux_tc_router_prefill_kernel`) reads the scaled rows instead. The
kernel is the fused kernel with its norm stage cut after the ``rstd`` multiply:
:func:`_gamma_stage_dloc` is ``rmsnorm_tkg._rmsnorm_tkg_dloc`` from its gamma multiply
on (the same ``tensor_tensor`` on the same bf16 tile, the same store, ``pe_transpose``
and core exchange), then the same nkilib ``router_topk`` call and the same ``noaux_tc``
stage (``router._noaux_tc_stage``).

Numerics. The fused kernel reads the collapse as bf16 rows from HBM; here the rows
stay in the graph, and neuronx-cc folds the ``.to(float32)`` that reads them into the
collapse's ``fp32 -> bf16`` convert, so :func:`router_rms_scale` first puts them back on
the bf16 grid by arithmetic (:func:`round_to_bf16_grid`). Without it the device router
reads the unrounded fp32 sum (rel_l2 2.8e-3 on the logits against the fused kernel's
1.5e-4, both against an fp64 router on the same rows). The nkilib norm rounds twice,
``bf16(bf16(x * rstd) * gamma)``
(``_rmsnorm_tkg_dloc`` writes ``x * rstd`` back into its bf16 tile before the gamma
multiply), with ``rstd = rsqrt(sum(x**2) * (1/H) + eps)``. The first rounding is here in
XLA, the second in the kernel. The gamma multiply cannot be in XLA too: on the device
neuronx-cc folds a ``bf16 -> fp32`` convert that follows an ``fp32 -> bf16`` one, so
``bf16(bf16(x * rstd) * gamma)`` written in torch runs there as one rounding (22% of
elements one bf16 step off the two-rounding value, 26 of 1024 non-tie rows routed to
another expert set). What moves is the fp32 summation order of ``sum(x**2)`` and the
``rsqrt`` implementation, so a row's ``rstd`` can differ in its last bit and an element
by one bf16 step where ``x * rstd`` sits on a rounding boundary. The tests bound the
effect on the selection.

Envelope. ``Glm5NextRoutedExperts.route_tokens`` asks :func:`prefill_route_admits`
after its decode branch, so ``route_tokens`` stays the block's one routing authority.
Prefill only: every call of at most ``router_decode.DECODE_ROUTE_MAX_TOKENS`` rows has
already taken the decode router there, and :func:`prefill_route_admits` declines such
a call anyway; ``VLLM_NEURON_MOE_PREFILL_ROUTER=0`` declines everything (8aa22fa's
route, the fused kernel). The launch counts into the ``noaux_tc`` router seam's family
(``router.noaux_tc_dispatch_counters``), as the decode router does; the kernel name
tells the two kernels apart.

Both LNC2 cores: yes. The launch is ``[2]``; nkilib's router shards the token rows
over the two programs (``shard_on_tokens``) and the noaux_tc stage covers the same
rows on each core, as in the fused kernel.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.router_topk.router_topk import XSBLayout_tp102__0
from nkilib.core.router_topk.router_topk import router_topk as _substrate_router_topk
from nkilib.core.subkernels.norm_tkg_utils import _DGE_MODE_NONE, pe_transpose
from nkilib.core.subkernels.rmsnorm_tkg import _DLOC_T_TILE_SIZE
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.common_types import RouterActFnType
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.core.utils.logging import get_logger

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

from . import router as _router_seam
from .router_decode import DECODE_ROUTE_MAX_TOKENS

#: ``0`` keeps 8aa22fa's prefill route (the fused kernel on the pre-norm rows).
PREFILL_ROUTER_ENV = "VLLM_NEURON_MOE_PREFILL_ROUTER"

#: The fused kernel's ``x_hbm_layout`` (``x`` arrives in SBUF from the norm stage).
_X_HBM_LAYOUT_FUSED = 0

#: Hidden extents come in whole 128-row partition blocks (nkilib ``router_topk``).
_H_BLOCK = 128


def prefill_router_enabled() -> bool:
    """``VLLM_NEURON_MOE_PREFILL_ROUTER`` is not ``0``."""
    return os.environ.get(PREFILL_ROUTER_ENV, "1") != "0"


def prefill_route_admits(hidden_states, router_weights, top_k: int) -> bool:
    """True when ``route_tokens`` routes this call here instead of the fused kernel.

    More than ``DECODE_ROUTE_MAX_TOKENS`` rows (prefill), bf16 activations and router
    weights, ``top_k == 8``, ``H`` in 128-row blocks, ``8 <= E <= 512``, on an NKI
    device or the simulator, with the switch on. Anything else keeps 8aa22fa's route.
    """
    if not isinstance(hidden_states, Tensor) or not isinstance(router_weights, Tensor):
        return False
    if hidden_states.dim() not in (2, 3) or router_weights.dim() != 2:
        return False
    hidden = int(hidden_states.shape[-1])
    if hidden <= 0:
        return False
    tokens = hidden_states.numel() // hidden
    experts = int(router_weights.shape[1])
    return (
        prefill_router_enabled()
        and tokens > DECODE_ROUTE_MAX_TOKENS
        and int(top_k) == _router_seam.NOAUX_TC_MAX8_WIDTH
        and hidden_states.dtype == torch.bfloat16
        and router_weights.dtype == torch.bfloat16
        and hidden % _H_BLOCK == 0
        and tuple(router_weights.shape) == (hidden, experts)
        and _router_seam.NOAUX_TC_MAX8_WIDTH <= experts <= _router_seam._NOAUX_TC_F_MAX
        and can_run_kernel(hidden_states)
    )


#: Veltkamp's splitting constant for a 24-bit significand cut to bf16's 8: ``2**16 + 1``.
_VELTKAMP_BF16 = 65537.0


def round_to_bf16_grid(x: Tensor) -> Tensor:
    """fp32 ``x`` rounded to the nearest bf16 value (ties to even), kept in fp32.

    Veltkamp's split, ``c - (c - x)`` with ``c = (2**16 + 1) * x``: plain fp32 arithmetic,
    so no compiler can fold it the way neuronx-cc folds a ``bf16 -> fp32`` convert that
    follows an ``fp32 -> bf16`` one. Exact for every finite ``|x| < 2**111``.
    """
    c = x * _VELTKAMP_BF16
    return c - (c - x)


def router_rms_scale(hidden_states: Tensor, eps: float) -> Tensor:
    """``bf16(x * rstd)``: the nkilib RMSNorm up to its first rounding, in torch.

    ``x`` is the bf16 row the fused kernel reads from HBM. In the graph ``hidden_states``
    is ``mhc_pre``'s ``fp32 -> bf16`` collapse, and the ``.to(float32)`` here would fold
    into it on the device (the norm would read the unrounded fp32 sum), so the row is put
    back on the bf16 grid by arithmetic (:func:`round_to_bf16_grid`), a no-op wherever the
    convert pair is honoured.

    Args:
        hidden_states: ``[T, H]`` bf16 pre-norm activations.
        eps: the RMSNorm epsilon.

    Returns:
        ``[T, H]`` bf16; the kernel applies the gain (the second rounding).
    """
    hidden = int(hidden_states.shape[-1])
    x = round_to_bf16_grid(hidden_states.to(torch.float32))
    rstd = torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) * (1.0 / hidden) + eps)
    return (x * rstd).to(hidden_states.dtype)


def _gamma_stage_dloc(scaled_hbm, gamma, output_hbm, output_sb):
    """``rmsnorm_tkg._rmsnorm_tkg_dloc`` from its gamma multiply on, ``sync_output=True``.

    ``scaled_hbm`` ``[T, H]`` holds ``bf16(x * rstd)``, the value nkilib's tile holds at
    that point. Each LNC core takes ``T // 2`` rows in 128-row tiles: the gain multiply
    (bf16 tile in place, as nkilib), the ``[T, H]`` store, the PE transpose into
    ``output_sb`` ``[128, T, H // 128]`` at the core's shard, then the exchange that gives
    each core the whole ``[128, T, H // 128]``. The lines are nkilib's, minus the
    ``activation_reduce`` / ``rsqrt`` / ``tensor_scalar`` that made ``x * rstd``.
    """
    t_rows, hidden = scaled_hbm.shape
    h0 = nl.tile_size.pmax
    h1 = hidden // h0
    tile_size = _DLOC_T_TILE_SIZE

    _, lnc, shard_id = get_verified_program_sharding_info("gamma_stage_dloc", (0, 1))
    kernel_assert(lnc == 2, "gamma_stage_dloc requires LNC=2")
    kernel_assert(t_rows % 2 == 0, "T must be even for LNC=2 sharding")
    t_shard = t_rows // 2
    kernel_assert(t_shard % tile_size == 0, f"T//2 must be divisible by {tile_size}")

    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=nl.tile_size.total_available_sbuf_size,
        logger=get_logger("gamma_stage_dloc"),
        use_auto_alloc=True,
    )
    sbm.open_scope(name="gamma_stage_dloc")
    num_tiles = t_shard // tile_size
    sb_shard_offset = shard_id * t_shard

    tile_buf = sbm.alloc_heap((tile_size, hidden), dtype=scaled_hbm.dtype, buffer=nl.sbuf)
    gamma_sb = sbm.alloc_heap((tile_size, hidden), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_sb, src=gamma.broadcast(dim=0, size=tile_size),
                  dge_mode=_DGE_MODE_NONE)

    for tile_idx in range(num_tiles):
        t_start = shard_id * t_shard + tile_idx * tile_size
        sb_t_start = sb_shard_offset + tile_idx * tile_size
        nisa.dma_copy(dst=tile_buf,
                      src=scaled_hbm.slice(dim=0, start=t_start, end=t_start + tile_size),
                      dge_mode=_DGE_MODE_NONE)
        nisa.tensor_tensor(tile_buf[...], tile_buf[...], gamma_sb[...], nl.multiply)
        nisa.dma_copy(dst=output_hbm.slice(dim=0, start=t_start, end=t_start + tile_size),
                      src=tile_buf, dge_mode=_DGE_MODE_NONE)
        src_3d = tile_buf.reshape_dim(dim=1, shape=[h0, h1])
        dst_3d = output_sb.slice(dim=1, start=sb_t_start, end=sb_t_start + tile_size)
        pe_transpose(src=src_3d, dst=dst_3d, tile_size=h0, dtype=scaled_hbm.dtype, sbm=sbm)

    sbm.pop_heap()  # gamma_sb
    sbm.pop_heap()  # tile_buf
    nisa.core_barrier(output_hbm, (0, 1))

    other_offset = (1 - shard_id) * t_shard
    nisa.sendrecv(
        dst=output_sb.slice(dim=1, start=other_offset, end=other_offset + t_shard),
        src=output_sb.slice(dim=1, start=sb_shard_offset, end=sb_shard_offset + t_shard),
        send_to_rank=1 - shard_id,
        recv_from_rank=1 - shard_id,
        pipe_id=0,
    )
    sbm.close_scope()


@nki.jit
def noaux_tc_router_prefill_kernel(
    scaled,
    gamma,
    router_weights,
    correction_bias,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
):
    """The gain, the router GEMM and the ``noaux_tc`` stage on ``[T, H]`` scaled rows.

    ``router._noaux_tc_rmsnorm_router_topk_nki`` with its norm stage started after the
    ``rstd`` multiply (:func:`_gamma_stage_dloc`); the router call and the ``noaux_tc``
    stage are the fused kernel's, argument for argument.
    """
    t_extent, h_extent = scaled.shape
    _, e_extent = router_weights.shape

    # One shard decision for the router and the `noaux_tc` stage, as in the fused kernel.
    shard_on_tokens = t_extent > 1

    router_logits = nl.ndarray((t_extent, e_extent), dtype=nl.float32,
                               buffer=nl.shared_hbm)
    norm_output = nl.ndarray((t_extent, h_extent), dtype=scaled.dtype,
                             buffer=nl.shared_hbm)
    # The nkilib router's own uncorrected outputs: written, not returned.
    substrate_index = nl.ndarray((t_extent, _router_seam.NOAUX_TC_MAX8_WIDTH),
                                 dtype=nl.int32, buffer=nl.shared_hbm)
    substrate_affinities = nl.ndarray((t_extent, e_extent), dtype=nl.bfloat16,
                                      buffer=nl.shared_hbm)
    # The corrected outputs.
    expert_index = nl.ndarray((t_extent, _router_seam.NOAUX_TC_MAX8_WIDTH),
                              dtype=nl.uint32, buffer=nl.shared_hbm)
    expert_affinities = nl.ndarray((t_extent, e_extent), dtype=nl.float32,
                                   buffer=nl.shared_hbm)

    norm_sb = nl.ndarray((nl.tile_size.pmax, t_extent, h_extent // nl.tile_size.pmax),
                         dtype=scaled.dtype, buffer=nl.sbuf)
    _gamma_stage_dloc(scaled, gamma, norm_output, norm_sb)

    # `w_bias=None`: the correction bias is not a projection bias; it enters the
    # selection score after the sigmoid, in the stage below.
    _substrate_router_topk(
        x=norm_sb,
        w=router_weights,
        w_bias=None,
        router_logits=router_logits,
        expert_affinities=substrate_affinities,
        expert_index=substrate_index,
        act_fn=RouterActFnType.SIGMOID,
        k=_router_seam.NOAUX_TC_MAX8_WIDTH,
        x_hbm_layout=_X_HBM_LAYOUT_FUSED,
        x_sb_layout=XSBLayout_tp102__0,
        router_pre_norm=False,
        norm_topk_prob=False,
        use_column_tiling=True,
        use_indirect_dma_scatter=True,
        use_PE_broadcast_w_bias=True,
        shard_on_tokens=shard_on_tokens,
        skip_store_expert_index=False,
        skip_store_router_logits=False,
    )

    _router_seam._noaux_tc_stage(
        router_logits_hbm=router_logits,
        correction_bias_hbm=correction_bias,
        expert_index_hbm=expert_index,
        expert_affinities_hbm=expert_affinities,
        num_tokens=t_extent,
        num_experts=e_extent,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        shard_on_tokens=shard_on_tokens,
    )

    return router_logits, expert_index, expert_affinities


def noaux_tc_router_prefill(
    hidden_states: Tensor,
    gamma: Tensor,
    router_weights: Tensor,
    correction_bias: Tensor,
    top_k: int = _router_seam.NOAUX_TC_MAX8_WIDTH,
    eps: float = 1e-6,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Route ``[T, H]`` (or ``[B, S, H]``) pre-norm prefill rows with ``noaux_tc``.

    The same contract as ``Glm5NextRoutedExperts.route_tokens``: ``(router_logits
    [T, E] fp32, expert_index [T, K] int32, expert_affinities [T, E] fp32)``, the
    affinities scattered (gate weight at each selected expert, zero elsewhere).

    A call :func:`prefill_route_admits` declines takes 8aa22fa's route
    (``router.noaux_tc_rmsnorm_router_topk``), which has its own torch fallback.

    Raises:
        NoauxTcRouterError: for a ``top_k`` or ``E`` the noaux_tc stage cannot serve,
            or a correction bias that is not ``[E]`` / ``[1, E]``.
    """
    hidden = int(hidden_states.shape[-1])
    tokens = hidden_states.numel() // hidden
    rows = hidden_states.reshape(tokens, hidden)
    num_experts = int(router_weights.shape[1])
    _router_seam._require_noaux_tc_extents(num_experts, int(top_k))
    bias = _router_seam._legalize_correction_bias(correction_bias, num_experts)

    if not prefill_route_admits(rows, router_weights, int(top_k)):
        logits, index, affinities, _substrate = _router_seam.noaux_tc_rmsnorm_router_topk(
            hidden_states=rows.reshape(1, tokens, hidden),
            gamma=gamma,
            router_weights=router_weights,
            correction_bias=bias,
            top_k=int(top_k),
            eps=eps,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
        )
        return logits, index, affinities

    scaled = router_rms_scale(rows, eps)
    # Whole 128-row tiles per core under the [2] launch. The pad repeats the last
    # real row (router._noaux_tc_pad_tokens), as 8aa22fa pads its pre-norm rows: the
    # norm is per row, so this is the same padded tensor scaled.
    t_pad = _router_seam._noaux_tc_pad_target(tokens, _router_seam._NOAUX_TC_T_MULTIPLE)
    padded = _router_seam._noaux_tc_pad_tokens(scaled, t_pad)
    # [1, H], as the fused kernel's wrapper legalises it for nkilib.
    gain = gamma.reshape(1, hidden)

    _router_seam._count_nki_dispatch()
    logits, index, affinities = wrap_nki(noaux_tc_router_prefill_kernel)[2](
        scaled=padded,
        gamma=gain,
        router_weights=router_weights,
        correction_bias=bias,
        norm_topk_prob=bool(norm_topk_prob),
        routed_scaling_factor=float(routed_scaling_factor),
    )
    return (
        logits[:tokens],
        index[:tokens].to(torch.int32),
        affinities[:tokens],
    )
