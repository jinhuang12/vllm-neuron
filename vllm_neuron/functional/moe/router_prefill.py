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

What this changes. The router's RMSNorm moves out of the kernel into torch
(:func:`router_norm`), next to the experts' own norm (``Glm5NextModel._rms_norm``),
which already reads the same pre-norm tensor. The collapse then feeds only XLA
elementwise consumers, the reduce fuses, and the router kernel
(:func:`noaux_tc_router_prefill_kernel`) reads the normalised rows instead. The
kernel is the fused kernel without its first stage: the same nkilib ``router_topk``
call with the same arguments and the same ``noaux_tc`` stage
(``router._noaux_tc_stage``), so given the same normalised bf16 rows it computes
the same logits, indices and gate weights.

Numerics. :func:`router_norm` keeps the nkilib RMSNorm's two bf16 roundings,
``bf16(bf16(x * rstd) * gamma)`` (``rmsnorm_tkg._rmsnorm_tkg_dloc`` writes ``x * rstd``
back into its bf16 input tile before the gamma multiply), and its scale
``rstd = rsqrt(sum(x**2) * (1/H) + eps)``. What moves is the fp32 summation order of
``sum(x**2)`` and the ``rsqrt`` implementation, so a row's ``rstd`` can differ in its
last bit and an element of the normalised row by one bf16 step where ``x * rstd`` sits
on a rounding boundary. The tests bound the effect on the selection.

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
import nki.language as nl
from nkilib.core.router_topk.router_topk import XSBLayout_tp102__0
from nkilib.core.router_topk.router_topk import router_topk as _substrate_router_topk
from nkilib.core.utils.common_types import RouterActFnType

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

from . import router as _router_seam
from .router_decode import DECODE_ROUTE_MAX_TOKENS

#: ``0`` keeps 8aa22fa's prefill route (the fused kernel on the pre-norm rows).
PREFILL_ROUTER_ENV = "VLLM_NEURON_MOE_PREFILL_ROUTER"

#: nkilib ``router_topk``'s ``XHBMLayout_T_H__1``: ``x`` in HBM as ``[T, H]``.
_X_HBM_LAYOUT_T_H = 1

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
        and int(top_k) == _router_seam.NOAUX_TC_K
        and hidden_states.dtype == torch.bfloat16
        and router_weights.dtype == torch.bfloat16
        and hidden % _H_BLOCK == 0
        and tuple(router_weights.shape) == (hidden, experts)
        and _router_seam.NOAUX_TC_K <= experts <= _router_seam._NOAUX_TC_F_MAX
        and can_run_kernel(hidden_states)
    )


def router_norm(hidden_states: Tensor, gamma: Tensor, eps: float) -> Tensor:
    """The router's RMSNorm, ``bf16(bf16(x * rstd) * gamma)``, as nkilib rounds it.

    Args:
        hidden_states: ``[T, H]`` bf16 pre-norm activations.
        gamma: ``[H]`` or ``[1, H]`` the FFN norm's gain.
        eps: the RMSNorm epsilon.

    Returns:
        ``[T, H]`` bf16, the rows the router GEMM consumes.
    """
    hidden = int(hidden_states.shape[-1])
    x = hidden_states.to(torch.float32)
    rstd = torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) * (1.0 / hidden) + eps)
    scaled = (x * rstd).to(hidden_states.dtype)
    gain = gamma.reshape(1, hidden).to(torch.float32)
    return (scaled.to(torch.float32) * gain).to(torch.bfloat16)


@nki.jit
def noaux_tc_router_prefill_kernel(
    normed,
    router_weights,
    correction_bias,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
):
    """Router GEMM and the ``noaux_tc`` stage on ``[T, H]`` normalised rows.

    ``router._noaux_tc_rmsnorm_router_topk_nki`` minus its RMSNorm stage: nkilib's
    router loads ``normed`` from HBM into the ``[128, T, H/128]`` SBUF layout the norm
    stage wrote there (``XSBLayout_tp102__0``), then the call and the stage are the
    fused kernel's, argument for argument.
    """
    t_extent, _h_extent = normed.shape
    _, e_extent = router_weights.shape

    # One shard decision for the router and the `noaux_tc` stage, as in the fused kernel.
    shard_on_tokens = t_extent > 1

    router_logits = nl.ndarray((t_extent, e_extent), dtype=nl.float32,
                               buffer=nl.shared_hbm)
    # The nkilib router's own uncorrected outputs: written, not returned.
    substrate_index = nl.ndarray((t_extent, _router_seam.NOAUX_TC_K), dtype=nl.int32,
                                 buffer=nl.shared_hbm)
    substrate_affinities = nl.ndarray((t_extent, e_extent), dtype=nl.bfloat16,
                                      buffer=nl.shared_hbm)
    # The corrected outputs.
    expert_index = nl.ndarray((t_extent, _router_seam.NOAUX_TC_K), dtype=nl.uint32,
                              buffer=nl.shared_hbm)
    expert_affinities = nl.ndarray((t_extent, e_extent), dtype=nl.float32,
                                   buffer=nl.shared_hbm)

    # `w_bias=None`: the correction bias is not a projection bias; it enters the
    # selection score after the sigmoid, in the stage below.
    _substrate_router_topk(
        x=normed,
        w=router_weights,
        w_bias=None,
        router_logits=router_logits,
        expert_affinities=substrate_affinities,
        expert_index=substrate_index,
        act_fn=RouterActFnType.SIGMOID,
        k=_router_seam.NOAUX_TC_K,
        x_hbm_layout=_X_HBM_LAYOUT_T_H,
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
    top_k: int = _router_seam.NOAUX_TC_K,
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

    normed = router_norm(rows, gamma, eps)
    # Whole 128-row tiles per core under the [2] launch. The pad repeats the last
    # real row (router._noaux_tc_pad_tokens), as 8aa22fa pads its pre-norm rows: the
    # norm is per row, so this is the same padded tensor normalised.
    t_pad = _router_seam._noaux_tc_pad_target(tokens, _router_seam._NOAUX_TC_T_MULTIPLE)
    padded = _router_seam._noaux_tc_pad_tokens(normed, t_pad)

    _router_seam._count_nki_dispatch()
    logits, index, affinities = wrap_nki(noaux_tc_router_prefill_kernel)[2](
        normed=padded,
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
