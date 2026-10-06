# SPDX-License-Identifier: Apache-2.0
"""Decode router: RMSNorm, router GEMM and ``noaux_tc`` top-8 in one NKI launch.

The prefill router (``router.noaux_tc_rmsnorm_router_topk``) puts one token per
partition and pads the token axis to 256 rows, so at decode it normalises 255 pad
rows, walks 4096 elements on one lane per norm instruction, runs the nkilib
router's own unused top-8 with its scatter, and round-trips the logits through
HBM. This kernel serves ``T <= 128`` real tokens with no pad:

* RMSNorm with the hidden axis on all 128 partitions, ``h = p * (H / 128) + j``
  (the nkilib ``tp102`` layout the old router matmul consumed). The squares are
  reduced per partition, summed across partitions by one fp32 ones-matmul, and
  the two bf16 roundings of the old norm (``x * rstd`` then ``* gamma``) are kept.
* Router GEMM: stationary = the normalised ``[128, T]`` column for each ``j``,
  moving = router weights ``[128, E]``, accumulated in PSUM in ``j`` order, as
  the old nkilib router did.
* ``noaux_tc`` on the ``[T, E]`` logits straight from PSUM: fp32 sigmoid, plus
  the correction bias, ``max8`` + ``nc_find_index8`` (the old instructions), the
  selected scores masked by ``nc_match_replace8`` (one instruction instead of an
  8-way one-hot loop), L1 renormalisation and ``routed_scaling_factor``.

One program (one physical core): the whole kernel is ~33 small matmuls and ~20
vector ops on a few KB, so a second core would add a cross-core exchange
(``sendrecv``) of the logits for at most the 288-column streaming time of half
the matmuls.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

# The decode router is the noaux_tc router seam's decode route, so it reports
# into that seam's counter family (``router.noaux_tc_dispatch_counters``) rather
# than opening a family no route check reads.
from . import router as _router_seam

#: Largest token count one launch serves: one token per partition in the noaux stage.
ROUTER_DECODE_MAX_TOKENS = 128

#: ``nisa.max8`` emits exactly 8 values per partition.
ROUTER_DECODE_K = 8

#: Guard term in the L1 denominator, as in ``router.NOAUX_TC_DENOM_EPS``.
_DENOM_EPS = 1e-20

#: Written by ``nc_match_replace8`` over the selected scores. A corrected score is
#: ``sigmoid + bias`` and is never this value.
_SELECTED = -3.0e38

#: The nkilib router's moving free-dim cap, which bounds E for one PSUM bank.
_MAX_EXPERTS = 512


@nki.jit
def noaux_router_decode_kernel(
    hidden,
    gamma,
    router_weights,
    correction_bias,
    eps: float = 1e-6,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
):
    """``[T, H]`` pre-norm activations -> logits, top-8 indices, scattered weights.

    Args:
        hidden: ``[T, H]`` bf16, ``1 <= T <= 128``, ``H % 128 == 0``.
        gamma: ``[1, H]`` RMSNorm gain.
        router_weights: ``[H, E]`` bf16, ``8 <= E <= 512``.
        correction_bias: ``[1, E]`` fp32 ``e_score_correction_bias``.

    Returns:
        ``(logits [T, E] fp32, expert_index [T, 8] int32, affinities [T, E] fp32)``.
    """
    tokens, hidden_size = hidden.shape
    experts = router_weights.shape[1]
    kernel_assert(1 <= tokens <= ROUTER_DECODE_MAX_TOKENS, "1 <= T <= 128")
    kernel_assert(hidden_size % 128 == 0, "H must be a multiple of 128")
    kernel_assert(ROUTER_DECODE_K <= experts <= _MAX_EXPERTS, "8 <= E <= 512")
    kernel_assert(router_weights.shape[0] == hidden_size, "router weights are [H, E]")
    cols = hidden_size // 128

    logits_hbm = nl.ndarray((tokens, experts), dtype=nl.float32, buffer=nl.shared_hbm)
    index_hbm = nl.ndarray((tokens, ROUTER_DECODE_K), dtype=nl.int32,
                           buffer=nl.shared_hbm)
    aff_hbm = nl.ndarray((tokens, experts), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- Loads: four independent DMAs. ----------------------------------- #
    # Partition p holds hidden elements p*cols .. p*cols+cols-1 of every token.
    x = nl.ndarray((128, tokens, cols), dtype=hidden.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=hidden.ap(
        pattern=[[cols, 128], [hidden_size, tokens], [1, cols]]))
    g = nl.ndarray((128, cols), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=g, src=gamma.ap(pattern=[[cols, 128], [1, cols]]))
    # Rows p*cols .. p*cols+cols-1 are one contiguous run per partition.
    w = nl.ndarray((128, cols, experts), dtype=router_weights.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=w.reshape((128, cols * experts)), src=router_weights.ap(
        pattern=[[cols * experts, 128], [1, cols * experts]]))
    bias = nl.ndarray((tokens, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias, src=correction_bias.ap(pattern=[[0, tokens], [1, experts]]))

    # ---- RMSNorm over the 128-partition layout. ---------------------------- #
    sq = nl.ndarray((128, tokens, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=x, data2=x, op=nl.multiply)
    part = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=part, data=sq, op=nl.add, axis=(2,))
    ones = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    total = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.psum)
    # Every output partition receives the full sum of squares of each token.
    nisa.nc_matmul(dst=total, stationary=ones, moving=part)
    rstd = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=total, scale=1.0 / hidden_size,
                    bias=eps)
    # Same two roundings as the old norm: bf16(x * rstd), then bf16(that * gamma).
    scaled = nl.ndarray((128, tokens, cols), dtype=hidden.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=scaled, data1=x, data2=rstd.ap(
        pattern=[[tokens, 128], [1, tokens], [0, cols]]), op=nl.multiply)
    normed = nl.ndarray((128, tokens, cols), dtype=router_weights.dtype,
                        buffer=nl.sbuf)
    nisa.tensor_tensor(dst=normed, data1=scaled, data2=g.ap(
        pattern=[[cols, 128], [0, tokens], [1, cols]]), op=nl.multiply)

    # ---- Router GEMM: [T, E] in one PSUM bank, accumulated in j order. ------ #
    logits_ps = nl.ndarray((tokens, experts), dtype=nl.float32, buffer=nl.psum)
    for j in range(cols):
        nisa.nc_matmul(dst=logits_ps, stationary=normed[:, :, j],
                       moving=w[:, j, :], accumulate=(j > 0))

    # ---- noaux_tc on T partitions. ----------------------------------------- #
    logits = nl.ndarray((tokens, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=logits, src=logits_ps)
    nisa.dma_copy(dst=logits_hbm, src=logits)
    scores = nl.ndarray((tokens, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=scores, op=nl.sigmoid, data=logits)
    choice = nl.ndarray((tokens, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=choice, data1=scores, data2=bias, op=nl.add)
    top8 = nl.ndarray((tokens, ROUTER_DECODE_K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8, src=choice)
    idx8 = nl.ndarray((tokens, ROUTER_DECODE_K), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=idx8, data=choice, vals=top8)
    index = nl.ndarray((tokens, ROUTER_DECODE_K), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=index, src=idx8)
    nisa.dma_copy(dst=index_hbm, src=index)

    # The first occurrence of each top-8 value -- the positions find_index8
    # reports -- becomes _SELECTED; the gate weight is the unbiased score there.
    marked = nl.ndarray((tokens, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_match_replace8(dst=marked, data=choice, vals=top8, imm=_SELECTED)
    selected = nl.ndarray((tokens, experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=selected, data=marked, op0=nl.equal,
                              operand0=_SELECTED, op1=nl.multiply, operand1=scores)
    out = nl.ndarray((tokens, experts), dtype=nl.float32, buffer=nl.sbuf)
    if norm_topk_prob:
        row_sum = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_sum, data=selected, op=nl.add, axis=(1,))
        denom = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=denom, data=row_sum, op0=nl.add, operand0=_DENOM_EPS)
        recip = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=recip, data=denom)
        nisa.tensor_scalar(dst=out, data=selected, op0=nl.multiply, operand0=recip,
                           op1=nl.multiply, operand1=float(routed_scaling_factor))
    else:
        nisa.tensor_scalar(dst=out, data=selected, op0=nl.multiply,
                           operand0=float(routed_scaling_factor))
    nisa.dma_copy(dst=aff_hbm, src=out)
    return logits_hbm, index_hbm, aff_hbm


#: The model call site's envelope: decode batches. Larger token counts (prefill)
#: keep the prefill router, whose layout serves them.
DECODE_ROUTE_MAX_TOKENS = 64


def decode_route_admits(hidden_states, router_weights, top_k: int) -> bool:
    """True when ``route_tokens`` takes this router instead of the prefill one."""
    if not isinstance(hidden_states, Tensor) or not isinstance(router_weights, Tensor):
        return False
    if hidden_states.dim() < 2:
        return False
    tokens = hidden_states.numel() // hidden_states.shape[-1]
    return tokens <= DECODE_ROUTE_MAX_TOKENS and can_run_router_decode(
        hidden_states, router_weights, top_k)


def can_run_router_decode(hidden_states: Tensor, router_weights: Tensor,
                          top_k: int) -> bool:
    """True when the decode kernel serves this call on an NKI device or simulator."""
    tokens = hidden_states.numel() // hidden_states.shape[-1]
    hidden_size = hidden_states.shape[-1]
    experts = router_weights.shape[-1]
    return (
        top_k == ROUTER_DECODE_K
        and 1 <= tokens <= ROUTER_DECODE_MAX_TOKENS
        and hidden_size % 128 == 0
        and tuple(router_weights.shape) == (hidden_size, experts)
        and ROUTER_DECODE_K <= experts <= _MAX_EXPERTS
        and router_weights.dtype == torch.bfloat16
        and can_run_kernel(hidden_states)
    )


def router_decode_torch_oracle(hidden_states, gamma, router_weights, correction_bias,
                               eps, norm_topk_prob, routed_scaling_factor):
    """Torch reference with the kernel's roundings; used when no device is present."""
    hidden_size = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, hidden_size)
    xf = x.to(torch.float32)
    rstd = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + eps)
    scaled = (xf * rstd).to(x.dtype).to(torch.float32)
    normed = (scaled * gamma.reshape(1, -1).to(torch.float32)).to(router_weights.dtype)
    logits = normed.to(torch.float32) @ router_weights.to(torch.float32)
    scores = logits.sigmoid()
    choice = scores + correction_bias.reshape(1, -1).to(torch.float32)
    index = torch.topk(choice, ROUTER_DECODE_K, dim=-1).indices
    weights = scores.gather(1, index)
    if norm_topk_prob:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + _DENOM_EPS)
    weights = weights * routed_scaling_factor
    affinities = torch.zeros_like(scores).scatter_(1, index, weights)
    return logits, index.to(torch.int32), affinities


def noaux_tc_router_decode(
    hidden_states: Tensor,
    gamma: Tensor,
    router_weights: Tensor,
    correction_bias: Tensor,
    top_k: int = ROUTER_DECODE_K,
    eps: float = 1e-6,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Decode-time ``noaux_tc`` routing of ``T <= 128`` tokens in one launch.

    Args:
        hidden_states: ``[T, H]`` or ``[B, S, H]`` pre-norm activations (bf16).
        gamma: ``[H]`` or ``[1, H]`` RMSNorm gain.
        router_weights: ``[H, E]`` bf16.
        correction_bias: ``[E]`` or ``[1, E]`` ``e_score_correction_bias``.

    Returns:
        ``(router_logits [T, E] fp32, expert_index [T, 8] int32,
        expert_affinities [T, E] fp32)``, the three tensors ``route_tokens``
        returns. ``expert_affinities`` is scattered: the gate weight at each
        selected expert's column, zero elsewhere.
    """
    hidden_size = hidden_states.shape[-1]
    experts = router_weights.shape[-1]
    x = hidden_states.reshape(-1, hidden_size)
    gamma = gamma.reshape(1, hidden_size)
    bias = correction_bias.reshape(1, experts).to(torch.float32)
    if not can_run_router_decode(x, router_weights, top_k):
        if top_k != ROUTER_DECODE_K:
            raise ValueError(f"top_k must be {ROUTER_DECODE_K}, got {top_k}")
        _router_seam._count_torch_fallback()
        return router_decode_torch_oracle(x, gamma, router_weights, bias, eps,
                                          norm_topk_prob, routed_scaling_factor)
    _router_seam._count_nki_dispatch()
    return wrap_nki(noaux_router_decode_kernel)(
        hidden=x.contiguous(),
        gamma=gamma.contiguous(),
        router_weights=router_weights.contiguous(),
        correction_bias=bias.contiguous(),
        eps=float(eps),
        norm_topk_prob=bool(norm_topk_prob),
        routed_scaling_factor=float(routed_scaling_factor),
    )
