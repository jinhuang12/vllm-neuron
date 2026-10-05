# SPDX-License-Identifier: Apache-2.0
"""Prepared-weight interface for the fused GLM expert kernel.

The model's routed-expert forward uses this entry point. The caller supplies
device routing for one compiled row bucket and performs the usual FP32 combine
of the returned expert contributions.
"""

import os

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from .moe_fused_fp8 import moe_fused_fp8_kernel
from .moe_fused_fp8_decode import compact_decode_kernel
from .expert_decode import (
    EXPERT_DECODE_MAX_TOKENS,
    can_run_expert_decode,
    default_programs,
    expert_decode_kernel,
    geometry,
    rank_operand,
)

from .fused_fp8_pack import PackedExperts, pack_experts
from .fused_fp8_config import select_tiles

__all__ = ["PackedExperts", "pack_experts", "fused_fp8_experts",
           "fused_fp8_decode_experts"]

_FUSED_EXPERTS = wrap_nki(moe_fused_fp8_kernel)[2]
_FUSED_DECODE_EXPERTS = wrap_nki(compact_decode_kernel)[2]
_TOKEN_DECODE_EXPERTS = {1: wrap_nki(expert_decode_kernel),
                         2: wrap_nki(expert_decode_kernel)[2]}
_DISPATCH_COUNT = 0


def reset_fused_dispatch_counters():
    """Reset the wrapper-entry counter, matching other MoE seams' test hooks."""
    global _DISPATCH_COUNT
    _DISPATCH_COUNT = 0


def fused_dispatch_counters():
    """Return (NKI entries, torch fallbacks); this seam has no fallback."""
    return _DISPATCH_COUNT, 0


@torch._dynamo.assume_constant_result
def _count_nki_dispatch():
    """Count eager/capture entries without adding a changing Dynamo guard."""
    global _DISPATCH_COUNT
    _DISPATCH_COUNT += 1


def fused_fp8_experts(hidden, packed, row_ids, expert_ids, affinity, bounds,
                      *, block_m=None, block_n=None, block_k=None):
    """Compute routed expert contributions in the caller's block order.

    ``row_ids`` has shape [blocks,q], with real token IDs or -1. ``expert_ids``
    has shape [blocks,1], and values in [0,E). Hidden states include a final
    zero padding row. Affinities use token-major [T+1,E] layout with the final
    row zero. ``bounds`` is FP32 [128,3], repeating gate upper, up lower, and
    up upper limits. Use infinities for an unbounded side.

    The returned FP32 [blocks*q,H] tensor includes zero padding rows.
    BLOCK_M/N/K specialize the compiled schedule; H/I and q come from shapes.
    BLOCK_N/K group 128-channel tiles without merging quantization scales.
    Routing values stay on device. Pack model-prepared CPU weights and their
    matching compensated scales once, then transfer the packed bank to device.
    """
    if hidden.ndim != 2 or hidden.shape[0] < 1 or hidden.shape[1] < 1 or hidden.shape[1] % 128 or hidden.dtype != torch.bfloat16:
        raise ValueError("hidden must be BF16 [T+1,H], positive H multiple of128")
    if row_ids.ndim != 2 or row_ids.shape[0] < 1 or row_ids.shape[1] < 1:
        raise ValueError("row_ids must have positive shape [blocks,q]")
    blocks, _ = row_ids.shape
    if packed.weights.ndim != 5 or packed.weights.shape[0] < 1:
        raise ValueError("weights must be a nonempty packed expert bank")
    experts, panels, contraction, nh, channels = packed.weights.shape
    if panels < 3 or panels % 3 or contraction != 128 or channels != 128 or nh * 128 != hidden.shape[1]:
        raise ValueError("Invalid packed H/I geometry")
    intermediate = (panels // 3) * 128
    tile_m, tile_n, tile_k = select_tiles(hidden.shape[1], intermediate, row_ids.shape[1],
                                         block_m, block_n, block_k)
    requirements = (
        (packed.weights, (experts, panels, 128, nh, 128), torch.float8_e4m3fn, "weights"),
        (packed.scales, (experts, panels, nh), torch.float32, "scales"),
        (expert_ids, (blocks, 1), torch.int32, "expert_ids"),
        (affinity, (hidden.shape[0], experts), torch.float32, "affinity"),
        (bounds, (128, 3), torch.float32, "bounds"),
    )
    for tensor, shape, dtype, name in requirements:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype} with shape {shape}")
    if row_ids.dtype != torch.int32:
        raise ValueError("row_ids must use int32")
    for tensor in (hidden, packed.weights, packed.scales, row_ids, expert_ids, affinity, bounds):
        if not tensor.is_contiguous() or tensor.device != hidden.device:
            raise ValueError("All operands must be contiguous and on the same device")
    _count_nki_dispatch()
    if (
        os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2"
        and tuple(hidden.shape) == (2, 4096)
        and tuple(row_ids.shape) == (8, 1)
        and experts == 18
        and intermediate == 512
        and (tile_m, tile_n, tile_k) == (1, 4096, 4096)
    ):
        return _FUSED_DECODE_EXPERTS(
            hidden, packed.weights, packed.scales, row_ids, expert_ids,
            affinity.reshape(-1, 1), bounds,
            BLOCK_N=tile_n, BLOCK_K=tile_k,
        ).reshape(-1, hidden.shape[1])
    return _FUSED_EXPERTS(
        hidden, packed.weights, packed.scales, row_ids, expert_ids,
        affinity.reshape(-1, 1), bounds,
        BLOCK_M=tile_m, BLOCK_N=tile_n, BLOCK_K=tile_k,
    ).reshape(-1, hidden.shape[1])


def fused_fp8_decode_experts(hidden, expert_affinities, packed, bounds,
                             expert_parallel_rank=0, *, programs=None,
                             out_dtype=None, weight_fp8=False):
    """This rank's routed-expert output for ``T <= 64`` decode tokens, one launch.

    The decode route of this seam: no mapping, no padding row, no combine. The
    kernel reads the router's global scattered ``[T, E_global]`` affinities and
    the rank's group id itself, visits each distinct local expert with a routed
    token once, and returns the fp32-accumulated sum in ``out_dtype``.

    Args:
        hidden: ``[T, H]`` bf16 expert inputs, real tokens only.
        expert_affinities: ``[T, E_global]`` fp32, as ``route_tokens`` returns;
            ``E_global`` is a multiple of the bank's expert count.
        packed: this rank's ``PackedExperts``.
        bounds: ``[128, 3]`` fp32 SwiGLU bounds, as for ``fused_fp8_experts``.
        expert_parallel_rank: the bank's group, an int or a one-element int
            tensor (the runner's device operand).
        programs: 1 or 2; default 2 under ``NEURON_LOGICAL_NC_CONFIG=2``.
        out_dtype: ``torch.bfloat16`` (default: ``hidden.dtype``) or fp32.
        weight_fp8: feed fp8 tiles to the PE instead of a bf16 DMA cast.

    Raises:
        ValueError: outside the kernel's envelope (see ``can_run_expert_decode``);
            like ``fused_fp8_experts`` this seam has no torch fallback.
    """
    out_dtype = hidden.dtype if out_dtype is None else out_dtype
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"out_dtype must be bf16 or fp32, got {out_dtype}")
    if tuple(bounds.shape) != (128, 3) or bounds.dtype != torch.float32:
        raise ValueError("bounds must be FP32 [128,3]")
    if not can_run_expert_decode(hidden, expert_affinities, packed):
        raise ValueError(
            f"fused_fp8_decode_experts serves 1..{EXPERT_DECODE_MAX_TOKENS} bf16 "
            f"tokens of a packed bank on an NKI device or simulator; got hidden "
            f"{tuple(hidden.shape)} {hidden.dtype}, affinities "
            f"{tuple(expert_affinities.shape)}")
    experts, ni, nh = geometry(packed)
    programs = default_programs(ni, nh) if programs is None else int(programs)
    if programs not in _TOKEN_DECODE_EXPERTS:
        raise ValueError(f"programs must be 1 or 2, got {programs}")
    _count_nki_dispatch()
    return _TOKEN_DECODE_EXPERTS[programs](
        hidden=hidden.contiguous(),
        affinity=expert_affinities.to(torch.float32).reshape(
            hidden.shape[0], -1, experts).contiguous(),
        rank=rank_operand(expert_parallel_rank, hidden.device),
        weights=packed.weights,
        scales=packed.scales,
        bounds=bounds.contiguous(),
        OUT_FP32=out_dtype == torch.float32,
        WEIGHT_FP8=bool(weight_fp8),
    )
