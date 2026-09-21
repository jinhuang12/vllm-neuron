# SPDX-License-Identifier: Apache-2.0
"""Prepared-weight interface for the fused GLM expert kernel.

The model's routed-expert forward uses this entry point. The caller supplies
device routing for one compiled row bucket and performs the usual FP32 combine
of the returned expert contributions.
"""

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from .moe_fused_fp8 import moe_fused_fp8_kernel

from .fused_fp8_pack import PackedExperts, pack_experts
from .fused_fp8_config import select_tiles

__all__ = ["PackedExperts", "pack_experts", "fused_fp8_experts"]

_FUSED_EXPERTS = wrap_nki(moe_fused_fp8_kernel)[2]
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
    return _FUSED_EXPERTS(
        hidden, packed.weights, packed.scales, row_ids, expert_ids,
        affinity.reshape(-1, 1), bounds,
        BLOCK_M=tile_m, BLOCK_N=tile_n, BLOCK_K=tile_k,
    ).reshape(-1, hidden.shape[1])
