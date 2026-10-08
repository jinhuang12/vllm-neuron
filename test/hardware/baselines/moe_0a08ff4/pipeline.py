# SPDX-License-Identifier: Apache-2.0
"""0a08ff4's decode entry point over the 0a08ff4 kernel copy next to this file.

The body is ``fused_fp8.fused_fp8_decode_experts`` as of 0a08ff4 with the seam's
dispatch counter removed. Operand preparation (fp32 ``[T, G, E]`` affinities,
``[1, 1]`` int32 rank, contiguous hidden) is unchanged.
"""

from __future__ import annotations

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from .expert_decode import (
    EXPERT_DECODE_MAX_TOKENS,
    can_run_expert_decode,
    default_programs,
    expert_decode_kernel,
    expert_decode_torch_oracle,
    geometry,
    rank_operand,
)

__all__ = ["decode_experts", "expert_decode_kernel", "expert_decode_torch_oracle"]

_KERNELS = {1: wrap_nki(expert_decode_kernel), 2: wrap_nki(expert_decode_kernel)[2]}


def decode_experts(hidden, expert_affinities, packed, bounds, expert_parallel_rank=0,
                   *, programs=None, out_dtype=None, weight_fp8=False):
    """``fused_fp8_decode_experts`` as of 0a08ff4 (see that docstring)."""
    out_dtype = hidden.dtype if out_dtype is None else out_dtype
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"out_dtype must be bf16 or fp32, got {out_dtype}")
    if tuple(bounds.shape) != (128, 3) or bounds.dtype != torch.float32:
        raise ValueError("bounds must be FP32 [128,3]")
    if not can_run_expert_decode(hidden, expert_affinities, packed):
        raise ValueError(
            f"0a08ff4 decode serves 1..{EXPERT_DECODE_MAX_TOKENS} bf16 tokens; got "
            f"hidden {tuple(hidden.shape)} {hidden.dtype}, affinities "
            f"{tuple(expert_affinities.shape)}")
    experts, ni, nh = geometry(packed)
    programs = default_programs(ni, nh) if programs is None else int(programs)
    if programs not in _KERNELS:
        raise ValueError(f"programs must be 1 or 2, got {programs}")
    return _KERNELS[programs](
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
