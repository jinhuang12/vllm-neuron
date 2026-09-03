# SPDX-License-Identifier: Apache-2.0
"""Plain-torch numerics reference for LD-11 ``NF.block_fp8_linear``.

This module exists to satisfy the c13 comparison contract: the per-triad
harness (``scripts/triad_numerics.py`` leg 1) grades the triad's torch
fallback against a PLAIN-TORCH reference, and it REFUSES any reference that
resolves inside the port package under test -- "a reference taken from the
branch under test is not plain torch, and a reference that IS the fallback
would grade it against itself".

It therefore lives OUTSIDE ``vllm_neuron/`` (top-level ``numerics/``
namespace package), imports nothing from the port, and is never imported by
port code. It is not part of the served graph and never runs on a
NeuronCore.

WHAT IT IMPLEMENTS -- port-plan.md iteration 5 §1(b), assessment §5 LD-11 row,
transcribed clause by clause:

    quantize ``x`` per row per 128-element K group to fp8 with a ue8m0
    (power-of-two) group scale ``s = 2**ceil(log2(amax / FP8_MAX))``, amax
    floored at 1e-4 and the code clamped to +-FP8_MAX; multiply the fp8
    operands blockwise; accumulate in fp32; scale each 128x128 output-block
    partial product by ``a_scale * w_scale``; cast to bf16 last.

Deliberately written in the SCALED-PARTIAL form -- integer-grid fp8 codes
multiplied, the fp32 partial then scaled -- because the triad's fallback is
written in the equivalent DEQUANTIZE-THEN-MATMUL form. Two spellings of the
declared math, so leg 1 compares two independent derivations instead of one
derivation against a paraphrase of itself. The two agree exactly in real
arithmetic (every scale is an exact power of two, so the scale multiply is
exact) and differ only in fp32 rounding of the summation, which is what the
declared 2e-05 fp32 clause bounds.

TWO CONSTANTS THIS MODULE PINS, AND WHY THEY ARE PINNED HERE

``fp8_max`` DEFAULTS TO 240.0, NOT THE REFERENCE IMPLEMENTATION'S 448.
``vllm_neuron.utils.dtype_utils.FP8_CLAMP_MAX`` is the fork's own
platform-resolved ceiling (240.0 on trn2, 448.0 on trn3), and the triad
imports exactly that. This module may not import it -- importing the port
package here is what the harness refuses -- so the venue's value is written
as a literal and the declaration pins the venue: every graded leg runs on
trn2 with ``NEURON_PLATFORM_TARGET_OVERRIDE=trn2`` exported BEFORE the first
import (R-24; the fork freezes ``FP8_CLAMP_MAX`` at import time and falls
back to 448.0 in bare CPU mode). If this reference said 448 while the triad
resolved 240, leg 1 would fail loudly rather than silently agree -- the
failure mode is safe, but the pin is stated so no reader has to derive it.

The fp8 CARRIER dtype is ``torch.float8_e4m3fn``. On this venue the bytes
inside that carrier hold LEGACY ``nl.float8_e4m3`` values (amax 240),
re-encoded at load by LD-24. Plain torch decodes the carrier as OCP
``float8_e4m3fn``, and that decode is EXACTLY right for these bytes: after
LD-24 every byte sits at exponent field <= 14, where the two grids agree
element for element (assessment §2.4(3)). That equality is the whole reason
a CPU-side reference is meaningful for this op at all; it was not true
before LD-24, and the iteration-1 ``plan_defect`` is the record of that.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

#: The venue's fp8 ceiling. See the module docstring: this mirrors
#: ``vllm_neuron.utils.dtype_utils.FP8_CLAMP_MAX`` on trn2 and may not import
#: it. Legacy ``nl.float8_e4m3`` amax, never OCP's 448.
LEGACY_FP8_E4M3_MAX = 240.0

#: The reference implementation's own amax floor (``dsv4_ref/kernel.py:47-49``).
AMAX_FLOOR = 1e-4


def _pow2_ceil_scale(amax: Tensor, fp8_max: float) -> Tensor:
    """``s = 2 ** ceil(log2(amax / fp8_max))`` via the IEEE-754 bit form.

    The bit form, not ``exp2(ceil(log2(v)))``: at an exact power of two fp32
    ``log2`` can land a hair above or below the integer and flip the ceiling,
    which is one whole exponent of quantization error. The reference
    implementation computes it bitwise for the same reason
    (``dsv4_ref/kernel.py:22-37``)::

        exp = (bits >> 23) & 0xFF ; man = bits & 0x7FFFFF
        ceil_log2 = exp - 127 + (1 if man else 0)
        pow2      = reinterpret_f32((ceil_log2 + 127) << 23)
    """
    v = (amax.to(torch.float32) * (1.0 / fp8_max)).contiguous()
    bits = v.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    ceil_log2 = exponent - 127 + (mantissa != 0).to(torch.int32)
    return ((ceil_log2 + 127) << 23).view(torch.float32)


def block_fp8_linear_reference(
    x: Tensor,
    weight_fp8: Tensor,
    weight_scale: Tensor,
    *,
    block_size=(128, 128),
    act_group_size: int = 128,
    accum_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.bfloat16,
    bias=None,
    fp8_max: float = LEGACY_FP8_E4M3_MAX,
    amax_floor: float = AMAX_FLOOR,
) -> Tensor:
    """``out[M, N] = dequant(quant(x)) @ dequant(weight_fp8).T`` per the contract.

    Signature matches the plan-fixed call form of ``NF.block_fp8_linear``
    (3 positionals + 5 keywords) so the harness can call reference and op with
    one argument list. ``block_size`` arrives from JSON as a list, so it is
    coerced rather than assumed to be a tuple.
    """
    block_n, block_k = (int(v) for v in tuple(block_size))
    group = int(act_group_size)
    if group != block_k:
        raise ValueError(
            f"the declared contract fixes act_group_size == block_k; got "
            f"{group} and {block_k}. With unequal values a 128x128 output-block "
            f"partial no longer has ONE activation scale per row and the "
            f"'scale each partial by a_scale * w_scale' clause is undefined."
        )
    if x.dim() != 2 or weight_fp8.dim() != 2:
        raise ValueError(
            f"reference is 2-D only (the op is: wo_a's bmm degenerates to "
            f"batch 1 per core); got x {tuple(x.shape)}, weight "
            f"{tuple(weight_fp8.shape)}"
        )
    m, k = x.shape
    n, k_w = weight_fp8.shape
    if k != k_w:
        raise ValueError(
            f"contraction mismatch: x is [{m}, {k}], weight is [{n}, {k_w}]"
        )
    if k % group != 0:
        raise ValueError(
            f"K={k} must be a whole number of {group}-element activation "
            f"groups (K_local % 128 == 0 is a hard invariant of this port)"
        )

    n_blocks = math.ceil(n / block_n)
    k_blocks = math.ceil(k / block_k)
    if tuple(weight_scale.shape) != (n_blocks, k_blocks):
        raise ValueError(
            f"weight_scale must be [ceil(N/{block_n}), ceil(K/{block_k})] = "
            f"[{n_blocks}, {k_blocks}]; got {tuple(weight_scale.shape)}"
        )

    # ---- activations: dynamic per-row, per-K-group fp8 quantization --------
    grouped = x.to(torch.float32).reshape(m, k // group, group)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=amax_floor)
    a_scale = _pow2_ceil_scale(amax, fp8_max)                # [M, K/group, 1]
    a_codes = (
        (grouped / a_scale)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
        .reshape(m, k)
    )                                                        # integer fp8 grid
    a_scale = a_scale.reshape(m, k // group)                  # [M, K/group]

    # ---- weights: the carrier's bytes, decoded, still on the fp8 grid -----
    w_codes = weight_fp8.to(torch.float32)                    # [N, K]
    w_scale = weight_scale.to(torch.float32)                  # [N/128, K/128]

    # ---- blockwise fp8 multiply, fp32 accumulation, partials scaled -------
    out = torch.zeros((m, n), dtype=torch.float32)
    for nb in range(n_blocks):
        n0, n1 = nb * block_n, min((nb + 1) * block_n, n)
        for kb in range(k_blocks):
            k0, k1 = kb * block_k, min((kb + 1) * block_k, k)
            partial = a_codes[:, k0:k1] @ w_codes[n0:n1, k0:k1].t()
            out[:, n0:n1] += (
                partial * a_scale[:, kb : kb + 1] * w_scale[nb, kb]
            )

    out = out.to(accum_dtype)
    if bias is not None:
        out = out + bias.to(accum_dtype)
    return out.to(out_dtype)
