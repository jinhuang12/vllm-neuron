# SPDX-License-Identifier: Apache-2.0
"""Dense blockwise-fp8 GEMM: a SCRATCH NKI kernel, authored here.

`inc-glm53f-026`. This module is the dense half of the campaign's block-quant
path -- the projections outside the MoE expert banks. It is **kernel-class**
under P13, and unlike its MoE sibling it is **SCRATCH rather than ADAPT**: G1
found no blockwise member in the substrate's ``QuantizationType`` (static and
row only at gen3), so a blockwise dense matmul is functionality ``nkilib`` does
not provide at any granularity. There is no vendor kernel to wrap, so the
arithmetic below is authored in NKI. The torch code in this module is the CPU
oracle and the constraint-violation fallback -- the two roles the plan's
substrate register admits (``design/increment-plan.md`` §4) -- and never the
shipped implementation.

What the kernel computes
------------------------
``out[M, N] = x[M, K] @ dequantise(weight[K, N])``, where ``weight`` is fp8-e4m3
carrying one fp32 scale per ``256 x 256`` block of ``(K, N)``::

    dequantise(weight)[k, n] = weight[k, n] * weight_scale[k // 256, n // 256]

Accumulate-then-scale, and why that order is load-bearing
---------------------------------------------------------
The two ``128``-wide contraction tiles of one ``256`` block accumulate in PSUM,
and the block scale is applied **after** that accumulation, then added into an
fp32 SBUF accumulator. This is deliberately the same order the MoE consumer
uses (``nkilib/core/moe/moe_cte/bwmm_shard_on_I.py:2113``-``:2151``), because
that order is what makes the F1 power-of-two precondition load-bearing:
``increments/evidence-071.md`` F1 measured **720 fp32 ulp** of retile-remapping
error under a non-pow2 ``256``-block scale against **0** under a pow2 one. A
scale-then-accumulate kernel would hide that, and the campaign's tolerance would
then certify something other than kernel error.

Precision, stated rather than implied
-------------------------------------
* fp8 weight tiles are upcast to bf16 **on load** (``nl.load(..., dtype=...)``).
  The upcast is bit-exact: e4m3's 3 significand bits and 4 exponent bits are
  both contained in bf16's 7 and 8. It is preferred over ``perf_mode="double_row"``
  fp8 matmul because that mode is gated on NeuronCore generation
  (``nisa.nc_matmul``: "On NeuronCore-v2, performance mode is not supported"),
  and a correctness increment should not depend on a performance gate.
* the PSUM tile and the SBUF accumulator are fp32, and the kernel returns
  **fp32**. Casting the result is the caller's choice; throwing precision away
  inside a kernel whose acceptance measures precision is not.

The scale operand layout, and the one place it is written
---------------------------------------------------------
``nisa.tensor_scalar`` broadcasts ``operand0`` along the free dimension from a
tile of shape ``(data.shape[0], 1)``, so a per-block scalar must arrive as a
``[TILE_SIZE, 1]`` column. :func:`to_kernel_scale_layout` is the single place
that replication and the flat block index are written; :func:`flat_scale_index`
is the index itself, so no consumer repeats the arithmetic.

Route
-----
Acceptance is Tier N: the NKI simulator, reached through this module's own
:func:`blockwise_fp8_mm` seam (``wrap_nki -> NKIHOPCaller -> HOP ->
DispatchKey.CPU -> nki.simulator.simulate_kernel``). The seam counts its
dispatches, and the counters are module-level state with module-level reset and
read functions **on purpose**: `inc-glm53f-033`'s route predicate is form R-2
over *this* seam, so a later increment's own test must be able to zero and read
these counters from another module. A test-local counter would satisfy this
increment and break that one.

Under F1 a numeric comparison alone cannot prove a kernel ran -- a torch
fallback would put torch on both sides of the comparison and pass green -- so
the counters below are acceptance criteria, not diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.moe.blockwise_fp8_retile import (
    BLOCK_QUANT_SIZE,
    TILE_SIZE,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: Contraction tiles per ``256`` scale block. Re-derived from the two constants
#: rather than written as ``2``, so the pair cannot drift from the quotient.
K_TILES_PER_BLOCK = BLOCK_QUANT_SIZE // TILE_SIZE

#: Tensor Engine operand bounds, from ``nl.tile_size``:
#: ``pmax=128``, ``gemm_stationary_fmax=128``, ``gemm_moving_fmax=512``, and
#: ``psum_bank_fmax=512`` fp32 elements per PSUM bank. ``BLOCK_QUANT_SIZE``
#: (256) is the moving free extent this kernel uses, so it sits inside both the
#: moving bound and the PSUM bank bound with room to spare.
STATIONARY_FMAX = 128
MOVING_FMAX = 512

__all__ = [
    "BLOCK_QUANT_SIZE",
    "TILE_SIZE",
    "BlockwiseFp8MmError",
    "blockwise_fp8_mm",
    "blockwise_fp8_mm_kernel",
    "blockwise_fp8_mm_torch_oracle",
    "can_run_blockwise_fp8_mm",
    "dispatch_counters",
    "flat_scale_index",
    "kernel_identity",
    "kernel_scale_shape",
    "reset_dispatch_counters",
    "scale_grid_shape",
    "to_kernel_scale_layout",
]


class BlockwiseFp8MmError(ValueError):
    """A geometry or layout this module refuses, named rather than coerced.

    Raised in preference to letting NKI trap at trace time: a refusal that names
    the offending extent is what a caller can act on, and a silently truncated
    extent would compute a different function than the one requested.
    """


# --------------------------------------------------------------------------- #
# The NKI kernel. SCRATCH: no vendor member provides blockwise dense matmul.    #
# --------------------------------------------------------------------------- #
@nki.jit
def blockwise_fp8_mm_kernel(x, weight, weight_scale_t):
    """``out[M, N] = x[M, K] @ dequantise(weight[K, N])``, in NKI.

    Args:
        x: ``[M, K]`` activations, bf16. ``M`` is tiled by ``TILE_SIZE`` over the
            PSUM partition axis and ``K`` by ``BLOCK_QUANT_SIZE`` over the
            contraction axis.
        weight: ``[K, N]`` fp8-e4m3 weights, already expressed against the
            ``256``-granular scales (that re-expression is `inc-glm53f-024`'s
            producer, not this kernel's business).
        weight_scale_t: ``[TILE_SIZE, (K // 256) * (N // 256)]`` fp32, the
            kernel operand layout :func:`to_kernel_scale_layout` builds: one
            column per ``256 x 256`` block, replicated across the partition axis
            because ``nisa.tensor_scalar`` broadcasts only along the free
            dimension.

    Returns:
        ``[M, N]`` fp32.

    Both operands of every ``nc_matmul`` carry the contraction extent on the
    **partition** axis, which is what the Tensor Engine contracts over: ``x`` is
    therefore loaded through ``nl.load_transpose2d`` (a DMA-side transpose)
    rather than transposed on chip.
    """
    m_extent, k_extent = x.shape
    _, n_extent = weight.shape
    n_n_blocks = n_extent // BLOCK_QUANT_SIZE
    n_k_blocks = k_extent // BLOCK_QUANT_SIZE

    out = nl.ndarray((m_extent, n_extent), dtype=nl.float32, buffer=nl.shared_hbm)
    # One load: the scale operand is (partitions x blocks) and tiny.
    scale_sb = nl.load(weight_scale_t)

    for m_tile in range(m_extent // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
        for n_block in range(n_n_blocks):
            n0 = n_block * BLOCK_QUANT_SIZE
            # fp32 accumulator over the K blocks, in SBUF: PSUM is reclaimed per
            # block so the block scale can be applied between blocks.
            acc = nl.ndarray(
                (TILE_SIZE, BLOCK_QUANT_SIZE), dtype=nl.float32, buffer=nl.sbuf
            )
            for k_block in range(n_k_blocks):
                psum = nl.ndarray(
                    (TILE_SIZE, BLOCK_QUANT_SIZE), dtype=nl.float32, buffer=nl.psum
                )
                for k_sub in range(K_TILES_PER_BLOCK):
                    k0 = k_block * BLOCK_QUANT_SIZE + k_sub * TILE_SIZE
                    # [K=TILE_SIZE partitions, M=TILE_SIZE free]
                    x_t = nl.load_transpose2d(
                        x[m0 : m0 + TILE_SIZE, k0 : k0 + TILE_SIZE]
                    )
                    # [K=TILE_SIZE partitions, N=BLOCK_QUANT_SIZE free], upcast
                    # from fp8 on the DMA.
                    w_tile = nl.load(
                        weight[k0 : k0 + TILE_SIZE, n0 : n0 + BLOCK_QUANT_SIZE],
                        dtype=nl.bfloat16,
                    )
                    # dst = stationary.T @ moving = [M, N]. Explicit accumulate
                    # flag rather than the compiler's inference, so the
                    # first-write-overwrites contract is visible here.
                    nisa.nc_matmul(
                        dst=psum[0:TILE_SIZE, 0:BLOCK_QUANT_SIZE],
                        stationary=x_t,
                        moving=w_tile,
                        accumulate=(k_sub > 0),
                    )
                flat = k_block * n_n_blocks + n_block
                if k_block == 0:
                    # First block initialises the accumulator, so no zeroing pass.
                    nisa.tensor_scalar(
                        dst=acc[0:TILE_SIZE, 0:BLOCK_QUANT_SIZE],
                        data=psum[0:TILE_SIZE, 0:BLOCK_QUANT_SIZE],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                    )
                else:
                    nisa.scalar_tensor_tensor(
                        dst=acc[0:TILE_SIZE, 0:BLOCK_QUANT_SIZE],
                        data=psum[0:TILE_SIZE, 0:BLOCK_QUANT_SIZE],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                        op1=nl.add,
                        operand1=acc[0:TILE_SIZE, 0:BLOCK_QUANT_SIZE],
                    )
            nl.store(
                out[m0 : m0 + TILE_SIZE, n0 : n0 + BLOCK_QUANT_SIZE],
                value=acc[0:TILE_SIZE, 0:BLOCK_QUANT_SIZE],
            )
    return out


# --------------------------------------------------------------------------- #
# Geometry and the scale-operand bridge.                                       #
# --------------------------------------------------------------------------- #
def _require_blocked(rows: int, cols: int, tokens: int) -> None:
    """Every extent condition the kernel above imposes, checked in one place.

    ``rows`` is ``K`` (the contraction extent), ``cols`` is ``N``, ``tokens`` is
    ``M``. Each condition names the line of the kernel that needs it, so a
    reader can check the refusal against the code rather than against prose.
    """
    problems: list[str] = []
    if tokens <= 0 or tokens % TILE_SIZE:
        problems.append(
            f"M={tokens} is not a positive multiple of TILE_SIZE={TILE_SIZE}; "
            f"the kernel tiles M over the PSUM partition axis and does not pad. "
            f"Padding tokens to a whole tile is the caller's, exactly as the MoE "
            f"consumer pads to block_size"
        )
    if rows <= 0 or rows % BLOCK_QUANT_SIZE:
        problems.append(
            f"K={rows} is not a positive multiple of "
            f"BLOCK_QUANT_SIZE={BLOCK_QUANT_SIZE}; the remainder has no block "
            f"scale"
        )
    if cols <= 0 or cols % BLOCK_QUANT_SIZE:
        problems.append(
            f"N={cols} is not a positive multiple of "
            f"BLOCK_QUANT_SIZE={BLOCK_QUANT_SIZE}; the remainder has no block "
            f"scale"
        )
    if BLOCK_QUANT_SIZE > MOVING_FMAX:
        # Structural, and asserted rather than assumed: if either constant ever
        # moves, the moving free extent must be re-tiled, not silently exceeded.
        problems.append(
            f"BLOCK_QUANT_SIZE={BLOCK_QUANT_SIZE} exceeds the Tensor Engine "
            f"moving free bound {MOVING_FMAX}"
        )
    if TILE_SIZE > STATIONARY_FMAX:
        problems.append(
            f"TILE_SIZE={TILE_SIZE} exceeds the Tensor Engine stationary free "
            f"bound {STATIONARY_FMAX}"
        )
    if problems:
        raise BlockwiseFp8MmError(
            "blockwise fp8 mm refuses this geometry: " + "; ".join(problems)
        )


def scale_grid_shape(rows: int, cols: int) -> tuple[int, int]:
    """``(K // 256, N // 256)`` -- the shape of the PUBLIC scale grid.

    This is the shape a caller supplies: one fp32 scale per ``256 x 256`` block
    of the weight, indexed ``[k_block, n_block]``. The kernel operand is a
    different shape; :func:`to_kernel_scale_layout` is the bridge.
    """
    if rows <= 0 or rows % BLOCK_QUANT_SIZE or cols <= 0 or cols % BLOCK_QUANT_SIZE:
        raise BlockwiseFp8MmError(
            f"weight extent [{rows},{cols}] is not a whole number of "
            f"{BLOCK_QUANT_SIZE}x{BLOCK_QUANT_SIZE} blocks"
        )
    return rows // BLOCK_QUANT_SIZE, cols // BLOCK_QUANT_SIZE


def flat_scale_index(k_block: int, n_block: int, n_n_blocks: int) -> int:
    """The kernel's flat block index: ``k_block`` major, ``n_block`` minor.

    Written once, here, and read by both the bridge and the kernel's own
    ``flat = k_block * n_n_blocks + n_block``. A consumer that needs the index
    calls this rather than repeating the arithmetic -- the defect that costs is
    a transposed flattening, which no range check can see
    (``increments/evidence-071.md`` §9.2).
    """
    return k_block * n_n_blocks + n_block


def kernel_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    """``(TILE_SIZE, n_blocks)`` -- the shape the kernel's scale operand needs."""
    n_k_blocks, n_n_blocks = scale_grid_shape(rows, cols)
    return TILE_SIZE, n_k_blocks * n_n_blocks


def to_kernel_scale_layout(weight_scale: Tensor, rows: int, cols: int) -> Tensor:
    """Bridge the public ``[K//256, N//256]`` grid to the kernel's operand.

    Two things happen here and nowhere else: the grid is flattened by
    :func:`flat_scale_index`, and each scalar is replicated across
    ``TILE_SIZE`` partitions because ``nisa.tensor_scalar`` broadcasts
    ``operand0`` only along the **free** dimension, from a tile of shape
    ``(data.shape[0], 1)``.

    Returns:
        ``[TILE_SIZE, (K//256) * (N//256)]`` fp32, contiguous.

    Raises:
        BlockwiseFp8MmError: if ``weight_scale`` is not the declared grid shape
            or not fp32. Checked rather than trusted: a mis-sized grid can
            reshape without error onto a different block-to-scale assignment,
            and the two orders are indistinguishable by any range check.
    """
    want = scale_grid_shape(rows, cols)
    if tuple(weight_scale.shape) != want:
        raise BlockwiseFp8MmError(
            f"weight_scale has shape {tuple(weight_scale.shape)}, expected "
            f"{want} for a [K={rows}, N={cols}] weight. Refusing to reshape: a "
            f"mis-sized scale grid can flatten onto a different "
            f"block-to-scale assignment without any error."
        )
    if weight_scale.dtype != torch.float32:
        raise BlockwiseFp8MmError(
            f"weight_scale must be fp32, got {weight_scale.dtype}"
        )
    n_k_blocks, n_n_blocks = want
    flat = torch.empty(
        n_k_blocks * n_n_blocks, dtype=torch.float32, device=weight_scale.device
    )
    for k_block in range(n_k_blocks):
        for n_block in range(n_n_blocks):
            flat[flat_scale_index(k_block, n_block, n_n_blocks)] = weight_scale[
                k_block, n_block
            ]
    return flat.unsqueeze(0).expand(TILE_SIZE, flat.numel()).contiguous()


# --------------------------------------------------------------------------- #
# The route seam and its counters.                                             #
# --------------------------------------------------------------------------- #
@dataclass
class _DispatchCounters:
    """What route actually ran, counted rather than inferred.

    ``nki_dispatch`` counts entries into the ``wrap_nki`` seam; ``torch_fallback``
    counts entries into the torch path. Two counters rather than one flag, so
    "the kernel ran" and "the fallback did not run" are independent readings and
    a test can require both.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: MODULE-LEVEL, and that is a contract rather than an implementation detail:
#: `inc-glm53f-033` counts this seam's dispatches from its OWN test module (form
#: R-2), so the counter must be resettable and readable from outside this
#: module and outside this increment's test.
_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters. Called at the start of each declared test case."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


def can_run_blockwise_fp8_mm(x: Tensor, rows: int, cols: int, tokens: int) -> bool:
    """Is the NKI route available *and* admissible for this geometry?

    Two independent conditions, deliberately not merged: ``can_run_kernel``
    answers "is there a device or a simulator", :func:`_require_blocked` answers
    "does this kernel accept these extents". A geometry the kernel cannot serve
    raises rather than falling back, because falling back would ship a torch
    path for kernel-class work (P13, D6).

    Raises:
        BlockwiseFp8MmError: if the geometry is inadmissible.
    """
    _require_blocked(rows, cols, tokens)
    return can_run_kernel(x)


def blockwise_fp8_mm(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
) -> Tensor:
    """Dense blockwise-fp8 GEMM. The seam the route predicate counts.

    Args:
        x: ``[M, K]`` activations, bf16.
        weight: ``[K, N]`` fp8-e4m3, expressed against ``weight_scale``.
        weight_scale: ``[K//256, N//256]`` fp32, one scale per weight block.

    Returns:
        ``[M, N]`` fp32.

    Raises:
        BlockwiseFp8MmError: on an inadmissible geometry or a mis-shaped scale
            grid.
    """
    tokens, rows = x.shape[-2], x.shape[-1]
    if weight.shape[-2] != rows:
        raise BlockwiseFp8MmError(
            f"x has K={rows} but weight has K={weight.shape[-2]}; the "
            f"contraction extents must agree"
        )
    cols = weight.shape[-1]

    if not can_run_blockwise_fp8_mm(x, rows, cols, tokens):
        _COUNTERS.torch_fallback += 1
        logger.debug(
            "blockwise_fp8_mm: NKI route unavailable, using the torch path "
            "(oracle / constraint-violation fallback, not the shipped path)"
        )
        return blockwise_fp8_mm_torch_oracle(x, weight, weight_scale)

    _COUNTERS.nki_dispatch += 1
    scale_t = to_kernel_scale_layout(weight_scale, rows, cols)
    return wrap_nki(blockwise_fp8_mm_kernel)(
        x=x, weight=weight, weight_scale_t=scale_t
    )


def blockwise_fp8_mm_torch_oracle(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
) -> Tensor:
    """Block-dequantise, then matmul -- in torch, in fp32.

    The independent formulation the plan's acceptance names: it dequantises
    **first** and contracts in one fp32 matmul, where the kernel contracts per
    ``256`` block and scales between blocks. Because the two disagree in
    arithmetic ORDER while agreeing in value, the comparison is a real check on
    the kernel's block-to-scale assignment rather than a restatement of it --
    and it never consults :func:`flat_scale_index`, so a transposed flattening
    in the bridge shows up as a numeric disagreement.

    This is also the constraint-violation fallback for
    :func:`blockwise_fp8_mm`. It is never the shipped kernel-class path (D6).

    Returns:
        ``[M, N]`` fp32.
    """
    rows, cols = weight.shape[-2], weight.shape[-1]
    want = scale_grid_shape(rows, cols)
    if tuple(weight_scale.shape) != want:
        raise BlockwiseFp8MmError(
            f"weight_scale has shape {tuple(weight_scale.shape)}, expected "
            f"{want} for a [K={rows}, N={cols}] weight"
        )
    dequantised = weight.to(torch.float32) * weight_scale.repeat_interleave(
        BLOCK_QUANT_SIZE, 0
    ).repeat_interleave(BLOCK_QUANT_SIZE, 1)
    return x.to(torch.float32) @ dequantised


def kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the NKI kernel, read off the object.

    Exposed so a test can assert the seam dispatches to the kernel this module
    authors, and so a substitution shows up as a changed reading rather than as
    silence.
    """
    func = getattr(blockwise_fp8_mm_kernel, "func", None)
    target = func if func is not None else blockwise_fp8_mm_kernel
    return target.__module__, target.__qualname__
