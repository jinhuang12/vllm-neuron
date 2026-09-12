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
carrying one fp32 scale per ``128 x 128`` block of ``(K, N)``::

    dequantise(weight)[k, n] = weight[k, n] * weight_scale[k // 128, n // 128]

That is the granularity the CHECKPOINT stores, and reading it directly is
`inc-glm53f-112`. The kernel formerly indexed by ``256`` blocks, which meant the
checkpoint's scales had to be retiled up to ``256`` before this kernel could use
them -- four scales replaced by one, which is arithmetic on a scale and is
exactly what the retile's error was.

One scale per product, so no arithmetic touches a scale
-------------------------------------------------------
At this granularity a scale block holds exactly ONE ``128``-wide contraction
tile (``K_TILES_PER_BLOCK == 1``), so each ``nc_matmul`` result is multiplied by
exactly one scale and then added into an fp32 SBUF accumulator. Nothing is
accumulated across two different scales, and no scale is ever rescaled,
compensated or combined with another.

The order is still written as accumulate-then-scale, and that is deliberate: the
form matches the MoE consumer's (``nkilib/core/moe/moe_cte/bwmm_shard_on_I.py``
``:2113``-``:2151``) and it stays correct if the granularity ever widens again.
What changed is that the order stopped being load-bearing. Under the ``256``
grid, ``increments/evidence-071.md`` F1 measured **720 fp32 ulp** of
retile-remapping error for a non-power-of-two block scale against **0** for a
power-of-two one; at ``128`` there is no remapping to be wrong, so the
acceptance reads EXACT equality instead of a tolerance.

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

from vllm_neuron.functional.moe.blockwise_fp8_retile import TILE_SIZE
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: The scale-block extent this module indexes by: the granularity the CHECKPOINT
#: stores, one fp32 scale per ``128 x 128`` block of the weight.
#:
#: `inc-glm53f-112`. It is declared here and equals ``TILE_SIZE`` rather than
#: being imported from ``blockwise_fp8_retile``, whose ``BLOCK_QUANT_SIZE`` is
#: the MoE consumer's ``256`` and is not this kernel's business. Two names for
#: one granularity is the drift D17.1 exists to prevent, so this module carries
#: exactly one, and no ``256`` is read by any expression in this file.
SCALE_BLOCK_SIZE = TILE_SIZE

#: Contraction tiles per scale block. Re-derived from the two constants rather
#: than written as ``1``, so the pair cannot drift from the quotient. At this
#: granularity the quotient IS ``1``: one contraction tile per scale block, which
#: is why no partial sum is ever accumulated across two different scales.
K_TILES_PER_BLOCK = SCALE_BLOCK_SIZE // TILE_SIZE

#: Tensor Engine operand bounds, from ``nl.tile_size``:
#: ``pmax=128``, ``gemm_stationary_fmax=128``, ``gemm_moving_fmax=512``, and
#: ``psum_bank_fmax=512`` fp32 elements per PSUM bank. ``SCALE_BLOCK_SIZE``
#: (128) is the moving free extent this kernel uses, so it sits inside both the
#: moving bound and the PSUM bank bound with room to spare.
STATIONARY_FMAX = 128
MOVING_FMAX = 512

__all__ = [
    "SCALE_BLOCK_SIZE",
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
            PSUM partition axis and ``K`` by ``SCALE_BLOCK_SIZE`` over the
            contraction axis.
        weight: ``[K, N]`` fp8-e4m3 weights, already expressed against the
            ``128``-granular scales the checkpoint itself stores, so no
            producer-side re-expression stands between the two).
        weight_scale_t: ``[TILE_SIZE, (K // 128) * (N // 128)]`` fp32, the
            kernel operand layout :func:`to_kernel_scale_layout` builds: one
            column per ``128 x 128`` block, replicated across the partition axis
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
    n_n_blocks = n_extent // SCALE_BLOCK_SIZE
    n_k_blocks = k_extent // SCALE_BLOCK_SIZE

    out = nl.ndarray((m_extent, n_extent), dtype=nl.float32, buffer=nl.shared_hbm)
    # One load: the scale operand is (partitions x blocks) and tiny.
    scale_sb = nl.load(weight_scale_t)

    for m_tile in range(m_extent // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
        for n_block in range(n_n_blocks):
            n0 = n_block * SCALE_BLOCK_SIZE
            # fp32 accumulator over the K blocks, in SBUF: PSUM is reclaimed per
            # block so the block scale can be applied between blocks.
            acc = nl.ndarray(
                (TILE_SIZE, SCALE_BLOCK_SIZE), dtype=nl.float32, buffer=nl.sbuf
            )
            for k_block in range(n_k_blocks):
                psum = nl.ndarray(
                    (TILE_SIZE, SCALE_BLOCK_SIZE), dtype=nl.float32, buffer=nl.psum
                )
                for k_sub in range(K_TILES_PER_BLOCK):
                    k0 = k_block * SCALE_BLOCK_SIZE + k_sub * TILE_SIZE
                    # [K=TILE_SIZE partitions, M=TILE_SIZE free]
                    x_t = nl.load_transpose2d(
                        x[m0 : m0 + TILE_SIZE, k0 : k0 + TILE_SIZE]
                    )
                    # [K=TILE_SIZE partitions, N=SCALE_BLOCK_SIZE free], upcast
                    # from fp8 on the DMA.
                    w_tile = nl.load(
                        weight[k0 : k0 + TILE_SIZE, n0 : n0 + SCALE_BLOCK_SIZE],
                        dtype=nl.bfloat16,
                    )
                    # dst = stationary.T @ moving = [M, N]. Explicit accumulate
                    # flag rather than the compiler's inference, so the
                    # first-write-overwrites contract is visible here.
                    nisa.nc_matmul(
                        dst=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        stationary=x_t,
                        moving=w_tile,
                        accumulate=(k_sub > 0),
                    )
                flat = k_block * n_n_blocks + n_block
                if k_block == 0:
                    # First block initialises the accumulator, so no zeroing pass.
                    nisa.tensor_scalar(
                        dst=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        data=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                    )
                else:
                    nisa.scalar_tensor_tensor(
                        dst=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        data=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                        op1=nl.add,
                        operand1=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                    )
            nl.store(
                out[m0 : m0 + TILE_SIZE, n0 : n0 + SCALE_BLOCK_SIZE],
                value=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
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
    if rows <= 0 or rows % SCALE_BLOCK_SIZE:
        problems.append(
            f"K={rows} is not a positive multiple of "
            f"SCALE_BLOCK_SIZE={SCALE_BLOCK_SIZE}; the remainder has no block "
            f"scale"
        )
    if cols <= 0 or cols % SCALE_BLOCK_SIZE:
        problems.append(
            f"N={cols} is not a positive multiple of "
            f"SCALE_BLOCK_SIZE={SCALE_BLOCK_SIZE}; the remainder has no block "
            f"scale"
        )
    if SCALE_BLOCK_SIZE > MOVING_FMAX:
        # Structural, and asserted rather than assumed: if either constant ever
        # moves, the moving free extent must be re-tiled, not silently exceeded.
        problems.append(
            f"SCALE_BLOCK_SIZE={SCALE_BLOCK_SIZE} exceeds the Tensor Engine "
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
    """``(K // 128, N // 128)`` -- the shape of the PUBLIC scale grid.

    This is the shape a caller supplies: one fp32 scale per ``128 x 128`` block
    of the weight, indexed ``[k_block, n_block]``. The kernel operand is a
    different shape; :func:`to_kernel_scale_layout` is the bridge.
    """
    if rows <= 0 or rows % SCALE_BLOCK_SIZE or cols <= 0 or cols % SCALE_BLOCK_SIZE:
        raise BlockwiseFp8MmError(
            f"weight extent [{rows},{cols}] is not a whole number of "
            f"{SCALE_BLOCK_SIZE}x{SCALE_BLOCK_SIZE} blocks"
        )
    return rows // SCALE_BLOCK_SIZE, cols // SCALE_BLOCK_SIZE


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
    """Bridge the public ``[K//128, N//128]`` grid to the kernel's operand.

    Two things happen here and nowhere else: the grid is flattened by
    :func:`flat_scale_index`, and each scalar is replicated across
    ``TILE_SIZE`` partitions because ``nisa.tensor_scalar`` broadcasts
    ``operand0`` only along the **free** dimension, from a tile of shape
    ``(data.shape[0], 1)``.

    Returns:
        ``[TILE_SIZE, (K//128) * (N//128)]`` fp32, contiguous.

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


def _checked_prebuilt_scale(prebuilt: Tensor, rows: int, cols: int) -> Tensor:
    """A caller-supplied kernel scale operand, or a refusal naming the shape.

    `inc-glm53f-090`. When the operand is built once at load time the bridge
    does not run, so the bridge's own checks do not run either. This is the net
    that replaces them at the seam.

    WHAT THIS CHECK CANNOT SEE, recorded because the gap is real and is covered
    elsewhere rather than ignored: :func:`kernel_scale_shape` pins ``TILE_SIZE``
    and the PRODUCT ``n_k_blocks * n_n_blocks``, so a transposed public grid --
    ``[16, 8]`` where ``[8, 16]`` was meant -- flattens to the same element count
    and passes here. That is the very defect :func:`flat_scale_index` warns
    about. It is caught at BUILD time instead: the builder calls
    :func:`to_kernel_scale_layout`, whose grid check compares the public grid
    against ``(rows, cols)`` before flattening. The two checks are complements,
    and the strong one still runs -- once, where the operand is made.
    """
    want = kernel_scale_shape(rows, cols)
    if tuple(prebuilt.shape) != want:
        raise BlockwiseFp8MmError(
            f"prebuilt_scale_t has shape {tuple(prebuilt.shape)}, expected "
            f"{want} = kernel_scale_shape(rows={rows}, cols={cols}). Refusing "
            f"to reshape or rebuild: an operand of the wrong extent means the "
            f"caller prepared it for a different weight."
        )
    if prebuilt.dtype != torch.float32:
        raise BlockwiseFp8MmError(
            f"prebuilt_scale_t must be fp32, got {prebuilt.dtype}; the kernel "
            f"reads block scales as fp32 and a narrower dtype computes a "
            f"different function without changing any shape"
        )
    return prebuilt


def blockwise_fp8_mm(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    *,
    prebuilt_scale_t: Tensor | None = None,
) -> Tensor:
    """Dense blockwise-fp8 GEMM. The seam the route predicate counts.

    Args:
        x: ``[M, K]`` activations, bf16.
        weight: ``[K, N]`` fp8-e4m3, expressed against ``weight_scale``.
        weight_scale: ``[K//128, N//128]`` fp32, one scale per weight block.
        prebuilt_scale_t: OPTIONAL, keyword-only. The kernel operand
            :func:`to_kernel_scale_layout` would have built, already built --
            shape :func:`kernel_scale_shape`, fp32. Supply it and the bridge is
            not entered on this call; omit it and this function behaves exactly
            as it did before `inc-glm53f-090`, which is why `-026`'s landed
            acceptance needs no edit. ``weight_scale`` is still required either
            way: the torch-oracle fallback consumes the PUBLIC grid, so a
            prebuilt operand cannot stand in for it.

    Returns:
        ``[M, N]`` fp32.

    Raises:
        BlockwiseFp8MmError: on an inadmissible geometry, a mis-shaped scale
            grid, or a prebuilt operand that is not
            ``kernel_scale_shape(rows, cols)`` fp32.
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
    # `inc-glm53f-090`: the operand's ARRIVAL FORM, and the one build site.
    # The counter increments only on the branch that actually builds, so a
    # reading of 0 over a forward step means the bridge did not run on it --
    # which is the whole claim. The attribute store is the form already proven
    # graph-safe at capture: ``_COUNTERS.nki_dispatch += 1`` one line above sits
    # in this same function and `-022` part 1 captured 2/2 shapes with it there.
    if prebuilt_scale_t is None:
        _BUILD_COUNTERS.scale_layout_builds += 1
        scale_t = to_kernel_scale_layout(weight_scale, rows, cols)
    else:
        scale_t = _checked_prebuilt_scale(prebuilt_scale_t, rows, cols)
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
    ``128`` block and scales between blocks. Because the two disagree in
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
        SCALE_BLOCK_SIZE, 0
    ).repeat_interleave(SCALE_BLOCK_SIZE, 1)
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


# --------------------------------------------------------------------------- #
# `inc-glm53f-090` OWNS EVERYTHING BELOW THIS LINE, AND IT SITS AT THE END OF   #
# THE FILE ON PURPOSE. Six other files cite this module by line -- eleven cites #
# in all -- and every one of them targets a line above `blockwise_fp8_mm`.      #
# Defining these three names further up would have shifted all of them by the   #
# height of this block, so a re-anchor sweep across five files outside this      #
# block's surface would have been owed for nothing. Python resolves module       #
# globals at CALL time, so the seam above reads them without a forward           #
# declaration. The names are absent from `__all__` for the same reason: adding   #
# two entries there shifts the whole file, and `__all__` gates only `import *`,  #
# which nothing in this campaign uses.                                          #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The scale-operand BUILD counter. `inc-glm53f-090` owns this and the operand's #
# arrival form below, and nothing else in this file.                            #
# --------------------------------------------------------------------------- #
@dataclass
class _BuildCounters:
    """How many kernel scale operands the seam built itself.

    SEPARATE from ``_DispatchCounters`` deliberately. That dataclass, its reset
    and its two-tuple reader are `inc-glm53f-026`'s, and `-026`'s landed
    acceptance reads ``dispatch_counters()`` as a two-tuple -- a third field
    would either change that shape or force an edit inside `-026`'s reset. A
    separate counter leaves `-026`'s surface untouched.

    WHAT IT COUNTS, stated so a reading cannot be over-read: builds THE SEAM
    performs, at the one site in :func:`blockwise_fp8_mm`. A direct call to
    :func:`to_kernel_scale_layout` from anywhere else is not counted. That is
    the right population for the claim it settles -- whether the per-forward
    path rebuilds the operand -- because the dense shared-expert path reaches
    the bridge only through this seam and imports the helper nowhere.
    """

    scale_layout_builds: int = 0


#: MODULE-LEVEL for the same reason ``_COUNTERS`` is, and it is the same
#: contract: `inc-glm53f-090` reads this counter from its OWN test module, so it
#: must be resettable and readable from outside this module.
_BUILD_COUNTERS = _BuildCounters()


def reset_scale_layout_builds() -> None:
    """Zero the build counter. Called at the start of each declared case."""
    _BUILD_COUNTERS.scale_layout_builds = 0


def scale_layout_builds() -> int:
    """How many operands the seam has built since the last reset."""
    return _BUILD_COUNTERS.scale_layout_builds
