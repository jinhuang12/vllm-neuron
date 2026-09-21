# SPDX-License-Identifier: Apache-2.0
"""Dense blockwise-fp8 GEMM: ``out[M, N] = x[M, K] @ dequantise(weight[K, N])``.

``weight`` is fp8-e4m3 with one fp32 scale per ``128 x 128`` block of ``(K, N)``,
so ``dequantise(weight)[k, n] = weight[k, n] * weight_scale[k // 128, n // 128]``
-- the granularity a checkpoint stores, which the kernel indexes directly instead
of re-tiling. A scale block spans exactly one ``128``-wide contraction tile, so
every ``nc_matmul`` result is multiplied by a single scale before it is added into
an fp32 SBUF accumulator: no partial sum crosses two scales.

fp8 weight tiles are upcast to bf16 on the load DMA. The upcast is bit-exact
(e4m3's 3 significand and 4 exponent bits fit bf16's 7 and 8) and avoids
``perf_mode="double_row"`` fp8 matmul, which NeuronCore-v2 does not support. PSUM,
the accumulator and the returned tensor are fp32; casting the result down is the
caller's choice.

:func:`blockwise_fp8_mm_torch_oracle` is the torch reference for the same product,
and the fallback taken when no Neuron device or simulator is available.
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

#: Scale-block extent this module indexes by: one fp32 scale per ``128 x 128``
#: weight block. Equals ``TILE_SIZE``; ``blockwise_fp8_retile.BLOCK_QUANT_SIZE``
#: (256) is the MoE path's granularity and is never read here.
SCALE_BLOCK_SIZE = TILE_SIZE

#: Contraction tiles per scale block -- ``1`` at this granularity. Derived from the
#: two constants so the pair cannot drift from the quotient.
K_TILES_PER_BLOCK = SCALE_BLOCK_SIZE // TILE_SIZE

#: Tensor Engine free-size bounds from ``nl.tile_size``: 128 stationary, 512 moving
#: (also the fp32 PSUM bank extent). ``SCALE_BLOCK_SIZE`` is the moving extent used
#: here, so it fits both with room to spare.
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
    """A geometry or layout this module refuses.

    Raised in preference to letting NKI trap at trace time: the message names the
    offending extent, and no extent is silently truncated into a different
    function than the one requested.
    """


@nki.jit
def blockwise_fp8_mm_kernel(x, weight, weight_scale_t):
    """``out[M, N] = x[M, K] @ dequantise(weight[K, N])``, in NKI.

    Both operands of every ``nc_matmul`` carry the contraction extent on the
    partition axis, which is what the Tensor Engine contracts over, so ``x`` is
    loaded through ``nl.load_transpose2d`` (a DMA-side transpose) rather than
    transposed on chip.

    Args:
        x: ``[M, K]`` activations, bf16. ``M`` is tiled by ``TILE_SIZE`` over the
            PSUM partition axis and ``K`` by ``SCALE_BLOCK_SIZE`` over the
            contraction axis.
        weight: ``[K, N]`` fp8-e4m3 weights, expressed against the
            ``128``-granular block scales the checkpoint itself stores.
        weight_scale_t: ``[TILE_SIZE, (K // 128) * (N // 128)]`` fp32, the operand
            layout :func:`to_kernel_scale_layout` builds: one column per
            ``128 x 128`` block, replicated across the partition axis because
            ``nisa.tensor_scalar`` broadcasts only along the free dimension.

    Returns:
        ``[M, N]`` fp32.
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
                    # dst = stationary.T @ moving = [M, N]. The accumulate flag is
                    # explicit rather than inferred by the compiler, so the
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


def _require_blocked(rows: int, cols: int, tokens: int) -> None:
    """Check every extent condition the kernel above imposes.

    ``rows`` is ``K`` (the contraction extent), ``cols`` is ``N``, ``tokens`` is
    ``M``.

    Raises:
        BlockwiseFp8MmError: naming every condition that failed.
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
        # Asserted rather than assumed: if either constant moves, the moving free
        # extent has to be re-tiled, not silently exceeded.
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
    """``(K // 128, N // 128)`` -- the shape of the public scale grid.

    This is the shape a caller supplies: one fp32 scale per ``128 x 128`` weight
    block, indexed ``[k_block, n_block]``. The kernel operand is a different
    shape; :func:`to_kernel_scale_layout` is the bridge.
    """
    if rows <= 0 or rows % SCALE_BLOCK_SIZE or cols <= 0 or cols % SCALE_BLOCK_SIZE:
        raise BlockwiseFp8MmError(
            f"weight extent [{rows},{cols}] is not a whole number of "
            f"{SCALE_BLOCK_SIZE}x{SCALE_BLOCK_SIZE} blocks"
        )
    return rows // SCALE_BLOCK_SIZE, cols // SCALE_BLOCK_SIZE


def flat_scale_index(k_block: int, n_block: int, n_n_blocks: int) -> int:
    """The kernel's flat block index: ``k_block`` major, ``n_block`` minor.

    Written once here and mirrored by the kernel's own
    ``flat = k_block * n_n_blocks + n_block``, so a consumer never repeats the
    arithmetic: a transposed flattening leaves every shape valid and no range
    check can see it.
    """
    return k_block * n_n_blocks + n_block


def kernel_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    """``(TILE_SIZE, n_blocks)`` -- the shape the kernel's scale operand needs."""
    n_k_blocks, n_n_blocks = scale_grid_shape(rows, cols)
    return TILE_SIZE, n_k_blocks * n_n_blocks


def to_kernel_scale_layout(weight_scale: Tensor, rows: int, cols: int) -> Tensor:
    """Bridge the public ``[K//128, N//128]`` grid to the kernel's operand.

    Flattens the grid by :func:`flat_scale_index` and replicates each scalar
    across ``TILE_SIZE`` partitions, because ``nisa.tensor_scalar`` broadcasts
    ``operand0`` only along the **free** dimension, from a tile of shape
    ``(data.shape[0], 1)``.

    Returns:
        ``[TILE_SIZE, (K//128) * (N//128)]`` fp32, contiguous.

    Raises:
        BlockwiseFp8MmError: if ``weight_scale`` is not the declared grid shape
            or not fp32. A mis-sized grid can reshape without error onto a
            different block-to-scale assignment, and the two orders are
            indistinguishable by any range check.
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


@dataclass
class _DispatchCounters:
    """Which route ran, counted rather than inferred.

    ``nki_dispatch`` counts entries into the ``wrap_nki`` path, ``torch_fallback``
    counts entries into the torch path. Two counters rather than one flag, so
    "the kernel ran" and "the fallback did not run" are independent readings.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: Module level so the counters can be zeroed and read from outside this module.
_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


# Counter stores stay off the traced graph: a store Dynamo traces turns into a
# value guard that fails on the first call after warmup.
@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch."""
    _COUNTERS.nki_dispatch += 1


def can_run_blockwise_fp8_mm(x: Tensor, rows: int, cols: int, tokens: int) -> bool:
    """Is the NKI route available *and* admissible for this geometry?

    Two independent conditions, deliberately not merged: ``can_run_kernel``
    answers "is there a device or a simulator", :func:`_require_blocked` answers
    "does this kernel accept these extents". A geometry the kernel cannot serve
    raises rather than falling back to torch.

    Raises:
        BlockwiseFp8MmError: if the geometry is inadmissible.
    """
    _require_blocked(rows, cols, tokens)
    return can_run_kernel(x)


def _checked_prebuilt_scale(prebuilt: Tensor, rows: int, cols: int) -> Tensor:
    """Accept a caller-supplied kernel scale operand, or refuse it by shape.

    When the operand is built once at load time the bridge does not run, so the
    bridge's own checks do not run either; this replaces them at the call site.

    The check here is the weaker of the two: :func:`kernel_scale_shape` pins
    ``TILE_SIZE`` and the product ``n_k_blocks * n_n_blocks``, so a transposed
    public grid -- ``[16, 8]`` where ``[8, 16]`` was meant -- has the same element
    count and passes. That case is caught where the operand is built instead,
    because :func:`to_kernel_scale_layout` compares the public grid against
    ``(rows, cols)`` before flattening.
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
    """Dense blockwise-fp8 GEMM: NKI kernel when available, torch otherwise.

    Args:
        x: ``[M, K]`` activations, bf16.
        weight: ``[K, N]`` fp8-e4m3, expressed against ``weight_scale``.
        weight_scale: ``[K//128, N//128]`` fp32, one scale per weight block.
        prebuilt_scale_t: Optional, keyword-only. The kernel operand
            :func:`to_kernel_scale_layout` would have built, already built --
            shape :func:`kernel_scale_shape`, fp32. Supply it and the bridge is
            not entered on this call. ``weight_scale`` is still required either
            way: the torch fallback consumes the public grid, so a prebuilt
            operand cannot stand in for it.

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
        _count_torch_fallback()
        logger.debug(
            "blockwise_fp8_mm: NKI route unavailable, using the torch path "
            "(oracle / constraint-violation fallback, not the shipped path)"
        )
        return blockwise_fp8_mm_torch_oracle(x, weight, weight_scale)

    _count_nki_dispatch()
    if prebuilt_scale_t is None:
        _count_scale_layout_build()
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

    Independent of the kernel in arithmetic order: it dequantises **first** and
    contracts in one fp32 matmul, where the kernel contracts one ``128``-wide
    block at a time and scales between blocks. It also never consults
    :func:`flat_scale_index`, so a transposed flattening in the bridge shows up
    as a numeric disagreement rather than as agreement by construction.

    This is also the fallback for :func:`blockwise_fp8_mm` when the NKI route is
    unavailable.

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
    """``(module, qualname)`` of the NKI kernel, read off the wrapped object."""
    func = getattr(blockwise_fp8_mm_kernel, "func", None)
    target = func if func is not None else blockwise_fp8_mm_kernel
    return target.__module__, target.__qualname__


@dataclass
class _BuildCounters:
    """How many kernel scale operands :func:`blockwise_fp8_mm` built itself.

    Separate from ``_DispatchCounters`` so that its reader keeps returning a
    two-tuple. Only builds at the one site in :func:`blockwise_fp8_mm` are
    counted; a direct call to :func:`to_kernel_scale_layout` from elsewhere is
    not, which is the right population for the question this answers -- whether
    the per-forward path rebuilds the operand.
    """

    scale_layout_builds: int = 0


#: Module level for the same reason ``_COUNTERS`` is: zeroed and read from
#: outside this module.
_BUILD_COUNTERS = _BuildCounters()


def reset_scale_layout_builds() -> None:
    """Zero the build counter."""
    _BUILD_COUNTERS.scale_layout_builds = 0


def scale_layout_builds() -> int:
    """How many operands have been built since the last reset."""
    return _BUILD_COUNTERS.scale_layout_builds


@torch._dynamo.assume_constant_result
def _count_scale_layout_build() -> None:
    """Count one scale-layout build, off the traced graph."""
    _BUILD_COUNTERS.scale_layout_builds += 1
