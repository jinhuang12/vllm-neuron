# SPDX-License-Identifier: Apache-2.0
"""Sinkhorn normalisation for mHC, as an NKI kernel.

The iterative row/column rescaling that turns a raw mHC affinity matrix into a
doubly stochastic one, before
:mod:`vllm_neuron.functional.mhc.hyper_connection` mixes the streams. The torch
code here is a CPU oracle, never the shipped path.

What the kernel computes
------------------------
Given a strictly positive affinity matrix ``A[M, N]``, it runs
:data:`SINKHORN_ITERS` iterations of alternating row-then-column rescaling::

    for _ in range(SINKHORN_ITERS):
        A[i, j] /= sum_j A[i, j]                         # every row sums to 1
        A[i, j] *= column_target(M, N) / sum_i A[i, j]   # every column sums to M/N

and returns fp32. The fixed point is doubly stochastic in the rectangular sense:
row sums ``1``, column sums ``M / N``, total mass ``M`` on both readings.
:func:`row_target` and :func:`column_target` are the one place those numbers are
written.

The iteration loop is a ``nl.sequential_range`` in the kernel body, so one call
is one dispatch. ``sequential_range`` is the construct that admits a loop-carried
dependency; ``nl.affine_range`` would assert the absence of exactly the
dependency this algorithm is built on.

The two reductions
------------------
Sinkhorn reduces along both axes, and in NKI the two directions cost very
different things, so each pass uses the primitive that suits it:

* Row sums reduce along the free axis: ``nl.sum(..., axis=1)``, and the reciprocal
  broadcasts back along the free axis from an ``[M, 1]`` tile, which is what
  ``nisa.tensor_scalar`` does with ``operand0``.
* Column sums reduce along the partition axis, which no elementwise engine does.
  The form is a matmul against a ones vector:
  ``nisa.nc_matmul(stationary=ones[M, 1], moving=A[M, N])`` contracts the
  partition axis and lands ``[1, N]`` column sums in PSUM. Scattering that
  ``[1, N]`` scale back over ``M`` partitions is the mirror-image problem, and
  ``nl.broadcast_to`` is the member that broadcasts on the partition axis where
  ``tensor_scalar`` does not.

The alternative shape, transposing the working tile twice per iteration so both
reductions fall on the free axis, costs 2 * :data:`SINKHORN_ITERS`
``nc_transpose`` ops for the same answer.

Serving more rows than one partition tile holds
-----------------------------------------------
``M`` is the token axis and runs to ``tokens * hc_mult``, far past the 128
partitions one tile has, so the kernel walks ``M`` in tiles inside the one
dispatch. The answer does not change: the row pass is already independent per
row, and the column sum stays a sum over every row of the matrix, accumulated in
one PSUM tile with ``accumulate=False`` on the first tile and ``True`` on the
rest.

Each iteration is therefore three passes over the tiles, and their order is what
keeps the answer identical to an untiled one: every tile is row-scaled, then the
column sums are accumulated over all tiles, then every tile is column-scaled by
the single ``[1, N]`` scale. A per-tile column scale would normalise 128 rows at
a time and compute a different matrix.

Tiles are cut on block boundaries. The mHC affinity matrix is block-diagonal:
token ``t`` owns rows and columns ``t * S`` to ``t * S + S - 1`` where ``S`` is
``hc_mult``. :func:`row_tile_extent` therefore rounds the tile height down to a
multiple of ``S``, so no token's ``S x S`` block is split across two tiles. At
``S = 4`` the tile height is the full 128 rows, so the rounding costs nothing
there and keeps the property true at stream counts that do not divide 128.

``M`` has no ceiling. ``N`` does: it stays one tile wide on the moving free axis.
So does the SBUF the working set occupies, since every tile stays live across the
iterations; a geometry too large to allocate fails in the allocator at trace time
rather than returning a wrong answer.

Taking the blocks instead of the block-diagonal matrix
------------------------------------------------------
A block-diagonal matrix's row and column sums are its blocks' row and column
sums: every off-diagonal entry is zero, and a zero stays zero under any row or
column scaling. So normalising ``block_diag(B_1..B_T)`` is exactly the ``T``
independent normalisations of ``B_1..B_T``, and the square matrix carries no
information the blocks do not.

It does carry cost. ``(T*S)^2`` fp32 values is 256 MB at 2048 tokens with
``S = 4``, against 128 KB for the blocks, and ``N = T * S`` rides the tensor
engine's moving free axis, which caps ``T`` at ``MOVING_FMAX // S``, or 128
tokens. :func:`sinkhorn_blocks_kernel` therefore takes ``[T, S, S]`` directly:
``T`` rides the partition axis in tiles walked inside the one dispatch, the
block's two axes are both free, and no extent bound on ``T`` remains.

Both kernels stay. The square one is the general ``[M, N]`` normalisation; the
batched one is what the mHC layer calls. They share the targets, the denominator
guard, the oracle and the dispatch counters, so no reading drifts between them.

Precision and the denominator guard
-----------------------------------
The working tile, both PSUM tiles and the returned tensor are fp32. The matrix
entries are ``O(1/N)``, and bf16's roughly three decimal digits could not express
agreement at ``atol=1e-5`` at all. fp32 is also sufficient rather than merely
chosen: the iteration is a contraction toward its fixed point, so a rounding
difference introduced at iteration ``k`` is damped by the iterations after it
instead of accumulating.

:data:`SINKHORN_DENOM_EPS` is added to both denominators as a divide-by-zero
guard for a degenerate all-zero row or column. It is numerically inert at the
scales this kernel runs on -- ``1e-30`` against sums of order ``1`` and ``M / N``
is 23 orders below fp32's resolution -- and it is applied identically in the
kernel and in the oracle, so it cannot manufacture a disagreement between them.
Callers are expected to supply strictly positive affinities; the guard turns an
undefined result into a finite one rather than pretending a zero row is
meaningful.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: The target's mHC stream count (``hc_mult``), and therefore the affinity
#: matrix's column extent. A named constant because both halves of mHC and the
#: layer wiring are sized by the same number; the kernels do not hardcode it.
MHC_STREAMS = 4

#: The normalisation iteration count, a trace-time constant so the whole loop
#: unrolls inside one dispatch.
SINKHORN_ITERS = 20

#: Divide-by-zero guard, added to both denominators. Inert at fp32; see the module
#: docstring.
SINKHORN_DENOM_EPS = 1e-30

#: Partition-axis bound, from ``nl.tile_size.pmax``. This bounds one row tile and
#: not ``M``: the kernel walks ``M`` in tiles of at most this height, so a matrix
#: with more rows is served rather than refused.
PARTITION_MAX = 128

#: Tensor engine moving free bound, from ``nl.tile_size.gemm_moving_fmax``. The
#: column-sum matmul carries ``N`` on the moving free axis.
MOVING_FMAX = 512

__all__ = [
    "MHC_STREAMS",
    "MOVING_FMAX",
    "PARTITION_MAX",
    "SINKHORN_DENOM_EPS",
    "SINKHORN_ITERS",
    "SinkhornError",
    "blocks_kernel_identity",
    "can_run_sinkhorn",
    "can_run_sinkhorn_blocks",
    "column_target",
    "dispatch_counters",
    "kernel_identity",
    "reset_dispatch_counters",
    "row_target",
    "row_tile_extent",
    "row_tiles",
    "sinkhorn_blocks_kernel",
    "sinkhorn_kernel",
    "sinkhorn_normalise",
    "sinkhorn_normalise_blocks",
    "sinkhorn_torch_oracle",
]


class SinkhornError(ValueError):
    """A geometry or dtype this module refuses, named rather than coerced.

    Raised in preference to letting NKI trap at trace time, because a refusal that
    names the offending extent is what a caller can act on. A geometry the kernel
    cannot serve raises rather than routing to the torch oracle.
    """


def row_target() -> float:
    """Every row's target sum: ``1.0``.

    A function rather than a bare constant so that the kernels, the oracle and any
    caller take both targets from one place and they cannot drift apart.
    """
    return 1.0


def _column_target_unchecked(rows: int, cols: int) -> float:
    """:func:`column_target`'s arithmetic with no refusal in it, for the kernels.

    NKI traces into every plain python helper a kernel body calls, and its tracer
    rejects ``raise``: "NKI does not support 'raise' statements; use 'if/else'
    control flow within kernels, or 'assert' for fatal errors". A kernel that
    called :func:`column_target` therefore fails to specialise. So the arithmetic
    lives here, the refusal stays in the public wrapper, and the two cannot drift
    because the wrapper computes nothing of its own.

    A caller outside a kernel body wants :func:`column_target`. This function
    trusts its arguments completely.
    """
    return rows / cols


def column_target(rows: int, cols: int) -> float:
    """Every column's target sum: ``rows / cols``.

    The rectangular generalisation of double stochasticity. With row sums at ``1``
    the total mass is ``rows``, so spreading it evenly over ``cols`` columns puts
    ``rows / cols`` in each; the two targets therefore agree on the total, which is
    what makes them a consistent pair.

    This is the checked path. Kernel bodies call
    :func:`_column_target_unchecked` instead, for the tracer reason written there;
    both return the same number.
    """
    if cols <= 0:
        raise SinkhornError(f"cols={cols} must be positive")
    return _column_target_unchecked(rows, cols)


def _row_tile_extent_unchecked(block: int) -> int:
    """:func:`row_tile_extent`'s arithmetic with no refusal in it, for the kernels.

    Same reason as :func:`_column_target_unchecked`: both of
    :func:`row_tile_extent`'s refusals are ``raise`` statements, which a traced
    kernel cannot carry. The checked wrapper returns exactly what this returns.
    """
    return (PARTITION_MAX // block) * block


def row_tile_extent(block: int = MHC_STREAMS) -> int:
    """How many rows one tile carries: ``PARTITION_MAX`` rounded down to ``block``.

    ``block`` is ``hc_mult``, the mHC stream count and the height of one token's
    ``S x S`` affinity block. Rounding down to a multiple of it keeps a block inside
    a single tile, because then every tile starts at a row index that is a multiple
    of ``block``.

    At ``hc_mult = 4`` the answer is 128, the full partition extent, so the
    alignment costs no rows. At a stream count that does not divide 128 the tile
    gives up the remainder instead of splitting a block.

    Raises:
        SinkhornError: if ``block`` is not positive, or is taller than one tile
            can hold, in which case no tile height satisfies the alignment.
    """
    if block < 1:
        raise SinkhornError(f"block={block} must be positive")
    if block > PARTITION_MAX:
        raise SinkhornError(
            f"block={block} exceeds PARTITION_MAX={PARTITION_MAX}; one token's "
            f"block must fit inside a single row tile, and no tile can be taller "
            f"than the partition axis"
        )
    return _row_tile_extent_unchecked(block)


def _row_tiles_unchecked(rows: int, block: int) -> list[tuple[int, int]]:
    """:func:`row_tiles`'s tile list with no refusal in it, for the kernels.

    Two things are absent because the NKI tracer, which follows this call from a
    kernel body, rejects both: a ``raise`` statement, and a list comprehension
    (reported as "unsupported expression"). The loop below therefore uses only
    ``for`` over ``range``, ``append``, a tuple and an ``if``/``else``, and no
    ``min`` -- its branch and ``min(height, rows - start)`` agree for every input.

    The kernels read the returned list by index under ``for idx in range(...)``
    rather than walking it, because a ``for`` whose loop variable is a tuple is
    refused too.
    """
    height = _row_tile_extent_unchecked(block)
    tiles = []
    for start in range(0, rows, height):
        remaining = rows - start
        if remaining < height:
            tiles.append((start, remaining))
        else:
            tiles.append((start, height))
    return tiles


def _row_tile_count_unchecked(rows: int, block: int) -> int:
    """How many row tiles :func:`_row_tiles_unchecked` returns, by arithmetic.

    This exists to be a kernel loop bound. A kernel that walked the tile list
    directly is refused with ``expecting simple variable``, once per ``for``
    statement whose loop variable is a tuple, so the kernels count with
    ``for idx in range(bound)`` instead. The bound must be a plain name rather than
    a call, which is why this is computed here and not written ``len(tiles)`` at
    the call site.

    The ceiling division matches the list's own length for every input.
    """
    height = _row_tile_extent_unchecked(block)
    return (rows + height - 1) // height


def row_tiles(rows: int, block: int = MHC_STREAMS) -> list[tuple[int, int]]:
    """The ``(start, height)`` row tiles the kernel walks, in order.

    The last tile is short whenever ``rows`` is not a multiple of the tile height,
    which is admitted: ``M`` need not be a whole number of blocks either.

    This is the checked path, and the :func:`row_tile_extent` call below is here
    for its refusals. Kernel bodies call :func:`_row_tiles_unchecked`, which
    computes the same list.
    """
    row_tile_extent(block)
    return _row_tiles_unchecked(rows, block)


@nki.jit
def sinkhorn_kernel(affinity, iters: int = SINKHORN_ITERS, block: int = MHC_STREAMS):
    """Doubly stochastic normalisation of ``affinity[M, N]``, in NKI.

    Args:
        affinity: ``[M, N]`` strictly positive affinities in HBM. ``M`` is walked
            in row tiles of :func:`row_tile_extent` rows, so it has no ceiling;
            ``N`` is the free extent and the moving free extent of the column-sum
            matmul, so it stays one tile wide.
        iters: normalisation iterations, a trace-time constant. Defaults to
            :data:`SINKHORN_ITERS`. It is a parameter so a caller can build the
            same algorithm at another iteration count, never so the iteration can
            be driven from the host.
        block: the mHC stream count ``S``, a trace-time constant. It sets the tile
            height through :func:`row_tile_extent` so that no token's ``S x S``
            block is split across two tiles. Defaults to :data:`MHC_STREAMS`; the
            mHC layer passes its own ``hc_mult``.

    Returns:
        ``[M, N]`` fp32, row sums ``1`` and column sums ``M / N``.

    The iteration loop is ``nl.sequential_range`` because every iteration reads
    what the previous one wrote, rewriting each working tile in place. The tile
    loops inside it are ordinary python loops over a trace-time tile list.
    """
    m_extent, n_extent = affinity.shape
    col_goal = m_extent / n_extent
    row_goal = 1.0
    # The unchecked core, because the tracer follows this call and NKI refuses a
    # traced `raise`. `_require_admissible` refuses an inadmissible shape before
    # any dispatch reaches here.
    tiles = _row_tiles_unchecked(int(m_extent), block)
    # The loop bound, as a plain name: every tile loop below counts rather than
    # walking `tiles`, because a `for` whose variable is a tuple is refused.
    tile_count = _row_tile_count_unchecked(int(m_extent), block)

    out = nl.ndarray((m_extent, n_extent), dtype=nl.float32, buffer=nl.shared_hbm)

    # The ones vector that turns a partition-axis reduction into a matmul. Built
    # once at the tallest tile's height and sliced per tile: it is loop-invariant
    # and every entry is 1, so a short tile contracts a prefix of it.
    first_tile = tiles[0]
    ones_col = nl.ndarray((first_tile[1], 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_col, value=1.0)

    # One working tile per row tile, upcast to fp32 on the load. These stay live
    # across the whole iteration loop because the iteration is loop-carried: tile
    # t's next row pass reads what its own last column pass wrote.
    working = []
    row_den = []
    row_scale = []
    for idx in range(tile_count):
        tile_geom = tiles[idx]
        start = tile_geom[0]
        height = tile_geom[1]
        tile = nl.ndarray((height, n_extent), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=tile,
            src=nl.load(affinity[start:start + height, 0:n_extent], dtype=nl.float32),
        )
        working.append(tile)
        row_den.append(nl.ndarray((height, 1), dtype=nl.float32, buffer=nl.sbuf))
        row_scale.append(nl.ndarray((height, 1), dtype=nl.float32, buffer=nl.sbuf))

    # The column tiles are allocated once outside the loop and reused: allocating
    # inside would ask for one live PSUM tile per iteration where one suffices, and
    # PSUM banks are the scarcest resource on the chip. There is one of each for
    # the whole matrix, not one per row tile, because the column sum is a sum over
    # every row.
    col_psum = nl.ndarray((1, n_extent), dtype=nl.float32, buffer=nl.psum)
    col_sum = nl.ndarray((1, n_extent), dtype=nl.float32, buffer=nl.sbuf)
    col_scale = nl.ndarray((1, n_extent), dtype=nl.float32, buffer=nl.sbuf)

    for _ in nl.sequential_range(iters):
        # Row pass: reduce along the free axis, scale rows to row_goal. Per row, so
        # per tile, and the tiles do not interact here.
        for idx in range(tile_count):
            row_sum = nl.sum(working[idx], axis=1, keepdims=True, dtype=nl.float32)
            nisa.tensor_scalar(
                dst=row_den[idx], data=row_sum, op0=nl.add,
                operand0=SINKHORN_DENOM_EPS,
            )
            # Reciprocal then multiply rather than a divide, so the multiply
            # broadcasts an [height, 1] operand along the free axis.
            nisa.reciprocal(dst=row_scale[idx], data=row_den[idx])
            nisa.tensor_scalar(
                dst=row_scale[idx], data=row_scale[idx], op0=nl.multiply,
                operand0=float(row_goal),
            )
            nisa.tensor_scalar(
                dst=working[idx], data=working[idx], op0=nl.multiply,
                operand0=row_scale[idx],
            )

        # Column pass: reduce along the partition axis with the ones matmul, across
        # all tiles into one PSUM tile. `accumulate` is False on the first tile and
        # True on the rest, the form for a contraction longer than the partition
        # axis. False on the first tile is also what clears the previous
        # iteration's sums, since this PSUM tile is reused across iterations.
        for idx in range(tile_count):
            tile_geom = tiles[idx]
            height = tile_geom[1]
            nisa.nc_matmul(
                dst=col_psum,
                stationary=ones_col[0:height, 0:1],
                moving=working[idx],
                accumulate=(idx > 0),
            )
        nisa.tensor_copy(dst=col_sum, src=col_psum)
        nisa.tensor_scalar(
            dst=col_sum, data=col_sum, op0=nl.add, operand0=SINKHORN_DENOM_EPS
        )
        nisa.reciprocal(dst=col_scale, data=col_sum)
        nisa.tensor_scalar(
            dst=col_scale, data=col_scale, op0=nl.multiply, operand0=float(col_goal)
        )
        # One scale for the whole matrix, applied to every tile. Scaling a tile by
        # its own column sums instead would normalise one tile's rows at a time and
        # compute a different matrix. `nl.broadcast_to` is the member that
        # broadcasts on the partition axis.
        for idx in range(tile_count):
            tile_geom = tiles[idx]
            height = tile_geom[1]
            col_scale_b = nl.broadcast_to(col_scale, (height, n_extent))
            nisa.tensor_tensor(
                dst=working[idx], data1=working[idx], data2=col_scale_b,
                op=nl.multiply,
            )

    for idx in range(tile_count):
        tile_geom = tiles[idx]
        start = tile_geom[0]
        height = tile_geom[1]
        nl.store(out[start:start + height, 0:n_extent], value=working[idx])
    return out


@nki.jit
def sinkhorn_blocks_kernel(affinity_blocks, iters: int = SINKHORN_ITERS):
    """Normalise ``[T, S, S]`` blocks, each one doubly stochastic on its own.

    Equivalent to running :func:`sinkhorn_kernel` on ``torch.block_diag`` of the
    same blocks, and much cheaper: the square matrix is ``(T*S)^2`` values, 256 MB
    of fp32 at 2048 tokens against 128 KB for the blocks. Every off-diagonal entry
    of it is zero, and a zero stays zero under row and column scaling, so the
    global normalisation is the ``T`` per-block normalisations.

    It also removes the last extent bound. In the square form ``N = T * S`` rides
    the tensor engine's moving free axis, capping ``T`` at ``MOVING_FMAX // S``.
    Here ``T`` rides the partition axis in tiles of at most :data:`PARTITION_MAX`
    tokens, walked inside this one dispatch, and the block's two axes are both
    free, so both normalisations are free-axis reductions: no ones-vector matmul,
    no partition-axis broadcast, and no bound on ``T``.

    Args:
        affinity_blocks: ``[T, S, S]`` strictly positive affinities in HBM.
        iters: normalisation iterations, a trace-time constant as in
            :func:`sinkhorn_kernel`. The loop stays inside this dispatch.

    Returns:
        ``[T, S, S]`` fp32. Every block has row sums :func:`row_target` and column
        sums :func:`column_target`, which for a square block is also 1.
    """
    t_extent, rows_per_block, cols_per_block = affinity_blocks.shape
    row_goal = row_target()
    # The unchecked core, for the tracer reason written on it.
    # `_require_blocks_admissible` is what refuses a bad shape.
    col_goal = _column_target_unchecked(int(rows_per_block), int(cols_per_block))
    # Decode uses one token block per call. Keep larger token batches on the
    # SDK default so prefill and boundary shapes retain the compiler-selected
    # engine placement that existing coverage exercises.
    block_scalar_engine = (
        nisa.vector_engine if int(t_extent) == 1 else nisa.unknown_engine
    )

    out = nl.ndarray(
        (t_extent, rows_per_block, cols_per_block),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )

    # The token tiles. `block=1` is not a special case: with the S x S block held
    # in the two free axes, a token is one partition row, so no alignment is needed
    # and the tile is the whole partition extent. The arithmetic is `row_tiles`'s,
    # so both kernels tile the partition axis one way.
    tiles = _row_tiles_unchecked(int(t_extent), 1)
    # The loop bound, as a plain name, for the reason written on the square
    # kernel's own bound.
    tile_count = _row_tile_count_unchecked(int(t_extent), 1)

    # Per token tile: one working tile per block ROW, plus that row's own
    # denominator and scale, plus one column accumulator for the whole tile. All
    # allocated before the iteration loop, because the iteration is loop-carried.
    work: list[list] = []
    row_den: list[list] = []
    row_scale: list[list] = []
    col_sum = []
    col_scale = []
    for idx in range(tile_count):
        tile_geom = tiles[idx]
        start = tile_geom[0]
        height = tile_geom[1]
        tile_rows = []
        den_rows = []
        scale_rows = []
        for i in range(rows_per_block):
            tile = nl.ndarray(
                (height, cols_per_block), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(
                dst=tile,
                src=nl.load(
                    affinity_blocks[start:start + height, i, 0:cols_per_block],
                    dtype=nl.float32,
                ),
            )
            tile_rows.append(tile)
            den_rows.append(
                nl.ndarray((height, 1), dtype=nl.float32, buffer=nl.sbuf)
            )
            scale_rows.append(
                nl.ndarray((height, 1), dtype=nl.float32, buffer=nl.sbuf)
            )
        work.append(tile_rows)
        row_den.append(den_rows)
        row_scale.append(scale_rows)
        col_sum.append(
            nl.ndarray((height, cols_per_block), dtype=nl.float32, buffer=nl.sbuf)
        )
        col_scale.append(
            nl.ndarray((height, cols_per_block), dtype=nl.float32, buffer=nl.sbuf)
        )

    for _ in nl.sequential_range(iters):
        for idx in range(tile_count):
            # Row pass. Block row i of every token in this tile is one [tokens, S]
            # tile, so its row sum is a free-axis reduction and the reciprocal
            # broadcasts back along the free axis, as in the square kernel.
            for i in range(rows_per_block):
                r_sum = nl.sum(
                    work[idx][i], axis=1, keepdims=True, dtype=nl.float32
                )
                nisa.tensor_scalar(
                    dst=row_den[idx][i], data=r_sum, op0=nl.add,
                    operand0=SINKHORN_DENOM_EPS, engine=block_scalar_engine,
                )
                nisa.reciprocal(dst=row_scale[idx][i], data=row_den[idx][i])
                nisa.tensor_scalar(
                    dst=row_scale[idx][i], data=row_scale[idx][i],
                    op0=nl.multiply, operand0=float(row_goal),
                    engine=block_scalar_engine,
                )
                nisa.tensor_scalar(
                    dst=work[idx][i], data=work[idx][i], op0=nl.multiply,
                    operand0=row_scale[idx][i], engine=block_scalar_engine,
                )

            # Column pass. A block's column sum runs over its rows, which here are
            # separate tiles, so it is an elementwise add of the S tiles rather than
            # a reduction along any axis: entry j of the accumulator is column j's
            # sum. The first row initialises the accumulator, so no memset pass is
            # needed.
            nisa.tensor_copy(dst=col_sum[idx], src=work[idx][0])
            for i in range(1, rows_per_block):
                nisa.tensor_tensor(
                    dst=col_sum[idx], data1=col_sum[idx], data2=work[idx][i],
                    op=nl.add,
                )
            nisa.tensor_scalar(
                dst=col_sum[idx], data=col_sum[idx], op0=nl.add,
                operand0=SINKHORN_DENOM_EPS, engine=block_scalar_engine,
            )
            nisa.reciprocal(dst=col_scale[idx], data=col_sum[idx])
            nisa.tensor_scalar(
                dst=col_scale[idx], data=col_scale[idx], op0=nl.multiply,
                operand0=float(col_goal), engine=block_scalar_engine,
            )
            for i in range(rows_per_block):
                nisa.tensor_tensor(
                    dst=work[idx][i], data1=work[idx][i], data2=col_scale[idx],
                    op=nl.multiply,
                )

    for idx in range(tile_count):
        tile_geom = tiles[idx]
        start = tile_geom[0]
        height = tile_geom[1]
        for i in range(rows_per_block):
            nl.store(
                out[start:start + height, i, 0:cols_per_block], value=work[idx][i]
            )
    return out


def _require_admissible(rows: int, cols: int, block: int = MHC_STREAMS) -> None:
    """Every extent condition :func:`sinkhorn_kernel` imposes, in one place.

    ``M`` deliberately has no upper bound: the kernel walks it in row tiles, so a
    partition-axis ceiling would refuse the extents the tiling exists to serve.
    What is still checked is that a token's block fits inside one tile, because the
    tiling is only correct if it cuts on block boundaries.
    """
    problems: list[str] = []
    if rows <= 0:
        problems.append(f"M={rows} must be positive")
    if block < 1:
        problems.append(f"block={block} must be positive")
    elif block > PARTITION_MAX:
        problems.append(
            f"block={block} exceeds PARTITION_MAX={PARTITION_MAX}; the row tiles "
            f"are cut on block boundaries so that no token's block is split "
            f"across two tiles, and a block taller than one tile leaves no tile "
            f"height that can do that. This refuses rather than splitting a block "
            f"quietly or routing to a torch path"
        )
    if cols <= 0:
        problems.append(f"N={cols} must be positive")
    elif cols > MOVING_FMAX:
        problems.append(
            f"N={cols} exceeds the Tensor Engine moving free bound "
            f"{MOVING_FMAX}; the column-sum matmul carries N on the moving free "
            f"axis"
        )
    if problems:
        raise SinkhornError(
            "sinkhorn normalisation refuses this geometry: " + "; ".join(problems)
        )


def _require_blocks_admissible(tokens: int, rows: int, cols: int) -> None:
    """Extent conditions for the ``[T, S, S]`` form: two of them, and no ceiling.

    ``T`` has no upper bound because it rides the partition axis in tiles walked
    inside the kernel. Nor does the block extent: it rides two free axes and never
    reaches the tensor engine's moving free axis, so :data:`MOVING_FMAX` has
    nothing to say about it. What bounds the block is SBUF, and a block too large
    to allocate fails in the allocator at trace time rather than returning a wrong
    answer.

    The blocks must be square. Row target ``1`` and column target ``rows / cols``
    describe the same fixed point only when the two extents agree, and mHC's blocks
    are ``S x S`` by construction.
    """
    problems: list[str] = []
    if tokens <= 0:
        problems.append(f"T={tokens} must be positive")
    if rows <= 0:
        problems.append(f"S={rows} must be positive")
    elif rows != cols:
        problems.append(
            f"the block is [{rows}, {cols}] and must be square; each block is one "
            f"token's S x S mHC affinity, and a doubly stochastic block needs its "
            f"row and column extents to agree. This refuses rather than routing "
            f"to a torch path"
        )
    if problems:
        raise SinkhornError(
            "batched sinkhorn normalisation refuses this geometry: "
            + "; ".join(problems)
        )


@dataclass
class _DispatchCounters:
    """Which path actually ran, counted rather than inferred.

    ``nki_dispatch`` counts entries into the ``wrap_nki`` call, ``torch_fallback``
    entries into the torch path. Two counters rather than one flag, so "the kernel
    ran" and "the fallback did not" are independent readings.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: Module level so a caller outside this module can reset and read it. A distinct
#: object from the combine kernel's.
_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _COUNTERS.nki_dispatch += 1


def can_run_sinkhorn(
    affinity: Tensor, rows: int, cols: int, block: int = MHC_STREAMS
) -> bool:
    """Is the NKI path available *and* admissible for this geometry?

    Two independent conditions: ``can_run_kernel`` answers whether a device or
    simulator exists, :func:`_require_admissible` whether this kernel accepts these
    extents. A geometry the kernel cannot serve raises rather than falling back.

    Args:
        affinity: the tensor whose device decides the path.
        rows: ``M``, the row extent. No ceiling; the kernel tiles it.
        cols: ``N``, the column extent.
        block: the mHC stream count the row tiles are aligned to.

    Raises:
        SinkhornError: if the geometry is inadmissible.
    """
    _require_admissible(rows, cols, block)
    return can_run_kernel(affinity)


def can_run_sinkhorn_blocks(
    affinity_blocks: Tensor, tokens: int, rows: int, cols: int
) -> bool:
    """Is the NKI path available *and* admissible for ``[T, S, S]`` blocks?

    The same two conditions as :func:`can_run_sinkhorn`, over
    :func:`_require_blocks_admissible`. Extents are arguments rather than read off
    the tensor so that a caller can ask whether ``T`` tokens are servable before it
    builds the blocks.

    Args:
        affinity_blocks: the tensor whose device decides the path.
        tokens: ``T``, the token extent. No ceiling; the kernel tiles it.
        rows: the block's row extent ``S``.
        cols: the block's column extent, which must equal ``rows``.

    Raises:
        SinkhornError: if the geometry is inadmissible.
    """
    _require_blocks_admissible(tokens, rows, cols)
    return can_run_kernel(affinity_blocks)


def sinkhorn_normalise(
    affinity: Tensor, iters: int = SINKHORN_ITERS, block: int = MHC_STREAMS
) -> Tensor:
    """Doubly stochastic normalisation of ``[M, N]``: one kernel dispatch per call.

    Args:
        affinity: ``[M, N]`` strictly positive affinities.
        iters: normalisation iterations, default :data:`SINKHORN_ITERS`. Passed to
            the kernel as a trace-time constant, so the loop stays inside the one
            dispatch.
        block: the mHC stream count ``S``, default :data:`MHC_STREAMS`. The row
            tiles are cut on multiples of it, so no token's ``S x S`` block is
            split. One call is still one dispatch however many tiles the row extent
            needs, since the tiling is inside the kernel.

    Returns:
        ``[M, N]`` fp32, row sums :func:`row_target` and column sums
        :func:`column_target`.

    Raises:
        SinkhornError: on an inadmissible geometry, a non-2-D input, or a
            non-positive ``iters``.
    """
    if affinity.dim() != 2:
        raise SinkhornError(
            f"affinity must be 2-D [M, N], got shape {tuple(affinity.shape)}; "
            f"the kernel maps M onto the partition axis and N onto the free axis"
        )
    if iters <= 0:
        raise SinkhornError(
            f"iters={iters} must be positive; the target declares "
            f"{SINKHORN_ITERS} normalisation iterations"
        )
    rows, cols = int(affinity.shape[0]), int(affinity.shape[1])

    if not can_run_sinkhorn(affinity, rows, cols, block):
        _count_torch_fallback()
        logger.debug(
            "sinkhorn_normalise: NKI route unavailable, using the torch path "
            "(oracle only, not the shipped path)"
        )
        return sinkhorn_torch_oracle(affinity, iters=iters)

    _count_nki_dispatch()
    return wrap_nki(sinkhorn_kernel)(affinity=affinity, iters=iters, block=block)


def sinkhorn_normalise_blocks(
    affinity_blocks: Tensor, iters: int = SINKHORN_ITERS
) -> Tensor:
    """Normalise ``[T, S, S]`` blocks: the entry point the mHC layer calls.

    This has no torch path, where :func:`sinkhorn_normalise` returns the oracle
    when the NKI path is unavailable. The difference is deliberate: an mHC layer
    that silently normalised 2048 tokens in torch would be slow in a way no numeric
    check could see, so an absent path raises here and ``torch_fallback`` stays
    zero however this is called.

    Args:
        affinity_blocks: ``[T, S, S]`` strictly positive affinities, one square
            block per token.
        iters: normalisation iterations, default :data:`SINKHORN_ITERS`, passed to
            the kernel as a trace-time constant. One call is one dispatch however
            many token tiles ``T`` needs.

    Returns:
        ``[T, S, S]`` fp32. Each block has row sums :func:`row_target` and column
        sums :func:`column_target`, which for a square block is also ``1``.

    Raises:
        SinkhornError: on a non-3-D input, a non-positive ``iters``, an
            inadmissible geometry, or an unavailable NKI route.
    """
    if affinity_blocks.dim() != 3:
        raise SinkhornError(
            f"affinity_blocks must be 3-D [T, S, S], got shape "
            f"{tuple(affinity_blocks.shape)}; the kernel maps T onto the partition "
            f"axis and the block onto the two free axes"
        )
    if iters <= 0:
        raise SinkhornError(
            f"iters={iters} must be positive; the target declares "
            f"{SINKHORN_ITERS} normalisation iterations"
        )
    tokens = int(affinity_blocks.shape[0])
    rows = int(affinity_blocks.shape[1])
    cols = int(affinity_blocks.shape[2])

    if not can_run_sinkhorn_blocks(affinity_blocks, tokens, rows, cols):
        raise SinkhornError(
            "the NKI route is unavailable (no device and no simulator), and the "
            "batched sinkhorn has no torch path to fall back to: normalising "
            f"{tokens} token blocks has no torch implementation. Enable the NKI "
            "simulator for a CPU-mode run."
        )

    _count_nki_dispatch()
    return wrap_nki(sinkhorn_blocks_kernel)(
        affinity_blocks=affinity_blocks, iters=iters
    )


def sinkhorn_torch_oracle(affinity: Tensor, iters: int = SINKHORN_ITERS) -> Tensor:
    """The same algorithm in torch, in fp32. The CPU oracle, never shipped.

    Independent of the kernel in the two ways that matter. It reduces with
    ``Tensor.sum`` along each axis, so it touches neither the ones-vector matmul the
    kernel uses for its column sums nor the partition-axis broadcast that scatters
    the result, which are the two places a partition-axis mistake could hide. And
    it divides where the kernel takes a reciprocal and multiplies, so the two round
    differently.

    It shares one thing with the kernel on purpose: :data:`SINKHORN_DENOM_EPS`,
    applied to both denominators, so the guard cannot manufacture a disagreement.

    Returns:
        ``[M, N]`` fp32.
    """
    rows, cols = int(affinity.shape[0]), int(affinity.shape[1])
    col_goal = column_target(rows, cols)
    row_goal = row_target()

    working = affinity.to(torch.float32).clone()
    for _ in range(iters):
        working = working * (
            row_goal / (working.sum(dim=1, keepdim=True) + SINKHORN_DENOM_EPS)
        )
        working = working * (
            col_goal / (working.sum(dim=0, keepdim=True) + SINKHORN_DENOM_EPS)
        )
    return working


def kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the square kernel, read off the object.

    Lets a caller check that this module dispatches to the kernel it authors, so a
    substitution shows up as a changed reading.
    """
    func = getattr(sinkhorn_kernel, "func", None)
    target = func if func is not None else sinkhorn_kernel
    return target.__module__, target.__qualname__


def blocks_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the batched kernel, read off the object.

    Kept separate from :func:`kernel_identity` so a caller can tell which of the two
    kernels ran instead of inferring it from a shape. Both unwrap ``.func``, because
    ``nki.jit`` returns a wrapper and the wrapped name is the one a substitution
    would change.
    """
    func = getattr(sinkhorn_blocks_kernel, "func", None)
    target = func if func is not None else sinkhorn_blocks_kernel
    return target.__module__, target.__qualname__
