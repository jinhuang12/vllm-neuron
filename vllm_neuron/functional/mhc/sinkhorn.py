# SPDX-License-Identifier: Apache-2.0
"""Sinkhorn normalisation for mHC: a SCRATCH NKI kernel, authored here.

`inc-glm53f-028`. This is WP8's normalisation half -- the iterative row/column
rescaling that turns a raw mHC affinity matrix into a doubly stochastic one
before `inc-glm53f-029`'s combine kernel mixes the hyper-connection streams.

It is **kernel-class** under P13, and it is **SCRATCH with ZERO precedent**:
the plan's substrate bullet records that ``nkilib`` has **0** hits for
sinkhorn / mhc / hadamard-as-implementation, and that the ``vendored_kernels/``
precedent covers version-lag vendoring only, not missing primitives. **No
"vendoring precedent" claim is available for this increment and none is made.**
There is no vendor kernel to wrap or adapt, so the arithmetic below is authored
in NKI. The torch code in this module is the CPU oracle -- one of the two roles
the plan's substrate register admits (``design/increment-plan.md`` §4) -- and
never the shipped implementation. **An iterative per-token normalisation written
in torch would be exactly the P13 fallback the rule forbids.**

What the kernel computes
------------------------
Given a strictly positive affinity matrix ``A[M, N]``, it runs
:data:`SINKHORN_ITERS` iterations of alternating row-then-column rescaling::

    for _ in range(SINKHORN_ITERS):
        A[i, j] /= sum_j A[i, j]                    # every row sums to 1
        A[i, j] *= column_target(M, N) / sum_i A[i, j]   # every column sums to M/N

and returns the result as fp32. The fixed point is **doubly stochastic** in the
rectangular sense: row sums ``1``, column sums ``M / N``, total mass ``M`` on
both readings. :func:`row_target` and :func:`column_target` are the single place
those two numbers are written, so a consumer and a test read them rather than
restating them.

Why the twenty iterations are INSIDE the kernel
-----------------------------------------------
The loop is a ``nl.sequential_range`` **in the kernel body**, so one call to the
:func:`sinkhorn_normalise` seam is one dispatch. That is a declared, counted
property rather than a stylistic one: the plan's route predicate reads **`1`
dispatch per declared case -- `1`, not `20`** -- precisely so that a host-driven
iteration loop, which would read `20`, is told apart from this design by the
instrument. ``nl.sequential_range`` is the construct that admits a
loop-carried dependency (the repository's own precedent is
``functional/argsort_unstable.py:206``, whose passes rewrite their input tile in
place); ``nl.affine_range`` would assert the absence of exactly the dependency
this algorithm is built on.

The two reductions, and why neither needs a partition-axis broadcast trick
-------------------------------------------------------------------------
Sinkhorn reduces along both axes, and in NKI those two directions cost very
different things. Each pass is therefore taken with the primitive that suits it:

* **row sums reduce along the FREE axis** -- ``nl.sum(..., axis=1)``, and the
  reciprocal broadcasts back along the free axis from an ``[M, 1]`` tile, which
  is exactly what ``nisa.tensor_scalar`` does with ``operand0``. This is
  ``functional/moe/router.py:1222-1236``'s idiom, reused rather than reinvented.
* **column sums reduce along the PARTITION axis**, which no elementwise engine
  does. The canonical NKI form is a matmul against a ones vector:
  ``nisa.nc_matmul(stationary=ones[M, 1], moving=A[M, N])`` contracts over the
  partition axis and lands ``[1, N]`` column sums in PSUM --
  ``functional/moe/topk_reduce.py:337-348``'s own "column-sum via a ones-moving
  matmul". Scattering the ``[1, N]`` scale back over ``M`` partitions is the
  mirror-image problem, and ``nl.broadcast_to`` is the member that does it:
  ``moe/router.py:1268-1269`` records that a ``[1, E]`` row's broadcast "is on the
  PARTITION axis, which ``nl.broadcast_to`` does and ``tensor_scalar`` does
  not."

Every primitive above is attested at a landed line of this repository, fp32
included -- ``nisa.nc_matmul`` runs on an explicitly fp32 operand at
``functional/vendored_kernels/rotational_topk/rotational_topk.py:384`` (the
``rotation_f32`` path). The alternative shape, transposing the working tile
twice per iteration so that both reductions fall on the free axis, was
considered and rejected: it costs 40 ``nc_transpose`` ops for the same answer.

Serving more rows than one partition tile holds
-----------------------------------------------
`inc-glm53f-028b`. ``M`` is the token axis, and serving needs prefill extents of
2048 tokens and more, so ``M`` runs to ``2048 * hc_mult`` and far past the 128
partitions one tile has. The kernel therefore walks ``M`` in tiles **inside the
kernel**. Nothing about the answer changes: the row pass is per row and so is
already independent per tile, and the column sum stays a sum over **every** row
of the matrix, accumulated in one PSUM tile across the tiles with
``accumulate=False`` on the first tile and ``True`` on the rest. That is this
repository's own landed form for a contraction longer than 128 --
``functional/attention/mla_projections.py:204-209`` -- reused rather than
invented, and it is why the targets :func:`row_target` and :func:`column_target`
are unchanged and the oracle is untouched.

Each iteration is three passes over the tiles rather than one, and the order is
what makes the answer identical to the untiled one: every tile is row-scaled
first, then the column sums are accumulated over all tiles, then every tile is
column-scaled by the one ``[1, N]`` scale. A per-tile column scale would
normalise 128 rows at a time and compute a different matrix.

**Tiles are cut on block boundaries.** The mHC affinity matrix is
block-diagonal: token ``t`` owns rows and columns ``t * S`` to ``t * S + S - 1``
where ``S`` is ``hc_mult`` (``model_fp8.py``'s ``mhc_pre`` builds it with
``torch.block_diag``). :func:`row_tile_extent` therefore rounds the tile height
DOWN to a multiple of ``S``, so every tile starts at a block boundary and no
token's ``S x S`` block is split across two tiles. At ``S = 4`` the tile height
is the full 128 rows; the rounding costs nothing there and keeps the property
true at stream counts that do not divide 128.

``M`` no longer has a declared ceiling, and that absence is the point of this
increment -- the same reading ``mla_projections.py:216-224`` records for its own
tiled axes. What remains bounded is ``N``, still one tile wide on the moving free
axis, and the SBUF the working set occupies: every tile stays live across the
iterations because each iteration reads what the last one wrote, so a geometry
too large to allocate fails at trace time in the allocator rather than returning
a wrong answer.

Taking the blocks instead of the block-diagonal matrix
-----------------------------------------------------
`inc-glm53f-028b`, second form. The square matrix above is **block-diagonal**, and
a block-diagonal matrix's row sums and column sums ARE its blocks' row sums and
column sums -- every off-diagonal entry is zero, and a zero stays zero under any
row or column scaling. So the global normalisation of ``block_diag(B_1..B_T)`` is
exactly the ``T`` independent normalisations of ``B_1..B_T``, and the square
matrix carries no information the blocks do not.

It does carry cost. ``(T*S)^2`` fp32 values is 256 MB at 2048 tokens with
``S = 4``, against 128 KB for the blocks, and ``N = T * S`` rides the Tensor
Engine's moving free axis, which caps ``T`` at ``MOVING_FMAX // S`` -- 128 tokens.
:func:`sinkhorn_blocks_kernel` therefore takes ``[T, S, S]`` directly. ``T`` rides
the PARTITION axis in tiles walked inside the one dispatch, the block's two axes
are both FREE, both normalisations become free-axis work, and no extent bound on
``T`` remains. The equivalence is a test, not a claim: ``T`` in ``{1, 3, 33}`` is
compared block-for-block against this module's own square kernel run on
``torch.block_diag`` of the same blocks.

Both kernels stay. The square one is the general ``[M, N]`` normalisation and the
acceptance case the plan declares; the batched one is what the mHC layer will
call once `inc-glm53f-030b` switches it over. They share the targets, the
denominator guard, the oracle and the dispatch counters, so no reading drifts
between them.

Precision, stated rather than implied
-------------------------------------
The working tile, both PSUM tiles and the returned tensor are **fp32**. This is
not a default: the acceptance compares against a torch oracle at ``atol=1e-5``
while the matrix entries are ``O(1/N)``, and bf16's ~3 decimal digits could not
express that difference at all. A normalisation kernel whose acceptance measures
precision does not throw precision away internally.

Sinkhorn is also self-correcting in a way that is worth naming, because it is
why fp32 is *sufficient* rather than merely chosen: the iteration is a
contraction toward its fixed point, so a rounding difference introduced at
iteration ``k`` is damped by the iterations after it instead of accumulating.
The kernel and the oracle therefore agree far inside the declared tolerance even
though they execute different instruction sequences.

The denominator guard
---------------------
:data:`SINKHORN_DENOM_EPS` is added to both denominators. It is a
divide-by-zero guard for a degenerate all-zero row or column, and it is
**numerically inert** at the scales this kernel runs on: ``1e-30`` against sums
of order ``1`` and ``M / N`` is 23 orders below fp32's ``~1.2e-7`` resolution.
It is applied **identically in the kernel and in the oracle**, so it cannot
manufacture a disagreement between them. Callers are expected to supply strictly
positive affinities; the guard converts an undefined result into a finite one
rather than pretending a zero row is meaningful.

Route
-----
Acceptance is Tier N: the NKI simulator, reached through this module's own
:func:`sinkhorn_normalise` seam (``wrap_nki -> NKIHOPCaller -> HOP ->
DispatchKey.CPU -> nki.simulator.simulate_kernel``). The seam counts its
dispatches, and the counters are module-level state with module-level reset and
read functions **on purpose**: `inc-glm53f-030`'s route predicate is form R-2
over *this* seam together with `inc-glm53f-029`'s, so a later increment's own
test must be able to zero and read these counters from another module. A
test-local counter would satisfy this increment and break that one. This mirrors
`inc-glm53f-026`'s landed placement (``functional/blockwise_fp8_mm.py:368-372``)
deliberately, so the two seams `inc-glm53f-030` reads present one shape.

Under F1 a numeric comparison alone cannot prove a kernel ran -- a torch
fallback would put torch on both sides of the comparison and pass green -- so
the counters below are acceptance criteria, not diagnostics.
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

#: The target's mHC stream count (``hc_mult 4``), and therefore the affinity
#: matrix's column extent. Recorded as a named constant because
#: `inc-glm53f-029`'s combine kernel and `inc-glm53f-030`'s layer wiring are
#: sized by the same number; the kernel itself does not hardcode it.
MHC_STREAMS = 4

#: The target's normalisation iteration count. The plan declares **20**, and it
#: is a trace-time constant so the whole loop unrolls INSIDE one dispatch.
SINKHORN_ITERS = 20

#: Divide-by-zero guard, added to both denominators. Inert at fp32 -- see the
#: module docstring's "denominator guard" section.
SINKHORN_DENOM_EPS = 1e-30

#: Partition-axis bound, from ``nl.tile_size.pmax``. This bounds ONE ROW TILE,
#: not ``M``: `inc-glm53f-028b` walks ``M`` in tiles of at most this height, so a
#: matrix with more rows than this is served rather than refused. Written as a
#: module constant so the tile arithmetic, the refusals below and any consumer
#: read one number.
PARTITION_MAX = 128

#: Tensor Engine moving free bound, from ``nl.tile_size.gemm_moving_fmax``. The
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

    Raised in preference to letting NKI trap at trace time: a refusal that names
    the offending extent is what a caller can act on. Refusing is also what P13
    requires here -- a geometry this kernel cannot serve must NOT quietly route
    to the torch oracle, because that would ship a torch path for kernel-class
    work (D6).
    """


# --------------------------------------------------------------------------- #
# The two targets, written once.                                              #
# --------------------------------------------------------------------------- #
def row_target() -> float:
    """Every row's target sum: ``1.0``.

    A function rather than a bare constant so that the kernel, the oracle, the
    acceptance's doubly-stochastic reading and `inc-glm53f-030` all take the
    number from one place. The plan's expected result is stated per axis
    ("within 1e-3 of *its* target"), so the two targets must not drift apart.
    """
    return 1.0


def _column_target_unchecked(rows: int, cols: int) -> float:
    """:func:`column_target`'s arithmetic with no refusal in it, for the kernels.

    WHY THIS EXISTS, and it is not a style choice. NKI traces INTO every plain
    Python helper a kernel body calls, and its tracer rejects ``raise``: the
    compiler's own words are "NKI does not support 'raise' statements; use
    'if/else' control flow within kernels, or 'assert' for fatal errors". A
    kernel that called :func:`column_target` therefore failed to specialize --
    measured, not predicted, on all eleven declared Tier C shapes under lease
    grant 083 (``increments/run028b-r2-trn2-1-at-8de8b588-20260908T161823Z.out``,
    where the compiler counted exactly the raises this module's traced helpers
    hold). So the arithmetic lives here, the refusal stays in the public wrapper,
    and the two cannot drift because the wrapper computes nothing of its own.

    A caller outside a kernel body wants :func:`column_target`. This function
    trusts its arguments completely.
    """
    return rows / cols


def column_target(rows: int, cols: int) -> float:
    """Every column's target sum: ``rows / cols``.

    The rectangular generalisation of double stochasticity. With row sums at
    ``1`` the total mass is ``rows``, so spreading that mass evenly over ``cols``
    columns puts ``rows / cols`` in each -- and the two readings agree on the
    total, which is what makes them a consistent pair of targets rather than two
    independent wishes. For the declared ``[64, 4]`` case this is ``16.0``.

    This is the checked path, for the seam and for tests. Kernel bodies call
    :func:`_column_target_unchecked` instead, for the tracer reason written
    there; the number both return is the same number.
    """
    if cols <= 0:
        raise SinkhornError(f"cols={cols} must be positive")
    return _column_target_unchecked(rows, cols)


# --------------------------------------------------------------------------- #
# The row tiling, written once and read by the kernel, the refusals and a test. #
# --------------------------------------------------------------------------- #
def _row_tile_extent_unchecked(block: int) -> int:
    """:func:`row_tile_extent`'s arithmetic with no refusal in it, for the kernels.

    Same reason as :func:`_column_target_unchecked`: both of
    :func:`row_tile_extent`'s refusals are ``raise`` statements, and a kernel that
    traced them did not compile. The rounding is written once, here, and the
    checked wrapper returns exactly what this returns.
    """
    return (PARTITION_MAX // block) * block


def row_tile_extent(block: int = MHC_STREAMS) -> int:
    """How many rows one tile carries: ``PARTITION_MAX`` rounded down to ``block``.

    ``block`` is ``hc_mult`` -- the mHC stream count, and the height of one
    token's ``S x S`` affinity block. Rounding the tile height down to a multiple
    of it is what keeps a block inside a single tile, because then every tile
    starts at a row index that is a multiple of ``block``.

    At the target's ``hc_mult 4`` the answer is 128, the full partition extent:
    4 divides 128, so the alignment costs no rows. At a stream count that does
    not divide 128 -- 3, 5, 6 -- the tile gives up the remainder instead of
    splitting a block.

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

    TWO THINGS ARE DELIBERATELY ABSENT and both were named by the compiler. The
    ``raise`` statements are gone because they reach here through
    :func:`row_tile_extent`, and the list comprehension is gone because the same
    refusal carried one "unsupported expression" on every shape -- one, on both
    kernels, where the ``f``-strings would have given two on the square path and
    three on the batched one. The loop below therefore uses only forms the tracer
    accepted in the kernel bodies themselves on that same run -- ``for`` over
    ``range``, ``append``, a tuple, and the ``if``/``else`` the compiler's own
    error text recommends -- and it uses no ``min``, which appeared exactly once
    in this module and only inside that comprehension. Which of the two was the
    unsupported expression is not settled here; neither survives, so neither can
    refuse the trace again.

    The arithmetic is unchanged: ``min(height, rows - start)`` and the branch
    below agree for every input.
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


def row_tiles(rows: int, block: int = MHC_STREAMS) -> list[tuple[int, int]]:
    """The ``(start, height)`` row tiles the kernel walks, in order.

    A function rather than a loop written twice, so the kernel and
    :mod:`test_sinkhorn`'s alignment reading take the same arithmetic. The last
    tile is short whenever ``rows`` is not a multiple of the tile height -- which
    is admitted, because ``M`` need not be a whole number of blocks either
    (``M = 129`` is one of the declared acceptance cases).

    This is the checked path. The call to :func:`row_tile_extent` below is here
    for its two refusals: a caller on the seam or in a test that asks for an
    impossible ``block`` is told so, exactly as before. Kernel bodies call
    :func:`_row_tiles_unchecked`, which computes the same list from the same
    core.
    """
    row_tile_extent(block)
    return _row_tiles_unchecked(rows, block)


# --------------------------------------------------------------------------- #
# The NKI kernel. SCRATCH: nkilib provides no sinkhorn member at any shape.     #
# --------------------------------------------------------------------------- #
@nki.jit
def sinkhorn_kernel(affinity, iters: int = SINKHORN_ITERS, block: int = MHC_STREAMS):
    """Doubly stochastic normalisation of ``affinity[M, N]``, in NKI.

    Args:
        affinity: ``[M, N]`` strictly positive affinities in HBM. ``M`` is walked
            in row tiles of :func:`row_tile_extent` rows, so it has no ceiling;
            ``N`` is the free extent and the moving free extent of the column-sum
            matmul, so it stays one tile wide.
        iters: normalisation iterations, a **trace-time** constant. Defaults to
            :data:`SINKHORN_ITERS`. Exposed so a test can build the same
            algorithm at other iteration counts to show the acceptance's
            threshold is armed -- never so a caller can drive the iteration from
            the host, which is the design the route predicate exists to exclude.
        block: the mHC stream count ``S``, a **trace-time** constant. It sets the
            tile height through :func:`row_tile_extent` so that no token's
            ``S x S`` block is split across two tiles. Defaults to
            :data:`MHC_STREAMS`; `inc-glm53f-030b`'s layer passes its own
            ``hc_mult`` rather than relying on the default.

    Returns:
        ``[M, N]`` fp32, row sums ``1`` and column sums ``M / N``.

    The iteration loop is ``nl.sequential_range`` because every iteration reads
    what the previous one wrote. Each working tile is rewritten in place, which
    ``nisa`` admits and this repository already relies on
    (``functional/moe/router.py:1212``, ``functional/argsort_unstable.py:216``).
    The tile loops inside it are ordinary Python loops over a trace-time tile
    list, the form ``mla_projections.py:187-212`` uses for its own tiled axes.
    """
    m_extent, n_extent = affinity.shape
    col_goal = m_extent / n_extent
    row_goal = 1.0
    # The UNCHECKED core, because the tracer follows this call. The checked
    # `row_tiles` raises, and NKI refuses a traced `raise`: the seam's
    # `_require_admissible` is where an inadmissible shape is refused, before any
    # dispatch reaches here.
    tiles = _row_tiles_unchecked(int(m_extent), block)

    out = nl.ndarray((m_extent, n_extent), dtype=nl.float32, buffer=nl.shared_hbm)

    # The ones vector that turns a partition-axis reduction into a matmul. Built
    # once at the tallest tile's height and sliced per tile: it is loop-invariant
    # and every entry is 1, so a short tile contracts a prefix of it.
    ones_col = nl.ndarray((tiles[0][1], 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_col, value=1.0)

    # One working tile per row tile, upcast to fp32 on the load. These stay live
    # across the whole iteration loop because the iteration is loop-carried: tile
    # t's next row pass reads what its own last column pass wrote.
    working = []
    row_den = []
    row_scale = []
    for start, height in tiles:
        tile = nl.ndarray((height, n_extent), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=tile,
            src=nl.load(affinity[start:start + height, 0:n_extent], dtype=nl.float32),
        )
        working.append(tile)
        row_den.append(nl.ndarray((height, 1), dtype=nl.float32, buffer=nl.sbuf))
        row_scale.append(nl.ndarray((height, 1), dtype=nl.float32, buffer=nl.sbuf))

    # The column tiles are allocated ONCE outside the loop and reused. With 20
    # iterations, allocating inside would ask for 20 live PSUM tiles where 1
    # suffices, and PSUM banks are the scarcest resource on the chip. There is
    # one of each for the WHOLE matrix, not one per row tile, because the column
    # sum is a sum over every row.
    col_psum = nl.ndarray((1, n_extent), dtype=nl.float32, buffer=nl.psum)
    col_sum = nl.ndarray((1, n_extent), dtype=nl.float32, buffer=nl.sbuf)
    col_scale = nl.ndarray((1, n_extent), dtype=nl.float32, buffer=nl.sbuf)

    for _ in nl.sequential_range(iters):
        # ---- row pass: reduce along the FREE axis, scale rows to row_goal ----
        # Per row, so per tile, and the tiles do not interact here.
        for idx, (_start, _height) in enumerate(tiles):
            row_sum = nl.sum(working[idx], axis=1, keepdims=True, dtype=nl.float32)
            nisa.tensor_scalar(
                dst=row_den[idx], data=row_sum, op0=nl.add,
                operand0=SINKHORN_DENOM_EPS,
            )
            # reciprocal then multiply, rather than a divide: `nisa.reciprocal` is
            # the member moe/router.py:1312 uses for exactly this shape, and the
            # multiply below broadcasts an [height, 1] operand along the free axis.
            nisa.reciprocal(dst=row_scale[idx], data=row_den[idx])
            nisa.tensor_scalar(
                dst=row_scale[idx], data=row_scale[idx], op0=nl.multiply,
                operand0=float(row_goal),
            )
            nisa.tensor_scalar(
                dst=working[idx], data=working[idx], op0=nl.multiply,
                operand0=row_scale[idx],
            )

        # ---- column pass: reduce along the PARTITION axis via the ones matmul #
        # ACROSS ALL TILES into one PSUM tile. `accumulate` is False on the first
        # tile and True on the rest, which is `mla_projections.py:204-209`'s form
        # for a contraction longer than the partition axis. False on the first
        # tile is also what clears the previous iteration's sums: this PSUM tile
        # is reused across all 20 iterations.
        for idx, (_start, height) in enumerate(tiles):
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
        # One scale for the whole matrix, applied to every tile. Scaling a tile
        # by its own column sums instead would normalise 128 rows at a time and
        # compute a different matrix. `nl.broadcast_to` is the member that
        # broadcasts on the PARTITION axis (moe/router.py:1268-1269).
        for idx, (_start, height) in enumerate(tiles):
            col_scale_b = nl.broadcast_to(col_scale, (height, n_extent))
            nisa.tensor_tensor(
                dst=working[idx], data1=working[idx], data2=col_scale_b,
                op=nl.multiply,
            )

    for idx, (start, height) in enumerate(tiles):
        nl.store(out[start:start + height, 0:n_extent], value=working[idx])
    return out


# --------------------------------------------------------------------------- #
# The BATCHED kernel: T independent S x S Sinkhorns, no square matrix built.     #
# --------------------------------------------------------------------------- #
@nki.jit
def sinkhorn_blocks_kernel(affinity_blocks, iters: int = SINKHORN_ITERS):
    """Normalise ``[T, S, S]`` blocks, each one doubly stochastic on its own.

    WHY THIS EXISTS ALONGSIDE :func:`sinkhorn_kernel`. The mHC layer holds ``T``
    blocks of ``S x S`` and used to hand the Sinkhorn one block-diagonal
    ``[T*S, T*S]`` matrix built with ``torch.block_diag``. Every off-diagonal
    entry of that matrix is zero, and a zero stays zero under row and column
    scaling, so the global normalisation IS the ``T`` per-block normalisations --
    a block-diagonal matrix's row sums and column sums are its blocks' row sums
    and column sums. The square matrix therefore carried no information and cost
    ``(T*S)^2`` values: 256 MB of fp32 at 2048 tokens, against 128 KB for the
    blocks themselves. This kernel takes the blocks.

    It also removes the last extent bound. In the square form ``N = T * S`` rode
    the Tensor Engine's moving free axis, so ``T`` was capped at
    ``MOVING_FMAX // S``. Here ``T`` rides the PARTITION axis in tiles of at most
    :data:`PARTITION_MAX` tokens, walked inside this one dispatch, and the block's
    two axes are both FREE. So both normalisations are free-axis reductions: no
    ones-vector matmul, no partition-axis broadcast, and no bound on ``T``.

    Args:
        affinity_blocks: ``[T, S, S]`` strictly positive affinities in HBM.
        iters: normalisation iterations, a **trace-time** constant, exactly as in
            :func:`sinkhorn_kernel`. The loop stays inside this dispatch.

    Returns:
        ``[T, S, S]`` fp32. Every block has row sums :func:`row_target` and
        column sums :func:`column_target`, which for a square block is also 1.

    The shape of the code is ``hyper_connection.py:185-232``'s, the sibling
    combine kernel: one 2-D tile per block row loaded with a middle-index slice
    (``:202``), ``range(S)`` loops unrolled at trace time so the whole block is
    one dispatch, and a middle-index store (``:230``).
    """
    t_extent, rows_per_block, cols_per_block = affinity_blocks.shape
    row_goal = row_target()
    # Both of these take the UNCHECKED cores, for the tracer reason written on
    # them. `_require_blocks_admissible` on the seam is what refuses a bad shape.
    col_goal = _column_target_unchecked(int(rows_per_block), int(cols_per_block))

    out = nl.ndarray(
        (t_extent, rows_per_block, cols_per_block),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )

    # The token tiles. `block=1` is not a special case: with the S x S block held
    # in the two FREE axes, a token is ONE partition row, so no alignment is
    # needed and the tile is the whole partition extent. The arithmetic is
    # `row_tiles`'s so that both kernels tile the partition axis one way.
    tiles = _row_tiles_unchecked(int(t_extent), 1)

    # Per token tile: one working tile per block ROW, plus that row's own
    # denominator and scale, plus one column accumulator for the whole tile. All
    # allocated before the iteration loop, because the iteration is loop-carried.
    work: list[list] = []
    row_den: list[list] = []
    row_scale: list[list] = []
    col_sum = []
    col_scale = []
    for start, height in tiles:
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
        for idx, (_start, _height) in enumerate(tiles):
            # ---- row pass. Block row i of every token in this tile is one
            # [tokens, S] tile, so its row sum is a FREE-axis reduction and the
            # reciprocal broadcasts back along the free axis -- the same
            # `tensor_scalar` idiom the square kernel uses for its row pass.
            for i in range(rows_per_block):
                r_sum = nl.sum(
                    work[idx][i], axis=1, keepdims=True, dtype=nl.float32
                )
                nisa.tensor_scalar(
                    dst=row_den[idx][i], data=r_sum, op0=nl.add,
                    operand0=SINKHORN_DENOM_EPS,
                )
                nisa.reciprocal(dst=row_scale[idx][i], data=row_den[idx][i])
                nisa.tensor_scalar(
                    dst=row_scale[idx][i], data=row_scale[idx][i],
                    op0=nl.multiply, operand0=float(row_goal),
                )
                nisa.tensor_scalar(
                    dst=work[idx][i], data=work[idx][i], op0=nl.multiply,
                    operand0=row_scale[idx][i],
                )

            # ---- column pass. A block's column sum runs over its ROWS, which
            # here are separate tiles, so it is an elementwise add of the S
            # tiles rather than a reduction along any axis: entry j of the
            # accumulator is column j's sum. The first row INITIALISES the
            # accumulator, so no memset pass is needed --
            # `hyper_connection.py:213-216`'s reason for the same shape.
            nisa.tensor_copy(dst=col_sum[idx], src=work[idx][0])
            for i in range(1, rows_per_block):
                nisa.tensor_tensor(
                    dst=col_sum[idx], data1=col_sum[idx], data2=work[idx][i],
                    op=nl.add,
                )
            nisa.tensor_scalar(
                dst=col_sum[idx], data=col_sum[idx], op0=nl.add,
                operand0=SINKHORN_DENOM_EPS,
            )
            nisa.reciprocal(dst=col_scale[idx], data=col_sum[idx])
            nisa.tensor_scalar(
                dst=col_scale[idx], data=col_scale[idx], op0=nl.multiply,
                operand0=float(col_goal),
            )
            for i in range(rows_per_block):
                nisa.tensor_tensor(
                    dst=work[idx][i], data1=work[idx][i], data2=col_scale[idx],
                    op=nl.multiply,
                )

    for idx, (start, height) in enumerate(tiles):
        for i in range(rows_per_block):
            nl.store(
                out[start:start + height, i, 0:cols_per_block], value=work[idx][i]
            )
    return out


# --------------------------------------------------------------------------- #
# Geometry admission.                                                          #
# --------------------------------------------------------------------------- #
def _require_admissible(rows: int, cols: int, block: int = MHC_STREAMS) -> None:
    """Every extent condition the kernel above imposes, checked in one place.

    Each condition names what in the kernel needs it, so a reader can check the
    refusal against the code rather than against prose.

    ``M`` HAS NO UPPER BOUND HERE, and the absence is deliberate: the kernel walks
    ``M`` in row tiles, so re-imposing a partition-axis ceiling would refuse the
    extents `inc-glm53f-028b` exists to serve. ``mla_projections.py:216-224``
    records the same reading for its own tiled axes. What is still checked is that
    a token's block fits inside one tile, because the tiling is only correct if it
    cuts on block boundaries.
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
    """Extent conditions for the ``[T, S, S]`` form. Two of them, and no ceiling.

    ``T`` HAS NO UPPER BOUND: it rides the partition axis in tiles walked inside
    the kernel, so a ceiling here would refuse the serving extents this form
    exists for. Neither does the block extent carry a declared bound -- it rides
    two FREE axes and never reaches the Tensor Engine's moving free axis, so
    :data:`MOVING_FMAX`, which the square kernel's column-sum matmul does hit,
    has nothing to say about it. What bounds the block is SBUF, and a block too
    large to allocate fails in the allocator at trace time rather than returning
    a wrong answer. No byte budget is asserted here because this repository
    declares none.

    The blocks must be SQUARE. Row target ``1`` and column target
    ``rows / cols`` describe the same fixed point only when the two extents
    agree, and mHC's blocks are ``S x S`` by construction, so a non-square block
    is a caller mistake worth naming rather than a case to generalise.
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
#: `inc-glm53f-030` counts this seam's dispatches from its OWN test module (form
#: R-2, together with `inc-glm53f-029`'s seam), so the counter must be
#: resettable and readable from outside this module and outside this increment's
#: test. `inc-glm53f-026` placed its counters this way for the same reason.
_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters. Called at the start of each declared test case."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


def can_run_sinkhorn(
    affinity: Tensor, rows: int, cols: int, block: int = MHC_STREAMS
) -> bool:
    """Is the NKI route available *and* admissible for this geometry?

    Two independent conditions, deliberately not merged: ``can_run_kernel``
    answers "is there a device or a simulator", :func:`_require_admissible`
    answers "does this kernel accept these extents". A geometry the kernel
    cannot serve raises rather than falling back, because falling back would
    ship a torch path for kernel-class work (P13, D6).

    Args:
        affinity: the tensor whose device decides the route.
        rows: ``M``, the row extent. No ceiling -- the kernel tiles it.
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
    """Is the NKI route available *and* admissible for ``[T, S, S]`` blocks?

    The same two independent conditions as :func:`can_run_sinkhorn`, over
    :func:`_require_blocks_admissible`. Extents are taken as arguments rather than
    read off the tensor so that a caller -- `inc-glm53f-030b`'s layer -- can ask
    whether ``T`` tokens are servable BEFORE it builds the blocks.

    Args:
        affinity_blocks: the tensor whose device decides the route.
        tokens: ``T``, the token extent. No ceiling -- the kernel tiles it.
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
    """Doubly stochastic normalisation. The seam the route predicate counts.

    Args:
        affinity: ``[M, N]`` strictly positive affinities.
        iters: normalisation iterations, default :data:`SINKHORN_ITERS`. Passed
            through to the kernel as a trace-time constant, so the loop stays
            inside the single dispatch this function counts.
        block: the mHC stream count ``S``, default :data:`MHC_STREAMS`. The row
            tiles are cut on multiples of it, so no token's ``S x S`` block is
            split. One call still means one dispatch however many tiles the row
            extent needs -- the tiling is inside the kernel, which is what
            `inc-glm53f-028b` requires and what the route predicate measures.

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
        _COUNTERS.torch_fallback += 1
        logger.debug(
            "sinkhorn_normalise: NKI route unavailable, using the torch path "
            "(oracle only, not the shipped path)"
        )
        return sinkhorn_torch_oracle(affinity, iters=iters)

    _COUNTERS.nki_dispatch += 1
    return wrap_nki(sinkhorn_kernel)(affinity=affinity, iters=iters, block=block)


def sinkhorn_normalise_blocks(
    affinity_blocks: Tensor, iters: int = SINKHORN_ITERS
) -> Tensor:
    """Normalise ``[T, S, S]`` blocks. The seam `inc-glm53f-030b` will call.

    THIS SEAM HAS NO TORCH PATH AT ALL, and the difference from
    :func:`sinkhorn_normalise` is deliberate rather than an oversight. That seam
    returns the oracle when the route is unavailable, which is why it counts a
    ``torch_fallback``; this one raises, so ``torch_fallback`` stays zero for it
    however it is called. Kernel-class work does not ship a torch path (P13, D6),
    and a route that is absent is a fact worth failing on rather than papering
    over -- an mHC layer that silently normalised 2048 tokens in torch would be
    slow in a way no numeric acceptance could see.

    Args:
        affinity_blocks: ``[T, S, S]`` strictly positive affinities, one square
            block per token.
        iters: normalisation iterations, default :data:`SINKHORN_ITERS`, passed to
            the kernel as a trace-time constant. One call is ONE dispatch however
            many token tiles ``T`` needs, which is what the route predicate reads.

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
            f"{tokens} token blocks is kernel-class work (P13). Enable the NKI "
            "simulator for a CPU-mode run."
        )

    _COUNTERS.nki_dispatch += 1
    return wrap_nki(sinkhorn_blocks_kernel)(
        affinity_blocks=affinity_blocks, iters=iters
    )


def sinkhorn_torch_oracle(affinity: Tensor, iters: int = SINKHORN_ITERS) -> Tensor:
    """The same algorithm in torch, in fp32. The CPU oracle -- never shipped.

    Independent of the kernel in the two ways that matter for the comparison to
    say something. It reduces with ``Tensor.sum`` along each axis, so it never
    touches the ones-vector matmul the kernel uses for its column sums nor the
    partition-axis broadcast that scatters the result -- the two places a
    partition-axis mistake could hide. And it divides where the kernel takes a
    reciprocal and multiplies, so the two paths round differently.

    It shares exactly one thing with the kernel on purpose:
    :data:`SINKHORN_DENOM_EPS`, applied to both denominators, so the guard cannot
    manufacture a disagreement.

    This is **never** the shipped kernel-class path (P13, D6). It exists to be
    compared against, and as the constraint-violation return for a route the
    seam refuses.

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
    """``(module, qualname)`` of the NKI kernel, read off the object.

    Exposed so a test can assert the seam dispatches to the kernel this module
    authors -- which is how SCRATCH is checkable rather than merely claimed --
    and so a substitution shows up as a changed reading rather than as silence.
    """
    func = getattr(sinkhorn_kernel, "func", None)
    target = func if func is not None else sinkhorn_kernel
    return target.__module__, target.__qualname__


def blocks_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the BATCHED kernel, read off the object.

    The same reading as :func:`kernel_identity` for the other kernel, kept
    separate so a test can tell which of the two a seam dispatched to instead of
    inferring it from a shape. ``mla_projections.py:290-300`` unwraps ``.func``
    the same way, because ``nki.jit`` returns a wrapper and the name a substitution
    would change is the wrapped one.
    """
    func = getattr(sinkhorn_blocks_kernel, "func", None)
    target = func if func is not None else sinkhorn_blocks_kernel
    return target.__module__, target.__qualname__
