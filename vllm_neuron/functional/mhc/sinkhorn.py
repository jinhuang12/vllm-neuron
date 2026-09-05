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

#: Partition-axis bound, from ``nl.tile_size.pmax``. The affinity matrix's row
#: extent occupies the partition axis in a single tile, so ``M`` may not exceed
#: it. Written as a module constant so the refusal below and any consumer read
#: one number.
PARTITION_MAX = 128

#: Tensor Engine moving free bound, from ``nl.tile_size.gemm_moving_fmax``. The
#: column-sum matmul carries ``N`` on the moving free axis.
MOVING_FMAX = 512

__all__ = [
    "MHC_STREAMS",
    "PARTITION_MAX",
    "SINKHORN_DENOM_EPS",
    "SINKHORN_ITERS",
    "SinkhornError",
    "can_run_sinkhorn",
    "column_target",
    "dispatch_counters",
    "kernel_identity",
    "reset_dispatch_counters",
    "row_target",
    "sinkhorn_kernel",
    "sinkhorn_normalise",
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


def column_target(rows: int, cols: int) -> float:
    """Every column's target sum: ``rows / cols``.

    The rectangular generalisation of double stochasticity. With row sums at
    ``1`` the total mass is ``rows``, so spreading that mass evenly over ``cols``
    columns puts ``rows / cols`` in each -- and the two readings agree on the
    total, which is what makes them a consistent pair of targets rather than two
    independent wishes. For the declared ``[64, 4]`` case this is ``16.0``.
    """
    if cols <= 0:
        raise SinkhornError(f"cols={cols} must be positive")
    return rows / cols


# --------------------------------------------------------------------------- #
# The NKI kernel. SCRATCH: nkilib provides no sinkhorn member at any shape.     #
# --------------------------------------------------------------------------- #
@nki.jit
def sinkhorn_kernel(affinity, iters: int = SINKHORN_ITERS):
    """Doubly stochastic normalisation of ``affinity[M, N]``, in NKI.

    Args:
        affinity: ``[M, N]`` strictly positive affinities in HBM. ``M`` occupies
            the partition axis in one tile, so ``M <= PARTITION_MAX``; ``N`` is
            the free extent and the moving free extent of the column-sum matmul.
        iters: normalisation iterations, a **trace-time** constant. Defaults to
            :data:`SINKHORN_ITERS`. Exposed so a test can build the same
            algorithm at other iteration counts to show the acceptance's
            threshold is armed -- never so a caller can drive the iteration from
            the host, which is the design the route predicate exists to exclude.

    Returns:
        ``[M, N]`` fp32, row sums ``1`` and column sums ``M / N``.

    The loop is ``nl.sequential_range`` because every iteration reads the tile
    the previous one wrote. The working tile is rewritten in place, which
    ``nisa`` admits and this repository already relies on
    (``functional/moe/router.py:1212``, ``functional/argsort_unstable.py:216``).
    """
    m_extent, n_extent = affinity.shape
    col_goal = m_extent / n_extent
    row_goal = 1.0

    out = nl.ndarray((m_extent, n_extent), dtype=nl.float32, buffer=nl.shared_hbm)

    # The ones vector that turns a partition-axis reduction into a matmul.
    # Built once: it is loop-invariant.
    ones_col = nl.ndarray((m_extent, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_col, value=1.0)

    # The working tile, upcast to fp32 on the load.
    working = nl.ndarray((m_extent, n_extent), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=working, src=nl.load(affinity, dtype=nl.float32))

    # Every scratch tile is allocated ONCE outside the loop and reused. With 20
    # iterations, allocating inside would ask for 20 live PSUM tiles where 1
    # suffices, and PSUM banks are the scarcest resource on the chip.
    col_psum = nl.ndarray((1, n_extent), dtype=nl.float32, buffer=nl.psum)
    col_sum = nl.ndarray((1, n_extent), dtype=nl.float32, buffer=nl.sbuf)
    col_scale = nl.ndarray((1, n_extent), dtype=nl.float32, buffer=nl.sbuf)
    row_den = nl.ndarray((m_extent, 1), dtype=nl.float32, buffer=nl.sbuf)
    row_scale = nl.ndarray((m_extent, 1), dtype=nl.float32, buffer=nl.sbuf)

    for _ in nl.sequential_range(iters):
        # ---- row pass: reduce along the FREE axis, scale rows to row_goal ----
        row_sum = nl.sum(working, axis=1, keepdims=True, dtype=nl.float32)
        nisa.tensor_scalar(
            dst=row_den, data=row_sum, op0=nl.add, operand0=SINKHORN_DENOM_EPS
        )
        # reciprocal then multiply, rather than a divide: `nisa.reciprocal` is
        # the member moe/router.py:1312 uses for exactly this shape, and the
        # multiply below broadcasts an [M, 1] operand along the free axis.
        nisa.reciprocal(dst=row_scale, data=row_den)
        nisa.tensor_scalar(
            dst=row_scale, data=row_scale, op0=nl.multiply, operand0=float(row_goal)
        )
        nisa.tensor_scalar(
            dst=working, data=working, op0=nl.multiply, operand0=row_scale
        )

        # ---- column pass: reduce along the PARTITION axis via the ones matmul #
        # `accumulate=False` is explicit rather than inferred: this PSUM tile is
        # reused across all 20 iterations, so an accumulating write would sum
        # every iteration's column sums together instead of replacing them.
        nisa.nc_matmul(
            dst=col_psum, stationary=ones_col, moving=working, accumulate=False
        )
        nisa.tensor_copy(dst=col_sum, src=col_psum)
        nisa.tensor_scalar(
            dst=col_sum, data=col_sum, op0=nl.add, operand0=SINKHORN_DENOM_EPS
        )
        nisa.reciprocal(dst=col_scale, data=col_sum)
        nisa.tensor_scalar(
            dst=col_scale, data=col_scale, op0=nl.multiply, operand0=float(col_goal)
        )
        # The [1, N] scale has to reach M partitions. `nl.broadcast_to` is the
        # member that broadcasts on the PARTITION axis (moe/router.py:1268-1269).
        col_scale_b = nl.broadcast_to(col_scale, (m_extent, n_extent))
        nisa.tensor_tensor(
            dst=working, data1=working, data2=col_scale_b, op=nl.multiply
        )

    nl.store(out[0:m_extent, 0:n_extent], value=working)
    return out


# --------------------------------------------------------------------------- #
# Geometry admission.                                                          #
# --------------------------------------------------------------------------- #
def _require_admissible(rows: int, cols: int) -> None:
    """Every extent condition the kernel above imposes, checked in one place.

    Each condition names what in the kernel needs it, so a reader can check the
    refusal against the code rather than against prose.
    """
    problems: list[str] = []
    if rows <= 0:
        problems.append(f"M={rows} must be positive")
    elif rows > PARTITION_MAX:
        problems.append(
            f"M={rows} exceeds PARTITION_MAX={PARTITION_MAX}; the affinity "
            f"matrix's row extent occupies the partition axis in a SINGLE tile "
            f"and this kernel does not tile M. Multi-tile M would also make the "
            f"column sum a cross-tile accumulation, which changes the kernel's "
            f"shape rather than its parameters -- so it is out of "
            f"`inc-glm53f-028`'s declared scope and routes to the lead, never to "
            f"a silent pad or a torch path"
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


def can_run_sinkhorn(affinity: Tensor, rows: int, cols: int) -> bool:
    """Is the NKI route available *and* admissible for this geometry?

    Two independent conditions, deliberately not merged: ``can_run_kernel``
    answers "is there a device or a simulator", :func:`_require_admissible`
    answers "does this kernel accept these extents". A geometry the kernel
    cannot serve raises rather than falling back, because falling back would
    ship a torch path for kernel-class work (P13, D6).

    Raises:
        SinkhornError: if the geometry is inadmissible.
    """
    _require_admissible(rows, cols)
    return can_run_kernel(affinity)


def sinkhorn_normalise(affinity: Tensor, iters: int = SINKHORN_ITERS) -> Tensor:
    """Doubly stochastic normalisation. The seam the route predicate counts.

    Args:
        affinity: ``[M, N]`` strictly positive affinities.
        iters: normalisation iterations, default :data:`SINKHORN_ITERS`. Passed
            through to the kernel as a trace-time constant, so the loop stays
            inside the single dispatch this function counts.

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

    if not can_run_sinkhorn(affinity, rows, cols):
        _COUNTERS.torch_fallback += 1
        logger.debug(
            "sinkhorn_normalise: NKI route unavailable, using the torch path "
            "(oracle only, not the shipped path)"
        )
        return sinkhorn_torch_oracle(affinity, iters=iters)

    _COUNTERS.nki_dispatch += 1
    return wrap_nki(sinkhorn_kernel)(affinity=affinity, iters=iters)


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
