# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for `inc-glm53f-028` -- the mHC Sinkhorn normalisation kernel.

Acceptance command (plan block ``#### inc-glm53f-028``, Tier N harness "as
`-025`")::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/mhc/test_sinkhorn.py \
      -q -s --timeout 60 -p no:cacheprovider

The two DECLARED arms, and which one is the real one
---------------------------------------------------
1. the **oracle arm** -- after the target's **20** iterations on a synthetic
   ``[64, 4]`` affinity matrix, simulated NKI output against a torch Sinkhorn
   oracle authored below, ``assert_close(rtol=1e-2, atol=1e-5)``;
2. the **doubly-stochastic arm** -- every row sum and every column sum within
   **1e-3** of its target, reported as worst-row and worst-column deviations.

The plan states plainly that arm 2 "is the real one: it does not depend on the
oracle being right", and this file is written to keep that true. Arm 2 compares
the kernel's output against the two TARGETS -- ``1`` per row and ``M / N`` per
column -- which are properties of the algorithm's definition, not of any
reference implementation. If the oracle below were wrong, arm 1 would go red and
arm 2 would not move.

`inc-glm53f-028b` -- the same two arms at serving row extents
------------------------------------------------------------
The kernel now walks ``M`` in row tiles, so both arms run again at
:data:`TILED_ROWS` -- ``129``, ``256``, ``512 * hc_mult`` and ``2048 * hc_mult``
-- in ``test_tiled_rows_match_the_oracle_and_stay_doubly_stochastic``. Nothing
about the arms changes: the same fixture construction, the same tolerances, the
same three route instruments, and one dispatch per case however many tiles the
extent needs. ``129`` is in the list because it is the first extent that does not
fit one tile AND is not a whole number of blocks, so a ragged last tile is
measured rather than assumed.

Two readings are new, and each one exists because the tiling could be wrong in a
way the arms alone would not name. ``test_row_tiles_are_cut_on_block_boundaries``
reads the kernel's own tile arithmetic and requires every tile to start on a
multiple of ``hc_mult``, so no token's block is split. And arm 2 is what catches
the tempting wrong tiling: a kernel that scaled each tile by its own column sums
would leave ``tile_rows / N`` in each column instead of ``M / N``, which arm 2
reads directly and arm 1 would nearly pass.

No tolerance number is invented, widened or narrowed anywhere in this file.
:data:`RTOL`, :data:`ATOL` and :data:`STOCHASTIC_TOL` are the plan's.

Why the route predicate is an acceptance criterion and not a diagnostic (F1)
---------------------------------------------------------------------------
Arm 1 compares simulated NKI output against a torch oracle. If the seam silently
took its torch path, *both* sides would be torch and the comparison would pass
green while measuring nothing about a kernel. So each declared case reads three
route instruments and reports each as a number:

1. the seam's own module-level dispatch counter (form R-1) -- ``nki_dispatch ==
   1``, ``torch_fallback == 0``;
2. ``can_run_kernel()`` -- ``True``;
3. real ``nki.simulator.simulate_kernel`` invocations on the F1 chain -- ``1``
   per kernel call. Instrument 3 counts the VENDOR entry point, so a bug in
   instrument 1 cannot fake it.

``1``, NOT ``20`` -- the discrimination this increment turns on
--------------------------------------------------------------
The plan singles this out: "**`1`, not `20`, is the declaration that matters**"
-- the twenty normalisation iterations run INSIDE the kernel, so a host-driven
iteration loop would read ``20`` and "the two readings tell the two designs
apart." A test that only asserted ``== 1`` would leave that claim resting on the
number's smallness. So
:func:`test_route_predicate_reads_one_not_twenty` builds the rejected design --
twenty single-iteration seam calls from the host, which computes the *same*
answer -- and shows the instrument reading ``20`` against this design's ``1``,
on one fixture, in one transcript. That is what makes ``1`` a measurement of
where the loop lives rather than a coincidence.

Every zero is armed. ``test_route_control_fallback_counter_discriminates`` shows
instrument 1 reading ``(0, 1)`` and instrument 3 reading ``0`` on the fallback
path; ``test_route_control_simulator_is_load_bearing`` shows the chain RAISING
rather than quietly computing torch when the simulator is off; and
``test_doubly_stochastic_bar_is_armed`` shows the ``1e-3`` bar rejecting a
truncated iteration count, so arm 2's pass is not a property of the threshold
being loose.

The batched ``[T, S, S]`` form, and how it is checked
----------------------------------------------------
`inc-glm53f-028b`, second form. ``sinkhorn_normalise_blocks`` normalises one
square block per token instead of the ``block_diag`` matrix of them. Its
correctness is checked in the way that leaves nothing to a claim: at ``T`` in
``{1, 3, 33}`` its output is compared BLOCK FOR BLOCK against this module's own
square kernel run on ``torch.block_diag`` of the same blocks -- the same call
``model_fp8.py:1147``'s ``mhc_pre`` makes. The two are not merely equal at the
fixed point; they are equal iteration for iteration, because a row of a
block-diagonal matrix has nonzeros only inside its own block, so the square
kernel's row and column sums ARE the per-block sums.

At ``T`` in ``{129, 2048}`` that comparison is not available, and its absence is
the point of the form rather than a gap in the test:
``test_the_square_form_cannot_serve_what_the_blocks_form_serves`` shows the
square seam REFUSING ``T = 129`` by name, because ``N = T * S`` passes the Tensor
Engine's moving free bound, and prints what the square matrix would have cost at
2048 tokens. Those two extents are therefore checked against a batched torch
oracle, which is itself cross-checked against the per-block oracle the landed
cases use.
"""

from __future__ import annotations

import ast
import importlib
import os
import pathlib
import sys

import pytest
import torch

import nki
import nki.simulator

from vllm_neuron.functional.mhc.sinkhorn import (
    MHC_STREAMS,
    MOVING_FMAX,
    PARTITION_MAX,
    SINKHORN_DENOM_EPS,
    SINKHORN_ITERS,
    SinkhornError,
    blocks_kernel_identity,
    can_run_sinkhorn,
    can_run_sinkhorn_blocks,
    column_target,
    dispatch_counters,
    kernel_identity,
    reset_dispatch_counters,
    row_target,
    row_tile_extent,
    row_tiles,
    sinkhorn_normalise,
    sinkhorn_normalise_blocks,
    sinkhorn_torch_oracle,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# --------------------------------------------------------------------------- #
# The declared fixture. [64, 4] is the plan's; 4 is read off MHC_STREAMS so     #
# the fixture and the target's `hc_mult` cannot drift apart.                    #
# --------------------------------------------------------------------------- #
M = 64
N = MHC_STREAMS  # 4

#: `inc-glm53f-028b`'s declared row extents, the ones serving needs. 129 is the
#: first row count that does not fit one partition tile, and it is deliberately
#: NOT a whole number of blocks -- 129 = 32 blocks and one row -- so the tiling
#: is measured on a ragged extent as well as on exact ones. The two largest are
#: written as ``tokens * MHC_STREAMS`` because that is what they are: 512 and 2048
#: tokens of the target's four streams each.
TILED_ROWS = (129, 256, 512 * MHC_STREAMS, 2048 * MHC_STREAMS)

#: Token counts for the BATCHED form's equivalence arm. Small on purpose: each
#: one is also run through the square kernel on ``block_diag`` of the same blocks,
#: which costs ``(T*S)^2``. ``1`` is the degenerate single block, ``3`` is a
#: handful, and ``33`` is the first that makes the SQUARE side tile as well
#: (``33 * 4 = 132`` rows, so two row tiles) -- so the equivalence is read across
#: the square kernel's tile seam too, not only inside one tile.
BLOCK_EQUIV_TOKENS = (1, 3, 33)

#: Token counts for the batched form's SCALE arm, where no square comparison
#: exists: ``129`` is the first ``T`` the square form refuses (``N = 516`` passes
#: :data:`MOVING_FMAX`), and ``2048`` is the serving extent this form exists for.
BLOCK_SCALE_TOKENS = (129, 2048)

#: The declared tolerance pair for the oracle arm, from the plan block.
RTOL = 1e-2
ATOL = 1e-5
#: The declared bound for the doubly-stochastic arm, from the plan block.
STOCHASTIC_TOL = 1e-3

_MODULE = "vllm_neuron.functional.mhc.sinkhorn"


class RouteInstrumentError(AssertionError):
    """A route reading that is not what the plan declares."""


class StochasticityError(AssertionError):
    """A row or column sum outside the declared 1e-3 of its target."""


class VacuousControlError(AssertionError):
    """A control whose input could not have made it fail.

    A zero over vacuous input measures nothing, so the control refuses to report
    a pass it did not earn.
    """


# --------------------------------------------------------------------------- #
# Route instrumentation. Counts the VENDOR entry point, so it is independent    #
# of the seam counter it cross-checks.                                          #
# --------------------------------------------------------------------------- #
class _SimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls for the duration."""

    def __init__(self) -> None:
        self.calls = 0
        self._real = None

    def __enter__(self) -> "_SimulatorCounter":
        self._real = nki.simulator.simulate_kernel
        real = self._real

        def counting(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        nki.simulator.simulate_kernel = counting
        return self

    def __exit__(self, *exc_info) -> None:
        nki.simulator.simulate_kernel = self._real


def _assert_route(sim: _SimulatorCounter, expected_dispatches: int, label: str) -> str:
    """Read all three route instruments and return the reading for the transcript."""
    nki_dispatch, torch_fallback = dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"[{label}] nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
        f"can_run_kernel={gate} simulate_kernel_calls={sim.calls}"
    )
    print(reading)
    if nki_dispatch != expected_dispatches:
        raise RouteInstrumentError(
            f"{label}: seam dispatch counter read {nki_dispatch}, declared "
            f"{expected_dispatches}. {reading}"
        )
    if torch_fallback != 0:
        raise RouteInstrumentError(
            f"{label}: torch-fallback counter read {torch_fallback}, declared "
            f"exactly 0 -- a fallback pass would compare torch against torch. "
            f"{reading}"
        )
    if gate is not True:
        raise RouteInstrumentError(
            f"{label}: can_run_kernel() read {gate!r}, declared True. {reading}"
        )
    if sim.calls != expected_dispatches:
        raise RouteInstrumentError(
            f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, "
            f"declared {expected_dispatches}. A numeric pass without a simulator "
            f"call is the F1 false green. {reading}"
        )
    return reading


# --------------------------------------------------------------------------- #
# Fixture and the TEST-AUTHORED oracle.                                        #
# --------------------------------------------------------------------------- #
def _affinity(seed: int = 21) -> torch.Tensor:
    """A synthetic strictly positive ``[64, 4]`` affinity matrix, fp32.

    ``exp`` of a bounded uniform draw, which is what an affinity matrix is in the
    target -- the exponential of a score -- and which guarantees strict
    positivity without a clamp. DETERMINISTIC by seed because two properties of
    this fixture are load-bearing and a draw could satisfy them by luck:

    * **strictly positive**, so Sinkhorn's fixed point exists and the guard
      :data:`SINKHORN_DENOM_EPS` is never what keeps the arithmetic finite;
    * **not already doubly stochastic**, so the 20 iterations have real work to
      do. :func:`test_fixture_is_not_already_normalised` measures that rather
      than assuming it -- an input that arrived normalised would let a kernel
      that did NOTHING pass arm 2.

    The ``[-1, 1]`` exponent range keeps the dynamic range at ``e**2 ~ 7.4``, so
    the row sums are well conditioned and no term dominates its row. That is the
    `inc-glm53f-025` attempt-1 conditioning lesson applied: a relative tolerance
    over a badly conditioned reduction measures cancellation, not the kernel.

    ``inc-glm53f-028b`` moved the draw into :func:`_affinity_rows` so the tiled
    row extents use this same construction at their own heights. The ``[64, 4]``
    fixture and every property claimed for it above are unchanged.
    """
    return _affinity_rows(M, seed)


def _affinity_rows(rows: int, seed: int = 21) -> torch.Tensor:
    """The same fixture at ``rows`` rows: ``exp`` of a bounded uniform draw, fp32.

    One construction for every row extent, so a tiled case cannot pass on an
    easier input than the single-tile case. The seed is fixed for the same reason
    it is fixed in :func:`_affinity`: strict positivity and "not already
    normalised" are load-bearing and a draw could satisfy them by luck.
    """
    generator = torch.Generator().manual_seed(seed)
    logits = (
        torch.rand((rows, N), generator=generator, dtype=torch.float32) * 2.0 - 1.0
    )
    return torch.exp(logits)


def _sinkhorn_oracle_authored_here(
    affinity: torch.Tensor, iters: int = SINKHORN_ITERS
) -> torch.Tensor:
    """The plan's "torch Sinkhorn oracle authored in the test", written here.

    Deliberately a DIFFERENT formulation from the module's
    :func:`sinkhorn_torch_oracle`, so arm 1 is not a module comparing itself
    against its own restatement. This one accumulates explicit per-axis scaling
    VECTORS and applies them to the ORIGINAL matrix at the end of each iteration
    -- the classical Sinkhorn-Knopp form -- where the module's oracle rescales
    the working matrix in place.

    The two agree in value while differing in what they multiply and in what
    order, which is what makes
    :func:`test_module_oracle_agrees_with_the_test_authored_oracle` a real
    cross-check rather than a tautology.

    :data:`SINKHORN_DENOM_EPS` is applied to both denominators, identically to
    the kernel and to the module's oracle, so the guard cannot manufacture a
    disagreement in either direction.
    """
    rows, cols = int(affinity.shape[0]), int(affinity.shape[1])
    base = affinity.to(torch.float64)
    u = torch.ones((rows, 1), dtype=torch.float64)
    v = torch.ones((1, cols), dtype=torch.float64)
    row_goal = row_target()
    col_goal = column_target(rows, cols)

    for _ in range(iters):
        scaled = base * u * v
        u = u * (row_goal / (scaled.sum(dim=1, keepdim=True) + SINKHORN_DENOM_EPS))
        scaled = base * u * v
        v = v * (col_goal / (scaled.sum(dim=0, keepdim=True) + SINKHORN_DENOM_EPS))

    return (base * u * v).to(torch.float32)


def _deviations(result: torch.Tensor) -> tuple[float, float]:
    """``(worst_row_deviation, worst_column_deviation)`` -- numbers, not verdicts.

    Each axis is compared against ITS OWN target, which is what the plan's
    "within 1e-3 of its target" requires: rows against :func:`row_target`,
    columns against :func:`column_target`.
    """
    rows, cols = int(result.shape[0]), int(result.shape[1])
    row_dev = float((result.sum(dim=1) - row_target()).abs().max())
    col_dev = float((result.sum(dim=0) - column_target(rows, cols)).abs().max())
    return row_dev, col_dev


def _report_stochasticity(result: torch.Tensor, label: str) -> tuple[float, float]:
    """Print the per-axis readings and return the two worst deviations."""
    rows, cols = int(result.shape[0]), int(result.shape[1])
    row_dev, col_dev = _deviations(result)
    row_sums = result.sum(dim=1)
    col_sums = result.sum(dim=0)
    print(
        f"[{label}] row_target={row_target()} column_target="
        f"{column_target(rows, cols)} iters={SINKHORN_ITERS}"
    )
    print(
        f"[{label}] worst_row_deviation={row_dev:.6e} "
        f"worst_column_deviation={col_dev:.6e} bound={STOCHASTIC_TOL}"
    )
    print(
        f"[{label}] row_sum_min={float(row_sums.min()):.9f} "
        f"row_sum_max={float(row_sums.max()):.9f} "
        f"column_sums={[round(float(v), 9) for v in col_sums]}"
    )
    print(
        f"[{label}] total_mass={float(result.sum()):.9f} expected={float(rows)} "
        f"output_min={float(result.min()):.6e}"
    )
    return row_dev, col_dev


# --------------------------------------------------------------------------- #
# `inc-glm53f-028b` -- the BATCHED `[T, S, S]` fixture, oracle and readings.     #
# --------------------------------------------------------------------------- #
def _affinity_blocks(tokens: int, seed: int = 21) -> torch.Tensor:
    """``[T, S, S]`` strictly positive affinities, one square block per token.

    Built by RESHAPING :func:`_affinity_rows`, so the batched fixture is the same
    draw as every other case in this file rather than a second construction that
    could be easier: ``T * S`` rows of ``S`` columns hold exactly ``T`` blocks of
    ``S x S``. Everything :func:`_affinity` claims -- strict positivity, a bounded
    ``e**2`` dynamic range, not already normalised -- therefore holds here, and
    the non-vacuity readings below measure the last one per case.
    """
    return _affinity_rows(tokens * N, seed).reshape(tokens, N, N)


def _block_diag_of(blocks: torch.Tensor) -> torch.Tensor:
    """``torch.block_diag`` of the blocks -- the matrix the target builds today.

    The same call ``model_fp8.py:1147``'s ``mhc_pre`` makes on its own blocks, so
    the equivalence arm compares against the form the layer currently produces
    rather than against a restatement of it.
    """
    return torch.block_diag(*blocks.unbind(0))


def _diagonal_blocks_of(
    matrix: torch.Tensor, tokens: int, streams: int
) -> torch.Tensor:
    """Cut the ``T`` diagonal ``S x S`` blocks back out of a ``[T*S, T*S]`` matrix.

    Written here rather than imported from ``model_fp8`` on purpose: that module's
    extractor is private, and this arm should read the diagonal the way the maths
    defines it, so a change there shows up as a disagreement instead of moving
    both sides at once.
    """
    return torch.stack(
        [
            matrix[t * streams : (t + 1) * streams, t * streams : (t + 1) * streams]
            for t in range(tokens)
        ]
    )


def _blocks_oracle_authored_here(
    blocks: torch.Tensor, iters: int = SINKHORN_ITERS
) -> torch.Tensor:
    """:func:`_sinkhorn_oracle_authored_here`, batched over the leading axis.

    The same classical Sinkhorn-Knopp formulation -- accumulate per-axis scaling
    VECTORS in float64 and apply them to the ORIGINAL blocks -- with one extra
    leading dimension, so the reductions are over axes 2 and 1 of ``[T, S, S]``.

    :func:`test_the_batched_oracle_agrees_with_the_per_block_oracle` measures that
    this batching did not change the answer, by running the landed per-block
    oracle block by block over the same input. That control is what lets the scale
    arm rest on this helper at ``T = 2048``, where a per-block python loop over the
    oracle would dominate the case's runtime.
    """
    base = blocks.to(torch.float64)
    tokens, rows, cols = (int(v) for v in base.shape)
    u = torch.ones((tokens, rows, 1), dtype=torch.float64)
    v = torch.ones((tokens, 1, cols), dtype=torch.float64)
    row_goal = row_target()
    col_goal = column_target(rows, cols)

    for _ in range(iters):
        scaled = base * u * v
        u = u * (row_goal / (scaled.sum(dim=2, keepdim=True) + SINKHORN_DENOM_EPS))
        scaled = base * u * v
        v = v * (col_goal / (scaled.sum(dim=1, keepdim=True) + SINKHORN_DENOM_EPS))

    return (base * u * v).to(torch.float32)


def _block_deviations(result: torch.Tensor) -> tuple[float, float]:
    """``(worst_row_deviation, worst_column_deviation)`` over ALL blocks.

    Per block and per axis, against that axis's own target -- the same reading
    :func:`_deviations` takes on a matrix, with the worst case over the ``T``
    blocks rather than an average, so one bad block cannot hide behind 2047 good
    ones.
    """
    _tokens, rows, cols = (int(v) for v in result.shape)
    row_dev = float((result.sum(dim=2) - row_target()).abs().max())
    col_dev = float((result.sum(dim=1) - column_target(rows, cols)).abs().max())
    return row_dev, col_dev


def _report_block_stochasticity(
    result: torch.Tensor, label: str
) -> tuple[float, float]:
    """Print the batched per-axis readings and return the two worst deviations."""
    tokens, rows, cols = (int(v) for v in result.shape)
    row_dev, col_dev = _block_deviations(result)
    print(
        f"[{label}] T={tokens} block=[{rows},{cols}] row_target={row_target()} "
        f"column_target={column_target(rows, cols)} iters={SINKHORN_ITERS}"
    )
    print(
        f"[{label}] worst_row_deviation={row_dev:.6e} "
        f"worst_column_deviation={col_dev:.6e} bound={STOCHASTIC_TOL}"
    )
    print(
        f"[{label}] total_mass={float(result.sum()):.6f} "
        f"expected={float(tokens * rows)} output_min={float(result.min()):.6e}"
    )
    return row_dev, col_dev


# --------------------------------------------------------------------------- #
# DECLARED ARM 1 -- the oracle arm.                                            #
# --------------------------------------------------------------------------- #
def test_output_matches_torch_oracle_after_twenty_iterations() -> None:
    """Simulated NKI output vs the test-authored torch Sinkhorn oracle.

    The plan's declared Expected: after the target's **20** iterations on a
    synthetic ``[64, 4]`` affinity matrix, ``assert_close(rtol=1e-2,
    atol=1e-5)``.
    """
    affinity = _affinity()
    assert tuple(affinity.shape) == (64, 4), tuple(affinity.shape)
    assert SINKHORN_ITERS == 20, SINKHORN_ITERS

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = sinkhorn_normalise(affinity)
    _assert_route(sim, 1, "oracle-arm")

    want = _sinkhorn_oracle_authored_here(affinity)
    got32 = got.to(torch.float32)

    abs_err = float((got32 - want).abs().max())
    rel_err = float(((got32 - want).abs() / (want.abs() + ATOL)).max())
    print(
        f"[oracle-arm] max_abs_error={abs_err:.6e} max_rel_error={rel_err:.6e} "
        f"rtol={RTOL} atol={ATOL} want_absmin={float(want.abs().min()):.6e} "
        f"want_absmax={float(want.abs().max()):.6e}"
    )
    if float(want.abs().max()) == 0.0:
        raise VacuousControlError(
            "the oracle produced an all-zero reference, so the comparison would "
            "pass over empty input; refusing to report a pass"
        )
    torch.testing.assert_close(got32, want, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# DECLARED ARM 2 -- the doubly-stochastic arm. The real one.                    #
# --------------------------------------------------------------------------- #
def test_doubly_stochastic_row_and_column_sums() -> None:
    """Every row sum and column sum within 1e-3 of its target.

    The plan's declared Expected, and the assertion it calls "the real one: it
    does not depend on the oracle being right". Nothing in this test consults any
    oracle -- the two targets come from the algorithm's definition through
    :func:`row_target` and :func:`column_target`.
    """
    affinity = _affinity()
    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = sinkhorn_normalise(affinity).to(torch.float32)
    _assert_route(sim, 1, "stochastic-arm")

    row_dev, col_dev = _report_stochasticity(got, "stochastic-arm")

    if not torch.isfinite(got).all():
        raise StochasticityError("the kernel returned non-finite values")
    if row_dev > STOCHASTIC_TOL:
        raise StochasticityError(
            f"worst row deviation {row_dev:.6e} exceeds the declared "
            f"{STOCHASTIC_TOL} against row target {row_target()}"
        )
    if col_dev > STOCHASTIC_TOL:
        raise StochasticityError(
            f"worst column deviation {col_dev:.6e} exceeds the declared "
            f"{STOCHASTIC_TOL} against column target {column_target(M, N)}"
        )


def test_fixture_is_not_already_normalised() -> None:
    """NON-VACUITY of arm 2: the INPUT must fail the bar the output passes.

    Without this reading, arm 2 would be satisfied by a kernel that returned its
    input untouched, and no other arm would notice. Measured on the same fixture
    arm 2 uses.
    """
    affinity = _affinity()
    row_dev, col_dev = _deviations(affinity)
    print(
        f"[fixture-non-vacuity] input worst_row_deviation={row_dev:.6e} "
        f"input worst_column_deviation={col_dev:.6e} bound={STOCHASTIC_TOL}"
    )
    assert row_dev > STOCHASTIC_TOL, (
        f"the fixture's rows already sum to within {STOCHASTIC_TOL} of "
        f"{row_target()} (worst {row_dev:.6e}), so a kernel returning its input "
        f"unchanged would pass arm 2"
    )
    assert col_dev > STOCHASTIC_TOL, (
        f"the fixture's columns already sum to within {STOCHASTIC_TOL} of "
        f"{column_target(M, N)} (worst {col_dev:.6e}), so arm 2 would be vacuous"
    )
    assert float(affinity.min()) > 0.0, "the fixture is not strictly positive"


def test_doubly_stochastic_bar_is_armed() -> None:
    """The 1e-3 bar must REJECT a truncated iteration count.

    Arm 2 passing tells us something only if the bar can fail. A single
    iteration computes a matrix whose columns are exact (it ends on a column
    pass) but whose rows are not yet converged, so it is the sharpest available
    probe of the bar: the same instrument, the same fixture, the same threshold,
    a different iteration count.

    This also measures WHY 20 is not arbitrary -- the transcript carries the
    deviation ladder.
    """
    affinity = _affinity()
    ladder = []
    for iters in (1, 2, 3, 5, SINKHORN_ITERS):
        reference = _sinkhorn_oracle_authored_here(affinity, iters=iters)
        row_dev, col_dev = _deviations(reference)
        ladder.append((iters, row_dev, col_dev))
        print(
            f"[bar-armed] iters={iters:2d} worst_row_deviation={row_dev:.6e} "
            f"worst_column_deviation={col_dev:.6e} bound={STOCHASTIC_TOL} "
            f"passes={row_dev <= STOCHASTIC_TOL and col_dev <= STOCHASTIC_TOL}"
        )

    one_row_dev = ladder[0][1]
    assert one_row_dev > STOCHASTIC_TOL, (
        f"a SINGLE iteration already lands inside {STOCHASTIC_TOL} (worst row "
        f"deviation {one_row_dev:.6e}), so the bar cannot distinguish a "
        f"converged result from a truncated one and arm 2 means nothing"
    )
    final_row_dev, final_col_dev = ladder[-1][1], ladder[-1][2]
    assert final_row_dev <= STOCHASTIC_TOL and final_col_dev <= STOCHASTIC_TOL, (
        f"the declared {SINKHORN_ITERS} iterations do not reach the declared "
        f"{STOCHASTIC_TOL}: row {final_row_dev:.6e}, column {final_col_dev:.6e}"
    )


# --------------------------------------------------------------------------- #
# `1`, NOT `20` -- the discrimination made live.                                #
# --------------------------------------------------------------------------- #
def test_route_predicate_reads_one_not_twenty() -> None:
    """One dispatch for twenty in-kernel iterations; twenty for a host loop.

    The plan: "**`1`, not `20`, is the declaration that matters** -- the target's
    twenty normalisation iterations run INSIDE the kernel, so a host-driven
    iteration loop would read `20` and the two readings tell the two designs
    apart."

    So both designs are RUN here, on one fixture, and both counters are
    reported. The rejected design is built the only way it could be -- twenty
    seam calls of one iteration each -- and it computes the same answer, which is
    exactly why a numeric comparison could never have told them apart and the
    counter is an acceptance criterion rather than a diagnostic.
    """
    affinity = _affinity()

    # THIS design: the loop is in the kernel.
    reset_dispatch_counters()
    with _SimulatorCounter() as sim_in_kernel:
        in_kernel = sinkhorn_normalise(affinity, iters=SINKHORN_ITERS)
    in_kernel_counters = dispatch_counters()
    _assert_route(sim_in_kernel, 1, "in-kernel-loop")

    # The REJECTED design: the loop is on the host.
    reset_dispatch_counters()
    with _SimulatorCounter() as sim_host:
        host_driven = affinity
        for _ in range(SINKHORN_ITERS):
            host_driven = sinkhorn_normalise(host_driven, iters=1)
    host_counters = dispatch_counters()
    print(
        f"[one-not-twenty] in_kernel_counters={in_kernel_counters} "
        f"in_kernel_simulate_kernel_calls={sim_in_kernel.calls} "
        f"host_loop_counters={host_counters} "
        f"host_loop_simulate_kernel_calls={sim_host.calls}"
    )

    assert in_kernel_counters == (1, 0), (
        f"the in-kernel design read {in_kernel_counters}, declared (1, 0)"
    )
    assert host_counters == (SINKHORN_ITERS, 0), (
        f"the host-driven design read {host_counters}, expected "
        f"({SINKHORN_ITERS}, 0); if it does not read {SINKHORN_ITERS} then the "
        f"counter does not discriminate the two designs and the plan's "
        f"'1, not 20' declaration is untestable"
    )
    assert sim_host.calls == SINKHORN_ITERS, sim_host.calls

    # And the two designs agree NUMERICALLY, which is the whole point: no
    # numeric arm could have caught the wrong one.
    agreement = float(
        (in_kernel.to(torch.float32) - host_driven.to(torch.float32)).abs().max()
    )
    print(
        f"[one-not-twenty] max_abs_difference_between_designs={agreement:.6e} "
        f"atol={ATOL} -- the numeric arms CANNOT tell these apart, the counter can"
    )
    torch.testing.assert_close(
        in_kernel.to(torch.float32),
        host_driven.to(torch.float32),
        rtol=RTOL,
        atol=ATOL,
    )


# --------------------------------------------------------------------------- #
# NON-VACUITY of the numeric comparison (D1.5): it must be able to FAIL.        #
# --------------------------------------------------------------------------- #
def test_numeric_comparison_is_armed() -> None:
    """Compare the kernel's output against a DIFFERENT matrix's normalisation.

    Without this arm, a small ``max_rel_error`` would be indistinguishable from
    an unwired comparison. The declared tolerance pair is used unchanged; only
    the reference's input is perturbed, and the perturbation is shown to have
    changed the reference before the failure is required.
    """
    affinity = _affinity()
    perturbed = affinity.clone()
    # Perturb ONE entry by a factor Sinkhorn cannot wash out: the fixed point
    # depends on the input up to per-axis scalings, and a single-entry change is
    # not a per-axis scaling.
    perturbed[0, 0] = perturbed[0, 0] * 4.0
    if torch.equal(perturbed, affinity):
        raise VacuousControlError("injection changed nothing -- control is vacuous")

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = sinkhorn_normalise(affinity).to(torch.float32)
    _assert_route(sim, 1, "armed-control")

    wrong = _sinkhorn_oracle_authored_here(perturbed)
    rel = float(((got - wrong).abs() / (wrong.abs() + ATOL)).max())
    print(
        f"[armed-control] affinity[0,0] scaled by 4.0: max_rel_error={rel:.6e} "
        f"vs rtol={RTOL}"
    )
    assert rel > RTOL, (
        f"quadrupling one affinity entry moved the comparison by only "
        f"{rel:.6e}, which is inside the declared rtol {RTOL}; the comparison is "
        f"therefore not a discriminator and a green arm 1 would mean nothing"
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, wrong, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# The module's oracle vs the test-authored one -- two formulations, one value.   #
# --------------------------------------------------------------------------- #
def test_module_oracle_agrees_with_the_test_authored_oracle() -> None:
    """The two torch formulations agree, so arm 1's reference is corroborated.

    The plan's Acceptance names an oracle "authored in the test", and the plan's
    Surface names a "torch oracle" in the module. Both exist, and they are
    written differently on purpose -- the module rescales a working matrix in
    place, this file accumulates explicit per-axis scaling vectors and applies
    them to the ORIGINAL matrix. Agreement between two different formulations is
    evidence about the reference; agreement between one formulation and itself
    would be nothing.

    This is also what lets `inc-glm53f-030` rely on the module's oracle: it is
    exercised here rather than shipped unmeasured.
    """
    affinity = _affinity()
    module_side = sinkhorn_torch_oracle(affinity)
    test_side = _sinkhorn_oracle_authored_here(affinity)
    delta = float((module_side - test_side).abs().max())
    print(
        f"[oracle-cross-check] max_abs_difference={delta:.6e} atol={ATOL} "
        f"module_absmax={float(module_side.abs().max()):.6e}"
    )
    torch.testing.assert_close(module_side, test_side, rtol=RTOL, atol=ATOL)

    # Both formulations must independently reach the declared bar, or the
    # agreement above would just mean they are wrong together.
    for label, reference in (("module", module_side), ("test-authored", test_side)):
        row_dev, col_dev = _deviations(reference)
        print(
            f"[oracle-cross-check] {label} worst_row_deviation={row_dev:.6e} "
            f"worst_column_deviation={col_dev:.6e} bound={STOCHASTIC_TOL}"
        )
        assert row_dev <= STOCHASTIC_TOL and col_dev <= STOCHASTIC_TOL, (
            f"the {label} oracle does not reach the declared {STOCHASTIC_TOL}"
        )


def test_targets_are_a_consistent_pair() -> None:
    """The two targets agree on the total mass, so arm 2 is not self-contradictory.

    ``M`` rows at ``row_target()`` and ``N`` columns at ``column_target(M, N)``
    must describe the same total. If they did not, no matrix could satisfy both
    and arm 2 would be unsatisfiable by construction rather than by defect.
    """
    from_rows = M * row_target()
    from_cols = N * column_target(M, N)
    print(
        f"[targets] rows {M} x {row_target()} = {from_rows}; columns {N} x "
        f"{column_target(M, N)} = {from_cols}"
    )
    assert from_rows == from_cols == float(M)
    assert column_target(64, 4) == 16.0


# --------------------------------------------------------------------------- #
# ROUTE CONTROLS -- so every zero above is a measurement.                       #
# --------------------------------------------------------------------------- #
def test_route_control_fallback_counter_discriminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the simulator disabled the seam takes the torch path, and it is COUNTED.

    This is the arm that makes ``torch_fallback == 0`` above meaningful: the
    counter is shown reading ``1`` and ``nki_dispatch`` reading ``0``, through
    the real gate rather than a mock. It is also the measured form of the plan's
    claim that a pure-torch implementation yields ``0`` dispatches.
    """
    affinity = _affinity()
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0, so this control is unarmed"
    )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = sinkhorn_normalise(affinity)
    nki_dispatch, torch_fallback = dispatch_counters()
    print(
        f"[route-control] nki_dispatch={nki_dispatch} "
        f"torch_fallback={torch_fallback} simulate_kernel_calls={sim.calls}"
    )
    assert nki_dispatch == 0, f"expected 0 NKI dispatches, got {nki_dispatch}"
    assert torch_fallback == 1, f"expected 1 torch fallback, got {torch_fallback}"
    assert sim.calls == 0, f"the simulator ran {sim.calls} times with it disabled"
    assert tuple(out.shape) == (M, N)


def test_route_control_simulator_is_load_bearing() -> None:
    """The NKI chain RAISES without the simulator rather than computing torch.

    Recorded because it is what forecloses the F1 false green BELOW this
    repository's seam: if the HOP silently degraded to a torch path, a green
    numeric comparison could not be attributed to a kernel at all, and no
    counter of this module's could detect it.
    """
    affinity = _affinity()
    saved = os.environ.get("NKI_SIMULATOR")
    os.environ["NKI_SIMULATOR"] = "0"
    try:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from vllm_neuron.functional.mhc.sinkhorn import sinkhorn_kernel

        with pytest.raises(RuntimeError) as excinfo:
            wrap_nki(sinkhorn_kernel)(affinity=affinity, iters=SINKHORN_ITERS)
    finally:
        if saved is None:
            os.environ.pop("NKI_SIMULATOR", None)
        else:
            os.environ["NKI_SIMULATOR"] = saved

    message = str(excinfo.value)
    print(f"[route-control] simulator_off_raise={message[:160]!r}")
    assert "simulator" in message.lower(), message


# --------------------------------------------------------------------------- #
# The counters are MODULE-LEVEL state -- `inc-glm53f-030` depends on it.         #
# --------------------------------------------------------------------------- #
def test_dispatch_counters_are_module_level_state_reachable_from_elsewhere() -> None:
    """Another module can zero and read these counters. `-030` needs exactly this.

    `inc-glm53f-030`'s route predicate is form R-2 over THIS seam together with
    `inc-glm53f-029`'s: its own test module resets and reads the counters this
    module owns, per layer call. A test-local counter, or one only this file
    could reset, would pass this increment and break that one.

    Measured rather than asserted by inspection: the module is re-acquired
    through ``importlib`` -- the same mechanism another test module's import uses
    -- one reference RESETS, the seam is driven, and the OTHER reference READS.
    Then the counter is shown to ACCUMULATE across calls, because `-030` reads a
    per-layer-call total and a counter that saturated at 1 could not supply one.
    """
    foreign = importlib.import_module(_MODULE)
    assert foreign is sys.modules[_MODULE]
    # Identity, not just equality: both references address one counter object.
    assert foreign.dispatch_counters is dispatch_counters
    assert foreign.reset_dispatch_counters is reset_dispatch_counters

    affinity = _affinity()
    foreign.reset_dispatch_counters()
    assert dispatch_counters() == (0, 0), (
        "a reset through the foreign reference did not zero the counters this "
        "test reads, so the state is not shared and -030's R-2 predicate cannot "
        "be taken over this seam"
    )
    with _SimulatorCounter() as sim:
        foreign.sinkhorn_normalise(affinity)
    after_one = dispatch_counters()
    with _SimulatorCounter() as sim2:
        foreign.sinkhorn_normalise(affinity)
    after_two = foreign.dispatch_counters()
    print(
        f"[cross-module] after_reset=(0, 0) after_one_call={after_one} "
        f"after_two_calls={after_two} sim_calls={sim.calls + sim2.calls}"
    )
    assert after_one == (1, 0), f"expected (1, 0) after one dispatch, got {after_one}"
    assert after_two == (2, 0), (
        f"expected (2, 0) after two dispatches, got {after_two}; the counter "
        f"does not accumulate across calls, so it cannot supply -030's "
        f"per-layer-call total"
    )
    foreign.reset_dispatch_counters()
    assert dispatch_counters() == (0, 0)


# --------------------------------------------------------------------------- #
# Seam identity and geometry refusals.                                          #
# --------------------------------------------------------------------------- #
def test_seam_dispatches_to_the_kernel_this_increment_authors() -> None:
    """The seam dispatches to THIS module's kernel, read off the object.

    SCRATCH is checkable here rather than merely claimed: the kernel's module is
    this repository's, not ``nkilib``'s. The plan records that nkilib has 0 hits
    for sinkhorn and that no vendoring-precedent claim is available for this
    increment, so a vendor module appearing here would contradict the substrate
    declaration.
    """
    module, qualname = kernel_identity()
    print(f"[identity] kernel={module}.{qualname}")
    assert module == _MODULE, module
    assert qualname == "sinkhorn_kernel", qualname
    assert not module.startswith("nkilib"), (
        "the seam dispatches to a vendor kernel, but this increment is SCRATCH "
        "with ZERO precedent: nkilib has 0 sinkhorn/mhc members"
    )


# --------------------------------------------------------------------------- #
# `inc-glm53f-028b` -- ROW EXTENTS PAST ONE PARTITION TILE.                     #
#                                                                               #
# RUNTIME NOTE, not a criterion: the largest declared case is 8192 rows, which   #
# the kernel walks as 64 row tiles of 128, and every one of the 20 iterations    #
# touches all 64. The declared acceptance command carries `--timeout 60` per     #
# test. If the simulator needs longer than that on the largest case, the reading #
# to report is the timeout with its measured duration -- not a widened timeout   #
# and not a dropped case, both of which would change what was accepted.          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows", TILED_ROWS)
def test_tiled_rows_match_the_oracle_and_stay_doubly_stochastic(rows: int) -> None:
    """Both declared arms at a row extent that needs more than one tile.

    ONE dispatch per case, whatever the tile count: the tiling is inside the
    kernel, so the route instruments read exactly what they read for the 64-row
    case. That is the property `inc-glm53f-028b` had to preserve -- a host-side
    loop over 128-row slices would compute the same numbers and read one dispatch
    per slice.

    Arm 1 is the oracle comparison at the declared tolerances; arm 2 is the
    doubly-stochastic reading against the two targets, and the column target
    ``M / N`` is what makes it a real check across tiles: a kernel that scaled
    each tile by its OWN column sums would put ``tile_rows / N`` in each column
    instead and fail here while arm 1 still nearly passed.
    """
    tiles = row_tiles(rows, MHC_STREAMS)
    affinity = _affinity_rows(rows)
    assert tuple(affinity.shape) == (rows, N), tuple(affinity.shape)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = sinkhorn_normalise(affinity).to(torch.float32)
    reading = _assert_route(sim, 1, f"tiled-M{rows}")
    print(
        f"[tiled-M{rows}] tiles={len(tiles)} "
        f"tile_extent={row_tile_extent(MHC_STREAMS)} "
        f"first={tiles[0]} last={tiles[-1]} {reading}"
    )

    # Arm 2 first, because it does not depend on the oracle being right.
    row_dev, col_dev = _report_stochasticity(got, f"tiled-M{rows}")
    if not torch.isfinite(got).all():
        raise StochasticityError(f"M={rows}: the kernel returned non-finite values")
    if row_dev > STOCHASTIC_TOL:
        raise StochasticityError(
            f"M={rows}: worst row deviation {row_dev:.6e} exceeds the declared "
            f"{STOCHASTIC_TOL} against row target {row_target()}"
        )
    if col_dev > STOCHASTIC_TOL:
        raise StochasticityError(
            f"M={rows}: worst column deviation {col_dev:.6e} exceeds the declared "
            f"{STOCHASTIC_TOL} against column target {column_target(rows, N)}"
        )

    # NON-VACUITY on this case's own input: the fixture must fail the bar its
    # output passes, or a kernel that returned its input would read green here.
    in_row_dev, in_col_dev = _deviations(affinity)
    print(
        f"[tiled-M{rows}] input_row_deviation={in_row_dev:.6e} "
        f"input_column_deviation={in_col_dev:.6e} bound={STOCHASTIC_TOL}"
    )
    if in_row_dev <= STOCHASTIC_TOL and in_col_dev <= STOCHASTIC_TOL:
        raise VacuousControlError(
            f"M={rows}: the fixture already satisfies the doubly-stochastic bar, "
            f"so arm 2 would pass on a kernel that did nothing"
        )

    want = _sinkhorn_oracle_authored_here(affinity)
    abs_err = float((got - want).abs().max())
    rel_err = float(((got - want).abs() / (want.abs() + ATOL)).max())
    print(
        f"[tiled-M{rows}] max_abs_error={abs_err:.6e} max_rel_error={rel_err:.6e} "
        f"rtol={RTOL} atol={ATOL}"
    )
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_row_tiles_are_cut_on_block_boundaries() -> None:
    """No token's ``S x S`` block is split, at every declared row extent.

    This reads the same :func:`row_tiles` arithmetic the kernel runs, so it is a
    check on the tiling the kernel actually uses and not on a restatement of it.
    Four things are read per extent: every tile starts on a multiple of the block,
    no tile is taller than one partition tile, the heights sum to ``M`` so no row
    is dropped or served twice, and the count is above one so the case genuinely
    exercises tiling.
    """
    extent = row_tile_extent(MHC_STREAMS)
    print(f"[tiling] block={MHC_STREAMS} tile_extent={extent} pmax={PARTITION_MAX}")
    assert extent % MHC_STREAMS == 0, extent
    assert 0 < extent <= PARTITION_MAX, extent

    for rows in TILED_ROWS:
        tiles = row_tiles(rows, MHC_STREAMS)
        starts = [start for start, _ in tiles]
        heights = [height for _, height in tiles]
        print(
            f"[tiling] M={rows} tiles={len(tiles)} heights_first={heights[0]} "
            f"heights_last={heights[-1]} sum={sum(heights)}"
        )
        assert len(tiles) > 1, f"M={rows} did not tile, so it measures nothing new"
        assert sum(heights) == rows, (rows, sum(heights))
        assert all(start % MHC_STREAMS == 0 for start in starts), starts
        assert all(0 < height <= PARTITION_MAX for height in heights), heights


def test_the_moved_bound_admits_the_serving_extent() -> None:
    """``can_run_sinkhorn`` no longer refuses ``M`` above one partition tile.

    The bound moved, so the reading that used to be a refusal is now an
    admission, and it is measured at the largest declared extent rather than just
    above the old ceiling. ``can_run_kernel`` decides the route; this test only
    requires that the GEOMETRY check raises nothing, which is the half
    `inc-glm53f-028b` changed.
    """
    for rows in (PARTITION_MAX + 1,) + TILED_ROWS:
        verdict = can_run_sinkhorn(torch.zeros(1), rows, N)
        print(f"[admits] M={rows} N={N} can_run_sinkhorn={verdict}")
        assert isinstance(verdict, bool)


def test_a_block_taller_than_one_tile_is_refused_by_name() -> None:
    """A stream count no tile height can align to is refused, not split.

    The one geometry the tiling adds a refusal for. It cannot arise at the
    target's ``hc_mult 4``; it is checked because the refusal is what keeps the
    block-alignment property true rather than merely intended.
    """
    with pytest.raises(SinkhornError) as excinfo:
        can_run_sinkhorn(torch.zeros(1), 512, N, block=PARTITION_MAX + 1)
    message = str(excinfo.value)
    print(f"[block-refusal] message={message!r}")
    assert f"block={PARTITION_MAX + 1}" in message, message
    assert f"PARTITION_MAX={PARTITION_MAX}" in message, message

    with pytest.raises(SinkhornError) as excinfo:
        row_tile_extent(0)
    assert "block=0 must be positive" in str(excinfo.value)


@pytest.mark.parametrize(
    ("rows", "cols", "needle"),
    [
        (0, 4, "M=0 must be positive"),
        (64, 0, "N=0 must be positive"),
        (64, 513, "exceeds the Tensor Engine moving free bound"),
    ],
)
def test_refuses_inadmissible_geometry_by_name(
    rows: int, cols: int, needle: str
) -> None:
    """Every refusal is a NAMED error carrying the offending extent.

    Refusing rather than falling back is what P13 requires: a torch path for
    kernel-class work would be a design defect, so an extent this kernel cannot
    serve raises.
    """
    with pytest.raises(SinkhornError) as excinfo:
        can_run_sinkhorn(torch.zeros(1), rows, cols)
    message = str(excinfo.value)
    assert needle in message, f"[M={rows},N={cols}] message was: {message}"


def test_seam_refuses_non_2d_and_non_positive_iters() -> None:
    """The seam's own argument refusals, named rather than coerced."""
    with pytest.raises(SinkhornError) as excinfo:
        sinkhorn_normalise(torch.ones((2, 3, 4), dtype=torch.float32))
    assert "must be 2-D" in str(excinfo.value)

    for bad in (0, -1):
        with pytest.raises(SinkhornError) as excinfo:
            sinkhorn_normalise(_affinity(), iters=bad)
        assert f"iters={bad} must be positive" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# `inc-glm53f-028b` -- THE BATCHED `[T, S, S]` FORM.                            #
#                                                                               #
# RUNTIME NOTE, not a criterion: the largest case is 2048 blocks, walked as 16   #
# token tiles of 128 with 20 iterations over all of them. As with the tiled      #
# cases above, a `--timeout 60` expiry is reported as the timeout with its        #
# measured duration -- never a widened timeout and never a dropped case.         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tokens", BLOCK_EQUIV_TOKENS)
def test_blocks_kernel_equals_the_square_kernel_on_the_block_diagonal(
    tokens: int,
) -> None:
    """The batched form gives the square form's answer, block for block.

    THE CLAIM UNDER TEST is the one the batched kernel is built on: normalising
    ``T`` blocks independently equals normalising ``block_diag`` of them, because
    a block-diagonal matrix's row and column sums are its blocks' row and column
    sums and a zero stays zero under any rescaling. If that were false, the mHC
    layer would compute a different attention mix after `inc-glm53f-030b` switches
    it over, and no per-block reading of the batched output alone could tell.

    So both kernels run here, on the SAME blocks, in one case: the batched one on
    ``[T, S, S]`` and the square one on ``torch.block_diag`` of those blocks --
    two dispatches, which the route instruments read as ``2`` rather than ``1``.
    The diagonal blocks of the square result are then compared against the batched
    result at the declared tolerances.
    """
    blocks = _affinity_blocks(tokens)
    matrix = _block_diag_of(blocks)
    assert tuple(blocks.shape) == (tokens, N, N), tuple(blocks.shape)
    assert tuple(matrix.shape) == (tokens * N, tokens * N), tuple(matrix.shape)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = sinkhorn_normalise_blocks(blocks).to(torch.float32)
        square = sinkhorn_normalise(matrix).to(torch.float32)
    reading = _assert_route(sim, 2, f"blocks-equiv-T{tokens}")
    print(
        f"[blocks-equiv-T{tokens}] square_rows={tokens * N} "
        f"square_tiles={len(row_tiles(tokens * N, MHC_STREAMS))} "
        f"blocks_tiles={len(row_tiles(tokens, 1))} {reading}"
    )

    # The off-diagonal of the square result must still be exactly zero, which is
    # the premise of the equivalence rather than a side observation. If the square
    # kernel had leaked mass off the blocks, the two sides could not agree and the
    # premise -- not the batched kernel -- would be what failed.
    mask = torch.ones_like(square, dtype=torch.bool)
    for t in range(tokens):
        mask[t * N : (t + 1) * N, t * N : (t + 1) * N] = False
    off_diagonal_max = float(square[mask].abs().max()) if bool(mask.any()) else 0.0
    print(f"[blocks-equiv-T{tokens}] off_diagonal_max={off_diagonal_max:.6e}")
    assert off_diagonal_max == 0.0, off_diagonal_max

    want = _diagonal_blocks_of(square, tokens, N)
    abs_err = float((got - want).abs().max())
    signal = float((want - blocks).abs().max())
    print(
        f"[blocks-equiv-T{tokens}] max_abs_error={abs_err:.6e} rtol={RTOL} "
        f"atol={ATOL} distance_from_the_unnormalised_input={signal:.6e}"
    )
    if signal <= ATOL:
        raise VacuousControlError(
            f"T={tokens}: the normalised blocks are within atol of the raw input, "
            f"so this comparison would pass on a kernel that returned its input"
        )
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)

    # Arm 2 on the batched output, per block: each block is doubly stochastic on
    # its own, which is what the layer consumes.
    row_dev, col_dev = _report_block_stochasticity(got, f"blocks-equiv-T{tokens}")
    if row_dev > STOCHASTIC_TOL or col_dev > STOCHASTIC_TOL:
        raise StochasticityError(
            f"T={tokens}: worst row deviation {row_dev:.6e} and column deviation "
            f"{col_dev:.6e} against the declared {STOCHASTIC_TOL}"
        )


@pytest.mark.parametrize("tokens", BLOCK_SCALE_TOKENS)
def test_blocks_kernel_at_serving_scale_matches_the_per_block_oracle(
    tokens: int,
) -> None:
    """The serving extents, against the batched torch oracle. ONE dispatch each.

    No square comparison runs here and none can: ``T = 129`` is refused by the
    square form (see
    :func:`test_the_square_form_cannot_serve_what_the_blocks_form_serves`) and
    ``T = 2048`` would need a 256 MB matrix. That is the whole reason the batched
    form exists, so these two extents are checked against the oracle instead --
    arm 2 first, because it does not depend on the oracle being right.
    """
    blocks = _affinity_blocks(tokens)
    tiles = row_tiles(tokens, 1)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = sinkhorn_normalise_blocks(blocks).to(torch.float32)
    reading = _assert_route(sim, 1, f"blocks-scale-T{tokens}")
    print(
        f"[blocks-scale-T{tokens}] token_tiles={len(tiles)} first={tiles[0]} "
        f"last={tiles[-1]} {reading}"
    )
    assert tuple(got.shape) == (tokens, N, N), tuple(got.shape)

    row_dev, col_dev = _report_block_stochasticity(got, f"blocks-scale-T{tokens}")
    if not torch.isfinite(got).all():
        raise StochasticityError(f"T={tokens}: the kernel returned non-finite values")
    if row_dev > STOCHASTIC_TOL:
        raise StochasticityError(
            f"T={tokens}: worst row deviation {row_dev:.6e} exceeds the declared "
            f"{STOCHASTIC_TOL} against row target {row_target()}"
        )
    if col_dev > STOCHASTIC_TOL:
        raise StochasticityError(
            f"T={tokens}: worst column deviation {col_dev:.6e} exceeds the declared "
            f"{STOCHASTIC_TOL} against column target {column_target(N, N)}"
        )

    # NON-VACUITY on this case's own input.
    in_row_dev, in_col_dev = _block_deviations(blocks)
    print(
        f"[blocks-scale-T{tokens}] input_row_deviation={in_row_dev:.6e} "
        f"input_column_deviation={in_col_dev:.6e} bound={STOCHASTIC_TOL}"
    )
    if in_row_dev <= STOCHASTIC_TOL and in_col_dev <= STOCHASTIC_TOL:
        raise VacuousControlError(
            f"T={tokens}: the fixture already satisfies the doubly-stochastic bar, "
            f"so this arm would pass on a kernel that did nothing"
        )

    want = _blocks_oracle_authored_here(blocks)
    abs_err = float((got - want).abs().max())
    rel_err = float(((got - want).abs() / (want.abs() + ATOL)).max())
    print(
        f"[blocks-scale-T{tokens}] max_abs_error={abs_err:.6e} "
        f"max_rel_error={rel_err:.6e} rtol={RTOL} atol={ATOL}"
    )
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_the_square_form_cannot_serve_what_the_blocks_form_serves() -> None:
    """The square form REFUSES the scale extents; the batched form admits them.

    This is the increment's reason, measured. ``N = T * S`` rides the Tensor
    Engine's moving free axis in the square kernel, so ``T`` above
    ``MOVING_FMAX // S`` is refused by name -- and the refusal is quoted here
    rather than described. The byte counts are printed for the same reason: the
    square matrix's cost is a number, not an adjective.
    """
    for tokens in BLOCK_SCALE_TOKENS:
        square_side = tokens * N
        square_bytes = square_side * square_side * 4
        blocks_bytes = tokens * N * N * 4
        print(
            f"[why-blocks] T={tokens} square=[{square_side},{square_side}] "
            f"square_fp32_bytes={square_bytes} blocks_fp32_bytes={blocks_bytes} "
            f"ratio={square_bytes // blocks_bytes}x"
        )
        with pytest.raises(SinkhornError) as excinfo:
            can_run_sinkhorn(torch.zeros(1), square_side, square_side)
        message = str(excinfo.value)
        print(f"[why-blocks] T={tokens} square_refusal={message!r}")
        assert f"N={square_side}" in message, message
        assert f"moving free bound {MOVING_FMAX}" in message, message

        verdict = can_run_sinkhorn_blocks(torch.zeros(1), tokens, N, N)
        print(f"[why-blocks] T={tokens} can_run_sinkhorn_blocks={verdict}")
        assert isinstance(verdict, bool)


def test_the_batched_oracle_agrees_with_the_per_block_oracle() -> None:
    """The batched oracle is the landed one, block by block. Not a new algorithm.

    The control that lets the scale arm rest on :func:`_blocks_oracle_authored_here`
    at 2048 blocks: at ``T = 3`` the landed per-block oracle is run on each block
    and the two are compared. The tolerance here is tighter than the kernel arm's
    on purpose -- both sides are float64 torch running the same formulation, so
    only reduction ORDER can differ, and a real disagreement would mean the
    batching changed the maths.
    """
    blocks = _affinity_blocks(3)
    batched = _blocks_oracle_authored_here(blocks)
    per_block = torch.stack(
        [_sinkhorn_oracle_authored_here(block) for block in blocks.unbind(0)]
    )
    abs_err = float((batched - per_block).abs().max())
    print(f"[oracle-control] batched_vs_per_block_max_abs_error={abs_err:.6e}")
    torch.testing.assert_close(batched, per_block, rtol=1e-6, atol=1e-7)


def test_blocks_seam_dispatches_to_the_kernel_this_increment_authors() -> None:
    """The batched seam dispatches to THIS module's batched kernel, read off it.

    Read separately from :func:`kernel_identity` so the two kernels are told apart
    by name rather than inferred from a shape, and so a seam quietly wired to the
    other one -- which for ``T <= 128`` would return plausible numbers -- shows up
    as a changed reading.
    """
    module, qualname = blocks_kernel_identity()
    print(f"[identity] blocks_kernel={module}.{qualname}")
    assert module == _MODULE, module
    assert qualname == "sinkhorn_blocks_kernel", qualname
    assert not module.startswith("nkilib"), (
        "the batched seam dispatches to a vendor kernel, but nkilib has 0 "
        "sinkhorn/mhc members"
    )
    assert blocks_kernel_identity() != kernel_identity(), (
        "the two seams report the same kernel identity, so neither reading tells "
        "which kernel ran"
    )


def test_the_blocks_seam_has_no_torch_path_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the route unavailable the batched seam RAISES; it never computes torch.

    The counterpart of :func:`test_route_control_fallback_counter_discriminates`,
    and the opposite outcome on purpose: kernel-class work ships no torch path
    (P13), so this seam has none to count. Both counters must read zero -- a
    dispatch that raised is not a dispatch, and a fallback that does not exist
    cannot be taken. An mHC layer that silently normalised 2048 tokens in torch
    would be slow in a way no numeric acceptance could see, which is why the
    absence is measured here rather than left to the docstring.
    """
    blocks = _affinity_blocks(2)
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0, so this control is unarmed"
    )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        with pytest.raises(SinkhornError) as excinfo:
            sinkhorn_normalise_blocks(blocks)
    nki_dispatch, torch_fallback = dispatch_counters()
    message = str(excinfo.value)
    print(
        f"[blocks-route-control] nki_dispatch={nki_dispatch} "
        f"torch_fallback={torch_fallback} simulate_kernel_calls={sim.calls} "
        f"raise={message!r}"
    )
    assert "no torch path" in message, message
    assert nki_dispatch == 0, nki_dispatch
    assert torch_fallback == 0, torch_fallback
    assert sim.calls == 0, sim.calls


@pytest.mark.parametrize(
    ("tokens", "rows", "cols", "needle"),
    [
        (0, N, N, "T=0 must be positive"),
        (4, 0, N, "S=0 must be positive"),
        (4, 3, 4, "must be square"),
    ],
)
def test_blocks_form_refuses_inadmissible_geometry_by_name(
    tokens: int, rows: int, cols: int, needle: str
) -> None:
    """Every batched refusal names the offending extent, and refuses rather than
    falling back (P13)."""
    with pytest.raises(SinkhornError) as excinfo:
        can_run_sinkhorn_blocks(torch.zeros(1), tokens, rows, cols)
    message = str(excinfo.value)
    assert needle in message, f"[T={tokens},block={rows}x{cols}] message: {message}"


def test_blocks_seam_refuses_non_3d_and_non_positive_iters() -> None:
    """The batched seam's own argument refusals, named rather than coerced."""
    with pytest.raises(SinkhornError) as excinfo:
        sinkhorn_normalise_blocks(_affinity())
    assert "must be 3-D" in str(excinfo.value)

    for bad in (0, -1):
        with pytest.raises(SinkhornError) as excinfo:
            sinkhorn_normalise_blocks(_affinity_blocks(2), iters=bad)
        assert f"iters={bad} must be positive" in str(excinfo.value)


def test_the_batched_form_has_no_declared_token_ceiling() -> None:
    """No ``T`` is refused for being large, at any extent this run can name.

    The absence of a ceiling is the property, so it is read rather than assumed:
    the scale extents, the old square ceiling, and one an order of magnitude past
    serving all pass the geometry check. ``can_run_kernel`` decides the route;
    this reads only the geometry half.
    """
    for tokens in BLOCK_SCALE_TOKENS + (MOVING_FMAX // N, 32768):
        verdict = can_run_sinkhorn_blocks(torch.zeros(1), tokens, N, N)
        tiles = row_tiles(tokens, 1)
        print(
            f"[no-ceiling] T={tokens} token_tiles={len(tiles)} "
            f"can_run_sinkhorn_blocks={verdict}"
        )
        assert isinstance(verdict, bool)
        assert sum(height for _start, height in tiles) == tokens


# --------------------------------------------------------------------------- #
# The trace-refusal guard. A simulator pass is not a compile.                  #
# --------------------------------------------------------------------------- #
_TRACE_HOSTILE_COMPREHENSIONS = (
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _module_functions(source: str) -> dict[str, ast.FunctionDef]:
    """Every top-level-visible function in ``source``, by name."""
    return {
        node.name: node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    }


def _traced_callees(functions: dict[str, ast.FunctionDef], entry: str) -> set[str]:
    """The functions NKI would trace into from ``entry``, transitively.

    A kernel body is not the whole traced program: the tracer follows every plain
    Python call it can resolve, so the closure is what has to be clean, not just
    the decorated function.
    """
    seen: set[str] = set()
    frontier = {entry}
    while frontier:
        name = frontier.pop()
        if name in seen or name not in functions:
            continue
        seen.add(name)
        for node in ast.walk(functions[name]):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in functions
            ):
                frontier.add(node.func.id)
    seen.discard(entry)
    return seen


def _trace_hostile_sites(
    functions: dict[str, ast.FunctionDef], names: set[str]
) -> tuple[list[str], list[str], list[str]]:
    """``(raises, comprehensions, min_calls)`` as ``function:line`` strings."""
    raises: list[str] = []
    comprehensions: list[str] = []
    min_calls: list[str] = []
    for name in sorted(names):
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Raise):
                raises.append(f"{name}:{node.lineno}")
            elif isinstance(node, _TRACE_HOSTILE_COMPREHENSIONS):
                comprehensions.append(f"{name}:{node.lineno}")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "min"
            ):
                min_calls.append(f"{name}:{node.lineno}")
    return raises, comprehensions, min_calls


_TRACE_FIXTURE = '''
def helper_with_a_raise(x):
    if x < 1:
        raise ValueError("no")
    return x


def helper_with_a_comprehension(n):
    return [(i, min(i, n)) for i in range(n)]


def fake_kernel(a):
    return helper_with_a_raise(a) + len(helper_with_a_comprehension(a))
'''


def test_no_kernel_traces_into_a_raise_or_a_comprehension() -> None:
    """Neither kernel's traced call graph carries a construct NKI refuses.

    WHY THIS IS A TEST AND NOT A CONVENTION. Both kernels compiled cleanly under
    the simulator and then failed to specialize on hardware, on all eleven
    declared Tier C shapes, with the compiler saying "NKI does not support 'raise'
    statements" plus one "unsupported expression" -- because the tracer follows a
    kernel's plain Python callees, and this module's tile-geometry helpers raised.
    The simulator cannot see that: it executes the helper, and a branch not taken
    is nothing. So the property is read here, off the module's own source, at a
    cost of no device and no compile.

    THREE CONSTRUCTS ARE REFUSED, and the third is a live question rather than
    caution. The ``raise`` count is settled -- the compiler counted two on the
    square path and three on the batched one, exactly the statements the helpers
    held. The single "unsupported expression" was one on BOTH paths, which rules
    out the ``f``-strings (two and three) and leaves the two constructs that
    appeared exactly once each in the shared helper: a list comprehension and a
    ``min`` call. Which one it was is not settled, so both are refused, and this
    docstring is the record of that open question rather than a claim it is
    closed.

    The reading is armed on a planted fixture whose answer is known by
    construction, because a walker that found nothing would pass this module
    whatever it contained.
    """
    module = importlib.import_module(_MODULE)
    source = pathlib.Path(module.__file__).read_text()
    functions = _module_functions(source)

    for entry in ("sinkhorn_kernel", "sinkhorn_blocks_kernel"):
        assert entry in functions, entry
        callees = _traced_callees(functions, entry)
        raises, comprehensions, min_calls = _trace_hostile_sites(functions, callees)
        print(
            f"[trace-guard] {entry} traced_callees={sorted(callees)} "
            f"raise={raises} comprehension={comprehensions} min={min_calls}"
        )
        assert callees, (
            f"{entry} resolved no traced callees, so this reading covers nothing"
        )
        assert raises == [], (
            f"{entry} traces into a raise at {raises}; NKI refuses a traced "
            "raise, and the seam is where an inadmissible shape is refused"
        )
        assert comprehensions == [], (
            f"{entry} traces into a comprehension at {comprehensions}, one of the "
            "two candidates for the compiler's unsupported expression"
        )
        assert min_calls == [], (
            f"{entry} traces into a min call at {min_calls}, the other candidate "
            "for the compiler's unsupported expression"
        )

    # THE WALKER MUST BE ABLE TO FIND ALL THREE, or the zeros above say nothing.
    fixture = _module_functions(_TRACE_FIXTURE)
    fixture_callees = _traced_callees(fixture, "fake_kernel")
    f_raises, f_comprehensions, f_mins = _trace_hostile_sites(fixture, fixture_callees)
    print(
        f"[trace-guard-control] traced_callees={sorted(fixture_callees)} "
        f"raise={f_raises} comprehension={f_comprehensions} min={f_mins}"
    )
    assert fixture_callees == {
        "helper_with_a_raise",
        "helper_with_a_comprehension",
    }, fixture_callees
    assert len(f_raises) == 1, f_raises
    assert len(f_comprehensions) == 1, f_comprehensions
    assert len(f_mins) == 1, f_mins
