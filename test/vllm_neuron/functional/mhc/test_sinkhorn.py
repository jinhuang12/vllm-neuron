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
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest
import torch

import nki
import nki.simulator

from vllm_neuron.functional.mhc.sinkhorn import (
    MHC_STREAMS,
    PARTITION_MAX,
    SINKHORN_DENOM_EPS,
    SINKHORN_ITERS,
    SinkhornError,
    can_run_sinkhorn,
    column_target,
    dispatch_counters,
    kernel_identity,
    reset_dispatch_counters,
    row_target,
    sinkhorn_normalise,
    sinkhorn_torch_oracle,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# --------------------------------------------------------------------------- #
# The declared fixture. [64, 4] is the plan's; 4 is read off MHC_STREAMS so     #
# the fixture and the target's `hc_mult` cannot drift apart.                    #
# --------------------------------------------------------------------------- #
M = 64
N = MHC_STREAMS  # 4

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
    """
    generator = torch.Generator().manual_seed(seed)
    logits = torch.rand((M, N), generator=generator, dtype=torch.float32) * 2.0 - 1.0
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


@pytest.mark.parametrize(
    ("rows", "cols", "needle"),
    [
        (PARTITION_MAX + 1, 4, f"exceeds PARTITION_MAX={PARTITION_MAX}"),
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
