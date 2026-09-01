# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for `inc-glm53f-029` -- the mHC combine kernel, `hc_mult 4`.

Acceptance command (plan block ``#### inc-glm53f-029``, Tier N harness "as
`-025`")::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/mhc/test_hyper_connection.py \
      --timeout 60 -p no:cacheprovider

The two DECLARED cases, and what each one can prove
---------------------------------------------------
1. the **tolerance case** -- simulated output against a torch oracle,
   ``assert_close(rtol=1e-2, atol=1e-5)``, ``1/1`` tiny case;
2. the **pass-through identity case** -- the ``hc_mult`` weights set to the
   pass-through pattern, asserted at ``atol 1e-5`` with **no relative slack**,
   which the plan says "catches an indexing error the tolerance case would
   absorb".

**Why case 2 catches what case 1 absorbs, stated because it is the point of the
case.** Case 1's reference is an oracle *authored here*. If the same ``i``/``j``
mistake were made in the kernel and in the oracle, case 1 passes green. Case 2 has
no authored reference at all: with ``comb_res_mix = I`` and ``post_layer_mix = 0``
the expected output **is the input tensor**, so any mis-indexing shows up against
a bit-exact expectation that no mistake of this file's could move. That is the
same structural property `-028`'s arm 2 has, and it is why the two cases arm each
other rather than repeating each other:

* case 2 alone would be satisfied by a kernel that ignored both weight tensors
  and copied ``residual`` through -- :func:`test_identity_case_is_not_vacuous`
  measures that such an implementation FAILS case 1;
* case 1 alone would be satisfied by a transposed ``comb_res_mix`` reading if the
  oracle shared the error -- :func:`test_tolerance_case_detects_a_transposed_mix`
  measures that the fixture's mix is asymmetric enough to separate them, and
  :func:`test_the_two_upstream_spellings_agree` corroborates the convention
  against a SECOND, independent statement of it from the pinned base.

No tolerance number is invented, widened or narrowed anywhere in this file.
:data:`RTOL` and :data:`ATOL` are the plan's. The base's own combine test compares
at ``atol=5e-2`` because it returns bf16; that number is not used here.

Why the route predicate is an acceptance criterion and not a diagnostic (F1)
---------------------------------------------------------------------------
Case 1 compares simulated NKI output against a torch oracle. If the seam silently
took its torch path, *both* sides would be torch and the comparison would pass
green while measuring nothing about a kernel. So each declared case reads three
route instruments and reports each as a number:

1. the seam's own module-level dispatch counter (form R-1) -- ``nki_dispatch ==
   1``, ``torch_fallback == 0``;
2. ``can_run_kernel()`` -- ``True``;
3. real ``nki.simulator.simulate_kernel`` invocations on the F1 chain -- ``1`` per
   kernel call. Instrument 3 counts the VENDOR entry point, so a bug in
   instrument 1 cannot fake it.

Every zero is armed. :func:`test_route_control_fallback_counter_discriminates`
shows instrument 1 reading ``(0, 1)`` and instrument 3 reading ``0`` on the
fallback path; :func:`test_route_control_simulator_is_load_bearing` shows the
chain RAISING rather than quietly computing torch when the simulator is off.

The counters are also shown INDEPENDENT of `-028`'s
(:func:`test_counters_are_independent_of_the_sinkhorn_seam`), because
`inc-glm53f-030` reads both seams per layer call and needs two numbers, not one.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest
import torch

import nki
import nki.simulator

from vllm_neuron.functional.mhc.hyper_connection import (
    MHC_STREAMS,
    PARTITION_MAX,
    HyperConnectionError,
    can_run_hyper_connection,
    dispatch_counters,
    hyper_connection_combine,
    hyper_connection_torch_oracle,
    kernel_identity,
    reset_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# --------------------------------------------------------------------------- #
# The declared tiny case. S is read off MHC_STREAMS so the fixture and the      #
# target's `hc_mult` cannot drift apart; H is a multiple of 256, the            #
# granularity the base's own fast path is gated on.                            #
# --------------------------------------------------------------------------- #
T = 64
S = MHC_STREAMS  # 4
H = 256

#: The declared tolerance pair for the tolerance case, from the plan block.
RTOL = 1e-2
ATOL = 1e-5
#: The identity case is declared at "atol 1e-5 exactly". Read as atol-only: the
#: contrast the plan draws is against "the tolerance case", so the identity case
#: carries NO relative slack. This is the STRICTER of the two readings, and it is
#: the one asserted; the measured value is reported so either reading is
#: checkable from the transcript.
IDENTITY_RTOL = 0.0

_MODULE = "vllm_neuron.functional.mhc.hyper_connection"
_SINKHORN_MODULE = "vllm_neuron.functional.mhc.sinkhorn"


class RouteInstrumentError(AssertionError):
    """A route reading that is not what the plan declares."""


class VacuousControlError(AssertionError):
    """A control whose input could not have made it fail.

    A zero over vacuous input measures nothing, so the control refuses to report
    a pass it did not earn.
    """


# --------------------------------------------------------------------------- #
# Route instrumentation. Counts the VENDOR entry point, so it is independent of  #
# the seam counter it cross-checks. Same shape as `-028`'s.                      #
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
# Fixtures.                                                                    #
# --------------------------------------------------------------------------- #
def _inputs(seed: int = 29, rows: int = T, hidden: int = H):
    """The declared tiny case's four tensors, fp32, DETERMINISTIC by seed.

    Two properties are load-bearing and a draw could satisfy them by luck, so both
    are measured rather than assumed
    (:func:`test_fixture_mix_is_asymmetric_and_row_stochastic`):

    * ``comb_res_mix`` is **row-stochastic**, which is what a Sinkhorn stage hands
      the combine, and which is also upstream's own pre-Sinkhorn form
      (``softmax(-1)``);
    * ``comb_res_mix`` is **ASYMMETRIC**, without which an ``i``/``j`` transpose
      would be invisible and the tolerance case could not catch the error the
      identity case is there to catch.

    ``residual`` and ``x`` are signed ``randn``, not made positive: a sign error
    in a mixing kernel should be visible, and the declared ``atol=1e-5`` is the
    binding term at these magnitudes so occasional near-zero outputs from
    cancellation do not inflate the comparison.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, hidden), generator=g, dtype=torch.float32)
    residual = torch.randn((rows, S, hidden), generator=g, dtype=torch.float32)
    post_layer_mix = torch.rand((rows, S, 1), generator=g, dtype=torch.float32)
    comb_res_mix = torch.softmax(
        torch.randn((rows, S, S), generator=g, dtype=torch.float32), dim=-1
    )
    return x, residual, post_layer_mix, comb_res_mix


def _pass_through_weights(rows: int = T):
    """The pass-through pattern: ``comb_res_mix = I_S`` and ``post_layer_mix = 0``.

    Defined HERE rather than in the module on purpose. The identity case's whole
    value is that its expectation -- the ``residual`` tensor itself -- is not
    authored by the code under test. Taking the pattern from the module would put
    the module back on both sides of the comparison.

    Under ``out_j = post_layer_mix_j * x + sum_i comb_ij * residual_i``: with
    ``comb = I`` the sum collapses to ``residual_j``, and with ``post = 0`` the
    layer-output term vanishes. So ``out == residual``, and in fp32 that is
    BIT-exact -- multiplying by exactly ``1.0`` and adding exact ``0.0`` are both
    lossless.
    """
    ident = torch.eye(S, dtype=torch.float32).expand(rows, S, S).contiguous()
    zero_post = torch.zeros((rows, S, 1), dtype=torch.float32)
    return zero_post, ident


def _oracle_authored_here(x, residual, post_layer_mix, comb_res_mix):
    """The base's SECOND spelling of the combine: ``bmm(comb.mT, residual)``.

    Transcribed from ``tests/kernels/test_mhc_kernels.py``'s ``mhc_post_ref``,
    whose own docstring sources it from the TileLang reference repository. The
    module's oracle uses the ``einsum`` spelling from
    ``vllm/model_executor/kernels/mhc/torch.py``, so the two references here are
    two INDEPENDENT statements of the ``i``/``j`` convention rather than one
    statement and its restatement -- and the ``.mT`` is where the convention lives.

    Kept in fp32 rather than cast to bf16 as the base's version does: that cast is
    what forces the base's own looser ``atol=5e-2``.
    """
    term2 = torch.bmm(comb_res_mix.mT.to(torch.float32), residual.to(torch.float32))
    return x.to(torch.float32).unsqueeze(-2) * post_layer_mix.to(torch.float32) + term2


def _errors(got, want) -> tuple[float, float]:
    """``(max_abs, max_rel)`` -- numbers, not verdicts."""
    got = got.to(torch.float32)
    want = want.to(torch.float32)
    max_abs = float((got - want).abs().max())
    max_rel = float(((got - want).abs() / (want.abs() + ATOL)).max())
    return max_abs, max_rel


# --------------------------------------------------------------------------- #
# DECLARED CASE 1 -- the tolerance case.                                       #
# --------------------------------------------------------------------------- #
def test_output_matches_torch_oracle_tiny_case() -> None:
    """Simulated NKI output vs a torch oracle, ``rtol=1e-2``, ``atol=1e-5``.

    The plan's declared Expected: "simulated output vs a torch oracle,
    ``assert_close(rtol=1e-2, atol=1e-5)``, 1/1 tiny case".
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs()
    assert tuple(residual.shape) == (T, S, H), tuple(residual.shape)
    assert S == 4, f"hc_mult read {S}, the target declares 4"

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    _assert_route(sim, 1, "tolerance-case")

    want = _oracle_authored_here(x, residual, post_layer_mix, comb_res_mix)
    max_abs, max_rel = _errors(got, want)
    print(
        f"[tolerance-case] max_abs_error={max_abs:.6e} max_rel_error={max_rel:.6e} "
        f"rtol={RTOL} atol={ATOL} want_absmin={float(want.abs().min()):.6e} "
        f"want_absmax={float(want.abs().max()):.6e}"
    )
    if float(want.abs().max()) == 0.0:
        raise VacuousControlError(
            "the oracle produced an all-zero reference, so the comparison would "
            "pass over empty input; refusing to report a pass"
        )
    assert tuple(got.shape) == (T, S, H), tuple(got.shape)
    torch.testing.assert_close(got.to(torch.float32), want, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# DECLARED CASE 2 -- the pass-through identity case, at atol 1e-5 exactly.       #
# --------------------------------------------------------------------------- #
def test_pass_through_identity_case_is_exact() -> None:
    """``comb = I``, ``post = 0`` => ``out == residual``, at atol 1e-5, no rtol.

    The plan's declared Expected: "an identity case (``hc_mult`` weights set to
    the pass-through pattern) asserted at **atol 1e-5** exactly, which catches an
    indexing error the tolerance case would absorb".

    The reference is the ``residual`` tensor itself, so nothing this file authored
    can be wrong in the same direction as the kernel.
    """
    x, residual, _, _ = _inputs()
    zero_post, ident = _pass_through_weights()

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = hyper_connection_combine(x, residual, zero_post, ident)
    _assert_route(sim, 1, "identity-case")

    max_abs, max_rel = _errors(got, residual)
    print(
        f"[identity-case] max_abs_error={max_abs:.6e} atol={ATOL} "
        f"rtol={IDENTITY_RTOL} max_rel_error={max_rel:.6e} "
        f"bit_exact={max_abs == 0.0} "
        f"residual_absmax={float(residual.abs().max()):.6e}"
    )
    if float(residual.abs().max()) == 0.0:
        raise VacuousControlError("the residual fixture is all zero")
    assert tuple(got.shape) == (T, S, H), tuple(got.shape)
    torch.testing.assert_close(
        got.to(torch.float32), residual, rtol=IDENTITY_RTOL, atol=ATOL
    )


def test_identity_case_is_not_vacuous() -> None:
    """A kernel that IGNORED both weight tensors would pass case 2. Case 1 stops it.

    Measured rather than argued: the trivial "return ``residual`` unchanged"
    implementation is compared against case 1's oracle on case 1's own fixture,
    and must MISS by more than the declared tolerance. Without this reading, case
    2 would be a test a do-nothing kernel satisfies and no other case in this file
    would say so.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs()
    want = _oracle_authored_here(x, residual, post_layer_mix, comb_res_mix)
    max_abs, max_rel = _errors(residual, want)
    print(
        f"[identity-non-vacuity] a pass-through implementation vs the tolerance "
        f"case's oracle: max_abs_error={max_abs:.6e} max_rel_error={max_rel:.6e} "
        f"rtol={RTOL} atol={ATOL}"
    )
    assert max_rel > RTOL, (
        f"returning `residual` unchanged sits within the declared rtol {RTOL} of "
        f"the tolerance case's oracle (max_rel {max_rel:.6e}), so case 2 could be "
        f"passed by a kernel that computes nothing and case 1 would not notice"
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            residual.to(torch.float32), want, rtol=RTOL, atol=ATOL
        )


# --------------------------------------------------------------------------- #
# The i/j convention: corroborated, and shown to be DISCRIMINABLE.               #
# --------------------------------------------------------------------------- #
def test_the_two_upstream_spellings_agree() -> None:
    """The base states the combine twice; the two statements must agree.

    ``einsum("...ij,...ih->...jh", comb, residual)`` -- the plain-torch backend,
    which this module's oracle uses -- against ``bmm(comb.mT, residual)`` -- the
    TileLang reference, which this file's oracle uses. The ``.mT`` is the entire
    content of the ``i``/``j`` convention, so agreement between the two spellings
    is what makes the convention corroborated rather than assumed.

    If these two ever disagreed, every numeric reading in this file would be
    resting on a convention nobody had checked.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs()
    einsum_side = hyper_connection_torch_oracle(
        x, residual, post_layer_mix, comb_res_mix
    )
    bmm_side = _oracle_authored_here(x, residual, post_layer_mix, comb_res_mix)
    max_abs, max_rel = _errors(einsum_side, bmm_side)
    print(
        f"[oracle-cross-check] einsum vs bmm(comb.mT, residual): "
        f"max_abs_difference={max_abs:.6e} max_rel={max_rel:.6e} atol={ATOL}"
    )
    torch.testing.assert_close(einsum_side, bmm_side, rtol=0.0, atol=ATOL)


def test_tolerance_case_detects_a_transposed_mix() -> None:
    """The fixture's mix is asymmetric enough that ``comb[j, i]`` reads WRONG.

    The identity case cannot catch a transpose -- ``I`` is symmetric -- so the
    transpose is case 1's to catch, and that only works if the fixture's mix is
    genuinely asymmetric. Measured on the declared fixture, with the declared
    tolerance unchanged: only the reference's weight tensor is transposed.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs()
    transposed = comb_res_mix.transpose(-1, -2).contiguous()
    if torch.equal(transposed, comb_res_mix):
        raise VacuousControlError(
            "the fixture's mix is symmetric, so transposing it changed nothing "
            "and this control cannot discriminate"
        )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    _assert_route(sim, 1, "transpose-control")

    wrong = _oracle_authored_here(x, residual, post_layer_mix, transposed)
    max_abs, max_rel = _errors(got, wrong)
    print(
        f"[transpose-control] kernel vs a TRANSPOSED-mix reference: "
        f"max_abs_error={max_abs:.6e} max_rel_error={max_rel:.6e} vs rtol={RTOL}"
    )
    assert max_rel > RTOL, (
        f"transposing the mix moved the comparison by only {max_rel:.6e}, inside "
        f"the declared rtol {RTOL}; case 1 therefore cannot catch an i/j swap and "
        f"nothing in this file could"
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            got.to(torch.float32), wrong, rtol=RTOL, atol=ATOL
        )


def test_cyclic_permutation_routes_streams_exactly() -> None:
    """An EXACT case that is also transpose-sensitive, unlike the identity.

    Not a substitute for the declared identity case -- an addition to it. A cyclic
    shift ``comb[i, j] = 1 iff j == (i + 1) mod S`` is a pass-through in the
    routing sense, so the expected output is again reference-free (a roll of
    ``residual`` along the stream axis, bit-exact), but the matrix is NOT
    symmetric, so it combines the identity case's exactness with the transpose
    sensitivity the identity case lacks.
    """
    x, residual, _, _ = _inputs()
    zero_post, _ = _pass_through_weights()
    perm = torch.zeros((S, S), dtype=torch.float32)
    for i in range(S):
        perm[i, (i + 1) % S] = 1.0
    perm_b = perm.expand(T, S, S).contiguous()
    expected = torch.roll(residual, shifts=1, dims=1)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = hyper_connection_combine(x, residual, zero_post, perm_b)
    _assert_route(sim, 1, "permutation-control")

    max_abs, _ = _errors(got, expected)
    against_untouched, _ = _errors(got, residual)
    print(
        f"[permutation-control] vs roll(residual, +1, stream): "
        f"max_abs_error={max_abs:.6e} atol={ATOL} bit_exact={max_abs == 0.0}; "
        f"vs the UNROLLED residual: max_abs={against_untouched:.6e} "
        f"-- large, so the roll really happened"
    )
    assert against_untouched > ATOL, (
        "the permutation produced the untouched residual, so the stream axis was "
        "not routed at all"
    )
    torch.testing.assert_close(got.to(torch.float32), expected, rtol=0.0, atol=ATOL)


def test_fixture_mix_is_asymmetric_and_row_stochastic() -> None:
    """Both load-bearing fixture properties, measured rather than assumed."""
    _, _, _, comb_res_mix = _inputs()
    row_sums = comb_res_mix.sum(dim=-1)
    worst_row = float((row_sums - 1.0).abs().max())
    asymmetry = float(
        (comb_res_mix - comb_res_mix.transpose(-1, -2)).abs().max()
    )
    print(
        f"[fixture] worst_row_sum_deviation={worst_row:.6e} "
        f"max_asymmetry={asymmetry:.6e} min_weight="
        f"{float(comb_res_mix.min()):.6e}"
    )
    assert worst_row <= 1e-6, f"the mix is not row-stochastic: {worst_row:.6e}"
    assert asymmetry > 1e-2, (
        f"the mix is near-symmetric (max asymmetry {asymmetry:.6e}), so an i/j "
        f"transpose would be invisible to the tolerance case"
    )
    assert float(comb_res_mix.min()) > 0.0, "a softmax row produced a zero weight"


def test_stream_count_matches_the_target_hc_mult() -> None:
    """``MHC_STREAMS`` is the target's ``hc_mult``, and it is imported, not restated.

    The constant lives in `-028`'s module, which records that this increment and
    `-030` are sized by the same number. Reading it through THIS module's
    namespace is what proves the import is in place rather than a second copy.
    """
    sinkhorn = importlib.import_module(_SINKHORN_MODULE)
    combine = importlib.import_module(_MODULE)
    print(
        f"[constants] MHC_STREAMS={combine.MHC_STREAMS} "
        f"PARTITION_MAX={combine.PARTITION_MAX} "
        f"shared_with_sinkhorn="
        f"{combine.MHC_STREAMS is sinkhorn.MHC_STREAMS}"
    )
    assert combine.MHC_STREAMS == 4
    assert combine.MHC_STREAMS == sinkhorn.MHC_STREAMS
    assert combine.PARTITION_MAX == sinkhorn.PARTITION_MAX == 128


# --------------------------------------------------------------------------- #
# NON-VACUITY of the numeric comparison (D1.5): it must be able to FAIL.         #
# --------------------------------------------------------------------------- #
def test_numeric_comparison_is_armed() -> None:
    """Compare the kernel's output against a PERTURBED reference.

    Without this arm, a small ``max_abs_error`` would be indistinguishable from an
    unwired comparison. The declared tolerance pair is used unchanged; only the
    reference's input is perturbed, and the perturbation is shown to have changed
    the reference before the failure is required.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs()
    perturbed = residual.clone()
    perturbed[0, 0, 0] = perturbed[0, 0, 0] + 4.0
    if torch.equal(perturbed, residual):
        raise VacuousControlError("injection changed nothing -- control is vacuous")

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    _assert_route(sim, 1, "armed-control")

    wrong = _oracle_authored_here(x, perturbed, post_layer_mix, comb_res_mix)
    max_abs, max_rel = _errors(got, wrong)
    print(
        f"[armed-control] residual[0,0,0] += 4.0: max_abs_error={max_abs:.6e} "
        f"max_rel_error={max_rel:.6e} vs rtol={RTOL} atol={ATOL}"
    )
    assert max_abs > ATOL, (
        f"perturbing one residual entry moved the comparison by only "
        f"{max_abs:.6e}, inside the declared atol {ATOL}; the comparison is "
        f"therefore not a discriminator and a green case 1 would mean nothing"
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            got.to(torch.float32), wrong, rtol=RTOL, atol=ATOL
        )


# --------------------------------------------------------------------------- #
# ROUTE CONTROLS -- so every zero above is a measurement.                        #
# --------------------------------------------------------------------------- #
def test_route_control_fallback_counter_discriminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the simulator disabled the seam takes the torch path, and it is COUNTED.

    This is the arm that makes ``torch_fallback == 0`` above meaningful: the
    counter is shown reading ``1`` and ``nki_dispatch`` reading ``0``, through the
    real gate rather than a mock. It is also the measured form of the plan's claim
    that a pure-torch implementation yields ``0`` dispatches.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs()
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0, so this control is unarmed"
    )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    nki_dispatch, torch_fallback = dispatch_counters()
    print(
        f"[route-control] nki_dispatch={nki_dispatch} "
        f"torch_fallback={torch_fallback} simulate_kernel_calls={sim.calls}"
    )
    assert nki_dispatch == 0, f"expected 0 NKI dispatches, got {nki_dispatch}"
    assert torch_fallback == 1, f"expected 1 torch fallback, got {torch_fallback}"
    assert sim.calls == 0, f"the simulator ran {sim.calls} times with it disabled"
    assert tuple(out.shape) == (T, S, H)


def test_route_control_simulator_is_load_bearing() -> None:
    """The NKI chain RAISES without the simulator rather than computing torch.

    Recorded because it is what forecloses the F1 false green BELOW this
    repository's seam: if the HOP silently degraded to a torch path, a green
    numeric comparison could not be attributed to a kernel at all, and no counter
    of this module's could detect it.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs()
    saved = os.environ.get("NKI_SIMULATOR")
    os.environ["NKI_SIMULATOR"] = "0"
    try:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from vllm_neuron.functional.mhc.hyper_connection import (
            hyper_connection_kernel,
        )

        with pytest.raises(RuntimeError) as excinfo:
            wrap_nki(hyper_connection_kernel)(
                x=x,
                residual=residual,
                post_layer_mix=post_layer_mix,
                comb_res_mix=comb_res_mix,
            )
    finally:
        if saved is None:
            os.environ.pop("NKI_SIMULATOR", None)
        else:
            os.environ["NKI_SIMULATOR"] = saved

    message = str(excinfo.value)
    print(f"[route-control] simulator_off_raise={message[:160]!r}")
    assert "simulator" in message.lower(), message


# --------------------------------------------------------------------------- #
# The counters are MODULE-LEVEL and INDEPENDENT -- `-030` depends on both.        #
# --------------------------------------------------------------------------- #
def test_dispatch_counters_are_module_level_state_reachable_from_elsewhere() -> None:
    """Another module can zero and read these counters. `-030` needs exactly this.

    `inc-glm53f-030`'s route predicate is form R-2 over THIS seam together with
    `-028`'s: its own test module resets and reads the counters this module owns,
    per layer call. A test-local counter, or one only this file could reset, would
    pass this increment and break that one.

    Measured rather than asserted by inspection: the module is re-acquired through
    ``importlib`` -- the same mechanism another test module's import uses -- one
    reference RESETS, the seam is driven, and the OTHER reference READS. Then the
    counter is shown to ACCUMULATE across calls, because `-030` reads a
    per-layer-call total and a counter saturating at 1 could not supply one.
    """
    foreign = importlib.import_module(_MODULE)
    assert foreign is sys.modules[_MODULE]
    # Identity, not just equality: both references address one counter object.
    assert foreign.dispatch_counters is dispatch_counters
    assert foreign.reset_dispatch_counters is reset_dispatch_counters

    x, residual, post_layer_mix, comb_res_mix = _inputs()
    foreign.reset_dispatch_counters()
    assert dispatch_counters() == (0, 0), (
        "a reset through the foreign reference did not zero the counters this "
        "test reads, so the state is not shared and -030's R-2 predicate cannot "
        "be taken over this seam"
    )
    with _SimulatorCounter() as sim:
        foreign.hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    after_one = dispatch_counters()
    with _SimulatorCounter() as sim2:
        foreign.hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    after_two = foreign.dispatch_counters()
    print(
        f"[cross-module] after_reset=(0, 0) after_one_call={after_one} "
        f"after_two_calls={after_two} sim_calls={sim.calls + sim2.calls}"
    )
    assert after_one == (1, 0), f"expected (1, 0) after one dispatch, got {after_one}"
    assert after_two == (2, 0), (
        f"expected (2, 0) after two dispatches, got {after_two}; the counter does "
        f"not accumulate across calls, so it cannot supply -030's per-layer-call "
        f"total"
    )
    foreign.reset_dispatch_counters()
    assert dispatch_counters() == (0, 0)


def test_counters_are_independent_of_the_sinkhorn_seam() -> None:
    """This seam's counter and `-028`'s must be two numbers, not one.

    `inc-glm53f-030`'s R-2 predicate reads BOTH seams per layer call and asserts
    each was entered exactly once. If the two modules shared a counter object --
    or if either reset cleared both -- that predicate would be unable to tell "both
    kernels ran once" from "one kernel ran twice", and `-030` would be built on a
    reading that cannot discriminate.

    Measured in BOTH directions on one fixture.
    """
    sinkhorn = importlib.import_module(_SINKHORN_MODULE)
    combine = importlib.import_module(_MODULE)

    assert combine._COUNTERS is not sinkhorn._COUNTERS, (
        "the two seams share one counter object, so -030 cannot read two numbers"
    )

    x, residual, post_layer_mix, comb_res_mix = _inputs(rows=8, hidden=32)
    g = torch.Generator().manual_seed(29)
    affinity = torch.exp(
        torch.rand((8, S), generator=g, dtype=torch.float32) * 2.0 - 1.0
    )

    combine.reset_dispatch_counters()
    sinkhorn.reset_dispatch_counters()

    # Drive THIS seam only.
    with _SimulatorCounter():
        combine.hyper_connection_combine(
            x, residual, post_layer_mix, comb_res_mix
        )
    after_combine = (combine.dispatch_counters(), sinkhorn.dispatch_counters())

    # Drive `-028`'s seam only.
    with _SimulatorCounter():
        sinkhorn.sinkhorn_normalise(affinity)
    after_sinkhorn = (combine.dispatch_counters(), sinkhorn.dispatch_counters())

    # A reset of ONE must not clear the OTHER.
    combine.reset_dispatch_counters()
    after_reset = (combine.dispatch_counters(), sinkhorn.dispatch_counters())

    print(
        f"[seam-independence] (combine, sinkhorn) after combine call="
        f"{after_combine} after sinkhorn call={after_sinkhorn} "
        f"after combine reset={after_reset}"
    )
    assert after_combine == ((1, 0), (0, 0)), (
        f"driving this seam moved -028's counter: {after_combine}"
    )
    assert after_sinkhorn == ((1, 0), (1, 0)), (
        f"driving -028's seam moved this one, or did not move its own: "
        f"{after_sinkhorn}"
    )
    assert after_reset == ((0, 0), (1, 0)), (
        f"resetting this seam cleared -028's counter too: {after_reset}"
    )
    sinkhorn.reset_dispatch_counters()


# --------------------------------------------------------------------------- #
# Seam identity and geometry refusals.                                          #
# --------------------------------------------------------------------------- #
def test_seam_dispatches_to_the_kernel_this_increment_authors() -> None:
    """The seam dispatches to THIS module's kernel, read off the object.

    SCRATCH is checkable here rather than merely claimed: the kernel's module is
    this repository's, not ``nkilib``'s. The plan records that nkilib has 0 hits
    for mhc and that no vendoring-precedent claim is available for this increment,
    so a vendor module appearing here would contradict the substrate declaration.
    """
    module, qualname = kernel_identity()
    print(f"[identity] kernel={module}.{qualname}")
    assert module == _MODULE, module
    assert qualname == "hyper_connection_kernel", qualname
    assert not module.startswith("nkilib"), (
        "the seam dispatches to a vendor kernel, but this increment is SCRATCH: "
        "nkilib has 0 mhc/hyper-connection members"
    )


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        ("rows_over_max", f"exceeds PARTITION_MAX={PARTITION_MAX}"),
        ("x_rows", "expected [T, H]"),
        ("x_hidden", "expected [T, H]"),
        ("x_rank", "x must be 2-D [T, H]"),
        ("post_shape", "expected [T, S, 1]"),
        ("comb_shape", "expected [T, S, S]"),
        ("residual_rank", "residual must be 3-D [T, S, H]"),
    ],
)
def test_refuses_inadmissible_geometry_by_name(mutate: str, needle: str) -> None:
    """Every refusal is a NAMED error carrying the offending shape.

    Refusing rather than falling back is what P13 requires: a torch path for
    kernel-class work would be a design defect, so a shape this kernel cannot
    serve raises. Measured on the BARE kernel, these same shapes trap inside NKI
    or numpy with messages that name no argument -- an out-of-bound access on
    "tensor `unnamed`", or a raw broadcast failure.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs(rows=8, hidden=32)
    if mutate == "rows_over_max":
        x, residual, post_layer_mix, comb_res_mix = _inputs(
            rows=PARTITION_MAX + 1, hidden=8
        )
    elif mutate == "x_rows":
        x = x[:-1]
    elif mutate == "x_hidden":
        x = x[:, :-1]
    elif mutate == "x_rank":
        x = x.unsqueeze(0)
    elif mutate == "post_shape":
        post_layer_mix = post_layer_mix.squeeze(-1)
    elif mutate == "comb_shape":
        comb_res_mix = comb_res_mix[:, :, :-1]
    elif mutate == "residual_rank":
        residual = residual.reshape(8, -1)

    with pytest.raises(HyperConnectionError) as excinfo:
        can_run_hyper_connection(x, residual, post_layer_mix, comb_res_mix)
    message = str(excinfo.value)
    assert needle in message, f"[{mutate}] message was: {message}"


def test_seam_refuses_before_the_kernel_traps() -> None:
    """The seam refuses on the same shapes, not just the gate helper.

    ``can_run_hyper_connection`` is what the refusal tests above drive directly.
    This measures that the SEAM reaches that check before it reaches the kernel,
    so a caller gets the named error rather than the NKI trap.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs(
        rows=PARTITION_MAX + 1, hidden=8
    )
    with pytest.raises(HyperConnectionError) as excinfo:
        hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    assert f"exceeds PARTITION_MAX={PARTITION_MAX}" in str(excinfo.value)
    print(f"[seam-refusal] {str(excinfo.value)[:140]!r}")
