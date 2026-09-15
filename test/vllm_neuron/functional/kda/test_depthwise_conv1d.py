# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for `inc-glm53f-034` -- the KDA prefill depthwise conv1d WRAP.

Acceptance command (plan block ``#### inc-glm53f-034``, Tier N harness "as
`-025`")::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/kda/test_depthwise_conv1d.py \
      -q -s --timeout 60 -p no:cacheprovider

The two DECLARED cases
----------------------
1. the **reference case** -- one tiny case, simulated NKI output against
   ``nkilib``'s own torch reference for this kernel at
   ``assert_close(rtol=1e-2, atol=1e-5)``, worst error reported;
2. the **unit-impulse case** -- asserted at ``atol=1e-5``, recovering the kernel
   taps exactly and so proving the wrap's argument order.

No tolerance number is invented, widened or narrowed anywhere in this file.
:data:`RTOL`, :data:`ATOL` and :data:`IMPULSE_ATOL` are the plan block's.

What makes case 2 an ARGUMENT-ORDER proof rather than a second numeric arm
-------------------------------------------------------------------------
Case 2 does not compare the kernel against the reference. It compares the kernel
against a **closed form computed in this file** from the taps alone: with a unit
impulse at input position ``p``, cross-correlation puts ``filter[p - q]`` at
output position ``q``, so the output row is the tap vector REVERSED and then
zero-padded. If the wrap passed the image where the filter belongs -- or the
filter where the image belongs -- that closed form would not come back. The
reference's agreement is recorded beside it as a second, independent reading.

:func:`test_impulse_expectation_is_not_order_blind` measures that the claim has
content: the forward taps and the reversed taps are shown to DIFFER, so
"reversed" is a falsifiable statement about order rather than a symmetric
coincidence.

Why the route predicate is an acceptance criterion and not a diagnostic (F1)
---------------------------------------------------------------------------
Case 1 compares simulated NKI output against a torch reference. If the seam
silently took its torch path, *both* sides would be torch and the comparison
would pass green while measuring nothing about a kernel. So each declared case
reads three route instruments and reports each as a number:

1. the seam's own module-level dispatch counter (form R-1) -- ``nki_dispatch ==
   1``, ``torch_fallback == 0``;
2. ``can_run_kernel()`` -- ``True``;
3. real ``nki.simulator.simulate_kernel`` invocations -- ``1`` per case.
   Instrument 3 counts the VENDOR entry point, so a bug in instrument 1 cannot
   fake it.

The plan block states the discrimination directly: "A pure-torch implementation
yields ``0`` and therefore cannot pass."

Every zero is armed. :func:`test_route_control_fallback_counter_discriminates`
shows instrument 1 reading ``(0, 1)`` and instrument 3 reading ``0`` on the
fallback path; :func:`test_route_control_simulator_is_load_bearing` shows the
chain RAISING rather than quietly computing torch when the simulator is off; and
:func:`test_feature_group_count_derivation_is_load_bearing` shows the substrate
kernel REFUSING at its own default, so the seam's derivation of that argument is
measured rather than assumed.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest
import torch

import nki
import nki.simulator

from vllm_neuron.functional.kda.depthwise_conv1d import (
    LNC_SHARDS,
    NO_PADDING,
    UNIT_DILATION,
    UNIT_STRIDE,
    KdaDepthwiseConv1dError,
    can_run_depthwise_conv1d,
    depthwise_conv1d,
    depthwise_conv1d_torch_reference,
    dispatch_counters,
    kernel_identity,
    output_width,
    reset_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# --------------------------------------------------------------------------- #
# The declared tiny case. C is a multiple of LNC_SHARDS by construction, read   #
# off the constant so the fixture and the module's refusal cannot drift apart.  #
# --------------------------------------------------------------------------- #
BATCH = 1
CHANNELS = 4 * LNC_SHARDS  # 8
WIDTH = 16
TAPS = 4
Q = output_width(WIDTH, TAPS)  # 13

#: The declared tolerance pair for the reference case, from the plan block.
RTOL = 1e-2
ATOL = 1e-5
#: The declared bound for the unit-impulse case, from the plan block.
IMPULSE_ATOL = 1e-5

#: Where the unit impulse sits. Interior, so the reversed tap window is fully
#: inside the input and the closed form below is exact rather than truncated.
IMPULSE_AT = TAPS - 1

_MODULE = "vllm_neuron.functional.kda.depthwise_conv1d"

#: The substrate member this increment WRAPS. Asserted, not assumed.
SUBSTRATE_MODULE = "nkilib.experimental.conv.depthwise_conv1d"
SUBSTRATE_QUALNAME = "depthwise_conv1d_implicit_gemm"


class RouteInstrumentError(AssertionError):
    """A route reading that is not what the plan declares."""


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
            f"declared {expected_dispatches}. A numeric pass without a "
            f"simulator call is the F1 false green. {reading}"
        )
    return reading


# --------------------------------------------------------------------------- #
# Fixtures.                                                                     #
# --------------------------------------------------------------------------- #
def _image(seed: int = 34) -> torch.Tensor:
    """A deterministic ``[N, C, 1, W]`` fp32 input.

    fp32 rather than bf16 because the declared ``atol`` is ``1e-5`` and bf16's
    ~3 decimal digits could not express a difference at that scale at all --
    the comparison would measure the storage format instead of the kernel. This
    is the `inc-glm53f-025` conditioning lesson applied: a tolerance must be
    measurable in the dtype it is measured in.

    Centred on zero so the convolution's sum is not dominated by a DC term,
    which would let a wrong tap ordering hide inside a large common offset.
    """
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand(
            (BATCH, CHANNELS, 1, WIDTH), generator=generator, dtype=torch.float32
        )
        * 2.0
        - 1.0
    )


def _filter(seed: int = 43) -> torch.Tensor:
    """A deterministic ``[C, 1, 1, S]`` fp32 tap set, one filter per channel."""
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand(
            (CHANNELS, 1, 1, TAPS), generator=generator, dtype=torch.float32
        )
        * 2.0
        - 1.0
    )


def _distinct_taps() -> torch.Tensor:
    """Taps that are distinct per position AND per channel, for the impulse case.

    Distinctness is what makes the impulse case an ORDER proof: with equal taps
    a reversed window and a forward window would agree, and the case would pass
    while measuring nothing. So all ``C * S`` values are consecutive integers,
    ``1 .. C*S`` -- every value appears exactly once, and any permutation of any
    two of them shows up as a changed reading.

    Consecutive integers rather than powers of two scaled per channel: that
    construction collides (``2 * 2 == 1 * 4``), and
    :func:`test_impulse_expectation_is_not_order_blind` measured it as 20
    distinct values out of 32 before this fixture was corrected.
    """
    values = torch.arange(1, CHANNELS * TAPS + 1, dtype=torch.float32)
    return values.reshape(CHANNELS, 1, 1, TAPS)


def _impulse_image() -> torch.Tensor:
    """``[N, C, 1, W]`` zeros with a single ``1.0`` per channel at :data:`IMPULSE_AT`."""
    img = torch.zeros((BATCH, CHANNELS, 1, WIDTH), dtype=torch.float32)
    img[0, :, 0, IMPULSE_AT] = 1.0
    return img


def _impulse_closed_form(filt: torch.Tensor) -> torch.Tensor:
    """The output a unit impulse must produce, computed from the taps ALONE.

    Cross-correlation places ``filter[p - q]`` at output position ``q`` for a
    unit impulse at input position ``p``. So output position ``0`` carries the
    LAST tap and position ``p`` carries the FIRST: the tap vector reversed, then
    zeros for every output position beyond ``p``.

    Deliberately built without calling the kernel, the reference, or any conv
    operator -- it is the independent expectation the argument-order claim rests
    on.
    """
    expected = torch.zeros((BATCH, CHANNELS, 1, Q), dtype=torch.float32)
    reversed_taps = torch.flip(filt.reshape(CHANNELS, TAPS), dims=(1,))
    span = min(TAPS, Q)
    expected[0, :, 0, :span] = reversed_taps[:, TAPS - span :]
    return expected


def _worst_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.to(torch.float32) - expected.to(torch.float32)).abs().max().item()


# --------------------------------------------------------------------------- #
# DECLARED CASE 1 -- the reference case.                                        #
# --------------------------------------------------------------------------- #
def test_reference_case_matches_the_substrates_own_torch_reference() -> None:
    """1/1 tiny case at ``assert_close(rtol=1e-2, atol=1e-5)``, worst error reported.

    The comparison target is ``nkilib``'s own reference for this kernel, reached
    through the module's :func:`depthwise_conv1d_torch_reference`, so neither
    side of this comparison is numerics this increment authored.

    ``rtol`` and ``atol`` are passed EXPLICITLY rather than left to
    ``assert_close``'s dtype default, on the registration's PIT-13 rule
    (``design/acceptance-preregistration.md`` §3): a defaulted tolerance is a
    tolerance nobody declared.
    """
    img, filt = _image(), _filter()
    expected = depthwise_conv1d_torch_reference(img, filt)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        actual = depthwise_conv1d(img, filt)
    reading = _assert_route(sim, 1, "reference-case")

    assert tuple(actual.shape) == (BATCH, CHANNELS, 1, Q), (
        f"shape {tuple(actual.shape)} is not the declared [N, C, 1, Q] "
        f"{(BATCH, CHANNELS, 1, Q)}"
    )
    assert actual.dtype == img.dtype, (
        f"output dtype {actual.dtype} is not the input dtype {img.dtype}"
    )
    worst = _worst_abs(actual, expected)
    print(
        f"[reference-case] geometry=N{BATCH}xC{CHANNELS}xW{WIDTH}xS{TAPS}->Q{Q} "
        f"worst_abs_error={worst:.3e} rtol={RTOL} atol={ATOL} | {reading}"
    )
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_reference_case_fixture_is_not_vacuous() -> None:
    """The declared case has real work in it, measured rather than assumed.

    Two ways this comparison could pass while measuring nothing, both closed
    here: an all-zero output would match an all-zero reference, and an output
    equal to a slice of the input would mean no convolution happened. Both are
    shown false on the same fixture the reference case uses.
    """
    img, filt = _image(), _filter()
    expected = depthwise_conv1d_torch_reference(img, filt)
    largest = expected.abs().max().item()
    print(f"[fixture] reference_abs_max={largest:.4f} nonzero={bool(largest > 0)}")
    if largest == 0.0:
        raise VacuousControlError(
            "the reference output is identically zero, so the declared "
            "comparison could pass over an empty result"
        )
    identity_slice = img[..., :Q]
    if torch.allclose(expected, identity_slice, rtol=RTOL, atol=ATOL):
        raise VacuousControlError(
            "the reference output equals a slice of the input, so a kernel that "
            "merely copied its input would pass the declared comparison"
        )


# --------------------------------------------------------------------------- #
# DECLARED CASE 2 -- the unit-impulse case.                                     #
# --------------------------------------------------------------------------- #
def test_unit_impulse_case_recovers_the_taps_and_proves_argument_order() -> None:
    """Unit impulse at ``atol=1e-5``: the taps come back, reversed, exactly.

    Compared against :func:`_impulse_closed_form`, which is built from the taps
    alone. That is what makes this an argument-order reading: a wrap that
    exchanged ``img_ref`` and ``filter_ref`` could not reproduce a tap vector it
    never received in that position.

    ``rtol=0`` is deliberate. The plan declares this case at ``atol 1e-5``, and
    a relative term would scale the bar with the taps' own magnitude, which is
    exactly the leniency an exact-recovery claim must not have.
    """
    filt = _distinct_taps()
    img = _impulse_image()
    expected = _impulse_closed_form(filt)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        actual = depthwise_conv1d(img, filt)
    reading = _assert_route(sim, 1, "unit-impulse-case")

    worst = _worst_abs(actual, expected)
    print(
        f"[unit-impulse-case] impulse_at={IMPULSE_AT} "
        f"taps_channel0={filt.reshape(CHANNELS, TAPS)[0].tolist()} "
        f"recovered_channel0={actual[0, 0, 0, :TAPS].tolist()} "
        f"expected_channel0={expected[0, 0, 0, :TAPS].tolist()} "
        f"worst_abs_error={worst:.3e} atol={IMPULSE_ATOL} | {reading}"
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=IMPULSE_ATOL)

    # Second, independent reading: the substrate's reference agrees too.
    reference = depthwise_conv1d_torch_reference(img, filt)
    reference_worst = _worst_abs(reference, expected)
    print(
        f"[unit-impulse-case] substrate_reference_vs_closed_form="
        f"{reference_worst:.3e}"
    )
    torch.testing.assert_close(reference, expected, rtol=0.0, atol=IMPULSE_ATOL)


def test_impulse_expectation_is_not_order_blind() -> None:
    """The reversed taps DIFFER from the forward taps, so "reversed" has content.

    Without this reading, the impulse case could pass on a palindromic tap set
    and prove nothing about order at all.
    """
    filt = _distinct_taps()
    flat = filt.reshape(CHANNELS, TAPS)
    forward = flat[0].tolist()
    reversed_taps = torch.flip(flat, dims=(1,))[0].tolist()
    unique = len(set(flat.flatten().tolist()))
    print(
        f"[impulse-arming] forward={forward} reversed={reversed_taps} "
        f"distinct_tap_values={unique}/{CHANNELS * TAPS}"
    )
    if forward == reversed_taps:
        raise VacuousControlError(
            "the tap vector is palindromic, so the impulse case cannot "
            "distinguish a reversed window from a forward one"
        )
    if unique != CHANNELS * TAPS:
        raise VacuousControlError(
            f"only {unique} of {CHANNELS * TAPS} tap values are distinct, so a "
            f"permutation could pass unnoticed"
        )


# --------------------------------------------------------------------------- #
# Route controls -- every zero above is armed here.                             #
# --------------------------------------------------------------------------- #
def test_route_control_fallback_counter_discriminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the simulator disabled the seam takes the torch path, and it is COUNTED.

    This is the arm that makes ``torch_fallback == 0`` above meaningful: the
    counter is shown reading ``1`` and ``nki_dispatch`` reading ``0``, through
    the real gate rather than a mock. It is also the measured form of the plan
    block's claim that a pure-torch implementation yields ``0`` dispatches.
    """
    img, filt = _image(), _filter()
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0, so this control is unarmed"
    )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = depthwise_conv1d(img, filt)
    nki_dispatch, torch_fallback = dispatch_counters()
    print(
        f"[route-control] nki_dispatch={nki_dispatch} "
        f"torch_fallback={torch_fallback} simulate_kernel_calls={sim.calls}"
    )
    assert nki_dispatch == 0, f"expected 0 NKI dispatches, got {nki_dispatch}"
    assert torch_fallback == 1, f"expected 1 torch fallback, got {torch_fallback}"
    assert sim.calls == 0, f"the simulator ran {sim.calls} times with it disabled"
    assert tuple(out.shape) == (BATCH, CHANNELS, 1, Q)


def test_route_control_simulator_is_load_bearing() -> None:
    """The NKI chain RAISES without the simulator rather than computing torch.

    Recorded because it is what forecloses the F1 false green BELOW this
    repository's seam: if the HOP silently degraded to a torch path, a green
    numeric comparison could not be attributed to a kernel at all, and no
    counter of this module's could detect it.
    """
    img, filt = _image(), _filter()
    saved = os.environ.get("NKI_SIMULATOR")
    os.environ["NKI_SIMULATOR"] = "0"
    try:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
        from nkilib.experimental.conv.depthwise_conv1d import (
            depthwise_conv1d_implicit_gemm,
        )

        with pytest.raises(RuntimeError) as excinfo:
            wrap_nki(depthwise_conv1d_implicit_gemm)(
                img_ref=img,
                filter_ref=filt,
                padding=NO_PADDING,
                stride=UNIT_STRIDE,
                feature_group_count=CHANNELS,
            )
    finally:
        if saved is None:
            os.environ.pop("NKI_SIMULATOR", None)
        else:
            os.environ["NKI_SIMULATOR"] = saved

    message = str(excinfo.value)
    print(f"[route-control] simulator_off_raise={message[:160]!r}")
    assert "simulator" in message.lower(), message


def test_feature_group_count_derivation_is_load_bearing() -> None:
    """The substrate kernel REFUSES its own default, so the seam's derivation counts.

    ``depthwise_conv1d_implicit_gemm`` defaults ``feature_group_count`` to ``1``
    and asserts it equals ``C``. The seam derives it from the image instead. This
    control drives the raw kernel at its default and records the refusal
    verbatim, so "the seam supplies that argument" is a measured property rather
    than a claim about code that happens to be present.
    """
    img, filt = _image(), _filter()
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.experimental.conv.depthwise_conv1d import (
        depthwise_conv1d_implicit_gemm,
    )

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - vendor raises AssertionError
        wrap_nki(depthwise_conv1d_implicit_gemm)(
            img_ref=img,
            filter_ref=filt,
            padding=NO_PADDING,
            stride=UNIT_STRIDE,
        )
    message = str(excinfo.value)
    print(
        f"[derivation-control] raised={type(excinfo.value).__name__} "
        f"message={message[:200]!r}"
    )
    assert "feature_group_count" in message, message

    # And the seam, which derives it, succeeds on the same inputs.
    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = depthwise_conv1d(img, filt)
    _assert_route(sim, 1, "derivation-control-seam")
    assert tuple(out.shape) == (BATCH, CHANNELS, 1, Q)


# --------------------------------------------------------------------------- #
# The WRAP is checkable: the seam dispatches to the SUBSTRATE's member.          #
# --------------------------------------------------------------------------- #
def test_seam_wraps_the_substrate_member_and_authors_no_kernel() -> None:
    """``kernel_identity()`` names ``nkilib``'s kernel, not anything authored here.

    This is what makes "WRAP, not SCRATCH" a reading rather than a claim: a
    substitution -- vendoring a copy, or quietly authoring a replacement --
    changes this reading instead of passing silently.
    """
    module, qualname = kernel_identity()
    print(f"[substrate] kernel_identity=({module}, {qualname})")
    assert module == SUBSTRATE_MODULE, (
        f"the seam dispatches to {module}, not to the substrate module "
        f"{SUBSTRATE_MODULE}"
    )
    assert qualname == SUBSTRATE_QUALNAME, (
        f"the seam dispatches to {qualname}, not to {SUBSTRATE_QUALNAME}"
    )
    assert not module.startswith("vllm_neuron"), (
        "the wrapped kernel lives inside this repository, which would make this "
        "increment SCRATCH rather than the declared WRAP"
    )


def test_reference_is_the_substrates_own_and_not_authored_here() -> None:
    """The comparison target is ``nkilib``'s reference, driven through the module.

    Measured by driving the vendor function directly with the same inputs and
    requiring bit-for-bit equality with what the module returns: if this module
    had authored its own torch convolution, the two would differ in the last
    bits at best.
    """
    from nkilib.experimental.conv.depthwise_conv1d_torch import (
        depthwise_conv1d_implicit_gemm_torch_ref,
    )

    img, filt = _image(), _filter()
    direct = depthwise_conv1d_implicit_gemm_torch_ref(
        img, filt, padding=NO_PADDING, stride=UNIT_STRIDE, feature_group_count=CHANNELS
    )["output"]
    through_module = depthwise_conv1d_torch_reference(img, filt)
    difference = _worst_abs(through_module, direct)
    print(f"[substrate] module_reference_vs_vendor_direct={difference:.3e}")
    assert torch.equal(through_module, direct), (
        f"the module's reference is not bit-identical to the vendor's own "
        f"(worst difference {difference:.3e}), so it is not a pure delegation"
    )


# --------------------------------------------------------------------------- #
# The counters are MODULE-LEVEL state, on -026's and -028's landed placement.    #
# --------------------------------------------------------------------------- #
def test_dispatch_counters_are_module_level_state_reachable_from_elsewhere() -> None:
    """Another module can zero and read these counters, and they accumulate.

    Measured rather than asserted by inspection: the module is re-acquired
    through ``importlib`` -- the same mechanism another test module's import uses
    -- one reference RESETS, the seam is driven, and the OTHER reference READS.
    """
    foreign = importlib.import_module(_MODULE)
    assert foreign is sys.modules[_MODULE]
    assert foreign.dispatch_counters is dispatch_counters
    assert foreign.reset_dispatch_counters is reset_dispatch_counters

    img, filt = _image(), _filter()
    foreign.reset_dispatch_counters()
    assert dispatch_counters() == (0, 0), (
        "a reset through the foreign reference did not zero the counters this "
        "test reads, so the state is not shared"
    )
    with _SimulatorCounter() as sim:
        foreign.depthwise_conv1d(img, filt)
    after_one = dispatch_counters()
    with _SimulatorCounter() as sim2:
        foreign.depthwise_conv1d(img, filt)
    after_two = foreign.dispatch_counters()
    print(
        f"[cross-module] after_reset=(0, 0) after_one_call={after_one} "
        f"after_two_calls={after_two} sim_calls={sim.calls + sim2.calls}"
    )
    assert after_one == (1, 0), f"expected (1, 0) after one dispatch, got {after_one}"
    assert after_two == (2, 0), (
        f"expected (2, 0) after two dispatches, got {after_two}; the counter "
        f"does not accumulate across calls"
    )
    foreign.reset_dispatch_counters()
    assert dispatch_counters() == (0, 0)


# --------------------------------------------------------------------------- #
# Geometry refusals, and the honest reading behind the channel-count one.       #
# --------------------------------------------------------------------------- #
def test_odd_channel_count_is_refused_by_this_module() -> None:
    """An odd ``C`` raises here, and the raise is THIS MODULE's, measured.

    The substrate's Notes require ``C`` divisible by ``NUM_SHARDS``, and its body
    indexes each shard from ``shard_id * (C // num_programs())``. The NKI
    SIMULATOR does not enforce it -- this test measures that too, by driving the
    raw kernel at an odd ``C`` and recording that it returns a correct result.

    So the refusal is this module's deliberate conservatism on the substrate's
    documented device requirement, and this reading says so out loud rather than
    letting a reader infer that the kernel failed.
    """
    odd_channels = CHANNELS + 1
    generator = torch.Generator().manual_seed(55)
    img = torch.rand(
        (BATCH, odd_channels, 1, WIDTH), generator=generator, dtype=torch.float32
    )
    filt = torch.rand(
        (odd_channels, 1, 1, TAPS), generator=generator, dtype=torch.float32
    )

    reset_dispatch_counters()
    with pytest.raises(KdaDepthwiseConv1dError) as excinfo:
        depthwise_conv1d(img, filt)
    assert "LNC_SHARDS" in str(excinfo.value), str(excinfo.value)
    assert dispatch_counters() == (0, 0), (
        "a refused geometry incremented a counter; a refusal must not dispatch "
        "and must not fall back"
    )

    # The honest half: the simulator itself accepts the same odd C.
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.experimental.conv.depthwise_conv1d import (
        depthwise_conv1d_implicit_gemm,
    )

    raw = wrap_nki(depthwise_conv1d_implicit_gemm)(
        img_ref=img,
        filter_ref=filt,
        padding=NO_PADDING,
        stride=UNIT_STRIDE,
        feature_group_count=odd_channels,
    )
    reference = depthwise_conv1d_implicit_gemm_torch_ref_odd(img, filt, odd_channels)
    worst = _worst_abs(torch.as_tensor(raw), reference)
    print(
        f"[refusal] odd_C={odd_channels} module_refused=True "
        f"simulator_accepted_the_same_call=True raw_worst_abs={worst:.3e} "
        f"(so the refusal is this module's, on the substrate's device Notes)"
    )


def depthwise_conv1d_implicit_gemm_torch_ref_odd(
    img: torch.Tensor, filt: torch.Tensor, channels: int
) -> torch.Tensor:
    """The substrate's reference at an odd ``C``, which this module refuses.

    A helper rather than a module function on purpose: the module's public
    reference derives ``feature_group_count`` behind an admission check that
    rejects this geometry, and relaxing that check to serve a control would
    weaken the refusal the control exists to measure.
    """
    from nkilib.experimental.conv.depthwise_conv1d_torch import (
        depthwise_conv1d_implicit_gemm_torch_ref,
    )

    return depthwise_conv1d_implicit_gemm_torch_ref(
        img,
        filt,
        padding=NO_PADDING,
        stride=UNIT_STRIDE,
        feature_group_count=channels,
    )["output"]


@pytest.mark.parametrize(
    ("label", "kwargs", "needle"),
    [
        ("height_padding", {"padding": ((1, 1), (0, 0))}, "height padding"),
        ("stride_h", {"stride": (2, 1)}, "stride_h"),
        ("rhs_dilation", {"rhs_dilation": (1, 2)}, "rhs_dilation"),
        ("lhs_dilation", {"lhs_dilation": (2, 1)}, "lhs_dilation"),
        ("batch_group_count", {"batch_group_count": 2}, "batch_group_count"),
    ],
)
def test_unsupported_options_are_refused(
    label: str, kwargs: dict, needle: str
) -> None:
    """Every option the substrate kernel asserts on is refused before dispatch.

    Refused, not coerced and not routed to torch: a silent fallback for
    kernel-class work is what P13 forbids. Each case names the option in its
    message, which is what a caller can act on.
    """
    img, filt = _image(), _filter()
    reset_dispatch_counters()
    with pytest.raises(KdaDepthwiseConv1dError) as excinfo:
        depthwise_conv1d(img, filt, **kwargs)
    message = str(excinfo.value)
    print(f"[refusal] {label}: {message[:150]}")
    assert needle in message, message
    assert dispatch_counters() == (0, 0), (
        f"{label}: a refused option incremented a counter"
    )


def test_shape_refusals_name_the_offending_extent() -> None:
    """A wrong rank or a mismatched channel extent is named, not coerced."""
    img, filt = _image(), _filter()

    with pytest.raises(KdaDepthwiseConv1dError) as rank_exc:
        depthwise_conv1d(img.reshape(BATCH, CHANNELS, WIDTH), filt)
    assert "4-D" in str(rank_exc.value), str(rank_exc.value)

    mismatched = _filter()[: CHANNELS - LNC_SHARDS]
    with pytest.raises(KdaDepthwiseConv1dError) as channel_exc:
        depthwise_conv1d(img, mismatched)
    assert "channel extent" in str(channel_exc.value), str(channel_exc.value)

    with pytest.raises(KdaDepthwiseConv1dError) as width_exc:
        depthwise_conv1d(img[..., : TAPS - 1], filt)
    assert "smaller than the kernel size" in str(width_exc.value), str(
        width_exc.value
    )
    print("[refusal] rank, channel-extent and padded-width refusals all named")


def test_dtype_mismatch_is_refused() -> None:
    """Mixed dtypes are refused, because the kernel contracts the two together."""
    img, filt = _image(), _filter()
    with pytest.raises(KdaDepthwiseConv1dError) as excinfo:
        depthwise_conv1d(img, filt.to(torch.bfloat16))
    assert "dtype" in str(excinfo.value), str(excinfo.value)
    print(f"[refusal] dtype: {str(excinfo.value)[:140]}")


def test_gate_reports_availability_separately_from_admissibility() -> None:
    """``can_run_depthwise_conv1d`` answers availability; it RAISES on inadmissible.

    The two conditions are deliberately not merged, so a caller can tell "no
    device or simulator" from "this kernel does not accept these extents".
    """
    img, filt = _image(), _filter()
    assert can_run_depthwise_conv1d(img, filt) is True

    with pytest.raises(KdaDepthwiseConv1dError):
        can_run_depthwise_conv1d(img, filt, batch_group_count=3)
    print("[gate] available=True on the declared case; inadmissible options raise")


def test_output_width_is_the_substrates_formula() -> None:
    """``Q`` is computed once, and the declared case's ``Q`` matches the shape."""
    assert Q == (WIDTH - TAPS) // 1 + 1
    assert output_width(WIDTH, TAPS, ((0, 0), (2, 2))) == (WIDTH + 4 - TAPS) + 1
    assert output_width(WIDTH, TAPS, NO_PADDING, (1, 2)) == (WIDTH - TAPS) // 2 + 1
    with pytest.raises(KdaDepthwiseConv1dError):
        output_width(TAPS - 1, TAPS)
    print(f"[geometry] Q={Q} from W={WIDTH} S={TAPS} at unit stride, no padding")
