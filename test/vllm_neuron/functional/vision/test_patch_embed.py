# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for `inc-glm53f-057` -- the WP10 vision patch embed WRAP.

Acceptance command (plan block ``#### inc-glm53f-057``, Tier N harness "as
`-025`")::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/vision/test_patch_embed.py \
      -q -s --timeout 60 -p no:cacheprovider

The two DECLARED cases
----------------------
1. the **reference case** -- one tiny case, simulated NKI output against
   ``nkilib``'s own torch reference for this kernel at
   ``assert_close(rtol=1e-2, atol=1e-5)``, worst error reported;
2. the **single-patch impulse case** -- asserted at ``atol=1e-5``, recovering the
   kernel weights exactly and so proving the ``K_d = 1`` degeneration is wired
   right.

No tolerance number is invented, widened or narrowed anywhere in this file.
:data:`RTOL`, :data:`ATOL` and :data:`IMPULSE_ATOL` are the plan block's, and no
other tolerance pair is authored here (P9).

Why the impulse case proves the DEGENERATION and not merely arithmetic
---------------------------------------------------------------------
The impulse case runs on an input that is exactly ONE patch: ``H = W = P``, so
with the seam's derived stride there is exactly one output position. A single
unit impulse at input channel ``c0`` and pixel ``(kh0, kw0)`` therefore makes the
whole output equal one slice of the filter, ``filters[0, kh0, kw0, c0, :]`` --
a closed form computed IN THIS FILE from the filter alone, never from the module
under test. Three things would break it: a depth axis that contracted anything
(``K_d`` not degenerate), a filter and an image passed to each other's argument,
or a filter index order read differently by the wrap than by this file.
:func:`test_impulse_expectation_is_not_index_order_blind` measures that the claim
has content -- the expected slice is shown to DIFFER from the transposed-index
slice and from another channel's slice, so "this slice" is falsifiable rather
than a symmetric coincidence. The vendor reference's agreement on the same
inputs is recorded beside it as a second, independent reading.

Why the route predicate is an acceptance criterion and not a diagnostic (F1)
---------------------------------------------------------------------------
Case 1 compares simulated NKI output against a torch reference. If the seam
silently took its torch path, *both* sides would be torch and the comparison
would pass green while measuring nothing about a kernel. So each declared case
reads three route instruments and reports each as a number:

1. the seam's own module-level dispatch counter (form R-1, the block's declared
   form) -- ``nki_dispatch == 1``, ``torch_fallback == 0``;
2. ``can_run_kernel()`` -- ``True``;
3. real ``nki.simulator.simulate_kernel`` invocations -- ``1`` per case.
   Instrument 3 counts the VENDOR entry point, so a bug in instrument 1 cannot
   fake it.

The plan block states the discrimination directly: "A pure-torch implementation
yields ``0`` and therefore cannot pass."

``kernel_identity()`` is read in this file, but **never as a route reading**
(D13.1): it certifies what the module imported, not what ran. It is used here
only as a substitution detector.

Every zero is armed. :func:`test_route_control_fallback_counter_discriminates`
shows instrument 1 reading ``(0, 1)`` and instrument 3 reading ``0`` on the
fallback path; :func:`test_route_control_simulator_is_load_bearing` shows the
chain RAISING rather than quietly computing torch when the simulator is off; and
:func:`test_stride_derivation_is_load_bearing` shows the substrate's DEFAULT
stride returning a different, overlapping-window extent on the same inputs, so
the seam's derivation of that argument is measured rather than assumed. (This
kernel carries no ``assert`` statements, so unlike `inc-glm53f-034`'s conv1d
there is no vendor refusal to record here -- the wrong argument is silently
wrong, which is why the control measures the extent instead of an exception.)

Why fp32 and not the tower's bf16
---------------------------------
The declared ``atol`` is ``1e-5``, which bf16's ~3 decimal digits cannot express
at all, so a bf16 comparison would measure the storage format instead of the
kernel (the `inc-glm53f-025` conditioning lesson, applied as `inc-glm53f-034`
landed it). The block's rev-231 rider records that the real tower's inputs are
bf16; :func:`test_bf16_tower_dtype_is_admitted_by_the_seam` ties that rider to
this code as a STRUCTURAL reading -- the gate admits bf16 and the seam
dispatches -- and authors no tolerance pair for it.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

import pytest
import torch

import nki
import nki.simulator

from vllm_neuron.accuracy.testing import assert_close
from vllm_neuron.functional.vision.patch_embed import (
    BATCH_RANGE,
    C_IN_RANGE,
    C_OUT_RANGE,
    LNC_SHARD,
    NO_PADDING,
    SINGLETON_D,
    UNIT_DILATION,
    VisionPatchEmbedError,
    can_run_patch_embed,
    dispatch_counters,
    kernel_identity,
    output_extents,
    patch_embed,
    patch_embed_torch_reference,
    patch_stride,
    reset_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# --------------------------------------------------------------------------- #
# The one tiny geometry both declared cases run on. C_in = 3 is the block's own #
# ground: the kernel documents "C_in: 3-1280", which the block cites as        #
# covering a 3-channel image tower.                                            #
# --------------------------------------------------------------------------- #
BATCH = 1
C_IN = 3
C_OUT = 8
PATCH = 4
#: Two patches per axis on the reference case, so the output extent is not 1 and
#: a stride error cannot hide in a degenerate grid.
IMAGE_SIDE = PATCH * 2
#: Exactly one patch on the impulse case, so the whole output IS one filter
#: slice.
IMPULSE_SIDE = PATCH

#: The plan block's registered pair for the reference case (P9). Not widened,
#: not narrowed, and no second pair is authored in this file.
RTOL = 1e-2
ATOL = 1e-5
#: The plan block's impulse-case tolerance.
IMPULSE_ATOL = 1e-5

#: The impulse position and channel. ``kh0 != kw0`` on purpose: it is what makes
#: the transposed-index control below able to fail.
IMPULSE_CHANNEL = 1
IMPULSE_H = 0
IMPULSE_W = 2

_MODULE = "vllm_neuron.functional.vision.patch_embed"
SUBSTRATE_MODULE = "nkilib.experimental.conv.conv3d"
SUBSTRATE_QUALNAME = "conv3d"
SUBSTRATE_REFERENCE_MODULE = "nkilib.experimental.conv.conv3d_torch"


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
# Fixtures. fp32: see the module docstring on why not bf16.                     #
# --------------------------------------------------------------------------- #
def _image(seed: int = 57) -> torch.Tensor:
    """A deterministic ``[B, C_in, 1, H, W]`` fp32 input."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        BATCH,
        C_IN,
        SINGLETON_D,
        IMAGE_SIDE,
        IMAGE_SIDE,
        generator=generator,
        dtype=torch.float32,
    )


def _filters(seed: int = 75) -> torch.Tensor:
    """A deterministic ``[1, P, P, C_in, C_out]`` fp32 patch projection."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        SINGLETON_D,
        PATCH,
        PATCH,
        C_IN,
        C_OUT,
        generator=generator,
        dtype=torch.float32,
    )


def _distinct_filters() -> torch.Tensor:
    """Filters whose every element differs, so an index-order error cannot hide.

    ``arange`` rather than random: two random slices could coincide by accident
    and the control below would report a pass it did not earn.
    """
    total = SINGLETON_D * PATCH * PATCH * C_IN * C_OUT
    return (
        torch.arange(1, total + 1, dtype=torch.float32)
        .reshape(SINGLETON_D, PATCH, PATCH, C_IN, C_OUT)
        .contiguous()
    )


def _impulse_image() -> torch.Tensor:
    """One patch, all zeros but a single unit at ``(c0, kh0, kw0)``."""
    x = torch.zeros(
        BATCH, C_IN, SINGLETON_D, IMPULSE_SIDE, IMPULSE_SIDE, dtype=torch.float32
    )
    x[0, IMPULSE_CHANNEL, 0, IMPULSE_H, IMPULSE_W] = 1.0
    return x


def _impulse_closed_form(filters: torch.Tensor) -> torch.Tensor:
    """The output a single unit impulse must produce, from the FILTER alone.

    With one patch of input and the seam's derived stride there is exactly one
    output position, and a unit impulse at ``(c0, kh0, kw0)`` selects one filter
    slice::

        out[0, :, 0, 0, 0] == filters[0, kh0, kw0, c0, :]

    Computed here from ``filters``, never from the module under test.
    """
    slice_ = filters[0, IMPULSE_H, IMPULSE_W, IMPULSE_CHANNEL, :]
    return slice_.reshape(1, C_OUT, 1, 1, 1).contiguous()


def _worst_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max())


# --------------------------------------------------------------------------- #
# C00 -- the module under test is the candidate, on the §88(iii) repaired form. #
# --------------------------------------------------------------------------- #
def test_c00_the_module_under_test_is_the_scratch_candidate() -> None:
    """Where ``vllm_neuron`` was imported from, asserted in both runner shapes.

    Two arms on purpose (DECISIONS §88 ruling (iii)): when the campaign
    harness's ``GLM53F_CANDIDATE_ROOT`` is SET, the stricter declared-root arm
    stands (§78.1 -- an in-venv editable install points at a checkout that is not
    the candidate, so a run without this assertion measured an unidentified
    tree). When it is UNSET -- a maintainer, CI, or a reviewer's plain ``pytest``
    -- the root is derived from this file's own tree instead, so this item is
    never red by construction outside the harness.
    """
    import vllm_neuron

    imported_from = Path(vllm_neuron.__file__).resolve()
    declared = os.environ.get("GLM53F_CANDIDATE_ROOT")
    print(f"[c00] VLLM_NEURON_IMPORTED_FROM={imported_from}")
    print(f"[c00] GLM53F_CANDIDATE_ROOT={declared!r}")

    if declared:
        root = Path(declared).resolve()
        arm = "declared-root"
    else:
        # test/vllm_neuron/functional/vision/<this file> -> repository root
        root = Path(__file__).resolve().parents[4]
        arm = "derived-from-this-file"
    print(f"[c00] arm={arm} root={root}")
    assert (root / "vllm_neuron").is_dir(), (
        f"{arm}: {root} does not contain a vllm_neuron package, so this arm "
        f"could not discriminate anything"
    )
    assert imported_from.is_relative_to(root), (
        f"{arm}: vllm_neuron imported from {imported_from}, which is not under "
        f"{root}. IS_UNDER_THE_SCRATCH_CANDIDATE=0"
    )
    print("[c00] IS_UNDER_THE_SCRATCH_CANDIDATE=1")


# --------------------------------------------------------------------------- #
# DECLARED CASE 1 -- the reference case.                                        #
# --------------------------------------------------------------------------- #
def test_reference_case_matches_the_substrates_own_torch_reference() -> None:
    """Simulated NKI output vs ``nkilib``'s conv3d torch reference, 1/1 case.

    The plan block's expected result: ``assert_close(rtol=1e-2, atol=1e-5)`` on
    one tiny case, with the route predicate reading 1 dispatch.
    """
    x, filters = _image(), _filters()
    expected_shape = (BATCH, C_OUT) + output_extents(
        (SINGLETON_D, IMAGE_SIDE, IMAGE_SIDE),
        (SINGLETON_D, PATCH, PATCH),
        NO_PADDING,
        patch_stride(PATCH),
        UNIT_DILATION,
    )

    reference = patch_embed_torch_reference(x, filters, PATCH)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        actual = patch_embed(x, filters, PATCH)
    _assert_route(sim, 1, "case1-reference")

    assert tuple(actual.shape) == expected_shape, (
        f"kernel returned {tuple(actual.shape)}, the substrate's own formula "
        f"says {expected_shape}"
    )
    assert tuple(reference.shape) == expected_shape

    result = assert_close(
        actual, reference, rtol=RTOL, atol=ATOL, name="patch_embed"
    )
    print(
        f"[case1-reference] cases=1/1 rtol={RTOL} atol={ATOL} "
        f"max_abs_error={result.max_abs_error:.3e} "
        f"max_rel_error={result.max_rel_error:.3e} "
        f"linf_rel={result.linf_rel:.3e} allclose={result.allclose} "
        f"worst_abs_recomputed={_worst_abs(actual, reference):.3e}"
    )
    assert result.allclose is True


def test_reference_case_fixture_is_not_vacuous() -> None:
    """The reference case's inputs and outputs could have made it fail.

    An all-zero input, an all-zero filter or a constant output would let case 1
    pass while measuring nothing. Each is measured here (§79.1: a set-based
    reading also asserts the set is non-empty).
    """
    x, filters = _image(), _filters()
    assert x.numel() > 0 and filters.numel() > 0, "an empty fixture measures nothing"
    if float(x.abs().max()) == 0.0:
        raise VacuousControlError("the reference image is all zeros")
    if float(filters.abs().max()) == 0.0:
        raise VacuousControlError("the reference filters are all zeros")

    reference = patch_embed_torch_reference(x, filters, PATCH)
    assert reference.numel() > 0, "the reference produced an empty tensor"
    spread = float(reference.max() - reference.min())
    print(
        f"[case1-vacuity] elements={reference.numel()} spread={spread:.6f} "
        f"input_absmax={float(x.abs().max()):.6f} "
        f"filter_absmax={float(filters.abs().max()):.6f}"
    )
    if spread <= 0.0:
        raise VacuousControlError(
            f"the reference output is constant (spread {spread}), so the "
            f"tolerance could not discriminate"
        )


# --------------------------------------------------------------------------- #
# DECLARED CASE 2 -- the single-patch impulse case.                             #
# --------------------------------------------------------------------------- #
def test_single_patch_impulse_recovers_the_kernel_weights() -> None:
    """One patch, one unit impulse: the output IS a filter slice, at atol 1e-5.

    This is the plan block's second declared case and its stated purpose --
    "recovering the kernel weights exactly, which proves the ``K_d = 1``
    degeneration is wired right". The expectation is computed in this file from
    the filter alone; the vendor reference's agreement is recorded beside it as
    a second, independent reading.
    """
    x = _impulse_image()
    filters = _distinct_filters()
    expected = _impulse_closed_form(filters)

    reference = patch_embed_torch_reference(x, filters, PATCH)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        actual = patch_embed(x, filters, PATCH)
    _assert_route(sim, 1, "case2-impulse")

    assert tuple(actual.shape) == (BATCH, C_OUT, 1, 1, 1), (
        f"one patch of input must give one output position, got "
        f"{tuple(actual.shape)}"
    )

    result = assert_close(
        actual, expected, rtol=0.0, atol=IMPULSE_ATOL, name="impulse_vs_closed_form"
    )
    ref_worst = _worst_abs(reference, expected)
    print(
        f"[case2-impulse] atol={IMPULSE_ATOL} "
        f"kernel_vs_closed_form_max_abs={_worst_abs(actual, expected):.3e} "
        f"allclose={result.allclose} "
        f"reference_vs_closed_form_max_abs={ref_worst:.3e} "
        f"recovered={actual.flatten().tolist()} "
        f"expected={expected.flatten().tolist()}"
    )
    assert result.allclose is True
    assert ref_worst <= IMPULSE_ATOL, (
        f"the VENDOR reference disagrees with this file's closed form by "
        f"{ref_worst:.3e} > {IMPULSE_ATOL}; the closed form, not the kernel, is "
        f"then the thing to fix"
    )


def test_impulse_expectation_is_not_index_order_blind() -> None:
    """"That slice" is falsifiable: the near-miss slices differ from it.

    If the transposed-index slice or another channel's slice equalled the
    expected one, case 2 would pass for a wrap that read the filter's axes in
    the wrong order.
    """
    filters = _distinct_filters()
    expected = _impulse_closed_form(filters).flatten()
    transposed = filters[0, IMPULSE_W, IMPULSE_H, IMPULSE_CHANNEL, :].flatten()
    other_channel = filters[
        0, IMPULSE_H, IMPULSE_W, (IMPULSE_CHANNEL + 1) % C_IN, :
    ].flatten()

    d_transposed = float((expected - transposed).abs().max())
    d_channel = float((expected - other_channel).abs().max())
    print(
        f"[case2-order-control] expected_vs_transposed_max_abs={d_transposed:.6f} "
        f"expected_vs_other_channel_max_abs={d_channel:.6f} "
        f"atol={IMPULSE_ATOL}"
    )
    assert expected.numel() == C_OUT and C_OUT > 1, (
        "a one-element expectation could not discriminate an order error"
    )
    if d_transposed <= IMPULSE_ATOL:
        raise VacuousControlError(
            f"the transposed-index slice is within atol of the expected one "
            f"({d_transposed:.3e}), so case 2 is order-blind as written"
        )
    if d_channel <= IMPULSE_ATOL:
        raise VacuousControlError(
            f"another channel's slice is within atol of the expected one "
            f"({d_channel:.3e}), so case 2 is channel-blind as written"
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
    x, filters = _image(), _filters()
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0, so this control is unarmed"
    )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = patch_embed(x, filters, PATCH)
    nki_dispatch, torch_fallback = dispatch_counters()
    print(
        f"[route-control] nki_dispatch={nki_dispatch} "
        f"torch_fallback={torch_fallback} simulate_kernel_calls={sim.calls}"
    )
    assert nki_dispatch == 0, f"expected 0 NKI dispatches, got {nki_dispatch}"
    assert torch_fallback == 1, f"expected 1 torch fallback, got {torch_fallback}"
    assert sim.calls == 0, f"the simulator ran {sim.calls} times with it disabled"
    assert tuple(out.shape) == (
        BATCH,
        C_OUT,
        1,
        IMAGE_SIDE // PATCH,
        IMAGE_SIDE // PATCH,
    )


def test_route_control_simulator_is_load_bearing() -> None:
    """The NKI chain RAISES without the simulator rather than computing torch.

    Recorded because it is what forecloses the F1 false green BELOW this
    repository's seam: if the HOP silently degraded to a torch path, a green
    numeric comparison could not be attributed to a kernel at all, and no
    counter of this module's could detect it.
    """
    x, filters = _image(), _filters()
    saved = os.environ.get("NKI_SIMULATOR")
    os.environ["NKI_SIMULATOR"] = "0"
    try:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
        from nkilib.experimental.conv.conv3d import conv3d

        with pytest.raises(RuntimeError) as excinfo:
            wrap_nki(conv3d)(
                x_in=x,
                filters=filters,
                bias=None,
                stride=patch_stride(PATCH),
                padding=NO_PADDING,
                dilation=UNIT_DILATION,
                activation_fn=None,
                lnc_shard=LNC_SHARD,
            )
    finally:
        if saved is None:
            os.environ.pop("NKI_SIMULATOR", None)
        else:
            os.environ["NKI_SIMULATOR"] = saved

    message = str(excinfo.value)
    print(f"[route-control] simulator_off_raise={message[:160]!r}")
    assert "simulator" in message.lower(), message


def test_stride_derivation_is_load_bearing() -> None:
    """The substrate's DEFAULT stride computes a different, overlapping answer.

    This kernel carries no ``assert`` statements, so unlike `inc-glm53f-034`'s
    conv1d there is no vendor refusal to record: the substrate default
    ``stride = (1, 1, 1)`` succeeds and returns a LARGER extent, because the
    patch windows then overlap at every pixel offset. That silent wrongness is
    exactly why the seam derives the stride, and it is measured here on the
    substrate's own formula rather than argued.
    """
    derived = output_extents(
        (SINGLETON_D, IMAGE_SIDE, IMAGE_SIDE),
        (SINGLETON_D, PATCH, PATCH),
        NO_PADDING,
        patch_stride(PATCH),
        UNIT_DILATION,
    )
    defaulted = output_extents(
        (SINGLETON_D, IMAGE_SIDE, IMAGE_SIDE),
        (SINGLETON_D, PATCH, PATCH),
        NO_PADDING,
        (1, 1, 1),
        UNIT_DILATION,
    )
    print(
        f"[stride-control] derived_stride={patch_stride(PATCH)} extents={derived} "
        f"substrate_default_stride=(1, 1, 1) extents={defaulted}"
    )
    assert derived == (1, IMAGE_SIDE // PATCH, IMAGE_SIDE // PATCH), derived
    assert defaulted != derived, (
        "the substrate default and the derived stride give the same extent, so "
        "this control cannot show the derivation matters"
    )

    # And the seam, which derives it, agrees with the substrate's formula.
    x, filters = _image(), _filters()
    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = patch_embed(x, filters, PATCH)
    _assert_route(sim, 1, "stride-control-seam")
    assert tuple(out.shape) == (BATCH, C_OUT) + derived


# --------------------------------------------------------------------------- #
# The WRAP is checkable: the seam dispatches to the SUBSTRATE's member.          #
# --------------------------------------------------------------------------- #
def test_seam_wraps_the_substrate_member_and_authors_no_kernel() -> None:
    """``kernel_identity()`` names ``nkilib``'s kernel, not anything authored here.

    A substitution detector, NOT a route reading (D13.1): it certifies what this
    module imported. The route reading is the counter, taken through the seam.
    """
    module, qualname = kernel_identity()
    print(f"[wrap-check] kernel_identity=({module}, {qualname})")
    assert module == SUBSTRATE_MODULE, module
    assert qualname == SUBSTRATE_QUALNAME, qualname
    assert not module.startswith("vllm_neuron"), (
        f"the seam dispatches to {module}, which is this repository's own code; "
        f"a WRAP must dispatch to the substrate"
    )


def test_reference_is_the_substrates_own_and_not_authored_here() -> None:
    """The oracle is ``nkilib``'s, so neither side of case 1 is this repo's code."""
    from nkilib.experimental.conv.conv3d_torch import conv3d_torch_ref

    print(
        f"[oracle-check] reference={conv3d_torch_ref.__module__}."
        f"{conv3d_torch_ref.__qualname__}"
    )
    assert conv3d_torch_ref.__module__ == SUBSTRATE_REFERENCE_MODULE
    assert not conv3d_torch_ref.__module__.startswith("vllm_neuron")

    # The wrap's reference path is that member and nothing else: the return-key
    # normalisation is the only thing between them.
    x, filters = _image(), _filters()
    raw = conv3d_torch_ref(
        x,
        filters,
        bias=None,
        stride=patch_stride(PATCH),
        padding=NO_PADDING,
        dilation=UNIT_DILATION,
        activation_fn=None,
        lnc_shard=LNC_SHARD,
    )
    assert set(raw) == {"out"}, (
        f"this kernel's reference returns {sorted(raw)}; the wrap unwraps 'out' "
        f"and the conv1d reference's 'output' key does not apply here"
    )
    assert torch.equal(patch_embed_torch_reference(x, filters, PATCH), raw["out"])


def test_dispatch_counters_are_module_level_state_reachable_from_elsewhere() -> None:
    """Another increment's test module can zero and read these counters.

    The placement `inc-glm53f-026` and `inc-glm53f-034` landed. A test-local
    counter would satisfy this increment and break the next one that reads this
    seam.
    """
    module = importlib.import_module(_MODULE)
    for name in ("reset_dispatch_counters", "dispatch_counters"):
        assert callable(getattr(module, name)), name
        assert name in module.__all__, f"{name} is not exported"
    module.reset_dispatch_counters()
    assert module.dispatch_counters() == (0, 0)
    print(f"[counter-placement] module={module.__name__} after_reset=(0, 0)")


# --------------------------------------------------------------------------- #
# This module contains no vision grid arithmetic (design-20260905-aq (iii)).     #
# --------------------------------------------------------------------------- #
def test_module_contains_no_vision_grid_arithmetic_and_no_model_import() -> None:
    """The transformers patch-grid rule is re-derived nowhere in this file.

    Ruling ``design-20260905-aq`` (iii): the fork consumes that arithmetic and
    re-derives it nowhere. This module satisfies it by containing none -- it
    knows about convolution extents and nothing about canvases, merge sizes or
    token budgets -- and by not importing across the ``functional/`` -> ``model/``
    boundary, which no module under ``functional/`` does.
    """
    source = Path(
        sys.modules[_MODULE].__file__
    ).read_text(encoding="utf-8")
    # Constructs, not words: a docstring may legitimately DISCUSS the grid, and
    # this file's does (§67 -- match the construct the claim names).
    code_lines = [
        line
        for line in source.splitlines()
        if not line.lstrip().startswith("#")
    ]
    code = "\n".join(code_lines)
    forbidden_calls = [
        r"\bsmart_resize\s*\(",
        r"\bimage_grid_spec\s*\(",
        r"\bvideo_grid_spec\s*\(",
        r"\bgrid_thw\b",
        r"\bmerge_size\b",
        r"\bspatial_merge",
        r"\bpixels_per_token\b",
    ]
    hits = {p: len(re.findall(p, code)) for p in forbidden_calls}
    imports = re.findall(r"^\s*(?:from|import)\s+vllm_neuron\.model", source, re.M)
    print(f"[aq-iii] forbidden_construct_hits={hits} model_imports={len(imports)}")
    assert sum(hits.values()) == 0, hits
    assert len(imports) == 0, imports

    # Firing control: the same recipe over a planted line reads 1, so the zeros
    # above are readings and not an empty scan (§64).
    planted = code + "\n    grid = smart_resize(h, w)\n"
    control = len(re.findall(r"\bsmart_resize\s*\(", planted))
    planted_import = "from vllm_neuron.model.glm5_next import x"
    control_import = len(
        re.findall(r"^\s*(?:from|import)\s+vllm_neuron\.model", planted_import, re.M)
    )
    print(
        f"[aq-iii-control] planted_call_hits={control} "
        f"planted_import_hits={control_import}"
    )
    assert control == 1 and control_import == 1, (
        "the scan does not fire on a planted violation, so its zeros mean nothing"
    )


# --------------------------------------------------------------------------- #
# Refusals: named, and grounded in the substrate's own documented ranges.       #
# --------------------------------------------------------------------------- #
def test_non_degenerate_filter_depth_is_refused_by_name() -> None:
    """``K_d = 2`` is refused: a real 3-D convolution is not this seam's call."""
    x = _image()
    filters = torch.zeros(2, PATCH, PATCH, C_IN, C_OUT, dtype=torch.float32)
    with pytest.raises(VisionPatchEmbedError) as excinfo:
        patch_embed(x, filters, PATCH)
    message = str(excinfo.value)
    print(f"[refusal] K_d=2 -> {message[:200]!r}")
    assert "K_d=2" in message, message
    assert f"must be {SINGLETON_D}" in message, message


# ``ids`` is given explicitly: without it pytest names each case after the repr
# of its lambda, which puts ``<lambda>`` inside the node id. These ids are the
# ADDED set a leg D ROW M gate names by name, so they are readable and stable.
@pytest.mark.parametrize(
    ("label", "make", "needle"),
    [
        (
            "window_not_equal_to_stride",
            lambda: (
                _image(),
                torch.zeros(
                    SINGLETON_D, PATCH + 1, PATCH + 1, C_IN, C_OUT,
                    dtype=torch.float32,
                ),
                PATCH,
            ),
            "must both equal patch_size",
        ),
        (
            "image_not_a_whole_number_of_patches",
            lambda: (
                torch.zeros(
                    BATCH, C_IN, SINGLETON_D, IMAGE_SIDE + 1, IMAGE_SIDE,
                    dtype=torch.float32,
                ),
                _filters(),
                PATCH,
            ),
            "whole number of",
        ),
        (
            "dtype_mismatch",
            lambda: (
                _image(),
                _filters().to(torch.bfloat16),
                PATCH,
            ),
            "differ",
        ),
        (
            "channel_disagreement",
            lambda: (
                _image(),
                torch.zeros(
                    SINGLETON_D, PATCH, PATCH, C_IN + 1, C_OUT,
                    dtype=torch.float32,
                ),
                PATCH,
            ),
            "does not match x_in C_in",
        ),
        (
            "c_in_below_the_documented_range",
            lambda: (
                torch.zeros(
                    BATCH, C_IN_RANGE[0] - 1, SINGLETON_D, IMAGE_SIDE, IMAGE_SIDE,
                    dtype=torch.float32,
                ),
                torch.zeros(
                    SINGLETON_D, PATCH, PATCH, C_IN_RANGE[0] - 1, C_OUT,
                    dtype=torch.float32,
                ),
                PATCH,
            ),
            "Intended Usage Range",
        ),
        (
            "c_out_below_the_documented_range",
            lambda: (
                _image(),
                torch.zeros(
                    SINGLETON_D, PATCH, PATCH, C_IN, C_OUT_RANGE[0] - 1,
                    dtype=torch.float32,
                ),
                PATCH,
            ),
            "Intended Usage Range",
        ),
        (
            "wrong_rank_input",
            lambda: (
                torch.zeros(BATCH, C_IN, IMAGE_SIDE, IMAGE_SIDE, dtype=torch.float32),
                _filters(),
                PATCH,
            ),
            "must be 5-D",
        ),
        (
            "padding_is_not_passed_through",
            lambda: (_image(), _filters(), PATCH),
            "must be (0, 0, 0, 0, 0, 0)",
        ),
    ],
    ids=[
        "window_not_equal_to_stride",
        "image_not_a_whole_number_of_patches",
        "dtype_mismatch",
        "channel_disagreement",
        "c_in_below_the_documented_range",
        "c_out_below_the_documented_range",
        "wrong_rank_input",
        "padding_is_not_passed_through",
    ],
)
def test_refusals_name_the_offending_extent(label, make, needle) -> None:
    """Every refusal says which extent or option it refused, and why."""
    x, filters, patch = make()
    kwargs = {}
    if label == "padding_is_not_passed_through":
        kwargs["padding"] = (0, 0, 1, 1, 0, 0)
    with pytest.raises(VisionPatchEmbedError) as excinfo:
        patch_embed(x, filters, patch, **kwargs)
    message = str(excinfo.value)
    print(f"[refusal] {label} -> {message[:200]!r}")
    assert needle in message, f"{label}: {needle!r} not in {message!r}"


def test_patch_size_outside_the_substrate_stride_range_is_refused() -> None:
    """``patch_stride`` refuses a patch size the kernel's stride range excludes."""
    with pytest.raises(VisionPatchEmbedError) as excinfo:
        patch_stride(0)
    print(f"[refusal] patch_size=0 -> {str(excinfo.value)[:160]!r}")
    assert "documented stride range" in str(excinfo.value)
    assert patch_stride(PATCH) == (1, PATCH, PATCH)


def test_gate_reports_availability_separately_from_admissibility() -> None:
    """An inadmissible call RAISES from the gate; it never reads False.

    Unavailable and inadmissible are different answers, and merging them would
    let a refused geometry take the torch path -- a torch path for kernel-class
    work (P13, D6).
    """
    x, filters = _image(), _filters()
    assert can_run_patch_embed(x, filters, PATCH) is True
    with pytest.raises(VisionPatchEmbedError):
        can_run_patch_embed(x, torch.zeros(2, PATCH, PATCH, C_IN, C_OUT), PATCH)
    print("[gate] admissible=True inadmissible=raises")


def test_output_extents_is_the_substrates_formula() -> None:
    """The formula this module states is the kernel's, on cases with known answers."""
    assert output_extents((1, 8, 8), (1, 4, 4), NO_PADDING, (1, 4, 4), (1, 1, 1)) == (
        1,
        2,
        2,
    )
    assert output_extents((1, 8, 8), (1, 4, 4), NO_PADDING, (1, 1, 1), (1, 1, 1)) == (
        1,
        5,
        5,
    )
    assert output_extents((1, 4, 4), (1, 4, 4), NO_PADDING, (1, 4, 4), (1, 1, 1)) == (
        1,
        1,
        1,
    )
    with pytest.raises(VisionPatchEmbedError):
        output_extents((1, 2, 2), (1, 4, 4), NO_PADDING, (1, 4, 4), (1, 1, 1))
    print("[formula] 3 known extents agree, 1 impossible extent refused")


def test_bf16_tower_dtype_is_admitted_by_the_seam() -> None:
    """The rev-231 rider tied to code: the real tower is bf16 and the seam takes it.

    STRUCTURAL only -- it asserts the gate admits bf16 and the seam dispatches
    once. It authors NO tolerance pair (P9): the declared numeric cases are fp32
    because ``atol=1e-5`` is not expressible in bf16 at all.
    """
    x = _image().to(torch.bfloat16)
    filters = _filters().to(torch.bfloat16)
    assert can_run_patch_embed(x, filters, PATCH) is True

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = patch_embed(x, filters, PATCH)
    _assert_route(sim, 1, "bf16-rider")
    print(
        f"[bf16-rider] in_dtype={x.dtype} out_dtype={out.dtype} "
        f"out_shape={tuple(out.shape)}"
    )
    assert tuple(out.shape) == (
        BATCH,
        C_OUT,
        1,
        IMAGE_SIDE // PATCH,
        IMAGE_SIDE // PATCH,
    )


def test_batch_range_bound_is_the_kernels_own() -> None:
    """The transcribed ranges are the kernel's, checked against its docstring."""
    from nkilib.experimental.conv.conv3d import conv3d as substrate

    inner = getattr(substrate, "func", None) or substrate
    doc = inner.__doc__ or ""
    for label, bounds in (
        ("B", BATCH_RANGE),
        ("C_in", C_IN_RANGE),
        ("C_out", C_OUT_RANGE),
    ):
        needle = f"{label}: {bounds[0]}-{bounds[1]}"
        print(f"[range-check] {needle!r} in the kernel docstring")
        assert needle in doc, (
            f"this module transcribes {needle!r}, which the kernel's own "
            f"docstring does not say"
        )
    assert len(doc) > 0, "an empty docstring could not have confirmed anything"
