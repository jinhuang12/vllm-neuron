# SPDX-License-Identifier: Apache-2.0
"""Tests for the KDA prefill depthwise conv1d wrapper.

Two numeric cases and a set of refusals.

The first case compares the simulated kernel against ``nkilib``'s own torch
reference for it, so neither side of the comparison is numerics authored here.
That comparison alone could pass while measuring nothing about a kernel: if the
module took its torch path, both sides would be torch. So each numeric case also
reads the module's dispatch counters and counts real
``nki.simulator.simulate_kernel`` calls, which is the vendor entry point and so
independent of the module's own counter.

The second case is an argument-order proof rather than a second numeric arm. It
compares the kernel against a closed form computed here from the taps alone: with
a unit impulse at input position ``p``, cross-correlation puts ``filter[p - q]``
at output position ``q``, so the output row is the tap vector reversed and then
zero-padded. A wrapper that passed the image where the filter belongs could not
reproduce a tap vector it never received in that position. The taps are all
distinct so that "reversed" is falsifiable rather than a symmetric coincidence.
"""

from __future__ import annotations

import os

import pytest
import torch

import nki.simulator

from vllm_neuron.functional.kda.depthwise_conv1d import (
    LNC_SHARDS,
    NO_PADDING,
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

#: The tiny case's geometry. ``C`` is read off ``LNC_SHARDS`` so the fixture and
#: the module's divisibility refusal cannot drift apart.
BATCH = 1
CHANNELS = 4 * LNC_SHARDS  # 8
WIDTH = 16
TAPS = 4
Q = output_width(WIDTH, TAPS)  # 13

RTOL = 1e-2
ATOL = 1e-5

#: The impulse case recovers taps exactly, so it is bounded absolutely with no
#: relative term: a relative term would scale the bar with the taps' own
#: magnitude, which is the leniency an exact-recovery claim must not have.
IMPULSE_ATOL = 1e-5

#: Where the unit impulse sits. Interior, so the reversed tap window is fully
#: inside the input and the closed form below is exact rather than truncated.
IMPULSE_AT = TAPS - 1

#: The ``nkilib`` member this module wraps.
SUBSTRATE_MODULE = "nkilib.experimental.conv.depthwise_conv1d"
SUBSTRATE_QUALNAME = "depthwise_conv1d_implicit_gemm"


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


def _assert_nki_ran(sim: _SimulatorCounter, expected: int, label: str) -> None:
    """Require that the NKI path, and only the NKI path, served the calls."""
    nki_dispatch, torch_fallback = dispatch_counters()
    assert nki_dispatch == expected, (
        f"{label}: the dispatch counter read {nki_dispatch}, expected {expected}"
    )
    assert torch_fallback == 0, (
        f"{label}: the torch-fallback counter read {torch_fallback}; a fallback "
        f"would compare torch against torch"
    )
    assert can_run_kernel(torch.zeros(1)) is True, f"{label}: no device or simulator"
    assert sim.calls == expected, (
        f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, expected "
        f"{expected}; a numeric pass without a simulator call measures no kernel"
    )


def _image(seed: int = 34) -> torch.Tensor:
    """A deterministic ``[N, C, 1, W]`` fp32 input.

    fp32 rather than bf16 because bf16's ~3 decimal digits could not express a
    difference at :data:`ATOL` at all, so the comparison would measure the storage
    format instead of the kernel. Centred on zero so the convolution's sum carries
    no DC term for a wrong tap ordering to hide inside.
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
        torch.rand((CHANNELS, 1, 1, TAPS), generator=generator, dtype=torch.float32)
        * 2.0
        - 1.0
    )


def _distinct_taps() -> torch.Tensor:
    """Taps distinct per position and per channel, for the impulse case.

    All ``C * S`` values are the consecutive integers ``1 .. C*S``, so every value
    appears exactly once and any permutation of any two shows up as a changed
    reading. Powers of two scaled per channel would not do: that construction
    collides, because ``2 * 2 == 1 * 4``.
    """
    values = torch.arange(1, CHANNELS * TAPS + 1, dtype=torch.float32)
    return values.reshape(CHANNELS, 1, 1, TAPS)


def _impulse_image() -> torch.Tensor:
    """``[N, C, 1, W]`` zeros with a single ``1.0`` per channel at :data:`IMPULSE_AT`."""
    img = torch.zeros((BATCH, CHANNELS, 1, WIDTH), dtype=torch.float32)
    img[0, :, 0, IMPULSE_AT] = 1.0
    return img


def _impulse_closed_form(filt: torch.Tensor) -> torch.Tensor:
    """The output a unit impulse must produce, computed from the taps alone.

    Built without calling the kernel, the reference, or any convolution operator,
    which is what makes it an independent expectation.
    """
    expected = torch.zeros((BATCH, CHANNELS, 1, Q), dtype=torch.float32)
    reversed_taps = torch.flip(filt.reshape(CHANNELS, TAPS), dims=(1,))
    span = min(TAPS, Q)
    expected[0, :, 0, :span] = reversed_taps[:, TAPS - span :]
    return expected


def test_the_kernel_matches_the_nkilib_torch_reference() -> None:
    """The tiny case agrees with ``nkilib``'s reference, in shape, dtype and value."""
    img, filt = _image(), _filter()
    expected = depthwise_conv1d_torch_reference(img, filt)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        actual = depthwise_conv1d(img, filt)
    _assert_nki_ran(sim, 1, "reference-case")

    assert tuple(actual.shape) == (BATCH, CHANNELS, 1, Q), (
        f"shape {tuple(actual.shape)} is not the expected [N, C, 1, Q] "
        f"{(BATCH, CHANNELS, 1, Q)}"
    )
    assert actual.dtype == img.dtype, (
        f"output dtype {actual.dtype} is not the input dtype {img.dtype}"
    )
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_a_unit_impulse_recovers_the_taps_reversed() -> None:
    """A unit impulse returns the tap vector reversed, which pins argument order.

    The closed form is also checked against ``nkilib``'s reference, so the two
    independent sides agree on it as well.
    """
    filt = _distinct_taps()
    img = _impulse_image()
    expected = _impulse_closed_form(filt)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        actual = depthwise_conv1d(img, filt)
    _assert_nki_ran(sim, 1, "unit-impulse-case")

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=IMPULSE_ATOL)

    reference = depthwise_conv1d_torch_reference(img, filt)
    torch.testing.assert_close(reference, expected, rtol=0.0, atol=IMPULSE_ATOL)


def test_the_torch_fallback_is_taken_and_counted_without_a_simulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no device or simulator the torch path runs and the fallback is counted.

    This is what makes ``torch_fallback == 0`` in the numeric cases meaningful:
    the counter is shown reading ``1`` and ``nki_dispatch`` reading ``0``, through
    the real gate rather than a mock.
    """
    img, filt = _image(), _filter()
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0, so this case is vacuous"
    )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = depthwise_conv1d(img, filt)

    assert dispatch_counters() == (0, 1), (
        f"expected 0 NKI dispatches and 1 torch fallback, got {dispatch_counters()}"
    )
    assert sim.calls == 0, f"the simulator ran {sim.calls} times with it disabled"
    assert tuple(out.shape) == (BATCH, CHANNELS, 1, Q)


def test_kernel_identity_names_the_nkilib_kernel() -> None:
    """``kernel_identity`` names ``nkilib``'s kernel, not anything authored here."""
    assert kernel_identity() == (SUBSTRATE_MODULE, SUBSTRATE_QUALNAME)


def test_the_torch_reference_delegates_to_the_nkilib_reference() -> None:
    """The module's reference is bit-identical to the ``nkilib`` one it wraps.

    Measured by driving the vendor function directly with the same inputs: a torch
    convolution authored here would differ in the last bits at best.
    """
    from nkilib.experimental.conv.depthwise_conv1d_torch import (
        depthwise_conv1d_implicit_gemm_torch_ref,
    )

    img, filt = _image(), _filter()
    direct = depthwise_conv1d_implicit_gemm_torch_ref(
        img, filt, padding=NO_PADDING, stride=UNIT_STRIDE, feature_group_count=CHANNELS
    )["output"]
    through_module = depthwise_conv1d_torch_reference(img, filt)
    assert torch.equal(through_module, direct), (
        "the module's reference is not bit-identical to the vendor's own, so it is "
        "not a pure delegation"
    )


def test_dispatch_counters_accumulate_across_calls() -> None:
    """The counters are module-level state: they accumulate until reset."""
    img, filt = _image(), _filter()
    reset_dispatch_counters()
    assert dispatch_counters() == (0, 0)

    depthwise_conv1d(img, filt)
    assert dispatch_counters() == (1, 0)

    depthwise_conv1d(img, filt)
    assert dispatch_counters() == (2, 0), (
        f"expected (2, 0) after two dispatches, got {dispatch_counters()}; the "
        f"counter does not accumulate across calls"
    )

    reset_dispatch_counters()
    assert dispatch_counters() == (0, 0)


def test_a_channel_count_the_shards_do_not_divide_is_refused() -> None:
    """An odd ``C`` raises, and the refusal neither dispatches nor falls back.

    The refusal is this module's, on the ``nkilib`` kernel's documented device
    requirement that ``C`` divide by the shard count: the NKI simulator does not
    enforce it, so nothing below this module would refuse.
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
    """Every option the ``nkilib`` kernel asserts on is refused before dispatch.

    Refused rather than coerced or routed to torch, and each message names the
    option so a caller can act on it.
    """
    img, filt = _image(), _filter()
    reset_dispatch_counters()
    with pytest.raises(KdaDepthwiseConv1dError) as excinfo:
        depthwise_conv1d(img, filt, **kwargs)
    assert needle in str(excinfo.value), str(excinfo.value)
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
    assert "smaller than the kernel size" in str(width_exc.value), str(width_exc.value)


def test_dtype_mismatch_is_refused() -> None:
    """Mixed dtypes are refused, because the kernel contracts the two together."""
    img, filt = _image(), _filter()
    with pytest.raises(KdaDepthwiseConv1dError) as excinfo:
        depthwise_conv1d(img, filt.to(torch.bfloat16))
    assert "dtype" in str(excinfo.value), str(excinfo.value)


def test_gate_reports_availability_separately_from_admissibility() -> None:
    """``can_run_depthwise_conv1d`` answers availability and raises on inadmissible.

    The two conditions are deliberately not merged, so a caller can tell "no device
    or simulator" from "this kernel does not accept these extents".
    """
    img, filt = _image(), _filter()
    assert can_run_depthwise_conv1d(img, filt) is True

    with pytest.raises(KdaDepthwiseConv1dError):
        can_run_depthwise_conv1d(img, filt, batch_group_count=3)


def test_output_width_follows_the_convolution_formula() -> None:
    """``output_width`` handles padding and dilation, and refuses a too-short input."""
    assert Q == (WIDTH - TAPS) // 1 + 1
    assert output_width(WIDTH, TAPS, ((0, 0), (2, 2))) == (WIDTH + 4 - TAPS) + 1
    assert output_width(WIDTH, TAPS, NO_PADDING, (1, 2)) == (WIDTH - TAPS) // 2 + 1
    with pytest.raises(KdaDepthwiseConv1dError):
        output_width(TAPS - 1, TAPS)
