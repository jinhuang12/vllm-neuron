# SPDX-License-Identifier: Apache-2.0
"""Tests for the vision patch embedding wrapper.

Two numeric cases.

The first compares the simulated kernel against ``nkilib``'s own conv3d torch
reference, so neither side is numerics authored here. That comparison alone could
pass while measuring nothing about a kernel: if the module took its torch path,
both sides would be torch. So each numeric case also reads the module's dispatch
counters and counts real ``nki.simulator.simulate_kernel`` calls, which is the
vendor entry point and so independent of the module's own counter.

The second runs on an input that is exactly one patch, ``H = W = P``, so with the
derived stride there is exactly one output position. A single unit impulse at input
channel ``c0`` and pixel ``(kh0, kw0)`` then makes the whole output equal one slice
of the filter, ``filters[0, kh0, kw0, c0, :]`` -- a closed form computed here from
the filter alone. Three faults would break it: a depth axis that contracted
anything, so the ``K_d = 1`` degeneration is not wired right; a filter and an image
passed to each other's argument; or a filter index order read differently by the
wrapper than here.

Both cases are fp32 rather than the tower's bf16, because ``atol = 1e-5`` is not
expressible in bf16 at all, so a bf16 comparison would measure the storage format
instead of the kernel. That the real tower's bf16 inputs are admitted is checked
structurally instead, without a second tolerance.
"""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import pytest
import torch

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

#: The tiny geometry both numeric cases run on. ``C_in = 3`` is a three-channel
#: image tower, inside the kernel's documented ``C_in: 3-1280``.
BATCH = 1
C_IN = 3
C_OUT = 8
PATCH = 4

#: Two patches per axis on the reference case, so the output extent is not 1 and a
#: stride error cannot hide in a degenerate grid.
IMAGE_SIDE = PATCH * 2

#: Exactly one patch on the impulse case, so the whole output is one filter slice.
IMPULSE_SIDE = PATCH

RTOL = 1e-2
ATOL = 1e-5

#: The impulse case recovers weights exactly, so it is bounded absolutely.
IMPULSE_ATOL = 1e-5

#: The impulse position and channel. ``kh0 != kw0`` on purpose, so a transposed
#: index order gives a different slice.
IMPULSE_CHANNEL = 1
IMPULSE_H = 0
IMPULSE_W = 2

_MODULE = "vllm_neuron.functional.vision.patch_embed"
SUBSTRATE_MODULE = "nkilib.experimental.conv.conv3d"
SUBSTRATE_QUALNAME = "conv3d"
SUBSTRATE_REFERENCE_MODULE = "nkilib.experimental.conv.conv3d_torch"


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

    ``arange`` rather than random: two random slices could coincide by accident and
    the impulse case would report a pass it did not earn.
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
    """The output a single unit impulse must produce, from the filter alone.

    ``out[0, :, 0, 0, 0] == filters[0, kh0, kw0, c0, :]``, computed here and never
    from the module under test.
    """
    slice_ = filters[0, IMPULSE_H, IMPULSE_W, IMPULSE_CHANNEL, :]
    return slice_.reshape(1, C_OUT, 1, 1, 1).contiguous()


def test_the_kernel_matches_the_nkilib_torch_reference() -> None:
    """The tiny case agrees with ``nkilib``'s conv3d reference, and in shape."""
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
    _assert_nki_ran(sim, 1, "reference-case")

    assert tuple(actual.shape) == expected_shape, (
        f"the kernel returned {tuple(actual.shape)}, the convolution formula says "
        f"{expected_shape}"
    )
    assert tuple(reference.shape) == expected_shape
    assert_close(actual, reference, rtol=RTOL, atol=ATOL, name="patch_embed")


def test_a_single_patch_impulse_recovers_the_filter_slice() -> None:
    """One patch and one unit impulse return one filter slice, exactly.

    The near-miss slices are required to differ from the expected one, so "this
    slice" is a falsifiable statement about index order rather than a coincidence.
    ``nkilib``'s reference is checked against the same closed form, as a second
    independent reading.
    """
    x = _impulse_image()
    filters = _distinct_filters()
    expected = _impulse_closed_form(filters)

    transposed = filters[0, IMPULSE_W, IMPULSE_H, IMPULSE_CHANNEL, :].flatten()
    other_channel = filters[
        0, IMPULSE_H, IMPULSE_W, (IMPULSE_CHANNEL + 1) % C_IN, :
    ].flatten()
    flat = expected.flatten()
    assert flat.numel() == C_OUT > 1, "a one-element expectation cannot discriminate"
    assert float((flat - transposed).abs().max()) > IMPULSE_ATOL, (
        "the transposed-index slice equals the expected one, so this case is "
        "order-blind as written"
    )
    assert float((flat - other_channel).abs().max()) > IMPULSE_ATOL, (
        "another channel's slice equals the expected one, so this case is "
        "channel-blind as written"
    )

    reference = patch_embed_torch_reference(x, filters, PATCH)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        actual = patch_embed(x, filters, PATCH)
    _assert_nki_ran(sim, 1, "impulse-case")

    assert tuple(actual.shape) == (BATCH, C_OUT, 1, 1, 1), (
        f"one patch of input must give one output position, got "
        f"{tuple(actual.shape)}"
    )
    assert_close(
        actual, expected, rtol=0.0, atol=IMPULSE_ATOL, name="impulse_vs_closed_form"
    )
    assert_close(
        reference, expected, rtol=0.0, atol=IMPULSE_ATOL,
        name="reference_vs_closed_form",
    )


def test_the_torch_fallback_is_taken_and_counted_without_a_simulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no device or simulator the torch path runs and the fallback is counted.

    This is what makes ``torch_fallback == 0`` in the numeric cases meaningful.
    """
    x, filters = _image(), _filters()
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0, so this case is vacuous"
    )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = patch_embed(x, filters, PATCH)

    assert dispatch_counters() == (0, 1), (
        f"expected 0 NKI dispatches and 1 torch fallback, got {dispatch_counters()}"
    )
    assert sim.calls == 0, f"the simulator ran {sim.calls} times with it disabled"
    assert tuple(out.shape) == (
        BATCH,
        C_OUT,
        1,
        IMAGE_SIDE // PATCH,
        IMAGE_SIDE // PATCH,
    )


def test_the_derived_patch_stride_gives_a_non_overlapping_grid() -> None:
    """The derived stride, not the kernel's default, is what makes patches disjoint.

    This kernel holds no ``assert`` statements, so the default ``stride = (1, 1, 1)``
    does not fail: it succeeds and returns a larger extent, because the patch windows
    then overlap at every pixel offset. That silent wrongness is why the stride is
    derived from the patch size, and it is read off the convolution formula here.
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
    assert derived == (1, IMAGE_SIDE // PATCH, IMAGE_SIDE // PATCH), derived
    assert defaulted != derived, (
        "the default and the derived stride give the same extent, so this case "
        "cannot show the derivation matters"
    )

    x, filters = _image(), _filters()
    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = patch_embed(x, filters, PATCH)
    _assert_nki_ran(sim, 1, "stride-case")
    assert tuple(out.shape) == (BATCH, C_OUT) + derived


def test_kernel_identity_names_the_nkilib_kernel() -> None:
    """``kernel_identity`` names ``nkilib``'s conv3d, not anything authored here.

    A substitution detector rather than a reading about what ran; the dispatch
    counters are what say that.
    """
    assert kernel_identity() == (SUBSTRATE_MODULE, SUBSTRATE_QUALNAME)


def test_the_torch_reference_delegates_to_the_nkilib_reference() -> None:
    """The module's reference is ``nkilib``'s, with only the return key normalised.

    This kernel's reference returns its result under ``"out"``, not the ``"output"``
    key the conv1d reference uses, so the unwrapping is checked rather than assumed.
    """
    from nkilib.experimental.conv.conv3d_torch import conv3d_torch_ref

    assert conv3d_torch_ref.__module__ == SUBSTRATE_REFERENCE_MODULE

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
        f"this kernel's reference returns {sorted(raw)}; the wrapper unwraps 'out'"
    )
    assert torch.equal(patch_embed_torch_reference(x, filters, PATCH), raw["out"])


def test_dispatch_counters_accumulate_across_calls() -> None:
    """The counters are module-level state: they accumulate until reset."""
    module = importlib.import_module(_MODULE)
    for name in ("reset_dispatch_counters", "dispatch_counters"):
        assert name in module.__all__, f"{name} is not exported"

    x, filters = _image(), _filters()
    reset_dispatch_counters()
    assert dispatch_counters() == (0, 0)

    patch_embed(x, filters, PATCH)
    assert dispatch_counters() == (1, 0)

    patch_embed(x, filters, PATCH)
    assert dispatch_counters() == (2, 0), (
        f"expected (2, 0) after two dispatches, got {dispatch_counters()}"
    )

    reset_dispatch_counters()
    assert dispatch_counters() == (0, 0)


def test_the_module_does_not_import_across_the_model_boundary() -> None:
    """Nothing under ``functional/`` imports from ``model/``, this module included.

    The patch grid -- canvas sizes, merge sizes, token budgets -- is the model's
    arithmetic. This module knows about convolution extents and nothing else, so the
    boundary holds by there being nothing to import.
    """
    module = importlib.import_module(_MODULE)
    source = Path(module.__file__).read_text(encoding="utf-8")
    imports = re.findall(r"^\s*(?:from|import)\s+vllm_neuron\.model", source, re.M)
    assert imports == [], imports


def test_a_non_degenerate_filter_depth_is_refused_by_name() -> None:
    """``K_d = 2`` is refused: a real 3-D convolution is not this wrapper's call."""
    x = _image()
    filters = torch.zeros(2, PATCH, PATCH, C_IN, C_OUT, dtype=torch.float32)
    with pytest.raises(VisionPatchEmbedError) as excinfo:
        patch_embed(x, filters, PATCH)
    message = str(excinfo.value)
    assert "K_d=2" in message, message
    assert f"must be {SINGLETON_D}" in message, message


# ``ids`` is given explicitly: without it pytest names each case after the repr of
# its lambda, which puts ``<lambda>`` inside the node id.
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
    assert needle in str(excinfo.value), f"{label}: {needle!r} not in {excinfo.value!r}"


def test_a_patch_size_outside_the_kernels_stride_range_is_refused() -> None:
    """``patch_stride`` refuses a patch size the kernel's stride range excludes."""
    with pytest.raises(VisionPatchEmbedError) as excinfo:
        patch_stride(0)
    assert "documented stride range" in str(excinfo.value)
    assert patch_stride(PATCH) == (1, PATCH, PATCH)


def test_gate_reports_availability_separately_from_admissibility() -> None:
    """An inadmissible call raises from the gate; it never reads False.

    Unavailable and inadmissible are different answers, and merging them would let a
    refused geometry take the torch path.
    """
    x, filters = _image(), _filters()
    assert can_run_patch_embed(x, filters, PATCH) is True
    with pytest.raises(VisionPatchEmbedError):
        can_run_patch_embed(x, torch.zeros(2, PATCH, PATCH, C_IN, C_OUT), PATCH)


def test_output_extents_follows_the_convolution_formula() -> None:
    """Three extents with known answers, and one impossible extent refused."""
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


def test_the_towers_bf16_dtype_is_admitted_and_dispatches() -> None:
    """bf16 inputs pass the gate and reach the kernel.

    Structural only, and it authors no tolerance: the numeric cases above are fp32
    because ``atol = 1e-5`` is not expressible in bf16.
    """
    x = _image().to(torch.bfloat16)
    filters = _filters().to(torch.bfloat16)
    assert can_run_patch_embed(x, filters, PATCH) is True

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = patch_embed(x, filters, PATCH)
    _assert_nki_ran(sim, 1, "bf16-case")
    assert tuple(out.shape) == (
        BATCH,
        C_OUT,
        1,
        IMAGE_SIDE // PATCH,
        IMAGE_SIDE // PATCH,
    )


def test_the_transcribed_extent_ranges_are_the_kernels_own() -> None:
    """The ranges this module states appear in the kernel's own docstring."""
    from nkilib.experimental.conv.conv3d import conv3d as substrate

    inner = getattr(substrate, "func", None) or substrate
    doc = inner.__doc__ or ""
    assert doc, "an empty docstring could not have confirmed anything"
    for label, bounds in (
        ("B", BATCH_RANGE),
        ("C_in", C_IN_RANGE),
        ("C_out", C_OUT_RANGE),
    ):
        needle = f"{label}: {bounds[0]}-{bounds[1]}"
        assert needle in doc, (
            f"this module transcribes {needle!r}, which the kernel's own docstring "
            f"does not say"
        )
