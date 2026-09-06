# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ``inc-glm53f-056`` -- WP10: vision config and preprocessing.

The declared acceptance (increment plan revision 232, design-entry ruling ``design-20260905-aq`` (iv)),
in its own words:

    3/3 image sizes -- the bridge's ``image_grid_thw`` and ``pixel_values`` row count equal BOTH the HF
    processor called directly as the oracle AND the closed form stated in the predictions: regime A (plain)
    at the non-multiple 784x1036; regime B (below the floor, upscaled) at 29x29 -> [1, 8, 8], 64 rows, NOT
    the plain form's 4x4; regime C (above the ceiling, downscaled by the processor's search) at one size the
    predictions name with its number; 2/2 video frame counts with ``do_sample_frames`` PINNED False:
    ``grid_t = ceil(F / 2)``; 0 inputs silently resize -- every resample the HF path performs is REPORTED by
    the bridge as a per-item record, with regimes B and C as the firing controls. Tier T exactness
    throughout: integer and index equality, no numeric pair authored.

The conjuncts are measured as C01 (3/3 images), C02 (2/2 videos), C03 (0 unreported resamples), C04 (the
registration, asserted positively) and C05 (the field config). C00 guards the import origin and C06 reports
every reading so no number here is silent.

WHERE THE EXPECTED NUMBERS COME FROM. Every value in ``REGISTERED_IMAGE_CASES`` and
``REGISTERED_VIDEO_CASES`` was computed from the transformers source and written into
``predictions-056-build.txt`` BEFORE this file existed, so this test cannot be a transcription of its own
answer. The two facts that make regime B's answer 8 and not 4 or 12 are worth stating: the image path tells
the resizer there are ``temporal_patch_size`` frames rather than one, and the below-floor scale for 29x29 is
``sqrt(25088 / (2 * 29 * 29)) = 112 / 29`` exactly.

HOW A SILENT RESAMPLE IS CAUGHT (C03). Nothing in the processor's output says whether the pixels were scaled
or merely padded, so C03 does not ask it to. It runs the same geometry twice over two different constant
inputs: every element the two runs agree on is padding, because padding does not depend on the content, and
every element they differ on is content. That count is an integer, needs no authored tolerance and no
authored pad value, and it separates "scaled to fill the canvas" from "padded with zeros" exactly.

THE PADDING CONTROL IS DELIBERATELY OUTSIDE THE REGISTERED SET. All three registered image sizes happen to
be multiples of the 28-pixel canvas alignment, so all three pad by zero and could not tell a working
pad-detector from a broken one. ``PADDING_CONTROL`` at 800x1000 is added as extra coverage -- it changes no
registered criterion -- purely so the instrument in C03 is shown to report a non-zero answer when there is
one to report.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch

from vllm_neuron.model.glm5_next.utils.vision_preprocessing import (
    REGIME_ABOVE_CEILING,
    REGIME_BELOW_FLOOR,
    REGIME_PLAIN,
    Glm5NextDummyInputsBuilder,
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
    GridConstants,
    build_mm_fields_config,
    describe_resample,
    image_grid_spec,
    video_grid_spec,
)

# ---------------------------------------------------------------------------
# The registered cases. (height, width, regime, grid_thw, patch rows, merged tokens, resampled)
# ---------------------------------------------------------------------------
REGISTERED_IMAGE_CASES = (
    ("B01_regime_A_plain", 784, 1036, REGIME_PLAIN, (1, 56, 74), 4144, 1036, False),
    ("B02_regime_B_below_floor", 29, 29, REGIME_BELOW_FLOOR, (1, 8, 8), 64, 16, True),
    ("B03_regime_C_above_ceiling", 2800, 2800, REGIME_ABOVE_CEILING, (1, 178, 178), 31684, 7921, True),
)

# (num_frames, height, width, grid_thw, patch rows, merged tokens)
REGISTERED_VIDEO_CASES = (
    ("B04_three_frames", 3, 112, 112, (2, 8, 8), 128, 32),
    ("B05_eight_frames", 8, 112, 112, (4, 8, 8), 256, 64),
)

#: Extra coverage, not a registered criterion: the one image size here that is NOT a multiple of the canvas
#: alignment, so the pad detector in C03 has something non-zero to find.
PADDING_CONTROL = ("B06_padding_control", 800, 1000)

#: The two constant pixel values C03 runs the same geometry over. Any two distinct values work; these are
#: mid-range so neither can be confused with the zero padding.
CONTENT_VALUE_A = 200
CONTENT_VALUE_B = 100


# ---------------------------------------------------------------------------
# Fixtures. Both oracles are the transformers classes themselves, constructed with no arguments: this test
# needs no checkpoint, no download and no network.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def glm5_next_pkg():
    import transformers.models.glm5_next as pkg

    return pkg


@pytest.fixture(scope="module")
def image_oracle(glm5_next_pkg):
    return glm5_next_pkg.Glm5NextImageProcessor()


@pytest.fixture(scope="module")
def video_oracle(glm5_next_pkg):
    return glm5_next_pkg.Glm5NextVideoProcessor()


@pytest.fixture(scope="module")
def image_consts(image_oracle):
    return GridConstants.from_processor(image_oracle)


@pytest.fixture(scope="module")
def video_consts(video_oracle):
    return GridConstants.from_processor(video_oracle)


def _run_image_oracle(oracle, height, width, value=CONTENT_VALUE_A):
    image = torch.full((3, height, width), value, dtype=torch.uint8)
    return oracle(images=[image], return_tensors="pt")


def _run_video_oracle(oracle, num_frames, height, width, value=CONTENT_VALUE_A):
    video = torch.full((num_frames, 3, height, width), value, dtype=torch.uint8)
    return oracle(videos=[video], do_sample_frames=False, return_tensors="pt")


# ---------------------------------------------------------------------------
# C00 -- the module under test is the candidate, not the venv's editable checkout.
# ---------------------------------------------------------------------------
def test_c00_the_module_under_test_is_the_scratch_candidate():
    """The import must resolve under the tree this run is measuring.

    An editable install can point at a different checkout than the one the tests live in, and importing vllm
    already imports this plugin through its platform entry point, so which tree answers ``import
    vllm_neuron`` is a property of the run rather than of the source. This names that tree and checks it.

    Two arms, and never a skip, because a skip here would hide exactly the mix-up it is looking for:

    * ``GLM53F_CANDIDATE_ROOT`` set -- the runner declared which tree it meant, and the import must resolve
      under that declaration. This is the arm the campaign's own acceptance runs.
    * unset -- a plain ``pytest`` from a checkout, which is how a maintainer, CI or a reviewer's clone runs
      this suite. The tree is then derived from this file's own location, so the run is still pinned to a
      named directory instead of passing whatever the interpreter happened to import.
    """
    import vllm_neuron

    resolved = Path(vllm_neuron.__file__).resolve()
    declared_root = os.environ.get("GLM53F_CANDIDATE_ROOT")
    if declared_root:
        root = Path(declared_root).resolve()
        source = "GLM53F_CANDIDATE_ROOT"
    else:
        # test/vllm_neuron/model/glm5_next/<this file> -- four parents up is the repository root.
        root = Path(__file__).resolve().parents[4]
        source = "this test file's own location"
    # The marker is pyproject.toml and NOT the package directory: the test tree mirrors the package, so
    # test/vllm_neuron/ is also a directory called vllm_neuron and a package-only marker would accept the
    # test directory as a repository root and hide an off-by-one in the derivation above.
    assert (root / "pyproject.toml").is_file() and (root / "vllm_neuron" / "__init__.py").is_file(), (
        f"{root}, taken from {source}, is not a repository root -- it holds no pyproject.toml beside a "
        "vllm_neuron package, so the check below would be asserting something about the wrong directory."
    )
    assert root in resolved.parents, (
        f"vllm_neuron resolved to {resolved}, which is not under {root} (taken from {source}). Some other "
        "checkout answered the import, so this run measured a tree it did not mean to."
    )


# ---------------------------------------------------------------------------
# C01 -- 3/3 image sizes, closed form and oracle and registered number all equal.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,height,width,regime,grid,rows,tokens,resampled", REGISTERED_IMAGE_CASES,
    ids=[c[0] for c in REGISTERED_IMAGE_CASES],
)
def test_c01_image_grid_matches_oracle_and_closed_form(
    image_oracle, image_consts, name, height, width, regime, grid, rows, tokens, resampled
):
    spec = image_grid_spec(image_consts, height, width)
    out = _run_image_oracle(image_oracle, height, width)

    oracle_grid = tuple(int(v) for v in out["image_grid_thw"][0].tolist())
    oracle_rows = int(out["pixel_values"].shape[0])

    assert spec.thw == grid, f"{name}: the bridge's grid {spec.thw} is not the registered {grid}"
    assert oracle_grid == grid, f"{name}: the oracle's grid {oracle_grid} is not the registered {grid}"
    assert spec.num_patch_rows == rows
    assert oracle_rows == rows, f"{name}: the oracle produced {oracle_rows} rows, registered {rows}"
    assert spec.num_merged_tokens == tokens
    assert spec.regime == regime, f"{name}: the bridge called it regime {spec.regime}, registered {regime}"

    record = describe_resample(
        image_consts, modality="image", num_frames=image_consts.temporal_patch_size,
        height=height, width=width,
    )
    assert record.resampled is resampled, f"{name}: {record.describe()}"


def test_c01_the_registered_regimes_are_three_different_ones():
    """The 3/3 would be worth little if all three sizes took the same branch."""
    regimes = {case[3] for case in REGISTERED_IMAGE_CASES}
    assert regimes == {REGIME_PLAIN, REGIME_BELOW_FLOOR, REGIME_ABOVE_CEILING}


def test_c01_regime_b_is_not_the_plain_answer(image_consts):
    """29x29 must not produce the 4x4 a plain reading of ceil(29/28) gives, nor the 12x12 a one-frame
    reading of the below-floor branch gives. Both wrong answers are named so the right one is not a
    coincidence."""
    spec = image_grid_spec(image_consts, 29, 29)
    assert spec.grid_h == 8 and spec.grid_w == 8
    assert (spec.grid_h, spec.grid_w) != (4, 4)
    assert (spec.grid_h, spec.grid_w) != (12, 12)


def test_c01_regime_c_honours_the_ceiling_it_was_cut_to(image_consts):
    """The cut-down canvas must sit under the token ceiling, and one alignment step more must not."""
    spec = image_grid_spec(image_consts, 2800, 2800)
    assert spec.num_merged_tokens <= image_consts.max_image_tokens
    one_step_more = spec.canvas_height + image_consts.factor
    refused = image_consts.temporal_patch_size * one_step_more * one_step_more
    assert refused > image_consts.ceiling_pixels


# ---------------------------------------------------------------------------
# C02 -- 2/2 video frame counts, do_sample_frames pinned False.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,num_frames,height,width,grid,rows,tokens", REGISTERED_VIDEO_CASES,
    ids=[c[0] for c in REGISTERED_VIDEO_CASES],
)
def test_c02_video_grid_matches_oracle_and_closed_form(
    video_oracle, video_consts, name, num_frames, height, width, grid, rows, tokens
):
    spec = video_grid_spec(video_consts, num_frames, height, width)
    out = _run_video_oracle(video_oracle, num_frames, height, width)

    oracle_grid = tuple(int(v) for v in out["video_grid_thw"][0].tolist())
    oracle_rows = int(out["pixel_values_videos"].shape[0])

    assert spec.thw == grid, f"{name}: the bridge's grid {spec.thw} is not the registered {grid}"
    assert oracle_grid == grid, f"{name}: the oracle's grid {oracle_grid} is not the registered {grid}"
    assert spec.num_patch_rows == rows
    assert oracle_rows == rows
    assert spec.num_merged_tokens == tokens


def test_c02_the_temporal_element_is_padded_frame_pairs(video_consts):
    """grid_t is ceil(F / temporal_patch_size), so an odd frame count rounds up rather than down."""
    for num_frames in (1, 2, 3, 4, 7, 8):
        spec = video_grid_spec(video_consts, num_frames, 112, 112)
        assert spec.grid_t == math.ceil(num_frames / video_consts.temporal_patch_size)
    assert video_grid_spec(video_consts, 3, 112, 112).grid_t != 3


def test_c02_the_video_ceiling_differs_from_the_image_ceiling(image_consts, video_consts):
    """A reading that used the image ceiling for video would be a value under the wrong condition, so the
    difference is measured rather than assumed."""
    assert video_consts.max_image_tokens != image_consts.max_image_tokens
    assert video_consts.temporal_patch_size == image_consts.temporal_patch_size


# ---------------------------------------------------------------------------
# C03 -- 0 inputs silently resize.
# ---------------------------------------------------------------------------
def _pad_and_content_elements(run_a, run_b, key):
    """Elements the two runs agree on are padding; the rest is content."""
    a, b = run_a[key], run_b[key]
    assert a.shape == b.shape, "the two runs must share their geometry"
    agree = int(torch.eq(a, b).sum())
    differ = int(a.numel() - agree)
    return agree, differ


@pytest.mark.parametrize(
    "name,height,width",
    [(c[0], c[1], c[2]) for c in REGISTERED_IMAGE_CASES] + [PADDING_CONTROL],
    ids=[c[0] for c in REGISTERED_IMAGE_CASES] + [PADDING_CONTROL[0]],
)
def test_c03_every_image_resample_is_reported(image_oracle, image_consts, name, height, width):
    record = describe_resample(
        image_consts, modality="image", num_frames=image_consts.temporal_patch_size,
        height=height, width=width,
    )
    run_a = _run_image_oracle(image_oracle, height, width, CONTENT_VALUE_A)
    run_b = _run_image_oracle(image_oracle, height, width, CONTENT_VALUE_B)
    pad_elements, content_elements = _pad_and_content_elements(run_a, run_b, "pixel_values")

    per_pixel = 3 * image_consts.temporal_patch_size
    canvas_area = record.canvas_height * record.canvas_width
    content_area = record.content_height * record.content_width

    assert content_elements == content_area * per_pixel, (
        f"{name}: the oracle's content covers {content_elements // per_pixel} pixels but the record says "
        f"{content_area}. {record.describe()}"
    )
    assert pad_elements == (canvas_area - content_area) * per_pixel, f"{name}: {record.describe()}"


@pytest.mark.parametrize(
    "name,num_frames,height,width",
    [(c[0], c[1], c[2], c[3]) for c in REGISTERED_VIDEO_CASES],
    ids=[c[0] for c in REGISTERED_VIDEO_CASES],
)
def test_c03_every_video_resample_is_reported(
    video_oracle, video_consts, name, num_frames, height, width
):
    record = describe_resample(
        video_consts, modality="video", num_frames=num_frames, height=height, width=width
    )
    spec = video_grid_spec(video_consts, num_frames, height, width)
    run_a = _run_video_oracle(video_oracle, num_frames, height, width, CONTENT_VALUE_A)
    run_b = _run_video_oracle(video_oracle, num_frames, height, width, CONTENT_VALUE_B)
    pad_elements, content_elements = _pad_and_content_elements(run_a, run_b, "pixel_values_videos")

    padded_frames = spec.grid_t * video_consts.temporal_patch_size
    per_pixel = 3 * padded_frames
    canvas_area = record.canvas_height * record.canvas_width
    content_area = record.content_height * record.content_width

    assert content_elements == content_area * per_pixel, f"{name}: {record.describe()}"
    assert pad_elements == (canvas_area - content_area) * per_pixel, f"{name}: {record.describe()}"


def test_c03_the_reporter_says_true_and_false_and_the_detector_finds_padding(image_oracle, image_consts):
    """The firing controls the acceptance names, in one place.

    A reporter that always said False would make the zero above worthless, and a pad detector that always
    answered zero would make the pad assertions worthless. Regimes B and C report True, regime A reports
    False, and the padding control produces a non-zero pad count.
    """
    flags = {}
    for name, height, width, _regime, _grid, _rows, _tokens, _resampled in REGISTERED_IMAGE_CASES:
        flags[name] = describe_resample(
            image_consts, modality="image", num_frames=image_consts.temporal_patch_size,
            height=height, width=width,
        ).resampled
    assert sorted(flags.values()) == [False, True, True], flags

    _name, height, width = PADDING_CONTROL
    record = describe_resample(
        image_consts, modality="image", num_frames=image_consts.temporal_patch_size,
        height=height, width=width,
    )
    assert record.resampled is False
    assert record.pad_bottom > 0 and record.pad_right > 0
    run_a = _run_image_oracle(image_oracle, height, width, CONTENT_VALUE_A)
    run_b = _run_image_oracle(image_oracle, height, width, CONTENT_VALUE_B)
    pad_elements, _content = _pad_and_content_elements(run_a, run_b, "pixel_values")
    assert pad_elements > 0, "the pad detector never reports padding, so its zeros mean nothing"


# ---------------------------------------------------------------------------
# C04 -- the registration, asserted positively.
# ---------------------------------------------------------------------------
def test_c04_the_model_class_carries_this_blocks_processor_factories():
    """A missing registration is SILENT: vLLM catches the unregistered-processor error, logs once and
    serves the model text-only with images dropped. So this asserts the attribute is present and that its
    three factories are this module's own classes, by identity."""
    from vllm_neuron.model.glm5_next import Glm5NextForConditionalGeneration

    factories = getattr(Glm5NextForConditionalGeneration, "_processor_factory", None)
    assert factories is not None, (
        "Glm5NextForConditionalGeneration carries no _processor_factory, so vLLM would treat this "
        "architecture as text-only and drop every image without raising."
    )
    assert factories.processor is Glm5NextMultiModalProcessor
    assert factories.info is Glm5NextProcessingInfo
    assert factories.dummy_inputs is Glm5NextDummyInputsBuilder


def test_c04_an_unregistered_class_carries_no_factories():
    """The firing control for the assertion above."""

    class NotRegistered:
        pass

    assert getattr(NotRegistered, "_processor_factory", None) is None


def test_c04_this_block_declares_no_model_side_protocol():
    """RG-20 owns the SupportsMultiModal declaration on the model class, so this block must not have added
    one. The test states the boundary rather than leaving it to a reviewer's memory."""
    from vllm_neuron.model.glm5_next import Glm5NextForConditionalGeneration

    assert getattr(Glm5NextForConditionalGeneration, "supports_multimodal", False) is not True


# ---------------------------------------------------------------------------
# C05 -- the field config slices the oracle's real output correctly.
# ---------------------------------------------------------------------------
def test_c05_the_field_config_sizes_match_the_oracle_row_counts(image_oracle, image_consts):
    out = _run_image_oracle(image_oracle, 784, 1036)
    fields = build_mm_fields_config(out, image_consts.merge_size)

    assert set(fields) == {
        "pixel_values",
        "image_embeds",
        "image_grid_thw",
        "pixel_values_videos",
        "video_embeds",
        "video_grid_thw",
    }
    grid = out["image_grid_thw"]
    rows = int(grid.prod(-1).sum())
    assert rows == int(out["pixel_values"].shape[0])
    assert int((grid.prod(-1) // image_consts.merge_length).sum()) == rows // image_consts.merge_length


def test_c05_an_absent_modality_sizes_to_nothing(image_consts):
    """A text-only or image-only batch must not make the video fields claim rows that do not exist."""
    fields = build_mm_fields_config({}, image_consts.merge_size)
    assert set(fields) >= {"pixel_values", "video_grid_thw"}


def test_c05_the_processor_declares_the_two_hooks_vllm_requires():
    """Both are abstract on vLLM's base, so a triple missing either could not be instantiated."""
    for name in ("_get_mm_fields_config", "_get_prompt_updates"):
        assert name in vars(Glm5NextMultiModalProcessor), f"{name} is not overridden"


def test_c05_image_limits_are_open_and_video_is_not_offered_yet():
    """The limits this bridge declares, stated as a test so a later change is deliberate.

    Called unbound with ``None`` for ``self`` on purpose: the method reads nothing off the instance, so the
    declaration can be checked without a vLLM processing context.
    """
    limits = Glm5NextProcessingInfo.get_supported_mm_limits(None)
    assert limits["image"] is None
    assert "video" not in limits


# ---------------------------------------------------------------------------
# C06 -- report every reading, so no number above is silent.
# ---------------------------------------------------------------------------
def test_c06_report_the_measured_readings(image_consts, video_consts, image_oracle, capsys):
    lines = [
        "READINGS for inc-glm53f-056, all read off the constructed processors:",
        f"  image  patch_size={image_consts.patch_size} merge_size={image_consts.merge_size} "
        f"temporal_patch_size={image_consts.temporal_patch_size} "
        f"patch_expand_factor={image_consts.patch_expand_factor}",
        f"  image  min_image_tokens={image_consts.min_image_tokens} "
        f"max_image_tokens={image_consts.max_image_tokens}",
        f"  video  max_image_tokens={video_consts.max_image_tokens}",
        f"  derived factor={image_consts.factor} pixels_per_token={image_consts.pixels_per_token} "
        f"floor_px={image_consts.floor_pixels} ceiling_px={image_consts.ceiling_pixels}",
    ]
    for name, height, width, regime, grid, rows, tokens, resampled in REGISTERED_IMAGE_CASES:
        spec = image_grid_spec(image_consts, height, width)
        record = describe_resample(
            image_consts, modality="image", num_frames=image_consts.temporal_patch_size,
            height=height, width=width,
        )
        lines.append(
            f"  {name}: {height}x{width} -> grid {spec.thw} rows {spec.num_patch_rows} "
            f"tokens {spec.num_merged_tokens} regime {spec.regime} | {record.describe()}"
        )
        assert (spec.thw, spec.num_patch_rows, spec.num_merged_tokens, spec.regime, record.resampled) == (
            grid, rows, tokens, regime, resampled
        )
    for name, num_frames, height, width, grid, rows, tokens in REGISTERED_VIDEO_CASES:
        spec = video_grid_spec(video_consts, num_frames, height, width)
        lines.append(
            f"  {name}: F={num_frames} {height}x{width} -> grid {spec.thw} rows {spec.num_patch_rows} "
            f"tokens {spec.num_merged_tokens} regime {spec.regime}"
        )
        assert (spec.thw, spec.num_patch_rows, spec.num_merged_tokens) == (grid, rows, tokens)

    name, height, width = PADDING_CONTROL
    record = describe_resample(
        image_consts, modality="image", num_frames=image_consts.temporal_patch_size,
        height=height, width=width,
    )
    lines.append(f"  {name} (extra coverage, not registered): {record.describe()}")

    lines.append(f"  padding-control pad = {record.pad_bottom} bottom, {record.pad_right} right")
    with capsys.disabled():
        print("\n" + "\n".join(lines))
