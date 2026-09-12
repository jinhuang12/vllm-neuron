# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ``inc-glm53f-056`` -- WP10: vision config and preprocessing.

The declared acceptance (increment plan revision 245, design-entry ruling ``design-20260905-aq`` (iv)),
in its own words:

WHAT THE REVISION CITE MEANS. It names the plan revision these tests were DERIVED AGAINST -- the revision
current when the run that last changed them started. It is not a claim that the plan has not moved since, and
it is not a version of this file. It is refreshed on any lap that touches this file, so a reader comparing the
tests to the plan knows which plan text to open. The acceptance quoted below is the ruling's own wording and
does not move with the cite.

    3/3 image sizes -- the bridge's ``image_grid_thw`` and ``pixel_values`` row count equal BOTH the HF
    processor called directly as the oracle AND the closed form stated in the predictions: regime A (plain)
    at the non-multiple 784x1036; regime B (below the floor, upscaled) at 29x29 -> [1, 8, 8], 64 rows, NOT
    the plain form's 4x4; regime C (above the ceiling, downscaled by the processor's search) at one size the
    predictions name with its number; 2/2 video frame counts with ``do_sample_frames`` PINNED False:
    ``grid_t = ceil(F / 2)``; 0 inputs silently resize -- every resample the HF path performs is REPORTED by
    the bridge as a per-item record, with regimes B and C as the firing controls. Tier T exactness
    throughout: integer and index equality, no numeric pair authored.

WHY THE QUOTE ABOVE STILL SAYS 2/2 AND THIS FILE MEASURES 5/5. The block above is ruling
``design-20260905-aq`` (iv) in its own words at plan revision 245, and it is left exactly as the ruling wrote
it -- editing a quotation to match a later count would make the file report that the ruling said something it
did not. Increment-plan revision 262 raised this table's registered count to 5/5 as part of
``inc-glm53f-061``: three rows were added to ``REGISTERED_VIDEO_CASES`` (F=1 and F=2 at 112x112, and F=1 at
60x40 below the floor), so C02 now runs five cases over four distinct frame counts instead of two cases over
two. The ruling's other numbers did not move. Everything below this line is this file's own prose and counts
the cases as they now stand.

The conjuncts are measured as C01 (3/3 images), C02 (5/5 videos), C03 (0 unreported resamples), C04 (the
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
    SMART_RESIZE_MODULE,
    SMART_RESIZE_NAME,
    Glm5NextDummyInputsBuilder,
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
    GridConstants,
    build_mm_fields_config,
    describe_resample,
    image_grid_spec,
    require_transformers_smart_resize,
    resolve_canvas,
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
#
# B07-B09 were added by ``inc-glm53f-061`` at increment-plan revision 262, which raised this table's
# registered count from 2/2 to 5/5. B04 and B05 are unchanged. The reason for the widening: with only two
# three-frames-or-more rows at one size, the video path was oracle-pinned at no clamp bound and at neither
# F=1 nor F=2 -- and F=1 is the exact frame count ``-061``'s cross-modality reading depends on. Every value
# in the three new rows was computed from upstream's own ``smart_resize`` text and printed in
# ``derive-061-registered-values-r1.out`` part 4, whose bytes are
# ``cc3b82bdcc9bc53198a17b2292e2b6a8cf92df53e8cca43ccf24af90da59afa6``; B04 and B05 are reproduced by that
# same derivation as its own check, so the three new rows and the two landed ones come from one rule.
REGISTERED_VIDEO_CASES = (
    ("B04_three_frames", 3, 112, 112, (2, 8, 8), 128, 32),
    ("B05_eight_frames", 8, 112, 112, (4, 8, 8), 256, 64),
    ("B07_one_frame", 1, 112, 112, (1, 8, 8), 64, 16),
    ("B08_two_frames", 2, 112, 112, (1, 8, 8), 64, 16),
    ("B09_one_frame_below_floor", 1, 60, 40, (1, 14, 10), 140, 35),
)

#: Extra coverage, not a registered criterion: the one image size here that is NOT a multiple of the canvas
#: alignment, so the pad detector in C03 has something non-zero to find.
PADDING_CONTROL = ("B06_padding_control", 800, 1000)

#: The two constant pixel values C03 runs the same geometry over. Any two distinct values work; these are
#: mid-range so neither can be confused with the zero padding.
CONTENT_VALUE_A = 200
CONTENT_VALUE_B = 100

# ---------------------------------------------------------------------------
# The R2 grid (ruling design-20260906-au). Every integer below was READ from
# ``derive-056r2-divergence.out``, the record filed with this repair, and is quoted here by case name rather
# than recomputed, so this file cannot become a transcription of its own answer.
#
# (case name, height, width, regime, count at patch_expand_factor 1, count at 2, HF's public counter)
#
# HF's public counter appears once, not twice, because it drops ``patch_expand_factor`` -- it answers the same
# number whatever the factor is, and that is the whole point of the divergence item below.
# ---------------------------------------------------------------------------
R2_CASES = (
    ("aligned_to_both_factors", 448, 448, REGIME_PLAIN, 1024, 1024, 1024),
    ("aligned_to_28_only", 420, 336, REGIME_PLAIN, 720, 768, 720),
    ("tiny_floor_bound", 60, 40, REGIME_BELOW_FLOOR, 80, 320, 80),
    ("tiny_square", 32, 32, REGIME_BELOW_FLOOR, 64, 256, 64),
    ("odd_mid_size", 500, 333, REGIME_PLAIN, 864, 864, 864),
    ("huge_ceiling_bound", 8000, 6000, REGIME_ABOVE_CEILING, 31724, 126896, 31724),
    ("wide_strip", 1400, 56, REGIME_PLAIN, 400, 400, 400),
)

#: The one R2 case whose real-pixel arm is skipped: an 8000x6000 uint8 image is 144 million pixels and
#: resampling it on CPU costs more than the reading is worth. Its ARITHMETIC arm still runs, and regime C is
#: already measured against the real processor by B03 at 2800x2800.
R2_TOO_BIG_FOR_REAL_PIXELS = "huge_ceiling_bound"


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
# C02 -- 5/5 video cases (plan rev 262; was 2/2), do_sample_frames pinned False.
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


# ---------------------------------------------------------------------------
# The R2 items (ruling design-20260906-au). Two things are measured here that C01-C06 could not measure: that
# the canvas arithmetic is CONSUMED from transformers rather than re-derived, and that the vLLM bridge has a
# failing path at all -- before R2 the four bridge methods were only checked by name.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def offline_tokenizer():
    """A real tokenizer, built in memory, because the HF processor refuses a duck type.

    ``Glm5NextProcessor.__init__`` type-checks its tokenizer argument and rejects anything that is not a
    ``PreTrainedTokenizerBase``, so the bridge cannot be exercised with a stub. This builds the smallest real
    fast tokenizer that round-trips the tokens the bridge cares about, with no checkpoint and no network: the
    fixture directory carries no tokenizer files, and downloading one would make this suite need a network.
    """
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = [
        "text",
        "<|image|>",
        "<|video|>",
        "<|begin_of_video|>",
        "<|end_of_video|>",
        "<|begin_of_image|>",
        "<|end_of_image|>",
    ]
    vocab = {word: index for index, word in enumerate(words)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="text"))
    backend.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(r"<\|[a-z_]+\|>|[a-z]+"), behavior="isolated"
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="text")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [word for word in words if word.startswith("<|")]}
    )
    tokenizer.image_token = "<|image|>"
    tokenizer.image_token_id = vocab["<|image|>"]
    tokenizer.video_token = "<|video|>"
    tokenizer.video_token_id = vocab["<|video|>"]
    return tokenizer


@pytest.fixture(scope="module")
def hf_processor(glm5_next_pkg, offline_tokenizer):
    """The real transformers processor, holding the real image and video sub-processors."""
    return glm5_next_pkg.Glm5NextProcessor(
        image_processor=glm5_next_pkg.Glm5NextImageProcessor(),
        tokenizer=offline_tokenizer,
        video_processor=glm5_next_pkg.Glm5NextVideoProcessor(),
    )


@pytest.fixture
def bridge(hf_processor, offline_tokenizer, monkeypatch):
    """The bridge under test, wired to the real transformers processor.

    vLLM resolves an HF processor from a model config and a checkpoint on disk, which this suite does not have.
    The ONE seam is therefore vLLM's own ``BaseProcessingInfo.get_hf_processor``, replaced by the real processor
    above. No fork method is replaced: the bridge methods below run their own code, and every grid they read is
    produced by transformers.
    """
    from vllm.config.multimodal import MultiModalConfig
    from vllm.multimodal.processing import context as vllm_context

    multimodal_config = MultiModalConfig()

    class _OfflineModelConfig:
        model = "glm5-next-offline-fixture"
        tokenizer = model
        trust_remote_code = False
        hf_config = None
        dtype = torch.float32

        def get_multimodal_config(self):
            return multimodal_config

    _OfflineModelConfig.multimodal_config = multimodal_config
    monkeypatch.setattr(
        vllm_context.BaseProcessingInfo,
        "get_hf_processor",
        lambda self, **kwargs: hf_processor,
    )
    ctx = vllm_context.InputProcessingContext(_OfflineModelConfig(), offline_tokenizer)
    info = Glm5NextProcessingInfo(ctx)
    return info, Glm5NextMultiModalProcessor(info, Glm5NextDummyInputsBuilder(info))


def test_m01_the_consumed_canvas_function_resolves_and_refuses_by_name(glm5_next_pkg, monkeypatch):
    """The canvas arithmetic is imported from transformers, and the import is guarded by a named refusal.

    The function is not on the package surface -- that is measured here rather than assumed, because it is the
    whole reason the fork names a private submodule -- so the refusal is what turns a moved private name into a
    message naming the module, the symbol and the remedy, instead of an ``AttributeError`` from inside a grid
    computation. The planted control deletes the symbol and checks the refusal actually fires: without it, a
    refusal that could never trigger would read exactly like a working one.
    """
    import importlib

    assert not hasattr(glm5_next_pkg, SMART_RESIZE_NAME), (
        f"transformers.models.glm5_next now exposes {SMART_RESIZE_NAME} on its package surface. The fork can "
        "stop naming the private submodule -- update the helper and delete this assertion's reason."
    )

    module = importlib.import_module(SMART_RESIZE_MODULE)
    resolved = require_transformers_smart_resize()
    assert resolved is getattr(module, SMART_RESIZE_NAME)
    assert callable(resolved)

    monkeypatch.delattr(module, SMART_RESIZE_NAME)
    with pytest.raises(RuntimeError) as excinfo:
        require_transformers_smart_resize()
    message = str(excinfo.value)
    assert SMART_RESIZE_MODULE in message and SMART_RESIZE_NAME in message, message


@pytest.mark.parametrize("case", R2_CASES, ids=[case[0] for case in R2_CASES])
def test_m02_the_consumed_canvas_and_the_fork_count_agree_with_transformers(
    image_oracle, image_consts, case
):
    """At this checkpoint's factor the fork's count equals transformers' own, on all seven sizes.

    Two oracles, because they answer different questions. ``get_number_of_image_patches`` is arithmetic and
    runs on every case including the huge one. The real processor is pixels, and its ``image_grid_thw`` and
    row count are what the served path actually produces.
    """
    name, height, width, _regime, count_at_factor_1, _count_at_factor_2, hf_count = case
    assert image_consts.patch_expand_factor == 1, (
        "this item states the agreeing case, which is patch_expand_factor 1; the divergence is m03"
    )

    spec = image_grid_spec(image_consts, height, width)
    assert spec.grid_h * spec.grid_w == count_at_factor_1, (
        f"{name}: the fork counts {spec.grid_h * spec.grid_w} patches where the filed derivation recorded "
        f"{count_at_factor_1}."
    )
    assert image_oracle.get_number_of_image_patches(height, width, {}) == hf_count
    assert spec.grid_h * spec.grid_w == image_oracle.get_number_of_image_patches(height, width, {})

    if name != R2_TOO_BIG_FOR_REAL_PIXELS:
        out = _run_image_oracle(image_oracle, height, width)
        assert out["image_grid_thw"].tolist() == [list(spec.thw)], name
        assert int(out["pixel_values"].shape[0]) == spec.num_patch_rows, name


def test_m03_the_fork_count_keeps_patch_expand_factor_where_transformers_drops_it(
    image_oracle, image_consts
):
    """The divergence, case by case, with the integers the filed derivation recorded.

    Transformers' public counter computes its alignment as ``patch_size * merge_size`` while its own
    preprocessing path multiplies by ``patch_expand_factor``. Consuming that counter would import the
    inconsistency, so the fork divides over the consumed canvas instead. This item is what makes that visible:
    at factor 2 the counter answers as though the factor were 1 -- it is handed the factor and ignores it --
    and four of the seven cases move while three do not. Both lists are asserted, so neither "they always
    differ" nor "they always agree" can satisfy this item.

    EVERY CASE IS PINNED WITH ITS REGIME, and the regime is read back from the fork's own answer rather than
    trusted from the table: the two moving-by-scale cases are regime B, the ceiling case is regime C, and the
    alignment-rounding case is regime A. A case that quietly changed regime would otherwise keep its integers
    and slide between the two lists unnoticed, which is the one way this item could go stale while passing.
    """
    doubled = GridConstants(
        patch_size=image_consts.patch_size,
        merge_size=image_consts.merge_size,
        temporal_patch_size=image_consts.temporal_patch_size,
        patch_expand_factor=2,
        min_image_tokens=image_consts.min_image_tokens,
        max_image_tokens=image_consts.max_image_tokens,
    )
    differing, agreeing = [], []
    for name, height, width, regime, _count_1, count_at_factor_2, hf_count in R2_CASES:
        spec = image_grid_spec(doubled, height, width)
        assert spec.regime == regime, (
            f"{name}: the filed derivation recorded regime {regime} at patch_expand_factor 2 and the fork now "
            f"answers {spec.regime}. A case that changed regime is not the case this item pinned."
        )
        assert spec.grid_h * spec.grid_w == count_at_factor_2, (
            f"{name} (regime {regime}): at patch_expand_factor 2 the fork counts "
            f"{spec.grid_h * spec.grid_w} where the filed derivation recorded {count_at_factor_2}."
        )
        assert (
            image_oracle.get_number_of_image_patches(height, width, {"patch_expand_factor": 2}) == hf_count
        ), (
            f"{name} (regime {regime}): transformers' counter changed its answer when handed the factor it "
            "used to ignore."
        )
        (differing if count_at_factor_2 != hf_count else agreeing).append((name, regime))

    assert differing == [
        ("aligned_to_28_only", REGIME_PLAIN),
        ("tiny_floor_bound", REGIME_BELOW_FLOOR),
        ("tiny_square", REGIME_BELOW_FLOOR),
        ("huge_ceiling_bound", REGIME_ABOVE_CEILING),
    ], differing
    assert agreeing == [
        ("aligned_to_both_factors", REGIME_PLAIN),
        ("odd_mid_size", REGIME_PLAIN),
        ("wide_strip", REGIME_PLAIN),
    ], agreeing


@pytest.mark.parametrize("case", R2_CASES, ids=[case[0] for case in R2_CASES])
def test_m04_the_regime_label_names_the_branch_transformers_would_take(image_consts, case):
    """The label the fork still owns, checked against this test's own statement of the rule.

    ``smart_resize`` returns a canvas and never says which branch produced it, so the label cannot be consumed
    and is pinned instead. The expected value is computed here, in the test, from the plainly aligned canvas
    and the two budgets -- which is what transformers compares against -- so the label is measured against a
    second statement of the rule rather than against the code that produces it.
    """
    name, height, width, regime, *_rest = case
    factor = image_consts.factor

    def rounded_up(value: int) -> int:
        return -(-value // factor) * factor

    plain_budget = image_consts.temporal_patch_size * rounded_up(height) * rounded_up(width)
    if plain_budget < image_consts.floor_pixels:
        expected = REGIME_BELOW_FLOOR
    elif plain_budget > image_consts.ceiling_pixels:
        expected = REGIME_ABOVE_CEILING
    else:
        expected = REGIME_PLAIN

    _canvas_height, _canvas_width, got = resolve_canvas(
        image_consts, num_frames=image_consts.temporal_patch_size, height=height, width=width
    )
    assert got == expected, f"{name}: plain budget {plain_budget} says {expected}, the fork says {got}"
    assert got == regime, f"{name}: the filed derivation recorded {regime}"


@pytest.mark.parametrize("num_frames", [1, 2, 3, 4, 5, 7, 8])
def test_m05_the_video_grid_frame_count_is_the_padding_transformers_patchifies(
    video_oracle, video_consts, num_frames
):
    """The video grid stays the fork's arithmetic, because transformers exposes no video counter, so it is
    pinned against the real video processor's own ``video_grid_thw`` across a frame sweep."""
    spec = video_grid_spec(video_consts, num_frames, 112, 112)
    out = _run_video_oracle(video_oracle, num_frames, 112, 112)
    assert out["video_grid_thw"].tolist() == [list(spec.thw)], f"F={num_frames}"


def test_m05_the_budget_rounding_and_the_grid_padding_are_different_rules(video_consts):
    """The firing control for the pair above: the two frame roundings must not be one function.

    The pixel budget rounds to the NEAREST whole temporal patch, as ``smart_resize`` does; the grid pads UP,
    as ``patchify`` does. At five frames they disagree, and a single shared helper would make this item fail.
    """
    padded = video_grid_spec(video_consts, 5, 112, 112).grid_t * video_consts.temporal_patch_size
    assert padded == 6
    assert video_consts.aligned_frames(5) == 4
    assert video_consts.aligned_frames(5) != padded


def test_m06_the_bridge_token_count_equals_the_processors_own_grid(bridge, image_oracle):
    """``get_num_image_tokens`` against the real processor's grid, which is what vLLM budgets against.

    A count that is too small truncates the placeholder run and the image is silently cropped; too large and
    vLLM raises on the mismatch. Neither had a failing path before this item.
    """
    info, _processor = bridge
    merge_length = image_oracle.merge_size**2
    for name, height, width, _regime, *_rest in R2_CASES:
        if name == R2_TOO_BIG_FOR_REAL_PIXELS:
            continue
        out = _run_image_oracle(image_oracle, height, width)
        expected = int(out["image_grid_thw"].prod()) // merge_length
        assert info.get_num_image_tokens(image_width=width, image_height=height) == expected, name


def test_m07_the_profiling_size_reaches_the_token_ceiling_and_is_not_a_square(bridge, image_oracle):
    """``get_image_size_with_most_features`` must reach the token ceiling EXACTLY, measured on the processor.

    vLLM profiles memory with this size and then refuses any request needing more tokens than the profiled run
    reserved for. A size that falls short therefore does not merely waste a little memory: the server rejects
    requests it has the capacity to serve.

    WHAT CHANGED AND WHY. The earlier version of this test asserted the size was SQUARE. That cannot be right.
    The ceiling bounds the token grid's AREA, and a square grid's area is a perfect square, so a ceiling that is
    not one is unreachable by any square whatsoever -- at this pin the ceiling is 8000, the best square is 89 by
    89, and 79 tokens were simply unreachable. The squareness assertion is gone and NO SHAPE RULE replaces it.
    The shape is whatever reaching the ceiling requires. Pinning a shape here is how the defect got in.

    EVERY READING COMES FROM THE PROCESSOR, AND THE TWO SIDES ARE INDEPENDENT. The expected values below are
    read off ``image_oracle``'s own attributes and its own ``image_grid_thw``; the answer under test comes from
    the fork. Neither side is derived from the other, so they can disagree.

    THE MESSAGES ARE PREFIXED WITH SHORT KEYS on purpose. The mutation battery gates each planted mutation on
    the SPECIFIC assertion it is supposed to redden, not merely on the test going red for some reason, and it
    needs a stable string to look for. Reading a key in a transcript is the difference between a control that
    fired and a control that fired for the reason claimed.

    COST. Three real-pixel runs, the largest 2240x2800. The regime-C case this suite already resamples is
    2800x2800, so nothing here is newly expensive.
    """
    info, _processor = bridge
    consts = info.get_grid_constants()
    frames = consts.temporal_patch_size
    merge_length = image_oracle.merge_size**2
    ceiling_tokens = image_oracle.max_image_tokens
    patch = image_oracle.patch_size

    size = info.get_image_size_with_most_features()

    def grid_of(height, width):
        return tuple(int(v) for v in _run_image_oracle(image_oracle, height, width)["image_grid_thw"][0])

    def grid_the_size_implies(height, width):
        return (1, height // patch, width // patch)

    # MAXIMALITY. The one assertion the pre-fix square search cannot satisfy: it returned 7921 tokens against a
    # ceiling of 8000. Read off the processor's own grid rather than any arithmetic in this repository.
    grid = grid_of(size.height, size.width)
    assert (grid[0] * grid[1] * grid[2]) // merge_length == ceiling_tokens, (
        f"m07 maximality: the profiling size reaches {(grid[0] * grid[1] * grid[2]) // merge_length} tokens "
        f"but the ceiling admits {ceiling_tokens}, so the server would refuse requests it could serve"
    )

    # And the fork's own counter, which is what vLLM budgets against, reads the same ceiling.
    assert info.get_num_image_tokens(image_width=size.width, image_height=size.height) == ceiling_tokens, (
        "m07 fork counter: the bridge's own token count for the profiling size disagrees with the ceiling"
    )

    # THE EQUALITY BOUNDARY, AS AN ACCEPTED-AND-CUT PAIR. This size's pixel budget lands exactly ON the ceiling,
    # and the processor cuts only when the budget is STRICTLY greater. So the size must come back as the canvas
    # it declared, and the next legal step must come back cut. Compared as GRIDS, never as token counts: one
    # step wider is cut back to this very canvas, so its token count is the ceiling too and a count comparison
    # would pass while measuring nothing.
    assert grid == grid_the_size_implies(size.height, size.width), (
        f"m07 accepted at the boundary: the processor returned {grid} for a size implying "
        f"{grid_the_size_implies(size.height, size.width)}, so the profiled canvas is not the declared one"
    )
    wider = size.width + consts.factor
    assert grid_of(size.height, wider) != grid_the_size_implies(size.height, wider), (
        "m07 one step wider is cut: the step past the profiling size was accepted whole, so the size below it "
        "is not at the ceiling's boundary"
    )

    # The inherited bounds and the alignment, unchanged. Both are algebraically implied by maximality at this
    # pin, because patch_expand_factor is 1 there and that makes factor equal the token unit, which in turn
    # makes the pixel budget a fixed multiple of the token count. They are kept because that coincidence is a
    # property of the pinned checkpoint and not of this code: on a checkpoint where the expand factor is not 1
    # the pixel bound carries something the token bound does not.
    assert frames * size.height * size.width <= consts.ceiling_pixels, (
        "m07 within the ceiling: the profiling size is over the pixel budget"
    )
    assert (
        frames * (size.height + consts.factor) * (size.width + consts.factor) > consts.ceiling_pixels
    ), "one more alignment step still fits, so this is not the largest size"
    assert size.height % consts.factor == 0 and size.width % consts.factor == 0


def test_m08_the_placeholder_replacement_is_the_items_own_token_count(bridge, image_oracle):
    """``_get_prompt_updates`` builds one replacement per image, sized by that image's own grid.

    The replacement callable is driven with the grid the real processor produced for one item. A wrong merge
    length or a grid read off the wrong item changes the length, and the length is asserted exactly.
    """
    from vllm.multimodal.parse import MultiModalDataParser

    info, processor = bridge
    height, width = 784, 1036
    out = _run_image_oracle(image_oracle, height, width)
    grid = out["image_grid_thw"]
    expected = int(grid.prod()) // image_oracle.merge_size**2

    class _Item:
        data = grid[0]

    mm_items = MultiModalDataParser().parse_mm_data(
        {"image": [torch.full((3, height, width), CONTENT_VALUE_A, dtype=torch.uint8)]}
    )
    updates = processor._get_prompt_updates(mm_items, {}, {"image": [{"image_grid_thw": _Item()}]})
    assert len(updates) == 1
    replacement = updates[0].replacement(0)
    assert len(replacement) == expected
    assert set(replacement) == {info.get_hf_processor().image_token_id}
    assert len(replacement) != int(grid.prod()), (
        "the replacement is as long as the PATCH count, so the 2x2 merge was never applied"
    )


def test_m09_apply_expands_one_image_to_its_own_token_run(bridge, image_oracle):
    """vLLM's own ``apply`` end to end, which is the path a served request takes.

    The fork overrides only the field config and the prompt updates, so this item is what shows the two of them
    compose correctly inside the base implementation: one image, one placeholder in the prompt, and a token run
    exactly as long as the grid says.
    """
    from vllm.multimodal.parse import MultiModalDataParser
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    info, processor = bridge
    height, width = 784, 1036
    image = torch.full((3, height, width), CONTENT_VALUE_A, dtype=torch.uint8)
    image_token = info.get_hf_processor().image_token
    image_token_id = info.get_hf_processor().image_token_id

    out = processor.apply(
        ProcessorInputs(
            prompt=f"text {image_token} text",
            mm_data_items=MultiModalDataParser().parse_mm_data({"image": [image]}),
            mm_uuid_items=None,
            hf_processor_mm_kwargs={},
            tokenization_kwargs={},
        ),
        TimingContext(),
    )

    oracle_grid = _run_image_oracle(image_oracle, height, width)["image_grid_thw"]
    expected_tokens = int(oracle_grid.prod()) // image_oracle.merge_size**2
    ids = out["prompt_token_ids"]
    assert sum(1 for token_id in ids if token_id == image_token_id) == expected_tokens
    assert out["mm_kwargs"]["image"][0]["image_grid_thw"].data.tolist() == oracle_grid[0].tolist()
    assert len(ids) > expected_tokens, "the text around the placeholder was dropped"
