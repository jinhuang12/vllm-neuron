# SPDX-License-Identifier: Apache-2.0
"""Tests for the video side of the vision preprocessing bridge.

An F-frame video packs to ``grid_t == ceil(F / temporal_patch_size)``, measured against the
real video processor with ``do_sample_frames`` pinned False. Inside its condition -- regime A
and at least ``floor_pixels`` raw pixels -- a 1-frame video produces the same tokens as the
same content as a still image, bit for bit; outside it the two paths differ, and each
divergence is paired with the same call made once more with one input changed, so the cause is
asserted rather than narrated.

Upstream's raw frame count reaches the canvas in two places, which is what the divergences are
made of: the below-floor rescue inside ``smart_resize`` scales by
``sqrt(min_pixels / (num_frames * h * w))`` while every other branch measures the budget over
the aligned frame count, and the resize tail one level up asks the image about
``temporal_patch_size * h * w`` and the video about its real frame count, so in a band of sizes
the image is padded with a zero border while the video is resampled to fill the same canvas.
The two token ceilings are a third, unrelated fact: 8000 tokens for images and 240000 for
videos, so a size between them is regime C on one path and regime A on the other.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from vllm_neuron.model.glm5_next.utils.vision_preprocessing import (
    REGIME_ABOVE_CEILING,
    REGIME_BELOW_FLOOR,
    REGIME_PLAIN,
    GridConstants,
    describe_resample,
    image_grid_spec,
    resolve_canvas,
    video_grid_spec,
)

# The frame counts under test. 1 and 2 are the counts a one-frame reading turns on, 3 pads up by
# one frame, 8 and 16 are exact, and 17 is a count where the pixel budget's rounding and the
# grid's padding give different answers.
FRAME_COUNTS = (1, 2, 3, 8, 16, 17)

# grid_t per frame count.
EXPECTED_GRID_T = {1: 1, 2: 1, 3: 2, 8: 4, 16: 8, 17: 9}

# The counts where the pixel budget's rounding and the grid's padding agree, and the counts
# where they differ, with both answers named: F -> (budget frame pairs, grid_t).
ROUNDINGS_AGREE_AT = (1, 2, 3, 8)
ROUNDINGS_DIFFER_AT = {5: (2, 3), 17: (8, 9)}

# The frame sweep's item size. 112x112 is regime A on this checkpoint, so its canvas does not
# move with the frame count and the sweep reads the frame factor alone.
FRAME_ARM_HEIGHT = 112
FRAME_ARM_WIDTH = 112

# The sizes both paths are compared on. (name, height, width, regime)
SIZE_CASES = (
    ("aligned_to_both_factors", 448, 448, REGIME_PLAIN),
    ("aligned_to_28_only", 420, 336, REGIME_PLAIN),
    ("tiny_floor_bound", 60, 40, REGIME_BELOW_FLOOR),
    ("tiny_square", 32, 32, REGIME_BELOW_FLOOR),
    ("odd_mid_size", 500, 333, REGIME_PLAIN),
    ("huge_ceiling_bound", 8000, 6000, REGIME_ABOVE_CEILING),
    ("wide_strip", 1400, 56, REGIME_PLAIN),
)

# The sizes where a 1-frame video is bit-identical to the still image. Named rather than
# filtered by regime, because the condition is regime A and at least ``floor_pixels`` raw
# pixels, and both halves are asserted per case.
EQUALITY_CASE_NAMES = (
    "aligned_to_both_factors",
    "aligned_to_28_only",
    "odd_mid_size",
    "wide_strip",
)

# Below the floor. (name, h, w, image canvas, video canvas at F=1). The counterfactual is the
# same video call told ``temporal_patch_size`` frames, which must land on the image canvas.
FLOOR_COUNTERFACTUAL_CASES = (
    ("tiny_floor_bound", 60, 40, (140, 112), (196, 140)),
    ("tiny_square", 32, 32, (112, 112), (168, 168)),
)

# Above the image ceiling and below the video ceiling: regime C against regime A. The
# counterfactual is the image canvas recomputed at the video ceiling, which must land on the
# video canvas and read regime A.
CEILING_CASE = ("huge_ceiling_bound", 8000, 6000)
CEILING_IMAGE_CANVAS = (2884, 2156)
CEILING_VIDEO_CANVAS = (8008, 6020)

# The band where both paths agree on the canvas, the grid and the row count and still differ in
# pixels: the image is padded with a zero border and the video is resampled to fill.
BAND_CASE = ("band_same_grid_different_pixels", 130, 130)
BAND_CANVAS = (140, 140)
BAND_GRID_HW = (10, 10)
BAND_PATCH_ROWS = 100
BAND_IMAGE_CONTENT = (130, 130)
BAND_VIDEO_CONTENT = (140, 140)

# Three frames at the first equality size. Two frames cannot tell the two paths apart, because
# two frames take grid_t 1 and the same canvas as one still image.
SHAPE_CONTROL_HEIGHT = 448
SHAPE_CONTROL_WIDTH = 448
SHAPE_CONTROL_FRAMES = 3
SHAPE_CONTROL_IMAGE_ROWS = 1024
SHAPE_CONTROL_VIDEO_ROWS = 2048

# (name, num_frames, height, width, grid_thw, patch rows, merged tokens) for the video path
# against the real video processor: two clamp-free sizes, F=1, F=2 and one below the floor.
VIDEO_ORACLE_CASES = (
    ("three_frames", 3, 112, 112, (2, 8, 8), 128, 32),
    ("eight_frames", 8, 112, 112, (4, 8, 8), 256, 64),
    ("one_frame", 1, 112, 112, (1, 8, 8), 64, 16),
    ("two_frames", 2, 112, 112, (1, 8, 8), 64, 16),
    ("one_frame_below_floor", 1, 60, 40, (1, 14, 10), 140, 35),
)

# A video row at the default video ceiling would need over 376 million pixels, so the ceiling is
# lowered on the processor instances instead and the size stays small. 64 tokens is 100352
# pixels, and 448x448 over two aligned frames is 401408, so both paths are above the ceiling and
# the processors' own budget search decides the canvas.
LOWERED_CEILING_TOKENS = 64
LOWERED_CEILING_HEIGHT = 448
LOWERED_CEILING_WIDTH = 448
LOWERED_CEILING_CANVAS = (224, 224)
LOWERED_CEILING_GRID_HW = (16, 16)
LOWERED_CEILING_PATCH_ROWS = 256

# The constant pixel value both paths are fed, and a second value for the content comparison.
# Mid-range, so neither can be confused with the zero padding.
CONTENT_VALUE_A = 200
CONTENT_VALUE_B = 100

# Bit-identical means exactly this, and no tolerance is authored anywhere in this file.
BIT_IDENTICAL_MAX_ABS_DIFF = 0.0


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


def _case(name):
    """One row of the size table by name, so a table reorder cannot change what a test measures."""
    for row in SIZE_CASES:
        if row[0] == name:
            return row
    raise AssertionError(f"{name} is not one of the sizes in SIZE_CASES")


def _run_image_oracle(oracle, height, width, value=CONTENT_VALUE_A):
    """One still image of constant content, through transformers' image processor."""
    image = torch.full((3, height, width), value, dtype=torch.uint8)
    return oracle(images=[image], return_tensors="pt")


def _run_video_oracle(oracle, num_frames, height, width, value=CONTENT_VALUE_A):
    """One video of constant content, with ``do_sample_frames`` pinned False.

    The video processor samples frames by default, which would move the frame count away from
    the one under test and make F an output rather than an input.
    """
    video = torch.full((num_frames, 3, height, width), value, dtype=torch.uint8)
    return oracle(videos=[video], do_sample_frames=False, return_tensors="pt")


def _closed_form_grid_t(num_frames: int, temporal_patch_size: int) -> int:
    """Pad F up to whole temporal patches, then divide. ``temporal_patch_size`` is read off the
    processor by the caller and never written as 2."""
    padded = num_frames + (-num_frames % temporal_patch_size)
    return padded // temporal_patch_size


def _budget_frame_pairs(consts: GridConstants, num_frames: int) -> int:
    """The frame pairs the PIXEL BUDGET counts, which rounds to the nearest whole patch."""
    return consts.aligned_frames(num_frames) // consts.temporal_patch_size


def _both_specs(image_consts, video_consts, height, width, num_frames):
    """The two paths' grids and resample records for one size, all four from the bridge."""
    image_spec = image_grid_spec(image_consts, height, width)
    video_spec = video_grid_spec(video_consts, num_frames, height, width)
    image_record = describe_resample(
        image_consts,
        modality="image",
        num_frames=image_consts.temporal_patch_size,
        height=height,
        width=width,
    )
    video_record = describe_resample(
        video_consts, modality="video", num_frames=num_frames, height=height, width=width
    )
    return image_spec, video_spec, image_record, video_record


@pytest.mark.parametrize("num_frames", FRAME_COUNTS)
def test_the_token_count_equals_the_closed_form(video_oracle, video_consts, num_frames):
    """The bridge, the closed form and the real processor all answer the same token count.

    The spatial factors come from the bridge's own spec, so the only thing this can be wrong
    about is the frame factor.
    """
    spec = video_grid_spec(video_consts, num_frames, FRAME_ARM_HEIGHT, FRAME_ARM_WIDTH)
    expected_grid_t = _closed_form_grid_t(num_frames, video_consts.temporal_patch_size)
    expected_tokens = expected_grid_t * spec.grid_h * spec.grid_w // video_consts.merge_length

    out = _run_video_oracle(video_oracle, num_frames, FRAME_ARM_HEIGHT, FRAME_ARM_WIDTH)
    oracle_grid = tuple(int(v) for v in out["video_grid_thw"][0].tolist())
    oracle_rows = int(out["pixel_values_videos"].shape[0])

    assert spec.grid_t == expected_grid_t, (
        f"F={num_frames}: the bridge's grid_t {spec.grid_t} is not the closed form's "
        f"{expected_grid_t}"
    )
    assert spec.grid_t == EXPECTED_GRID_T[num_frames], (
        f"F={num_frames}: the bridge's grid_t {spec.grid_t} is not the expected "
        f"{EXPECTED_GRID_T[num_frames]}"
    )
    assert spec.num_merged_tokens == expected_tokens, (
        f"F={num_frames}: the bridge packs {spec.num_merged_tokens} tokens, the closed form "
        f"expects {expected_tokens}"
    )
    assert oracle_grid == spec.thw, (
        f"F={num_frames}: the real processor's grid {oracle_grid} is not the bridge's {spec.thw}"
    )
    assert oracle_rows == spec.num_patch_rows, (
        f"F={num_frames}: the real processor produced {oracle_rows} patch rows, the bridge "
        f"expects {spec.num_patch_rows}"
    )


def test_the_grid_padding_is_not_the_pixel_budget_rounding(video_consts):
    """The grid pads the frame count up; the pixel budget rounds it to the nearest patch.

    The two agree at most frame counts, which is why the sweep above also runs F=17: with only
    counts where the two forms agree, a grid that used the budget's rounding would pass
    everywhere.
    """
    for num_frames in ROUNDINGS_AGREE_AT:
        spec = video_grid_spec(video_consts, num_frames, FRAME_ARM_HEIGHT, FRAME_ARM_WIDTH)
        budget = _budget_frame_pairs(video_consts, num_frames)
        assert budget == spec.grid_t, (
            f"the two roundings differ at F={num_frames}, where they are expected to agree"
        )

    for num_frames, (expect_budget, expect_grid) in sorted(ROUNDINGS_DIFFER_AT.items()):
        spec = video_grid_spec(video_consts, num_frames, FRAME_ARM_HEIGHT, FRAME_ARM_WIDTH)
        budget = _budget_frame_pairs(video_consts, num_frames)
        assert budget == expect_budget, (
            f"the pixel budget counts {budget} frame pairs at F={num_frames}, not the "
            f"{expect_budget} its nearest-rounding gives"
        )
        assert spec.grid_t == expect_grid, (
            f"the grid gave grid_t {spec.grid_t} at F={num_frames}, not {expect_grid}"
        )
        assert budget != spec.grid_t, (
            f"the two roundings agree at F={num_frames}, so the sweep above could not tell the "
            "pixel budget's nearest-rounding from the grid's pad-up rounding"
        )


def test_the_temporal_patch_size_is_read_and_not_assumed(video_consts):
    """The grid pads to the configured temporal patch size, not to a hard-coded 2.

    With three frames per temporal patch, 5 and 6 frames give two patches and 7 gives three;
    a grid that assumed pairs would answer 3, 3 and 4. The size is regime A at both settings,
    and the spatial factors are asserted unchanged, so the frame factor is the only thing the
    comparison reads.
    """
    assert video_consts.temporal_patch_size != 3, "the processor already uses patches of 3"
    consts = replace(video_consts, temporal_patch_size=3)
    default = video_grid_spec(video_consts, 5, FRAME_ARM_HEIGHT, FRAME_ARM_WIDTH)

    for num_frames, expected in ((5, 2), (6, 2), (7, 3)):
        assert _closed_form_grid_t(num_frames, 3) == expected
        spec = video_grid_spec(consts, num_frames, FRAME_ARM_HEIGHT, FRAME_ARM_WIDTH)
        assert spec.grid_t == expected, (
            f"F={num_frames} over temporal patches of 3 gave grid_t {spec.grid_t}, not {expected}"
        )
        assert spec.regime == REGIME_PLAIN
        assert (spec.grid_h, spec.grid_w) == (default.grid_h, default.grid_w)

    assert video_grid_spec(consts, 5, FRAME_ARM_HEIGHT, FRAME_ARM_WIDTH).grid_t != default.grid_t, (
        "five frames gave the same grid_t at patches of 3 and of 2, so the configured value is "
        "not what decides the temporal element"
    )


@pytest.mark.parametrize("name", EQUALITY_CASE_NAMES)
def test_one_frame_video_is_bit_identical_to_the_still_image(
    image_oracle, video_oracle, image_consts, video_consts, name
):
    """The two paths' grids are equal and their pixel tensors differ by 0.0.

    The condition is asserted before the equality: regime A on both paths, and a raw-frame pixel
    count at or above the floor budget. Both are read from the constants rather than trusted
    from the case list, because a size that drifted into the band below the floor budget would
    be a counterexample to the equality rather than an instance of it.
    """
    _name, height, width, regime = _case(name)
    image_spec, video_spec, _image_record, _video_record = _both_specs(
        image_consts, video_consts, height, width, num_frames=1
    )
    raw_pixels = height * width

    assert image_spec.regime == regime, (
        f"{name}: the bridge calls {height}x{width} regime {image_spec.regime}, but the case "
        f"table says regime {regime}. The table is stale."
    )
    assert image_spec.regime == REGIME_PLAIN and video_spec.regime == REGIME_PLAIN, (
        f"{name}: the equality is stated for regime A on both paths, and this size reads "
        f"{image_spec.regime} / {video_spec.regime}"
    )
    assert raw_pixels >= video_consts.floor_pixels, (
        f"{name}: {height}x{width} is {raw_pixels} raw pixels, below the floor budget "
        f"{video_consts.floor_pixels}. Inside that band upstream's resize tail pads the image "
        "and resamples the video, so the two paths differ there by design; see the band test."
    )

    grid_equal = image_spec.thw == video_spec.thw

    image_pixels = _run_image_oracle(image_oracle, height, width)["pixel_values"]
    video_pixels = _run_video_oracle(video_oracle, 1, height, width)["pixel_values_videos"]
    assert tuple(image_pixels.shape) == tuple(video_pixels.shape), (
        f"{name}: the two paths produced different token counts inside the equality's "
        f"condition, so there is no elementwise difference to measure. The image path is "
        f"{tuple(image_pixels.shape)} on a "
        f"{image_spec.canvas_height}x{image_spec.canvas_width} canvas, the 1-frame video path is "
        f"{tuple(video_pixels.shape)} on a "
        f"{video_spec.canvas_height}x{video_spec.canvas_width} canvas. Both are regime A and at "
        f"or above the floor budget, so the frame count should not have reached the canvas."
    )

    max_abs_diff = (image_pixels.double() - video_pixels.double()).abs().max().item()
    assert max_abs_diff == BIT_IDENTICAL_MAX_ABS_DIFF, (
        f"{name}: the 1-frame video's tokens differ from the still image's by "
        f"{max_abs_diff:.6e}, and bit-identical means exactly {BIT_IDENTICAL_MAX_ABS_DIFF}"
    )
    assert grid_equal, (
        f"{name}: the still image's grid {image_spec.thw} and the 1-frame video's "
        f"{video_spec.thw} differ, so the video path forked instead of degenerating. Canvases "
        f"were {image_spec.canvas_height}x{image_spec.canvas_width} and "
        f"{video_spec.canvas_height}x{video_spec.canvas_width}, regimes {image_spec.regime} and "
        f"{video_spec.regime}."
    )


@pytest.mark.parametrize(
    "name,height,width,image_canvas,video_canvas",
    FLOOR_COUNTERFACTUAL_CASES,
    ids=[c[0] for c in FLOOR_COUNTERFACTUAL_CASES],
)
def test_below_the_floor_the_video_canvas_is_larger_and_the_frame_count_is_why(
    image_consts, video_consts, name, height, width, image_canvas, video_canvas
):
    """Below the floor a 1-frame video gets a strictly larger canvas, and the frame count is why.

    The cause is asserted rather than narrated: the same call is made once more with the frame
    count changed to ``temporal_patch_size`` -- which is what upstream's image path passes -- and
    the divergence must vanish onto the image canvas. Nothing but the frame count moves between
    the two video calls.
    """
    tps = video_consts.temporal_patch_size
    got_image = resolve_canvas(
        image_consts, num_frames=image_consts.temporal_patch_size, height=height, width=width
    )
    got_video = resolve_canvas(video_consts, num_frames=1, height=height, width=width)
    counterfactual = resolve_canvas(video_consts, num_frames=tps, height=height, width=width)

    assert (got_image[0], got_image[1]) == image_canvas, (
        f"{name}: the still path's canvas is {got_image[0]}x{got_image[1]}, expected "
        f"{image_canvas[0]}x{image_canvas[1]}"
    )
    assert (got_video[0], got_video[1]) == video_canvas, (
        f"{name}: the 1-frame video's canvas is {got_video[0]}x{got_video[1]}, expected "
        f"{video_canvas[0]}x{video_canvas[1]}"
    )
    assert got_image[2] == REGIME_BELOW_FLOOR and got_video[2] == REGIME_BELOW_FLOOR, (
        f"{name}: this size is below the floor on both paths and reads "
        f"{got_image[2]} / {got_video[2]}"
    )
    assert got_video[0] > got_image[0] and got_video[1] > got_image[1], (
        f"{name}: the 1-frame video's canvas {got_video[0]}x{got_video[1]} is not strictly larger "
        f"than the still image's {got_image[0]}x{got_image[1]}. Upstream's below-floor rescue "
        "scales by sqrt(floor_pixels / (num_frames * h * w)) on the raw frame count, so one frame "
        "must ask for more upscaling than two."
    )
    assert (counterfactual[0], counterfactual[1]) == (got_image[0], got_image[1]), (
        f"{name}: told {tps} frames instead of 1, the video path still lands on "
        f"{counterfactual[0]}x{counterfactual[1]} rather than the still image's "
        f"{got_image[0]}x{got_image[1]}. The frame count is then not the cause of this divergence."
    )


def test_above_the_image_ceiling_the_two_ceilings_are_why(image_consts, video_consts):
    """The still path is cut down and the video path is not, because the two ceilings differ.

    The frame count explains none of this size, so the counterfactual changes the ceiling and not
    the frame count: the still path recomputed at the video ceiling must land on the video canvas
    and read regime A. This is arithmetic only -- 8000x6000 is 144 million pixels and resampling
    it twice on CPU costs more than the reading is worth -- and the two grid functions are pinned
    to real pixels at a lowered ceiling below instead.
    """
    name, height, width = CEILING_CASE
    got_image = resolve_canvas(
        image_consts, num_frames=image_consts.temporal_patch_size, height=height, width=width
    )
    got_video = resolve_canvas(video_consts, num_frames=1, height=height, width=width)
    counterfactual = resolve_canvas(
        image_consts,
        num_frames=image_consts.temporal_patch_size,
        height=height,
        width=width,
        ceiling_pixels=video_consts.ceiling_pixels,
    )

    assert video_consts.max_image_tokens != image_consts.max_image_tokens, (
        "the two processors declare the same max_image_tokens, so this divergence has no cause "
        "to assert and cannot exist"
    )
    assert (got_image[0], got_image[1]) == CEILING_IMAGE_CANVAS, (
        f"{name}: the still path's canvas is {got_image[0]}x{got_image[1]}, expected "
        f"{CEILING_IMAGE_CANVAS[0]}x{CEILING_IMAGE_CANVAS[1]}"
    )
    assert got_image[2] == REGIME_ABOVE_CEILING, (
        f"{name}: the still path reads {got_image[2]}, expected above the ceiling"
    )
    assert (got_video[0], got_video[1]) == CEILING_VIDEO_CANVAS, (
        f"{name}: the 1-frame video's canvas is {got_video[0]}x{got_video[1]}, expected "
        f"{CEILING_VIDEO_CANVAS[0]}x{CEILING_VIDEO_CANVAS[1]}"
    )
    assert got_video[2] == REGIME_PLAIN, (
        f"{name}: the video path reads {got_video[2]}, expected regime A -- 8000x6000 is above "
        "the image ceiling and below the video ceiling"
    )
    assert (counterfactual[0], counterfactual[1]) == CEILING_VIDEO_CANVAS, (
        f"{name}: given the video ceiling, the still path lands on "
        f"{counterfactual[0]}x{counterfactual[1]} rather than the video canvas "
        f"{CEILING_VIDEO_CANVAS[0]}x{CEILING_VIDEO_CANVAS[1]}. The ceiling is then not the cause "
        "of this divergence."
    )
    assert counterfactual[2] == REGIME_PLAIN, (
        f"{name}: given the video ceiling, the still path still reads {counterfactual[2]} rather "
        "than regime A, so the ceiling is not what put it above the bound"
    )


def test_in_the_band_the_grids_agree_and_the_content_does_not(image_consts, video_consts):
    """Same canvas, same grid, same row count, different content -- upstream's resize tail.

    The band is stated from the constants: at or above ``floor_pixels / temporal_patch_size``, so
    the image's tail test passes and its scale is capped at 1.0, and below ``floor_pixels``, so
    the 1-frame video's tail test does not and its scale is left uncapped. The image is then
    padded with a zero border and the video is resampled to fill the canvas. Only the content is
    asserted here; the pixel difference is measured further down.
    """
    name, height, width = BAND_CASE
    tps = video_consts.temporal_patch_size
    raw_pixels = height * width
    image_spec, video_spec, image_record, video_record = _both_specs(
        image_consts, video_consts, height, width, num_frames=1
    )

    assert video_consts.floor_pixels // tps <= raw_pixels < video_consts.floor_pixels, (
        f"{name}: {raw_pixels} raw pixels is outside the band "
        f"[{video_consts.floor_pixels // tps}, {video_consts.floor_pixels}), so the two paths' "
        "tail tests do not disagree here and this measures nothing"
    )
    assert image_consts.temporal_patch_size * raw_pixels >= image_consts.floor_pixels, (
        f"{name}: the image tail test must pass here, capping its scale at 1.0"
    )
    assert 1 * raw_pixels < video_consts.floor_pixels, (
        f"{name}: the 1-frame video tail test must fail here, leaving its scale uncapped"
    )

    assert (image_spec.canvas_height, image_spec.canvas_width) == BAND_CANVAS, (
        f"{name}: the still path's canvas is {image_spec.canvas_height}x{image_spec.canvas_width}, "
        f"expected {BAND_CANVAS[0]}x{BAND_CANVAS[1]}"
    )
    assert (video_spec.canvas_height, video_spec.canvas_width) == BAND_CANVAS, (
        f"{name}: the video path's canvas is {video_spec.canvas_height}x{video_spec.canvas_width}, "
        f"expected {BAND_CANVAS[0]}x{BAND_CANVAS[1]}"
    )
    assert image_spec.regime == REGIME_PLAIN and video_spec.regime == REGIME_PLAIN, (
        f"{name}: this size is regime A on both paths and reads {image_spec.regime} / "
        f"{video_spec.regime}; regime A alone is not sufficient for the paths to agree"
    )
    assert image_spec.thw == video_spec.thw, (
        f"{name}: the grids {image_spec.thw} and {video_spec.thw} differ, so this would be an "
        "ordinary shape divergence rather than the same-grid, different-pixels case it is here for"
    )
    assert (image_spec.grid_h, image_spec.grid_w) == BAND_GRID_HW
    assert image_spec.num_patch_rows == video_spec.num_patch_rows == BAND_PATCH_ROWS, (
        f"{name}: rows are {image_spec.num_patch_rows} and {video_spec.num_patch_rows}, expected "
        f"{BAND_PATCH_ROWS} on both"
    )

    assert (image_record.content_height, image_record.content_width) == BAND_IMAGE_CONTENT, (
        f"{name}: the still path's content is "
        f"{image_record.content_height}x{image_record.content_width}, expected "
        f"{BAND_IMAGE_CONTENT[0]}x{BAND_IMAGE_CONTENT[1]} -- the input size, padded and not resized"
    )
    assert (video_record.content_height, video_record.content_width) == BAND_VIDEO_CONTENT, (
        f"{name}: the video path's content is "
        f"{video_record.content_height}x{video_record.content_width}, expected "
        f"{BAND_VIDEO_CONTENT[0]}x{BAND_VIDEO_CONTENT[1]} -- the whole canvas, resampled"
    )
    assert (image_record.content_height, image_record.content_width) != (
        video_record.content_height,
        video_record.content_width,
    ), f"{name}: the two contents agree, so there is no divergence here to assert"


def test_a_three_frame_video_is_not_bit_identical_to_the_still_image(
    image_oracle, video_oracle, image_consts, video_consts
):
    """Three frames must pack twice the rows of one still image.

    Three and not two: at F=2 the video takes grid_t 1 and the same canvas as one still image, so
    the shapes are necessarily equal, which the last assertion measures rather than states.
    """
    height, width = SHAPE_CONTROL_HEIGHT, SHAPE_CONTROL_WIDTH
    image_spec, video_spec, _image_record, _video_record = _both_specs(
        image_consts, video_consts, height, width, num_frames=SHAPE_CONTROL_FRAMES
    )
    image_pixels = _run_image_oracle(image_oracle, height, width)["pixel_values"]
    video_pixels = _run_video_oracle(
        video_oracle, SHAPE_CONTROL_FRAMES, height, width
    )["pixel_values_videos"]

    assert tuple(image_pixels.shape) != tuple(video_pixels.shape), (
        f"a {SHAPE_CONTROL_FRAMES}-frame video packed to the same shape as one still image, so "
        "the shape gate cannot tell one frame from three"
    )
    assert image_spec.num_patch_rows == SHAPE_CONTROL_IMAGE_ROWS, (
        f"the still image packs {image_spec.num_patch_rows} rows, expected "
        f"{SHAPE_CONTROL_IMAGE_ROWS}"
    )
    assert video_spec.num_patch_rows == SHAPE_CONTROL_VIDEO_ROWS, (
        f"the {SHAPE_CONTROL_FRAMES}-frame video packs {video_spec.num_patch_rows} rows, expected "
        f"{SHAPE_CONTROL_VIDEO_ROWS}"
    )
    assert video_spec.grid_t == 2 * image_spec.grid_t, (
        f"{SHAPE_CONTROL_FRAMES} frames gave grid_t {video_spec.grid_t} where one image gives "
        f"{image_spec.grid_t}"
    )
    assert (video_spec.canvas_height, video_spec.canvas_width) == (
        image_spec.canvas_height,
        image_spec.canvas_width,
    ), "this size must be one where the canvas does not move, so only the frame count does"

    two_frame_spec = video_grid_spec(video_consts, 2, height, width)
    assert two_frame_spec.thw == image_spec.thw, (
        "a 2-frame video no longer packs to the same grid as one still image, so F=2 would in "
        "fact tell the two paths apart and the reason for using F=3 no longer holds"
    )


def test_in_the_band_the_shapes_agree_and_the_pixels_differ(image_oracle, video_oracle):
    """Equal shapes and unequal pixels, from one and the same content.

    Both paths are given the identical constant frame at 130x130 and produce the same shape, and
    the tensors still differ, because the image is padded with a zero border and the video is
    resampled to fill the canvas. A comparison that only looked at shape would read this as
    agreement.
    """
    name, height, width = BAND_CASE
    image_pixels = _run_image_oracle(image_oracle, height, width, value=CONTENT_VALUE_A)[
        "pixel_values"
    ]
    video_pixels = _run_video_oracle(video_oracle, 1, height, width, value=CONTENT_VALUE_A)[
        "pixel_values_videos"
    ]
    assert tuple(image_pixels.shape) == tuple(video_pixels.shape), (
        f"{name}: the two paths produced {tuple(image_pixels.shape)} and "
        f"{tuple(video_pixels.shape)}. This comparison needs equal shapes -- its whole point is a "
        "pixel difference that a shape gate cannot see."
    )
    max_abs_diff = (image_pixels.double() - video_pixels.double()).abs().max().item()
    assert max_abs_diff > BIT_IDENTICAL_MAX_ABS_DIFF, (
        f"{name}: the same constant content read as bit-identical across the two paths at equal "
        "shapes, so either the comparison cannot see pixel values at all, or upstream stopped "
        "padding the image while resampling the video"
    )


@pytest.mark.parametrize(
    "name,num_frames,height,width,grid,rows,tokens",
    VIDEO_ORACLE_CASES,
    ids=[row[0] for row in VIDEO_ORACLE_CASES],
)
def test_the_fork_video_grid_equals_the_upstream_video_processor(
    video_oracle, video_consts, name, num_frames, height, width, grid, rows, tokens
):
    """The fork's video grid, the expected numbers and the real processor all agree.

    Per modality: the video path is compared with the video processor and never with the image
    processor.
    """
    spec = video_grid_spec(video_consts, num_frames, height, width)
    out = _run_video_oracle(video_oracle, num_frames, height, width)
    oracle_grid = tuple(int(v) for v in out["video_grid_thw"][0].tolist())
    oracle_rows = int(out["pixel_values_videos"].shape[0])

    assert spec.thw == grid, f"{name}: the fork's grid {spec.thw} is not the expected {grid}"
    assert oracle_grid == grid, (
        f"{name}: the real video processor's grid {oracle_grid} is not the expected {grid}"
    )
    assert spec.num_patch_rows == rows and oracle_rows == rows, (
        f"{name}: rows are fork {spec.num_patch_rows} and oracle {oracle_rows}, expected {rows}"
    )
    assert spec.num_merged_tokens == tokens, (
        f"{name}: the fork packs {spec.num_merged_tokens} tokens, expected {tokens}"
    )


def test_a_lowered_ceiling_pins_both_grid_functions_at_regime_c(glm5_next_pkg):
    """Both grid functions measured end to end at regime C, on real pixels.

    Nothing in the fork changes: the constructor keyword becomes an attribute on the processor
    instance, ``GridConstants.from_processor`` reads it, and the same instances are the oracle.
    The first two assertions read the attribute back, because a silently ignored keyword would
    leave both processors answering at the default ceiling and every assertion below would pass
    while measuring regime A under a regime C label.
    """
    height, width = LOWERED_CEILING_HEIGHT, LOWERED_CEILING_WIDTH
    image_oracle_n = glm5_next_pkg.Glm5NextImageProcessor(
        max_image_tokens=LOWERED_CEILING_TOKENS
    )
    video_oracle_n = glm5_next_pkg.Glm5NextVideoProcessor(
        max_image_tokens=LOWERED_CEILING_TOKENS
    )
    consts_image = GridConstants.from_processor(image_oracle_n)
    consts_video = GridConstants.from_processor(video_oracle_n)

    assert consts_image.max_image_tokens == LOWERED_CEILING_TOKENS, (
        f"Glm5NextImageProcessor(max_image_tokens={LOWERED_CEILING_TOKENS}) carries "
        f"{consts_image.max_image_tokens}, so the keyword did not reach the attribute and this "
        "test would measure the default ceiling while reporting a lowered one"
    )
    assert consts_video.max_image_tokens == LOWERED_CEILING_TOKENS, (
        f"Glm5NextVideoProcessor(max_image_tokens={LOWERED_CEILING_TOKENS}) carries "
        f"{consts_video.max_image_tokens}, so the keyword did not reach the attribute"
    )

    image_spec = image_grid_spec(consts_image, height, width)
    video_spec = video_grid_spec(consts_video, 1, height, width)
    image_out = _run_image_oracle(image_oracle_n, height, width)
    video_out = _run_video_oracle(video_oracle_n, 1, height, width)
    oracle_image_grid = tuple(int(v) for v in image_out["image_grid_thw"][0].tolist())
    oracle_video_grid = tuple(int(v) for v in video_out["video_grid_thw"][0].tolist())
    oracle_image_rows = int(image_out["pixel_values"].shape[0])
    oracle_video_rows = int(video_out["pixel_values_videos"].shape[0])

    assert image_spec.regime == REGIME_ABOVE_CEILING, (
        f"the still path reads {image_spec.regime} at a ceiling of {LOWERED_CEILING_TOKENS} "
        "tokens, so the lowered ceiling did not bind"
    )
    assert video_spec.regime == REGIME_ABOVE_CEILING, (
        f"the video path reads {video_spec.regime} at a ceiling of {LOWERED_CEILING_TOKENS} tokens"
    )
    assert (image_spec.canvas_height, image_spec.canvas_width) == LOWERED_CEILING_CANVAS, (
        f"the still path's canvas is {image_spec.canvas_height}x{image_spec.canvas_width}, "
        f"expected {LOWERED_CEILING_CANVAS[0]}x{LOWERED_CEILING_CANVAS[1]}"
    )
    assert (video_spec.canvas_height, video_spec.canvas_width) == LOWERED_CEILING_CANVAS, (
        f"the video path's canvas is {video_spec.canvas_height}x{video_spec.canvas_width}, "
        f"expected {LOWERED_CEILING_CANVAS[0]}x{LOWERED_CEILING_CANVAS[1]}"
    )
    assert (image_spec.grid_h, image_spec.grid_w) == LOWERED_CEILING_GRID_HW
    assert (video_spec.grid_h, video_spec.grid_w) == LOWERED_CEILING_GRID_HW
    assert image_spec.num_patch_rows == LOWERED_CEILING_PATCH_ROWS
    assert video_spec.num_patch_rows == LOWERED_CEILING_PATCH_ROWS

    assert oracle_image_grid == image_spec.thw, (
        f"at the lowered ceiling the real image processor's grid {oracle_image_grid} is not the "
        f"fork's {image_spec.thw}"
    )
    assert oracle_video_grid == video_spec.thw, (
        f"at the lowered ceiling the real video processor's grid {oracle_video_grid} is not the "
        f"fork's {video_spec.thw}"
    )
    assert oracle_image_rows == image_spec.num_patch_rows, (
        f"the real image processor produced {oracle_image_rows} rows, the fork expects "
        f"{image_spec.num_patch_rows}"
    )
    assert oracle_video_rows == video_spec.num_patch_rows, (
        f"the real video processor produced {oracle_video_rows} rows, the fork expects "
        f"{video_spec.num_patch_rows}"
    )
    assert image_spec.thw == video_spec.thw, (
        f"at one shared ceiling the two paths still disagree -- image {image_spec.thw}, video "
        f"{video_spec.thw}. The ceiling divergence above is attributed to the two ceilings "
        "differing, so making them equal must remove it."
    )
