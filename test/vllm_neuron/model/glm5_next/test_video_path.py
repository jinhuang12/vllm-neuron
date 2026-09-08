# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ``inc-glm53f-061`` -- WP10: the video path.

THE REGISTERED CRITERION, VERBATIM. Copied from ``design/increment-plan.md`` line 1333 at plan
revision 260, whose bytes are
``ee09a7de20fd3251e4b3106d3c6c97e89e7472ffc1fcf439827cb92f38599a7a``. No criteria pin was minted
for ``-061``, so the plan digest is the pin this file cites; the acceptance driver greps the quote
below as a fixed string. Markdown emphasis and all -- a transliteration would let the quote drift
from the criterion while still reading correctly:

**Expected:** an F-frame video packs to a token count equal to the closed-form expectation exactly in **3/3** frame counts; a 1-frame video produces a token sequence **bit-identical** to the same content as a still image (max abs diff == 0.0), which proves the video path degenerates correctly instead of forking.

The two conjuncts are measured as V01 (3/3 frame counts) and V04 (the 1-frame bit-identity).
V02 fixes the reference the closed form is checked against, V03 and V05 are firing controls, V06
exercises the guards, and V07 reports every reading so no number here is silent.

THIS INCREMENT IS TEST-ONLY. The ruling that scoped it refused a packer with no caller and a video
branch that would duplicate handling already landed: ``video_grid_spec`` landed with ``-056`` and
per-frame ``compute_attention_bounds`` landed with ``-060``. Nothing is authored here to give a
file surface a body, so this file measures landed code and adds none.

WHAT IS CONSUMED AND WHAT IS UNDER TEST. Ruling ``design-20260905-aq`` (iii) forbids re-deriving
the grid in a test. So the SPATIAL factors are consumed from ``-056``'s landed module and never
recomputed here; the one quantity this file states as a closed form is the FRAME factor, which is
the increment's own subject. The form is the plan rider's grid form ``grid_t = (F + (-F % tps)) //
tps``, with ``tps`` read off the processor and never typed as 2.

THE ROUNDING DECOY, AND WHY V03 EXISTS. The module carries two roundings of the same quantity, and
they are not interchangeable. ``GridConstants.aligned_frames`` rounds to the NEAREST whole temporal
patch, which is what the PIXEL BUDGET is measured over; the grid pads UP by ``-F % tps``. They
agree at F=1 and F=3 and they disagree at F=17, where nearest-rounding gives 8 and the grid gives
9. A test that used the budget form would read a green at F=1 and F=3 and be wrong about every odd
frame count above 3. V03 asserts the two forms DISAGREE at F=17, so this file is on the record as
measuring the grid form and not the budget form.

WHERE THE FRAME-COUNT REFERENCE COMES FROM. The five ``grid_t`` points in ``RECORDED_GRID_T`` were
read from the real processor on the leased host and filed as
``read-056-r3-closedform.out`` lines 68-76, whose bytes are
``c5603c91f0e091b7386e95b4851588873977b7cc1a52a245140f93780a261700``. They are quoted here by
frame count rather than recomputed, so V01's closed form is checked against a world reading and
cannot become a transcription of its own answer.

THE SEVEN CASES ARE ``-056``'s R2 SET. Their sizes and regimes are quoted from
``test_vision_preprocessing.py`` lines 107-114, which read every integer from
``derive-056r2-divergence.out``. Only the size and the regime label are carried across, never a
token count: this file compares two paths against each other, so an absolute count from another
increment would add a stale number without adding a reading. V04's first act is to check each
quoted regime against the regime ``-056``'s landed ``resolve_canvas`` computes for that size, so a
stale row here fails loudly instead of mislabelling a reading.

A FORECAST RED IS RECORDED IN THE PREDICTIONS, NOT SOFTENED HERE. transformers hands its image
resizer ``num_frames=temporal_patch_size`` and its video resizer the real frame count
(``read-056-r7-imagepath.out``; ``predictions-056-build.txt`` lines 26-29). In the two clamped
regimes the below-floor scale is ``sqrt(floor_px / (num_frames * h * w))``, so it reads a different
canvas at F=1 than at F=2 -- and ``test_vision_preprocessing.py`` lines 243-249 already assert that
asymmetry as landed behaviour, naming the 12x12 that a one-frame reading of 29x29 produces where
the image path gives 8x8. V04 is therefore FORECAST to diverge on the below-floor cases. The
criterion is registered and stays exactly as registered: this file asserts ``max abs diff == 0.0``
per case as written, reports the divergence with both canvases beside each other, and the routing
of that red belongs to the lead.

WHEN THE SHAPES DIFFER THERE IS NO MAX ABS DIFF. Two tensors of different row counts have no
elementwise difference at all, so V04 does not attempt one. It gates on shape first and reports a
shape divergence with both shapes named, then computes the difference only where the shapes agree.
A test that subtracted them regardless would report an error where the reading is a measurement.

NO CANDIDATE-ROOT REQUIREMENT. V06's origin guard has two arms and never skips: with
``GLM53F_CANDIDATE_ROOT`` set the import must resolve under the declared tree, and unset the tree
is derived from this file's own location, which is how a plain ``pytest`` from a checkout runs. It
is the form landed in ``test_vision_preprocessing.py`` and it passes in both partitions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from vllm_neuron.model.glm5_next.utils.vision_preprocessing import (
    REGIME_ABOVE_CEILING,
    REGIME_BELOW_FLOOR,
    REGIME_PLAIN,
    GridConstants,
    describe_resample,
    image_grid_spec,
    video_grid_spec,
)

# ---------------------------------------------------------------------------
# Criterion 1: the frame counts.
# ---------------------------------------------------------------------------
#: The three REGISTERED frame counts, one per shape the grid form can take: below one temporal
#: patch, odd and padding up by one patch, and odd and large enough that the budget form's
#: nearest-rounding gives a different answer. Every one of the three has a recorded reference.
REGISTERED_FRAME_COUNTS = (1, 3, 17)

#: Extra coverage, not a registered criterion: the two even counts, which pad by nothing. They are
#: reported and asserted alongside the three, and they change no registered number.
EXTRA_FRAME_COUNTS = (2, 16)

#: ``grid_t`` read off the real processor and filed as ``read-056-r3-closedform.out`` lines 68-76.
#: Quoted by frame count, never recomputed.
RECORDED_GRID_T = {1: 1, 2: 1, 3: 2, 16: 8, 17: 9}

#: The one frame count where the budget form and the grid form must disagree, with the two answers
#: the record and the module's own docstring name.
DECOY_FRAME_COUNT = 17
DECOY_BUDGET_FORM_GRID_T = 8
DECOY_GRID_FORM_GRID_T = 9

#: The frame-count arm's item size. 112x112 is regime A on this checkpoint, so its canvas does not
#: move with the frame count and the arm reads the frame factor alone. It is the size ``-056``'s
#: own registered video cases use.
FRAME_ARM_HEIGHT = 112
FRAME_ARM_WIDTH = 112

# ---------------------------------------------------------------------------
# Criterion 2: the seven cases, from ``-056``'s R2 set. (name, height, width, regime)
# ---------------------------------------------------------------------------
R2_CASES = (
    ("aligned_to_both_factors", 448, 448, REGIME_PLAIN),
    ("aligned_to_28_only", 420, 336, REGIME_PLAIN),
    ("tiny_floor_bound", 60, 40, REGIME_BELOW_FLOOR),
    ("tiny_square", 32, 32, REGIME_BELOW_FLOOR),
    ("odd_mid_size", 500, 333, REGIME_PLAIN),
    ("huge_ceiling_bound", 8000, 6000, REGIME_ABOVE_CEILING),
    ("wide_strip", 1400, 56, REGIME_PLAIN),
)

#: The one case whose REAL-PIXEL arm is skipped, for the reason ``-056`` skipped it: an 8000x6000
#: uint8 image is 144 million pixels and resampling it twice on CPU costs more than the reading is
#: worth. Its ARITHMETIC arm still runs, so the case is not dropped from the seven.
R2_TOO_BIG_FOR_REAL_PIXELS = "huge_ceiling_bound"

#: The constant pixel value both paths are fed, and a second value V05 uses to prove the
#: comparison reads content and not only shape. Mid-range, so neither can be confused with the
#: zero padding.
CONTENT_VALUE_A = 200
CONTENT_VALUE_B = 100

#: The criterion's own number. Bit-identical means exactly this and no tolerance is authored.
BIT_IDENTICAL_MAX_ABS_DIFF = 0.0


# ---------------------------------------------------------------------------
# Fixtures. Both processors are the transformers classes themselves, constructed with no
# arguments: this test needs no checkpoint, no download and no network.
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
    """One still image of constant content, through transformers' image processor."""
    image = torch.full((3, height, width), value, dtype=torch.uint8)
    return oracle(images=[image], return_tensors="pt")


def _run_video_oracle(oracle, num_frames, height, width, value=CONTENT_VALUE_A):
    """One video of constant content, with ``do_sample_frames`` PINNED False.

    The pin is the plan rider's: the video processor samples frames by default, which would move
    the frame count away from the one under test and make F an output rather than an input.
    """
    video = torch.full((num_frames, 3, height, width), value, dtype=torch.uint8)
    return oracle(videos=[video], do_sample_frames=False, return_tensors="pt")


def _closed_form_grid_t(num_frames: int, temporal_patch_size: int) -> int:
    """The plan rider's grid form: pad F UP to whole temporal patches, then divide.

    ``temporal_patch_size`` is read off the processor by the caller. This is the one quantity this
    file states rather than consumes, because it is the increment's subject.
    """
    padded = num_frames + (-num_frames % temporal_patch_size)
    return padded // temporal_patch_size


def _budget_form_grid_t(consts: GridConstants, num_frames: int) -> int:
    """The DECOY: the module's pixel-budget rounding, which is nearest and not up.

    Used only by V03, to show the two forms are distinguishable. Never used as an expectation.
    """
    return consts.aligned_frames(num_frames) // consts.temporal_patch_size


def _record(*fields):
    """Print one pipe-delimited record the acceptance driver reads by label and field.

    The driver gates on these numbers rather than on pytest's verdict, so a green file whose
    achieved reading was not the registered one cannot read as a pass. A record is located by its
    label, never by column position, because ``-q`` and ``-v`` indent differently.
    """
    print("VIDEO|" + "|".join(str(f) for f in fields))


def _paths_report(image_spec, video_spec, image_record, video_record, num_frames):
    """The per-case reading, in the fields the ruling named: F, regime, canvas, grid_h, grid_w.

    Both paths are printed side by side and unconditionally, so a divergence is settleable from
    the transcript alone without re-running anything.
    """
    return (
        "frames", num_frames,
        "image_regime", image_spec.regime,
        "video_regime", video_spec.regime,
        "image_canvas", f"{image_spec.canvas_height}x{image_spec.canvas_width}",
        "video_canvas", f"{video_spec.canvas_height}x{video_spec.canvas_width}",
        "image_grid_hw", f"{image_spec.grid_h}x{image_spec.grid_w}",
        "video_grid_hw", f"{video_spec.grid_h}x{video_spec.grid_w}",
        "image_rows", image_spec.num_patch_rows,
        "video_rows", video_spec.num_patch_rows,
        "image_content", f"{image_record.content_height}x{image_record.content_width}",
        "video_content", f"{video_record.content_height}x{video_record.content_width}",
        "image_resampled", image_record.resampled,
        "video_resampled", video_record.resampled,
    )


def _both_specs(image_consts, video_consts, height, width, num_frames):
    """The two paths' grids and resample records for one size, all four consumed from ``-056``."""
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


# ---------------------------------------------------------------------------
# V01 -- criterion 1: the token count equals the closed form, in 3/3 frame counts.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_frames", REGISTERED_FRAME_COUNTS + EXTRA_FRAME_COUNTS)
def test_v01_the_token_count_equals_the_closed_form(video_oracle, video_consts, num_frames):
    """One frame count: the bridge, the closed form, the recorded reference and the real
    processor all answer the same token count.

    The spatial factors come from the bridge's own spec, so the only thing this arm can be wrong
    about is the frame factor -- which is what the increment is for.
    """
    spec = video_grid_spec(video_consts, num_frames, FRAME_ARM_HEIGHT, FRAME_ARM_WIDTH)
    expected_grid_t = _closed_form_grid_t(num_frames, video_consts.temporal_patch_size)
    expected_tokens = expected_grid_t * spec.grid_h * spec.grid_w // video_consts.merge_length

    out = _run_video_oracle(video_oracle, num_frames, FRAME_ARM_HEIGHT, FRAME_ARM_WIDTH)
    oracle_grid = tuple(int(v) for v in out["video_grid_thw"][0].tolist())
    oracle_rows = int(out["pixel_values_videos"].shape[0])

    _record(
        "v01",
        "frames", num_frames,
        "registered", num_frames in REGISTERED_FRAME_COUNTS,
        "grid_t", spec.grid_t,
        "closed_form_grid_t", expected_grid_t,
        "recorded_grid_t", RECORDED_GRID_T[num_frames],
        "oracle_grid", ",".join(str(v) for v in oracle_grid),
        "tokens", spec.num_merged_tokens,
        "closed_form_tokens", expected_tokens,
        "rows", spec.num_patch_rows,
        "oracle_rows", oracle_rows,
        "regime", spec.regime,
    )

    assert spec.grid_t == expected_grid_t, (
        f"F={num_frames}: the bridge's grid_t {spec.grid_t} is not the closed form's "
        f"{expected_grid_t}"
    )
    assert spec.grid_t == RECORDED_GRID_T[num_frames], (
        f"F={num_frames}: the bridge's grid_t {spec.grid_t} is not the "
        f"{RECORDED_GRID_T[num_frames]} recorded from the real processor in "
        "read-056-r3-closedform.out"
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


def test_v01_the_three_registered_counts_are_three_different_grid_shapes(video_consts):
    """3/3 would be worth little if the three counts produced one answer three times."""
    grid_ts = {
        _closed_form_grid_t(f, video_consts.temporal_patch_size) for f in REGISTERED_FRAME_COUNTS
    }
    assert len(grid_ts) == len(REGISTERED_FRAME_COUNTS), (
        f"the three registered frame counts collapse to grid_t {sorted(grid_ts)}"
    )


# ---------------------------------------------------------------------------
# V02 -- the reference the closed form is checked against.
# ---------------------------------------------------------------------------
def test_v02_every_registered_count_has_a_recorded_reference():
    """No registered frame count may be checked against the closed form alone.

    The closed form is authored in this file. Without a world reading beside it, V01 would be
    comparing the module to a rule this file made up.
    """
    for num_frames in REGISTERED_FRAME_COUNTS:
        assert num_frames in RECORDED_GRID_T, (
            f"F={num_frames} is registered but has no recorded reference in "
            "read-056-r3-closedform.out"
        )


def test_v02_the_recorded_points_agree_with_the_closed_form(video_consts):
    """All five recorded points, registered and extra, must satisfy the grid form.

    This is the closed form's own evidence: five readings from the real processor, none of them
    produced by this file.
    """
    for num_frames, recorded in sorted(RECORDED_GRID_T.items()):
        derived = _closed_form_grid_t(num_frames, video_consts.temporal_patch_size)
        _record("v02", "frames", num_frames, "recorded", recorded, "closed_form", derived)
        assert derived == recorded, (
            f"F={num_frames}: the closed form gives {derived}, the processor recorded {recorded}"
        )


# ---------------------------------------------------------------------------
# V03 -- control: the budget form and the grid form must DISAGREE at F=17.
# ---------------------------------------------------------------------------
def test_v03_the_budget_rounding_is_not_the_grid_rounding(video_consts):
    """The decoy control. If these two ever agree at F=17, V01 cannot tell them apart.

    The module carries both roundings deliberately and documents why. This asserts the difference
    is real at the one frame count where it shows, with both answers named, so V01's green is
    known to be the grid form's green.
    """
    budget = _budget_form_grid_t(video_consts, DECOY_FRAME_COUNT)
    grid = _closed_form_grid_t(DECOY_FRAME_COUNT, video_consts.temporal_patch_size)
    _record(
        "v03",
        "frames", DECOY_FRAME_COUNT,
        "budget_form", budget,
        "grid_form", grid,
        "expect_budget", DECOY_BUDGET_FORM_GRID_T,
        "expect_grid", DECOY_GRID_FORM_GRID_T,
    )
    assert budget == DECOY_BUDGET_FORM_GRID_T, (
        f"the budget form gave {budget} at F={DECOY_FRAME_COUNT}, not the "
        f"{DECOY_BUDGET_FORM_GRID_T} the module's nearest-rounding documents"
    )
    assert grid == DECOY_GRID_FORM_GRID_T, (
        f"the grid form gave {grid} at F={DECOY_FRAME_COUNT}, not "
        f"{DECOY_GRID_FORM_GRID_T}"
    )
    assert budget != grid, (
        f"the two roundings agree at F={DECOY_FRAME_COUNT}, so V01 could not tell the pixel "
        "budget's nearest-rounding from the grid's pad-up rounding"
    )


# ---------------------------------------------------------------------------
# V04 -- criterion 2: a 1-frame video is bit-identical to the same content as a still image.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,height,width,regime", R2_CASES, ids=[c[0] for c in R2_CASES]
)
def test_v04_one_frame_video_is_bit_identical_to_the_still_image(
    image_oracle, video_oracle, image_consts, video_consts, name, height, width, regime
):
    """One case: the two paths' grids are equal and their pixel tensors differ by exactly 0.0.

    Every field the ruling named is printed before any assertion runs, so a red is settleable
    from the transcript without a second run. The regime quoted in the case table is checked
    against the regime the landed module computes, so a stale row cannot mislabel the reading.
    """
    image_spec, video_spec, image_record, video_record = _both_specs(
        image_consts, video_consts, height, width, num_frames=1
    )
    _record("v04", "case", name, "size", f"{height}x{width}", *_paths_report(
        image_spec, video_spec, image_record, video_record, num_frames=1
    ))

    assert image_spec.regime == regime, (
        f"{name}: the landed module calls {height}x{width} regime {image_spec.regime}, but this "
        f"file's case table quotes regime {regime}. The table is stale."
    )

    # The arithmetic arm, which every case runs. grid_t is 1 on both paths by construction, so a
    # divergence here is a divergence in the spatial canvas and nothing else.
    grid_equal = image_spec.thw == video_spec.thw
    _record(
        "v04_grid", name,
        "image_thw", ",".join(str(v) for v in image_spec.thw),
        "video_thw", ",".join(str(v) for v in video_spec.thw),
        "equal", grid_equal,
    )

    if name == R2_TOO_BIG_FOR_REAL_PIXELS:
        _record("v04_pixels", name, "skipped", "144_million_pixels_costs_more_than_the_reading")
    else:
        image_out = _run_image_oracle(image_oracle, height, width)
        video_out = _run_video_oracle(video_oracle, 1, height, width)
        image_pixels = image_out["pixel_values"]
        video_pixels = video_out["pixel_values_videos"]
        shapes_equal = tuple(image_pixels.shape) == tuple(video_pixels.shape)
        _record(
            "v04_pixels", name,
            "image_shape", "x".join(str(v) for v in image_pixels.shape),
            "video_shape", "x".join(str(v) for v in video_pixels.shape),
            "shapes_equal", shapes_equal,
        )
        if shapes_equal:
            # Only here is an elementwise difference defined.
            max_abs_diff = (
                (image_pixels.double() - video_pixels.double()).abs().max().item()
            )
            _record("v04_diff", name, "max_abs_diff", f"{max_abs_diff:.6e}")
            assert max_abs_diff == BIT_IDENTICAL_MAX_ABS_DIFF, (
                f"{name}: the 1-frame video's tokens differ from the still image's by "
                f"{max_abs_diff:.6e}, and bit-identical means exactly "
                f"{BIT_IDENTICAL_MAX_ABS_DIFF}"
            )
        else:
            _record("v04_diff", name, "max_abs_diff", "undefined_shapes_differ")
            pytest.fail(
                f"{name}: the two paths produced different token counts, so there is no "
                f"elementwise difference to measure. The image path is "
                f"{tuple(image_pixels.shape)} on a "
                f"{image_spec.canvas_height}x{image_spec.canvas_width} canvas, the 1-frame video "
                f"path is {tuple(video_pixels.shape)} on a "
                f"{video_spec.canvas_height}x{video_spec.canvas_width} canvas. transformers "
                "hands its image resizer num_frames=temporal_patch_size "
                "(image_grid_spec passes consts.temporal_patch_size) and its video resizer the "
                "real frame count (video_grid_spec passes num_frames), and the below-floor scale "
                "sqrt(floor_px / (num_frames * h * w)) reads a different canvas for the two. "
                "This is a finding about the frame-count handoff, not a tolerance to widen."
            )

    assert grid_equal, (
        f"{name}: the still image's grid {image_spec.thw} and the 1-frame video's "
        f"{video_spec.thw} differ, so the video path forked instead of degenerating. Canvases "
        f"were {image_spec.canvas_height}x{image_spec.canvas_width} and "
        f"{video_spec.canvas_height}x{video_spec.canvas_width}, regimes {image_spec.regime} and "
        f"{video_spec.regime}."
    )


def test_v04_the_seven_cases_cover_all_three_regimes():
    """Seven cases in one regime would say nothing about the clamped branches.

    The clamped regimes are the ones where the frame count reaches the canvas at all, so a set
    without them could not measure the criterion where it is at risk.
    """
    assert len(R2_CASES) == 7, f"expected -056's seven-case set, got {len(R2_CASES)}"
    regimes = {case[3] for case in R2_CASES}
    assert regimes == {REGIME_PLAIN, REGIME_BELOW_FLOOR, REGIME_ABOVE_CEILING}, (
        f"the seven cases cover only {sorted(regimes)}"
    )


# ---------------------------------------------------------------------------
# V05 -- controls: the bit-identity comparison can fail, on shape and on content.
# ---------------------------------------------------------------------------
def test_v05_a_two_frame_video_is_not_bit_identical_to_the_still_image(
    image_oracle, video_oracle, image_consts, video_consts
):
    """V04 can fail on shape. Two frames must pack twice the rows of one still image.

    Without this, a V04 green could mean the comparison never looked at the frame count.
    """
    name, height, width, _regime = R2_CASES[0]
    image_spec, video_spec, image_record, video_record = _both_specs(
        image_consts, video_consts, height, width, num_frames=2
    )
    image_pixels = _run_image_oracle(image_oracle, height, width)["pixel_values"]
    video_pixels = _run_video_oracle(video_oracle, 2, height, width)["pixel_values_videos"]
    _record("v05_shape", name, *_paths_report(
        image_spec, video_spec, image_record, video_record, num_frames=2
    ), "image_shape", "x".join(str(v) for v in image_pixels.shape),
        "video_shape", "x".join(str(v) for v in video_pixels.shape))
    assert tuple(image_pixels.shape) != tuple(video_pixels.shape), (
        "a 2-frame video packed to the same shape as one still image, so V04's shape gate could "
        "not tell one frame from two"
    )
    assert video_spec.grid_t == 2 * image_spec.grid_t, (
        f"two frames gave grid_t {video_spec.grid_t} where one image gives {image_spec.grid_t}"
    )


def test_v05_different_content_is_not_bit_identical(image_oracle, video_oracle):
    """V04 can fail on content too, not only on shape.

    Both arms here have identical shapes by construction, so this is the control that shows the
    max-abs-diff reading is looking at pixels and not merely agreeing because the shapes matched.
    """
    name, height, width, _regime = R2_CASES[0]
    image_pixels = _run_image_oracle(image_oracle, height, width, value=CONTENT_VALUE_A)[
        "pixel_values"
    ]
    video_pixels = _run_video_oracle(video_oracle, 1, height, width, value=CONTENT_VALUE_B)[
        "pixel_values_videos"
    ]
    if tuple(image_pixels.shape) != tuple(video_pixels.shape):
        pytest.skip(
            "this size's two paths already differ in shape, so a content control cannot be "
            "measured on it; the shape divergence is V04's reading"
        )
    max_abs_diff = (image_pixels.double() - video_pixels.double()).abs().max().item()
    _record(
        "v05_content", name,
        "value_a", CONTENT_VALUE_A,
        "value_b", CONTENT_VALUE_B,
        "max_abs_diff", f"{max_abs_diff:.6e}",
    )
    assert max_abs_diff > BIT_IDENTICAL_MAX_ABS_DIFF, (
        f"{name}: two different constant contents read as bit-identical, so V04's comparison "
        "cannot see pixel values at all"
    )


# ---------------------------------------------------------------------------
# V06 -- the guards.
# ---------------------------------------------------------------------------
def test_v06_the_module_under_test_is_the_tree_this_run_means():
    """Two arms, and never a skip, because a skip would hide the mix-up it looks for.

    An editable install can point at a different checkout than the one these tests live in, so
    which tree answers ``import vllm_neuron`` is a property of the run, not of the source.
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
    # The marker is pyproject.toml and NOT the package directory: the test tree mirrors the
    # package, so test/vllm_neuron/ is also a directory called vllm_neuron and a package-only
    # marker would accept the test directory as a repository root.
    assert (root / "pyproject.toml").is_file() and (
        root / "vllm_neuron" / "__init__.py"
    ).is_file(), (
        f"{root}, taken from {source}, is not a repository root -- it holds no pyproject.toml "
        "beside a vllm_neuron package, so the check below would be asserting something about the "
        "wrong directory."
    )
    assert root in resolved.parents, (
        f"vllm_neuron resolved to {resolved}, which is not under {root} (taken from {source}). "
        "Some other checkout answered the import, so this run measured a tree it did not mean to."
    )


def test_v06_a_zero_frame_video_is_not_silently_one(video_consts):
    """Zero frames must not pad up to a whole temporal patch and read as content.

    WHAT IS ASSERTED AND WHAT IS ONLY REPORTED. The closed form is asserted, because it is
    arithmetic this file owns: it gives 0 at F=0, and if it ever gave 1 an empty video would occupy
    a token and the packing would be off by a frame pair for everything after it.

    The end-to-end call is REPORTED and not asserted. It reaches transformers' ``smart_resize``
    with ``num_frames=0``, and what that does is in no record this lane holds -- an empty video is
    not part of either registered criterion, so an exception there would turn a green acceptance
    red for a question -061 never asked. The reading is printed either way, which is what makes the
    behaviour known for the next increment that needs it.
    """
    assert _closed_form_grid_t(0, video_consts.temporal_patch_size) == 0, (
        "the closed form must give grid_t 0 for a video with no frames"
    )
    try:
        spec = video_grid_spec(video_consts, 0, FRAME_ARM_HEIGHT, FRAME_ARM_WIDTH)
    except Exception as exc:  # noqa: BLE001 -- the point is to report, never to classify
        _record("v06", "frames", 0, "end_to_end", f"raised {type(exc).__name__}: {exc}")
    else:
        _record(
            "v06", "frames", 0,
            "end_to_end_grid_t", spec.grid_t,
            "tokens", spec.num_merged_tokens,
            "regime", spec.regime,
        )


def test_v06_the_temporal_patch_size_is_read_and_not_assumed(video_consts):
    """The closed form must answer correctly at another temporal patch size.

    If it only works at 2, the form has the checkpoint's constant baked into it rather than
    reading it, and a checkpoint with a different temporal size would be packed wrong.
    """
    assert _closed_form_grid_t(5, 3) == 2, "F=5 over patches of 3 pads to 6 and gives 2"
    assert _closed_form_grid_t(6, 3) == 2, "F=6 over patches of 3 is exact and gives 2"
    assert _closed_form_grid_t(7, 3) == 3, "F=7 over patches of 3 pads to 9 and gives 3"
    _record("v06", "tps_read_from_processor", video_consts.temporal_patch_size)


# ---------------------------------------------------------------------------
# V07 -- report every reading.
# ---------------------------------------------------------------------------
def test_v07_report_the_measured_readings(
    image_oracle, video_oracle, image_consts, video_consts, capsys
):
    """One place a person can read what this run actually measured."""
    import transformers
    import vllm_neuron

    _record(
        "env",
        "transformers", transformers.__version__,
        "module", vllm_neuron.__file__,
        "torch", torch.__version__,
        "temporal_patch_size", video_consts.temporal_patch_size,
        "merge_length", video_consts.merge_length,
        "factor", video_consts.factor,
    )
    lines = [
        "",
        "inc-glm53f-061 -- video path readings",
        f"  module resolved from   {vllm_neuron.__file__}",
        f"  transformers reference {transformers.__version__}",
        f"  temporal_patch_size {video_consts.temporal_patch_size}, "
        f"merge_length {video_consts.merge_length}, canvas factor {video_consts.factor}",
        "",
        f"  criterion 1 -- token count, registered at F={REGISTERED_FRAME_COUNTS} "
        f"(extra coverage {EXTRA_FRAME_COUNTS}), item {FRAME_ARM_HEIGHT}x{FRAME_ARM_WIDTH}",
    ]
    for num_frames in REGISTERED_FRAME_COUNTS + EXTRA_FRAME_COUNTS:
        spec = video_grid_spec(video_consts, num_frames, FRAME_ARM_HEIGHT, FRAME_ARM_WIDTH)
        closed = _closed_form_grid_t(num_frames, video_consts.temporal_patch_size)
        budget = _budget_form_grid_t(video_consts, num_frames)
        mark = "registered" if num_frames in REGISTERED_FRAME_COUNTS else "extra"
        lines.append(
            f"    F={num_frames:<3} ({mark:<10}) grid_t {spec.grid_t} "
            f"(closed form {closed}, recorded {RECORDED_GRID_T[num_frames]}, "
            f"budget-form decoy {budget}), {spec.num_merged_tokens} tokens over "
            f"{spec.num_patch_rows} patch rows, regime {spec.regime}"
        )
    lines += [
        "",
        "  criterion 2 -- 1-frame bit-identity, over -056's seven cases",
    ]
    for name, height, width, _regime in R2_CASES:
        image_spec, video_spec, image_record, video_record = _both_specs(
            image_consts, video_consts, height, width, num_frames=1
        )
        verdict = "grids agree" if image_spec.thw == video_spec.thw else "GRIDS DIVERGE"
        lines += [
            f"    {name} ({height}x{width}, regime {image_spec.regime}): {verdict}",
            f"      still image     canvas {image_spec.canvas_height}x"
            f"{image_spec.canvas_width}, grid {image_spec.grid_h}x{image_spec.grid_w}, "
            f"{image_spec.num_patch_rows} rows, content "
            f"{image_record.content_height}x{image_record.content_width}"
            f"{' (resampled)' if image_record.resampled else ' (padded only)'}",
            f"      1-frame video   canvas {video_spec.canvas_height}x"
            f"{video_spec.canvas_width}, grid {video_spec.grid_h}x{video_spec.grid_w}, "
            f"{video_spec.num_patch_rows} rows, content "
            f"{video_record.content_height}x{video_record.content_width}"
            f"{' (resampled)' if video_record.resampled else ' (padded only)'}",
        ]
        if name == R2_TOO_BIG_FOR_REAL_PIXELS:
            lines.append(
                "      real pixels     skipped, 144 million pixels; the arithmetic arm ran"
            )
            continue
        image_pixels = _run_image_oracle(image_oracle, height, width)["pixel_values"]
        video_pixels = _run_video_oracle(video_oracle, 1, height, width)[
            "pixel_values_videos"
        ]
        if tuple(image_pixels.shape) == tuple(video_pixels.shape):
            diff = (image_pixels.double() - video_pixels.double()).abs().max().item()
            lines.append(
                f"      real pixels     both {tuple(image_pixels.shape)}, "
                f"max abs diff {diff:.3e} (asserted == "
                f"{BIT_IDENTICAL_MAX_ABS_DIFF})"
            )
        else:
            lines.append(
                f"      real pixels     image {tuple(image_pixels.shape)} vs video "
                f"{tuple(video_pixels.shape)}; shapes differ, so no elementwise difference "
                "exists to report"
            )
        _record(
            "v07", name,
            "image_thw", ",".join(str(v) for v in image_spec.thw),
            "video_thw", ",".join(str(v) for v in video_spec.thw),
            "grids_equal", image_spec.thw == video_spec.thw,
        )
    with capsys.disabled():
        print("\n".join(lines))
