# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ``inc-glm53f-061`` -- WP10: the video path.

THE REGISTERED CRITERION, VERBATIM. Copied from ``design/increment-plan.md`` line 1334 at plan
revision 262, whose bytes are
``c71ff306583ed57b7e92e50b14350b9b2edc8e1d7a7e1e6b2caec414e7dc9249``. No criteria pin was minted
for ``-061``, so the plan digest is the pin this file cites; the acceptance driver greps the quote
below as a fixed string. Markdown emphasis, arrows and all -- a transliteration would let the quote
drift from the criterion while still reading correctly:

**Expected:** (1) an F-frame video packs to `grid_t == ceil(F / temporal_patch_size)` and to the closed-form token count exactly in **5/5** frame counts F ∈ {1, 2, 3, 8, 17} with `do_sample_frames` pinned False (17 is KEPT: it separates the grid form from the `aligned_frames // tps` decoy, which agree at 1, 2, 3, 8 and differ at 5 and 17); (2) in regime A AND `h·w ≥ min_image_tokens · pixels_per_token` (25088), a 1-frame video produces a token sequence **bit-identical** to the same content as a still image (max abs diff == 0.0) — **4/4** unclamped cases (aligned_to_both_factors, aligned_to_28_only, odd_mid_size, wide_strip); (3) the reference's divergences are ASSERTED by counterfactual on `resolve_canvas`'s existing parameters — **4/4**: below the floor the video canvas is strictly larger (60x40 → 140x112 vs 196x140; 32x32 → 112x112 vs 168x168) and the same call told 2 frames makes the divergence vanish (140x112 / 112x112 — the raw-frame-count rescue on `smart_resize`'s floor line); above the image ceiling the still path is resampled (regime C, 2884x2156) and the video path is not (regime A, 8008x6020), and the image at the video ceiling gives 8008x6020 regime A (`max_image_tokens` 8000 vs 240000); in the band 12544 ≤ h·w < 25088 with a padded canvas (130x130 → 140x140, regime A both) the image is padded and the 1-frame video is up-scaled — same grid, same rows, DIFFERENT pixels (the second raw-frame-count read in upstream's resize tail; the fork mirrors it in `describe_resample`); (4) per-modality oracle parity: the fork's image grid equals the upstream image processor's and the fork's video grid equals the upstream video processor's in **5/5** registered video cases (B04 F=3 112x112, B05 F=8 112x112, F=1 112x112, F=2 112x112, F=1 60x40 floor — real pixels, `do_sample_frames` False) plus **1/1** lowered-ceiling pair: `Glm5NextImageProcessor(max_image_tokens=N)` and `Glm5NextVideoProcessor(max_image_tokens=N)` at a small size with `GridConstants.from_processor` driving `image_grid_spec` / `video_grid_spec` end to end at regime C against those same instances (zero production change; the default ceilings stay covered by the arithmetic-only row and `-056`'s landed differing-ceilings assertion); (5) controls **2/2**: a 3-frame video is NOT bit-identical to the still (grid_t 2, 2048 vs 1024 rows — F=2 cannot fire), and the 130x130 F=1 band case is same-shape/different-pixels. **A cross-modality equality outside (2)'s condition is NOT the criterion**: upstream itself does not produce it. P4 zero over added lines; `PROVENANCE_061=PASS`.

WHERE EACH OF THE FIVE REGISTERED GROUPS IS MEASURED. (1) the 5/5 frame counts are V01, with V02
holding their references and V03 the decoy control; (2) the 4/4 unclamped equality cases are V04;
(3) the 4/4 counterfactual divergences are V04C (the two floor cases), V04D (the ceiling case) and
V04E (the band case); (4) the 5/5 oracle-parity cases and the 1/1 lowered-ceiling pair are V08;
(5) the 2/2 controls are V05A (three frames are not bit-identical) and V05B (the band case's real
pixels). V06 exercises the guards and V07 reports every reading, so no number here is silent.

WHY THIS CRITERION REPLACED THE ONE ``-061`` WAS FIRST GIVEN, AND WHY THAT IS A FINDING RATHER THAN
A RETREAT. Revision 260 asked for a 1-frame video to be bit-identical to the same still image at
every clamp bound. It was measured under grant 105 and 4 of 23 items went red
(``run-061-r11.out``). The reading that mattered is that the failing comparison had no fork code in
it: it ran the two transformers processors against each other. Upstream itself does not produce
that equality, by two mechanisms quoted in ``contradiction-061-r11-run-105.md`` section 5b and
confirmed independently in ``design-061-plan-rev-r1-findings.md``. So the criterion was wrong and
the port was right, and making the fork satisfy the old law would have meant making it disagree
with upstream -- which is the opposite of what this campaign is for.

THIS INCREMENT IS TEST-ONLY, AND ``vision_preprocessing.py`` MUST NOT CHANGE. The ruling that
scoped it refused a packer with no caller and a video branch that would duplicate handling already
landed: ``video_grid_spec`` landed with ``-056`` and per-frame ``compute_attention_bounds`` landed
with ``-060``. Revision 262 adds the stronger statement: the module is FAITHFUL to upstream on both
raw-frame-count reads and on both token ceilings, so an edit there would be the defect and not the
fix. Nothing is authored here to give a file surface a body.

THE TWO MECHANISMS, WHICH ARE THE WHOLE SUBJECT OF PARTS (2), (3) AND (5). Upstream hands its image
resizer ``num_frames=temporal_factor`` and its video resizer the real frame count, and the raw
count then reaches the answer in exactly two places:

* the BELOW-FLOOR RESCUE inside ``smart_resize``, whose scale is
  ``sqrt(min_pixels / (num_frames * h * w))`` -- every other branch of that function measures the
  budget over ``aligned_frames``, which maps one frame to two. This is why a 1-frame video below
  the floor gets a strictly LARGER canvas than the same still image;
* the RESIZE TAIL, one level up, where the image asks
  ``temporal_factor * h * w >= pixels_per_token * min_image_tokens``
  (``image_processing_glm5_next.py`` line 174) and the video asks the same question of
  ``videos.shape[1] * h * w`` (``video_processing_glm5_next.py`` line 265). When the answer differs
  the image's scale is capped at 1.0 and the video's is not, so the image is PADDED with a zero
  border while the video is RESAMPLED to fill the canvas -- same canvas, same grid, same row count,
  different pixels.

The two ceilings are the third fact and belong to neither mechanism: the image processor declares
``max_image_tokens`` 8000 and the video processor 240000, so a size above the image ceiling and
below the video ceiling is regime C on one path and regime A on the other. The frame count explains
none of that, which is the misattribution ``contradiction-061-r11-run-105.md`` section 9 records
and corrects.

WHY PART (2) CARRIES A CONDITION AND NOT JUST "REGIME A". The resize tail above means the
bit-identity law is false inside the band ``floor_pixels / temporal_patch_size <= h*w <
floor_pixels``, where both paths land on the same canvas in regime A and still differ in pixels.
130x130 is such a size, and it is registered as part (3)'s fourth divergence and part (5)'s second
control rather than being excluded and forgotten. Part (2)'s four cases all satisfy
``h*w >= floor_pixels``, and V04 ASSERTS that precondition per case from the constants rather than
trusting the case list, so a case that drifted into the band would fail loudly instead of quietly
becoming a counterexample to the law it is meant to demonstrate.

WHY PART (3) IS A COUNTERFACTUAL AND NOT A SENTENCE. "The divergence is asserted with its cause"
has no assertable form: a test can satisfy it with a message string, and a future upstream that
diverged for a different reason would still read green. So each divergence is paired with the same
call made once more with one input changed -- the frame count for the floor cases, the ceiling for
the ceiling case -- and the assertion is that the divergence VANISHES. That pins the cause to the
input that produced it, on ``resolve_canvas``'s existing parameters and with no production change.

WHY THE LOWERED-CEILING PAIR EXISTS AND WHY IT NEEDS NO NEW CODE. A video row at the DEFAULT video
ceiling would need more than 376 million pixels, which is over a gigabyte of uint8 and not worth
the reading. Both processor base classes set every declared init keyword as an attribute, so
constructing the two processors with ``max_image_tokens=N`` lowers the ceiling on the instances
themselves; ``GridConstants.from_processor`` then reads N, and the two grid functions run end to
end at regime C against those same instances as the oracle. V08 asserts the keyword actually
reached the attribute before it measures anything, because a kwarg that was silently ignored would
leave the test measuring the default ceiling and reporting a pass.

WHAT IS CONSUMED AND WHAT IS UNDER TEST. Ruling ``design-20260905-aq`` (iii) forbids re-deriving
the grid in a test. So the SPATIAL factors are consumed from ``-056``'s landed module and never
recomputed here; the one quantity this file states as a closed form is the FRAME factor, which is
the increment's own subject. The form is the plan rider's grid form ``grid_t = (F + (-F % tps)) //
tps``, with ``tps`` read off the processor and never typed as 2. Every threshold -- the floor
budget, the band edges, the ceilings -- is read off ``GridConstants`` and never typed as an
integer.

THE ROUNDING DECOY, AND WHY V03 EXISTS. The module carries two roundings of the same quantity, and
they are not interchangeable. ``GridConstants.aligned_frames`` rounds to the NEAREST whole temporal
patch, which is what the PIXEL BUDGET is measured over; the grid pads UP by ``-F % tps``. They
agree at F=1, 2, 3 and 8 -- every registered count except one -- and they disagree at F=5 and F=17,
where nearest-rounding gives 2 and 8 where the grid gives 3 and 9. This is why revision 262 KEPT
F=17 after the first draft dropped it: without 17 the whole registered set sits where the two forms
agree, and a ``video_grid_spec`` that used the budget form would pass part (1) 5/5 and part (4) 5/5
against the real oracle. V03 asserts the two forms agree where they must and disagree where they
must, so this file is on the record as measuring the grid form.

WHERE THE FRAME-COUNT REFERENCES COME FROM, AND WHY F=8's IS DIFFERENT. Five ``grid_t`` points were
read from the real processor on the leased host and filed as ``read-056-r3-closedform.out`` lines
68-76, whose bytes are
``c5603c91f0e091b7386e95b4851588873977b7cc1a52a245140f93780a261700``. F=8 is not among them: its
reference is ``-056``'s landed ``B05_eight_frames`` row, itself pinned to the real video processor
by ``test_c02_video_grid_matches_oracle_and_closed_form``. Both sources are named per frame count in
``RECORDED_GRID_T_SOURCE`` and V02 prints them, so no reference is anonymous and none is recomputed
here.

THE SEVEN CASES ARE ``-056``'s R2 SET, NOW PARTITIONED RATHER THAN NARROWED. Their sizes and regimes
are quoted from ``test_vision_preprocessing.py``, which read every integer from
``derive-056r2-divergence.out``. Revision 262 splits them: four are part (2)'s equality cases and
three are part (3)'s divergences. V04B asserts the partition is a partition -- every case in exactly
one side, all seven accounted for -- because narrowing a case set is precisely how a criterion gets
hollowed out, and the check makes that impossible to do quietly.

WHERE THE NUMBERS IN THIS FILE'S TABLES CAME FROM. Every canvas, grid, row count and token count
below was computed from upstream's own ``smart_resize`` and resize-tail text by
``derive-061-registered-values-r1.py`` and printed in
``derive-061-registered-values-r1.out``, whose bytes are
``cc3b82bdcc9bc53198a17b2292e2b6a8cf92df53e8cca43ccf24af90da59afa6``, ``CHECKS=88 PASSED=88
FAILED=0``. That derivation reproduces all 28 canvas and content readings already filed in
``run-061-r11.out`` as its own check, so a value here that was mistyped shows up as a mismatch
against a world transcript rather than as a confident wrong expectation.

WHEN THE SHAPES DIFFER THERE IS NO MAX ABS DIFF. Two tensors of different row counts have no
elementwise difference at all, so no test here attempts one. Each pixel comparison gates on shape
first and reports both shapes, then computes the difference only where the shapes agree. A test that
subtracted them regardless would report an error where the reading is a measurement.

NO CANDIDATE-ROOT REQUIREMENT. V06's origin guard has two arms and never skips: with
``GLM53F_CANDIDATE_ROOT`` set the import must resolve under the declared tree, and unset the tree
is derived from this file's own location, which is how a plain ``pytest`` from a checkout runs. It
is the form landed in ``test_vision_preprocessing.py`` and it passes in both partitions.
"""

from __future__ import annotations

import ast
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
    resolve_canvas,
    video_grid_spec,
)

# ---------------------------------------------------------------------------
# Part (1): the frame counts. 5/5 at plan revision 262.
# ---------------------------------------------------------------------------
#: The five REGISTERED frame counts. 1 and 2 are the counts a 1-frame reading depends on, 3 pads up
#: by one frame, 8 is exact and large, and 17 is the one count where the budget-form decoy gives a
#: different answer. Every one has a recorded reference in ``RECORDED_GRID_T``.
REGISTERED_FRAME_COUNTS = (1, 2, 3, 8, 17)

#: Extra coverage, not a registered criterion: one even count that pads by nothing and has a
#: recorded reference. It is reported and asserted alongside the five and changes no registered
#: number.
EXTRA_FRAME_COUNTS = (16,)

#: ``grid_t`` per frame count, every value read from the world and never recomputed here.
RECORDED_GRID_T = {1: 1, 2: 1, 3: 2, 8: 4, 16: 8, 17: 9}

#: Which world reading each point came from. F=8 is the odd one out and says so.
RECORDED_GRID_T_SOURCE = {
    1: "read-056-r3-closedform.out lines 68-76 (real processor, leased host)",
    2: "read-056-r3-closedform.out lines 68-76 (real processor, leased host)",
    3: "read-056-r3-closedform.out lines 68-76 (real processor, leased host)",
    8: "test_vision_preprocessing.py REGISTERED_VIDEO_CASES B05_eight_frames, oracle-pinned by C02",
    16: "read-056-r3-closedform.out lines 68-76 (real processor, leased host)",
    17: "read-056-r3-closedform.out lines 68-76 (real processor, leased host)",
}

#: The distinct ``grid_t`` the five registered counts produce. It is FOUR and not five, because the
#: pad-up form maps both F=1 and F=2 to one frame pair -- which is the form's own behaviour and the
#: reason F=1 and F=2 are both worth registering. Named as a set so the spread cannot shrink
#: silently: a form that collapsed 3 and 8 together would fail here.
REGISTERED_GRID_T_VALUES = {1, 2, 4, 9}

#: The counts where the budget form and the grid form MUST agree, and the counts where they must
#: differ, with both answers named. F=17's pair is the module's own docstring and the plan; F=5's is
#: ``design-061-plan-rev-r1-findings.md`` and ``-056``'s ``test_m05``, which sweeps F=5 against the
#: real video processor and so pins the grid form's 3 in the world.
DECOY_AGREEMENT_FRAME_COUNTS = (1, 2, 3, 8)
DECOY_DISAGREEMENT_FRAME_COUNTS = {5: (2, 3), 17: (8, 9)}  # F -> (budget form, grid form)

#: The frame-count arm's item size. 112x112 is regime A on this checkpoint, so its canvas does not
#: move with the frame count and the arm reads the frame factor alone. It is the size ``-056``'s
#: own registered video cases use.
FRAME_ARM_HEIGHT = 112
FRAME_ARM_WIDTH = 112

# ---------------------------------------------------------------------------
# The seven cases, from ``-056``'s R2 set, partitioned by revision 262.
# (name, height, width, regime)
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

#: Part (2): the 4/4 cases where the bit-identity law holds. Named, not filtered by regime, because
#: the law's condition is regime A AND ``h*w >= floor_pixels`` and V04 asserts both per case.
UNCLAMPED_EQUALITY_CASE_NAMES = (
    "aligned_to_both_factors",
    "aligned_to_28_only",
    "odd_mid_size",
    "wide_strip",
)

#: The three clamped cases of the seven, which diverge and are asserted by counterfactual. This is NOT
#: part (3)'s registered count: part (3) is 4/4, because its fourth divergence is the band case, which is
#: 130x130 and is not one of ``-056``'s seven.
CLAMPED_DIVERGENCE_CASE_NAMES = ("tiny_floor_bound", "tiny_square", "huge_ceiling_bound")

#: Part (3) cases 1 and 2. (name, h, w, image canvas, video canvas at F=1). The counterfactual is
#: the same video call told ``temporal_patch_size`` frames, which must land on the image canvas.
FLOOR_COUNTERFACTUAL_CASES = (
    ("tiny_floor_bound", 60, 40, (140, 112), (196, 140)),
    ("tiny_square", 32, 32, (112, 112), (168, 168)),
)

#: Part (3) case 3. Above the image ceiling and below the video ceiling: regime C against regime A.
#: The counterfactual is the image canvas recomputed at the VIDEO ceiling, which must land on the
#: video canvas and read regime A.
CEILING_CASE = ("huge_ceiling_bound", 8000, 6000)
CEILING_IMAGE_CANVAS = (2884, 2156)
CEILING_VIDEO_CANVAS = (8008, 6020)

#: Part (3) case 4 and part (5) control 2: the band, where both paths agree on the canvas, the grid
#: and the row count and still differ in pixels. The image is padded with a zero border and the
#: video is resampled to fill.
BAND_CASE = ("band_same_grid_different_pixels", 130, 130)
BAND_CANVAS = (140, 140)
BAND_GRID_HW = (10, 10)
BAND_PATCH_ROWS = 100
BAND_IMAGE_CONTENT = (130, 130)
BAND_VIDEO_CONTENT = (140, 140)

#: Part (5) control 1: three frames at the first unclamped size. F=2 cannot fire this control,
#: because two frames take ``grid_t`` 1 and the same canvas as one still image, so the shapes are
#: necessarily equal -- which V05A asserts rather than merely stating.
SHAPE_CONTROL_HEIGHT = 448
SHAPE_CONTROL_WIDTH = 448
SHAPE_CONTROL_FRAMES = 3
SHAPE_CONTROL_IMAGE_ROWS = 1024
SHAPE_CONTROL_VIDEO_ROWS = 2048

# ---------------------------------------------------------------------------
# Part (4): oracle parity over ``-056``'s registered video cases, plus the lowered-ceiling pair.
# ---------------------------------------------------------------------------
#: The sibling file that OWNS the registered video case table. Revision 262 widened it from 2 rows
#: to 5 as part of this increment, and this file reads it rather than restating it, so the five
#: registered cases live in exactly one place.
CASES_OWNER_FILENAME = "test_vision_preprocessing.py"
CASES_OWNER_SYMBOL = "REGISTERED_VIDEO_CASES"


def _read_registered_video_cases():
    """Read ``-056``'s registered video case table out of its source, without importing it.

    An AST read rather than an import for two reasons: importing another test module to borrow a
    constant makes this file's collection depend on that file's collection, and the table is a plain
    literal, so nothing needs to run for it to be read. A missing or non-literal table raises here,
    at import, rather than producing an empty parametrisation that would read as a pass over zero
    cases.
    """
    source_path = Path(__file__).with_name(CASES_OWNER_FILENAME)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == CASES_OWNER_SYMBOL for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(
        f"{CASES_OWNER_FILENAME} has no module-level {CASES_OWNER_SYMBOL} assignment, so part (4)'s "
        "registered cases could not be read from the file that owns them"
    )


#: (name, num_frames, height, width, grid_thw, patch rows, merged tokens), read from the owner.
REGISTERED_VIDEO_CASES = _read_registered_video_cases()

#: The drift guard for the line above, and NOT a second source of truth. Part (4) registers five
#: named cases; if the owner's table stops being exactly these five, this file must not quietly
#: measure a different set under the same registered count.
EXPECTED_REGISTERED_VIDEO_CASES = (
    ("B04_three_frames", 3, 112, 112, (2, 8, 8), 128, 32),
    ("B05_eight_frames", 8, 112, 112, (4, 8, 8), 256, 64),
    ("B07_one_frame", 1, 112, 112, (1, 8, 8), 64, 16),
    ("B08_two_frames", 2, 112, 112, (1, 8, 8), 64, 16),
    ("B09_one_frame_below_floor", 1, 60, 40, (1, 14, 10), 140, 35),
)

#: The lowered ceiling, in merged tokens, and the size measured at it. 64 tokens is 100352 pixels,
#: and 448x448 over two aligned frames is 401408, so both paths are above the ceiling and the
#: processors' own budget search decides the canvas. The size is small enough to feed real pixels.
LOWERED_CEILING_TOKENS = 64
LOWERED_CEILING_HEIGHT = 448
LOWERED_CEILING_WIDTH = 448
LOWERED_CEILING_CANVAS = (224, 224)
LOWERED_CEILING_GRID_HW = (16, 16)
LOWERED_CEILING_PATCH_ROWS = 256

#: The one case whose REAL-PIXEL arm is skipped, for the reason ``-056`` skipped it: an 8000x6000
#: uint8 image is 144 million pixels and resampling it twice on CPU costs more than the reading is
#: worth. Its ARITHMETIC arm still runs, so the case is not dropped from the seven.
R2_TOO_BIG_FOR_REAL_PIXELS = "huge_ceiling_bound"

#: The constant pixel value both paths are fed, and a second value the extra content control uses to
#: prove the comparison reads content and not only shape. Mid-range, so neither can be confused with
#: the zero padding.
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


def _case(name):
    """One row of the seven-case set by name, so a table reorder cannot change what a test measures."""
    for row in R2_CASES:
        if row[0] == name:
            return row
    raise AssertionError(f"{name} is not one of -056's seven R2 cases")


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
# V01 -- part (1): the token count equals the closed form, in 5/5 frame counts.
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
        f"{RECORDED_GRID_T[num_frames]} recorded from the world in "
        f"{RECORDED_GRID_T_SOURCE[num_frames]}"
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


def test_v01_the_five_registered_counts_spread_over_four_grid_shapes(video_consts):
    """5/5 would be worth little if the five counts produced one answer five times.

    The spread is FOUR and not five on purpose: the pad-up form maps F=1 and F=2 onto the same
    single frame pair, which is exactly why both are registered -- a 1-frame video and a 2-frame
    video are the two inputs a cross-modality reading can confuse. The four values are named, so a
    form that collapsed any OTHER pair would fail here rather than reading as a green 5/5.
    """
    grid_ts = {
        _closed_form_grid_t(f, video_consts.temporal_patch_size) for f in REGISTERED_FRAME_COUNTS
    }
    _record("v01_spread", "counts", len(REGISTERED_FRAME_COUNTS), "distinct_grid_t", sorted(grid_ts))
    assert grid_ts == REGISTERED_GRID_T_VALUES, (
        f"the five registered frame counts give grid_t {sorted(grid_ts)}, not the registered "
        f"{sorted(REGISTERED_GRID_T_VALUES)}"
    )
    assert (
        _closed_form_grid_t(1, video_consts.temporal_patch_size)
        == _closed_form_grid_t(2, video_consts.temporal_patch_size)
    ), "F=1 and F=2 must share one frame pair; if they stopped sharing it, the spread above is wrong"


# ---------------------------------------------------------------------------
# V02 -- the references the closed form is checked against.
# ---------------------------------------------------------------------------
def test_v02_every_registered_count_has_a_recorded_reference_and_a_named_source():
    """No registered frame count may be checked against the closed form alone.

    The closed form is authored in this file. Without a world reading beside it, V01 would be
    comparing the module to a rule this file made up. The source is asserted as well as the value,
    because F=8's reference comes from a different reading than the other four and an unnamed
    reference cannot be checked by anyone.
    """
    for num_frames in REGISTERED_FRAME_COUNTS:
        assert num_frames in RECORDED_GRID_T, (
            f"F={num_frames} is registered but has no recorded reference"
        )
        source = RECORDED_GRID_T_SOURCE.get(num_frames, "")
        assert source.strip(), f"F={num_frames} has a recorded value with no named source"
        _record("v02_source", "frames", num_frames, "grid_t", RECORDED_GRID_T[num_frames],
                "source", source)


def test_v02_the_recorded_points_agree_with_the_closed_form(video_consts):
    """All six recorded points, registered and extra, must satisfy the grid form.

    This is the closed form's own evidence: six readings from the world, none of them produced by
    this file.
    """
    for num_frames, recorded in sorted(RECORDED_GRID_T.items()):
        derived = _closed_form_grid_t(num_frames, video_consts.temporal_patch_size)
        _record("v02", "frames", num_frames, "recorded", recorded, "closed_form", derived)
        assert derived == recorded, (
            f"F={num_frames}: the closed form gives {derived}, the world recorded {recorded} in "
            f"{RECORDED_GRID_T_SOURCE[num_frames]}"
        )


def test_v02_f8s_reference_is_the_landed_row_it_claims_to_be():
    """F=8's recorded ``grid_t`` must be the one ``-056``'s landed table actually holds.

    This is the check that keeps ``RECORDED_GRID_T_SOURCE`` honest for the one point whose source is
    another file: if B05's row ever changed, F=8 would still read 4 here and the source string would
    be a claim about a row that no longer says it.
    """
    b05 = [row for row in REGISTERED_VIDEO_CASES if row[0] == "B05_eight_frames"]
    assert b05, "B05_eight_frames is gone from the owner's table, so F=8 has no reference"
    name, num_frames, _height, _width, grid, _rows, _tokens = b05[0]
    _record("v02_f8", "row", name, "frames", num_frames, "grid_t", grid[0])
    assert num_frames == 8, f"{name} is no longer an 8-frame row; it is F={num_frames}"
    assert grid[0] == RECORDED_GRID_T[8], (
        f"{name} records grid_t {grid[0]} where this file's reference for F=8 says "
        f"{RECORDED_GRID_T[8]}"
    )


# ---------------------------------------------------------------------------
# V03 -- control: the budget form and the grid form agree and disagree where they must.
# ---------------------------------------------------------------------------
def test_v03_the_budget_rounding_is_not_the_grid_rounding(video_consts):
    """The decoy control, over the whole registered set and not one point.

    The module carries both roundings deliberately and documents why. Four of the five registered
    counts sit where the two forms AGREE, so without F=17 the registered set could not tell them
    apart at all. This asserts the agreement where it is expected and the difference where it shows,
    with every answer named, so V01's green is known to be the grid form's green.
    """
    for num_frames in DECOY_AGREEMENT_FRAME_COUNTS:
        budget = _budget_form_grid_t(video_consts, num_frames)
        grid = _closed_form_grid_t(num_frames, video_consts.temporal_patch_size)
        _record("v03_agree", "frames", num_frames, "budget_form", budget, "grid_form", grid)
        assert budget == grid, (
            f"the two roundings differ at F={num_frames}, where this file records them as agreeing; "
            "the registered set's decoy separation is stated by the disagreement points below"
        )

    for num_frames, (expect_budget, expect_grid) in sorted(
        DECOY_DISAGREEMENT_FRAME_COUNTS.items()
    ):
        budget = _budget_form_grid_t(video_consts, num_frames)
        grid = _closed_form_grid_t(num_frames, video_consts.temporal_patch_size)
        _record(
            "v03_disagree",
            "frames", num_frames,
            "budget_form", budget,
            "grid_form", grid,
            "expect_budget", expect_budget,
            "expect_grid", expect_grid,
        )
        assert budget == expect_budget, (
            f"the budget form gave {budget} at F={num_frames}, not the {expect_budget} the "
            "module's nearest-rounding documents"
        )
        assert grid == expect_grid, (
            f"the grid form gave {grid} at F={num_frames}, not {expect_grid}"
        )
        assert budget != grid, (
            f"the two roundings agree at F={num_frames}, so V01 could not tell the pixel budget's "
            "nearest-rounding from the grid's pad-up rounding"
        )

    assert 17 in DECOY_DISAGREEMENT_FRAME_COUNTS and 17 in REGISTERED_FRAME_COUNTS, (
        "F=17 must stay registered AND be a disagreement point: it is the only registered count "
        "that separates the grid form from the budget form"
    )


# ---------------------------------------------------------------------------
# V04 -- part (2): 4/4 unclamped cases where a 1-frame video IS bit-identical to the still image.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", UNCLAMPED_EQUALITY_CASE_NAMES)
def test_v04_one_frame_video_is_bit_identical_to_the_still_image(
    image_oracle, video_oracle, image_consts, video_consts, name
):
    """One unclamped case: the two paths' grids are equal and their pixel tensors differ by 0.0.

    The law's CONDITION is asserted before the law: regime A on both paths, and a raw-frame pixel
    count at or above the floor budget. Both are read from the constants rather than trusted from
    the case list, because a case that drifted into the band below the floor budget would be a
    counterexample to the law rather than an instance of it, and would then read as a defect in the
    port.
    """
    _name, height, width, regime = _case(name)
    image_spec, video_spec, image_record, video_record = _both_specs(
        image_consts, video_consts, height, width, num_frames=1
    )
    raw_pixels = height * width
    _record("v04", "case", name, "size", f"{height}x{width}",
            "raw_pixels", raw_pixels, "floor_pixels", video_consts.floor_pixels,
            *_paths_report(image_spec, video_spec, image_record, video_record, num_frames=1))

    assert image_spec.regime == regime, (
        f"{name}: the landed module calls {height}x{width} regime {image_spec.regime}, but this "
        f"file's case table quotes regime {regime}. The table is stale."
    )
    assert image_spec.regime == REGIME_PLAIN and video_spec.regime == REGIME_PLAIN, (
        f"{name}: part (2) is stated for regime A on both paths, and this case reads "
        f"{image_spec.regime} / {video_spec.regime}"
    )
    assert raw_pixels >= video_consts.floor_pixels, (
        f"{name}: {height}x{width} is {raw_pixels} raw pixels, below the floor budget "
        f"{video_consts.floor_pixels}. Inside that band upstream's resize tail pads the image and "
        "resamples the video, so bit-identity is FALSE of the reference here and this case does not "
        "belong to part (2). See the band case."
    )

    grid_equal = image_spec.thw == video_spec.thw
    _record(
        "v04_grid", name,
        "image_thw", ",".join(str(v) for v in image_spec.thw),
        "video_thw", ",".join(str(v) for v in video_spec.thw),
        "equal", grid_equal,
    )

    image_pixels = _run_image_oracle(image_oracle, height, width)["pixel_values"]
    video_pixels = _run_video_oracle(video_oracle, 1, height, width)["pixel_values_videos"]
    shapes_equal = tuple(image_pixels.shape) == tuple(video_pixels.shape)
    _record(
        "v04_pixels", name,
        "image_shape", "x".join(str(v) for v in image_pixels.shape),
        "video_shape", "x".join(str(v) for v in video_pixels.shape),
        "shapes_equal", shapes_equal,
    )
    assert shapes_equal, (
        f"{name}: the two paths produced different token counts inside part (2)'s condition, so "
        f"there is no elementwise difference to measure. The image path is "
        f"{tuple(image_pixels.shape)} on a "
        f"{image_spec.canvas_height}x{image_spec.canvas_width} canvas, the 1-frame video path is "
        f"{tuple(video_pixels.shape)} on a "
        f"{video_spec.canvas_height}x{video_spec.canvas_width} canvas. Both are regime A and at or "
        f"above the floor budget, so the frame count should not have reached the canvas at all."
    )

    max_abs_diff = (image_pixels.double() - video_pixels.double()).abs().max().item()
    _record("v04_diff", name, "max_abs_diff", f"{max_abs_diff:.6e}")
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


def test_v04_the_seven_cases_are_partitioned_and_none_was_dropped():
    """Revision 262 split the seven cases; this asserts the split is a partition.

    Narrowing a case set is how a criterion gets hollowed out, so the equality-case names and the
    divergence-case names must together be exactly the seven, with no case in both and none left
    out. Without this check, part (2) could be made green by deleting a case rather than by the port
    being right.
    """
    all_names = {case[0] for case in R2_CASES}
    equality = set(UNCLAMPED_EQUALITY_CASE_NAMES)
    divergence = set(CLAMPED_DIVERGENCE_CASE_NAMES)
    _record("v04_partition", "seven", len(all_names), "equality", len(equality),
            "divergence", len(divergence))
    assert len(R2_CASES) == 7, f"expected -056's seven-case set, got {len(R2_CASES)}"
    assert not equality & divergence, (
        f"these cases are in both halves of the partition: {sorted(equality & divergence)}"
    )
    assert equality | divergence == all_names, (
        f"the partition misses {sorted(all_names - (equality | divergence))} and invents "
        f"{sorted((equality | divergence) - all_names)}"
    )
    assert len(equality) == 4 and len(divergence) == 3, (
        f"part (2) registers 4 equality cases and part (3) 3 clamped cases; this file holds "
        f"{len(equality)} and {len(divergence)}"
    )
    regimes = {case[3] for case in R2_CASES}
    assert regimes == {REGIME_PLAIN, REGIME_BELOW_FLOOR, REGIME_ABOVE_CEILING}, (
        f"the seven cases cover only {sorted(regimes)}"
    )


# ---------------------------------------------------------------------------
# V04C -- part (3) cases 1 and 2: the floor divergence, and the counterfactual that removes it.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,height,width,image_canvas,video_canvas",
    FLOOR_COUNTERFACTUAL_CASES,
    ids=[c[0] for c in FLOOR_COUNTERFACTUAL_CASES],
)
def test_v04c_below_the_floor_the_video_canvas_is_larger_and_the_frame_count_is_why(
    image_consts, video_consts, name, height, width, image_canvas, video_canvas
):
    """Below the floor a 1-frame video gets a strictly larger canvas, and the frame count is the cause.

    The cause is asserted, not narrated: the SAME call is made once more with the frame count
    changed to ``temporal_patch_size`` -- which is exactly what upstream's image path passes -- and
    the divergence must vanish onto the image canvas. Nothing but the frame count moves between the
    two video calls, so if the divergence survived, the frame count would not be the cause and this
    file's account of the mechanism would be wrong.
    """
    tps = video_consts.temporal_patch_size
    got_image = resolve_canvas(
        image_consts, num_frames=image_consts.temporal_patch_size, height=height, width=width
    )
    got_video = resolve_canvas(video_consts, num_frames=1, height=height, width=width)
    counterfactual = resolve_canvas(video_consts, num_frames=tps, height=height, width=width)

    _record(
        "v04c", name, "size", f"{height}x{width}",
        "image_canvas", f"{got_image[0]}x{got_image[1]}", "image_regime", got_image[2],
        "video_canvas", f"{got_video[0]}x{got_video[1]}", "video_regime", got_video[2],
        "counterfactual_frames", tps,
        "counterfactual_canvas", f"{counterfactual[0]}x{counterfactual[1]}",
        "counterfactual_regime", counterfactual[2],
    )

    assert (got_image[0], got_image[1]) == image_canvas, (
        f"{name}: the still path's canvas is {got_image[0]}x{got_image[1]}, registered as "
        f"{image_canvas[0]}x{image_canvas[1]}"
    )
    assert (got_video[0], got_video[1]) == video_canvas, (
        f"{name}: the 1-frame video's canvas is {got_video[0]}x{got_video[1]}, registered as "
        f"{video_canvas[0]}x{video_canvas[1]}"
    )
    assert got_image[2] == REGIME_BELOW_FLOOR and got_video[2] == REGIME_BELOW_FLOOR, (
        f"{name}: this case is registered as below the floor on both paths and reads "
        f"{got_image[2]} / {got_video[2]}"
    )
    assert got_video[0] > got_image[0] and got_video[1] > got_image[1], (
        f"{name}: the 1-frame video's canvas {got_video[0]}x{got_video[1]} is not strictly larger "
        f"than the still image's {got_image[0]}x{got_image[1]}. Upstream's below-floor rescue "
        "scales by sqrt(floor_pixels / (num_frames * h * w)) on the RAW frame count, so one frame "
        "must ask for more upscaling than two."
    )
    assert (counterfactual[0], counterfactual[1]) == (got_image[0], got_image[1]), (
        f"{name}: told {tps} frames instead of 1, the video path still lands on "
        f"{counterfactual[0]}x{counterfactual[1]} rather than the still image's "
        f"{got_image[0]}x{got_image[1]}. The frame count is then NOT the cause of this divergence, "
        "and the mechanism this file records is wrong."
    )


# ---------------------------------------------------------------------------
# V04D -- part (3) case 3: the ceiling divergence, and the counterfactual that removes it.
# ---------------------------------------------------------------------------
def test_v04d_above_the_image_ceiling_the_two_ceilings_are_why(image_consts, video_consts):
    """The still path is cut down and the video path is not, because the two ceilings differ.

    The frame count explains none of this case, which is the misattribution
    ``contradiction-061-r11-run-105.md`` section 9 corrects. So the counterfactual changes the
    CEILING and not the frame count: the still path recomputed at the video ceiling must land on
    the video canvas and read regime A. This arm is arithmetic only -- 8000x6000 is 144 million
    pixels and resampling it twice on CPU costs more than the reading is worth -- and the two grid
    functions are pinned to real pixels at a lowered ceiling by V08 instead.
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

    _record(
        "v04d", name, "size", f"{height}x{width}",
        "image_max_image_tokens", image_consts.max_image_tokens,
        "video_max_image_tokens", video_consts.max_image_tokens,
        "image_canvas", f"{got_image[0]}x{got_image[1]}", "image_regime", got_image[2],
        "video_canvas", f"{got_video[0]}x{got_video[1]}", "video_regime", got_video[2],
        "counterfactual_canvas", f"{counterfactual[0]}x{counterfactual[1]}",
        "counterfactual_regime", counterfactual[2],
        "real_pixels", "skipped_144_million_pixels_costs_more_than_the_reading",
    )

    assert video_consts.max_image_tokens != image_consts.max_image_tokens, (
        "the two processors declare the same max_image_tokens, so this case has no cause to assert "
        "and part (3)'s ceiling divergence cannot exist"
    )
    assert (got_image[0], got_image[1]) == CEILING_IMAGE_CANVAS, (
        f"{name}: the still path's canvas is {got_image[0]}x{got_image[1]}, registered as "
        f"{CEILING_IMAGE_CANVAS[0]}x{CEILING_IMAGE_CANVAS[1]}"
    )
    assert got_image[2] == REGIME_ABOVE_CEILING, (
        f"{name}: the still path reads {got_image[2]}, registered as above the ceiling"
    )
    assert (got_video[0], got_video[1]) == CEILING_VIDEO_CANVAS, (
        f"{name}: the 1-frame video's canvas is {got_video[0]}x{got_video[1]}, registered as "
        f"{CEILING_VIDEO_CANVAS[0]}x{CEILING_VIDEO_CANVAS[1]}"
    )
    assert got_video[2] == REGIME_PLAIN, (
        f"{name}: the video path reads {got_video[2]}, registered as regime A -- 8000x6000 is above "
        "the image ceiling and below the video ceiling"
    )
    assert (counterfactual[0], counterfactual[1]) == CEILING_VIDEO_CANVAS, (
        f"{name}: given the video ceiling, the still path lands on "
        f"{counterfactual[0]}x{counterfactual[1]} rather than the video canvas "
        f"{CEILING_VIDEO_CANVAS[0]}x{CEILING_VIDEO_CANVAS[1]}. The ceiling is then NOT the cause of "
        "this divergence."
    )
    assert counterfactual[2] == REGIME_PLAIN, (
        f"{name}: given the video ceiling, the still path still reads {counterfactual[2]} rather "
        "than regime A, so the ceiling is not what put it above the bound"
    )


# ---------------------------------------------------------------------------
# V04E -- part (3) case 4: the band, where the grids agree and the pixels cannot.
# ---------------------------------------------------------------------------
def test_v04e_in_the_band_the_grids_agree_and_the_content_does_not(image_consts, video_consts):
    """Same canvas, same grid, same row count, different content -- upstream's resize tail.

    This is the case that made revision 260's law false, and it is registered rather than excluded.
    The band is stated from the constants: at or above ``floor_pixels / temporal_patch_size``, so
    the image's tail test passes, and below ``floor_pixels``, so the 1-frame video's tail test does
    not. The image is then padded with a zero border and the video is resampled to fill the canvas.
    Only the content is asserted here; V05B measures the resulting pixel difference.
    """
    name, height, width = BAND_CASE
    tps = video_consts.temporal_patch_size
    raw_pixels = height * width
    image_spec, video_spec, image_record, video_record = _both_specs(
        image_consts, video_consts, height, width, num_frames=1
    )

    _record(
        "v04e", name, "size", f"{height}x{width}",
        "raw_pixels", raw_pixels,
        "band_low", video_consts.floor_pixels // tps,
        "band_high", video_consts.floor_pixels,
        "image_tail_test", image_consts.temporal_patch_size * raw_pixels,
        "video_tail_test", 1 * raw_pixels,
        *_paths_report(image_spec, video_spec, image_record, video_record, num_frames=1),
    )

    assert video_consts.floor_pixels // tps <= raw_pixels < video_consts.floor_pixels, (
        f"{name}: {raw_pixels} raw pixels is outside the band "
        f"[{video_consts.floor_pixels // tps}, {video_consts.floor_pixels}), so the two paths' tail "
        "tests do not disagree here and this case measures nothing"
    )
    assert image_consts.temporal_patch_size * raw_pixels >= image_consts.floor_pixels, (
        f"{name}: the image tail test must PASS here, capping its scale at 1.0"
    )
    assert 1 * raw_pixels < video_consts.floor_pixels, (
        f"{name}: the 1-frame video tail test must FAIL here, leaving its scale uncapped"
    )

    assert (image_spec.canvas_height, image_spec.canvas_width) == BAND_CANVAS, (
        f"{name}: the still path's canvas is {image_spec.canvas_height}x{image_spec.canvas_width}, "
        f"registered as {BAND_CANVAS[0]}x{BAND_CANVAS[1]}"
    )
    assert (video_spec.canvas_height, video_spec.canvas_width) == BAND_CANVAS, (
        f"{name}: the video path's canvas is {video_spec.canvas_height}x{video_spec.canvas_width}, "
        f"registered as {BAND_CANVAS[0]}x{BAND_CANVAS[1]}"
    )
    assert image_spec.regime == REGIME_PLAIN and video_spec.regime == REGIME_PLAIN, (
        f"{name}: the band case is regime A on both paths and reads {image_spec.regime} / "
        f"{video_spec.regime}; that is the whole point -- regime A is not sufficient for bit-identity"
    )
    assert image_spec.thw == video_spec.thw, (
        f"{name}: the grids {image_spec.thw} and {video_spec.thw} differ, so this case would be an "
        "ordinary shape divergence rather than the same-grid/different-pixels case it is registered "
        "as"
    )
    assert (image_spec.grid_h, image_spec.grid_w) == BAND_GRID_HW
    assert image_spec.num_patch_rows == video_spec.num_patch_rows == BAND_PATCH_ROWS, (
        f"{name}: rows are {image_spec.num_patch_rows} and {video_spec.num_patch_rows}, registered "
        f"as {BAND_PATCH_ROWS} on both"
    )

    assert (image_record.content_height, image_record.content_width) == BAND_IMAGE_CONTENT, (
        f"{name}: the still path's content is "
        f"{image_record.content_height}x{image_record.content_width}, registered as "
        f"{BAND_IMAGE_CONTENT[0]}x{BAND_IMAGE_CONTENT[1]} -- the input size, padded and not resized"
    )
    assert (video_record.content_height, video_record.content_width) == BAND_VIDEO_CONTENT, (
        f"{name}: the video path's content is "
        f"{video_record.content_height}x{video_record.content_width}, registered as "
        f"{BAND_VIDEO_CONTENT[0]}x{BAND_VIDEO_CONTENT[1]} -- the whole canvas, resampled"
    )
    assert (image_record.content_height, image_record.content_width) != (
        video_record.content_height,
        video_record.content_width,
    ), f"{name}: the two contents agree, so there is no divergence here to assert"


# ---------------------------------------------------------------------------
# V05 -- part (5): the 2/2 controls. The comparison must be able to fail, on shape and on pixels.
# ---------------------------------------------------------------------------
def test_v05a_a_three_frame_video_is_not_bit_identical_to_the_still_image(
    image_oracle, video_oracle, image_consts, video_consts
):
    """Control 1. V04 can fail on shape: three frames must pack twice the rows of one still image.

    The control is at THREE frames and not two. At F=2 the video takes ``grid_t`` 1 and the same
    canvas as one still image, so the shapes are necessarily equal and the control could never
    fire -- which this test asserts, so the reason the control moved is measured rather than
    claimed.
    """
    height, width = SHAPE_CONTROL_HEIGHT, SHAPE_CONTROL_WIDTH
    image_spec, video_spec, image_record, video_record = _both_specs(
        image_consts, video_consts, height, width, num_frames=SHAPE_CONTROL_FRAMES
    )
    image_pixels = _run_image_oracle(image_oracle, height, width)["pixel_values"]
    video_pixels = _run_video_oracle(
        video_oracle, SHAPE_CONTROL_FRAMES, height, width
    )["pixel_values_videos"]
    _record("v05a", "size", f"{height}x{width}", *_paths_report(
        image_spec, video_spec, image_record, video_record, num_frames=SHAPE_CONTROL_FRAMES
    ), "image_shape", "x".join(str(v) for v in image_pixels.shape),
        "video_shape", "x".join(str(v) for v in video_pixels.shape))

    assert tuple(image_pixels.shape) != tuple(video_pixels.shape), (
        f"a {SHAPE_CONTROL_FRAMES}-frame video packed to the same shape as one still image, so "
        "V04's shape gate could not tell one frame from three"
    )
    assert image_spec.num_patch_rows == SHAPE_CONTROL_IMAGE_ROWS, (
        f"the still image packs {image_spec.num_patch_rows} rows, registered as "
        f"{SHAPE_CONTROL_IMAGE_ROWS}"
    )
    assert video_spec.num_patch_rows == SHAPE_CONTROL_VIDEO_ROWS, (
        f"the {SHAPE_CONTROL_FRAMES}-frame video packs {video_spec.num_patch_rows} rows, registered "
        f"as {SHAPE_CONTROL_VIDEO_ROWS}"
    )
    assert video_spec.grid_t == 2 * image_spec.grid_t, (
        f"{SHAPE_CONTROL_FRAMES} frames gave grid_t {video_spec.grid_t} where one image gives "
        f"{image_spec.grid_t}"
    )
    assert (video_spec.canvas_height, video_spec.canvas_width) == (
        image_spec.canvas_height,
        image_spec.canvas_width,
    ), "the control's size must be one where the canvas does not move, so only the frame count does"

    two_frame_spec = video_grid_spec(video_consts, 2, height, width)
    _record("v05a_why_not_two", "video_thw", ",".join(str(v) for v in two_frame_spec.thw),
            "image_thw", ",".join(str(v) for v in image_spec.thw))
    assert two_frame_spec.thw == image_spec.thw, (
        "a 2-frame video no longer packs to the same grid as one still image, so F=2 would in fact "
        "fire this control and the reason it was moved to F=3 no longer holds"
    )


def test_v05b_in_the_band_the_shapes_agree_and_the_pixels_differ(image_oracle, video_oracle):
    """Control 2. V04 can fail on PIXELS with the shapes agreeing, from one and the same content.

    This is the strongest form of the pixel control: both paths are given the identical constant
    frame at 130x130, both produce the same shape, and the tensors still differ -- because the image
    is padded with a zero border and the video is resampled to fill the canvas. A comparison that
    only looked at shape would read this as a pass, so it is the case that proves the max-abs-diff
    reading is looking at pixels.
    """
    name, height, width = BAND_CASE
    image_pixels = _run_image_oracle(image_oracle, height, width, value=CONTENT_VALUE_A)[
        "pixel_values"
    ]
    video_pixels = _run_video_oracle(video_oracle, 1, height, width, value=CONTENT_VALUE_A)[
        "pixel_values_videos"
    ]
    shapes_equal = tuple(image_pixels.shape) == tuple(video_pixels.shape)
    _record(
        "v05b", name, "size", f"{height}x{width}",
        "image_shape", "x".join(str(v) for v in image_pixels.shape),
        "video_shape", "x".join(str(v) for v in video_pixels.shape),
        "shapes_equal", shapes_equal,
        "content_value", CONTENT_VALUE_A,
    )
    assert shapes_equal, (
        f"{name}: the two paths produced {tuple(image_pixels.shape)} and "
        f"{tuple(video_pixels.shape)}. This control needs EQUAL shapes -- its whole point is a "
        "pixel difference that a shape gate cannot see -- so an unequal shape here means the band "
        "case is no longer the case it is registered as."
    )
    max_abs_diff = (image_pixels.double() - video_pixels.double()).abs().max().item()
    _record("v05b_diff", name, "max_abs_diff", f"{max_abs_diff:.6e}")
    assert max_abs_diff > BIT_IDENTICAL_MAX_ABS_DIFF, (
        f"{name}: the same constant content read as bit-identical across the two paths at equal "
        "shapes, so the max-abs-diff reading cannot see pixel values at all -- or upstream stopped "
        "padding the image while resampling the video, which would make part (3)'s fourth "
        "divergence obsolete"
    )


def test_v05_different_content_is_not_bit_identical(image_oracle, video_oracle):
    """Extra coverage, not a registered control: two DIFFERENT constant contents must differ.

    V05B supersedes this as the registered pixel control, because it produces a difference from one
    and the same content. This is kept as the cheaper, weaker form: it would still catch a
    comparison that returned 0.0 unconditionally.
    """
    name, height, width, _regime = _case(UNCLAMPED_EQUALITY_CASE_NAMES[0])
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
    not part of any registered criterion, so an exception there would turn a green acceptance red
    for a question -061 never asked. The reading is printed either way, which is what makes the
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
# V08 -- part (4): per-modality oracle parity, 5/5 registered cases plus the 1/1 ceiling pair.
# ---------------------------------------------------------------------------
def test_v08_the_registered_case_table_is_the_five_this_increment_registered():
    """The drift guard on the table this file reads from ``-056``.

    Part (4) registers five named cases. This file reads them from the file that owns them rather
    than restating them, so this is the check that the owner still holds exactly those five: a row
    added, removed or resized there would otherwise change what a registered 5/5 means without
    anything failing.
    """
    _record("v08_table", "count", len(REGISTERED_VIDEO_CASES),
            "names", ",".join(row[0] for row in REGISTERED_VIDEO_CASES))
    assert len(REGISTERED_VIDEO_CASES) == 5, (
        f"part (4) registers 5 video cases; {CASES_OWNER_FILENAME} holds "
        f"{len(REGISTERED_VIDEO_CASES)}"
    )
    assert tuple(REGISTERED_VIDEO_CASES) == EXPECTED_REGISTERED_VIDEO_CASES, (
        f"{CASES_OWNER_FILENAME}'s {CASES_OWNER_SYMBOL} has drifted from the five rows part (4) "
        f"registers.\n  owner:      {tuple(REGISTERED_VIDEO_CASES)}\n  registered: "
        f"{EXPECTED_REGISTERED_VIDEO_CASES}"
    )


@pytest.mark.parametrize(
    "name,num_frames,height,width,grid,rows,tokens",
    REGISTERED_VIDEO_CASES,
    ids=[row[0] for row in REGISTERED_VIDEO_CASES],
)
def test_v08_the_fork_video_grid_equals_the_upstream_video_processors(
    video_oracle, video_consts, name, num_frames, height, width, grid, rows, tokens
):
    """One registered case: the fork's video grid, the registered row and the real processor agree.

    This is the criterion that actually protects the port, and it is PER MODALITY: the video path is
    compared with the video processor and never with the image processor. Revision 262 widened the
    set to five because two rows at one size left the video path oracle-pinned at no clamp bound and
    at neither F=1 nor F=2 -- and F=1 is the exact input parts (2), (3) and (5) turn on.
    """
    spec = video_grid_spec(video_consts, num_frames, height, width)
    out = _run_video_oracle(video_oracle, num_frames, height, width)
    oracle_grid = tuple(int(v) for v in out["video_grid_thw"][0].tolist())
    oracle_rows = int(out["pixel_values_videos"].shape[0])

    _record(
        "v08_video", name, "frames", num_frames, "size", f"{height}x{width}",
        "fork_thw", ",".join(str(v) for v in spec.thw),
        "oracle_thw", ",".join(str(v) for v in oracle_grid),
        "registered_thw", ",".join(str(v) for v in grid),
        "fork_rows", spec.num_patch_rows, "oracle_rows", oracle_rows, "registered_rows", rows,
        "fork_tokens", spec.num_merged_tokens, "registered_tokens", tokens,
        "regime", spec.regime,
    )
    assert spec.thw == grid, f"{name}: the fork's grid {spec.thw} is not the registered {grid}"
    assert oracle_grid == grid, (
        f"{name}: the real video processor's grid {oracle_grid} is not the registered {grid}"
    )
    assert spec.num_patch_rows == rows and oracle_rows == rows, (
        f"{name}: rows are fork {spec.num_patch_rows} and oracle {oracle_rows}, registered {rows}"
    )
    assert spec.num_merged_tokens == tokens, (
        f"{name}: the fork packs {spec.num_merged_tokens} tokens, registered {tokens}"
    )


def test_v08_the_lowered_ceiling_pair_pins_both_grid_functions_at_regime_c(glm5_next_pkg):
    """The 1/1 ceiling pair: both grid functions measured end to end at regime C, on real pixels.

    A video row at the DEFAULT video ceiling would need over 376 million pixels, so the ceiling is
    lowered on the processor INSTANCES instead and the size stays small. Nothing in the fork
    changes: the constructor keyword becomes an attribute, ``GridConstants.from_processor`` reads
    it, and the same instances are the oracle.

    THE FIRST TWO ASSERTIONS ARE THE SEAM CHECK AND MUST COME FIRST. If the keyword were silently
    ignored, both processors would still answer -- at the DEFAULT ceiling -- and every assertion
    below would pass while measuring regime A under a regime C label. So the attribute is read back
    before anything is measured.
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

    _record(
        "v08_ceiling", "tokens", LOWERED_CEILING_TOKENS, "size", f"{height}x{width}",
        "ceiling_pixels", consts_image.ceiling_pixels,
        "image_regime", image_spec.regime, "video_regime", video_spec.regime,
        "image_canvas", f"{image_spec.canvas_height}x{image_spec.canvas_width}",
        "video_canvas", f"{video_spec.canvas_height}x{video_spec.canvas_width}",
        "fork_image_thw", ",".join(str(v) for v in image_spec.thw),
        "oracle_image_thw", ",".join(str(v) for v in oracle_image_grid),
        "fork_video_thw", ",".join(str(v) for v in video_spec.thw),
        "oracle_video_thw", ",".join(str(v) for v in oracle_video_grid),
        "fork_image_rows", image_spec.num_patch_rows, "oracle_image_rows", oracle_image_rows,
        "fork_video_rows", video_spec.num_patch_rows, "oracle_video_rows", oracle_video_rows,
    )

    assert image_spec.regime == REGIME_ABOVE_CEILING, (
        f"the still path reads {image_spec.regime} at a ceiling of {LOWERED_CEILING_TOKENS} tokens; "
        "the pair is registered at regime C, so the lowered ceiling did not bind"
    )
    assert video_spec.regime == REGIME_ABOVE_CEILING, (
        f"the video path reads {video_spec.regime} at a ceiling of {LOWERED_CEILING_TOKENS} tokens"
    )
    assert (image_spec.canvas_height, image_spec.canvas_width) == LOWERED_CEILING_CANVAS, (
        f"the still path's canvas is {image_spec.canvas_height}x{image_spec.canvas_width}, "
        f"registered as {LOWERED_CEILING_CANVAS[0]}x{LOWERED_CEILING_CANVAS[1]}"
    )
    assert (video_spec.canvas_height, video_spec.canvas_width) == LOWERED_CEILING_CANVAS, (
        f"the video path's canvas is {video_spec.canvas_height}x{video_spec.canvas_width}, "
        f"registered as {LOWERED_CEILING_CANVAS[0]}x{LOWERED_CEILING_CANVAS[1]}"
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
        f"at ONE shared ceiling the two paths still disagree -- image {image_spec.thw}, video "
        f"{video_spec.thw}. Part (3)'s ceiling divergence is attributed to the two ceilings "
        "DIFFERING, so making them equal must remove it."
    )


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
        "floor_pixels", video_consts.floor_pixels,
        "image_max_image_tokens", image_consts.max_image_tokens,
        "video_max_image_tokens", video_consts.max_image_tokens,
    )
    tps = video_consts.temporal_patch_size
    lines = [
        "",
        "inc-glm53f-061 -- video path readings (increment-plan revision 262)",
        f"  module resolved from   {vllm_neuron.__file__}",
        f"  transformers reference {transformers.__version__}",
        f"  temporal_patch_size {tps}, merge_length {video_consts.merge_length}, "
        f"canvas factor {video_consts.factor}",
        f"  floor budget {video_consts.floor_pixels} px; ceilings "
        f"{image_consts.max_image_tokens} tokens image, {video_consts.max_image_tokens} video",
        "",
        f"  part (1) -- token count, registered at F={REGISTERED_FRAME_COUNTS} "
        f"(extra coverage {EXTRA_FRAME_COUNTS}), item {FRAME_ARM_HEIGHT}x{FRAME_ARM_WIDTH}",
    ]
    for num_frames in REGISTERED_FRAME_COUNTS + EXTRA_FRAME_COUNTS:
        spec = video_grid_spec(video_consts, num_frames, FRAME_ARM_HEIGHT, FRAME_ARM_WIDTH)
        closed = _closed_form_grid_t(num_frames, tps)
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
        "  part (2) -- 1-frame bit-identity, over the 4 unclamped cases (regime A and at or above "
        "the floor budget)",
    ]
    for name in UNCLAMPED_EQUALITY_CASE_NAMES:
        _n, height, width, _regime = _case(name)
        image_spec, video_spec, image_record, video_record = _both_specs(
            image_consts, video_consts, height, width, num_frames=1
        )
        verdict = "grids agree" if image_spec.thw == video_spec.thw else "GRIDS DIVERGE"
        image_pixels = _run_image_oracle(image_oracle, height, width)["pixel_values"]
        video_pixels = _run_video_oracle(video_oracle, 1, height, width)["pixel_values_videos"]
        if tuple(image_pixels.shape) == tuple(video_pixels.shape):
            diff = (image_pixels.double() - video_pixels.double()).abs().max().item()
            pixel_line = (
                f"both {tuple(image_pixels.shape)}, max abs diff {diff:.3e} "
                f"(asserted == {BIT_IDENTICAL_MAX_ABS_DIFF})"
            )
        else:
            pixel_line = (
                f"image {tuple(image_pixels.shape)} vs video {tuple(video_pixels.shape)}; shapes "
                "differ, so no elementwise difference exists to report"
            )
        lines += [
            f"    {name} ({height}x{width}, {height * width} px, regime {image_spec.regime}): "
            f"{verdict}",
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
            f"      real pixels     {pixel_line}",
        ]
        _record(
            "v07", name,
            "image_thw", ",".join(str(v) for v in image_spec.thw),
            "video_thw", ",".join(str(v) for v in video_spec.thw),
            "grids_equal", image_spec.thw == video_spec.thw,
        )

    lines += [
        "",
        "  part (3) -- the 4 divergences, each with the counterfactual that removes it",
    ]
    for name, height, width, _ic, _vc in FLOOR_COUNTERFACTUAL_CASES:
        got_image = resolve_canvas(
            image_consts, num_frames=image_consts.temporal_patch_size, height=height, width=width
        )
        got_video = resolve_canvas(video_consts, num_frames=1, height=height, width=width)
        counter = resolve_canvas(video_consts, num_frames=tps, height=height, width=width)
        lines.append(
            f"    {name} ({height}x{width}): still {got_image[0]}x{got_image[1]} vs 1-frame video "
            f"{got_video[0]}x{got_video[1]} ({got_video[2]}); told {tps} frames the video gives "
            f"{counter[0]}x{counter[1]}, so the frame count is the cause"
        )
    ceiling_name, ceiling_h, ceiling_w = CEILING_CASE
    got_image = resolve_canvas(
        image_consts, num_frames=image_consts.temporal_patch_size, height=ceiling_h, width=ceiling_w
    )
    got_video = resolve_canvas(video_consts, num_frames=1, height=ceiling_h, width=ceiling_w)
    counter = resolve_canvas(
        image_consts,
        num_frames=image_consts.temporal_patch_size,
        height=ceiling_h,
        width=ceiling_w,
        ceiling_pixels=video_consts.ceiling_pixels,
    )
    lines.append(
        f"    {ceiling_name} ({ceiling_h}x{ceiling_w}): still {got_image[0]}x{got_image[1]} "
        f"({got_image[2]}) vs 1-frame video {got_video[0]}x{got_video[1]} ({got_video[2]}); given "
        f"the video ceiling the still path gives {counter[0]}x{counter[1]} ({counter[2]}), so the "
        "two ceilings are the cause and the frame count is not"
    )
    band_name, band_h, band_w = BAND_CASE
    band_image_spec, band_video_spec, band_image_rec, band_video_rec = _both_specs(
        image_consts, video_consts, band_h, band_w, num_frames=1
    )
    lines.append(
        f"    {band_name} ({band_h}x{band_w}, {band_h * band_w} px, band "
        f"[{video_consts.floor_pixels // tps}, {video_consts.floor_pixels})): both "
        f"{band_image_spec.canvas_height}x{band_image_spec.canvas_width} regime "
        f"{band_image_spec.regime}, both {band_image_spec.num_patch_rows} rows; content "
        f"{band_image_rec.content_height}x{band_image_rec.content_width} padded vs "
        f"{band_video_rec.content_height}x{band_video_rec.content_width} resampled"
    )

    lines += [
        "",
        f"  part (4) -- per-modality oracle parity over {len(REGISTERED_VIDEO_CASES)} registered "
        f"video cases, read from {CASES_OWNER_FILENAME}",
    ]
    for name, num_frames, height, width, grid, rows, tokens in REGISTERED_VIDEO_CASES:
        spec = video_grid_spec(video_consts, num_frames, height, width)
        lines.append(
            f"    {name}: F={num_frames} {height}x{width} -> grid {spec.thw} rows "
            f"{spec.num_patch_rows} tokens {spec.num_merged_tokens} regime {spec.regime} "
            f"(registered {grid}, {rows}, {tokens})"
        )
    lines.append(
        f"    lowered-ceiling pair: {LOWERED_CEILING_TOKENS} tokens at "
        f"{LOWERED_CEILING_HEIGHT}x{LOWERED_CEILING_WIDTH} -> canvas "
        f"{LOWERED_CEILING_CANVAS[0]}x{LOWERED_CEILING_CANVAS[1]}, grid "
        f"{LOWERED_CEILING_GRID_HW[0]}x{LOWERED_CEILING_GRID_HW[1]}, "
        f"{LOWERED_CEILING_PATCH_ROWS} rows on both paths (V08 measures it)"
    )

    lines += [
        "",
        "  part (5) -- the 2 controls",
        f"    three frames at {SHAPE_CONTROL_HEIGHT}x{SHAPE_CONTROL_WIDTH}: "
        f"{SHAPE_CONTROL_VIDEO_ROWS} video rows against {SHAPE_CONTROL_IMAGE_ROWS} image rows, so "
        "the shape gate fires",
        f"    {band_name}: equal shapes and unequal pixels from one and the same content",
    ]
    with capsys.disabled():
        print("\n".join(lines))
