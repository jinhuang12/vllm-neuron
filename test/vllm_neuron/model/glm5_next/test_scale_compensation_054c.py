"""``inc-glm53f-054c``: the dense and shared scale grids are compensated exactly once.

WHAT THIS MEASURES, AND WHY IT STARTS FROM CHECKPOINT BYTES. The trn2 load is a matched pair -- squeeze
the weight BYTES into the 240 range, multiply the per-block scale grid by the inverse factor -- and before
this increment only the first half ran for the dense MLP and the shared expert. Their grids are attached
out-of-band and raw, so the product the kernel multiplied was the SQUEEZE FACTOR times the checkpoint's
numbers -- ``240/448`` when this file was written, an exact ``1/2`` since ``inc-glm53f-054e``.
A test that hands the seam PREPARED operands cannot see that, because the defect lives in the step that
turns checkpoint-format tensors into prepared ones. So every reading here starts from fp8-e4m3fn bytes and
a ``128``-tile grid and drives the real loader and the real prep.

THE REFERENCE IS THE CHECKPOINT'S OWN NUMBERS, NOT THE LOAD PATH'S. It dequantises the RAW bytes against
the RAW grid: no squeeze, no compensation. The landed reference in ``test_load_weights.py`` applies the
squeeze to its own reference, which makes both sides carry the factor and agrees with the defect -- that
one is corrected under this increment, and this file states the convention it should have had.

THE CONTROL RUNS IN BOTH DIRECTIONS, and each arm asserts the NUMBER it expects. A missing multiply reads
``0.5`` per projection since ``-054e`` (it was ``240/448 = 0.5357143``); a doubled one reads ``2.0``. An
arm that only required "some
failure" would be satisfied by an unrelated breakage.

VACUITY IS THE REAL RISK HERE. ``needs_240_downscale()`` resolves the clamp maximum AT IMPORT TIME and
answers 448 on a bare CPU-mode run, which makes the squeeze, the compensation, this test and its controls
all no-ops that pass. The override therefore belongs in the PROCESS INVOCATION and never in a fixture or
``monkeypatch.setenv`` (``design/acceptance-preregistration.md`` PIT-24), and every reading below asserts
the gate is engaged before it reads a number.

Run::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    pytest test/vllm_neuron/model/glm5_next/test_scale_compensation_054c.py \\
      -s -rA --timeout 120 -p no:cacheprovider
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.blockwise_fp8_mm import SCALE_BLOCK_SIZE
from vllm_neuron.model.glm5_next.quantization import DEFAULT_WEIGHT_BLOCK_SIZE
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    MAPPED_KEY_QUANTISED_WEIGHT,
    BlockScaleCompensation,
    classify_mapped_keys,
    compensate_block_scales,
    dequantise_blockwise,
    loader_for_mapped_keys,
    needs_240_downscale,
)

#: The two magnitudes the pair is built from -- the legacy and OCP e4m3 maxima.
FP8_E4M3_MAX = 240.0
FP8_E4M3FN_MAX = 448.0

#: What a MISSING compensation costs, per projection and through one MLP.
#:
#: ``inc-glm53f-054e`` made the squeeze the largest POWER OF TWO that fits 448 inside 240
#: rather than the ratio of the two, so this is 0.5 and not 240/448. It is written as the
#: literal it is: the ratio would be the wrong relation, and the load path's own constant is
#: pinned against the same literal in ``test_weight_loaders.py``.
#:
#: WHICH COMPARISONS BECOME EXACT. ``x 1/2`` is a pure exponent shift, so every fp8 value at
#: or above ``2**-5`` survives the round trip bit-exactly; the eight that do not are the odd
#: multiples of ``2**-9``. :func:`_ratio` is a ratio of SUMS weighted by magnitude, so those
#: eight contribute essentially nothing and every reading below becomes exact up to fp32
#: summation noise, where at 240/448 the same statistic sat about 1.4% high. THE BANDS ARE
#: NOT RE-TUNED for that -- a band is not this file's to move, and gaining margin is not a
#: reason to touch one. The exact-element count is PRINTED instead, so the gain is visible
#: in the transcript rather than folded into a tolerance.
SQUEEZE = 0.5                                     # was FP8_E4M3_MAX / FP8_E4M3FN_MAX
MLP_SQUEEZE = SQUEEZE**3                          # 0.125, was 0.1537445335276968

#: The PRODUCER's block, kept only to size the extents below in whole 256 blocks so
#: this file's geometry does not move. Nothing dequantises at it any more.
BLOCK = 256
#: THE CHECKPOINT'S TILE, AND SINCE ``inc-glm53f-112`` THE GRID THE MODULE CARRIES.
#: IMPORTED from the dense kernel rather than typed: the publish now hands the
#: checkpoint's own grid to ``blockwise_fp8_mm``, which indexes at
#: ``SCALE_BLOCK_SIZE``, so a literal here would be a second place for that number to
#: live and would go stale the day the kernel's granularity moves again.
TILE = SCALE_BLOCK_SIZE
HIDDEN = 2 * BLOCK          # 512, a whole number of 256-blocks so the retile runs
INTERMEDIATE = 2 * BLOCK    # 512
FP8 = torch.float8_e4m3fn
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _impl():
    """Import the modeling module inside a test body -- this package's convention."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


@pytest.fixture(autouse=True)
def _the_gate_is_engaged() -> None:
    """Refuse to report anything on a platform where the whole pair is a no-op.

    THIS IS NOT A SETUP STEP, IT IS THE VACUITY GUARD. On a bare CPU-mode run the clamp
    maximum resolves to 448, ``needs_240_downscale()`` answers False, the squeeze and the
    compensation both become no-ops, and every equality below holds for the wrong reason --
    including both control arms, which would stop failing. The override cannot be set here:
    it is read at IMPORT time, so a fixture runs too late (PIT-24). All this can do is
    refuse, loudly, naming the invocation that fixes it.
    """
    if not needs_240_downscale():
        pytest.fail(
            "VACUOUS RUN REFUSED: needs_240_downscale() is False, so the 240 squeeze and "
            "the compensation are both no-ops and every reading in this file "
            "would pass without measuring anything -- the controls included. The clamp "
            "maximum is resolved at IMPORT time, so this cannot be fixed from a fixture. "
            "Re-run with NEURON_PLATFORM_TARGET_OVERRIDE=trn2 in the process invocation."
        )


def _checkpoint_tensors(rows: int, cols: int, seed: int):
    """One projection in CHECKPOINT format: fp8-e4m3fn bytes and its own 128-tile grid.

    The grid is deliberately NON-uniform and away from 1.0, so a missing or doubled
    compensation cannot hide behind a scale that happens to be neutral.
    """
    torch.manual_seed(seed)
    raw_bytes = (torch.randn(rows, cols) * 40.0).to(FP8)
    grid = torch.rand(rows // TILE, cols // TILE, dtype=torch.float32) * 0.02 + 0.01
    return raw_bytes, grid


def _reference_matrix(raw_bytes: torch.Tensor, raw_grid: torch.Tensor) -> torch.Tensor:
    """The checkpoint's OWN numbers: raw bytes against the raw grid, un-squeezed.

    No ``downscale_fp8_weight_bytes`` and no ``compensate_block_scales``. This is the
    convention the landed reference in ``test_load_weights.py`` got wrong by applying the
    squeeze to itself, and the whole point of the comparison is that the load path has to
    reproduce THIS rather than its own transformed copy.
    """
    return dequantise_blockwise(raw_bytes, raw_grid, DEFAULT_WEIGHT_BLOCK_SIZE).to(
        torch.float32
    )


def _stored_bytes_through_the_real_loader(raw_bytes: torch.Tensor) -> torch.Tensor:
    """The bytes the ACTUAL loader stores, taken from the actual dispatch.

    ``loader_for_mapped_keys`` is asked for a quantised-weight mapping and the branch it
    picks is ASSERTED, so a future dispatch change cannot silently route this reading
    through a loader that does not squeeze.
    """
    keys = [
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv",
    ]
    assert classify_mapped_keys(keys) == MAPPED_KEY_QUANTISED_WEIGHT, (
        "this reading depends on the quantised-weight branch of loader_for_mapped_keys; "
        "the mapping above no longer classifies as one"
    )
    loader = loader_for_mapped_keys(keys, param_name="gate_proj_weight")
    stored = loader.load([raw_bytes], 0)
    assert stored.dtype is FP8, f"the loader stored {stored.dtype}, not fp8 bytes"
    return stored


def _loaded_module(kind: str):
    """A shared expert or a dense MLP, built from the package's own config type."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    # BOTH intermediate widths are set. The shared expert reads
    # ``moe_intermediate_size`` and the dense MLP reads ``intermediate_size``, and while
    # neither decides anything here -- the prep takes its extents off the ATTACHED weight
    # -- leaving one at its 12288 default would put a module's own declared width at odds
    # with the tensors this test hands it, which reads like an oversight.
    text_config = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTERMEDIATE,
        intermediate_size=INTERMEDIATE,
        n_shared_experts=1,
    )
    if kind == "shared":
        return _impl().Glm5NextSharedExperts(text_config)
    return _impl().Glm5NextDenseMLP(text_config)


def _attach_checkpoint_format(module, seed: int) -> dict[str, tuple]:
    """Put checkpoint-format weights and RAW grids on the module, as the load path does.

    The weights go through the real loader's squeeze; the grids are attached raw, which is
    exactly what ``_load_out_of_band_scales`` does at ``model_fp8.py:7743``. The grid is a
    plain attribute and not a parameter, deliberately, because that is how this tree holds
    scale grids.
    """
    truth = {}
    for index, leaf in enumerate(PROJECTIONS):
        rows, cols = (
            (INTERMEDIATE, HIDDEN) if leaf == "down_proj" else (HIDDEN, INTERMEDIATE)
        )
        raw_bytes, raw_grid = _checkpoint_tensors(rows, cols, seed + index)
        stored = _stored_bytes_through_the_real_loader(raw_bytes)
        weight_name = f"{leaf}_weight"
        grid_name = _impl().Glm5NextForConditionalGeneration._sibling_scale_grid_name(
            weight_name
        )
        setattr(module, weight_name, torch.nn.Parameter(stored, requires_grad=False))
        setattr(module, grid_name, raw_grid.clone())
        truth[leaf] = (raw_bytes, raw_grid, weight_name, grid_name)
    return truth


def _effective_matrix(module, weight_name: str, grid_name: str) -> torch.Tensor:
    """Rebuild what the kernel will multiply, undoing ONLY the compute-frame transpose.

    The prep leaves the weight and its grid transposed into the kernel's frame, so both
    are transposed back here and nothing else is touched.

    THE GRANULARITY IS THE CHECKPOINT'S OWN, RE-PINNED BY ``inc-glm53f-112``. The prep
    used to coarsen the 128 grid onto a public 256 one, so this rebuild ran at 256. It
    publishes the checkpoint's grid unchanged now, so the rebuild runs at ``TILE`` --
    which is the dense kernel's ``SCALE_BLOCK_SIZE``, imported. Nothing else about this
    reading moves: the compensation is still applied by the load path and still read
    here, and the reference is still the checkpoint's own numbers.
    """
    weight = getattr(module, weight_name)
    grid = getattr(module, grid_name)
    return dequantise_blockwise(
        weight.data.t().contiguous(), grid.t().contiguous(), (TILE, TILE)
    ).to(torch.float32)


def _ratio(got: torch.Tensor, want: torch.Tensor) -> float:
    """One number for how far the load path is from the checkpoint, by total magnitude.

    A ratio of sums rather than a mean of ratios: the grid is non-uniform and a per-element
    mean would be dominated by the smallest reference entries, where fp8 noise is largest.
    """
    return float(got.abs().sum() / want.abs().sum())


def _prepped(kind: str, seed: int = 11):
    """Drive the REAL seam: attach checkpoint format, then call the module's own prep."""
    module = _loaded_module(kind)
    truth = _attach_checkpoint_format(module, seed)
    retiled = module.retile_checkpoint_scale_grids()
    return module, truth, retiled


# --------------------------------------------------------------------------- #
# 1. The effective matrix matches the checkpoint, through the real seam.       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["shared", "dense"])
def test_the_effective_matrix_matches_the_checkpoint_not_the_squeezed_copy(kind) -> None:
    module, truth, retiled = _prepped(kind)
    assert retiled == len(PROJECTIONS), (
        f"the prep retiled {retiled} projections, not {len(PROJECTIONS)}; a skipped "
        f"retile would make the comparison below read a granularity nobody serves"
    )
    for leaf, (raw_bytes, raw_grid, weight_name, grid_name) in truth.items():
        want = _reference_matrix(raw_bytes, raw_grid)
        got = _effective_matrix(module, weight_name, grid_name)
        ratio = _ratio(got, want)
        exact = int(torch.eq(got, want).sum())
        print(f"S054C|{kind}|{leaf}|EFFECTIVE_OVER_CHECKPOINT|{ratio:.7f}")
        print(
            f"S054C|{kind}|{leaf}|ELEMENTS_BIT_EXACT|{exact}/{want.numel()}"
            f"|gated_on_none|118 of the 126 fp8 magnitudes can match at x 1/2, 14 at 240/448"
        )
        assert ratio == pytest.approx(1.0, rel=0.02), (
            f"{kind} {leaf}: the effective matrix is {ratio:.7f} of the checkpoint's. "
            f"A missing compensation reads {SQUEEZE:.7f}"
        )


# --------------------------------------------------------------------------- #
# 2. The compensation is applied EXACTLY once per grid -- neither 0 nor 2.     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["shared", "dense"])
def test_the_compensation_is_counted_once_per_grid(kind) -> None:
    module, truth, _ = _prepped(kind)
    # The health attribute is read off the module's OWN class constant, never guessed by
    # name shape: each class declares where its record lands and a sniffing test would
    # start passing on the wrong dictionary the moment either constant changed.
    health_attr = (
        module.SHARED_RETILE_HEALTH_ATTR
        if kind == "shared"
        else module.DENSE_RETILE_HEALTH_ATTR
    )
    health = getattr(module, health_attr)
    for leaf in truth:
        record = health[f"{leaf}_weight"]
        applied = record["scale_compensations_applied"]
        print(
            f"S054C|{kind}|{leaf}|COMPENSATIONS_APPLIED|{applied}"
            f"|PLATFORM_APPLIED|{record['scale_compensated']}"
            f"|BLOCKS_FLOORED|{record['scale_blocks_floored']}"
        )
        assert applied == 1, f"{kind} {leaf} compensated {applied} times, not once"
        assert record["scale_compensated"] is True, (
            f"{kind} {leaf}: the platform gate reports the multiply did not happen, so "
            f"the count of 1 is a call that did nothing"
        )


# --------------------------------------------------------------------------- #
# 3. THE CONTROL, BOTH DIRECTIONS, each arm naming the number it expects.      #
# --------------------------------------------------------------------------- #
def _identity_compensation(grid: torch.Tensor) -> BlockScaleCompensation:
    """The defect, restored: hand the grid back untouched."""
    flat = grid.to(torch.float32)
    return BlockScaleCompensation(
        scale_inv=flat,
        applied=False,
        below_minval_before=0,
        below_minval_after=0,
        floored_blocks=(),
    )


def _doubled_compensation(grid: torch.Tensor) -> BlockScaleCompensation:
    """The opposite defect: apply the factor twice."""
    once = compensate_block_scales(grid)
    return BlockScaleCompensation(
        scale_inv=compensate_block_scales(once.scale_inv).scale_inv,
        applied=True,
        below_minval_before=once.below_minval_before,
        below_minval_after=once.below_minval_after,
        floored_blocks=once.floored_blocks,
    )


@pytest.mark.parametrize(
    "arm,replacement,expected",
    [
        ("removed", _identity_compensation, SQUEEZE),
        ("doubled", _doubled_compensation, 1.0 / SQUEEZE),
    ],
)
def test_the_control_fails_at_the_predicted_number(
    monkeypatch: pytest.MonkeyPatch, arm, replacement, expected
) -> None:
    """With the compensation removed OR doubled the comparison must fail, at a NAMED value.

    The patch replaces the name ``_publish_compute_frame_operands`` resolves, so the seam
    under test is the real one and only the arithmetic moves.
    """
    monkeypatch.setattr(_impl(), "compensate_block_scales", replacement)
    module, truth, _ = _prepped("shared")
    for leaf, (raw_bytes, raw_grid, weight_name, grid_name) in truth.items():
        want = _reference_matrix(raw_bytes, raw_grid)
        got = _effective_matrix(module, weight_name, grid_name)
        ratio = _ratio(got, want)
        print(f"S054C|control-{arm}|{leaf}|EFFECTIVE_OVER_CHECKPOINT|{ratio:.7f}")
        assert ratio == pytest.approx(expected, rel=0.02), (
            f"control arm '{arm}' on {leaf} read {ratio:.7f}, not the predicted "
            f"{expected:.7f}; a control that fails at an unpredicted number is not a "
            f"control over this defect"
        )
        assert ratio != pytest.approx(1.0, rel=0.02), (
            f"control arm '{arm}' on {leaf} still matches the checkpoint, so the "
            f"comparison in this file cannot tell the defect from the fix"
        )


# --------------------------------------------------------------------------- #
# 4. One MLP output, so the reading is not only about a stored tensor.         #
# --------------------------------------------------------------------------- #
def test_one_mlp_output_carries_no_leftover_squeeze() -> None:
    """y = down @ ((gate @ x) * (up @ x)), the load path against the checkpoint.

    Three projections means the missing factor would appear three times, which is the
    ``SQUEEZE**3`` the landed test's own docstring recorded as its inverse in the
    other direction. Computed from the effective matrices rather than by running the
    kernel: this increment corrects a LOAD-path value and owes no kernel dispatch.
    """
    module, truth, _ = _prepped("shared")
    torch.manual_seed(7)
    x = torch.randn(HIDDEN, 8, dtype=torch.float32)

    def mlp(matrices: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = (matrices["gate_proj"] @ x) * (matrices["up_proj"] @ x)
        return matrices["down_proj"] @ hidden

    want = mlp(
        {leaf: _reference_matrix(t[0], t[1]).t() for leaf, t in truth.items()}
    )
    got = mlp(
        {
            leaf: _effective_matrix(module, t[2], t[3]).t()
            for leaf, t in truth.items()
        }
    )
    ratio = _ratio(got, want)
    print(f"S054C|shared|MLP_OUTPUT_OVER_CHECKPOINT|{ratio:.7f}")
    print(f"S054C|shared|MLP_OUTPUT_IF_UNCOMPENSATED|{MLP_SQUEEZE:.7f}")
    assert ratio == pytest.approx(1.0, rel=0.05), (
        f"the MLP output is {ratio:.7f} of the checkpoint's; an uncompensated load "
        f"reads {MLP_SQUEEZE:.7f}"
    )
