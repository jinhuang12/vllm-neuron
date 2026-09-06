# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ``inc-glm53f-059b`` -- WP10: the vision tower's 2-D rope.

THE REGISTERED CRITERION, VERBATIM. These are the plan's own 163 bytes, copied from
``design/increment-plan.md`` at criteria pin
``2f95bb021cdb2366cf1f29288cc2a63ccd260e724a05ac11bf2772df490dc310``, and the acceptance driver
greps for them as a fixed string. Markdown emphasis and all -- a transliteration would let the
quote drift from the criterion while still reading correctly:

2-D RoPE matches a torch reference at `assert_close(rtol=1e-2, atol=1e-5)` and its **rotation-norm invariance** holds within **1e-5**, which is oracle-independent.

WHAT THE RULING CHANGED, AND WHAT IT DID NOT. Design entry ``design-20260906-av`` §95 (ii)
re-ruled the invariance SCOPE to the FULL head, on this seat's own withdrawn rider. ``1e-5`` is
unchanged, ``rtol=1e-2, atol=1e-5`` is unchanged, and no number above moved -- a criterion
change would be the lead's and the user's, never this seat's. The grid is CONSUMED from
``-056``'s landed ``vision_preprocessing.py`` and never re-derived (``design-20260905-aq``
(iii)).

The conjuncts are measured as R01 (the reference comparison, 3/3 grids) and R03 (the norm
invariance). R02 fixes the output contract, R04 and R05 are firing controls, R06 exercises the
guards, and R07 reports every reading so no number here is silent.

THE REFERENCE IS NOT WRITTEN HERE. It is transformers 5.16.1's own chain --
``get_vision_position_ids`` + ``Glm5NextVisionRotaryEmbedding`` + ``cat(rotary, rotary)`` +
``cos``/``sin`` -- and the rotation in R03 is applied by transformers'
``apply_rotary_pos_emb_vision``. A hand-written oracle would put the very re-derivation ruling
``design-20260905-aq`` (iii) forbids into the test instead of the module, and it would agree
with the module by construction rather than by evidence.

WHY THE FULL HEAD. The module's docstring carries the derivation; the measurement is R03, and
R04 is its control: restricted to one half of a head the norm is NOT invariant, because the
rotation pairs a channel in the first half with a channel in the second. That is the reading
that withdrew this seat's earlier rider, and R04 fails loudly if it ever stops being true.

WHY THE INPUT IS NORMALISED IN R03. "Within 1e-5" does not say absolute or relative. With unit
vectors the two readings are the same number, so the item cannot be argued either way. The
raw-scale deviation is reported in R07 as well, unasserted, so both numbers are on the record.

NO CANDIDATE-ROOT ASSERT. This file deliberately carries no environment-coupled guard: the
whole-suite partition invocation runs with ``GLM53F_CANDIDATE_ROOT`` unset, and a test that
fails there would be measuring the harness. R07 REPORTS where the module resolved from; the
acceptance driver is what asserts the tree it measured.
"""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from vllm_neuron.model.glm5_next.utils.vision_preprocessing import (
    GridConstants,
    image_grid_spec,
    video_grid_spec,
)
from vllm_neuron.model.glm5_next.utils.vision_rope import (
    DEFAULT_VISION_ROPE_THETA,
    Glm5NextVisionRopeError,
    compute_vision_rotary_pos_emb,
    vision_position_ids,
)

#: The registered pair, from the design entry. No other numeric pair is authored here.
RTOL = 1e-2
ATOL = 1e-5

#: The registered norm-invariance bound, measured over the full head.
NORM_BOUND = 1e-5

#: This checkpoint's vision tower: 1024 wide over 16 heads, merge blocks of 2.
HEAD_DIM = 64
MERGE_SIZE = 2

#: Item sizes handed to ``-056``'s grid functions. The grids themselves are never typed here.
IMAGE_SIZES = (("A01_regime_A", 784, 1036), ("A02_regime_B", 29, 29))
VIDEO_SIZES = (("A03_video_three_frames", 3, 112, 112),)


@pytest.fixture(scope="module")
def hf():
    """transformers 5.16.1's own GLM-5.3-Flash vision pieces -- the reference, not a copy."""
    import transformers
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextVisionRotaryEmbedding,
        apply_rotary_pos_emb_vision,
    )
    from transformers.vision_utils import get_vision_position_ids

    return {
        "version": transformers.__version__,
        "rotary": Glm5NextVisionRotaryEmbedding,
        "apply": apply_rotary_pos_emb_vision,
        "position_ids": get_vision_position_ids,
    }


@pytest.fixture(scope="module")
def grids():
    """Every grid under test, CONSUMED from ``-056``'s landed module.

    ``GridConstants`` reads the six patch numbers off the transformers processor, so nothing
    about the grid is stated in this file -- only the pixel size of the item.
    """
    import transformers.models.glm5_next as pkg

    image_consts = GridConstants.from_processor(pkg.Glm5NextImageProcessor())
    video_consts = GridConstants.from_processor(pkg.Glm5NextVideoProcessor())

    out = []
    for name, height, width in IMAGE_SIZES:
        spec = image_grid_spec(image_consts, height, width)
        out.append((name, torch.tensor([spec.thw], dtype=torch.long), spec))
    for name, frames, height, width in VIDEO_SIZES:
        spec = video_grid_spec(video_consts, frames, height, width)
        out.append((name, torch.tensor([spec.thw], dtype=torch.long), spec))
    return tuple(out)


def _reference_cos_sin(hf, grid_thw, head_dim, merge_size):
    """The reference ``(cos, sin)``, built by transformers' own objects end to end."""
    position_ids = hf["position_ids"](grid_thw, merge_size)
    rotary = hf["rotary"](head_dim // 2)(position_ids)
    emb = torch.cat((rotary, rotary), dim=-1)
    return emb.cos(), emb.sin()


def _record(*fields):
    """Print one pipe-delimited record the acceptance driver reads by label and field.

    The driver gates on these numbers rather than on pytest's verdict, so a green file whose
    achieved error was worse than the registered bound cannot read as a pass. A record is
    located by its label, never by column position, because ``-q`` and ``-v`` indent
    differently.
    """
    print("ROPE|" + "|".join(str(f) for f in fields))


def _unit_heads(num_patches, num_heads, head_dim, seed):
    """Random query vectors, each head normalised to length 1."""
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(
        num_patches, num_heads, head_dim, generator=generator, dtype=torch.float32
    )
    return q / q.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# R01 -- the reference comparison, over 3/3 grids.
# ---------------------------------------------------------------------------
def test_r01_cos_and_sin_match_the_transformers_chain(hf, grids):
    """3/3 grids: the module's pair equals transformers' own pair at the registered tolerance."""
    assert len(grids) == 3, f"expected 3 grids under test, got {len(grids)}"
    _record("pair", "rtol", RTOL, "atol", ATOL, "norm_bound", NORM_BOUND)
    for name, grid_thw, spec in grids:
        cos, sin = compute_vision_rotary_pos_emb(grid_thw, HEAD_DIM, MERGE_SIZE)
        cos_ref, sin_ref = _reference_cos_sin(hf, grid_thw, HEAD_DIM, MERGE_SIZE)
        _record(
            "grid", name,
            "thw", ",".join(str(v) for v in spec.thw),
            "regime", spec.regime,
            "rows", spec.num_patch_rows,
        )
        _record(
            "r01", name,
            "cos_max", f"{(cos - cos_ref).abs().max().item():.6e}",
            "sin_max", f"{(sin - sin_ref).abs().max().item():.6e}",
        )
        assert_close(cos, cos_ref, rtol=RTOL, atol=ATOL, msg=f"{name}: cos")
        assert_close(sin, sin_ref, rtol=RTOL, atol=ATOL, msg=f"{name}: sin")


def test_r01_the_position_ids_match_the_transformers_layout(hf, grids):
    """The index layout is exact, not toleranced: a permutation is right or it is wrong."""
    for name, grid_thw, _spec in grids:
        mine = vision_position_ids(grid_thw, MERGE_SIZE)
        theirs = hf["position_ids"](grid_thw, MERGE_SIZE)
        _record("ids", name, "equal", torch.equal(mine, theirs), "rows", mine.shape[0])
        assert torch.equal(mine, theirs), f"{name}: position ids differ from transformers'"


def test_r01_the_three_grids_are_three_different_shapes(grids):
    """Three grids that agree on shape would be one grid measured three times."""
    shapes = {tuple(g[1][0].tolist()) for g in grids}
    assert len(shapes) == 3, f"the grids collapse to {shapes}"
    temporal = {tuple(g[1][0].tolist())[0] for g in grids}
    assert temporal == {1, 2}, f"no video grid among them: temporal elements {temporal}"


# ---------------------------------------------------------------------------
# R02 -- the output contract.
# ---------------------------------------------------------------------------
def test_r02_the_pair_is_full_head_width_float32(grids):
    """``[total_patches, head_dim]`` float32, the packing the fork's attention applies."""
    for name, grid_thw, spec in grids:
        cos, sin = compute_vision_rotary_pos_emb(grid_thw, HEAD_DIM, MERGE_SIZE)
        expected = (spec.num_patch_rows, HEAD_DIM)
        assert tuple(cos.shape) == expected, f"{name}: cos shape {tuple(cos.shape)}"
        assert tuple(sin.shape) == expected, f"{name}: sin shape {tuple(sin.shape)}"
        assert cos.dtype is torch.float32 and sin.dtype is torch.float32, f"{name}: dtype"


def test_r02_the_two_halves_of_the_pair_are_equal(grids):
    """Both halves of a head must see the same angle, or the rotation is not a rotation."""
    for name, grid_thw, _spec in grids:
        cos, sin = compute_vision_rotary_pos_emb(grid_thw, HEAD_DIM, MERGE_SIZE)
        for label, tensor in (("cos", cos), ("sin", sin)):
            first, second = tensor.chunk(2, dim=-1)
            assert torch.equal(first, second), f"{name}: {label} halves differ"


def test_r02_a_wider_head_gets_more_frequencies_not_a_wider_pass_through(hf, grids):
    """The contract holds at another head width, so 64 is not baked in."""
    _name, grid_thw, spec = grids[1]
    cos, sin = compute_vision_rotary_pos_emb(grid_thw, 128, MERGE_SIZE)
    assert tuple(cos.shape) == (spec.num_patch_rows, 128)
    cos_ref, sin_ref = _reference_cos_sin(hf, grid_thw, 128, MERGE_SIZE)
    assert_close(cos, cos_ref, rtol=RTOL, atol=ATOL)
    assert_close(sin, sin_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# R03 -- rotation-norm invariance, over the FULL head.
# ---------------------------------------------------------------------------
def test_r03_the_full_head_norm_is_invariant(hf, grids):
    """Applying the rope changes no head's length, within the registered bound.

    This is oracle-independent: it asks whether the operation is a rotation, not whether it
    agrees with anyone. The vectors are unit length, so the bound reads the same absolutely and
    relatively. The rotation is transformers' own ``apply_rotary_pos_emb_vision``.
    """
    for name, grid_thw, spec in grids:
        cos, sin = compute_vision_rotary_pos_emb(grid_thw, HEAD_DIM, MERGE_SIZE)
        q = _unit_heads(spec.num_patch_rows, 4, HEAD_DIM, seed=59)
        rotated, _ = hf["apply"](q, q.clone(), cos, sin)
        before = q.double().norm(dim=-1)
        after = rotated.double().norm(dim=-1)
        worst = (after - before).abs().max().item()
        _record(
            "r03", name,
            "unit_worst", f"{worst:.6e}",
            "bound", NORM_BOUND,
            "heads", 4,
            "rows", spec.num_patch_rows,
        )
        assert worst <= NORM_BOUND, f"{name}: worst full-head norm change {worst:.3e}"


# ---------------------------------------------------------------------------
# R04 -- control: restricted to one half, the norm is NOT invariant.
# ---------------------------------------------------------------------------
def test_r04_a_single_half_is_not_norm_invariant(hf, grids):
    """The control that withdrew this seat's rider: the halves mix, so a half is not preserved.

    If this ever passes as an invariance, the rotation has stopped pairing the halves and R03's
    scope would be wrong -- which is exactly the mistake this control exists to catch.
    """
    _name, grid_thw, spec = grids[0]
    cos, sin = compute_vision_rotary_pos_emb(grid_thw, HEAD_DIM, MERGE_SIZE)
    q = _unit_heads(spec.num_patch_rows, 4, HEAD_DIM, seed=59)
    rotated, _ = hf["apply"](q, q.clone(), cos, sin)
    half = HEAD_DIM // 2
    before = q[..., :half].double().norm(dim=-1)
    after = rotated[..., :half].double().norm(dim=-1)
    worst = (after - before).abs().max().item()
    _record("r04", _name, "half_worst", f"{worst:.6e}", "floor", 1e-3, "half_width", half)
    assert worst > 1e-3, (
        "the first half of the head kept its length, so the rotation is not pairing the two "
        f"halves; worst change was only {worst:.3e}"
    )


# ---------------------------------------------------------------------------
# R05 -- control: a row-major layout must FAIL R01's comparison.
# ---------------------------------------------------------------------------
def test_r05_a_row_major_layout_fails_the_same_comparison(hf, grids):
    """R01 can fail. A layout that forgets the merge blocks disagrees at the same tolerance."""
    _name, grid_thw, _spec = grids[0]
    temporal, height, width = (int(v) for v in grid_thw[0].tolist())
    rows, columns = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )
    row_major = torch.stack([rows.flatten(), columns.flatten()], dim=-1).repeat(
        temporal, 1
    )
    wrong = hf["rotary"](HEAD_DIM // 2)(row_major)
    wrong_cos = torch.cat((wrong, wrong), dim=-1).cos()
    cos, _sin = compute_vision_rotary_pos_emb(grid_thw, HEAD_DIM, MERGE_SIZE)
    _record(
        "r05", _name,
        "row_major_cos_max", f"{(cos - wrong_cos).abs().max().item():.6e}",
        "atol", ATOL,
    )
    assert not torch.allclose(cos, wrong_cos, rtol=RTOL, atol=ATOL), (
        "a row-major position layout matched the block-major one, so R01 could not tell an "
        "index bug from a correct layout"
    )


# ---------------------------------------------------------------------------
# R06 -- the guards.
# ---------------------------------------------------------------------------
def test_r06_a_head_that_does_not_divide_by_four_is_refused(grids):
    _name, grid_thw, _spec = grids[1]
    with pytest.raises(Glm5NextVisionRopeError, match="multiple of 4"):
        compute_vision_rotary_pos_emb(grid_thw, 6, MERGE_SIZE)


def test_r06_a_grid_that_is_not_whole_merge_blocks_is_refused():
    with pytest.raises(Glm5NextVisionRopeError, match="whole merge blocks"):
        compute_vision_rotary_pos_emb(
            torch.tensor([[1, 5, 4]], dtype=torch.long), HEAD_DIM, MERGE_SIZE
        )


def test_r06_an_empty_or_misshaped_grid_is_refused():
    with pytest.raises(Glm5NextVisionRopeError, match="empty"):
        vision_position_ids(torch.zeros((0, 3), dtype=torch.long), MERGE_SIZE)
    with pytest.raises(Glm5NextVisionRopeError, match=r"\[num_items, 3\]"):
        vision_position_ids(torch.tensor([1, 56, 74], dtype=torch.long), MERGE_SIZE)


# ---------------------------------------------------------------------------
# R07 -- report every reading.
# ---------------------------------------------------------------------------
def test_r07_report_the_measured_readings(hf, grids, capsys):
    """One place a person can read what this run actually measured."""
    import vllm_neuron

    _record(
        "env",
        "transformers", hf["version"],
        "module", vllm_neuron.__file__,
        "torch", torch.__version__,
        "head_dim", HEAD_DIM,
        "merge", MERGE_SIZE,
        "theta", DEFAULT_VISION_ROPE_THETA,
    )
    lines = [
        "",
        "inc-glm53f-059b -- vision rope readings",
        f"  module resolved from   {vllm_neuron.__file__}",
        f"  transformers reference {hf['version']}",
        f"  head_dim {HEAD_DIM}, merge {MERGE_SIZE}, theta {DEFAULT_VISION_ROPE_THETA}",
        f"  registered pair rtol={RTOL} atol={ATOL}, norm bound {NORM_BOUND} over the full head",
    ]
    for name, grid_thw, spec in grids:
        cos, sin = compute_vision_rotary_pos_emb(grid_thw, HEAD_DIM, MERGE_SIZE)
        cos_ref, sin_ref = _reference_cos_sin(hf, grid_thw, HEAD_DIM, MERGE_SIZE)
        q_unit = _unit_heads(spec.num_patch_rows, 4, HEAD_DIM, seed=59)
        rotated_unit, _ = hf["apply"](q_unit, q_unit.clone(), cos, sin)
        unit_worst = (
            (rotated_unit.double().norm(dim=-1) - q_unit.double().norm(dim=-1))
            .abs()
            .max()
            .item()
        )
        generator = torch.Generator().manual_seed(59)
        q_raw = torch.randn(
            spec.num_patch_rows, 4, HEAD_DIM, generator=generator, dtype=torch.float32
        )
        rotated_raw, _ = hf["apply"](q_raw, q_raw.clone(), cos, sin)
        raw_before = q_raw.double().norm(dim=-1)
        raw_worst = (rotated_raw.double().norm(dim=-1) - raw_before).abs().max().item()
        _record(
            "r07", name,
            "unit_worst", f"{unit_worst:.6e}",
            "raw_worst", f"{raw_worst:.6e}",
            "raw_mean_len", f"{raw_before.mean().item():.6f}",
        )
        lines += [
            f"  {name}: grid {tuple(spec.thw)} regime {spec.regime}, "
            f"{spec.num_patch_rows} patch rows",
            f"    max abs diff vs transformers: cos {(cos - cos_ref).abs().max():.3e}, "
            f"sin {(sin - sin_ref).abs().max():.3e}",
            f"    full-head norm change: unit vectors {unit_worst:.3e} (asserted <= "
            f"{NORM_BOUND}), raw N(0,1) {raw_worst:.3e} (reported, mean length "
            f"{raw_before.mean():.3f})",
        ]
    with capsys.disabled():
        print("\n".join(lines))
