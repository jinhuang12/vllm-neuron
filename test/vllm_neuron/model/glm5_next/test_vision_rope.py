# SPDX-License-Identifier: Apache-2.0
"""Tests for the vision tower's 2-D rotary embedding.

The reference is transformers' own chain -- ``get_vision_position_ids``, then
``Glm5NextVisionRotaryEmbedding``, then ``cat(rotary, rotary)`` -- executed here rather than
restated as a formula, and the rotation is applied by ``apply_rotary_pos_emb_vision``. The
grids come from the preprocessing module, so no grid is written out in this file.
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
    Glm5NextVisionRopeError,
    compute_vision_rotary_pos_emb,
    vision_position_ids,
)

RTOL = 1e-2
ATOL = 1e-5

#: Bound on how much the rotation may change a head's length, measured over the full head.
NORM_BOUND = 1e-5

#: This checkpoint's vision tower: 1024 wide over 16 heads, merge blocks of 2.
HEAD_DIM = 64
MERGE_SIZE = 2

#: Pixel sizes handed to the grid helpers, which own the grid arithmetic.
IMAGE_SIZES = (("regime_a", 784, 1036), ("regime_b", 29, 29))
VIDEO_SIZES = (("video_three_frames", 3, 112, 112),)


@pytest.fixture(scope="module")
def hf():
    """transformers' own vision rope pieces, used as the reference."""
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextVisionRotaryEmbedding,
        apply_rotary_pos_emb_vision,
    )
    from transformers.vision_utils import get_vision_position_ids

    return {
        "rotary": Glm5NextVisionRotaryEmbedding,
        "apply": apply_rotary_pos_emb_vision,
        "position_ids": get_vision_position_ids,
    }


@pytest.fixture(scope="module")
def grids():
    """Every grid under test, built by the preprocessing module.

    ``GridConstants`` reads the six patch numbers off the transformers processor, so only the
    pixel size of an item is named in this file.
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


def _unit_heads(num_patches, num_heads, head_dim, seed):
    """Random query vectors, each head normalised to length 1."""
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(
        num_patches, num_heads, head_dim, generator=generator, dtype=torch.float32
    )
    return q / q.norm(dim=-1, keepdim=True)


def test_cos_and_sin_match_the_transformers_chain(hf, grids):
    """The module's pair equals transformers' own pair on every grid."""
    assert len(grids) == 3, f"expected 3 grids under test, got {len(grids)}"
    for name, grid_thw, _spec in grids:
        cos, sin = compute_vision_rotary_pos_emb(grid_thw, HEAD_DIM, MERGE_SIZE)
        cos_ref, sin_ref = _reference_cos_sin(hf, grid_thw, HEAD_DIM, MERGE_SIZE)
        assert_close(cos, cos_ref, rtol=RTOL, atol=ATOL, msg=f"{name}: cos")
        assert_close(sin, sin_ref, rtol=RTOL, atol=ATOL, msg=f"{name}: sin")


def test_the_position_ids_match_the_transformers_layout(hf, grids):
    """The index layout is exact, not toleranced: a permutation is right or it is wrong."""
    for name, grid_thw, _spec in grids:
        mine = vision_position_ids(grid_thw, MERGE_SIZE)
        theirs = hf["position_ids"](grid_thw, MERGE_SIZE)
        assert torch.equal(mine, theirs), f"{name}: position ids differ from transformers'"


def test_the_three_grids_are_three_different_shapes(grids):
    """Three grids that agreed on shape would be one grid measured three times."""
    shapes = {tuple(grid_thw[0].tolist()) for _name, grid_thw, _spec in grids}
    assert len(shapes) == 3, f"the grids collapse to {shapes}"
    temporal = {shape[0] for shape in shapes}
    assert temporal == {1, 2}, f"no video grid among them: temporal elements {temporal}"


def test_the_pair_is_full_head_width_float32(grids):
    """``[total_patches, head_dim]`` float32, the packing the fork's attention applies."""
    for name, grid_thw, spec in grids:
        cos, sin = compute_vision_rotary_pos_emb(grid_thw, HEAD_DIM, MERGE_SIZE)
        expected = (spec.num_patch_rows, HEAD_DIM)
        assert tuple(cos.shape) == expected, f"{name}: cos shape {tuple(cos.shape)}"
        assert tuple(sin.shape) == expected, f"{name}: sin shape {tuple(sin.shape)}"
        assert cos.dtype is torch.float32 and sin.dtype is torch.float32, f"{name}: dtype"


def test_the_two_halves_of_the_pair_are_equal(grids):
    """Both halves of a head must see the same angle, or the rotation is not a rotation."""
    for name, grid_thw, _spec in grids:
        cos, sin = compute_vision_rotary_pos_emb(grid_thw, HEAD_DIM, MERGE_SIZE)
        for label, tensor in (("cos", cos), ("sin", sin)):
            first, second = tensor.chunk(2, dim=-1)
            assert torch.equal(first, second), f"{name}: {label} halves differ"


def test_a_wider_head_gets_more_frequencies_not_a_wider_pass_through(hf, grids):
    """The contract holds at another head width, so 64 is not baked in."""
    _name, grid_thw, spec = grids[1]
    cos, sin = compute_vision_rotary_pos_emb(grid_thw, 128, MERGE_SIZE)
    assert tuple(cos.shape) == (spec.num_patch_rows, 128)
    cos_ref, sin_ref = _reference_cos_sin(hf, grid_thw, 128, MERGE_SIZE)
    assert_close(cos, cos_ref, rtol=RTOL, atol=ATOL)
    assert_close(sin, sin_ref, rtol=RTOL, atol=ATOL)


def test_the_full_head_norm_is_invariant(hf, grids):
    """Applying the rope changes no head's length by more than ``NORM_BOUND``.

    This asks whether the operation is a rotation, not whether it agrees with anyone. The
    vectors are unit length, so the bound reads the same absolutely and relatively. The
    rotation is transformers' own ``apply_rotary_pos_emb_vision``.
    """
    for name, grid_thw, spec in grids:
        cos, sin = compute_vision_rotary_pos_emb(grid_thw, HEAD_DIM, MERGE_SIZE)
        q = _unit_heads(spec.num_patch_rows, 4, HEAD_DIM, seed=59)
        rotated, _ = hf["apply"](q, q.clone(), cos, sin)
        before = q.double().norm(dim=-1)
        after = rotated.double().norm(dim=-1)
        worst = (after - before).abs().max().item()
        assert worst <= NORM_BOUND, f"{name}: worst full-head norm change {worst:.3e}"


def test_a_single_half_is_not_norm_invariant(hf, grids):
    """Half a head is not length-preserving, because the rotation pairs the two halves.

    A channel in the first half rotates against a channel in the second, so the invariance
    holds over the whole head and not over either half of it.
    """
    _name, grid_thw, spec = grids[0]
    cos, sin = compute_vision_rotary_pos_emb(grid_thw, HEAD_DIM, MERGE_SIZE)
    q = _unit_heads(spec.num_patch_rows, 4, HEAD_DIM, seed=59)
    rotated, _ = hf["apply"](q, q.clone(), cos, sin)
    half = HEAD_DIM // 2
    before = q[..., :half].double().norm(dim=-1)
    after = rotated[..., :half].double().norm(dim=-1)
    worst = (after - before).abs().max().item()
    assert worst > 1e-3, (
        "the first half of the head kept its length, so the rotation is not pairing the two "
        f"halves; worst change was only {worst:.3e}"
    )


def test_a_row_major_layout_fails_the_same_comparison(hf, grids):
    """A layout that forgets the merge blocks disagrees at the same tolerance."""
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
    assert not torch.allclose(cos, wrong_cos, rtol=RTOL, atol=ATOL), (
        "a row-major position layout matched the block-major one, so the comparison above "
        "could not tell an index bug from a correct layout"
    )


def test_a_head_that_does_not_divide_by_four_is_refused(grids):
    _name, grid_thw, _spec = grids[1]
    with pytest.raises(Glm5NextVisionRopeError, match="multiple of 4"):
        compute_vision_rotary_pos_emb(grid_thw, 6, MERGE_SIZE)


def test_a_grid_that_is_not_whole_merge_blocks_is_refused():
    with pytest.raises(Glm5NextVisionRopeError, match="whole merge blocks"):
        compute_vision_rotary_pos_emb(
            torch.tensor([[1, 5, 4]], dtype=torch.long), HEAD_DIM, MERGE_SIZE
        )


def test_an_empty_or_misshaped_grid_is_refused():
    with pytest.raises(Glm5NextVisionRopeError, match="empty"):
        vision_position_ids(torch.zeros((0, 3), dtype=torch.long), MERGE_SIZE)
    with pytest.raises(Glm5NextVisionRopeError, match=r"\[num_items, 3\]"):
        vision_position_ids(torch.tensor([1, 56, 74], dtype=torch.long), MERGE_SIZE)
