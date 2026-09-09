# SPDX-License-Identifier: Apache-2.0
"""Acceptance for ``inc-glm53f-060``: the vision tower and the embed merge.

WHAT EACH ITEM SETTLES, and which registered expectation it answers.

  C00  The modules under test resolve from the tree this run means to measure.
  C01  The tower's output equals the transformers 5.16.1 ``Glm5NextVisionModel``
       on a tiny random-init config with every weight COPIED across, over one
       regime-A grid from the landed seven-case set, at
       ``assert_close(rtol=1e-2, atol=1e-5)``. Two arms, so a failure says WHERE:
       arm A compares the merged output (the reference's ``pooler_output``, this
       tower's whole forward) and arm B compares the pre-merger stage (the
       reference's ``last_hidden_state``), isolating the merger from everything
       upstream of it.
  C02  The two temporal slices of a REAL IMAGE processor's patch row are the same
       pixels. Measured against the shipped image processor, not asserted from
       the source. It is NOT a guard on the tower and its old wording wrongly
       said it was -- see the item's own docstring and C09.
  C03  The rope positions tell a transposed grid apart. Review note B91 N5
       records that neither the landed adapter nor the reference can distinguish
       a (4, 2) grid from a (2, 4) one, because both treat any four consecutive
       tokens as one merge block. The position ids CAN, and this item measures
       that on a non-square grid, elementwise against the reference helper.
  C04  The merge writes every placeholder position and nothing else: index-set
       equality, N/N placeholders carrying their own values, 0 misplaced, and
       every other row bit-identical. Second phase: positions outside this
       rank's shard are dropped rather than wrapped.
  C05  Two image items in ONE merge land in their own spans (review note
       B90 N4).
  C06  The encoder-cache snapshot round-trips at max abs diff EXACTLY 0.0.
  C07  A text-only request snapshots 0 encoder-cache entries.
  C08  This module's activation map has not drifted from the adapter's.
  C09  `inc-glm53f-111`. A two-frame video with DIFFERING frames, through the real
       ``Glm5NextVideoProcessor``, embeds through the tower to the reference
       ``nn.Conv3d``'s own answer at the registered pair -- so frame ``2k+1`` of
       every temporal pair reaches the patch embedding. Its control that fires:
       the pre-`-111` computation (filters folded over the temporal axis, slice 0
       fed) must NOT meet the pair on those same rows, and the item raises on a
       vacuous reading rather than passing quietly.
  C10  `inc-glm53f-111`. The cause, named: with the second temporal filter slice
       zeroed, the folded single-slice computation and the reference agree again
       at the pair. That is what makes C09's gap the dropped slice and not a
       dtype, layout or tolerance artefact.
  C11  `inc-glm53f-111`. Image non-regression: image-processor rows through the
       per-slice path still equal the reference at the registered pair. The
       identity is algebraic -- for copies, ``sum_t conv2d(x_0, W_t)`` and
       ``conv2d(x_0, sum_t W_t)`` are the same map -- but summation order differs
       in finite precision, so it is MEASURED at the pair and never asserted
       bit-equal.

HOW THE CACHE ITEMS ARE DRIVEN, and why not through a request. C06 and C07 call
the runner's four snapshot methods UNBOUND, on a ``SimpleNamespace`` carrying
only the attributes those methods read. That is the form
``test/vllm_neuron/worker/test_get_kv_cache_spec_hybrid.py:206-224`` already
uses on this branch, and it is deliberate here: the only production caller of
``_snapshot_encoder_entries`` sits inside ``execute_model``, which on this
branch still reaches ``inc-glm53f-013``'s raising stub. Driven through
``execute_model`` these items would fail on that raise instead of on an
assertion, which would say nothing about the cache.

THE FOUR CACHE METHODS ARE PIN CODE AND THIS FILE DOES NOT CHANGE THEM. They
are complete and architecture-generic at this block's parent, so C06 and C07
certify behaviour rather than new work -- see the increment's evidence record.
"""

from __future__ import annotations

import copy
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.testing import assert_close

# The oracle is imported plainly and on purpose. transformers is a hard floor for
# this fork -- ``requirements/core.txt:18`` declares
# ``transformers>=5.16.1,<6.0.0`` -- so a conditional import would let this whole
# file go green without ever executing the specification it claims to measure.
from transformers.models.glm5_next import configuration_glm5_next, modeling_glm5_next

from vllm_neuron.functional.vision.patch_embed import patch_embed as _seam_patch_embed
from vllm_neuron.model.glm5_next.config import Glm5NextVisionConfig
from vllm_neuron.model.glm5_next.utils.merge_vision_embeds import (
    Glm5NextVisionMergeError,
    merge_vision_embeddings,
)
from vllm_neuron.model.glm5_next.utils.vision_preprocessing import (
    GridConstants,
    image_grid_spec,
    require_transformers_glm5_next,
)
from vllm_neuron.model.glm5_next.utils.vision_rope import vision_position_ids
from vllm_neuron.model.glm5_next.vision_encoder import (
    _ACTIVATIONS,
    _PATCH_ROWS_PER_CALL,
    Glm5NextVisionEncoder,
    compute_attention_bounds,
    patch_embed_filters_from_conv3d,
)

#: float32 throughout. The tolerance pair is registered at rtol 1e-2 / atol 1e-5,
#: but the two sides share copied weights, so anything near that bound would mean
#: a real disagreement rather than accumulated rounding.
DTYPE = torch.float32

#: The registered tolerance pair for C01, from the increment's own block.
RTOL, ATOL = 1e-2, 1e-5

#: The regime-A case C01 runs, taken from the landed seven-case set in
#: ``test_vision_preprocessing.py`` (``R2_CASES``, entry ``wide_strip``). It is
#: chosen for two properties at once: it is regime A (no resample), and its grid
#: is NON-SQUARE, which is what C03 needs. Its patch count also exceeds the patch
#: embed seam's per-call batch ceiling, so the tower's chunking is exercised
#: rather than skipped.
REGIME_A_CASE = ("wide_strip", 1400, 56)

#: The geometry fields stay at the checkpoint's real values so the landed grid
#: helper applies unchanged; only the widths and the depth shrink.
TINY = dict(
    depth=2,
    hidden_size=32,
    num_heads=2,
    intermediate_size=48,
    in_channels=3,
    patch_size=14,
    temporal_patch_size=2,
    spatial_merge_size=2,
    out_hidden_size=24,
    projection_intermediate_size=40,
    attention_bias=True,
    hidden_act="silu",
    swiglu_limit=10.0,
    rms_norm_eps=1e-05,
)


def _emit(tag: str, **values) -> None:
    """Print one reading as a bracketed line so the transcript carries the number.

    A test that only says "passed" leaves the reader to trust it. These lines put
    the measured error, the tolerance pair it was judged against and each counted
    value into the transcript, where the acceptance driver reads them back.
    """
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"[{tag}] {body}", flush=True)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    # Detached before the scalar read: some of these are module parameters, and
    # torch warns when a tensor still carrying requires_grad becomes a float.
    return float((a.detach() - b.detach()).abs().max())


def _repo_root_from_this_file() -> Path:
    """Name the checkout this test file itself ships in, without the environment.

    Same derivation the sibling vision test uses, with the same depth assertion:
    if this file is ever moved to another depth the assertion fires, which is the
    right outcome.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "vllm_neuron" / "__init__.py").is_file() and (
            parent / "test"
        ).is_dir():
            assert parent == here.parents[4], (
                f"the derived root {parent} is not parents[4] {here.parents[4]}; "
                "this file has moved depth and the derivation needs re-reading"
            )
            return parent
    raise AssertionError(f"no checkout root above {here} holds vllm_neuron/ and test/")


# ---------------------------------------------------------------------------
# Fixtures shared by the tower items.
# ---------------------------------------------------------------------------
def _fork_config() -> Glm5NextVisionConfig:
    return Glm5NextVisionConfig(**TINY)


def _reference_config():
    """The reference's own config object, carrying the same tiny numbers.

    ``_attn_implementation`` is pinned to eager because the flash path needs a
    CUDA build; eager takes the reference's chunked per-segment branch
    (``modeling_glm5_next.py:1642-1663``), which is the same mathematics on CPU.
    """
    cfg = configuration_glm5_next.Glm5NextVisionConfig(**TINY)
    cfg._attn_implementation = "eager"
    return cfg


def _reference_tower():
    """Build the reference tower and put it in eval mode with a fixed seed."""
    torch.manual_seed(60)
    ref = modeling_glm5_next.Glm5NextVisionModel(_reference_config()).to(DTYPE)
    ref.eval()
    return ref


def _fork_tower_with_copied_weights(cfg: Glm5NextVisionConfig, ref) -> Glm5NextVisionEncoder:
    """Copy every reference weight into a fork tower, and prove the copy is total.

    Three parameters need a transform and the rest match by name. The two
    coverage assertions at the end are the point of this helper: a comparison
    whose weights were only PARTLY copied would pass or fail for reasons that
    have nothing to do with the tower, so both sides are accounted for
    name-by-name rather than trusted.
    """
    tower = Glm5NextVisionEncoder(cfg, dtype=DTYPE)
    tower.eval()

    ref_params = dict(ref.named_parameters())
    consumed: set[str] = set()
    written: set[str] = set()

    def take(name: str) -> torch.Tensor:
        consumed.add(name)
        return ref_params[name].detach()

    with torch.no_grad():
        # (1) patch embed: [C_out, C_in, T, P, P] -> [T, P, P, C_in, C_out], a
        # pure axis reorder that keeps the temporal slices SEPARATE. Nothing is
        # summed in the weight since `inc-glm53f-111`; the tower sums the seam's
        # outputs instead, one call per temporal slice (C09, C10, C11).
        tower.patch_embed_filters.copy_(
            patch_embed_filters_from_conv3d(take("patch_embed.proj.weight"))
        )
        written.add("patch_embed_filters")
        tower.patch_embed_bias.copy_(take("patch_embed.proj.bias"))
        written.add("patch_embed_bias")

        # (2) the block stack: every name matches the reference's.
        for idx in range(cfg.depth):
            for leaf in (
                "norm1.weight",
                "norm2.weight",
                "attn.qkv.weight",
                "attn.qkv.bias",
                "attn.proj.weight",
                "attn.proj.bias",
                "attn.q_norm.weight",
                "attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.gate_proj.bias",
                "mlp.up_proj.weight",
                "mlp.up_proj.bias",
                "mlp.down_proj.weight",
                "mlp.down_proj.bias",
            ):
                name = f"blocks.{idx}.{leaf}"
                _assign(tower, name, take(name))
                written.add(name)

        # (3) the adapter: the landed module owns its own conv transform.
        tower.adapter.post_layernorm.weight.copy_(take("post_layernorm.weight"))
        written.add("adapter.post_layernorm.weight")
        tower.adapter.downsample_weight.copy_(
            tower.adapter.downsample_weight_from_conv(take("downsample.weight"))
        )
        written.add("adapter.downsample_weight")
        tower.adapter.downsample_bias.copy_(take("downsample.bias"))
        written.add("adapter.downsample_bias")
        for leaf in (
            "proj.weight",
            "post_projection_norm.weight",
            "post_projection_norm.bias",
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
        ):
            name = f"merger.{leaf}"
            _assign(tower, f"adapter.{name}", take(name))
            written.add(f"adapter.{name}")

    fork_names = {name for name, _ in tower.named_parameters()}
    missing_fork = sorted(fork_names - written)
    missing_ref = sorted(set(ref_params) - consumed)
    _emit(
        "c01 weight-copy",
        fork_params=len(fork_names),
        fork_written=len(written),
        ref_params=len(ref_params),
        ref_consumed=len(consumed),
        FORK_PARAMS_LEFT_UNWRITTEN=len(missing_fork),
        REF_PARAMS_LEFT_UNCONSUMED=len(missing_ref),
    )
    assert not missing_fork, f"fork parameters never written: {missing_fork}"
    assert not missing_ref, f"reference parameters never read: {missing_ref}"
    return tower


def _assign(module: torch.nn.Module, dotted: str, value: torch.Tensor) -> None:
    """Copy ``value`` into the parameter at ``dotted`` on ``module``."""
    target = module
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    getattr(target, parts[-1]).copy_(value)


def _grid_for_case(height: int, width: int) -> torch.Tensor:
    """Derive the case's grid through the landed helper, not from a literal.

    Hardcoding the grid would make this file a second source for a number the
    preprocessing module already owns and tests.
    """
    package = require_transformers_glm5_next()
    consts = GridConstants.from_processor(package.Glm5NextImageProcessor())
    spec = image_grid_spec(consts, height, width)
    return torch.tensor([spec.thw], dtype=torch.long)


def _patch_rows(grid_thw: torch.Tensor, cfg: Glm5NextVisionConfig, seed: int):
    """Build patch rows whose temporal slices are copies, as the processor does.

    Returns ``(pixel_values, total_patches)`` with ``pixel_values`` shaped
    ``[total_patches, in_channels * temporal_patch_size * patch_size ** 2]`` and
    the temporal axis a broadcast of one slice -- the same construction
    ``image_processing_glm5_next.py:206-213`` performs, which C02 measures on the
    real processor.
    """
    total = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())
    generator = torch.Generator().manual_seed(seed)
    one_slice = torch.randn(
        total,
        cfg.in_channels,
        1,
        cfg.patch_size,
        cfg.patch_size,
        generator=generator,
        dtype=DTYPE,
    )
    rows = one_slice.expand(
        total, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size
    )
    return rows.reshape(total, -1).contiguous(), total


# ---------------------------------------------------------------------------
# C00 -- the modules under test are the tree this run means to measure.
# ---------------------------------------------------------------------------
def test_c00_the_modules_under_test_are_the_scratch_candidate():
    """Both modules this block adds must resolve under the measured tree."""
    import vllm_neuron

    from vllm_neuron.model.glm5_next import vision_encoder as tower_module
    from vllm_neuron.model.glm5_next.utils import (
        merge_vision_embeds as merge_module,
    )

    declared_root = os.environ.get("GLM53F_CANDIDATE_ROOT")
    root = (
        Path(declared_root).resolve() if declared_root else _repo_root_from_this_file()
    )
    _emit(
        "c00 import-origin",
        root=root,
        root_source="declared" if declared_root else "derived",
    )
    for name, module in (
        ("vllm_neuron", vllm_neuron),
        ("vision_encoder", tower_module),
        ("merge_vision_embeds", merge_module),
    ):
        resolved = Path(module.__file__).resolve()
        _emit(
            "c00 import-origin",
            module=name,
            IS_UNDER_THE_SCRATCH_CANDIDATE=int(root in resolved.parents),
            resolved=resolved,
        )
        assert root in resolved.parents, (
            f"{name} resolved to {resolved}, which is not under {root}"
        )


# ---------------------------------------------------------------------------
# C01 -- the tower equals the reference model on a regime-A grid, two arms.
# ---------------------------------------------------------------------------
def test_c01_the_tower_matches_the_reference_model_on_a_regime_a_grid():
    """Arm A at the merged output, arm B at the pre-merger stage."""
    name, height, width = REGIME_A_CASE
    cfg = _fork_config()
    grid_thw = _grid_for_case(height, width)
    pixel_values, total = _patch_rows(grid_thw, cfg, seed=1060)

    ref = _reference_tower()
    tower = _fork_tower_with_copied_weights(cfg, ref)

    _emit(
        "c01 case",
        name=name,
        height=height,
        width=width,
        grid=tuple(grid_thw[0].tolist()),
        patch_rows=total,
        GRID_IS_NON_SQUARE=int(int(grid_thw[0][1]) != int(grid_thw[0][2])),
        PATCH_ROWS_EXCEED_ONE_SEAM_CALL=int(total > 128),
    )

    with torch.no_grad():
        reference = ref(pixel_values, grid_thw=grid_thw)
        actual_pooled = tower(pixel_values, grid_thw)

    expected_pooled = reference.pooler_output
    assert actual_pooled.shape == expected_pooled.shape, (
        f"arm A shape {tuple(actual_pooled.shape)} != reference "
        f"{tuple(expected_pooled.shape)}"
    )
    _emit(
        "c01 arm-A pooler_output",
        max_abs_diff=_max_abs(actual_pooled, expected_pooled),
        rtol=RTOL,
        atol=ATOL,
        shape=tuple(actual_pooled.shape),
    )
    assert_close(actual_pooled, expected_pooled, rtol=RTOL, atol=ATOL)

    # Arm B: stop both sides one stage earlier, at the reference's
    # last_hidden_state (post-downsample, pre-merger). A mismatch here and not in
    # arm A means the merger; a mismatch in both means something upstream.
    with torch.no_grad():
        blocks_out = _fork_blocks_only(tower, pixel_values, grid_thw)
        actual_pre_merger = tower.adapter.regroup_and_downsample(
            tower.adapter.post_layernorm(blocks_out)
        )
    expected_pre_merger = reference.last_hidden_state
    _emit(
        "c01 arm-B last_hidden_state",
        max_abs_diff=_max_abs(actual_pre_merger, expected_pre_merger),
        rtol=RTOL,
        atol=ATOL,
        shape=tuple(actual_pre_merger.shape),
    )
    assert_close(actual_pre_merger, expected_pre_merger, rtol=RTOL, atol=ATOL)


def _fork_blocks_only(
    tower: Glm5NextVisionEncoder, pixel_values: torch.Tensor, grid_thw: torch.Tensor
) -> torch.Tensor:
    """Run the tower's patch embed, rope and block stack, stopping before the adapter."""
    from vllm_neuron.model.glm5_next.utils.vision_rope import (
        compute_vision_rotary_pos_emb,
    )

    hidden_states = tower.patch_embed(pixel_values)
    cos, sin = compute_vision_rotary_pos_emb(
        grid_thw, tower.head_dim, tower.spatial_merge_size
    )
    cos = cos.to(dtype=hidden_states.dtype)
    sin = sin.to(dtype=hidden_states.dtype)
    bound_min, bound_max = compute_attention_bounds(grid_thw)
    for block in tower.blocks:
        hidden_states = block(hidden_states, cos, sin, bound_min, bound_max)
    return hidden_states


# ---------------------------------------------------------------------------
# C02 -- the real processor's temporal slices are the same pixels.
# ---------------------------------------------------------------------------
def test_c02_the_two_temporal_patch_slices_are_the_same_pixels():
    """An IMAGE patch row's temporal slices are copies. Images only.

    Measured on the shipped IMAGE processor's own output, which broadcasts one
    frame across the temporal axis (``image_processing_glm5_next.py:206-215``).

    THIS ITEM IS NOT A GUARD ON THE TOWER, and it used to claim it was. Its
    earlier wording said that if a processor ever made the temporal slices
    differ "this item is what fires first"; it cannot, because it constructs
    ``Glm5NextImageProcessor()`` below and the image processor is exactly the one
    that always copies. The VIDEO processor does not copy -- it is a plain
    ``view`` over the real frame axis (``video_processing_glm5_next.py:286-316``)
    -- and the tower no longer depends on copies either way: since
    ``inc-glm53f-111`` it calls the patch-embed seam once per temporal slice with
    that slice's own filter and sums the outputs, which is the reference
    convolution's arithmetic for copies and non-copies alike. C09 is the item
    that measures the video case, and C11 is the image non-regression.

    What this item still earns its place for: it pins the IMAGE processor's
    broadcast as a measured property of the shipped code rather than a claim read
    off the source, so a future processor change that stopped broadcasting for
    images is reported here by name.
    """
    package = require_transformers_glm5_next()
    processor = package.Glm5NextImageProcessor()
    consts = GridConstants.from_processor(processor)

    # A small regime case keeps the resample cheap; the property under test is
    # not size-dependent.
    height, width = 112, 112
    image = torch.randint(0, 256, (3, height, width), dtype=torch.uint8)
    out = processor(images=[image], return_tensors="pt")
    pixel_values = out["pixel_values"]
    grid_thw = out["image_grid_thw"]

    # The processor may hand back a 2-D [total_patches, patch_dim] block or a
    # batched 3-D one, so the leading axes are collapsed and only the patch_dim
    # split is named. TINY["in_channels"] is the checkpoint's real 3.
    rows = pixel_values.reshape(
        -1,
        int(TINY["in_channels"]),
        int(consts.temporal_patch_size),
        int(consts.patch_size),
        int(consts.patch_size),
    )
    slice_gap = 0.0
    for temporal in range(1, int(consts.temporal_patch_size)):
        slice_gap = max(slice_gap, _max_abs(rows[:, :, 0], rows[:, :, temporal]))
    _emit(
        "c02 temporal-slices",
        grid=tuple(grid_thw[0].tolist()),
        patch_rows=int(pixel_values.shape[0]),
        temporal_patch_size=int(consts.temporal_patch_size),
        MAX_ABS_GAP_BETWEEN_SLICES=slice_gap,
    )
    assert slice_gap == 0.0, (
        f"the temporal slices of an IMAGE patch row differ by {slice_gap}; the "
        "image processor is expected to broadcast one frame across that axis "
        "(image_processing_glm5_next.py:206-215), so this is a change in the "
        "shipped image processor, not a tower defect -- the tower embeds each "
        "temporal slice with its own filter either way (C09, C11)"
    )


# ---------------------------------------------------------------------------
# C03 -- the rope positions tell a transposed grid apart (B91 N5).
# ---------------------------------------------------------------------------
def test_c03_the_rope_positions_distinguish_a_transposed_grid():
    """Elementwise against the reference helper, then a discrimination check."""
    from transformers.vision_utils import get_vision_position_ids

    cfg = _fork_config()
    _name, height, width = REGIME_A_CASE
    grid_thw = _grid_for_case(height, width)
    transposed = grid_thw.clone()
    transposed[0, 1], transposed[0, 2] = grid_thw[0, 2], grid_thw[0, 1]

    mine = vision_position_ids(grid_thw, cfg.spatial_merge_size)
    theirs = get_vision_position_ids(grid_thw, cfg.spatial_merge_size)
    mine_t = vision_position_ids(transposed, cfg.spatial_merge_size)

    mismatches = int((mine != theirs).any(dim=-1).sum())
    _emit(
        "c03 rope-positions",
        grid=tuple(grid_thw[0].tolist()),
        transposed=tuple(transposed[0].tolist()),
        rows=int(mine.shape[0]),
        ROWS_DISAGREEING_WITH_THE_REFERENCE=mismatches,
    )
    assert mine.shape == theirs.shape, (
        f"shape {tuple(mine.shape)} != reference {tuple(theirs.shape)}"
    )
    assert mismatches == 0, f"{mismatches} position rows disagree with the reference"

    # The discrimination B91 N5 asks for: this grid is non-square, so a
    # transposed grid must NOT produce the same position ids. If it did, nothing
    # downstream could catch a transposed grid either.
    assert int(grid_thw[0][1]) != int(grid_thw[0][2]), (
        "the chosen case must be non-square for this discrimination to mean anything"
    )
    differing = int((mine != mine_t).any(dim=-1).sum())
    _emit(
        "c03 transpose-discrimination",
        ROWS_THAT_CHANGE_UNDER_TRANSPOSE=differing,
        rows=int(mine.shape[0]),
    )
    assert differing > 0, (
        "a transposed grid produced identical position ids, so the patch order "
        "carries no orientation and B91 N5's gap is real in this file too"
    )


# ---------------------------------------------------------------------------
# C04 / C05 -- the merge writes exactly the placeholder positions.
# ---------------------------------------------------------------------------
def _merge_case(spans: list[tuple[int, int]], local_len: int, width: int, block_size: int):
    """Lay out cache blocks and positions for the given placeholder spans.

    Every slot a span does not claim is given a position outside the shard, which
    the merge must route to the discarded dummy row.
    """
    generator = torch.Generator().manual_seed(600)
    sentinel = local_len + 7  # outside the shard, and not the shard boundary
    positions: list[int] = []
    values: list[torch.Tensor] = []
    for start, length in spans:
        positions.extend(range(start, start + length))
        values.append(
            torch.randn(length, width, generator=generator, dtype=DTYPE) + 10.0
        )
    filled = torch.cat(values, dim=0)

    num_blocks = -(-len(positions) // block_size)
    padded = num_blocks * block_size
    pad = padded - len(positions)
    positions.extend([sentinel] * pad)
    rows = torch.cat(
        [filled, torch.zeros(pad, width, dtype=DTYPE)], dim=0
    ) if pad else filled

    blocks = tuple(
        rows[i * block_size : (i + 1) * block_size] for i in range(num_blocks)
    )
    position_grid = torch.tensor(positions, dtype=torch.int64).reshape(
        num_blocks, block_size
    )
    return blocks, position_grid, filled


def _run_merge(spans, local_len, width, block_size, rank=0):
    blocks, position_grid, filled = _merge_case(spans, local_len, width, block_size)
    base = (
        torch.arange(local_len, dtype=DTYPE).unsqueeze(1).expand(local_len, width) + 1.0
    ).contiguous()
    merged = merge_vision_embeddings(base.clone(), blocks, position_grid, rank)
    return base, merged, filled


def test_c04_the_merge_writes_every_placeholder_and_nothing_else():
    """Index-set equality, N/N placeholders, 0 misplaced, rest bit-identical."""
    local_len, width, block_size = 40, 24, 8
    spans = [(5, 8)]
    base, merged, filled = _run_merge(spans, local_len, width, block_size)

    expected_positions = {p for start, length in spans for p in range(start, start + length)}
    changed = {int(i) for i in (merged != base).any(dim=1).nonzero().flatten()}
    misplaced = sorted(changed - expected_positions)
    unwritten = sorted(expected_positions - changed)

    _emit(
        "c04 index-set",
        placeholders=len(expected_positions),
        WRITTEN=len(changed & expected_positions),
        MISPLACED=len(misplaced),
        UNWRITTEN=len(unwritten),
        INDEX_SETS_EQUAL=int(changed == expected_positions),
    )
    assert changed == expected_positions, (
        f"misplaced={misplaced} unwritten={unwritten}"
    )

    written_rows = merged[sorted(expected_positions)]
    _emit("c04 values", max_abs_diff=_max_abs(written_rows, filled))
    assert torch.equal(written_rows, filled), (
        "a placeholder row does not carry its own embedding"
    )

    untouched = sorted(set(range(local_len)) - expected_positions)
    _emit(
        "c04 untouched",
        rows=len(untouched),
        BIT_IDENTICAL=int(torch.equal(merged[untouched], base[untouched])),
    )
    assert torch.equal(merged[untouched], base[untouched])

    # Phase two: on a rank whose shard holds none of these positions, every write
    # goes to the dummy row and the hidden states come back untouched.
    base2, merged2, _ = _run_merge(spans, local_len, width, block_size, rank=3)
    _emit(
        "c04 out-of-shard",
        rank=3,
        ROWS_CHANGED=int((merged2 != base2).any(dim=1).sum()),
    )
    assert torch.equal(merged2, base2), (
        "positions outside this rank's shard were written instead of dropped"
    )


def test_c05_two_image_items_merge_into_their_own_spans():
    """B90 N4: one merge carrying two image items."""
    local_len, width, block_size = 60, 24, 8
    spans = [(4, 8), (30, 12)]
    base, merged, filled = _run_merge(spans, local_len, width, block_size)

    expected_positions = [p for start, length in spans for p in range(start, start + length)]
    changed = {int(i) for i in (merged != base).any(dim=1).nonzero().flatten()}
    _emit(
        "c05 two-items",
        items=len(spans),
        spans=spans,
        placeholders=len(expected_positions),
        INDEX_SETS_EQUAL=int(changed == set(expected_positions)),
    )
    assert changed == set(expected_positions)
    assert torch.equal(merged[expected_positions], filled), (
        "the two items' rows are not each in their own span"
    )


def test_c05b_a_width_disagreement_refuses_instead_of_slicing():
    """A fat cache row is a tower/decoder disagreement, not something to truncate."""
    local_len, width, block_size = 16, 24, 8
    blocks, position_grid, _ = _merge_case([(0, 8)], local_len, width * 2, block_size)
    base = torch.zeros(local_len, width, dtype=DTYPE)
    with pytest.raises(Glm5NextVisionMergeError, match="does not equal hidden_states width"):
        merge_vision_embeddings(base, blocks, position_grid, 0)
    with pytest.raises(Glm5NextVisionMergeError, match="at least one cache block"):
        merge_vision_embeddings(base, (), position_grid, 0)


# ---------------------------------------------------------------------------
# C06 / C07 -- the encoder-cache snapshot cluster, driven unbound.
# ---------------------------------------------------------------------------
class _FakeEncoderCache:
    """Only the three methods the snapshot cluster actually calls.

    ``contains`` and ``get`` are read by ``_snapshot_encoder_entries``
    (``neuron_model_runner.py:9078-9081``) and ``items`` by
    ``get_encoder_cache`` (``:9121``). Nothing else is provided, so an item that
    silently started depending on more of the real cache would fail here rather
    than pass on a coincidence.
    """

    def __init__(self, entries: dict[str, torch.Tensor]) -> None:
        self._entries = dict(entries)

    def contains(self, mm_hash: str) -> bool:
        return mm_hash in self._entries

    def get(self, mm_hash: str):
        return self._entries.get(mm_hash)

    def items(self):
        return self._entries.items()


def _runner_class():
    from vllm_neuron.vllm.worker import neuron_model_runner

    return neuron_model_runner.NeuronModelRunner


def _runner_stand_in(entries, requests):
    """Exactly the four attributes the snapshot cluster reads."""
    return SimpleNamespace(
        _encoder_cache_snapshot=None,
        _encoder_cache_snapshot_enabled=False,
        requests=requests,
        encoder_cache=_FakeEncoderCache(entries),
    )


def _request(mm_hashes: list[str]):
    return SimpleNamespace(
        mm_features=[SimpleNamespace(identifier=h) for h in mm_hashes]
    )


def test_c06_the_encoder_cache_snapshot_round_trips_exactly():
    """Max abs diff EXACTLY 0.0 across the serialise / restore boundary."""
    runner = _runner_class()
    generator = torch.Generator().manual_seed(606)
    entries = {
        "img-a": torch.randn(8, 24, generator=generator, dtype=DTYPE),
        "img-b": torch.randn(12, 24, generator=generator, dtype=DTYPE),
    }
    requests = {"req-0": _request(["img-a", "img-b"])}
    stand_in = _runner_stand_in(entries, requests)
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="req-0")],
        scheduled_encoder_inputs={},
    )

    runner.enable_encoder_cache_snapshot(stand_in)
    assert stand_in._encoder_cache_snapshot_enabled is True
    assert stand_in._encoder_cache_snapshot is None

    runner._snapshot_encoder_entries(stand_in, scheduler_output)
    runner._snapshot_encoder_entries(stand_in, scheduler_output)  # accumulates
    blob = runner.get_encoder_cache(stand_in)

    restored = {
        mm_hash: torch.load(io.BytesIO(payload), weights_only=True)
        for mm_hash, payload in blob.items()
    }
    worst = max(_max_abs(restored[k], entries[k]) for k in entries)
    _emit(
        "c06 round-trip",
        entries_in=len(entries),
        ENTRIES_OUT=len(restored),
        keys_match=int(set(restored) == set(entries)),
        MAX_ABS_DIFF=worst,
    )
    assert set(restored) == set(entries), (
        f"round-trip keys {sorted(restored)} != {sorted(entries)}"
    )
    assert worst == 0.0, f"round-trip changed a value by {worst}"

    runner.clear_encoder_cache_snapshot(stand_in)
    _emit(
        "c06 cleared",
        enabled=int(stand_in._encoder_cache_snapshot_enabled),
        snapshot_is_none=int(stand_in._encoder_cache_snapshot is None),
    )
    assert stand_in._encoder_cache_snapshot_enabled is False
    assert stand_in._encoder_cache_snapshot is None


def test_c07_a_text_only_request_snapshots_no_entries():
    """0 entries, and the snapshot is an empty dict rather than a fallback."""
    runner = _runner_class()
    entries = {"img-a": torch.ones(4, 24, dtype=DTYPE)}
    requests = {"req-text": _request([])}
    stand_in = _runner_stand_in(entries, requests)
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="req-text")],
        scheduled_encoder_inputs={},
    )

    runner.enable_encoder_cache_snapshot(stand_in)
    runner._snapshot_encoder_entries(stand_in, scheduler_output)
    blob = runner.get_encoder_cache(stand_in)

    _emit(
        "c07 text-only",
        cache_holds=len(entries),
        SNAPSHOT_ENTRIES=len(stand_in._encoder_cache_snapshot),
        RETURNED_ENTRIES=len(blob),
        snapshot_is_dict=int(isinstance(stand_in._encoder_cache_snapshot, dict)),
    )
    # The empty dict matters twice over: it is the 0-entry reading, and it is
    # what keeps get_encoder_cache on the snapshot branch. Had the snapshot
    # stayed None, the method would have fallen back to the live cache and
    # returned the unrelated entry above.
    assert isinstance(stand_in._encoder_cache_snapshot, dict)
    assert len(stand_in._encoder_cache_snapshot) == 0
    assert len(blob) == 0, f"a text-only request produced {len(blob)} cache entries"


# ---------------------------------------------------------------------------
# C08 -- the activation map has not drifted from the adapter's.
# ---------------------------------------------------------------------------
def test_c08_the_activation_map_matches_the_adapters():
    """The tower keeps its own copy because the adapter's is private to it."""
    from vllm_neuron.model.glm5_next import vision_adapter

    theirs = vision_adapter._ACTIVATIONS
    _emit(
        "c08 activation-map",
        tower=sorted(_ACTIVATIONS),
        adapter=sorted(theirs),
        KEYS_EQUAL=int(set(_ACTIVATIONS) == set(theirs)),
    )
    assert set(_ACTIVATIONS) == set(theirs), (
        "the tower's activation map has drifted from the adapter's"
    )
    for key in _ACTIVATIONS:
        assert _ACTIVATIONS[key] is theirs[key], f"{key} maps to a different class"


# ---------------------------------------------------------------------------
# C09-C11 -- `inc-glm53f-111`: every temporal slice reaches the patch embedding.
#
# The tower used to fold the reference convolution's weight over the temporal
# axis and feed slice 0 only. That is exact when the slices are copies, which the
# IMAGE processor guarantees and the VIDEO processor does not: `patchify` is a
# plain `view` over the real frame axis (`video_processing_glm5_next.py:286-316`),
# so a two-frame video puts frames 0 and 1 in one patch row and the fold dropped
# frame 1. These three items measure the fix, name its cause, and pin that the
# image path did not move.
# ---------------------------------------------------------------------------
def _real_video_rows():
    """Two frames that DIFFER, through the shipped video processor.

    The frames are two different constants, so the temporal slices of every patch
    row must differ; a helper that filled both frames alike would make C09
    vacuous no matter what the tower did, which is the trap the `-110` reading
    was rewritten to avoid.
    """
    package = require_transformers_glm5_next()
    processor = package.Glm5NextVideoProcessor()
    consts = GridConstants.from_processor(processor)
    frames = torch.stack(
        [
            torch.full((3, 112, 112), 40, dtype=torch.uint8),
            torch.full((3, 112, 112), 200, dtype=torch.uint8),
        ]
    )
    input_frames_differ = int(bool((frames[0] != frames[1]).any()))
    out = processor(videos=[frames], do_sample_frames=False, return_tensors="pt")
    rows = out["pixel_values_videos"]
    rows = rows.reshape(-1, rows.shape[-1]).to(DTYPE)
    return rows, out["video_grid_thw"], consts, input_frames_differ


def _slice_gap(rows: torch.Tensor, consts) -> float:
    """Largest absolute difference between temporal slice 0 and slice 1 of a row."""
    reshaped = rows.reshape(
        rows.shape[0],
        int(TINY["in_channels"]),
        int(consts.temporal_patch_size),
        int(consts.patch_size),
        int(consts.patch_size),
    )
    gap = 0.0
    for temporal in range(1, int(consts.temporal_patch_size)):
        gap = max(gap, _max_abs(reshaped[:, :, 0], reshaped[:, :, temporal]))
    return gap


def _folded_single_slice_embed(
    tower: Glm5NextVisionEncoder,
    pixel_values: torch.Tensor,
    filters: torch.Tensor | None = None,
) -> torch.Tensor:
    """The pre-`-111` computation, kept here so the fix has something to beat.

    Line for line what the tower did before this increment: sum the per-slice
    filters into one depth-1 filter, feed temporal slice 0, call the same seam.
    It is built from the tower's OWN filters (or from ``filters`` when a caller
    wants a modified weight), so it cannot drift away from the weights the tower
    is holding.
    """
    source = tower.patch_embed_filters.detach() if filters is None else filters
    folded = source.sum(dim=0, keepdim=True)
    total = pixel_values.shape[0]
    rows = pixel_values.reshape(
        total,
        tower.in_channels,
        tower.temporal_patch_size,
        tower.patch_size,
        tower.patch_size,
    )[:, :, :1]
    out = []
    for start in range(0, total, _PATCH_ROWS_PER_CALL):
        chunk = rows[start : start + _PATCH_ROWS_PER_CALL]
        embedded = _seam_patch_embed(
            chunk,
            folded,
            tower.patch_size,
            bias=tower.patch_embed_bias.detach(),
        )
        out.append(embedded.reshape(chunk.shape[0], tower.hidden_size))
    return torch.cat(out, dim=0)


def _meets_the_registered_pair(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Ask the SAME instrument the criterion uses, rather than a hand-rolled bound.

    A control that decided "close enough" with its own comparison could disagree
    with the assertion the criterion is judged by. This calls ``assert_close`` at
    the registered pair and reports whether it held.
    """
    try:
        assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    except AssertionError:
        return False
    return True


def test_c09_a_two_frame_video_embeds_both_temporal_slices():
    """The fix, measured against the reference convolution on real video rows.

    The reference ``Glm5NextVisionPatchEmbed`` is a real ``nn.Conv3d`` whose
    kernel depth is ``temporal_patch_size`` and whose stride equals its kernel
    (``modeling_glm5_next.py:1720-1721``), so it computes
    ``sum_t W[:, :, t] * x_t`` over the whole temporal axis. The tower must reach
    the same value through the depth-1 seam, called once per slice.

    Two input controls keep the reading from being vacuous -- the temporal axis
    really carries two slices, and the two frames really differ -- and one control
    FIRES: the pre-`-111` folded computation must fail the registered pair on
    these same rows. If it were to pass, the fix would be unmeasurable here and
    the item raises saying so instead of going green.
    """
    cfg = _fork_config()
    ref = _reference_tower()
    tower = _fork_tower_with_copied_weights(cfg, ref)

    rows, grid, consts, input_frames_differ = _real_video_rows()
    gap = _slice_gap(rows, consts)

    with torch.no_grad():
        expected = ref.patch_embed(rows)
        actual = tower.patch_embed(rows)
        folded = _folded_single_slice_embed(tower, rows)

    fold_error = _max_abs(folded, expected)
    fold_meets = _meets_the_registered_pair(folded, expected)
    _emit(
        "v-slice-fold",
        GAP=fold_error,
        FOLDED_MEETS_THE_PAIR=int(fold_meets),
        RTOL=RTOL,
        ATOL=ATOL,
    )
    _emit(
        "c09 video-temporal-slices",
        grid=tuple(grid[0].tolist()),
        patch_rows=int(rows.shape[0]),
        temporal_patch_size=int(consts.temporal_patch_size),
        INPUT_FRAMES_DIFFER=input_frames_differ,
        VIDEO_TEMPORAL_SLICE_GAP=gap,
        PER_SLICE_MAX_ABS_ERROR=_max_abs(actual, expected),
        rtol=RTOL,
        atol=ATOL,
    )

    assert int(consts.temporal_patch_size) == 2, (
        "the temporal axis carries one slice, so a per-slice sum and a "
        "single-slice fold are the same computation and this item measures nothing"
    )
    assert input_frames_differ == 1, (
        "the two input frames are identical, so the video rows carry no temporal "
        "signal and this item measures nothing"
    )
    assert gap > 0.0, (
        f"the video processor returned patch rows whose temporal slices agree "
        f"(gap {gap}); the fold this item is built to beat would be exact on "
        "these rows and the reading would be vacuous"
    )
    assert not fold_meets, (
        f"the pre-inc-glm53f-111 fold (filters summed over the temporal axis, "
        f"slice 0 fed) MET the registered pair on these video rows at max abs "
        f"error {fold_error}, so this item cannot tell the fix from the defect it "
        "replaces; the reading is vacuous and must not be recorded as a pass"
    )

    assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_c10_zeroing_the_second_temporal_filter_restores_the_fold():
    """The cause, named: C09's gap IS the dropped slice.

    With ``W[:, :, 1]`` zeroed the reference convolution has nothing to
    contribute from the second temporal slice, so summing the filters over that
    axis and feeding slice 0 becomes exact again -- and the old computation meets
    the registered pair on the very rows where C09 showed it failing. That rules
    out a dtype, a layout error or a tolerance that is simply too tight as the
    explanation for C09.
    """
    cfg = _fork_config()
    ref = _reference_tower()
    tower = _fork_tower_with_copied_weights(cfg, ref)
    rows, _, consts, _ = _real_video_rows()

    zeroed_weight = ref.patch_embed.proj.weight.detach().clone()
    zeroed_weight[:, :, 1] = 0.0
    zeroed_reference = copy.deepcopy(ref.patch_embed)
    with torch.no_grad():
        zeroed_reference.proj.weight.copy_(zeroed_weight)
        expected = zeroed_reference(rows)
        folded = _folded_single_slice_embed(
            tower, rows, filters=patch_embed_filters_from_conv3d(zeroed_weight)
        )

    _emit(
        "c10 zeroed-second-slice",
        SECOND_FILTER_SLICE_ABS_SUM=float(zeroed_weight[:, :, 1].abs().sum()),
        FIRST_FILTER_SLICE_ABS_SUM=float(zeroed_weight[:, :, 0].abs().sum()),
        FOLD_MAX_ABS_ERROR=_max_abs(folded, expected),
        rtol=RTOL,
        atol=ATOL,
    )

    assert float(zeroed_weight[:, :, 1].abs().sum()) == 0.0, (
        "the second temporal filter slice was not actually zeroed, so this item "
        "is not the control it claims to be"
    )
    assert float(zeroed_weight[:, :, 0].abs().sum()) > 0.0, (
        "the first temporal filter slice is all zeros, so both sides would agree "
        "on an empty computation and the control would be vacuous"
    )
    assert _slice_gap(rows, consts) > 0.0, (
        "these are not the differing-slice rows C09 measured, so agreement here "
        "would not explain C09's gap"
    )

    assert_close(folded, expected, rtol=RTOL, atol=ATOL)


def test_c11_the_image_path_is_unchanged_by_the_per_slice_sum():
    """Image non-regression, measured -- not asserted bit-equal.

    For rows whose temporal slices are copies, ``sum_t conv2d(x_0, W_t)`` and
    ``conv2d(x_0, sum_t W_t)`` are the same linear map, so the per-slice sum
    cannot change the image answer in exact arithmetic. Finite precision does not
    inherit that: the two differ in summation order, and float32 addition is not
    associative. So this is judged at the REGISTERED pair, exactly like C01, and
    the fold's own error is printed beside it rather than being required to be
    zero.
    """
    cfg = _fork_config()
    ref = _reference_tower()
    tower = _fork_tower_with_copied_weights(cfg, ref)

    package = require_transformers_glm5_next()
    processor = package.Glm5NextImageProcessor()
    consts = GridConstants.from_processor(processor)
    image = torch.randint(0, 256, (3, 112, 112), dtype=torch.uint8)
    out = processor(images=[image], return_tensors="pt")
    rows = out["pixel_values"]
    rows = rows.reshape(-1, rows.shape[-1]).to(DTYPE)

    with torch.no_grad():
        expected = ref.patch_embed(rows)
        actual = tower.patch_embed(rows)
        folded = _folded_single_slice_embed(tower, rows)

    _emit(
        "c11 image-non-regression",
        grid=tuple(out["image_grid_thw"][0].tolist()),
        patch_rows=int(rows.shape[0]),
        IMAGE_TEMPORAL_SLICE_GAP=_slice_gap(rows, consts),
        PER_SLICE_MAX_ABS_ERROR=_max_abs(actual, expected),
        FOLDED_MAX_ABS_ERROR=_max_abs(folded, expected),
        rtol=RTOL,
        atol=ATOL,
    )

    assert _slice_gap(rows, consts) == 0.0, (
        "the image processor's temporal slices differ here, so this is not the "
        "copies case the algebraic identity is about (see C02)"
    )

    assert_close(actual, expected, rtol=RTOL, atol=ATOL)
