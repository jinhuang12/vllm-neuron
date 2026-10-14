# SPDX-License-Identifier: Apache-2.0
"""Tests for the vision tower's inputs and for merging its embeddings into the decoder's hidden states.

The tower tests check the processor's patch rows and the rope positions against transformers'
reference. The merge tests check that exactly the placeholder positions are written, that
positions outside this rank's shard are dropped, and that the encoder-cache snapshot
round-trips unchanged.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
import torch

# transformers is a hard floor for this fork -- requirements/core.txt declares
# transformers>=5.16.1,<6.0.0 -- so the skip is taken here rather than in a fixture, and an
# older transformers skips this file by name instead of failing every other file's collection
# with an import error. Skipping is the only alternative to importing: nothing below means
# anything without the reference, so this file must never report a pass when it could not
# measure against it.
pytest.importorskip(
    "transformers.models.glm5_next",
    reason="transformers>=5.16.1 carries the glm5_next reference this file measures against",
)

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
from vllm_neuron.model.glm5_next.vision_encoder import _ACTIVATIONS

# The dtype of every tensor these tests build.
DTYPE = torch.float32

# The size the rope test runs. It resamples nothing, and its grid is non-square.
REGIME_A_CASE = ("wide_strip", 1400, 56)

# The geometry fields stay at the checkpoint's real values so the grid helper applies
# unchanged; only the widths and the depth shrink.
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


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    # Detached before the scalar read: some of these are module parameters, and torch warns
    # when a tensor still carrying requires_grad becomes a float.
    return float((a.detach() - b.detach()).abs().max())


def _fork_config() -> Glm5NextVisionConfig:
    return Glm5NextVisionConfig(**TINY)


def _grid_for_case(height: int, width: int) -> torch.Tensor:
    """Derive the case's grid through the preprocessing helper, not from a literal."""
    package = require_transformers_glm5_next()
    consts = GridConstants.from_processor(package.Glm5NextImageProcessor())
    spec = image_grid_spec(consts, height, width)
    return torch.tensor([spec.thw], dtype=torch.long)


def test_the_two_temporal_patch_slices_are_the_same_pixels():
    """The patch embedding's temporal sum is exact only if this holds.

    Measured on the shipped processor's own output. If a future checkpoint or processor made
    the temporal slices differ, the tower's single-slice input would silently drop half the
    signal, and this is what fires first.
    """
    package = require_transformers_glm5_next()
    processor = package.Glm5NextImageProcessor()
    consts = GridConstants.from_processor(processor)

    # A small size keeps the resample cheap; the property under test is not size-dependent.
    height, width = 112, 112
    image = torch.randint(0, 256, (3, height, width), dtype=torch.uint8)
    out = processor(images=[image], return_tensors="pt")
    pixel_values = out["pixel_values"]

    # The processor may hand back a 2-D [total_patches, patch_dim] block or a batched 3-D one,
    # so the leading axes are collapsed and only the patch_dim split is named. 3 is the
    # checkpoint's channel count.
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
    assert slice_gap == 0.0, (
        f"the temporal slices of a patch row differ by {slice_gap}; the tower's patch "
        "embedding sums the reference convolution over that axis and feeds one slice, which "
        "is exact only while the slices are copies"
    )


def test_the_rope_positions_distinguish_a_transposed_grid():
    """Elementwise against the reference helper, then a transposed grid must differ.

    Neither the adapter nor the reference can tell a (4, 2) grid from a (2, 4) one, because
    both treat any four consecutive tokens as one merge block. The position ids can.
    """
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
    assert mine.shape == theirs.shape, (
        f"shape {tuple(mine.shape)} != reference {tuple(theirs.shape)}"
    )
    assert mismatches == 0, f"{mismatches} position rows disagree with the reference"

    assert int(grid_thw[0][1]) != int(grid_thw[0][2]), (
        "the chosen size must be non-square for the comparison below to mean anything"
    )
    differing = int((mine != mine_t).any(dim=-1).sum())
    assert differing > 0, (
        "a transposed grid produced identical position ids, so the patch order carries no "
        "orientation at all"
    )


def _merge_case(spans: list[tuple[int, int]], local_len: int, width: int, block_size: int):
    """Lay out cache blocks and positions for the given placeholder spans.

    Every slot a span does not claim is given a position outside the shard, which the merge
    must route to the discarded dummy row.
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


def test_the_merge_writes_every_placeholder_and_nothing_else():
    """Index-set equality, every placeholder carrying its own row, the rest bit-identical."""
    local_len, width, block_size = 40, 24, 8
    spans = [(5, 8)]
    base, merged, filled = _run_merge(spans, local_len, width, block_size)

    expected_positions = {p for start, length in spans for p in range(start, start + length)}
    changed = {int(i) for i in (merged != base).any(dim=1).nonzero().flatten()}
    misplaced = sorted(changed - expected_positions)
    unwritten = sorted(expected_positions - changed)

    assert changed == expected_positions, (
        f"misplaced={misplaced} unwritten={unwritten}"
    )

    written_rows = merged[sorted(expected_positions)]
    assert torch.equal(written_rows, filled), (
        "a placeholder row does not carry its own embedding"
    )

    untouched = sorted(set(range(local_len)) - expected_positions)
    assert torch.equal(merged[untouched], base[untouched])

    # On a rank whose shard holds none of these positions, every write goes to the dummy row
    # and the hidden states come back untouched.
    base2, merged2, _ = _run_merge(spans, local_len, width, block_size, rank=3)
    assert torch.equal(merged2, base2), (
        "positions outside this rank's shard were written instead of dropped"
    )


def test_two_image_items_merge_into_their_own_spans():
    """One merge carrying two image items."""
    local_len, width, block_size = 60, 24, 8
    spans = [(4, 8), (30, 12)]
    base, merged, filled = _run_merge(spans, local_len, width, block_size)

    expected_positions = [p for start, length in spans for p in range(start, start + length)]
    changed = {int(i) for i in (merged != base).any(dim=1).nonzero().flatten()}
    assert changed == set(expected_positions)
    assert torch.equal(merged[expected_positions], filled), (
        "the two items' rows are not each in their own span"
    )


def test_a_width_disagreement_refuses_instead_of_slicing():
    """A fat cache row is a tower/decoder disagreement, not something to truncate."""
    local_len, width, block_size = 16, 24, 8
    blocks, position_grid, _ = _merge_case([(0, 8)], local_len, width * 2, block_size)
    base = torch.zeros(local_len, width, dtype=DTYPE)
    with pytest.raises(Glm5NextVisionMergeError, match="does not equal hidden_states width"):
        merge_vision_embeddings(base, blocks, position_grid, 0)
    with pytest.raises(Glm5NextVisionMergeError, match="at least one cache block"):
        merge_vision_embeddings(base, (), position_grid, 0)


class _FakeEncoderCache:
    """Only the three methods the snapshot cluster calls.

    ``contains`` and ``get`` are read by ``_snapshot_encoder_entries`` and ``items`` by
    ``get_encoder_cache``. Nothing else is provided, so a test that started depending on more
    of the real cache would fail here rather than pass on a coincidence.
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
    """Exactly the four attributes the snapshot cluster reads.

    The methods are called unbound on this stand-in because the only production caller of
    ``_snapshot_encoder_entries`` sits inside ``execute_model``, which on this branch still
    reaches a raising stub: driven through it these tests would fail on that raise instead of
    on an assertion about the cache.
    """
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


def test_the_encoder_cache_snapshot_round_trips_exactly():
    """Max abs diff exactly 0.0 across the serialise / restore boundary."""
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
    assert set(restored) == set(entries), (
        f"round-trip keys {sorted(restored)} != {sorted(entries)}"
    )
    assert worst == 0.0, f"round-trip changed a value by {worst}"

    runner.clear_encoder_cache_snapshot(stand_in)
    assert stand_in._encoder_cache_snapshot_enabled is False
    assert stand_in._encoder_cache_snapshot is None


def test_a_text_only_request_snapshots_no_entries():
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

    # The empty dict matters twice over: it is the zero-entry reading, and it is what keeps
    # get_encoder_cache on the snapshot branch. Had the snapshot stayed None, the method would
    # have fallen back to the live cache and returned the unrelated entry above.
    assert isinstance(stand_in._encoder_cache_snapshot, dict)
    assert len(stand_in._encoder_cache_snapshot) == 0
    assert len(blob) == 0, f"a text-only request produced {len(blob)} cache entries"


def test_the_activation_map_matches_the_adapters():
    """The tower keeps its own copy because the adapter's is private to it."""
    from vllm_neuron.model.glm5_next import vision_adapter

    theirs = vision_adapter._ACTIVATIONS
    assert set(_ACTIVATIONS) == set(theirs), (
        "the tower's activation map has drifted from the adapter's"
    )
    for key in _ACTIVATIONS:
        assert _ACTIVATIONS[key] is theirs[key], f"{key} maps to a different class"
