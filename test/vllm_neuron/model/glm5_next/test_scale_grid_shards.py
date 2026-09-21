# SPDX-License-Identifier: Apache-2.0
"""The block-FP8 scale grid of a sharded MLA projection shards with its weight.

The widths here are wide enough that a grid has blocks to divide; on the
default miniature every grid is a single tile and a whole grid describes a
rank's half as well as the whole tensor."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from vllm_neuron.model.glm5_next import model_fp8 as _MODEL_FP8
from vllm_neuron.model.glm5_next import weight_loaders_fp8 as _WL_FP8
from vllm_neuron.model.glm5_next.config import Glm5NextConfig, Glm5NextTextConfig
from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextForConditionalGeneration
from vllm_neuron.model.glm5_next.quantization import DEFAULT_WEIGHT_BLOCK_SIZE
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    FP8_SCALE_SUFFIX,
    Glm5NextWeightMapError,
    block_grid_shape,
)

from .test_load_weights import (  # noqa: F401 -- fixtures are used by name
    MINI_ALL_DENSE_FIRST_K,
    MINI_LAYERS,
    MINI_ROUTED_EXPERTS,
    _keys_of,
    _mappings_for,
    _mla_key_overrides,
    _write_miniature_checkpoint,
    keep_the_loaded_tensors,
    single_rank_process_group,
)
from .test_tensor_parallel_shards import (
    SHARD_INTERMEDIATE,
    SHARD_LINEAR_ATTN,
    SHARD_WORLD,
    _LEAF_WEIGHT_SUFFIX,
    _max_abs_diff,
    _seed_page_cache_signal,
    _shard_key_overrides,
    _shard_pattern,
)


#: Head widths of two 128-tiles each, so every grid has blocks to divide.
GRID_SHARD_MLA_WIDTHS = dict(
    hidden_size=128,
    num_attention_heads=4,
    qk_nope_head_dim=256,
    qk_rope_head_dim=0,
    v_head_dim=256,
    q_lora_rank=32,
    kv_lora_rank=32,
)

GRID_SHARD_LEAVES = ("q_b_proj", "o_proj")


def _grid_shard_config() -> Glm5NextConfig:
    """``_shard_config`` with MLA head widths wide enough for a grid to have blocks to divide."""
    return Glm5NextConfig(
        text_config=Glm5NextTextConfig(
            num_hidden_layers=MINI_LAYERS,
            n_routed_experts=MINI_ROUTED_EXPERTS,
            n_shared_experts=0,
            first_k_dense_replace=MINI_ALL_DENSE_FIRST_K,
            tie_word_embeddings=False,
            linear_attn_config=SHARD_LINEAR_ATTN,
            intermediate_size=SHARD_INTERMEDIATE,
            **GRID_SHARD_MLA_WIDTHS,
        )
    )


def _grid_shard_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint at the closed-form shapes, with position-identifying grids for ``q_b_proj`` and ``o_proj``."""
    config = _grid_shard_config()
    mappings = _mappings_for(config)
    reference = Glm5NextForConditionalGeneration(config)

    grids: dict[str, torch.Tensor] = {}
    for path, module in reference.named_modules():
        if type(module).__name__ != "Glm5NextMLAAttention":
            continue
        widths = {
            name: (idim, odim) for name, idim, odim in module.projection_widths()
        }
        for leaf in GRID_SHARD_LEAVES:
            grid_param = f"{path}.{leaf}_{FP8_SCALE_SUFFIX}"
            if grid_param not in mappings:
                continue
            idim, odim = widths[leaf]
            shape = block_grid_shape((odim, idim), DEFAULT_WEIGHT_BLOCK_SIZE)
            dim = _mla_grid_shard_dim(leaf)
            for key in _keys_of(mappings, grid_param):
                grids[key] = _shard_pattern(shape, dim, torch.float32)
    assert len(grids) == len(GRID_SHARD_LEAVES), (
        f"wrote position-identifying values for {len(grids)} grids, not "
        f"{len(GRID_SHARD_LEAVES)}, so one projection is absent from the map"
    )

    # The MLA projections are written at their own closed-form shapes; the
    # shard table's tensors cover the other families only.
    module_shapes = _mla_key_overrides(reference, mappings)
    shard_over = {
        key: tensor
        for key, tensor in _shard_key_overrides(reference, mappings).items()
        if key not in module_shapes
    }
    extra_overrides = {**shard_over, **grids}

    directory = tmp_path / "grid-shard"
    written = _write_miniature_checkpoint(
        directory, mappings, reference, extra_overrides=extra_overrides
    )
    assert written, "the grid fixture wrote no tensors"
    return directory


def _load_grid_at_world(
    directory: Path, world_size: int, rank: int, monkeypatch
) -> Glm5NextForConditionalGeneration:
    """Build the wide-head model at a synthetic world size and rank, then load."""
    monkeypatch.setattr(_MODEL_FP8, "_resolve_world_size", lambda: world_size)
    monkeypatch.setattr(_MODEL_FP8, "_resolve_rank", lambda: rank)
    model = Glm5NextForConditionalGeneration(_grid_shard_config())
    assert model.world_size == world_size, (
        f"the model resolved world size {model.world_size}, not the patched "
        f"{world_size}"
    )
    _seed_page_cache_signal()
    model.load_weights(str(directory), torch.device("cpu"), None)
    return model


def _mla_grid_shard_dim(leaf: str) -> int:
    """The dim the production table shards this projection's weight on."""
    declared = _MODEL_FP8._SHARD_GEOMETRY["Glm5NextMLAAttention"][
        f"{leaf}{_LEAF_WEIGHT_SUFFIX}"
    ]
    return declared.shard_dim


def _mla_grids(
    model: Glm5NextForConditionalGeneration,
) -> dict[str, tuple[tuple[int, ...], torch.Tensor]]:
    """``dotted leaf -> (weight shape, grid)`` per sharded scaled projection."""
    found: dict[str, tuple[tuple[int, ...], torch.Tensor]] = {}
    for path, module in model.named_modules():
        if type(module).__name__ != "Glm5NextMLAAttention":
            continue
        for leaf in GRID_SHARD_LEAVES:
            weight = getattr(module, f"{leaf}{_LEAF_WEIGHT_SUFFIX}", None)
            grid = getattr(module, f"{leaf}_{FP8_SCALE_SUFFIX}", None)
            if weight is None or grid is None:
                continue
            found[f"{path}.{leaf}"] = (tuple(weight.shape), grid.data)
    return found


def test_a_sharded_projections_scale_grid_shards_with_its_weight(
    keep_the_loaded_tensors, tmp_path: Path, monkeypatch, single_rank_process_group
) -> None:
    """Each rank's grid is the block slice matching its weight shard; with grid geometry withheld the load refuses."""
    from vllm_neuron.functional.attention import mla_projections

    mla_projections.reset_mla_projection_dispatch_counters()

    directory = _grid_shard_checkpoint(tmp_path)

    whole = _load_grid_at_world(directory, 1, 0, monkeypatch)
    whole_grids = _mla_grids(whole)
    assert whole_grids, "no MLA module reported both a weight and a scale grid"

    per_rank = {
        rank: _mla_grids(_load_grid_at_world(directory, SHARD_WORLD, rank, monkeypatch))
        for rank in range(SHARD_WORLD)
    }
    for rank, found in sorted(per_rank.items()):
        assert set(found) == set(whole_grids), (
            f"rank {rank} reported different subjects from the world-1 load: only "
            f"whole {sorted(set(whole_grids) - set(found))}, only rank "
            f"{sorted(set(found) - set(whole_grids))}"
        )

    narrowed = 0
    for dotted, (_, whole_grid) in sorted(whole_grids.items()):
        leaf = dotted.rpartition(".")[2]
        dim = _mla_grid_shard_dim(leaf)
        for rank, found in sorted(per_rank.items()):
            weight_shape, grid = found[dotted]
            expected = block_grid_shape(weight_shape, DEFAULT_WEIGHT_BLOCK_SIZE)
            assert tuple(grid.shape) == expected, (
                f"{dotted} loaded a {weight_shape} weight at rank {rank} of world "
                f"size {SHARD_WORLD} with a {tuple(grid.shape)} grid; a blockwise "
                f"grid holds one value per tile, so this weight's grid is "
                f"{expected}. A grid describing the unsharded tensor scales the "
                f"wrong blocks"
            )
            extent = expected[dim]
            mine = whole_grid.narrow(dim, rank * extent, extent)
            difference = _max_abs_diff(grid, mine)
            assert difference == 0.0, (
                f"{dotted} at rank {rank} is not blocks "
                f"[{rank * extent}:{(rank + 1) * extent}] of dim {dim} of the "
                f"unsharded grid: they differ by {difference}. The shape is right, "
                f"so this is an offset defect: the rank is being handed another "
                f"rank's tile scales"
            )
        if tuple(whole_grid.shape) != tuple(per_rank[0][dotted][1].shape):
            narrowed += 1
    assert narrowed == len(whole_grids), (
        f"only {narrowed} of {len(whole_grids)} grids differ from their unsharded "
        f"shape, so for the rest the test would pass whether or not the grid was "
        f"sharded"
    )

    landed = _MODEL_FP8._shard_geometry_for

    def without_grid_geometry(module, leaf, world_size):
        if leaf.endswith(f"_{FP8_SCALE_SUFFIX}"):
            return None
        return landed(module, leaf, world_size)

    monkeypatch.setattr(_MODEL_FP8, "_shard_geometry_for", without_grid_geometry)
    with pytest.raises(Glm5NextWeightMapError) as refusal:
        _load_grid_at_world(directory, SHARD_WORLD, 0, monkeypatch)
    message = str(refusal.value)
    assert "scale grid shape" in message, (
        f"the load refused for a reason other than the missing grid geometry: "
        f"{message}"
    )

    dispatches = mla_projections.mla_projection_dispatch_counters()
    assert dispatches == (0, 0), (
        f"the projection seam counted {dispatches} (nki, torch_fallback) during "
        f"the loads; weight loading dequantises and shards, it does not project"
    )


def test_every_sharded_mla_head_width_is_a_whole_quant_block() -> None:
    """Each sharded MLA head width is a whole checkpoint tile and a whole consumer block, so whole-head shards need no pad."""
    config = Glm5NextTextConfig()

    head_widths = {
        "q_b_proj_weight": (0, config.qk_nope_head_dim + config.qk_rope_head_dim),
        "kv_b_proj_weight": (0, config.qk_nope_head_dim + config.v_head_dim),
        "o_proj_weight": (1, config.v_head_dim),
    }

    def part_tile(shard_dim: int, head_width: int) -> int:
        """The remainder one head leaves in a checkpoint tile; 0 is whole."""
        return head_width % DEFAULT_WEIGHT_BLOCK_SIZE[shard_dim]

    consumer_block = _WL_FP8.dense_consumer_block_quant_size()

    def part_block(head_width: int, block: int = 0) -> int:
        """The remainder one head leaves in a consumer block; 0 is whole.

        ``block`` defaults to the product's own granularity. The control below
        passes a moved one, so the predicate can be read on a width that fails
        this rule and no other -- impossible at the production numbers, where the
        consumer's block and the checkpoint's tile are one number.
        """
        return head_width % (block or consumer_block)

    offenders = {
        leaf: width
        for leaf, (dim, width) in head_widths.items()
        if part_tile(dim, width)
    }

    block_offenders = {
        leaf: width
        for leaf, (_dim, width) in head_widths.items()
        if part_block(width)
    }

    # THE PART-TILE PREDICATE, PUT TO A WIDTH IT MUST FLAG. A DeepSeek-style split
    # of 128 nope plus 64 rope gives a head 192, one and a half tiles. Without this
    # the empty result above would also be what a predicate that had stopped
    # discriminating returns.
    control_widths = {"q_b_proj_weight": (0, 128 + 64)}
    control_offenders = {
        leaf: width
        for leaf, (dim, width) in control_widths.items()
        if part_tile(dim, width)
    }
    assert control_offenders, (
        "the part-tile predicate did not flag a 192-wide head against a "
        f"{DEFAULT_WEIGHT_BLOCK_SIZE} block, so it cannot detect a violation and "
        "the empty result above says nothing"
    )

    # THE PART-BLOCK PREDICATE, AT A MOVED GRANULARITY. At the production numbers
    # the consumer's block IS the checkpoint tile, so no width fails one rule
    # alone; a control taking a width that fails both would read a green belonging
    # to the tile rule. So the width stays three tiles and the BLOCK moves: fed a
    # doubled block, part_block must flag it while part_tile clears it.
    moved_block = 2 * DEFAULT_WEIGHT_BLOCK_SIZE[0]
    escaping_width = 3 * DEFAULT_WEIGHT_BLOCK_SIZE[0]
    escaping_tile = part_tile(0, escaping_width)
    escaping_block = part_block(escaping_width, moved_block)
    assert escaping_tile == 0, (
        f"the {escaping_width}-wide control does not clear the "
        f"{DEFAULT_WEIGHT_BLOCK_SIZE} tile, so the tile rule would answer for it "
        f"and this control would be reading the wrong rule"
    )
    assert escaping_block, (
        f"the {escaping_width}-wide control leaves no remainder in a "
        f"{moved_block}-row consumer block, so the part-block predicate could not "
        f"answer non-empty and its empty result above says nothing"
    )

    assert offenders == {}, (
        f"an MLA head width is not a whole number of checkpoint tiles: "
        f"{offenders}, against {DEFAULT_WEIGHT_BLOCK_SIZE}. These three shard on "
        f"whole heads, so a part tile means a rank's shard ends inside a block "
        f"whose single scale cannot be split between ranks and "
        f"shard_geometry_for_grid refuses the load; padding a head-bearing axis "
        f"would invent columns belonging to no head"
    )

    assert block_offenders == {}, (
        f"an MLA head width is not a whole number of the consumer's blocks: "
        f"{block_offenders}, against {consumer_block} from "
        f"dense_consumer_block_quant_size(). Such a width clears the "
        f"{DEFAULT_WEIGHT_BLOCK_SIZE} checkpoint tile and is still refused by "
        f"shard_geometry_for_grid's second check"
    )


def test_only_one_weight_block_size_is_supported() -> None:
    """The compensating grid loader relies on the default block size being the only supported one."""
    from vllm_neuron.model.glm5_next import quantization as _quant

    supported = _quant.SUPPORTED_WEIGHT_BLOCK_SIZES
    default = _quant.DEFAULT_WEIGHT_BLOCK_SIZE

    geom_default = inspect.signature(
        _WL_FP8.shard_geometry_for_grid
    ).parameters["block_size"].default

    # Which of the two loaders can be handed a block size, read off the live
    # signatures rather than restated, so a signature change moves this reading.
    def takes_block_size(fn: object) -> bool:
        return "block_size" in inspect.signature(fn).parameters  # type: ignore[arg-type]

    sibling_takes = takes_block_size(_WL_FP8.sharded_scale_grid_loader)
    mine_takes = takes_block_size(_WL_FP8.compensating_sharded_scale_grid_loader)

    assert len(supported) == 1 and supported == frozenset({default}), (
        f"the supported block sizes are {sorted(supported)} rather than exactly "
        f"[{default}]. compensating_sharded_scale_grid_loader takes no block_size "
        f"and always uses the default, which is safe only while the default is "
        f"the one reachable value; give it the parameter and forward it into "
        f"shard_geometry_for_grid, as sharded_scale_grid_loader does"
    )
    assert geom_default == default, (
        f"shard_geometry_for_grid defaults block_size to {geom_default}, not "
        f"{default}; the compensating loader omits the argument, so the default "
        f"is the divisor its path uses"
    )
    assert sibling_takes and not mine_takes, (
        f"the two loaders' signatures moved: sharded_scale_grid_loader takes "
        f"block_size={sibling_takes}, compensating_sharded_scale_grid_loader takes "
        f"block_size={mine_takes}. This test exists to explain that exact "
        f"asymmetry, so a change to either signature means the explanation needs "
        f"rewriting rather than re-asserting"
    )
