# SPDX-License-Identifier: Apache-2.0
"""Sharding read off the checkpoint's own width: the shared expert, the routed
bank and the dense MLP, at a world of four with expert-parallel degree two.

The bank divides by the ranks of one expert-parallel group and takes its
column from the group, refusing an extent that is not a whole consumer block;
the dense MLP pads to one. The last tests load the same checkpoint at world
size 1 so the shared expert's scale prep runs inside a completing load."""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import pytest
import torch

from vllm_neuron.model.glm5_next import factory as _FACTORY
from vllm_neuron.model.glm5_next import model_fp8 as _MODEL_FP8
from vllm_neuron.model.glm5_next import weight_loaders_fp8 as _WL_FP8
from vllm_neuron.model.glm5_next.config import Glm5NextConfig, Glm5NextTextConfig
from vllm_neuron.model.glm5_next.model_fp8 import (
    Glm5NextForConditionalGeneration,
    Glm5NextSharedExperts,
    _WEIGHT_LEAF_SUFFIX,
    _scale_prep_leaves,
)
from vllm_neuron.model.glm5_next.quantization import DEFAULT_WEIGHT_BLOCK_SIZE
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    FP8_SCALE_SUFFIX,
    Glm5NextExpertBankNotLoadableError,
    block_grid_shape,
    compensate_block_scales,
    dequantise_blockwise,
    downscale_fp8_weight_bytes,
    scale_keys,
)
from vllm_neuron.parallel import neuron_parallel_state as _NPS

from .test_load_weights import (  # noqa: F401 -- fixtures are used by name
    MINI_FIRST_K_DENSE,
    MINI_LAYERS,
    MINI_MLA_WIDTHS,
    MINI_ROUTED_EXPERTS,
    MINI_SHARED_EXPERTS,
    _checkpoint_plain_dtype,
    _keys_of,
    _mappings_for,
    _write_miniature_checkpoint,
    keep_the_loaded_tensors,
    single_rank_process_group,
)
from .test_tensor_parallel_shards import (
    DEFERRED_EP_GROUP_CLASSES,
    DEFERRED_FAMILIES,
    SHARD_DENSE_LEAVES,
    SHARD_FAMILIES,
    SHARD_GRID_ATTRIBUTES,
    SHARD_INTERMEDIATE,
    SHARD_LINEAR_ATTN,
    SHARD_ONE_DIMENSIONAL,
    SHARD_OTHER_EXTENT,
    SHARED_INTERMEDIATE,
    _COARSENED_AT_LOAD_CLASSES,
    _REPUBLISHED_CLASSES,
    _as_the_loader_left_it,
    _deferred_leaves,
    _in_the_loader_frame,
    _loaded,
    _max_abs_diff,
    _pow2_block_grid_pattern,
    _seed_page_cache_signal,
    _shard_pattern,
    _sharded_leaves,
)


SHARD_EP_WORLD = 4
SHARD_EP_DEGREE = 2
SHARD_TP_PER_EP = SHARD_EP_WORLD // SHARD_EP_DEGREE

#: The unsharded extent of every two-dimensional tensor: one consumer block.
DEFERRED_NARROW = 256

#: A dense width narrower than four consumer blocks, so some ranks are all pad.
PAD_DENSE_INTERMEDIATE = 256


def _deferred_families_at(
    dense_intermediate: int,
) -> dict[tuple[str, str], tuple[int, int]]:
    """``SHARD_FAMILIES`` merged with ``DEFERRED_FAMILIES``, the dense three at ``dense_intermediate``."""
    merged: dict[tuple[str, str], tuple[int, int]] = {
        **SHARD_FAMILIES,
        **DEFERRED_FAMILIES,
    }
    for leaf in SHARD_DENSE_LEAVES:
        key = ("Glm5NextDenseMLP", leaf)
        shard_dim, _full = merged[key]
        merged[key] = (shard_dim, dense_intermediate)
    return merged


def _deferred_full_shape(
    family: str, leaf: str, shard_dim: int, full: int
) -> tuple[int, ...]:
    """``_shard_full_shape`` with ``DEFERRED_NARROW`` as the unsharded extent."""
    if leaf in SHARD_ONE_DIMENSIONAL:
        return (full,)
    other = SHARD_OTHER_EXTENT.get((family, leaf), DEFERRED_NARROW)
    return (full, other) if shard_dim == 0 else (other, full)


def _padded_shard_extent(full: int, num_shards: int, block: int) -> int:
    """One rank's extent after the width is rounded up to a multiple of ``num_shards * block``."""
    step = num_shards * block
    return (math.ceil(full / step) * step) // num_shards


def _deferred_config(
    shared_experts: int = MINI_SHARED_EXPERTS,
    dense_intermediate: int = SHARD_INTERMEDIATE,
) -> Glm5NextConfig:
    """The shard fixture with a shared expert; ``dense_intermediate`` reaches only ``Glm5NextDenseMLP``."""
    return Glm5NextConfig(
        text_config=Glm5NextTextConfig(
            num_hidden_layers=MINI_LAYERS,
            n_routed_experts=MINI_ROUTED_EXPERTS,
            n_shared_experts=shared_experts,
            first_k_dense_replace=MINI_FIRST_K_DENSE,
            tie_word_embeddings=False,
            linear_attn_config=SHARD_LINEAR_ATTN,
            intermediate_size=dense_intermediate,
            **MINI_MLA_WIDTHS,
        )
    )


def _deferred_key_overrides(
    model: Glm5NextForConditionalGeneration,
    mappings: dict[str, str | list[str]],
    *,
    ramp_grids: bool = False,
    dense_intermediate: int = SHARD_INTERMEDIATE,
) -> dict[str, torch.Tensor]:
    """Full tensors for the deferred families and the shard table's; a bank grid is pow2 per block unless ``ramp_grids``."""
    overrides: dict[str, torch.Tensor] = {}
    every_family = _deferred_families_at(dense_intermediate)
    for path, module in model.named_modules():
        cls = type(module).__name__
        for (family, leaf), (shard_dim, full) in every_family.items():
            if cls != family:
                continue
            param = f"{path}.{leaf}"
            if param not in mappings:
                continue
            keys = _keys_of(mappings, param)
            scales = scale_keys(keys)
            weights = [key for key in keys if key not in scales]
            shape = _deferred_full_shape(family, leaf, shard_dim, full)
            if not scales:
                for key in weights:
                    overrides[key] = _shard_pattern(
                        shape, shard_dim, _checkpoint_plain_dtype(leaf)
                    )
                continue
            grid_shape = block_grid_shape(shape, DEFAULT_WEIGHT_BLOCK_SIZE)
            for key in weights:
                overrides[key] = _shard_pattern(
                    shape, shard_dim, torch.float8_e4m3fn
                )
            coarsened = not ramp_grids and family in _COARSENED_AT_LOAD_CLASSES
            for key in scales:
                if coarsened:
                    overrides[key] = _pow2_block_grid_pattern(grid_shape, shard_dim)
                else:
                    overrides[key] = _shard_pattern(
                        grid_shape, shard_dim, torch.float32
                    )
    return overrides


def _deferred_checkpoint(
    tmp_path: Path,
    *,
    ramp_grids: bool = False,
    name: str = "deferred",
    dense_intermediate: int = SHARD_INTERMEDIATE,
) -> tuple[Path, dict, dict]:
    """One checkpoint holding every full tensor the tests below read."""
    config = _deferred_config(dense_intermediate=dense_intermediate)
    mappings = _mappings_for(config)
    reference = Glm5NextForConditionalGeneration(config)
    overrides = _deferred_key_overrides(
        reference,
        mappings,
        ramp_grids=ramp_grids,
        dense_intermediate=dense_intermediate,
    )
    directory = tmp_path / name
    _write_miniature_checkpoint(
        directory, mappings, reference, extra_overrides=overrides
    )
    return directory, overrides, mappings


class _FixtureGroup:
    """The two fields the loader reads off a ``GroupCoordinator``."""

    def __init__(self, rank_in_group: int, world_size: int) -> None:
        self.rank_in_group = rank_in_group
        self.world_size = world_size


def _mesh_answers(world_size: int, ep_degree: int, rank: int) -> tuple[int, int]:
    """``(ep_rank, column)`` for one global rank, from the package's own mesh builder."""
    rows, _columns = _NPS._build_ep_group_ranks(world_size, ep_degree)
    for row_index, row in enumerate(rows):
        if rank in row:
            return row_index, row.index(rank)
    raise AssertionError(
        f"global rank {rank} is in none of the {len(rows)} EP-TP rows the package "
        f"built for world {world_size} at expert-parallel degree {ep_degree}"
    )


class _DeferredLoad(NamedTuple):
    """One load's outcome: the model, and how many scale operands the shared expert's prep built (``None`` without one)."""

    model: Glm5NextForConditionalGeneration
    prepared: int | None


def _load_at_ep(
    directory: Path,
    world_size: int,
    rank: int,
    ep_degree: int,
    monkeypatch,
    shared_experts: int = MINI_SHARED_EXPERTS,
    dense_intermediate: int = SHARD_INTERMEDIATE,
) -> _DeferredLoad:
    """Load at a synthetic world size, rank and expert-parallel degree, and check the shared expert's published grid."""
    ep_rank, column = _mesh_answers(world_size, ep_degree, rank)
    monkeypatch.setattr(_MODEL_FP8, "_resolve_world_size", lambda: world_size)
    monkeypatch.setattr(_MODEL_FP8, "_resolve_rank", lambda: rank)
    monkeypatch.setattr(_FACTORY, "_resolve_ep_degree", lambda given: ep_degree)
    monkeypatch.setattr(_NPS, "get_neuron_ep_rank", lambda: ep_rank)
    monkeypatch.setattr(
        _NPS,
        "get_neuron_ep_tp_group",
        lambda: _FixtureGroup(column, world_size // ep_degree),
    )
    model = Glm5NextForConditionalGeneration(
        _deferred_config(shared_experts, dense_intermediate=dense_intermediate)
    )
    assert model.world_size == world_size, (
        f"the model resolved world size {model.world_size}, not the patched "
        f"{world_size}"
    )
    _seed_page_cache_signal()
    if shared_experts == 0:
        model.load_weights(str(directory), torch.device("cpu"), None)
        return _DeferredLoad(model, None)

    model.load_weights(str(directory), torch.device("cpu"), None)

    block = _WL_FP8.dense_consumer_block_quant_size()
    rows = _padded_shard_extent(SHARED_INTERMEDIATE, world_size, block)
    cols = DEFERRED_NARROW
    tile_grid = (
        rows // DEFAULT_WEIGHT_BLOCK_SIZE[0],
        cols // DEFAULT_WEIGHT_BLOCK_SIZE[1],
    )
    public_grid = (rows // block, cols // block)

    shared = [
        (path, module)
        for path, module in model.named_modules()
        if type(module).__name__ == "Glm5NextSharedExperts"
    ]
    assert shared, (
        f"this load was asked for {shared_experts} shared experts and built no "
        f"Glm5NextSharedExperts module, so there is no prep here to read"
    )

    built: set[int] = set()
    for path, module in shared:
        prepared = getattr(module, Glm5NextSharedExperts.PREPARED_SCALE_OPERANDS_ATTR)
        built.add(len(prepared))
        health = getattr(module, Glm5NextSharedExperts.SHARED_RETILE_HEALTH_ATTR)
        record = health["gate_proj_weight"]
        assert record["published"] is True, (
            f"{path}.gate_proj_weight was not published: {record.get('reason')}. At "
            f"[{rows},{cols}] both extents are whole {block} blocks, so a skip "
            f"here means the step could not read the extents it was given"
        )
        assert record["retiled"] is False, (
            f"{path}.gate_proj_weight reports a retile; the dense load path "
            f"coarsens nothing"
        )
        assert tuple(record["checkpoint_grid"]) == tile_grid, (
            f"{path} retiled from grid {tuple(record['checkpoint_grid'])}, not the "
            f"checkpoint-tile grid {tile_grid} this world size produces"
        )
        assert tuple(record["public_grid"]) == public_grid, (
            f"{path} published grid {tuple(record['public_grid'])}, not the public "
            f"grid {public_grid} the prep demands at [K={rows}, N={cols}]"
        )
        compute_grid = tuple(reversed(public_grid))
        grid = getattr(module, f"gate_proj_{FP8_SCALE_SUFFIX}")
        assert tuple(grid.shape) == compute_grid, (
            f"{path}.gate_proj_{FP8_SCALE_SUFFIX} is {tuple(grid.shape)} on the "
            f"module after the load, not the {compute_grid} the republish leaves "
            f"(the published {public_grid} transposed); the retile has to replace "
            f"the attribute the prep reads, not a copy of it"
        )
        weight = _loaded(model, f"{path}.gate_proj_weight")
        assert tuple(weight.shape) == (cols, rows), (
            f"{path}.gate_proj_weight is {tuple(weight.shape)} after the load; the "
            f"loader delivers [{rows},{cols}] and the republish has to leave the "
            f"[K={cols}, N={rows}] frame the seam contracts on, or the prepared "
            f"scale operand above was built from the other frame"
        )

    assert built == {3}, (
        f"the shared experts built {sorted(built)} scale operands, not 3 each. "
        f"Three projections, one operand apiece, and the load completed -- so a "
        f"shortfall means a projection was skipped rather than refused"
    )

    for path, module in model.named_modules():
        if type(module).__name__ != "Glm5NextDenseMLP":
            continue
        health = getattr(module, module.DENSE_RETILE_HEALTH_ATTR, None)
        assert health is not None, (
            f"{path} carries no republish health record after a real load, so the "
            f"load-time prep loop did not reach Glm5NextDenseMLP"
        )
        for leaf, record in health.items():
            assert record.get("transposed") is True, (
                f"{path}.{leaf} was not republished into the compute frame: "
                f"{record}"
            )
            assert tuple(record["compute_frame"]) == tuple(
                reversed(tuple(record["loader_frame"]))
            ), (
                f"{path}.{leaf} went from {record['loader_frame']} to "
                f"{record['compute_frame']}, which is not that pair transposed"
            )
    return _DeferredLoad(model, 3)


def test_every_deferred_family_lands_at_its_declared_per_rank_shape(
    keep_the_loaded_tensors, tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """At world 4 with degree 2 the dense, shared and bank families each divide by their own divisor."""
    directory, _, _ = _deferred_checkpoint(tmp_path)
    block = _WL_FP8.dense_consumer_block_quant_size()

    loads = {
        rank: _load_at_ep(
            directory, SHARD_EP_WORLD, rank, SHARD_EP_DEGREE, monkeypatch
        )
        for rank in range(SHARD_EP_WORLD)
    }
    models = {rank: load.model for rank, load in loads.items()}
    prepped = [rank for rank, load in loads.items() if load.prepared == 3]
    assert sorted(prepped) == sorted(models), (
        f"only ranks {sorted(prepped)} built the shared expert's three scale "
        f"operands, of {sorted(models)}. A rank that built none either never "
        f"reached the prep or the load-path retile did not publish its grid"
    )
    whole_load = _load_at_ep(directory, 1, 0, 1, monkeypatch)
    whole = whole_load.model
    assert whole_load.prepared == 3, (
        f"the world-size-1 load built {whole_load.prepared} scale operands, not 3"
    )

    control = _load_at_ep(
        directory, SHARD_EP_WORLD, 0, SHARD_EP_DEGREE, monkeypatch, shared_experts=0
    )
    assert control.prepared is None, (
        f"the no-shared-expert control reported {control.prepared} prepared "
        f"operands, so the count above is not the shared expert's prep"
    )
    control_shards = _sharded_leaves(control.model) + [
        row
        for row in _deferred_leaves(control.model)
        if type(row[1]).__name__ in DEFERRED_EP_GROUP_CLASSES
    ]
    assert control_shards, "the control load attached no sharded family"

    expected_local_experts = MINI_ROUTED_EXPERTS // SHARD_EP_DEGREE
    checked = 0
    for rank, model in models.items():
        for path, module, leaf, shard_dim, full in _deferred_leaves(model):
            cls = type(module).__name__
            dotted = f"{path}.{leaf}"
            got = tuple(
                _as_the_loader_left_it(module, leaf, _loaded(model, dotted)).shape
            )
            if cls in DEFERRED_EP_GROUP_CLASSES:
                per_rank = full // SHARD_TP_PER_EP
                base = list(_deferred_full_shape(cls, leaf, shard_dim, full))
                base[shard_dim] = per_rank
                expected = (expected_local_experts, *base)
                divisor = (
                    f"tp_per_ep {SHARD_TP_PER_EP} with no padding; an "
                    f"inadmissible extent is refused, not rounded up"
                )
            else:
                per_rank = _padded_shard_extent(full, SHARD_EP_WORLD, block)
                expected = list(_deferred_full_shape(cls, leaf, shard_dim, full))
                expected[shard_dim] = per_rank
                expected = tuple(expected)
                divisor = (
                    f"world {SHARD_EP_WORLD} after rounding up to a multiple of "
                    f"{block} per rank"
                )
            assert got == expected, (
                f"{dotted} loaded {got} at rank {rank}; this file's rule says "
                f"{expected} -- full extent {full} on dim {shard_dim}, divided by "
                f"{divisor}"
            )
            checked += 1
    assert checked == SHARD_EP_WORLD * len(_deferred_leaves(models[0]))
    assert checked > 0, "no deferred family was present on the tree"

    whole_checked = 0
    for path, module, leaf, shard_dim, full in _deferred_leaves(whole):
        dotted = f"{path}.{leaf}"
        got = tuple(
            _as_the_loader_left_it(module, leaf, _loaded(whole, dotted)).shape
        )
        base = list(
            _deferred_full_shape(type(module).__name__, leaf, shard_dim, full)
        )
        expected = (
            (MINI_ROUTED_EXPERTS, *base)
            if type(module).__name__ in DEFERRED_EP_GROUP_CLASSES
            else tuple(base)
        )
        assert got == expected, (
            f"{dotted} loaded {got} at world size 1, not the whole {expected}"
        )
        whole_checked += 1
    assert whole_checked > 0, "the world-size-1 control read no deferred family"
    assert whole_checked == len(_deferred_leaves(whole))


def test_the_mla_three_take_their_other_extent_from_the_table() -> None:
    """The three MLA head-width families resolve their real other extent.

    WHY A DIRECT READ RATHER THAN A SHAPE ASSERTION. This fixture's writer and
    its expectations share one function, :func:`_deferred_full_shape`, so a wrong
    width agrees with itself and the per-rank shape assertions above cannot see
    it. Reading the function itself is the only place the disagreement shows.

    THE KDA CONTROL IS THE FIRING HALF. ``o_proj_weight`` is declared by both
    ``Glm5NextKDAAttention`` and ``Glm5NextMLAAttention``, and only the second
    declares an other extent, so a lookup keyed on the leaf alone would hand the
    KDA row the MLA row's ``hidden_size``. The resolved extent is asserted on its
    own and not only through the whole tuple, because ``_KDA_FULL`` and
    :data:`DEFERRED_NARROW` are both 256 in this fixture and a tuple comparison
    alone could not say which of the two it had read.
    """
    expected_other = {
        ("Glm5NextMLAAttention", "q_b_proj_weight"): MINI_MLA_WIDTHS["q_lora_rank"],
        ("Glm5NextMLAAttention", "kv_b_proj_weight"): MINI_MLA_WIDTHS["kv_lora_rank"],
        ("Glm5NextMLAAttention", "o_proj_weight"): MINI_MLA_WIDTHS["hidden_size"],
        ("Glm5NextKDAAttention", "o_proj_weight"): DEFERRED_NARROW,
    }
    readings: list[str] = []
    others: dict[tuple[str, str], int] = {}
    for (family, leaf), want_other in expected_other.items():
        shard_dim, full = SHARD_FAMILIES[(family, leaf)]
        shape = _deferred_full_shape(family, leaf, shard_dim, full)
        assert len(shape) == 2, (
            f"{family}.{leaf} is not a two-dimensional leaf here, so the other "
            f"extent has no place to be read: {shape}"
        )
        other = shape[1 - shard_dim]
        assert shape[shard_dim] == full, (
            f"{family}.{leaf} put {shape[shard_dim]} on its shard dim {shard_dim} "
            f"rather than the full extent {full}"
        )
        assert other == want_other, (
            f"{family}.{leaf} resolved its other extent to {other}; this file's "
            f"table says {want_other}. Were the lookup keyed on the leaf alone, "
            f"both o_proj_weight rows would take "
            f"{MINI_MLA_WIDTHS['hidden_size']} and this reading would go red"
        )
        others[(family, leaf)] = other
        readings.append(f"{family}.{leaf}={shape}")
    assert len(readings) == len(expected_other) > 0

    # AND NOT ALL THROUGH THE FALLBACK. Without this, a function that ignored the
    # table entirely could still satisfy every assertion above on a fixture whose
    # declared widths happened to equal DEFERRED_NARROW.
    fell_back = sorted(
        key
        for key, value in others.items()
        if key[0] == "Glm5NextMLAAttention" and value == DEFERRED_NARROW
    )
    assert not fell_back, (
        f"an MLA row still reads the {DEFERRED_NARROW} fallback instead of its "
        f"declared extent: {fell_back}"
    )


def test_the_group_reassembles_every_deferred_family_bit_identically(
    keep_the_loaded_tensors, tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Shards reassembled over the right axes equal the checkpoint tensor; a ragged degree and a part-block shard refuse."""
    directory, overrides, mappings = _deferred_checkpoint(tmp_path)
    loads = {
        rank: _load_at_ep(
            directory, SHARD_EP_WORLD, rank, SHARD_EP_DEGREE, monkeypatch
        )
        for rank in range(SHARD_EP_WORLD)
    }
    models = {rank: load.model for rank, load in loads.items()}
    prepped = [rank for rank, load in loads.items() if load.prepared == 3]
    assert sorted(prepped) == sorted(models), (
        f"only ranks {sorted(prepped)} built the shared expert's three scale "
        f"operands, of {sorted(models)}. A rank that built none either never "
        f"reached the prep or the load-path retile did not publish its grid"
    )

    reassembled = 0
    for path, module, leaf, shard_dim, full in _deferred_leaves(models[0]):
        cls = type(module).__name__
        dotted = f"{path}.{leaf}"
        keys = _keys_of(mappings, dotted)
        scales = scale_keys(keys)
        weight_keys = [key for key in keys if key not in scales]
        if cls in DEFERRED_EP_GROUP_CLASSES:
            rows, _cols = _NPS._build_ep_group_ranks(
                SHARD_EP_WORLD, SHARD_EP_DEGREE
            )
            per_expert = []
            for row in rows:
                columns = [_loaded(models[rank], dotted) for rank in row]
                per_expert.append(torch.cat(columns, dim=shard_dim + 1))
            joined = torch.cat(per_expert, dim=0)
            expected = torch.stack(
                [overrides[key] for key in weight_keys]
            )
            trim = [slice(None)] * joined.dim()
            trim[shard_dim + 1] = slice(0, full)
            got = joined[tuple(trim)]
        else:
            got = torch.cat(
                [
                    _as_the_loader_left_it(
                        module, leaf, _loaded(models[rank], dotted)
                    )
                    for rank in range(SHARD_EP_WORLD)
                ],
                dim=shard_dim,
            )
            expected = overrides[weight_keys[0]]
            trim = [slice(None)] * got.dim()
            trim[shard_dim] = slice(0, full)
            got = got[tuple(trim)]
        reference = downscale_fp8_weight_bytes(expected)
        assert got.shape == reference.shape, (
            f"{dotted} reassembled to {tuple(got.shape)}, not the checkpoint's "
            f"{tuple(reference.shape)}"
        )
        diff = _max_abs_diff(got, reference)
        assert diff == 0.0, (
            f"{dotted} reassembled with max abs diff {diff}, not 0.0 -- the shards "
            f"do not put the checkpoint tensor back together"
        )
        reassembled += 1
    assert reassembled > 0, "no deferred family was present on the tree"
    assert reassembled == len(_deferred_leaves(models[0]))

    from vllm_neuron.model.glm5_next.factory import (
        RaggedExpertPartitionError,
        require_uniform_expert_partition,
    )

    # One more rank than experts cannot partition them evenly.
    ragged_degree = MINI_ROUTED_EXPERTS + 1
    assert MINI_ROUTED_EXPERTS % ragged_degree != 0, (
        f"{MINI_ROUTED_EXPERTS} experts divide evenly by {ragged_degree}, so this "
        f"is not the ragged case"
    )
    with pytest.raises(RaggedExpertPartitionError):
        require_uniform_expert_partition(MINI_ROUTED_EXPERTS, ragged_degree)

    # Three tiles is whole tiles but not whole two-tile consumer blocks, so
    # only the consumer's rule can refuse it.
    tile = DEFAULT_WEIGHT_BLOCK_SIZE[0]
    moved_block = 2 * tile
    monkeypatch.setattr(
        _WL_FP8, "dense_consumer_block_quant_size", lambda: moved_block
    )
    not_a_whole_block = 3 * tile
    assert not_a_whole_block % tile == 0, (
        f"{not_a_whole_block} rows is not a whole number of {tile}-row checkpoint "
        f"tiles, so a refusal here could be the tile rule's rather than the "
        f"consumer's and this arm would not read what it names"
    )
    assert not_a_whole_block % moved_block != 0, (
        f"{not_a_whole_block} rows IS a whole number of the moved consumer block "
        f"{moved_block}, so this arm would read no refusal at all"
    )
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as unusable:
        _WL_FP8.shard_geometry_for_grid(
            _WL_FP8.ShardGeometry(
                shard_dim=0,
                shard_size=not_a_whole_block,
                num_shards=SHARD_TP_PER_EP,
            ),
            param_name="probe.experts.gate_proj_weight_scale_inv",
        )
    message = str(unusable.value)
    assert "probe.experts.gate_proj_weight_scale_inv" in message
    assert f"{not_a_whole_block} rows along dim 0" in message, (
        f"the refusal does not name the width this arm fed it: {message[:200]}"
    )
    assert f"{moved_block}-row" in message and "consumer" in message.lower(), (
        f"the refusal does not name the consumer's {moved_block}-row block, so it "
        f"is not the consumer's rule that refused: {message[:200]}"
    )


def test_the_pad_is_fp8_zero_with_a_unit_grid_and_dequantises_exactly(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Padded ranks hold fp8 zero under the compensated unit grid, and the pad changes no computed number."""
    directory, overrides, mappings = _deferred_checkpoint(
        tmp_path, name="deferred-pad", dense_intermediate=PAD_DENSE_INTERMEDIATE
    )
    block = _WL_FP8.dense_consumer_block_quant_size()
    loads = {
        rank: _load_at_ep(
            directory,
            SHARD_EP_WORLD,
            rank,
            SHARD_EP_DEGREE,
            monkeypatch,
            dense_intermediate=PAD_DENSE_INTERMEDIATE,
        )
        for rank in range(SHARD_EP_WORLD)
    }
    models = {rank: load.model for rank, load in loads.items()}
    prepped = [rank for rank, load in loads.items() if load.prepared == 3]
    assert sorted(prepped) == sorted(models), (
        f"only ranks {sorted(prepped)} built the shared expert's three scale "
        f"operands, of {sorted(models)}. A rank that built none either never "
        f"reached the prep or the load-path retile did not publish its grid"
    )

    dense_paths = sorted(
        {
            path
            for path, module in models[0].named_modules()
            if type(module).__name__ == "Glm5NextDenseMLP"
        }
    )
    assert dense_paths, "the fixture built no dense MLP"

    per_rank = _padded_shard_extent(PAD_DENSE_INTERMEDIATE, SHARD_EP_WORLD, block)
    real_ranks = PAD_DENSE_INTERMEDIATE // per_rank
    assert real_ranks < SHARD_EP_WORLD, (
        f"every one of the {SHARD_EP_WORLD} ranks holds a real row of the "
        f"{PAD_DENSE_INTERMEDIATE}-row dense width at the consumer's {block}-row "
        f"block, so no rank is wholly padding"
    )

    zero_ranks = 0
    for path in dense_paths:
        for leaf in SHARD_DENSE_LEAVES:
            _shard_dim, _full = _deferred_families_at(PAD_DENSE_INTERMEDIATE)[
                ("Glm5NextDenseMLP", leaf)
            ]
            attribute = SHARD_GRID_ATTRIBUTES[SHARD_DENSE_LEAVES.index(leaf)]
            for rank in range(real_ranks, SHARD_EP_WORLD):
                weight = _loaded(models[rank], f"{path}.{leaf}")
                grid = getattr(models[rank].get_submodule(path), attribute)
                as_float = weight.to(torch.float32)
                assert bool((as_float == 0.0).all()), (
                    f"{path}.{leaf} at rank {rank} sits wholly past the real "
                    f"{PAD_DENSE_INTERMEDIATE} rows, so every element must be fp8 "
                    f"zero; "
                    f"max abs is {as_float.abs().max().item()}"
                )
                pad = compensate_block_scales(torch.ones_like(grid))
                assert bool(torch.equal(grid.to(torch.float32), pad.scale_inv)), (
                    f"{path}.{leaf}'s grid at rank {rank} must be the stored pad "
                    f"value {pad.scale_inv.flatten()[0].item()} past the real rows; "
                    f"it holds values from {grid.min().item()} to "
                    f"{grid.max().item()}. Twice that value means the grid was "
                    f"compensated twice"
                )
                if pad.applied:
                    assert not bool((grid.to(torch.float32) == 1.0).all()), (
                        f"{path}.{leaf}'s grid at rank {rank} still reads all 1.0 "
                        f"with the 240 clamp engaged, so the load path did not "
                        f"compensate it"
                    )
                zero_ranks += 1
    assert zero_ranks > 0, "no padded rank was read"

    shared_per_rank = _padded_shard_extent(SHARED_INTERMEDIATE, SHARD_EP_WORLD, block)
    assert shared_per_rank * SHARD_EP_WORLD == SHARED_INTERMEDIATE, (
        f"the shared expert's {SHARED_INTERMEDIATE} padded to "
        f"{shared_per_rank * SHARD_EP_WORLD}; only the dense MLP pads here"
    )

    path = dense_paths[0]
    hidden = int(_in_the_loader_frame(models[0], f"{path}.gate_proj_weight").shape[1])
    torch.manual_seed(0)
    x = torch.randn(hidden, dtype=torch.float32)

    inexact: dict[tuple[int, str], int] = {}
    coarsened: dict[tuple[int, str], int] = {}
    for rank in range(SHARD_EP_WORLD):
        module = models[rank].get_submodule(path)
        health = getattr(module, module.DENSE_RETILE_HEALTH_ATTR, None)
        assert health is not None, (
            f"{path} at rank {rank} carries no republish health record after a real "
            f"load, so the load-time prep loop never reached Glm5NextDenseMLP"
        )
        for leaf, record in health.items():
            if record.get("retiled"):
                coarsened[(rank, leaf)] = int(record["inexact_rescales"])
            if not record.get("published"):
                continue
            inexact[(rank, leaf)] = int(record["inexact_rescales"])
    assert not coarsened, (
        f"{len(coarsened)} dense projections report a coarsening: "
        f"{sorted(coarsened)}. The dense load path publishes the checkpoint's own "
        f"grid, so a non-empty set means a 256 retile is back on this path"
    )
    assert inexact, (
        f"no dense projection was published at world size {SHARD_EP_WORLD}"
    )
    worst_inexact = max(inexact.values())
    assert worst_inexact == 0, (
        f"the republish rescaled {worst_inexact} tiles inexactly, so the layout "
        f"change altered weight values: "
        f"{sorted(key for key, count in inexact.items() if count)}"
    )

    def _dequantised(rank: int, leaf: str) -> torch.Tensor:
        """One rank's shard dequantised at the grid the module carries, in the loader's frame."""
        module = models[rank].get_submodule(path)
        attribute = SHARD_GRID_ATTRIBUTES[SHARD_DENSE_LEAVES.index(leaf)]
        weight = _as_the_loader_left_it(
            module, leaf, _loaded(models[rank], f"{path}.{leaf}")
        )
        grid = _as_the_loader_left_it(module, attribute, getattr(module, attribute))
        block = (
            weight.shape[0] // grid.shape[0],
            weight.shape[1] // grid.shape[1],
        )
        return dequantise_blockwise(weight, grid, block).to(torch.float32)

    gate_padded = torch.cat(
        [_dequantised(rank, "gate_proj_weight") for rank in range(SHARD_EP_WORLD)],
        dim=0,
    )
    up_padded = torch.cat(
        [_dequantised(rank, "up_proj_weight") for rank in range(SHARD_EP_WORLD)],
        dim=0,
    )
    down_padded = torch.cat(
        [_dequantised(rank, "down_proj_weight") for rank in range(SHARD_EP_WORLD)],
        dim=1,
    )
    h_padded = (gate_padded @ x) * (up_padded @ x)
    y_padded = down_padded @ h_padded

    def _reference(leaf: str, *, uncompensated: bool = False) -> torch.Tensor:
        """The checkpoint's own tensor through the squeeze and, unless ``uncompensated``, the grid compensation."""
        keys = _keys_of(mappings, f"{path}.{leaf}")
        scales = scale_keys(keys)
        weight_key = next(key for key in keys if key not in scales)
        grid = overrides[scales[0]]
        if not uncompensated:
            grid = compensate_block_scales(grid).scale_inv
        return dequantise_blockwise(
            downscale_fp8_weight_bytes(overrides[weight_key]),
            grid,
            DEFAULT_WEIGHT_BLOCK_SIZE,
        ).to(torch.float32)

    h_whole = (_reference("gate_proj_weight") @ x) * (_reference("up_proj_weight") @ x)
    y_whole = _reference("down_proj_weight") @ h_whole

    stacks = {
        "gate_proj_weight": gate_padded,
        "up_proj_weight": up_padded,
        "down_proj_weight": down_padded,
    }
    conventions: dict[str, tuple[float, float]] = {}
    for leaf, stack in stacks.items():
        whole = _reference(leaf)
        other = _reference(leaf, uncompensated=True)
        real = (
            stack[:, : whole.shape[1]]
            if leaf == "down_proj_weight"
            else stack[: whole.shape[0]]
        )
        raw_diff = _max_abs_diff(real, whole)
        uncompensated_diff = _max_abs_diff(real, other)
        conventions[leaf] = (raw_diff, uncompensated_diff)
    for leaf, (raw_diff, uncompensated_diff) in conventions.items():
        assert uncompensated_diff != 0.0, (
            f"{leaf} matches the uncompensated form as well as the compensated "
            f"one, so the compensation is not reaching this grid"
        )
        assert raw_diff == 0.0, (
            f"{leaf}'s real rows differ from the checkpoint's own tensor by "
            f"{raw_diff}; the load path changes this tensor's layout, never its "
            f"values. Check first whether a 256 coarsening is back on this path"
        )

    def _stacked(leaf: str, ranks: range, dim: int) -> torch.Tensor:
        return torch.cat([_dequantised(rank, leaf) for rank in ranks], dim=dim)

    real_only = range(real_ranks)
    h_real = (
        _stacked("gate_proj_weight", real_only, 0) @ x
    ) * (_stacked("up_proj_weight", real_only, 0) @ x)
    y_real = _stacked("down_proj_weight", real_only, 1) @ h_real
    pad_only_diff = _max_abs_diff(y_padded, y_real)
    assert pad_only_diff == 0.0, (
        f"the padded ranks change the output by {pad_only_diff} against the same "
        f"module tensors with those ranks left out, so the pad itself contributes"
    )

    assert h_padded.shape[0] > h_whole.shape[0], (
        f"the padded intermediate is {h_padded.shape[0]} wide and the unpadded "
        f"{h_whole.shape[0]}; if they matched, no pad was exercised"
    )
    tail = h_padded[h_whole.shape[0] :]
    assert tail.abs().max().item() == 0.0, (
        "the padded intermediate's tail is not exactly zero, so the padded rows are "
        "contributing to the down projection"
    )
    assert _max_abs_diff(y_padded, y_whole) == 0.0, (
        "the padded shards compute a different output from the unpadded reference; "
        "the pad is not exact"
    )


def test_the_six_families_differ_between_ranks_and_nothing_else_does(
    keep_the_loaded_tensors, tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Between two ranks the parameters that differ are exactly the declared-sharded set, the six included by name."""
    directory, _overrides, _mappings = _deferred_checkpoint(tmp_path)
    load0 = _load_at_ep(directory, SHARD_EP_WORLD, 0, SHARD_EP_DEGREE, monkeypatch)
    load1 = _load_at_ep(directory, SHARD_EP_WORLD, 1, SHARD_EP_DEGREE, monkeypatch)
    rank0, rank1 = load0.model, load1.model
    recorded = [load0.prepared, load1.prepared]
    assert load0.prepared == 3 and load1.prepared == 3, (
        f"the two loads built {recorded} scale operands, not three each, so they "
        f"are not the same kind of load and their difference is not a shard reading"
    )

    declared_sharded = {
        f"{path}.{leaf}"
        for path, _module, leaf, _dim, _full in _deferred_leaves(rank0)
    } | {
        f"{path}.{leaf}"
        for path, _module, leaf, _dim, _full in _sharded_leaves(rank0)
    }
    assert declared_sharded, "this file declares no sharded family"

    left = dict(rank0.named_parameters())
    right = dict(rank1.named_parameters())
    assert set(left) == set(right), (
        f"the two ranks registered different parameter names, so a family could "
        f"drop out of the comparison unseen. Only at rank 0: "
        f"{sorted(set(left) - set(right))[:3]}; only at rank 1: "
        f"{sorted(set(right) - set(left))[:3]}"
    )
    common = sorted(left)
    assert common, "the two ranks share no parameter name, so no comparison happened"

    moved = {
        name
        for name in common
        if tuple(left[name].shape) != tuple(right[name].shape)
        or _max_abs_diff(left[name].data, right[name].data) != 0.0
    }
    still = set(common) - moved
    assert moved, "no parameter differs between the two ranks, so nothing is sharded"
    assert still, "every parameter differs, so the replicated set is empty"

    declared_but_still = sorted(name for name in declared_sharded if name in still)
    assert not declared_but_still, (
        f"{len(declared_but_still)} declared-sharded families read identically on "
        f"both ranks, first {declared_but_still[:3]} -- they are still replicated"
    )

    moved_but_undeclared = sorted(moved - declared_sharded)
    assert not moved_but_undeclared, (
        f"{len(moved_but_undeclared)} families differ between ranks without being "
        f"declared sharded, first {moved_but_undeclared[:3]}"
    )

    found_keys = {
        (type(module).__name__, leaf)
        for _path, module, leaf, _dim, _full in _deferred_leaves(rank0)
    }
    missing_keys = sorted(set(DEFERRED_FAMILIES) - found_keys)
    six = sorted(
        f"{path}.{leaf}"
        for path, _module, leaf, _dim, _full in _deferred_leaves(rank0)
    )
    assert not missing_keys, (
        f"the tree holds no parameter for {missing_keys}, so a class the table "
        f"names was never built"
    )
    assert all(name in moved for name in six), (
        f"one of the six did not move between ranks: "
        f"{[name for name in six if name not in moved][:3]}"
    )


def test_the_shard_column_comes_from_the_group_and_refuses_a_disagreement(
    monkeypatch,
) -> None:
    """On the non-contiguous 64-rank mesh the loader reads the group's row and column and refuses a modulo disagreement."""
    world = 64
    disagreeing_degree = 8
    registered_degree = 16
    tp_per_ep = world // disagreeing_degree

    assert _NPS.uses_noncontiguous_mesh(world, tp_per_ep), (
        f"the package does not call world {world} with row {tp_per_ep} "
        f"non-contiguous, so there is no disagreement with the modulo to read"
    )

    row_rank, _ = _mesh_answers(world, disagreeing_degree, 12)
    assert row_rank != 12 // tp_per_ep, (
        f"the group puts rank 12 in row {row_rank} and the division also says "
        f"{12 // tp_per_ep}; they agree, so this reading distinguishes nothing"
    )
    monkeypatch.setattr(_NPS, "get_neuron_ep_rank", lambda: row_rank)
    owner = type(
        "Owner", (), {"ep_degree": disagreeing_degree, "tp_degree": world}
    )()
    to_partition_rank = _WL_FP8._expert_parallel_rank_map(
        owner, "probe.experts.gate_proj_weight"
    )
    got_row = to_partition_rank(12)
    assert got_row == row_rank, (
        f"the rank map returned {got_row} where the group says {row_rank}"
    )
    assert got_row != 12 // tp_per_ep, (
        f"the rank map returned the divided answer {12 // tp_per_ep}"
    )

    _, column_of_4 = _mesh_answers(world, disagreeing_degree, 4)
    assert column_of_4 != 4 % tp_per_ep, (
        f"the group puts rank 4 at column {column_of_4} and the modulo also says "
        f"{4 % tp_per_ep}; they agree, so there is no disagreement to refuse"
    )
    monkeypatch.setattr(
        _NPS,
        "get_neuron_ep_tp_group",
        lambda: _FixtureGroup(column_of_4, tp_per_ep),
    )
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
        _WL_FP8._expert_parallel_shard_column(
            4, tp_per_ep, disagreeing_degree, "probe.experts.gate_proj_weight"
        )
    message = str(refusal.value)
    assert message.startswith("probe.experts.gate_proj_weight "), (
        f"the refusal does not name the parameter first: {message}"
    )
    assert f"at column {column_of_4}" in message, (
        f"the refusal does not say which column the group reported: {message}"
    )
    assert f"= {4 % tp_per_ep} " in message or message.rstrip().endswith(
        f"= {4 % tp_per_ep}"
    ), f"the refusal does not say which column the consumer derives: {message}"

    registered_tp_per_ep = world // registered_degree
    agreements = 0
    for rank in range(world):
        _, column = _mesh_answers(world, registered_degree, rank)
        if column == rank % registered_tp_per_ep:
            agreements += 1
    assert agreements == world, (
        f"only {agreements} of {world} ranks agree with the modulo at degree "
        f"{registered_degree}, so the production degree would refuse at load"
    )
    _, registered_column = _mesh_answers(world, registered_degree, 12)
    monkeypatch.setattr(
        _NPS,
        "get_neuron_ep_tp_group",
        lambda: _FixtureGroup(registered_column, registered_tp_per_ep),
    )
    accepted = _WL_FP8._expert_parallel_shard_column(
        12, registered_tp_per_ep, registered_degree, "probe.experts.gate_proj_weight"
    )
    assert accepted == registered_column, (
        f"the column reader returned {accepted} where the group says "
        f"{registered_column} at the registered degree"
    )

    def _the_group_must_not_be_asked():
        raise AssertionError(
            "Neuron EP-TP group is not initialized. "
            "Call initialize_neuron_parallel_state() with ep_degree > 1."
        )

    monkeypatch.setattr(_NPS, "get_neuron_ep_tp_group", _the_group_must_not_be_asked)
    at_degree_one = [
        _WL_FP8._expert_parallel_shard_column(
            rank, world, 1, "probe.experts.gate_proj_weight"
        )
        for rank in (0, 1, 12, 63)
    ]
    assert at_degree_one == [0, 1, 12, 63], (
        f"at expert-parallel degree 1 the group is the whole world, so each rank's "
        f"column is its own rank; the reader returned {at_degree_one}"
    )


BANKPAD_PARAM = "layers.0.mlp.experts.gate_proj_weight"
BANKPAD_SIBLING_PARAM = "layers.0.mlp.gate_proj_weight"


class _BankpadSlice:
    """A checkpoint slice stub supporting ``get_shape`` and ``__getitem__``."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = tuple(shape)
        self._tensor = torch.zeros(self._shape, dtype=torch.float32)

    def get_shape(self) -> list[int]:
        return list(self._shape)

    def __getitem__(self, key):
        return self._tensor[key]


def _bankpad_bank_geometry(ep_degree: int, world_size: int):
    """The bank's geometry from the production table, for a synthetic owner."""
    owner = type("Glm5NextRoutedExperts", (), {"ep_degree": ep_degree})()
    return _MODEL_FP8._shard_geometry_for(owner, "gate_proj_weight", world_size)


def below_and_above(extent: int, required: int) -> tuple[int, int]:
    """The admissible neighbours of ``extent``, computed independently of the loader."""
    below = (extent // required) * required
    return below, below + required


def test_the_bank_refuses_a_part_block_extent_while_a_sibling_family_loads_it() -> None:
    """A bank shard narrower than a whole block is refused by name; a family declaring no requirement loads the same extent."""
    world = 4
    block = _WL_FP8.consumer_block_quant_size()
    geometry = _bankpad_bank_geometry(ep_degree=1, world_size=world)
    assert geometry.pad_to_multiple_of is None, (
        f"the bank declares a pad of {geometry.pad_to_multiple_of}; it must "
        f"declare a requirement instead"
    )
    assert geometry.require_multiple_of == block, (
        f"the bank declares require_multiple_of={geometry.require_multiple_of}, not "
        f"the consumer's own {block}; the number must be imported, never typed"
    )
    assert geometry.num_shards == world, (
        f"at expert-parallel degree 1 the bank must divide by the world ({world}), "
        f"and this geometry says {geometry.num_shards}"
    )

    full = 512
    experts = 2
    expected_extent = full // world
    assert expected_extent % block != 0, (
        f"{full} over {world} ranks is {expected_extent}, a whole {block} block, "
        f"so no refusal can fire"
    )
    stack_whole = lambda _slices, _rank: torch.zeros((experts, full, 8))
    transform = _WL_FP8._column_of_each_expert(geometry, BANKPAD_PARAM, stack_whole)
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
        transform([], 0)
    message = str(refusal.value)
    assert message.startswith(f"{BANKPAD_PARAM} "), (
        f"the refusal does not name the parameter first: {message}"
    )
    assert f"loads {expected_extent} rows per rank" in message, (
        f"the refusal does not say the per-rank extent it refused: {message}"
    )
    assert f"{block}-row block" in message, (
        f"the refusal does not name the consumer's block: {message}"
    )
    assert below_and_above(expected_extent, block)[0] == 0, (
        f"{expected_extent} is at least one whole {block} block, so the "
        f"nearest-neighbours arm fires instead of the sub-block arm"
    )
    assert f"narrower than one whole block" in message, (
        f"the refusal took the nearest-neighbours arm at a sub-block extent: {message}"
    )
    assert f"smallest admissible per-rank extent is {block}" in message, (
        f"the refusal does not name a usable smallest extent: {message}"
    )
    assert f"full width of {block * world}" in message, (
        f"the refusal does not name an admissible full width: {message}"
    )

    sibling = _WL_FP8.DeferredShardGeometry(shard_dim=0, num_shards=world)
    assert sibling.require_multiple_of is None, (
        "the default of require_multiple_of moved; a family that declares nothing "
        "must stay unbound"
    )
    loaded = _WL_FP8._sharding_loader(sibling, BANKPAD_SIBLING_PARAM).transform(
        [_BankpadSlice((full, 8))], 0
    )
    got = tuple(loaded.shape)
    assert got == (expected_extent, 8), (
        f"the sibling family loaded {got}, not {(expected_extent, 8)}; it must "
        f"load the same unaligned extent the bank was just refused for"
    )
    assert got[0] % block != 0, (
        f"the sibling's extent {got[0]} is a whole {block} block, so it was never a "
        f"candidate for the refusal"
    )


def test_a_wider_part_block_extent_names_both_admissible_neighbours() -> None:
    """A refused extent wider than one block names the admissible widths on either side."""
    world = 4
    ep_degree = 1
    block = _WL_FP8.consumer_block_quant_size()
    geometry = _bankpad_bank_geometry(ep_degree=ep_degree, world_size=world)
    tp_per_ep = world // ep_degree
    assert geometry.num_shards == tp_per_ep, (
        f"the bank's rank count is {geometry.num_shards} where tp_per_ep is "
        f"{tp_per_ep}"
    )

    full = 1536
    unpadded = full // tp_per_ep
    padded = _padded_shard_extent(full, tp_per_ep, block)
    below, above = below_and_above(unpadded, block)
    assert padded != unpadded, (
        f"padding {full} over {tp_per_ep} ranks gives {padded}, the same as the "
        f"unpadded {unpadded}, so a pad would be a no-op at this width"
    )
    assert unpadded % block != 0, (
        f"{unpadded} is a whole {block} block, so the unpadded shard is admissible "
        f"and there is nothing here to refuse"
    )
    assert below >= block, (
        f"flooring {unpadded} to a multiple of {block} gives {below}, under one whole "
        f"block, so the refusal would take the narrower-than-a-block arm instead"
    )

    stack_whole = lambda _slices, _rank: torch.zeros((2, full, 8))
    transform = _WL_FP8._column_of_each_expert(geometry, BANKPAD_PARAM, stack_whole)
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
        transform([], 0)
    message = str(refusal.value)
    assert f"admissible are {below} and {above}" in message, (
        f"the refusal does not name both admissible neighbours {below} and "
        f"{above}: {message}"
    )


def test_a_bank_grid_column_comes_from_the_group_not_the_modulo(
    monkeypatch,
) -> None:
    """The grid's converted geometry keeps the expert-parallel degree, so its column is read from the group."""
    world = 64
    disagreeing_degree = 8
    tp_per_ep = world // disagreeing_degree
    weight_geometry = _bankpad_bank_geometry(
        ep_degree=disagreeing_degree, world_size=world
    )
    converted = _WL_FP8.shard_geometry_for_grid(
        weight_geometry, BANKPAD_PARAM, DEFAULT_WEIGHT_BLOCK_SIZE
    )
    assert converted.expert_parallel_degree == disagreeing_degree, (
        f"the grid conversion carries degree {converted.expert_parallel_degree} "
        f"where the weight declared {disagreeing_degree}; the column reader will "
        f"short-circuit and read no group"
    )

    assert _NPS.uses_noncontiguous_mesh(world, tp_per_ep), (
        f"the package does not call world {world} with row {tp_per_ep} "
        f"non-contiguous, so there is no disagreement with the modulo to read"
    )
    _, column_of_4 = _mesh_answers(world, disagreeing_degree, 4)
    assert column_of_4 != 4 % tp_per_ep, (
        f"the group puts rank 4 at column {column_of_4} and the modulo also says "
        f"{4 % tp_per_ep}; they agree, so there is no disagreement to refuse"
    )
    monkeypatch.setattr(
        _NPS, "get_neuron_ep_tp_group", lambda: _FixtureGroup(column_of_4, tp_per_ep)
    )
    stack_whole = lambda _slices, _rank: torch.zeros((2, 2048 // 128, 8))
    transform = _WL_FP8._column_of_each_expert(converted, BANKPAD_PARAM, stack_whole)
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
        transform([], 4)
    message = str(refusal.value)
    assert f"at column {column_of_4}" in message, (
        f"the refusal does not report the group's column, so the grid path did not "
        f"reach the group: {message}"
    )

    registered_degree = 16
    registered_tp_per_ep = world // registered_degree
    registered_geometry = _WL_FP8.shard_geometry_for_grid(
        _bankpad_bank_geometry(ep_degree=registered_degree, world_size=world),
        BANKPAD_PARAM,
        DEFAULT_WEIGHT_BLOCK_SIZE,
    )
    _, registered_column = _mesh_answers(world, registered_degree, 12)
    monkeypatch.setattr(
        _NPS,
        "get_neuron_ep_tp_group",
        lambda: _FixtureGroup(registered_column, registered_tp_per_ep),
    )
    rows = 2048 // 128
    accepted = _WL_FP8._column_of_each_expert(
        registered_geometry,
        BANKPAD_PARAM,
        lambda _slices, _rank: torch.zeros((2, rows, 8)),
    )([], 12)
    assert tuple(accepted.shape) == (2, rows // registered_tp_per_ep, 8), (
        f"the registered degree loaded {tuple(accepted.shape)}; at "
        f"{registered_tp_per_ep} ranks the grid's {rows} rows must divide to "
        f"{rows // registered_tp_per_ep}"
    )


def test_the_grid_conversion_carries_the_degree_through_both_returns() -> None:
    """Both exits of ``shard_geometry_for_grid``'s deferred branch keep the declared degree."""
    degree = 8
    block = DEFAULT_WEIGHT_BLOCK_SIZE[0]
    readings = []
    for label, pad in (("no_pad", None), ("converted_pad", block * 2)):
        geometry = _WL_FP8.DeferredShardGeometry(
            shard_dim=0,
            num_shards=4,
            pad_to_multiple_of=pad,
            expert_parallel_degree=degree,
        )
        converted = _WL_FP8.shard_geometry_for_grid(
            geometry, BANKPAD_PARAM, DEFAULT_WEIGHT_BLOCK_SIZE
        )
        readings.append((label, converted.expert_parallel_degree))
    assert len(readings) == 2, (
        f"this item read {len(readings)} of the conversion's two deferred exits"
    )
    assert all(value == degree for _label, value in readings), (
        f"the conversion did not carry the declared degree {degree} through both "
        f"returns: {readings}"
    )


BLOCKED_WORLD = 1
BLOCKED_EP_DEGREE = 1


def _load_blocked(
    directory: Path,
    monkeypatch,
    shared_experts: int = MINI_SHARED_EXPERTS,
) -> Glm5NextForConditionalGeneration:
    """Load the whole-block checkpoint at world size 1."""
    monkeypatch.setattr(_MODEL_FP8, "_resolve_world_size", lambda: BLOCKED_WORLD)
    monkeypatch.setattr(_MODEL_FP8, "_resolve_rank", lambda: 0)
    monkeypatch.setattr(
        _FACTORY, "_resolve_ep_degree", lambda given: BLOCKED_EP_DEGREE
    )
    monkeypatch.setattr(_NPS, "get_neuron_ep_rank", lambda: 0)
    monkeypatch.setattr(
        _NPS,
        "get_neuron_ep_tp_group",
        lambda: _FixtureGroup(0, BLOCKED_WORLD // BLOCKED_EP_DEGREE),
    )
    model = Glm5NextForConditionalGeneration(_deferred_config(shared_experts))
    assert model.world_size == BLOCKED_WORLD, (
        f"the model resolved world size {model.world_size}, not the patched "
        f"{BLOCKED_WORLD}"
    )
    _seed_page_cache_signal()
    model.load_weights(str(directory), torch.device("cpu"), None)
    return model


def _modules_named(
    model: Glm5NextForConditionalGeneration, class_name: str
) -> list[tuple[str, torch.nn.Module]]:
    """Every ``(path, module)`` whose type has this name."""
    return [
        (path, module)
        for path, module in model.named_modules()
        if type(module).__name__ == class_name
    ]


def test_the_shared_expert_prep_completes_a_load_and_publishes_its_grids(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """A load with a shared expert completes, publishes each grid at the implied shape and prepares every bank."""
    directory, _overrides, _mappings = _deferred_checkpoint(tmp_path)
    block = _WL_FP8.dense_consumer_block_quant_size()

    model = _load_blocked(directory, monkeypatch)

    shared = _modules_named(model, "Glm5NextSharedExperts")
    assert shared, "this configuration built no Glm5NextSharedExperts module"

    for path, module in shared:
        prepared = getattr(module, Glm5NextSharedExperts.PREPARED_SCALE_OPERANDS_ATTR)
        assert len(prepared) == 3, (
            f"{path} carries {len(prepared)} prepared scale operands, not 3; the "
            f"prep builds one per projection and the load completed, so a "
            f"shortfall means a projection was skipped rather than refused"
        )

        health = getattr(module, Glm5NextSharedExperts.SHARED_RETILE_HEALTH_ATTR)
        leaves = _scale_prep_leaves(module)
        assert len(leaves) == 3, (
            f"{path} offers {leaves} to the prep loop, not three leaves"
        )
        for leaf in leaves:
            record = health[leaf]
            assert record["published"] is True, (
                f"{path}.{leaf} was not published: {record.get('reason')}. On this "
                f"fixture every MoE extent is a whole {block} block, so a skip "
                f"here means the step could not read the extents it was given"
            )
            assert record["retiled"] is False, (
                f"{path}.{leaf} reports a retile; the dense load path publishes "
                f"the checkpoint's own grid and coarsens nothing"
            )
            weight = getattr(module, leaf)
            rows, cols = int(weight.shape[0]), int(weight.shape[1])
            implied = (rows // block, cols // block)
            grid_name = f"{leaf[: -len(_WEIGHT_LEAF_SUFFIX)]}_{FP8_SCALE_SUFFIX}"
            grid = getattr(module, grid_name)
            assert tuple(grid.shape) == implied, (
                f"{path}.{grid_name} is {tuple(grid.shape)} after the load; a "
                f"[{rows},{cols}] weight implies the public grid {implied} at "
                f"the consumer's {block}-block granularity"
            )
            assert grid.dtype is torch.float32, (
                f"{path}.{grid_name} is {grid.dtype}; the scale grids stay fp32 "
                f"through the retile"
            )

            assert record["emitted_unsupplied"] == 0, (
                f"{path}.{leaf} emitted {record['emitted_unsupplied']} slots the "
                f"input did not supply, so the retiled layout does not decode "
                f"back to the grid it was given"
            )
            assert record["input_scales_dropped"] == 0, (
                f"{path}.{leaf} dropped {record['input_scales_dropped']} input "
                f"scales, so the coarser grid cannot reproduce them bit-exactly"
            )

    banks = _modules_named(model, "Glm5NextRoutedExperts")
    assert banks, (
        "this configuration built no routed bank; first_k_dense_replace must "
        "leave at least one MoE layer"
    )
    bank_class = type(banks[0][1])
    for path, module in banks:
        prepared = getattr(module, bank_class.PREPARED_KERNEL_OPERANDS_ATTR, None)
        assert prepared, (
            f"{path} carries no prepared kernel operands after a completing "
            f"load. The prep loop's gate is a type test and this class now "
            f"defines prepare_scale_operands, so the loop must have visited it"
        )
        assert set(prepared) == {"packed_weights", "packed_scales"}, (
            f"{path} carries {len(prepared)} prepared kernel operands, not the "
            f"two packed banks block_quant_expert_mm takes: {sorted(prepared)}"
        )

    control = _load_blocked(directory, monkeypatch, shared_experts=0)
    assert not _modules_named(control, "Glm5NextSharedExperts"), (
        "the control built a shared-expert module at n_shared_experts=0"
    )
    assert _modules_named(control, "Glm5NextRoutedExperts"), (
        "the control built no routed bank either, so it varies more than the one "
        "field it declares"
    )


def test_a_ramp_scale_grid_loads_and_publishes_only_finite_values(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """A grid that ramps per tile loads and publishes no non-finite value."""
    _pow2_directory, pow2_overrides, _pow2_mappings = _deferred_checkpoint(tmp_path)
    ramp_directory, ramp_overrides, _ramp_mappings = _deferred_checkpoint(
        tmp_path, ramp_grids=True, name="deferred-ramp"
    )

    # THE ONE FIELD THIS ITEM VARIES, measured rather than declared: the two
    # checkpoints differ on the grids of the retiled families and nowhere else.
    # The comparison goes through fp32 because the weight tensors are fp8, where a
    # dtype-native equality is not something this file assumes it has.
    def _same(left: torch.Tensor, right: torch.Tensor) -> bool:
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            return False
        return bool(torch.equal(left.to(torch.float32), right.to(torch.float32)))

    assert sorted(ramp_overrides) == sorted(pow2_overrides), (
        "the two fixtures do not even hold the same keys, so they differ in more "
        "than the grid family this item varies"
    )
    differing = sorted(
        key
        for key, tensor in ramp_overrides.items()
        if not _same(tensor, pow2_overrides[key])
    )
    assert differing, (
        "the ramp fixture and the pow2 fixture hold identical tensors, so this "
        "item varies nothing and the readings below would not be attributable to "
        "the grid"
    )
    assert all(key.endswith(FP8_SCALE_SUFFIX) for key in differing), (
        f"the two fixtures differ on tensors that are not scale grids: "
        f"{[key for key in differing if not key.endswith(FP8_SCALE_SUFFIX)][:6]}. "
        f"This item varies the grid family and must vary nothing else"
    )

    ramp_loaded = _load_blocked(ramp_directory, monkeypatch)
    assert _modules_named(ramp_loaded, "Glm5NextSharedExperts"), (
        "the ramp load built no shared-expert module"
    )
    published = list(ramp_loaded.named_parameters()) + list(ramp_loaded.named_buffers())
    nonfinite = [
        name
        for name, tensor in published
        if not bool(torch.isfinite(tensor.detach().to(torch.float32)).all())
    ]
    assert published, "the completed ramp load published no parameter or buffer"
    assert not nonfinite, (
        f"the ramp load published non-finite values in {nonfinite[:6]}; the "
        f"publish rescales no byte, so a ramping grid must arrive finite"
    )


def test_a_ramp_scale_grid_dequantises_exactly_through_the_publish(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """On a ramp grid each published projection dequantises to the checkpoint's own numbers at the checkpoint's block."""
    ramp_directory, overrides, mappings = _deferred_checkpoint(
        tmp_path, ramp_grids=True, name="deferred-ramp-dense"
    )
    loaded = _load_blocked(ramp_directory, monkeypatch)

    read: list[str] = []
    for path, module in loaded.named_modules():
        if type(module).__name__ not in _REPUBLISHED_CLASSES:
            continue
        for attribute_name in (
            "SHARED_RETILE_HEALTH_ATTR",
            "DENSE_RETILE_HEALTH_ATTR",
        ):
            attribute = getattr(type(module), attribute_name, None)
            if attribute is None:
                continue
            health = getattr(module, attribute, None)
            if not health:
                continue
            for leaf, record in health.items():
                dotted = f"{path}.{leaf}"
                assert record.get("published") is True, (
                    f"{dotted} was not published on a ramp grid: "
                    f"{record.get('reason')}. The publish reads extents and never "
                    f"values, so a legal grid it declines to publish means the step "
                    f"could not read the extents it was handed"
                )
                assert record.get("retiled") is False, (
                    f"{dotted} reports a coarsening on the dense path, so the 256 "
                    f"retile is back"
                )
                assert int(record.get("inexact_rescales", 0)) == 0, (
                    f"{dotted} rescaled {record['inexact_rescales']} tiles "
                    f"inexactly; the publish rescales nothing"
                )

                keys = _keys_of(mappings, dotted)
                scales = scale_keys(keys)
                weight_key = next(key for key in keys if key not in scales)
                grid_name = f"{leaf[: -len(_WEIGHT_LEAF_SUFFIX)]}_{FP8_SCALE_SUFFIX}"

                weight = _as_the_loader_left_it(module, leaf, _loaded(loaded, dotted))
                grid = _as_the_loader_left_it(
                    module, grid_name, getattr(module, grid_name)
                )
                block = (
                    weight.shape[0] // grid.shape[0],
                    weight.shape[1] // grid.shape[1],
                )
                got = dequantise_blockwise(weight, grid, block).to(torch.float32)

                reference = dequantise_blockwise(
                    downscale_fp8_weight_bytes(overrides[weight_key]),
                    compensate_block_scales(overrides[scales[0]]).scale_inv,
                    DEFAULT_WEIGHT_BLOCK_SIZE,
                ).to(torch.float32)
                uncompensated = dequantise_blockwise(
                    downscale_fp8_weight_bytes(overrides[weight_key]),
                    overrides[scales[0]],
                    DEFAULT_WEIGHT_BLOCK_SIZE,
                ).to(torch.float32)

                real = got[: reference.shape[0], : reference.shape[1]]
                paired_diff = _max_abs_diff(real, reference)
                uncompensated_diff = _max_abs_diff(real, uncompensated)

                assert block == tuple(DEFAULT_WEIGHT_BLOCK_SIZE), (
                    f"{dotted} carries a grid at {block} granularity, not the "
                    f"checkpoint's own {tuple(DEFAULT_WEIGHT_BLOCK_SIZE)}; a coarser "
                    f"block here means the coarsening is back"
                )
                assert real.shape == reference.shape, (
                    f"{dotted} loaded to {tuple(got.shape)}, whose leading extents do "
                    f"not cover the checkpoint's {tuple(reference.shape)}"
                )
                assert uncompensated_diff != 0.0, (
                    f"{dotted} matches the uncompensated reference as well as the "
                    f"compensated one, so the compensation is not reaching this grid"
                )
                assert paired_diff == 0.0, (
                    f"{dotted} dequantises to something {paired_diff} away from the "
                    f"checkpoint's own numbers on a ramp grid; the load path changes "
                    f"this tensor's layout, never its values"
                )
                read.append(dotted)

    assert read, "the ramp load published no dense or shared-expert health record"

    nonfinite = [
        name
        for name, tensor in list(loaded.named_parameters())
        + list(loaded.named_buffers())
        if not bool(torch.isfinite(tensor.detach().to(torch.float32)).all())
    ]
    assert not nonfinite, (
        f"the ramp load published non-finite values in {nonfinite[:6]}"
    )
