# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel sharding of the loaded weights at a synthetic world size.

``SHARD_FAMILIES`` states which family shards on which dim and how wide it is,
independently of the production table, so the two can disagree. Every test
loads a real checkpoint whose tensors identify their own position, at world
size 1 and at world size 2, and compares the ranks against the whole tensor."""

from __future__ import annotations

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
    Glm5NextExpertBankNotLoadableError,
    block_grid_shape,
    scale_keys,
)

from .test_expert_bank_load import (
    _blocked_bank_overrides,
    _scale_grid_attribute,
)
from .test_load_weights import (  # noqa: F401 -- fixtures are used by name
    MINI_ALL_DENSE_FIRST_K,
    MINI_CHECKPOINT_FILE,
    MINI_FIRST_K_DENSE,
    MINI_LAYERS,
    MINI_MLA_WIDTHS,
    MINI_ROUTED_EXPERTS,
    _checkpoint_plain_dtype,
    _keys_of,
    _mappings_for,
    _write_miniature_checkpoint,
    keep_the_loaded_tensors,
    single_rank_process_group,
)


SHARD_LINEAR_ATTN = {
    "num_heads": 8,
    "head_dim": 32,
    "short_conv_kernel_size": 4,
    "gate_lower_bound": -5.0,
}

#: The dense MLP width: two ranks each take whole 128-row blocks of it.
SHARD_INTERMEDIATE = 512

SHARD_WORLD = 2

#: The unsharded extent of every two-dimensional shard tensor.
SHARD_NARROW = 8

_KDA_FULL = SHARD_LINEAR_ATTN["num_heads"] * SHARD_LINEAR_ATTN["head_dim"]
_KDA_HEADS = SHARD_LINEAR_ATTN["num_heads"]

_LEAF_WEIGHT_SUFFIX = "_weight"

_MLA_HEADS = MINI_MLA_WIDTHS["num_attention_heads"]
_MLA_Q_B_FULL = _MLA_HEADS * (
    MINI_MLA_WIDTHS["qk_nope_head_dim"] + MINI_MLA_WIDTHS["qk_rope_head_dim"]
)
_MLA_KV_B_FULL = _MLA_HEADS * (
    MINI_MLA_WIDTHS["qk_nope_head_dim"] + MINI_MLA_WIDTHS["v_head_dim"]
)
_MLA_O_PROJ_FULL = _MLA_HEADS * MINI_MLA_WIDTHS["v_head_dim"]

SHARD_FAMILIES: dict[tuple[str, str], tuple[int, int]] = {
    ("Glm5NextKDAAttention", "q_proj_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "k_proj_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "v_proj_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "f_b_proj_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "g_b_proj_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "q_conv1d_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "k_conv1d_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "v_conv1d_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "o_proj_weight"): (1, _KDA_FULL),
    ("Glm5NextKDAAttention", "b_proj_weight"): (0, _KDA_HEADS),
    ("Glm5NextKDAAttention", "A_log"): (0, _KDA_HEADS),
    ("Glm5NextKDAAttention", "dt_bias"): (0, _KDA_FULL),
    ("Glm5NextDenseMLP", "gate_proj_weight"): (0, SHARD_INTERMEDIATE),
    ("Glm5NextDenseMLP", "up_proj_weight"): (0, SHARD_INTERMEDIATE),
    ("Glm5NextDenseMLP", "down_proj_weight"): (1, SHARD_INTERMEDIATE),
    ("Glm5NextMLAAttention", "q_b_proj_weight"): (0, _MLA_Q_B_FULL),
    ("Glm5NextMLAAttention", "kv_b_proj_weight"): (0, _MLA_KV_B_FULL),
    ("Glm5NextMLAAttention", "o_proj_weight"): (1, _MLA_O_PROJ_FULL),
}

SHARD_OTHER_EXTENT: dict[tuple[str, str], int] = {
    ("Glm5NextMLAAttention", "q_b_proj_weight"): MINI_MLA_WIDTHS["q_lora_rank"],
    ("Glm5NextMLAAttention", "kv_b_proj_weight"): MINI_MLA_WIDTHS["kv_lora_rank"],
    ("Glm5NextMLAAttention", "o_proj_weight"): MINI_MLA_WIDTHS["hidden_size"],
}

SHARD_ONE_DIMENSIONAL = ("A_log", "dt_bias")

SHARD_DENSE_LEAVES = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")

SHARD_GRID_ATTRIBUTES = (
    "gate_proj_weight_scale_inv",
    "up_proj_weight_scale_inv",
    "down_proj_weight_scale_inv",
)


def _shard_config(first_k_dense: int) -> Glm5NextConfig:
    """The shard fixture: the stacked configuration with the linear-attention and MLP widths shrunk."""
    return Glm5NextConfig(
        text_config=Glm5NextTextConfig(
            num_hidden_layers=MINI_LAYERS,
            n_routed_experts=MINI_ROUTED_EXPERTS,
            n_shared_experts=0,
            first_k_dense_replace=first_k_dense,
            tie_word_embeddings=False,
            linear_attn_config=SHARD_LINEAR_ATTN,
            intermediate_size=SHARD_INTERMEDIATE,
            **MINI_MLA_WIDTHS,
        )
    )


def _shard_full_shape(
    family: str, leaf: str, shard_dim: int, full: int
) -> tuple[int, ...]:
    """The full (unsharded) shape this family's checkpoint tensor is written at."""
    if leaf in SHARD_ONE_DIMENSIONAL:
        return (full,)
    other = SHARD_OTHER_EXTENT.get((family, leaf), SHARD_NARROW)
    return (full, other) if shard_dim == 0 else (other, full)


def _shard_pattern(
    shape: tuple[int, ...], dim: int, dtype: torch.dtype
) -> torch.Tensor:
    """A tensor whose value identifies its position along ``dim``.

    fp8 values are the bytes ``(i % 119) + 8`` reinterpreted, so no two rows
    collapse under rounding: bytes 1-7 are subnormal and 127 and 255 are NaN.
    """
    extent = shape[dim]
    index = torch.arange(extent, dtype=torch.int64)
    if dtype is torch.float8_e4m3fn:
        line = ((index % 119) + 8).to(torch.uint8).view(torch.float8_e4m3fn)
    else:
        line = (index + 1).to(dtype)
    view = [1] * len(shape)
    view[dim] = extent
    return line.reshape(view).expand(shape).contiguous()


_COARSENED_AT_LOAD_CLASSES = ("Glm5NextRoutedExperts",)

_POW2_GRID_EXPONENTS = tuple(range(-3, 5))


def _tiles_per_producer_block() -> int:
    """How many 128-row checkpoint tiles one routed-bank block covers."""
    from vllm_neuron.functional.moe.blockwise_fp8_retile import BLOCK_QUANT_SIZE

    tiles, remainder = divmod(BLOCK_QUANT_SIZE, DEFAULT_WEIGHT_BLOCK_SIZE[0])
    assert remainder == 0 and len(set(DEFAULT_WEIGHT_BLOCK_SIZE)) == 1, (
        f"the producer's {BLOCK_QUANT_SIZE} block is not a whole number of the "
        f"checkpoint's {DEFAULT_WEIGHT_BLOCK_SIZE} tiles, so no grid this fixture "
        f"writes is one the coarsening reproduces"
    )
    return tiles


def _pow2_block_grid_pattern(shape: tuple[int, ...], dim: int) -> torch.Tensor:
    """A checkpoint-tile grid whose scales agree inside every bank block and are exact powers of two."""
    per_block = _tiles_per_producer_block()
    extent = shape[dim]
    block = torch.arange(extent, dtype=torch.int64) // per_block
    exponents = torch.tensor(_POW2_GRID_EXPONENTS, dtype=torch.int64)
    line = torch.ldexp(
        torch.ones(extent, dtype=torch.float32), exponents[block % exponents.numel()]
    )
    view = [1] * len(shape)
    view[dim] = extent
    return line.reshape(view).expand(shape).contiguous()


def _shard_key_overrides(
    model: Glm5NextForConditionalGeneration,
    mappings: dict[str, str | list[str]],
) -> dict[str, torch.Tensor]:
    """Position-identifying full tensors for every family in ``SHARD_FAMILIES``, per checkpoint key."""
    overrides: dict[str, torch.Tensor] = {}
    for path, module in model.named_modules():
        cls = type(module).__name__
        for (family, leaf), (shard_dim, full) in SHARD_FAMILIES.items():
            if cls != family:
                continue
            param = f"{path}.{leaf}"
            if param not in mappings:
                continue
            keys = _keys_of(mappings, param)
            scales = scale_keys(keys)
            weight_key = next(key for key in keys if key not in scales)
            shape = _shard_full_shape(family, leaf, shard_dim, full)
            if not scales and leaf.endswith(_LEAF_WEIGHT_SUFFIX):
                grid_param = (
                    f"{param[: -len(_LEAF_WEIGHT_SUFFIX)]}_{FP8_SCALE_SUFFIX}"
                )
                if grid_param in mappings:
                    scales = _keys_of(mappings, grid_param)
            if scales:
                overrides[weight_key] = _shard_pattern(
                    shape, shard_dim, torch.float8_e4m3fn
                )
                overrides[scales[0]] = _shard_pattern(
                    block_grid_shape(shape, DEFAULT_WEIGHT_BLOCK_SIZE),
                    shard_dim,
                    torch.float32,
                )
            else:
                overrides[weight_key] = _shard_pattern(
                    shape, shard_dim, _checkpoint_plain_dtype(leaf)
                )
    return overrides


def _seed_page_cache_signal() -> None:
    """Add the one-file miniature's page-cache key to the store, which a second rank would otherwise wait for."""
    store = torch.distributed.distributed_c10d._get_default_store()
    store.add(MINI_CHECKPOINT_FILE, 1)


def _load_at_world(
    directory: Path,
    world_size: int,
    rank: int,
    monkeypatch,
    first_k_dense: int,
) -> Glm5NextForConditionalGeneration:
    """Build a model at a synthetic world size and rank, then load the checkpoint."""
    monkeypatch.setattr(_MODEL_FP8, "_resolve_world_size", lambda: world_size)
    monkeypatch.setattr(_MODEL_FP8, "_resolve_rank", lambda: rank)
    model = Glm5NextForConditionalGeneration(_shard_config(first_k_dense))
    assert model.world_size == world_size, (
        f"the model resolved world size {model.world_size}, not the patched "
        f"{world_size}"
    )
    _seed_page_cache_signal()
    model.load_weights(str(directory), torch.device("cpu"), None)
    return model


def _shard_checkpoint(
    tmp_path: Path, first_k_dense: int
) -> tuple[Path, dict[str, torch.Tensor], dict]:
    """Write one checkpoint holding the full tensors; return it, the tensors and the map."""
    config = _shard_config(first_k_dense)
    mappings = _mappings_for(config)
    reference = Glm5NextForConditionalGeneration(config)
    overrides = _shard_key_overrides(reference, mappings)
    overrides.update(_blocked_bank_overrides(reference, mappings))
    assert len(overrides) >= len(SHARD_FAMILIES), (
        f"only {len(overrides)} checkpoint keys were given shard tensors, fewer "
        f"than the {len(SHARD_FAMILIES)} families this file declares sharded, so "
        f"some family is missing from the map and would be measured against a "
        f"constant"
    )
    directory = tmp_path / f"shard-{first_k_dense}"
    _write_miniature_checkpoint(
        directory, mappings, reference, extra_overrides=overrides
    )
    return directory, overrides, mappings


def _sharded_leaves(
    model: Glm5NextForConditionalGeneration,
) -> list[tuple[str, torch.nn.Module, str, int, int]]:
    """Every ``(path, module, leaf, shard_dim, full)`` in ``SHARD_FAMILIES`` present on the tree."""
    found: list[tuple[str, torch.nn.Module, str, int, int]] = []
    for path, module in model.named_modules():
        cls = type(module).__name__
        declared = getattr(module, "declared_param_names", ())
        for (family, leaf), (shard_dim, full) in SHARD_FAMILIES.items():
            if cls == family and leaf in declared:
                found.append((path, module, leaf, shard_dim, full))
    return found


def _loaded(model: Glm5NextForConditionalGeneration, dotted: str) -> torch.Tensor:
    loaded = dict(model.named_parameters())
    assert dotted in loaded, f"{dotted} is not in named_parameters() after the load"
    return loaded[dotted].data


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    """Max abs difference in fp32, so fp8 pairs can be compared."""
    if left.numel() == 0:
        return 0.0
    return (left.to(torch.float32) - right.to(torch.float32)).abs().max().item()


_REPUBLISHED_CLASSES = ("Glm5NextDenseMLP", "Glm5NextSharedExperts")


def _as_the_loader_left_it(
    module: torch.nn.Module, name: str, tensor: torch.Tensor
) -> torch.Tensor:
    """Undo the compute-frame transpose the dense MLP and shared expert apply after loading."""
    if type(module).__name__ not in _REPUBLISHED_CLASSES:
        return tensor
    if not name.startswith(("gate_proj", "up_proj", "down_proj")):
        return tensor
    if tensor.dim() != 2:
        return tensor
    return tensor.t()


def _in_the_loader_frame(
    model: Glm5NextForConditionalGeneration, dotted: str
) -> torch.Tensor:
    path, leaf = dotted.rsplit(".", 1)
    return _as_the_loader_left_it(
        model.get_submodule(path), leaf, _loaded(model, dotted)
    )


def test_shard_every_sharded_family_lands_at_its_declared_per_rank_shape(
    keep_the_loaded_tensors, tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Every declared family loads at its full extent divided by the world size, on both ranks."""
    directory, _, _ = _shard_checkpoint(tmp_path, MINI_ALL_DENSE_FIRST_K)

    whole = _load_at_world(directory, 1, 0, monkeypatch, MINI_ALL_DENSE_FIRST_K)
    control = {
        f"{path}.{leaf}": tuple(
            _as_the_loader_left_it(
                module, leaf, _loaded(whole, f"{path}.{leaf}")
            ).shape
        )
        for path, module, leaf, _, _ in _sharded_leaves(whole)
    }

    per_rank = {
        rank: _load_at_world(
            directory, SHARD_WORLD, rank, monkeypatch, MINI_ALL_DENSE_FIRST_K
        )
        for rank in range(SHARD_WORLD)
    }
    leaves = _sharded_leaves(per_rank[0])
    assert leaves, "no sharded family is present in the tree"

    checked = 0
    families_seen: set[tuple[str, str]] = set()
    for rank, sharded in sorted(per_rank.items()):
        for path, module, leaf, shard_dim, full in _sharded_leaves(sharded):
            dotted = f"{path}.{leaf}"
            full_shape = _shard_full_shape(
                type(module).__name__, leaf, shard_dim, full
            )
            expected = list(full_shape)
            expected[shard_dim] = full // SHARD_WORLD
            got = tuple(
                _as_the_loader_left_it(module, leaf, _loaded(sharded, dotted)).shape
            )
            assert got == tuple(expected), (
                f"{dotted} loaded {got} at rank {rank} of world size {SHARD_WORLD}; "
                f"this file declares it sharded on dim {shard_dim} of a full "
                f"{full_shape}, so its per-rank shape is {tuple(expected)}"
            )
            assert control[dotted] == full_shape, (
                f"{dotted} loaded {control[dotted]} at world size 1 but the "
                f"checkpoint holds {full_shape}; the control is not reading the "
                f"whole tensor, so the comparison above means nothing"
            )
            families_seen.add((type(module).__name__, leaf))
            checked += 1

    moved = {
        rank: sum(
            1
            for dotted, shape in control.items()
            if shape != tuple(_in_the_loader_frame(sharded, dotted).shape)
        )
        for rank, sharded in sorted(per_rank.items())
    }
    assert checked == SHARD_WORLD * len(leaves)
    assert len(families_seen) == len(SHARD_FAMILIES), (
        f"the load exercised {len(families_seen)} of the {len(SHARD_FAMILIES)} "
        f"families this file declares sharded; missing "
        f"{sorted(set(SHARD_FAMILIES) - families_seen)}"
    )
    for rank, count in moved.items():
        assert count == len(control), (
            f"only {count} of {len(control)} shapes differ between world size 1 and "
            f"rank {rank} of world size {SHARD_WORLD}; a shape that does not move "
            f"was not sharded"
        )


def test_shard_the_two_ranks_reassemble_every_family_bit_identically(
    keep_the_loaded_tensors, tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Rank 0 and rank 1 concatenated along the shard dim equal the world-size-1 tensor exactly."""
    directory, _, _ = _shard_checkpoint(tmp_path, MINI_ALL_DENSE_FIRST_K)
    dense = MINI_ALL_DENSE_FIRST_K

    whole = _load_at_world(directory, 1, 0, monkeypatch, dense)
    rank0 = _load_at_world(directory, SHARD_WORLD, 0, monkeypatch, dense)
    rank1 = _load_at_world(directory, SHARD_WORLD, 1, monkeypatch, dense)

    leaves = _sharded_leaves(whole)
    assert leaves, "no sharded family is present in the tree"

    exact = 0
    rejoined = 0
    worst = 0.0
    ranks_differ = 0
    for path, module, leaf, shard_dim, full in leaves:
        dotted = f"{path}.{leaf}"
        reference = _as_the_loader_left_it(module, leaf, _loaded(whole, dotted))
        per_rank = full // SHARD_WORLD
        for rank, model in ((0, rank0), (1, rank1)):
            mine = _as_the_loader_left_it(module, leaf, _loaded(model, dotted))
            expected = reference.narrow(shard_dim, rank * per_rank, per_rank)
            assert tuple(mine.shape) == tuple(expected.shape), (
                f"{dotted} at rank {rank} is {tuple(mine.shape)} and the matching "
                f"slice of the whole tensor is {tuple(expected.shape)}"
            )
            difference = _max_abs_diff(mine, expected)
            worst = max(worst, difference)
            assert difference == 0.0, (
                f"{dotted} at rank {rank} differs from indices "
                f"[{rank * per_rank}:{(rank + 1) * per_rank}] of dim {shard_dim} of "
                f"the world-size-1 tensor by {difference}; a shard is a slice, so "
                f"any difference at all is an indexing defect"
            )
            exact += 1

        reassembled = torch.cat(
            [
                _in_the_loader_frame(rank0, dotted),
                _in_the_loader_frame(rank1, dotted),
            ],
            dim=shard_dim,
        )
        assert tuple(reassembled.shape) == tuple(reference.shape), (
            f"{dotted}: the two ranks' tensors concatenate to "
            f"{tuple(reassembled.shape)}, and the world-size-1 tensor is "
            f"{tuple(reference.shape)}"
        )
        joined = _max_abs_diff(reassembled, reference)
        worst = max(worst, joined)
        assert joined == 0.0, (
            f"{dotted}: rank 0 and rank 1 concatenated along dim {shard_dim} differ "
            f"from the unsharded tensor by {joined}"
        )
        rejoined += 1

        if (
            _max_abs_diff(
                _in_the_loader_frame(rank0, dotted).narrow(shard_dim, 0, 1),
                _in_the_loader_frame(rank1, dotted).narrow(shard_dim, 0, 1),
            )
            != 0.0
        ):
            ranks_differ += 1

    assert exact == 2 * len(leaves)
    assert rejoined == len(leaves)
    assert worst == 0.0
    assert ranks_differ == len(leaves), (
        f"only {ranks_differ} of {len(leaves)} families hold different data on the "
        f"two ranks; for the rest, rank 1 could be reading rank 0's rows and every "
        f"assertion above would still pass"
    )


def test_shard_the_scale_grid_follows_its_weight_and_refuses_misalignment(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """A sharded weight's grid is the matching block slice, and a shard that is not whole blocks is refused by name."""
    directory, overrides, mappings = _shard_checkpoint(
        tmp_path, MINI_ALL_DENSE_FIRST_K
    )
    layout = MINI_ALL_DENSE_FIRST_K

    whole = _load_at_world(directory, 1, 0, monkeypatch, layout)
    rank0 = _load_at_world(directory, SHARD_WORLD, 0, monkeypatch, layout)
    rank1 = _load_at_world(directory, SHARD_WORLD, 1, monkeypatch, layout)

    dense = [
        path
        for path, module in whole.named_modules()
        if type(module).__name__ == "Glm5NextDenseMLP"
    ]
    assert dense, "the shard fixture built no dense MLP"

    for leaf, literal in zip(SHARD_DENSE_LEAVES, SHARD_GRID_ATTRIBUTES, strict=True):
        from_model = whole._sibling_scale_grid_name(leaf)
        from_file = _scale_grid_attribute(leaf)
        assert from_model == literal, (
            f"the model derives {from_model!r} as {leaf}'s grid attribute; this "
            f"file names it {literal!r}. One of the two is wrong, and a test that "
            f"only asked the code could not tell which"
        )
        assert from_file == literal, (
            f"this file's own helper derives {from_file!r}, not {literal!r}"
        )

    followed = 0
    grids = 0
    for path in dense:
        for leaf, attribute in zip(
            SHARD_DENSE_LEAVES, SHARD_GRID_ATTRIBUTES, strict=True
        ):
            shard_dim, full = SHARD_FAMILIES[("Glm5NextDenseMLP", leaf)]
            grid_key = scale_keys(_keys_of(mappings, f"{path}.{leaf}"))[0]
            written = overrides[grid_key]
            blocks = (full // SHARD_WORLD) // DEFAULT_WEIGHT_BLOCK_SIZE[shard_dim]
            assert blocks >= 1, (
                f"{path}.{leaf}'s per-rank extent is narrower than one block, so "
                f"there is no aligned grid shard to read"
            )
            whole_module = whole.get_submodule(path)
            reference = _as_the_loader_left_it(
                whole_module, attribute, getattr(whole_module, attribute)
            )
            assert tuple(reference.shape) == tuple(written.shape), (
                f"{path}.{attribute} arrived {tuple(reference.shape)} at world size "
                f"1 but the checkpoint holds {tuple(written.shape)}"
            )
            grids += 1
            for rank, model in ((0, rank0), (1, rank1)):
                rank_module = model.get_submodule(path)
                mine = getattr(rank_module, attribute, None)
                assert mine is not None, (
                    f"{path}.{attribute} does not exist after the rank-{rank} load, "
                    f"so a sharded weight's grid never arrived at all"
                )
                mine = _as_the_loader_left_it(rank_module, attribute, mine)
                expected = reference.narrow(shard_dim, rank * blocks, blocks)
                assert tuple(mine.shape) == tuple(expected.shape), (
                    f"{path}.{attribute} is {tuple(mine.shape)} at rank {rank}; its "
                    f"weight is sharded on dim {shard_dim}, so its grid must be "
                    f"{tuple(expected.shape)} -- a full grid here would describe "
                    f"blocks this rank does not hold, and the dequant would scale "
                    f"the wrong ones"
                )
                difference = _max_abs_diff(mine, expected)
                assert difference == 0.0, (
                    f"{path}.{attribute} at rank {rank} differs from the matching "
                    f"blocks of the whole grid by {difference}"
                )
                assert mine.dtype is torch.float32, (
                    f"{path}.{attribute} arrived {mine.dtype}, not fp32"
                )
                followed += 1

    moved = sum(
        1
        for path in dense
        for attribute in SHARD_GRID_ATTRIBUTES
        if _max_abs_diff(
            getattr(rank0.get_submodule(path), attribute),
            getattr(rank1.get_submodule(path), attribute),
        )
        != 0.0
    )
    assert grids == len(dense) * len(SHARD_GRID_ATTRIBUTES)
    assert followed == 2 * grids
    assert moved == grids, (
        f"only {moved} of {grids} grids differ between the two ranks, so a grid "
        f"that ignored the rank entirely would pass the checks above"
    )

    aligned = _WL_FP8.shard_geometry_for_grid(
        _WL_FP8.ShardGeometry(
            shard_dim=0,
            shard_size=2 * DEFAULT_WEIGHT_BLOCK_SIZE[0],
            num_shards=SHARD_WORLD,
        ),
        param_name="probe.gate_proj_weight_scale_inv",
    )
    assert aligned.shard_size == 2, (
        f"an aligned shard of 256 rows gave "
        f"{aligned.shard_size} grid rows, not 2"
    )
    misaligned = DEFAULT_WEIGHT_BLOCK_SIZE[0] + 1
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
        _WL_FP8.shard_geometry_for_grid(
            _WL_FP8.ShardGeometry(
                shard_dim=0, shard_size=misaligned, num_shards=SHARD_WORLD
            ),
            param_name="probe.gate_proj_weight_scale_inv",
        )
    message = str(refusal.value)
    assert "probe.gate_proj_weight_scale_inv" in message, (
        f"the refusal does not name the parameter: {message}"
    )
    assert str(DEFAULT_WEIGHT_BLOCK_SIZE[0]) in message, (
        f"the refusal does not name the block boundary it enforced: {message}"
    )

    # Three tiles is whole tiles but not whole two-tile consumer blocks, so
    # only the consumer's rule can refuse it.
    tile = DEFAULT_WEIGHT_BLOCK_SIZE[0]
    moved_block = 2 * tile
    monkeypatch.setattr(
        _WL_FP8, "dense_consumer_block_quant_size", lambda: moved_block
    )
    sharp = 3 * tile
    assert sharp % tile == 0, (
        f"{sharp} is not a whole number of {tile}-row tiles, so the tile rule would "
        f"refuse it first and this arm would be reading the wrong gate"
    )
    assert sharp % moved_block != 0, (
        f"{sharp} IS a whole number of {moved_block}-row consumer blocks, so the "
        f"refusal below cannot fire and this control means nothing"
    )
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as consumer_refusal:
        _WL_FP8.shard_geometry_for_grid(
            _WL_FP8.ShardGeometry(
                shard_dim=0,
                shard_size=sharp,
                num_shards=SHARD_WORLD,
            ),
            param_name="probe.gate_proj_weight_scale_inv",
        )
    consumer_message = str(consumer_refusal.value)
    assert "probe.gate_proj_weight_scale_inv" in consumer_message, (
        f"the consumer refusal does not name the parameter: {consumer_message}"
    )
    assert "consumer" in consumer_message.lower(), (
        f"the refusal that fired is not the consumer's: {consumer_message}"
    )
    assert str(moved_block) in consumer_message, (
        f"the consumer refusal does not name the {moved_block}-row block it "
        f"enforced: {consumer_message}"
    )


def test_shard_the_unsharded_families_are_untouched_both_directions(
    keep_the_loaded_tensors, tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """The parameters that differ between world sizes are exactly the declared-sharded set."""
    directory, _, _ = _shard_checkpoint(tmp_path, MINI_FIRST_K_DENSE)
    with_bank = MINI_FIRST_K_DENSE

    whole = _load_at_world(directory, 1, 0, monkeypatch, with_bank)
    rank0 = _load_at_world(directory, SHARD_WORLD, 0, monkeypatch, with_bank)

    left = dict(whole.named_parameters())
    right = dict(rank0.named_parameters())
    assert set(left) == set(right), (
        f"the two loads registered different parameter names, so no family can be "
        f"compared. Only at world 1: {sorted(set(left) - set(right))[:3]}; only at "
        f"world {SHARD_WORLD}: {sorted(set(right) - set(left))[:3]}"
    )

    declared = {
        f"{path}.{leaf}" for path, _, leaf, _, _ in _sharded_leaves(whole)
    } | {
        f"{path}.{leaf}"
        for path, module, leaf, _, _ in _deferred_leaves(whole)
        if type(module).__name__ in DEFERRED_EP_GROUP_CLASSES
    }
    identical = {
        name
        for name in left
        if tuple(left[name].shape) == tuple(right[name].shape)
        and _max_abs_diff(left[name].data, right[name].data) == 0.0
    }
    moved = set(left) - identical

    assert moved == declared, (
        f"the set that moved between world sizes is not the set this file declares "
        f"sharded. Moved but not declared: {sorted(moved - declared)[:5]}. Declared "
        f"but did not move: {sorted(declared - moved)[:5]}"
    )
    assert identical == set(left) - declared, (
        "the set that stayed identical is not exactly the complement of the sharded "
        "set, so some parameter is neither replicated nor sharded"
    )

    by_owner: dict[str, set[str]] = {}
    for name in sorted(identical):
        owner = type(whole.get_submodule(name.rpartition(".")[0])).__name__
        by_owner.setdefault(owner, set()).add(name.rpartition(".")[2])
    by_owner_moved: dict[str, set[str]] = {}
    for name in sorted(moved):
        owner = type(whole.get_submodule(name.rpartition(".")[0])).__name__
        by_owner_moved.setdefault(owner, set()).add(name.rpartition(".")[2])
    mla = sorted(by_owner.get("Glm5NextMLAAttention", ()))
    routed = sorted(by_owner.get("Glm5NextRoutedExperts", ()))
    shared = sorted(by_owner.get("Glm5NextSharedExperts", ()))
    assert mla, (
        "no MLA parameter stayed identical; the latent projections and layernorms "
        "have no head axis to shard and must stay replicated"
    )
    bank_projections = {
        leaf
        for (family, leaf) in DEFERRED_FAMILIES
        if family in DEFERRED_EP_GROUP_CLASSES
    }
    routed_that_moved = sorted(by_owner_moved.get("Glm5NextRoutedExperts", ()))
    projections_still_replicated = sorted(set(routed) & bank_projections)
    routers_still_replicated = sorted(set(routed) - bank_projections)
    assert projections_still_replicated == [], (
        f"a routed-expert projection stayed identical across the two world sizes: "
        f"{projections_still_replicated}. At world 2 with expert-parallel degree 1 "
        f"every expert is local and the intermediate width divides across the "
        f"whole world, so all three bank families must differ"
    )
    assert sorted(set(routed_that_moved) & bank_projections) == sorted(
        bank_projections
    ), (
        f"the three bank projections did not all move: moved "
        f"{sorted(set(routed_that_moved) & bank_projections)} of "
        f"{sorted(bank_projections)}"
    )
    assert routers_still_replicated, (
        "no router stayed identical; both routers are replicated, so the sharded "
        "set has widened past the bank's three families"
    )
    assert shared == [], (
        "this fixture built a shared-expert module; _shard_config declares none, "
        "so the shared expert's three families would need counting here"
    )

    bank_rank1 = _load_at_world(directory, SHARD_WORLD, 1, monkeypatch, with_bank)
    banks = [
        (path, module)
        for path, module in bank_rank1.named_modules()
        if type(module).__name__ == "Glm5NextRoutedExperts"
    ]
    assert banks, "this configuration built no Glm5NextRoutedExperts module"
    loaded_at_rank1 = dict(bank_rank1.named_parameters())
    leading: list[tuple[str, int, int]] = []
    for path, module in banks:
        for leaf in getattr(module, "declared_param_names", ()):
            dotted = f"{path}.{leaf}"
            if dotted not in loaded_at_rank1 or loaded_at_rank1[dotted].dim() < 2:
                continue
            leading.append(
                (
                    dotted,
                    int(loaded_at_rank1[dotted].shape[0]),
                    int(module.num_local_experts),
                )
            )
    assert len(leading) >= 1, (
        "the rank-1 load completed but no bank parameter with a leading expert "
        "axis was found"
    )
    assert banks[0][1].ep_degree == 1, (
        f"the expert-parallel degree read {banks[0][1].ep_degree}, not 1"
    )
    assert banks[0][1].num_local_experts == MINI_ROUTED_EXPERTS, (
        f"at expert-parallel degree 1 every expert is local, so rank 1 should own "
        f"all {MINI_ROUTED_EXPERTS}; the module declares "
        f"{banks[0][1].num_local_experts}"
    )
    for dotted, axis, expected in leading:
        assert axis == expected, (
            f"{dotted} loaded a leading axis of {axis} at rank 1 while its module "
            f"declares {expected} local experts, so the rank got a different "
            f"number of experts than the partition assigns it"
        )

    with pytest.raises(ValueError) as still_bounded:
        banks[0][1].expert_partition.local_expert_indices(1)
    bounded_text = str(still_bounded.value)
    assert "outside the partition" in bounded_text, (
        f"the partition no longer refuses an out-of-range rank: {bounded_text}; "
        f"without the bound a bank at a real expert-parallel degree would be "
        f"placed by guess"
    )
    assert "rank 1" in bounded_text and "1 ranks" in bounded_text, (
        f"the refusal does not name both the rank it was given and the size of "
        f"the partition it was checked against: {bounded_text}"
    )


#: Widths that divide by a world of four in whole consumer blocks (shared
#: expert) and by two ranks in whole checkpoint tiles (routed bank).
SHARED_INTERMEDIATE = 2048
BANK_INTERMEDIATE = 512

#: The families sharded off the checkpoint's own width rather than the table.
DEFERRED_FAMILIES: dict[tuple[str, str], tuple[int, int]] = {
    ("Glm5NextSharedExperts", "gate_proj_weight"): (0, SHARED_INTERMEDIATE),
    ("Glm5NextSharedExperts", "up_proj_weight"): (0, SHARED_INTERMEDIATE),
    ("Glm5NextSharedExperts", "down_proj_weight"): (1, SHARED_INTERMEDIATE),
    ("Glm5NextRoutedExperts", "gate_proj_weight"): (0, BANK_INTERMEDIATE),
    ("Glm5NextRoutedExperts", "up_proj_weight"): (0, BANK_INTERMEDIATE),
    ("Glm5NextRoutedExperts", "down_proj_weight"): (1, BANK_INTERMEDIATE),
}

DEFERRED_WHOLE_WORLD_CLASSES = ("Glm5NextDenseMLP", "Glm5NextSharedExperts")
DEFERRED_EP_GROUP_CLASSES = ("Glm5NextRoutedExperts",)


def _deferred_leaves(
    model: Glm5NextForConditionalGeneration,
) -> list[tuple[str, torch.nn.Module, str, int, int]]:
    """Every ``(path, module, leaf, shard_dim, full)`` in ``DEFERRED_FAMILIES`` present on the tree."""
    found: list[tuple[str, torch.nn.Module, str, int, int]] = []
    for path, module in model.named_modules():
        cls = type(module).__name__
        declared = getattr(module, "declared_param_names", ())
        for (family, leaf), (shard_dim, full) in DEFERRED_FAMILIES.items():
            if cls == family and leaf in declared:
                found.append((path, module, leaf, shard_dim, full))
    return found
