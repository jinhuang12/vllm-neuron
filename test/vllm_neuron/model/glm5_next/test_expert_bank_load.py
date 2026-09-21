# SPDX-License-Identifier: Apache-2.0
"""Loading a routed expert bank: stacking, per-rank selection and its scale grids.

A bank arrives as one map entry holding every expert's weight and scale key.
The load must stack one rank's experts, keep each expert's own bytes, and hand
the bank's three scale grids over as plain attributes with one row per expert."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_neuron.model.glm5_next.config import Glm5NextConfig, Glm5NextTextConfig
from vllm_neuron.model.glm5_next.model_fp8 import (
    Glm5NextForConditionalGeneration,
    Glm5NextSharedExperts,
    _WEIGHT_LEAF_SUFFIX,
    _is_fp8_dtype,
    _scale_prep_leaves,
)
from vllm_neuron.model.glm5_next.quantization import DEFAULT_WEIGHT_BLOCK_SIZE
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    Glm5NextExpertBankNotLoadableError,
    MAPPED_KEY_STACKED_BANK,
    bank_layout,
    block_grid_shape,
    classify_mapped_keys,
    compensate_block_scales,
    downscale_fp8_weight_bytes,
    loader_for_mapped_keys,
    scale_keys,
    stacked_expert_bank_loader,
    stacked_expert_scale_loader,
)
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

from .test_load_weights import (  # noqa: F401 -- fixtures are used by name
    MINI_CHECKPOINT_FILE,
    MINI_FIRST_K_DENSE,
    MINI_LAYERS,
    MINI_MLA_WIDTHS,
    MINI_ROUTED_EXPERTS,
    _dense_config,
    _dense_model,
    _implied_numels,
    _keys_of,
    _mappings_for,
    _not_lazy_count,
    _write_miniature_checkpoint,
    keep_the_loaded_tensors,
    single_rank_process_group,
)


STACKED_EP_DEGREE = 2
STACKED_EXPERTS_PER_RANK = MINI_ROUTED_EXPERTS // STACKED_EP_DEGREE

#: Bank extents at whole consumer blocks, so the bank's grids have blocks to divide.
BLOCKED_BANK_OUT = 512
BLOCKED_BANK_IN = 256

BLOCKED_BANK_FAMILIES: dict[tuple[str, str], tuple[int, int]] = {
    ("Glm5NextRoutedExperts", "gate_proj_weight"): (0, BLOCKED_BANK_OUT),
    ("Glm5NextRoutedExperts", "up_proj_weight"): (0, BLOCKED_BANK_OUT),
    ("Glm5NextRoutedExperts", "down_proj_weight"): (1, BLOCKED_BANK_OUT),
}


def _blocked_bank_overrides(
    model: Glm5NextForConditionalGeneration,
    mappings: dict[str, str | list[str]],
) -> dict[str, torch.Tensor]:
    """The bank's checkpoint tensors at whole-block extents, values unchanged."""
    overrides: dict[str, torch.Tensor] = {}
    for path, module in model.named_modules():
        cls = type(module).__name__
        for (family, leaf), (shard_dim, full) in BLOCKED_BANK_FAMILIES.items():
            if cls != family:
                continue
            param = f"{path}.{leaf}"
            if param not in mappings:
                continue
            keys = _keys_of(mappings, param)
            scales = scale_keys(keys)
            weights = [key for key in keys if key not in scales]
            shape = (
                (full, BLOCKED_BANK_IN)
                if shard_dim == 0
                else (BLOCKED_BANK_IN, full)
            )
            grid_shape = block_grid_shape(shape, DEFAULT_WEIGHT_BLOCK_SIZE)
            for key in weights:
                overrides[key] = torch.ones(
                    shape, dtype=torch.bfloat16
                ).to(torch.float8_e4m3fn)
            for key in scales:
                overrides[key] = torch.full(grid_shape, 0.5, dtype=torch.float32)
    return overrides


def _stacked_checkpoint(
    directory: Path,
    mappings: dict[str, str | list[str]],
    model: Glm5NextForConditionalGeneration,
) -> int:
    """``_write_miniature_checkpoint`` with the bank at whole-block extents."""
    return _write_miniature_checkpoint(
        directory,
        mappings,
        model,
        extra_overrides=_blocked_bank_overrides(model, mappings),
    )


def _stacked_config() -> Glm5NextConfig:
    """A routed configuration with no shared expert, so the load completes."""
    return Glm5NextConfig(
        text_config=Glm5NextTextConfig(
            num_hidden_layers=MINI_LAYERS,
            n_routed_experts=MINI_ROUTED_EXPERTS,
            n_shared_experts=0,
            first_k_dense_replace=MINI_FIRST_K_DENSE,
            tie_word_embeddings=False,
            **MINI_MLA_WIDTHS,
        )
    )


def _stacked_model() -> Glm5NextForConditionalGeneration:
    return Glm5NextForConditionalGeneration(_stacked_config())


def _bank_entries(mappings: dict[str, str | list[str]]) -> dict[str, list[str]]:
    """Every map entry the classifier calls an expert bank."""
    return {
        name: _keys_of(mappings, name)
        for name in mappings
        if classify_mapped_keys(mappings[name]) == MAPPED_KEY_STACKED_BANK
    }


def _checkpoint_tensor(directory: Path, key: str) -> torch.Tensor:
    """One checkpoint tensor, read through ``get_slice`` as the loaders read it."""
    with safe_open(str(directory / MINI_CHECKPOINT_FILE), framework="pt") as opened:
        return opened.get_slice(key)[:]


def _slice_pairs(directory: Path, keys: list[str]) -> list[torch.Tensor]:
    """A bank entry's keys as tensors, in checkpoint order."""
    return [_checkpoint_tensor(directory, key) for key in keys]


def _distinguish_bank_experts(directory: Path, keys: list[str]) -> dict[str, float]:
    """Rewrite the file so expert ``e`` holds ``e + 1`` in its weight and half that in its grid; return the values."""
    path = directory / MINI_CHECKPOINT_FILE
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as opened:
        for key in opened.keys():
            tensors[key] = opened.get_tensor(key)

    written: dict[str, float] = {}
    layout = bank_layout(keys, param_name="the bank entry under test")
    for expert in range(layout.experts):
        weight_key = keys[layout.weight_at[expert]]
        scale_key = keys[layout.scale_at[expert]]
        value = float(expert + 1)
        tensors[weight_key] = torch.full(
            tuple(tensors[weight_key].shape), value, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        tensors[scale_key] = torch.full(
            tuple(tensors[scale_key].shape), value / 2.0, dtype=torch.float32
        )
        written[weight_key] = value
        written[scale_key] = value / 2.0

    save_file(tensors, str(path))
    return written


def _stacked_bank_geometry(experts: int, per_rank: int):
    """A stand-in owner declaring a fixed expert partition."""

    class Owner:
        num_routed_experts = experts

        def local_expert_indices(self, rank: int) -> tuple[int, ...]:
            return tuple(range(rank * per_rank, (rank + 1) * per_rank))

    return Owner()


def test_the_stacked_bank_delivers_every_expert_or_refuses_by_name(
    keep_the_loaded_tensors, tmp_path, single_rank_process_group
) -> None:
    """A loaded bank holds every expert its slices imply; an owner with no geometry is refused by name."""
    directory = tmp_path / "stacked"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    _stacked_checkpoint(directory, mappings, model)
    banks = _bank_entries(mappings)

    assert banks, (
        "this configuration produced no expert-bank entry, so the bank loader is "
        "never reached"
    )

    before = _not_lazy_count(model)
    model.load_weights(str(directory), torch.device("cpu"), None)
    implied = _implied_numels(directory, mappings)
    loaded = dict(model.named_parameters())

    mismatches = {}
    axes = {}
    for name in sorted(banks):
        parameter = loaded[name]
        axes[name] = int(parameter.shape[0])
        if parameter.numel() != implied[name]:
            mismatches[name] = (parameter.numel(), implied[name])

    assert before == 0, (
        f"{before} parameters already held a real tensor before the load"
    )
    assert mismatches == {}, (
        f"{len(mismatches)} bank parameters received a different number of "
        f"elements than their own checkpoint slices imply, e.g. "
        f"{[(k, *mismatches[k]) for k in sorted(mismatches)[:3]]} as "
        f"(parameter, loaded, implied) -- an expert was dropped or invented"
    )
    assert set(axes.values()) == {MINI_ROUTED_EXPERTS}, (
        f"a bank's leading axis is not the expert count: read "
        f"{sorted(set(axes.values()))}, expected [{MINI_ROUTED_EXPERTS}]. At "
        f"expert-parallel degree 1 one rank owns every expert, so every bank "
        f"stacks E of them"
    )

    name = sorted(banks)[0]
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
        stacked_expert_bank_loader(banks[name], param_name=name, owner=object())
    message = str(refusal.value)
    assert name in message, (
        f"the refusal does not name the parameter it refused: {message}"
    )
    assert str(len(banks[name])) in message, (
        f"the refusal does not name the entry's key count: {message}"
    )
    assert "declares no expert geometry" in message.lower(), (
        f"the refusal does not name the missing declaration: {message}"
    )


def test_the_stacked_bank_holds_each_expert_bit_identically(
    tmp_path, single_rank_process_group
) -> None:
    """Each stacked expert row is bit-identical to its own checkpoint tensor through the same squeeze."""
    directory = tmp_path / "stacked"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    _write_miniature_checkpoint(directory, mappings, model)
    banks = _bank_entries(mappings)
    name = sorted(banks)[0]
    keys = banks[name]
    layout = bank_layout(keys, param_name=name)
    _distinguish_bank_experts(directory, keys)

    weight_keys = [keys[i] for i in layout.weight_at]
    scale_key_list = [keys[i] for i in layout.scale_at]

    owner = model.get_submodule(name.rsplit(".", 1)[0])
    slices = _slice_pairs(directory, keys)
    stacked = stacked_expert_bank_loader(
        keys, param_name=name, owner=owner
    ).transform(slices, 0)
    grids = stacked_expert_scale_loader(
        keys, param_name=name, owner=owner
    ).transform(slices, 0)

    weight_refs = [
        downscale_fp8_weight_bytes(_checkpoint_tensor(directory, key))
        for key in weight_keys
    ]
    scale_refs = [
        compensate_block_scales(_checkpoint_tensor(directory, key)).scale_inv
        for key in scale_key_list
    ]

    def worst(stack, refs) -> float:
        return max(
            float((stack[e].to(torch.float32) - refs[e].to(torch.float32)).abs().max())
            for e in range(len(refs))
        )

    aligned_weight = worst(stacked, weight_refs)
    aligned_scale = worst(grids, scale_refs)
    rotated_weight = worst(stacked, weight_refs[1:] + weight_refs[:1])
    rotated_scale = worst(grids, scale_refs[1:] + scale_refs[:1])

    assert stacked.shape[0] == layout.experts, (
        f"the stack holds {stacked.shape[0]} experts, not {layout.experts}"
    )
    assert grids.shape[0] == layout.experts, (
        f"the scale stack holds {grids.shape[0]} rows, not {layout.experts}"
    )
    assert aligned_weight == 0.0, (
        f"expert weights are not bit-identical to their own checkpoint tensors "
        f"through the same squeeze: worst abs diff {aligned_weight}. A stack is "
        f"a permutation of its inputs, so this is an indexing defect"
    )
    assert aligned_scale == 0.0, (
        f"expert scale rows are not bit-identical to their own grids through the "
        f"same compensation: worst abs diff {aligned_scale}"
    )
    assert rotated_weight > 0.0 and rotated_scale > 0.0, (
        f"the rotation control did not move (weight {rotated_weight}, scale "
        f"{rotated_scale}), so the equalities above would hold for a stack in "
        f"any order and certify nothing about indexing"
    )


def test_the_stacked_bank_gives_each_rank_its_declared_experts(
    tmp_path, single_rank_process_group
) -> None:
    """At expert-parallel degree 2 the two ranks hold disjoint halves whose union is every expert."""
    directory = tmp_path / "stacked"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    _write_miniature_checkpoint(directory, mappings, model)
    banks = _bank_entries(mappings)
    name = sorted(banks)[0]
    keys = banks[name]
    layout = bank_layout(keys, param_name=name)
    _distinguish_bank_experts(directory, keys)
    slices = _slice_pairs(directory, keys)
    references = [
        downscale_fp8_weight_bytes(_checkpoint_tensor(directory, keys[i]))
        for i in layout.weight_at
    ]

    def experts_in(stack) -> list[int]:
        """Which global expert each stacked row is, decided by its bytes."""
        found = []
        for row in range(stack.shape[0]):
            matches = [
                e
                for e in range(layout.experts)
                if float(
                    (stack[row].to(torch.float32) - references[e].to(torch.float32))
                    .abs()
                    .max()
                )
                == 0.0
            ]
            assert len(matches) == 1, (
                f"stacked row {row} matches {len(matches)} of the "
                f"{layout.experts} expert references, so it cannot be said which "
                f"expert it holds"
            )
            found.append(matches[0])
        return found

    split = _stacked_bank_geometry(layout.experts, STACKED_EXPERTS_PER_RANK)
    loader = stacked_expert_bank_loader(keys, param_name=name, owner=split)
    rank0 = experts_in(loader.transform(slices, 0))
    rank1 = experts_in(loader.transform(slices, 1))

    whole = _stacked_bank_geometry(layout.experts, layout.experts)
    degree1 = experts_in(
        stacked_expert_bank_loader(keys, param_name=name, owner=whole).transform(
            slices, 0
        )
    )

    assert len(rank0) == STACKED_EXPERTS_PER_RANK, (
        f"rank 0 holds {len(rank0)} experts at degree {STACKED_EP_DEGREE}, not "
        f"{STACKED_EXPERTS_PER_RANK}"
    )
    assert len(rank1) == STACKED_EXPERTS_PER_RANK, (
        f"rank 1 holds {len(rank1)} experts at degree {STACKED_EP_DEGREE}, not "
        f"{STACKED_EXPERTS_PER_RANK}"
    )
    assert set(rank0) & set(rank1) == set(), (
        f"the two ranks share experts {sorted(set(rank0) & set(rank1))}, so the "
        f"same weight was loaded twice and the partition is not a partition"
    )
    assert set(rank0) | set(rank1) == set(range(layout.experts)), (
        f"the two ranks together hold {sorted(set(rank0) | set(rank1))}, not all "
        f"{layout.experts} experts, so an expert reached no rank at all"
    )
    assert degree1 == list(range(layout.experts)), (
        f"the degree-1 control holds {degree1} rather than every expert in "
        f"order, so the selection is not reading the declared geometry"
    )


def test_a_well_formed_bank_gets_a_loader_and_malformed_entries_refuse(tmp_path) -> None:
    """A well-formed bank gets a loader and an fp8 placeholder; odd, non-alternating and scale-free entries refuse by name."""
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    banks = _bank_entries(mappings)
    name = sorted(banks)[0]
    keys = banks[name]
    owner = model.get_submodule(name.rsplit(".", 1)[0])

    kind = classify_mapped_keys(keys)
    loader = loader_for_mapped_keys(keys, param_name=name, owner=owner)
    dtype = model._placeholder_dtype(keys, param_name=name, mappings=mappings)
    assert kind == MAPPED_KEY_STACKED_BANK, (
        f"a well-formed bank classifies {kind!r}, not {MAPPED_KEY_STACKED_BANK!r}"
    )
    assert loader is not None and loader.transform is not None, (
        "a well-formed bank did not get a transforming loader"
    )
    assert _is_fp8_dtype(dtype), (
        f"a bank placeholder is typed {dtype}, not fp8. The classifier's second "
        f"consumer does not know the fourth kind, so every bank load would warn "
        f"on a dtype mismatch against a loader that delivers fp8"
    )

    weight, scale = keys[0], keys[1]
    malformed = {
        "an odd key count": (keys[:-1], "odd count"),
        "a scale where a weight belongs": ([scale] + keys[1:], "do not alternate"),
        "several weights and no scale": ([weight, weight + ".dup"], "0 scale keys"),
    }
    refusals = {}
    for defect, (entry, expected) in malformed.items():
        with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
            loader_for_mapped_keys(entry, param_name=name, owner=owner)
        message = str(refusal.value)
        refusals[defect] = message
        assert name in message, (
            f"the refusal for {defect} does not name the parameter: {message}"
        )
        # Compared lower-cased, so the refusal's wording may change case freely.
        assert expected in message.lower(), (
            f"the refusal for {defect} does not name the defect "
            f"({expected!r} absent): {message}"
        )
        assert str(len(entry)) in message, (
            f"the refusal for {defect} does not name the key count: {message}"
        )
    assert len(refusals) == 3, "one of the malformed shapes did not refuse"


BANKSCALE_LEAVES = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")


def _scale_grid_attribute(leaf: str) -> str:
    """The attribute a weight leaf's scale grid arrives under, asked of the model's own naming rule."""
    return Glm5NextForConditionalGeneration._sibling_scale_grid_name(leaf)


def _alter_one_scale_slice(directory: Path, key: str) -> float:
    """Add 1.0 to every element of one scale key in the written file."""
    path = directory / MINI_CHECKPOINT_FILE
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as opened:
        for name in opened.keys():
            tensors[name] = opened.get_tensor(name)
    tensors[key] = tensors[key].to(torch.float32) + 1.0
    save_file(tensors, str(path))
    return float(tensors[key].flatten()[0])


def _bank_owner_paths(banks: dict[str, list[str]]) -> list[str]:
    """The module paths that own the bank entries."""
    return sorted({name.rsplit(".", 1)[0] for name in banks})


def test_the_bank_scale_grids_arrive_as_plain_attributes_row_per_expert(
    keep_the_loaded_tensors, tmp_path, single_rank_process_group
) -> None:
    """The bank's three grids arrive as plain attributes with one row per local expert, each row its expert's own grid."""
    device = torch.device("cpu")
    directory = tmp_path / "bankscale"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    _stacked_checkpoint(directory, mappings, model)
    banks = _bank_entries(mappings)
    assert banks, (
        "this configuration produced no expert-bank entry, so the bank branch of "
        "the scale read is never reached"
    )
    owner_paths = _bank_owner_paths(banks)
    subject = sorted(banks)[0]
    subject_keys = banks[subject]
    layout = bank_layout(subject_keys, param_name=subject)
    _distinguish_bank_experts(directory, subject_keys)

    model.load_weights(str(directory), device, None)

    parameter_names = set(dict(model.named_parameters()))
    present: dict[str, tuple[int, ...]] = {}
    missing: list[str] = []
    as_parameters: list[str] = []
    axes: dict[str, int] = {}
    for path in owner_paths:
        module = model.get_submodule(path)
        for leaf in BANKSCALE_LEAVES:
            attribute = _scale_grid_attribute(leaf)
            dotted = f"{path}.{attribute}"
            grid = getattr(module, attribute, None)
            if grid is None:
                missing.append(dotted)
                continue
            present[dotted] = tuple(grid.shape)
            axes[dotted] = int(grid.shape[0]) - int(module.num_local_experts)
            if dotted in parameter_names:
                as_parameters.append(dotted)

    assert missing == [], (
        f"{len(missing)} bank scale grids never arrived, e.g. {missing[:3]}; the "
        f"reader's bank branch did not run or stored them under other names"
    )
    assert len(present) == 3 * len(owner_paths), (
        f"expected three grids on each of {len(owner_paths)} bank modules, found "
        f"{len(present)}"
    )
    assert as_parameters == [], (
        f"{len(as_parameters)} scale grids were registered as parameters, e.g. "
        f"{as_parameters[:3]}. Registering one adds a name to named_parameters() "
        f"that the weight map does not carry, which is the map widening the "
        f"plain-attribute convention exists to avoid"
    )
    assert set(axes.values()) == {0}, (
        f"a grid's leading axis is not its module's num_local_experts: read "
        f"differences {sorted(set(axes.values()))}, expected [0]"
    )

    owner = model.get_submodule(subject.rsplit(".", 1)[0])
    attribute = _scale_grid_attribute(subject.rsplit(".", 1)[1])
    grid = getattr(owner, attribute)
    references = [
        compensate_block_scales(
            _checkpoint_tensor(directory, subject_keys[position])
        ).scale_inv
        for position in layout.scale_at
    ]
    unequal = [
        expert
        for expert in range(layout.experts)
        if not torch.equal(grid[expert], references[expert])
    ]
    assert unequal == [], (
        f"rows {unequal} are not bit-identical to their own expert's grid "
        f"through compensate_block_scales. A stack is a permutation of its "
        f"inputs, so this is an indexing defect and not a tolerance question"
    )

    without_banks = {
        name: keys for name, keys in mappings.items() if name not in banks
    }
    full_read = _stacked_model()._load_out_of_band_scales(
        SafetensorsCheckpoint(str(directory)), mappings, device
    )
    lone_read = _stacked_model()._load_out_of_band_scales(
        SafetensorsCheckpoint(str(directory)), without_banks, device
    )
    assert full_read - lone_read == len(banks), (
        f"the reader's return rose by {full_read - lone_read} with the bank "
        f"entries in the map, not by the {len(banks)} bank entries it read"
    )
    assert len(banks) == 3 * len(owner_paths), (
        f"{len(banks)} bank entries over {len(owner_paths)} bank modules is not "
        f"three per bank, so the rise above is not the per-bank rise"
    )

    control_directory = tmp_path / "bankscale-control"
    control_model = _stacked_model()
    _stacked_checkpoint(control_directory, mappings, control_model)
    _distinguish_bank_experts(control_directory, subject_keys)
    altered_expert = layout.experts - 1
    altered_key = subject_keys[layout.scale_at[altered_expert]]
    _alter_one_scale_slice(control_directory, altered_key)
    control_model.load_weights(str(control_directory), device, None)
    control_grid = getattr(
        control_model.get_submodule(subject.rsplit(".", 1)[0]), attribute
    )
    moved = [
        expert
        for expert in range(layout.experts)
        if not torch.equal(grid[expert], control_grid[expert])
    ]
    assert moved == [altered_expert], (
        f"altering expert {altered_expert}'s scale slice moved rows {moved}. "
        f"Zero rows means the reader is not reading this file; every row means "
        f"it is not reading per expert"
    )


def test_the_scale_prep_leaves_are_the_weights_whose_grid_is_present(
    tmp_path, single_rank_process_group
) -> None:
    """``_scale_prep_leaves`` returns the weight leaves whose grid is attached, and skips ``router_weight``."""
    device = torch.device("cpu")
    directory = tmp_path / "bankscale-leaves"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    _stacked_checkpoint(directory, mappings, model)
    banks = _bank_entries(mappings)
    assert banks, "this configuration produced no expert-bank entry"
    model.load_weights(str(directory), device, None)

    bank = model.get_submodule(_bank_owner_paths(banks)[0])
    declared = tuple(getattr(bank, "declared_param_names", ()))
    leaves = _scale_prep_leaves(bank)

    assert leaves == list(BANKSCALE_LEAVES), (
        f"the helper returned {leaves} for a loaded bank, not the three "
        f"projection leaves {list(BANKSCALE_LEAVES)} whose grids arrived"
    )
    assert "router_weight" in declared, (
        "the bank no longer declares router_weight, so there is no grid-free "
        "weight leaf for the helper to skip"
    )
    assert "router_weight" not in leaves, (
        "the helper offered router_weight, a weight leaf with no scale grid"
    )

    shared = Glm5NextSharedExperts(_dense_config().text_config)
    bare = _scale_prep_leaves(shared)
    for leaf in BANKSCALE_LEAVES:
        setattr(shared, _scale_grid_attribute(leaf), torch.zeros(1, 1))
    attached = _scale_prep_leaves(shared)

    assert attached == list(BANKSCALE_LEAVES), (
        f"the helper returned {attached} for a shared expert with its three grids "
        f"attached"
    )
    assert bare == [], (
        f"the helper returned {bare} for a shared expert with no grids attached, "
        f"so it is reading the class rather than the attributes"
    )


def test_the_prep_loop_visits_every_module_that_declares_a_prep(
    keep_the_loaded_tensors, tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """``_run_load_time_preps`` visits exactly the modules whose type declares a prep and hands the bank six operands."""
    device = torch.device("cpu")
    directory = tmp_path / "bankscale-loop"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    _stacked_checkpoint(directory, mappings, model)
    banks = _bank_entries(mappings)
    owner_paths = _bank_owner_paths(banks)
    assert owner_paths, "this configuration built no bank module"
    model.load_weights(str(directory), device, None)

    expected_projection = sum(
        1
        for _, module in model.named_modules()
        if hasattr(type(module), "prepare_projection_weights")
    )
    expected_scale = sum(
        1
        for _, module in model.named_modules()
        if hasattr(type(module), "prepare_scale_operands")
    )
    projection_calls, scale_calls = model._run_load_time_preps(device)

    assert (projection_calls, scale_calls) == (expected_projection, expected_scale), (
        f"the loop returned ({projection_calls}, {scale_calls}) where its own two "
        f"gates select ({expected_projection}, {expected_scale}) modules, so it "
        f"visited something other than what it tests for"
    )
    assert scale_calls == len(owner_paths), (
        f"the loop ran {scale_calls} scale preps over {len(owner_paths)} bank "
        f"modules on a configuration with no shared-expert module. Every scale "
        f"prep on this tree is a bank's, so the two must agree: fewer means a "
        f"bank was skipped, more means something else declared a prep"
    )

    dense_directory = tmp_path / "bankscale-loop-dense"
    dense = _dense_model()
    dense_mappings = _mappings_for(_dense_config())
    _write_miniature_checkpoint(dense_directory, dense_mappings, dense)
    dense.load_weights(str(dense_directory), device, None)
    dense_banks = _bank_entries(dense_mappings)
    dense_expected = (
        sum(
            1
            for _, module in dense.named_modules()
            if hasattr(type(module), "prepare_projection_weights")
        ),
        sum(
            1
            for _, module in dense.named_modules()
            if hasattr(type(module), "prepare_scale_operands")
        ),
    )
    dense_pair = dense._run_load_time_preps(device)
    assert dense_banks == {}, (
        f"the all-dense configuration produced {len(dense_banks)} bank entries, "
        f"so it is not the bank-free tree this control needs"
    )
    assert dense_pair == dense_expected, (
        f"the loop returned {dense_pair} on the dense tree where its own gates "
        f"select {dense_expected}"
    )

    handed: dict[str, list[str]] = {}

    def stub(self, **operands) -> None:
        handed[type(self).__name__] = sorted(operands)

    bank_type = type(model.get_submodule(owner_paths[0]))
    monkeypatch.setattr(bank_type, "prepare_scale_operands", stub, raising=False)
    planted_projection, planted_scale = model._run_load_time_preps(device)

    assert planted_scale == len(owner_paths), (
        f"with a prep planted on {bank_type.__name__} the loop ran "
        f"{planted_scale} scale preps over {len(owner_paths)} bank modules; the "
        f"stub stands in for the real prep on the same type, so the count it "
        f"produces must be the same one the real prep produced above"
    )
    assert planted_projection == projection_calls, (
        f"planting a scale prep moved the projection count from "
        f"{projection_calls} to {planted_projection}"
    )
    expected_operands = sorted(
        [leaf for leaf in BANKSCALE_LEAVES]
        + [f"{leaf[: -len(_WEIGHT_LEAF_SUFFIX)]}_scale" for leaf in BANKSCALE_LEAVES]
    )
    assert handed.get(bank_type.__name__) == expected_operands, (
        f"the planted prep was handed {handed.get(bank_type.__name__)}, not the "
        f"six operands {expected_operands} the three arrived grids imply"
    )
