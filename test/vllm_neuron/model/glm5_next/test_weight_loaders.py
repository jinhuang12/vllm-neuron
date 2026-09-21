# SPDX-License-Identifier: Apache-2.0
"""Tests for the GLM-5.3-Flash checkpoint-to-parameter weight mapping.

Three subjects, in order: the key map and shard index over a 4-layer miniature of
the real stack, the same map over the published 76,108-key checkpoint index, and
the blockwise-FP8 scale arithmetic the load path applies on a 240-max platform.

The miniature's checkpoint keys are enumerated by hand in
:func:`_hf_keys_for_layer` rather than taken from ``build_weight_mappings``, so
"every key is mapped" is the agreement of two independent enumerations and not a
tautology. ``hf_state_to_fake_slices`` supplies the slice map and has no notion of
a shard, so :func:`_partition_into_shards` here is what composes the shards, and
the duplicate tests run on the cross-shard path the helper cannot see.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from test.vllm_neuron.model.glm5_next.fixtures import weight_index
from test.vllm_neuron.model.utils import FakeSafeSlice, hf_state_to_fake_slices
from vllm_neuron.model.glm5_next import weight_loaders_fp8 as _fp8_module
from vllm_neuron.model.glm5_next.config import (
    DSA_LAYER_TYPE,
    KDA_LAYER_TYPE,
    Glm5NextConfig,
    Glm5NextTextConfig,
)
from vllm_neuron.model.glm5_next.quantization import (
    DEFAULT_WEIGHT_BLOCK_SIZE,
    QuantScheme,
    QuantizationSpec,
    keeps_bf16,
)
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    ABSENT_KEY_FAMILIES,
    GROUNDED,
    KEY_FAMILY_PROVENANCE,
    MINVAL,
    PROVISIONAL,
    DuplicateShardKeyError,
    Glm5NextShardIndex,
    block_agreement,
    block_grid_shape,
    blockwise_scale_loader,
    build_weight_mappings,
    check_key_coverage,
    compensate_block_scales,
    dequantise_blockwise,
    downscale_fp8_weight_bytes,
    needs_240_downscale,
    report_floored_blocks,
    resolved_fp8_clamp_max,
    scale_keys,
    squeeze_blockwise_fp8,
    wrap_with_blockwise_fp8_downscale,
)
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

# --------------------------------------------------------------------------- #
# The miniature fixture's shape.
# --------------------------------------------------------------------------- #

#: How many shard files the miniature is split over.
NUM_SHARDS = 3

#: Layers 0-3 of the real stack, with the expert count scaled down to 4.
MINI_LAYERS = 4
MINI_ROUTED_EXPERTS = 4
MINI_SHARED_EXPERTS = 1
FIRST_K_DENSE = 3

SCALE_SUFFIX = "weight_scale_inv"


# --------------------------------------------------------------------------- #
# The checkpoint key enumeration, written independently of the mapping builder.
# --------------------------------------------------------------------------- #


def _hf_keys_for_layer(
    *, is_dsa: bool, is_moe: bool, n_routed: int, n_shared: int
) -> list[str]:
    """Every checkpoint key one layer of the miniature holds, without its prefix.

    ``hf_state_to_fake_slices`` applies the ``model.layers.<i>.`` prefix. The
    suffixes are spelled out rather than built, so this enumeration shares no code
    with the mapping builder it is checked against.
    """
    keys = ["input_layernorm.weight", "post_attention_layernorm.weight"]

    # Multi-hyper-connections: six bare tensors on every layer of the real stack.
    keys += [
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
    ]

    if is_dsa:
        # Only these four carry a scale companion.
        for leaf in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
            keys += [f"self_attn.{leaf}.weight", f"self_attn.{leaf}.{SCALE_SUFFIX}"]
        keys += [
            "self_attn.kv_b_proj.weight",  # real projection, no scale in this ckpt
            "self_attn.q_a_layernorm.weight",
            "self_attn.kv_a_layernorm.weight",
        ]
        # The indexer carries no scale companion in this checkpoint.
        keys += [
            "self_attn.indexer.wq_b.weight",
            "self_attn.indexer.wk.weight",
            "self_attn.indexer.k_norm.weight",
            "self_attn.indexer.k_norm.bias",
            "self_attn.indexer.weights_proj.weight",
            "self_attn.indexer.index_kpool_compress_ape",
            "self_attn.indexer.index_kpool_compress_gate",
        ]
    else:
        # The KDA family is 15 ``self_attn`` leaves and no scale companion.
        for leaf in (
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "f_b_proj",
            "g_a_proj",
            "g_b_proj",
            "q_conv1d",
            "k_conv1d",
            "v_conv1d",
            "o_norm",
            "o_proj",
        ):
            keys += [f"self_attn.{leaf}.weight"]
        keys += ["self_attn.A_log", "self_attn.dt_bias"]

    if is_moe:
        keys += ["mlp.gate.weight", "mlp.gate.e_score_correction_bias"]
        for expert_id in range(n_routed):
            for leaf in ("gate_proj", "up_proj", "down_proj"):
                keys += [
                    f"mlp.experts.{expert_id}.{leaf}.weight",
                    f"mlp.experts.{expert_id}.{leaf}.{SCALE_SUFFIX}",
                ]
        if n_shared:
            for leaf in ("gate_proj", "up_proj", "down_proj"):
                keys += [
                    f"mlp.shared_experts.{leaf}.weight",
                    f"mlp.shared_experts.{leaf}.{SCALE_SUFFIX}",
                ]
    else:
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            keys += [f"mlp.{leaf}.weight", f"mlp.{leaf}.{SCALE_SUFFIX}"]

    return keys


def _non_layer_hf_keys(*, tie_word_embeddings: bool) -> list[str]:
    """Checkpoint keys outside the layer stack (``layer_idx=None``)."""
    keys = ["model.embed_tokens.weight", "model.norm.weight"]
    if not tie_word_embeddings:
        keys.append("lm_head.weight")
    return keys


#: The prefix ``hf_state_to_fake_slices`` applies, and the prefix the published
#: checkpoint index uses for the same tensors.
MODULE_PREFIX = "model."
CKPT_PREFIX = "model.language_model."


def _into_checkpoint_namespace(key: str) -> str:
    """Move one module-namespace key into the checkpoint's namespace.

    ``hf_state_to_fake_slices`` qualifies a layer key as ``model.layers.<i>.<leaf>``
    while the checkpoint puts every text-model tensor under
    ``model.language_model.``. The rewrite runs after the shared helper rather than
    inside it, because other models use that helper. ``lm_head.weight`` sits outside
    the text-model prefix in the checkpoint, so this is a prefix rewrite and not a
    blanket one.
    """
    if key.startswith(CKPT_PREFIX):
        return key
    if key.startswith(MODULE_PREFIX):
        return CKPT_PREFIX + key[len(MODULE_PREFIX) :]
    return key


# --------------------------------------------------------------------------- #
# The fixture: the slice map from the helper, the shard composition from here.
# --------------------------------------------------------------------------- #


def _fake_state(keys: list[str]) -> dict[str, torch.Tensor]:
    """A synthetic HF state dict. Shapes are irrelevant to key routing."""
    return {key: torch.zeros(2, 2) for key in keys}


def _build_slice_map(cfg: Glm5NextTextConfig) -> dict[str, FakeSafeSlice]:
    """The flat ``{checkpoint_key: FakeSafeSlice}`` map for the miniature.

    One ``hf_state_to_fake_slices`` call per layer, so the helper applies each
    layer's own prefix, plus one with ``layer_idx=None`` for the rest.
    """
    raw: dict[str, FakeSafeSlice] = {}
    raw.update(
        hf_state_to_fake_slices(
            _fake_state(
                _non_layer_hf_keys(tie_word_embeddings=cfg.tie_word_embeddings)
            ),
            None,
        )
    )
    for layer_id, layer_type in enumerate(cfg.layer_types):
        layer_keys = _hf_keys_for_layer(
            is_dsa=layer_type == DSA_LAYER_TYPE,
            is_moe=layer_id >= cfg.first_k_dense_replace,
            n_routed=cfg.n_routed_experts,
            n_shared=cfg.n_shared_experts,
        )
        raw.update(hf_state_to_fake_slices(_fake_state(layer_keys), layer_id))
    # The helper qualifies into the module namespace and the checkpoint is one
    # namespace over. The rewrite is injective on this key set, asserted below, so
    # the key count cannot change under it.
    slice_map = {_into_checkpoint_namespace(key): sl for key, sl in raw.items()}
    assert len(slice_map) == len(raw), "namespace rewrite collided two keys"
    return slice_map


def _shard_name(shard_id: int, total: int) -> str:
    return f"model-{shard_id + 1:05d}-of-{total:05d}.safetensors"


def _partition_into_shards(
    keys: list[str], num_shards: int = NUM_SHARDS
) -> dict[str, list[str]]:
    """Split a flat key list into contiguous per-shard key lists.

    Contiguously rather than round-robin, because that is how a published shard set
    is laid out.
    """
    per_shard, remainder = divmod(len(keys), num_shards)
    shards: dict[str, list[str]] = {}
    start = 0
    for shard_id in range(num_shards):
        size = per_shard + (1 if shard_id < remainder else 0)
        shards[_shard_name(shard_id, num_shards)] = keys[start : start + size]
        start += size
    assert start == len(keys), "partition dropped keys"
    return shards


@pytest.fixture
def mini_config() -> Glm5NextTextConfig:
    """The miniature text config: layers 0-3 of the real stack."""
    return Glm5NextTextConfig(
        num_hidden_layers=MINI_LAYERS,
        n_routed_experts=MINI_ROUTED_EXPERTS,
        n_shared_experts=MINI_SHARED_EXPERTS,
        first_k_dense_replace=FIRST_K_DENSE,
        tie_word_embeddings=False,
    )


@pytest.fixture
def slice_map(mini_config: Glm5NextTextConfig) -> dict[str, FakeSafeSlice]:
    return _build_slice_map(mini_config)


@pytest.fixture
def shard_index(slice_map: dict[str, FakeSafeSlice]) -> Glm5NextShardIndex:
    return Glm5NextShardIndex.from_shard_key_lists(
        _partition_into_shards(list(slice_map))
    )


@pytest.fixture
def coverage(shard_index, mini_config):
    return check_key_coverage(shard_index, build_weight_mappings(mini_config))


# --------------------------------------------------------------------------- #
# The miniature holds the shape the tests below assume.
# --------------------------------------------------------------------------- #


def test_miniature_config_reproduces_the_real_layer_phases(mini_config) -> None:
    """Layers 0-2 are KDA + dense; layer 3 is DSA + MoE, as in the real stack."""
    assert list(mini_config.layer_types) == [
        KDA_LAYER_TYPE,
        KDA_LAYER_TYPE,
        KDA_LAYER_TYPE,
        DSA_LAYER_TYPE,
    ]
    assert mini_config.attention_layer_split == (3, 1)
    assert mini_config.dsa_layer_indices == [3]
    assert mini_config.first_k_dense_replace == FIRST_K_DENSE


def test_shard_index_reports_the_fixture_shard_count(shard_index) -> None:
    """The index reports one shard per shard file it was built from."""
    assert shard_index.num_shards == NUM_SHARDS
    assert len(shard_index.per_shard_counts()) == NUM_SHARDS


# --------------------------------------------------------------------------- #
# Key coverage over the miniature.
# --------------------------------------------------------------------------- #


def test_every_checkpoint_key_in_the_fixture_is_mapped(coverage, slice_map) -> None:
    """``check_key_coverage`` maps every key the fixture holds."""
    assert coverage.unique_checkpoint_key_count == len(slice_map)
    assert coverage.mapped_key_count == len(slice_map)
    assert coverage.coverage_fraction == 1.0


def test_no_key_and_no_parameter_is_left_unmatched(coverage) -> None:
    """Neither direction of the comparison reports a leftover."""
    assert coverage.unmatched_checkpoint_keys == ()
    assert coverage.unmatched_parameters == {}
    assert coverage.unmatched_count == 0
    assert coverage.is_complete


def test_unmatched_counters_report_a_missing_and_an_extra_key(
    mini_config, slice_map
) -> None:
    """Each counter moves when its own side of the comparison is perturbed.

    Dropping a key the mapping asks for leaves an unmatched parameter while the
    coverage fraction stays 1.0, because a key that is absent cannot be unmapped.
    Adding a key no parameter asks for moves both readings.
    """
    mappings = build_weight_mappings(mini_config)
    all_keys = list(slice_map)

    # Drop one checkpoint key the mapping asks for: an unmatched parameter. The
    # checkpoint side is namespaced and the parameter name is not, which is why the
    # two literals below differ.'
    dropped = "model.language_model.layers.3.mlp.shared_experts.down_proj.weight"
    assert dropped in slice_map, "fixture no longer holds the key this arm drops"
    short_index = Glm5NextShardIndex.from_shard_key_lists(
        _partition_into_shards([k for k in all_keys if k != dropped])
    )
    short = check_key_coverage(short_index, mappings)
    assert len(short.unmatched_parameters) == 1
    assert short.unmatched_parameters == {
        "model.layers.3.mlp.shared_experts.down_proj_weight": (dropped,)
    }
    assert short.unmatched_count == 1
    assert short.coverage_fraction == 1.0  # an absent key cannot be unmapped
    assert not short.is_complete

    # Add a checkpoint key no parameter asks for: an unmatched key, and this is the
    # direction that moves the coverage fraction.
    extra = "model.layers.3.mlp.experts.999.gate_proj.weight"
    assert extra not in slice_map
    long_index = Glm5NextShardIndex.from_shard_key_lists(
        _partition_into_shards([*all_keys, extra])
    )
    long = check_key_coverage(long_index, mappings)
    assert long.unmatched_checkpoint_keys == (extra,)
    assert long.unmatched_count == 1
    assert long.coverage_fraction < 1.0
    assert long.mapped_key_count == len(slice_map)
    assert not long.is_complete


# --------------------------------------------------------------------------- #
# Duplicate keys across shard files.
# --------------------------------------------------------------------------- #


def test_a_clean_shard_set_reports_no_duplicate(coverage, shard_index) -> None:
    """A shard set with no repeated key reports no duplicate."""
    assert coverage.duplicated_keys == {}
    assert coverage.duplicated_count == 0
    assert shard_index.duplicated_keys() == {}
    shard_index.require_no_duplicates()  # must not raise


def test_a_cross_shard_duplicate_is_reported_by_the_shard_index(
    mini_config, slice_map
) -> None:
    """One key in two shard files is reported by the index, and raises under strict.

    The fixture builder cannot see such a duplicate, since it runs per layer on a
    flat dict, and a flattened ``{key: file}`` map cannot represent one either. Both
    are checked here, so the report is attributable to the index alone.
    """
    # Layer 0 is a KDA layer, whose output projection is ``self_attn.o_proj``.
    duplicated = "model.language_model.layers.0.self_attn.o_proj.weight"
    assert duplicated in slice_map

    shards = _partition_into_shards(list(slice_map))
    shard_names = list(shards)
    home = next(name for name in shard_names if duplicated in shards[name])
    other = next(name for name in shard_names if name != home)
    shards[other] = [*shards[other], duplicated]  # now in two shard files

    # The helper's own within-layer duplicate guard stays silent: a cross-shard
    # duplicate is invisible to it.
    rebuilt = _build_slice_map(mini_config)
    assert set(rebuilt) == set(slice_map)

    # A flattened ``{key: file}`` map cannot hold the duplicate at all.
    flattened: dict[str, str] = {}
    for shard, keys in shards.items():
        for key in keys:
            flattened[key] = shard
    assert len(flattened) == len(slice_map)
    collapsed = Glm5NextShardIndex.from_weight_map(flattened)
    assert collapsed.duplicated_keys() == {}, "the lossy direction must lose it"

    # Leg 3 -- the loader reports it off the per-shard key lists.
    index = Glm5NextShardIndex.from_shard_key_lists(shards)
    reported = index.duplicated_keys()
    assert set(reported) == {duplicated}
    assert set(reported[duplicated]) == {home, other}

    dirty = check_key_coverage(index, build_weight_mappings(mini_config))
    assert dirty.duplicated_count == 1
    assert not dirty.is_complete

    with pytest.raises(DuplicateShardKeyError, match=duplicated):
        index.require_no_duplicates()
    with pytest.raises(DuplicateShardKeyError):
        check_key_coverage(index, build_weight_mappings(mini_config), strict=True)


# --------------------------------------------------------------------------- #
# Per-shard key counts.
# --------------------------------------------------------------------------- #


def test_per_shard_counts_sum_to_the_fixture_key_count(
    shard_index, coverage, slice_map
) -> None:
    """The per-shard counts add up to the number of keys in the fixture."""
    counts = shard_index.per_shard_counts()
    assert len(counts) == NUM_SHARDS
    assert all(count > 0 for count in counts.values())
    assert sum(counts.values()) == len(slice_map)
    assert shard_index.total_shard_key_count == len(slice_map)
    assert coverage.total_shard_key_count == len(slice_map)
    assert shard_index.unique_key_count == len(slice_map)


def test_per_shard_sum_exceeds_the_unique_count_when_a_key_repeats(slice_map) -> None:
    """The sum over shards exceeds the unique count when a key sits in two shards."""
    shards = _partition_into_shards(list(slice_map))
    victim = next(iter(shards[_shard_name(0, NUM_SHARDS)]))
    last = _shard_name(NUM_SHARDS - 1, NUM_SHARDS)
    shards[last] = [*shards[last], victim]

    index = Glm5NextShardIndex.from_shard_key_lists(shards)
    assert index.total_shard_key_count == len(slice_map) + 1
    assert index.unique_key_count == len(slice_map)


# --------------------------------------------------------------------------- #
# The mapping's own surface: family tags, list-valued parameters, index parsing.
# --------------------------------------------------------------------------- #


def test_every_key_family_carries_a_provenance_tag() -> None:
    """Every key family carries a provenance tag, and every absence names a reason."""
    assert KEY_FAMILY_PROVENANCE
    assert set(KEY_FAMILY_PROVENANCE.values()) <= {GROUNDED, PROVISIONAL}
    # Both families' leaf names are read off the published index, so both are
    # grounded rather than inferred from a sibling architecture.
    assert {"dsa_indexer", "kda_linear_attention"} <= {
        name for name, tag in KEY_FAMILY_PROVENANCE.items() if tag == GROUNDED
    }
    # The hyper-connection family is present in the index and mapped, so the only
    # absent family is the vision tower.
    assert set(ABSENT_KEY_FAMILIES) == {"vision_tower"}
    assert all(reason.strip() for reason in ABSENT_KEY_FAMILIES.values())


def test_a_moe_expert_parameter_maps_to_a_list_of_keys(mini_config) -> None:
    """A fused expert parameter maps to the list of its per-expert keys.

    This checkpoint stores one tensor per expert, so the fused parameter cannot map
    to a single key.
    """
    mappings = build_weight_mappings(mini_config)
    param = "model.layers.3.mlp.experts.gate_proj_weight"
    keys = mappings[param]
    assert isinstance(keys, list)
    # 4 experts x (weight + scale)
    assert len(keys) == MINI_ROUTED_EXPERTS * 2
    assert len(scale_keys(keys)) == MINI_ROUTED_EXPERTS
    # A single-key parameter stays a bare string.
    assert isinstance(mappings["model.norm_weight"], str)


def test_no_rope_projection_is_mapped(mini_config) -> None:
    """``mla_use_nope`` with ``qk_rope_head_dim == 0``: no rotary head slice."""
    mappings = build_weight_mappings(mini_config)
    every_key = [
        key
        for value in mappings.values()
        for key in (value if isinstance(value, list) else [value])
    ]
    assert mini_config.qk_rope_head_dim == 0
    assert [key for key in every_key if "rope" in key] == []


def test_shard_index_reads_a_weight_map_out_of_index_json(slice_map) -> None:
    """``from_index_json`` reads a ``weight_map`` and agrees on the counts."""
    shards = _partition_into_shards(list(slice_map))
    weight_map = {key: shard for shard, keys in shards.items() for key in keys}
    index = Glm5NextShardIndex.from_index_json(json.dumps({"weight_map": weight_map}))
    assert index.num_shards == NUM_SHARDS
    assert index.unique_key_count == len(slice_map)
    assert sum(index.per_shard_counts().values()) == len(slice_map)


# =========================================================================== #
# The published checkpoint index and config
# =========================================================================== #
#
# The tests above run on the 4-layer miniature; the tests below run on the
# published index itself -- 76,108 keys over 62 shards, rebuilt by
# `fixtures/weight_index.py` -- and on the published config in
# `fixtures/hf-config.json`. Every denominator is derived from those fixtures, so a
# fixture change moves the expectation with it instead of disagreeing with a
# literal.

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_CONFIG_PATH = FIXTURES_DIR / "hf-config.json"

#: The two families outside the text model: the draft layer and the vision tower.
#: Prefixes, not counts -- the counts are read off the fixture.
MTP_LAYER_PREFIX = "model.language_model.layers.45."
VISION_PREFIX = "model.visual."

#: The published config's own digest and byte count. Usable as expected values
#: because the fixture is a byte-identical copy of the published file, recorded in
#: its provenance sidecar.
VENDOR_CONFIG_SHA256 = (
    "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
)
VENDOR_CONFIG_BYTES = 69416


@pytest.fixture(scope="module")
def real_weight_map() -> dict[str, str]:
    """The published index's own ``weight_map``: ``{checkpoint_key: shard}``."""
    return weight_index.weight_map()


@pytest.fixture(scope="module")
def real_text_config() -> Glm5NextTextConfig:
    """The text config built from the published config rather than from defaults."""
    raw = json.loads(REAL_CONFIG_PATH.read_text())
    return Glm5NextTextConfig.from_hf_config(raw["text_config"])


@pytest.fixture(scope="module")
def real_in_scope(real_weight_map) -> dict[str, str]:
    """The in-scope keys: the whole map less the draft layer and the vision tower."""
    return {
        key: shard
        for key, shard in real_weight_map.items()
        if not key.startswith(MTP_LAYER_PREFIX) and not key.startswith(VISION_PREFIX)
    }


@pytest.fixture(scope="module")
def real_index(real_in_scope) -> Glm5NextShardIndex:
    """A shard index over the in-scope keys, built from per-shard key lists.

    ``from_shard_key_lists`` and not ``from_weight_map``: a flattened map cannot
    represent a cross-shard duplicate, so the duplicate test below would read zero
    from the container rather than from the index.
    """
    per_shard: dict[str, list[str]] = {}
    for key, shard in real_in_scope.items():
        per_shard.setdefault(shard, []).append(key)
    return Glm5NextShardIndex.from_shard_key_lists(per_shard)


@pytest.fixture(scope="module")
def real_mappings(real_text_config) -> dict[str, str | list[str]]:
    """The mapping under test, over the published 45-layer schedule."""
    return build_weight_mappings(real_text_config)


@pytest.fixture(scope="module")
def real_referenced(real_mappings) -> frozenset[str]:
    """Every checkpoint key the mapping references, flattened."""
    return frozenset(
        key
        for value in real_mappings.values()
        for key in (value if isinstance(value, list) else [value])
    )


@pytest.fixture(scope="module")
def real_coverage(real_index, real_mappings):
    return check_key_coverage(real_index, real_mappings)


def _layers_of(text_config: Glm5NextTextConfig, family: str) -> list[int]:
    """Layer indices of one attention family, by equality on ``layer_types``."""
    return [i for i, t in enumerate(text_config.layer_types) if t == family]


def _self_attn_keys(referenced: frozenset[str], layers: list[int]) -> list[str]:
    """Referenced ``self_attn`` keys on the named layers, checkpoint namespace."""
    prefixes = tuple(f"model.language_model.layers.{i}.self_attn." for i in layers)
    return sorted(key for key in referenced if key.startswith(prefixes))


def _leaf_after(key: str, marker: str) -> str:
    return key.split(marker, 1)[1]


# --------------------------------------------------------------------------- #
# The in-scope key population.
# --------------------------------------------------------------------------- #


def test_in_scope_population_is_the_map_less_the_two_excluded_families(
    real_weight_map, real_in_scope
) -> None:
    """74,001 = 76,108 - 1,760 - 347, each term counted off the fixture.

    The three parts are asserted to sum to the whole and the two exclusions to be
    disjoint, so a family dropped twice or not at all shows up here.
    """
    total = len(real_weight_map)
    mtp = [k for k in real_weight_map if k.startswith(MTP_LAYER_PREFIX)]
    vision = [k for k in real_weight_map if k.startswith(VISION_PREFIX)]
    in_scope = len(real_in_scope)

    assert total == 76_108
    assert len(mtp) == 1_760
    assert len(vision) == 347
    assert in_scope == 74_001
    # The sum is what makes the subtraction a partition.
    assert in_scope + len(mtp) + len(vision) == total
    # The two exclusions are disjoint, so no key was subtracted twice.
    assert set(mtp).isdisjoint(vision)


def test_real_index_coverage_is_complete(real_coverage, real_in_scope) -> None:
    """74,001 of 74,001 in-scope keys are mapped, with nothing unmatched.

    The two unmatched lists are asserted empty before the count, so a failure names
    the keys and not only the number.
    """
    assert real_coverage.unmatched_checkpoint_keys == ()
    assert real_coverage.unmatched_parameters == {}
    assert real_coverage.unmatched_count == 0
    assert real_coverage.unique_checkpoint_key_count == len(real_in_scope)
    assert real_coverage.mapped_key_count == len(real_in_scope)
    assert real_coverage.coverage_fraction == 1.0
    assert real_coverage.is_complete


def test_every_scale_key_has_its_weight_partner_mapped(
    real_in_scope, real_referenced
) -> None:
    """Every in-scope ``weight_scale_inv`` key has its ``weight`` partner mapped.

    36,467 scale keys are in scope, so the empty orphan list is a reading over a
    non-empty set, and dropping one partner shows the same check reporting one.
    """
    scales = sorted(scale_keys(real_in_scope))
    scale_population = len(scales)

    def orphans(referenced: frozenset[str]) -> list[str]:
        out = []
        for key in scales:
            partner = key[: -len(f".{SCALE_SUFFIX}")] + ".weight"
            if partner not in referenced:
                out.append(key)
        return out

    real_orphans = orphans(real_referenced)

    # The denominator first: an empty orphan list over no scale key says nothing.
    assert scale_population == 36_467
    assert scale_population > 0
    assert real_orphans == []

    # Drop one partner, and the same check must report that key as an orphan.
    victim = scales[0]
    victim_partner = victim[: -len(f".{SCALE_SUFFIX}")] + ".weight"
    assert victim_partner in real_referenced
    poisoned = orphans(frozenset(real_referenced - {victim_partner}))
    assert poisoned == [victim], "the orphan check did not report a missing partner"


def test_real_index_has_no_duplicated_key(
    real_index, real_coverage, real_in_scope
) -> None:
    """No key of the 62 shards is duplicated, and the same check can report one.

    Read off per-shard key lists: a flattened map cannot represent a duplicate, so an
    empty report from one would be a property of the container.
    """
    assert real_index.duplicated_keys() == {}
    assert real_coverage.duplicated_count == 0
    real_index.require_no_duplicates()  # must not raise
    # Sum over shards equals the unique count exactly when nothing is duplicated.
    assert real_index.total_shard_key_count == len(real_in_scope)
    assert real_index.unique_key_count == len(real_in_scope)

    # The same method over a per-shard set with one key placed in a second shard.
    per_shard = {shard: list(keys) for shard, keys in real_index.shard_keys.items()}
    shard_names = list(per_shard)
    assert len(shard_names) > 1
    victim = per_shard[shard_names[0]][0]
    per_shard[shard_names[-1]] = [*per_shard[shard_names[-1]], victim]
    dirty = Glm5NextShardIndex.from_shard_key_lists(per_shard)
    assert set(dirty.duplicated_keys()) == {victim}
    assert dirty.total_shard_key_count == len(real_in_scope) + 1
    with pytest.raises(DuplicateShardKeyError, match=victim):
        dirty.require_no_duplicates()


# --------------------------------------------------------------------------- #
# The three attention and hyper-connection families, key by key.
# --------------------------------------------------------------------------- #


def test_kda_family_is_fifteen_leaves_with_no_scale(
    real_text_config, real_referenced, real_in_scope
) -> None:
    """510 = 15 x 34 keys, with no scale companion anywhere in the family.

    The checkpoint carries 15 ``self_attn.*`` leaves per KDA layer, not the
    ``qwen3_next`` ``linear_attn`` convention. Both the leaf set and the layer count
    are derived from the fixture, so the mapping is compared against the checkpoint
    and not against this module's constants.
    """
    kda_layers = _layers_of(real_text_config, KDA_LAYER_TYPE)
    marker = ".self_attn."

    mapped = _self_attn_keys(real_referenced, kda_layers)
    mapped_leaves = {_leaf_after(key, marker) for key in mapped}

    fixture = _self_attn_keys(frozenset(real_in_scope), kda_layers)
    fixture_leaves = {_leaf_after(key, marker) for key in fixture}

    assert len(kda_layers) == 34
    assert mapped_leaves == fixture_leaves  # the checkpoint decides the leaf set
    assert len(mapped_leaves) == 15
    assert len(mapped) == 15 * len(kda_layers) == 510
    assert scale_keys(mapped) == ()
    # The absence is the checkpoint's, not the mapping's opinion of it.
    assert scale_keys(fixture) == ()
    # No leaf follows the ``linear_attn`` convention.
    assert not any("linear_attn" in key for key in real_referenced)


def test_hyper_connection_family_is_six_tensors_on_every_layer(
    real_text_config, real_referenced, real_weight_map
) -> None:
    """270 = 6 x 45 keys, six on every text layer and none on the draft layer.

    The per-layer six is checked on each layer rather than in aggregate, so six on
    one layer and none on another cannot average into the total.
    """
    leaves = sorted(
        {
            key.rsplit(".", 1)[1]
            for key in real_weight_map
            if key.rsplit(".", 1)[1].startswith(("hc_attn_", "hc_ffn_"))
        }
    )
    assert leaves == [
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
    ]

    mapped = sorted(key for key in real_referenced if key.rsplit(".", 1)[1] in leaves)
    num_layers = len(real_text_config.layer_types)
    assert num_layers == 45
    assert len(mapped) == 6 * num_layers == 270

    per_layer = {
        i: [
            key
            for key in mapped
            if key.startswith(f"model.language_model.layers.{i}.")
        ]
        for i in range(num_layers)
    }
    assert sorted({len(v) for v in per_layer.values()}) == [6]

    # None on the draft layer, read off the fixture: that layer is out of scope.
    mtp_mhc = [
        key
        for key in real_weight_map
        if key.startswith(MTP_LAYER_PREFIX) and key.rsplit(".", 1)[1] in leaves
    ]
    assert mtp_mhc == []

    # The family is mapped, so it is tagged and not listed as absent.
    assert "multi_hyper_connections" not in ABSENT_KEY_FAMILIES
    assert KEY_FAMILY_PROVENANCE["multi_hyper_connections"] == GROUNDED


def test_dsa_family_is_eighteen_leaves_with_four_scaled(
    real_text_config, real_referenced, real_in_scope
) -> None:
    """198 = 18 x 11 keys, of which exactly 44 = 4 x 11 carry a scale.

    18 counts tensor leaves as a key map emits them -- 14 tensors, four of them with
    a ``weight_scale_inv`` companion -- rather than the 12 sub-modules they sit under.
    The four scaled leaves are named, because asking for a scale the checkpoint does
    not supply fails the load.
    """
    dsa_layers = _layers_of(real_text_config, DSA_LAYER_TYPE)
    marker = ".self_attn."

    mapped = _self_attn_keys(real_referenced, dsa_layers)
    fixture = _self_attn_keys(frozenset(real_in_scope), dsa_layers)

    assert len(dsa_layers) == 11
    assert {_leaf_after(k, marker) for k in mapped} == {
        _leaf_after(k, marker) for k in fixture
    }
    assert len(mapped) == 18 * len(dsa_layers) == 198

    scaled = sorted(
        {
            _leaf_after(key, marker)[: -len(f".{SCALE_SUFFIX}")]
            for key in scale_keys(mapped)
        }
    )
    assert len(scale_keys(mapped)) == 4 * len(dsa_layers) == 44
    assert scaled == ["kv_a_proj_with_mqa", "o_proj", "q_a_proj", "q_b_proj"]

    # The three corrections, each as its own presence reading.
    per_layer_leaves = {_leaf_after(key, marker) for key in mapped}
    assert "indexer.wq_b.weight" in per_layer_leaves
    assert "indexer.wq.weight" not in per_layer_leaves
    assert "indexer.k_norm.bias" in per_layer_leaves
    assert "indexer.index_kpool_compress_ape" in per_layer_leaves
    assert "indexer.index_kpool_compress_gate" in per_layer_leaves
    # And no indexer or kv_b_proj scale is requested anywhere.
    assert not any("indexer" in key for key in scale_keys(mapped))
    assert not any("kv_b_proj" in key for key in scale_keys(mapped))


# --------------------------------------------------------------------------- #
# (h) Both fixtures are pinned by digest and byte count -- 2/2.
# --------------------------------------------------------------------------- #


def test_rebuilt_weight_index_matches_the_published_digests() -> None:
    """The rebuilt index reproduces the published key set, shard map and totals.

    ``fixtures/weight_index.py`` rebuilds the 76,108-key index from the layer
    schedule in ``hf-config.json`` instead of vendoring 8.4 MB of JSON. Both
    digests it carries were taken from the published file, so an edit to the
    rebuild that moves one key or one shard assignment fails here.
    """
    assert weight_index.key_set_digest() == weight_index.KEY_SET_SHA256
    assert weight_index.weight_map_digest() == weight_index.WEIGHT_MAP_SHA256

    mapping = weight_index.weight_map()
    assert len(mapping) == weight_index.TOTAL_KEYS == 76_108
    assert len(set(mapping.values())) == weight_index.NUM_SHARDS == 62

    sidecar = FIXTURES_DIR / "model.safetensors.index.json.provenance.json"
    recorded = json.loads(sidecar.read_text())
    assert recorded["weight_map_sha256"] == weight_index.WEIGHT_MAP_SHA256
    assert recorded["key_set_sha256"] == weight_index.KEY_SET_SHA256
    assert recorded["keys"] == weight_index.TOTAL_KEYS
    assert recorded["shards"] == weight_index.NUM_SHARDS


def test_real_config_fixture_is_pinned_by_digest() -> None:
    """``hf-config.json`` is a byte copy of the published config, pinned by sha256.

    The fork compares against the published ``modules_to_not_convert`` skip list
    and the real ``text_config``, so this fixture is kept whole rather than
    trimmed, and the digest stops a quiet edit that would make a comparison pass.
    """
    data = REAL_CONFIG_PATH.read_bytes()
    got_sha = hashlib.sha256(data).hexdigest()
    assert got_sha == VENDOR_CONFIG_SHA256, f"hf-config.json: sha256 {got_sha}"
    assert len(data) == VENDOR_CONFIG_BYTES

    sidecar = FIXTURES_DIR / "hf-config.json.provenance.json"
    recorded = json.loads(sidecar.read_text())
    assert recorded["sha256"] == got_sha
    assert recorded["bytes"] == len(data)
    assert recorded["fixture_form"] == "byte-identical copy"


# =========================================================================== #
# Blockwise-FP8 scale loading under the 240-max squeeze
# =========================================================================== #
#
# On a platform whose fp8 clamp is 240.0, the load path multiplies the stored
# bytes by 1/2 and the per-block scale by 2, so the dequantised product is
# unchanged while every stored byte fits inside the clamp. The tests below run on
# a synthetic [256,256] weight with four [128,128] fp32 block scales.
#
# Agreement is read per block, with the block's own absolute maximum as the
# reference -- max|after - before| <= atol + rtol * max|before| -- because a
# global normalisation would mask a single block. The per-element difference is
# larger than rtol by construction: at an exact 1/2 the only inexact magnitudes
# are the odd multiples of 2**-9, each missing by one minimum subnormal, and the
# smallest subnormal halves onto a tie and rounds to zero.
#
# --------------------------------------------------------------------------- #
# The values the squeeze is defined by.
# --------------------------------------------------------------------------- #

#: Largest finite magnitude of the legacy ``nl.float8_e4m3``, which is the clamp a
#: trn2 platform resolves and the bound the stored bytes must fit inside.
FP8_PLATFORM_CLAMP = 240.0

#: Largest finite magnitude of OCP ``float8_e4m3fn``, the checkpoint's own space.
FP8_OCP_MAX = 448.0

#: The squeeze factor and its exact inverse: the largest power of two that fits 448
#: inside 240. Written as literals rather than read from the module, so this file
#: states the value the load path must hold.
FP8_DOWNSCALE = 0.5
FP8_COMPENSATION = 2.0

#: The range ratio, used only as the alternative factor the exactness count below
#: must be able to tell apart from an exact power of two.
FP8_RANGE_RATIO = FP8_PLATFORM_CLAMP / FP8_OCP_MAX

#: Smallest positive ``float8_e4m3fn`` magnitude. It is the grid step the eight
#: inexact magnitudes are odd multiples of, and the exact residual they miss by.
FP8_MIN_SUBNORMAL = 2.0**-9

#: The synthetic weight's shape and the checkpoint's block shape.
FP8_WEIGHT_SHAPE = (256, 256)
FP8_BLOCK_SIZE = (128, 128)

#: The tolerances every fp8 comparison here is read at.
FP8_RTOL = 3e-2
FP8_ATOL = 1e-5

#: The four fp32 block scales, one per ``[128,128]`` tile, spread over 1.5 orders of
#: magnitude and all above ``MINVAL`` so the floor engages on none of them.
FP8_BLOCK_SCALES = ((2.5e-3, 7.5e-4), (1.25e-2, 4.0e-4))

#: A scale four orders of magnitude below ``MINVAL`` even after the compensation has
#: multiplied it up, and the tile it is placed in.
FP8_TINY_SCALE = 1e-9
FP8_TINY_BLOCK = (1, 1)


# --------------------------------------------------------------------------- #
# The synthetic weight. Deterministic by construction -- no RNG, no seed.
# --------------------------------------------------------------------------- #


def _fp8_representable_magnitudes() -> torch.Tensor:
    """Every finite positive magnitude of ``float8_e4m3fn``, read off the format.

    All 256 byte values are viewed as the dtype, the non-finite ones are dropped and
    the absolute values de-duplicated, so a torch build with a different grid moves
    this list instead of disagreeing with it. On torch 2.11 it is 126 magnitudes from
    0.001953125 to 448.0, seven of them subnormal.
    """
    every_byte = (
        torch.arange(256, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    finite = every_byte[torch.isfinite(every_byte)]
    magnitudes = torch.unique(finite.abs())
    return magnitudes[magnitudes > 0]


def _squeeze_and_restore(
    magnitudes: torch.Tensor, down: float, up: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the load path's arithmetic over a tensor of magnitudes at a given factor.

    The steps are ``downscale_fp8_weight_bytes``'s own and in its order: fp32
    multiply, clamp at the platform bound, cast to the stored dtype, then the
    compensation the dequantisation applies to the block scale. Returns the stored
    bytes and the restored fp32 values.
    """
    stored = (
        (magnitudes.to(torch.float32) * down)
        .clamp(-FP8_PLATFORM_CLAMP, FP8_PLATFORM_CLAMP)
        .to(_fp8_module._FP8_DTYPE)
    )
    return stored, stored.to(torch.float32) * up


def _fp8_full_range_tile(variant: int) -> torch.Tensor:
    """One ``[128,128]`` fp32 tile holding every representable e4m3fn magnitude.

    Built from the enumerated grid rather than from a ramp: the 126 positive
    magnitudes, their negatives and zero are 253 values, repeated to fill 16384
    elements, which is 64 whole cycles plus 192, so coverage does not depend on where
    the truncation falls. A ramp reaches only 88 of the 126 and misses the whole low
    end, subnormals included, which is where the squeeze is worst per element.

    Every tile's absolute maximum is 448, which is what a per-block quantiser
    produces, and 256.0 is present because it is the byte with the worst absolute
    re-quantisation error. The four variants are rearrangements that preserve both.
    """
    rows, cols = FP8_BLOCK_SIZE
    positive = _fp8_representable_magnitudes()
    signed = torch.cat([-positive.flip(0), torch.zeros(1), positive])
    repeats = (rows * cols) // int(signed.numel()) + 1
    grid = signed.repeat(repeats)[: rows * cols].reshape(rows, cols)
    if variant == 0:
        return grid
    if variant == 1:
        return torch.flip(grid, dims=(1,))
    if variant == 2:
        return grid.t().contiguous()
    return -grid


def _fp8_synthetic_weight() -> torch.Tensor:
    """The synthetic `[256,256]` fp8 weight: four full-range tiles."""
    rows, cols = FP8_WEIGHT_SHAPE
    block_rows, block_cols = FP8_BLOCK_SIZE
    dense = torch.zeros(rows, cols, dtype=torch.float32)
    for variant, (grid_row, grid_col) in enumerate(
        ((0, 0), (0, 1), (1, 0), (1, 1))
    ):
        dense[
            grid_row * block_rows : (grid_row + 1) * block_rows,
            grid_col * block_cols : (grid_col + 1) * block_cols,
        ] = _fp8_full_range_tile(variant)
    return dense.to(torch.float8_e4m3fn)


def _fp8_scale_grid(scales=FP8_BLOCK_SCALES) -> torch.Tensor:
    """The `[2,2]` fp32 `weight_scale_inv` grid."""
    return torch.tensor(scales, dtype=torch.float32)


def _fp8_as_stored(value: float) -> torch.Tensor:
    """A Python float as the fp32 scale grid stores it.

    The grid is fp32 while ``MINVAL`` is a Python float, so the stored floor is
    ``float32(1e-5)`` and not the binary64 literal. Returning a tensor keeps the
    comparison exact in the dtype that holds the number.
    """
    return torch.tensor(value, dtype=torch.float32)


def _fp8_tiles(weight_shape=FP8_WEIGHT_SHAPE) -> tuple[tuple[int, int], ...]:
    grid_rows, grid_cols = block_grid_shape(weight_shape, FP8_BLOCK_SIZE)
    return tuple(
        (row, col) for row in range(grid_rows) for col in range(grid_cols)
    )


@pytest.fixture
def fp8_downscale_weight() -> torch.Tensor:
    return _fp8_synthetic_weight()


@pytest.fixture
def fp8_downscale_scales() -> torch.Tensor:
    return _fp8_scale_grid()


# --------------------------------------------------------------------------- #
# The gate the squeeze runs behind.
# --------------------------------------------------------------------------- #


def test_the_squeeze_is_gated_on_the_resolved_platform_clamp() -> None:
    """The squeeze applies only where the resolved platform clamp is 240.0.

    On a 448-max platform an unconditional rescale would corrupt correct weights.
    This states the condition the tests below are read under: the suite pins the
    platform to trn2 before collection, so the gate is engaged here. A 448-max
    platform is not reachable from this process, so the other arm is not tested.
    """
    clamp = resolved_fp8_clamp_max()
    assert clamp == FP8_PLATFORM_CLAMP, (
        f"resolved clamp is {clamp}, not {FP8_PLATFORM_CLAMP}; this run did not "
        "pin the trn2 platform"
    )
    assert needs_240_downscale() is True


# --------------------------------------------------------------------------- #
# The synthetic weight holds the properties the readings below rest on.
# --------------------------------------------------------------------------- #


def test_the_synthetic_weight_is_full_range_and_block_shaped(
    fp8_downscale_weight, fp8_downscale_scales
) -> None:
    """The shapes, and the three properties the readings below rest on.

    Every tile's absolute maximum is 448, the top of the format, so the bytes before
    the squeeze are as far outside the 240 clamp as the format allows. Not all of them
    are inside that clamp already, or a fraction of 1.0 after the squeeze would say
    nothing. And every one of the 126 representable magnitudes is present, the seven
    subnormals among them, because the smallest subnormal is where the squeeze is
    worst per element.
    """
    assert tuple(fp8_downscale_weight.shape) == FP8_WEIGHT_SHAPE
    assert fp8_downscale_weight.dtype is torch.float8_e4m3fn
    assert block_grid_shape(FP8_WEIGHT_SHAPE, FP8_BLOCK_SIZE) == (2, 2)
    assert tuple(fp8_downscale_scales.shape) == (2, 2)
    assert fp8_downscale_scales.numel() == 4
    assert fp8_downscale_scales.dtype is torch.float32

    dense = fp8_downscale_weight.to(torch.float32)
    block_rows, block_cols = FP8_BLOCK_SIZE
    for grid_row, grid_col in _fp8_tiles():
        tile = dense[
            grid_row * block_rows : (grid_row + 1) * block_rows,
            grid_col * block_cols : (grid_col + 1) * block_cols,
        ]
        tile_max = float(tile.abs().max().item())
        assert tile_max == FP8_OCP_MAX, (
            f"tile {(grid_row, grid_col)} tops out at {tile_max}, not "
            f"{FP8_OCP_MAX}; the bytes before the squeeze must reach above the "
            f"{FP8_PLATFORM_CLAMP} clamp, and {FP8_OCP_MAX} is the furthest the "
            f"format allows"
        )

    fraction_within = float((dense.abs() <= FP8_PLATFORM_CLAMP).to(torch.float32).mean())
    assert fraction_within < 1.0, (
        "every byte is already inside the clamp before the squeeze, so a fraction of "
        "1.0 afterwards would hold without the squeeze doing anything"
    )
    # Every scale is above the floor, so no reading below is the floor's doing.
    assert bool((fp8_downscale_scales >= MINVAL).all())

    # Coverage is compared against the grid the tile is built from rather than
    # against a literal count, so a torch build with a different grid moves both
    # sides together.
    representable = _fp8_representable_magnitudes()
    present = torch.unique(dense.abs())
    present = present[present > 0]
    missing = representable[~torch.isin(representable, present)]
    subnormals = representable[representable < 2.0 ** -6]
    subnormals_present = int(torch.isin(subnormals, present).sum())
    assert missing.numel() == 0, (
        f"the weight misses {missing.numel()} of {representable.numel()} "
        f"representable magnitudes, smallest missing {float(missing.min())}; the "
        f"per-element error would then be measured over a subset of the format"
    )
    assert subnormals_present == subnormals.numel(), (
        f"only {subnormals_present} of {subnormals.numel()} subnormals are present; "
        f"the smallest subnormal is where the squeeze is worst per element, since "
        f"halving it lands on a tie and round-to-nearest-even sends it to zero"
    )


# --------------------------------------------------------------------------- #
# The dequantised product survives the squeeze.
# --------------------------------------------------------------------------- #


def test_dequantisation_agrees_per_block_across_the_squeeze(
    fp8_downscale_weight, fp8_downscale_scales
) -> None:
    """All four blocks agree within rtol 3e-2 / atol 1e-5, read block by block.

    ``rtol`` and ``atol`` are passed explicitly because the fork's tolerance map has
    no fp8 entry and would fall back to the bf16 pair.
    """
    assert (FP8_RTOL, FP8_ATOL) == (3e-2, 1e-5)

    before = dequantise_blockwise(
        fp8_downscale_weight, fp8_downscale_scales, FP8_BLOCK_SIZE
    )
    squeeze = squeeze_blockwise_fp8(
        fp8_downscale_weight, fp8_downscale_scales, FP8_BLOCK_SIZE
    )
    assert squeeze.applied is True
    after = dequantise_blockwise(squeeze.weight, squeeze.scale_inv, FP8_BLOCK_SIZE)

    reports = block_agreement(
        before, after, block_size=FP8_BLOCK_SIZE, rtol=FP8_RTOL, atol=FP8_ATOL
    )
    assert len(reports) == 4, "the [256,256]/[128,128] grid is 4 blocks"

    failed = [report for report in reports if not report.within]
    assert not failed, "blocks outside rtol 3e-2 / atol 1e-5: " + "; ".join(
        f"{r.index} normalised={r.normalised_diff:.6f} "
        f"max_abs_diff={r.max_abs_diff:.6e} tolerance={r.tolerance:.6e}"
        for r in failed
    )

    # The residual is named exactly, over the stored bytes before the block scale
    # multiplies in: after the scale the difference is between two fp32 products that
    # round independently, so no exact comparison is available there. Before it, every
    # value is an fp8 magnitude times a power of two, and the squeeze is exact except
    # on the odd multiples of the minimum subnormal, where it is off by exactly one.
    restored_bytes = squeeze.weight.to(torch.float32) * FP8_COMPENSATION
    element_error = (restored_bytes - fp8_downscale_weight.to(torch.float32)).abs()
    residuals = torch.unique(element_error)
    nonzero_residuals = residuals[residuals > 0]
    assert nonzero_residuals.numel() == 1, (
        f"the squeeze produced {nonzero_residuals.numel()} distinct non-zero element "
        f"residuals, not one: {[float(v) for v in nonzero_residuals]}. At an exact "
        f"x 1/2 every inexact magnitude misses by exactly one minimum subnormal, so "
        f"more than one value means the factor is not a power of two"
    )
    assert float(nonzero_residuals[0]) == FP8_MIN_SUBNORMAL, (
        f"the single residual is {float(nonzero_residuals[0])}, not the minimum "
        f"subnormal {FP8_MIN_SUBNORMAL}"
    )

    # The same reading at the range ratio must not be one value: 240/448 re-rounds
    # mantissas rather than shifting exponents, so it produces many distinct
    # residuals. The helper is the one asserted byte-identical to
    # ``downscale_fp8_weight_bytes`` below, so this is not a second implementation.
    _, ratio_restored = _squeeze_and_restore(
        fp8_downscale_weight.to(torch.float32),
        FP8_RANGE_RATIO,
        1.0 / FP8_RANGE_RATIO,
    )
    ratio_error = (
        ratio_restored - fp8_downscale_weight.to(torch.float32)
    ).abs()
    ratio_residuals = torch.unique(ratio_error)
    ratio_nonzero = ratio_residuals[ratio_residuals > 0]
    assert ratio_nonzero.numel() > 1, (
        f"at 240/448 the squeeze produced {ratio_nonzero.numel()} distinct non-zero "
        f"residuals; the reading above cannot tell an exact power of two from a range "
        f"ratio unless this arm reads many"
    )


def test_every_stored_byte_is_inside_the_platform_clamp(
    fp8_downscale_weight, fp8_downscale_scales
) -> None:
    """Every stored byte is inside the clamp, read as a fraction rather than all()."""
    squeeze = squeeze_blockwise_fp8(
        fp8_downscale_weight, fp8_downscale_scales, FP8_BLOCK_SIZE
    )
    assert squeeze.weight.dtype is torch.float8_e4m3fn
    assert tuple(squeeze.weight.shape) == FP8_WEIGHT_SHAPE
    assert squeeze.fraction_within_240 == 1.0
    assert squeeze.max_abs_stored <= FP8_PLATFORM_CLAMP

    # The stored maximum is exactly 448 times the factor, so a squeeze that shrank
    # the bytes further still fails here. At an exact 1/2 that is 224, which leaves 16
    # counts of headroom under the clamp rather than reaching it.
    assert squeeze.max_abs_stored == FP8_OCP_MAX * FP8_DOWNSCALE
    assert squeeze.max_abs_stored < FP8_PLATFORM_CLAMP


def test_no_block_scale_falls_below_minval(
    fp8_downscale_weight, fp8_downscale_scales
) -> None:
    """No scale falls below ``MINVAL``, counted before and after the squeeze.

    The count after the squeeze alone would be guaranteed by the floor, so the count
    before it is what makes this a reading: no scale of this grid was near the floor.
    The test below is the case where the floor does engage.
    """
    assert MINVAL == 1e-5
    squeeze = squeeze_blockwise_fp8(
        fp8_downscale_weight, fp8_downscale_scales, FP8_BLOCK_SIZE
    )
    assert squeeze.below_minval_before == 0
    assert squeeze.below_minval_after == 0
    assert squeeze.floored_blocks == ()

    # The compensation is the exact inverse of the byte squeeze and applies to every
    # tile, not only to the tiles that needed clamping. Both halves are powers of two,
    # so the round trip is exact in fp32 and the equality needs no tolerance.
    expected = fp8_downscale_scales * FP8_COMPENSATION
    assert torch.equal(squeeze.scale_inv, expected)


# --------------------------------------------------------------------------- #
# The MINVAL floor, where it does engage.
# --------------------------------------------------------------------------- #


def test_the_minval_floor_engages_on_a_tiny_block_scale(
    fp8_downscale_weight,
) -> None:
    """One tiny block scale engages the floor, on that tile alone.

    The stored scale becomes exactly ``MINVAL`` and the tile is named in the report,
    the other three tiles keep their plain compensated scale, and the block-agreement
    reading turns False on the floored tile only -- the same check and the same
    tolerances as the test above, with a different answer.
    """
    scales = [list(row) for row in FP8_BLOCK_SCALES]
    scales[FP8_TINY_BLOCK[0]][FP8_TINY_BLOCK[1]] = FP8_TINY_SCALE
    tiny_grid = _fp8_scale_grid(tuple(tuple(row) for row in scales))

    before = dequantise_blockwise(fp8_downscale_weight, tiny_grid, FP8_BLOCK_SIZE)
    squeeze = squeeze_blockwise_fp8(fp8_downscale_weight, tiny_grid, FP8_BLOCK_SIZE)

    # The floor engaged on exactly the one block that needed it. The stored value is
    # compared in the grid's own dtype, so the equality is exact.
    assert squeeze.below_minval_before == 1
    assert squeeze.floored_blocks == (FP8_TINY_BLOCK,)
    assert squeeze.below_minval_after == 0
    assert torch.equal(squeeze.scale_inv[FP8_TINY_BLOCK], _fp8_as_stored(MINVAL))
    # The floor is what put it there: the compensated value it replaced is four
    # orders of magnitude below the floor at the factor the load path applies.
    assert FP8_TINY_SCALE * FP8_COMPENSATION < MINVAL

    # Every other tile carries the plain compensated scale, compared tile by tile
    # against the grid the transform started from.
    compensated = tiny_grid * FP8_COMPENSATION
    for grid_row, grid_col in _fp8_tiles():
        if (grid_row, grid_col) == FP8_TINY_BLOCK:
            continue
        assert torch.equal(
            squeeze.scale_inv[grid_row, grid_col], compensated[grid_row, grid_col]
        )

    # The block-agreement reading moves, and only on the floored tile.
    after = dequantise_blockwise(squeeze.weight, squeeze.scale_inv, FP8_BLOCK_SIZE)
    by_index = {
        report.index: report
        for report in block_agreement(
            before, after, block_size=FP8_BLOCK_SIZE, rtol=FP8_RTOL, atol=FP8_ATOL
        )
    }
    assert by_index[FP8_TINY_BLOCK].within is False, (
        "the floored block's dequantisation must not agree: the floor raised its "
        "scale by four orders of magnitude"
    )
    assert all(
        by_index[tile].within for tile in _fp8_tiles() if tile != FP8_TINY_BLOCK
    )


# --------------------------------------------------------------------------- #
# The same arithmetic through the loaders, on fake checkpoint slices.
# --------------------------------------------------------------------------- #


class _Fp8RecordingHandler(logging.Handler):
    """Collects formatted records off the loader module's own logger object."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _capture_fp8_loader_log():
    """Attach a handler to ``weight_loaders_fp8.logger`` itself.

    Not through ``caplog``: its handler lives on the root logger, so any propagation
    setting in vLLM's logging configuration would make this read zero records for a
    reason unrelated to the code under test.
    """
    handler = _Fp8RecordingHandler()
    target = _fp8_module.logger
    previous_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)


FP8_FLOOR_MARKER = "fp8 block-scale floor ENGAGED"
FP8_LOADER_PARAM_NAME = "model.layers.0.mlp.down_proj.weight_scale_inv"


def test_the_scale_loader_logs_an_engaged_floor(
    fp8_downscale_scales,
) -> None:
    """A load that floors a tile leaves a log record on the load path.

    Read as a difference rather than as the presence of one message: the same loader
    runs twice over the same slice, once on a grid where no tile needs the floor and
    once with one tile's scale set below it, so a loader that logged on every load
    would fail the first half. The record's text is checked too, because a record that
    names neither the parameter nor the tile sends a reader nowhere.
    """
    # The grid where no tile needs the floor must produce no record.
    quiet_loader = blockwise_scale_loader(FP8_LOADER_PARAM_NAME)
    with _capture_fp8_loader_log() as quiet:
        quiet_grid = quiet_loader.load([FakeSafeSlice(fp8_downscale_scales)], 0)
    quiet_records = [m for m in quiet.messages if FP8_FLOOR_MARKER in m]
    assert quiet_records == [], (
        "the loader reported a floor on a grid where no tile needed one, so the "
        "record does not distinguish a floored load from a clean one"
    )

    # One tile below the floor, through the same loader.
    scales = [list(row) for row in FP8_BLOCK_SCALES]
    scales[FP8_TINY_BLOCK[0]][FP8_TINY_BLOCK[1]] = FP8_TINY_SCALE
    tiny_grid = _fp8_scale_grid(tuple(tuple(row) for row in scales))

    loud_loader = blockwise_scale_loader(FP8_LOADER_PARAM_NAME)
    with _capture_fp8_loader_log() as loud:
        loaded = loud_loader.load([FakeSafeSlice(tiny_grid)], 0)
    loud_records = [m for m in loud.messages if FP8_FLOOR_MARKER in m]
    assert len(loud_records) == 1, (
        f"a load that floored tile {FP8_TINY_BLOCK} must leave exactly one record; "
        f"got {len(loud_records)}"
    )

    # The record names the parameter and the floored tile.
    message = loud_records[0]
    assert FP8_LOADER_PARAM_NAME in message, message
    assert f"({FP8_TINY_BLOCK[0]},{FP8_TINY_BLOCK[1]})" in message, message
    assert "MINVAL" in message, message

    # The logging changes no returned value.
    assert torch.equal(loaded, compensate_block_scales(tiny_grid).scale_inv)
    assert torch.equal(loaded[FP8_TINY_BLOCK], _fp8_as_stored(MINVAL))
    assert torch.equal(quiet_grid, compensate_block_scales(fp8_downscale_scales).scale_inv)

    # A loader built without a parameter name says so, rather than logging "None".
    with _capture_fp8_loader_log() as unnamed:
        blockwise_scale_loader().load([FakeSafeSlice(tiny_grid)], 0)
    unnamed_records = [m for m in unnamed.messages if FP8_FLOOR_MARKER in m]
    assert len(unnamed_records) == 1
    assert "an unnamed parameter" in unnamed_records[0], unnamed_records[0]
    assert "None" not in unnamed_records[0], unnamed_records[0]

    # The reporter's return value carries the same fact without the log line.
    assert report_floored_blocks(compensate_block_scales(tiny_grid)) is True
    assert report_floored_blocks(compensate_block_scales(fp8_downscale_scales)) is False


def test_the_scale_loader_compensates_through_a_fake_slice(
    fp8_downscale_scales,
) -> None:
    """The `weight_scale_inv` loader returns the compensated, floored grid."""
    loader = blockwise_scale_loader()
    loaded = loader.load([FakeSafeSlice(fp8_downscale_scales)], 0)

    assert loaded.dtype is torch.float32
    assert tuple(loaded.shape) == (2, 2)
    assert torch.equal(loaded, compensate_block_scales(fp8_downscale_scales).scale_inv)
    # The second reading names the factor instead of deriving it from the module, so a
    # change to the module fails here rather than following along.
    assert torch.equal(loaded, fp8_downscale_scales * FP8_COMPENSATION)


def test_the_weight_loader_squeezes_through_a_fake_slice(
    fp8_downscale_weight,
) -> None:
    """The wrapped weight loader stores bytes inside the platform clamp.

    It wraps the plain loader here, which is the shape the model file wraps a sharding
    loader in: the wrapper composes with the transform it is given rather than
    replacing it.
    """
    wrapped = wrap_with_blockwise_fp8_downscale(SafetensorsWeightLoader())
    loaded = wrapped.load([FakeSafeSlice(fp8_downscale_weight)], 0)

    assert loaded.dtype is torch.float8_e4m3fn
    assert tuple(loaded.shape) == FP8_WEIGHT_SHAPE
    dense = loaded.to(torch.float32)
    # The squeezed maximum is 448 times the factor, so 224, which is asserted against
    # the derived value and separately against the clamp: the squeeze leaves headroom
    # under the clamp rather than reaching it.
    assert float(dense.abs().max().item()) == FP8_OCP_MAX * FP8_DOWNSCALE
    assert float(dense.abs().max().item()) < FP8_PLATFORM_CLAMP
    assert torch.equal(
        dense, downscale_fp8_weight_bytes(fp8_downscale_weight).to(torch.float32)
    )


# --------------------------------------------------------------------------- #
# The quantization spec the block geometry comes from.
# --------------------------------------------------------------------------- #


def test_the_quantization_spec_parses_the_block_config() -> None:
    """``QuantizationSpec`` resolves the blockwise scheme and its block shape.

    Blockwise rather than per-tensor: the scheme carries a 2-D block shape, and that
    shape is what the loaders index the scale grid with, so a config that omits it must
    raise here rather than surface as a shape mismatch inside a transform.
    """
    spec = QuantizationSpec.from_hf_quantization_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
    )
    assert spec is not None
    assert spec.linear_scheme is QuantScheme.FP8_BLOCK_DYNAMIC
    assert spec.kv_cache_scheme is QuantScheme.NONE
    assert spec.weight_block_size == FP8_BLOCK_SIZE == DEFAULT_WEIGHT_BLOCK_SIZE
    assert spec.activation_scheme == "dynamic"
    assert spec.is_block_quantized
    # A spec parsed from a block config with no skip list has an empty one, and both
    # call shapes -- with and without a layer index -- then get the linear scheme. What
    # the real checkpoint's skip list does to those answers is measured further down.
    assert spec.modules_to_not_convert == ()
    assert spec.get_scheme(0, "model.layers.0.self_attn.o_proj") is spec.linear_scheme
    assert spec.get_scheme(None, "lm_head") is spec.linear_scheme

    # An unquantized checkpoint gives None rather than a spec with no scheme.
    assert QuantizationSpec.from_hf_quantization_config(None) is None
    assert QuantizationSpec.from_hf_quantization_config({}) is None

    # The same fields reached through the model config give the same spec.
    from_config = QuantizationSpec.from_model_config(Glm5NextConfig())
    assert from_config == spec

    with pytest.raises(ValueError, match="quant_method"):
        QuantizationSpec.from_hf_quantization_config({"quant_method": "awq"})
    with pytest.raises(ValueError, match="activation_scheme"):
        QuantizationSpec.from_hf_quantization_config(
            {"quant_method": "fp8", "activation_scheme": "static"}
        )
    with pytest.raises(ValueError, match="weight_block_size"):
        QuantizationSpec.from_hf_quantization_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128],
            }
        )


# =========================================================================== #
# The checkpoint's FP8 skip list
# =========================================================================== #
#
# The published config names 1,509 modules to keep in BF16, and the checkpoint
# carries no `weight_scale_inv` companion for any of them. The tests below compare
# the config's own skip list against the index's own scale keys; neither side is
# computed from the other, and both fixtures are pinned by digest above.
#
# The skip entries are module names (`model.layers.0.self_attn.q_proj`) while the
# index keys are checkpoint names (`model.language_model.layers.0...`). The
# substring rule resolves one against the other only because `language_model` ends
# in `model`, so the tests spell both namespaces out rather than leaving the match
# implicit.

# --------------------------------------------------------------------------- #
# Counts read off the published config and index, each cross-checked below
# against a number derived from the fixture.
# --------------------------------------------------------------------------- #

#: The five ``quantization_config`` fields the published config carries, the number
#: of skip-list entries, and the fp8 format it names.
NUM_QUANT_CONFIG_FIELDS = 5
NUM_SKIP_ENTRIES = 1509
FP8_FMT = "e4m3"

#: The in-scope base tensors, split into those with a scale companion and those
#: the skip list keeps in BF16.
NUM_BASE_TENSORS = 37534
NUM_QUANTIZED_TENSORS = 36467
NUM_BF16_TENSORS = 1067

#: How the checkpoint spells a scale companion: the base key plus this tail.
#: Appending a tail rather than substituting a leaf keeps the rule total over every
#: base tensor -- a parameter that is not a ``.weight``, such as ``A_log`` or a
#: hyper-connection tensor, gets a name the index cannot contain, which is right,
#: since it has no scale companion.
SCALE_COMPANION_TAIL = "_scale_inv"

#: A token the published skip list does not carry, used to show the suppression
#: moving; the number of scale requests it removes is derived from the config.
SYNTHETIC_SKIP_TOKEN = "shared_experts"
SYNTHETIC_SKIP_DROP = 126


@pytest.fixture(scope="module")
def real_raw_config() -> dict[str, Any]:
    """The published config, parsed."""
    return json.loads(REAL_CONFIG_PATH.read_text())


@pytest.fixture(scope="module")
def real_quant_config(real_raw_config) -> dict[str, Any]:
    """The published `quantization_config` block, verbatim."""
    return real_raw_config["quantization_config"]


@pytest.fixture(scope="module")
def real_config() -> Glm5NextConfig:
    """The config the adapter produces from the published config.

    ``Glm5NextConfig.from_configs`` on a fresh parse, so the adapter is what is
    measured and nothing it touches can reach the expectation side of an assertion.
    """
    return Glm5NextConfig.from_configs(json.loads(REAL_CONFIG_PATH.read_text()))


@pytest.fixture(scope="module")
def real_spec(real_config) -> QuantizationSpec:
    """The spec built through the whole chain: file to config to spec.

    Driving ``from_model_config`` means a bridge that forwarded the skip list's field
    only partway fails here rather than passing with an empty skip list.
    """
    spec = QuantizationSpec.from_model_config(real_config)
    assert spec is not None
    return spec


@pytest.fixture(scope="module")
def real_base_tensors(real_in_scope) -> list[str]:
    """The in-scope base tensors: every in-scope key that is not a scale key."""
    return sorted(
        key for key in real_in_scope if not key.endswith(f".{SCALE_SUFFIX}")
    )


def _layer_index_of(key: str) -> int | None:
    """The layer number inside a key, or `None` for a key outside any block."""
    parts = key.split(".")
    if "layers" in parts:
        position = parts.index("layers") + 1
        if position < len(parts) and parts[position].isdigit():
            return int(parts[position])
    return None


def _scale_companion_of(key: str) -> str:
    """The scale key the checkpoint would carry for `key`, spelled its way."""
    return key + SCALE_COMPANION_TAIL


def _score_spec_against_index(
    spec: QuantizationSpec, base_keys: list[str], index_keys: frozenset[str]
) -> dict[str, Any]:
    """Score one spec against the index over every base tensor.

    One pass, four counters: where the two answers agree and disagree, and how the
    population splits into quantized and unquantized tensors.
    """
    agreements = disagreements = quantized = unquantized = 0
    examples: list[str] = []
    for key in base_keys:
        has_scale = _scale_companion_of(key) in index_keys
        unquantised = spec.get_scheme(_layer_index_of(key), key) is QuantScheme.NONE
        if unquantised == (not has_scale):
            agreements += 1
        else:
            disagreements += 1
            if len(examples) < 5:
                examples.append(key)
        quantized += has_scale
        unquantized += not has_scale
    return {
        "agreements": agreements,
        "disagreements": disagreements,
        "quantized": quantized,
        "unquantized": unquantized,
        "examples": examples,
    }


def _requested_scale_keys(mappings: dict[str, str | list[str]]) -> list[str]:
    """Every `weight_scale_inv` key the mapping asks the checkpoint for."""
    return sorted(
        {
            key
            for value in mappings.values()
            for key in (value if isinstance(value, list) else [value])
            if key.endswith(f".{SCALE_SUFFIX}")
        }
    )


def _base_tensor_of(scale_key: str) -> str:
    """The base tensor a requested scale key belongs to."""
    assert scale_key.endswith(SCALE_COMPANION_TAIL)
    return scale_key[: -len(SCALE_COMPANION_TAIL)]


def test_config_lifts_the_skip_list_and_the_fp8_format(
    real_quant_config, real_config
) -> None:
    """The adapter lifts all five ``quantization_config`` fields.

    Every expectation is read off the published config rather than typed in: the entry
    count is ``len()`` of the fixture's own list and the format string is the fixture's
    own value. The constants above are asserted beside them, which is sound only
    because the fixture is pinned by the published file's digest.
    """
    assert sorted(real_quant_config) == [
        "activation_scheme",
        "fmt",
        "modules_to_not_convert",
        "quant_method",
        "weight_block_size",
    ]
    assert len(real_quant_config) == NUM_QUANT_CONFIG_FIELDS

    declared_skip = real_quant_config["modules_to_not_convert"]
    assert len(declared_skip) == NUM_SKIP_ENTRIES

    # The lift loses no entry and reorders none.
    assert real_config.modules_to_not_convert == declared_skip
    assert real_config.fmt == real_quant_config["fmt"] == FP8_FMT

    # The three fields that were already lifted still are.
    assert real_config.quant_method == real_quant_config["quant_method"]
    assert real_config.activation_scheme == real_quant_config["activation_scheme"]
    assert real_config.weight_block_size == real_quant_config["weight_block_size"]

    # The list's composition, because a bare module name cannot match a qualified
    # entry: nine entries are bare tokens and the other 1,500 are dotted paths.
    bare = sorted(entry for entry in declared_skip if "." not in entry)
    assert len(bare) == 9
    assert "lm_head" in bare
    assert len(declared_skip) - len(bare) == 1500


def test_skip_list_agrees_with_the_index_scale_keys(
    real_spec, real_base_tensors, real_in_scope, real_quant_config
) -> None:
    """The config's skip list and the index's scale keys agree on all 37,534 tensors.

    ``get_scheme`` returns the unquantized scheme for a base tensor exactly when that
    tensor has no scale companion in the index. The left side is the 1,509-entry list
    in the config, the right side a key census over the 76,108-key index, and neither
    is derived from the other. The same census against a spec built without the skip
    list disagrees 1,067 times, one per BF16 tensor.
    """
    # The two spellings of the one suffix, pinned together so neither can drift.
    assert _scale_companion_of("m.weight") == f"m.{SCALE_SUFFIX}"

    index_keys = frozenset(real_in_scope)
    scored = _score_spec_against_index(real_spec, real_base_tensors, index_keys)

    assert scored["disagreements"] == 0, scored["examples"]
    assert scored["agreements"] == len(real_base_tensors)
    assert len(real_base_tensors) == NUM_BASE_TENSORS
    assert scored["quantized"] == NUM_QUANTIZED_TENSORS
    assert scored["unquantized"] == NUM_BF16_TENSORS
    assert scored["quantized"] + scored["unquantized"] == len(real_base_tensors)
    assert len(real_spec.modules_to_not_convert) == NUM_SKIP_ENTRIES

    blind_spec = QuantizationSpec.from_hf_quantization_config(
        {
            key: value
            for key, value in real_quant_config.items()
            if key != "modules_to_not_convert"
        }
    )
    assert blind_spec is not None
    assert blind_spec.modules_to_not_convert == ()
    blind = _score_spec_against_index(blind_spec, real_base_tensors, index_keys)
    assert blind["disagreements"] == NUM_BF16_TENSORS > 0
    assert blind["agreements"] == NUM_QUANTIZED_TENSORS


def test_named_projections_resolve_to_the_expected_scheme(
    real_spec, real_text_config, real_in_scope
) -> None:
    """Two projections the checkpoint keeps in BF16, and one it quantizes.

    The third case is what makes this a discrimination: a rule that answered
    "unquantized" for every name would fail on ``q_a_proj``. Each case is also checked
    against the index, so the expected answer comes from the checkpoint.

    A bare name such as ``layers.0.self_attn.q_proj`` matches none of the 1,500
    qualified entries, since those are dotted paths under ``model.``, so every case is
    asserted in both qualified namespaces and in the bare spelling.
    """
    cases = (
        ("layers.0.self_attn.q_proj", QuantScheme.NONE, KDA_LAYER_TYPE),
        ("layers.3.self_attn.kv_b_proj", QuantScheme.NONE, DSA_LAYER_TYPE),
        (
            "layers.3.self_attn.q_a_proj",
            QuantScheme.FP8_BLOCK_DYNAMIC,
            DSA_LAYER_TYPE,
        ),
    )
    rows = []
    for short, expected, family in cases:
        layer = int(short.split(".")[1])
        assert real_text_config.layer_types[layer] == family

        module_ns = f"model.{short}"
        checkpoint_ns = f"model.language_model.{short}"
        for name in (module_ns, checkpoint_ns, f"{checkpoint_ns}.weight"):
            got = real_spec.get_scheme(layer, name)
            assert got is expected, f"{name}: {got.name} != {expected.name}"

        # The index's own answer for the same projection.
        has_scale = _scale_companion_of(f"{checkpoint_ns}.weight") in real_in_scope
        assert has_scale == (expected is QuantScheme.FP8_BLOCK_DYNAMIC)

        # A bare name matches no qualified entry, so it reads as quantized.
        assert real_spec.get_scheme(layer, short) is real_spec.linear_scheme

        rows.append(
            {
                "case": short,
                "layer_type": family,
                "scheme": expected.value,
                "index_has_scale_key": has_scale,
            }
        )

    assert len(rows) == 3
    assert len({row["scheme"] for row in rows}) == 2


def test_no_scale_companion_is_requested_for_a_bf16_tensor(
    real_text_config, real_quant_config, real_in_scope
) -> None:
    """The mapping requests no scale key for a BF16 tensor, and none the index lacks.

    Two censuses over the map the published skip list produces: requested scale keys
    whose base tensor the skip list keeps in BF16, and requested scale keys the index
    does not contain. Both are empty, and the 36,467 requests that remain are exactly
    the index's own scale-key population.

    The suppression is shown with ``shared_experts``, a token the published list does
    not carry, and the number of requests it removes is derived from the config: one
    per shared-expert leaf on each MoE layer. Switching the published list off changes
    no request, because the builder already maps no scale for those families.
    """
    skip = tuple(real_quant_config["modules_to_not_convert"])
    honoured = build_weight_mappings(real_text_config, modules_to_not_convert=skip)
    requested = _requested_scale_keys(honoured)

    kept_bf16 = [key for key in requested if keeps_bf16(_base_tensor_of(key), skip)]
    assert kept_bf16 == []
    # The same question in the other spelling, the module name without the parameter
    # leaf, so the two forms cannot disagree unnoticed.
    assert [
        key
        for key in requested
        if keeps_bf16(_base_tensor_of(key).removesuffix(".weight"), skip)
    ] == []

    absent = [key for key in requested if key not in real_in_scope]
    assert absent == []

    index_scales = {
        key for key in real_in_scope if key.endswith(f".{SCALE_SUFFIX}")
    }
    assert len(requested) == len(index_scales) == NUM_QUANTIZED_TENSORS

    probe = skip + (SYNTHETIC_SKIP_TOKEN,)
    moe_layers = (
        real_text_config.num_hidden_layers - real_text_config.first_k_dense_replace
    )
    shared_leaves = 3
    unsuppressed = _requested_scale_keys(
        build_weight_mappings(real_text_config, modules_to_not_convert=())
    )
    fires = [key for key in unsuppressed if keeps_bf16(_base_tensor_of(key), probe)]
    assert len(fires) == moe_layers * shared_leaves
    assert len(fires) == SYNTHETIC_SKIP_DROP > 0

    suppressed = _requested_scale_keys(
        build_weight_mappings(real_text_config, modules_to_not_convert=probe)
    )
    assert [key for key in suppressed if keeps_bf16(_base_tensor_of(key), probe)] == []
    assert len(suppressed) == len(requested) - len(fires)


# =========================================================================== #
# Why the squeeze factor is an exact power of two
# =========================================================================== #
#
# The load path squeezes fp8 bytes by 1/2 and multiplies the block scale by 2
# rather than using the range ratio 240/448, because `value -> squeeze -> fp8 ->
# compensate` is bit-exact for 118 of the 126 positive magnitudes at 1/2 and for
# only 14 at the ratio. The test below ties three things together: the counting
# helper produces the same bytes as `downscale_fp8_weight_bytes`, so it is not a
# second implementation; the same helper at the range ratio reads 14, so the count
# tells the two factors apart; and the platform gate is engaged, since otherwise
# the production function returns its input unchanged.

#: The magnitudes that survive the round trip at each factor, and the count that
#: does not at an exact power of two. Derived by exact-fraction emulation of
#: e4m3fn round-to-nearest-even, independently of torch.
FP8_EXACT_ROUND_TRIPS = 118
FP8_EXACT_ROUND_TRIPS_AT_RANGE_RATIO = 14
FP8_INEXACT_MAGNITUDES = 8

#: The magnitude below which every inexact value sits. They are the odd multiples
#: of ``FP8_MIN_SUBNORMAL``.
FP8_INEXACT_CEILING = 2.0**-5


def test_the_squeeze_factor_is_an_exact_power_of_two() -> None:
    """The factor is the largest power of two that fits, and 118 magnitudes survive."""
    assert needs_240_downscale() is True, (
        "the platform gate is disengaged, so downscale_fp8_weight_bytes returns its "
        "input unchanged and every count below would measure nothing. The clamp is "
        "resolved at import time, so the platform must be pinned in the process "
        "environment rather than by a fixture."
    )

    down = _fp8_module._FP8_WEIGHT_DOWNSCALE
    up = _fp8_module._FP8_SCALE_COMPENSATION

    # The module holds the pinned pair, and both halves are exact powers of two:
    # ``.hex()`` reads the bits rather than trusting a decimal literal.
    assert (down, up) == (FP8_DOWNSCALE, FP8_COMPENSATION)
    assert down.hex() == "0x1.0000000000000p-1"
    assert up.hex() == "0x1.0000000000000p+1"
    assert down * up == 1.0

    # It is the largest such factor: the next power of two up leaves the format's
    # maximum outside the bound the kernel reads.
    assert FP8_OCP_MAX * down <= FP8_PLATFORM_CLAMP
    assert FP8_OCP_MAX * (down * 2.0) > FP8_PLATFORM_CLAMP

    magnitudes = _fp8_representable_magnitudes()
    assert magnitudes.numel() == 126

    # The counting helper produces the production path's bytes. Compared as uint8,
    # which asks about the stored bytes and nothing else.
    stored, restored = _squeeze_and_restore(magnitudes, down, up)
    production = downscale_fp8_weight_bytes(magnitudes)
    assert torch.equal(production.view(torch.uint8), stored.view(torch.uint8))

    exact = restored == magnitudes.to(torch.float32)
    exact_count = int(exact.sum().item())
    inexact = magnitudes[~exact]

    # The eight that do not survive are the odd multiples of the minimum subnormal:
    # halving one lands exactly between two grid points, and round-to-nearest-even
    # resolves the tie away from it.
    odd_multiples = [
        float(value) / FP8_MIN_SUBNORMAL for value in inexact.tolist()
    ]
    smallest_restored = float(
        _squeeze_and_restore(
            torch.tensor([FP8_MIN_SUBNORMAL]), down, up
        )[1].item()
    )

    # The same helper at the range ratio must read 14, or the count above cannot tell
    # an exact power of two from a range ratio.
    _, ratio_restored = _squeeze_and_restore(
        magnitudes, FP8_RANGE_RATIO, 1.0 / FP8_RANGE_RATIO
    )
    ratio_exact_count = int((ratio_restored == magnitudes.to(torch.float32)).sum().item())

    zero_restored = float(
        _squeeze_and_restore(torch.tensor([0.0]), down, up)[1].item()
    )

    assert exact_count == FP8_EXACT_ROUND_TRIPS
    assert ratio_exact_count == FP8_EXACT_ROUND_TRIPS_AT_RANGE_RATIO
    assert exact_count > ratio_exact_count
    assert inexact.numel() == FP8_INEXACT_MAGNITUDES
    assert float(inexact.max().item()) < FP8_INEXACT_CEILING
    assert all(multiple % 2 == 1 for multiple in odd_multiples)
    assert smallest_restored == 0.0
    assert zero_restored == 0.0

    # The stored maximum is the derived product, so the headroom the factor leaves
    # under the clamp is asserted rather than assumed.
    assert float(stored.to(torch.float32).max().item()) == FP8_OCP_MAX * down
    assert FP8_OCP_MAX * down < FP8_PLATFORM_CLAMP

