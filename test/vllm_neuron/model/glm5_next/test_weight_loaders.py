# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ``inc-glm53f-011`` -- WP1: the weight-loader skeleton.

The declared acceptance (increment plan revision 10, L3194), verbatim:

    "against a synthetic 3-shard fake index built with ``hf_state_to_fake_slices``
    (``inc-glm53f-002``), the loader maps **100% of HF keys with 0 unmatched and
    0 duplicated**, and reports a per-shard count summing exactly to the
    fixture's key count."

Four conjuncts, measured below as C01-C04:

* **C01** -- 100% of the fixture's HF keys are mapped.
* **C02** -- 0 unmatched, both directions.
* **C03** -- 0 duplicated.
* **C04** -- the per-shard counts sum exactly to the fixture's key count.

Every test function in this module carries ``skeleton`` in its name. That is not
cosmetic: the declared invocation filters with ``-k skeleton``, and a ``-k``
expression that matches *some* tests exits 0 while silently shrinking the run --
a total miss exits 5 and is caught, a partial miss is not. The two collected-item
counts (filtered and unfiltered) are recorded in this increment's evidence record
so a dropped conjunct cannot hide. ``inc-glm53f-012`` later adds this file's
numerics partition, selected by its own ``-k`` expression; no *test name* here
contains that expression, so the two partitions stay disjoint. (``-k`` matches
item keywords -- module, class, function and marker names -- not docstrings, so
this paragraph naming it is inert.)

WHAT CERTIFIES WHAT -- the point of this file
---------------------------------------------
``hf_state_to_fake_slices`` already raises on two keys that qualify to the same
name (``test/vllm_neuron/model/utils.py:137-138``). A duplicate test that only
tripped *that* would certify the fixture builder and say nothing about the
loader. So the duplicate conjunct here is exercised on the **cross-shard** path:
one key placed in two shard files, which the helper structurally cannot see
because it is called per layer on a flat dict with no shard concept at all.
:func:`test_skeleton_duplicate_is_certified_by_the_loader_not_the_fixture` proves
all three legs -- the helper does not raise, the pin's own flattened
``{key: file}`` dict silently loses the duplicate, and the loader reports it.

THE 3-SHARD COMPOSITION IS AUTHORED HERE, NOT BY THE HELPER
-----------------------------------------------------------
``hf_state_to_fake_slices(state_dict, layer_idx)`` returns a flat
``{checkpoint_key: FakeSafeSlice}`` map. It has no notion of a shard. The fixture
is therefore "built **with**" the helper, not "built **by**" it: the helper
supplies the slice map, and :func:`_partition_into_shards` below is the 3-shard
composition, written here. C04 measures exactly that composition.

THE KEY UNIVERSE IS ENUMERATED INDEPENDENTLY OF THE MAPPING BUILDER
-------------------------------------------------------------------
If the fixture's checkpoint keys were generated from
``build_weight_mappings``, "100% mapped" would be a tautology -- one enumeration
agreeing with itself. :func:`_hf_keys_for_layer` is therefore a separate,
literal enumeration written in this file, in a deliberately different style
(spelled-out ``.weight`` / ``.weight_scale_inv`` suffixes, no reuse of any
private helper from the module under test). C01/C02 are the *agreement* of two
independently authored enumerations, so a typo on either side fails them --
which :func:`test_skeleton_unmatched_counters_can_report_nonzero` confirms by
perturbing each side in turn.

FIXTURE PROVENANCE, AND ITS LIMIT
---------------------------------
Key *names* are shaped by the recorded intake evidence for
``zai-org/GLM-5.3-Flash`` rev ``04c4e9e9``
(``artifacts/run/intake-preflight/03-glm53flash-weights.md``, sha256
``2d3c1912...``): 45 layers on a 3:1 KDA/DSA schedule, ``first_k_dense_replace``
3, 288 routed + 1 shared expert, blockwise FP8 ``[128, 128]``. The fixture is a
faithful **miniature**, not a sample: ``num_hidden_layers=4`` reproduces layers
0-3 of the real stack exactly (three ``linear_attention`` layers with dense MLPs,
then the first ``deepseek_sparse_attention`` layer with the first MoE MLP), and
the expert count is scaled to 4.

The limit, stated plainly: the checkpoint's own
``model.safetensors.index.json`` was **never downloaded** (intake deliberately
fetched no weights), so no leaf name here is verbatim-sourced. Families whose
leaf names are unconfirmed are tagged ``PROVISIONAL`` by the module under test,
and :func:`test_skeleton_key_families_are_all_tagged` holds that tagging to
being total. The 62 shards in the increment heading are title prose: no conjunct
pins them, the acceptance says 3, and
:func:`test_skeleton_index_is_three_shards_not_sixty_two` pins 3.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from test.vllm_neuron.model.utils import FakeSafeSlice, hf_state_to_fake_slices
from vllm_neuron.model.glm5_next.config import (
    DSA_LAYER_TYPE,
    KDA_LAYER_TYPE,
    Glm5NextTextConfig,
)
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    ABSENT_KEY_FAMILIES,
    GROUNDED,
    KEY_FAMILY_PROVENANCE,
    PROVISIONAL,
    DuplicateShardKeyError,
    Glm5NextShardIndex,
    build_weight_mappings,
    check_key_coverage,
    scale_keys,
)

# --------------------------------------------------------------------------- #
# Declared values.
# --------------------------------------------------------------------------- #

#: The acceptance says 3. The increment heading's "62 shards" is title prose and
#: is pinned by no conjunct -- see the module docstring.
DECLARED_NUM_SHARDS = 3

#: The miniature. Layers 0-3 of the real stack, experts scaled to 4.
MINI_LAYERS = 4
MINI_ROUTED_EXPERTS = 4
MINI_SHARED_EXPERTS = 1
FIRST_K_DENSE = 3

SCALE_SUFFIX = "weight_scale_inv"

_RESULTS_PATH = Path(
    os.environ.get("VLLM_NEURON_INC011_RESULTS_JSON")
    or Path(tempfile.gettempdir()) / "vllm_neuron_inc011_predicates.json"
)
_RESULTS: dict[str, Any] = {}
_RESULTS_PATH.write_text("{}\n")  # truncate stale values from an earlier run


def _record(**values: Any) -> None:
    _RESULTS.update(values)
    _RESULTS_PATH.write_text(
        json.dumps(_RESULTS, indent=2, sort_keys=True, default=str) + "\n"
    )


# --------------------------------------------------------------------------- #
# The independent HF key enumeration (see the module docstring).
# --------------------------------------------------------------------------- #


def _hf_keys_for_layer(
    *, is_dsa: bool, is_moe: bool, n_routed: int, n_shared: int
) -> list[str]:
    """Every checkpoint key one layer of the miniature holds, UNQUALIFIED.

    Unqualified because ``hf_state_to_fake_slices`` is what applies the
    ``model.layers.<i>.`` prefix. Written as literal suffixes on purpose: this
    enumeration must not share code with the mapping builder it is checked
    against.
    """
    keys = ["input_layernorm.weight", "post_attention_layernorm.weight"]

    if is_dsa:
        for leaf in (
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
        ):
            keys += [f"self_attn.{leaf}.weight", f"self_attn.{leaf}.{SCALE_SUFFIX}"]
        keys += ["self_attn.q_a_layernorm.weight", "self_attn.kv_a_layernorm.weight"]
        for leaf in ("wq", "wk"):
            keys += [
                f"self_attn.indexer.{leaf}.weight",
                f"self_attn.indexer.{leaf}.{SCALE_SUFFIX}",
            ]
        keys += [
            "self_attn.indexer.k_norm.weight",
            "self_attn.indexer.weights_proj.weight",
        ]
    else:
        for leaf in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
            keys += [f"linear_attn.{leaf}.weight", f"linear_attn.{leaf}.{SCALE_SUFFIX}"]
        keys += [
            "linear_attn.conv1d.weight",
            "linear_attn.norm.weight",
            "linear_attn.conv1d.bias",
            "linear_attn.dt_bias",
            "linear_attn.A_log",
        ]

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


# --------------------------------------------------------------------------- #
# The fixture: slice map via the helper, 3-shard composition authored here.
# --------------------------------------------------------------------------- #


def _fake_state(keys: list[str]) -> dict[str, torch.Tensor]:
    """A synthetic HF state dict. Shapes are irrelevant to key routing."""
    return {key: torch.zeros(2, 2) for key in keys}


def _build_slice_map(cfg: Glm5NextTextConfig) -> dict[str, FakeSafeSlice]:
    """The flat ``{checkpoint_key: FakeSafeSlice}`` map, built WITH the helper.

    One ``hf_state_to_fake_slices`` call per layer (so the helper applies each
    layer's own prefix) plus one with ``layer_idx=None`` for the rest.
    """
    slice_map: dict[str, FakeSafeSlice] = {}
    slice_map.update(
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
        slice_map.update(hf_state_to_fake_slices(_fake_state(layer_keys), layer_id))
    return slice_map


def _shard_name(shard_id: int, total: int) -> str:
    return f"model-{shard_id + 1:05d}-of-{total:05d}.safetensors"


def _partition_into_shards(
    keys: list[str], num_shards: int = DECLARED_NUM_SHARDS
) -> dict[str, list[str]]:
    """Split a flat key list into contiguous per-shard key lists.

    THIS is the shard composition -- the helper supplies no shard concept. Split
    contiguously rather than round-robin because that is how a real HF shard set
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
    """The miniature text config -- layers 0-3 of the real stack."""
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
# The miniature is the miniature it claims to be.
# --------------------------------------------------------------------------- #


def test_skeleton_miniature_reproduces_the_real_stack_phase(mini_config) -> None:
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
    _record(miniature_layer_types=list(mini_config.layer_types))


def test_skeleton_index_is_three_shards_not_sixty_two(shard_index) -> None:
    """HAZARD 4: the acceptance says 3. The heading's 62 is unsourced prose."""
    assert shard_index.num_shards == DECLARED_NUM_SHARDS
    assert len(shard_index.per_shard_counts()) == DECLARED_NUM_SHARDS
    _record(num_shards=shard_index.num_shards)


# --------------------------------------------------------------------------- #
# C01 -- 100% of the fixture's HF keys are mapped.
# --------------------------------------------------------------------------- #


def test_skeleton_maps_one_hundred_percent_of_hf_keys(coverage, slice_map) -> None:
    """C01. Certified by ``check_key_coverage`` over the loader's own mapping."""
    assert coverage.unique_checkpoint_key_count == len(slice_map)
    assert coverage.mapped_key_count == len(slice_map)
    assert coverage.coverage_fraction == 1.0
    _record(
        c01_fixture_key_count=len(slice_map),
        c01_mapped_key_count=coverage.mapped_key_count,
        c01_coverage_fraction=coverage.coverage_fraction,
    )


# --------------------------------------------------------------------------- #
# C02 -- 0 unmatched, both directions.
# --------------------------------------------------------------------------- #


def test_skeleton_reports_zero_unmatched(coverage) -> None:
    """C02. Both directions reported separately, then jointly."""
    assert coverage.unmatched_checkpoint_keys == ()
    assert coverage.unmatched_parameters == {}
    assert coverage.unmatched_count == 0
    assert coverage.is_complete
    _record(
        c02_unmatched_count=coverage.unmatched_count,
        c02_num_parameters=len(coverage.matched_parameters),
    )


def test_skeleton_unmatched_counters_can_report_nonzero(
    mini_config, slice_map
) -> None:
    """C02 non-vacuity: a counter that cannot fail is not a measurement.

    Perturbs each side in turn, which is also what proves the two enumerations
    are independent -- if they shared code, neither perturbation could disagree.

    The two arms are deliberately complementary, and they show that C01 and C02
    are separate conjuncts rather than one restated: arm (a) removes a key the
    mapping wants, which C02 catches while C01's fraction stays 1.0 (every key
    still *present* is still mapped); arm (b) adds a key nobody wants, which
    moves both.
    """
    mappings = build_weight_mappings(mini_config)
    all_keys = list(slice_map)

    # (a) drop one checkpoint key the mapping asks for -> unmatched parameter.
    dropped = "model.layers.3.mlp.shared_experts.down_proj.weight"
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
    assert short.coverage_fraction == 1.0  # C01 cannot see a key that is absent
    assert not short.is_complete

    # (b) add a checkpoint key no parameter asks for -> unmatched key, and this
    # is the direction that does move the coverage fraction.
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

    _record(
        c02_arm_a_unmatched=short.unmatched_count,
        c02_arm_b_unmatched=long.unmatched_count,
    )


# --------------------------------------------------------------------------- #
# C03 -- 0 duplicated, and the LOADER is what certifies it.
# --------------------------------------------------------------------------- #


def test_skeleton_reports_zero_duplicated(coverage, shard_index) -> None:
    """C03 on the clean fixture."""
    assert coverage.duplicated_keys == {}
    assert coverage.duplicated_count == 0
    assert shard_index.duplicated_keys() == {}
    shard_index.require_no_duplicates()  # must not raise
    _record(c03_duplicated_count=coverage.duplicated_count)


def test_skeleton_duplicate_is_certified_by_the_loader_not_the_fixture(
    mini_config, slice_map
) -> None:
    """HAZARD 2, all three legs, on the CROSS-SHARD path.

    Leg 1: the fixture builder does not raise -- it never sees the duplicate,
    because it is called per layer on a flat dict with no shard concept.
    Leg 2: the pin's own flattened ``{key: file}`` dict silently loses it.
    Leg 3: the loader reports it, and raises under ``strict``.
    """
    duplicated = "model.layers.0.linear_attn.out_proj.weight"
    assert duplicated in slice_map

    shards = _partition_into_shards(list(slice_map))
    shard_names = list(shards)
    home = next(name for name in shard_names if duplicated in shards[name])
    other = next(name for name in shard_names if name != home)
    shards[other] = [*shards[other], duplicated]  # now in TWO shard files

    # Leg 1 -- the helper is not the certifier. Rebuilding the slice map runs
    # the helper's own within-layer duplicate guard (utils.py:137-138) and it
    # stays silent, because a cross-shard duplicate is invisible to it.
    rebuilt = _build_slice_map(mini_config)
    assert set(rebuilt) == set(slice_map)

    # Leg 2 -- the pin's flattening loses it (checkpoints.py:226-227).
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

    _record(
        c03_cross_shard_duplicate_shards=sorted(reported[duplicated]),
        c03_flattened_lost_it=collapsed.duplicated_keys() == {},
    )


# --------------------------------------------------------------------------- #
# C04 -- per-shard counts sum exactly to the fixture's key count.
# --------------------------------------------------------------------------- #


def test_skeleton_per_shard_counts_sum_to_fixture_key_count(
    shard_index, coverage, slice_map
) -> None:
    """C04, on the composition authored in this file (HAZARD 3)."""
    counts = shard_index.per_shard_counts()
    assert len(counts) == DECLARED_NUM_SHARDS
    assert all(count > 0 for count in counts.values())
    assert sum(counts.values()) == len(slice_map)
    assert shard_index.total_shard_key_count == len(slice_map)
    assert coverage.total_shard_key_count == len(slice_map)
    assert shard_index.unique_key_count == len(slice_map)
    _record(
        c04_per_shard_counts=counts,
        c04_sum=sum(counts.values()),
        c04_fixture_key_count=len(slice_map),
    )


def test_skeleton_per_shard_sum_exceeds_unique_when_duplicated(slice_map) -> None:
    """C04 non-vacuity: the sum is what makes a cross-shard duplicate visible."""
    shards = _partition_into_shards(list(slice_map))
    victim = next(iter(shards[_shard_name(0, DECLARED_NUM_SHARDS)]))
    last = _shard_name(DECLARED_NUM_SHARDS - 1, DECLARED_NUM_SHARDS)
    shards[last] = [*shards[last], victim]

    index = Glm5NextShardIndex.from_shard_key_lists(shards)
    assert index.total_shard_key_count == len(slice_map) + 1
    assert index.unique_key_count == len(slice_map)
    _record(
        c04_dirty_sum=index.total_shard_key_count,
        c04_dirty_unique=index.unique_key_count,
    )


# --------------------------------------------------------------------------- #
# Skeleton surface guards.
# --------------------------------------------------------------------------- #


def test_skeleton_key_families_are_all_tagged() -> None:
    """Every declared family carries a provenance tag, and absences are named."""
    assert KEY_FAMILY_PROVENANCE
    assert set(KEY_FAMILY_PROVENANCE.values()) <= {GROUNDED, PROVISIONAL}
    assert {"dsa_indexer", "kda_linear_attention"} <= {
        name for name, tag in KEY_FAMILY_PROVENANCE.items() if tag == PROVISIONAL
    }
    assert set(ABSENT_KEY_FAMILIES) == {"multi_hyper_connections", "vision_tower"}
    assert all(reason.strip() for reason in ABSENT_KEY_FAMILIES.values())
    _record(
        provisional_families=sorted(
            name for name, tag in KEY_FAMILY_PROVENANCE.items() if tag == PROVISIONAL
        )
    )


def test_skeleton_moe_expert_parameter_consumes_a_key_list(mini_config) -> None:
    """The list-valued mapping branch is real: one param, many expert keys.

    This checkpoint stores one tensor per expert, unlike the fork's only extant
    MoE precedent, so the fused expert parameter must map to a list.
    """
    mappings = build_weight_mappings(mini_config)
    param = "model.layers.3.mlp.experts.gate_proj_weight"
    keys = mappings[param]
    assert isinstance(keys, list)
    # 4 experts x (weight + scale)
    assert len(keys) == MINI_ROUTED_EXPERTS * 2
    assert len(scale_keys(keys)) == MINI_ROUTED_EXPERTS
    # A single-key parameter stays a bare string, as the pin's shape requires.
    assert isinstance(mappings["model.norm_weight"], str)
    _record(c_moe_expert_key_count=len(keys))


def test_skeleton_no_rope_projection_is_mapped(mini_config) -> None:
    """``mla_use_nope`` with ``qk_rope_head_dim == 0``: no rotary head slice."""
    mappings = build_weight_mappings(mini_config)
    every_key = [
        key
        for value in mappings.values()
        for key in (value if isinstance(value, list) else [value])
    ]
    assert mini_config.qk_rope_head_dim == 0
    assert [key for key in every_key if "rope" in key] == []


def test_skeleton_index_json_round_trip(slice_map) -> None:
    """``from_index_json`` reads a ``weight_map`` and agrees on the counts."""
    shards = _partition_into_shards(list(slice_map))
    weight_map = {key: shard for shard, keys in shards.items() for key in keys}
    index = Glm5NextShardIndex.from_index_json(json.dumps({"weight_map": weight_map}))
    assert index.num_shards == DECLARED_NUM_SHARDS
    assert index.unique_key_count == len(slice_map)
    assert sum(index.per_shard_counts().values()) == len(slice_map)


def test_skeleton_reports_the_measured_readings(coverage, shard_index, slice_map) -> None:
    """Write the measured values out; pytest swallows stdout on a pass."""
    _record(
        report_fixture_key_count=len(slice_map),
        report_num_shards=shard_index.num_shards,
        report_per_shard_counts=shard_index.per_shard_counts(),
        report_num_parameters=len(coverage.matched_parameters),
        report_scale_key_count=len(scale_keys(slice_map)),
        report_results_path=str(_RESULTS_PATH),
    )
    assert _RESULTS_PATH.exists()
