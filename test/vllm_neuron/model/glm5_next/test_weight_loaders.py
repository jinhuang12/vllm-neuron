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


# =========================================================================== #
# inc-glm53f-012 -- WP1: block-fp8 scale loading with the 240-max downscale
# =========================================================================== #
#
# THE `-k` PARTITION, AND WHY THIS SECTION IS APPENDED
# ---------------------------------------------------
# `inc-glm53f-011` owns `-k skeleton` above; this increment owns
# `-k fp8_downscale` below. Neither selection can collect the other's items, so
# neither increment's counted predicate can be satisfied or broken by the other's
# tests. Every test name below carries `fp8_downscale` and none carries
# `skeleton`. The section -- imports included -- is appended rather than merged
# into the header block for the same reason the module under test appends its own
# half: the plan declares this increment's change a PURE ADDITION, and an insert
# into the header moves every `-011` line below it.
#
# THE DECLARED ACCEPTANCE (increment plan revision 12, L3582), verbatim:
#
#     "for a synthetic `[256,256]` weight with 4 `[128,128]` fp32 block scales,
#     after downscale-and-compensate the **dequantised** tensor matches the
#     pre-squeeze dequantisation with **rtol 3e-2 / atol 1e-5 per block** (4/4
#     blocks pass, reported per block -- per-block because the global-abs-max
#     normalisation would otherwise mask it, D3); every stored byte satisfies
#     `abs(x) <= 240.0` in **100%** of elements; and **0** scales fall below
#     `MINVAL = 1e-5`."
#
# Three counted conjuncts, measured below as C1-C3, plus the declared regression
# case (L3583: "a regression case with one deliberately tiny block scale
# asserting the `MINVAL` floor engages").
#
# WHAT "MATCHES WITH rtol 3e-2 PER BLOCK" IS READ AS, AND WHY IT MATTERS
# ---------------------------------------------------------------------
# The criterion's own clause names its comparison: "per-block because the
# GLOBAL-ABS-MAX NORMALISATION would otherwise mask it". So the predicate is the
# abs-max-normalised difference with the BLOCK's absolute maximum as the
# reference -- `max|after - before| <= atol + rtol * max|before|` over each
# block -- rather than a per-element relative comparison.
#
# This is not a convenient reading, it is the only one the declared tolerance can
# carry, and the arithmetic says so: squeezing an fp8 byte by 240/448 and
# re-quantising costs up to 6.25% of that byte's own magnitude (e4m3 has three
# mantissa bits, so half the grid spacing is 1/16). A per-element reading is
# therefore breached by construction -- `byte 256.0 -> 144.0 -> 268.8` is a 5.0%
# element error that no fixture choice avoids. Block-normalised, the same worst
# element contributes `12.8 / 448 = 0.0286`, inside `3e-2`. Both numbers are
# measured and recorded below (`worst_element_relative` is reported on every
# block and gated on none), so the reading is visible in the evidence instead of
# being implicit in a pass.
#
# The margin is thin (0.0286 against 0.0300) and it DEPENDS on each block
# reaching the top of the OCP range: a block whose maximum is 416 instead of 448
# measures 0.0308 and would fail. That is a property of blockwise quantisation
# rather than a fixture trick -- a per-block scale is chosen so the block's
# values use the full fp8 range -- but it is load-bearing, so
# `test_fp8_downscale_fixture_is_full_range_and_block_shaped` asserts it out loud
# rather than leaving the pass resting on an accident.
#
# NEITHER TOLERANCE IS THIS FILE'S TO MOVE. `rtol 3e-2` traces to the frozen
# acceptance pre-registration's single uniform loosening factor of 3
# ("Registered value: 3"), and `atol 1e-5` is that precedent's uniform atol. The
# declared falsification route is that a measurement exceeding a registered
# tolerance is a FAIL whose remedy is a user decision.

from vllm_neuron.model.glm5_next.config import Glm5NextConfig
from vllm_neuron.model.glm5_next.quantization import (
    DEFAULT_WEIGHT_BLOCK_SIZE,
    QuantScheme,
    QuantizationSpec,
)
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    MINVAL,
    block_agreement,
    block_grid_shape,
    blockwise_scale_loader,
    compensate_block_scales,
    dequantise_blockwise,
    downscale_fp8_weight_bytes,
    needs_240_downscale,
    resolved_fp8_clamp_max,
    squeeze_blockwise_fp8,
    wrap_with_blockwise_fp8_downscale,
)
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

# --------------------------------------------------------------------------- #
# Declared values -- every one traced to its source, none invented here.
# --------------------------------------------------------------------------- #

#: Legacy `nl.float8_e4m3` max finite magnitude: the clamp trn2 resolves, and the
#: bound C2 counts against (plan L3582; `dtype_utils.py:18`).
FP8_DECLARED_CLAMP = 240.0

#: OCP `float8_e4m3fn` max finite magnitude -- the checkpoint's scale space
#: (`dtype_utils.py:19`).
FP8_OCP_MAX = 448.0

#: The acceptance's synthetic weight shape and block shape (plan L3582).
FP8_WEIGHT_SHAPE = (256, 256)
FP8_BLOCK_SIZE = (128, 128)

#: The registered tolerances. NOT this file's to widen -- see the section header.
FP8_RTOL = 3e-2
FP8_ATOL = 1e-5

#: The 4 fp32 block scales, one per `[128,128]` tile. Deliberately spread over
#: 1.5 orders of magnitude and all well above `MINVAL`, so C3's zero is a census
#: over a real population rather than a restatement of the fixture, and so the
#: smallest-scaled block cannot hide behind the largest one.
FP8_BLOCK_SCALES = ((2.5e-3, 7.5e-4), (1.25e-2, 4.0e-4))

#: The regression case's deliberately tiny scale, four orders below `MINVAL`
#: even after the 448/240 compensation has multiplied it up.
FP8_TINY_SCALE = 1e-9

#: Which tile the regression case makes tiny.
FP8_TINY_BLOCK = (1, 1)

_FP8_RESULTS_PATH = Path(
    os.environ.get("VLLM_NEURON_INC012_RESULTS_JSON")
    or Path(tempfile.gettempdir()) / "vllm_neuron_inc012_predicates.json"
)
#: This partition's own results file, kept separate from `-011`'s so a reader of
#: either record cannot mistake one increment's measurements for the other's.
#: Written lazily on the first record -- no import-time side effect -- and the
#: first write replaces the whole file, so a stale value from an earlier run
#: cannot survive into this one.
_FP8_RESULTS: dict[str, Any] = {}


def _record_fp8(**values: Any) -> None:
    _FP8_RESULTS.update(values)
    _FP8_RESULTS_PATH.write_text(
        json.dumps(_FP8_RESULTS, indent=2, sort_keys=True, default=str) + "\n"
    )


# --------------------------------------------------------------------------- #
# The synthetic fixture. Deterministic by construction -- no RNG, no seed.
# --------------------------------------------------------------------------- #


def _fp8_full_range_tile(variant: int) -> torch.Tensor:
    """One `[128,128]` fp32 tile sweeping the whole OCP e4m3fn magnitude range.

    A linspace from `-448` to `+448` hits both endpoints exactly (both are
    representable), and casting to fp8 collapses the 16384 samples onto the
    grid -- so the tile contains essentially every representable magnitude,
    including `256.0`, the byte whose re-quantisation error is the worst in the
    range. The four variants are range-preserving rearrangements: every tile's
    absolute maximum is `448`, which is what a per-block quantiser produces and
    what C1's normalisation leans on.
    """
    rows, cols = FP8_BLOCK_SIZE
    ramp = torch.linspace(
        -FP8_OCP_MAX, FP8_OCP_MAX, rows * cols, dtype=torch.float32
    ).reshape(rows, cols)
    if variant == 0:
        return ramp
    if variant == 1:
        return torch.flip(ramp, dims=(1,))
    if variant == 2:
        return ramp.t().contiguous()
    return -ramp


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
    """A Python float as the fp32 scale grid actually stores it.

    The scale grid is fp32 and `MINVAL` is a Python float (binary64), so a stored
    floor is `float32(1e-5)` = `9.999999747378752e-06`, not the binary64 literal
    `1e-5`. Comparing a widened readback against the literal asserts a value the
    grid's dtype cannot hold, which is a claim about float64 rather than about the
    floor. Equality is still EXACT -- it is just exact in the dtype that stores
    the number, which is why this returns a tensor to compare with `torch.equal`
    rather than a tolerance to compare within.
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
# The gate: CONDITIONAL on the resolved clamp (lead-ruling-012-downscale-gate).
# --------------------------------------------------------------------------- #


def test_fp8_downscale_gate_follows_the_resolved_platform_clamp() -> None:
    """The squeeze is gated on the vendor's resolved clamp, and the gate is live.

    `approvals/lead-ruling-012-downscale-gate.md`: the downscale "applies IFF the
    resolved platform clamp is 240.0 … On a 448.0-max platform an unconditional
    240/448 rescale would corrupt correct weights". This pins the antecedent on
    the instrument the other conjuncts run on, so a later reader knows C1-C3 were
    measured with the gate TRUE rather than measured through a gate nobody
    checked.

    The complementary arm is deliberately absent. The pinned invocation fixes
    `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` before collection
    (`test/conftest.py:23-25`), so this instrument cannot discriminate a gated
    implementation from an ungated one, and the ruling declines a trn3/448
    discriminating test as outside this increment's scope (ruling item 3). The
    limitation is recorded, not repaired here.
    """
    clamp = resolved_fp8_clamp_max()
    assert clamp == FP8_DECLARED_CLAMP, (
        f"resolved clamp is {clamp}, not {FP8_DECLARED_CLAMP}; the acceptance "
        "environment did not pin trn2"
    )
    assert needs_240_downscale() is True
    _record_fp8(gate_resolved_clamp=clamp, gate_needs_downscale=True)


# --------------------------------------------------------------------------- #
# Fixture integrity -- what C1's and C2's non-vacuity rest on.
# --------------------------------------------------------------------------- #


def test_fp8_downscale_fixture_is_full_range_and_block_shaped(
    fp8_downscale_weight, fp8_downscale_scales
) -> None:
    """The declared shapes, and the two properties the conjuncts lean on.

    Property 1 (C1's): every tile's absolute maximum is the top of the OCP range.
    Block-normalised agreement is sensitive to it -- a tile topping out at 416
    measures 0.0308 against a 0.0300 tolerance -- so it is asserted rather than
    assumed. Property 2 (C2's): the pre-squeeze bytes are NOT already inside the
    240 range, or "100% within 240" afterwards would certify nothing.
    """
    assert tuple(fp8_downscale_weight.shape) == FP8_WEIGHT_SHAPE
    assert fp8_downscale_weight.dtype is torch.float8_e4m3fn
    assert block_grid_shape(FP8_WEIGHT_SHAPE, FP8_BLOCK_SIZE) == (2, 2)
    assert tuple(fp8_downscale_scales.shape) == (2, 2)
    assert fp8_downscale_scales.numel() == 4
    assert fp8_downscale_scales.dtype is torch.float32

    dense = fp8_downscale_weight.to(torch.float32)
    block_rows, block_cols = FP8_BLOCK_SIZE
    tile_maxima = {}
    for grid_row, grid_col in _fp8_tiles():
        tile = dense[
            grid_row * block_rows : (grid_row + 1) * block_rows,
            grid_col * block_cols : (grid_col + 1) * block_cols,
        ]
        tile_max = float(tile.abs().max().item())
        tile_maxima[f"{grid_row},{grid_col}"] = tile_max
        assert tile_max == FP8_OCP_MAX, (
            f"tile {(grid_row, grid_col)} tops out at {tile_max}, not "
            f"{FP8_OCP_MAX}; C1's block normalisation depends on this"
        )

    fraction_within = float((dense.abs() <= FP8_DECLARED_CLAMP).to(torch.float32).mean())
    assert fraction_within < 1.0, (
        "the pre-squeeze fixture is already inside the 240 range, so C2 would "
        "pass without the squeeze doing anything"
    )
    # Every declared scale is above the floor, so C3's zero is not the floor's doing.
    assert bool((fp8_downscale_scales >= MINVAL).all())
    _record_fp8(
        fixture_tile_maxima=tile_maxima,
        fixture_fraction_within_240_before=fraction_within,
        fixture_scales=fp8_downscale_scales.flatten().tolist(),
    )


# --------------------------------------------------------------------------- #
# C1 -- the dequantised tensor survives the squeeze, 4/4 blocks, per block.
# --------------------------------------------------------------------------- #


def test_fp8_downscale_c1_dequantisation_agrees_per_block(
    fp8_downscale_weight, fp8_downscale_scales
) -> None:
    """C1. Reported per block (D3), never aggregated.

    `rtol`/`atol` are passed explicitly, as every fp8 comparison in this campaign
    must be: the fork's tolerance map has no fp8 entry and falls back to the bf16
    pair (pre-registration PIT-13).
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

    # Non-vacuity: atol alone carries no block, so the rtol term is what passes
    # them. Without this, a fixture with tiny scales would pass on atol and C1
    # would certify nothing about the 240/448 squeeze at all.
    for report in reports:
        assert report.max_abs_diff > FP8_ATOL, (
            f"block {report.index} passes on atol alone "
            f"(max_abs_diff={report.max_abs_diff:.3e} <= atol={FP8_ATOL:.3e})"
        )

    _record_fp8(
        c1_blocks_within=len([r for r in reports if r.within]),
        c1_blocks_total=len(reports),
        c1_per_block={
            f"{r.index[0]},{r.index[1]}": {
                "normalised_diff": r.normalised_diff,
                "max_abs_diff": r.max_abs_diff,
                "max_abs_before": r.max_abs_before,
                "tolerance": r.tolerance,
                "within": r.within,
                "worst_element_relative": r.worst_element_relative,
            }
            for r in reports
        },
        c1_rtol=FP8_RTOL,
        c1_atol=FP8_ATOL,
        # Disclosed, gated on nothing: the worst per-element relative difference
        # across all four blocks. It exceeds rtol, which is exactly why the
        # criterion's own clause names a normalised comparison.
        c1_worst_element_relative_disclosed=max(
            r.worst_element_relative for r in reports
        ),
    )


# --------------------------------------------------------------------------- #
# C2 -- every stored byte is inside the 240 range, in 100% of elements.
# --------------------------------------------------------------------------- #


def test_fp8_downscale_c2_every_stored_byte_is_within_240(
    fp8_downscale_weight, fp8_downscale_scales
) -> None:
    """C2, as a fraction rather than an `all()`, so the shortfall is legible."""
    squeeze = squeeze_blockwise_fp8(
        fp8_downscale_weight, fp8_downscale_scales, FP8_BLOCK_SIZE
    )
    assert squeeze.weight.dtype is torch.float8_e4m3fn
    assert tuple(squeeze.weight.shape) == FP8_WEIGHT_SHAPE
    assert squeeze.fraction_within_240 == 1.0
    assert squeeze.max_abs_stored <= FP8_DECLARED_CLAMP

    # The bound is reached, not merely respected: 448 maps to exactly 240, so a
    # squeeze that quietly over-shrank the bytes would show up here.
    assert squeeze.max_abs_stored == FP8_DECLARED_CLAMP
    _record_fp8(
        c2_fraction_within_240=squeeze.fraction_within_240,
        c2_max_abs_stored=squeeze.max_abs_stored,
        c2_element_count=squeeze.weight.numel(),
    )


# --------------------------------------------------------------------------- #
# C3 -- a counted zero: no stored scale falls below MINVAL.
# --------------------------------------------------------------------------- #


def test_fp8_downscale_c3_no_scale_falls_below_minval(
    fp8_downscale_weight, fp8_downscale_scales
) -> None:
    """C3, on the synthetic case only.

    Counted on BOTH populations. The stored count alone would be near-tautological
    -- the floor guarantees it -- so the input census is what makes the zero a
    measurement: no declared scale was near the floor, and none was floored.
    The regression case below is the arm that proves the counter can move.
    """
    assert MINVAL == 1e-5
    squeeze = squeeze_blockwise_fp8(
        fp8_downscale_weight, fp8_downscale_scales, FP8_BLOCK_SIZE
    )
    assert squeeze.below_minval_before == 0
    assert squeeze.below_minval_after == 0
    assert squeeze.floored_blocks == ()

    # The compensation is the exact inverse of the byte squeeze, and it is
    # applied to every tile -- not just to the tiles that needed clamping.
    expected = fp8_downscale_scales * (FP8_OCP_MAX / FP8_DECLARED_CLAMP)
    assert torch.equal(squeeze.scale_inv, expected)
    _record_fp8(
        c3_minval=MINVAL,
        c3_scales_below_minval_before=squeeze.below_minval_before,
        c3_scales_below_minval_after=squeeze.below_minval_after,
        c3_floored_blocks=list(squeeze.floored_blocks),
        c3_scale_population=squeeze.scale_inv.numel(),
    )


# --------------------------------------------------------------------------- #
# The declared regression case -- the MINVAL floor engages, and locally.
# --------------------------------------------------------------------------- #


def test_fp8_downscale_minval_floor_engages_on_a_tiny_block_scale(
    fp8_downscale_weight,
) -> None:
    """Plan L3583: "a regression case with one deliberately tiny block scale
    asserting the `MINVAL` floor engages".

    Three legs. The floor ENGAGES (the stored scale is exactly `MINVAL`, and the
    block is named in the report). It is LOCAL (the other three tiles are
    untouched and still agree). And it is VISIBLE in C1's own predicate: the
    floored tile's agreement is False, which is what proves C1 is a measurement
    that can fail rather than a formality -- the same predicate, the same
    tolerances, a different answer.
    """
    scales = [list(row) for row in FP8_BLOCK_SCALES]
    scales[FP8_TINY_BLOCK[0]][FP8_TINY_BLOCK[1]] = FP8_TINY_SCALE
    tiny_grid = _fp8_scale_grid(tuple(tuple(row) for row in scales))

    before = dequantise_blockwise(fp8_downscale_weight, tiny_grid, FP8_BLOCK_SIZE)
    squeeze = squeeze_blockwise_fp8(fp8_downscale_weight, tiny_grid, FP8_BLOCK_SIZE)

    # Leg 1 -- the floor engaged, on exactly the one block that needed it. The
    # stored value is compared in the grid's own dtype (see `_fp8_as_stored`):
    # exactly the fp32 floor, not merely close to it.
    assert squeeze.below_minval_before == 1
    assert squeeze.floored_blocks == (FP8_TINY_BLOCK,)
    assert squeeze.below_minval_after == 0
    floored_value = float(squeeze.scale_inv[FP8_TINY_BLOCK].item())
    assert torch.equal(squeeze.scale_inv[FP8_TINY_BLOCK], _fp8_as_stored(MINVAL))
    # And the floor is what put it there: the compensated value it replaced was
    # four orders of magnitude below the floor.
    assert FP8_TINY_SCALE * (FP8_OCP_MAX / FP8_DECLARED_CLAMP) < MINVAL

    # Leg 2 -- local: every other tile carries the plain compensated scale,
    # compared tile-by-tile against the same grid the transform started from.
    compensated = tiny_grid * (FP8_OCP_MAX / FP8_DECLARED_CLAMP)
    for grid_row, grid_col in _fp8_tiles():
        if (grid_row, grid_col) == FP8_TINY_BLOCK:
            continue
        assert torch.equal(
            squeeze.scale_inv[grid_row, grid_col], compensated[grid_row, grid_col]
        )

    # Leg 3 -- C1's predicate moves, and only on the floored tile.
    after = dequantise_blockwise(squeeze.weight, squeeze.scale_inv, FP8_BLOCK_SIZE)
    by_index = {
        report.index: report
        for report in block_agreement(
            before, after, block_size=FP8_BLOCK_SIZE, rtol=FP8_RTOL, atol=FP8_ATOL
        )
    }
    assert by_index[FP8_TINY_BLOCK].within is False, (
        "the floored block's dequantisation must NOT agree -- the floor raised "
        "its scale by four orders of magnitude on purpose"
    )
    assert all(
        by_index[tile].within for tile in _fp8_tiles() if tile != FP8_TINY_BLOCK
    )

    _record_fp8(
        regression_tiny_scale=FP8_TINY_SCALE,
        regression_floored_blocks=list(squeeze.floored_blocks),
        regression_floored_stored_value=floored_value,
        regression_floored_block_within=by_index[FP8_TINY_BLOCK].within,
        regression_other_blocks_within=[
            by_index[tile].within
            for tile in _fp8_tiles()
            if tile != FP8_TINY_BLOCK
        ],
    )


# --------------------------------------------------------------------------- #
# The checkpoint path -- the same two halves, driven through fake slices.
# --------------------------------------------------------------------------- #


def test_fp8_downscale_scale_loader_compensates_through_a_fake_slice(
    fp8_downscale_scales,
) -> None:
    """The `weight_scale_inv` loader returns the compensated, floored grid."""
    loader = blockwise_scale_loader()
    loaded = loader.load([FakeSafeSlice(fp8_downscale_scales)], 0)

    assert loaded.dtype is torch.float32
    assert tuple(loaded.shape) == (2, 2)
    assert torch.equal(loaded, compensate_block_scales(fp8_downscale_scales).scale_inv)
    assert torch.equal(
        loaded, fp8_downscale_scales * (FP8_OCP_MAX / FP8_DECLARED_CLAMP)
    )
    _record_fp8(loader_scale_grid=loaded.flatten().tolist())


def test_fp8_downscale_weight_loader_squeezes_through_a_fake_slice(
    fp8_downscale_weight,
) -> None:
    """The wrapped weight loader stores bytes inside the 240 range.

    Wraps the identity loader, which is the shape the model file will wrap a
    sharding loader in: the wrapper composes with whatever transform it is given
    rather than replacing it.
    """
    wrapped = wrap_with_blockwise_fp8_downscale(SafetensorsWeightLoader())
    loaded = wrapped.load([FakeSafeSlice(fp8_downscale_weight)], 0)

    assert loaded.dtype is torch.float8_e4m3fn
    assert tuple(loaded.shape) == FP8_WEIGHT_SHAPE
    dense = loaded.to(torch.float32)
    assert float(dense.abs().max().item()) == FP8_DECLARED_CLAMP
    assert torch.equal(
        dense, downscale_fp8_weight_bytes(fp8_downscale_weight).to(torch.float32)
    )
    _record_fp8(loader_weight_max_abs=float(dense.abs().max().item()))


# --------------------------------------------------------------------------- #
# The quantization spec this half's block geometry comes from.
# --------------------------------------------------------------------------- #


def test_fp8_downscale_quantization_spec_parses_the_block_config() -> None:
    """The new `quantization.py` resolves the blockwise scheme and its geometry.

    Blockwise, not per-tensor: the scheme carries a 2-D block shape, and that
    shape is what the loaders index the scale grid with, so a config that fails
    to declare it must raise here rather than surface as a shape mismatch inside
    a transform.
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
    # The lookup is uniform over both call shapes, in-block and outside.
    assert (
        spec.get_scheme(0, "linear_attn.out_proj") is QuantScheme.FP8_BLOCK_DYNAMIC
    )
    assert spec.get_scheme(None, "lm_head") is QuantScheme.FP8_BLOCK_DYNAMIC

    # An unquantized checkpoint is None, not a NONE-scheme spec.
    assert QuantizationSpec.from_hf_quantization_config(None) is None
    assert QuantizationSpec.from_hf_quantization_config({}) is None

    # The fields config.py already lifted round-trip into the same spec.
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
    _record_fp8(
        spec_linear_scheme=spec.linear_scheme.value,
        spec_kv_cache_scheme=spec.kv_cache_scheme.value,
        spec_weight_block_size=list(spec.weight_block_size),
        spec_activation_scheme=spec.activation_scheme,
    )


def test_fp8_downscale_reports_the_measured_readings(
    fp8_downscale_weight, fp8_downscale_scales
) -> None:
    """Write this partition's measured values out; pytest swallows stdout."""
    squeeze = squeeze_blockwise_fp8(
        fp8_downscale_weight, fp8_downscale_scales, FP8_BLOCK_SIZE
    )
    _record_fp8(
        report_resolved_clamp=resolved_fp8_clamp_max(),
        report_applied=squeeze.applied,
        report_weight_shape=list(FP8_WEIGHT_SHAPE),
        report_block_size=list(FP8_BLOCK_SIZE),
        report_scale_grid_shape=list(block_grid_shape(FP8_WEIGHT_SHAPE, FP8_BLOCK_SIZE)),
        report_results_path=str(_FP8_RESULTS_PATH),
    )
    assert _FP8_RESULTS_PATH.exists()
