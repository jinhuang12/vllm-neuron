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

import contextlib
import hashlib
import json
import logging
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

    # Multi-hyper-connections: six bare tensors on every layer of the real stack.
    # inc-glm53f-078 -- declared ABSENT before the real index was on disk.
    keys += [
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
    ]

    if is_dsa:
        # inc-glm53f-078: only these four carry a scale companion.
        for leaf in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
            keys += [f"self_attn.{leaf}.weight", f"self_attn.{leaf}.{SCALE_SUFFIX}"]
        keys += [
            "self_attn.kv_b_proj.weight",  # real projection, no scale in this ckpt
            "self_attn.q_a_layernorm.weight",
            "self_attn.kv_a_layernorm.weight",
        ]
        # inc-glm53f-078: wq_b not wq, no indexer scales, plus the three leaves
        # nothing mapped before.
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
        # inc-glm53f-078: the KDA family is 15 self_attn leaves and no scales,
        # not the qwen3_next linear_attn convention this file first guessed.
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


#: The module-tree prefix ``hf_state_to_fake_slices`` applies, and the
#: checkpoint prefix the real index actually uses (``inc-glm53f-078``).
MODULE_PREFIX = "model."
CKPT_PREFIX = "model.language_model."


def _into_checkpoint_namespace(key: str) -> str:
    """Move one module-namespace key into the checkpoint's namespace.

    ``hf_state_to_fake_slices`` (``test/vllm_neuron/model/utils.py``) qualifies
    every layer key as ``model.layers.<i>.<leaf>``, which is the module tree's
    namespace and what this file's miniature was built in. The real checkpoint
    puts every text-model tensor under ``model.language_model.`` instead, so the
    miniature is moved after the shared helper runs rather than by changing the
    helper -- that helper is another increment's surface and other models use it.

    ``lm_head.weight`` is outside the text-model prefix in the real index and is
    left alone, which is why this is a prefix rewrite and not a blanket one.
    """
    if key.startswith(CKPT_PREFIX):
        return key
    if key.startswith(MODULE_PREFIX):
        return CKPT_PREFIX + key[len(MODULE_PREFIX) :]
    return key


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
    # inc-glm53f-078: the helper qualifies into the module namespace, the real
    # checkpoint is one namespace over. Rewriting is injective on this key set
    # (asserted), so the count cannot change under it.
    slice_map = {_into_checkpoint_namespace(key): sl for key, sl in raw.items()}
    assert len(slice_map) == len(raw), "namespace rewrite collided two keys"
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
    # inc-glm53f-078 re-namespaced the checkpoint side; the parameter name below
    # is unchanged, which is exactly the split this literal now demonstrates.
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
    # inc-glm53f-078: layer 0 is KDA, and the KDA family's real output projection
    # is ``self_attn.o_proj`` in the checkpoint namespace.
    duplicated = "model.language_model.layers.0.self_attn.o_proj.weight"
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
    # inc-glm53f-078 re-tagged both: each leaf is now a name the published index
    # itself carries, measured rather than guessed from a sibling architecture.
    assert {"dsa_indexer", "kda_linear_attention"} <= {
        name for name, tag in KEY_FAMILY_PROVENANCE.items() if tag == GROUNDED
    }
    # inc-glm53f-078 removed multi_hyper_connections: the family is present in
    # the index and is now mapped, so declaring it absent would be false.
    assert set(ABSENT_KEY_FAMILIES) == {"vision_tower"}
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
# inc-glm53f-078 -- WP1 REPAIR: the real checkpoint index and config as fixtures
# =========================================================================== #
#
# WHY THESE ITEMS ARE HERE AND NOT IN A NEW FILE
# ----------------------------------------------
# `inc-glm53f-078` is a SECOND WRITER on `inc-glm53f-011`'s `-k skeleton` side
# (plan section 11 row A.1, partitioned by concern: `-011` owns the shard index
# and the key map's shape, `-078` owns the checkpoint-key namespace and the
# family names). Every item below carries `skeleton` and none carries
# `fp8_downscale`, so `inc-glm53f-012`'s selection cannot collect them.
#
# WHAT IS DIFFERENT ABOUT THEM
# ----------------------------
# Everything above runs on a 4-layer miniature this file authors. These eight
# run on the REAL published checkpoint index -- 76,108 keys over 62 shards --
# landed as `fixtures/model.safetensors.index.json`. That is the whole point of
# the increment: `-011` wrote the key map against no checkpoint at all.
#
# EIGHT ITEMS, ONE PER COUNTED CONJUNCT, NO PARAMETRIZE (section 6 rule 6), and
# every denominator is DERIVED from the fixture rather than typed in, so a
# fixture that changed would move the expectation with it instead of silently
# disagreeing with a literal.

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_INDEX_PATH = FIXTURES_DIR / "model.safetensors.index.json"
REAL_CONFIG_PATH = FIXTURES_DIR / "hf-config.json"

#: The two families excluded from this increment's scope, each with its reason.
#: Prefixes, not counts: the counts are read off the fixture below.
MTP_LAYER_PREFIX = "model.language_model.layers.45."
VISION_PREFIX = "model.visual."

#: The vendor's own numbers for the two fixtures, fixed before this increment
#: ran. They are usable as expected values ONLY because each fixture is a
#: byte-identical copy of the published file -- see each fixture's provenance
#: sidecar. A digest this increment computed after editing a file would certify
#: nothing.
VENDOR_INDEX_SHA256 = (
    "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05"
)
VENDOR_INDEX_BYTES = 8406613
VENDOR_CONFIG_SHA256 = (
    "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
)
VENDOR_CONFIG_BYTES = 69416


@pytest.fixture(scope="module")
def real_weight_map() -> dict[str, str]:
    """The published index's own ``weight_map``: ``{checkpoint_key: shard}``."""
    return json.loads(REAL_INDEX_PATH.read_text())["weight_map"]


@pytest.fixture(scope="module")
def real_text_config() -> Glm5NextTextConfig:
    """The text config built from the real published config, not from defaults."""
    raw = json.loads(REAL_CONFIG_PATH.read_text())
    return Glm5NextTextConfig.from_hf_config(raw["text_config"])


@pytest.fixture(scope="module")
def real_in_scope(real_weight_map) -> dict[str, str]:
    """The in-scope key population: the whole map less the two named families."""
    return {
        key: shard
        for key, shard in real_weight_map.items()
        if not key.startswith(MTP_LAYER_PREFIX) and not key.startswith(VISION_PREFIX)
    }


@pytest.fixture(scope="module")
def real_index(real_in_scope) -> Glm5NextShardIndex:
    """A shard index over the in-scope keys, built the FAITHFUL way.

    ``from_shard_key_lists`` off per-shard lists rather than
    ``from_weight_map``, because the flattened direction is documented as unable
    to report a cross-shard duplicate at all -- and conjunct (d) is a duplicate
    reading, so it has to run on the direction that can report one.
    """
    per_shard: dict[str, list[str]] = {}
    for key, shard in real_in_scope.items():
        per_shard.setdefault(shard, []).append(key)
    return Glm5NextShardIndex.from_shard_key_lists(per_shard)


@pytest.fixture(scope="module")
def real_mappings(real_text_config) -> dict[str, str | list[str]]:
    """The mapping under test, over the real 45-layer schedule."""
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
# (a) The in-scope population is the two exclusions subtracted, and they sum.
# --------------------------------------------------------------------------- #


def test_skeleton_real_index_population_is_the_two_exclusions_subtracted(
    real_weight_map, real_in_scope
) -> None:
    """(a) 74,001 = 76,108 - 1,760 - 347, each term counted off the fixture.

    The two exclusions are named and counted rather than assumed, and the three
    parts are asserted to sum to the whole: a subtraction nobody adds back up is
    how a silently-dropped family hides.
    """
    total = len(real_weight_map)
    mtp = [k for k in real_weight_map if k.startswith(MTP_LAYER_PREFIX)]
    vision = [k for k in real_weight_map if k.startswith(VISION_PREFIX)]
    in_scope = len(real_in_scope)

    assert total == 76_108
    assert len(mtp) == 1_760
    assert len(vision) == 347
    assert in_scope == 74_001
    # The sum, which is what makes the subtraction a partition.
    assert in_scope + len(mtp) + len(vision) == total
    # And the two exclusions are disjoint, so no key was subtracted twice.
    assert set(mtp).isdisjoint(vision)

    _record(
        c078a_total_weight_map_keys=total,
        c078a_mtp_excluded=len(mtp),
        c078a_vision_excluded=len(vision),
        c078a_in_scope=in_scope,
    )


# --------------------------------------------------------------------------- #
# (b) check_key_coverage reports k/N == 100% over that population.
# --------------------------------------------------------------------------- #


def test_skeleton_real_index_coverage_is_one_hundred_percent(
    real_coverage, real_in_scope
) -> None:
    """(b) 74,001 / 74,001 mapped, 0 unmatched in either direction.

    A shortfall here is ``evidence_contradicts_design`` to the lead and never a
    fixture edit or a parameter rename (plan section 11 row A.3). The two
    unmatched lists are asserted empty before the count, so a failure names the
    keys rather than only the number.
    """
    assert real_coverage.unmatched_checkpoint_keys == ()
    assert real_coverage.unmatched_parameters == {}
    assert real_coverage.unmatched_count == 0
    assert real_coverage.unique_checkpoint_key_count == len(real_in_scope)
    assert real_coverage.mapped_key_count == len(real_in_scope)
    assert real_coverage.coverage_fraction == 1.0
    assert real_coverage.is_complete

    _record(
        c078b_unique_checkpoint_keys=real_coverage.unique_checkpoint_key_count,
        c078b_mapped_key_count=real_coverage.mapped_key_count,
        c078b_coverage_fraction=real_coverage.coverage_fraction,
        c078b_unmatched_count=real_coverage.unmatched_count,
        c078b_parameter_count=len(real_coverage.matched_parameters),
    )


# --------------------------------------------------------------------------- #
# (c) Zero orphan scale keys, over a scale population that is not empty.
# --------------------------------------------------------------------------- #


def test_skeleton_real_index_has_no_orphan_scale_keys(
    real_in_scope, real_referenced
) -> None:
    """(c) Every in-scope ``weight_scale_inv`` has its ``weight`` partner mapped.

    NON-VACUITY (design decision D1.5) is the scale population itself: 36,467
    scale keys are in scope, so the zero is a reading over a non-empty set. The
    control goes further and shows the predicate can report non-zero, by asking
    it about a scale key whose partner is deliberately not in the mapping.
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

    # The non-empty denominator, first: a zero over nothing is not a reading.
    assert scale_population == 36_467
    assert scale_population > 0
    assert real_orphans == []

    # FIRING CONTROL: drop one partner and the same predicate must report it.
    victim = scales[0]
    victim_partner = victim[: -len(f".{SCALE_SUFFIX}")] + ".weight"
    assert victim_partner in real_referenced
    poisoned = orphans(frozenset(real_referenced - {victim_partner}))
    assert poisoned == [victim], "the orphan predicate cannot report non-zero"

    _record(
        c078c_scale_population=scale_population,
        c078c_orphan_scale_keys=len(real_orphans),
        c078c_control_orphans=len(poisoned),
    )


# --------------------------------------------------------------------------- #
# (d) Zero duplicated keys, on require_no_duplicates over per-shard lists.
# --------------------------------------------------------------------------- #


def test_skeleton_real_index_has_no_duplicated_keys(
    real_index, real_coverage, real_in_scope
) -> None:
    """(d) 0 duplicates across the 62 shards, and the check can report one.

    Read off per-shard key lists, not off the flattened map: the flattened
    direction is documented as unable to represent a duplicate at all, so a zero
    from it would be an artefact of the container. The control re-runs the same
    method with one key placed in a second shard.
    """
    assert real_index.duplicated_keys() == {}
    assert real_coverage.duplicated_count == 0
    real_index.require_no_duplicates()  # must not raise
    # Sum over shards equals the unique count exactly when nothing is duplicated.
    assert real_index.total_shard_key_count == len(real_in_scope)
    assert real_index.unique_key_count == len(real_in_scope)

    # FIRING CONTROL: the same method over a deliberately dirty per-shard set.
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

    _record(
        c078d_num_shards=real_index.num_shards,
        c078d_duplicated_count=real_coverage.duplicated_count,
        c078d_control_duplicated_count=len(dirty.duplicated_keys()),
    )


# --------------------------------------------------------------------------- #
# (e) The KDA family: 15 leaves on each of 34 layers, 510 keys, 0 scales.
# --------------------------------------------------------------------------- #


def test_skeleton_real_kda_family_is_fifteen_leaves_and_no_scales(
    real_text_config, real_referenced, real_in_scope
) -> None:
    """(e) 510 = 15 x 34, and not one scale companion anywhere in the family.

    This is the family the skeleton got wholly wrong: it mapped the
    ``qwen3_next`` gated-delta convention (``linear_attn.in_proj_qkvz`` and
    friends), and the checkpoint carries 15 ``self_attn.*`` leaves instead. Both
    the leaf set and the layer count are derived, and the mapping's leaf set is
    asserted equal to the FIXTURE's, so agreement is with the checkpoint rather
    than with this module's own constants.
    """
    kda_layers = _layers_of(real_text_config, KDA_LAYER_TYPE)
    marker = ".self_attn."

    mapped = _self_attn_keys(real_referenced, kda_layers)
    mapped_leaves = {_leaf_after(key, marker) for key in mapped}

    fixture = _self_attn_keys(frozenset(real_in_scope), kda_layers)
    fixture_leaves = {_leaf_after(key, marker) for key in fixture}

    assert len(kda_layers) == 34
    assert mapped_leaves == fixture_leaves  # the checkpoint is the authority
    assert len(mapped_leaves) == 15
    assert len(mapped) == 15 * len(kda_layers) == 510
    assert scale_keys(mapped) == ()
    # And the absence is the checkpoint's, not the mapping's opinion of it.
    assert scale_keys(fixture) == ()
    # No leaf survives from the old convention.
    assert not any("linear_attn" in key for key in real_referenced)

    _record(
        c078e_kda_layers=len(kda_layers),
        c078e_kda_keys=len(mapped),
        c078e_kda_distinct_leaves=sorted(mapped_leaves),
        c078e_kda_scale_keys=len(scale_keys(mapped)),
    )


# --------------------------------------------------------------------------- #
# (f) The mHC family: 270 keys, 6 per layer over 0-44, 0 on the MTP layer.
# --------------------------------------------------------------------------- #


def test_skeleton_real_mhc_family_is_six_per_layer_and_none_on_the_mtp_layer(
    real_text_config, real_referenced, real_weight_map
) -> None:
    """(f) 270 = 6 x 45, zero on layer 45, and no longer declared absent.

    The declaration this replaces was honest and wrong: the leaf names were
    settled in the published index all along. The per-layer six is checked on
    every layer rather than in aggregate, so 6-and-0 on two layers could not
    average into the total.
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

    # 0 on the MTP layer, read off the FIXTURE (that layer is not in the map).
    mtp_mhc = [
        key
        for key in real_weight_map
        if key.startswith(MTP_LAYER_PREFIX) and key.rsplit(".", 1)[1] in leaves
    ]
    assert mtp_mhc == []

    # The family is no longer declared absent, and it is tagged.
    assert "multi_hyper_connections" not in ABSENT_KEY_FAMILIES
    assert KEY_FAMILY_PROVENANCE["multi_hyper_connections"] == GROUNDED

    _record(
        c078f_mhc_leaves=leaves,
        c078f_mhc_keys=len(mapped),
        c078f_mhc_per_layer=sorted({len(v) for v in per_layer.values()}),
        c078f_mhc_on_mtp_layer=len(mtp_mhc),
    )


# --------------------------------------------------------------------------- #
# (g) The DSA half: 198 keys over 11 layers, exactly 44 scaled, four named.
# --------------------------------------------------------------------------- #


def test_skeleton_real_dsa_half_is_eighteen_leaves_with_four_scaled(
    real_text_config, real_referenced, real_in_scope
) -> None:
    """(g) 198 = 18 x 11 keys, of which exactly 44 = 4 x 11 carry a scale.

    18 counts TENSOR LEAVES as a key map emits them -- 14 tensors, four of which
    have a ``weight_scale_inv`` companion -- and not sub-modules; the same 18 sit
    under 12 distinct sub-modules. The four scaled leaves are named, because
    asking for a scale the checkpoint does not supply is one of the four defects
    this increment repairs.
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

    _record(
        c078g_dsa_layers=len(dsa_layers),
        c078g_dsa_keys=len(mapped),
        c078g_dsa_scale_keys=len(scale_keys(mapped)),
        c078g_dsa_scaled_leaves=scaled,
        c078g_dsa_distinct_leaves=len(per_layer_leaves),
    )


# --------------------------------------------------------------------------- #
# (h) Both fixtures are pinned by digest and byte count -- 2/2.
# --------------------------------------------------------------------------- #


def test_skeleton_real_fixtures_are_pinned_by_digest() -> None:
    """(h) 2/2: one sha256 and one byte count per fixture, read off disk.

    The expected values are the VENDOR's own numbers, fixed before this
    increment ran, and they can be used as expected values only because each
    fixture is a byte-identical copy of the published file. This is what makes
    plan section 11 row A.3 mechanical for both files instead of a promise: a
    later hand cannot quietly edit either fixture to make a comparison pass.

    The two provenance sidecars are deliberately NOT hashed and add no conjunct:
    a sidecar says where the bytes came from, and what the bytes ARE is what
    this item settles.
    """
    readings = []
    for path, want_sha, want_bytes in (
        (REAL_INDEX_PATH, VENDOR_INDEX_SHA256, VENDOR_INDEX_BYTES),
        (REAL_CONFIG_PATH, VENDOR_CONFIG_SHA256, VENDOR_CONFIG_BYTES),
    ):
        data = path.read_bytes()
        got_sha = hashlib.sha256(data).hexdigest()
        assert got_sha == want_sha, f"{path.name}: sha256 {got_sha} != {want_sha}"
        assert len(data) == want_bytes, f"{path.name}: {len(data)} != {want_bytes} B"
        readings.append({"fixture": path.name, "sha256": got_sha, "bytes": len(data)})

    assert len(readings) == 2

    # Each fixture has its provenance sidecar beside it, carrying the same
    # numbers this item just measured. Read, not hashed.
    for reading in readings:
        sidecar = FIXTURES_DIR / f"{reading['fixture']}.provenance.json"
        assert sidecar.is_file(), f"missing provenance sidecar for {reading['fixture']}"
        recorded = json.loads(sidecar.read_text())
        assert recorded["sha256"] == reading["sha256"]
        assert recorded["bytes"] == reading["bytes"]
        assert recorded["fixture_form"] == "BYTE-IDENTICAL COPY"
        assert "ZERO network access was used to build this file" in recorded["note"]

    _record(c078h_fixture_digests=readings, c078h_fixtures_pinned=len(readings))


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
from vllm_neuron.model.glm5_next import weight_loaders_fp8 as _fp8_module
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    MINVAL,
    block_agreement,
    block_grid_shape,
    blockwise_scale_loader,
    compensate_block_scales,
    dequantise_blockwise,
    downscale_fp8_weight_bytes,
    needs_240_downscale,
    report_floored_blocks,
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


def _fp8_representable_magnitudes() -> torch.Tensor:
    """Every finite positive magnitude of ``float8_e4m3fn``, enumerated not derived.

    Read off the format itself: all 256 byte values are viewed as the dtype, the
    non-finite ones are dropped, and the absolute values are de-duplicated. That is a
    measurement of the grid rather than a restatement of the spec, so a torch build
    with a different grid would move this fixture instead of silently disagreeing with
    it.

    Measured on the campaign venv (torch 2.11.0): **254** finite bytes, **126**
    positive magnitudes, smallest ``0.001953125``, largest ``448.0``, and **seven**
    subnormals below ``2**-6``.
    """
    every_byte = (
        torch.arange(256, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    finite = every_byte[torch.isfinite(every_byte)]
    magnitudes = torch.unique(finite.abs())
    return magnitudes[magnitudes > 0]


def _fp8_full_range_tile(variant: int) -> torch.Tensor:
    """One `[128,128]` fp32 tile holding EVERY representable OCP e4m3fn magnitude.

    REBUILT BY ``inc-glm53f-012``'s R2 ROUND, for finding
    ``B08-F2-fixture-misses-low-end-grid-understates-disclosed-error``. It used to be
    `torch.linspace(-448, +448, 16384)` cast to fp8, and its docstring claimed the
    cast collapsed those samples onto "essentially every representable magnitude".
    That is false, and the review measured why: the ramp's sample spacing is about
    0.0547, so no sample lands below it, and the cast reaches **88 of the 126**
    positive magnitudes. The 38 it misses are the entire low end, including all seven
    subnormals. The consequence was not cosmetic -- the per-element figure this
    fixture produces was routed to the lead as "the measured per-element worst", and
    over the missing low end the real worst is thirteen times larger.

    The tile is now built from the enumerated grid itself
    (:func:`_fp8_representable_magnitudes`): the 126 positive magnitudes, their
    negatives and zero, which is 253 values, repeated to fill 16384 elements. 16384 is
    64 whole cycles of 253 plus 192, so coverage does not depend on where the
    truncation falls. Measured: the tile covers **126 of 126** magnitudes, against 88
    for the old ramp, and its absolute maximum is still exactly ``448``.

    Two properties the conjuncts lean on are preserved on purpose. Every tile's
    absolute maximum is ``448``, which is what a per-block quantiser produces and what
    C1's normalisation leans on. And ``256.0`` is still present -- the byte whose
    re-quantisation gives the worst *absolute* error, 12.8, which is what C1 actually
    measures. C1's block-normalised worst is unchanged at ``0.0285714`` under both
    fixtures, so this rebuild moves the disclosed per-element figure and no tolerance.

    The four variants are range-preserving rearrangements, as before.
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

    PROPERTY 3 IS NEW, from finding ``B08-F2``: the fixture's magnitude COVERAGE. The
    old fixture's docstring claimed it held essentially every representable magnitude
    and, as the review put it, that claim "is also asserted by no test" -- this test
    checked the maxima and the 240 fraction and said nothing about coverage, so the
    claim could be false for as long as nobody recomputed it. It is now a reading:
    every one of the 126 positive magnitudes must be present, the seven subnormals
    among them, and the count is recorded so a future fixture change shows up as a
    number rather than as a still-green test.
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

    # Property 3 -- coverage, per finding B08-F2. Compared against the grid this
    # fixture is built from, not against a literal count, so the assertion states the
    # property ("all of them") rather than a number that would need editing if a torch
    # build ever changed the grid.
    representable = _fp8_representable_magnitudes()
    present = torch.unique(dense.abs())
    present = present[present > 0]
    missing = representable[~torch.isin(representable, present)]
    subnormals = representable[representable < 2.0 ** -6]
    subnormals_present = int(torch.isin(subnormals, present).sum())
    _record_fp8(
        fixture_representable_magnitudes=int(representable.numel()),
        fixture_magnitudes_present=int(present.numel()),
        fixture_magnitudes_missing=[float(x) for x in missing],
        fixture_subnormals_total=int(subnormals.numel()),
        fixture_subnormals_present=subnormals_present,
        fixture_smallest_magnitude_present=float(present.min()),
    )
    assert missing.numel() == 0, (
        f"the fixture misses {missing.numel()} of {representable.numel()} "
        f"representable magnitudes, smallest missing {float(missing.min())}; the "
        f"per-element figure this fixture discloses would be taken over a subset"
    )
    assert subnormals_present == subnormals.numel(), (
        f"only {subnormals_present} of {subnormals.numel()} subnormals are present; "
        f"the smallest subnormal is where the 240/448 squeeze is worst per element"
    )
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
        #
        # RE-REPORTED BY THE R2 ROUND, for finding B08-F2. This figure was 0.0667 and
        # was routed to the lead as "the measured per-element worst". It was the worst
        # over the OLD fixture's 88-magnitude subset, not over the grid: the ramp
        # reached no magnitude below 0.0273, and the squeeze is at its worst on the
        # smallest subnormal. On the rebuilt fixture the same code measures 0.8667 --
        # thirteen times larger, and now over all 126 magnitudes. NO TOLERANCE MOVED:
        # this value gates nothing, and C1's block-normalised worst is 0.0285714 under
        # both fixtures.
        c1_worst_element_relative_disclosed=max(
            r.worst_element_relative for r in reports
        ),
        # The population the figure was taken over, recorded beside it so the number
        # cannot be quoted again without its denominator.
        c1_worst_element_population_magnitudes=int(
            _fp8_representable_magnitudes().numel()
        ),
        c1_worst_element_population_is_the_full_grid=True,
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


class _Fp8RecordingHandler(logging.Handler):
    """Collects formatted records off the loader module's own logger object."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _capture_fp8_loader_log():
    """Attach to ``weight_loaders_fp8.logger`` directly.

    Directly, not through ``caplog``, for the reason
    ``test_platform_hybrid_config.py:217-220`` already records: ``caplog``'s handler
    lives on the root logger, so a propagation setting anywhere in vLLM's logging
    configuration would make this differential read 0 for a reason that has nothing to
    do with the code under test.
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


def test_fp8_downscale_scale_loader_records_an_engaged_floor(
    fp8_downscale_scales,
) -> None:
    """A load that floors a tile leaves a record on the load path, not just in a test.

    THE FINDING THIS ANSWERS. ``B08-F1-minval-floor-silent-on-loader-path``: the module
    said twice that the floor is "reported per block rather than applied silently", but
    ``blockwise_scale_loader``'s transform returned ``.scale_inv`` and dropped the whole
    census, so a load that inflated an entire tile of weights by about 5.4x left no
    trace anywhere outside this file. The review's requirement was that the fix
    "demonstrate that a load which floors a tile leaves a trace outside the test suite".

    THE READING IS A DIFFERENTIAL, not the presence of one message. The same loader is
    driven twice through the same fake slice: once with the declared grid, where no tile
    needs the floor and the record count must be 0, and once with one tile's scale set
    to ``1e-9``, where it must be 1. A test that only asserted the second half would
    pass against a loader that warned on every load, which would be its own defect.

    IT ALSO CHECKS WHAT THE MESSAGE SAYS, because the finding asked for a warning that
    "names the parameter and the floored tiles". A record that fired but named neither
    would satisfy a count and not the requirement.
    """
    # Leg 1 -- the negative half: the declared grid floors nothing, so it says nothing.
    quiet_loader = blockwise_scale_loader(FP8_LOADER_PARAM_NAME)
    with _capture_fp8_loader_log() as quiet:
        quiet_grid = quiet_loader.load([FakeSafeSlice(fp8_downscale_scales)], 0)
    quiet_records = [m for m in quiet.messages if FP8_FLOOR_MARKER in m]
    _record_fp8(f1_records_on_an_unfloored_load=len(quiet_records))
    assert quiet_records == [], (
        "the loader reported a floor on a grid where no tile needed one, so the "
        "record does not distinguish a floored load from a clean one"
    )

    # Leg 2 -- the positive half: one tile below the floor, through the same loader.
    scales = [list(row) for row in FP8_BLOCK_SCALES]
    scales[FP8_TINY_BLOCK[0]][FP8_TINY_BLOCK[1]] = FP8_TINY_SCALE
    tiny_grid = _fp8_scale_grid(tuple(tuple(row) for row in scales))

    loud_loader = blockwise_scale_loader(FP8_LOADER_PARAM_NAME)
    with _capture_fp8_loader_log() as loud:
        loaded = loud_loader.load([FakeSafeSlice(tiny_grid)], 0)
    loud_records = [m for m in loud.messages if FP8_FLOOR_MARKER in m]
    _record_fp8(
        f1_records_on_a_floored_load=len(loud_records),
        f1_message=loud_records[0] if loud_records else None,
    )
    assert len(loud_records) == 1, (
        f"a load that floored tile {FP8_TINY_BLOCK} must leave exactly one record; "
        f"got {len(loud_records)}"
    )

    # Leg 3 -- the record says which parameter and which tile, per the finding.
    message = loud_records[0]
    assert FP8_LOADER_PARAM_NAME in message, message
    assert f"({FP8_TINY_BLOCK[0]},{FP8_TINY_BLOCK[1]})" in message, message
    assert "MINVAL" in message, message

    # Leg 4 -- the returned tensor is unchanged by the reporting. The floor still
    # floors; the only difference is that it is now audible.
    assert torch.equal(loaded, compensate_block_scales(tiny_grid).scale_inv)
    assert torch.equal(loaded[FP8_TINY_BLOCK], _fp8_as_stored(MINVAL))
    assert torch.equal(quiet_grid, compensate_block_scales(fp8_downscale_scales).scale_inv)

    # Leg 5 -- an unnamed caller gets a record that says so, rather than "None".
    with _capture_fp8_loader_log() as unnamed:
        blockwise_scale_loader().load([FakeSafeSlice(tiny_grid)], 0)
    unnamed_records = [m for m in unnamed.messages if FP8_FLOOR_MARKER in m]
    _record_fp8(f1_unnamed_message=unnamed_records[0] if unnamed_records else None)
    assert len(unnamed_records) == 1
    assert "an unnamed parameter" in unnamed_records[0], unnamed_records[0]
    assert "None" not in unnamed_records[0], unnamed_records[0]

    # Leg 6 -- the reporter's own return value distinguishes the two cases, so a
    # caller that wants the fact without the log line can have it.
    assert report_floored_blocks(compensate_block_scales(tiny_grid)) is True
    assert report_floored_blocks(compensate_block_scales(fp8_downscale_scales)) is False


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
    #
    # `inc-glm53f-079` SUPERSEDED the two assertions that stood here. They said
    # `get_scheme(0, "linear_attn.out_proj")` and `get_scheme(None, "lm_head")`
    # were both block-FP8. Both are false against the real checkpoint --
    # `lm_head` is skip-listed and has no scale key, and `linear_attn.out_proj`
    # is not a name this checkpoint contains at all -- and the standing answers
    # for those call shapes are that increment's conjunct (c), measured against
    # the checkpoint. What is still this item's to say is that a spec parsed from
    # a block config carrying NO skip list has an empty one, and that both call
    # shapes then get `linear_scheme`.
    assert spec.modules_to_not_convert == ()
    assert spec.get_scheme(0, "model.layers.0.self_attn.o_proj") is spec.linear_scheme
    assert spec.get_scheme(None, "lm_head") is spec.linear_scheme

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


# =========================================================================== #
# inc-glm53f-079 -- WP6 REPAIR: the checkpoint's FP8 skip list is honoured
# =========================================================================== #
#
# WHY THIS SECTION IS APPENDED AT THE END
# ---------------------------------------
# `inc-glm53f-079` is the FOURTH writer on this file (plan section 11 row A.1,
# partitioned by pytest selection and by concern: `-011` owns the shard index and
# the key map's shape, `-078` the checkpoint-key namespace and the family names,
# `-012` the `-k fp8_downscale` numerics, and this increment whether a family asks
# for a scale companion AT ALL). It appends rather than inserts, so it moves no
# landed line of the three sections above it. Every item below carries `skeleton`
# and none carries `fp8_downscale`, so the two selections stay disjoint.
#
# WHAT IS BEING SETTLED
# ---------------------
# The checkpoint names 1,509 modules to keep in BF16 and, before this increment,
# the plugin read none of them: `config.py` lifted three of the five
# `quantization_config` fields, and `QuantizationSpec.get_scheme` discarded both
# of its arguments and answered "block-FP8" for `lm_head`, for the KDA
# projections and for the DSA indexer alike. The four items below are the four
# counted conjuncts of the declared acceptance, ONE ITEM EACH, NO PARAMETRIZE
# (section 6 rule 6), and every denominator is derived from a fixture.
#
# THE AGREEMENT IS BETWEEN TWO INSTRUMENTS, WHICH IS WHAT MAKES IT FALSIFIABLE
# ---------------------------------------------------------------------------
# Conjunct (b) compares the checkpoint's DECLARED policy -- the 1,509-entry list
# in `hf-config.json` -- against the checkpoint's ACTUAL scale keys -- presence
# or absence of a `weight_scale_inv` companion in the 76,108-key index. Neither
# side is computed from the other, both are vendor files this directory pins by
# digest (`-078` conjunct (h)), and a predicate that answered one way for
# everything fails on one arm or the other.
#
# THE TWO NAMESPACES (`-078`)
# ---------------------------
# The skip entries are MODULE-namespace (`model.layers.0.self_attn.q_proj`); the
# index keys are CHECKPOINT-namespace (`model.language_model.layers.0...`). The
# fork's substring rule resolves one against the other only because
# `language_model` ends in the literal `model`. That alignment is recorded rather
# than designed, so conjunct (c) asserts the qualified spellings out loud instead
# of leaving the match implicit.

from vllm_neuron.model.glm5_next.quantization import keeps_bf16

# --------------------------------------------------------------------------- #
# Declared values. Each is CROSS-CHECKED against a number derived from a
# fixture, never used as the expectation on its own.
# --------------------------------------------------------------------------- #

#: The five `quantization_config` fields the published config carries, and the
#: two the adapter used to drop.
C079_DECLARED_QUANT_CONFIG_KEYS = 5
C079_DECLARED_SKIP_ENTRIES = 1509
C079_DECLARED_FMT = "e4m3"

#: Conjunct (b)'s population and its two non-vacuity arms (D1.5).
C079_DECLARED_BASE_TENSORS = 37534
C079_DECLARED_QUANTIZED = 36467
C079_DECLARED_UNQUANTIZED = 1067

#: How the checkpoint spells a scale companion: the base key plus this tail.
#: Appending a tail rather than substituting a leaf makes the rule TOTAL over
#: every base tensor -- a parameter that is not a `.weight` (`A_log`, `dt_bias`,
#: the six hyper-connection tensors) gets a name the index cannot contain, which
#: is the right answer, since it has no scale companion.
C079_SCALE_COMPANION_TAIL = "_scale_inv"

#: Conjunct (d)'s firing control. `shared_experts` is a token the real skip list
#: does NOT carry, so adding it makes the suppressed count move; the number of
#: requests it removes is derived from the config below, not typed in here.
C079_SYNTHETIC_SKIP_TOKEN = "shared_experts"
C079_DECLARED_SYNTHETIC_DROP = 126


@pytest.fixture(scope="module")
def real_raw_config() -> dict[str, Any]:
    """The published config, parsed. `-078` landed the file; this reads it."""
    return json.loads(REAL_CONFIG_PATH.read_text())


@pytest.fixture(scope="module")
def real_quant_config(real_raw_config) -> dict[str, Any]:
    """The published `quantization_config` block, verbatim."""
    return real_raw_config["quantization_config"]


@pytest.fixture(scope="module")
def real_config() -> Glm5NextConfig:
    """The config the REAL adapter produces from the REAL published config.

    `Glm5NextConfig.from_configs` on a fresh parse -- the same entry point
    `test_config.py` drives -- so conjunct (a) measures the adapter rather than a
    dict this file assembled. Fresh parse, so nothing the adapter touches can
    reach the expectation side of an assertion.
    """
    return Glm5NextConfig.from_configs(json.loads(REAL_CONFIG_PATH.read_text()))


@pytest.fixture(scope="module")
def real_spec(real_config) -> QuantizationSpec:
    """The spec built through the whole chain: file -> config -> spec.

    `from_model_config` is the bridge the increment taught to forward the fourth
    name, so driving it here means a bridge that forwarded three of four would
    fail these items rather than pass them with an empty skip list.
    """
    spec = QuantizationSpec.from_model_config(real_config)
    assert spec is not None
    return spec


@pytest.fixture(scope="module")
def real_base_tensors(real_in_scope) -> list[str]:
    """The in-scope BASE tensors: every in-scope key that is not a scale key."""
    return sorted(
        key for key in real_in_scope if not key.endswith(f".{SCALE_SUFFIX}")
    )


def _c079_layer_index(key: str) -> int | None:
    """The layer number inside a key, or `None` for a key outside any block."""
    parts = key.split(".")
    if "layers" in parts:
        position = parts.index("layers") + 1
        if position < len(parts) and parts[position].isdigit():
            return int(parts[position])
    return None


def _c079_companion(key: str) -> str:
    """The scale key the checkpoint would carry for `key`, spelled its way."""
    return key + C079_SCALE_COMPANION_TAIL


def _c079_score(
    spec: QuantizationSpec, base_keys: list[str], index_keys: frozenset[str]
) -> dict[str, Any]:
    """Score one spec against the index over every base tensor.

    One pass, four counters: agreements and disagreements of the two
    independently produced answers, and the population split that keeps both
    arms of the comparison non-empty.
    """
    agreements = disagreements = quantized = unquantized = 0
    examples: list[str] = []
    for key in base_keys:
        has_scale = _c079_companion(key) in index_keys
        unquantised = spec.get_scheme(_c079_layer_index(key), key) is QuantScheme.NONE
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


def _c079_requested_scales(mappings: dict[str, str | list[str]]) -> list[str]:
    """Every `weight_scale_inv` key the mapping asks the checkpoint for."""
    return sorted(
        {
            key
            for value in mappings.values()
            for key in (value if isinstance(value, list) else [value])
            if key.endswith(f".{SCALE_SUFFIX}")
        }
    )


def _c079_base_of(scale_key: str) -> str:
    """The base tensor a requested scale key belongs to."""
    assert scale_key.endswith(C079_SCALE_COMPANION_TAIL)
    return scale_key[: -len(C079_SCALE_COMPANION_TAIL)]


def test_skeleton_config_lifts_the_skip_list_and_the_fp8_format(
    real_quant_config, real_config
) -> None:
    """(a) 5/5: the adapter lifts all five `quantization_config` fields.

    The two it used to drop are `modules_to_not_convert` and `fmt`. Both
    expectations are READ OFF the published config rather than typed in: the
    entry count is `len()` of the fixture's own list, and the format string is
    the fixture's own value. The declared numbers are asserted beside them as a
    cross-check that this is the file the plan measured -- legitimate only
    because `-078` pins the fixture by the vendor's digest.
    """
    assert sorted(real_quant_config) == [
        "activation_scheme",
        "fmt",
        "modules_to_not_convert",
        "quant_method",
        "weight_block_size",
    ]
    assert len(real_quant_config) == C079_DECLARED_QUANT_CONFIG_KEYS

    declared_skip = real_quant_config["modules_to_not_convert"]
    assert len(declared_skip) == C079_DECLARED_SKIP_ENTRIES

    # The lift loses no entry and reorders none.
    assert real_config.modules_to_not_convert == declared_skip
    assert real_config.fmt == real_quant_config["fmt"] == C079_DECLARED_FMT

    # The three fields that were already lifted still are.
    assert real_config.quant_method == real_quant_config["quant_method"]
    assert real_config.activation_scheme == real_quant_config["activation_scheme"]
    assert real_config.weight_block_size == real_quant_config["weight_block_size"]

    # The list's composition, measured, because `get_scheme` documents that a
    # bare module name cannot match a qualified entry and conjunct (c) asserts
    # it. Nine entries are bare tokens; the other 1,500 are dotted paths.
    bare = sorted(entry for entry in declared_skip if "." not in entry)
    assert len(bare) == 9
    assert "lm_head" in bare
    assert len(declared_skip) - len(bare) == 1500

    _record(
        c079a_quant_config_keys=sorted(real_quant_config),
        c079a_skip_entries=len(declared_skip),
        c079a_fmt=real_config.fmt,
        c079a_bare_entries=bare,
        c079a_qualified_entries=len(declared_skip) - len(bare),
    )


def test_skeleton_skip_list_agrees_with_the_index_scale_keys(
    real_spec, real_base_tensors, real_in_scope, real_quant_config
) -> None:
    """(b) 37,534/37,534: the declared policy and the actual scale keys agree.

    `get_scheme` returns the unquantized scheme for a base tensor exactly when
    that tensor has no scale companion in the index. The left side is the
    1,509-entry list in the config; the right side is a key census over the
    76,108-key index. Neither is derived from the other.

    D1.5 CONTROL, IN THIS ITEM: the same census against a spec built from the
    same block with the skip list REMOVED. The disagreement counter has to
    become non-zero -- it becomes 1,067, one per BF16 tensor -- or the agreement
    above would be true of any predicate at all.
    """
    # The two spellings of one suffix, pinned together so neither can drift.
    assert _c079_companion("m.weight") == f"m.{SCALE_SUFFIX}"

    index_keys = frozenset(real_in_scope)
    scored = _c079_score(real_spec, real_base_tensors, index_keys)

    assert scored["disagreements"] == 0, scored["examples"]
    assert scored["agreements"] == len(real_base_tensors)
    assert len(real_base_tensors) == C079_DECLARED_BASE_TENSORS
    assert scored["quantized"] == C079_DECLARED_QUANTIZED
    assert scored["unquantized"] == C079_DECLARED_UNQUANTIZED
    assert scored["quantized"] + scored["unquantized"] == len(real_base_tensors)
    assert len(real_spec.modules_to_not_convert) == C079_DECLARED_SKIP_ENTRIES

    blind_spec = QuantizationSpec.from_hf_quantization_config(
        {
            key: value
            for key, value in real_quant_config.items()
            if key != "modules_to_not_convert"
        }
    )
    assert blind_spec is not None
    assert blind_spec.modules_to_not_convert == ()
    blind = _c079_score(blind_spec, real_base_tensors, index_keys)
    assert blind["disagreements"] == C079_DECLARED_UNQUANTIZED > 0
    assert blind["agreements"] == C079_DECLARED_QUANTIZED

    _record(
        c079b_base_tensors=len(real_base_tensors),
        c079b_agreements=scored["agreements"],
        c079b_disagreements=scored["disagreements"],
        c079b_quantized=scored["quantized"],
        c079b_unquantized=scored["unquantized"],
        c079b_control_disagreements=blind["disagreements"],
        c079b_control_agreements=blind["agreements"],
    )


def test_skeleton_three_named_cases_pin_the_scheme(
    real_spec, real_text_config, real_in_scope
) -> None:
    """(c) 3/3: two projections the checkpoint keeps in BF16, and one it does not.

    The third case is why this is a discrimination and not a refusal: a
    predicate that answered "unquantized" for every name would fail on
    `q_a_proj`. Each case is also checked against the index, so the expected
    answer traces to the checkpoint rather than to this docstring.

    THE BLOCK SPELLS THESE CASES LEAF-STYLE (`layers.0.self_attn.q_proj`) AND
    THAT SPELLING MATCHES NO ENTRY. The 1,500 qualified entries are dotted paths
    under `model.`, so a bare name cannot be a substring of one; the item
    asserts every case in both qualified namespaces and asserts the bare
    spelling's answer out loud rather than leaving the requirement implicit.
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
    for short, want, family in cases:
        layer = int(short.split(".")[1])
        assert real_text_config.layer_types[layer] == family

        module_ns = f"model.{short}"
        checkpoint_ns = f"model.language_model.{short}"
        for name in (module_ns, checkpoint_ns, f"{checkpoint_ns}.weight"):
            got = real_spec.get_scheme(layer, name)
            assert got is want, f"{name}: {got.name} != {want.name}"

        # The index's own answer for the same projection.
        has_scale = _c079_companion(f"{checkpoint_ns}.weight") in real_in_scope
        assert has_scale == (want is QuantScheme.FP8_BLOCK_DYNAMIC)

        # A bare name matches no qualified entry, so it reads as quantized.
        assert real_spec.get_scheme(layer, short) is real_spec.linear_scheme

        rows.append(
            {
                "case": short,
                "layer_type": family,
                "scheme": want.value,
                "index_has_scale_key": has_scale,
            }
        )

    assert len(rows) == 3
    assert len({row["scheme"] for row in rows}) == 2
    _record(c079c_cases=rows, c079c_named_cases=len(rows))


def test_skeleton_no_scale_companion_is_requested_for_a_bf16_tensor(
    real_text_config, real_quant_config, real_in_scope
) -> None:
    """(d) 2/2: no scale request for a BF16 tensor, and none the index lacks.

    Two censuses over the map the real skip list produces: the number of
    requested scale keys whose base tensor the skip list keeps in BF16, and the
    number of requested scale keys the index does not contain. Both are 0, and
    the 36,467 requests that remain are exactly the index's own scale-key
    population -- the non-vacuity arm.

    D1.5 CONTROL, IN THIS ITEM, AND WHY IT IS SYNTHETIC. The suppression is
    measured with a token the real list does not carry, `shared_experts`, and the
    count it removes is derived from the config: one scale request per shared
    expert leaf on each MoE layer. Against the real list alone the counter cannot
    fire, because `-078` already made every BF16 family structurally unquantised
    in this builder -- so switching the real list off changes no request, and a
    control resting on it would prove nothing about this increment's predicate.
    """
    skip = tuple(real_quant_config["modules_to_not_convert"])
    honoured = build_weight_mappings(real_text_config, modules_to_not_convert=skip)
    requested = _c079_requested_scales(honoured)

    kept_bf16 = [key for key in requested if keeps_bf16(_c079_base_of(key), skip)]
    assert kept_bf16 == []
    # The predicate's other spelling -- the module name without the parameter
    # leaf -- so the two forms cannot disagree unnoticed.
    assert [
        key
        for key in requested
        if keeps_bf16(_c079_base_of(key).removesuffix(".weight"), skip)
    ] == []

    absent = [key for key in requested if key not in real_in_scope]
    assert absent == []

    index_scales = {
        key for key in real_in_scope if key.endswith(f".{SCALE_SUFFIX}")
    }
    assert len(requested) == len(index_scales) == C079_DECLARED_QUANTIZED

    probe = skip + (C079_SYNTHETIC_SKIP_TOKEN,)
    moe_layers = (
        real_text_config.num_hidden_layers - real_text_config.first_k_dense_replace
    )
    shared_leaves = 3
    unsuppressed = _c079_requested_scales(
        build_weight_mappings(real_text_config, modules_to_not_convert=())
    )
    fires = [key for key in unsuppressed if keeps_bf16(_c079_base_of(key), probe)]
    assert len(fires) == moe_layers * shared_leaves
    assert len(fires) == C079_DECLARED_SYNTHETIC_DROP > 0

    suppressed = _c079_requested_scales(
        build_weight_mappings(real_text_config, modules_to_not_convert=probe)
    )
    assert [key for key in suppressed if keeps_bf16(_c079_base_of(key), probe)] == []
    assert len(suppressed) == len(requested) - len(fires)

    _record(
        c079d_scale_requests=len(requested),
        c079d_index_scale_keys=len(index_scales),
        c079d_requests_for_a_bf16_tensor=len(kept_bf16),
        c079d_requested_but_absent=len(absent),
        c079d_control_fires=len(fires),
        c079d_control_suppressed_requests=len(suppressed),
        c079d_control_token=C079_SYNTHETIC_SKIP_TOKEN,
    )
