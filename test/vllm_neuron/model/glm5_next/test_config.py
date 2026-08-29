# SPDX-License-Identifier: Apache-2.0
"""
`inc-glm53f-008` acceptance — GLM-5.3-Flash model config.

THE DECLARED PREDICATE (increment plan revision 10, block L3170-3176): parsing
the pinned checkpoint's `config.json` yields **exactly** ten exact equalities,
asserted as **10/10**.

THE COMPOSITION OF THE TEN, stated because the plan pins the number and not its
composition (scope-lap-011 §8.2): the plan's enumerated clauses total ten ONLY
IF the KDA/DSA clause counts as ONE paired equality. It is therefore asserted
here as one pair, `(kda, dsa) == (34, 11)` -- never as two independent
equalities, which would make a correct test report eleven and fail a pinned ten.
The declared number is NOT changed by this file.

FIXTURE PROVENANCE (lead ruling, scope-lap-011 §8.3): the trimmed fixture
derives from `artifacts/run/intake-preflight/03-glm53flash-weights.md`
transcript 3 -- the intake evidence that fetched `config.json` (69,416 B) from
HuggingFace `zai-org/GLM-5.3-Flash` with recorded proof that `main` was the
pinned revision `04c4e9e95c5da8862dced7e5056455116f83a7e0` at fetch time (that
artifact's L28 and L45). This file performs ZERO network access; the fixture's
bytes are pinned by digest below, so a silent edit fails loudly.

FALSIFIABILITY: every counted or compared reading here carries an arm that
would fail if the reading were vacuous. The ten equalities each get a mutation
arm proving the extractor reads the fixture rather than returning a constant;
the 34/11 split is counted from three independent bases; and the two-family
partition is proved exhaustive, because an unrecognised family name would be
dropped by a counting pass and inflate the other family's count silently.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from vllm_neuron.model.glm5_next.config import (
    DSA_LAYER_TYPE,
    KDA_LAYER_TYPE,
    Glm5NextConfig,
    Glm5NextTextConfig,
    default_layer_types,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"

# Pinned so an edit to the fixture cannot silently move a declared value.
FIXTURE_SHA256 = "f3d8790f18a18ffc95015dcc8869ac25c8d49129a383ccd3e0b4d07183bd6802"

# BASE 3 for the layer schedule: the DSA layer indices as ENUMERATED in the
# intake record (03-glm53flash-weights.md L119, `full_attn_layers (DSA)`).
INTAKE_RECORDED_DSA_INDICES = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]

# The plan's declared split and declared conjunct count.
EXPECTED_LAYER_SPLIT = (34, 11)
DECLARED_EQUALITY_COUNT = 10


def _raw() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def raw() -> dict:
    return _raw()


@pytest.fixture(scope="module")
def cfg(raw) -> Glm5NextConfig:
    return Glm5NextConfig.from_configs(copy.deepcopy(raw))


# ---------------------------------------------------------------------------
# The ten conjuncts, as a table: (id, what, extract, expected, mutate)
#
# `mutate` edits a deep copy of the raw HF dict so that THIS conjunct's
# extracted value must change. It is what proves the extractor reads the
# fixture instead of returning a constant.
# ---------------------------------------------------------------------------


def _m_num_hidden_layers(r):
    # num_hidden_layers and layer_types must stay length-consistent, so the
    # mutation drops one layer from both -- otherwise the config rejects the
    # dict before the extractor ever runs.
    r["text_config"]["num_hidden_layers"] = 44
    r["text_config"]["layer_types"].pop()


def _m_layer_split(r):
    # Flip the first KDA layer to DSA: still a valid exhaustive partition of
    # 45 layers, but the pair becomes (33, 12).
    r["text_config"]["layer_types"][0] = DSA_LAYER_TYPE


def _m_kv_lora_rank(r):
    r["text_config"]["kv_lora_rank"] = 256


def _m_qk_rope_head_dim(r):
    r["text_config"]["qk_rope_head_dim"] = 64


def _m_n_routed_experts(r):
    r["text_config"]["n_routed_experts"] = 256


def _m_n_shared_experts(r):
    r["text_config"]["n_shared_experts"] = 2


def _m_num_experts_per_tok(r):
    r["text_config"]["num_experts_per_tok"] = 4


def _m_hc_mult(r):
    r["text_config"]["hc_mult"] = 2


def _m_weight_block_size(r):
    r["quantization_config"]["weight_block_size"] = [64, 64]


def _m_activation_scheme(r):
    r["quantization_config"]["activation_scheme"] = "static"


EQUALITIES = [
    (
        "C01",
        "num_hidden_layers == 45",
        lambda c: c.text_config.num_hidden_layers,
        45,
        _m_num_hidden_layers,
    ),
    (
        "C02",
        "KDA/DSA 3:1 over 45 layers -> (kda, dsa) == (34, 11)  [ONE paired equality]",
        lambda c: c.text_config.attention_layer_split,
        EXPECTED_LAYER_SPLIT,
        _m_layer_split,
    ),
    (
        "C03",
        "kv_lora_rank == 512",
        lambda c: c.text_config.kv_lora_rank,
        512,
        _m_kv_lora_rank,
    ),
    (
        "C04",
        "qk_rope_head_dim == 0",
        lambda c: c.text_config.qk_rope_head_dim,
        0,
        _m_qk_rope_head_dim,
    ),
    (
        "C05",
        "n_routed_experts == 288",
        lambda c: c.text_config.n_routed_experts,
        288,
        _m_n_routed_experts,
    ),
    (
        "C06",
        "n_shared_experts == 1",
        lambda c: c.text_config.n_shared_experts,
        1,
        _m_n_shared_experts,
    ),
    (
        "C07",
        "num_experts_per_tok == 8",
        lambda c: c.text_config.num_experts_per_tok,
        8,
        _m_num_experts_per_tok,
    ),
    ("C08", "hc_mult == 4", lambda c: c.text_config.hc_mult, 4, _m_hc_mult),
    (
        "C09",
        "weight_block_size == [128, 128]",
        lambda c: c.weight_block_size,
        [128, 128],
        _m_weight_block_size,
    ),
    (
        "C10",
        'activation_scheme == "dynamic"',
        lambda c: c.activation_scheme,
        "dynamic",
        _m_activation_scheme,
    ),
]


def test_fixture_is_the_pinned_bytes():
    """Tamper detection: the fixture's digest is pinned in this file."""
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    print(f"\n[fixture] path={FIXTURE_PATH}")
    print(f"[fixture] sha256={digest}")
    print(f"[fixture] bytes={FIXTURE_PATH.stat().st_size}")
    assert digest == FIXTURE_SHA256


def test_the_declared_conjunct_table_holds_exactly_ten_equalities():
    """The pinned 10 is the table's own length, so the two cannot drift apart."""
    ids = [row[0] for row in EQUALITIES]
    assert len(ids) == len(set(ids)), f"duplicate conjunct ids: {ids}"
    assert len(EQUALITIES) == DECLARED_EQUALITY_COUNT


def test_ten_exact_equalities(cfg):
    """THE DECLARED ACCEPTANCE: 10/10 exact equalities, all evaluated.

    Every conjunct is evaluated before any assertion fires, so a failure
    reports the whole table rather than stopping at the first mismatch.
    """
    results = []
    for cid, what, extract, expected, _ in EQUALITIES:
        actual = extract(cfg)
        ok = actual == expected and type(actual) is type(expected)
        results.append((cid, what, actual, expected, ok))
        print(f"[{cid}] {'PASS' if ok else 'FAIL'}  {what}  actual={actual!r}")

    passed = [r for r in results if r[4]]
    failed = [r for r in results if not r[4]]
    print(f"\n[10/10] passed={len(passed)}/{len(results)}  failed={len(failed)}")

    assert not failed, "mismatched conjuncts: " + "; ".join(
        f"{cid}: actual={actual!r} expected={expected!r}"
        for cid, _, actual, expected, _ in failed
    )
    assert len(passed) == DECLARED_EQUALITY_COUNT


@pytest.mark.parametrize("cid,what,extract,expected,mutate", EQUALITIES)
def test_each_equality_is_falsifiable(raw, cid, what, extract, expected, mutate):
    """Non-vacuity: mutating the fixture must move THIS conjunct's value.

    A conjunct that still reads its expected value from a mutated config is
    reading something other than the config, and its pass means nothing.
    """
    mutated = copy.deepcopy(raw)
    mutate(mutated)
    actual = extract(Glm5NextConfig.from_configs(mutated))
    print(f"[{cid}] mutated -> {actual!r} (expected to differ from {expected!r})")
    assert actual != expected, f"{cid} ({what}) is vacuous: unmoved by its mutation"


def test_layer_schedule_agrees_across_three_independent_bases(cfg):
    """The 34/11 split, counted from three bases that do not share a mechanism.

    Base 1 -- the checked-in 45-entry `layer_types` array in the fixture.
    Base 2 -- the config's generated 3:1 interleave rule.
    Base 3 -- the DSA layer indices enumerated in the intake record.

    Three bases, because a single counting pass is a claim about the data's
    surface form and inherits that claim's risk.
    """
    text = cfg.text_config
    base1 = text.attention_layer_split
    generated = default_layer_types(text.num_hidden_layers)
    base2 = (
        sum(1 for t in generated if t == KDA_LAYER_TYPE),
        sum(1 for t in generated if t == DSA_LAYER_TYPE),
    )
    base3 = (
        text.num_hidden_layers - len(INTAKE_RECORDED_DSA_INDICES),
        len(INTAKE_RECORDED_DSA_INDICES),
    )
    print(f"\n[split] base1(fixture array)={base1}")
    print(f"[split] base2(generated 3:1 rule)={base2}")
    print(f"[split] base3(intake DSA indices)={base3}")
    print(f"[split] dsa indices measured={text.dsa_layer_indices}")

    assert base1 == EXPECTED_LAYER_SPLIT
    assert base2 == EXPECTED_LAYER_SPLIT
    assert base3 == EXPECTED_LAYER_SPLIT
    assert text.layer_types == generated
    assert text.dsa_layer_indices == INTAKE_RECORDED_DSA_INDICES
    # The pair sums to the layer count, which is what makes it ONE fact.
    assert sum(base1) == text.num_hidden_layers == 45


def test_the_two_family_partition_is_exhaustive(cfg):
    """No layer falls outside {KDA, DSA}: the complement is empty.

    This is the arm that stops the silent-narrowing fault. Family names are
    compared by EQUALITY -- 'attention' is a substring of both names, so a
    substring screen would mis-partition the stack and still report 45.
    """
    text = cfg.text_config
    known = {KDA_LAYER_TYPE, DSA_LAYER_TYPE}
    complement = [t for t in text.layer_types if t not in known]
    print(f"\n[partition] families observed={sorted(set(text.layer_types))}")
    print(f"[partition] complement (outside {sorted(known)})={complement}")
    assert complement == []
    assert len(text.kda_layer_indices) + len(text.dsa_layer_indices) == len(
        text.layer_types
    )
    assert set(text.kda_layer_indices).isdisjoint(text.dsa_layer_indices)


def test_an_unrecognised_layer_family_is_rejected():
    """The partition guard is real, not decorative."""
    schedule = default_layer_types(45)
    schedule[0] = "sliding_window_attention"
    with pytest.raises(ValueError, match="unrecognised attention families"):
        Glm5NextTextConfig(num_hidden_layers=45, layer_types=schedule)


def test_a_length_mismatched_schedule_is_rejected():
    """A schedule that disagrees with num_hidden_layers cannot pass silently."""
    with pytest.raises(ValueError, match="layer_types has 44 entries"):
        Glm5NextTextConfig(num_hidden_layers=45, layer_types=default_layer_types(44))


def test_no_weights_are_referenced_by_the_fixture(raw):
    """The fixture is config only, no weights (plan L3173)."""
    blob = json.dumps(raw)
    for banned in ("safetensors", "model.safetensors.index.json", "weight_map"):
        assert banned not in blob, f"fixture references weights via {banned!r}"
