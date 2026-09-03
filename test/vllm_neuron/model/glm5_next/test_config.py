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

`inc-glm53f-080` re-transcribed the fixture's `text_config` from the in-repo
copy of that same vendor config, `fixtures/hf-config.json`, which `inc-glm53f-078`
lands and pins by its own digest. The pin below therefore moved; the ten
equalities above did not, because every value they read is unchanged. The
`-080` section at the end of this file carries that increment's four conjuncts.

FALSIFIABILITY: every counted or compared reading here carries an arm that
would fail if the reading were vacuous. The ten equalities each get a mutation
arm proving the extractor reads the fixture rather than returning a constant;
the 34/11 split is counted from three independent bases; and the two-family
partition is proved exhaustive, because an unrecognised family name would be
dropped by a counting pass and inflate the other family's count silently.
"""

import contextlib
import copy
import hashlib
import json
import logging
from dataclasses import fields
from pathlib import Path

import pytest

from vllm_neuron.model.glm5_next import config as config_module
from vllm_neuron.model.glm5_next.config import (
    DSA_LAYER_TYPE,
    KDA_LAYER_TYPE,
    Glm5NextConfig,
    Glm5NextTextConfig,
    default_layer_types,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"

# Pinned so an edit to the fixture cannot silently move a declared value.
# Moved by `inc-glm53f-080`, which re-transcribed the fixture's `text_config`
# from the in-repo vendor copy; the previous pin was
# f3d8790f18a18ffc95015dcc8869ac25c8d49129a383ccd3e0b4d07183bd6802.
FIXTURE_SHA256 = "5ed24d23a3e14a038352e1bdc21fd25fc90ff2291d3f6a310acf5d4036665a1d"

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


# ===========================================================================
# `inc-glm53f-080` acceptance -- WP1/WP7 repair.
#
# THE DEFECT, in one sentence: the real text config carries 58 keys, the fork's
# dataclass modelled only some of them, and the adapter dropped every other key
# without a word -- one of the dropped keys was the model's own RMSNorm epsilon.
#
# THE MODELLED AND DROPPED COUNTS ARE NOT WRITTEN IN THIS PROSE, since
# `inc-glm53f-033` repair round 2. They used to be, as "modelled 30 ... dropped
# the other 28", which cannot be right about both halves at once next to a pinned
# count of 26: 58 minus 30 is 28. Conjunct (c) below DERIVES the dropped set from
# the vendor config and the dataclass, so the count has one home and this prose
# is not a second one.
#
# FOUR counted conjuncts, ONE item each, no `parametrize` (plan section 6 rule
# 6). Every expected value is DERIVED here from the two pinned fixtures; the
# figures pinned as constants beside the derivation are named for what they are,
# so the two cannot drift apart silently.
# ===========================================================================

# `inc-glm53f-078` lands this byte-identical copy of the vendor config and pins
# it by this digest in its own conjunct (h). This section READS it and never
# writes it: it is the only side of the comparison that speaks for the vendor.
REAL_CONFIG_PATH = FIXTURE_PATH.parent / "hf-config.json"
REAL_CONFIG_SHA256 = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"

# Four of the five figures below are READINGS OF THE VENDOR CONFIG: the key
# count, the two layer-list lengths and the quantisation-config key count are
# facts about `hf-config.json` and move only if the checkpoint does.
C080_REAL_TEXT_KEYS = 58
C080_KDA_LAYERS = 34
C080_FULL_ATTN_LAYERS = 11
C080_QUANT_CONFIG_KEYS = 4

# THE FIFTH IS DIFFERENT, AND THE COMMENT SAYING OTHERWISE WAS WRONG. This is
# THIS FILE's reading of the COMPLEMENT -- the vendor's keys that
# `Glm5NextTextConfig` does not declare -- so it moves whenever the dataclass
# models one more of the checkpoint's keys. It is NOT a figure the plan declares:
# the plan's `-080` row registers that the adapter "lifts the checkpoint's
# `rms_norm_eps` and names every key it drops", and carries no dropped-key count
# at all. Conjunct (c) derives the set and this constant pins the count beside the
# derivation, which is the only reason to keep it.
#
# 26 -> 25 AT `inc-glm53f-033` REPAIR ROUND 2, BECAUSE THAT ROUND MODELLED ONE
# MORE KEY: it added `swiglu_limit` to `Glm5NextTextConfig`, the bound the
# checkpoint clamps both shared-expert projections with. 58 real keys, 33 modelled
# after the lift, 25 dropped -- and `swiglu_limit` is now absent from the log,
# which conjunct (c) asserts BY NAME.
C080_DROPPED_KEYS = 25

# The checkpoint's two epsilons. They are DIFFERENT numbers, which is the whole
# point of the repair: one field cannot carry both.
C080_RMS_NORM_EPS = 1e-05
C080_HC_EPS = 1e-06

# Conjunct (d)'s two readings. The non-default value is what makes reading 1
# falsifiable: a hard-wired `1e-05` in the resolution would fail it.
C080_NON_DEFAULT_RMS_NORM_EPS = 3e-05
C080_EXPLICIT_OVERRIDE_EPS = 1e-6


def _real_text_config() -> dict:
    """The vendor config's `text_config`, digest-checked before it is trusted."""
    digest = hashlib.sha256(REAL_CONFIG_PATH.read_bytes()).hexdigest()
    assert digest == REAL_CONFIG_SHA256, f"hf-config.json moved: sha256={digest}"
    return json.loads(REAL_CONFIG_PATH.read_text())["text_config"]


class _RecordingHandler(logging.Handler):
    """Collects records off the config module's own logger object."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _capture_drop_log():
    """Attach to `config.logger` directly, not through `caplog`.

    Directly, because `caplog`'s handler lives on the root logger, so a
    propagation setting anywhere in vLLM's logging configuration would make
    this read zero for a reason that has nothing to do with the code under
    test. This is the pattern `test_platform_quant_validation.py` landed.
    """
    handler = _RecordingHandler()
    target = config_module.logger
    previous_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.WARNING)
    try:
        yield handler
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)


class _RouterBankStub:
    """Only the two attributes `route_tokens` reads off `self`.

    A real `Glm5NextRoutedExperts` is not built, because the seam is replaced
    by a recorder and nothing downstream of it runs: no kernel is entered and
    no accelerator is reached.
    """

    def __init__(self) -> None:
        self.router_weight = object()
        self.router_bias = object()


class _SeamRecorder:
    """Stands in for the router seam and records the `eps=` it was handed.

    Returns the FOUR values the caller unpacks -- logits, expert index, expert
    affinities, substrate index -- so `route_tokens` completes normally.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return ("logits", "expert_index", "expert_affinities", "substrate_index")


def test_c080_a_the_two_epsilons_are_two_distinct_values(cfg):
    """(a) `rms_norm_eps == 1e-05`, `hc_eps == 1e-06`, and the two are unequal.

    The inequality is the conjunct. Before this block the fork carried one
    epsilon field, so every RMSNorm that reached for a config epsilon got the
    mHC number; the two values being distinct is what proves that is over.
    """
    text = cfg.text_config
    real = _real_text_config()
    print(f"\n[C080-a] config rms_norm_eps={text.rms_norm_eps!r}")
    print(f"[C080-a] config hc_eps={text.hc_eps!r}")
    print(f"[C080-a] vendor rms_norm_eps={real['rms_norm_eps']!r}")
    print(f"[C080-a] vendor hc_eps={real['hc_eps']!r}")

    assert text.rms_norm_eps == C080_RMS_NORM_EPS
    assert text.hc_eps == C080_HC_EPS
    assert text.rms_norm_eps != text.hc_eps
    # Both readings are the vendor's own, so neither is a dataclass default
    # that happens to agree with the fixture.
    assert real["rms_norm_eps"] == C080_RMS_NORM_EPS
    assert real["hc_eps"] == C080_HC_EPS


def test_c080_b_the_retranscribed_fixture_agrees_with_the_vendor_config(raw):
    """(b) 58 keys, none absent from the vendor config, both layer lists present.

    Plus the declared negative that keeps ONE source of truth for the skip
    list: `quantization_config` stays at its 4 keys with
    `modules_to_not_convert` ABSENT, so the 1,509-entry list exists in this
    repository exactly once -- in `hf-config.json`.
    """
    real = _real_text_config()
    fixture_text = raw["text_config"]
    absent = sorted(k for k in real if k not in fixture_text)
    extra = sorted(k for k in fixture_text if k not in real)
    lac = fixture_text["linear_attn_config"]
    quant = raw["quantization_config"]
    real_quant = json.loads(REAL_CONFIG_PATH.read_text())["quantization_config"]

    print(f"\n[C080-b] fixture text_config keys={len(fixture_text)}")
    print(f"[C080-b] vendor  text_config keys={len(real)}")
    print(f"[C080-b] vendor keys the fixture lacks={absent}")
    print(f"[C080-b] fixture keys the vendor lacks={extra}")
    print(f"[C080-b] kda_layers={len(lac['kda_layers'])}")
    print(f"[C080-b] full_attn_layers={len(lac['full_attn_layers'])}")
    print(f"[C080-b] quantization_config keys={sorted(quant)}")

    assert len(fixture_text) == C080_REAL_TEXT_KEYS
    assert len(real) == C080_REAL_TEXT_KEYS
    assert absent == []
    assert extra == []
    assert len(lac["kda_layers"]) == C080_KDA_LAYERS
    assert len(lac["full_attn_layers"]) == C080_FULL_ATTN_LAYERS
    assert lac["kda_layers"] == real["linear_attn_config"]["kda_layers"]
    assert lac["full_attn_layers"] == real["linear_attn_config"]["full_attn_layers"]
    # The two lists partition the 45 layers, and the DSA half is the same set
    # the intake record enumerated -- two bases, one answer.
    assert len(lac["kda_layers"]) + len(lac["full_attn_layers"]) == 45
    assert lac["full_attn_layers"] == INTAKE_RECORDED_DSA_INDICES
    # ONE source of truth for the skip list: present there, absent here.
    assert len(quant) == C080_QUANT_CONFIG_KEYS
    assert "modules_to_not_convert" not in quant
    assert "modules_to_not_convert" in real_quant


def test_c080_c_the_filter_names_every_key_it_drops():
    """(c) The drop log names exactly the keys the dataclass does not model.

    The expected set is DERIVED from the vendor config and the dataclass, never
    typed in: it is the vendor's keys that are neither a `fields(cls)` name nor
    the key the `dtype` -> `torch_dtype` remap consumes. `dtype` is therefore
    absent from the log, because the adapter reads it.

    Non-vacuity (D1.5): the set must be NON-EMPTY, so a log that named nothing
    fails this item.

    TWO KEYS ARE ASSERTED BY NAME, one per repair that modelled them:
    `rms_norm_eps` (`inc-glm53f-080`, this block's own) and `swiglu_limit`
    (`inc-glm53f-033` repair round 2, the SwiGLU bound the shared expert clamps
    with). Both must be declared fields and neither may appear in the log. The
    by-name half matters because the count alone cannot say WHICH key left: a
    dataclass that dropped one field and added another would keep the count and
    break the model.
    """
    real = _real_text_config()
    field_names = {f.name for f in fields(Glm5NextTextConfig)}
    # The remap's own condition, restated once from the data rather than
    # asserted by name: it fires only when the HF dict carries `dtype`, lacks
    # `torch_dtype`, and the dataclass declares `torch_dtype`.
    remapped = set()
    if "dtype" in real and "torch_dtype" not in real and "torch_dtype" in field_names:
        remapped.add("dtype")
    expected = sorted(set(real) - field_names - remapped)

    with _capture_drop_log() as handler:
        built = Glm5NextTextConfig.from_hf_config(copy.deepcopy(real))

    warnings = [r for r in handler.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"expected one drop-log record, got {len(warnings)}"
    logged = sorted(warnings[0].args[-1].split(", "))

    # The two keys asserted by name, printed as a set difference against the
    # vendor's own key list so a reader sees the population each claim is made
    # over, not just the verdict.
    lifted = ("rms_norm_eps", "swiglu_limit")
    in_vendor = sorted(k for k in lifted if k in real)
    in_fields = sorted(k for k in lifted if k in field_names)
    in_log = sorted(k for k in lifted if k in logged)

    print(f"\n[C080-c] logged {len(logged)} dropped keys={logged}")
    print(f"[C080-c] expected {len(expected)} dropped keys={expected}")
    print(f"[C080-c] message={warnings[0].getMessage()}")
    print(f"[C080-c] remap consumed={sorted(remapped)}")
    print(f"[C080-c] built torch_dtype={built.torch_dtype}")
    print(f"[C080-c] vendor text_config keys={len(real)}")
    print(f"[C080-c] lifted keys the vendor declares={in_vendor}")
    print(f"[C080-c] lifted keys the dataclass models={in_fields}")
    print(f"[C080-c] lifted keys still in the drop log={in_log}")

    assert logged == expected
    assert len(logged) == C080_DROPPED_KEYS
    assert logged, "the drop log named nothing, so this item would be vacuous"
    # The remap really did fire, which is the ground for `dtype` not being in
    # the log: the value landed on the dataclass field.
    assert str(built.torch_dtype) == "torch.bfloat16"
    assert "dtype" not in logged
    # The two lifted keys: the vendor declares both, the dataclass models both,
    # and neither is dropped. `rms_norm_eps` is this block's own repair;
    # `swiglu_limit` is `inc-glm53f-033` repair round 2's, and it is the reason
    # the count above reads 25 rather than 26.
    assert in_vendor == sorted(lifted), (
        f"the vendor config does not declare {sorted(set(lifted) - set(in_vendor))}, "
        f"so this claim would be about a key the checkpoint never had"
    )
    assert in_fields == sorted(lifted)
    assert in_log == []
    assert "rms_norm_eps" not in logged
    assert "rms_norm_eps" in field_names
    assert "swiglu_limit" not in logged
    assert "swiglu_limit" in field_names
    # And the count really is the complement's size, recomputed from the two
    # sides rather than trusted from the constant.
    assert len(logged) == len(real) - len(set(real) & (field_names | remapped))


def test_c080_d_the_seam_receives_the_config_epsilon(raw, cfg, monkeypatch):
    """(d) The seam gets the CONFIG's epsilon on a call that passes no `eps`.

    Reading 1 -- the production call shape, `route_tokens(hidden, gamma,
    text_config)`. It is measured on a config whose `rms_norm_eps` is set to a
    NON-DEFAULT `3e-05`, so a hard-wired `1e-05` in the resolution fails it;
    the dataclass default `1e-05` is recorded alongside as the fixture reading.

    Reading 2 -- the explicit-override control (D1.5): `eps=1e-6` is still
    delivered unchanged. This is what proves the resolution honours a caller,
    and it is what keeps `inc-glm53f-032`'s landed call honoured.

    Both values are read back from the RECORDED call, never from the signature
    default. No kernel runs and no accelerator is reached.
    """
    from vllm_neuron.functional.moe import router as router_module
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextRoutedExperts

    mutated = copy.deepcopy(raw)
    mutated["text_config"]["rms_norm_eps"] = C080_NON_DEFAULT_RMS_NORM_EPS
    non_default_text = Glm5NextConfig.from_configs(mutated).text_config
    fixture_reading = cfg.text_config.rms_norm_eps

    recorder = _SeamRecorder()
    monkeypatch.setattr(
        router_module, "noaux_tc_rmsnorm_router_topk", recorder, raising=True
    )
    bank = _RouterBankStub()
    hidden, gamma = object(), object()

    # Reading 1: the production call shape -- no `eps` argument at all. This is
    # byte-for-byte the shape `-032`'s landed test calls with.
    Glm5NextRoutedExperts.route_tokens(bank, hidden, gamma, non_default_text)
    # Reading 2: the explicit override.
    Glm5NextRoutedExperts.route_tokens(
        bank, hidden, gamma, non_default_text, eps=C080_EXPLICIT_OVERRIDE_EPS
    )

    assert len(recorder.calls) == 2, f"seam entered {len(recorder.calls)} times"
    reading1 = recorder.calls[0]["eps"]
    reading2 = recorder.calls[1]["eps"]

    print(f"\n[C080-d] reading 1 (no eps passed) seam eps={reading1!r}")
    print(f"[C080-d] config rms_norm_eps={non_default_text.rms_norm_eps!r}")
    print(f"[C080-d] fixture default reading={fixture_reading!r}")
    print(f"[C080-d] reading 2 (eps passed) seam eps={reading2!r}")

    # Reading 1: the seam got the config's number, not a literal.
    assert reading1 == C080_NON_DEFAULT_RMS_NORM_EPS
    assert reading1 == non_default_text.rms_norm_eps
    assert reading1 != C080_RMS_NORM_EPS, "a hard-wired 1e-05 would pass vacuously"
    # ... and the fixture's own reading is the checkpoint's 1e-05.
    assert fixture_reading == C080_RMS_NORM_EPS
    # Reading 2: the override survives, so the resolution is not a clamp.
    assert reading2 == C080_EXPLICIT_OVERRIDE_EPS
    assert reading2 != reading1
    # The recorded call is the one this method made, so the recorded `eps`
    # belongs to it and to nothing else.
    assert recorder.calls[0]["hidden_states"] is hidden
    assert recorder.calls[0]["gamma"] is gamma
    assert recorder.calls[0]["correction_bias"] is bank.router_bias
    assert recorder.calls[0]["router_weights"] is bank.router_weight
