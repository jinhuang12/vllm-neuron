# SPDX-License-Identifier: Apache-2.0
"""Tests for parsing the GLM-5.3-Flash checkpoint's ``config.json``.

The first half reads a trimmed copy of the published config and pins the architecture values
the rest of the model depends on, each with a mutation arm proving the value is read from the
file rather than returned as a constant. The second half reads the untrimmed published config
beside it and covers the key filter: the two distinct epsilons, the keys the dataclass does
not model, and the epsilon the router seam is handed. No test here touches the network.
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

# The checkpoint's full-attention (DSA) layer indices, as the published config enumerates them.
DSA_LAYER_INDICES = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]

# (num_kda, num_dsa) over the 45 layers: one fact about one 3:1 schedule, so it is asserted as
# one pair rather than as two independent counts.
EXPECTED_LAYER_SPLIT = (34, 11)


def _raw() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def raw() -> dict:
    return _raw()


@pytest.fixture(scope="module")
def cfg(raw) -> Glm5NextConfig:
    return Glm5NextConfig.from_configs(copy.deepcopy(raw))


# The pinned values, as a table: (what, extract, expected, mutate).
#
# `mutate` edits a deep copy of the raw HF dict so that this row's extracted value must change.
# It is what proves the extractor reads the config instead of returning a constant.


def _m_num_hidden_layers(r):
    # num_hidden_layers and layer_types must stay length-consistent, so the mutation drops one
    # layer from both -- otherwise the config rejects the dict before the extractor runs.
    r["text_config"]["num_hidden_layers"] = 44
    r["text_config"]["layer_types"].pop()


def _m_layer_split(r):
    # Flip the first KDA layer to DSA: still a valid exhaustive partition of 45 layers, but the
    # pair becomes (33, 12).
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
        "num_hidden_layers == 45",
        lambda c: c.text_config.num_hidden_layers,
        45,
        _m_num_hidden_layers,
    ),
    (
        "KDA/DSA 3:1 over 45 layers -> (kda, dsa) == (34, 11)",
        lambda c: c.text_config.attention_layer_split,
        EXPECTED_LAYER_SPLIT,
        _m_layer_split,
    ),
    (
        "kv_lora_rank == 512",
        lambda c: c.text_config.kv_lora_rank,
        512,
        _m_kv_lora_rank,
    ),
    (
        "qk_rope_head_dim == 0",
        lambda c: c.text_config.qk_rope_head_dim,
        0,
        _m_qk_rope_head_dim,
    ),
    (
        "n_routed_experts == 288",
        lambda c: c.text_config.n_routed_experts,
        288,
        _m_n_routed_experts,
    ),
    (
        "n_shared_experts == 1",
        lambda c: c.text_config.n_shared_experts,
        1,
        _m_n_shared_experts,
    ),
    (
        "num_experts_per_tok == 8",
        lambda c: c.text_config.num_experts_per_tok,
        8,
        _m_num_experts_per_tok,
    ),
    ("hc_mult == 4", lambda c: c.text_config.hc_mult, 4, _m_hc_mult),
    (
        "weight_block_size == [128, 128]",
        lambda c: c.weight_block_size,
        [128, 128],
        _m_weight_block_size,
    ),
    (
        'activation_scheme == "dynamic"',
        lambda c: c.activation_scheme,
        "dynamic",
        _m_activation_scheme,
    ),
]


def test_the_parsed_config_carries_the_checkpoints_values(cfg):
    """Every row of the table above, all evaluated before any of them fails.

    A failure then reports the whole table rather than stopping at the first mismatch.
    """
    assert EQUALITIES, "the table above is empty, so this test would read nothing"
    failed = []
    for what, extract, expected, _ in EQUALITIES:
        actual = extract(cfg)
        if not (actual == expected and type(actual) is type(expected)):
            failed.append((what, actual, expected))

    assert not failed, "mismatched values: " + "; ".join(
        f"{what}: actual={actual!r} expected={expected!r}"
        for what, actual, expected in failed
    )


@pytest.mark.parametrize(
    "what,extract,expected,mutate",
    EQUALITIES,
    ids=[
        "num_hidden_layers",
        "attention_layer_split",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "hc_mult",
        "weight_block_size",
        "activation_scheme",
    ],
)
def test_each_value_moves_when_the_config_moves(raw, what, extract, expected, mutate):
    """Mutating the config must move this row's value.

    A row that still reads its expected value from a mutated config is reading something other
    than the config, and its pass means nothing.
    """
    mutated = copy.deepcopy(raw)
    mutate(mutated)
    actual = extract(Glm5NextConfig.from_configs(mutated))
    assert actual != expected, f"{what} is unmoved by its mutation, so it reads a constant"


def test_the_layer_schedule_agrees_across_three_bases(cfg):
    """The 34/11 split, counted from three bases that do not share a mechanism.

    The checked-in 45-entry ``layer_types`` array, the config's generated 3:1 interleave rule,
    and the checkpoint's own enumeration of the DSA layer indices.
    """
    text = cfg.text_config
    base1 = text.attention_layer_split
    generated = default_layer_types(text.num_hidden_layers)
    base2 = (
        sum(1 for t in generated if t == KDA_LAYER_TYPE),
        sum(1 for t in generated if t == DSA_LAYER_TYPE),
    )
    base3 = (
        text.num_hidden_layers - len(DSA_LAYER_INDICES),
        len(DSA_LAYER_INDICES),
    )

    assert base1 == EXPECTED_LAYER_SPLIT
    assert base2 == EXPECTED_LAYER_SPLIT
    assert base3 == EXPECTED_LAYER_SPLIT
    assert text.layer_types == generated
    assert text.dsa_layer_indices == DSA_LAYER_INDICES
    # The pair sums to the layer count, which is what makes it one fact.
    assert sum(base1) == text.num_hidden_layers == 45


def test_the_two_family_partition_is_exhaustive(cfg):
    """No layer falls outside {KDA, DSA}: the complement is empty.

    Family names are compared by equality, because 'attention' is a substring of both names and
    a substring screen would mis-partition the stack while still reporting 45 layers.
    """
    text = cfg.text_config
    known = {KDA_LAYER_TYPE, DSA_LAYER_TYPE}
    complement = [t for t in text.layer_types if t not in known]
    assert complement == []
    assert len(text.kda_layer_indices) + len(text.dsa_layer_indices) == len(
        text.layer_types
    )
    assert set(text.kda_layer_indices).isdisjoint(text.dsa_layer_indices)


def test_an_unrecognised_layer_family_is_rejected():
    """A family name outside {KDA, DSA} must be refused rather than dropped."""
    schedule = default_layer_types(45)
    schedule[0] = "sliding_window_attention"
    with pytest.raises(ValueError, match="unrecognised attention families"):
        Glm5NextTextConfig(num_hidden_layers=45, layer_types=schedule)


def test_a_length_mismatched_schedule_is_rejected():
    """A schedule that disagrees with num_hidden_layers cannot pass silently."""
    with pytest.raises(ValueError, match="layer_types has 44 entries"):
        Glm5NextTextConfig(num_hidden_layers=45, layer_types=default_layer_types(44))


def test_no_weights_are_referenced_by_the_fixture(raw):
    """The fixture is config only, no weights."""
    blob = json.dumps(raw)
    for banned in ("safetensors", "model.safetensors.index.json", "weight_map"):
        assert banned not in blob, f"fixture references weights via {banned!r}"


# ---------------------------------------------------------------------------
# The key filter, read against the untrimmed published config beside the fixture.
# ---------------------------------------------------------------------------

# A byte-identical copy of the published config, pinned by digest. This half reads it and
# never writes it: it is the only side of the comparison that speaks for the checkpoint.
REAL_CONFIG_PATH = FIXTURE_PATH.parent / "hf-config.json"
REAL_CONFIG_SHA256 = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"

# Readings of the published config: the text_config key count and the quantization_config key
# count. They move only if the checkpoint does.
PUBLISHED_TEXT_CONFIG_KEYS = 58
QUANT_CONFIG_KEYS = 4

# How many of the published text_config keys ``Glm5NextTextConfig`` does not declare. The
# derivation below computes the set; this pins its size beside it, so a population that moves
# for an undeclared reason fails even though the derivation would follow it.
UNMODELLED_TEXT_CONFIG_KEYS = 18

# The indexer's dials. They are declared in the published text_config and modelled as dataclass
# fields, so none of them may appear in the drop log. Asserted by name because a count cannot
# say which key left: a dataclass that dropped one field while adding another keeps the count
# and breaks the model.
INDEXER_CONFIG_KEYS = (
    "index_topk",
    "index_n_heads",
    "index_head_dim",
    "index_kpool",
    "index_kpool_compress",
    "index_kpool_always_select_tail",
    "index_share_for_mtp_iteration",
)

# The checkpoint's two epsilons. They are different numbers, which is why one field cannot
# carry both.
RMS_NORM_EPS = 1e-05
HC_EPS = 1e-06

# A value the dataclass does not default to, so a hard-wired 1e-05 in the resolution fails, and
# the override the second reading passes explicitly.
NON_DEFAULT_RMS_NORM_EPS = 3e-05
EXPLICIT_OVERRIDE_EPS = 1e-6


def _real_text_config() -> dict:
    """The published config's ``text_config``, digest-checked before it is trusted."""
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
    """Attach to ``config.logger`` directly, not through ``caplog``.

    ``caplog``'s handler lives on the root logger, so a propagation setting anywhere in vLLM's
    logging configuration would make this read zero for a reason that has nothing to do with
    the code under test.
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
    """Only the two attributes ``route_tokens`` reads off ``self``.

    A real ``Glm5NextRoutedExperts`` is not built, because the seam is replaced by a recorder
    and nothing downstream of it runs: no kernel is entered and no accelerator is reached.
    """

    def __init__(self) -> None:
        self.router_weight = object()
        self.router_bias = object()


class _SeamRecorder:
    """Stands in for the router seam and records the ``eps=`` it was handed.

    Returns the four values the caller unpacks -- logits, expert index, expert affinities,
    substrate index -- so ``route_tokens`` completes normally.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return ("logits", "expert_index", "expert_affinities", "substrate_index")


def test_the_two_epsilons_are_two_distinct_values(cfg):
    """``rms_norm_eps == 1e-05``, ``hc_eps == 1e-06``, and the two are unequal.

    The inequality is the point: with one epsilon field on the fork config, every RMSNorm that
    reached for a config epsilon got the mHC number instead.
    """
    text = cfg.text_config
    real = _real_text_config()

    assert text.rms_norm_eps == RMS_NORM_EPS
    assert text.hc_eps == HC_EPS
    assert text.rms_norm_eps != text.hc_eps
    # Both readings are the checkpoint's own, so neither is a dataclass default that happens to
    # agree with the fixture.
    assert real["rms_norm_eps"] == RMS_NORM_EPS
    assert real["hc_eps"] == HC_EPS


def test_the_trimmed_fixture_agrees_with_the_published_config(raw):
    """58 keys, none absent from the published config, both layer lists present.

    ``quantization_config`` stays at its four keys with ``modules_to_not_convert`` absent, so
    the 1,509-entry skip list is carried by ``hf-config.json`` alone.
    """
    real = _real_text_config()
    fixture_text = raw["text_config"]
    absent = sorted(k for k in real if k not in fixture_text)
    extra = sorted(k for k in fixture_text if k not in real)
    lac = fixture_text["linear_attn_config"]
    quant = raw["quantization_config"]
    real_quant = json.loads(REAL_CONFIG_PATH.read_text())["quantization_config"]

    assert len(fixture_text) == PUBLISHED_TEXT_CONFIG_KEYS
    assert len(real) == PUBLISHED_TEXT_CONFIG_KEYS
    assert absent == [], f"the published config declares keys the fixture lacks: {absent}"
    assert extra == [], f"the fixture declares keys the published config lacks: {extra}"
    assert len(lac["kda_layers"]) == EXPECTED_LAYER_SPLIT[0]
    assert len(lac["full_attn_layers"]) == EXPECTED_LAYER_SPLIT[1]
    assert lac["kda_layers"] == real["linear_attn_config"]["kda_layers"]
    assert lac["full_attn_layers"] == real["linear_attn_config"]["full_attn_layers"]
    # The two lists partition the 45 layers, and the DSA half is the same set the published
    # config enumerates -- two bases, one answer.
    assert len(lac["kda_layers"]) + len(lac["full_attn_layers"]) == 45
    assert lac["full_attn_layers"] == DSA_LAYER_INDICES
    # One home for the skip list: present there, absent here.
    assert len(quant) == QUANT_CONFIG_KEYS
    assert "modules_to_not_convert" not in quant
    assert "modules_to_not_convert" in real_quant


def test_the_filter_names_every_key_it_drops():
    """The drop log names exactly the published keys the dataclass does not model.

    The expected set is derived from the published config and the dataclass, never typed in: it
    is the published keys that are neither a ``fields(cls)`` name nor the key the ``dtype`` ->
    ``torch_dtype`` remap consumes. ``dtype`` is therefore absent from the log, because the
    adapter reads it.

    ``rms_norm_eps`` and ``swiglu_limit`` are asserted by name as well: both must be declared
    fields and neither may appear in the log.
    """
    real = _real_text_config()
    field_names = {f.name for f in fields(Glm5NextTextConfig)}
    # The remap's own condition, restated from the data rather than asserted by name: it fires
    # only when the HF dict carries `dtype`, lacks `torch_dtype`, and the dataclass declares
    # `torch_dtype`.
    remapped = set()
    if "dtype" in real and "torch_dtype" not in real and "torch_dtype" in field_names:
        remapped.add("dtype")
    expected = sorted(set(real) - field_names - remapped)

    with _capture_drop_log() as handler:
        built = Glm5NextTextConfig.from_hf_config(copy.deepcopy(real))

    warnings = [r for r in handler.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"expected one drop-log record, got {len(warnings)}"
    logged = sorted(warnings[0].args[-1].split(", "))

    lifted = ("rms_norm_eps", "swiglu_limit")
    in_vendor = sorted(k for k in lifted if k in real)
    in_fields = sorted(k for k in lifted if k in field_names)
    in_log = sorted(k for k in lifted if k in logged)

    assert logged == expected
    assert len(logged) == UNMODELLED_TEXT_CONFIG_KEYS
    assert logged, "the drop log named nothing, so the checks below would read an empty list"
    # The remap really did fire, which is the ground for `dtype` not being in the log: the
    # value reached the dataclass field.
    assert str(built.torch_dtype) == "torch.bfloat16"
    assert "dtype" not in logged
    assert in_vendor == sorted(lifted), (
        f"the published config does not declare {sorted(set(lifted) - set(in_vendor))}, so this "
        f"claim would be about a key the checkpoint never had"
    )
    assert in_fields == sorted(lifted)
    assert in_log == []
    # The same two claims again by name, so neither depends on the tuple above still holding
    # both keys: a shortened tuple would quietly narrow the three readings before it.
    assert "rms_norm_eps" not in logged, (
        "rms_norm_eps is dropped again, so every RMSNorm that reaches for the config epsilon "
        "takes a default instead of the checkpoint's number"
    )
    assert "rms_norm_eps" in field_names, "Glm5NextTextConfig no longer declares rms_norm_eps"
    assert "swiglu_limit" not in logged, (
        "swiglu_limit is dropped again, so the shared expert clamps at a default rather than "
        "the checkpoint's bound"
    )
    assert "swiglu_limit" in field_names, "Glm5NextTextConfig no longer declares swiglu_limit"

    indexer_in_vendor = sorted(k for k in INDEXER_CONFIG_KEYS if k in real)
    indexer_in_fields = sorted(k for k in INDEXER_CONFIG_KEYS if k in field_names)
    indexer_in_log = sorted(k for k in INDEXER_CONFIG_KEYS if k in logged)
    assert indexer_in_vendor == sorted(INDEXER_CONFIG_KEYS), (
        f"the published config does not declare "
        f"{sorted(set(INDEXER_CONFIG_KEYS) - set(indexer_in_vendor))}"
    )
    assert indexer_in_fields == sorted(INDEXER_CONFIG_KEYS), (
        f"Glm5NextTextConfig does not model "
        f"{sorted(set(INDEXER_CONFIG_KEYS) - set(indexer_in_fields))}; the indexer reads its "
        f"dials off this dataclass, so an unmodelled dial is a default the checkpoint never set"
    )
    assert indexer_in_log == [], (
        f"{indexer_in_log} are still named in the drop log while the dataclass declares them, "
        f"which means the filter and the fields disagree about the same key"
    )
    # `indexer_rope_interleave` is the near miss: it is spelled `indexer_`, not `index_`, and
    # the config records why it stays unmodelled.
    assert "indexer_rope_interleave" in logged, (
        "indexer_rope_interleave left the drop log, but the config records it as deliberately "
        "unmodelled; if it is modelled now, that decision moved"
    )
    # And the count really is the complement's size, recomputed from the two sides rather than
    # trusted from the constant.
    assert len(logged) == len(real) - len(set(real) & (field_names | remapped))


def test_the_router_seam_receives_the_config_epsilon(raw, cfg, monkeypatch):
    """The seam gets the config's epsilon on a call that passes no ``eps``.

    The first reading is the production call shape, ``route_tokens(hidden, gamma, text_config)``,
    measured on a config whose ``rms_norm_eps`` is a non-default ``3e-05`` so that a hard-wired
    ``1e-05`` fails. The second reading passes ``eps=1e-6`` and must see it delivered unchanged.

    Both values are read back from the recorded call, never from a signature default. No kernel
    runs and no accelerator is reached.
    """
    from vllm_neuron.functional.moe import router as router_module
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextRoutedExperts

    mutated = copy.deepcopy(raw)
    mutated["text_config"]["rms_norm_eps"] = NON_DEFAULT_RMS_NORM_EPS
    non_default_text = Glm5NextConfig.from_configs(mutated).text_config
    fixture_reading = cfg.text_config.rms_norm_eps

    recorder = _SeamRecorder()
    monkeypatch.setattr(
        router_module, "noaux_tc_rmsnorm_router_topk", recorder, raising=True
    )
    bank = _RouterBankStub()
    hidden, gamma = object(), object()

    # Reading 1: the production call shape -- no `eps` argument at all.
    Glm5NextRoutedExperts.route_tokens(bank, hidden, gamma, non_default_text)
    # Reading 2: the explicit override.
    Glm5NextRoutedExperts.route_tokens(
        bank, hidden, gamma, non_default_text, eps=EXPLICIT_OVERRIDE_EPS
    )

    assert len(recorder.calls) == 2, f"seam entered {len(recorder.calls)} times"
    reading1 = recorder.calls[0]["eps"]
    reading2 = recorder.calls[1]["eps"]

    # Reading 1: the seam got the config's number, not a literal.
    assert reading1 == NON_DEFAULT_RMS_NORM_EPS
    assert reading1 == non_default_text.rms_norm_eps
    assert reading1 != RMS_NORM_EPS, "a hard-wired 1e-05 would pass without reading the config"
    # ... and the fixture's own reading is the checkpoint's 1e-05.
    assert fixture_reading == RMS_NORM_EPS
    # Reading 2: the override survives, so the resolution is not a clamp.
    assert reading2 == EXPLICIT_OVERRIDE_EPS
    assert reading2 != reading1
    # The recorded call is the one this method made, so the recorded `eps` belongs to it.
    assert recorder.calls[0]["hidden_states"] is hidden
    assert recorder.calls[0]["gamma"] is gamma
    assert recorder.calls[0]["correction_bias"] is bank.router_bias
    assert recorder.calls[0]["router_weights"] is bank.router_weight
