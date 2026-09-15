# SPDX-License-Identifier: Apache-2.0
"""`NeuronConfig` knobs for the GLM-5.3-Flash (Glm5Next) hybrid stack.

Covers the knobs `inc-glm53f-010` adds across three families -- hybrid
KDA/DSA, mHC, and blockwise FP8.

Two things about this file are deliberate and load-bearing:

1. **N is stated, not counted from whatever happens to be present.** ``N``
   below is fixed by this increment; ``test_knob_count_is_n`` asserts
   ``len(GLM5NEXT_KNOBS) == N`` so the table cannot silently grow or shrink.
   The counted assertions then run N/N over that table.
2. **The target class is pinned by a negative control.** ``NeuronConfig`` is a
   dataclass, so an unknown keyword raises ``TypeError`` from the dataclass
   machinery. Its file-mate ``OnDeviceSamplingConfig`` takes ``**kwargs`` and
   swallows unknown keys, so the same test written against that class would go
   green while the "unknown knob raises" property was false. The control
   measures that difference instead of trusting it.
"""

import dataclasses

import pytest

from vllm_neuron.model.neuron_config import (
    NeuronConfig,
    OnDeviceSamplingConfig,
    TensorCaptureConfig,
    TensorReplacementConfig,
    VisionNeuronConfig,
)

# The knob count this increment fixes. Stated here as a number so the table
# below is checked against a declaration rather than against itself.
N = 8

# (name, declared default, probe value, family). The probe value must differ
# from the default -- `test_probe_values_differ_from_defaults` enforces it, so
# a round-trip cannot pass vacuously.
GLM5NEXT_KNOBS = [
    ("enable_hybrid_kv_cache", False, True, "hybrid-KDA/DSA"),
    ("hybrid_kv_block_size", None, 128, "hybrid-KDA/DSA"),
    ("kda_state_dtype", None, "bfloat16", "hybrid-KDA/DSA"),
    ("kda_state_chunk_size", None, 64, "hybrid-KDA/DSA"),
    ("mhc_sinkhorn_iters", None, 12, "mHC"),
    ("mhc_eps", None, 1e-05, "mHC"),
    ("blockwise_fp8", False, True, "block-fp8"),
    ("block_quant_scale_min", None, 1e-04, "block-fp8"),
]

# `NeuronConfig`'s dataclass fields at the campaign's target base, read by
# runtime introspection (`dataclasses.fields`) rather than by a regex over the
# source. A name-pattern screen misses `all2all_backend` and `fp8_packed_kv`
# because both contain a digit; introspection cannot miss either.
BASE_FIELDS = (
    "ep_degree",
    "attention_dp_size",
    "embedding_dp_size",
    "lm_head_dp_size",
    "mlp_dp_size",
    "on_device_sampling_config",
    "max_logprobs",
    "tensor_capture",
    "tensor_replacement",
    "kv_segment_size_buckets",
    "debug_logits_dir",
    "quantization",
    "all2all_backend",
    "modules_to_not_convert",
    "num_batched_tokens_buckets",
    "num_seqs_buckets",
    "decode_context_length_buckets",
    "enable_structured_outputs",
    "fp8_packed_kv",
)

KNOB_NAMES = [name for name, _, _, _ in GLM5NEXT_KNOBS]


def _field_names(cls):
    return [f.name for f in dataclasses.fields(cls)]


# --------------------------------------------------------------------------
# The table itself: N is declared, and the table is held to it.
# --------------------------------------------------------------------------


def test_knob_count_is_n():
    """N/N is a count against a declaration, so the table cannot drift."""
    assert len(GLM5NEXT_KNOBS) == N
    assert len(set(KNOB_NAMES)) == N  # no duplicate name inflating the count


def test_every_family_is_covered():
    """All three families the increment names are represented."""
    assert {family for _, _, _, family in GLM5NEXT_KNOBS} == {
        "hybrid-KDA/DSA",
        "mHC",
        "block-fp8",
    }


def test_probe_values_differ_from_defaults():
    """Falsifiability arm for the round-trip: no probe may equal its default."""
    for name, default, probe, _ in GLM5NEXT_KNOBS:
        assert probe != default, name


# --------------------------------------------------------------------------
# Conjunct 1 -- every knob is a real field and round-trips through construction
# --------------------------------------------------------------------------


def test_knobs_are_fields_of_neuron_config():
    """N/N: each tabled name resolves to an actual dataclass field.

    Falsifiability arm for the count: a typo in the table fails here rather
    than passing as a `setattr` on an instance that accepts anything.
    """
    fields = _field_names(NeuronConfig)
    missing = [name for name in KNOB_NAMES if name not in fields]
    assert missing == []


def test_new_fields_are_exactly_the_table():
    """Complement screen: the added field set equals the table, no extras.

    `test_knobs_are_fields_of_neuron_config` proves table -> class. This proves
    class -> table, so a field added by accident cannot hide outside the
    derivation record.
    """
    added = [f for f in _field_names(NeuronConfig) if f not in BASE_FIELDS]
    assert sorted(added) == sorted(KNOB_NAMES)


def test_field_count_delta():
    """Before/after with the base stated: 19 base fields + N knobs."""
    assert len(BASE_FIELDS) == 19
    assert len(_field_names(NeuronConfig)) == 19 + N


@pytest.mark.parametrize("name,default,probe,family", GLM5NEXT_KNOBS)
def test_knob_round_trips_through_construction(name, default, probe, family):
    """N/N: a value passed to the constructor reads back unchanged."""
    config = NeuronConfig(**{name: probe})
    assert getattr(config, name) == probe


@pytest.mark.parametrize("name,default,probe,family", GLM5NEXT_KNOBS)
def test_knob_round_trips_through_from_dict(name, default, probe, family):
    """N/N: and through `from_dict`, the documented additional_config path.

    A field absent from `from_dict` would be unreachable from
    `additional_config['neuron_config']` -- present as a field, dead as a knob.
    """
    config = NeuronConfig.from_dict({name: probe})
    assert getattr(config, name) == probe


# --------------------------------------------------------------------------
# Conjunct 2 -- N/N exact default values
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,default,probe,family", GLM5NEXT_KNOBS)
def test_knob_default_is_exact(name, default, probe, family):
    """N/N exact defaults, by identity for None and bool.

    Identity rather than equality: `0 == False` and `0.0 == False` both hold,
    so an equality check would accept a numeric zero where a bool is declared.
    """
    config = NeuronConfig()
    actual = getattr(config, name)
    if default is None:
        assert actual is None
    elif isinstance(default, bool):
        assert actual is default
    else:
        assert actual == default


@pytest.mark.parametrize("name,default,probe,family", GLM5NEXT_KNOBS)
def test_default_assertion_is_not_vacuous(name, default, probe, family):
    """Falsifiability arm for the defaults: an override must NOT read as default.

    This is the arm that kills the wrong-class failure mode. A class that
    swallowed constructor keywords would return the default here too, so every
    default assertion above would pass while no knob was settable at all.
    """
    config = NeuronConfig(**{name: probe})
    assert getattr(config, name) != default


def test_defaults_leave_existing_fields_untouched():
    """The added knobs change no pre-existing default."""
    config = NeuronConfig()
    assert config.fp8_packed_kv is False
    assert config.enable_structured_outputs is False
    assert config.ep_degree == 1
    assert config.quantization is None
    assert config.all2all_backend is None


# --------------------------------------------------------------------------
# Conjunct 3 -- an unknown knob raises, 1/1, and on the right class
# --------------------------------------------------------------------------


def test_unknown_knob_raises():
    """1/1: constructing `NeuronConfig` with an unknown knob raises.

    `TypeError` from the dataclass machinery satisfies "raises" as written; no
    custom error is authored for this.
    """
    with pytest.raises(TypeError):
        NeuronConfig(definitely_not_a_glm5next_knob=1)


def test_negative_control_kwargs_class_does_not_raise():
    """The control that pins the class: the file-mate SWALLOWS the same key.

    Read-only on `OnDeviceSamplingConfig` -- this measures the pin's existing
    behaviour, and it is why the conjunct above must be asserted against
    `NeuronConfig` specifically.
    """
    sampling = OnDeviceSamplingConfig(definitely_not_a_glm5next_knob=1)
    assert not hasattr(sampling, "definitely_not_a_glm5next_knob")


def test_knobs_landed_on_exactly_one_class():
    """None of the knobs leaked onto a sibling config class in the same file."""
    for cls in (TensorCaptureConfig, TensorReplacementConfig, VisionNeuronConfig):
        overlap = [name for name in KNOB_NAMES if name in _field_names(cls)]
        assert overlap == [], f"{cls.__name__}: {overlap}"
    sampling = OnDeviceSamplingConfig()
    for name in KNOB_NAMES:
        assert not hasattr(sampling, name), name
