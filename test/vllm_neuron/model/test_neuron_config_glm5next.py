# SPDX-License-Identifier: Apache-2.0
"""``NeuronConfig`` knobs for the glm-5.3-Flash hybrid stack.

Each knob's default, its type and the refusal an out-of-range value raises.
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


# (name, declared default, probe value). Every probe differs from its default,
# so a round-trip cannot pass by reading back a value it never set.
GLM5NEXT_KNOBS = [
    ("enable_hybrid_kv_cache", False, True),
    ("hybrid_kv_block_size", None, 128),
    ("kda_state_dtype", None, "bfloat16"),
    ("kda_state_chunk_size", None, 64),
    ("mhc_sinkhorn_iters", None, 12),
    ("mhc_eps", None, 1e-05),
    ("blockwise_fp8", False, True),
    ("block_quant_scale_min", None, 1e-04),
]

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

KNOB_NAMES = [name for name, _, _ in GLM5NEXT_KNOBS]

#: Capabilities the model code sets for itself. They are fields of the same
#: dataclass but are not user-settable knobs, so the table below excludes them.
INTERNAL_FIELDS = ("_model_supports_independent_prefill_buckets",)


def _field_names(cls):
    return [f.name for f in dataclasses.fields(cls)]


def test_knobs_are_fields_of_neuron_config():
    """Each tabled name resolves to an actual dataclass field."""
    fields = _field_names(NeuronConfig)
    missing = [name for name in KNOB_NAMES if name not in fields]
    assert missing == []


def test_new_fields_are_exactly_the_table():
    """No knob outside the table was added alongside them."""
    added = [
        f for f in _field_names(NeuronConfig)
        if f not in BASE_FIELDS + INTERNAL_FIELDS
    ]
    assert sorted(added) == sorted(KNOB_NAMES)


@pytest.mark.parametrize("name,default,probe", GLM5NEXT_KNOBS)
def test_knob_round_trips_through_construction(name, default, probe):
    """A value passed to the constructor reads back unchanged."""
    assert probe != default, f"{name}: probe equals the default, so this is vacuous"
    config = NeuronConfig(**{name: probe})
    assert getattr(config, name) == probe


@pytest.mark.parametrize("name,default,probe", GLM5NEXT_KNOBS)
def test_knob_round_trips_through_from_dict(name, default, probe):
    """And through ``from_dict``, the documented ``additional_config`` path."""
    config = NeuronConfig.from_dict({name: probe})
    assert getattr(config, name) == probe


@pytest.mark.parametrize("name,default,probe", GLM5NEXT_KNOBS)
def test_knob_default_is_exact(name, default, probe):
    """Each knob's default, by identity for ``None`` and for bools."""
    config = NeuronConfig()
    actual = getattr(config, name)
    if default is None:
        assert actual is None
    elif isinstance(default, bool):
        assert actual is default
    else:
        assert actual == default


def test_defaults_leave_existing_fields_untouched():
    """The added knobs change no pre-existing default."""
    config = NeuronConfig()
    assert config.fp8_packed_kv is False
    assert config.enable_structured_outputs is False
    assert config.ep_degree == 1
    assert config.quantization is None
    assert config.all2all_backend is None


def test_unknown_knob_raises():
    """Constructing ``NeuronConfig`` with an unknown knob raises."""
    with pytest.raises(TypeError):
        NeuronConfig(definitely_not_a_glm5next_knob=1)


def test_sampling_config_ignores_unknown_keys():
    """Its file-mate swallows the same key, so the refusal above is class-specific."""
    sampling = OnDeviceSamplingConfig(definitely_not_a_glm5next_knob=1)
    assert not hasattr(sampling, "definitely_not_a_glm5next_knob")


def test_knobs_are_declared_on_exactly_one_class():
    """None of the knobs leaked onto a sibling config class in the same file."""
    for cls in (TensorCaptureConfig, TensorReplacementConfig, VisionNeuronConfig):
        overlap = [name for name in KNOB_NAMES if name in _field_names(cls)]
        assert overlap == [], f"{cls.__name__}: {overlap}"
    sampling = OnDeviceSamplingConfig()
    for name in KNOB_NAMES:
        assert not hasattr(sampling, name), name
