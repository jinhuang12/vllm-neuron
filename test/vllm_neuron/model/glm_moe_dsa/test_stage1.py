# SPDX-License-Identifier: Apache-2.0
"""Focused CPU/meta tests for GLM-5.2 Stage 1 registration."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig
from vllm import ModelRegistry

from vllm_neuron.model.glm_moe_dsa import (
    GlmMoeDsaConfig,
    GlmMoeDsaForCausalLM,
)
from vllm_neuron.model.registry import get_models
from vllm_neuron.vllm.platform import NeuronPlatform

MODEL_PATH_VALUE = os.environ.get("GLM52_MODEL_PATH")
MODEL_PATH = Path(MODEL_PATH_VALUE or ".")


@pytest.fixture(scope="module")
def hf_config():
    if MODEL_PATH_VALUE is None:
        pytest.skip("GLM52_MODEL_PATH is required for pinned-model tests")
    return AutoConfig.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False
    )


def test_config_adapter_matches_pinned_contract(hf_config):
    config = GlmMoeDsaConfig.from_configs(hf_config)

    assert config.hidden_size == 6144
    assert config.num_attention_heads == 64
    assert config.q_lora_rank == 2048
    assert config.kv_lora_rank == 512
    assert config.qk_nope_head_dim == 192
    assert config.qk_rope_head_dim == 64
    assert config.v_head_dim == 256
    assert config.num_hidden_layers == 78
    assert config.n_routed_experts == 256
    assert config.num_experts_per_tok == 8
    assert config.first_k_dense_replace == 3
    assert config.index_topk == 2048
    assert config.model_type == "glm_moe_dsa"
    assert config.quantization_config["quant_method"] == "fp8"
    assert config.quantization_config["fmt"] == "e4m3"


def test_model_registry_resolution():
    registered = dict(get_models())
    assert registered["GlmMoeDsaForCausalLM"] is GlmMoeDsaForCausalLM

    ModelRegistry.register_model("GlmMoeDsaForCausalLM", GlmMoeDsaForCausalLM)
    registry_config = SimpleNamespace(
        model_impl="auto", convert_type="none", runner_type="generate"
    )
    resolved_cls, resolved_arch = ModelRegistry.resolve_model_cls(
        "GlmMoeDsaForCausalLM", model_config=registry_config
    )

    assert resolved_cls is GlmMoeDsaForCausalLM
    assert resolved_arch == "GlmMoeDsaForCausalLM"


def test_platform_keeps_upstream_glm_registration_until_worker_start(monkeypatch):
    """ModelConfig must inspect vLLM's generate-capable GLM class first."""

    monkeypatch.delenv("VLLM_NEURON_SYNTHETIC_MODEL", raising=False)
    with patch(
        "vllm.model_executor.models.registry.ModelRegistry.register_model"
    ) as register_model:
        NeuronPlatform.pre_register_and_update()

    register_model.assert_not_called()
    assert dict(get_models())["GlmMoeDsaForCausalLM"] is GlmMoeDsaForCausalLM


def test_meta_factory_builds_only_main_layers(hf_config):
    tp_group = SimpleNamespace(device_group=object())
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=64),
        patch("torch.distributed.get_rank", return_value=0),
        patch(
            "vllm_neuron.model.glm_moe_dsa.factory.GlmMoeDsaForCausalLM."
            "_resolve_tensor_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm_neuron.model.glm_moe_dsa.model._resolve_tp_groups",
            return_value=(tp_group, tp_group.device_group),
        ),
        torch.device("meta"),
    ):
        model = GlmMoeDsaForCausalLM.from_configs(
            hf_config, neuron_config=None, tensor_parallel_size=64
        )

    assert len(model.model.layers) == 78
    assert model.main_layer_indices == tuple(range(78))
    assert model.excluded_mtp_layer_indices == (78,)
    assert [layer.layer_idx for layer in model.model.layers[:3]] == [0, 1, 2]
    assert all(layer.is_dense for layer in model.model.layers[:3])
    assert model.model.layers[3].is_moe
    parameters = dict(model.named_parameters())
    assert len(parameters) == 3_660
    assert sum(name.endswith(".weight_scale_inv") for name in parameters) == 1_566
    assert sum(parameter.numel() for parameter in parameters.values()) > 0


def test_tp64_positive_and_negative_validation(hf_config):
    tp_group = SimpleNamespace(device_group=object())
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=64),
        patch("torch.distributed.get_rank", return_value=0),
        patch(
            "vllm_neuron.model.glm_moe_dsa.factory.GlmMoeDsaForCausalLM."
            "_resolve_tensor_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm_neuron.model.glm_moe_dsa.model._resolve_tp_groups",
            return_value=(tp_group, tp_group.device_group),
        ),
        torch.device("meta"),
    ):
        model = GlmMoeDsaForCausalLM.from_configs(
            hf_config, neuron_config=None, tensor_parallel_size=64
        )
    assert model.tensor_parallel_size == 64

    with pytest.raises(ValueError, match="tensor_parallel_size=64"):
        GlmMoeDsaForCausalLM.from_configs(
            hf_config, neuron_config=None, tensor_parallel_size=32
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config.update(model_type="other"), "model_type"),
        (
            lambda config: config.update(architectures=["OtherForCausalLM"]),
            "architectures",
        ),
        (
            lambda config: config["quantization_config"].update(
                quant_method="unsupported"
            ),
            "quantization_config",
        ),
    ],
)
def test_rejects_wrong_model_identity_or_quantization(hf_config, mutation, message):
    config_dict = hf_config.to_dict()
    config_dict["quantization_config"] = dict(config_dict["quantization_config"])
    mutation(config_dict)

    with pytest.raises(ValueError, match=message):
        GlmMoeDsaConfig.from_configs(config_dict)
