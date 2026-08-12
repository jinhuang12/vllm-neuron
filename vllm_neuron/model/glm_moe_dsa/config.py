# SPDX-License-Identifier: Apache-2.0
"""Focused Hugging Face config adapter for the pinned GLM-5.2 checkpoint."""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, cast

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class GlmMoeDsaConfig:
    """Neuron-side view of the GLM-5.2 architecture used by this port.

    ``num_hidden_layers`` contains the 78 main decoder layers. The one MTP
    layer follows them at index 78 and is deliberately outside main execution.
    """

    model_type: str = "glm_moe_dsa"
    vocab_size: int = 154880
    hidden_size: int = 6144
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_nextn_predict_layers: int = 1
    num_attention_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    n_group: int = 1
    topk_group: int = 1
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    hidden_act: str = "silu"
    first_k_dense_replace: int = 3
    index_topk: int = 2048
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    rope_parameters: dict[str, Any] = field(default_factory=dict)
    quantization_config: dict[str, Any] = field(default_factory=dict)
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    _EXPECTED_ARCHITECTURE = "GlmMoeDsaForCausalLM"
    _CHECKPOINT_QUANTIZATION_FIELD = "glm52_checkpoint_quantization_config"
    _EXPECTED_VALUES: ClassVar[dict[str, Any]] = {
        "model_type": "glm_moe_dsa",
        "vocab_size": 154880,
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "num_attention_heads": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "num_hidden_layers": 78,
        "num_nextn_predict_layers": 1,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "hidden_act": "silu",
        "first_k_dense_replace": 3,
        "index_topk": 2048,
        "max_position_embeddings": 1048576,
    }
    _EXPECTED_QUANTIZATION: ClassVar[dict[str, Any]] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }

    def __post_init__(self) -> None:
        mismatches = [
            f"{name}={getattr(self, name)!r} (expected {expected!r})"
            for name, expected in self._EXPECTED_VALUES.items()
            if getattr(self, name) != expected
        ]
        if mismatches:
            raise ValueError(
                "Unsupported GLM-5.2 architecture values: " + ", ".join(mismatches)
            )
        if self.torch_dtype is not torch.bfloat16:
            raise ValueError(
                f"GLM-5.2 requires torch_dtype=torch.bfloat16; got {self.torch_dtype!r}"
            )

    @property
    def main_layer_indices(self) -> tuple[int, ...]:
        """Indices used by main decoder execution."""
        return tuple(range(self.num_hidden_layers))

    @property
    def mtp_layer_indices(self) -> tuple[int, ...]:
        """MTP indices reserved for later work and excluded from main execution."""
        start = self.num_hidden_layers
        return tuple(range(start, start + self.num_nextn_predict_layers))

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict[str, Any] | str | Path,
        neuron_config: NeuronConfig | None = None,
    ) -> "GlmMoeDsaConfig":
        config_dict = cls._as_dict(hf_config)
        architectures = config_dict.get("architectures") or []
        if cls._EXPECTED_ARCHITECTURE not in architectures:
            raise ValueError(
                "GLM-5.2 requires architectures to contain "
                f"{cls._EXPECTED_ARCHITECTURE!r}; got {architectures!r}"
            )

        quantization_config = config_dict.get("quantization_config")
        if quantization_config is None:
            # vLLM validates the generic Hugging Face quantization label before
            # the Neuron worker can select this architecture-specific loader.
            # Launch code may preserve the checkpoint metadata here while it
            # clears only the generic field through hf_overrides.
            quantization_config = config_dict.get(cls._CHECKPOINT_QUANTIZATION_FIELD)
        if not isinstance(quantization_config, dict):
            raise TypeError(
                "GLM-5.2 requires the pinned FP8 quantization_config mapping"
            )
        quantization_mismatches = [
            f"{name}={quantization_config.get(name)!r} (expected {expected!r})"
            for name, expected in cls._EXPECTED_QUANTIZATION.items()
            if quantization_config.get(name) != expected
        ]
        if quantization_mismatches:
            raise ValueError(
                "Unsupported GLM-5.2 quantization_config: "
                + ", ".join(quantization_mismatches)
            )

        field_names = {item.name for item in fields(cls)}
        filtered = {
            name: value for name, value in config_dict.items() if name in field_names
        }
        filtered["quantization_config"] = quantization_config

        if "torch_dtype" not in filtered and "dtype" in config_dict:
            filtered["torch_dtype"] = config_dict["dtype"]
        dtype = filtered.get("torch_dtype")
        if isinstance(dtype, str):
            try:
                filtered["torch_dtype"] = getattr(torch, dtype)
            except AttributeError as error:
                raise ValueError(f"Unsupported torch dtype {dtype!r}") from error

        filtered["neuron_config"] = neuron_config
        return cls(**filtered)

    @staticmethod
    def _as_dict(
        hf_config: PretrainedConfig | dict[str, Any] | str | Path,
    ) -> dict[str, Any]:
        if isinstance(hf_config, (str, Path)):
            with Path(hf_config).open() as config_file:
                return cast(dict[str, Any], json.load(config_file))
        if isinstance(hf_config, PretrainedConfig):
            return cast(dict[str, Any], hf_config.to_dict())
        if isinstance(hf_config, dict):
            return dict(hf_config)
        raise TypeError(f"Unsupported config type: {type(hf_config)!r}")
