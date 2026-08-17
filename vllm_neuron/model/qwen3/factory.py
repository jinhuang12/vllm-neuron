# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3 model selection based on platform and configuration."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3ForCausalLM(nn.Module):
    """Factory that validates config and selects the Qwen3 implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)

        # Qwen3-Embedding-8B's HF architectures field is "Qwen3ForCausalLM", so
        # both the generative and pooling models resolve to this factory.
        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
        if model_config.runner_type == "pooling":
            # Two pooling cases share this arch, split by the checkpoint:
            #   - native sentence-transformers embedding checkpoint (has
            #     modules.json, e.g. Qwen3-Embedding-8B) -> dedicated hand-written
            #     Qwen3ForEmbedding.
            #   - a plain generative checkpoint run with --convert embed -> the
            #     generic pooling adapter (applied at the load_model seam), so
            #     the factory just builds the generative model here.
            from vllm.transformers_utils.config import get_pooling_config

            is_native_st_embedding = (
                get_pooling_config(model_config.model, model_config.revision)
                is not None
            )
            if is_native_st_embedding:
                from .model_embedding import Qwen3ForEmbedding as Model
            else:
                from .model import Qwen3ForCausalLM as Model
        else:
            from .model import Qwen3ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the configuration is supported.

        Qwen3-32B dense currently ships BF16 only. Reject quantized configs
        up front so users get a clear error instead of a cryptic failure
        deep in weight loading.
        """
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for Qwen3-32B "
                "dense. Only BF16 (quantization=None or 'bf16') is implemented."
            )
