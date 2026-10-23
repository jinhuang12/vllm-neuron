# SPDX-License-Identifier: Apache-2.0
"""Factory for Kimi K3 model selection."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class KimiK3ForConditionalGeneration(nn.Module):
    """Factory that selects the Kimi K3 language model implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry requirements. The
    implementation module is imported only when a model is built, so the
    registry lookup imports no model code.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        """Delegate the forward pass to the selected implementation."""
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None
    ) -> nn.Module:
        """Create the model from configs and return the implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Build the language model implementation."""
        from .model import KimiK3ForConditionalGeneration as Model

        return Model.from_configs(hf_config, neuron_config)
