# SPDX-License-Identifier: Apache-2.0
"""Factory for GLM-5.3-Flash (``glm5_next``) implementation selection.

Follows the package convention every other arch in this tree uses
(``qwen3/factory.py``, ``qwen3_vl/factory.py``): this module *defines* the
arch-named class that the registry registers, the class extends ``nn.Module``
so vLLM's ``ModelRegistry`` accepts it, and the concrete implementation module
is imported lazily inside the selection classmethod.

The lazy import is load-bearing, not stylistic: importing this module -- and
looking its class up through ``vllm_neuron.model.registry`` -- must never pull
in model code or allocate weights.
"""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig


class Glm5NextForConditionalGeneration(nn.Module):
    """Factory that selects the GLM-5.3-Flash implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry requirements.

    The model runner passes ``text_neuron_config`` and ``vision_neuron_config``
    separately because the text decoder and the vision encoder carry their own
    parallelism and compilation settings -- the same split that
    ``Glm5NextConfig.from_configs`` already models.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def embed_input_ids(self, input_ids):
        """Boundary member: present so config-time interface validation passes.

        This class is a selection seam, never a compute path -- the selected
        implementation owns embedding. No call site for this method exists in
        ``vllm_neuron``, so the raise is the permanent contract.
        """
        raise NotImplementedError(
            "Glm5NextForConditionalGeneration is a selection factory; "
            "embed_input_ids belongs to the selected implementation."
        )

    def compute_logits(self, hidden_states):
        """Boundary member: present so config-time interface validation passes.

        Same contract as ``embed_input_ids`` -- the selected implementation
        owns logits, and no call site for this method exists in
        ``vllm_neuron``.
        """
        raise NotImplementedError(
            "Glm5NextForConditionalGeneration is a selection factory; "
            "compute_logits belongs to the selected implementation."
        )

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> nn.Module:
        return cls._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
        vision_neuron_config: VisionNeuronConfig | None,
    ) -> nn.Module:
        # Blockwise-FP8 is the only weight format in scope for this checkpoint,
        # so there is a single implementation module and no format branch here.
        # The import stays local so that registration and arch lookup work
        # without importing model code or allocating weights.
        from .model_fp8 import Glm5NextForConditionalGeneration as Model

        return Model.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )
