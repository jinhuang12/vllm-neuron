# SPDX-License-Identifier: Apache-2.0
"""Factory for Kimi K3 model selection."""

import math

import torch.nn as nn
from transformers import PretrainedConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_neuron.model.interfaces import SupportsEncoderBudget, SupportsSpatialMerge
from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig

from .config import KimiK3VisionConfig


# Upstream's K3 processor classes live in a package whose import also loads the
# upstream model, so each factory imports them on first use.
def _processing_info(ctx):
    """Build upstream's K3 processing info."""
    from vllm.models.kimi_k3.common.mm_preprocess import KimiK3ProcessingInfo

    return KimiK3ProcessingInfo(ctx)


def _dummy_inputs(info):
    """Build upstream's K3 dummy-input builder."""
    from vllm.models.kimi_k3.common.mm_preprocess import KimiK3DummyInputsBuilder

    return KimiK3DummyInputsBuilder(info)


def _processor(info, dummy_inputs, *, cache=None):
    """Build the K3 image processor that gives uint8 patches."""
    from .vision_processor import KimiK3Uint8Processor

    return KimiK3Uint8Processor(info, dummy_inputs, cache=cache)


@MULTIMODAL_REGISTRY.register_processor(
    _processor, info=_processing_info, dummy_inputs=_dummy_inputs
)
class KimiK3ForConditionalGeneration(
    nn.Module, SupportsSpatialMerge, SupportsEncoderBudget
):
    """Factory that selects the Kimi K3 model implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry requirements. The
    implementation module is imported only when a model is built, so the
    registry lookup imports no model code. The model runner passes
    ``text_neuron_config`` and ``encoder_neuron_config`` for a multimodal model.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config, text_neuron_config, encoder_neuron_config
        )

    def forward(self, *args, **kwargs):
        """Delegate the forward pass to the selected implementation."""
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> nn.Module:
        """Create the model from configs and return the implementation directly."""
        return cls._select_implementation(
            hf_config, text_neuron_config, encoder_neuron_config
        )

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
        encoder_neuron_config: EncoderNeuronConfig | None,
    ) -> nn.Module:
        """Build the language model, with the vision tower when an encoder config is given."""
        if encoder_neuron_config is None:
            from .model import KimiK3ForConditionalGeneration as Model

            return Model.from_configs(hf_config, text_neuron_config)
        from .vision_encoder import KimiK3VisionForConditionalGeneration as Model

        return Model.from_configs(hf_config, text_neuron_config, encoder_neuron_config)

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        """Return how many patches merge into one image token."""
        return KimiK3VisionConfig.from_hf_config(hf_config.vision_config).merge_factor

    @classmethod
    def get_max_encoder_embed_budget(
        cls,
        hf_config: PretrainedConfig,
        encoder_neuron_config: EncoderNeuronConfig,
    ) -> int | None:
        """Return the image tokens the largest bucket holds, or None with no buckets."""
        # Vision buckets count patches; the merge turns them into image tokens.
        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return None
        return max(buckets) // cls.get_vision_token_merge_factor(hf_config)

    @classmethod
    def get_max_encoder_blocks_per_request(
        cls,
        hf_config: PretrainedConfig,
        encoder_neuron_config: EncoderNeuronConfig,
    ) -> int:
        """Return the encoder-cache blocks the largest bucket fills, or 0 with no buckets."""
        from vllm_neuron.utils.mm_utils import get_encoder_cache_block_size

        budget = cls.get_max_encoder_embed_budget(hf_config, encoder_neuron_config)
        if budget is None:
            return 0
        cache_block_size = get_encoder_cache_block_size(
            hf_config, encoder_neuron_config.encoder_attention_block_size
        )
        return math.ceil(budget / cache_block_size)
