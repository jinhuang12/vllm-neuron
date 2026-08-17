# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3-VL model selection based on platform and configuration."""

import torch.nn as nn
from transformers import PretrainedConfig
from vllm.multimodal.inputs import MultiModalKwargsItem

from vllm_neuron.model.interfaces import (
    SupportsDisaggEncoder,
    SupportsMaxPixels,
    SupportsSpatialMerge,
)
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig


class Qwen3VLForConditionalGeneration(
    nn.Module, SupportsSpatialMerge, SupportsMaxPixels, SupportsDisaggEncoder
):
    """Factory that validates config and selects the appropriate Qwen3-VL implementation.

    The model runner passes `text_neuron_config` and `vision_neuron_config`
    for multimodal models.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

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
        cls._validate_config(hf_config, text_neuron_config)

        quantization = text_neuron_config.quantization if text_neuron_config else None

        if quantization == "mxfp8":
            from .model_mxfp8 import Qwen3VLForConditionalGeneration as Model
        else:
            from .model_bf16 import Qwen3VLForConditionalGeneration as Model

        return Model.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        return hf_config.vision_config.spatial_merge_size**2

    @classmethod
    def get_max_pixels_token_count(
        cls, hf_config: PretrainedConfig, max_pixels: int
    ) -> int:
        patch_size = hf_config.vision_config.patch_size
        return max_pixels // (patch_size**2)

    @classmethod
    def get_epd_kwargs(cls, item: MultiModalKwargsItem) -> MultiModalKwargsItem:
        # M-RoPE needs image_grid_thw on LM pool, send over HTTP
        return MultiModalKwargsItem({"image_grid_thw": item["image_grid_thw"]})

    @classmethod
    def _validate_config(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
    ) -> None:
        # Reject an unrecognized quantization string at config-load time. Without
        # this, a typo (e.g. "mxfp-8") silently routes to bf16 and produces a
        # baffling baseline measurement instead of failing loudly. None / "bf16"
        # are the default bf16 path; "mxfp8" selects the CPU-dequant model
        # (model_mxfp8).
        quantization = text_neuron_config.quantization if text_neuron_config else None
        if quantization not in (None, "bf16", "mxfp8"):
            raise ValueError(
                f"Unknown quantization {quantization!r} for Qwen3-VL. "
                "Expected one of: None, 'bf16', 'mxfp8'."
            )
