# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from transformers import PretrainedConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem

from vllm_neuron.model.neuron_config import VisionNeuronConfig


@runtime_checkable
class SupportsVisionWarmup(Protocol):
    """Models that support vision encoder warmup implement this interface."""

    def build_vision_synthetic_inputs(
        self,
        bucket: int,
        vision_neuron_config: VisionNeuronConfig,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Construct shape-only tensors matching the vision encoder forward signature."""
        ...


@runtime_checkable
class SupportsSpatialMerge(Protocol):
    """Models that perform spatial token merging in the vision encoder."""

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        """Return how many raw vision tokens collapse into one embed token."""
        ...


@runtime_checkable
class SupportsMaxPixels(Protocol):
    """Models whose image processor supports a max_pixels resize cap.

    Models that honor it own the math that maps the pixel budget to a raw
    (pre-merge) vision-token count, since patch tiling differs across
    architectures.
    """

    @classmethod
    def get_max_pixels_token_count(
        cls, hf_config: PretrainedConfig, max_pixels: int
    ) -> int:
        """Convert a max_pixels cap into a raw vision-token count.

        Args:
            hf_config: HuggingFace PretrainedConfig for the model.
            max_pixels: Per-image pixel cap from mm_processor_kwargs.

        Returns:
            Raw (pre-merge) vision token count for that pixel budget.
        """
        ...


@runtime_checkable
class SupportsDisaggEncoder(Protocol):
    """Models split across an encoder pool and a language-model pool (EPD).

    The LM pool pulls the encoded embedding out-of-band (over NIXL), so it needs
    only the lightweight per-item metadata, not the heavy encoder-side tensors.
    """

    @classmethod
    def get_epd_kwargs(cls, item: MultiModalKwargsItem) -> MultiModalKwargsItem:
        """Return the subset of an item's mm kwargs the LM pool needs.

        Args:
            item: The full preprocessed mm kwargs for one media item.

        Returns:
            The LM-relevant subset (e.g. image_grid_thw for M-RoPE), dropping
            encoder-only tensors like pixel_values that the LM pool pulls over
            NIXL instead.
        """
        ...


@runtime_checkable
class SupportsMRoPE(Protocol):
    """Models that provide 3D multimodal rotary position embeddings."""

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """Compute 3D M-RoPE positions for the given prompt.

        Args:
            input_tokens: Full prompt token IDs.
            mm_features: Multimodal features attached to the request.

        Returns:
            (positions, delta) where positions is [3, seq_len] int64 tensor
            and delta is the offset for computing decode-phase positions.
        """
        ...
