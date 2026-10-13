# SPDX-License-Identifier: Apache-2.0
"""Vision encoder utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from transformers import PretrainedConfig

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalKwargsItem


def get_vision_token_merge_factor(hf_config: PretrainedConfig) -> int:
    """Return the vision token merge factor for a model.

    Looks up the model class via registry and calls its
    SupportsSpatialMerge.get_vision_token_merge_factor classmethod.
    Returns 1 if the model doesn't implement the protocol.

    Args:
        hf_config: HuggingFace PretrainedConfig for the model.

    Returns:
        Number of raw vision tokens that collapse into one embed token.
        E.g., 4 for Qwen3-VL (spatial_merge_size=2, so 2x2=4).
    """
    from vllm_neuron.model.interfaces import SupportsSpatialMerge
    from vllm_neuron.model.registry import get_models

    arch = hf_config.architectures[0]
    model_cls = dict(get_models()).get(arch)
    if model_cls and issubclass(model_cls, SupportsSpatialMerge):
        return model_cls.get_vision_token_merge_factor(hf_config)
    return 1


def get_max_pixels_token_count(
    hf_config: PretrainedConfig, max_pixels: int
) -> int | None:
    """Return the raw vision-token count for a max_pixels cap, or None.

    Looks up the model class via registry and calls its
    SupportsMaxPixels.get_max_pixels_token_count classmethod. Returns None
    when the model does not implement the protocol.

    Args:
        hf_config: HuggingFace PretrainedConfig for the model.
        max_pixels: Per-image pixel cap from mm_processor_kwargs.

    Returns:
        Raw (pre-merge) vision token count, or None if unsupported.
    """
    from vllm_neuron.model.interfaces import SupportsMaxPixels
    from vllm_neuron.model.registry import get_models

    arch = hf_config.architectures[0]
    model_cls = dict(get_models()).get(arch)
    if model_cls and issubclass(model_cls, SupportsMaxPixels):
        return model_cls.get_max_pixels_token_count(hf_config, max_pixels)
    return None


@dataclass(frozen=True)
class VisionEmbedLimit:
    """The most embeddings one image may produce on this server.

    The vision encoder is compiled for the ``num_vision_tokens_buckets`` sizes, which
    count raw (pre-merge) vision tokens. The language model receives one embedding per
    ``merge_factor`` raw tokens, so an image with more than
    ``max_bucket // merge_factor`` embeddings fits no compiled bucket.
    """

    #: Largest entry of ``vision_neuron_config.num_vision_tokens_buckets``.
    max_bucket: int
    #: Raw vision tokens per embedding (:func:`get_vision_token_merge_factor`).
    merge_factor: int

    @property
    def embeds(self) -> int:
        """The limit, in embeddings: what ``PlaceholderRange.get_num_embeds`` counts."""
        return max(1, self.max_bucket // self.merge_factor)


def get_vision_embed_limit(vllm_config) -> VisionEmbedLimit | None:
    """The per-image embedding limit for a served config, or None without vision buckets.

    The one source for both places that enforce the limit: the request check
    (``NeuronPlatform.validate_request``, through the admission policy) and the
    scheduler's per-step encoder budget (``NeuronScheduler``). Read after the platform
    has resolved ``vision_neuron_config``.
    """
    vision_config = (vllm_config.additional_config or {}).get("vision_neuron_config")
    buckets = vision_config.get("num_vision_tokens_buckets") if vision_config else None
    if not buckets:
        return None
    merge_factor = get_vision_token_merge_factor(vllm_config.model_config.hf_config)
    return VisionEmbedLimit(max_bucket=max(buckets), merge_factor=max(merge_factor, 1))


def get_epd_kwargs(
    hf_config: PretrainedConfig, item: "MultiModalKwargsItem"
) -> "MultiModalKwargsItem":
    """Return the subset of an item's mm kwargs the EPD LM pool needs.

    Looks up the model class via registry and calls its
    SupportsDisaggEncoder.get_epd_kwargs classmethod, which drops encoder-only
    tensors (e.g. pixel_values) the LM pool pulls out-of-band. Returns the item
    unchanged when the model doesn't implement the protocol.

    Args:
        hf_config: HuggingFace PretrainedConfig for the model.
        item: The full preprocessed mm kwargs for one media item.

    Returns:
        The LM-relevant mm kwargs item.
    """
    from vllm_neuron.model.interfaces import SupportsDisaggEncoder
    from vllm_neuron.model.registry import get_models

    arch = hf_config.architectures[0]
    model_cls = dict(get_models()).get(arch)
    if model_cls and issubclass(model_cls, SupportsDisaggEncoder):
        return model_cls.get_epd_kwargs(item)
    return item
