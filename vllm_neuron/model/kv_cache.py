# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch


@dataclass
class LayerSpec:
    """
    Defines the KV cache specification for a single transformer layer.

    Used to specify the memory requirements and configuration for storing
    key-value pairs in the attention mechanism of a transformer layer.
    """

    name: str
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window_size: int | None = None
    chunk_size: int | None = None

    # A linear-attention (KDA) layer holds no key/value pair: it holds a
    # short-convolution state and a recurrent state, described by these four
    # fields and read by the layers that allocate the buffers. Shapes and dtypes
    # come in pairs, matching vLLM's MambaStateShapeCalculator.kda_state_shape
    # and MambaStateDtypeCalculator.kda_state_dtype. The conv state's extent
    # order follows vLLM's VLLM_SSM_CONV_STATE_LAYOUT ("SD" by default), so a
    # producer must store the order it read rather than assume one.
    kda_conv_state_shape: tuple[int, int] | None = None
    kda_recurrent_state_shape: tuple[int, int, int] | None = None
    kda_conv_state_dtype: torch.dtype | None = None
    kda_recurrent_state_dtype: torch.dtype | None = None
    # A latent-attention (MLA) layer caches one compressed vector per token and
    # has no value half to store, so its page holds one buffer where a key/value
    # layer's holds two. Set for the layers the allocator has to describe with
    # MLAAttentionSpec, whose page carries no second term.
    latent_kv: bool = False


@dataclass
class KVSpec:
    """
    Defines the KV cache needs of a model by specifying all layer configurations.

    Contains a list of LayerSpec objects that collectively define the complete
    KV cache requirements for an entire transformer model.
    """

    layers: list[LayerSpec]
