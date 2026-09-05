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

    # ---- KDA recurrent-state geometry (hybrid KDA/DSA stacks) -------------
    # A linear-attention (KDA) layer does not hold a key/value pair: it holds a
    # short-convolution state and a recurrent state. These four fields are the
    # vocabulary for describing them; the layers that allocate the buffers are
    # the readers. All four are optional and default to None, so a model wired
    # for a uniform cache keeps its resolved spec byte-for-byte, and they are
    # APPENDED after chunk_size so the six-argument positional form above stays
    # unbroken by construction rather than by luck.
    #
    # Names and arities follow vLLM's own state calculators -- the fork's
    # authority for this geometry, not a local coinage:
    # MambaStateShapeCalculator.kda_state_shape returns
    # (conv_state_shape, recurrent_state_shape) at ranks 2 and 3, and
    # MambaStateDtypeCalculator.kda_state_dtype returns the matching dtype
    # pair. The kda_state_ prefix is the fork's landed NeuronConfig convention
    # (neuron_config.py:184,188). The conv state's extent ORDER is chosen by
    # vLLM's VLLM_SSM_CONV_STATE_LAYOUT ("SD" by default), so a producer must
    # store the order it read rather than assume one.
    kda_conv_state_shape: tuple[int, int] | None = None
    kda_recurrent_state_shape: tuple[int, int, int] | None = None
    kda_conv_state_dtype: torch.dtype | None = None
    kda_recurrent_state_dtype: torch.dtype | None = None


@dataclass
class KVSpec:
    """
    Defines the KV cache needs of a model by specifying all layer configurations.

    Contains a list of LayerSpec objects that collectively define the complete
    KV cache requirements for an entire transformer model.
    """

    layers: list[LayerSpec]
