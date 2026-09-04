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


@dataclass
class KVSpec:
    """
    Defines the KV cache needs of a model by specifying all layer configurations.

    Contains a list of LayerSpec objects that collectively define the complete
    KV cache requirements for an entire transformer model.
    """

    layers: list[LayerSpec]


def resolve_layer_cache_dtype(
    layer_dtype: torch.dtype, requested_dtype: torch.dtype
) -> torch.dtype:
    """Apply a global KV dtype only to floating-point cache entries.

    Model-declared nonfloating entries carry encoded bytes or bookkeeping
    values. Reinterpreting those entries as a floating cache dtype changes
    their storage contract.
    """

    if not layer_dtype.is_floating_point:
        return layer_dtype
    return requested_dtype
