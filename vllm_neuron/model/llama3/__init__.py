# SPDX-License-Identifier: Apache-2.0
from .config import LlamaConfig
from .factory import Eagle3LlamaForCausalLM, LlamaForCausalLM
from .model import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from .quantization import QuantizationSpec, QuantScheme, resolve_attention_mlp_classes

__all__ = [
    "LlamaConfig",
    "LlamaForCausalLM",
    "Eagle3LlamaForCausalLM",
    "LlamaAttention",
    "LlamaMLP",
    "LlamaRMSNorm",
    "LlamaRotaryEmbedding",
    "QuantizationSpec",
    "QuantScheme",
    "resolve_attention_mlp_classes",
]
