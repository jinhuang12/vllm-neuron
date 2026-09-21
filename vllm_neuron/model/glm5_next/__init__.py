# SPDX-License-Identifier: Apache-2.0
from .config import (
    DSA_LAYER_TYPE,
    KDA_LAYER_TYPE,
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
    default_layer_types,
)
from .factory import Glm5NextForConditionalGeneration
from .utils.vision_preprocessing import register_glm5_next_multimodal

# Bind the vision bridge to the model class here, in the package vLLM's own lazy
# name imports to resolve the architecture. That import is what runs this call, so
# it has happened before vLLM can take the class attribute -- no import order to
# arrange.
register_glm5_next_multimodal(Glm5NextForConditionalGeneration)

__all__ = [
    "DSA_LAYER_TYPE",
    "Glm5NextConfig",
    "Glm5NextForConditionalGeneration",
    "Glm5NextTextConfig",
    "Glm5NextVisionConfig",
    "KDA_LAYER_TYPE",
    "default_layer_types",
    "register_glm5_next_multimodal",
]
