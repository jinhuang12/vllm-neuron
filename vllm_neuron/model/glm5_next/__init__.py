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

# Bind the vision bridge to the model class here, in the package that vLLM's own lazy name
# "vllm_neuron.model.glm5_next:Glm5NextForConditionalGeneration" imports. That import is what resolves the
# architecture, so this call has run before vLLM can take the class attribute -- no import order to arrange.
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
