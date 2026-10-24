# SPDX-License-Identifier: Apache-2.0
"""Give every process the plugin's uint8 Kimi K3 image processor.

The plugin registers its model classes with vLLM's model registry only in the
worker, so the API process resolves upstream's K3 class and builds its
normalizing processor, while the plugin's patch embed reads raw uint8 values.
This patch wraps the multimodal registry's class lookup: where the lookup
resolves upstream's K3 class, it returns the plugin's K3 class, whose processor
factories build the uint8 processor. Every other model keeps the lookup's
result. The plugin's class is imported at the first K3 lookup, not at patch time.
"""

from typing import Any

_UPSTREAM_PACKAGE = "vllm.models.kimi_k3"
_CLASS_NAME = "KimiK3ForConditionalGeneration"

_applied = False
_original_get_model_cls: Any = None


def _get_model_cls(self, model_config):
    """Resolve the model class; upstream's K3 class resolves to the plugin's."""
    model_cls = _original_get_model_cls(self, model_config)
    if model_cls.__name__ == _CLASS_NAME and model_cls.__module__.startswith(_UPSTREAM_PACKAGE + "."):
        from vllm_neuron.model.kimi_k3.factory import KimiK3ForConditionalGeneration

        return KimiK3ForConditionalGeneration
    return model_cls


def apply_kimi_k3_processor_patch() -> None:
    """Wrap ``MultiModalRegistry._get_model_cls`` once."""
    global _applied, _original_get_model_cls

    if _applied:
        return
    from vllm.multimodal.registry import MultiModalRegistry

    _original_get_model_cls = MultiModalRegistry._get_model_cls
    MultiModalRegistry._get_model_cls = _get_model_cls
    _applied = True
