# SPDX-License-Identifier: Apache-2.0
"""Config-time architecture registration for ``Glm5Next``.

The subject is ``NeuronPlatform.pre_register_and_update``, which
``EngineArgs.create_engine_config`` calls before it resolves the model, so the
fork's factory class must win the architecture lookup for the fixture checkpoint
and no Transformers fallback may be taken.

The engine config is built once per process and every test reads that one
resolution: building it is the expensive part and nothing here mutates it.
"""

from __future__ import annotations

import os
from pathlib import Path

from vllm.model_executor.models.interfaces_base import (
    is_text_generation_model,
    is_vllm_model,
)
from vllm.model_executor.models.registry import ModelRegistry

from vllm_neuron.model.glm5_next import Glm5NextForConditionalGeneration
from vllm_neuron.vllm.platform import NeuronPlatform

ARCH = "Glm5NextForConditionalGeneration"
FIXTURE = "test/vllm_neuron/model/glm5_next/fixtures/"
DEFINING_MODULE = "vllm_neuron.model.glm5_next.factory"

_RESOLUTION: dict | None = None


def _resolve_once() -> dict:
    """Build the engine config once per process and keep every observable."""
    global _RESOLUTION
    if _RESOLUTION is not None:
        return _RESOLUTION

    from vllm.engine.arg_utils import EngineArgs

    out: dict = {
        "raised": None,
        "config": None,
        "arch_in_process": None,
        "resolved_name": None,
        "resolved_module": None,
    }
    try:
        out["config"] = EngineArgs(model=FIXTURE).create_engine_config()
    except BaseException as exc:  # kept, then asserted on below
        out["raised"] = f"{type(exc).__name__}: {exc}"

    if out["config"] is not None:
        model_config = out["config"].model_config
        out["arch_in_process"] = model_config.architecture
        # Both readings below come from one class object out of one
        # resolve_model_cls call, never off _model_info.architecture, which a
        # warm model-info cache reconstructs from JSON. That is what keeps this
        # a reading about the registered class rather than about the cache.
        resolved = ModelRegistry.resolve_model_cls(
            model_config.architectures, model_config
        )[0]
        out["resolved_name"] = resolved.__name__
        out["resolved_module"] = resolved.__module__

    _RESOLUTION = out
    return out


def test_glm5next_resolves_before_validation() -> None:
    """The fixture checkpoint resolves to the fork's class, not to a fallback."""
    assert Path(FIXTURE).is_dir(), f"fixture unreachable from cwd: {FIXTURE}"

    r = _resolve_once()

    # With the architecture unregistered this call raises before it reaches the
    # model at all, so a config that built is itself the reading that the hook
    # ran ahead of ModelConfig validation.
    assert r["raised"] is None, f"create_engine_config raised: {r['raised']}"
    assert r["config"] is not None

    # One value, two readings: the config's own attribute, and the class the
    # registry hands back when asked for that architecture.
    assert r["arch_in_process"] == ARCH
    assert r["resolved_name"] == ARCH
    assert r["resolved_module"] == DEFINING_MODULE


def test_registry_contains_glm5next_after_hook(monkeypatch) -> None:
    """The registration hook adds the architecture and not the synthetic model."""
    ambient = os.environ.get("VLLM_NEURON_SYNTHETIC_MODEL")
    assert ambient != "1", (
        "this test is stated for VLLM_NEURON_SYNTHETIC_MODEL unset; the ambient "
        f"value was {ambient!r}"
    )
    monkeypatch.delenv("VLLM_NEURON_SYNTHETIC_MODEL", raising=False)

    NeuronPlatform.pre_register_and_update()
    archs = set(ModelRegistry.get_supported_archs())

    assert ARCH in archs
    assert "SyntheticNeuronModel" not in archs


def test_no_transformers_fallback_is_taken() -> None:
    """The fork's class resolves, not the Transformers multi-modal fallback."""
    r = _resolve_once()
    module = r["resolved_module"]
    name = r["resolved_name"]
    assert module is not None, f"nothing resolved; create_engine_config: {r['raised']}"
    assert not module.startswith("transformers."), f"__module__ was {module!r}"
    assert name != "TransformersMultiModalMoEForCausalLM", f"__name__ was {name!r}"


def test_factory_class_satisfies_vllm_model_interface() -> None:
    """The factory class satisfies vLLM's model interface without a device."""
    cls = Glm5NextForConditionalGeneration

    assert is_vllm_model(cls) is True
    assert is_text_generation_model(cls) is True

    from vllm.utils.func_utils import supports_kw

    assert supports_kw(cls.__init__, "vllm_config") is True
    assert callable(getattr(cls, "embed_input_ids", None))
    assert callable(getattr(cls, "compute_logits", None))
