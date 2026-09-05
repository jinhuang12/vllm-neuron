# SPDX-License-Identifier: Apache-2.0
"""CONFIG-TIME architecture registration for ``Glm5Next`` (``inc-glm53f-074``).

Four items, one per declared case, **no** ``parametrize`` -- the declared
collected count is 4 and stays derivable before the run.

The subject is ``NeuronPlatform.pre_register_and_update``, which
``EngineArgs.create_engine_config`` calls as its first statement
(``vllm/engine/arg_utils.py`` L1794). Item 1 records the full D16.3
resolution observable set, (iii) and (iv) mandatory.

Coldness is bounded to item 1: ``VLLM_CACHE_ROOT`` is freshened once per
command invocation, so only the first ``create_engine_config()`` in the process
is cold. ``_resolve_once`` therefore builds the config **exactly once** and
every item reads that one resolution. Nothing under ``VLLM_CACHE_ROOT`` is
deleted, truncated or bypassed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from vllm import envs
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

# The four cache-path narrations (registry.py L857/L865/L921/L928). Recorded
# for attribution, never asserted and never counted.
CACHE_PATHS = ("not found", "is stale", "from cache", "miss. Loading model instead")

_RESOLUTION: dict | None = None


def _resolve_once() -> dict:
    """Build the engine config ONCE per process and record every observable."""
    global _RESOLUTION
    if _RESOLUTION is not None:
        return _RESOLUTION

    from vllm.engine.arg_utils import EngineArgs

    modelinfos = Path(envs.VLLM_CACHE_ROOT) / "modelinfos"
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Collect(level=logging.DEBUG)
    # vLLM pins ``propagate = False`` on its own root logger (logger.py L67),
    # so the handler attaches there and not to Python's root.
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.addHandler(handler)
    out: dict = {
        "cache_root": os.environ.get("VLLM_CACHE_ROOT"),
        "modelinfos": str(modelinfos),
        "modelinfos_before": modelinfos.exists(),
        "logger_level": logging.getLevelName(vllm_logger.getEffectiveLevel()),
        "raised": None,
        "config": None,
        "arch_in_process": None,
        "resolved_name": None,
        "resolved_module": None,
    }
    try:
        out["config"] = EngineArgs(model=FIXTURE).create_engine_config()
    except BaseException as exc:  # recorded, then adjudicated by item 1
        out["raised"] = f"{type(exc).__name__}: {exc}"
    finally:
        vllm_logger.removeHandler(handler)

    out["modelinfos_after"] = modelinfos.exists()
    out["records"] = records
    out["arch_log_lines"] = [r for r in records if "Resolved architecture:" in r]
    out["cache_path_lines"] = [
        r for r in records if any(p in r for p in CACHE_PATHS) and "model info" in r
    ]
    if out["config"] is not None:
        model_config = out["config"].model_config
        out["arch_in_process"] = model_config.architecture
        # (iii)/(iv) read two properties of ONE class object from ONE
        # resolve_model_cls call -- never off _model_info.architecture, which a
        # warm hit reconstructs from JSON.
        resolved = ModelRegistry.resolve_model_cls(
            model_config.architectures, model_config
        )[0]
        out["resolved_name"] = resolved.__name__
        out["resolved_module"] = resolved.__module__

    _RESOLUTION = out
    return out


def test_glm5next_resolves_before_validation() -> None:
    """Item 1 -- D16.3 (i)-(iv) as ONE conjunction, plus the coldness readings."""
    assert os.environ.get("VLLM_CACHE_ROOT"), (
        "VLLM_CACHE_ROOT must be set inline on the acceptance command so this "
        "invocation gets a fresh, empty root"
    )
    assert Path(FIXTURE).is_dir(), f"fixture unreachable from cwd: {FIXTURE}"

    r = _resolve_once()

    # Coldness, bounded to this item: three world-produced readings.
    assert r["modelinfos_before"] is False, f"root was not fresh: {r['modelinfos']}"
    assert r["modelinfos_after"] is True, f"no modelinfos written: {r['modelinfos']}"
    cold = [line for line in r["cache_path_lines"] if "miss. Loading model" in line]
    assert cold, f"no cold-miss narration; cache lines were {r['cache_path_lines']}"

    # (i) returns a VllmConfig and raises nothing.
    assert r["raised"] is None, f"create_engine_config raised: {r['raised']}"
    assert r["config"] is not None

    # (ii) one value, two readings: in-process attribute and the log record.
    assert r["arch_in_process"] == ARCH
    wanted = f"Resolved architecture: {ARCH}"
    assert wanted in r["arch_log_lines"], f"log records were {r['arch_log_lines']}"

    # (iii) MANDATORY -- the resolved class object's own __name__.
    assert r["resolved_name"] == ARCH

    # (iv) MANDATORY -- the verbatim dotted DEFINING module, not the package.
    assert r["resolved_module"] == DEFINING_MODULE


def test_registry_contains_glm5next_after_hook(monkeypatch) -> None:
    """Item 2 -- the hook registers the arch, 1 of 2 candidate keys, never 2."""
    ambient = os.environ.get("VLLM_NEURON_SYNTHETIC_MODEL")
    assert ambient != "1", (
        "the differential below is stated for VLLM_NEURON_SYNTHETIC_MODEL unset; "
        f"ambient value was {ambient!r}"
    )
    monkeypatch.delenv("VLLM_NEURON_SYNTHETIC_MODEL", raising=False)

    NeuronPlatform.pre_register_and_update()
    archs = set(ModelRegistry.get_supported_archs())

    candidates = (ARCH, "SyntheticNeuronModel")
    present = [name for name in candidates if name in archs]
    assert ARCH in archs
    assert "SyntheticNeuronModel" not in archs
    assert len(present) == 1, f"expected 1 of 2 candidate keys, got {present}"


def test_no_transformers_fallback_is_taken() -> None:
    """Item 3 -- branch 2 stays shut: the fork's class resolves, not HF's."""
    r = _resolve_once()
    module = r["resolved_module"]
    name = r["resolved_name"]
    assert module is not None, f"nothing resolved; create_engine_config: {r['raised']}"
    assert not module.startswith("transformers."), f"__module__ was {module!r}"
    assert name != "TransformersMultiModalMoEForCausalLM", f"__name__ was {name!r}"


def test_factory_class_satisfies_vllm_model_interface() -> None:
    """Item 4 -- device-free readings on the class object; no engine built."""
    cls = Glm5NextForConditionalGeneration

    assert is_vllm_model(cls) is True
    assert is_text_generation_model(cls) is True

    from vllm.utils.func_utils import supports_kw

    assert supports_kw(cls.__init__, "vllm_config") is True
    assert callable(getattr(cls, "embed_input_ids", None))
    assert callable(getattr(cls, "compute_logits", None))
