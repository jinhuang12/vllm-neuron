# SPDX-License-Identifier: Apache-2.0
"""Let vLLM accept GLM-5.3-Flash as its own multi-token-prediction (MTP) draft.

``--speculative-config '{"method": "mtp", "num_speculative_tokens": k}'`` names
the draft the checkpoint carries itself: the layer past the stack that
``text_config.num_nextn_predict_layers`` declares, which this tree's
``Glm5NextMultiTokenPredictor`` holds. ``vllm.config.speculative.SpeculativeConfig``
resolves that method in ``__post_init__``: the draft model is the target
checkpoint, loaded a second time as a ``ModelConfig`` with
``hf_overrides=SpeculativeConfig.hf_config_override``, and the method is then
confirmed only if the draft's ``hf_config.model_type`` is in the module-global
``MTPModelTypes`` ``Literal``; anything else raises ``NotImplementedError("Unsupported
speculative method: 'mtp'")``. Upstream's override has a branch per known MTP
family (``Glm4MoeForCausalLM`` -> ``glm4_moe_mtp`` and so on) and none for
``Glm5NextForConditionalGeneration``, so the GLM-5.3-Flash draft keeps
``model_type = "glm5_next"``, which is not in the list, and the method is refused.

This patch adds that one family the way upstream adds its own:

* ``MTPModelTypes`` gains :data:`GLM5_NEXT_MTP_MODEL_TYPE`. Rebinding the module
  attribute suffices: both readers, ``get_args(MTPModelTypes)`` in ``__post_init__``
  and the one in the method check, are inside the target module and look the
  global up at call time. The derived
  ``Literal`` aliases upstream builds at import (``EagleModelTypes``,
  ``SpeculativeMethod``) are typing annotations only and are left as they are.
* ``SpeculativeConfig.hf_config_override`` is wrapped: upstream's override runs
  first and its result is returned untouched for every architecture it knows;
  for the GLM-5.3-Flash architecture the draft's copy of the config is retyped to
  :data:`GLM5_NEXT_MTP_MODEL_TYPE` and given ``n_predict``, the draft layer count
  the checkpoint's text config declares. A checkpoint that declares no draft layer
  is refused by name. The ``architectures`` entry is deliberately kept: no draft
  model class exists or is loaded (the runner builds its proposer from the root
  model's own head, ``vllm_neuron/vllm/spec_decode/mtp.py``), and ``ModelConfig``
  validates the draft's architecture against the model registry, where the plugin
  registers exactly that one class.

The retyping has to be live in the process that builds the engine config, before
``SpeculativeConfig.__post_init__`` runs. ``EngineArgs.create_engine_config``
builds the speculative config before the ``VllmConfig`` whose ``__post_init__``
calls ``NeuronPlatform.check_and_update_config``, so a platform hook is too late,
while importing ``vllm_neuron`` is early enough in every process -- hence the
explicit ``apply_spec_config_patch()`` call from ``vllm_neuron/__init__.py``.
In the production import order (``vllm`` first) the plugin is loaded from inside
``vllm.utils.torch_utils``'s own initialisation, before ``vllm.config`` exists,
so the target module cannot be imported eagerly there; binding is then deferred
to the moment the import system finishes loading it, the same way
``kv_spec_patch`` defers (see :func:`_install_deferred`).
"""

from __future__ import annotations

import typing

from vllm.logger import init_logger

logger = init_logger(__name__)

_TARGET_MODULE = "vllm.config.speculative"
_TYPES_ATTR = "MTPModelTypes"
_CONFIG_ATTR = "SpeculativeConfig"
_OVERRIDE_ATTR = "hf_config_override"

#: The architecture the plugin registers for GLM-5.3-Flash
#: (``NeuronPlatform.pre_register_and_update``); the draft config keeps it.
GLM5_NEXT_ARCHITECTURE = "Glm5NextForConditionalGeneration"
#: The draft's ``model_type`` once retyped, spelled like upstream's own
#: ``<family>_mtp`` entries.
GLM5_NEXT_MTP_MODEL_TYPE = "glm5_next_mtp"
#: The text-config field that declares the checkpoint's draft layer count; the
#: same field the head's weight map reads (``weight_loaders_fp8.mtp_layer_indices_for``).
DRAFT_LAYERS_FIELD = "num_nextn_predict_layers"

# Idempotence guards, as in kv_spec_patch: wiring (`_applied`) and binding
# (`_bound`) can happen at different moments when the target module is not
# importable yet.
_applied = False
_bound = False
_original_override = None
_deferred = False


class SpecConfigPatchTargetError(RuntimeError):
    """The speculative-config symbols are not where this patch expects them.

    Raised at apply time rather than skipped: an absent retyping surfaces later as
    upstream's own ``Unsupported speculative method: 'mtp'`` with nothing pointing
    back at this patch.
    """


def _draft_layer_count(hf_config) -> int:
    """The checkpoint's declared draft layer count, read off its text config."""
    text_config = hf_config.get_text_config() if hasattr(hf_config, "get_text_config") else hf_config
    declared = getattr(text_config, DRAFT_LAYERS_FIELD, None)
    if declared is None or int(declared) <= 0:
        raise ValueError(
            f"{GLM5_NEXT_ARCHITECTURE} checkpoint declares "
            f"{DRAFT_LAYERS_FIELD}={declared!r}, so it carries no multi-token-prediction "
            f"draft layer and cannot serve speculative method 'mtp'; GLM-5.3-Flash "
            f"declares {DRAFT_LAYERS_FIELD}=1 in its text_config."
        )
    return int(declared)


def _hf_config_override_with_glm5_next(hf_config):
    """Upstream's draft-config override, plus the GLM-5.3-Flash family.

    The order is load-bearing: upstream runs first and its result is returned
    untouched for every architecture it already handles.
    """
    hf_config = _original_override(hf_config)
    if hf_config.architectures[0] != GLM5_NEXT_ARCHITECTURE:
        return hf_config
    n_predict = _draft_layer_count(hf_config)
    hf_config.model_type = GLM5_NEXT_MTP_MODEL_TYPE
    hf_config.update({"n_predict": n_predict})
    return hf_config


def _install(speculative) -> None:
    """Rebind the two module symbols. Called eagerly, or later from the loader hook."""
    global _original_override, _bound

    if _bound:
        return

    config_cls = getattr(speculative, _CONFIG_ATTR, None)
    original = getattr(config_cls, _OVERRIDE_ATTR, None)
    types = getattr(speculative, _TYPES_ATTR, None)
    if not callable(original) or types is None:
        raise SpecConfigPatchTargetError(
            f"{_TARGET_MODULE}.{_CONFIG_ATTR}.{_OVERRIDE_ATTR} ({original!r}) or "
            f"{_TARGET_MODULE}.{_TYPES_ATTR} ({types!r}) is missing, so GLM-5.3-Flash "
            "cannot be registered as an MTP target and --speculative-config "
            "method 'mtp' would be refused by upstream. Check whether the symbols "
            "moved or were renamed at this vLLM pin."
        )

    _original_override = original
    # Exposes the wrapped callable, so the chain can be inspected without
    # reaching for this module's private names.
    _hf_config_override_with_glm5_next.__wrapped__ = original
    setattr(config_cls, _OVERRIDE_ATTR, staticmethod(_hf_config_override_with_glm5_next))

    entries = typing.get_args(types)
    if GLM5_NEXT_MTP_MODEL_TYPE not in entries:
        setattr(speculative, _TYPES_ATTR, typing.Literal[(*entries, GLM5_NEXT_MTP_MODEL_TYPE)])

    _bound = True
    logger.debug(
        "Neuron: %s registered as an MTP model type and %s.%s wrapped for %s.",
        GLM5_NEXT_MTP_MODEL_TYPE,
        _CONFIG_ATTR,
        _OVERRIDE_ATTR,
        GLM5_NEXT_ARCHITECTURE,
    )


def _install_deferred() -> None:
    """Patch as soon as the target module finishes loading.

    A meta-path finder for exactly one module name delegates to the real finder
    and wraps only its loader, so :func:`_install` runs the moment the module body
    has executed. Deterministic, and it imports nothing eagerly. The mechanism is
    the one ``kv_spec_patch._install_deferred`` documents; it is repeated here
    rather than shared because each patch binds to a different module and the
    finder is one-shot.
    """
    import importlib.abc
    import sys

    global _deferred
    _deferred = True

    already = sys.modules.get(_TARGET_MODULE)
    if already is not None and hasattr(already, _TYPES_ATTR):
        _install(already)
        return

    class _LoaderProxy(importlib.abc.Loader):
        """Upstream's own loader, plus one call after the module body runs."""

        def __init__(self, inner, finder):
            self._inner = inner
            self._finder = finder

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            _install(module)
            try:
                sys.meta_path.remove(self._finder)
            except ValueError:
                pass

        def __getattr__(self, name):
            return getattr(self._inner, name)

    class _TargetFinder(importlib.abc.MetaPathFinder):
        """Claims exactly one module name, then hands the work back upstream."""

        def find_spec(self, fullname, path=None, target=None):
            if _bound or fullname != _TARGET_MODULE:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _LoaderProxy(spec.loader, self)
                    return spec
            return None

    sys.meta_path.insert(0, _TargetFinder())
    logger.debug(
        "Neuron: MTP registration of %s deferred to a meta-path loader hook; %s was "
        "not importable at plugin-registration time.",
        GLM5_NEXT_ARCHITECTURE,
        _TARGET_MODULE,
    )


def apply_spec_config_patch() -> None:
    """Register GLM-5.3-Flash as an MTP target, eagerly if possible and deferred if not.

    Idempotent: repeated calls leave exactly one wrapper layer and one added type.
    """
    global _applied

    if _applied:
        return
    _applied = True

    try:
        import vllm.config.speculative as speculative
    except ImportError:
        _install_deferred()
        return

    _install(speculative)
