# SPDX-License-Identifier: Apache-2.0
"""Let recurrent-state layers take part in vLLM's KV-cache page-size unification.

``vllm.v1.core.kv_cache_utils.unify_kv_cache_spec_page_size`` makes every layer
of a hybrid model report one page size, because ``KVCacheManager`` allocates
blocks of a single size. For a layer whose page is below the maximum it either
keeps the physical page and multiplies the logical ``block_size`` (when the
maximum page is divisible by the layer's), or pads the physical page up to the
maximum. The padding branch is gated on ``isinstance(spec, AttentionSpec) and
spec.indexes_kv_by_block_stride``, because a padded page is read through a
strided view that not every backend handles. Anything else raises
``NotImplementedError``.

A linear-attention (KDA) layer holds no key/value history: it holds a
short-convolution state plus a recurrent state, which the runner reports as a
``MambaSpec``. ``MambaSpec`` derives from ``KVCacheSpec`` and not from
``AttentionSpec``, so the padding branch excludes it by type whatever its
geometry, and a recurrent-state page is not in general a divisor of the
attention page, so the re-blocking branch does not apply either. A hybrid
KDA + attention model therefore hits the raise at engine start, before a single
block is allocated.

``MambaSpec`` already carries the field the padding branch uses,
``page_size_padded``, and its ``page_size_bytes`` property returns that override
verbatim once set; upstream simply never offers the branch to a non-attention
spec. So this patch implements no new remedy and no grouping of its own: it
applies upstream's own ``replace(spec, page_size_padded=max_page)`` to the
recurrent-state layers the gate excludes, then hands the result back to the
upstream function, which decides everything else. The wrapper calls the original
first and returns its result untouched, so every input upstream can already
unify keeps its behaviour, and when there is nothing to pad upstream's
``NotImplementedError`` is re-raised as it stands.

The widening has to be live in the EngineCore subprocess, where
``EngineCore._initialize_kv_caches`` reaches the target through
``get_kv_cache_configs`` and ``get_kv_cache_groups``. That subprocess never
calls ``NeuronPlatform.check_and_update_config``, so a platform hook cannot
reach it, while importing ``vllm_neuron`` does, in every process including
spawn-mode children -- hence the explicit ``apply_kv_spec_patch()`` call from
``vllm_neuron/__init__.py``. Rebinding the module attribute suffices: the call
site is a module-global lookup inside the target module itself, and no other
module under ``vllm/`` holds a ``from ... import`` copy of the symbol.

Binding cannot always happen at import time; see :func:`_install_deferred`.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from dataclasses import replace

from vllm.logger import init_logger

logger = init_logger(__name__)

_TARGET_MODULE = "vllm.v1.core.kv_cache_utils"
_TARGET_ATTR = "unify_kv_cache_spec_page_size"
_PAD_FIELD = "page_size_padded"

# Idempotence guards: spawn re-imports and repeated plugin discovery both
# re-enter the caller, and wrapping a wrapper would stack layers. Two flags are
# needed because wiring and binding can happen at different moments -- when the
# target module is not importable yet, the wiring is done (`_applied`) while the
# rebinding still waits on the import system (`_bound`).
_applied = False
_bound = False
_original_unify = None

# Records whether the rebinding took the deferred route. Diagnostic only: the
# eager route works again as soon as upstream's circular import goes away, so
# nothing branches on this.
_deferred = False


class KvSpecPatchTargetError(RuntimeError):
    """The KV-spec unification callable is not where this patch expects it.

    Raised at apply time rather than skipped: a silently absent widening
    surfaces much later as upstream's own ``NotImplementedError`` inside the
    EngineCore subprocess, with nothing pointing back at this patch.
    """


def _padding_candidates(kv_cache_spec):
    """Find the recurrent-state layers upstream's padding branch excludes.

    A layer qualifies only when its page is below the maximum (a layer already
    at the maximum needs nothing), the maximum is not divisible by its page
    (upstream's re-blocking branch is preferred wherever it applies), it is a
    ``MambaSpec`` and not an ``AttentionSpec`` (upstream's gate owns every
    attention spec, including the opt-in flag it requires), and it carries a
    ``page_size_padded`` field that is still unset, so only a field upstream
    defined is ever filled and no value another caller chose is overwritten.

    Returns:
        ``(widened_spec_dict, padded_layer_names, max_page_size)``, or ``None``
        when no layer qualifies, in which case the caller re-raises upstream's
        error untouched.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # Upstream returns early on a uniform set and cannot have raised, so
        # reaching here at all means the error came from somewhere else.
        return None
    max_page_size = max(page_sizes)

    widened = {}
    padded = []
    for layer_name, spec in kv_cache_spec.items():
        page_size = spec.page_size_bytes
        if (
            page_size != max_page_size
            and max_page_size % page_size != 0
            and isinstance(spec, MambaSpec)
            and not isinstance(spec, AttentionSpec)
            and any(f.name == _PAD_FIELD for f in dataclass_fields(spec))
            and getattr(spec, _PAD_FIELD) is None
        ):
            widened[layer_name] = replace(spec, **{_PAD_FIELD: max_page_size})
            padded.append(layer_name)
        else:
            widened[layer_name] = spec

    if not padded:
        return None
    return widened, padded, max_page_size


def _unify_kv_cache_spec_page_size_widened(kv_cache_spec):
    """Upstream's unification, with its own padding branch offered one class wider.

    The order is load-bearing: the original runs first and its result is
    returned untouched, so any spec set upstream can already unify behaves
    exactly as it did before.
    """
    try:
        return _original_unify(kv_cache_spec)
    except NotImplementedError:
        candidates = _padding_candidates(kv_cache_spec)
        if candidates is None:
            # Nothing here this patch can pad, so upstream's error is the right
            # answer and is re-raised rather than reworded.
            raise

    # Outside the handler, so a fault here is not chained onto upstream's error.
    widened, padded, max_page_size = candidates
    logger.info(
        "Neuron: padded the physical KV page of %d recurrent-state layer(s) to "
        "%d bytes so a hybrid spec set can share one page size (%s). Upstream's "
        "own page_size_padded remedy, applied to MambaSpec; logical block sizes "
        "are untouched.",
        len(padded),
        max_page_size,
        ", ".join(padded[:4]) + (", ..." if len(padded) > 4 else ""),
    )
    return _original_unify(widened)


def _install(kv_cache_utils) -> None:
    """Rebind the module attribute. Called eagerly, or later from the loader hook."""
    global _original_unify, _bound

    if _bound:
        return

    original = getattr(kv_cache_utils, _TARGET_ATTR, None)
    if not callable(original):
        raise KvSpecPatchTargetError(
            f"{_TARGET_MODULE}.{_TARGET_ATTR} is missing or not callable "
            f"(got {original!r}). The KV-spec page-size widening cannot be "
            "applied, so a hybrid recurrent-state model would fail later inside "
            "the EngineCore subprocess with upstream's own NotImplementedError. "
            "Check whether the symbol moved or was renamed at this vLLM pin."
        )

    _original_unify = original
    # Exposes the wrapped callable, so the chain can be inspected without
    # reaching for this module's private names.
    _unify_kv_cache_spec_page_size_widened.__wrapped__ = original
    setattr(kv_cache_utils, _TARGET_ATTR, _unify_kv_cache_spec_page_size_widened)

    _bound = True
    logger.debug(
        "Neuron: KV-spec page-size widening applied to %s.%s (wrap, not replace).",
        _TARGET_MODULE,
        _TARGET_ATTR,
    )


def _install_deferred() -> None:
    """Patch as soon as the target module finishes loading.

    Importing ``vllm.v1.core.kv_cache_utils`` eagerly from here works when
    ``vllm_neuron`` is imported first, but breaks plugin loading outright when
    ``vllm`` is imported first: ``load_general_plugins`` runs while
    ``vllm.utils.torch_utils`` is still initializing, and the eager import walks
    ``kv_cache_utils -> vllm.config -> vllm.config.cache -> vllm.utils.torch_utils``
    back into that partially initialized module, which fails with ``cannot
    import name 'is_quantized_kv_cache' from partially initialized module`` and
    leaves the Neuron platform unregistered.

    An ``import`` audit hook cannot cover that case either: CPython raises the
    audit event before the module body runs, so ``sys.modules`` does not hold
    the target yet and the hook has nothing to rebind; it would only bind on
    some later import that happens to follow.

    So the deferral hangs off the import system itself: a meta-path finder for
    exactly one module name, which delegates to the real finder and wraps only
    its loader, so :func:`_install` runs the moment the module body has finished
    executing. Deterministic, and it still imports nothing eagerly.
    """
    import importlib.abc
    import sys

    global _deferred
    _deferred = True

    # The module may already be loaded -- partially, which is how we got here.
    # If the symbol is present, bind now: the finder never fires for a module
    # already in sys.modules.
    already = sys.modules.get(_TARGET_MODULE)
    if already is not None and hasattr(already, _TARGET_ATTR):
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
            # Upstream runs first and unchanged; if it raises, nothing is
            # installed and the import fails exactly as it would have.
            self._inner.exec_module(module)
            _install(module)
            # One-shot: leave no finder behind on the import path.
            try:
                sys.meta_path.remove(self._finder)
            except ValueError:
                pass

        def __getattr__(self, name):
            # get_code, is_package, get_source, ... all stay upstream's.
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
            # Nobody can load it; say so and let the real machinery raise.
            return None

    sys.meta_path.insert(0, _TargetFinder())
    logger.debug(
        "Neuron: KV-spec widening deferred to a meta-path loader hook; %s was "
        "not importable at plugin-registration time (circular import).",
        _TARGET_MODULE,
    )


def apply_kv_spec_patch() -> None:
    """Wire the KV-spec widening, eagerly if possible and deferred if not.

    Idempotent: repeated calls leave exactly one wrapper layer installed.
    """
    global _applied

    if _applied:
        return
    # Set before importing anything: the import below can re-enter plugin
    # discovery, and a second entry must not install a second hook.
    _applied = True

    try:
        from vllm.v1.core import kv_cache_utils
    except ImportError:
        # Circular import during plugin registration. Defer rather than let the
        # whole plugin fail to load.
        _install_deferred()
        return

    _install(kv_cache_utils)
