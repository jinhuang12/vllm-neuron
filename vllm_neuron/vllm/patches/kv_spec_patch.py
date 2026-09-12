# SPDX-License-Identifier: Apache-2.0
"""Admit recurrent-state layers into vLLM's KV-cache page-size unification.

WHAT UPSTREAM DOES. ``vllm.v1.core.kv_cache_utils.unify_kv_cache_spec_page_size``
makes every layer of a hybrid model report one page size, because
``KVCacheManager`` can only allocate blocks of a single size. For a layer whose
page is below the maximum it has two remedies and one refusal:

1. if the maximum page is divisible by the layer's page, it keeps the layer's
   physical page and multiplies its logical ``block_size``;
2. otherwise it pads the layer's *physical* page to the maximum -- but only for
   ``isinstance(layer_spec, AttentionSpec) and layer_spec.indexes_kv_by_block_stride``,
   because a padded page is read through a strided view that not every backend
   handles;
3. anything else raises ``NotImplementedError``.

WHY A NEURON HYBRID MODEL LANDS ON THE REFUSAL. A linear-attention (KDA) layer
holds no key/value history; it holds a short-convolution state plus a recurrent
state, which the runner reports as a ``MambaSpec``. ``MambaSpec`` derives from
``KVCacheSpec``, **not** from ``AttentionSpec``, so remedy 2 excludes it by type
no matter what its geometry is -- and a recurrent-state page is not in general a
divisor of the attention page, so remedy 1 does not apply either. A hybrid
KDA + attention model therefore reaches the raise, at engine start, before a
single block is allocated.

WHAT THIS PATCH CHANGES, AND HOW LITTLE. ``MambaSpec`` **already carries the
field remedy 2 uses** -- ``page_size_padded`` -- and its ``page_size_bytes``
property already returns that override verbatim once set. Upstream simply never
offers the remedy to a non-attention spec. So this patch does not implement a
new remedy, does not re-implement upstream's arithmetic and does not decide any
grouping: it applies upstream's OWN ``replace(spec, page_size_padded=max_page)``
to the recurrent-state layers upstream's gate excludes, and hands the result
back to the upstream function, which decides everything.

WRAP, NOT REPLACE. The wrapper calls the original FIRST and returns its result
untouched whenever the original succeeds, so every input upstream can already
unify keeps pin behaviour exactly. Only after the original has raised
``NotImplementedError`` does the wrapper look for recurrent-state layers to pad,
and if it finds none it re-raises upstream's own error rather than inventing
one. Upstream's refusal is narrowed, never removed.

WHY IMPORT TIME. The widening must be live in the **EngineCore subprocess**:
``EngineCore._initialize_kv_caches`` (``vllm/v1/engine/core.py:294``) calls
``get_kv_cache_configs``, which calls ``get_kv_cache_groups``, whose sole call of
the target sits at ``kv_cache_utils.py:1751``. That subprocess never calls
``NeuronPlatform.check_and_update_config``, so a platform hook cannot reach it;
importing ``vllm_neuron`` does, in every process including spawn-mode children.
Hence an explicit ``apply_kv_spec_patch()`` call from ``vllm_neuron/__init__.py``
(never the dead ``patches/__init__.py::apply_patches()`` stub).

The rebinding is on the module attribute, and the call at ``:1751`` is a
module-global lookup inside that same module, so it resolves to the wrapper. No
other module under ``vllm/`` holds a ``from ... import`` copy of the symbol,
which is what makes a single module-attribute rebinding sufficient.

The *binding moment* is not always import time, and that is deliberate: when
``vllm`` is imported before ``vllm_neuron`` the target module is mid-initialisation
and importing it here breaks plugin loading, so the rebinding defers until the
import system finishes loading the target. :func:`_install_deferred` records the
two measurements that forced that shape.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from dataclasses import replace

from vllm.logger import init_logger

logger = init_logger(__name__)

_TARGET_MODULE = "vllm.v1.core.kv_cache_utils"
_TARGET_ATTR = "unify_kv_cache_spec_page_size"
_PAD_FIELD = "page_size_padded"

# Idempotence guards. Spawn re-imports and repeated plugin discovery both
# re-enter this module's caller, and wrapping a wrapper would stack layers.
#
# TWO flags, because wiring and binding can happen at different moments: when the
# target module is not importable at plugin-registration time the wiring is done
# (`_applied`) while the rebinding still waits on the import system (`_bound`).
# One flag would either install a second hook or let a second wrapper stack.
_applied = False
_bound = False
_original_unify = None

# Diagnostic only, never a branch condition: True once the deferred route was
# taken, so a test can RECORD which of the two routes production actually used
# instead of inferring it. Asserting on it would false-fire the day upstream's
# circular import goes away and the eager route starts working.
_deferred = False


class KvSpecPatchTargetError(RuntimeError):
    """The KV-spec unification callable is not where this patch expects it.

    Raised loudly at apply time rather than skipped: if the widening is silently
    absent, the failure surfaces much later as upstream's own
    ``NotImplementedError`` inside the EngineCore subprocess, with nothing
    pointing back at this patch.
    """


def _padding_candidates(kv_cache_spec):
    """Recurrent-state layers upstream refuses but its own remedy would fit.

    Returns ``(widened_spec_dict, padded_layer_names, max_page_size)``, or
    ``None`` when this patch has nothing to offer -- in which case the caller
    re-raises upstream's refusal untouched.

    A layer is a candidate only when ALL of the following hold, which is
    deliberately narrower than "not an AttentionSpec":

    * its page is below the maximum page (a layer already at the maximum is
      returned verbatim by upstream and needs nothing);
    * the maximum page is NOT divisible by its page -- upstream's re-blocking
      remedy is strictly preferred where it applies, and this patch never
      displaces it;
    * it is a ``MambaSpec``, the recurrent-state class the runner reports for a
      linear-attention layer, and NOT an ``AttentionSpec`` -- upstream's own
      gate owns every attention spec, including the opt-in it requires, and this
      patch does not widen that gate;
    * it actually carries the ``page_size_padded`` field and has not already had
      it set, so the patch only ever fills a field upstream defined and never
      overwrites a value someone else chose.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # Upstream returns early on a uniform set and cannot have raised, so
        # reaching here at all would mean the refusal came from somewhere else.
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
    """Upstream's unification, with its own padding remedy offered one class wider.

    Order is load-bearing: the original runs first and its result is returned
    untouched, so no input upstream can already unify changes behaviour here.
    """
    try:
        return _original_unify(kv_cache_spec)
    except NotImplementedError:
        candidates = _padding_candidates(kv_cache_spec)
        if candidates is None:
            # Nothing this patch admits. Upstream's refusal is the right answer
            # and is re-raised verbatim rather than reworded.
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
    # Makes the wrapper chain machine-readable: exactly one layer over the
    # original, checkable without importing this module's private names.
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

    WHY THIS EXISTS -- a MEASURED failure, not a precaution. Importing
    ``vllm.v1.core.kv_cache_utils`` eagerly from here works when
    ``vllm_neuron`` is imported first, and **breaks plugin loading outright**
    when ``vllm`` is imported first, which is the production path: vLLM's
    ``load_general_plugins`` runs while ``vllm.utils.torch_utils`` is still
    initialising, and the eager import walks
    ``kv_cache_utils -> vllm.config -> vllm.config.cache -> vllm.utils.torch_utils``
    back into that partially initialised module. The observed result was
    ``ImportError: cannot import name 'is_quantized_kv_cache' from partially
    initialized module`` and ``Failed to load plugin neuron`` -- the Neuron
    platform not registering at all.

    WHY NOT THE AUDIT-HOOK PATTERN THE INVENTORY PRESCRIBES. Porter rule 7 says
    "circular-import timing at registration -- only the DCP patch has an
    audit-hook fallback -- copy that pattern if your patch must run at
    registration", and that was tried first, copied from
    ``vllm_neuron/vllm/platform.py:67-90``. It was MEASURED not to bind: CPython
    raises the ``import`` audit event BEFORE the module body runs, so at hook
    time ``sys.modules`` does not yet hold the target and the hook does nothing;
    it can only bind on some LATER import that happens to follow. The reading
    was ``APPLIED=True BOUND=False`` with the attribute still upstream's.

    So the deferral hangs off the import system itself rather than off an audit
    event: a meta-path finder for exactly one module name, which delegates to
    the real finder and wraps only its loader, so :func:`_install` runs the
    moment the module body has finished executing. Deterministic, and it still
    imports nothing eagerly. (This is a note for the lead about the inventory's
    rule 7 and about the landed DCP patch, which appears to carry the same
    latent hole; no landed code is changed here.)
    """
    import importlib.abc
    import sys

    global _deferred
    _deferred = True

    # It may already be loaded -- e.g. partially, which is how we got here. If
    # the symbol is present, bind now; the finder would never fire for a module
    # that is already in sys.modules.
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
            # Upstream runs first and unchanged; if it raises, we install
            # nothing and the import fails exactly as it would have.
            self._inner.exec_module(module)
            _install(module)
            # One-shot: leave no finder behind on the hot import path.
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
    # Set before doing any importing, exactly as the DCP precedent does: the
    # import below can re-enter plugin discovery, and a second entry must not
    # install a second hook.
    _applied = True

    try:
        from vllm.v1.core import kv_cache_utils
    except ImportError:
        # Circular import during plugin registration. Defer rather than let the
        # whole plugin fail to load.
        _install_deferred()
        return

    _install(kv_cache_utils)
