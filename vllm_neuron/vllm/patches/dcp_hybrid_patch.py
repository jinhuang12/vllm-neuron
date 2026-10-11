# SPDX-License-Identifier: Apache-2.0
"""Let vLLM's engine start a hybrid KV cache under decode context parallelism.

``EngineCore.__init__`` asks ``vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes``
for the scheduler's token granularity and the block-hash granularity before it builds
the scheduler. For a KV cache config with more than one group it refuses any context
parallelism outright::

    if dcp != 1 or pcp != 1:
        raise ValueError(
            "Hybrid KV cache groups with multiple block sizes do not "
            "support context parallelism (dcp_world_size/pcp_world_size > 1)."
        )

A model with linear-attention (KDA) layers next to its latent-attention layers always
has more than one group -- its recurrent-state layers form their own groups -- so it
cannot start with ``decode_context_parallel_size > 1`` at all, whatever its attention
supports.

Everything downstream of that function already sizes every group per rank. Each
single-type manager multiplies its block size by the context-parallel world size
(``SingleTypeKVCacheManager.__init__``: ``self.block_size *= dcp_world_size *
pcp_world_size``), and the runner's block table divides every group's width by it
(``MultiGroupBlockTable``: ``cdiv(max_model_len, block_size * total_cp_world_size)``).
What the refusal withholds is only the pair of granularities. For one group upstream
answers ``cache_config.block_size * dcp * pcp`` for both; for several groups at DCP=1
it answers the LCM of the group block sizes. This patch answers, for several groups at
DCP > 1, the LCM of the block sizes the managers really allocate, ``block_size * dcp``
per group: that is ``dcp`` times upstream's own DCP=1 answer, and it reduces to
upstream's single-group rule when there is one group. The hash granularity is set
equal to it, which is what upstream returns whenever block hashing is inactive.

Why that is safe for a recurrent + attention cache:

* An attention group is upstream's own per-rank rule: ``ceil(tokens / (block_size *
  dcp))`` blocks per request, the count the runner's block table is sized for.
* A recurrent group's page size is untouched, and its blocks are pool accounting
  only: the runner keeps each recurrent layer's state in a bank of one slot per
  request on every rank, addressed by request slot and never by block id
  (``kv_cache_allocations``). With ``--mamba-block-size <max_model_len>``, the served
  line, a recurrent group takes exactly one block per request at every DCP, as at
  DCP=1. With a shorter recurrent block its count shrinks by ``dcp`` exactly as the
  runner's block-table width for that group does, so the two stay consistent.
* The scheduler granularity is consumed, with prefix caching off and no KV connector,
  only as the invariant ``KVCacheCoordinator.__init__`` asserts (every group's block
  size divides it) and the retention interval the coordinator validates against it.

What is lifted is deliberately narrow; everything else keeps a refusal:

* PCP > 1, a single group, or any group that is not a full-attention spec or a
  ``"none"``-mode recurrent spec: upstream's own error is re-raised unchanged.
* Prefix caching on: refused by name. ``HybridKVCacheCoordinator.__init__`` asserts
  ``dcp_world_size == 1`` ("DCP not support hybrid attn now.") and
  ``MambaManager.find_longest_cache_hit`` asserts it too ("DCP not support mamba
  now."), so the engine would otherwise die on a bare assertion one step later.
* A KV connector: refused by name. Connectors consume ``Request.block_hashes`` at the
  hash granularity, and no granularity has been shown to line up with every group's
  per-rank blocks on a hybrid cache.

The wrapper calls the original first and returns its result untouched, so DCP=1 and
every input upstream already resolves keep their behaviour; the patch acts only on the
refusal above. Upstream raises at DCP > 1 with several groups before it computes
anything, so that refusal is recognised by the conditions it is raised under rather
than by its text.

Drop this patch when ``resolve_kv_cache_block_sizes`` stops refusing context
parallelism for several groups: the original then returns for that input and the
wrapper is inert.

The engine imports the target by name (``from vllm.v1.core.kv_cache_utils import
resolve_kv_cache_block_sizes`` in ``vllm/v1/engine/core.py``), so its own binding is
rebound too when that module is already loaded; one loaded later picks up the rebound
attribute. The connector modules that also import it by name are refused here anyway
and keep upstream's refusal. Like ``kv_spec_patch``, this is applied at import time
from ``vllm_neuron/__init__.py``, because the EngineCore subprocess never calls
``NeuronPlatform.check_and_update_config``.
"""

from __future__ import annotations

import math
import sys

from vllm.logger import init_logger

logger = init_logger(__name__)

_TARGET_MODULE = "vllm.v1.core.kv_cache_utils"
_TARGET_ATTR = "resolve_kv_cache_block_sizes"
#: The engine's call site holds its own ``from ... import`` binding of the target.
_ENGINE_MODULE = "vllm.v1.engine.core"
#: The recurrent cache mode that keeps no prefix-cache state snapshots; the
#: prefix-caching modes ("align", "all") are not lifted.
_RECURRENT_CACHE_MODE = "none"

# Idempotence guards, as in kv_spec_patch: wiring (`_applied`) and binding
# (`_bound`) can happen at different moments when the target module is not
# importable yet.
_applied = False
_bound = False
_original_resolve = None


class DcpHybridPatchTargetError(RuntimeError):
    """The block-size resolver is not where this patch expects it.

    Raised at apply time rather than skipped: a silently absent patch surfaces much
    later as upstream's refusal inside the EngineCore subprocess, with nothing
    pointing back at this module.
    """


def _lifted_dcp(kv_cache_config, vllm_config) -> int | None:
    """Return the DCP size when ``vllm_config`` is the case upstream refuses, else None.

    That is DCP > 1, PCP == 1 and more than one KV cache group: at that input
    upstream raises before it reads anything else.
    """
    parallel_config = vllm_config.parallel_config
    dcp = parallel_config.decode_context_parallel_size
    if (
        dcp > 1
        and parallel_config.prefill_context_parallel_size == 1
        and len(kv_cache_config.kv_cache_groups) > 1
    ):
        return dcp
    return None


def _check_lifted(kv_cache_config, vllm_config, upstream_error: ValueError) -> None:
    """Raise unless every group and feature of this config is one the patch lifts."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    specs = [group.kv_cache_spec for group in kv_cache_config.kv_cache_groups]
    if not all(
        isinstance(spec, FullAttentionSpec)
        or (
            isinstance(spec, MambaSpec)
            and spec.mamba_cache_mode == _RECURRENT_CACHE_MODE
        )
        for spec in specs
    ):
        raise upstream_error

    dcp = vllm_config.parallel_config.decode_context_parallel_size
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError(
            f"Prefix caching is not supported with decode_context_parallel_size="
            f"{dcp} on a hybrid KV cache: vLLM's HybridKVCacheCoordinator asserts "
            "'DCP not support hybrid attn now.' and MambaManager asserts 'DCP not "
            "support mamba now.', so this needs DCP support in vLLM's hybrid "
            "prefix-cache coordinator first. Serve with --no-enable-prefix-caching."
        ) from upstream_error
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is not None:
        raise ValueError(
            f"A KV connector ({kv_transfer_config.kv_connector}) is not supported "
            f"with decode_context_parallel_size={dcp} on a hybrid KV cache: "
            "connectors read Request.block_hashes at the hash block size, and no "
            "hash block size has been shown to match every group's per-rank blocks, "
            "so disaggregated prefill needs its own port onto a hybrid DCP cache. "
            "Serve DCP without --kv-transfer-config."
        ) from upstream_error


def _resolve_kv_cache_block_sizes_dcp(kv_cache_config, vllm_config):
    """Upstream's resolver, answering the hybrid DCP case it refuses.

    The order is load-bearing: the original runs first and its result is returned
    untouched, so every input upstream already resolves behaves exactly as before.
    """
    try:
        return _original_resolve(kv_cache_config, vllm_config)
    except ValueError as exc:
        dcp = _lifted_dcp(kv_cache_config, vllm_config)
        if dcp is None:
            raise
        upstream_error = exc

    # Outside the handler, so a refusal here names upstream's error as its cause
    # instead of being reported as a fault while handling it.
    _check_lifted(kv_cache_config, vllm_config, upstream_error)
    # Each manager allocates blocks of ``spec.block_size * dcp`` tokens
    # (SingleTypeKVCacheManager.__init__), so this is their LCM.
    block_size = math.lcm(
        *(
            group.kv_cache_spec.block_size * dcp
            for group in kv_cache_config.kv_cache_groups
        )
    )
    logger.info(
        "Neuron: resolved the scheduler and hash block size of a %d-group hybrid KV "
        "cache at decode_context_parallel_size=%d to %d tokens (the LCM of every "
        "group's per-rank block, block_size x dcp); upstream refuses context "
        "parallelism for more than one group.",
        len(kv_cache_config.kv_cache_groups),
        dcp,
        block_size,
    )
    return block_size, block_size


def _install(kv_cache_utils) -> None:
    """Rebind the target and the engine's copy. Called eagerly, or from the loader hook."""
    global _original_resolve, _bound

    if _bound:
        return

    original = getattr(kv_cache_utils, _TARGET_ATTR, None)
    if not callable(original):
        raise DcpHybridPatchTargetError(
            f"{_TARGET_MODULE}.{_TARGET_ATTR} is missing or not callable (got "
            f"{original!r}). The hybrid DCP block-size patch cannot be applied, so "
            "decode_context_parallel_size > 1 would fail at engine start with "
            "upstream's refusal. Check whether the symbol moved or was renamed at "
            "this vLLM pin."
        )
    engine = sys.modules.get(_ENGINE_MODULE)
    engine_binding = getattr(engine, _TARGET_ATTR, None)
    if engine_binding is not None and engine_binding is not original:
        raise DcpHybridPatchTargetError(
            f"{_ENGINE_MODULE}.{_TARGET_ATTR} is {engine_binding!r}, not upstream's "
            f"{_TARGET_MODULE}.{_TARGET_ATTR}; something else rebound the engine's "
            "call site, so this patch would not reach it."
        )

    _original_resolve = original
    _resolve_kv_cache_block_sizes_dcp.__wrapped__ = original
    setattr(kv_cache_utils, _TARGET_ATTR, _resolve_kv_cache_block_sizes_dcp)
    if engine_binding is not None:
        setattr(engine, _TARGET_ATTR, _resolve_kv_cache_block_sizes_dcp)

    _bound = True
    logger.debug(
        "Neuron: hybrid DCP block-size resolution applied to %s.%s%s (wrap, not "
        "replace).",
        _TARGET_MODULE,
        _TARGET_ATTR,
        f" and {_ENGINE_MODULE}" if engine_binding is not None else "",
    )


def _install_deferred() -> None:
    """Patch as soon as the target module finishes loading.

    Same situation and remedy as ``kv_spec_patch._install_deferred``, whose docstring
    gives the reasoning: when ``vllm`` is imported first, this plugin loads while
    ``vllm.utils.torch_utils`` is still initialising and the target module cannot be
    imported yet, so a one-shot meta-path finder for exactly that module wraps its
    loader and runs :func:`_install` once the module body has executed.

    The finder takes itself off ``sys.meta_path`` before it asks the remaining
    finders, so a finder that walks ``sys.meta_path`` in turn -- kv_spec_patch's,
    for the same module -- cannot re-enter it.
    """
    import importlib.abc

    already = sys.modules.get(_TARGET_MODULE)
    if already is not None and hasattr(already, _TARGET_ATTR):
        _install(already)
        return

    class _LoaderProxy(importlib.abc.Loader):
        """The real loader, plus one call after the module body runs."""

        def __init__(self, inner):
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            # Upstream runs first and unchanged; if it raises, nothing is installed
            # and the import fails exactly as it would have.
            self._inner.exec_module(module)
            _install(module)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    class _TargetFinder(importlib.abc.MetaPathFinder):
        """Claims exactly one module name, once, then hands the work back."""

        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET_MODULE:
                return None
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            for finder in list(sys.meta_path):
                find_spec = getattr(finder, "find_spec", None)
                if find_spec is None:
                    continue
                spec = find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _LoaderProxy(spec.loader)
                    return spec
            return None

    sys.meta_path.insert(0, _TargetFinder())
    logger.debug(
        "Neuron: hybrid DCP block-size patch deferred to a meta-path loader hook; %s "
        "was not importable at plugin-registration time (circular import).",
        _TARGET_MODULE,
    )


def apply_dcp_hybrid_patch() -> None:
    """Wire the hybrid DCP block-size resolution, eagerly if possible and deferred if not.

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
        _install_deferred()
        return

    _install(kv_cache_utils)
