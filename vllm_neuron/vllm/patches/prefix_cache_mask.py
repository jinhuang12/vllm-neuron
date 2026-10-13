# SPDX-License-Identifier: Apache-2.0
"""Hash a prefix-cache block only if it ends inside its request's prompt (GLM-5.3-Flash).

vLLM's "align" mode hashes every full block, a block that a decode step fills
included: ``SingleTypeKVCacheManager.cache_blocks`` hands every full block to
``BlockPool.cache_full_blocks``, which skips only null blocks and the blocks that
``reachable_block_mask`` masks out. A later request whose prompt repeats the earlier
prompt and output would then hit a block that a decode step filled. A block that ends
at or before its request's prompt covers prompt tokens only, which a prefill
computes; this module limits a GLM-5.3-Flash hit to such blocks.

:func:`install` gives each single-type manager of one ``KVCacheManager`` its own
``reachable_block_mask``: the manager's own mask AND "block ``i`` ends at or before
``num_prompt_tokens``". ``cache_blocks`` calls the method on the instance, so the
install reaches that engine's manager only, and vLLM's classes stay as they are.
``NeuronScheduler.__init__`` installs it on its own manager for
``Glm5NextForConditionalGeneration`` with prefix caching on, and imports this module
only then.

Every group is masked, the latent (MLA) group as well as the KDA groups. vLLM keeps
a second block under a hash it already holds and a lookup returns the first one, so
a latent block that a decode step filled could be returned in place of a later
prefill's copy. A hybrid hit is the length every group matches, and a KDA block that
ends inside a prompt has every latent block below it inside that prompt too. So
masking the latent group loses only a hit on a latent block a decode step filled.

The cost is the decoded region: a later turn reuses the earlier prompt, not the
earlier output, and prefills that output again.

The names of upstream's mask parameters are checked when the mask is installed. A
change is refused by name (:class:`PrefixCacheMaskTargetError`) at engine start,
rather than masking with the wrong argument.
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

logger = init_logger(__name__)

#: The parameters of upstream's ``reachable_block_mask``, in order. ``cache_blocks``
#: passes all of them by keyword; ``num_prompt_tokens`` is the one this mask reads.
MASK_PARAMETERS = (
    "start_block",
    "end_block",
    "alignment_tokens",
    "kv_cache_spec",
    "use_eagle",
    "retention_interval",
    "num_prompt_tokens",
)


class PrefixCacheMaskTargetError(RuntimeError):
    """Upstream's block-mask method is missing, or takes other parameters than this mask."""


def check_targets() -> None:
    """Refuse by name unless both managers' ``reachable_block_mask`` take :data:`MASK_PARAMETERS`.

    ``MambaManager`` (the KDA groups) has its own method; ``FullAttentionManager`` (the
    latent group) resolves the base class's.
    """
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager

    for manager_class in (MambaManager, FullAttentionManager):
        target = f"{manager_class.__module__}.{manager_class.__name__}.reachable_block_mask"
        method = getattr(manager_class, "reachable_block_mask", None)
        if not callable(method):
            raise PrefixCacheMaskTargetError(
                f"{target} is missing or not callable (got {method!r}), so prefix caching "
                "cannot be limited to the blocks that end inside their prompt. Check whether the "
                "method moved or was renamed at this vLLM pin."
            )
        names = tuple(inspect.signature(method).parameters)
        if names != MASK_PARAMETERS:
            raise PrefixCacheMaskTargetError(
                f"{target} takes {names}, not {MASK_PARAMETERS}. The prefix-cache mask "
                "reads num_prompt_tokens from that call, so it is not installed on a "
                "method whose parameters changed. Check the method at this vLLM pin."
            )


def prompt_only_mask(
    start_block: int, end_block: int, block_size: int, num_prompt_tokens: int | None
) -> list[bool]:
    """``True`` for each block ``start_block .. end_block - 1`` that ends inside the prompt.

    Block ``i`` holds tokens ``i * block_size .. (i + 1) * block_size - 1``. Without a
    prompt length no block is kept: hashing fewer blocks only loses hits.
    """
    prompt_tokens = num_prompt_tokens or 0
    return [(index + 1) * block_size <= prompt_tokens for index in range(start_block, end_block)]


def _masked(
    own_mask: Any,
    start_block: int,
    end_block: int,
    alignment_tokens: int | None,
    kv_cache_spec: Any,
    use_eagle: bool,
    retention_interval: int | None = None,
    num_prompt_tokens: int | None = None,
) -> list[bool]:
    upstream = own_mask(
        start_block=start_block,
        end_block=end_block,
        alignment_tokens=alignment_tokens,
        kv_cache_spec=kv_cache_spec,
        use_eagle=use_eagle,
        retention_interval=retention_interval,
        num_prompt_tokens=num_prompt_tokens,
    )
    mine = prompt_only_mask(start_block, end_block, kv_cache_spec.block_size, num_prompt_tokens)
    if upstream is None:
        return mine
    return [keep and inside for keep, inside in zip(upstream, mine, strict=True)]


def install(kv_cache_manager: KVCacheManager) -> None:
    """Mask every group of ``kv_cache_manager`` to the blocks that end inside their prompt."""
    check_targets()
    managers = kv_cache_manager.coordinator.single_type_managers
    for manager in managers:
        current = vars(manager).get("reachable_block_mask")
        if isinstance(current, functools.partial) and current.func is _masked:
            continue
        manager.reachable_block_mask = functools.partial(_masked, manager.reachable_block_mask)
    logger.info(
        "Prefix caching: %d KV cache group(s) hash a block only if it ends inside its "
        "request's prompt; a later turn reuses "
        "the earlier prompt, not the earlier output.",
        len(managers),
    )
