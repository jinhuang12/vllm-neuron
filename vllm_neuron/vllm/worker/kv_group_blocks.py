# SPDX-License-Identifier: Apache-2.0
"""The blocks a request holds in a KV cache group beyond its sequence's pages.

Two sizes of the worker and the runner must agree with vLLM's scheduler on how many
blocks a request holds in each KV cache group: the need-sized KV pool
(``NeuronWorker._kv_cache_need_bytes``, handed to vLLM as the available KV memory)
and the per-group rows of the runner's ``InputBatch`` block table
(``NeuronModelRunner.initialize_kv_cache``). Both read the figure from here, so a
spec class is priced once.
"""
from __future__ import annotations

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)


def draft_blocks_per_request(spec: KVCacheSpec, where: str) -> int:
    """Return the blocks a request holds in a KV cache group beyond its sequence's pages.

    A ``MambaSpec`` group holds ``num_speculative_blocks`` draft blocks: one state row
    per draft token, which vLLM's ``MambaSpec.max_memory_usage_bytes`` prices (as
    ``1 + num_speculative_blocks`` blocks in the ``none`` cache mode the plugin
    serves) and the scheduler hands out. An ``AttentionSpec`` group holds none. A
    ``UniformTypeKVCacheSpecs`` group -- one group of layers of one type at differing
    geometries, the layout vLLM builds for a drafter whose KV geometry differs from
    the target's (vLLM PR 25101; the runner allocates it) -- holds what its member
    specs hold, which vLLM's grouping keeps uniform. A spec of any other class has no
    pricing here and is refused by name: priced as attention it could hand vLLM a
    pool, or a block-table row, the request does not fit.

    Args:
        spec: The group's KV cache spec.
        where: The group, named for the refusal.

    Raises:
        ValueError: ``spec`` is of a class with no pricing here, or a uniform-type
            group's members do not agree on their draft blocks.
    """
    if isinstance(spec, MambaSpec):
        return int(spec.num_speculative_blocks)
    if isinstance(spec, AttentionSpec):
        return 0
    if isinstance(spec, UniformTypeKVCacheSpecs):
        per_member = {
            draft_blocks_per_request(member, f"{where}, layer {name!r}")
            for name, member in spec.kv_cache_specs.items()
        }
        if len(per_member) != 1:
            raise ValueError(
                f"{where} is a UniformTypeKVCacheSpecs group whose layers hold "
                f"differing draft blocks per request ({sorted(per_member)}); the "
                "KV pool and the block table price one figure per group"
            )
        return per_member.pop()
    raise ValueError(
        f"{where} is a {type(spec).__name__}: the KV pool and the block table price "
        "attention groups (the sequence's pages), MambaSpec groups (the sequence's "
        "pages plus one block per draft token) and uniform-type groups of those "
        "only, so this spec class needs its own pricing before it is served"
    )
