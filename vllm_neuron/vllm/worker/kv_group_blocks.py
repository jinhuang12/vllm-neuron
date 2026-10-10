# SPDX-License-Identifier: Apache-2.0
"""The blocks a request holds in a KV cache group: its sequence's pages and draft blocks.

Two sizes of the worker and the runner must agree with vLLM's scheduler on how many
blocks a request holds in each KV cache group: the need-sized KV pool
(``NeuronWorker._kv_cache_need_bytes``, handed to vLLM as the available KV memory)
and the per-group rows of the runner's ``InputBatch`` block table
(``NeuronModelRunner.initialize_kv_cache``). Both read the figure from here, so a
spec class is priced once. :func:`draft_blocks_per_request` gives the blocks beyond the
sequence's pages; :func:`group_blocks_per_request` gives the whole figure on one rank,
pages included, at any decode context parallel size.
"""
from __future__ import annotations

from vllm.utils.math_utils import cdiv
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


def group_blocks_per_request(spec: KVCacheSpec, max_model_len: int, where: str, *,
                             dcp: int = 1) -> int:
    """Return the blocks a request of ``max_model_len`` tokens holds in a KV cache group
    on one rank: the sequence's pages plus the group's draft blocks.

    ``dcp`` is the decode context parallel size (vLLM's
    ``decode_context_parallel_size``). Each rank of a decode context parallel group
    keeps ``1 / dcp`` of an attention group's tokens, so an ``AttentionSpec`` group
    holds ``cdiv(max_model_len, block_size * dcp)`` pages per rank. For full and MLA
    attention that is vLLM's own per-rank figure
    (``FullAttentionSpec.max_memory_usage_bytes``, ``MultiGroupBlockTable``); vLLM
    does not serve sliding-window attention under decode context parallelism
    (``SlidingWindowSpec.max_memory_usage_bytes`` asserts ``dcp == 1``). A
    ``MambaSpec`` group is not divided: the runner keeps each recurrent layer's state
    in a bank of one slot per request on every rank, so the group holds
    ``cdiv(max_model_len, block_size)`` pages plus its draft blocks at every ``dcp``.
    This models the served line, where ``--mamba-block-size`` is ``max_model_len``: a
    recurrent group then holds one block (plus its draft blocks) per request at any
    ``dcp``, as the hybrid DCP block-size patch resolves it. Off
    that line, with a shorter recurrent block, vLLM's managers and block table divide
    a recurrent group by ``dcp`` as well (the same patch's caveat); this figure is
    then larger than the blocks vLLM hands out, never smaller, so a pool or a
    block-table row sized on it still holds every request. A
    ``UniformTypeKVCacheSpecs`` group is divided as its layers are; at ``dcp > 1`` its
    layers must agree on whether they are divided.

    At ``dcp = 1`` the figure is ``cdiv(max_model_len, spec.block_size) +
    draft_blocks_per_request(spec, where)`` for every spec class priced here.

    Args:
        spec: The group's KV cache spec.
        max_model_len: The tokens of the longest request the group serves.
        where: The group, named for the refusal.
        dcp: The decode context parallel size, a positive int; 1 (the default) is no
            decode context parallelism.

    Raises:
        ValueError: ``dcp`` is not a positive int; ``spec`` is of a class with no
            pricing here (see :func:`draft_blocks_per_request`); or, at ``dcp > 1``, a
            uniform-type group mixes layers that are divided over the ranks with
            layers that are not.
    """
    if isinstance(dcp, bool) or not isinstance(dcp, int) or dcp < 1:
        raise ValueError(
            f"{where}: dcp is the decode context parallel size, a positive int; "
            f"got {dcp!r}"
        )
    draft_blocks = draft_blocks_per_request(spec, where)
    ranks = dcp if dcp > 1 and _divided_over_ranks(spec, where) else 1
    return cdiv(max_model_len, spec.block_size * ranks) + draft_blocks


def _divided_over_ranks(spec: KVCacheSpec, where: str) -> bool:
    """Whether a decode context parallel rank keeps only its share of the group's tokens.

    ``spec`` is of a class :func:`draft_blocks_per_request` prices: an attention spec is
    divided, a ``MambaSpec`` is not, and a uniform-type group is divided as all of its
    layers are.
    """
    if isinstance(spec, UniformTypeKVCacheSpecs):
        per_member = {
            _divided_over_ranks(member, f"{where}, layer {name!r}")
            for name, member in spec.kv_cache_specs.items()
        }
        if len(per_member) != 1:
            raise ValueError(
                f"{where} is a UniformTypeKVCacheSpecs group of attention layers, which "
                "decode context parallelism divides over the ranks, and MambaSpec "
                "layers, which it does not; the KV pool and the block table price one "
                "figure per group"
            )
        return per_member.pop()
    return isinstance(spec, AttentionSpec)
