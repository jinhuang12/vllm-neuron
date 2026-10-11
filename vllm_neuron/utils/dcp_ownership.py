# SPDX-License-Identifier: Apache-2.0
"""Which rank of a decode-context-parallel (DCP) group holds each KV row and indexer pool.

Under DCP the ``dcp_size`` ranks of a DCP group split every sequence's context by
block, the interleave the runner's slot mapping uses for prefill and decode alike
(``cp_kv_cache_interleave_size == block_size``):

* token ``pos`` lies in block ``b = pos // block_size``;
* block ``b`` is owned by rank ``b % dcp_size``, which stores it as its local block
  ``b // dcp_size``;
* so the token's local row on its owner is
  ``(b // dcp_size) * block_size + pos % block_size``.

A rank's owned blocks are packed in order, and every one of them except the last
is whole, so the local rows a rank holds for a context are ``0 .. L - 1`` with ``L``
the rank's :func:`owned_length`. The local row of a rank's next token is the count
of tokens it already holds. The lengths of the ranks of a group sum to the
context.

The DSA indexer's pools (``index_kpool`` consecutive tokens each) go with their
block. ``index_kpool`` divides ``block_size``, so no pool straddles two blocks, and
pool ``p`` lies in block ``p // pools_per_block`` with
``pools_per_block = block_size // index_kpool``. Its owner stores it as local pool
``(b // dcp_size) * pools_per_block + p % pools_per_block``.

At ``dcp_size == 1`` every map is the identity. Every function is integer torch
arithmetic, so the runner can call it on host tensors and a traced model on device
tensors. Positions, pools and lengths must be non-negative: a traced caller cannot
check that, so the runner checks it when it builds them.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _check_geometry(block_size: int, dcp_size: int) -> None:
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive int; got {block_size!r}")
    if not isinstance(dcp_size, int) or dcp_size <= 0:
        raise ValueError(f"dcp_size must be a positive int; got {dcp_size!r}")


def _check_rank(rank: int | Tensor, dcp_size: int) -> None:
    if not torch.is_tensor(rank) and not 0 <= int(rank) < dcp_size:
        raise ValueError(f"rank must lie in [0, dcp_size={dcp_size}); got {rank!r}")


def pools_per_block(*, block_size: int, index_kpool: int) -> int:
    """Return the indexer pools one KV block holds: ``block_size // index_kpool``.

    Raises:
        ValueError: ``index_kpool`` is not a positive divisor of ``block_size``. A
            pool would then straddle two blocks, which two ranks can own.
    """
    if not isinstance(index_kpool, int) or index_kpool <= 0:
        raise ValueError(f"index_kpool must be a positive int; got {index_kpool!r}")
    if not isinstance(block_size, int) or block_size <= 0 or block_size % index_kpool:
        raise ValueError(
            f"index_kpool={index_kpool} must divide block_size={block_size!r}: DCP "
            f"assigns whole blocks to ranks, so a pool that straddles two blocks "
            f"has no single owner"
        )
    return block_size // index_kpool


def token_owner(positions: Tensor, *, block_size: int, dcp_size: int) -> Tensor:
    """Return the rank that holds each token: ``(positions // block_size) % dcp_size``.

    Args:
        positions: integer tensor of absolute token positions, any shape.
        block_size: tokens per KV block (the DCP interleave).
        dcp_size: ranks in the DCP group.

    Returns:
        A tensor of ``positions``' shape and dtype, values in ``[0, dcp_size)``.
    """
    _check_geometry(block_size, dcp_size)
    return (positions // block_size) % dcp_size


def token_local_row(positions: Tensor, *, block_size: int, dcp_size: int) -> Tensor:
    """Return each token's row in its owner's local KV axis.

    ``(b // dcp_size) * block_size + positions % block_size`` with
    ``b = positions // block_size``. Same shape and dtype as ``positions``.
    """
    _check_geometry(block_size, dcp_size)
    return (positions // block_size // dcp_size) * block_size + positions % block_size


def token_position(
    local_rows: Tensor, owner: int | Tensor, *, block_size: int, dcp_size: int
) -> Tensor:
    """Return the absolute position of local row ``local_rows`` on rank ``owner``.

    The inverse of (:func:`token_owner`, :func:`token_local_row`):
    ``((local_rows // block_size) * dcp_size + owner) * block_size
    + local_rows % block_size``. ``owner`` is an int or a tensor that broadcasts
    against ``local_rows``.
    """
    _check_geometry(block_size, dcp_size)
    _check_rank(owner, dcp_size)
    local_block = local_rows // block_size
    return (local_block * dcp_size + owner) * block_size + local_rows % block_size


def pool_owner(
    pools: Tensor, *, block_size: int, dcp_size: int, index_kpool: int
) -> Tensor:
    """Return the rank that holds each indexer pool: the owner of its block.

    Args:
        pools: integer tensor of absolute pool indices (``position // index_kpool``).
        block_size: tokens per KV block.
        dcp_size: ranks in the DCP group.
        index_kpool: tokens per pool; must divide ``block_size``.
    """
    _check_geometry(block_size, dcp_size)
    per_block = pools_per_block(block_size=block_size, index_kpool=index_kpool)
    return (pools // per_block) % dcp_size


def pool_local_index(
    pools: Tensor, *, block_size: int, dcp_size: int, index_kpool: int
) -> Tensor:
    """Return each pool's index in its owner's local pool axis.

    ``(b // dcp_size) * pools_per_block + pools % pools_per_block`` with
    ``b = pools // pools_per_block``. Same shape and dtype as ``pools``.
    """
    _check_geometry(block_size, dcp_size)
    per_block = pools_per_block(block_size=block_size, index_kpool=index_kpool)
    return (pools // per_block // dcp_size) * per_block + pools % per_block


def pool_global_index(
    local_pools: Tensor,
    owner: int | Tensor,
    *,
    block_size: int,
    dcp_size: int,
    index_kpool: int,
) -> Tensor:
    """Return the absolute pool index of local pool ``local_pools`` on rank ``owner``.

    The inverse of (:func:`pool_owner`, :func:`pool_local_index`):
    ``((local_pools // pools_per_block) * dcp_size + owner) * pools_per_block
    + local_pools % pools_per_block``.
    """
    _check_geometry(block_size, dcp_size)
    _check_rank(owner, dcp_size)
    per_block = pools_per_block(block_size=block_size, index_kpool=index_kpool)
    local_block = local_pools // per_block
    return (local_block * dcp_size + owner) * per_block + local_pools % per_block


def owned_length(
    context_lens: Tensor, rank: int | Tensor, *, block_size: int, dcp_size: int
) -> Tensor:
    """Return how many of the first ``context_lens`` tokens rank ``rank`` holds.

    The ``f = context_lens // block_size`` whole blocks give the rank
    ``(f + dcp_size - 1 - rank) // dcp_size`` of them, and the partial block ``f``
    (``context_lens % block_size`` tokens) is the rank's when ``f % dcp_size ==
    rank``. That count is also the local row of the rank's next token: for the
    owner of position ``pos``, ``owned_length(pos, owner) == token_local_row(pos)``.
    For any other rank it is a whole number of blocks.

    Args:
        context_lens: integer tensor of context lengths, any shape.
        rank: the rank, an int or a tensor that broadcasts against ``context_lens``.
        block_size: tokens per KV block.
        dcp_size: ranks in the DCP group.
    """
    _check_geometry(block_size, dcp_size)
    _check_rank(rank, dcp_size)
    whole = context_lens // block_size
    partial = context_lens % block_size
    blocks = (whole + (dcp_size - 1) - rank) // dcp_size
    own_partial = torch.where(whole % dcp_size == rank, partial, torch.zeros_like(partial))
    return blocks * block_size + own_partial


def owned_lengths(context_lens: Tensor, *, block_size: int, dcp_size: int) -> Tensor:
    """Return every rank's :func:`owned_length`: ``[*context_lens.shape, dcp_size]``.

    The last axis sums to ``context_lens``.
    """
    ranks = torch.arange(dcp_size, dtype=context_lens.dtype, device=context_lens.device)
    return owned_length(
        context_lens.unsqueeze(-1), ranks, block_size=block_size, dcp_size=dcp_size
    )


def ownership_stride(*, block_size: int, dcp_size: int) -> int:
    """Return the positions of a sequence one block-table entry covers: ``block_size * dcp_size``.

    An entry names one block on every rank of the group, and the ranks' blocks of
    entry ``e`` are the consecutive blocks ``e * dcp_size .. e * dcp_size +
    dcp_size - 1`` of the sequence. So a rank needs ``ceil(context /
    ownership_stride)`` blocks for a context of ``context`` tokens. That is vLLM's
    own per-rank block rule: its KV cache manager scales each group's block size by
    the DCP size (``SingleTypeKVCacheManager.__init__``). Every per-rank block count
    divides by this one number.
    """
    _check_geometry(block_size, dcp_size)
    return block_size * dcp_size


def local_context_capacity(context: int, *, block_size: int, dcp_size: int) -> int:
    """Return the local rows one rank needs for a context of ``context`` tokens.

    Whole blocks, as the block table addresses them:
    ``ceil(context / ownership_stride) * block_size``. This is the most any rank of
    the group holds, rounded up to its last block. It is the bound a rank-local
    store (the latent window, the indexer's pool store, the decode candidate axis)
    is sized by.
    """
    stride = ownership_stride(block_size=block_size, dcp_size=dcp_size)
    if not isinstance(context, int) or context < 0:
        raise ValueError(f"context must be a non-negative int; got {context!r}")
    return -(-context // stride) * block_size
