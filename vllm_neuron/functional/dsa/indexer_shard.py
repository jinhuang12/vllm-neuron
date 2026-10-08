# SPDX-License-Identifier: Apache-2.0
"""Query-row sharding of the DSA prefill selection across the tensor-parallel ranks.

As built, every rank runs the indexer's score GEMM, causal bound, top-k and sentinel
ordering on every query row of the chunk: the indexer weights and the pooled keys are
replicated, so all ranks compute the same ``[T, k]`` pool ids. The selection is a function
of each row alone, so the rows can be divided instead::

    rank r of d:  rows r*R .. r*R + R - 1   (R = ceil(T / d); the last ranks pad)
        score  [R, C]  ->  bound  ->  top-k  ->  sentinel  ->  order   =  [R, k]
    all ranks:   all-gather on dim 0  ->  [d*R, k]  ->  first T rows

The sharding contract: every rank still holds the whole candidate axis (the same ``C``
pooled keys), so each row's chain sees exactly the operands it saw before, and the
kernels are row-tiled with no cross-row arithmetic, so a row selects the same pool-id set.
Within equal scores the selector's slot order may differ, because the vendored top-k picks
its tile shape from its row count. The expansion after the gather runs on all ``T`` rows,
as before, with each row's own length.

The rank is a device operand, not a python int: one prefill graph serves every rank (the
runner passes its own rank to the model as a tensor for the same reason), so a rank read
at trace time would compile rank 0's rows into every rank's graph. A slice cannot take a
device start row, so the cut is an NKI kernel (``shard_rows.dsa_take_rank_rows``) that
builds the row index on device and gathers the rows by indirect DMA.

The all-gather contract: the ids cross the collective as the selector's own int32, in the
group's ``rank_in_group`` order. An all-gather only moves bytes, so no cast is needed on
either side; when ``d * R == T`` the gathered block is the chunk as it stands, and only a
padded plan takes the leading ``T`` rows, a view.

``VLLM_NEURON_DSA_INDEXER_SHARD=0`` (``vllm_neuron/envs.py``) restores the replicated path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from vllm_neuron import envs
from vllm_neuron.functional.dsa.score_gemm import TOKEN_TILE
from vllm_neuron.functional.dsa.shard_rows import dsa_take_rank_rows, rank_row_index

#: Query rows one row tile of the selection kernels holds: the score GEMM's stationary
#: free size (``score_gemm.TOKEN_TILE``, ``nl.tile_size.pmax`` = 128 on trn2), which is
#: also the SBUF partition height the causal bound and the sentinel ordering walk. A chunk
#: of at most this many rows is one tile on every rank already, so sharding it buys
#: nothing.
ROW_TILE = TOKEN_TILE

#: The dtype of the selector's pool ids, which the all-gather moves unchanged.
POOL_ID_DTYPE = torch.int32


class IndexerShardError(ValueError):
    """A shard plan or a gather this module refuses: an empty chunk, a non-positive degree,
    an operand or local block of the wrong shape, or ids of another dtype."""


def indexer_shard_enabled() -> bool:
    """Whether the prefill selection may shard (``envs.VLLM_NEURON_DSA_INDEXER_SHARD``)."""
    return bool(envs.VLLM_NEURON_DSA_INDEXER_SHARD)


@dataclass(frozen=True)
class RowShard:
    """How a chunk's query rows divide over the ranks.

    ``tokens`` is the chunk's row count ``T``, ``degree`` the rank count ``d`` sharing it,
    and ``rows`` the rows each rank selects, ``R = ceil(T / d)``. Every rank runs ``R``
    rows so every rank's graph is the same shape; the ``d * R - T`` trailing rows of the
    last ranks repeat row ``T - 1`` and are dropped after the gather.
    """

    tokens: int
    degree: int
    rows: int


def row_shard(tokens: int, degree: int) -> RowShard:
    """The plan for ``tokens`` rows over ``degree`` ranks."""
    tokens, degree = int(tokens), int(degree)
    if tokens < 1 or degree < 1:
        raise IndexerShardError(
            f"a shard needs at least one row and one rank; got tokens={tokens}, "
            f"degree={degree}"
        )
    return RowShard(tokens=tokens, degree=degree, rows=-(-tokens // degree))


def shard_degree(tokens: int, world_size: int) -> int:
    """How many ranks share a chunk's selection: the world, or 1 for the replicated path.

    1 when the switch is off, at one rank, and for a chunk of at most :data:`ROW_TILE`
    rows, where each rank would still run one row tile per kernel and only pay the
    collective.
    """
    if not indexer_shard_enabled() or int(world_size) <= 1:
        return 1
    if int(tokens) <= ROW_TILE:
        return 1
    return int(world_size)


def local_row_index(shard: RowShard, rank: Tensor | int, device: torch.device) -> Tensor:
    """``[R]`` int64: the chunk rows rank ``rank`` selects, padding clamped to row ``T - 1``.

    The reference for the row cut :func:`select_local_rows` makes on device
    (``shard_rows.rank_row_index``). ``rank`` is a one-element integer tensor or a python
    int. The clamp keeps every pad row a legal row of the chunk, so its chain computes
    ordinary values that the gather then drops.
    """
    return rank_row_index(shard.tokens, shard.rows, rank, device)


def select_local_rows(
    select: Callable[[Tensor, Tensor, Tensor], Tensor],
    query: Tensor,
    weights: Tensor,
    seq_lens: Tensor,
    shard: RowShard,
    rank: Tensor | int,
) -> Tensor:
    """One rank's share of a selection: ``select`` applied to that rank's rows.

    Args:
        select: ``(query, weights, seq_lens) -> [rows, k]`` int32 pool ids for any number of
            rows, each row selected from its own operands only (the caller binds the
            candidate keys, which every rank holds whole).
        query: ``[T, ...]`` per-row query, any dtype.
        weights: ``[T, ...]`` per-row gate, any dtype.
        seq_lens: ``[T]`` (or ``[T, 1]``) per-row causal length.
        shard: the chunk's plan; ``T`` must be ``shard.tokens``.
        rank: this rank's index, see :func:`local_row_index`.

    Returns:
        ``[R, k]``: ``select`` on rows ``rank * R .. rank * R + R - 1`` (pad rows clamped).

    The three operands' rows are cut on device by ``shard_rows.dsa_take_rank_rows``, in
    one NKI launch.
    """
    for name, operand in (("query", query), ("weights", weights), ("seq_lens", seq_lens)):
        if int(operand.shape[0]) != shard.tokens:
            raise IndexerShardError(
                f"{name} has {int(operand.shape[0])} rows; the plan is for {shard.tokens}"
            )
    return select(*dsa_take_rank_rows((query, weights, seq_lens), rank, shard.rows))


def gather_rows(local: Tensor, group, shard: RowShard) -> Tensor:
    """Every rank's ``[R, k]`` int32 pool ids, gathered into the chunk's ``[T, k]`` int32.

    ``group`` is the tensor-parallel ``GroupCoordinator`` whose ``rank_in_group`` order the
    rows were cut in; its ``all_gather`` concatenates on ``dim=0`` in that order, which is
    the order :func:`local_row_index` assigned. The ids cross as they are; a padded plan
    keeps the leading ``T`` rows (a view of the gathered block).
    """
    if int(local.shape[0]) != shard.rows or local.dim() != 2:
        raise IndexerShardError(
            f"a rank gathers [{shard.rows}, k] ids; got shape {tuple(local.shape)}"
        )
    if local.dtype != POOL_ID_DTYPE:
        raise IndexerShardError(
            f"the gather moves the selector's {POOL_ID_DTYPE} ids; got {local.dtype}"
        )
    whole = group.all_gather(local, dim=0)
    if shard.degree * shard.rows == shard.tokens:
        return whole
    return whole[: shard.tokens]
