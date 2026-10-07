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
at trace time would compile rank 0's rows into every rank's graph.

The all-gather contract: the ids cross the collective as float32, by value, in the
group's ``rank_in_group`` order. Float32 is the dtype the model's other collectives move,
and every id here (``-1`` and the pool ids ``< C``) is an integer of magnitude at most
``2 ** 24``, which float32 carries exactly; :func:`gather_rows` refuses a wider id range. A
bit-cast would also be exact in the bytes, but it would turn ``-1`` into a NaN pattern,
and nothing promises that a NaN crosses a collective unchanged.

``VLLM_NEURON_DSA_INDEXER_SHARD=0`` (``vllm_neuron/envs.py``) restores the replicated path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from vllm_neuron import envs
from vllm_neuron.functional.dsa.score_gemm import TOKEN_TILE

#: Query rows one row tile of the selection kernels holds: the score GEMM's stationary
#: free size (``score_gemm.TOKEN_TILE``, ``nl.tile_size.pmax`` = 128 on trn2), which is
#: also the SBUF partition height the causal bound and the sentinel ordering walk. A chunk
#: of at most this many rows is one tile on every rank already, so sharding it buys
#: nothing.
ROW_TILE = TOKEN_TILE

#: Every integer of magnitude at most ``2 ** 24`` is exact in float32 (24-bit significand).
FP32_EXACT_INT = 2**24


class IndexerShardError(ValueError):
    """A shard plan or a gather this module refuses: an empty chunk, a non-positive degree,
    a rank operand or local block of the wrong shape, or ids float32 cannot carry exactly."""


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

    ``rank`` is a one-element integer tensor on the traced path (one graph for every rank)
    or a python int in an eager caller. The clamp keeps every pad row a legal row of the
    chunk, so its chain computes ordinary values that the gather then drops.
    """
    offsets = torch.arange(shard.rows, device=device, dtype=torch.int64)
    if torch.is_tensor(rank):
        if rank.numel() != 1 or rank.is_floating_point():
            raise IndexerShardError(
                f"the rank operand must be one integer element; got shape "
                f"{tuple(rank.shape)} dtype {rank.dtype}"
            )
        start = rank.to(device=device, dtype=torch.int64).reshape(()) * shard.rows
    else:
        start = int(rank) * shard.rows
    return (offsets + start).clamp_max(shard.tokens - 1)


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
    """
    for name, operand in (("query", query), ("weights", weights), ("seq_lens", seq_lens)):
        if int(operand.shape[0]) != shard.tokens:
            raise IndexerShardError(
                f"{name} has {int(operand.shape[0])} rows; the plan is for {shard.tokens}"
            )
    rows = local_row_index(shard, rank, query.device)
    return select(query.index_select(0, rows), weights.index_select(0, rows),
                  seq_lens.index_select(0, rows))


def gather_rows(local: Tensor, group, shard: RowShard, *, id_bound: int) -> Tensor:
    """Every rank's ``[R, k]`` int32 pool ids, gathered into the chunk's ``[T, k]`` int32.

    ``group`` is the tensor-parallel ``GroupCoordinator`` whose ``rank_in_group`` order the
    rows were cut in; its ``all_gather`` concatenates on ``dim=0`` in that order, which is
    the order :func:`local_row_index` assigned. ``id_bound`` is the candidate count, the
    exclusive upper bound of every id, and must leave float32 exact.
    """
    if int(local.shape[0]) != shard.rows or local.dim() != 2:
        raise IndexerShardError(
            f"a rank gathers [{shard.rows}, k] ids; got shape {tuple(local.shape)}"
        )
    if int(id_bound) > FP32_EXACT_INT:
        raise IndexerShardError(
            f"ids up to {int(id_bound) - 1} cross the gather as float32, which is exact only "
            f"up to {FP32_EXACT_INT}"
        )
    whole = group.all_gather(local.to(torch.float32), dim=0)
    return whole[: shard.tokens].to(torch.int32)
