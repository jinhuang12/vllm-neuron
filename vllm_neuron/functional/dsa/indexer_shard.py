# SPDX-License-Identifier: Apache-2.0
"""Query-row sharding of the DSA prefill selection across the tensor-parallel ranks.

As built, every rank runs the indexer's score GEMM, causal bound, top-k and sentinel
ordering on every query row of the chunk: the indexer weights and the pooled keys are
replicated, so all ranks compute the same ``[T, select_k]`` pool ids. The selection is a
function of each row alone, so the rows can be divided instead::

    rank r of d:  rows r*R .. r*R + R - 1   (R = ceil(T / d); the last rank pads)
        score  [R, C]  ->  bound  ->  top-k  ->  sentinel  ->  order   =  [R, select_k]
    all ranks:   all-gather on dim 0  ->  [d*R, select_k]  ->  first T rows

Every rank still holds the whole candidate axis (the same ``C`` pooled keys), so each
row's chain sees exactly the operands it saw before, and the kernels are row-tiled with
no cross-row arithmetic: a row's pool ids are the same pool ids. The expansion after the
gather runs on all ``T`` rows, as before, with each row's own length.

The rank is a device operand, not a python int: one prefill graph serves every rank (the
runner passes its own rank to the model as a tensor for the same reason), so a rank read
at trace time would compile rank 0's rows into every rank's graph.

The ids cross the collective as float32, by value. That is the dtype the model's other
collectives move, and every id here (``-1`` and the pool ids ``< C``) is an integer below
``2 ** 24``, which float32 carries exactly; :func:`gather_rows` refuses a wider id range.
A bit-cast would also be exact in the bytes, but it would turn ``-1`` into a NaN pattern,
and nothing promises that a NaN crosses a collective unchanged.

``VLLM_NEURON_DSA_INDEXER_SHARD=0`` restores the replicated path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import Tensor

#: The kill switch. ``0`` restores the replicated selection; anything else, or unset,
#: shards. Read at the call site on every trace, the convention of
#: ``VLLM_NEURON_KDA_FUSED_DECODE`` (``functional/kda/fused_decode.py``).
SHARD_ENV = "VLLM_NEURON_DSA_INDEXER_SHARD"

#: Query rows one SBUF tile holds. The score GEMM, the causal bound, the sentinel writer
#: and the ordering all walk the rows in tiles of this height, so a chunk of at most this
#: many rows is one tile on every rank already and sharding it buys nothing there.
PARTITION_MAX = 128

#: Integers float32 represents exactly: every integer of magnitude ``<= 2 ** 24``.
FP32_EXACT_INT = 2**24


class IndexerShardError(ValueError):
    """A shard plan or a gather this module refuses: an empty chunk, a non-positive degree,
    or ids float32 cannot carry exactly."""


def indexer_shard_enabled() -> bool:
    """Whether the prefill selection may shard. ``VLLM_NEURON_DSA_INDEXER_SHARD=0`` is off."""
    return os.environ.get(SHARD_ENV, "1") != "0"


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

    1 when the switch is off, at one rank, and for a chunk of at most
    :data:`PARTITION_MAX` rows, where each rank would still run one row tile per kernel
    and only pay the collective.
    """
    if not indexer_shard_enabled() or int(world_size) <= 1:
        return 1
    if int(tokens) <= PARTITION_MAX:
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
        start = rank.to(device=device, dtype=torch.int64).reshape(()) * shard.rows
    else:
        start = int(rank) * shard.rows
    return (offsets + start).clamp_max(shard.tokens - 1)


def take_rows(operand: Tensor, index: Tensor) -> Tensor:
    """``operand``'s rows at ``index``, on the leading axis."""
    return operand.index_select(0, index)


def gather_rows(local: Tensor, group, shard: RowShard, *, id_bound: int) -> Tensor:
    """Every rank's ``[R, k]`` int32 pool ids, gathered into the chunk's ``[T, k]`` int32.

    ``group`` is the tensor-parallel ``GroupCoordinator`` whose ``rank_in_group`` order the
    rows were cut in; its ``all_gather`` concatenates on ``dim=0`` in that order, which is
    the order :func:`local_row_index` assigned. ``id_bound`` is the candidate count, the
    exclusive upper bound of every id, and must leave float32 exact.
    """
    if int(id_bound) > FP32_EXACT_INT:
        raise IndexerShardError(
            f"ids up to {int(id_bound) - 1} cross the gather as float32, which is exact only "
            f"up to {FP32_EXACT_INT}"
        )
    whole = group.all_gather(local.to(torch.float32), dim=0)
    return whole[: shard.tokens].to(torch.int32)
