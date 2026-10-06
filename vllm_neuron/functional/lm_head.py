# SPDX-License-Identifier: Apache-2.0
"""Vocab-parallel lm_head: shard the vocabulary, gather the logits on device.

At TP=64 a replicated ``[154880, 4096]`` bf16 head makes every rank read the
same 1268.8 MB per decode step. Sharded by vocabulary row, each rank reads its
own ``154880 / 64 = 2420`` rows (19.8 MB) and an all-gather over the
tensor-parallel group returns the full ``[B, 154880]`` logits to every rank, so
the sampler sees the same shape and dtype it saw with the replicated head.

The loader does the sharding (``lm_head_weight`` is declared vocab-parallel in
the model's shard table, with :func:`vocab_shard_width` as its width), so the
graph carries no rank: each rank's graph input is simply its own rows, and the
all-gather concatenates the ranks' local logits in group order, which is the
order the loader cut them in.
"""

from __future__ import annotations

import torch
from torch import Tensor


class VocabParallelHeadError(ValueError):
    """A head or vocabulary geometry the vocab-parallel route refuses."""


def vocab_shard_rows(vocab_size: int, world_size: int) -> int:
    """Rows of the head each rank holds; refuses a vocabulary that does not divide.

    A remainder would need a padded last shard and a slice after the gather;
    the real geometry (154880 / 64 = 2420) needs neither, so it is refused by
    name rather than handled by a branch nothing exercises.
    """
    if world_size < 1 or vocab_size < 1:
        raise VocabParallelHeadError(
            f"vocab_size={vocab_size} and world_size={world_size} must be positive"
        )
    if vocab_size % world_size:
        raise VocabParallelHeadError(
            f"vocab_size={vocab_size} does not divide over {world_size} ranks; "
            f"the vocab-parallel head has no padded-shard path"
        )
    return vocab_size // world_size


def vocab_shard_width(module: torch.nn.Module, world_size: int) -> int:
    """The shard-table width function for ``lm_head_weight``: rows per rank."""
    return vocab_shard_rows(int(module.text_config.vocab_size), world_size)


def vocab_parallel_logits(
    rows: Tensor,
    head: Tensor,
    *,
    vocab_size: int,
    group,
) -> Tensor:
    """``[B, vocab]`` logits from a whole head or from this rank's vocab shard.

    Args:
        rows: ``[B, H]`` hidden states selected for sampling.
        head: ``[vocab, H]`` (replicated, or tied to the embedding table) or
            ``[vocab / world, H]`` (this rank's vocab shard).
        vocab_size: the model's vocabulary, which the result always spans.
        group: the tensor-parallel ``GroupCoordinator``, or ``None`` at one rank.

    Returns:
        ``[B, vocab]`` in the shared dtype of ``rows`` and ``head``, on every rank.

    Raises:
        VocabParallelHeadError: when ``head`` is neither the whole vocabulary nor
            exactly one rank's shard of it.
    """
    head_rows = int(head.shape[0])
    if head_rows == vocab_size:
        return torch.nn.functional.linear(rows, head)
    world = int(getattr(group, "world_size", 1)) if group is not None else 1
    if world <= 1 or head_rows * world != vocab_size:
        raise VocabParallelHeadError(
            f"lm_head has {head_rows} rows; expected the whole vocabulary "
            f"({vocab_size}) or one of {world} vocab shards"
        )
    local = torch.nn.functional.linear(rows, head)
    return group.all_gather(local, dim=-1)
