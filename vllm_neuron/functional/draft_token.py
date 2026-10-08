# SPDX-License-Identifier: Apache-2.0
"""TP-correct greedy draft token from a vocab-shard head's ``(max, argmax)`` pair.

At TP=64 each rank holds ``154880 / 64 = 2420`` rows of the head
(``functional/lm_head.py``). A rank's local ``argmax`` over its shard logits is an
index into its own shard, so handing it on as a token id is wrong on 63 of 64 ranks
(mtp.md H2). The route here keeps the head sharded and moves two numbers per row:

    shard logits [B, rows]  ->  (max, argmax) per row [B, 2]   (the output tail
                                 kernel's epilogue, functional/mtp/tail_out.py)
      -> all-gather over the group [B, 2 * world]
      -> the rank with the largest max wins; id = rank * rows + its local argmax

which is what the full ``argmax`` over the gathered ``[B, vocab]`` logits would
return (H5: ``2 x 64`` values per row cross the wire, not ``154880``). Ties resolve
to the lowest id, the convention of ``torch.argmax`` on the full logits: within a
shard the kernel's ``nc_find_index8`` (and ``shard_pair``'s ``argmax``) pick the
lowest index, and across shards the lowest rank -- which holds the lowest ids -- wins
an equal max.

The pair is fp32 (an index below ``2**24`` is exact in it); the logits it was taken
from are the head's own dtype, as ``vocab_parallel_logits`` computes the trunk's.

A whole head (``shard_rows == vocab_size``: one rank, or a replicated head) needs no
collective: the local index is the id. Any other row count is refused by name.
"""

from __future__ import annotations

import torch
from torch import Tensor


class DraftTokenError(ValueError):
    """A pair or head geometry the sharded draft-token route refuses."""


def draft_token_ids(
    pair: Tensor,
    *,
    shard_rows: int,
    vocab_size: int,
    group,
) -> Tensor:
    """``[B]`` int32 global greedy token ids from this rank's ``(max, argmax)`` pair.

    Args:
        pair: ``[B, 2]`` fp32, per row the shard's largest logit and the index of
            its first occurrence within the shard
            (:func:`vllm_neuron.functional.mtp.tail_out.shard_pair`).
        shard_rows: the head rows this rank holds: ``vocab_size`` (the whole head)
            or ``vocab_size / group.world_size`` (this rank's shard).
        vocab_size: the model's vocabulary; the ids returned lie in ``[0, vocab)``.
        group: the tensor-parallel ``GroupCoordinator`` (``world_size``,
            ``all_gather(tensor, dim)``), or ``None`` at one rank.

    Raises:
        DraftTokenError: when ``pair`` is not ``[B, 2]`` fp32, or ``shard_rows`` is
            neither the whole vocabulary nor exactly one of ``group.world_size``
            shards of it.
    """
    if pair.dim() != 2 or pair.shape[1] != 2:
        raise DraftTokenError(
            f"the draft pair is [B, 2] (max, argmax) per row; got {tuple(pair.shape)}"
        )
    if pair.dtype != torch.float32:
        raise DraftTokenError(f"the draft pair crosses the wire as float32; got {pair.dtype}")
    vocab = int(vocab_size)
    shard_rows = int(shard_rows)
    if shard_rows == vocab:
        return pair[:, 1].to(torch.int32)
    world = int(getattr(group, "world_size", 1)) if group is not None else 1
    if world <= 1 or shard_rows * world != vocab:
        raise DraftTokenError(
            f"the head has {shard_rows} rows; expected the whole vocabulary "
            f"({vocab}) or one of {world} vocab shards of {vocab // max(world, 1)} rows"
        )
    gathered = group.all_gather(pair, dim=-1)  # [B, 2 * world], rank-major
    maxes = gathered[:, 0::2]
    args = gathered[:, 1::2]
    owner = maxes.argmax(dim=-1)  # the lowest rank among equal maxes
    local = torch.gather(args, 1, owner.reshape(-1, 1)).reshape(-1).to(torch.int64)
    return (owner.to(torch.int64) * shard_rows + local).to(torch.int32)
