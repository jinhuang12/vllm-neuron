# SPDX-License-Identifier: Apache-2.0
"""TP-correct greedy draft token from a vocab-shard head.

At TP=64 each rank holds ``154880 / 64 = 2420`` rows of the head
(``functional/lm_head.py``). A rank's local ``argmax`` over its shard logits is an
index into its own shard, so handing it on as a token id is wrong on 63 of 64 ranks
(mtp.md H2). The route here keeps the head sharded and moves two numbers per row:

    shard logits [B, rows]  ->  (max, argmax) per row [B, 2]
      -> all-gather over the group [B, 2 * world]
      -> the rank with the largest max wins; id = rank * rows + its local argmax

which is what the full ``argmax`` over the gathered ``[B, vocab]`` logits would
return (H5: ``2 x 64`` values per row cross the wire, not ``154880``). Ties resolve
to the lowest id, the convention of ``torch.argmax`` on the full logits: within a
shard ``argmax`` picks the lowest index, and across shards the lowest rank -- which
holds the lowest ids -- wins an equal max.

The shard logits are computed in the head's own dtype, as ``vocab_parallel_logits``
computes the trunk's (bf16 in, bf16 out), so the draft's greedy token is taken on
the same logits the trunk's sampler sees; the pair that crosses the wire is fp32.

A whole head (``[vocab, H]``, one rank or a replicated head) takes the plain
``argmax`` with no collective. Any other row count is refused by name.
"""

from __future__ import annotations

import torch
from torch import Tensor


class DraftTokenError(ValueError):
    """A head geometry the sharded draft-token route refuses."""


def draft_token_ids(
    rows: Tensor,
    head: Tensor,
    *,
    vocab_size: int,
    group,
) -> Tensor:
    """``[B]`` int32 global greedy token ids for ``rows`` against ``head``.

    Args:
        rows: ``[B, H]`` hidden states (the shared-head-normed draft rows).
        head: ``[vocab, H]`` (whole) or ``[vocab / world, H]`` (this rank's shard).
        vocab_size: the model's vocabulary; the ids returned lie in ``[0, vocab)``.
        group: the tensor-parallel ``GroupCoordinator`` (``world_size``,
            ``all_gather(tensor, dim)``), or ``None`` at one rank.

    Raises:
        DraftTokenError: when ``head`` is neither the whole vocabulary nor exactly
            one of ``group.world_size`` shards of it.
    """
    vocab = int(vocab_size)
    shard_rows = int(head.shape[0])
    logits = torch.nn.functional.linear(rows.to(head.dtype), head).to(torch.float32)
    if shard_rows == vocab:
        return logits.argmax(dim=-1).to(torch.int32)
    world = int(getattr(group, "world_size", 1)) if group is not None else 1
    if world <= 1 or shard_rows * world != vocab:
        raise DraftTokenError(
            f"the head has {shard_rows} rows; expected the whole vocabulary "
            f"({vocab}) or one of {world} vocab shards of {vocab // max(world, 1)} rows"
        )
    local_max, local_arg = logits.max(dim=-1)
    pair = torch.stack([local_max, local_arg.to(torch.float32)], dim=-1)  # [B, 2]
    gathered = group.all_gather(pair, dim=-1)  # [B, 2 * world], rank-major
    maxes = gathered[:, 0::2]
    args = gathered[:, 1::2]
    owner = maxes.argmax(dim=-1)  # the lowest rank among equal maxes
    local = torch.gather(args, 1, owner.reshape(-1, 1)).reshape(-1).to(torch.int64)
    return (owner.to(torch.int64) * shard_rows + local).to(torch.int32)
