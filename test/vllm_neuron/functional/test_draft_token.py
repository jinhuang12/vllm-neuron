# SPDX-License-Identifier: Apache-2.0
"""TP-correct greedy draft token from a vocab-shard head: ``functional/draft_token.py``.

At TP=64 each rank holds ``154880 / 64 = 2420`` rows of the head. A rank's local
``argmax`` is an index into its own shard, so handing it on as a token id is wrong on
63 of 64 ranks. The route under test takes the shard's ``(max, argmax)``
pair -- the output tail kernel's second result (``functional/mtp/tail_out.py``), here
built by its torch route ``shard_pair`` from the shard logits -- all-gathers that
``[B, 2]`` pair once over the group and resolves the global id as
``rank * shard_rows + local_index`` of the rank holding the largest max (H5: the gather
moves ``2 x 64`` values per row, not ``154880``).

The reference is the replicated head's ``argmax`` over the full vocabulary, which is
what every rank must return. Ties resolve to the lowest id, the same convention as
``torch.argmax`` on the full logits, so the two agree bit for bit on tied rows too.
The logits are taken in the head's dtype (bf16, as the trunk's ``vocab_parallel_logits``
takes them) on both sides, so the comparison reads the id recovery, not a rounding.

The group is simulated in two rounds, the pattern of
``test_tiny_glm5next_vocab_parallel_sampling.py``: round one records each rank's local
operand and returns zeros; round two returns the rank-ordered concatenation of what
every rank recorded. A rank that gathered twice, or gathered a different operand in the
two rounds, fails inside the group.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.draft_token import DraftTokenError, draft_token_ids
from vllm_neuron.functional.mtp.tail_out import shard_pair

VOCAB = 154880
TP = 64
HIDDEN = 64
SHARD_ROWS = VOCAB // TP


class _SimulatedTensorParallelGroup:
    """``world`` ranks of one tensor-parallel group, run one after another."""

    def __init__(self, world: int):
        self.world_size = int(world)
        self.rank: int | None = None
        self.gathering = False
        self.recorded: dict[int, torch.Tensor] = {}
        self.gathers: list[int] = []

    def all_gather(self, local: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim in (-1, local.dim() - 1), f"the draft gathers on the last dim, got {dim}"
        self.gathers.append(self.rank)
        if not self.gathering:
            assert self.rank not in self.recorded, f"rank {self.rank} gathered twice"
            self.recorded[self.rank] = local.detach().clone()
            shape = list(local.shape)
            shape[-1] *= self.world_size
            return local.new_zeros(shape)
        assert torch.equal(local, self.recorded[self.rank]), (
            f"rank {self.rank}'s local operand changed between the two rounds"
        )
        return torch.cat([self.recorded[r] for r in range(self.world_size)], dim=-1)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def _head(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(VOCAB, HIDDEN, generator=gen) * HIDDEN**-0.5).to(torch.bfloat16)


def _rows(seed: int, batch: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, HIDDEN, generator=gen).to(torch.bfloat16)


def _pair(rows: torch.Tensor, shard: torch.Tensor) -> torch.Tensor:
    """The shard's ``[B, 2]`` fp32 ``(max, argmax)``: what the output tail kernel hands over."""
    return shard_pair(torch.nn.functional.linear(rows, shard).to(torch.float32))


def _every_rank(rows: torch.Tensor, head: torch.Tensor) -> tuple[list[torch.Tensor], _SimulatedTensorParallelGroup]:
    """``draft_token_ids`` on all ``TP`` ranks, each on its own shard's pair, in two rounds."""
    shards = head.reshape(TP, SHARD_ROWS, HIDDEN)
    group = _SimulatedTensorParallelGroup(TP)
    for group.gathering in (False, True):
        results = []
        for rank in range(TP):
            group.rank = rank
            results.append(draft_token_ids(_pair(rows, shards[rank]), shard_rows=SHARD_ROWS,
                                           vocab_size=VOCAB, group=group))
    return results, group


def _full_logits(rows: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """``[B, VOCAB]`` fp32: the head's logits in the head's own dtype, shard by shard.

    Computed the way the route under test computes each shard (``F.linear`` in the
    head's dtype, as ``vocab_parallel_logits`` computes the trunk's) and concatenated
    in rank order, so the reference is the replicated head's greedy token over the
    SAME logits every rank sees; what the test reads is the recovery of the global id,
    not the rounding of a 154880-wide bf16 GEMV."""
    shards = head.reshape(TP, SHARD_ROWS, HIDDEN)
    return torch.cat(
        [torch.nn.functional.linear(rows, shards[r]).to(torch.float32) for r in range(TP)],
        dim=-1,
    )


def _reference(rows: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """The replicated head's greedy token over the full vocabulary."""
    return _full_logits(rows, head).argmax(dim=-1)


@pytest.mark.parametrize("batch", [1, 3, 5])
def test_every_rank_recovers_the_global_greedy_token(batch: int) -> None:
    rows, head = _rows(7_101 + batch, batch), _head(7_002)
    want = _reference(rows, head)
    assert bool((want >= SHARD_ROWS).any()), (
        "the reference's tokens all sit in shard 0, where a local argmax is already "
        "global; this seed cannot tell the two apart"
    )
    results, group = _every_rank(rows, head)
    for rank, got in enumerate(results):
        assert got.dtype == torch.int32 and tuple(got.shape) == (batch,), (
            f"rank {rank}: draft ids must be [{batch}] int32, got {got.dtype} {tuple(got.shape)}"
        )
        assert torch.equal(got.to(torch.int64), want), (
            f"rank {rank} returned {got.tolist()}, the replicated head says {want.tolist()}"
        )
    assert sorted(group.gathers) == sorted(list(range(TP)) * 2), (
        f"each rank must gather exactly once per round: {group.gathers}"
    )


def test_the_local_shard_argmax_is_not_the_answer() -> None:
    """The control for H2: the per-shard argmax disagrees with the global id on most ranks."""
    rows, head = _rows(7_011, 4), _head(7_012)
    want = _reference(rows, head)
    shards = head.reshape(TP, SHARD_ROWS, HIDDEN)
    disagreeing = sum(
        int(not torch.equal(torch.nn.functional.linear(rows, shards[r]).argmax(-1).to(torch.int64), want))
        for r in range(TP)
    )
    assert disagreeing >= TP - 4, (
        f"only {disagreeing} of {TP} ranks' local argmax disagree with the global id; "
        f"the defect under test would then be invisible"
    )


def test_a_tie_across_shards_resolves_to_the_lowest_id_like_argmax() -> None:
    """Two identical head rows in two different shards: the full argmax picks the lower id."""
    rows, head = _rows(7_021, 2), _head(7_022).clone()
    low, high = 9 * SHARD_ROWS + 17, 41 * SHARD_ROWS + 5
    boost = rows[0].to(torch.float32) / rows[0].to(torch.float32).norm() * 8.0
    head[low] = boost.to(torch.bfloat16)
    head[high] = head[low]
    want = _reference(rows, head)
    assert int(want[0]) == low, f"the reference must pick the lower tied id {low}, got {int(want[0])}"
    full = _full_logits(rows, head)
    assert torch.equal(full[0, low], full[0, high]), "the two rows must tie exactly"
    results, _ = _every_rank(rows, head)
    for rank, got in enumerate(results):
        assert int(got[0]) == low, f"rank {rank} resolved the tie to {int(got[0])}, not {low}"
        assert int(got[1]) == int(want[1])


def test_a_whole_head_is_served_without_a_gather() -> None:
    rows, head = _rows(7_031, 3), _head(7_032)
    group = _SimulatedTensorParallelGroup(TP)
    group.rank = 0
    got = draft_token_ids(_pair(rows, head), shard_rows=VOCAB, vocab_size=VOCAB, group=group)
    assert torch.equal(got.to(torch.int64), _reference(rows, head))
    assert got.dtype == torch.int32
    assert group.gathers == [], "a whole head needs no collective"
    alone = draft_token_ids(_pair(rows, head), shard_rows=VOCAB, vocab_size=VOCAB, group=None)
    assert torch.equal(alone, got)


def test_a_head_that_is_neither_whole_nor_one_shard_is_refused() -> None:
    rows, head = _rows(7_041, 2), _head(7_042)
    group = _SimulatedTensorParallelGroup(TP)
    group.rank = 0
    pair = _pair(rows, head[:SHARD_ROWS])
    with pytest.raises(DraftTokenError, match="shard"):
        draft_token_ids(pair, shard_rows=SHARD_ROWS + 1, vocab_size=VOCAB, group=group)
    with pytest.raises(DraftTokenError, match="shard"):
        draft_token_ids(pair, shard_rows=SHARD_ROWS, vocab_size=VOCAB, group=None)
    assert issubclass(DraftTokenError, ValueError)


def test_an_operand_that_is_not_a_pair_is_refused_by_name() -> None:
    rows, head = _rows(7_045, 2), _head(7_046)
    group = _SimulatedTensorParallelGroup(TP)
    group.rank = 0
    logits = torch.nn.functional.linear(rows, head[:SHARD_ROWS]).to(torch.float32)
    with pytest.raises(DraftTokenError, match=r"\[B, 2\]"):
        draft_token_ids(logits, shard_rows=SHARD_ROWS, vocab_size=VOCAB, group=group)
    with pytest.raises(DraftTokenError, match="float32"):
        draft_token_ids(_pair(rows, head[:SHARD_ROWS]).to(torch.bfloat16), shard_rows=SHARD_ROWS,
                        vocab_size=VOCAB, group=group)


def test_the_gathered_operand_is_two_values_per_row_not_the_vocabulary() -> None:
    """H5: what crosses the wire is ``(max, argmax)`` per row, not the shard logits."""
    rows, head = _rows(7_051, 3), _head(7_052)
    _, group = _every_rank(rows, head)
    for rank, local in group.recorded.items():
        assert tuple(local.shape) == (3, 2), (
            f"rank {rank} gathered {tuple(local.shape)}; the route gathers [B, 2]"
        )
