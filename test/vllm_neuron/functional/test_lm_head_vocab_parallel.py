# SPDX-License-Identifier: Apache-2.0
"""Vocab-parallel lm_head: each rank projects its 1/64 of the vocabulary and an
all-gather returns the full ``[B, 154880]`` logits on every rank.

The gathered logits must equal the replicated head's logits: every logit is
the same dot product over the same 4096 elements whether it is computed in the
full GEMM or in its rank's shard, and the gather only moves values. With
integer-valued operands every partial sum is exact in fp32, so the comparison is
bitwise and independent of the GEMM's summation order. With random operands the
CPU GEMM may block the full and the shard GEMM differently, so the bound there is
one bf16 ulp of the logit.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.lm_head import (
    VocabParallelHeadError,
    vocab_parallel_logits,
    vocab_shard_rows,
    vocab_shard_width,
)

VOCAB = 154880
HIDDEN = 4096
TP = 64


class _FakeTensorParallelGroup:
    """``GroupCoordinator.all_gather`` across ``world`` simulated ranks.

    Every rank's shard is known up front, so the gather a rank performs returns
    the rank-ordered concatenation of all ranks' local logits, which is what the
    device all-gather over the TP group returns.
    """

    def __init__(self, rows, shards):
        self.world_size = len(shards)
        self._locals = [torch.nn.functional.linear(rows, s) for s in shards]
        self.calls = 0

    def all_gather(self, local, dim=-1):
        self.calls += 1
        assert any(torch.equal(local, mine) for mine in self._locals)
        return torch.cat(self._locals, dim=dim)


@pytest.fixture(scope="module")
def head():
    g = torch.Generator().manual_seed(17)
    return torch.randint(-3, 4, (VOCAB, HIDDEN), generator=g).to(torch.bfloat16)


@pytest.fixture(scope="module")
def random_head():
    g = torch.Generator().manual_seed(18)
    return (torch.randn((VOCAB, HIDDEN), generator=g) * 0.02).to(torch.bfloat16)


def _rows(tokens, integer=True):
    g = torch.Generator().manual_seed(tokens)
    if integer:
        return torch.randint(-3, 4, (tokens, HIDDEN), generator=g).to(torch.bfloat16)
    return torch.randn((tokens, HIDDEN), generator=g).to(torch.bfloat16)


def test_shard_rows_divide_the_real_vocab():
    assert vocab_shard_rows(VOCAB, TP) == 2420
    assert vocab_shard_rows(VOCAB, 1) == VOCAB
    with pytest.raises(VocabParallelHeadError):
        vocab_shard_rows(VOCAB + 1, TP)


@pytest.mark.parametrize("tokens", [1, 4])
def test_gathered_logits_equal_the_replicated_head(head, tokens):
    rows = _rows(tokens)
    replicated = vocab_parallel_logits(rows, head, vocab_size=VOCAB, group=None)
    assert torch.equal(replicated, torch.nn.functional.linear(rows, head))

    width = vocab_shard_rows(VOCAB, TP)
    shards = [head[r * width:(r + 1) * width] for r in range(TP)]
    group = _FakeTensorParallelGroup(rows, shards)
    for rank in (0, 1, 37, TP - 1):
        gathered = vocab_parallel_logits(
            rows, shards[rank], vocab_size=VOCAB, group=group
        )
        # Full logits on every rank, same shape and dtype the sampler saw.
        assert gathered.shape == (tokens, VOCAB)
        assert gathered.dtype == replicated.dtype == torch.bfloat16
        torch.testing.assert_close(gathered, replicated, rtol=0, atol=0)
    assert group.calls == 4


@pytest.mark.parametrize("tokens", [1, 4])
def test_gathered_logits_track_the_replicated_head_on_random_data(
    random_head, tokens
):
    rows = _rows(tokens, integer=False)
    replicated = torch.nn.functional.linear(rows, random_head)
    width = vocab_shard_rows(VOCAB, TP)
    shards = [random_head[r * width:(r + 1) * width] for r in range(TP)]
    group = _FakeTensorParallelGroup(rows, shards)
    gathered = vocab_parallel_logits(rows, shards[3], vocab_size=VOCAB, group=group)
    # The gather itself is exact: rank r's columns are rank r's local logits.
    for rank in range(TP):
        assert torch.equal(
            gathered[:, rank * width:(rank + 1) * width],
            torch.nn.functional.linear(rows, shards[rank]),
        )
    ulp = torch.finfo(torch.bfloat16).eps * replicated.float().abs().clamp_min(1e-3)
    assert ((gathered.float() - replicated.float()).abs() <= ulp).all()


def test_replicated_head_does_not_gather(head):
    rows = torch.ones((1, HIDDEN), dtype=torch.bfloat16)

    class _NoGather:
        world_size = TP

        def all_gather(self, *_):
            raise AssertionError("a replicated head must not gather")

    vocab_parallel_logits(rows, head, vocab_size=VOCAB, group=_NoGather())


def test_a_head_that_is_neither_whole_nor_one_shard_is_refused(head):
    rows = torch.ones((1, HIDDEN), dtype=torch.bfloat16)
    with pytest.raises(VocabParallelHeadError):
        vocab_parallel_logits(rows, head[:1000], vocab_size=VOCAB, group=None)

    class _Group:
        world_size = TP

    with pytest.raises(VocabParallelHeadError):
        vocab_parallel_logits(rows, head[:1000], vocab_size=VOCAB, group=_Group())


def test_root_lm_head_geometry_is_a_vocab_shard():
    # The loader shards lm_head_weight on dim 0, 2420 rows per rank at TP=64,
    # and slices rank r's rows [r * 2420, (r + 1) * 2420) -- the order the
    # all-gather concatenates in.
    from vllm_neuron.model.glm5_next import model_fp8
    from vllm_neuron.utils.weight_loader import sharding_weight_loader

    class Glm5NextForConditionalGeneration(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_config = type("T", (), {"vocab_size": VOCAB})()

    root = Glm5NextForConditionalGeneration()
    geometry = model_fp8._shard_geometry_for(root, "lm_head_weight", TP)
    assert (geometry.shard_dim, geometry.shard_size, geometry.num_shards) == (
        0, 2420, TP,
    )
    assert model_fp8._shard_geometry_for(root, "lm_head_weight", 1) is None
    assert vocab_shard_width(root, TP) == 2420

    full = torch.arange(VOCAB * 2, dtype=torch.float32).reshape(VOCAB, 2)
    loader = sharding_weight_loader(
        shard_dim=geometry.shard_dim,
        shard_size=geometry.shard_size,
        num_shards=geometry.num_shards,
    )
    class _CheckpointSlice:  # the safetensors slice interface the loader reads
        def get_shape(self):
            return list(full.shape)

        def __getitem__(self, index):
            return full[index]

    for rank in (0, 5, TP - 1):
        got = loader.transform([_CheckpointSlice()], rank)
        assert torch.equal(got, full[rank * 2420:(rank + 1) * 2420])
