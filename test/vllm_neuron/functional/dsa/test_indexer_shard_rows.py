# SPDX-License-Identifier: Apache-2.0
"""The row cut of the sharded DSA prefill selection (``shard_rows.dsa_take_rank_rows``).

The NKI kernel builds a rank's row index on device and gathers each source's rows by
indirect DMA, up to ``SOURCES_PER_LAUNCH`` sources per launch. On the CPU NKI simulator it
must return, bit for bit, what ``index_select`` on the reference index (``rank_row_index``)
returns, for the selection's three operands in one launch, for exact plans and padded ones,
and for the first, a middle and the last rank. Every such call must take the NKI route; the
torch route serves only what the kernel does not take.
"""

from __future__ import annotations

import pytest
import torch

#: ``(T, d)`` plans: production (1024 rows over 64 or 8 ranks), a 2048-row chunk, padded
#: plans (``T % d != 0``) and a plan wider than one 128-partition tile (300 over 2).
PLANS = ((1024, 64), (1024, 8), (2048, 8), (1029, 64), (130, 64), (300, 2))


def _operand_layout():
    """The selection's three operands at the production dials: query ``[T, heads, dim]``
    bf16, gate ``[T, heads]`` fp32, lengths ``[T]`` int32, as ``(dtype, trailing dims)``."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    cfg = Glm5NextTextConfig()
    heads, dim = int(cfg.index_n_heads), int(cfg.index_head_dim)
    return ((torch.bfloat16, (heads, dim)), (torch.float32, (heads,)), (torch.int32, ()))


def _source(tokens: int, dtype: torch.dtype, tail: tuple[int, ...], seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if dtype == torch.int32:
        # Every bit pattern, so a cut that converted through a float would show.
        info = torch.iinfo(torch.int32)
        return torch.randint(info.min, info.max, (tokens, *tail), generator=gen, dtype=dtype)
    return (torch.randn(tokens, *tail, generator=gen) * 100).to(dtype)


def _rank(rank: int) -> torch.Tensor:
    return torch.tensor([rank], dtype=torch.int32)


def _assert_cut(got, sources, tokens, rows, rank):
    from vllm_neuron.functional.dsa import shard_rows as SR

    index = SR.rank_row_index(tokens, rows, rank, torch.device("cpu"))
    assert len(got) == len(sources)
    for cut, src in zip(got, sources):
        assert cut.dtype == src.dtype and torch.equal(cut, src.index_select(0, index)), (
            rank, rows, src.dtype)


@pytest.mark.parametrize("tokens,degree", PLANS, ids=[f"T{t}-d{d}" for t, d in PLANS])
def test_one_launch_takes_each_ranks_rows_of_the_three_operands_bit_for_bit(tokens, degree):
    from vllm_neuron.functional.dsa import shard_rows as SR
    from vllm_neuron.functional.dsa.indexer_shard import row_shard

    rows = row_shard(tokens, degree).rows
    sources = tuple(_source(tokens, dtype, tail, seed=tokens * 31 + degree + j)
                    for j, (dtype, tail) in enumerate(_operand_layout()))
    for rank in sorted({0, degree // 2, degree - 1}):
        SR.reset_shard_rows_dispatch_counters()
        got = SR.dsa_take_rank_rows(sources, _rank(rank), rows)
        assert SR.shard_rows_dispatch_counters() == (1, 0), "not one NKI launch"
        _assert_cut(got, sources, tokens, rows, rank)


def test_more_sources_than_one_launch_takes_split_over_launches():
    from vllm_neuron.functional.dsa import shard_rows as SR

    count = SR.SOURCES_PER_LAUNCH + 1
    tokens, rows, rank = 200, 70, 2
    sources = tuple(_source(tokens, torch.float32, (j + 1,), seed=j) for j in range(count))
    SR.reset_shard_rows_dispatch_counters()
    got = SR.dsa_take_rank_rows(sources, _rank(rank), rows)
    launches = -(-count // SR.SOURCES_PER_LAUNCH)
    assert SR.shard_rows_dispatch_counters() == (launches, 0)
    _assert_cut(got, sources, tokens, rows, rank)


def test_pad_rows_repeat_the_last_real_row():
    from vllm_neuron.functional.dsa import shard_rows as SR

    src = torch.arange(10 * 4, dtype=torch.float32).reshape(10, 4)
    (got,) = SR.dsa_take_rank_rows((src,), _rank(3), 3)
    assert torch.equal(got, src[[9, 9, 9]])


def test_the_kernel_reads_the_sources_in_place(monkeypatch):
    """The ``[T, F]`` rows the kernel reads are views of the sources, not copies."""
    from vllm_neuron.functional.dsa import shard_rows as SR

    seen = []
    real = SR.wrap_nki

    def recording(kernel):
        caller = real(kernel)

        def call(*args):
            seen.extend(a for a in args if torch.is_tensor(a))
            return caller(*args)

        return call

    monkeypatch.setattr(SR, "wrap_nki", recording)
    sources = tuple(_source(64, dtype, tail, seed=j)
                    for j, (dtype, tail) in enumerate(_operand_layout()))
    SR.dsa_take_rank_rows(sources, _rank(1), 16)
    read = seen[1:]  # the rank operand comes first
    assert len(read) == len(sources)
    for view, src in zip(read, sources):
        assert view.untyped_storage().data_ptr() == src.untyped_storage().data_ptr()


def test_the_torch_route_serves_what_the_kernel_does_not_take():
    from vllm_neuron.functional.dsa import shard_rows as SR

    wide = torch.arange(10 * 4, dtype=torch.int64).reshape(10, 4)
    plain = wide.to(torch.float32)
    # An 8-byte dtype beside a supported one, a python-int rank, an int64 rank operand.
    for sources, rank in (((plain, wide), _rank(1)), ((plain,), 1),
                          ((plain,), torch.tensor([1]))):
        SR.reset_shard_rows_dispatch_counters()
        got = SR.dsa_take_rank_rows(sources, rank, 4)
        assert SR.shard_rows_dispatch_counters() == (0, 1)
        for cut, src in zip(got, sources):
            assert cut.dtype == src.dtype and torch.equal(cut, src[4:8])


@pytest.mark.parametrize("rank", [torch.tensor([1, 2], dtype=torch.int32),
                                  torch.tensor([1.0])])
def test_a_rank_operand_that_is_not_one_integer_is_refused(rank):
    from vllm_neuron.functional.dsa import shard_rows as SR

    with pytest.raises(SR.ShardRowsError):
        SR.dsa_take_rank_rows((torch.zeros(4, 2),), rank, 2)


@pytest.mark.parametrize("sources,rows", [
    ((torch.zeros(4, 2),), 0),
    ((torch.zeros(()),), 1),
    ((), 1),
    ((torch.zeros(4, 2), torch.zeros(5, 2)), 1),
], ids=["no-rows", "no-row-axis", "no-source", "two-row-counts"])
def test_a_cut_without_a_plan_is_refused(sources, rows):
    from vllm_neuron.functional.dsa import shard_rows as SR

    with pytest.raises(SR.ShardRowsError):
        SR.dsa_take_rank_rows(sources, _rank(0), rows)
