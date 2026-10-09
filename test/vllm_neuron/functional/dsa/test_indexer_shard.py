# SPDX-License-Identifier: Apache-2.0
"""Query-sharded DSA prefill selection: the same pool ids per row as the replicated path.

The as-built prefill runs the indexer's score, causal bound, top-k and ordering on every
query row of the chunk on every rank. Sharded, rank ``r`` of ``d`` runs that chain on rows
``[r * R, (r + 1) * R)`` only, with ``R = ceil(T / d)``, and an all-gather of the
``[R, select_k]`` pool ids rebuilds the whole ``[T, select_k]`` on every rank.

What this file pins:

1. The plan arithmetic, the row index a rank takes and the gather's row order and dtype.
2. For ``T in {128, 1024, 2048} x C in {512, 2048, 16384} x d in {1, 8, 64}``, with the
   ``d`` ranks run as separate calls on the CPU NKI simulator, every row selects the same
   pool-id SET as the replicated path, the sentinels sit in the same trailing columns in
   the same number, and the expanded token indices agree per row. Ties are counted, not
   assumed away (see :func:`test_the_tie_census_is_recorded`).
3. The kill switch (``envs.VLLM_NEURON_DSA_INDEXER_SHARD``) is on unless set to ``0``.

The model-level wiring (the rank operand, the gate, the gather at the indexer's forward)
is pinned by ``test/vllm_neuron/model/glm5_next/test_indexer_shard_forward.py``.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import pytest
import torch

from test.vllm_neuron.functional.dsa import indexer_shard_case as case

TOKENS = (128, 1024, 2048)
CANDS = (512, 2048, 16384)
DEGREES = (1, 8, 64)
SEED = 20261007
CASES = [(t, c, d) for t in TOKENS for c in CANDS for d in DEGREES]
SHARD_ENV = "VLLM_NEURON_DSA_INDEXER_SHARD"


# ---------------------------------------------------------------------------------------
# 1. Plan, row index, gather
# ---------------------------------------------------------------------------------------


def test_the_plan_gives_every_rank_the_same_row_count_and_covers_the_chunk():
    from vllm_neuron.functional.dsa.indexer_shard import row_shard

    for tokens, degree, rows in ((1024, 64, 16), (1024, 8, 128), (2048, 64, 32),
                                 (128, 64, 2), (1000, 64, 16), (5, 8, 1), (7, 1, 7)):
        shard = row_shard(tokens, degree)
        assert (shard.tokens, shard.degree, shard.rows) == (tokens, degree, rows)
        assert shard.rows * shard.degree >= tokens > (shard.rows - 1) * shard.degree


@pytest.mark.parametrize("tokens,degree", [(0, 8), (8, 0), (-1, 1)])
def test_the_plan_refuses_an_empty_chunk_or_degree(tokens, degree):
    from vllm_neuron.functional.dsa.indexer_shard import IndexerShardError, row_shard

    with pytest.raises(IndexerShardError):
        row_shard(tokens, degree)


def test_a_rank_takes_its_own_rows_and_padding_repeats_the_last_real_row():
    from vllm_neuron.functional.dsa.indexer_shard import local_row_index, row_shard

    shard = row_shard(10, 4)  # R = 3, the last rank holds row 9 and two pad rows
    got = [local_row_index(shard, r, torch.device("cpu")).tolist() for r in range(4)]
    assert got == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 9, 9]]
    # The rank arrives as a device operand on the traced path: one graph for every rank.
    as_tensor = [
        local_row_index(shard, torch.tensor([r], dtype=torch.int32), torch.device("cpu"))
        for r in range(4)
    ]
    assert [t.tolist() for t in as_tensor] == got
    assert all(t.dtype == torch.int64 for t in as_tensor)


@pytest.mark.parametrize("rank", [torch.tensor([1, 2], dtype=torch.int32),
                                  torch.tensor([1.0])])
def test_a_rank_operand_that_is_not_one_integer_is_refused(rank):
    from vllm_neuron.functional.dsa.indexer_shard import local_row_index, row_shard
    from vllm_neuron.functional.dsa.shard_rows import ShardRowsError

    with pytest.raises(ShardRowsError):
        local_row_index(row_shard(10, 4), rank, torch.device("cpu"))


def test_a_rank_selects_its_own_rows_of_every_operand():
    from vllm_neuron.functional.dsa.indexer_shard import (
        IndexerShardError,
        row_shard,
        select_local_rows,
    )

    shard = row_shard(10, 4)
    query = torch.arange(10 * 3, dtype=torch.float32).reshape(10, 3)
    weights = torch.arange(10 * 2, dtype=torch.float32).reshape(10, 2) + 100
    seq_lens = torch.arange(10, dtype=torch.int32) + 1000
    seen = []

    def select(q, w, s):
        seen.append((q, w, s))
        return s.reshape(-1, 1)

    out = select_local_rows(select, query, weights, seq_lens, shard, torch.tensor([3]))
    q, w, s = seen[0]
    assert torch.equal(q, query[[9, 9, 9]]) and torch.equal(w, weights[[9, 9, 9]])
    assert torch.equal(out.flatten(), seq_lens[[9, 9, 9]])
    with pytest.raises(IndexerShardError):  # an operand cut for another chunk
        select_local_rows(select, query[:9], weights, seq_lens, shard, 0)


class _ConcatGroup:
    """A ``world``-rank group whose all-gather concatenates prepared per-rank operands.

    It checks that the caller hands it this rank's own operand, already in the dtype the
    gather moves, and returns what a real all-gather on ``dim=0`` would.
    """

    def __init__(self, per_rank: list[torch.Tensor], rank: int):
        self.world_size = len(per_rank)
        self.rank = rank
        self.per_rank = per_rank
        self.calls: list[tuple[torch.dtype, int]] = []

    def all_gather(self, local: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self.calls.append((local.dtype, dim))
        assert dim == 0, f"the selection gathers query rows, dim 0; got dim={dim}"
        mine = self.per_rank[self.rank].to(local.dtype)
        assert torch.equal(local, mine), f"rank {self.rank} gathered rows it does not own"
        return torch.cat([p.to(local.dtype) for p in self.per_rank], dim=0)


def test_the_gather_restores_row_order_trims_padding_and_keeps_int32():
    from vllm_neuron.functional.dsa.indexer_shard import gather_rows, row_shard

    shard = row_shard(10, 4)
    full = torch.arange(10 * 6, dtype=torch.int32).reshape(10, 6) - 7  # holds -1 and others
    padded = torch.cat([full, full[-1:].expand(2, 6)])
    per_rank = list(padded.split(shard.rows))
    for rank in range(4):
        group = _ConcatGroup(per_rank, rank)
        out = gather_rows(per_rank[rank], group, shard)
        assert out.dtype == torch.int32 and torch.equal(out, full)
        # One collective per call, of the selector's own int32 ids: no cast on either side.
        assert group.calls == [(torch.int32, 0)]


def test_an_exact_plan_returns_the_gathered_block_itself():
    from vllm_neuron.functional.dsa.indexer_shard import gather_rows, row_shard

    shard = row_shard(8, 4)
    per_rank = list((torch.arange(8 * 3, dtype=torch.int32).reshape(8, 3) - 1).split(shard.rows))
    group = _ConcatGroup(per_rank, 1)
    gathered = torch.cat(per_rank)
    group.all_gather = lambda local, dim=-1: gathered
    # ``d * R == T``: no slice, no copy -- the collective's output is the chunk's ids.
    assert gather_rows(per_rank[1], group, shard) is gathered


def test_the_gather_refuses_ids_of_another_dtype():
    from vllm_neuron.functional.dsa.indexer_shard import IndexerShardError, gather_rows, row_shard

    shard = row_shard(4, 2)
    for dtype in (torch.float32, torch.int64):
        local = torch.zeros(2, 3, dtype=dtype)
        with pytest.raises(IndexerShardError):
            gather_rows(local, _ConcatGroup([local, local], 0), shard)


def test_the_gather_refuses_a_block_of_the_wrong_height():
    from vllm_neuron.functional.dsa.indexer_shard import IndexerShardError, gather_rows, row_shard

    shard = row_shard(4, 2)
    wrong = torch.zeros(3, 3, dtype=torch.int32)
    with pytest.raises(IndexerShardError):
        gather_rows(wrong, _ConcatGroup([wrong, wrong], 0), shard)


@pytest.mark.parametrize("value,enabled", [(None, True), ("1", True), ("0", False)])
def test_the_kill_switch_is_on_unless_zero(value, enabled, monkeypatch):
    from vllm_neuron import envs
    from vllm_neuron.functional.dsa.indexer_shard import indexer_shard_enabled

    assert SHARD_ENV in dir(envs)  # registered, so it is read through envs only
    if value is None:
        monkeypatch.delenv(SHARD_ENV, raising=False)
    else:
        monkeypatch.setenv(SHARD_ENV, value)
    assert indexer_shard_enabled() is enabled


def test_the_kill_switch_refuses_a_value_that_is_not_a_number(monkeypatch):
    """The ``envs`` boolean convention: ``0`` / ``1``; anything else is an error."""
    from vllm_neuron.functional.dsa.indexer_shard import indexer_shard_enabled

    monkeypatch.setenv(SHARD_ENV, "yes")
    with pytest.raises(ValueError):
        indexer_shard_enabled()


@pytest.mark.parametrize("tokens,world,switch,want", [
    (1024, 64, "1", 64), (2048, 64, "1", 64), (1024, 16, "1", 16), (1024, 64, "0", 1),
    (1024, 1, "1", 1), (1, 64, "1", 1),
])
def test_the_degree_is_the_world_above_one_row_tile_and_one_otherwise(
    tokens, world, switch, want, monkeypatch
):
    from vllm_neuron.functional.dsa.indexer_shard import ROW_TILE, shard_degree

    monkeypatch.setenv(SHARD_ENV, switch)
    assert shard_degree(tokens, world) == want
    # One row tile stays replicated; one row more shards.
    assert shard_degree(ROW_TILE, world) == 1
    assert shard_degree(ROW_TILE + 1, world) == (world if switch == "1" else 1)


def test_the_score_kernel_is_row_independent_bit_for_bit():
    """A row slice scores exactly what the same rows score inside the whole chunk."""
    from vllm_neuron.functional.dsa import score_gemm

    query, keys, weights, _ = case.case_operands(384, 2048, SEED)
    indexer = case.make_indexer(2048)
    slices = ((0, 16), (16, 32), (200, 328), (370, 384))
    score_gemm.reset_score_gemm_dispatch_counters()
    whole = indexer.score_pools(query, keys, weights)
    for lo, hi in slices:
        part = indexer.score_pools(query[lo:hi], keys, weights[lo:hi])
        assert torch.equal(part, whole[lo:hi]), (lo, hi)
    nki, fallback = score_gemm.score_gemm_dispatch_counters()
    # One call for the whole chunk and one per slice, all on the NKI route.
    assert (nki, fallback) == (1 + len(slices), 0), "a call left the NKI route"


# ---------------------------------------------------------------------------------------
# 2. The equality matrix
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matrix():
    """Run every replicated case and every rank of every sharded case once, in parallel."""
    if os.environ.get("NKI_SIMULATOR") != "1":
        pytest.skip("the NKI simulator is off, so the kernels under test would not run")
    saved = {k: os.environ.get(k) for k in case.worker_environment()}
    os.environ.update(case.worker_environment())
    workers = max(1, min(48, (os.cpu_count() or 2) // 2))
    results = {"replicated": {}, "ranks": {}}
    try:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {}
            for t in TOKENS:
                for c in CANDS:
                    futures[("replicated", t, c)] = pool.submit(case.replicated_task, t, c, SEED)
                    for d in DEGREES:
                        for r in range(d):
                            futures[("rank", t, c, d, r)] = pool.submit(
                                case.rank_task, t, c, d, r, SEED
                            )
            for key, future in futures.items():
                value = future.result(timeout=3600)
                if key[0] == "replicated":
                    results["replicated"][key[1:]] = value
                else:
                    results["ranks"][key[1:]] = value
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return results


def _gathered(matrix, tokens, cands, degree):
    """The ``d`` ranks' locals through the production gather, as every rank sees them."""
    from vllm_neuron.functional.dsa.indexer_shard import gather_rows, row_shard

    shard = row_shard(tokens, degree)
    per_rank = [matrix["ranks"][(tokens, cands, degree, r)][0] for r in range(degree)]
    for local in per_rank:
        assert local.dtype == torch.int32 and tuple(local.shape) == (shard.rows, local.shape[1])
    outs = [gather_rows(per_rank[r], _ConcatGroup(per_rank, r), shard) for r in range(degree)]
    for other in outs[1:]:
        assert torch.equal(other, outs[0]), "every rank must see the same gathered selection"
    return outs[0]


def _slot_scores(bounded: torch.Tensor, pool_ids: torch.Tensor) -> torch.Tensor:
    """The bounded score each slot selected, ``-inf`` at a ``-1`` slot."""
    picked = bounded.gather(1, pool_ids.clamp_min(0).to(torch.int64))
    return torch.where(pool_ids >= 0, picked, torch.full_like(picked, float("-inf")))


def _row_sets(pool_ids: torch.Tensor) -> list[frozenset[int]]:
    return [frozenset(v for v in row.tolist() if v >= 0) for row in pool_ids]


@pytest.mark.parametrize("tokens,cands,degree", CASES,
                         ids=[f"T{t}-C{c}-d{d}" for t, c, d in CASES])
def test_the_sharded_rows_select_the_same_pool_sets(matrix, tokens, cands, degree):
    want, bounded, counters = matrix["replicated"][(tokens, cands)]
    got = _gathered(matrix, tokens, cands, degree)
    indexer = case.make_indexer(cands)
    k = indexer.select_k()
    assert tuple(got.shape) == tuple(want.shape) == (tokens, k)
    # The as-built reference really ran the kernels it stands for.
    for family in ("score_gemm", "causal_bound", "causal_sentinel", "topk_select",
                   "sentinel_order"):
        assert counters[family] == (1, 0), (family, counters[family])
    for r in range(degree):
        rank_counters = matrix["ranks"][(tokens, cands, degree, r)][1]
        for family, value in rank_counters.items():
            assert value == (1, 0), (f"rank {r}", family, value)
        # The row cut: one NKI launch takes the three operands (query, weights, seq_lens).
        assert matrix["ranks"][(tokens, cands, degree, r)][2] == (1, 0), f"rank {r}"

    bad = [i for i, (a, b) in enumerate(zip(_row_sets(want), _row_sets(got))) if a != b]
    assert not bad, f"{len(bad)} row(s) select a different pool set, first {bad[:5]}"
    # Where the two put an id in a different slot, the scores there are equal: any slot
    # difference is a tie-break inside the selector, never a different ranking.
    assert torch.equal(_slot_scores(bounded, got), _slot_scores(bounded, want))


@pytest.mark.parametrize("tokens,cands,degree", CASES,
                         ids=[f"T{t}-C{c}-d{d}" for t, c, d in CASES])
def test_the_sentinels_and_the_expansion_are_unchanged(matrix, tokens, cands, degree):
    want, _bounded, _ = matrix["replicated"][(tokens, cands)]
    got = _gathered(matrix, tokens, cands, degree)
    _query, _keys, _weights, seq_lens = case.case_operands(tokens, cands, SEED)
    indexer = case.make_indexer(cands)
    k, pool = indexer.select_k(), indexer.index_kpool

    # Sentinels: as many per row as the row's own length leaves pools unfilled, and all of
    # them trailing (sentinel_order's canonical form), on both paths.
    expect = [max(0, k - min(int(s) // pool, cands)) for s in seq_lens]
    for name, ids in (("replicated", want), ("sharded", got)):
        negatives = (ids < 0).sum(dim=1).tolist()
        assert negatives == expect, (name, [i for i, (a, b) in enumerate(zip(negatives, expect))
                                            if a != b][:5])
        reals = k - torch.tensor(expect)
        cols = torch.arange(k)[None, :]
        assert torch.equal(ids < 0, cols >= reals[:, None]), f"{name}: a sentinel is not trailing"
    assert torch.equal(got < 0, want < 0)

    # The expansion runs on the gathered rows with every row's own length, so the tail
    # columns and the -1 columns land where the replicated path put them.
    exp_want = indexer.expand_indices(want, seq_lens)
    exp_got = indexer.expand_indices(got, seq_lens)
    assert exp_got.shape == exp_want.shape and exp_got.dtype == torch.int32
    assert torch.equal((exp_got < 0).sum(dim=1), (exp_want < 0).sum(dim=1))
    assert _row_sets(exp_got) == _row_sets(exp_want)


@pytest.mark.parametrize("tokens,cands", [(t, c) for t in TOKENS for c in CANDS],
                         ids=[f"T{t}-C{c}" for t in TOKENS for c in CANDS])
def test_the_replicated_selection_is_the_true_top_k(matrix, tokens, cands):
    """Both paths agree, and what they agree on is the torch top-k of the kernel's scores."""
    want, bounded, _ = matrix["replicated"][(tokens, cands)]
    from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL_MARK

    k = int(want.shape[1])
    values, indices = torch.topk(bounded, k, dim=-1)
    oracle = torch.where(values > BOUND_FILL_MARK, indices.to(torch.int32), -1)
    tied = _tied_rows(bounded, k)
    differ = [i for i, (a, b) in enumerate(zip(_row_sets(want), _row_sets(oracle)))
              if a != b and i not in tied]
    assert not differ, f"rows {differ[:5]} are not the top-k of their own bounded scores"


def _tied_rows(bounded: torch.Tensor, k: int) -> set[int]:
    """Rows whose k-th and (k+1)-th real scores are equal: the set there is a tie-break."""
    from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL_MARK

    top = torch.topk(bounded, min(k + 1, int(bounded.shape[1])), dim=-1).values
    if int(top.shape[1]) <= k:
        return set()
    real = (top[:, k - 1] > BOUND_FILL_MARK) & (top[:, k] > BOUND_FILL_MARK)
    return set(torch.nonzero(real & (top[:, k - 1] == top[:, k])).flatten().tolist())


def test_the_tie_census_is_recorded(matrix, record_property):
    """Where the set could legitimately differ: a real-score tie at the k-th slot.

    The sentinel's own ties (every bound-filled column holds one value) cannot change a
    set, because each becomes ``-1``; only a tie between two visible scores at the
    boundary can. This counts such rows per case and records how often the two paths'
    rows also match position for position, which is the stronger, unpromised property.
    """
    census = {}
    for (tokens, cands), (want, bounded, _) in matrix["replicated"].items():
        tied = _tied_rows(bounded, int(want.shape[1]))
        for degree in DEGREES:
            got = _gathered(matrix, tokens, cands, degree)
            exact = int((got == want).all(dim=1).sum())
            census[f"T{tokens}-C{cands}-d{degree}"] = {
                "rows": tokens, "tied_rows": len(tied), "positional_equal_rows": exact}
    record_property("tie_census", census)
    assert all(v["tied_rows"] == 0 for v in census.values()), census
