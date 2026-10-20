# SPDX-License-Identifier: Apache-2.0
"""The DSA decode indexer's ``T``-row step under decode context parallelism (DCP).

A speculative verify step hands the indexer ``T = 1 + k`` rows per request at positions
``p_b .. p_b + T - 1``. Under DCP a group of ``CP`` ranks holds every sequence's pooled
keys by 128-token block (``vllm_neuron.utils.dcp_ownership``), so a step's rows can lie in
two ranks' blocks. Each row is the one-row DCP step at its own position. On rank ``rank``
row ``r = b * T + t`` takes:

* ``local[r] = owned_length(p_b + t, rank)``: the row's local row where the rank owns it,
  else the rows the rank holds before it (whole blocks);
* ``owner[r] = token_owner(p_b + t) == rank``;
* ``length[r] = owned_length(p_b + t + 1, rank)``: its causal length on the rank.

That is the per-row operand pair of the T-row DCP attention entry, built from the same
ownership calls. What this file pins, against the CP = 1 answer for the same step:

* **Operands** against a token-by-token count of the interleave: every row owned by
  exactly one rank, each rank's owned rows one run.
* **The score kernel's per-row position form** (``dsa_decode_scores_rows`` with
  ``position`` ``[B * T]``): fed ``p_b + t`` it equals today's ``[B]`` form bit for bit;
  fed a rank's ``local`` on the rank's own bank it gives the CP = 1 scores of the pools the
  rank owns at their local columns, ``BOUND_FILL`` past each row's owned length. A row
  another rank owns stands nothing in; a row the rank owns stands its pool in for itself
  and every later row of its request, those another rank owns included.
* **The selection chain** per rank -- the scores, ``dsa_decode_select_pools`` at the owned
  lengths, the DCP merge kernel over every rank's list, ``dsa_index_expand`` at the owned
  lengths: the union of the ranks' tokens, mapped back to global positions, is the CP = 1
  ``dsa_decode_select`` token set of every row, exactly. Steps that cross a block, a
  request that leaves ranks with an empty window, CP in {2, 4}, T in {1, 2, 4}.
* **The full-window edge**: a row another rank owns on a rank whose window is exactly full
  has ``local == candidates * pool_size``; the kernel takes it and stands nothing in.
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.functional.dsa.dsa_decode_case import decode_config
from vllm_neuron.functional.dsa import dcp_candidate_merge as DCM
from vllm_neuron.functional.dsa import decode_select as DS
from vllm_neuron.functional.dsa import decode_trow as TR
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL
from vllm_neuron.functional.dsa.index_expand import dsa_index_expand
from vllm_neuron.utils import dcp_ownership as OWN
from vllm_neuron.utils.neuron_utils import can_run_kernel

_CFG = decode_config()
POOL = int(_CFG.index_kpool)
#: Pools the selection keeps: ``index_topk`` tokens in whole pools.
K = int(_CFG.index_topk) // POOL
HEADS = int(_CFG.index_n_heads)
DIM = int(_CFG.index_head_dim)
#: ``hybrid_kv_block_size`` of the serve recipes: the DCP interleave unit.
BLOCK = 128
PPB = BLOCK // POOL
CPS = (2, 4)
ROWS = (1, 2, 4)
#: The selecting window: 1024 pools, more than ``K`` on one rank and over the group.
CTX = 4096
SLOTS_TOTAL = 6


def _starts(rows: int) -> list[int]:
    """Row 0's positions: two steps that cross a block at ``rows >= 2`` (one closing a pool
    on the first block, both inside the window), and a short request whose context lies in
    blocks 0 and 1, which leaves ranks 2 and 3 of a 4-rank group an empty window."""
    back = rows // 2 if rows > 1 else 1
    return [3968 - back, 2432 - back, 128 - back]


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"


# ---------------------------------------------------------------------------------------
# Operands
# ---------------------------------------------------------------------------------------


def _row_positions(start: torch.Tensor, rows: int) -> torch.Tensor:
    return (start.to(torch.int64)[:, None] + torch.arange(rows)[None, :]).reshape(-1)


def _rank_operands(start: torch.Tensor, rows: int, rank: int, cp: int):
    """``(local, owner, length)`` of every row on ``rank``: int64, bool, int64, ``[B * T]``."""
    at = _row_positions(start, rows)
    local = OWN.owned_length(at, rank, block_size=BLOCK, dcp_size=cp)
    owner = OWN.token_owner(at, block_size=BLOCK, dcp_size=cp) == rank
    length = OWN.owned_length(at + 1, rank, block_size=BLOCK, dcp_size=cp)
    return local, owner, length


def _count(position: int, rank: int, cp: int) -> int:
    """Tokens before ``position`` that ``rank`` holds, counted one by one."""
    return sum(1 for x in range(position) if (x // BLOCK) % cp == rank)


@pytest.mark.parametrize("cp", CPS)
@pytest.mark.parametrize("rows", ROWS)
def test_row_operands_count_the_interleave(cp, rows):
    start = torch.tensor(_starts(rows) + [255, 256, 511], dtype=torch.int64)
    at = _row_positions(start, rows).tolist()
    held = torch.zeros(len(at), dtype=torch.int64)
    for rank in range(cp):
        local, owner, length = _rank_operands(start, rows, rank, cp)
        assert local.tolist() == [_count(p, rank, cp) for p in at], (cp, rank)
        assert length.tolist() == [_count(p + 1, rank, cp) for p in at], (cp, rank)
        assert owner.tolist() == [(p // BLOCK) % cp == rank for p in at], (cp, rank)
        # The owner's local position is the row's local row; anyone else's is whole blocks.
        rows_on_owner = OWN.token_local_row(torch.tensor(at), block_size=BLOCK, dcp_size=cp)
        assert torch.equal(local[owner], rows_on_owner[owner])
        assert bool((local[~owner] % BLOCK == 0).all())
        # A row's length is its position plus one where held, its position elsewhere.
        assert torch.equal(length, local + owner.to(torch.int64))
        # Each request's owned rows are one run (a step of T <= 4 meets at most two
        # blocks, and two consecutive blocks have two owners).
        for b in range(int(start.shape[0])):
            flags = owner[b * rows:(b + 1) * rows].to(torch.int64).tolist()
            ones = [t for t, f in enumerate(flags) if f]
            assert ones == list(range(ones[0], ones[0] + len(ones))) if ones else True
        held += owner.to(torch.int64)
    assert held.tolist() == [1] * len(at)


# ---------------------------------------------------------------------------------------
# The score kernel's per-row position form
# ---------------------------------------------------------------------------------------


def _bf16(gen, *shape, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(torch.bfloat16)


def _global_case(starts, rows, candidates, *, seed):
    gen = torch.Generator().manual_seed(seed)
    batch = len(starts)
    return {
        "query": _bf16(gen, batch * rows, HEADS, DIM),
        "weights": torch.randn(batch * rows, HEADS, generator=gen) * HEADS ** -0.5,
        "bank": _bf16(gen, SLOTS_TOTAL, candidates + 1, DIM, scale=0.5),
        "slots": torch.randperm(SLOTS_TOTAL, generator=gen)[:batch].to(torch.int32),
        "start": torch.tensor(starts, dtype=torch.int64),
        "pooled": _bf16(gen, batch * rows, DIM, scale=0.5),
        "candidates": candidates,
        "rows": rows,
    }


def _scores(case, bank, position, candidates, monkeypatch, lnc=None):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    TR.reset_decode_trow_dispatch_counters()
    out = TR.dsa_decode_scores_rows(
        case["query"], case["weights"], bank, case["slots"], position, case["pooled"],
        candidates=candidates, pool_size=POOL)
    assert TR.decode_trow_dispatch_counters() == (1, 0)
    return out


@pytest.mark.parametrize("lnc", (None, "2"))
@pytest.mark.parametrize("batch", (1, 4))
@pytest.mark.parametrize("rows", (2, 4))
def test_the_row_form_fed_consecutive_positions_is_todays_form(rows, batch, lnc,
                                                               monkeypatch):
    starts = [4090, 2100, 2101, 2102][:batch]
    case = _global_case(starts, rows, 1024, seed=10 * rows + batch)
    today = _scores(case, case["bank"], case["start"], 1024, monkeypatch, lnc)
    per_row = _scores(case, case["bank"], _row_positions(case["start"], rows), 1024,
                      monkeypatch, lnc)
    assert torch.equal(per_row, today)
    oracle = TR.dsa_decode_scores_rows_torch_oracle(
        case["query"], case["weights"], case["bank"], case["slots"],
        _row_positions(case["start"], rows), case["pooled"], candidates=1024,
        pool_size=POOL)
    assert torch.allclose(oracle, today, rtol=1e-5, atol=1e-5)


def _local_candidates(ctx: int, cp: int) -> int:
    return OWN.local_context_capacity(ctx, block_size=BLOCK, dcp_size=cp) // POOL


def _global_pools(local_pools: torch.Tensor, rank: int, cp: int) -> torch.Tensor:
    return OWN.pool_global_index(local_pools, rank, block_size=BLOCK, dcp_size=cp,
                                 index_kpool=POOL)


def _local_bank(bank: torch.Tensor, rank: int, cp: int, local_c: int, *, seed: int):
    """``rank``'s store: local pool ``p`` holds global pool ``pool_global_index(p, rank)``
    where that is on the global store, noise past it and on the trash row."""
    gen = torch.Generator().manual_seed(seed)
    out = _bf16(gen, int(bank.shape[0]), local_c + 1, DIM, scale=0.5)
    gids = _global_pools(torch.arange(local_c), rank, cp)
    inside = gids < int(bank.shape[1]) - 1
    out[:, :local_c][:, inside] = bank[:, gids[inside]]
    return out


def _expected_local_scores(want, length, rank, cp, local_c):
    """The CP = 1 scores of ``rank``'s pools at their local columns, ``BOUND_FILL`` where a
    local pool is not complete at the row's owned length."""
    gids = _global_pools(torch.arange(local_c), rank, cp)
    complete = (torch.arange(local_c)[None, :] + 1) * POOL <= length[:, None]
    picked = want[:, gids.clamp(max=int(want.shape[1]) - 1)]
    return torch.where(complete, picked, torch.full_like(picked, BOUND_FILL))


@pytest.mark.parametrize("cp", CPS)
@pytest.mark.parametrize("rows", ROWS)
def test_each_ranks_row_scores_are_the_cp1_scores_of_its_pools(cp, rows, monkeypatch):
    case = _global_case(_starts(rows), rows, CTX // POOL, seed=100 + 10 * cp + rows)
    want = _scores(case, case["bank"], case["start"], CTX // POOL, monkeypatch)
    local_c = _local_candidates(CTX, cp)
    for rank in range(cp):
        local, _owner, length = _rank_operands(case["start"], rows, rank, cp)
        bank = _local_bank(case["bank"], rank, cp, local_c, seed=rank)
        position = local if rows > 1 else local.to(torch.int32)
        got = _scores(case, bank, position, local_c, monkeypatch)
        assert torch.equal(got, _expected_local_scores(want, length, rank, cp, local_c)), (
            cp, rank)


def test_a_full_window_takes_a_row_another_rank_owns(monkeypatch):
    """DCP 2, one request at 126 .. 129 on a 256-token context: rank 0's window is one page
    (32 pools), and its rows 2 and 3 (positions 128, 129, rank 1's) sit at local 128."""
    cp, rows = 2, 4
    case = _global_case([126], rows, 64, seed=5)
    want = _scores(case, case["bank"], case["start"], 64, monkeypatch)
    local_c = _local_candidates(256, cp)
    assert local_c == 32
    for rank in range(cp):
        local, _owner, length = _rank_operands(case["start"], rows, rank, cp)
        if rank == 0:
            assert local.tolist() == [126, 127, 128, 128]
        bank = _local_bank(case["bank"], rank, cp, local_c, seed=7 + rank)
        got = _scores(case, bank, local, local_c, monkeypatch)
        assert torch.equal(got, _expected_local_scores(want, length, rank, cp, local_c))


def _edge_case(form: str):
    """One request whose last row sits at the axis end, ``candidates * pool`` (64 pools):
    today's form at row 0's position, or the per-row form."""
    case = _global_case([253], 4, 64, seed=9)
    position = case["start"] if form == "request" else _row_positions(case["start"], 4)
    return case, position


@pytest.mark.parametrize("form", ("request", "row"))
def test_a_row_at_the_axis_end_is_taken_in_both_forms(form, monkeypatch):
    case, position = _edge_case(form)
    assert int(_row_positions(case["start"], 4)[-1]) == 64 * POOL
    got = _scores(case, case["bank"], position, 64, monkeypatch)
    oracle = TR.dsa_decode_scores_rows_torch_oracle(
        case["query"], case["weights"], case["bank"], case["slots"], position,
        case["pooled"], candidates=64, pool_size=POOL)
    assert torch.allclose(got, oracle, rtol=1e-5, atol=1e-5)
    # The row at the axis end keeps every pool (the one at 255 closed by row 2 included).
    assert bool((got[-1] != BOUND_FILL).all())


@pytest.mark.parametrize("form", ("request", "row"))
def test_a_row_past_the_axis_end_is_refused_by_name(form):
    case, position = _edge_case(form)
    position = position + 1
    for entry in (TR.dsa_decode_scores_rows, TR.dsa_decode_scores_rows_torch_oracle):
        with pytest.raises(TR.DecodeTrowError,
                           match=r"every row's position must lie in the window \[0, "
                                 r"candidates \* pool_size\] = \[0, 256\]"):
            entry(case["query"], case["weights"], case["bank"], case["slots"], position,
                  case["pooled"], candidates=64, pool_size=POOL)


def test_a_position_vector_of_another_length_is_refused_by_name():
    case = _global_case([126, 300], 4, 128, seed=10)
    with pytest.raises(TR.DecodeTrowError, match=r"position must be \[2\].*or \[8\]"):
        TR.dsa_decode_scores_rows(
            case["query"], case["weights"], case["bank"], case["slots"],
            torch.arange(4), case["pooled"], candidates=128, pool_size=POOL)


# ---------------------------------------------------------------------------------------
# The selection chain: local scores, local top-k, the merge, the expansion
# ---------------------------------------------------------------------------------------


def _token_sets(tokens: torch.Tensor) -> list[set[int]]:
    return [set(v for v in row.tolist() if v >= 0) for row in tokens]


@pytest.mark.parametrize("cp", CPS)
@pytest.mark.parametrize("rows", ROWS)
def test_the_merged_rank_selections_are_the_cp1_selection(cp, rows, monkeypatch):
    case = _global_case(_starts(rows), rows, CTX // POOL, seed=200 + 10 * cp + rows)
    bounded = _scores(case, case["bank"], case["start"], CTX // POOL, monkeypatch)
    seq_lens = _row_positions(case["start"], rows) + 1
    want = _token_sets(DS.dsa_decode_select(bounded, seq_lens, select_k=K, pool_size=POOL))
    local_c = _local_candidates(CTX, cp)
    lists, lengths = [], []
    for rank in range(cp):
        local, _owner, length = _rank_operands(case["start"], rows, rank, cp)
        bank = _local_bank(case["bank"], rank, cp, local_c, seed=300 + rank)
        position = local if rows > 1 else local.to(torch.int32)
        mine = _scores(case, bank, position, local_c, monkeypatch)
        _tokens, values, ids = DS.dsa_decode_select_pools(mine, length, select_k=K,
                                                          pool_size=POOL)
        lists.append((values, ids))
        lengths.append(length)
    values = torch.stack([v for v, _ in lists])
    ids = torch.stack([i for _, i in lists])
    union = [set() for _ in want]
    for rank in range(cp):
        DCM.reset_dcp_merge_select_dispatch_counters()
        merged = DCM.dsa_dcp_merge_select(values, ids, cp_rank=rank, pools_per_block=PPB)
        assert DCM.dcp_merge_select_dispatch_counters()[1] == 0
        tokens = dsa_index_expand(merged, lengths[rank].to(torch.int32), POOL)
        for r, row in enumerate(tokens.tolist()):
            mine = [t for t in row if t >= 0]
            # A rank names only rows it holds, each once over the group.
            assert all(t < int(lengths[rank][r]) for t in mine), (cp, rank, r)
            got = {int(g) for g in OWN.token_position(
                torch.tensor(mine, dtype=torch.int64), rank, block_size=BLOCK,
                dcp_size=cp).tolist()}
            assert not (got & union[r]), (cp, rank, r)
            union[r] |= got
    for r, (got, expected) in enumerate(zip(union, want)):
        assert got == expected, (cp, rows, r, sorted(got ^ expected)[:8])
    # The empty windows: at CP = 4 the short request (blocks 0 and 1) leaves ranks 2 and 3
    # nothing, and every row of it still selects its whole context.
    short = slice(2 * rows, 3 * rows)
    if cp == 4:
        assert all(int(n) == 0 for rank in (2, 3) for n in lengths[rank][short])
    assert [len(s) for s in union[short]] == [int(n) for n in seq_lens[short]]
