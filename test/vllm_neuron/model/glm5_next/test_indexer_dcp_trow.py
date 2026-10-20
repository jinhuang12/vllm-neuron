# SPDX-License-Identifier: Apache-2.0
"""``Glm5NextDSAIndexer.forward_requests`` with ``T`` rows per request under DCP.

A speculative verify step hands the indexer ``T = 1 + k`` rows per request, and a
speculative config allocates its ring deeper than ``index_kpool``
(``decode_trow.indexer_ring_depth``). Under decode context parallelism every rank of a
``CP``-rank group runs the same step on its own pooled-key store (the 128-token blocks it
owns, ``vllm_neuron.utils.dcp_ownership``) and the replicated ring, and the group merges
the ranks' local selections (``dcp_candidate_merge``). These tests run that step for every
rank of a group with a simulated group, two rounds: the first records what each rank hands
the merge's two all-gathers, the second returns the concatenation a real all-gather
returns (and checks the rank handed the same tensors again). The reference is the same
step at CP = 1 on the global store. For each CP in {2, 4}, T in {1, 2, 4} and ring depth in
{``index_kpool``, the speculative depth}, three requests -- two whose rows cross a 128-token
block (two ranks own parts of the step, one row closing a pool on the first block) and one
short one that leaves ranks 2 and 3 of a 4-rank group an empty window:

* the ring of every rank is the CP = 1 ring, bit for bit (the ring stays replicated);
* each rank's store holds every pool the step closes in a block the rank owns at the
  pool's local id, the CP = 1 value bit for bit, and is unchanged everywhere else but its
  trash row: a row another rank owns writes the trash row;
* the selection: per row, the union over the ranks of their selected local rows, mapped
  to global positions, is the CP = 1 selection's token set, exactly, and no token is
  named by two ranks; in the bypass regime the same holds for the causal fill.

``T = 4`` on the ``index_kpool``-deep ring stays refused at every CP size by the ring
rule (``max_rows_for``). Pool scores (the masked attention's one-rank form) are refused by
name under DCP. ``decode_trow.DCP_SPECULATIVE_ROWS``, which the platform's speculative
refusal under DCP reads, is pinned to this path: it is true and the path runs.
"""

from __future__ import annotations

import contextlib
import functools

import pytest
import torch

import vllm_neuron.model.glm5_next.model_fp8 as model_fp8
from test.vllm_neuron.functional.dsa.dsa_batch_case import pool_rows
from test.vllm_neuron.functional.dsa.dsa_decode_case import build_attention, decode_config
from vllm_neuron.functional.dsa import decode_trow as TR
from vllm_neuron.functional.dsa.decode_tail_update import max_rows_for
from vllm_neuron.utils import dcp_ownership as OWN
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: The latent bank's block: the DCP interleave.
BLOCK = 128
CPS = (2, 4)
ROWS = (1, 2, 4)
SLOTS_TOTAL = 6
#: (global window, regime): 1024 pools select (more than select_k = 512 over the group);
#: 512 pools are the bypass regime (selection keeps every pool).
WINDOWS = {"selecting": 4096, "bypass": 2048}


@pytest.fixture(autouse=True)
def _simulator(monkeypatch):
    """The kernels run on the CPU simulator, as in ``test_indexer_shard_forward``."""
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1"


@functools.lru_cache(maxsize=1)
def _module():
    return build_attention(model_fp8, decode_config(hidden_size=512, q_lora_rank=256))


def _depths():
    pool = int(_module().indexer.index_kpool)
    return (pool, TR.indexer_ring_depth(pool, 3))


def _starts(rows: int, window: int) -> list[int]:
    """Row 0's positions: two steps that end on a block boundary's far side at ``rows >= 2``
    (the window's last block and one inside it; the row before the boundary closes a
    pool), and a short request in blocks 0 and 1. At ``rows = 1`` each closes a pool."""
    back = rows // 2 if rows > 1 else 1
    blocks = window // BLOCK
    return [(blocks - 1) * BLOCK - back, (blocks // 2 + 3) * BLOCK - back, BLOCK - back]


class _ReplayGroup:
    """One DCP group's all-gathers for ranks run one after another, in two rounds."""

    def __init__(self, world: int):
        self.world_size = int(world)
        self.rank_in_group = 0
        self.replay = False
        self.recorded: dict[tuple[int, int], torch.Tensor] = {}
        self.calls = 0

    def begin(self, rank: int) -> None:
        self.rank_in_group = int(rank)
        self.calls = 0

    def all_gather(self, local: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 0, dim
        key = (self.rank_in_group, self.calls)
        self.calls += 1
        if not self.replay:
            self.recorded[key] = local.detach().clone()
            return torch.cat([local] * self.world_size, dim=0)
        assert torch.equal(local, self.recorded[key]), f"rank {key[0]} changed its input"
        return torch.cat([self.recorded[(r, key[1])] for r in range(self.world_size)], dim=0)


@contextlib.contextmanager
def _as_rank(indexer, rank: int, cp: int, group: _ReplayGroup, monkeypatch):
    saved = (indexer.dcp_size, getattr(indexer, indexer.SHARD_RANK_ATTR, None))
    indexer.dcp_size = int(cp)
    setattr(indexer, indexer.SHARD_RANK_ATTR, torch.full((1,), int(rank), dtype=torch.int32))
    group.begin(rank)
    monkeypatch.setattr(model_fp8, "_resolve_dcp_kv_group", lambda: (group, int(rank)))
    try:
        yield
    finally:
        indexer.dcp_size, rank_attr = saved
        setattr(indexer, indexer.SHARD_RANK_ATTR, rank_attr)


def _case(rows: int, depth: int, window: int, *, seed: int) -> dict:
    module = _module()
    indexer = module.indexer
    pool, dim = int(indexer.index_kpool), int(indexer.index_head_dim)
    gen = torch.Generator().manual_seed(seed)
    starts = _starts(rows, window)
    batch = len(starts)
    hidden = (torch.randn(batch * rows, int(module.hidden_size), generator=gen) * 0.5
              ).to(torch.bfloat16)
    q_latent = module.project_query_latent(hidden)
    return {
        "hidden": hidden,
        "q_latent": q_latent,
        "projected": indexer.project_stage(hidden, q_latent, query_side=True,
                                           rotate_query=False),
        "pool_bank": (torch.randn(SLOTS_TOTAL, pool_rows(window, pool), dim, generator=gen)
                      * 0.5).to(torch.bfloat16),
        "tail_bank": (torch.randn(SLOTS_TOTAL, 2, depth, dim, generator=gen) * 0.5
                      ).to(torch.bfloat16),
        "slots": torch.randperm(SLOTS_TOTAL, generator=gen)[:batch].to(torch.int32),
        "start": torch.tensor(starts, dtype=torch.int64),
        "rows": rows,
        "window": window,
    }


def _step(case, pool_bank, tail_bank, max_seq_len, **options):
    rows = case["rows"]
    seq_lens = (case["start"][:, None] + torch.arange(rows)[None, :] + 1).reshape(-1)
    return _module().indexer.forward_requests(
        case["hidden"], case["q_latent"], pool_bank, tail_bank, case["slots"],
        seq_lens.to(torch.int32), case["start"], max_seq_len=max_seq_len,
        projected=case["projected"], query_rotated=False, **options)


def _local_store(store: torch.Tensor, window: int, rank: int, cp: int, *,
                 seed: int) -> torch.Tensor:
    """``rank``'s pooled store beside the CP = 1 ``store`` of a ``window``-token window:
    local pool ``p`` holds global pool ``pool_global_index(p, rank)`` where that is a global
    row (not its trash), noise past it and on the trash row."""
    pool = int(_module().indexer.index_kpool)
    local_rows = pool_rows(
        OWN.local_context_capacity(window, block_size=BLOCK, dcp_size=cp), pool)
    gen = torch.Generator().manual_seed(seed)
    out = (torch.randn(int(store.shape[0]), local_rows, int(store.shape[2]), generator=gen)
           * 0.5).to(store.dtype)
    gids = OWN.pool_global_index(torch.arange(local_rows - 1), rank, block_size=BLOCK,
                                 dcp_size=cp, index_kpool=pool)
    inside = gids < int(store.shape[1]) - 1
    out[:, :local_rows - 1][:, inside] = store[:, gids[inside]]
    return out


def _run_group(case, cp, monkeypatch):
    """Every rank's step, two rounds; returns per rank ``(selection, store, ring, before)``."""
    indexer = _module().indexer
    local_window = OWN.local_context_capacity(case["window"], block_size=BLOCK, dcp_size=cp)
    group = _ReplayGroup(cp)
    stores = {rank: _local_store(case["pool_bank"], case["window"], rank, cp,
                                 seed=50 + rank)
              for rank in range(cp)}
    results = {}
    for replay in (False, True):
        group.replay = replay
        for rank in range(cp):
            store, ring = stores[rank].clone(), case["tail_bank"].clone()
            with _as_rank(indexer, rank, cp, group, monkeypatch):
                out = _step(case, store, ring, local_window, page_size=BLOCK)
            results[rank] = (out, store, ring, stores[rank])
    return results


def _global_tokens(local_tokens: torch.Tensor, rank: int, cp: int) -> list[set[int]]:
    out = []
    for row in local_tokens.tolist():
        mine = torch.tensor([t for t in row if t >= 0], dtype=torch.int64)
        out.append(set(OWN.token_position(mine, rank, block_size=BLOCK,
                                          dcp_size=cp).tolist()))
    return out


def _closed_pools(case) -> list[tuple[int, int]]:
    """``(request, global pool)`` of every pool the step's rows close."""
    pool = int(_module().indexer.index_kpool)
    closed = []
    for b, start in enumerate(case["start"].tolist()):
        for t in range(case["rows"]):
            if (start + t) % pool == pool - 1:
                closed.append((b, (start + t) // pool))
    return closed


def _forms():
    for rows in ROWS:
        for depth_name in ("served", "speculative"):
            yield rows, depth_name


@pytest.mark.parametrize("regime", sorted(WINDOWS))
@pytest.mark.parametrize("cp", CPS)
@pytest.mark.parametrize("rows, depth_name", list(_forms()))
def test_every_rank_writes_its_pools_and_merges_to_the_cp1_selection(
        rows, depth_name, cp, regime, monkeypatch):
    served, speculative = _depths()
    depth = served if depth_name == "served" else speculative
    pool = int(_module().indexer.index_kpool)
    if rows > max_rows_for(depth, pool):
        pytest.skip("refused by the ring rule; test_the_ring_rule_still_refuses")
    window = WINDOWS[regime]
    case = _case(rows, depth, window, seed=1000 * cp + 10 * rows + depth + window)
    store_1, ring_1 = case["pool_bank"].clone(), case["tail_bank"].clone()
    want = _step(case, store_1, ring_1, window)
    results = _run_group(case, cp, monkeypatch)
    union = [set() for _ in range(int(want.shape[0]))]
    closed = _closed_pools(case)
    for rank, (out, store, ring, before) in results.items():
        # The ring is replicated.
        assert torch.equal(ring, ring_1), (cp, rank)
        # The store: the closed pools of this rank's blocks at their local ids, the
        # CP = 1 value; every other row unchanged but the trash row.
        expected = before.clone()
        for b, gpool in closed:
            if int(OWN.pool_owner(torch.tensor(gpool), block_size=BLOCK, dcp_size=cp,
                                  index_kpool=pool)) == rank:
                local = int(OWN.pool_local_index(torch.tensor(gpool), block_size=BLOCK,
                                                 dcp_size=cp, index_kpool=pool))
                slot = int(case["slots"][b])
                expected[slot, local] = store_1[slot, gpool]
        assert torch.equal(store[:, :-1], expected[:, :-1]), (cp, rank)
        # The selection: this rank's rows only, each token on one rank.
        assert out is not None and tuple(out.shape) == tuple(want.shape), (cp, rank)
        for r, got in enumerate(_global_tokens(out, rank, cp)):
            assert not (got & union[r]), (cp, rank, r)
            union[r] |= got
    for r, row in enumerate(want.tolist()):
        expected = {t for t in row if t >= 0}
        assert union[r] == expected, (cp, rows, depth, regime, r,
                                      sorted(union[r] ^ expected)[:8])


@pytest.mark.parametrize("cp", (1,) + CPS)
def test_the_ring_rule_still_refuses(cp, monkeypatch):
    served, _speculative = _depths()
    case = _case(4, served, WINDOWS["selecting"], seed=7)
    group = _ReplayGroup(max(cp, 1))
    indexer = _module().indexer
    with _as_rank(indexer, 0, cp, group, monkeypatch):
        with pytest.raises(model_fp8.Glm5NextDSAIndexerError, match="takes back on rollback"):
            _step(case, case["pool_bank"].clone(), case["tail_bank"].clone(),
                  WINDOWS["selecting"], **({"page_size": BLOCK} if cp > 1 else {}))


@pytest.mark.parametrize("cp", CPS)
def test_pool_scores_are_refused_by_name_under_dcp(cp, monkeypatch):
    _served, speculative = _depths()
    case = _case(4, speculative, WINDOWS["selecting"], seed=8)
    group = _ReplayGroup(cp)
    indexer = _module().indexer
    with _as_rank(indexer, 0, cp, group, monkeypatch):
        with pytest.raises(model_fp8.Glm5NextDSAIndexerError, match="pool scores"):
            _step(case, case["pool_bank"].clone(), case["tail_bank"].clone(),
                  OWN.local_context_capacity(WINDOWS["selecting"], block_size=BLOCK,
                                             dcp_size=cp),
                  page_size=BLOCK, pool_scores=True)


def test_the_capability_flag_vouches_for_this_path(monkeypatch):
    """The platform admits the speculative form under DCP on ``DCP_SPECULATIVE_ROWS``;
    with it set the indexer must take a verify step of T = 4 on the speculative ring."""
    assert TR.DCP_SPECULATIVE_ROWS is True
    _served, speculative = _depths()
    for cp in CPS:
        case = _case(4, speculative, WINDOWS["selecting"], seed=11 + cp)
        results = _run_group(case, cp, monkeypatch)
        assert all(out is not None and tuple(out.shape)[0] == 4 * len(case["start"])
                   for out, _store, _ring, _before in results.values())
