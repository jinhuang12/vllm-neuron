# SPDX-License-Identifier: Apache-2.0
"""The indexer's prefill forward with the query-sharded selection, at the tiny geometry.

``Glm5NextDSAIndexer.forward``'s prefill leg divides the selection's query rows over the
tensor-parallel ranks and all-gathers the pool ids (``functional/dsa/indexer_shard.py``).
These tests run that wiring with a simulated group: every rank's forward runs one after
another, twice. The first round records what each rank hands the gather; the second
returns the concatenation a real all-gather returns. A rank's result must then select, row for
row, what the replicated forward selects.

The rank enters as a device operand bound at weight preparation, because one prefill graph
serves every rank. ``test_the_rank_operand_is_a_graph_input`` checks that a compile lifts
it as an input rather than folding rank 0's value in.

The kernel-level equality over the production shapes is
``test/vllm_neuron/functional/dsa/test_indexer_shard.py``.
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.model.glm5_next.test_dsa_layer import (
    POOL_SIZE,
    _impl,
    _materialise_indexer,
    _tiny_text_config,
    prefill_slot_mapping,
)

#: Above one 128-row tile, so the gate shards; four ranks of 64 rows.
TOKENS = 256
#: A select_k of 16 pools out of 64 candidates: the selecting regime with real choices.
INDEX_TOPK = 64
PAGE_SIZE = 4
#: The pooled store: one row per complete pool of the chunk, plus spare rows the chunk
#: does not fill, so the store is not sized to the chunk exactly.
POOL_SPARE_ROWS = 8
POOL_ROWS = TOKENS // POOL_SIZE + POOL_SPARE_ROWS


@pytest.fixture(autouse=True)
def _simulator(monkeypatch):
    """The kernels run on the CPU simulator here, as in ``test_mhc_layer``: the selection
    under test is the NKI chain, not its torch fallback. The kill switch starts at its
    default (on) whatever the caller exported; the tests that need it off set it."""
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.delenv("VLLM_NEURON_DSA_INDEXER_SHARD", raising=False)


class _SimulatedGroup:
    """``world`` ranks of one tensor-parallel group, run one after another (two rounds)."""

    def __init__(self, world: int):
        self.world_size = int(world)
        self.rank_in_group = 0
        self.rank: int | None = None
        self.gathering = False
        self.recorded: dict[int, torch.Tensor] = {}
        self.calls: list[tuple[int | None, torch.dtype, int, tuple[int, ...]]] = []

    def all_gather(self, local: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self.calls.append((self.rank, local.dtype, dim, tuple(local.shape)))
        if not self.gathering:
            assert self.rank not in self.recorded, f"rank {self.rank} gathered twice"
            self.recorded[self.rank] = local.detach().clone()
            shape = list(local.shape)
            shape[dim] *= self.world_size
            return local.new_zeros(shape)
        assert torch.equal(local, self.recorded[self.rank]), (
            f"rank {self.rank}'s rows changed between the two rounds"
        )
        return torch.cat([self.recorded[r] for r in range(self.world_size)], dim=dim)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def _indexer(seed: int = 7):
    model_fp8 = _impl()
    cfg = _tiny_text_config(index_topk=INDEX_TOPK)
    indexer = model_fp8.Glm5NextDSAIndexer(cfg)
    _materialise_indexer(indexer, torch.Generator().manual_seed(seed))
    return indexer, cfg


def _operands(cfg, tokens: int = TOKENS, seed: int = 11) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {
        "hidden": torch.randn(tokens, int(cfg.hidden_size), generator=gen),
        "q_latent": torch.randn(tokens, int(cfg.q_lora_rank), generator=gen),
        "pool_cache": torch.zeros(POOL_ROWS, int(cfg.index_head_dim), dtype=torch.bfloat16),
        "seq_lens": torch.arange(1, tokens + 1, dtype=torch.int32),
        "slot_mapping": prefill_slot_mapping(tokens, int(cfg.index_kpool)),
        "prefill_tail": torch.zeros(2, int(cfg.index_kpool), int(cfg.index_head_dim),
                                    dtype=torch.bfloat16),
    }


def _prefill(indexer, ops: dict, *, tokens: int = TOKENS):
    """One prefill forward on fresh copies of the carried state; returns (indices, state)."""
    pool_cache = ops["pool_cache"].clone()
    tail = ops["prefill_tail"].clone()
    out = indexer.forward(
        ops["hidden"], ops["q_latent"], pool_cache, ops["seq_lens"],
        max_seq_len=tokens, page_size=PAGE_SIZE, slot_mapping=ops["slot_mapping"],
        prefill_tail=tail, prefill_end_position=tokens,
    )
    return out, (pool_cache, tail)


def _row_sets(ids: torch.Tensor) -> list[frozenset[int]]:
    return [frozenset(v for v in row.tolist() if v >= 0) for row in ids]


def _same_selection(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Per row, the same selected set and the same number of ``-1`` columns.

    Not positional equality: the vendored top-k orders tied scores by a rule that
    depends on its row count, and the tiny geometry's four indexer heads tie often (a
    score is exactly 0 whenever all four rectified dots are). The kernel-level test
    shows the slots differ only among equal scores.
    """
    return (got.shape == want.shape and got.dtype == want.dtype
            and torch.equal((got < 0).sum(dim=1), (want < 0).sum(dim=1))
            and _row_sets(got) == _row_sets(want))


def _bind(monkeypatch, model_fp8, indexer, group, rank: int) -> None:
    """Bind ``rank`` as the indexer's rank operand, the way preparation does in a worker."""
    group.rank_in_group = rank
    monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: group)
    assert indexer.prepare_projection_weights() == len(indexer.projection_widths())
    operand = getattr(indexer, indexer.SHARD_RANK_ATTR)
    assert operand.dtype == torch.int32 and operand.tolist() == [rank]


def test_preparation_binds_this_ranks_operand_and_none_at_one_rank(monkeypatch):
    model_fp8 = _impl()
    indexer, _cfg = _indexer()
    monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: None)
    assert indexer.prepare_projection_weights() == len(indexer.projection_widths())
    assert getattr(indexer, indexer.SHARD_RANK_ATTR) is None

    group = _SimulatedGroup(8)
    _bind(monkeypatch, model_fp8, indexer, group, 3)
    operand = getattr(indexer, indexer.SHARD_RANK_ATTR)
    weight = indexer._prepared_weight("wq_b")
    assert operand.device == weight.device and tuple(operand.shape) == (1,)
    # A plain attribute, like the prepared weights: not a buffer, so not in a checkpoint.
    assert indexer.SHARD_RANK_ATTR not in dict(indexer.named_buffers())
    assert indexer.SHARD_RANK_ATTR not in indexer.state_dict()


@pytest.mark.parametrize("world", [2, 4])
def test_every_rank_returns_the_replicated_selection(world, monkeypatch):
    model_fp8 = _impl()
    indexer, cfg = _indexer()
    ops = _operands(cfg)

    monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: None)
    assert indexer.prepare_projection_weights() == len(indexer.projection_widths())
    want, want_state = _prefill(indexer, ops)
    assert want.dtype == torch.int32 and int(want.shape[0]) == TOKENS

    group = _SimulatedGroup(world)
    rows = -(-TOKENS // world)
    for gathering in (False, True):
        group.gathering = gathering
        for rank in range(world):
            group.rank = rank
            _bind(monkeypatch, model_fp8, indexer, group, rank)
            got, state = _prefill(indexer, ops)
            if not gathering:
                continue
            assert _same_selection(got, want), f"rank {rank} of {world} selected differently"
            # The write stage is not sharded: every rank stores the same pools and ring.
            for mine, theirs in zip(state, want_state):
                assert torch.equal(mine, theirs)
    # One gather per forward, on the query rows, of this rank's R rows as the selector's int32.
    k = indexer.select_k()
    assert [c[0] for c in group.calls] == list(range(world)) * 2
    assert all(c[1:] == (torch.int32, 0, (rows, k)) for c in group.calls), group.calls[:2]


def test_a_rank_hands_the_gather_exactly_its_own_rows(monkeypatch):
    """Rank r's recorded rows are rows r*R .. r*R+R-1 of the replicated pool ids."""
    model_fp8 = _impl()
    indexer, cfg = _indexer()
    ops = _operands(cfg)
    monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: None)
    assert indexer.prepare_projection_weights() == len(indexer.projection_widths())

    seen = []
    real = indexer.expand_indices

    def spy(pool_ids, seq_lens):
        seen.append(pool_ids.clone())
        return real(pool_ids, seq_lens)

    monkeypatch.setattr(indexer, "expand_indices", spy)
    _prefill(indexer, ops)
    replicated = seen[-1]
    world = 4
    group = _SimulatedGroup(world)
    for rank in range(world):
        group.rank = rank
        _bind(monkeypatch, model_fp8, indexer, group, rank)
        _prefill(indexer, ops)
    rows = TOKENS // world
    for rank in range(world):
        mine = group.recorded[rank].to(torch.int32)
        assert _same_selection(mine, replicated[rank * rows:(rank + 1) * rows]), rank


def _gathers_with(monkeypatch, *, tokens=TOKENS, max_seq_len=None, decode=False, env=None,
                  bind=True) -> int:
    """How many gathers one forward performs under a 4-rank group."""
    model_fp8 = _impl()
    indexer, cfg = _indexer()
    group = _SimulatedGroup(4)
    group.rank = 1
    if bind:
        _bind(monkeypatch, model_fp8, indexer, group, 1)
    else:
        monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: None)
        assert indexer.prepare_projection_weights() == len(indexer.projection_widths())
        monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: group)
    if env is not None:
        monkeypatch.setenv("VLLM_NEURON_DSA_INDEXER_SHARD", env)
    ops = _operands(cfg, tokens=tokens)
    if decode:
        position = tokens - 1
        indexer.forward(
            ops["hidden"][-1:], ops["q_latent"][-1:], ops["pool_cache"].clone(),
            ops["seq_lens"][-1:], max_seq_len=tokens, page_size=PAGE_SIZE,
            tail=ops["prefill_tail"].clone(), position=position,
        )
    else:
        indexer.forward(
            ops["hidden"], ops["q_latent"], ops["pool_cache"].clone(), ops["seq_lens"],
            max_seq_len=tokens if max_seq_len is None else max_seq_len, page_size=PAGE_SIZE,
            slot_mapping=ops["slot_mapping"],
        )
    return len(group.calls)


def test_the_selecting_prefill_gathers_once(monkeypatch):
    assert _gathers_with(monkeypatch) == 1


def test_the_kill_switch_restores_the_replicated_path(monkeypatch):
    assert _gathers_with(monkeypatch, env="0") == 0


def test_a_chunk_of_one_row_tile_stays_replicated(monkeypatch):
    assert _gathers_with(monkeypatch, tokens=128) == 0


def test_the_decode_leg_stays_replicated(monkeypatch):
    assert _gathers_with(monkeypatch, decode=True) == 0


def test_the_bypass_regime_selects_nothing_and_gathers_nothing(monkeypatch):
    # 64 candidates at a 16-pool budget selects; 16 candidates (64 tokens) does not.
    assert _gathers_with(monkeypatch, tokens=TOKENS, max_seq_len=64) == 0


def test_an_unbound_rank_operand_stays_replicated(monkeypatch):
    """A group that appears after preparation has no rank operand to shard with."""
    assert _gathers_with(monkeypatch, bind=False) == 0


def test_the_rank_operand_is_a_graph_input(monkeypatch):
    """A compile reads the bound operand at run time: one graph, each rank's own rows,
    through the production row cut."""
    from vllm_neuron.functional.dsa.indexer_shard import row_shard
    from vllm_neuron.functional.dsa.shard_rows import dsa_take_rank_rows

    model_fp8 = _impl()
    indexer, _cfg = _indexer()
    group = _SimulatedGroup(4)
    _bind(monkeypatch, model_fp8, indexer, group, 0)
    shard = row_shard(16, 4)

    class Probe(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            rank = getattr(self.inner, self.inner.SHARD_RANK_ATTR)
            return dsa_take_rank_rows((x,), rank, shard.rows)[0]

    graphs = []

    def backend(gm, example_inputs):
        graphs.append([n.target for n in gm.graph.nodes if n.op == "placeholder"])
        return gm.forward

    probe = torch.compile(Probe(indexer), backend=backend, fullgraph=True)
    x = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    for rank in range(4):
        getattr(indexer, indexer.SHARD_RANK_ATTR).fill_(rank)
        assert torch.equal(probe(x), x[rank * 4:(rank + 1) * 4]), rank
    assert len(graphs) == 1, f"{len(graphs)} graphs for four ranks"
    assert len(graphs[0]) == 2, graphs  # x and the rank operand, nothing folded
