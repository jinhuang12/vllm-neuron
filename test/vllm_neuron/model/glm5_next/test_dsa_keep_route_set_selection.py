# SPDX-License-Identifier: Apache-2.0
"""The DSA prefill selection as a set on the dense-masked keep route, without its ids.

On a prefill chunk whose keep route is ``DENSE_MASKED`` (``dsa_keep_table.py``) the keep
table is the selection's one reader. Its bytes are a function of each row's set of pool
ids (``functional/dsa/selection_keep.py``), so ``dsa_set_selection.keep_route_selects_a_set``
lets the indexer skip the top-k sort and the sentinel order there, and the prefill forward
asks the indexer for the table alone: ``_select_pool_ids`` makes the bits from each rank's
own ids and gathers no ids, and ``attend`` stands a ``-1`` fill in for the absent index
operand, whose shape alone the seam reads on that route. These tests pin each part:

* the predicate, against the switch ``VLLM_NEURON_DSA_SET_SELECTION`` and the route;
* ``_select_pool_ids`` and the indexer forward: ``(None, bits)`` with the same bits as a call
  that returns the ids, a refusal for a call that wants neither, and one all-gather (the
  bits) on the row-sharded path;
* ``attend``: the stand-in operand handed to the seam, and no masked-window refusal for a
  set that a keep table reads;
* one DCP-1 prefill chunk on the route: the attention output and the carried caches are
  bit-equal with the switch off, while the indexer is asked for no ids and for a set only
  with the switch on; a collecting forward still asks for the ordered ids.

The chunk's geometry is the smallest that takes the route at the checkpoint's per-rank
attention shape (one head, latent 512, bf16, 128-row pages, pools of 4): the masked-window
kernel holds the window resident and refuses above 72 pages at latent 512, and the
dense-masked pass serves whole 4-page blocks up to ``VLLM_NEURON_MLA_DENSE_MASKED_MAX_PAGES``.
So the window is 76 pages. ``index_topk`` is 512, a ``select_k`` of 128 pools: the keep
kernel takes the ids 128 at a time. One forward costs about 15 s on the simulator.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from test.vllm_neuron.functional.dsa.dsa_decode_case import build_attention, decode_config
from test.vllm_neuron.model.glm5_next.test_dsa_layer import _materialise_indexer
from vllm_neuron.functional.attention import mla_dense_masked, mla_sparse
from vllm_neuron.functional.dsa import selection_keep
from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.glm5_next.dsa_keep_table import KeepTableRoute, prefill_keep_table_route
from vllm_neuron.model.glm5_next.dsa_set_selection import (
    keep_route_selects_a_set,
    prefill_selects_a_set,
)

SWITCH = "VLLM_NEURON_DSA_SET_SELECTION"
#: A ``select_k`` of 128 pools at pools of 4: the keep kernel's ids come 128 per matmul.
INDEX_TOPK = 512
#: The latent bank's page, which the dense-masked pass requires (one 128-row key chunk).
PAGE = 128
#: The smallest window the masked-window kernel refuses at latent 512 (above 72 pages) that
#: is whole 4-page blocks of the dense-masked pass.
WINDOW_PAGES = 76
BANK_PAGES = WINDOW_PAGES + 4
#: One 128-query tile, the last rows of a context that fills the window.
CHUNK = 128
CONTEXT = WINDOW_PAGES * PAGE
START = CONTEXT - CHUNK

#: The ``_select_pool_ids`` case: 256 rows, so a 2-rank group shards them 128 per rank, at the
#: end of a 1024-token context, so every row ranks 192 to 256 complete pools of 256.
SELECT_ROWS = 256
SELECT_WORLD = 2
SELECT_RANK = 1
SELECT_CONTEXT = 1024


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    """The kernels run on the CPU simulator, and every switch these routes read starts at
    its default whatever the caller exported; the tests that turn one off set it."""
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    for name in (SWITCH, "VLLM_NEURON_MLA_DENSE_MASKED", "VLLM_NEURON_MLA_DENSE_MASKED_MAX_PAGES",
                 "VLLM_NEURON_DSA_INDEXER_SHARD"):
        monkeypatch.delenv(name, raising=False)


def _config():
    return replace(decode_config(hidden_size=512, q_lora_rank=256), index_topk=INDEX_TOPK)


# --------------------------------------------------------------------------- predicate


@pytest.mark.parametrize("switch, on", [(None, True), ("1", True), ("0", False)])
@pytest.mark.parametrize("route", [None, KeepTableRoute.DENSE_MASKED])
def test_only_the_dense_masked_route_selects_a_set_and_only_with_the_switch_on(
    switch, on, route, monkeypatch
):
    if switch is not None:
        monkeypatch.setenv(SWITCH, switch)
    assert keep_route_selects_a_set(route) is (on and route is KeepTableRoute.DENSE_MASKED)


# --------------------------------------------------------------------------- _select_pool_ids


class _RecordingGroup:
    """A tensor-parallel group whose all-gather records what it moves and tiles it.

    The tiled rows stand in for the other ranks' rows: these tests read what one rank
    hands the wire, not what the others return."""

    def __init__(self, world: int, rank: int):
        self.world_size = int(world)
        self.rank_in_group = int(rank)
        self.moved: list[torch.Tensor] = []

    def all_gather(self, local: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self.moved.append(local.detach().clone())
        return torch.cat([local] * self.world_size, dim=dim)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def _selecting_indexer():
    indexer = model_fp8.Glm5NextDSAIndexer(_config())
    _materialise_indexer(indexer, torch.Generator().manual_seed(17))
    return indexer


def _select_operands(indexer, seed: int = 23) -> dict:
    """Scores' operands for :data:`SELECT_ROWS` rows over every complete pool of the context."""
    gen = torch.Generator().manual_seed(int(seed))
    heads, head_dim = int(indexer.index_n_heads), int(indexer.index_head_dim)
    pools = SELECT_CONTEXT // int(indexer.index_kpool)
    return {
        "query": (torch.randn(SELECT_ROWS, heads, head_dim, generator=gen) * 0.5).to(
            torch.bfloat16),
        "candidate_keys": (torch.randn(pools, head_dim, generator=gen) * 0.5).to(torch.bfloat16),
        "weights": torch.randn(SELECT_ROWS, heads, generator=gen) * heads ** -0.5,
        "seq_lens": torch.arange(SELECT_CONTEXT - SELECT_ROWS + 1, SELECT_CONTEXT + 1,
                                 dtype=torch.int32),
    }


def test_a_keep_only_selection_returns_no_ids_and_the_bits_of_the_ids_it_selected():
    indexer = _selecting_indexer()
    ops = _select_operands(indexer)
    pools = int(ops["candidate_keys"].shape[0])

    ids, want = indexer._select_pool_ids(**ops, shardable=False, ordered=False, keep_bits=True)
    got_ids, got = indexer._select_pool_ids(
        **ops, shardable=False, ordered=False, keep_bits=True, ids_wanted=False
    )

    assert got_ids is None
    assert tuple(ids.shape) == (SELECT_ROWS, indexer.select_k()) and ids.dtype == torch.int32
    assert got.dtype == torch.uint8 and tuple(got.shape) == (SELECT_ROWS, pools // 8)
    assert torch.equal(got, want)
    assert torch.equal(got, selection_keep.selection_keep_reference(ids, pools))


def test_a_selection_with_no_reader_is_refused_by_name_before_it_scores(monkeypatch):
    indexer = _selecting_indexer()
    ops = _select_operands(indexer)

    def scored(*args, **kwargs):
        raise AssertionError("the refused selection scored its rows")

    monkeypatch.setattr(indexer, "score_pools", scored)
    with pytest.raises(model_fp8.Glm5NextDSAIndexerError) as caught:
        indexer._select_pool_ids(**ops, shardable=False, keep_bits=False, ids_wanted=False)
    assert "keep_bits" in str(caught.value) and "ids_wanted" in str(caught.value)


def test_a_sharded_keep_only_selection_gathers_the_bits_alone(monkeypatch):
    indexer = _selecting_indexer()
    group = _RecordingGroup(SELECT_WORLD, SELECT_RANK)
    monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: group)
    assert indexer.prepare_projection_weights() == len(indexer.projection_widths())
    assert getattr(indexer, indexer.SHARD_RANK_ATTR).tolist() == [SELECT_RANK]
    ops = _select_operands(indexer)
    pools = int(ops["candidate_keys"].shape[0])
    rows = SELECT_ROWS // SELECT_WORLD

    ids, bits = indexer._select_pool_ids(**ops, shardable=True, ordered=False, keep_bits=True)
    with_ids = list(group.moved)
    group.moved.clear()
    none, alone = indexer._select_pool_ids(
        **ops, shardable=True, ordered=False, keep_bits=True, ids_wanted=False
    )

    assert [(t.dtype, tuple(t.shape)) for t in with_ids] == [
        (torch.int32, (rows, indexer.select_k())), (torch.uint8, (rows, pools // 8)),
    ]
    assert tuple(ids.shape) == (SELECT_ROWS, indexer.select_k())
    assert none is None
    assert [(t.dtype, tuple(t.shape)) for t in group.moved] == [(torch.uint8, (rows, pools // 8))]
    assert torch.equal(group.moved[0], with_ids[1])
    assert torch.equal(group.moved[0], selection_keep.selection_keep_reference(with_ids[0], pools))
    assert torch.equal(alone, bits) and tuple(alone.shape) == (SELECT_ROWS, pools // 8)


def test_the_indexer_forward_returns_no_ids_beside_a_keep_table_it_alone_reads():
    indexer = _selecting_indexer()
    cfg = _config()
    pool = int(cfg.index_kpool)
    gen = torch.Generator().manual_seed(29)
    tokens = CHUNK
    positions = torch.arange(SELECT_CONTEXT - tokens, SELECT_CONTEXT)
    pools = SELECT_CONTEXT // pool
    pool_cache = (torch.randn(-(-(pools + 1) // PAGE) * PAGE, int(cfg.index_head_dim),
                              generator=gen) * 0.5).to(torch.bfloat16)
    hidden = torch.randn(tokens, int(cfg.hidden_size), generator=gen)
    q_latent = torch.randn(tokens, int(cfg.q_lora_rank), generator=gen)

    def forward(indices_wanted: bool):
        return indexer.forward(
            hidden, q_latent, pool_cache.clone(), (positions + 1).to(torch.int32),
            max_seq_len=SELECT_CONTEXT, page_size=PAGE,
            slot_mapping=torch.where((positions + 1) % pool == 0, positions // pool, -1).to(
                torch.int32),
            indices_wanted=indices_wanted, ordered=False, keep_bits=True,
        )

    indices, want = forward(True)
    none, got = forward(False)

    assert tuple(indices.shape) == (tokens, indexer.bypass_width())
    assert none is None
    assert got.dtype == torch.uint8 and tuple(got.shape) == (tokens, pools // 8)
    assert torch.equal(got, want)


# --------------------------------------------------------------------------- attend


def _window_operands(cfg, seed: int = 7) -> dict:
    """The keep-route chunk: the last :data:`CHUNK` rows of a context that fills a
    :data:`WINDOW_PAGES`-page window, its pages shuffled through a larger bank, and every
    earlier pool of the context already in the pooled-key store."""
    gen = torch.Generator().manual_seed(int(seed))
    pool = int(cfg.index_kpool)
    table = torch.randperm(BANK_PAGES, generator=gen)[:WINDOW_PAGES].to(torch.int32)
    positions = torch.arange(START, CONTEXT)
    pools = CONTEXT // pool
    return {
        "normed_hidden_states": (torch.randn(CHUNK, int(cfg.hidden_size), generator=gen)
                                 * 0.5).to(torch.bfloat16),
        "latent_cache": (torch.randn(BANK_PAGES * PAGE, 1, int(cfg.kv_lora_rank), generator=gen)
                         * 0.5).to(torch.bfloat16),
        "pool_cache": (torch.randn(-(-(pools + 1) // PAGE) * PAGE, int(cfg.index_head_dim),
                                   generator=gen) * 0.5).to(torch.bfloat16),
        "seq_lens": (positions + 1).to(torch.int32),
        "start_position": torch.tensor(START, dtype=torch.int64),
        "softmax_scale": float(int(cfg.qk_nope_head_dim) + int(cfg.qk_rope_head_dim)) ** -0.5,
        "max_seq_len": CONTEXT,
        "page_size": PAGE,
        "block_table_row": table.reshape(WINDOW_PAGES, 1),
        "latent_slots": table[positions // PAGE].to(torch.int64) * PAGE + positions % PAGE,
        "slot_mapping": torch.where((positions + 1) % pool == 0, positions // pool, -1).to(
            torch.int32),
    }


def _cloned(ops: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in ops.items()}


def test_this_files_chunk_takes_the_keep_route_and_not_the_masked_window():
    """The geometry the forward tests below stand on, asked as the prefill forward asks it."""
    cfg = _config()
    attention = model_fp8.Glm5NextMLAAttention(cfg)
    indexer = attention.indexer
    common = dict(heads=attention._heads_per_rank(), latent=int(cfg.kv_lora_rank),
                  pool_size=int(cfg.index_kpool), index_cols=indexer.bypass_width(),
                  q_dtype=torch.bfloat16, kv_dtype=torch.bfloat16)
    assert not prefill_selects_a_set(rows=CHUNK, window_rows=CONTEXT, **common)
    assert prefill_keep_table_route(
        rows=CHUNK, kv_latent=int(cfg.kv_lora_rank), bank_rows=BANK_PAGES * PAGE,
        window_pages=WINDOW_PAGES, page_size=PAGE, candidates=CONTEXT // int(cfg.index_kpool),
        **common,
    ) is KeepTableRoute.DENSE_MASKED


@pytest.mark.parametrize("selection_is_a_set", [False, True])
def test_attend_hands_the_seam_a_sentinel_fill_in_place_of_the_absent_ids(
    selection_is_a_set, monkeypatch
):
    """``topk_indices=None`` beside a keep table: the seam gets a ``SENTINEL_INDEX`` fill of
    the selection's shape, ``[tokens, bypass_width]`` int32, and the table as given. A set
    read through the table is the keep route's to check, so no masked-window refusal."""
    cfg = _config()
    attention = build_attention(model_fp8, cfg)
    ops = _window_operands(cfg)
    pools = CONTEXT // int(cfg.index_kpool)
    keep_bits = torch.randint(0, 256, (CHUNK, pools // 8), dtype=torch.uint8,
                              generator=torch.Generator().manual_seed(31))
    handed = []

    def seam(q_lift, c_kv, topk_indices, softmax_scale, **kwargs):
        handed.append((topk_indices, kwargs.get("keep_bits")))
        return q_lift.new_zeros(q_lift.shape, dtype=torch.float32)

    monkeypatch.setattr(mla_sparse, "mla_sparse_attention", seam)
    out = attention.attend(
        ops["normed_hidden_states"], ops["latent_cache"], ops["start_position"], None,
        ops["softmax_scale"], block_table_row=ops["block_table_row"],
        latent_slots=ops["latent_slots"], page_size=PAGE,
        selection_is_a_set=selection_is_a_set, keep_bits=keep_bits,
        keep_route=KeepTableRoute.DENSE_MASKED,
    )

    assert len(handed) == 1
    stand_in, table = handed[0]
    assert stand_in.dtype == torch.int32
    assert tuple(stand_in.shape) == (CHUNK, attention.indexer.bypass_width())
    assert torch.equal(stand_in, torch.full_like(stand_in, mla_sparse.SENTINEL_INDEX))
    assert table is keep_bits
    assert tuple(out.shape) == (CHUNK, int(cfg.hidden_size))


# --------------------------------------------------------------------------- the prefill forward


class _IndexerCalls:
    """Records the selection keywords of each indexer call and the ids its keep table reads."""

    KEYS = ("indices_wanted", "ordered", "keep_bits")

    def __init__(self, monkeypatch, indexer):
        self.calls: list[dict] = []
        self.table_ids: list[torch.Tensor] = []
        self.dense_masked = 0
        real_forward = indexer.forward
        real_keep = selection_keep.dsa_selection_keep
        real_pass = mla_dense_masked.mla_dense_masked_attention

        def forward(*args, **kwargs):
            self.calls.append({k: kwargs.get(k) for k in self.KEYS})
            return real_forward(*args, **kwargs)

        def keep(pool_ids, pools):
            self.table_ids.append(pool_ids.clone())
            return real_keep(pool_ids, pools)

        def dense_pass(*args, **kwargs):
            self.dense_masked += 1
            return real_pass(*args, **kwargs)

        monkeypatch.setattr(indexer, "forward", forward)
        monkeypatch.setattr(selection_keep, "dsa_selection_keep", keep)
        monkeypatch.setattr(mla_dense_masked, "mla_dense_masked_attention", dense_pass)


def _row_sets(ids: torch.Tensor) -> list[frozenset[int]]:
    return [frozenset(v for v in row.tolist() if v >= 0) for row in ids]


def test_a_keep_route_chunk_attends_bit_equal_with_the_switch_off_and_gathers_no_ids(
    monkeypatch
):
    cfg = _config()
    attention = build_attention(model_fp8, cfg)
    ops = _window_operands(cfg)
    seen = _IndexerCalls(monkeypatch, attention.indexer)

    runs = {}
    for switch in (None, "0"):
        if switch is None:
            monkeypatch.delenv(SWITCH, raising=False)
        else:
            monkeypatch.setenv(SWITCH, switch)
        state = _cloned(ops)
        out = attention.forward(state.pop("normed_hidden_states"), **state)
        runs[switch] = (out, state["latent_cache"], state["pool_cache"])

    assert seen.calls == [
        {"indices_wanted": False, "ordered": False, "keep_bits": True},
        {"indices_wanted": False, "ordered": True, "keep_bits": True},
    ]
    assert seen.dense_masked == 2
    (out_on, bank_on, store_on), (out_off, bank_off, store_off) = runs[None], runs["0"]
    assert tuple(out_on.shape) == (CHUNK, int(cfg.hidden_size))
    assert bool(torch.isfinite(out_on.float()).all()) and bool(out_on.ne(0).any())
    assert torch.equal(out_on, out_off)
    assert torch.equal(bank_on, bank_off) and torch.equal(store_on, store_off)
    # The two tables read the same sets in different orders: the switch reached the
    # selection, and its order did not reach the output.
    set_ids, ordered_ids = seen.table_ids
    assert _row_sets(set_ids) == _row_sets(ordered_ids)
    assert not torch.equal(set_ids, ordered_ids)


def test_a_collecting_keep_route_chunk_asks_for_the_ordered_ids(monkeypatch):
    """The layer-stream dump hands the index rows on as the indexer emitted them, so a
    collecting forward on the keep route keeps their order and gathers them beside the table."""
    cfg = _config()
    attention = build_attention(model_fp8, cfg)
    ops = _window_operands(cfg)
    seen = _IndexerCalls(monkeypatch, attention.indexer)
    collector: list[torch.Tensor] = []

    state = _cloned(ops)
    attention.forward(state.pop("normed_hidden_states"), **state, collector=collector)

    assert seen.calls == [{"indices_wanted": True, "ordered": True, "keep_bits": True}]
    assert seen.dense_masked == 1
