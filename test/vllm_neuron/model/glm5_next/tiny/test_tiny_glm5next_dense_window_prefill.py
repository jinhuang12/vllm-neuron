# SPDX-License-Identifier: Apache-2.0
"""The DSA prefill short regime at the attention call site, on the tiny MLA geometry.

``Glm5NextMLAAttention.forward`` takes the dense window when the bound its graph can
prove -- ``min(max_seq_len, window rows)`` -- is within the identity bound
(``select_k * index_kpool + index_kpool - 1``; 2 pools x 4 + 3 = 11 tokens at these
dials). The tests read which kernels ran, that the output matches the as-built sparse
path, that the indexer's stores (pooled keys, ring) and the latent bank are written
exactly as before, and that a chunk past the bound keeps selecting.

The chunk windows here are sized per chunk, ``ceil((prefix + chunk) / page)`` pages,
the way a runner that serves each chunk on the smallest segment holding its prefix
sizes them. The window is a graph shape, so it is what a traced graph can decide on.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_dense_window_prefill.py
"""

from __future__ import annotations

import os

import pytest
import torch

from test.vllm_neuron.functional.attention.neuron_device_nodes import (
    open_neuron_device_nodes,
)
from test.vllm_neuron.model.glm5_next.tiny.test_tiny_glm5next_forward import (
    MLA_HIDDEN_SIZE,
    MLA_INDEX_HEAD_DIM,
    MLA_INDEX_KPOOL,
    MLA_PAGE_SIZE,
    MLA_SOFTMAX_SCALE,
    MLA_TOPK_POOLS,
    _mla_attention_fixture,
    _read_seam_counters,
    _reset_seam_counters,
)
from vllm_neuron.functional.attention import mla_dense_window as DW
from vllm_neuron.functional.dsa.decode_bypass import bypass_max_context

pytestmark = [pytest.mark.fast]

#: The kill switch (``vllm_neuron/envs.py``); ``0`` restores the sparse path.
KILL_SWITCH = "VLLM_NEURON_MLA_DENSE_WINDOW"
#: The identity bound at the tiny dials: ``select_k`` complete pools plus a full tail.
BOUND = MLA_TOPK_POOLS * MLA_INDEX_KPOOL + MLA_INDEX_KPOOL - 1
#: A model length well past the bound: the decision must come from the window.
MAX_MODEL_LEN = 64
#: Latent bank pages and pooled-key store rows: room for the widest window and the
#: longest prompt below (24 rows, 6 pooled keys).
BANK_PAGES = 8
POOL_ROWS = 32

#: The ruled tolerance for a bf16 layer output: two bf16 units at the reference's peak.
RTOL = 1e-2
BF16_UNITS = 2


@pytest.fixture(autouse=True)
def _simulator(monkeypatch):
    """Both kernels run in the NKI simulator; an explicit ``NKI_SIMULATOR`` still wins."""
    if "NKI_SIMULATOR" not in os.environ:
        monkeypatch.setenv("NKI_SIMULATOR", "1")


@pytest.fixture(autouse=True)
def _no_neuron_device_node():
    """Fail a test after which this process holds a Neuron device open (CPU only)."""
    yield
    held = open_neuron_device_nodes()
    assert not held, f"a CPU-only test left Neuron device nodes open: {held}"


def _atol(reference: torch.Tensor) -> float:
    return BF16_UNITS * 2.0 ** -8 * float(reference.float().abs().max())


class _State:
    """One request's carried state: latent bank, pooled-key store, ring."""

    def __init__(self, attention, dtype=torch.bfloat16):
        self.latent = torch.zeros(BANK_PAGES * MLA_PAGE_SIZE, attention.NUM_LATENT_KV_HEADS,
                                  int(attention.head_size), dtype=dtype)
        self.pool = torch.zeros(POOL_ROWS, MLA_INDEX_HEAD_DIM, dtype=torch.bfloat16)
        self.tail = torch.zeros(2, MLA_INDEX_KPOOL, MLA_INDEX_HEAD_DIM, dtype=torch.bfloat16)

    def clone(self):
        other = _State.__new__(_State)
        other.latent, other.pool, other.tail = (
            self.latent.clone(), self.pool.clone(), self.tail.clone())
        return other


def _pages(rows: int) -> int:
    """Pages of a window that holds ``rows`` rows: the per-chunk sizing the docstring states."""
    return -(-int(rows) // MLA_PAGE_SIZE)


#: A short chunk, its window, a window past the bound, and a window one row past it.
SHORT = 8
SHORT_PAGES = _pages(SHORT)
WIDE_PAGES = _pages(3 * SHORT)
PAST_BOUND_PAGES = _pages(BOUND + 1)


def _hidden(tokens: int, seed: int, dtype=torch.bfloat16) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(tokens, MLA_HIDDEN_SIZE, generator=gen) * 0.5).to(dtype)


def _chunk(attention, state, hidden, *, start: int, real: int, window_pages: int,
           max_seq_len: int = MAX_MODEL_LEN, active_rows: int | None = None):
    """One prefill chunk through ``forward``, with the operands the runner's carrier holds."""
    tokens = int(hidden.shape[0])
    positions = torch.arange(tokens) + start
    slot = torch.full((tokens,), -1, dtype=torch.int32)
    for i in range(real):
        if (int(positions[i]) + 1) % MLA_INDEX_KPOOL == 0:
            slot[i] = int(positions[i]) // MLA_INDEX_KPOOL
    latent_slots = torch.minimum(positions, torch.tensor(start + real - 1)).to(torch.int64)
    return attention.forward(
        hidden,
        latent_cache=state.latent,
        pool_cache=state.pool,
        seq_lens=(positions + 1).to(torch.int32),
        start_position=start,
        softmax_scale=MLA_SOFTMAX_SCALE,
        max_seq_len=max_seq_len,
        page_size=MLA_PAGE_SIZE,
        block_table_row=torch.arange(window_pages, dtype=torch.int32).reshape(-1, 1),
        latent_slots=latent_slots,
        slot_mapping=slot,
        prefill_tail=state.tail,
        prefill_end_position=start + real,
        **({"active_mla_query_rows": active_rows} if active_rows is not None else {}),
    )


def _routes():
    counts = _read_seam_counters()
    return {
        "dense_window": DW.mla_dense_window_dispatch_counters()[0],
        "mla_sparse": counts["mla_sparse"][0],
        "score_gemm": counts["dsa_score_gemm"][0],
        "topk_select": counts["dsa_topk_select"][0],
        "index_expand": counts["dsa_index_expand"][0],
        "kpool_hadamard": counts["dsa_kpool_hadamard"][0],
        "mla_projection": counts["mla_projection"][0],
    }


def _reset():
    _reset_seam_counters()
    DW.reset_mla_dense_window_dispatch_counters()


def _run(attention, state, hidden, **kw):
    _reset()
    out = _chunk(attention, state, hidden, **kw)
    return out, _routes()


def _assert_same_state(a: _State, b: _State) -> None:
    torch.testing.assert_close(a.latent, b.latent, rtol=0.0, atol=0.0)
    torch.testing.assert_close(a.pool, b.pool, rtol=0.0, atol=0.0)
    torch.testing.assert_close(a.tail, b.tail, rtol=0.0, atol=0.0)


@pytest.fixture(scope="module")
def attention():
    module, _raw, _gains, _cfg = _mla_attention_fixture()
    return module


def test_bound_matches_the_decode_bypass_at_these_dials(attention):
    """The bound these tests place their windows around, and the windows themselves."""
    assert SHORT_PAGES * MLA_PAGE_SIZE <= BOUND < PAST_BOUND_PAGES * MLA_PAGE_SIZE
    assert WIDE_PAGES * MLA_PAGE_SIZE > BOUND
    indexer = attention.indexer
    assert indexer.select_k() == MLA_TOPK_POOLS
    assert BOUND == bypass_max_context(int(indexer.index_topk), int(indexer.index_kpool))


def test_short_chunk_takes_the_dense_window(attention, monkeypatch):
    """An 8-token chunk in an 8-row window: no selection, no sparse gathers, same answer."""
    hidden = _hidden(SHORT, seed=1)
    on_state, off_state = _State(attention), _State(attention)

    monkeypatch.delenv(KILL_SWITCH, raising=False)
    dense, on = _run(attention, on_state, hidden, start=0, real=SHORT, window_pages=SHORT_PAGES)
    monkeypatch.setenv(KILL_SWITCH, "0")
    sparse, off = _run(attention, off_state, hidden, start=0, real=SHORT, window_pages=SHORT_PAGES)

    assert on["dense_window"] == 1 and on["mla_sparse"] == 0
    assert on["score_gemm"] == on["topk_select"] == on["index_expand"] == 0
    assert off["dense_window"] == 0 and off["mla_sparse"] == 1
    assert off["score_gemm"] == off["topk_select"] == off["index_expand"] == 1
    # The indexer's query side is skipped: the rotation (one kpool_hadamard dispatch)
    # and two projections (wq_b, weights_proj); its write stage still runs.
    assert on["kpool_hadamard"] == off["kpool_hadamard"] - 1 >= 1
    assert on["mla_projection"] == off["mla_projection"] - 2

    _assert_same_state(on_state, off_state)
    assert tuple(dense.shape) == (SHORT, MLA_HIDDEN_SIZE)
    torch.testing.assert_close(dense.float(), sparse.float(), rtol=RTOL, atol=_atol(sparse))


def test_padded_chunk_takes_the_dense_window(attention, monkeypatch):
    """A 6-token prompt padded to 8 operator rows: the real rows match the sparse path."""
    hidden = _hidden(SHORT, seed=2)
    on_state, off_state = _State(attention), _State(attention)
    monkeypatch.delenv(KILL_SWITCH, raising=False)
    dense, on = _run(attention, on_state, hidden, start=0, real=6, window_pages=SHORT_PAGES)
    monkeypatch.setenv(KILL_SWITCH, "0")
    sparse, off = _run(attention, off_state, hidden, start=0, real=6, window_pages=SHORT_PAGES)
    assert on["dense_window"] == 1 and off["mla_sparse"] == 1
    _assert_same_state(on_state, off_state)
    torch.testing.assert_close(dense[:6].float(), sparse[:6].float(), rtol=RTOL,
                               atol=_atol(sparse[:6]))


def test_active_query_rows_take_the_dense_window(attention, monkeypatch):
    """The selected token bucket (``active_mla_query_rows``) attends only its prefix of the
    query rows and leaves zero rows after it, on both paths."""
    hidden = _hidden(SHORT, seed=7)
    on_state, off_state = _State(attention), _State(attention)
    monkeypatch.delenv(KILL_SWITCH, raising=False)
    dense, on = _run(attention, on_state, hidden, start=0, real=6, window_pages=SHORT_PAGES,
                     active_rows=6)
    monkeypatch.setenv(KILL_SWITCH, "0")
    sparse, off = _run(attention, off_state, hidden, start=0, real=6, window_pages=SHORT_PAGES,
                       active_rows=6)
    assert on["dense_window"] == 1 and on["mla_sparse"] == 0
    assert off["dense_window"] == 0 and off["mla_sparse"] == 1
    _assert_same_state(on_state, off_state)
    torch.testing.assert_close(dense[:6].float(), sparse[:6].float(), rtol=RTOL,
                               atol=_atol(sparse[:6]))
    torch.testing.assert_close(dense[6:], sparse[6:], rtol=0.0, atol=0.0)


def test_three_chunk_prompt_dense_then_sparse(attention, monkeypatch):
    """Chunk 1 (context 8) takes the dense window; chunks 2 and 3 (contexts 16, 23) are
    past the bound and select. With the switch off all three select. The carried state
    after chunk 1 is bitwise the same, so chunks 2 and 3 are bitwise the same too."""
    # (start, real tokens, window pages): each window holds the prefix plus the chunk.
    chunks = [(start, real, _pages(start + real)) for start, real in ((0, 8), (8, 8), (16, 7))]
    assert chunks[0][2] * MLA_PAGE_SIZE <= BOUND < chunks[1][2] * MLA_PAGE_SIZE
    hidden = [_hidden(real, seed=10 + i) for i, (_, real, _) in enumerate(chunks)]

    def prompt(enabled: bool):
        if enabled:
            monkeypatch.delenv(KILL_SWITCH, raising=False)
        else:
            monkeypatch.setenv(KILL_SWITCH, "0")
        state = _State(attention)
        outs, routes, states = [], [], []
        for (start, real, pages), h in zip(chunks, hidden):
            out, route = _run(attention, state, h, start=start, real=real, window_pages=pages)
            outs.append(out)
            routes.append(route)
            states.append(state.clone())
        return outs, routes, states

    on_out, on_routes, on_states = prompt(True)
    off_out, off_routes, off_states = prompt(False)

    assert [r["dense_window"] for r in on_routes] == [1, 0, 0]
    assert [r["mla_sparse"] for r in on_routes] == [0, 1, 1]
    assert [r["topk_select"] for r in on_routes] == [0, 1, 1]
    assert [r["dense_window"] for r in off_routes] == [0, 0, 0]
    assert [r["mla_sparse"] for r in off_routes] == [1, 1, 1]

    for i in range(3):
        _assert_same_state(on_states[i], off_states[i])
    torch.testing.assert_close(on_out[0].float(), off_out[0].float(), rtol=RTOL,
                               atol=_atol(off_out[0]))
    torch.testing.assert_close(on_out[1], off_out[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(on_out[2], off_out[2], rtol=0.0, atol=0.0)


def test_wide_window_keeps_selecting(attention, monkeypatch):
    """The same short chunk in a window wider than the bound (one fixed segment for every
    chunk, as the as-built bs=64 line has): the graph cannot prove the identity, so it
    selects. This is the conservative answer, not a missed one."""
    monkeypatch.delenv(KILL_SWITCH, raising=False)
    _, route = _run(attention, _State(attention), _hidden(SHORT, seed=3), start=0, real=SHORT,
                    window_pages=WIDE_PAGES)
    assert route["dense_window"] == 0 and route["mla_sparse"] == 1
    assert route["topk_select"] == 1


def test_short_model_length_bounds_a_wide_window(attention, monkeypatch):
    """``max_seq_len`` at the bound proves the identity whatever the window's width."""
    monkeypatch.delenv(KILL_SWITCH, raising=False)
    _, route = _run(attention, _State(attention), _hidden(SHORT, seed=4), start=0, real=SHORT,
                    window_pages=WIDE_PAGES, max_seq_len=BOUND)
    assert route["dense_window"] == 1 and route["mla_sparse"] == 0


def test_one_token_past_the_bound_selects(attention, monkeypatch):
    """A window one row past the bound (rounded up to a page) selects, even for a short
    chunk whose own rows are all inside it."""
    monkeypatch.delenv(KILL_SWITCH, raising=False)
    _, route = _run(attention, _State(attention), _hidden(SHORT, seed=5), start=0, real=SHORT,
                    window_pages=PAST_BOUND_PAGES)
    assert route["dense_window"] == 0 and route["mla_sparse"] == 1


def test_fp32_cache_keeps_the_sparse_path(attention, monkeypatch):
    """The dense kernel takes the stored 2-byte query and cache only; an fp32 layer keeps
    the as-built route even in the short regime."""
    monkeypatch.delenv(KILL_SWITCH, raising=False)
    state = _State(attention, dtype=torch.float32)
    _, route = _run(attention, state, _hidden(SHORT, seed=6, dtype=torch.float32), start=0,
                    real=SHORT, window_pages=SHORT_PAGES)
    assert route["dense_window"] == 0 and route["mla_sparse"] == 1
