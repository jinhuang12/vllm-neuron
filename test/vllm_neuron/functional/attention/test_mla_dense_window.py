# SPDX-License-Identifier: Apache-2.0
"""The dense-window MLA prefill kernel against the as-built sparse path, in the simulator.

While a query's causal context has at most ``select_k`` complete pools (2051 tokens at
``index_topk`` 2048, ``index_kpool`` 4), the DSA top-k keeps every pool, so the selected
token set is the whole causal prefix. :func:`mla_dense_window_attention` attends that
prefix directly over the paged window. These tests read it against:

* the as-built sparse path, ``mla_sparse_attention`` on the index rows the as-built
  selection chain emits for the same rows (the low-precision default and the fp32 kill
  switch ``VLLM_NEURON_MLA_SPARSE_FP32=1``);
* the bypass form of the same rows (the causal fill, ascending);
* a float64 oracle.

The identity itself is checked on the selection chain as built (bound, top-k, sentinel,
order, expansion): every row at or under the bound selects exactly ``0 .. seq_len - 1``,
and the first row past it does not. The chain runs on its torch routes with the NKI
kernels switched off; the simulator switch is never turned off.

Every test checks afterwards that the process holds no ``/dev/neuron*`` node open: this
module runs on the CPU only.

Tolerances: per row, ``||dense - sparse|| / ||sparse||`` <= :data:`ROW_REL` on at least
:data:`ROW_SHARE` of the rows. The full-chunk cases run the sparse kernel through the
simulator and take about a minute per 1024 query rows.
"""

from __future__ import annotations

import contextlib
import os

import pytest
import torch

from test.vllm_neuron.functional.attention.neuron_device_nodes import (
    open_neuron_device_nodes,
)

from vllm_neuron.functional.attention import mla_dense_window as DW
from vllm_neuron.functional.attention import mla_sparse as MS
from vllm_neuron.functional.dsa.decode_bypass import bypass_max_context
from vllm_neuron.functional.dsa.index_expand import index_expand_width
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

#: The served geometry, from the checkpoint's config.
_CONFIG = Glm5NextTextConfig()
LATENT = int(_CONFIG.kv_lora_rank)
KPOOL = int(_CONFIG.index_kpool)
SELECT_K = int(_CONFIG.index_topk) // KPOOL
#: The served KV block size (the runner's block_size on every GLM-5.3-Flash line).
PAGE = 128
#: Any positive scale; the kernel and every reference take it as an argument.
SCALE = 0.0625
#: The identity bound: ``select_k`` complete pools plus a full open tail.
BOUND = SELECT_K * KPOOL + KPOOL - 1
#: The expanded index width the as-built chain emits for ``select_k`` pools.
WIDTH = index_expand_width(SELECT_K, KPOOL)
#: The engine bound the runner hands the indexer on the p1 line (max_model_len).
MAX_MODEL_LEN = 4096
#: Bank pages; more than any case's window, so the window is a scattered subset.
BANK_PAGES = 64

#: Per-row relative error against each sparse reference, and the share of rows that must
#: meet it. The acceptance bar (1e-4 on 99.9% of rows).
ROW_REL = 1e-4
ROW_SHARE = 0.999
#: Relative L2 of the whole output against the float64 oracle. The low-precision sparse
#: body's own error against it is about 6e-6 (bf16 hi/lo split of p, about 16 bits); the
#: bar is 5x that.
ORACLE_REL_L2 = 3e-5


def _rel_l2(got, want) -> float:
    got, want = got.double(), want.double()
    return float((got - want).norm() / want.norm().clamp_min(1e-30))


def _row_rel(got, want) -> torch.Tensor:
    """Per query-row relative error over the heads and the latent, float64."""
    got = got.double().reshape(got.shape[0], -1)
    want = want.double().reshape(want.shape[0], -1)
    return (got - want).norm(dim=1) / want.norm(dim=1).clamp_min(1e-30)


def _case(tokens: int, start: int, pages: int, heads: int, seed: int):
    """``(q, bank, table, written, offset, seq_lens, window)`` for one prefill chunk.

    The chunk's ``tokens`` rows sit at window rows ``start .. start + tokens - 1`` and are
    overlaid from ``written``; the window is ``pages`` random pages of a larger bank, so a
    kernel that read the bank in order would read the wrong rows.
    """
    gen = torch.Generator().manual_seed(seed)
    q = (torch.randn(tokens, heads, LATENT, generator=gen) * 0.5).to(torch.bfloat16)
    bank = torch.randn(BANK_PAGES * PAGE, LATENT, generator=gen).to(torch.bfloat16)
    table = torch.randperm(BANK_PAGES, generator=gen)[:pages].reshape(pages, 1).to(torch.int32)
    written = torch.randn(tokens, LATENT, generator=gen).to(torch.bfloat16)
    offset = torch.tensor([[start]], dtype=torch.int32)
    seq_lens = torch.arange(tokens, dtype=torch.int32) + start + 1
    window = bank.reshape(-1, PAGE, LATENT)[table[:, 0].long()].reshape(pages * PAGE, LATENT).clone()
    window[start:start + tokens] = written
    return q, bank, table, written, offset, seq_lens, window


@pytest.fixture(autouse=True)
def _no_neuron_device_node():
    """Fail a test after which this process holds a Neuron device open."""
    yield
    held = open_neuron_device_nodes()
    assert not held, f"a CPU-only test left Neuron device nodes open: {held}"


def test_device_node_guard_sees_an_open_node(monkeypatch, tmp_path):
    """The guard reads this process's descriptors: a held file under the prefix shows up,
    and is gone once closed. A temporary file stands in for a device node."""
    from test.vllm_neuron.functional.attention import neuron_device_nodes as N

    node = tmp_path / "neuron7"
    node.write_bytes(b"")
    monkeypatch.setattr(N, "_DEVICE_PREFIX", os.path.realpath(tmp_path / "neuron"))
    with open(node, "rb"):
        assert N.open_neuron_device_nodes() == [os.path.realpath(node)]
    assert N.open_neuron_device_nodes() == []


#: ``can_run_kernel`` refuses on this switch before it reads the simulator switch, so
#: every DSA selection op takes its torch route and nothing reaches NKI.
_KERNELS_OFF = "VLLM_NEURON_DISABLE_NKI_KERNELS"


@contextlib.contextmanager
def _torch_routes():
    """The selection chain's torch routes, for the duration of the block."""
    saved = os.environ.get(_KERNELS_OFF)
    os.environ[_KERNELS_OFF] = "1"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(_KERNELS_OFF, None)
        else:
            os.environ[_KERNELS_OFF] = saved


def _asbuilt_indices(seq_lens: torch.Tensor, seed: int) -> torch.Tensor:
    """The index rows the as-built prefill chain emits for these rows: bound, top-k,
    sentinel, order, expansion -- on random pool scores over ``max_model_len // 4``
    candidates, which is what the runner's ``max_seq_len`` gives the indexer. Each op runs
    its torch route (:func:`_torch_routes`); each kernel is held to that route in its
    own tests."""
    from vllm_neuron.functional.dsa.causal_bound import dsa_causal_bound, dsa_causal_sentinel
    from vllm_neuron.functional.dsa.index_expand import dsa_index_expand
    from vllm_neuron.functional.dsa.sentinel_order import dsa_sentinel_order
    from vllm_neuron.functional.dsa.topk_select import dsa_topk_select

    gen = torch.Generator().manual_seed(seed)
    rows = int(seq_lens.shape[0])
    candidates = MAX_MODEL_LEN // KPOOL
    scores = torch.randn(rows, candidates, generator=gen, dtype=torch.float32)
    with _torch_routes():
        bounded = dsa_causal_bound(scores, seq_lens.reshape(-1, 1), KPOOL)
        values, ids = dsa_topk_select(bounded, SELECT_K)
        pool_ids = dsa_causal_sentinel(values, ids.to(torch.int32), candidates)
        pool_ids = dsa_sentinel_order(pool_ids)
        return dsa_index_expand(pool_ids, seq_lens, KPOOL)


def _bypass_indices(seq_lens: torch.Tensor) -> torch.Tensor:
    from vllm_neuron.functional.dsa.causal_fill import dsa_causal_fill

    return dsa_causal_fill(seq_lens.to(torch.int32) - 1, WIDTH)


def _oracle(q, window, seq_lens) -> torch.Tensor:
    """float64: each row attends window rows ``0 .. seq_len - 1``."""
    qd, cd = q.double(), window.double()
    out = torch.empty(q.shape, dtype=torch.float64)
    for s in range(q.shape[0]):
        n = int(seq_lens[s])
        keys = cd[:n]
        weights = torch.softmax(torch.einsum("hl,kl->hk", qd[s], keys) * SCALE, dim=-1)
        out[s] = weights @ keys
    return out


def _sparse(q, bank, indices, table, written, offset):
    return MS.mla_sparse_attention(q, bank, indices, SCALE, block_table_row=table,
                                   written=written, write_offset=offset, page_size=PAGE)


def _dense(q, bank, seq_lens, table, written, offset, **kw):
    return DW.mla_dense_window_attention(q, bank, seq_lens, SCALE, block_table_row=table,
                                         written=written, write_offset=offset,
                                         page_size=PAGE, **kw)


def _set_lnc2(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")


# ---------------------------------------------------------------------------------------
# The identity, on the selection chain as built.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("lengths", [
    (1, 2, 3, 4, 5, 127, 128, 129, 1021, 1024, 2047, 2048),
    (2049, 2050, 2051),
])
def test_asbuilt_selection_keeps_the_whole_causal_prefix(lengths):
    """At or under the bound, the as-built chain selects exactly ``0 .. seq_len - 1``.

    The set is a property of the composition, so the chain's torch routes decide it.
    """
    seq_lens = torch.tensor(lengths, dtype=torch.int32)
    idx = _asbuilt_indices(seq_lens, seed=11)
    assert tuple(idx.shape) == (len(lengths), WIDTH)
    for row, n in enumerate(lengths):
        got = idx[row]
        live = got[got >= 0]
        assert int((got < -1).sum()) == 0
        assert live.numel() == n, (n, live.numel())
        assert torch.equal(torch.sort(live).values, torch.arange(n, dtype=live.dtype))


def test_asbuilt_selection_drops_tokens_one_past_the_bound():
    """One token past the bound completes pool ``select_k + 1``; the chain then keeps
    ``select_k`` pools and the tail, so it drops a whole pool: the dense path is no longer
    the as-built answer there, which is why the decision refuses it."""
    seq_lens = torch.tensor([BOUND + 1, BOUND + 5], dtype=torch.int32)
    idx = _asbuilt_indices(seq_lens, seed=12)
    for row, n in enumerate(seq_lens.tolist()):
        live = idx[row][idx[row] >= 0]
        assert live.numel() == SELECT_K * KPOOL + n % KPOOL
        assert live.numel() < n


def test_bound_constant_matches_the_decode_bypass():
    assert BOUND == bypass_max_context(int(_CONFIG.index_topk), KPOOL)


# ---------------------------------------------------------------------------------------
# The kernel against the as-built sparse path.
# ---------------------------------------------------------------------------------------

#: ``(tokens, start, pages, heads)``: the p1 chunk (1024 rows in a 2048-row window), the
#: second chunk of a 2048-token prompt (prefix 1024), a 2048-row single chunk, a short
#: bucket, a chunk whose real length is not a multiple of 4 or 128 and whose start is not
#: pool-aligned, and two heads per rank (TP=32).
CASES = [
    pytest.param(128, 0, 1, 1, id="t128-s0-w128-h1"),
    pytest.param(1024, 0, 16, 1, id="t1024-s0-w2048-h1"),
    pytest.param(1024, 1024, 16, 1, id="t1024-s1024-w2048-h1"),
    pytest.param(2048, 0, 16, 1, id="t2048-s0-w2048-h1"),
    pytest.param(125, 3, 2, 1, id="t125-s3-w256-h1"),
    pytest.param(1021, 1027, 17, 1, id="t1021-s1027-w2176-h1"),
    pytest.param(256, 0, 2, 2, id="t256-s0-w256-h2"),
    pytest.param(130, 1, 3, 2, id="t130-s1-w384-h2"),
]

_SPARSE_CACHE: dict = {}


def _sparse_outputs(key, operands, indices_kind: str, fp32: bool, monkeypatch):
    cache_key = (key, indices_kind, fp32)
    if cache_key not in _SPARSE_CACHE:
        q, bank, table, written, offset, seq_lens, _ = operands
        if indices_kind == "asbuilt":
            indices = _asbuilt_indices(seq_lens, seed=7)
        else:
            indices = _bypass_indices(seq_lens)
        with monkeypatch.context() as m:
            if fp32:
                m.setenv("VLLM_NEURON_MLA_SPARSE_FP32", "1")
            else:
                m.delenv("VLLM_NEURON_MLA_SPARSE_FP32", raising=False)
            _SPARSE_CACHE[cache_key] = _sparse(q, bank, indices, table, written, offset)
    return _SPARSE_CACHE[cache_key]


_DENSE_CACHE: dict = {}


def _dense_output(key, operands):
    if key not in _DENSE_CACHE:
        q, bank, table, written, offset, seq_lens, _ = operands
        _DENSE_CACHE[key] = _dense(q, bank, seq_lens, table, written, offset)
    return _DENSE_CACHE[key]


@pytest.mark.parametrize("tokens,start,pages,heads", CASES)
def test_dense_window_equals_asbuilt_sparse(monkeypatch, tokens, start, pages, heads):
    _set_lnc2(monkeypatch)
    key = (tokens, start, pages, heads)
    operands = _case(tokens, start, pages, heads, seed=tokens * 7 + start * 3 + heads)
    q, *_, seq_lens, window = operands
    assert int(seq_lens.max()) <= BOUND

    dense = _dense_output(key, operands)
    assert dense.dtype == torch.float32 and tuple(dense.shape) == tuple(q.shape)
    assert torch.isfinite(dense).all()

    oracle = _oracle(q, window, seq_lens)
    assert _rel_l2(dense, oracle) <= ORACLE_REL_L2

    for fp32 in (False, True):
        sparse = _sparse_outputs(key, operands, "asbuilt", fp32, monkeypatch)
        rel = _row_rel(dense, sparse)
        share = float((rel <= ROW_REL).double().mean())
        assert share >= ROW_SHARE, (fp32, share, float(rel.max()))
        assert _rel_l2(sparse, oracle) <= ORACLE_REL_L2


@pytest.mark.parametrize("tokens,start,pages,heads", [CASES[0], CASES[4], CASES[6]])
def test_dense_window_equals_the_bypass_form(monkeypatch, tokens, start, pages, heads):
    """The causal-fill rows (what the bypass regime hands the sparse kernel) give the same
    answer: the column order of the selected set does not matter beyond rounding."""
    _set_lnc2(monkeypatch)
    key = (tokens, start, pages, heads)
    operands = _case(tokens, start, pages, heads, seed=tokens * 7 + start * 3 + heads)
    dense = _dense_output(key, operands)
    sparse = _sparse_outputs(key, operands, "bypass", False, monkeypatch)
    rel = _row_rel(dense, sparse)
    assert float((rel <= ROW_REL).double().mean()) >= ROW_SHARE, float(rel.max())


def test_one_program_equals_two(monkeypatch):
    """The two-program launch splits query tiles only: the same arithmetic per row."""
    _set_lnc2(monkeypatch)
    operands = _case(512, 0, 4, 1, seed=5)
    q, bank, table, written, offset, seq_lens, _ = operands
    DW.reset_mla_dense_window_dispatch_counters()
    two = _dense(q, bank, seq_lens, table, written, offset)
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG")
    one = _dense(q, bank, seq_lens, table, written, offset)
    assert DW.mla_dense_window_route_counts() == (2, 1)
    torch.testing.assert_close(one, two, rtol=0.0, atol=0.0)


def test_stale_window_rows_do_not_reach_the_softmax(monkeypatch):
    """Rows past a query's context are other pages or stale slots; huge values there must
    change nothing, because masked scores are written by predicate, not by a bias."""
    _set_lnc2(monkeypatch)
    q, bank, table, written, offset, seq_lens, _ = _case(128, 0, 2, 1, seed=9)
    base = _dense(q, bank, seq_lens, table, written, offset)
    poisoned = bank.clone()
    second_page = int(table[1, 0])
    poisoned[second_page * PAGE:(second_page + 1) * PAGE] = 3.0e38
    got = _dense(q, poisoned, seq_lens, table, written, offset)
    torch.testing.assert_close(got, base, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------------------
# What the seam refuses.
# ---------------------------------------------------------------------------------------


def test_refuses_a_row_past_its_staged_rows():
    q, bank, table, written, offset, seq_lens, _ = _case(128, 0, 1, 1, seed=3)
    with pytest.raises(DW.MlaDenseWindowError, match="seq_lens"):
        _dense(q, bank, seq_lens + 1, table, written, offset)


def test_refuses_fp32_operands():
    q, bank, table, written, offset, seq_lens, _ = _case(128, 0, 1, 1, seed=3)
    with pytest.raises(DW.MlaDenseWindowError, match="2-byte"):
        _dense(q.float(), bank.float(), seq_lens, table, written.float(), offset)


def test_refuses_more_key_rows_than_the_kernel_holds():
    q, _, _, written, offset, seq_lens, _ = _case(128, 0, 1, 1, seed=3)
    wide = torch.arange(DW.MAX_KEY_ROWS // PAGE + 1, dtype=torch.int32).reshape(-1, 1)
    with pytest.raises(DW.MlaDenseWindowError, match="key rows"):
        DW.mla_dense_window_attention(q, torch.zeros(wide.shape[0] * PAGE, LATENT,
                                                     dtype=torch.bfloat16),
                                      seq_lens, SCALE, block_table_row=wide,
                                      written=written, write_offset=offset, page_size=PAGE)


def test_serves_matches_the_seam_refusals():
    """The trace-time gate accepts exactly the geometry the seam accepts."""
    bf, fp = torch.bfloat16, torch.float16
    rows = DW.MAX_KEY_ROWS
    assert DW.dense_window_serves(bf, bf, 1, LATENT, rows)
    assert DW.dense_window_serves(fp, fp, DW.ROW_TILE, DW.LATENT_TILE, 1)
    assert not DW.dense_window_serves(bf, bf, 1, LATENT, rows + 1)
    assert not DW.dense_window_serves(bf, bf, 1, LATENT, 0)
    assert not DW.dense_window_serves(torch.float32, torch.float32, 1, LATENT, rows)
    assert not DW.dense_window_serves(bf, fp, 1, LATENT, rows)
    assert not DW.dense_window_serves(bf, bf, DW.ROW_TILE + 1, LATENT, rows)
    assert not DW.dense_window_serves(bf, bf, 1, LATENT - 12, rows)
    assert not DW.dense_window_serves(bf, bf, 1, DW.LATENT_MAX + DW.LATENT_TILE, rows)
