# SPDX-License-Identifier: Apache-2.0
"""The worker's KV budget prices the deeper indexer rings of a speculative server.

An mtp server's runner allocates the two DSA indexer rings (``tail``, ``pad_tail``;
``NeuronModelRunner._glm5next_side_caches``) ``R = indexer_ring_depth(index_kpool, k)``
rows deep instead of ``index_kpool`` (the depth table: 8 for k in 2..5 against a pool
of 4), so a slot's side caches grow by ``2 rings x 2 x (R - index_kpool) x
index_head_dim x 2 B`` per sparse layer. ``neuron_worker._indexer_side_cache_bytes``
prices the side caches for the fit check (``_kv_cache_footprint_bytes``,
``determine_available_memory``) and must hand the pricer the server's ``k``
(``num_speculative_tokens`` of the runner's ``method == "mtp"`` speculative config, 0 otherwise),
or the check under-prices exactly that growth.

Expectations are derived from the fixture's layer count, the config's indexer dials and
the ring depth the runner's own helper returns. The depth helper lives in
``vllm_neuron.functional.dsa.decode_trow``; on a tree without it the plumbing is
pinned with a fixed stand-in depth and the served-figure test is a strict xfail.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_kv_budget.py
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
from vllm.config import SpeculativeConfig

from vllm_neuron.vllm.worker import neuron_worker
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner, indexer_side_cache_bytes

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv
from test.vllm_neuron.worker import test_kv_budget_side_caches as side

K = 3
SEQS = 64
LENGTH = 8192
MIB = 1024**2
RING_HELPER_PRESENT = (
    importlib.util.find_spec("vllm_neuron.functional.dsa.decode_trow") is not None
)
SEAM = pytest.mark.xfail(
    condition=not RING_HELPER_PRESENT,
    reason="indexer_ring_depth is not in this tree; the runner's "
    "ring helper cannot price a speculative server here",
    strict=True,
)


def _spec(method: str, k: int) -> SimpleNamespace:
    """A served speculative config of ``method`` with ``k`` drafts. ``use_eagle`` (read by
    the runner's KV spec) answers as vLLM's own rule does for the method."""
    spec = SimpleNamespace(method=method, num_speculative_tokens=k)
    spec.use_eagle = lambda: SpeculativeConfig.use_eagle(spec)
    return spec


def _mtp(runner, k: int) -> None:
    """Give the runner an mtp speculative config of ``k`` drafts (the worker reads the
    served speculative config off ``model_runner.speculative_config``, as it does for
    the draft-token budget)."""
    runner.speculative_config = _spec("mtp", k)


def _ring_growth_bytes(*, slots: int, ring_rows: int) -> int:
    """``tail`` + ``pad_tail`` growth over the pool-deep rings: two rings of
    ``[slots, 2, ring_rows, index_head_dim]`` bf16 per sparse layer."""
    extra_rows = ring_rows - side.INDEX_KPOOL
    return kv.MLA_LAYERS * slots * 2 * 2 * extra_rows * side.INDEX_HEAD_DIM * side.BF16_BYTES


def _fixed_depth(monkeypatch, ring_rows: int) -> None:
    """Stand in for the ring-depth helper: ``ring_rows`` when drafting, the pool otherwise."""
    monkeypatch.setattr(
        NeuronModelRunner,
        "_glm5next_indexer_ring_rows",
        staticmethod(lambda pool, k: int(ring_rows) if int(k) > 0 else int(pool)),
    )


def _priced(worker) -> int:
    return neuron_worker._indexer_side_cache_bytes(worker.vllm_config, worker.model_runner)


def test_a_plain_server_prices_the_pool_deep_rings() -> None:
    worker = side._worker(seqs=SEQS, length=LENGTH)
    assert worker.model_runner.speculative_config is None
    assert _priced(worker) == side.hand_side_cache_bytes(seqs=SEQS, length=LENGTH)


def test_an_mtp_server_prices_the_rings_the_runner_allocates(monkeypatch) -> None:
    ring_rows = 2 * side.INDEX_KPOOL
    _fixed_depth(monkeypatch, ring_rows)
    worker = side._worker(seqs=SEQS, length=LENGTH)
    plain = _priced(worker)
    _mtp(worker.model_runner, K)
    speculative = _priced(worker)
    assert speculative - plain == _ring_growth_bytes(slots=SEQS, ring_rows=ring_rows)
    # The same price the pricer gives when handed k directly: the caller forwards k, nothing else.
    runner = worker.model_runner
    assert speculative == indexer_side_cache_bytes(
        runner.get_kv_cache_spec(), runner.model.text_config,
        max_seq_len=LENGTH, request_slots=SEQS, speculative_tokens=K,
    )


def test_a_speculative_config_of_another_method_prices_the_plain_rings(monkeypatch) -> None:
    _fixed_depth(monkeypatch, 2 * side.INDEX_KPOOL)
    worker = side._worker(seqs=SEQS, length=LENGTH)
    plain = _priced(worker)
    worker.model_runner.speculative_config = _spec("ngram", K)
    assert _priced(worker) == plain


def test_the_fit_check_counts_the_deeper_rings(monkeypatch) -> None:
    """``_kv_cache_footprint_bytes`` (what ``determine_available_memory`` compares to the
    budget) grows by exactly the ring growth when a drafting server's rings deepen. The
    same ``k`` also adds the recurrent layers' checkpoint rows (``MambaSpec
    .num_speculative_blocks``, priced by vLLM's own block sizing); both footprints here
    carry those, so the difference is the rings alone."""
    worker = side._worker(seqs=SEQS, length=LENGTH)
    _mtp(worker.model_runner, K)
    need = worker._kv_cache_need_bytes()
    _fixed_depth(monkeypatch, side.INDEX_KPOOL)
    pool_deep = worker._kv_cache_footprint_bytes(need)
    ring_rows = 2 * side.INDEX_KPOOL
    _fixed_depth(monkeypatch, ring_rows)
    assert worker._kv_cache_footprint_bytes(need) - pool_deep == _ring_growth_bytes(
        slots=SEQS, ring_rows=ring_rows
    )


@SEAM
def test_the_served_bs64_line_prices_the_tables_growth_at_k_3() -> None:
    """With the depth table (8 rows at k = 3 against a pool of 4) the bs=64 line's
    fit check prices 2.75 MiB more per rank; per slot the figure the table states
    at 65 slots (+2.79 MiB)."""
    from vllm_neuron.functional.dsa.decode_trow import indexer_ring_depth

    worker = side._worker(seqs=SEQS, length=LENGTH)
    plain = _priced(worker)
    _mtp(worker.model_runner, K)
    growth = _priced(worker) - plain
    ring_rows = NeuronModelRunner._glm5next_indexer_ring_rows(side.INDEX_KPOOL, K)
    assert ring_rows == indexer_ring_depth(side.INDEX_KPOOL, K) > side.INDEX_KPOOL
    assert growth == _ring_growth_bytes(slots=SEQS, ring_rows=ring_rows)
    per_slot = growth / SEQS
    assert round(65 * per_slot / MIB, 2) == 2.79
