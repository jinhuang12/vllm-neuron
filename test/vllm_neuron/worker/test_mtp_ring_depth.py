# SPDX-License-Identifier: Apache-2.0
"""The indexer ring's depth on a speculative server.

A verify step of ``T = 1 + k`` rows writes ``T`` ring rows ahead of the request's
position; on the 4-row ring a row of an accepted position is overwritten once
``T >= 3`` (worker-58's proof, team-lead ruling 10:20Z). The depth comes from
worker-58's one helper, ``functional.dsa.decode_trow.indexer_ring_depth(index_kpool,
num_speculative_tokens)``, and the runner allocates ``tail`` and ``pad_tail`` as
``[slots, 2, R, index_head_dim]`` with it; a server that drafts nothing keeps today's
``index_kpool`` rows, bit-identical, without importing the helper. The KV budget's
side-cache price follows the same depth. The helper is worker-58's and is stood in
for here by a module stub.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_ring_depth.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner, indexer_side_cache_bytes

HELPER_MODULE = "vllm_neuron.functional.dsa.decode_trow"
POOL = 4
WIDTH = 8
SLOTS = 2
MAX_SEQ_LEN = 16
K = 3


@pytest.fixture
def helper(monkeypatch):
    """worker-58's helper, stood in: twice the pool on a drafting server."""
    monkeypatch.setitem(
        sys.modules, HELPER_MODULE,
        SimpleNamespace(indexer_ring_depth=lambda pool, k: 2 * int(pool) if k > 0 else int(pool)),
    )


def test_a_plain_server_keeps_the_pool_deep_ring_without_the_helper(monkeypatch):
    monkeypatch.delitem(sys.modules, HELPER_MODULE, raising=False)
    assert NeuronModelRunner._glm5next_indexer_ring_rows(POOL, 0) == POOL


def test_a_speculative_server_takes_the_helpers_depth(helper):
    assert NeuronModelRunner._glm5next_indexer_ring_rows(POOL, K) == 2 * POOL


def _banks():
    return [
        {"family": "self_attn", "name": "layers.0", "latent_cache": torch.zeros((1,), dtype=torch.bfloat16)},
        {"family": "linear_attn", "name": "layers.1"},
    ]


def test_the_side_caches_allocate_the_ring_at_the_given_depth():
    side = NeuronModelRunner._glm5next_side_caches(
        _banks(), index_kpool=POOL, index_head_dim=WIDTH, max_seq_len=MAX_SEQ_LEN,
        request_slots=SLOTS, ring_rows=2 * POOL,
    )
    assert tuple(side[0]["tail"].shape) == (SLOTS, 2, 2 * POOL, WIDTH)
    assert tuple(side[0]["pad_tail"].shape) == (SLOTS, 2, 2 * POOL, WIDTH)
    assert tuple(side[0]["pool_cache"].shape) == (SLOTS, MAX_SEQ_LEN // POOL + 1, WIDTH)
    assert side[1] == {}


def test_the_default_depth_is_the_pool():
    side = NeuronModelRunner._glm5next_side_caches(
        _banks(), index_kpool=POOL, index_head_dim=WIDTH, max_seq_len=MAX_SEQ_LEN, request_slots=SLOTS,
    )
    assert tuple(side[0]["tail"].shape) == (SLOTS, 2, POOL, WIDTH)


def test_a_depth_below_the_pool_is_refused_by_name():
    with pytest.raises(ValueError, match="ring"):
        NeuronModelRunner._glm5next_side_caches(
            _banks(), index_kpool=POOL, index_head_dim=WIDTH, max_seq_len=MAX_SEQ_LEN,
            request_slots=SLOTS, ring_rows=POOL - 1,
        )


def test_the_live_set_of_a_speculative_runner_is_as_deep_as_the_helper_says(helper):
    runner = SimpleNamespace(
        is_mtp_spec=True,
        drafter=SimpleNamespace(num_speculative_tokens=K),
        model=SimpleNamespace(text_config=SimpleNamespace(index_kpool=POOL, index_head_dim=WIDTH)),
        max_model_len=MAX_SEQ_LEN,
        max_num_reqs=SLOTS,
        _glm5next_request_slot_capacity=lambda banks: SLOTS,
        # The class's own allocator and depth helpers, bound to the stub.
        _glm5next_side_caches=NeuronModelRunner._glm5next_side_caches,
        _glm5next_indexer_ring_rows=NeuronModelRunner._glm5next_indexer_ring_rows,
    )
    runner._glm5next_speculative_tokens = (
        lambda: NeuronModelRunner._glm5next_speculative_tokens(runner)
    )
    side = NeuronModelRunner._glm5next_live_side_caches(runner, _banks())
    assert side[0]["tail"].shape[2] == 2 * POOL
    assert runner._glm5next_checkpoint_rows == {}


def test_the_budget_prices_the_deeper_ring(helper):
    spec = {"layers.0": FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=WIDTH, dtype=torch.bfloat16)}
    text_config = SimpleNamespace(index_kpool=POOL, index_head_dim=WIDTH)
    plain = indexer_side_cache_bytes(spec, text_config, max_seq_len=MAX_SEQ_LEN, request_slots=SLOTS)
    deep = indexer_side_cache_bytes(
        spec, text_config, max_seq_len=MAX_SEQ_LEN, request_slots=SLOTS, speculative_tokens=K,
    )
    # Two rings (tail, pad_tail), each [slots, 2, R, width] bf16: the extra rows' bytes.
    assert deep - plain == 2 * SLOTS * 2 * (2 * POOL - POOL) * WIDTH * torch.finfo(torch.bfloat16).bits // 8
