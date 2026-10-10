# SPDX-License-Identifier: Apache-2.0
"""The runner's block table holds every block the scheduler hands a drafting request.

``NeuronModelRunner.initialize_kv_cache`` builds vLLM's ``InputBatch``, whose
``MultiGroupBlockTable`` sizes each KV cache group's row at ``cdiv(max_model_len,
block_size)`` blocks unless told otherwise. A speculative server's recurrent
(``MambaSpec``) groups hand each request ``1 + num_speculative_blocks`` blocks -- the
blocks the KV need prices and vLLM's admission counts -- so a recurrent row must be
``num_speculative_blocks`` wider, or the first request's ``add_row`` overflows it:
"could not broadcast input array from shape (4,) into shape (1,)" with 3 drafts and
a recurrent block the length of the sequence (``gpu_input_batch.py:379`` under
``_update_states``). The GPU runner passes ``max_num_blocks_per_req`` per group for
this; the Neuron runner must size its rows the same way.

Expectations are derived from vLLM's own grouping, admission and per-spec memory
figures on the fixture's specs; no served figure is an expectation.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_block_table_drafter_seam.py
"""
from __future__ import annotations

import pytest

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv
from test.vllm_neuron.worker.test_kv_budget_drafter_seam import K, _layers
from test.vllm_neuron.worker.test_mtp_kv_budget import _mtp

#: Two requests of 512 tokens: four attention blocks per request, small CPU buffers.
SEQS, LENGTH = 2, 512


def _runner_with_its_input_batch(k: int, monkeypatch):
    """The fixture's drafter layout at ``k`` drafts, taken through vLLM's own admission
    to a ``KVCacheConfig`` and through the real ``initialize_kv_cache`` (real buffers,
    real ``InputBatch``)."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

    from vllm_neuron.vllm.worker import neuron_model_runner as module

    layers, text_config = _layers(drafter=True)
    runner = kv.fake_runner(
        layers, max_num_seqs=SEQS, max_model_len=LENGTH, text_config=text_config
    )
    if k:
        _mtp(runner, k)
    worker = kv.fake_worker(runner)
    (config,) = get_kv_cache_configs(
        worker.vllm_config, [runner.get_kv_cache_spec()], [worker._kv_cache_need_bytes()]
    )
    monkeypatch.setattr(module, "has_kv_transfer_group", lambda: False)
    module.NeuronModelRunner.initialize_kv_cache(runner, config)
    return runner, worker, config


def _blocks_per_request(vllm_config, config) -> tuple[list[int], ...]:
    """One request's block ids per group, as many as vLLM's own per-spec figure
    (``max_memory_usage_bytes`` over the page) says a request holds in that group."""
    ids = []
    for group in config.kv_cache_groups:
        spec = group.kv_cache_spec
        held = spec.max_memory_usage_bytes(vllm_config) // spec.page_size_bytes
        ids.append(list(range(1, held + 1)))
    return tuple(ids)


@pytest.mark.parametrize("k", [0, K], ids=["plain", "spec3"])
def test_every_groups_row_takes_the_blocks_a_request_holds(k: int, monkeypatch) -> None:
    """Each request slot's row in every group takes the blocks vLLM says a request
    holds there (the sequence's pages; ``1 + k`` for a recurrent group), and each
    row is exactly that wide. k = 0 is the control that fits today."""
    runner, _, config = _runner_with_its_input_batch(k, monkeypatch)
    table = runner.input_batch.block_table
    blocks = _blocks_per_request(runner.vllm_config, config)
    assert len(table.block_tables) == len(config.kv_cache_groups) == len(blocks)
    for row in range(SEQS):
        table.add_row(blocks, row)
    for group_table, ids in zip(table.block_tables, blocks):
        assert group_table.block_table.np[SEQS - 1, : len(ids)].tolist() == ids
        assert group_table.max_num_blocks_per_req == len(ids)


@pytest.mark.parametrize("k", [0, K], ids=["plain", "spec3"])
def test_the_input_batch_admits_a_request_the_way_the_engine_adds_one(k: int, monkeypatch) -> None:
    """``_update_states`` turns the scheduler's ``NewRequestData`` into a
    ``CachedRequestState`` and calls the real ``InputBatch.add_request`` -- the frame
    that overflowed on the served line. The same call, on the InputBatch the real
    ``initialize_kv_cache`` built, with the per-group block ids the scheduler hands a
    drafting request, registers the request and files its blocks."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    runner, _, config = _runner_with_its_input_batch(k, monkeypatch)
    blocks = _blocks_per_request(runner.vllm_config, config)
    request = CachedRequestState(
        req_id="drafting-0",
        prompt_token_ids=list(range(LENGTH // 2)),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=blocks,
        num_computed_tokens=0,
        output_token_ids=[],
    )
    index = runner.input_batch.add_request(request)
    assert runner.input_batch.req_id_to_index["drafting-0"] == index
    for group_table, ids in zip(runner.input_batch.block_table.block_tables, blocks):
        assert group_table.num_blocks_per_row[index] == len(ids)
        assert group_table.block_table.np[index, : len(ids)].tolist() == ids


def test_the_block_table_and_the_kv_need_price_a_request_alike(monkeypatch) -> None:
    """One arithmetic, two callers: the need-sized KV pool prices a request at the sum
    of the per-group row widths the block table was built with (plus the pool's null
    block), so the pool vLLM admits and the rows the engine fills agree."""
    runner, worker, config = _runner_with_its_input_batch(K, monkeypatch)
    from vllm.v1.core.kv_cache_utils import get_uniform_page_size

    rows = [table.max_num_blocks_per_req for table in runner.input_batch.block_table.block_tables]
    page = get_uniform_page_size([group.kv_cache_spec for group in config.kv_cache_groups])
    layers_per_pool = max(len(group.layer_names) for group in config.kv_cache_groups)
    assert worker._kv_cache_need_bytes() == (sum(rows) * SEQS + 1) * page * layers_per_pool
