# SPDX-License-Identifier: Apache-2.0
"""The DSA pool store: one buffer per sparse layer, built with prefix caching on only.

A sparse (DSA) layer's indexer keeps its pooled keys per request slot (``pool_cache``,
one row per ``index_kpool`` tokens). Prefix caching needs those rows per KV block as
well, and this store is their buffer: ``runner.dsa_pool_store[layer_name]``, a
``[num_blocks, block_size // index_kpool, index_head_dim]`` buffer in the latent
cache's dtype, indexed by the block id of the layer's latent group. Block ``b`` is for
the pool rows of tokens ``b * block_size .. (b + 1) * block_size - 1``. Nothing in this
tree reads or writes it.

The store is outside vLLM's KV cache tensors, so the worker prices it beside them, in
the footprint that has to fit the device, from the builder the runner allocates with.
It is not in the need the worker returns: vLLM turns that into blocks.

Read here, on the real ``initialize_kv_cache`` with the served TP 64 specs (no model
weights, small pool), and on the worker's budget path:

1. prefix caching off: no store, and no byte priced for it;
2. on: one zeroed, contiguous buffer per sparse layer, in its own storage, the
   KV cache buffers and the bound caches unchanged;
3. the worker's price is the runner's allocation, and the footprint is the KV cache
   tensors, the side caches and the store;
4. a block size that ``index_kpool`` does not divide is refused by name, and a model
   whose text config declares no indexer has no store.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/vllm/worker/test_dsa_pool_store.py
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv
from vllm_neuron.vllm.worker.neuron_worker import (
    _dsa_pool_store_bytes,
    _indexer_side_cache_bytes,
)

pytestmark = [pytest.mark.fast]

MAX_NUM_SEQS, MAX_MODEL_LEN = 2, 512


@pytest.fixture(scope="module")
def prefix_caching_point():
    """A runner fake with prefix caching on (128-token KDA blocks), its worker and KV config."""
    layers, _ = kv.glm53f_layer_specs()
    runner = kv.fake_runner(layers, max_num_seqs=MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN,
                            gate_knobs=False)
    assert runner.vllm_config.cache_config.enable_prefix_caching is True
    worker = kv.fake_worker(runner)
    return runner, worker, kv.served_kv_cache_config(runner, worker)


def _initialize(monkeypatch, runner, config, *, prefix_caching: bool):
    """The real ``initialize_kv_cache`` with the cache config's flag as given."""
    from vllm_neuron.vllm.worker import neuron_model_runner as module

    monkeypatch.setattr(module, "InputBatch", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(module, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(runner.vllm_config.cache_config, "enable_prefix_caching",
                        prefix_caching)
    runner.model.bound.clear()
    caches = module.NeuronModelRunner.initialize_kv_cache(runner, config)
    return caches, runner.dsa_pool_store


def _sparse_specs(config):
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    return {name: group.kv_cache_spec for group in config.kv_cache_groups
            for name in group.layer_names if isinstance(group.kv_cache_spec, MLAAttentionSpec)}


def _form(caches):
    return {
        name: [(tuple(t.shape), t.stride(), t.dtype, t.storage_offset(),
                t.untyped_storage().nbytes()) for t in tensors]
        for name, tensors in caches.items()
    }


def _buffers(caches):
    return {t.untyped_storage().data_ptr() for tensors in caches.values() for t in tensors}


def test_with_prefix_caching_off_no_store_is_built_or_priced(monkeypatch, prefix_caching_point):
    runner, worker, config = prefix_caching_point
    _, store = _initialize(monkeypatch, runner, config, prefix_caching=False)
    need = worker._kv_cache_need_bytes(log=False)

    assert store == {}
    assert _dsa_pool_store_bytes(worker.vllm_config, runner, need, config) == 0
    assert _dsa_pool_store_bytes(worker.vllm_config, runner, need) == 0


def test_each_sparse_layer_gets_a_zeroed_block_indexed_store_of_its_own(
    monkeypatch, prefix_caching_point
):
    runner, _, config = prefix_caching_point
    caches, store = _initialize(monkeypatch, runner, config, prefix_caching=True)
    text = runner.model.text_config
    sparse = _sparse_specs(config)

    assert set(store) == set(sparse) and len(store) == kv.MLA_LAYERS
    rows = kv.BLOCK_SIZE_TOKENS // int(text.index_kpool)
    for name, tensor in store.items():
        assert tuple(tensor.shape) == (config.num_blocks, rows, int(text.index_head_dim))
        assert tensor.dtype == sparse[name].dtype == torch.bfloat16
        assert tensor.device == torch.device(runner.device)
        assert tensor.is_contiguous() and not tensor.any()
        # One row per index_kpool tokens of block b, 8192 bytes per block at TP 64.
        assert tensor[0].nbytes == rows * int(text.index_head_dim) * 2 == 8192
    storages = [tensor.untyped_storage().data_ptr() for tensor in store.values()]
    assert len(set(storages)) == len(storages)
    assert not set(storages) & _buffers(caches)


def test_the_store_leaves_the_kv_cache_buffers_and_the_bound_caches_as_they_are(
    monkeypatch, prefix_caching_point
):
    runner, _, config = prefix_caching_point
    off, _ = _initialize(monkeypatch, runner, config, prefix_caching=False)
    on, store = _initialize(monkeypatch, runner, config, prefix_caching=True)

    assert store
    assert _form(on) == _form(off)
    assert runner.model.bound == [on]


def test_the_worker_prices_the_store_the_runner_allocates(monkeypatch, prefix_caching_point):
    runner, worker, config = prefix_caching_point
    _, store = _initialize(monkeypatch, runner, config, prefix_caching=True)
    allocated = sum(tensor.nbytes for tensor in store.values())
    need = worker._kv_cache_need_bytes(log=False)

    assert _dsa_pool_store_bytes(worker.vllm_config, runner, need, config) == allocated
    assert _dsa_pool_store_bytes(worker.vllm_config, runner, need) == allocated
    assert allocated == kv.MLA_LAYERS * config.num_blocks * 8192
    assert worker._kv_cache_footprint_bytes(need) == (
        sum(worker._kv_cache_allocation_sizes(need))
        + _indexer_side_cache_bytes(worker.vllm_config, runner)
        + allocated
    )


def test_with_prefix_caching_off_the_footprint_has_no_store(monkeypatch) -> None:
    layers, _ = kv.glm53f_layer_specs()
    runner = kv.fake_runner(layers, max_num_seqs=MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN)
    assert runner.vllm_config.cache_config.enable_prefix_caching is False
    worker = kv.fake_worker(runner)
    need = worker._kv_cache_need_bytes(log=False)

    assert worker._kv_cache_footprint_bytes(need) == (
        sum(worker._kv_cache_allocation_sizes(need))
        + _indexer_side_cache_bytes(worker.vllm_config, runner)
    )


def test_a_block_size_that_index_kpool_does_not_divide_is_refused(prefix_caching_point):
    from vllm_neuron.vllm.worker.neuron_model_runner import dsa_pool_store_shapes

    _, _, config = prefix_caching_point
    text = SimpleNamespace(index_kpool=3, index_head_dim=128)

    with pytest.raises(ValueError, match=r"block size 128, which index_kpool 3 does not divide"):
        dsa_pool_store_shapes(config, text)


def test_a_model_whose_text_config_declares_no_indexer_has_no_store(prefix_caching_point):
    from vllm_neuron.vllm.worker.neuron_model_runner import dsa_pool_store_shapes

    _, _, config = prefix_caching_point

    assert dsa_pool_store_shapes(config, SimpleNamespace()) == {}
    assert dsa_pool_store_shapes(config, None) == {}


@pytest.mark.parametrize("prefix_caching", [False, True], ids=["off", "on"])
def test_the_budget_line_names_the_store_only_where_prefix_caching_builds_it(
    monkeypatch, caplog, prefix_caching
) -> None:
    """With prefix caching off the budget line reads as before the store existed."""
    # The device budget path, as served: none of the CPU-mode or budget knobs set.
    for name in ("VLLM_NEURON_CPU_MODE", "VLLM_NEURON_CPU_COMPILE",
                 "VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB",
                 "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION"):
        monkeypatch.delenv(name, raising=False)
    layers, _ = kv.glm53f_layer_specs()
    runner = kv.fake_runner(layers, max_num_seqs=MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN,
                            gate_knobs=not prefix_caching)
    assert runner.vllm_config.cache_config.enable_prefix_caching is prefix_caching
    worker = kv.fake_worker(runner)
    side = _indexer_side_cache_bytes(worker.vllm_config, runner)
    store = kv.MLA_LAYERS * kv.served_kv_cache_config(runner, worker).num_blocks * 8192

    with caplog.at_level(logging.INFO):
        worker.determine_available_memory()

    (line,) = [r.getMessage() for r in caplog.records if "KV cache budget in" in r.getMessage()]
    if prefix_caching:
        assert (f"(DSA indexer side caches {side / kv.GIB:.3f} GiB, DSA pool store "
                f"{store / kv.GIB:.3f} GiB), admitted=") in line
    else:
        assert f"(DSA indexer side caches {side / kv.GIB:.3f} GiB), admitted=" in line
        assert "pool store" not in line
