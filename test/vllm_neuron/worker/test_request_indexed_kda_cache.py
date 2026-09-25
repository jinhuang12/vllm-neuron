# SPDX-License-Identifier: Apache-2.0
"""Physical request-state allocation, independent of scheduler token pages.

These tests use real CPU storage and the production allocation/footprint paths.
They do not certify captured device writes or concurrent serving behavior.
"""

from copy import deepcopy
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm_neuron.model.kv_cache import LayerSpec
from vllm_neuron.vllm.worker import neuron_model_runner as runner_module
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

CONV_SHAPE = (3, 12)
RECURRENT_SHAPE = (2, 4, 4)
RECURRENT_NAMES = ("layers.0.attn", "layers.2.attn")
LATENT_NAME = "layers.1.attn"
BLOCK_SIZE = 16


def _fixture(*, capacity, num_blocks=17, opt_in=True):
    latent = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
        sliding_window=None,
        attention_chunk_size=None,
    )
    recurrent = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(CONV_SHAPE, RECURRENT_SHAPE),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=latent.page_size_bytes,
    )
    layers = [
        LayerSpec(
            name=name,
            num_kv_heads=1,
            head_size=32,
            dtype=torch.bfloat16,
            kda_conv_state_shape=CONV_SHAPE,
            kda_recurrent_state_shape=RECURRENT_SHAPE,
            kda_conv_state_dtype=torch.bfloat16,
            kda_recurrent_state_dtype=torch.float32,
        )
        for name in RECURRENT_NAMES
    ]
    layers.append(LayerSpec(LATENT_NAME, 1, 32, torch.bfloat16, latent_kv=True))
    specs = {name: recurrent for name in RECURRENT_NAMES}
    specs[LATENT_NAME] = latent
    model = SimpleNamespace(get_kv_spec=lambda: SimpleNamespace(layers=layers))
    if opt_in:
        model.request_indexed_kda_state = True
    # Recurrent layers sharing a scheduler pool still need private state banks.
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * latent.page_size_bytes,
                shared_by=[LATENT_NAME, *RECURRENT_NAMES],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)
            for name, spec in specs.items()
        ],
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(block_size=BLOCK_SIZE, cache_dtype="auto"),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            scheduler_config=SimpleNamespace(max_num_seqs=capacity),
        ),
        neuron_config=SimpleNamespace(fp8_packed_kv=False),
        speculative_config=None,
        drafter=None,
        device=torch.device("cpu"),
        max_num_reqs=capacity,
        max_model_len=num_blocks * BLOCK_SIZE,
        max_num_batched_tokens=256,
        vocab_size=128,
        is_pooling_model=False,
        model=model,
        get_kv_cache_spec=lambda: specs,
        _kv_cache_full_tensors={},
    )
    runner._kv_cache_is_fp8_packed = MethodType(
        runner_module.NeuronModelRunner._kv_cache_is_fp8_packed, runner
    )
    runner._k_cache_alloc_shape = runner_module.NeuronModelRunner._k_cache_alloc_shape
    return runner, config, layers


def _allocate(runner, config, monkeypatch):
    bound = []
    runner.model.bind_kv_cache = bound.append
    monkeypatch.setattr(runner_module, "InputBatch", lambda **kwargs: kwargs)
    monkeypatch.setattr(runner_module, "has_kv_transfer_group", lambda: False)
    caches = runner_module.NeuronModelRunner.initialize_kv_cache(runner, config)
    assert bound == [caches]
    return caches


def _footprint(runner, config, monkeypatch):
    from vllm.v1.core import kv_cache_utils

    # Use the same scheduler configuration in the worker and allocation paths.
    # Existing test_kv_cache_budget.py covers the vendor's block-budget solver.
    monkeypatch.setattr(
        kv_cache_utils, "get_kv_cache_groups", lambda *_: config.kv_cache_groups
    )
    monkeypatch.setattr(
        kv_cache_utils, "get_kv_cache_config_from_groups", lambda *_: config
    )
    worker = SimpleNamespace(vllm_config=runner.vllm_config, model_runner=runner)
    return NeuronWorker._kv_cache_footprint_bytes(worker, 1)


def _owned_bytes(caches):
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for carriers in caches.values()
        for tensor in carriers
    }
    return sum(storages.values())


@pytest.mark.parametrize("capacity", [1, 2, 8])
def test_physical_storage_matches_worker_footprint(capacity, monkeypatch):
    runner, config, _ = _fixture(capacity=capacity)
    before = deepcopy(config)
    caches = _allocate(runner, config, monkeypatch)
    page = config.kv_cache_groups[0].kv_cache_spec.page_size_bytes

    expected = config.num_blocks * page + len(RECURRENT_NAMES) * capacity * page
    assert _owned_bytes(caches) == _footprint(runner, config, monkeypatch) == expected
    assert config == before  # Physical compaction must not change scheduler specs.
    assert caches[LATENT_NAME][0].shape == (config.num_blocks, 1, BLOCK_SIZE, 32)
    assert len(caches[LATENT_NAME]) == 1
    for name in RECURRENT_NAMES:
        conv, recurrent = caches[name]
        assert conv.shape == (capacity, *CONV_SHAPE)
        assert recurrent.shape == (capacity, *RECURRENT_SHAPE)
        assert conv.untyped_storage().nbytes() == capacity * page
        assert (
            conv.untyped_storage().data_ptr() == recurrent.untyped_storage().data_ptr()
        )


@pytest.mark.parametrize("capacity", [1, 2, 8])
def test_context_growth_changes_only_latent_storage(capacity, monkeypatch):
    observations = []
    for num_blocks in (17, 65):
        runner, config, _ = _fixture(capacity=capacity, num_blocks=num_blocks)
        caches = _allocate(runner, config, monkeypatch)
        observations.append(
            (
                caches[LATENT_NAME][0].untyped_storage().nbytes(),
                [
                    caches[name][0].untyped_storage().nbytes()
                    for name in RECURRENT_NAMES
                ],
            )
        )
    assert observations[0][1] == observations[1][1]
    assert observations[1][0] * 17 == observations[0][0] * 65


@pytest.mark.parametrize("capacity", [1, 2, 8])
def test_preserves_dtypes_padded_strides_and_isolated_slots(capacity, monkeypatch):
    runner, config, _ = _fixture(capacity=capacity)
    caches = _allocate(runner, config, monkeypatch)
    page = config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
    pointers = {caches[LATENT_NAME][0].untyped_storage().data_ptr()}
    for layer_index, name in enumerate(RECURRENT_NAMES):
        conv, recurrent = caches[name]
        pointer = conv.untyped_storage().data_ptr()
        assert pointer not in pointers
        pointers.add(pointer)
        assert conv.dtype == torch.bfloat16
        assert recurrent.dtype == torch.float32
        assert conv.stride() == (page // 2, *torch.empty(CONV_SHAPE).stride())
        assert recurrent.stride() == (page // 4, *torch.empty(RECURRENT_SHAPE).stride())
        assert conv.storage_offset() == 0
        assert recurrent.storage_offset() * 4 == conv[0].numel() * 2
        for slot in range(capacity):
            assert conv[slot].data_ptr() % 4 == 0
            assert recurrent[slot].data_ptr() % 4 == 0
            conv[slot].fill_(layer_index * 20 + slot + 1)
            recurrent[slot].fill_(layer_index * 20 + slot + 101)
    # Check after all writes, so cross-slot or cross-layer aliases cannot pass.
    for layer_index, name in enumerate(RECURRENT_NAMES):
        conv, recurrent = caches[name]
        for slot in range(capacity):
            assert torch.all(conv[slot] == layer_index * 20 + slot + 1)
            assert torch.all(recurrent[slot] == layer_index * 20 + slot + 101)
        raw = torch.empty(0, dtype=torch.uint8).set_(conv.untyped_storage())
        payload = conv[0].numel() * 2 + recurrent[0].numel() * 4
        assert torch.count_nonzero(raw.reshape(capacity, page)[:, payload:]) == 0
    assert torch.count_nonzero(caches[LATENT_NAME][0]) == 0


def test_requires_explicit_capability_and_complete_named_fields():
    runner, _, layers = _fixture(capacity=2, opt_in=False)
    capacities = runner_module.request_indexed_kda_capacities
    assert capacities(runner.model, 2) == {}
    runner.model.request_indexed_kda_state = True
    assert capacities(runner.model, 2) == dict.fromkeys(RECURRENT_NAMES, 2)
    layers[0] = replace(layers[0], kda_recurrent_state_dtype=None)
    with pytest.raises(ValueError, match="incomplete KDA state"):
        capacities(runner.model, 2)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "2"])
def test_refuses_invalid_model_capacity(capacity):
    runner, _, _ = _fixture(capacity=2)
    with pytest.raises(ValueError, match="positive integer"):
        runner_module.request_indexed_kda_capacities(runner.model, capacity)


def test_refuses_duplicate_layer_names():
    runner, _, layers = _fixture(capacity=2)
    layers.append(layers[0])
    with pytest.raises(ValueError, match="duplicate KV layer name"):
        runner_module.request_indexed_kda_capacities(runner.model, 2)


@pytest.mark.parametrize("name", ["missing.layer", LATENT_NAME])
def test_refuses_capacity_for_unknown_or_attention_layer(name):
    _, config, _ = _fixture(capacity=2)
    with pytest.raises(ValueError, match="unknown or non-recurrent"):
        runner_module.kv_cache_allocations(config, request_state_capacities={name: 2})


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "2"])
def test_refuses_invalid_allocation_capacity(capacity):
    _, config, _ = _fixture(capacity=2)
    with pytest.raises(ValueError, match="positive integer"):
        runner_module.kv_cache_allocations(
            config, request_state_capacities={RECURRENT_NAMES[0]: capacity}
        )


@pytest.mark.parametrize("owner_count", [0, 2])
def test_refuses_missing_or_duplicate_physical_owner(owner_count):
    _, config, _ = _fixture(capacity=2)
    name = RECURRENT_NAMES[0]
    config.kv_cache_tensors[0].shared_by.remove(name)
    config.kv_cache_tensors[0].shared_by.extend([name] * owner_count)
    with pytest.raises(ValueError, match="exactly one raw allocation"):
        runner_module.kv_cache_allocations(config, request_state_capacities={name: 2})


def test_legacy_recurrent_models_keep_token_pool_capacity(monkeypatch):
    runner, config, _ = _fixture(capacity=2, opt_in=False)
    caches = _allocate(runner, config, monkeypatch)
    page = config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
    expected = 3 * config.num_blocks * page
    assert _owned_bytes(caches) == _footprint(runner, config, monkeypatch) == expected
    for name in RECURRENT_NAMES:
        assert caches[name][0].shape[0] == config.num_blocks
    assert runner_module.kv_cache_allocations(config) == [
        (config.num_blocks * page, [LATENT_NAME]),
        *((config.num_blocks * page, [name]) for name in RECURRENT_NAMES),
    ]
