# SPDX-License-Identifier: Apache-2.0
"""Stage 5 integration tests for the GLM-5.2 main model and runner surface."""

from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoConfig

import vllm_neuron.model.glm_moe_dsa.model as model_module
import vllm_neuron.nn.cpl as cpl_module
import vllm_neuron.vllm.worker.neuron_model_runner as runner_module
from vllm_neuron.model.glm_moe_dsa.cache import (
    INDEXER_CACHE_BYTES,
    MAIN_INDEXER_LAYER_INDICES,
    MLA_CACHE_HEAD_SIZE,
    build_glm_mla_cache_spec,
    write_paged_cache,
)
from vllm_neuron.model.glm_moe_dsa.config import GlmMoeDsaConfig
from vllm_neuron.model.glm_moe_dsa.factory import GlmMoeDsaForCausalLM
from vllm_neuron.model.glm_moe_dsa.model import (
    GlmMoeDsaDecoderLayer,
    WeightLoadReport,
)
from vllm_neuron.model.glm_moe_dsa.model import (
    GlmMoeDsaForCausalLM as GlmMoeDsaModelForCausalLM,
)
from vllm_neuron.model.glm_moe_dsa.weight_loaders import (
    Disposition,
    load_checkpoint_manifest,
    local_load_plan,
)
from vllm_neuron.model.kv_cache import (
    KVSpec,
    LayerSpec,
    resolve_layer_cache_dtype,
)
from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig
from vllm_neuron.nn import ColumnParallelLinear
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

CHECKPOINT_DIR_VALUE = os.environ.get("GLM52_MODEL_PATH")
CHECKPOINT_DIR = Path(CHECKPOINT_DIR_VALUE or ".")
OWNED_FILES = (
    "model.py",
    "factory.py",
    "__init__.py",
    "test_stage5.py",
)


def _require_checkpoint_dir() -> Path:
    if CHECKPOINT_DIR_VALUE is None:
        pytest.skip("GLM52_MODEL_PATH is required for pinned-checkpoint tests")
    return CHECKPOINT_DIR


def _tiny_config() -> GlmMoeDsaConfig:
    config = copy.copy(GlmMoeDsaConfig())
    config.vocab_size = 64
    config.hidden_size = 16
    config.intermediate_size = 32
    config.num_hidden_layers = 1
    config.num_nextn_predict_layers = 0
    config.num_attention_heads = 1
    config.q_lora_rank = 16
    config.qk_nope_head_dim = 8
    config.v_head_dim = 8
    config.first_k_dense_replace = 1
    config.index_topk = 4
    return config


class _RecordingCollective:
    def __init__(self) -> None:
        self.device_group = object()
        self.calls: list[torch.Size] = []

    def all_reduce(self, values: torch.Tensor) -> torch.Tensor:
        self.calls.append(values.shape)
        return values * 2


class _LocalAttention(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.indexer = None

    def project(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        del positions
        batch, sequence, _ = hidden_states.shape
        return SimpleNamespace(
            q_lora=torch.zeros(batch, sequence, 1),
            queries=torch.zeros(batch, sequence, 1, 1),
            latent_cache=torch.zeros(batch, sequence, 576),
        )

    def attend(self, queries, latent_cache, selected_indices):
        del latent_cache, selected_indices
        return torch.zeros(queries.shape[0], queries.shape[1], self.hidden_size)


class _DenseLocalContribution(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(hidden_states)


class _MoeLocalContribution(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, *, is_decode: bool) -> torch.Tensor:
        del is_decode
        return torch.ones_like(hidden_states)


@pytest.fixture(scope="module")
def production_model() -> GlmMoeDsaModelForCausalLM:
    with (_require_checkpoint_dir() / "config.json").open() as config_file:
        config = GlmMoeDsaConfig.from_configs(json.load(config_file))
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=64),
        patch("torch.distributed.get_rank", return_value=0),
        torch.device("meta"),
    ):
        return GlmMoeDsaModelForCausalLM(config, 64, tp_group=_RecordingCollective())


@pytest.fixture(scope="module")
def manifest():
    checkpoint_dir = _require_checkpoint_dir()
    return load_checkpoint_manifest(checkpoint_dir / "model.safetensors.index.json")


def test_full_meta_model_has_exact_main_topology(production_model) -> None:
    assert len(production_model.model.layers) == 78
    assert [layer.layer_idx for layer in production_model.model.layers] == list(
        range(78)
    )
    assert sum(layer.is_dense for layer in production_model.model.layers) == 3
    assert sum(layer.is_moe for layer in production_model.model.layers) == 75
    assert production_model.excluded_mtp_layer_indices == (78,)
    assert production_model.model.norm.weight.shape == (6144,)
    assert production_model.lm_head.weight.shape == (154880 // 64, 6144)
    assert all(parameter.is_meta for parameter in production_model.parameters())
    assert production_model.model.layers[3].mlp.experts.global_expert_ids == (
        0,
        1,
        2,
        3,
    )


def test_registry_factory_selects_full_stage5_model(monkeypatch) -> None:
    hf_config = AutoConfig.from_pretrained(
        _require_checkpoint_dir(), local_files_only=True, trust_remote_code=False
    )
    monkeypatch.setattr(
        GlmMoeDsaForCausalLM,
        "_resolve_tensor_parallel_rank",
        staticmethod(lambda: 0),
    )
    collective = _RecordingCollective()
    monkeypatch.setattr(
        model_module,
        "_resolve_tp_groups",
        lambda tp_group, tensor_parallel_size: (
            collective,
            collective.device_group,
        ),
    )
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=64),
        patch("torch.distributed.get_rank", return_value=0),
        torch.device("meta"),
    ):
        model = GlmMoeDsaForCausalLM.from_configs(
            hf_config, neuron_config=None, tensor_parallel_size=64
        )
    assert isinstance(model, GlmMoeDsaModelForCausalLM)
    assert len(model.model.layers) == 78


def test_tiny_one_layer_cpu_forward_is_finite() -> None:
    torch.manual_seed(51)
    model = GlmMoeDsaModelForCausalLM(_tiny_config(), 1)
    input_ids = torch.tensor([1, 7, 11, 9], dtype=torch.int64)
    positions = torch.arange(4, dtype=torch.int64)
    logits = model(
        input_ids,
        positions,
        sampling_positions=torch.tensor([3], dtype=torch.int64),
    )
    assert logits.shape == (1, 64)
    assert logits.dtype is torch.bfloat16
    assert torch.isfinite(logits).all()


def test_tp64_requires_distributed_state_or_explicit_group() -> None:
    with (
        patch("torch.distributed.is_initialized", return_value=False),
        pytest.raises(RuntimeError, match="requires initialized distributed state"),
        torch.device("meta"),
    ):
        GlmMoeDsaModelForCausalLM(_tiny_config(), 64)


def _paged_meta(slot_mapping: torch.Tensor, *, max_query_len: int, cached_seq_len: int):
    mla = {
        "slot_mapping": slot_mapping,
        "block_size": 4,
        "block_table_tensor": torch.tensor([[0, 1]], dtype=torch.int32),
        "max_query_len": max_query_len,
        "decode_token_threshold": 1,
        "cached_seq_len": cached_seq_len,
    }
    return {
        "model.layers.0.self_attn.mla_cache": mla,
        "model.layers.0.self_attn.indexer.k_cache": dict(mla),
    }


def _tiny_bound_model() -> GlmMoeDsaModelForCausalLM:
    model = GlmMoeDsaModelForCausalLM(_tiny_config(), 1)
    layer = model.model.layers[0]
    layer.mla_k_cache = torch.zeros(
        2, 1, 4, MLA_CACHE_HEAD_SIZE, dtype=torch.float8_e4m3fn
    )
    layer.mla_v_cache = torch.zeros_like(layer.mla_k_cache)
    layer.indexer_k_cache = torch.zeros(2, 1, 4, INDEXER_CACHE_BYTES, dtype=torch.uint8)
    layer.indexer_v_cache = torch.zeros_like(layer.indexer_k_cache)
    return model


def test_prefill_writes_only_scheduler_slot_mapping() -> None:
    model = _tiny_bound_model()
    layer = model.model.layers[0]
    before = layer.mla_k_cache.clone()
    model(
        torch.tensor([1, 2, 3]),
        torch.tensor([0, 1, 2]),
        attn_metadata=_paged_meta(
            torch.tensor([1, -1, 5]), max_query_len=3, cached_seq_len=0
        ),
        sampling_positions=torch.tensor([2]),
    )
    changed = (layer.mla_k_cache != before).any(dim=-1).squeeze(1)
    assert changed.flatten().nonzero().flatten().tolist() == [1, 5]
    assert layer.indexer_k_cache.flatten(0, 2).any(
        dim=-1
    ).nonzero().flatten().tolist() == [1, 5]
    assert torch.count_nonzero(layer.mla_v_cache.float()) == 0
    assert torch.count_nonzero(layer.indexer_v_cache) == 0


def test_sentinel_cannot_overwrite_a_valid_final_cache_slot() -> None:
    cache = torch.zeros(2, 1, 32, 4, dtype=torch.float8_e4m3fn)
    values = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [7.0, 7.0, 7.0, 7.0]],
        dtype=torch.bfloat16,
    )
    write_paged_cache(
        cache,
        values,
        torch.tensor([63, -1], dtype=torch.int64),
        32,
    )
    torch.testing.assert_close(
        cache[1, 0, 31].float(),
        values[0].to(torch.float8_e4m3fn).float(),
    )
    assert torch.count_nonzero(cache.float()) == 4
    indexer_cache = torch.zeros(2, 1, 32, 4, dtype=torch.uint8)
    indexer_values = torch.tensor(
        [[1, 2, 3, 4], [255, 255, 255, 255]], dtype=torch.uint8
    )
    write_paged_cache(
        indexer_cache,
        indexer_values,
        torch.tensor([63, -1], dtype=torch.int64),
        32,
    )
    assert torch.equal(indexer_cache[1, 0, 31], indexer_values[0])
    assert torch.count_nonzero(indexer_cache) == 4


def test_prior_cache_changes_one_token_decode_logits() -> None:
    model = _tiny_bound_model()
    meta = _paged_meta(torch.tensor([3]), max_query_len=1, cached_seq_len=3)
    args = (torch.tensor([7]), torch.tensor([3]))
    first = model(*args, attn_metadata=meta, sampling_positions=torch.tensor([0]))
    model.model.layers[0].mla_k_cache[0, 0, :3].fill_(3)
    second = model(*args, attn_metadata=meta, sampling_positions=torch.tensor([0]))
    assert not torch.equal(first, second)


def test_cached_decode_matches_full_sequence_last_position() -> None:
    torch.manual_seed(52)
    full = _tiny_bound_model()
    cached = copy.deepcopy(full)
    ids = torch.tensor([1, 7, 11, 9])
    expected = full(ids, torch.arange(4), sampling_positions=torch.tensor([3]))
    cached(
        ids[:3],
        torch.arange(3),
        attn_metadata=_paged_meta(
            torch.tensor([0, 1, 2]), max_query_len=3, cached_seq_len=0
        ),
        sampling_positions=torch.tensor([2]),
    )
    actual = cached(
        ids[3:],
        torch.tensor([3]),
        attn_metadata=_paged_meta(torch.tensor([3]), max_query_len=1, cached_seq_len=3),
        sampling_positions=torch.tensor([0]),
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_two_request_cache_slots_are_isolated() -> None:
    model = _tiny_bound_model()
    layer = model.model.layers[0]
    layer.mla_k_cache = torch.zeros(
        4, 1, 4, MLA_CACHE_HEAD_SIZE, dtype=torch.float8_e4m3fn
    )
    layer.mla_v_cache = torch.full_like(layer.mla_k_cache, 17)
    layer.indexer_k_cache = torch.zeros(4, 1, 4, INDEXER_CACHE_BYTES, dtype=torch.uint8)
    layer.indexer_v_cache = torch.full_like(layer.indexer_k_cache, 19)
    meta = _paged_meta(
        torch.tensor([12, 13, 0, -1]),
        max_query_len=2,
        cached_seq_len=torch.tensor([0, 0]),
    )
    meta["model.layers.0.self_attn.mla_cache"]["block_table_tensor"] = torch.tensor(
        [[3, 1], [0, 2]], dtype=torch.int32
    )
    meta["model.layers.0.self_attn.indexer.k_cache"]["block_table_tensor"] = (
        torch.tensor([[3, 1], [0, 2]], dtype=torch.int32)
    )
    model(
        torch.tensor([5, 6, 7, 0]),
        torch.tensor([0, 1, 0, 0]),
        attn_metadata=meta,
        sampling_positions=torch.tensor([1, 2]),
    )
    changed = layer.mla_k_cache.flatten(0, 2).any(dim=-1)
    assert changed.nonzero().flatten().tolist() == [0, 12, 13]
    assert not torch.equal(layer.mla_k_cache[3, 0, 0], layer.mla_k_cache[0, 0, 0])
    assert torch.all(layer.mla_v_cache == 17)
    assert torch.all(layer.indexer_v_cache == 19)
    decode_meta = _paged_meta(
        torch.tensor([14, 1]), max_query_len=1, cached_seq_len=torch.tensor([2, 1])
    )
    decode_meta["model.layers.0.self_attn.mla_cache"]["block_table_tensor"] = (
        torch.tensor([[3, 1], [0, 2]], dtype=torch.int32)
    )
    decode_meta["model.layers.0.self_attn.indexer.k_cache"]["block_table_tensor"] = (
        torch.tensor([[3, 1], [0, 2]], dtype=torch.int32)
    )
    baseline_model = copy.deepcopy(model)
    baseline = baseline_model(
        torch.tensor([8, 9]),
        torch.tensor([2, 1]),
        attn_metadata=decode_meta,
        sampling_positions=torch.tensor([0, 1]),
    )
    request1_changed = copy.deepcopy(model)(
        torch.tensor([8, 10]),
        torch.tensor([2, 1]),
        attn_metadata=decode_meta,
        sampling_positions=torch.tensor([0, 1]),
    )
    request0_changed = copy.deepcopy(model)(
        torch.tensor([11, 9]),
        torch.tensor([2, 1]),
        attn_metadata=decode_meta,
        sampling_positions=torch.tensor([0, 1]),
    )
    torch.testing.assert_close(request1_changed[0], baseline[0])
    assert not torch.equal(request1_changed[1], baseline[1])
    torch.testing.assert_close(request0_changed[1], baseline[1])
    assert not torch.equal(request0_changed[0], baseline[0])


def test_metadata_controls_prefill_decode_dispatch_not_token_count() -> None:
    model = _tiny_bound_model()
    seen = []
    original = model.model.layers[0].forward

    def recording_forward(*args, **kwargs):
        seen.append(kwargs.get("is_decode"))
        return original(*args, **kwargs)

    model.model.layers[0].forward = recording_forward
    model(
        torch.tensor([3]),
        torch.tensor([0]),
        attn_metadata=_paged_meta(torch.tensor([0]), max_query_len=4, cached_seq_len=0),
        sampling_positions=torch.tensor([0]),
    )
    assert seen == [False]


def test_tp64_local_shards_are_gathered_for_host_logits() -> None:
    group = object()
    local = torch.tensor([[-3.0]])
    gathered = torch.full((1, 64), -9.0)
    gathered[0, 63] = 11.0
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=64),
        patch("torch.distributed.get_rank", return_value=0),
        patch.object(cpl_module, "all_gather_tensor", return_value=gathered) as gather,
    ):
        head = ColumnParallelLinear(
            1, 64, bias=False, gather_output=True, tp_group=group
        )
        head.weight.data.fill_(-3)
        logits = head(torch.ones(1, 1))
    gather.assert_called_once()
    assert torch.argmax(logits.float(), dim=-1).item() == 63


@pytest.mark.parametrize(("layer_idx", "is_moe"), [(0, False), (3, True)])
def test_decoder_sums_local_mlp_once_across_common_group(
    layer_idx: int, is_moe: bool
) -> None:
    config = _tiny_config()
    config.num_hidden_layers = 4
    config.num_attention_heads = 2
    config.first_k_dense_replace = 3
    collective = _RecordingCollective()
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_rank", return_value=0),
        torch.device("meta"),
    ):
        layer = GlmMoeDsaDecoderLayer(
            config,
            layer_idx,
            tensor_parallel_size=2,
            expert_parallel_size=2,
            expert_parallel_rank=0,
            tp_group=collective.device_group,
            collective_group=collective,
        )

    layer.input_layernorm = torch.nn.Identity()
    layer.post_attention_layernorm = torch.nn.Identity()
    layer.self_attn = _LocalAttention(config.hidden_size)
    layer.mlp = _MoeLocalContribution() if is_moe else _DenseLocalContribution()
    hidden_states = torch.zeros(1, 2, config.hidden_size)
    output, _ = layer(hidden_states, torch.arange(2).unsqueeze(0))

    assert layer.is_moe is is_moe
    assert layer.tp_group is collective
    assert collective.calls == [torch.Size([1, 2, config.hidden_size])]
    torch.testing.assert_close(output, torch.full_like(output, 2))


def test_runner_forward_signature_and_cache_binding(production_model) -> None:
    parameters = inspect.signature(production_model.forward).parameters
    required = {
        "input_ids",
        "positions",
        "inputs_embeds",
        "is_token_ids",
        "attn_metadata",
        "sampling_positions",
        "sampling_params",
        "spec_decode_metadata",
        "logit_mask",
        "rank",
        "kwargs",
    }
    assert required <= set(parameters)

    spec = production_model.get_kv_spec()
    assert len(spec.layers) == 78 + len(MAIN_INDEXER_LAYER_INDICES)
    assert all("layers.78" not in layer.name for layer in spec.layers)
    mla = [layer for layer in spec.layers if layer.name.endswith("mla_cache")]
    indexer = [layer for layer in spec.layers if layer.name.endswith("indexer.k_cache")]
    assert len(mla) == 78
    assert all(layer.head_size == MLA_CACHE_HEAD_SIZE for layer in mla)
    assert len(indexer) == len(MAIN_INDEXER_LAYER_INDICES)
    assert all(layer.head_size == INDEXER_CACHE_BYTES for layer in indexer)

    caches = {layer.name: [torch.empty(1), torch.empty(1)] for layer in spec.layers}
    production_model.bind_kv_cache(caches)
    first = production_model.model.layers[0]
    assert first.mla_k_cache is caches["model.layers.0.self_attn.mla_cache"][0]
    assert (
        first.indexer_k_cache is caches["model.layers.0.self_attn.indexer.k_cache"][0]
    )
    with pytest.raises(ValueError, match="not initialized"):
        production_model.bind_kv_cache({})


def test_runner_fp8_cache_inventory_preserves_packed_indexer_dtype() -> None:
    runner = object.__new__(NeuronModelRunner)
    runner.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=32, cache_dtype="fp8"),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    runner.model = SimpleNamespace(
        get_kv_spec=lambda: build_glm_mla_cache_spec(dtype=torch.bfloat16)
    )
    runner.speculative_config = None

    inventory = runner.get_kv_cache_spec()
    mla = {name: spec for name, spec in inventory.items() if name.endswith("mla_cache")}
    indexer = {
        name: spec
        for name, spec in inventory.items()
        if name.endswith("indexer.k_cache")
    }
    assert len(inventory) == 99
    assert len(mla) == 78
    assert len(indexer) == 21
    assert all(spec.dtype is torch.float8_e4m3fn for spec in mla.values())
    assert all(spec.dtype is torch.uint8 for spec in indexer.values())
    assert not any(spec.dtype is torch.float64 for spec in inventory.values())

    bytes_per_token_rank = sum(
        2
        * spec.num_kv_heads
        * spec.head_size
        * torch.empty((), dtype=spec.dtype).element_size()
        for spec in inventory.values()
    )
    assert bytes_per_token_rank == 95_400
    observed_cache_byte_budget = 7_106_918_400
    token_capacity = observed_cache_byte_budget // bytes_per_token_rank
    assert token_capacity == 74_496
    assert token_capacity >= 2_048 * 32


@pytest.mark.parametrize(
    "declared_dtype",
    (torch.uint8, torch.int8, torch.int32, torch.int64, torch.bool),
)
def test_cache_dtype_override_preserves_nonfloating_contracts(
    declared_dtype: torch.dtype,
) -> None:
    assert (
        resolve_layer_cache_dtype(declared_dtype, torch.float8_e4m3fn) is declared_dtype
    )


@pytest.mark.parametrize(
    "declared_dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn),
)
def test_cache_dtype_override_applies_to_floating_contracts(
    declared_dtype: torch.dtype,
) -> None:
    assert (
        resolve_layer_cache_dtype(declared_dtype, torch.float8_e4m3fn)
        is torch.float8_e4m3fn
    )


def test_runner_auto_cache_uses_model_dtype_and_preserves_indexer() -> None:
    runner = object.__new__(NeuronModelRunner)
    runner.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=32, cache_dtype="auto"),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    runner.model = SimpleNamespace(
        get_kv_spec=lambda: build_glm_mla_cache_spec(dtype=torch.bfloat16)
    )
    runner.speculative_config = None

    inventory = runner.get_kv_cache_spec()
    assert all(
        spec.dtype is torch.bfloat16
        for name, spec in inventory.items()
        if name.endswith("mla_cache")
    )
    assert all(
        spec.dtype is torch.uint8
        for name, spec in inventory.items()
        if name.endswith("indexer.k_cache")
    )


def test_runner_rejects_invalid_global_cache_dtype() -> None:
    runner = object.__new__(NeuronModelRunner)
    runner.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=32, cache_dtype="uint8"),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    runner.model = SimpleNamespace(
        get_kv_spec=lambda: build_glm_mla_cache_spec(dtype=torch.bfloat16)
    )
    runner.speculative_config = None

    with pytest.raises(ValueError, match="Unsupported kv_cache_dtype 'uint8'"):
        runner.get_kv_cache_spec()


def test_runner_fp8_cache_dtype_rule_applies_to_eagle_drafter(monkeypatch) -> None:
    class FakeEagleProposer:
        def __init__(self, spec: KVSpec) -> None:
            self.model = SimpleNamespace(get_kv_spec=lambda: spec)

    monkeypatch.setattr(runner_module, "EagleProposer", FakeEagleProposer)
    runner = object.__new__(NeuronModelRunner)
    runner.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=32, cache_dtype="fp8"),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    runner.model = SimpleNamespace(get_kv_spec=lambda: KVSpec(layers=[]))
    runner.speculative_config = SimpleNamespace(use_eagle=lambda: True)
    runner.drafter = FakeEagleProposer(
        KVSpec(
            layers=[
                LayerSpec("draft.float", 1, 16, torch.bfloat16),
                LayerSpec("draft.bytes", 1, 4, torch.uint8),
            ]
        )
    )

    inventory = runner.get_kv_cache_spec()
    assert inventory["draft.float"].dtype is torch.float8_e4m3fn
    assert inventory["draft.bytes"].dtype is torch.uint8


def test_checkpoint_mapping_exactly_delegates_rank_zero_and_accounts_mtp(
    production_model, manifest, monkeypatch
) -> None:
    mappings = production_model.checkpoint_mappings(manifest)
    report = production_model._account_manifest(manifest, mappings)
    local_plan = local_load_plan(manifest, ep_rank=0)
    expected_sources = {entry.key for entry in local_plan}
    actual_sources = {
        key
        for source in mappings.values()
        for key in (source if isinstance(source, list) else [source])
    }
    assert actual_sources == expected_sources
    assert report.mapped_sources == report.local_keys == len(expected_sources)
    assert report.indexed_keys == 118_629
    assert report.skipped_mtp == 1_569
    assert report.load_targets == sum(
        entry.info.disposition is Disposition.LOAD_TARGET for entry in local_plan
    )
    assert report.fp8_scales == sum(
        entry.info.disposition is Disposition.FP8_SCALE for entry in local_plan
    )
    assert all("model.layers.78" not in key for key in actual_sources)
    assert (
        mappings["model.layers.3.mlp.experts.experts.0.gate_proj.weight"]
        == "model.layers.3.mlp.experts.0.gate_proj.weight"
    )
    assert (
        mappings["model.layers.3.mlp.experts.experts.0.gate_proj.weight_scale_inv"]
        == "model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv"
    )
    assert (
        mappings["model.layers.3.mlp.router.gate.weight"]
        == "model.layers.3.mlp.gate.weight"
    )

    calls: dict[str, Any] = {}

    class FakeCheckpoint:
        def __init__(self, checkpoint_path, cache_dir):
            calls["init"] = (checkpoint_path, cache_dir)

        def load_sharded(self, rank, world_size, model, mappings, device, strict):
            calls["load"] = (rank, world_size, mappings, device, strict)
            return SimpleNamespace(
                state_dict={
                    name: parameter for name, parameter in model.named_parameters()
                }
            )

    monkeypatch.setattr(model_module, "load_checkpoint_manifest", lambda _: manifest)
    monkeypatch.setattr(model_module, "SafetensorsCheckpoint", FakeCheckpoint)
    production_model.load_weights("/checkpoint", torch.device("meta"), "/cache")
    assert calls["init"] == ("/checkpoint", "/cache")
    rank, world_size, delegated, device, strict = calls["load"]
    assert (rank, world_size, device.type, strict) == (0, 64, "meta", True)
    assert delegated == mappings
    assert production_model.last_load_report == report


def test_lightweight_loader_accounts_all_index_keys_without_payloads(
    production_model,
) -> None:
    production_model.load_weights_lite(
        str(CHECKPOINT_DIR), torch.device("cpu"), cache_dir=None
    )
    report = production_model.last_load_report
    assert isinstance(report, WeightLoadReport)
    assert report.indexed_keys == 118_629
    assert report.local_keys == report.load_targets + report.fp8_scales
    assert report.skipped_mtp == 1_569
    assert report.mapped_sources == 0


def test_fp8_loaders_retain_storage_and_adjust_trn2_range() -> None:
    model = GlmMoeDsaModelForCausalLM(_tiny_config(), 1)
    values = torch.arange(130 * 130, dtype=torch.float32).reshape(130, 130)
    values = ((values % 31) - 15).to(torch.float8_e4m3fn)
    inverse_scale = torch.tensor([[0.5, 1.0], [2.0, 4.0]])
    weight_loader = model._fp8_weight_loader(
        "model.layers.0.mlp.gate_proj.weight",
        (130, 130),
        shard_dim=None,
    )
    scale_loader = model._fp8_scale_loader(
        "model.layers.0.mlp.gate_proj.weight_scale_inv",
        (130, 130),
        shard_dim=None,
    )
    actual_weight = weight_loader.load([values], rank=0)
    actual_scale = scale_loader.load([inverse_scale], rank=0)
    assert actual_weight.dtype is torch.float8_e4m3fn
    torch.testing.assert_close(
        actual_weight.float(),
        (values.float() * (240.0 / 448.0)).to(torch.float8_e4m3fn).float(),
    )
    torch.testing.assert_close(actual_scale, inverse_scale * (448.0 / 240.0))


@pytest.mark.parametrize("output_length", [1, 16, 32])
def test_logits_match_reference(output_length: int) -> None:
    torch.manual_seed(52 + output_length)
    model = GlmMoeDsaModelForCausalLM(_tiny_config(), 1)
    hidden = torch.randn(output_length, 16, dtype=torch.bfloat16)
    logits = model.compute_logits(hidden)
    reference_logits = F.linear(hidden, model.lm_head.weight)
    torch.testing.assert_close(logits, reference_logits)
    assert logits.shape == (output_length, 64)


def test_on_device_sampling_uses_shared_sampler_and_sharded_logits(monkeypatch) -> None:
    calls: list[tuple[Any, ...]] = []

    class RecordingSampler(torch.nn.Module):
        def __init__(self, sampling_config, process_group=None) -> None:
            super().__init__()
            calls.append(("init", sampling_config, process_group))

        def forward(
            self,
            logits,
            sampling_params=None,
            logit_mask=None,
            tp_rank=None,
        ):
            calls.append(("forward", logits, sampling_params, logit_mask, tp_rank))
            return torch.argmax(logits.float(), dim=-1).to(torch.int32)

    config = _tiny_config()
    config.neuron_config = SimpleNamespace(
        on_device_sampling_config=OnDeviceSamplingConfig(all_greedy=True)
    )
    monkeypatch.setattr(model_module, "Sampler", RecordingSampler)
    model = GlmMoeDsaModelForCausalLM(config, 1)
    assert model.lm_head.gather_output is False

    sampling_params = torch.tensor([[1.0, 1.0, 0.0]])
    logit_mask = torch.ones(1, config.vocab_size, dtype=torch.bool)
    rank = torch.tensor(0, dtype=torch.int32)
    sampled, gathered_logits = model(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        sampling_positions=torch.tensor([1]),
        sampling_params=sampling_params,
        logit_mask=logit_mask,
        rank=rank,
    )

    assert gathered_logits is None
    assert sampled.shape == (1,)
    assert calls[0][0] == "init"
    _, local_logits, actual_params, actual_mask, actual_rank = calls[1]
    assert local_logits.shape == (1, config.vocab_size)
    assert actual_params is sampling_params
    assert actual_mask is logit_mask
    assert actual_rank is rank


def test_owned_files_compile_and_have_no_forbidden_runtime_imports() -> None:
    package = Path(__file__).parents[4] / "vllm_neuron" / "model" / "glm_moe_dsa"
    forbidden = (
        "neuronx_" + "distributed",
        "neuronx_" + "distributed_" + "inference",
        "NxD" + "Inference",
    )
    sources = [
        *(package / filename for filename in OWNED_FILES[:-1]),
        Path(__file__),
    ]
    for source_path in sources:
        source = source_path.read_text()
        compile(source, str(source_path), "exec")
        assert not any(token in source for token in forbidden)
