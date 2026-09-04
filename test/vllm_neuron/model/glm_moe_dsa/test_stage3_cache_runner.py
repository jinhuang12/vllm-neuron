# SPDX-License-Identifier: Apache-2.0
"""Stage 3 cache-capacity and runner-bucket contract tests."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm_neuron.model.glm_moe_dsa.cache as cache_module
import vllm_neuron.model.glm_moe_dsa.model as model_module
import vllm_neuron.vllm.worker.neuron_model_runner as runner_module
from vllm_neuron.model.glm_moe_dsa.cache import (
    INDEXER_CACHE_PART_BYTES,
    MLA_CACHE_PART_SIZE,
    build_glm_mla_cache_spec,
    gather_paged_cache_pair,
    write_paged_cache_pair,
)
from vllm_neuron.model.glm_moe_dsa.config import GlmMoeDsaConfig
from vllm_neuron.model.glm_moe_dsa.indexer import (
    GlmMoeDsaIndexer,
    IndexerProjection,
    causal_position_only_indices,
    causal_topk_indices,
    pack_indexer_keys,
    paged_key_positions,
    unpack_indexer_keys,
)
from vllm_neuron.model.glm_moe_dsa.model import (
    GlmMoeDsaDecoderLayer,
    _short_context_indexer_bypass,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner


def _fp8_paired_cache_bytes_per_token_rank() -> int:
    spec = build_glm_mla_cache_spec(dtype=torch.float8_e4m3fn)
    return sum(
        2
        * layer.num_kv_heads
        * layer.head_size
        * torch.empty((), dtype=layer.dtype).element_size()
        for layer in spec.layers
    )


def test_fp8_paired_cache_inventory_has_expected_bytes_per_token() -> None:
    assert _fp8_paired_cache_bytes_per_token_rank() == 47_700


class _RecordingIndexer(torch.nn.Module):
    def __init__(self, projection: IndexerProjection) -> None:
        super().__init__()
        self.projection = projection
        self.selector = GlmMoeDsaIndexer(
            hidden_size=4,
            q_lora_rank=4,
            num_heads=1,
            head_dim=128,
            topk=2048,
        )
        self.topk = self.selector.topk
        self.paged_calls: list[dict[str, object]] = []
        self.nonpaged_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.project_calls = 0

    def project(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
    ) -> IndexerProjection:
        del hidden_states, q_lora, positions
        self.project_calls += 1
        return self.projection

    def select(
        self,
        projection: IndexerProjection,
        cached_keys: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        self.nonpaged_calls.append((query_positions, key_positions))
        return self.selector.select(
            projection,
            cached_keys,
            query_positions,
            key_positions,
        )

    def select_paged(
        self,
        projection: IndexerProjection,
        cached_keys: torch.Tensor,
        query_positions: torch.Tensor,
        block_table: torch.Tensor,
        *,
        block_size: int,
        physical_block_count: int,
    ) -> torch.Tensor:
        self.paged_calls.append(
            {
                "cached_keys": cached_keys,
                "query_positions": query_positions,
                "block_table": block_table,
                "block_size": block_size,
                "physical_block_count": physical_block_count,
            }
        )
        return self.selector.select_paged(
            projection,
            cached_keys,
            query_positions,
            block_table,
            block_size=block_size,
            physical_block_count=physical_block_count,
        )


class _DecoderAttentionStub(torch.nn.Module):
    def __init__(self, indexer: _RecordingIndexer, projection: object) -> None:
        super().__init__()
        self.indexer = indexer
        self.projection = projection

    def project(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> object:
        del hidden_states, positions
        return self.projection

    def should_use_selected_latent_mla(self, *args, **kwargs) -> bool:
        return True

    def attend_selected_latents(
        self,
        queries: torch.Tensor,
        selected_indices: torch.Tensor,
        *args,
    ) -> torch.Tensor:
        del selected_indices, args
        return torch.zeros(
            queries.shape[0],
            queries.shape[1],
            4,
            dtype=queries.dtype,
            device=queries.device,
        )

    def attend(
        self,
        queries: torch.Tensor,
        attention_latents: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        del attention_latents, selected_indices
        return torch.zeros(
            queries.shape[0],
            queries.shape[1],
            4,
            dtype=queries.dtype,
            device=queries.device,
        )


class _ZeroMlp(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


def _decoder_layer_for_cache_test(
    indexer: _RecordingIndexer,
    attention_projection: object,
    *,
    physical_block_count: int,
    block_size: int,
    short_context_indexer_bypass: bool = False,
) -> GlmMoeDsaDecoderLayer:
    layer = object.__new__(GlmMoeDsaDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.layer_idx = 0
    layer.is_dense = True
    layer.is_moe = False
    layer.world_size = 1
    layer.tp_group = None
    layer.short_context_indexer_bypass = short_context_indexer_bypass
    layer.input_layernorm = torch.nn.Identity()
    layer.post_attention_layernorm = torch.nn.Identity()
    layer.self_attn = _DecoderAttentionStub(indexer, attention_projection)
    layer.mlp = _ZeroMlp()
    layer.mla_k_cache = torch.zeros(
        physical_block_count,
        1,
        block_size,
        MLA_CACHE_PART_SIZE,
        dtype=torch.bfloat16,
    )
    layer.mla_v_cache = torch.zeros_like(layer.mla_k_cache)
    layer.indexer_k_cache = torch.zeros(
        physical_block_count,
        1,
        block_size,
        INDEXER_CACHE_PART_BYTES,
        dtype=torch.uint8,
    )
    layer.indexer_v_cache = torch.zeros_like(layer.indexer_k_cache)
    return layer


@pytest.mark.parametrize(
    ("max_model_len", "expected"),
    ((None, False), (0, False), (2048, True), (2049, False)),
)
def test_short_context_bypass_boundary_uses_runtime_model_limit_only(
    max_model_len: int | None,
    expected: bool,
) -> None:
    neuron_config = NeuronConfig()
    neuron_config.max_model_len = max_model_len
    config = GlmMoeDsaConfig(neuron_config=neuron_config)
    assert _short_context_indexer_bypass(config) is expected


def test_user_neuron_config_cannot_spoof_runtime_model_limit() -> None:
    config = NeuronConfig.from_dict({"max_model_len": 1})
    assert config.max_model_len is None
    runner_source = inspect.getsource(NeuronModelRunner.__init__)
    assert "self.neuron_config.max_model_len = self.max_model_len" in runner_source


def test_position_only_selection_is_exact_for_short_context() -> None:
    torch.manual_seed(2055)
    queries = torch.randn(2, 3, 2, 8)
    keys = torch.randn(2, 5, 8)
    head_weights = torch.randn(2, 3, 2)
    query_positions = torch.tensor([[5, 7, 9], [101, 102, 104]])
    key_positions = torch.tensor([[3, 5, 7, 9, 11], [99, 101, 103, 105, 107]])

    position_only = causal_position_only_indices(
        query_positions,
        key_positions,
        topk=8,
    )
    legacy_short_path = causal_topk_indices(
        queries,
        keys,
        head_weights,
        query_positions,
        key_positions,
        topk=8,
    )
    key_indices = torch.arange(5).view(1, 1, 5)
    expected = torch.where(
        key_positions.unsqueeze(1) <= query_positions.unsqueeze(-1),
        key_indices,
        -torch.ones_like(key_indices),
    )
    expected = torch.nn.functional.pad(expected, (0, 3), value=-1)

    assert torch.equal(position_only, expected)
    assert torch.equal(position_only, legacy_short_path)


def _short_context_attention_projection(
    batch: int,
    query_count: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        q_lora=torch.zeros(batch, query_count, 4),
        queries=torch.zeros(batch, query_count, 1, 4),
        latent_cache=torch.zeros(batch, query_count, MLA_CACHE_PART_SIZE * 2),
    )


def _short_context_metadata(
    positions: torch.Tensor,
    *,
    block_table: torch.Tensor,
    block_size: int,
) -> dict[str, dict[str, torch.Tensor | int]]:
    cache_metadata: dict[str, torch.Tensor | int] = {
        "slot_mapping": positions.reshape(-1).to(torch.int32),
        "block_size": block_size,
        "block_table_tensor": block_table,
    }
    return {
        "model.layers.0.self_attn.mla_cache": dict(cache_metadata),
        "model.layers.0.self_attn.indexer.k_cache": dict(cache_metadata),
    }


def test_short_context_bypass_does_no_indexer_projection_or_cache_work(
    monkeypatch,
) -> None:
    block_size = 32
    physical_block_count = 64
    positions = torch.tensor([[511]], dtype=torch.int64)
    block_table = torch.arange(physical_block_count, dtype=torch.int32).unsqueeze(0)
    index_projection = IndexerProjection(
        queries=torch.randn(1, 1, 1, 128),
        keys=torch.randn(1, 1, 128),
        head_weights=torch.randn(1, 1, 1),
    )
    indexer = _RecordingIndexer(index_projection)
    layer = _decoder_layer_for_cache_test(
        indexer,
        _short_context_attention_projection(1, 1),
        physical_block_count=physical_block_count,
        block_size=block_size,
        short_context_indexer_bypass=True,
    )
    layer.indexer_k_cache.fill_(17)
    layer.indexer_v_cache.fill_(29)
    before_k = layer.indexer_k_cache.clone()
    before_v = layer.indexer_v_cache.clone()

    def fail_indexer_transform(*args, **kwargs):
        raise AssertionError("short-context bypass touched indexer cache data")

    monkeypatch.setattr(model_module, "pack_indexer_keys", fail_indexer_transform)
    monkeypatch.setattr(model_module, "unpack_indexer_keys", fail_indexer_transform)
    monkeypatch.setattr(model_module, "gather_paged_cache_pair", fail_indexer_transform)

    _, selected = layer(
        torch.zeros(1, 1, 4),
        positions,
        is_decode=True,
        attn_metadata=_short_context_metadata(
            positions,
            block_table=block_table,
            block_size=block_size,
        ),
    )

    assert indexer.project_calls == 0
    assert not indexer.paged_calls
    assert not indexer.nonpaged_calls
    assert torch.equal(layer.indexer_k_cache, before_k)
    assert torch.equal(layer.indexer_v_cache, before_v)
    assert torch.equal(selected[0, 0, :512], torch.arange(512))
    assert torch.all(selected[0, 0, 512:] == -1)


def test_short_context_bypass_reuses_prior_layer_selection() -> None:
    block_size = 32
    physical_block_count = 64
    positions = torch.tensor([[1024]], dtype=torch.int64)
    block_table = torch.arange(physical_block_count, dtype=torch.int32).unsqueeze(0)
    indexer = _RecordingIndexer(
        IndexerProjection(
            queries=torch.zeros(1, 1, 1, 128),
            keys=torch.zeros(1, 1, 128),
            head_weights=torch.ones(1, 1, 1),
        )
    )
    layer = _decoder_layer_for_cache_test(
        indexer,
        _short_context_attention_projection(1, 1),
        physical_block_count=physical_block_count,
        block_size=block_size,
        short_context_indexer_bypass=True,
    )
    prior_selection = torch.arange(2048).view(1, 1, 2048)

    _, selected = layer(
        torch.zeros(1, 1, 4),
        positions,
        selected_indices=prior_selection,
        is_decode=True,
        attn_metadata=_short_context_metadata(
            positions,
            block_table=block_table,
            block_size=block_size,
        ),
    )

    assert selected is prior_selection
    assert indexer.project_calls == 0
    assert not indexer.paged_calls


@pytest.mark.parametrize(
    ("start", "query_count"),
    ((0, 512), (512, 512), (1024, 512), (1536, 512), (2047, 1)),
    ids=("prefill-0", "prefill-1", "prefill-2", "prefill-3", "decode"),
)
def test_short_context_bypass_preserves_segmented_prefill_and_decode(
    start: int,
    query_count: int,
) -> None:
    block_size = 32
    physical_block_count = 64
    positions = torch.arange(start, start + query_count).unsqueeze(0)
    allocated_blocks = (start + query_count + block_size - 1) // block_size
    block_table = torch.full((1, physical_block_count), -1, dtype=torch.int32)
    block_table[0, :allocated_blocks] = torch.arange(
        allocated_blocks, dtype=torch.int32
    )
    indexer = _RecordingIndexer(
        IndexerProjection(
            queries=torch.zeros(1, query_count, 1, 128),
            keys=torch.zeros(1, query_count, 128),
            head_weights=torch.ones(1, query_count, 1),
        )
    )
    layer = _decoder_layer_for_cache_test(
        indexer,
        _short_context_attention_projection(1, query_count),
        physical_block_count=physical_block_count,
        block_size=block_size,
        short_context_indexer_bypass=True,
    )

    _, selected = layer(
        torch.zeros(1, query_count, 4),
        positions,
        is_decode=query_count == 1,
        attn_metadata=_short_context_metadata(
            positions,
            block_table=block_table,
            block_size=block_size,
        ),
    )
    key_positions = paged_key_positions(
        block_table,
        block_size=block_size,
        physical_block_count=physical_block_count,
    )
    expected = causal_position_only_indices(
        positions,
        key_positions,
        topk=2048,
    )
    assert torch.equal(selected, expected)
    assert indexer.project_calls == 0
    assert torch.count_nonzero(layer.indexer_k_cache) == 0
    assert torch.count_nonzero(layer.indexer_v_cache) == 0


def test_short_context_bypass_fails_closed_when_paged_capacity_exceeds_topk() -> None:
    block_size = 32
    physical_block_count = 65
    positions = torch.tensor([[0]], dtype=torch.int64)
    block_table = torch.arange(physical_block_count, dtype=torch.int32).unsqueeze(0)
    indexer = _RecordingIndexer(
        IndexerProjection(
            queries=torch.zeros(1, 1, 1, 128),
            keys=torch.zeros(1, 1, 128),
            head_weights=torch.ones(1, 1, 1),
        )
    )
    layer = _decoder_layer_for_cache_test(
        indexer,
        _short_context_attention_projection(1, 1),
        physical_block_count=physical_block_count,
        block_size=block_size,
        short_context_indexer_bypass=True,
    )

    with pytest.raises(AssertionError, match="paged capacity"):
        layer(
            torch.zeros(1, 1, 4),
            positions,
            attn_metadata=_short_context_metadata(
                positions,
                block_table=block_table,
                block_size=block_size,
            ),
        )


def test_over_2048_context_keeps_scored_indexer_path(monkeypatch) -> None:
    torch.manual_seed(2056)
    queries = torch.randn(1, 1, 1, 4)
    keys = torch.randn(1, 2049, 4)
    head_weights = torch.ones(1, 1, 1)
    query_positions = torch.tensor([[2048]])
    key_positions = torch.arange(2049).unsqueeze(0)
    direct_matmul = torch.matmul
    matmul_calls = 0

    def recording_matmul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        nonlocal matmul_calls
        matmul_calls += 1
        return direct_matmul(lhs, rhs)

    monkeypatch.setattr(torch, "matmul", recording_matmul)
    selected = causal_topk_indices(
        queries,
        keys,
        head_weights,
        query_positions,
        key_positions,
        topk=2,
    )
    assert selected.shape == (1, 1, 2)
    assert matmul_calls > 0


def test_decoder_routes_b2_paged_cache_metadata_to_streaming_dsa() -> None:
    torch.manual_seed(2052)
    batch = 2
    block_size = 32
    logical_token_count = 2304
    logical_block_count = logical_token_count // block_size
    physical_block_count = 160
    valid_lengths = torch.tensor([2177, 2209], dtype=torch.int64)
    positions = (valid_lengths - 1).unsqueeze(-1)

    request_blocks = torch.stack(
        (
            torch.arange(1, 74, dtype=torch.int64).roll(11),
            torch.arange(81, 154, dtype=torch.int64).roll(19),
        )
    )
    block_table = torch.full(
        (batch, logical_block_count),
        -1,
        dtype=torch.int32,
    )
    logical_keys = torch.randn(batch, logical_token_count, 128)
    packed_keys = pack_indexer_keys(logical_keys)
    written_slots = []
    written_values = []
    for request_index in range(batch):
        valid_length = int(valid_lengths[request_index])
        valid_block_count = (valid_length + block_size - 1) // block_size
        block_table[request_index, :valid_block_count] = request_blocks[
            request_index, :valid_block_count
        ].to(torch.int32)
        logical_positions = torch.arange(valid_length, dtype=torch.int64)
        written_slots.append(
            request_blocks[request_index, logical_positions // block_size] * block_size
            + logical_positions % block_size
        )
        written_values.append(packed_keys[request_index, :valid_length])

    current_slots = torch.stack(
        [
            request_blocks[index, positions[index, 0] // block_size] * block_size
            + positions[index, 0] % block_size
            for index in range(batch)
        ]
    )
    current_keys = torch.stack(
        [logical_keys[index, positions[index, 0]] for index in range(batch)]
    ).unsqueeze(1)
    index_projection = IndexerProjection(
        queries=torch.randn(batch, 1, 1, 128),
        keys=current_keys,
        head_weights=torch.randn(batch, 1, 1),
    )
    indexer = _RecordingIndexer(index_projection)
    attention_projection = SimpleNamespace(
        q_lora=torch.zeros(batch, 1, 4),
        queries=torch.zeros(batch, 1, 1, 4),
        latent_cache=torch.zeros(batch, 1, MLA_CACHE_PART_SIZE * 2),
    )
    layer = _decoder_layer_for_cache_test(
        indexer,
        attention_projection,
        physical_block_count=physical_block_count,
        block_size=block_size,
    )
    write_paged_cache_pair(
        layer.indexer_k_cache,
        layer.indexer_v_cache,
        torch.cat(written_values),
        torch.cat(written_slots),
        block_size,
    )
    metadata = {
        "model.layers.0.self_attn.mla_cache": {
            "slot_mapping": current_slots,
            "block_size": block_size,
            "block_table_tensor": block_table,
        },
        "model.layers.0.self_attn.indexer.k_cache": {
            "slot_mapping": current_slots,
            "block_size": block_size,
            "block_table_tensor": block_table,
        },
    }

    hidden_states = torch.randn(batch, 1, 4)
    output, selected = layer(
        hidden_states,
        positions,
        is_decode=True,
        attn_metadata=metadata,
    )

    assert torch.equal(output, hidden_states)
    assert len(indexer.paged_calls) == 1
    assert not indexer.nonpaged_calls
    call = indexer.paged_calls[0]
    assert torch.equal(call["query_positions"], positions)
    assert torch.equal(call["block_table"], block_table)
    assert call["block_size"] == block_size
    assert call["physical_block_count"] == physical_block_count
    expected_cached_keys = unpack_indexer_keys(
        gather_paged_cache_pair(
            layer.indexer_k_cache,
            layer.indexer_v_cache,
            block_table,
        ),
        dtype=current_keys.dtype,
    )
    torch.testing.assert_close(call["cached_keys"], expected_cached_keys)

    assert positions.tolist() == [[2176], [2208]]
    assert all(int(length) % block_size == 1 for length in valid_lengths)
    assert torch.any(block_table < 0)
    assert not torch.equal(
        block_table[0, :68], torch.arange(68, dtype=block_table.dtype)
    )
    assert selected.shape == (batch, 1, 2048)
    assert torch.all((selected[0] >= 0) & (selected[0] < valid_lengths[0]))
    assert torch.all((selected[1] >= 0) & (selected[1] < valid_lengths[1]))
    assert not torch.equal(selected[0], selected[1])


def test_decoder_nonpaged_dsa_reuses_absolute_positions() -> None:
    positions = torch.tensor([[4096], [8192]], dtype=torch.int64)
    index_projection = IndexerProjection(
        queries=torch.zeros(2, 1, 1, 128),
        keys=torch.zeros(2, 1, 128),
        head_weights=torch.ones(2, 1, 1),
    )
    indexer = _RecordingIndexer(index_projection)
    attention_projection = SimpleNamespace(
        q_lora=torch.zeros(2, 1, 4),
        queries=torch.zeros(2, 1, 1, 4),
        latent_cache=torch.zeros(2, 1, MLA_CACHE_PART_SIZE * 2),
    )
    layer = _decoder_layer_for_cache_test(
        indexer,
        attention_projection,
        physical_block_count=1,
        block_size=1,
    )

    layer(torch.zeros(2, 1, 4), positions)

    assert len(indexer.nonpaged_calls) == 1
    query_positions, key_positions = indexer.nonpaged_calls[0]
    assert query_positions is positions
    assert key_positions is positions
    forward_source = inspect.getsource(GlmMoeDsaDecoderLayer.forward)
    assert "indexer.select_paged(" in forward_source
    assert "torch.arange" not in forward_source


@pytest.mark.skipif(
    not {
        "GLM_STAGE3_CACHE_CAPACITY_EXPECTED_IDENTITY_JSON",
        "GLM_STAGE3_CACHE_CAPACITY_EVIDENCE_JSON",
    }.issubset(os.environ),
    reason="requires frozen expected identity and sealed allocator receipt",
)
def test_exact_environment_fp8_cache_capacity_covers_c32_t4096() -> None:
    """Join a frozen candidate identity to its sealed allocator receipt."""

    expected_path = Path(os.environ["GLM_STAGE3_CACHE_CAPACITY_EXPECTED_IDENTITY_JSON"])
    evidence_path = Path(os.environ["GLM_STAGE3_CACHE_CAPACITY_EVIDENCE_JSON"])
    expected = json.loads(expected_path.read_text())
    evidence = json.loads(evidence_path.read_text())

    expected_fields = {
        "status",
        "integrated_candidate_revision",
        "environment_fingerprint",
        "kv_cache_config_fingerprint",
        "cache_source_sha256",
        "runner_source_sha256",
        "hardware_topology",
        "workload",
    }
    assert expected_fields <= expected.keys()
    assert expected["status"] == "frozen"
    assert all(
        isinstance(expected[field], str) and expected[field]
        for field in (
            "integrated_candidate_revision",
            "environment_fingerprint",
            "kv_cache_config_fingerprint",
        )
    )
    current_cache_sha256 = hashlib.sha256(
        Path(cache_module.__file__).read_bytes()
    ).hexdigest()
    current_runner_sha256 = hashlib.sha256(
        Path(runner_module.__file__).read_bytes()
    ).hexdigest()
    assert expected["cache_source_sha256"] == current_cache_sha256
    assert expected["runner_source_sha256"] == current_runner_sha256
    assert expected["hardware_topology"]["accelerator_family"] == "trn2"
    assert expected["hardware_topology"]["neuron_core_count"] == 64
    assert expected["workload"] == {
        "max_num_seqs": 32,
        "max_model_len": 4096,
        "cache_dtype": "fp8",
    }

    evidence_fields = {
        "status",
        "producer_status",
        "receipt_fingerprint",
        "integrated_candidate_revision",
        "environment_fingerprint",
        "kv_cache_config_fingerprint",
        "cache_source_sha256",
        "runner_source_sha256",
        "hardware_topology",
        "workload",
        "allocator_result",
    }
    assert evidence_fields <= evidence.keys()
    assert evidence["status"] == "sealed"
    assert evidence["producer_status"] == "completed"
    assert isinstance(evidence["receipt_fingerprint"], str)
    assert evidence["receipt_fingerprint"]
    for field in (
        "integrated_candidate_revision",
        "environment_fingerprint",
        "kv_cache_config_fingerprint",
        "cache_source_sha256",
        "runner_source_sha256",
        "hardware_topology",
        "workload",
    ):
        assert evidence[field] == expected[field]

    bytes_per_token_rank = _fp8_paired_cache_bytes_per_token_rank()
    allocator_result = evidence["allocator_result"]
    assert allocator_result["status"] == "completed"
    assert allocator_result["bytes_per_token_rank"] == bytes_per_token_rank == 47_700
    assert (
        allocator_result["kv_cache_config_fingerprint"]
        == expected["kv_cache_config_fingerprint"]
    )
    available_cache_bytes = allocator_result["available_cache_bytes_per_rank"]
    assert isinstance(available_cache_bytes, int) and available_cache_bytes > 0
    required_cache_bytes = 32 * 4096 * bytes_per_token_rank
    assert allocator_result["required_cache_bytes_per_rank"] == required_cache_bytes
    assert (
        allocator_result["token_capacity_per_rank"]
        == available_cache_bytes // bytes_per_token_rank
    )
    assert available_cache_bytes >= required_cache_bytes


def _runner_for_bucket_contract() -> NeuronModelRunner:
    runner = object.__new__(NeuronModelRunner)
    runner.device = torch.device("cpu")
    runner.drafter = None
    runner.speculative_config = None
    runner.max_model_len = 8192
    runner._dcp_size = 1
    runner.cp_world_size = 1
    runner._cp_rank = 0
    runner.neuron_config = SimpleNamespace(
        decode_context_length_buckets=[2048, 4096, 8192],
        kv_segment_size_buckets=None,
        apply_prefill_dcp=False,
    )
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=32),
                layer_names=["model.layers.0.self_attn.indexer.k_cache"],
            )
        ]
    )
    return runner


@pytest.mark.parametrize(
    ("context_bucket", "expected_blocks"),
    ((2048, 64), (4096, 128), (8192, 256)),
)
def test_c32_decode_runner_bucket_and_block_table_contract(
    context_bucket: int,
    expected_blocks: int,
) -> None:
    runner = _runner_for_bucket_contract()
    metadata = runner._build_warmup_attention_metadata(
        num_tokens=32,
        num_reqs=32,
        cached_seq_len=0,
        ctx_bucket=context_bucket,
        device=torch.device("cpu"),
    )["model.layers.0.self_attn.indexer.k_cache"]

    assert metadata["block_size"] == 32
    assert metadata["max_blocks_per_seq"] == expected_blocks
    assert metadata["block_table_tensor"].shape == (32, expected_blocks)
    assert metadata["slot_mapping"].shape == (32,)
    assert metadata["full_block_table_tensor"].shape == (32, 256)
    assert torch.equal(
        metadata["block_table_tensor"][0],
        torch.arange(expected_blocks, dtype=torch.int32),
    )
    assert torch.equal(
        metadata["block_table_tensor"],
        metadata["block_table_tensor"][0].expand(32, -1),
    )


@pytest.mark.parametrize(
    ("max_decode_context", "expected_blocks"),
    ((2047, 64), (2048, 128), (4095, 128), (4096, 256), (8191, 256)),
)
def test_runtime_decode_bucket_pick_matches_warmup_block_width(
    max_decode_context: int,
    expected_blocks: int,
) -> None:
    runner = _runner_for_bucket_contract()
    assert (
        runner._decode_ctx_blocks_from_max_decode_ctx_len(
            max_decode_ctx_len=max_decode_context,
            block_size=32,
            max_num_draft_tokens=0,
        )
        == expected_blocks
    )
