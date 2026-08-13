# SPDX-License-Identifier: Apache-2.0
"""Focused Stage 3 tests for GLM-5.2 sparse MLA and dual caches."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch

import vllm_neuron.model.glm_moe_dsa.indexer as indexer_module
from vllm_neuron.model.glm_moe_dsa.attention import (
    GlmMoeDsaAttention,
    _sparse_score_matmul,
    sparse_attention,
)
from vllm_neuron.model.glm_moe_dsa.block_fp8 import (
    FP8_INVERSE_SCALE_ADJUSTMENT,
    FP8_STORAGE_SCALE,
    BlockFP8ColumnParallelLinear,
    dequantize_block_fp8,
)
from vllm_neuron.model.glm_moe_dsa.cache import (
    INDEXER_CACHE_BYTES,
    INDEXER_CACHE_PART_BYTES,
    MAIN_INDEXER_LAYER_INDICES,
    MLA_CACHE_HEAD_SIZE,
    MLA_CACHE_PART_SIZE,
    DualCacheState,
    build_glm_mla_cache_spec,
)
from vllm_neuron.model.glm_moe_dsa.indexer import (
    GlmMoeDsaIndexer,
    apply_interleaved_rope,
    causal_topk_indices,
    latest_indexer_layer,
    pack_indexer_keys,
    quantize_ue8m0_fp8,
    rotary_cos_sin,
    unpack_indexer_keys,
)
from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear
from vllm_neuron.utils.weight_loader import get_weight_loader

PREFILL_LENGTHS = (16, 128, 512, 2048)
DECODE_VARIANTS = tuple(
    (batch, output_tokens) for batch in (1, 8, 32) for output_tokens in (1, 16, 32)
)


def _causal_selection(batch: int, queries: int, keys: int) -> torch.Tensor:
    result = torch.full((batch, queries, keys), -1, dtype=torch.int64)
    for query in range(queries):
        result[:, query, : query + 1] = torch.arange(query + 1)
    return result


def test_indexer_query_slices_preserve_split_semantics_without_split_view() -> None:
    source = inspect.getsource(GlmMoeDsaIndexer.project)
    assert "q.split(" not in source
    assert "k.split(" not in source
    assert "q[..., : self.rope_dim].contiguous()" in source
    assert "q[..., self.rope_dim :].contiguous()" in source
    assert "k[..., : self.rope_dim].contiguous()" in source
    assert "k[..., self.rope_dim :].contiguous()" in source

    torch.manual_seed(307)
    indexer = GlmMoeDsaIndexer(
        hidden_size=16,
        q_lora_rank=16,
        num_heads=2,
        head_dim=128,
        rope_dim=64,
    )
    hidden = torch.randn(1, 3, 16)
    q_lora = torch.randn(1, 3, 16)
    positions = torch.tensor([[20, 21, 22]], dtype=torch.int64)

    actual = indexer.project(hidden, q_lora, positions)
    q = indexer.wq_b(q_lora).view(1, 3, 2, 128)
    q_pe, q_nope = q.split((64, 64), dim=-1)
    cos, sin = rotary_cos_sin(positions, 64, theta=indexer.rope_theta, dtype=q.dtype)
    expected_queries = torch.cat(
        (apply_interleaved_rope(q_pe, cos, sin), q_nope), dim=-1
    )
    k = indexer.k_norm(indexer.wk(hidden))
    k_pe, k_nope = k.split((64, 64), dim=-1)
    expected_keys = torch.cat((apply_interleaved_rope(k_pe, cos, sin), k_nope), dim=-1)
    torch.testing.assert_close(actual.queries, expected_queries, rtol=0, atol=0)
    torch.testing.assert_close(actual.keys, expected_keys, rtol=0, atol=0)


def test_mla_projection_slices_preserve_split_semantics_without_split_views() -> None:
    source = inspect.getsource(GlmMoeDsaAttention.project)
    assert "q.split(" not in source
    assert "latent_raw.split(" not in source
    assert "q[..., : self.qk_nope_head_dim].contiguous()" in source
    assert "q[..., self.qk_nope_head_dim :].contiguous()" in source
    assert "latent_raw[..., : self.kv_lora_rank].contiguous()" in source
    assert "latent_raw[..., self.kv_lora_rank :].contiguous()" in source

    torch.manual_seed(311)
    attention = GlmMoeDsaAttention(
        hidden_size=16,
        q_lora_rank=16,
        kv_lora_rank=512,
        local_heads=1,
        num_heads=1,
        qk_nope_head_dim=8,
        qk_rope_head_dim=64,
        v_head_dim=6,
    )
    hidden = torch.randn(1, 3, 16)
    positions = torch.tensor([[20, 21, 22]], dtype=torch.int64)

    actual = attention.project(hidden, positions)
    q_lora = attention.q_a_layernorm(attention.q_a_proj(hidden))
    q = attention.q_b_proj(q_lora).view(1, 3, 1, 72)
    q_nope, q_pe = q.split((8, 64), dim=-1)
    latent_raw = attention.kv_a_proj_with_mqa(hidden)
    kv_latent, k_pe = latent_raw.split((512, 64), dim=-1)
    kv_latent = attention.kv_a_layernorm(kv_latent)
    kv = attention.kv_b_proj(kv_latent).view(1, 3, 1, 14)
    k_nope, values = kv.split((8, 6), dim=-1)
    cos, sin = rotary_cos_sin(positions, 64, theta=attention.rope_theta, dtype=q.dtype)
    q_pe = apply_interleaved_rope(q_pe, cos, sin)
    k_pe = apply_interleaved_rope(k_pe, cos, sin)
    expected = (
        q_lora,
        torch.cat((q_nope, q_pe), dim=-1),
        torch.cat((kv_latent, k_pe), dim=-1),
        torch.cat((k_nope, k_pe.unsqueeze(-2)), dim=-1),
        values,
    )
    actual_tensors = (
        actual.q_lora,
        actual.queries,
        actual.latent_cache,
        actual.keys,
        actual.values,
    )
    for actual_tensor, expected_tensor in zip(actual_tensors, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def _reference_sparse_attention(q, k, v, selected, scale):
    output = torch.zeros(
        *q.shape[:-1], v.shape[-1], dtype=torch.float32, device=q.device
    )
    for batch in range(q.shape[0]):
        for query in range(q.shape[1]):
            valid = selected[batch, query]
            valid = valid[(valid >= 0) & (valid < k.shape[1])].unique()
            if valid.numel() == 0:
                continue
            for head in range(q.shape[2]):
                score = (
                    q[batch, query, head].float() @ k[batch, valid, head].float().T
                ) * scale
                weight = torch.softmax(score, dim=-1)
                output[batch, query, head] = weight @ v[batch, valid, head].float()
    return output.to(q.dtype)


def _production_block_fp8_attention() -> GlmMoeDsaAttention:
    module = GlmMoeDsaAttention(
        hidden_size=16,
        q_lora_rank=16,
        kv_lora_rank=512,
        local_heads=1,
        num_heads=1,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=8,
        fp8_weights=True,
        dtype=torch.bfloat16,
    )
    assert isinstance(module.kv_b_proj, BlockFP8ColumnParallelLinear)
    assert module.kv_b_proj.weight.shape == (72, 512)
    assert module.kv_b_proj.weight_scale_inv.shape == (1, 4)
    weight_values = (
        (torch.arange(72 * 512, dtype=torch.int32).reshape(72, 512) % 31) - 15
    ).to(torch.float32) * 0.0625
    with torch.no_grad():
        module.kv_b_proj.weight.copy_(
            (weight_values * FP8_STORAGE_SCALE).to(torch.float8_e4m3fn)
        )
        module.kv_b_proj.weight_scale_inv.copy_(
            torch.tensor([[0.5, 0.75, 1.0, 1.25]], dtype=torch.float32)
            * FP8_INVERSE_SCALE_ADJUSTMENT
        )
    return module


def test_default_rope_matches_independent_complex_reference() -> None:
    torch.manual_seed(1)
    values = torch.randn(2, 7, 3, 64)
    positions = torch.arange(7).expand(2, -1)
    cos, sin = rotary_cos_sin(positions, 64)
    actual = apply_interleaved_rope(values, cos, sin)

    pairs = values.view(2, 7, 3, 32, 2)
    complex_values = torch.view_as_complex(pairs.contiguous())
    rotation = torch.complex(cos, sin).unsqueeze(-2)
    expected = torch.view_as_real(complex_values * rotation).flatten(-2)
    torch.testing.assert_close(actual, expected)


def test_sparse_attention_matches_gather_reference() -> None:
    torch.manual_seed(2)
    q = torch.randn(2, 5, 2, 8)
    k = torch.randn(2, 7, 2, 8)
    v = torch.randn(2, 7, 2, 6)
    selected = torch.tensor(
        [
            [[0, -1, -1], [0, 1, -1], [0, 2, 1], [3, 1, 0], [4, 2, 0]],
            [[0, -1, -1], [1, 0, -1], [2, 0, 1], [3, 2, 0], [4, 3, 1]],
        ]
    )
    scale = 8**-0.5
    actual = sparse_attention(q, k, v, selected, scale=scale)
    expected = _reference_sparse_attention(q, k, v, selected, scale)
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)

    all_padding = torch.full_like(selected, -1)
    padded_output = sparse_attention(q, k, v, all_padding, scale=scale)
    assert torch.count_nonzero(padded_output) == 0


@pytest.mark.parametrize(
    ("case_name", "indices"),
    (
        ("boundary", [0, 3]),
        ("duplicates", [2, 2, 2]),
        ("negative", [-1, -17, 1]),
        ("high", [4, 10004]),
        ("mixed", [3, 4, 10004]),
        ("all_duplicates", [2] * 2048),
    ),
)
def test_sparse_attention_bounds_and_duplicate_semantics(
    case_name: str, indices: list[int]
) -> None:
    del case_name
    q = torch.tensor([[[[1.0, -0.5]]]])
    k = torch.tensor([[[[0.25, 0.5]], [[-0.5, 0.75]], [[1.0, -1.0]], [[0.5, 1.5]]]])
    v = torch.tensor([[[[1.0]], [[2.0]], [[4.0]], [[8.0]]]])
    selected = torch.tensor(indices, dtype=torch.int64).view(1, 1, -1)
    scale = 2**-0.5
    actual = sparse_attention(q, k, v, selected, scale=scale)
    expected = _reference_sparse_attention(q, k, v, selected, scale)
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_indexer_cache_pack_round_trip_and_exact_width() -> None:
    torch.manual_seed(3)
    keys = torch.randn(4, 9, 128, dtype=torch.float32)
    packed = pack_indexer_keys(keys)
    restored = unpack_indexer_keys(packed, dtype=torch.float32)
    assert packed.dtype is torch.uint8
    assert packed.shape == (4, 9, INDEXER_CACHE_BYTES)
    scale = packed[..., 128:].contiguous().view(torch.float32)
    raw = keys.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-4) / 448.0
    expected_scale = torch.exp2(torch.ceil(torch.log2(raw)))
    torch.testing.assert_close(scale, expected_scale)
    torch.testing.assert_close(torch.log2(scale), torch.log2(scale).round())
    torch.testing.assert_close(restored, keys, rtol=0.13, atol=0.03)


def test_mixed_fp8_mla_uint8_indexer_cache_multistep_round_trip() -> None:
    torch.manual_seed(31)
    cache = DualCacheState.allocate(
        max_batch=2,
        max_sequence_length=4,
        dtype=torch.float8_e4m3fn,
    )
    expected_mla = torch.randn(2, 2, MLA_CACHE_HEAD_SIZE) * 0.5
    expected_indexer = torch.randn(2, 2, 128) * 0.5

    for position in range(2):
        slots = torch.tensor([[0, position], [1, position]], dtype=torch.int64)
        packed_indexer = pack_indexer_keys(expected_indexer[:, position])
        cache.write(slots, expected_mla[:, position], packed_indexer)

    mla, mla_lengths = cache.read_mla(torch.tensor([0, 1]))
    packed, indexer_lengths = cache.read_indexer(torch.tensor([0, 1]))
    assert mla.dtype is torch.float8_e4m3fn
    assert packed.dtype is torch.uint8
    assert mla_lengths.tolist() == [2, 2]
    assert indexer_lengths.tolist() == [2, 2]
    assert torch.float64 not in {mla.dtype, packed.dtype}
    torch.testing.assert_close(mla[:, :2].float(), expected_mla, rtol=0.13, atol=0.03)
    torch.testing.assert_close(
        unpack_indexer_keys(packed[:, :2], dtype=torch.float32),
        expected_indexer,
        rtol=0.13,
        atol=0.03,
    )


def test_mixed_cache_read_is_fullgraph_dynamo_capturable() -> None:
    class MixedCacheRead(torch.nn.Module):
        def forward(self, mla_cache, packed_indexer):
            kv_latent, rotary = mla_cache.split((512, 64), dim=-1)
            indexer = unpack_indexer_keys(packed_indexer, dtype=torch.bfloat16)
            return kv_latent.to(torch.bfloat16), rotary.to(torch.bfloat16), indexer

    torch.manual_seed(32)
    mla_cache = torch.randn(2, 4, MLA_CACHE_HEAD_SIZE).to(torch.float8_e4m3fn)
    indexer_keys = torch.randn(2, 4, 128, dtype=torch.bfloat16)
    packed_indexer = pack_indexer_keys(indexer_keys)
    eager = MixedCacheRead()(mla_cache, packed_indexer)
    compiled = torch.compile(MixedCacheRead(), backend="eager", fullgraph=True)
    captured = compiled(mla_cache, packed_indexer)
    for actual, expected in zip(captured, eager, strict=True):
        assert actual.dtype is torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_dynamo_cache_key_separates_bf16_and_fp8_mla_inputs() -> None:
    signatures = []

    def recording_backend(graph_module, example_inputs):
        signatures.append(tuple(tensor.dtype for tensor in example_inputs))
        return graph_module.forward

    class CacheRead(torch.nn.Module):
        def forward(self, mla_cache):
            return mla_cache.to(torch.bfloat16).split((512, 64), dim=-1)

    torch._dynamo.reset()
    compiled = torch.compile(
        CacheRead(), backend=recording_backend, fullgraph=True, dynamic=False
    )
    compiled(torch.zeros(1, 4, MLA_CACHE_HEAD_SIZE, dtype=torch.bfloat16))
    compiled(torch.zeros(1, 4, MLA_CACHE_HEAD_SIZE, dtype=torch.float8_e4m3fn))
    assert signatures == [(torch.bfloat16,), (torch.float8_e4m3fn,)]


def test_cached_key_concat_supports_fp8_and_bf16_under_fake_tensor_and_dynamo() -> None:
    class CachedKeyConcat(torch.nn.Module):
        def forward(self, projected_key, cache):
            cached_rotary = cache[..., 512:].to(projected_key.dtype)
            return torch.cat((projected_key, cached_rotary), dim=-1)

    operation = CachedKeyConcat()
    for cache_dtype in (torch.float8_e4m3fn, torch.bfloat16):
        projected = torch.randn(6, 64, dtype=torch.bfloat16)
        cache = torch.randn(6, MLA_CACHE_HEAD_SIZE, dtype=torch.bfloat16).to(
            cache_dtype
        )
        expected = operation(projected, cache)
        compiled = torch.compile(operation, backend="eager", fullgraph=True)
        actual = compiled(projected, cache)
        assert actual.dtype is torch.bfloat16
        assert actual.shape == (6, 128)
        assert torch.float64 not in {projected.dtype, cache.dtype, actual.dtype}
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    from torch._subclasses.fake_tensor import FakeTensorMode

    mode = FakeTensorMode()
    with mode:
        projected = torch.empty(6, 64, dtype=torch.bfloat16)
        cache = torch.empty(6, MLA_CACHE_HEAD_SIZE, dtype=torch.float8_e4m3fn)
        result = operation(projected, cache)
        assert result.dtype is torch.bfloat16
        assert result.shape == (6, 128)


def test_cached_fp8_expansion_multistep_matches_bf16_projection(monkeypatch) -> None:
    import vllm_neuron.model.glm_moe_dsa.attention as attention_module

    class ProjectionKernelStub:
        def __getitem__(self, _grid):
            return self

        def __call__(self, cache2d, weight):
            return torch.nn.functional.linear(cache2d[:, :512].to(weight.dtype), weight)

    monkeypatch.setattr(attention_module, "can_run_kernel", lambda _tensor: True)
    monkeypatch.setattr(
        attention_module,
        "wrap_nki",
        lambda _kernel: (None, ProjectionKernelStub()),
    )
    module = GlmMoeDsaAttention(
        hidden_size=16,
        q_lora_rank=16,
        kv_lora_rank=512,
        local_heads=1,
        num_heads=1,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=8,
    ).to(dtype=torch.bfloat16)
    torch.manual_seed(48)
    source = torch.randn(1, 3, MLA_CACHE_HEAD_SIZE, dtype=torch.bfloat16)

    for cache_dtype in (torch.float8_e4m3fn, torch.bfloat16):
        cache = source.to(cache_dtype)
        for step in range(1, 4):
            current = cache[:, :step]
            projected = torch.nn.functional.linear(
                current[..., :512].to(torch.bfloat16), module.kv_b_proj.weight
            )
            expected_keys = torch.cat(
                (
                    projected[..., :64],
                    current[..., 512:].to(torch.bfloat16),
                ),
                dim=-1,
            ).unsqueeze(-2)
            expected_values = projected[..., 64:].unsqueeze(-2)
            actual_keys, actual_values = module.expand_cached_latents(current)
            assert actual_keys.dtype is torch.bfloat16
            assert actual_values.dtype is torch.bfloat16
            assert torch.float64 not in {actual_keys.dtype, actual_values.dtype}
            torch.testing.assert_close(actual_keys, expected_keys, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                actual_values, expected_values, rtol=0.0, atol=0.0
            )


def test_production_block_fp8_cached_expansion_mixed_cache_multistep() -> None:
    module = _production_block_fp8_attention()
    cache = DualCacheState.allocate(
        max_batch=1,
        max_sequence_length=3,
        dtype=torch.float8_e4m3fn,
    )
    torch.manual_seed(49)
    mla_source = torch.randn(1, 3, MLA_CACHE_HEAD_SIZE, dtype=torch.bfloat16) * 0.5
    indexer_source = torch.randn(1, 3, 128, dtype=torch.bfloat16) * 0.5
    for position in range(3):
        slots = torch.tensor([[0, position]], dtype=torch.int64)
        cache.write(
            slots,
            mla_source[:, position],
            pack_indexer_keys(indexer_source[:, position]),
        )

    mla_cache, mla_lengths = cache.read_mla(torch.tensor([0]))
    indexer_cache, indexer_lengths = cache.read_indexer(torch.tensor([0]))
    assert mla_cache.dtype is torch.float8_e4m3fn
    assert indexer_cache.dtype is torch.uint8
    assert mla_lengths.tolist() == [3]
    assert indexer_lengths.tolist() == [3]
    torch.testing.assert_close(
        unpack_indexer_keys(indexer_cache[:, :3], dtype=torch.bfloat16),
        indexer_source,
        rtol=0.13,
        atol=0.03,
    )

    dequantized_weight = dequantize_block_fp8(
        module.kv_b_proj.weight,
        module.kv_b_proj.weight_scale_inv,
        row_offset=module.kv_b_proj.row_offset,
        col_offset=module.kv_b_proj.col_offset,
    ).to(torch.bfloat16)
    for step in range(1, 4):
        current = mla_cache[:, :step]
        projected = torch.nn.functional.linear(
            current[..., :512].to(torch.bfloat16), dequantized_weight
        )
        expected_keys = torch.cat(
            (projected[..., :64], current[..., 512:].to(torch.bfloat16)), dim=-1
        ).unsqueeze(-2)
        expected_values = projected[..., 64:].unsqueeze(-2)
        actual_keys, actual_values = module.expand_cached_latents(current)
        assert actual_keys.dtype is torch.bfloat16
        assert actual_values.dtype is torch.bfloat16
        assert torch.float64 not in {
            mla_cache.dtype,
            indexer_cache.dtype,
            actual_keys.dtype,
            actual_values.dtype,
        }
        torch.testing.assert_close(actual_keys, expected_keys, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_values, expected_values, rtol=0.0, atol=0.0)


def test_production_block_fp8_cached_expansion_fake_tensor_and_dynamo() -> None:
    module = _production_block_fp8_attention()
    cached_latents = (
        torch.randn(1, 3, MLA_CACHE_HEAD_SIZE, dtype=torch.bfloat16)
        .clamp(-4.0, 4.0)
        .to(torch.float8_e4m3fn)
    )
    expected = module.expand_cached_latents(cached_latents)
    captured_graph = {}

    def capture_backend(graph_module, _example_inputs):
        captured_graph["module"] = graph_module
        return graph_module.forward

    compiled = torch.compile(
        module.expand_cached_latents,
        backend=capture_backend,
        fullgraph=True,
    )
    actual = compiled(cached_latents)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.dtype is torch.bfloat16
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)

    nodes = tuple(captured_graph["module"].graph.nodes)
    forbidden_split_sections = {(512, 64), (64, 8)}
    assert not any(
        node.op == "call_method"
        and node.target == "split"
        and len(node.args) > 1
        and node.args[1] in forbidden_split_sections
        for node in nodes
    )
    cache_slices = tuple(
        node.args[1]
        for node in nodes
        if node.op == "call_function"
        and node.target.__name__ == "getitem"
        and isinstance(node.args[1], tuple)
        and node.args[1]
        and node.args[1][0] is Ellipsis
    )
    assert (Ellipsis, slice(None, 512)) in cache_slices
    assert (Ellipsis, slice(512, None)) in cache_slices
    assert (Ellipsis, slice(None, 64)) in cache_slices
    assert (Ellipsis, slice(64, None)) in cache_slices

    from torch._subclasses.fake_tensor import FakeTensorMode

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        fake_cache = torch.empty(1, 3, MLA_CACHE_HEAD_SIZE, dtype=torch.float8_e4m3fn)
        fake_keys, fake_values = module.expand_cached_latents(fake_cache)
        assert fake_keys.dtype is torch.bfloat16
        assert fake_values.dtype is torch.bfloat16
        assert fake_keys.shape == (1, 3, 1, 128)
        assert fake_values.shape == (1, 3, 1, 8)


def test_arithmetic_pack_matches_pinned_fp8_and_fp32_bytes() -> None:
    torch.manual_seed(30)
    keys = torch.randn(2, 257, 128) * 100.0
    packed = pack_indexer_keys(keys)
    quantized, scale = quantize_ue8m0_fp8(keys, eps=1.0e-4)
    expected = torch.cat(
        (
            quantized.contiguous().view(torch.uint8),
            scale.contiguous().view(torch.uint8),
        ),
        dim=-1,
    )
    assert torch.equal(packed, expected)
    torch.testing.assert_close(
        unpack_indexer_keys(packed, dtype=torch.float32),
        quantized.float() * scale,
    )


def test_arithmetic_pack_matches_pinned_fp8_boundary_bytes() -> None:
    epsilon_above_272 = torch.nextafter(torch.tensor(272.0), torch.tensor(float("inf")))
    boundary = torch.tensor(
        [256.0, 272.0, epsilon_above_272, 288.0, 336.0, 400.0, 432.0, 448.0]
    )
    boundary = torch.cat((boundary, -boundary))
    keys = torch.zeros(1, 128, dtype=torch.float32)
    keys[0, : boundary.numel()] = boundary

    packed = pack_indexer_keys(keys)
    expected_values = keys.to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
    expected_scale = torch.ones(1, 1, dtype=torch.float32)
    expected = torch.cat(
        (expected_values, expected_scale.contiguous().view(torch.uint8)), dim=-1
    )

    assert torch.equal(
        packed[0, 128:], torch.tensor([0, 0, 128, 63], dtype=torch.uint8)
    )
    assert torch.equal(packed, expected)
    torch.testing.assert_close(
        unpack_indexer_keys(packed, dtype=torch.float32),
        keys.to(torch.float8_e4m3fn).float(),
    )


def test_causal_topk_matches_independent_score_reference() -> None:
    torch.manual_seed(31)
    queries = torch.randn(1, 3, 2, 4)
    keys = torch.randn(1, 5, 4)
    head_weights = torch.randn(1, 3, 2)
    query_positions = torch.tensor([[0, 2, 4]])
    key_positions = torch.arange(5).unsqueeze(0)
    actual = causal_topk_indices(
        queries,
        keys,
        head_weights,
        query_positions,
        key_positions,
        topk=2,
    )

    expected_rows = []
    q_absmax = queries.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10)
    q_scale = torch.exp2(torch.ceil(torch.log2(q_absmax / 448.0)))
    q_quant = (queries / q_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    k_absmax = keys.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-4)
    k_scale = torch.exp2(torch.ceil(torch.log2(k_absmax / 448.0)))
    k_quant = (keys / k_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    for query in range(3):
        score = torch.zeros(5)
        for head in range(2):
            # Transformers computes ReLU on each scaled head score before
            # applying the learned head weight and reducing across heads.
            per_head = (q_quant[0, query, head].float() @ k_quant[0].float().T) * (
                q_scale[0, query, head, 0] * k_scale[0, :, 0] * 4**-0.5
            )
            score += torch.relu(per_head) * (head_weights[0, query, head] * 2**-0.5)
        score[key_positions[0] > query_positions[0, query]] = float("-inf")
        values, indices = torch.topk(score, 2)
        expected_rows.append(torch.where(torch.isfinite(values), indices, -1))
    expected = torch.stack(expected_rows).unsqueeze(0)
    assert torch.equal(actual, expected)


def test_causal_topk_applies_relu_before_head_reduction() -> None:
    """Counterexample for the pre-fix signed per-head score reduction."""

    queries = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    keys = torch.tensor([[[-10.0, -20.0], [1.0, 0.0]]])
    head_weights = torch.tensor([[[1.0, -1.0]]])
    positions = torch.tensor([[1]])
    key_positions = torch.tensor([[0, 1]])

    actual = causal_topk_indices(
        queries,
        keys,
        head_weights,
        positions,
        key_positions,
        topk=1,
    )

    # Upstream ReLU removes both negative scores for key 0, so key 1 wins.
    # The old signed reduction ranked key 0 first: -10 - (-20) > 1 - 0.
    assert torch.equal(actual, torch.tensor([[[1]]]))


def _independent_dsa_membership_reference(
    queries: torch.Tensor,
    keys: torch.Tensor,
    head_weights: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    """Compute pinned DSA membership without the streaming implementation."""

    q_absmax = queries.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10)
    q_scale = torch.exp2(torch.ceil(torch.log2(q_absmax / 448.0)))
    q_quant = (queries / q_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    k_absmax = keys.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-4)
    k_scale = torch.exp2(torch.ceil(torch.log2(k_absmax / 448.0)))
    k_quant = (keys / k_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    logits = torch.zeros(
        queries.shape[0],
        queries.shape[1],
        keys.shape[1],
        dtype=torch.float32,
    )
    transposed_keys = k_quant.float().transpose(1, 2)
    for head in range(queries.shape[2]):
        per_head = torch.matmul(
            q_quant[:, :, head].float(),
            transposed_keys,
        )
        per_head = per_head * q_scale[:, :, head]
        per_head = per_head * k_scale.transpose(1, 2)
        logits += torch.relu(per_head * queries.shape[-1] ** -0.5) * (
            head_weights[:, :, head].unsqueeze(-1) * queries.shape[2] ** -0.5
        )
    logits = logits.masked_fill(
        key_positions.unsqueeze(1) > query_positions.unsqueeze(-1),
        torch.finfo(torch.float32).min,
    )
    return torch.topk(logits, topk, dim=-1).indices


def test_pinned_ue8m0_top2048_membership_probe() -> None:
    """Pinned-style 4096-key probe that detects raw-scale/full-Q scoring."""

    torch.manual_seed(123)
    queries = torch.randn(1, 1, 32, 128)
    keys = torch.randn(1, 4096, 128)
    head_weights = torch.randn(1, 1, 32)
    positions = torch.tensor([[4095]])
    key_positions = torch.arange(4096).unsqueeze(0)
    actual = causal_topk_indices(
        queries,
        keys,
        head_weights,
        positions,
        key_positions,
        topk=2048,
    )
    expected = _independent_dsa_membership_reference(
        queries,
        keys,
        head_weights,
        positions,
        key_positions,
        topk=2048,
    )
    assert torch.equal(actual.sort().values, expected.sort().values)


def test_long_context_dsa_streams_exact_scores_in_bounded_key_tiles(
    monkeypatch,
) -> None:
    """>2048 DSA preserves exact top-k without a full-width QK tensor."""

    torch.manual_seed(2049)
    queries = torch.randn(1, 2, 2, 128)
    keys = torch.randn(1, 2305, 128)
    head_weights = torch.randn(1, 2, 2)
    query_positions = torch.tensor([[1023, 2304]])
    key_positions = torch.arange(2305).unsqueeze(0)
    topk = 16

    q_absmax = queries.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10)
    q_scale = torch.exp2(torch.ceil(torch.log2(q_absmax / 448.0)))
    q_quant = (queries / q_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    k_absmax = keys.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-4)
    k_scale = torch.exp2(torch.ceil(torch.log2(k_absmax / 448.0)))
    k_quant = (keys / k_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    direct_scores = torch.zeros(1, 2, 2305)
    for query in range(2):
        for head in range(2):
            per_head = (q_quant[0, query, head].float() @ k_quant[0].float().T) * (
                q_scale[0, query, head, 0] * k_scale[0, :, 0] * 128**-0.5
            )
            direct_scores[0, query] += torch.relu(per_head) * (
                head_weights[0, query, head] * 2**-0.5
            )
    direct_scores = direct_scores.masked_fill(
        key_positions.unsqueeze(1) > query_positions.unsqueeze(-1),
        torch.finfo(torch.float32).min,
    )
    expected = torch.topk(direct_scores, topk, dim=-1).indices

    direct_matmul = torch.matmul
    qk_widths = []

    def recording_matmul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        qk_widths.append(rhs.shape[-1])
        return direct_matmul(lhs, rhs)

    original_topk = indexer_module.neuron_topk
    merge_widths = []

    def recording_topk(tensor: torch.Tensor, *args, **kwargs):
        merge_widths.append(tensor.shape[-1])
        return original_topk(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "matmul", recording_matmul)
    monkeypatch.setattr(indexer_module, "neuron_topk", recording_topk)
    actual = causal_topk_indices(
        queries,
        keys,
        head_weights,
        query_positions,
        key_positions,
        topk=topk,
    )

    assert torch.equal(actual.sort().values, expected.sort().values)
    assert max(qk_widths) == 256
    assert max(merge_widths) <= topk + 256
    assert len(qk_widths) == 10


def test_causal_topk_bypasses_scores_for_four_absolute_prefill_segments(
    monkeypatch,
) -> None:
    """Four 512-token segments select absolute causal indices through 2048."""

    def fail_if_scored(*args, **kwargs):
        raise AssertionError("the <=2048 causal bypass must not score QK")

    monkeypatch.setattr(torch, "matmul", fail_if_scored)
    keys = torch.zeros(1, 2048, 2)
    key_positions = torch.arange(2048).unsqueeze(0)

    for segment in range(4):
        start = segment * 512
        stop = start + 512
        query_positions = torch.arange(start, stop).unsqueeze(0)
        queries = torch.zeros(1, 512, 1, 2)
        head_weights = torch.ones(1, 512, 1)
        selected = causal_topk_indices(
            queries,
            keys,
            head_weights,
            query_positions,
            key_positions,
            topk=2048,
        )

        expected_indices = torch.arange(2048).view(1, 1, -1)
        expected = torch.where(
            expected_indices <= query_positions.unsqueeze(-1),
            expected_indices,
            -torch.ones_like(expected_indices),
        )
        assert torch.equal(selected, expected)


def test_cache_spec_has_separate_mla_and_indexer_entries_and_no_mtp() -> None:
    spec = build_glm_mla_cache_spec()
    mla = [layer for layer in spec.layers if layer.name.endswith("mla_cache")]
    indexer = [layer for layer in spec.layers if layer.name.endswith("indexer.k_cache")]
    assert len(mla) == 78
    assert all(layer.num_kv_heads == 1 for layer in mla)
    assert all(layer.head_size == MLA_CACHE_PART_SIZE for layer in mla)
    assert all(layer.dtype is torch.bfloat16 for layer in mla)
    assert len(indexer) == len(MAIN_INDEXER_LAYER_INDICES) == 21
    assert all(layer.head_size == INDEXER_CACHE_PART_BYTES for layer in indexer)
    assert all(layer.dtype is torch.uint8 for layer in indexer)
    assert not any("layers.78." in layer.name for layer in spec.layers)


def test_indexer_schedule_reuses_latest_selection_and_excludes_mtp() -> None:
    assert latest_indexer_layer(0) == 0
    assert latest_indexer_layer(3) == 2
    assert latest_indexer_layer(73) == 70
    assert latest_indexer_layer(74) == 74
    assert latest_indexer_layer(77) == 74
    with pytest.raises(ValueError, match="outside main execution"):
        latest_indexer_layer(78)


def test_q_and_kv_lora_projection_reconstructs_cached_latents() -> None:
    torch.manual_seed(4)
    module = GlmMoeDsaAttention(
        hidden_size=12,
        q_lora_rank=8,
        kv_lora_rank=512,
        local_heads=1,
        qk_nope_head_dim=8,
        qk_rope_head_dim=64,
        v_head_dim=6,
    )
    hidden = torch.randn(2, 5, 12)
    positions = torch.arange(5).expand(2, -1)
    projection = module.project(hidden, positions)
    assert projection.q_lora.shape == (2, 5, 8)
    assert projection.queries.shape == (2, 5, 1, 72)
    assert projection.latent_cache.shape == (2, 5, MLA_CACHE_HEAD_SIZE)
    assert projection.values.shape == (2, 5, 1, 6)

    def rms_reference(values, weight, eps):
        return (
            values
            * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
            * weight
        )

    q_lora = torch.nn.functional.linear(hidden, module.q_a_proj.weight)
    q_lora = rms_reference(q_lora, module.q_a_layernorm.weight, 1.0e-5)
    q = torch.nn.functional.linear(q_lora, module.q_b_proj.weight).view(2, 5, 1, 72)
    q_nope, q_pe = q.split((8, 64), dim=-1)
    latent_raw = torch.nn.functional.linear(hidden, module.kv_a_proj_with_mqa.weight)
    kv_latent, k_pe = latent_raw.split((512, 64), dim=-1)
    kv_latent = rms_reference(kv_latent, module.kv_a_layernorm.weight, 1.0e-5)
    cos, sin = rotary_cos_sin(positions, 64)
    q_pairs = torch.view_as_complex(q_pe.view(2, 5, 1, 32, 2).contiguous())
    k_pairs = torch.view_as_complex(k_pe.view(2, 5, 32, 2).contiguous())
    rotation = torch.complex(cos, sin)
    q_pe = torch.view_as_real(q_pairs * rotation.unsqueeze(-2)).flatten(-2)
    k_pe = torch.view_as_real(k_pairs * rotation).flatten(-2)
    expected_queries = torch.cat((q_nope, q_pe), dim=-1)
    expected_cache = torch.cat((kv_latent, k_pe), dim=-1)
    torch.testing.assert_close(projection.q_lora, q_lora)
    torch.testing.assert_close(projection.queries, expected_queries)
    torch.testing.assert_close(projection.latent_cache, expected_cache)

    reconstructed_keys, reconstructed_values = module.expand_cached_latents(
        projection.latent_cache
    )
    torch.testing.assert_close(reconstructed_keys, projection.keys)
    torch.testing.assert_close(reconstructed_values, projection.values)
    selected = _causal_selection(2, 5, 5)
    output = module.attend(projection.queries, projection.latent_cache, selected)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()


def test_expand_cached_latents_preserves_projection_checkpoint_contract(
    monkeypatch,
) -> None:
    def assert_equivalent(
        module: GlmMoeDsaAttention, latent_cache: torch.Tensor
    ) -> None:
        registered_weight = module.kv_b_proj.weight
        weight_shape = registered_weight.shape
        weight_before = registered_weight.detach().clone()
        loader = get_weight_loader(registered_weight)
        state_before = {
            name: tuple(tensor.shape) for name, tensor in module.state_dict().items()
        }
        kv_latent = latent_cache[..., : module.kv_lora_rank]
        k_pe = latent_cache[..., module.kv_lora_rank :]
        original_kv = torch.nn.functional.linear(kv_latent, registered_weight).view(
            *latent_cache.shape[:-1],
            module.local_heads,
            module.qk_nope_head_dim + module.v_head_dim,
        )
        expected_k_nope, expected_values = original_kv.split(
            (module.qk_nope_head_dim, module.v_head_dim), dim=-1
        )
        expected_keys = torch.cat(
            (
                expected_k_nope,
                k_pe.unsqueeze(-2).expand(
                    *k_pe.shape[:-1],
                    module.local_heads,
                    module.qk_rope_head_dim,
                ),
            ),
            dim=-1,
        )

        actual_keys, actual_values = module.expand_cached_latents(latent_cache)

        torch.testing.assert_close(actual_keys, expected_keys)
        torch.testing.assert_close(actual_values, expected_values)
        assert module.kv_b_proj.weight is registered_weight
        assert module.kv_b_proj.weight.shape == weight_shape
        assert get_weight_loader(module.kv_b_proj.weight) is loader
        assert torch.equal(module.kv_b_proj.weight, weight_before)
        assert {
            name: tuple(tensor.shape) for name, tensor in module.state_dict().items()
        } == state_before

    torch.manual_seed(41)
    artificial = GlmMoeDsaAttention(
        hidden_size=16,
        q_lora_rank=16,
        kv_lora_rank=512,
        local_heads=1,
        num_heads=1,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=8,
    )
    assert artificial.kv_b_proj.weight.shape == (72, 512)
    assert_equivalent(artificial, torch.randn(2, 7, MLA_CACHE_HEAD_SIZE))

    group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _=None: 64)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _=None: 7)
    pinned = GlmMoeDsaAttention(
        hidden_size=6144,
        q_lora_rank=2048,
        kv_lora_rank=512,
        local_heads=1,
        num_heads=64,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        tp_group=group,
    )
    assert pinned.kv_b_proj.weight.shape == (448, 512)
    assert_equivalent(pinned, torch.randn(1, 5, MLA_CACHE_HEAD_SIZE))


@pytest.mark.parametrize(
    ("query_count", "key_count"),
    (
        (1, 1),
        (256, 256),
        (512, 512),
        (2048, 2048),
        (1, 2048),
        (8, 2048),
        (32, 2048),
        (257, 513),
    ),
)
def test_sparse_score_tiling_matches_direct_matmul(
    query_count: int,
    key_count: int,
) -> None:
    torch.manual_seed(2048 + query_count + key_count)
    queries = torch.randint(-2, 3, (1, query_count, 1, 8)).to(torch.bfloat16)
    keys = torch.randint(-2, 3, (1, key_count, 1, 8)).to(torch.bfloat16)

    expected = torch.matmul(
        queries.float().permute(0, 2, 1, 3),
        keys.float().permute(0, 2, 3, 1),
    )
    actual = _sparse_score_matmul(queries, keys)

    assert torch.equal(actual, expected)


def test_sparse_score_tiling_only_splits_query_and_key_axes(monkeypatch) -> None:
    queries = torch.randn(1, 257, 1, 8, dtype=torch.bfloat16)
    keys = torch.randn(1, 513, 1, 8, dtype=torch.bfloat16)
    direct_matmul = torch.matmul
    calls = []

    def recording_matmul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        calls.append((lhs.shape[-2:], rhs.shape[-2:]))
        return direct_matmul(lhs, rhs)

    monkeypatch.setattr(torch, "matmul", recording_matmul)
    _sparse_score_matmul(queries, keys)

    assert calls == [
        (torch.Size((256, 8)), torch.Size((8, 256))),
        (torch.Size((256, 8)), torch.Size((8, 256))),
        (torch.Size((256, 8)), torch.Size((8, 1))),
        (torch.Size((1, 8)), torch.Size((8, 256))),
        (torch.Size((1, 8)), torch.Size((8, 256))),
        (torch.Size((1, 8)), torch.Size((8, 1))),
    ]


@pytest.mark.parametrize("prompt_tokens", PREFILL_LENGTHS)
def test_required_sparse_prefill_variant(prompt_tokens: int) -> None:
    """Four canonical sparse-prefill work items."""

    torch.manual_seed(10 + prompt_tokens)
    q = torch.randn(1, prompt_tokens, 1, 4)
    k = torch.randn(1, prompt_tokens, 4)
    weights = torch.ones(1, prompt_tokens, 1)
    positions = torch.arange(prompt_tokens)
    selected = causal_topk_indices(q, k, weights, positions, positions, topk=2048)
    values = torch.randn(1, prompt_tokens, 1, 3)
    output = sparse_attention(q, k.unsqueeze(2), values, selected)

    assert selected.shape == (1, prompt_tokens, 2048)
    assert output.shape == (1, prompt_tokens, 1, 3)
    assert torch.isfinite(output).all()
    valid_counts = (selected >= 0).sum(dim=-1)
    torch.testing.assert_close(
        valid_counts, torch.arange(1, prompt_tokens + 1).unsqueeze(0)
    )
    gathered_positions = torch.where(selected >= 0, selected, 0)
    assert torch.all(
        torch.where(
            selected >= 0,
            gathered_positions <= positions.view(1, -1, 1),
            True,
        )
    )


@pytest.mark.parametrize(("batch", "output_tokens"), DECODE_VARIANTS)
def test_required_sparse_decode_variant(batch: int, output_tokens: int) -> None:
    """Nine canonical sparse-decode work items."""

    torch.manual_seed(1000 + batch * 100 + output_tokens)
    prompt_tokens = 8
    total = prompt_tokens + output_tokens
    q = torch.randn(batch, output_tokens, 1, 4)
    k = torch.randn(batch, total, 4)
    weights = torch.randn(batch, output_tokens, 1)
    query_positions = torch.arange(prompt_tokens, total).expand(batch, -1)
    key_positions = torch.arange(total).expand(batch, -1)
    selected = causal_topk_indices(
        q, k, weights, query_positions, key_positions, topk=2048
    )
    values = torch.randn(batch, total, 1, 5)
    output = sparse_attention(q, k.unsqueeze(2), values, selected)

    assert selected.shape == (batch, output_tokens, 2048)
    assert output.shape == (batch, output_tokens, 1, 5)
    assert torch.isfinite(output).all()
    assert torch.equal(
        (selected >= 0).sum(dim=-1),
        query_positions + 1,
    )


@pytest.mark.parametrize("tokens", PREFILL_LENGTHS)
def test_required_prefill_cache_variant(tokens: int) -> None:
    """Four canonical prefill cache work items."""

    cache = DualCacheState.allocate(max_batch=2, max_sequence_length=tokens + 1)
    slots = torch.stack(
        (torch.zeros(tokens, dtype=torch.int64), torch.arange(tokens)), dim=1
    )
    mla = torch.arange(tokens, dtype=torch.float32).unsqueeze(1).expand(-1, 576)
    indexer = torch.arange(tokens, dtype=torch.uint8).unsqueeze(1).expand(-1, 132)
    cache.write(slots, mla, indexer)

    assert cache.lengths.tolist() == [tokens, 0]
    expected_mla = torch.arange(tokens).to(torch.bfloat16).float()
    torch.testing.assert_close(cache.mla[0, :tokens, 0].float(), expected_mla)
    assert torch.count_nonzero(cache.mla[1]) == 0
    assert cache.indexer is not None
    assert torch.count_nonzero(cache.indexer[1]) == 0


@pytest.mark.parametrize("batch", (1, 8, 32))
def test_required_decode_cache_variant(batch: int) -> None:
    """Three canonical decode-cache work items and prefill continuity."""

    cache = DualCacheState.allocate(max_batch=batch, max_sequence_length=3)
    request = torch.arange(batch, dtype=torch.int64)
    prefill_slots = torch.stack((request, torch.zeros_like(request)), dim=1)
    prefill_mla = request.to(torch.float32).unsqueeze(1).expand(-1, 576)
    prefill_indexer = request.to(torch.uint8).unsqueeze(1).expand(-1, 132)
    cache.write(prefill_slots, prefill_mla, prefill_indexer)

    decode_slots = torch.stack((request, torch.ones_like(request)), dim=1)
    decode_mla = (request + 100).to(torch.float32).unsqueeze(1).expand(-1, 576)
    decode_indexer = (request + 100).to(torch.uint8).unsqueeze(1).expand(-1, 132)
    cache.write(decode_slots, decode_mla, decode_indexer)

    assert cache.lengths.tolist() == [2] * batch
    torch.testing.assert_close(cache.mla[:, 0, 0].float(), request.float())
    torch.testing.assert_close(cache.mla[:, 1, 0].float(), (request + 100).float())
    assert cache.indexer is not None
    assert torch.equal(cache.indexer[:, 0, 0], request.to(torch.uint8))
    assert torch.equal(cache.indexer[:, 1, 0], (request + 100).to(torch.uint8))


def test_no_forbidden_distributed_runtime_references() -> None:
    package = Path(__file__).parents[4] / "vllm_neuron" / "model" / "glm_moe_dsa"
    distributed_name = "neuronx_" + "distributed"
    forbidden = (distributed_name, distributed_name + "_inference", "Nx" + "DI")
    sources = [
        package / "attention.py",
        package / "indexer.py",
        package / "cache.py",
        Path(__file__),
    ]
    for source in sources:
        text = source.read_text()
        assert not any(token in text for token in forbidden)


def test_tp64_attention_projection_shapes_and_loaders(monkeypatch) -> None:
    group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _=None: 64)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _=None: 7)

    module = GlmMoeDsaAttention(
        hidden_size=6144,
        q_lora_rank=2048,
        kv_lora_rank=512,
        local_heads=1,
        num_heads=64,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        tp_group=group,
    )

    assert isinstance(module.q_b_proj, ColumnParallelLinear)
    assert isinstance(module.kv_b_proj, ColumnParallelLinear)
    assert isinstance(module.o_proj, RowParallelLinear)
    assert module.q_b_proj.weight.shape == (256, 2048)
    assert module.kv_b_proj.weight.shape == (448, 512)
    assert module.o_proj.weight.shape == (6144, 256)

    class VirtualSlice:
        def __init__(self, shape):
            self.shape = shape
            self.requests = []

        def get_shape(self):
            return self.shape

        def __getitem__(self, item):
            self.requests.append(item)
            rows = torch.arange(*item[0].indices(self.shape[0]), dtype=torch.int64)
            cols = torch.arange(*item[1].indices(self.shape[1]), dtype=torch.int64)
            return rows[:, None] * self.shape[1] + cols[None, :]

    q_full = VirtualSlice((16384, 2048))
    q_shard = get_weight_loader(module.q_b_proj.weight).load([q_full], 7)
    assert q_full.requests == [(slice(7 * 256, 8 * 256), slice(None))]
    assert q_shard.shape == (256, 2048)
    assert q_shard[0, 0] == (7 * 256) * 2048
    assert q_shard[-1, -1] == (8 * 256) * 2048 - 1

    kv_full = VirtualSlice((28672, 512))
    kv_shard = get_weight_loader(module.kv_b_proj.weight).load([kv_full], 7)
    assert kv_full.requests == [(slice(7 * 448, 8 * 448), slice(None))]
    assert kv_shard.shape == (448, 512)
    assert kv_shard[0, 0] == (7 * 448) * 512
    assert kv_shard[-1, -1] == (8 * 448) * 512 - 1

    o_full = VirtualSlice((6144, 16384))
    o_shard = get_weight_loader(module.o_proj.weight).load([o_full], 7)
    assert o_full.requests == [(slice(None), slice(7 * 256, 8 * 256))]
    assert o_shard.shape == (6144, 256)
    assert o_shard[0, 0] == 7 * 256
    assert o_shard[-1, -1] == 6144 * 16384 - (64 - 8) * 256 - 1
    assert hasattr(module.q_a_proj.weight, "weight_loader")
    assert hasattr(module.kv_a_proj_with_mqa.weight, "weight_loader")


def test_o_proj_uses_the_full_tp_group_collective(monkeypatch) -> None:
    from vllm_neuron.nn import rpl

    group = object()
    calls = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _: 0)
    monkeypatch.setattr(rpl, "_NATIVE", True)

    def record_all_reduce(tensor, reduceOp, group):
        calls.append((reduceOp, group))
        return tensor

    monkeypatch.setattr(rpl, "all_reduce", record_all_reduce)
    module = GlmMoeDsaAttention(
        hidden_size=8,
        q_lora_rank=4,
        kv_lora_rank=512,
        local_heads=1,
        num_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=64,
        v_head_dim=4,
        tp_group=group,
    )
    module.o_proj(torch.ones(1, 1, 4))
    assert calls == [("sum", group)]


@pytest.mark.skipif(
    os.environ.get("GLM_STAGE3_RUN_NEURON_COMPILE") != "1",
    reason="explicit scoped Neuron compile smoke",
)
def test_neuron_compile_and_execute_nki_expansion_n72_n448(monkeypatch) -> None:
    """Compile NKI cached expansion for artificial and pinned local widths."""

    artificial = GlmMoeDsaAttention(
        hidden_size=16,
        q_lora_rank=16,
        kv_lora_rank=512,
        local_heads=1,
        num_heads=1,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=8,
    )
    group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _=None: 64)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _=None: 7)
    pinned = GlmMoeDsaAttention(
        hidden_size=6144,
        q_lora_rank=2048,
        kv_lora_rank=512,
        local_heads=1,
        num_heads=64,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        tp_group=group,
    )

    class ExpansionSmoke(torch.nn.Module):
        def __init__(self, attention):
            super().__init__()
            self.attention = attention

        def forward(self, cached_latents):
            return self.attention.expand_cached_latents(cached_latents)

    device = torch.device("neuron:0")
    for label, attention in (("N72", artificial), ("N448", pinned)):
        torch.manual_seed(43)
        cached_latents = torch.randn(1, 2048, MLA_CACHE_HEAD_SIZE, dtype=torch.bfloat16)
        attention = attention.to(dtype=torch.bfloat16)
        smoke = ExpansionSmoke(attention)
        expected = smoke(cached_latents)
        compiled = torch.compile(
            smoke.to(device),
            backend="vllm_neuron",
            fullgraph=True,
            dynamic=False,
            options={
                "compiler_workdir": os.path.join(
                    os.environ["GLM_STAGE3_COMPILE_DIR"], label.lower()
                )
            },
        )
        actual = compiled(cached_latents.to(device))
        actual_cpu = tuple(tensor.cpu() for tensor in actual)
        for kind, actual_tensor, expected_tensor in zip(
            ("KEYS", "VALUES"), actual_cpu, expected, strict=True
        ):
            assert actual_tensor.shape == expected_tensor.shape, kind
            assert torch.isfinite(actual_tensor).all(), kind
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=5.0e-2, atol=5.0e-2
            )


@pytest.mark.skipif(
    os.environ.get("GLM_STAGE3_RUN_NEURON_COMPILE") != "1",
    reason="explicit scoped Neuron compile smoke",
)
def test_neuron_compile_and_execute_mixed_fp8_mla_cache_expansion() -> None:
    """Compile one FP8 MLA cache read through the production NKI expansion."""

    attention = GlmMoeDsaAttention(
        hidden_size=16,
        q_lora_rank=16,
        kv_lora_rank=512,
        local_heads=1,
        num_heads=1,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=8,
    ).to(dtype=torch.bfloat16)

    class ExpansionSmoke(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, cached_latents):
            return self.module.expand_cached_latents(cached_latents)

    torch.manual_seed(44)
    cached_latents = (
        torch.randn(1, 128, MLA_CACHE_HEAD_SIZE, dtype=torch.bfloat16)
        .clamp(-4.0, 4.0)
        .to(torch.float8_e4m3fn)
    )
    smoke = ExpansionSmoke(attention)
    expected = smoke(cached_latents.to(torch.bfloat16))
    device = torch.device("neuron:0")
    compiled = torch.compile(
        smoke.to(device),
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={
            "compiler_workdir": os.path.join(
                os.environ["GLM_STAGE3_COMPILE_DIR"], "mixed_fp8_mla"
            )
        },
    )
    actual = tuple(tensor.cpu() for tensor in compiled(cached_latents.to(device)))
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.isfinite(actual_tensor).all()
        torch.testing.assert_close(
            actual_tensor.float(),
            expected_tensor.float(),
            rtol=0.13,
            atol=0.08,
        )


@pytest.mark.skipif(
    os.environ.get("GLM_STAGE3_RUN_NEURON_COMPILE") != "1",
    reason="explicit scoped Neuron compile smoke",
)
def test_neuron_compile_and_execute_production_block_fp8_cached_expansion() -> None:
    """Compile the pinned FP8-weight branch with an FP8 MLA cache."""

    attention = _production_block_fp8_attention()
    assert isinstance(attention.kv_b_proj, BlockFP8ColumnParallelLinear)

    class ExpansionSmoke(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, cached_latents):
            return self.module.expand_cached_latents(cached_latents)

    torch.manual_seed(50)
    cached_latents = (
        torch.randn(1, 128, MLA_CACHE_HEAD_SIZE, dtype=torch.bfloat16)
        .clamp(-4.0, 4.0)
        .to(torch.float8_e4m3fn)
    )
    smoke = ExpansionSmoke(attention)
    expected = smoke(cached_latents)
    assert torch.count_nonzero(attention.kv_b_proj.weight.float()) > 0
    assert torch.isfinite(attention.kv_b_proj.weight_scale_inv).all()
    assert torch.all(attention.kv_b_proj.weight_scale_inv > 0)
    device = torch.device("neuron:0")
    compiled = torch.compile(
        smoke.to(device),
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={
            "compiler_workdir": os.path.join(
                os.environ["GLM_STAGE3_COMPILE_DIR"], "mixed_fp8_block_weights"
            )
        },
    )
    actual = tuple(tensor.cpu() for tensor in compiled(cached_latents.to(device)))
    for kind, actual_tensor, expected_tensor in zip(
        ("KEYS", "VALUES"), actual, expected, strict=True
    ):
        assert torch.isfinite(actual_tensor).all(), kind
        assert actual_tensor.dtype is torch.bfloat16
        torch.testing.assert_close(
            actual_tensor.float(), expected_tensor.float(), rtol=0.13, atol=0.08
        )


@pytest.mark.skipif(
    os.environ.get("GLM_STAGE3_RUN_LONG_CONTEXT_DSA") != "1",
    reason="explicit long-context streaming DSA hardware smoke",
)
def test_neuron_compile_and_execute_long_context_streaming_dsa(monkeypatch) -> None:
    """Compile the smallest production-shaped >2048 DSA selection graph."""

    class StreamingDsaSmoke(torch.nn.Module):
        def forward(
            self,
            queries,
            keys,
            head_weights,
            query_positions,
            key_positions,
        ):
            return causal_topk_indices(
                queries,
                keys,
                head_weights,
                query_positions,
                key_positions,
                topk=2048,
            )

    torch.manual_seed(2050)
    queries = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)
    keys = torch.randn(1, 4096, 128, dtype=torch.bfloat16)
    head_weights = torch.randn(1, 1, 32, dtype=torch.bfloat16)
    query_positions = torch.tensor([[4095]], dtype=torch.int64)
    key_positions = torch.arange(4096, dtype=torch.int64).unsqueeze(0)
    smoke = StreamingDsaSmoke()
    expected = _independent_dsa_membership_reference(
        queries,
        keys,
        head_weights,
        query_positions,
        key_positions,
        topk=2048,
    )

    direct_matmul = torch.matmul
    original_topk = indexer_module.neuron_topk
    qk_widths = []
    merge_widths = []

    def recording_matmul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        qk_widths.append(rhs.shape[-1])
        return direct_matmul(lhs, rhs)

    def recording_topk(tensor: torch.Tensor, *args, **kwargs):
        merge_widths.append(tensor.shape[-1])
        return original_topk(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "matmul", recording_matmul)
        patch.setattr(indexer_module, "neuron_topk", recording_topk)
        eager = smoke(
            queries,
            keys,
            head_weights,
            query_positions,
            key_positions,
        )
    assert torch.equal(eager.sort().values, expected.sort().values)
    assert qk_widths == [indexer_module._DSA_SCORE_TILE_SIZE] * 16
    assert max(merge_widths) <= 2048 + indexer_module._DSA_SCORE_TILE_SIZE

    device = torch.device("neuron:0")
    compiled = torch.compile(
        smoke.to(device),
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={
            "compiler_workdir": os.environ["GLM_STAGE3_LONG_CONTEXT_DSA_COMPILE_DIR"]
        },
    )
    actual = compiled(
        queries.to(device),
        keys.to(device),
        head_weights.to(device),
        query_positions.to(device),
        key_positions.to(device),
    ).cpu()

    assert actual.shape == (1, 1, 2048)
    assert torch.equal(actual.sort().values, expected.sort().values)


@pytest.mark.skipif(
    os.environ.get("GLM_STAGE3_RUN_NEURON_COMPILE") != "1",
    reason="explicit scoped Neuron compile smoke",
)
def test_neuron_compile_and_execute_full_stage3_subgraph() -> None:
    """Compile and execute DSA projection/selection plus MLA attention."""

    class Smoke(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.indexer = GlmMoeDsaIndexer(
                hidden_size=16,
                q_lora_rank=16,
                num_heads=32,
                head_dim=128,
                rope_dim=64,
                topk=2048,
            )
            self.attention = GlmMoeDsaAttention(
                hidden_size=16,
                q_lora_rank=16,
                kv_lora_rank=512,
                local_heads=1,
                num_heads=1,
                qk_nope_head_dim=64,
                qk_rope_head_dim=64,
                v_head_dim=8,
            )

        def forward(
            self,
            hidden_states,
            cached_indexer_cache,
            cached_latents,
            positions,
            key_positions,
        ):
            attention_projection = self.attention.project(hidden_states, positions)
            indexer_projection = self.indexer.project(
                hidden_states, attention_projection.q_lora, positions
            )
            packed_current_key = pack_indexer_keys(indexer_projection.keys)
            cached_indexer_keys = unpack_indexer_keys(
                cached_indexer_cache, dtype=hidden_states.dtype
            )
            selected = self.indexer.select(
                indexer_projection,
                cached_indexer_keys,
                positions,
                key_positions,
            )
            output = self.attention.attend(
                attention_projection.queries, cached_latents, selected
            )
            return packed_current_key, selected, output

    torch.manual_seed(99)
    hidden_states = torch.randn(1, 1, 16, dtype=torch.bfloat16)
    cached_indexer_keys = torch.randn(1, 2048, 128, dtype=torch.bfloat16)
    cached_indexer_cache = pack_indexer_keys(cached_indexer_keys)
    cached_latents = torch.randn(1, 2048, 576, dtype=torch.bfloat16)
    positions = torch.tensor([[2047]], dtype=torch.int64)
    key_positions = torch.arange(2048, dtype=torch.int64).unsqueeze(0)
    smoke = Smoke().to(dtype=torch.bfloat16)
    expected = smoke(
        hidden_states,
        cached_indexer_cache,
        cached_latents,
        positions,
        key_positions,
    )

    device = torch.device("neuron:0")
    compile_dir = os.environ["GLM_STAGE3_COMPILE_DIR"]
    compiled = torch.compile(
        smoke.to(device),
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={"compiler_workdir": compile_dir},
    )
    actual = compiled(
        hidden_states.to(device),
        cached_indexer_cache.to(device),
        cached_latents.to(device),
        positions.to(device),
        key_positions.to(device),
    )
    actual_packed = actual[0].cpu()
    expected_packed = expected[0]
    actual_unpacked = unpack_indexer_keys(actual_packed, dtype=torch.bfloat16)
    expected_unpacked = unpack_indexer_keys(expected_packed, dtype=torch.bfloat16)
    torch.testing.assert_close(
        actual_unpacked,
        expected_unpacked,
        rtol=1.3e-1,
        atol=3.0e-2,
    )
    actual_selected = actual[1].cpu()
    expected_selected = expected[1]
    actual_sorted = torch.sort(actual_selected, dim=-1).values
    expected_sorted = torch.sort(expected_selected, dim=-1).values
    actual_unique = torch.unique(actual_selected)
    in_range = bool(((actual_selected >= 0) & (actual_selected < 2048)).all().item())
    set_equal = torch.equal(actual_sorted, expected_sorted)
    assert actual_selected.shape == (1, 1, 2048)
    assert in_range
    assert actual_unique.numel() == 2048
    assert set_equal

    actual_output = actual[2].cpu()
    torch.testing.assert_close(
        actual_output,
        expected[2],
        rtol=5.0e-2,
        atol=5.0e-2,
    )
