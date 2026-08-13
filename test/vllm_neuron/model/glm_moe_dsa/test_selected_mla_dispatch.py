# SPDX-License-Identifier: Apache-2.0
"""CPU/static dispatch gates for the opt-in selected-latent MLA path."""

from __future__ import annotations

import inspect

import pytest
import torch

import vllm_neuron.model.glm_moe_dsa.attention as attention_module
from vllm_neuron.model.glm_moe_dsa.attention import (
    SELECTED_LATENT_MLA_ENV,
    GlmMoeDsaAttention,
)
from vllm_neuron.model.glm_moe_dsa.model import GlmMoeDsaDecoderLayer


class _OutputIdentity(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


def _pinned_attention() -> GlmMoeDsaAttention:
    attention = GlmMoeDsaAttention(
        hidden_size=8,
        q_lora_rank=8,
        kv_lora_rank=512,
        local_heads=1,
        num_heads=64,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        fp8_weights=True,
        dtype=torch.bfloat16,
        device="cpu",
    )
    attention.o_proj = _OutputIdentity()
    return attention


def _pinned_inputs(*, is_decode: bool = True):
    del is_decode
    queries = torch.zeros(1, 1, 1, 256, dtype=torch.bfloat16)
    k_cache = torch.zeros(258, 1, 16, 288, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.remainder(
        torch.arange(255, -1, -1, dtype=torch.int32) * 73 + 1,
        257,
    ).view(1, 256)
    selected = torch.arange(2048, dtype=torch.int64).view(1, 1, 2048)
    return queries, selected, k_cache, v_cache, block_table


def test_default_off_uses_existing_dense_attention(monkeypatch) -> None:
    monkeypatch.delenv(SELECTED_LATENT_MLA_ENV, raising=False)
    attention = _pinned_attention()
    queries, selected, k_cache, v_cache, block_table = _pinned_inputs()
    assert not attention.should_use_selected_latent_mla(
        queries,
        selected,
        mla_k_cache=k_cache,
        mla_v_cache=v_cache,
        block_table=block_table,
        block_size=16,
        is_decode=True,
    )

    dense_cache = torch.zeros(1, 4, 576, dtype=torch.bfloat16)
    fallback_calls: list[str] = []

    def expand(_cache: torch.Tensor):
        fallback_calls.append("expand")
        keys = torch.zeros(1, 4, 1, 256, dtype=torch.bfloat16)
        values = torch.zeros_like(keys)
        return keys, values

    def fallback(*args, **kwargs):
        del args, kwargs
        fallback_calls.append("attention")
        return torch.zeros_like(queries)

    def selected_kernel(*args, **kwargs):
        del args, kwargs
        raise AssertionError("default-off path must not launch selected MLA")

    monkeypatch.setattr(attention, "expand_cached_latents", expand)
    monkeypatch.setattr(attention_module, "sparse_attention", fallback)
    monkeypatch.setattr(attention_module, "selected_latent_mla_decode", selected_kernel)
    output = attention.attend(queries, dense_cache, selected)

    assert output.shape == (1, 1, 256)
    assert fallback_calls == ["expand", "attention"]


@pytest.mark.parametrize("row_offset", (0, 64))
def test_opt_in_long_decode_dispatches_without_dense_expansion(
    monkeypatch,
    row_offset: int,
) -> None:
    monkeypatch.setenv(SELECTED_LATENT_MLA_ENV, "1")
    monkeypatch.setattr(attention_module, "can_run_kernel", lambda tensor: True)
    attention = _pinned_attention()
    attention.kv_b_proj.row_offset = row_offset
    queries, selected, k_cache, v_cache, block_table = _pinned_inputs()
    launches: list[dict[str, object]] = []

    def fail_dense(*args, **kwargs):
        del args, kwargs
        raise AssertionError("opt-in selected MLA must not expand the full cache")

    def selected_kernel(
        actual_queries,
        actual_k_cache,
        actual_v_cache,
        actual_block_table,
        actual_selected,
        actual_weight,
        actual_scales,
        *,
        block_size,
        row_offset,
    ):
        launches.append(
            {
                "queries": actual_queries,
                "k_cache": actual_k_cache,
                "v_cache": actual_v_cache,
                "block_table": actual_block_table,
                "selected_dtype": actual_selected.dtype,
                "weight": actual_weight,
                "scales": actual_scales,
                "block_size": block_size,
                "row_offset": row_offset,
            }
        )
        return torch.ones_like(actual_queries)

    monkeypatch.setattr(attention, "expand_cached_latents", fail_dense)
    monkeypatch.setattr(attention_module, "sparse_attention", fail_dense)
    monkeypatch.setattr(attention_module, "selected_latent_mla_decode", selected_kernel)

    assert attention.should_use_selected_latent_mla(
        queries,
        selected,
        mla_k_cache=k_cache,
        mla_v_cache=v_cache,
        block_table=block_table,
        block_size=16,
        is_decode=True,
    )
    output = attention.attend_selected_latents(
        queries,
        selected,
        k_cache,
        v_cache,
        block_table,
        16,
    )

    assert torch.equal(output, torch.ones(1, 1, 256, dtype=torch.bfloat16))
    assert len(launches) == 1
    launch = launches[0]
    assert launch["k_cache"] is k_cache
    assert launch["v_cache"] is v_cache
    assert launch["block_table"] is block_table
    assert launch["selected_dtype"] is torch.int32
    assert launch["weight"] is attention.kv_b_proj.weight
    assert launch["scales"] is attention.kv_b_proj.weight_scale_inv
    assert launch["block_size"] == 16
    assert launch["row_offset"] == row_offset


@pytest.mark.parametrize(
    ("is_decode", "query_count", "logical_blocks"),
    ((False, 1, 256), (True, 2, 256), (True, 1, 128), (True, 1, 257)),
)
def test_opt_in_unsupported_shapes_preserve_fallback(
    monkeypatch,
    is_decode: bool,
    query_count: int,
    logical_blocks: int,
) -> None:
    monkeypatch.setenv(SELECTED_LATENT_MLA_ENV, "1")
    monkeypatch.setattr(attention_module, "can_run_kernel", lambda tensor: True)
    attention = _pinned_attention()
    queries = torch.zeros(1, query_count, 1, 256, dtype=torch.bfloat16)
    selected = torch.zeros(1, query_count, 2048, dtype=torch.int64)
    k_cache = torch.zeros(2, 1, 16, 288, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(1, logical_blocks, dtype=torch.int32)

    assert not attention.should_use_selected_latent_mla(
        queries,
        selected,
        mla_k_cache=k_cache,
        mla_v_cache=v_cache,
        block_table=block_table,
        block_size=16,
        is_decode=is_decode,
    )


def test_opt_in_long_decode_invalid_contract_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv(SELECTED_LATENT_MLA_ENV, "1")
    monkeypatch.setattr(attention_module, "can_run_kernel", lambda tensor: True)
    attention = _pinned_attention()
    queries, selected, k_cache, v_cache, block_table = _pinned_inputs()
    attention.num_heads = 32

    with pytest.raises(ValueError, match="production contract violation"):
        attention.should_use_selected_latent_mla(
            queries,
            selected,
            mla_k_cache=k_cache,
            mla_v_cache=v_cache,
            block_table=block_table,
            block_size=16,
            is_decode=True,
        )


def test_model_gathers_only_in_selected_mla_fallback_branch() -> None:
    source = inspect.getsource(GlmMoeDsaDecoderLayer.forward)
    decision = source.index("use_selected_latent_mla =")
    fallback = source.index("if not use_selected_latent_mla:", decision)
    dispatch = source.index("if use_selected_latent_mla:", fallback)
    dense = source.index("else:", dispatch)

    assert "gather_paged_cache_pair" not in source[decision:fallback]
    assert "gather_paged_cache_pair" in source[fallback:dispatch]
    assert "gather_paged_cache_pair" not in source[dispatch:dense]
    assert "attend_selected_latents" in source[dispatch:dense]
    assert "selected_latent_mla_decode" not in source


def test_dispatch_decision_is_fullgraph_static() -> None:
    source = inspect.getsource(GlmMoeDsaAttention.should_use_selected_latent_mla)

    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert "logical_key_count != 4096" in source
