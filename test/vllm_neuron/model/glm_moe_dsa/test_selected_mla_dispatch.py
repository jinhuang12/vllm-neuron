# SPDX-License-Identifier: Apache-2.0
"""CPU/static dispatch gates for the opt-in selected-latent MLA path."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch

import vllm_neuron.model.glm_moe_dsa.attention as attention_module
from vllm_neuron.model.glm_moe_dsa.attention import (
    SELECTED_LATENT_MLA_CONTEXT_BUCKETS,
    SELECTED_LATENT_MLA_ENV,
    GlmMoeDsaAttention,
)
from vllm_neuron.model.glm_moe_dsa.model import GlmMoeDsaDecoderLayer

from test_sparse_mla import (
    _deterministic_inputs,
    _poison_unselected_physical_rows,
    _tiled_online_reference,
)


class _OutputIdentity(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


class _IntegratedSelectedMlaDecodeProbe(torch.nn.Module):
    """Exact guarded attention boundary for the proven row-offset-64 graph."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
    ) -> None:
        super().__init__()
        self.attention = _pinned_attention()
        self.attention.kv_b_proj.row_offset = 64
        with torch.no_grad():
            self.attention.kv_b_proj.weight.copy_(weight)
            self.attention.kv_b_proj.weight_scale_inv.copy_(weight_scale_inv)

    def forward(
        self,
        queries: torch.Tensor,
        mla_k_cache: torch.Tensor,
        mla_v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        enabled = self.attention.should_use_selected_latent_mla(
            queries,
            selected_indices,
            mla_k_cache=mla_k_cache,
            mla_v_cache=mla_v_cache,
            block_table=block_table,
            block_size=16,
            is_decode=True,
        )
        if not enabled:
            raise RuntimeError("exact selected-latent MLA contract was not enabled")
        return self.attention.attend_selected_latents(
            queries,
            selected_indices,
            mla_k_cache,
            mla_v_cache,
            block_table,
            16,
        )


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


def _pinned_inputs(
    *,
    is_decode: bool = True,
    logical_length: int = 4096,
    cache_dtype: torch.dtype = torch.float8_e4m3fn,
):
    del is_decode
    assert logical_length % 16 == 0
    queries = torch.zeros(1, 1, 1, 256, dtype=torch.bfloat16)
    k_cache = torch.zeros(258, 1, 16, 288, dtype=torch.bfloat16).to(cache_dtype)
    v_cache = torch.zeros_like(k_cache)
    logical_blocks = logical_length // 16
    block_table = torch.remainder(
        torch.arange(logical_blocks - 1, -1, -1, dtype=torch.int32) * 73 + 1,
        257,
    ).view(1, logical_blocks)
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


@pytest.mark.parametrize("logical_length", SELECTED_LATENT_MLA_CONTEXT_BUCKETS)
@pytest.mark.parametrize("row_offset", (0, 64))
def test_opt_in_long_decode_dispatches_without_dense_expansion(
    monkeypatch,
    row_offset: int,
    logical_length: int,
) -> None:
    monkeypatch.setenv(SELECTED_LATENT_MLA_ENV, "1")
    monkeypatch.setattr(attention_module, "can_run_kernel", lambda tensor: True)
    attention = _pinned_attention()
    attention.kv_b_proj.row_offset = row_offset
    queries, selected, k_cache, v_cache, block_table = _pinned_inputs(
        logical_length=logical_length
    )
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
    assert k_cache.dtype is torch.float8_e4m3fn


@pytest.mark.parametrize(
    ("is_decode", "query_count", "logical_blocks"),
    ((False, 1, 256), (True, 2, 256), (True, 1, 128)),
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


@pytest.mark.parametrize("logical_length", (4096 - 16, 6144, 16384))
def test_opt_in_unsupported_long_bucket_fails_closed(
    monkeypatch,
    logical_length: int,
) -> None:
    monkeypatch.setenv(SELECTED_LATENT_MLA_ENV, "1")
    monkeypatch.setattr(attention_module, "can_run_kernel", lambda tensor: True)
    attention = _pinned_attention()
    queries, selected, k_cache, v_cache, block_table = _pinned_inputs(
        logical_length=logical_length
    )

    with pytest.raises(
        ValueError, match=f"unsupported context bucket {logical_length}"
    ):
        attention.should_use_selected_latent_mla(
            queries,
            selected,
            mla_k_cache=k_cache,
            mla_v_cache=v_cache,
            block_table=block_table,
            block_size=16,
            is_decode=True,
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
    assert "logical_key_count <= 2048" in source
    assert "logical_key_count not in SELECTED_LATENT_MLA_CONTEXT_BUCKETS" in source
    assert SELECTED_LATENT_MLA_CONTEXT_BUCKETS == (4096, 8192)


def test_integrated_fullgraph_probe_has_no_dense_mla_path() -> None:
    probe_source = inspect.getsource(_IntegratedSelectedMlaDecodeProbe.forward)
    selected_source = inspect.getsource(GlmMoeDsaAttention.attend_selected_latents)

    assert "should_use_selected_latent_mla" in probe_source
    assert "attend_selected_latents" in probe_source
    assert "gather_paged_cache_pair" not in probe_source + selected_source
    assert "expand_cached_latents" not in probe_source + selected_source
    assert "selected_latent_mla_decode" in selected_source


def test_integrated_hardware_probe_uses_one_pinned_fullgraph() -> None:
    source = inspect.getsource(
        test_integrated_selected_latent_mla_t4096_q1_k2048_row64_neuron
    )

    assert 'backend="vllm_neuron"' in source
    assert "fullgraph=True" in source
    assert "dynamic=False" in source
    assert 'torch.device("neuron:0")' in source
    assert 'os.environ["GLM_STAGE3_SELECTED_MLA_INTEGRATED_COMPILE_DIR"]' in source
    assert "compiled(*device_inputs)" in source
    assert "compiled(" in source[source.index("all_padding") :]


@pytest.mark.skipif(
    os.environ.get("GLM_STAGE3_SELECTED_MLA_INTEGRATED_HARDWARE") != "1"
    or not Path("/dev/neuron0").exists(),
    reason=(
        "requires explicit GLM_STAGE3_SELECTED_MLA_INTEGRATED_HARDWARE=1 on Neuron"
    ),
)
def test_integrated_selected_latent_mla_t4096_q1_k2048_row64_neuron() -> None:
    if os.environ.get(SELECTED_LATENT_MLA_ENV) != "1":
        pytest.fail(f"{SELECTED_LATENT_MLA_ENV}=1 must be set before module creation")

    device = torch.device("neuron:0")
    (
        queries,
        logical_cache,
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected,
        weight,
        weight_scale_inv,
    ) = _deterministic_inputs()
    reference = _tiled_online_reference(
        queries,
        logical_cache,
        selected,
        block_table,
        weight,
        weight_scale_inv,
        row_offset=64,
    ).squeeze(-2)
    poisoned_k, poisoned_v = _poison_unselected_physical_rows(
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected,
    )

    module = _IntegratedSelectedMlaDecodeProbe(weight, weight_scale_inv).eval()
    assert module.attention.enable_selected_latent_mla
    module = module.to(device)
    compile_root = Path(os.environ["GLM_STAGE3_SELECTED_MLA_INTEGRATED_COMPILE_DIR"])
    compiled = torch.compile(
        module,
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={"compiler_workdir": str(compile_root / "t4096-k2048-row64")},
    )
    device_inputs = (
        queries.to(device),
        poisoned_k.to(device),
        poisoned_v.to(device),
        block_table.to(device),
        selected.to(device),
    )
    actual = compiled(*device_inputs).cpu()

    all_padding = torch.full_like(selected, -1)
    padded_actual = compiled(
        queries.to(device),
        torch.full(mla_k_cache.shape, 240.0, dtype=torch.bfloat16)
        .to(mla_k_cache.dtype)
        .to(device),
        torch.full(mla_v_cache.shape, 240.0, dtype=torch.bfloat16)
        .to(mla_v_cache.dtype)
        .to(device),
        block_table.to(device),
        all_padding.to(device),
    ).cpu()

    assert actual.shape == (2, 1, 256)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(padded_actual, torch.zeros_like(padded_actual))
    torch.testing.assert_close(actual, reference, rtol=2.0e-2, atol=2.0e-2)
