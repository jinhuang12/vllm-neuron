# SPDX-License-Identifier: Apache-2.0
"""CPU and static contracts for the 512-wide selected-MLA specialization."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

import vllm_neuron.model.glm_moe_dsa.attention as attention_module
from vllm_neuron.model.glm_moe_dsa.attention import (
    SELECTED_LATENT_MLA_ENV,
    GlmMoeDsaAttention,
)
from vllm_neuron.model.glm_moe_dsa.model import GlmMoeDsaDecoderLayer
from vllm_neuron.model.glm_moe_dsa.sparse_mla import (
    SELECTED_LATENT_MLA_BLOCK_SIZES,
    SELECTED_LATENT_MLA_LONG_WIDTH,
    SELECTED_LATENT_MLA_SHORT_CONTEXT_BUCKETS,
    SELECTED_LATENT_MLA_SHORT_WIDTH,
    _selected_latent_mla_decode_nki,
    validate_selected_latent_mla_decode_contract,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.bucket_utils import validate_decode_context_length_buckets
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner


_CACHE_HALF_WIDTH = 288


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


def _dispatch_inputs(
    *,
    batch_size: int,
    logical_length: int,
    block_size: int,
    cache_dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    assert logical_length % block_size == 0
    queries = torch.zeros(batch_size, 1, 1, 256, dtype=torch.bfloat16)
    physical_block_count = max(4, logical_length // block_size + 2)
    k_cache = torch.zeros(
        physical_block_count,
        1,
        block_size,
        _CACHE_HALF_WIDTH,
        dtype=torch.bfloat16,
    ).to(cache_dtype)
    v_cache = torch.zeros_like(k_cache)
    logical_block_count = logical_length // block_size
    logical_blocks = torch.arange(logical_block_count, dtype=torch.int32)
    block_table = torch.stack(
        [
            torch.remainder(
                logical_blocks * (2 * batch + 1) + batch,
                physical_block_count,
            )
            for batch in range(batch_size)
        ]
    )
    selected = torch.full(
        (batch_size, 1, SELECTED_LATENT_MLA_LONG_WIDTH),
        -1,
        dtype=torch.int64,
    )
    selected[:, 0, :logical_length] = torch.arange(logical_length)
    if logical_length > 24:
        selected[:, 0, 23] = selected[:, 0, 22]
    return queries, selected, k_cache, v_cache, block_table


@pytest.mark.parametrize(
    (
        "batch_size",
        "logical_length",
        "block_size",
        "cache_dtype",
        "row_offset",
    ),
    (
        (1, 128, 16, torch.bfloat16, 0),
        (32, 128, 32, torch.float8_e4m3fn, 64),
        (1, 512, 32, torch.float8_e4m3fn, 0),
        (32, 512, 32, torch.bfloat16, 64),
    ),
)
def test_short_decode_dispatches_exactly_512_indices_without_dense_expansion(
    monkeypatch,
    batch_size: int,
    logical_length: int,
    block_size: int,
    cache_dtype: torch.dtype,
    row_offset: int,
) -> None:
    monkeypatch.setenv(SELECTED_LATENT_MLA_ENV, "1")
    monkeypatch.setattr(attention_module, "can_run_kernel", lambda tensor: True)
    attention = _pinned_attention()
    attention.kv_b_proj.row_offset = row_offset
    queries, selected, k_cache, v_cache, block_table = _dispatch_inputs(
        batch_size=batch_size,
        logical_length=logical_length,
        block_size=block_size,
        cache_dtype=cache_dtype,
    )
    launches: list[torch.Tensor] = []

    def fail_dense(*args, **kwargs):
        del args, kwargs
        raise AssertionError("selected MLA must not expand or gather the full cache")

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
        del actual_weight, actual_scales
        assert actual_k_cache is k_cache
        assert actual_v_cache is v_cache
        assert actual_block_table is block_table
        assert block_size in SELECTED_LATENT_MLA_BLOCK_SIZES
        assert row_offset in (0, 64)
        launches.append(actual_selected)
        return torch.zeros_like(actual_queries)

    monkeypatch.setattr(attention, "expand_cached_latents", fail_dense)
    monkeypatch.setattr(attention_module, "sparse_attention", fail_dense)
    monkeypatch.setattr(attention_module, "selected_latent_mla_decode", selected_kernel)

    assert attention.should_use_selected_latent_mla(
        queries,
        selected,
        mla_k_cache=k_cache,
        mla_v_cache=v_cache,
        block_table=block_table,
        block_size=block_size,
        is_decode=True,
    )
    output = attention.attend_selected_latents(
        queries,
        selected,
        k_cache,
        v_cache,
        block_table,
        block_size,
    )

    assert output.shape == (batch_size, 1, 256)
    assert len(launches) == 1
    assert launches[0].shape == (batch_size, 1, SELECTED_LATENT_MLA_SHORT_WIDTH)
    assert launches[0].dtype is torch.int32
    assert torch.equal(launches[0], selected[..., :512].to(torch.int32))
    assert k_cache.dtype is cache_dtype


@pytest.mark.parametrize(
    ("logical_length", "block_size"),
    ((16, 16), (480, 32)),
)
def test_non_emittable_short_shapes_preserve_fallback(
    monkeypatch,
    logical_length: int,
    block_size: int,
) -> None:
    monkeypatch.setenv(SELECTED_LATENT_MLA_ENV, "1")
    monkeypatch.setattr(attention_module, "can_run_kernel", lambda tensor: True)
    attention = _pinned_attention()
    queries, selected, k_cache, v_cache, block_table = _dispatch_inputs(
        batch_size=1,
        logical_length=logical_length,
        block_size=block_size,
        cache_dtype=torch.bfloat16,
    )

    assert not attention.should_use_selected_latent_mla(
        queries,
        selected,
        mla_k_cache=k_cache,
        mla_v_cache=v_cache,
        block_table=block_table,
        block_size=block_size,
        is_decode=True,
    )


@pytest.mark.parametrize("logical_length", (544, 1024, 2048))
def test_contexts_above_512_and_at_most_2048_preserve_fallback(
    monkeypatch,
    logical_length: int,
) -> None:
    monkeypatch.setenv(SELECTED_LATENT_MLA_ENV, "1")
    monkeypatch.setattr(attention_module, "can_run_kernel", lambda tensor: True)
    attention = _pinned_attention()
    queries, selected, k_cache, v_cache, block_table = _dispatch_inputs(
        batch_size=1,
        logical_length=logical_length,
        block_size=32,
        cache_dtype=torch.bfloat16,
    )

    assert not attention.should_use_selected_latent_mla(
        queries,
        selected,
        mla_k_cache=k_cache,
        mla_v_cache=v_cache,
        block_table=block_table,
        block_size=32,
        is_decode=True,
    )


def test_q32_and_default_off_preserve_fallback(monkeypatch) -> None:
    monkeypatch.setattr(attention_module, "can_run_kernel", lambda tensor: True)
    inputs = _dispatch_inputs(
        batch_size=1,
        logical_length=512,
        block_size=32,
        cache_dtype=torch.float8_e4m3fn,
    )
    _, selected, k_cache, v_cache, block_table = inputs

    monkeypatch.setenv(SELECTED_LATENT_MLA_ENV, "1")
    attention = _pinned_attention()
    q32 = torch.zeros(1, 32, 1, 256, dtype=torch.bfloat16)
    selected_q32 = selected.expand(1, 32, -1)
    assert not attention.should_use_selected_latent_mla(
        q32,
        selected_q32,
        mla_k_cache=k_cache,
        mla_v_cache=v_cache,
        block_table=block_table,
        block_size=32,
        is_decode=True,
    )

    monkeypatch.delenv(SELECTED_LATENT_MLA_ENV, raising=False)
    default_off = _pinned_attention()
    assert not default_off.should_use_selected_latent_mla(
        inputs[0],
        selected,
        mla_k_cache=k_cache,
        mla_v_cache=v_cache,
        block_table=block_table,
        block_size=32,
        is_decode=True,
    )


def test_short_contract_accepts_only_the_512_wide_kernel_input() -> None:
    attention = _pinned_attention()
    queries, selected, k_cache, v_cache, block_table = _dispatch_inputs(
        batch_size=1,
        logical_length=512,
        block_size=32,
        cache_dtype=torch.float8_e4m3fn,
    )
    selected_512 = attention._selected_latent_mla_indices(
        selected,
        block_table=block_table,
        block_size=32,
    )
    validate_selected_latent_mla_decode_contract(
        queries,
        k_cache,
        v_cache,
        block_table,
        selected_512,
        attention.kv_b_proj.weight,
        attention.kv_b_proj.weight_scale_inv,
        block_size=32,
        row_offset=64,
    )

    with pytest.raises(ValueError, match="selected width must be 512"):
        validate_selected_latent_mla_decode_contract(
            queries,
            k_cache,
            v_cache,
            block_table,
            selected.to(torch.int32),
            attention.kv_b_proj.weight,
            attention.kv_b_proj.weight_scale_inv,
            block_size=32,
            row_offset=64,
        )
    with pytest.raises(ValueError, match="512- or 2048-wide"):
        attention._selected_latent_mla_indices(
            selected[..., :768],
            block_table=block_table,
            block_size=32,
    )


class _RuntimeBlockTable:
    def __init__(self, batch_size: int, max_model_len: int, block_size: int) -> None:
        block_count = max_model_len // block_size
        self._device = torch.arange(block_count, dtype=torch.int32).repeat(
            batch_size, 1
        )
        self.slot_mapping = SimpleNamespace(
            gpu=torch.arange(batch_size, dtype=torch.int64)
        )

    def get_device_tensor(self, request_count: int) -> torch.Tensor:
        return self._device[:request_count]


def test_runner_480_input_32_output_lifetime_dispatches_width512(
    monkeypatch,
) -> None:
    """Join the real runner bucket pick to selected-MLA dispatch.

    Future hardware A/B runs use this same ``[128, 512]`` runner config in
    both arms. The baseline leaves ``GLM_ENABLE_EXPERIMENTAL_SELECTED_LATENT_MLA``
    unset; the candidate sets it to ``1``.
    """

    max_model_len = 2048
    block_size = 32
    input_length = 480
    output_length = 32
    decode_buckets = validate_decode_context_length_buckets(
        [128, 512], max_model_len
    )
    runner = object.__new__(NeuronModelRunner)
    runner.device = torch.device("cpu")
    runner.drafter = None
    runner.speculative_config = None
    runner.max_model_len = max_model_len
    runner._dcp_size = 1
    runner.cp_world_size = 1
    runner._cp_rank = 0
    runner._is_synthetic_model = False
    runner.neuron_config = NeuronConfig.from_dict(
        {"decode_context_length_buckets": decode_buckets}
    )
    layer_name = "model.layers.0.self_attn.mla_cache"
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=block_size),
                layer_names=[layer_name],
            )
        ]
    )
    runner.input_batch = SimpleNamespace(
        block_table=[_RuntimeBlockTable(32, max_model_len, block_size)]
    )

    runtime_metadata = None
    for generated_tokens in range(output_length):
        max_decode_context = input_length + generated_tokens
        assert (
            runner._decode_ctx_bucket_from_max_decode_ctx_len(
                max_decode_context,
                max_num_draft_tokens=0,
            )
            == 512
        )
        runtime_metadata = runner._build_attention_metadata(
            padded_num_reqs=32,
            total_num_scheduled_tokens=32,
            max_query_len=1,
            max_num_draft_tokens=0,
            max_decode_ctx_len=max_decode_context,
        )[layer_name]
        assert runtime_metadata["block_size"] == block_size
        assert runtime_metadata["block_table_tensor"].shape == (32, 16)
        assert runtime_metadata["max_blocks_per_seq"] * block_size == 512

    assert runtime_metadata is not None
    monkeypatch.setenv(SELECTED_LATENT_MLA_ENV, "1")
    monkeypatch.setattr(attention_module, "can_run_kernel", lambda tensor: True)
    attention = _pinned_attention()
    queries = torch.zeros(32, 1, 1, 256, dtype=torch.bfloat16)
    selected = torch.full((32, 1, 2048), -1, dtype=torch.int64)
    selected[..., :512] = torch.arange(512)
    k_cache = torch.zeros(64, 1, block_size, _CACHE_HALF_WIDTH).to(
        torch.float8_e4m3fn
    )
    v_cache = torch.zeros_like(k_cache)

    assert attention.should_use_selected_latent_mla(
        queries,
        selected,
        mla_k_cache=k_cache,
        mla_v_cache=v_cache,
        block_table=runtime_metadata["block_table_tensor"],
        block_size=runtime_metadata["block_size"],
        is_decode=True,
    )


def _address_case(
    batch_size: int,
    cache_dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    block_size = 32
    logical_length = 512
    physical_block_count = 19
    logical_blocks = torch.arange(logical_length // block_size, dtype=torch.int32)
    block_table = torch.stack(
        [
            torch.remainder(logical_blocks * (2 * batch + 3) + batch, 17)
            for batch in range(batch_size)
        ]
    )
    block_table[:, 1] = block_table[:, 0]
    block_table[:, 3] = -1
    block_table[:, 7] = physical_block_count + 5

    selected = torch.stack(
        [
            torch.arange(logical_length, dtype=torch.int32)
            if batch % 2 == 0
            else torch.arange(logical_length - 1, -1, -1, dtype=torch.int32)
            for batch in range(batch_size)
        ]
    ).unsqueeze(1)
    boundaries = torch.tensor((0, 15, 16, 31, 32, 47, 63, 64, 95, 96))
    selected[:, 0, : boundaries.numel()] = boundaries
    selected[:, 0, 17] = 3 * block_size + 5
    selected[:, 0, 19] = 7 * block_size + 6
    selected[:, 0, 23] = selected[:, 0, 22]
    selected[:, 0, 29] = logical_length + 9
    selected[:, :, -31:] = -1

    oracle_rows = torch.zeros_like(selected, dtype=torch.int64)
    oracle_valid = torch.zeros_like(selected, dtype=torch.bool)
    table_values = block_table.tolist()
    for batch, batch_selection in enumerate(selected[:, 0].tolist()):
        for column, logical_row in enumerate(batch_selection):
            if logical_row < 0 or logical_row >= logical_length:
                continue
            logical_block, row_in_block = divmod(logical_row, block_size)
            physical_block = table_values[batch][logical_block]
            if physical_block < 0 or physical_block >= physical_block_count:
                continue
            oracle_rows[batch, 0, column] = physical_block * block_size + row_in_block
            oracle_valid[batch, 0, column] = True

    poison = 240.0 if cache_dtype is torch.float8_e4m3fn else torch.nan
    physical = torch.full(
        (physical_block_count, block_size, 2 * _CACHE_HALF_WIDTH),
        poison,
        dtype=torch.bfloat16,
    )
    chosen_rows = torch.unique(oracle_rows[oracle_valid])
    markers = (torch.remainder(chosen_rows, 17).to(torch.bfloat16) - 8).div(4)
    physical.view(-1, 2 * _CACHE_HALF_WIDTH)[chosen_rows] = markers[:, None]
    return (
        block_table,
        selected,
        physical.to(cache_dtype),
        oracle_rows,
        oracle_valid,
    )


@pytest.mark.parametrize("batch_size", (1, 32))
@pytest.mark.parametrize("cache_dtype", (torch.bfloat16, torch.float8_e4m3fn))
def test_block32_alias_invalid_duplicate_padding_and_poison_semantics(
    batch_size: int,
    cache_dtype: torch.dtype,
) -> None:
    block_table, selected, physical, oracle_rows, oracle_valid = _address_case(
        batch_size,
        cache_dtype,
    )
    physical_block_count = physical.shape[0]
    logical_length = block_table.shape[1] * 32
    safe = selected.clamp(0, logical_length - 1)
    logical_blocks = torch.bitwise_right_shift(safe, 5)
    rows_in_block = torch.bitwise_and(safe, 31)
    table = block_table[:, None, :].expand(-1, selected.shape[1], -1)
    physical_blocks = torch.gather(table, 2, logical_blocks.to(torch.int64))
    actual_valid = (selected >= 0) & (selected < logical_length)
    actual_valid &= (physical_blocks >= 0) & (
        physical_blocks < physical_block_count
    )
    actual_rows = (physical_blocks * 32 + rows_in_block).clamp(
        0, physical_block_count * 32 - 1
    )
    observable_rows = torch.where(
        actual_valid,
        actual_rows,
        torch.zeros_like(actual_rows),
    )

    assert torch.equal(observable_rows, oracle_rows)
    assert torch.equal(actual_valid, oracle_valid)
    assert torch.equal(block_table[:, 0], block_table[:, 1])
    assert torch.equal(selected[:, :, 22], selected[:, :, 23])
    assert not actual_valid[:, :, 17].any()
    assert not actual_valid[:, :, 19].any()
    assert not actual_valid[:, :, 29].any()
    assert not actual_valid[:, :, -31:].any()

    flat = physical.reshape(-1, 2 * _CACHE_HALF_WIDTH)
    selected_row_mask = torch.zeros(flat.shape[0], dtype=torch.bool)
    selected_row_mask[oracle_rows[oracle_valid]] = True
    if cache_dtype is torch.bfloat16:
        assert torch.isnan(flat[~selected_row_mask]).all()
    else:
        assert (flat[~selected_row_mask].float() == 240.0).all()
    fingerprints = torch.stack((flat[:, 0], flat[:, -1]), dim=-1).to(torch.bfloat16)
    actual = torch.where(
        actual_valid[..., None], fingerprints[observable_rows], torch.zeros(())
    )
    expected = torch.where(
        oracle_valid[..., None], fingerprints[oracle_rows], torch.zeros(())
    )
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()

    all_padding = torch.full_like(selected, -1)
    padding_valid = (all_padding >= 0) & (all_padding < logical_length)
    padded = torch.where(
        padding_valid[..., None], fingerprints[actual_rows], torch.zeros(())
    )
    assert torch.equal(padded, torch.zeros_like(padded))


def test_kernel_and_dispatch_are_static_and_do_not_materialize_dense_cache() -> None:
    kernel_source = inspect.getsource(_selected_latent_mla_decode_nki)
    selected_source = inspect.getsource(GlmMoeDsaAttention.attend_selected_latents)
    model_source = inspect.getsource(GlmMoeDsaDecoderLayer.forward)
    dispatch = model_source.index("if use_selected_latent_mla:")
    fallback = model_source.index("else:", dispatch)

    assert SELECTED_LATENT_MLA_SHORT_CONTEXT_BUCKETS == (128, 512)
    assert SELECTED_LATENT_MLA_SHORT_WIDTH // 128 == 4
    assert SELECTED_LATENT_MLA_LONG_WIDTH // 128 == 16
    assert "selected_count // _SELECTED_TILE" in kernel_source
    assert "nl.dynamic_range" not in kernel_source
    assert "register_" not in kernel_source
    assert "if block_size == 16:" in kernel_source
    assert "block_shift = 5" in kernel_source
    assert "block_row_mask = 31" in kernel_source
    assert kernel_source.count("vector_offset=physical_rows") == 2
    assert kernel_source.count("buffer=nl.shared_hbm") == 1
    assert "latent_cache" not in kernel_source
    assert "selected_latent_mla_decode" in selected_source
    assert "expand_cached_latents" not in selected_source
    assert "gather_paged_cache_pair" not in selected_source
    assert "gather_paged_cache_pair" not in model_source[dispatch:fallback]
    assert "attend_selected_latents" in model_source[dispatch:fallback]
