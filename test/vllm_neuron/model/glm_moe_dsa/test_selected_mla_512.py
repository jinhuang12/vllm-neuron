# SPDX-License-Identifier: Apache-2.0
"""CPU and static contracts for the 512-wide selected-MLA specialization."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

import vllm_neuron.model.glm_moe_dsa.attention as attention_module
import vllm_neuron.model.glm_moe_dsa.sparse_mla as sparse_mla_module
from vllm_neuron.model.glm_moe_dsa.attention import (
    SELECTED_LATENT_MLA_B32_WEIGHT_REUSE_ENV,
    SELECTED_LATENT_MLA_ENV,
    GlmMoeDsaAttention,
)
from vllm_neuron.model.glm_moe_dsa.model import GlmMoeDsaDecoderLayer
from vllm_neuron.model.glm_moe_dsa.sparse_mla import (
    SELECTED_LATENT_MLA_BLOCK_SIZES,
    SELECTED_LATENT_MLA_LONG_WIDTH,
    SELECTED_LATENT_MLA_SHORT_CONTEXT_BUCKETS,
    SELECTED_LATENT_MLA_SHORT_WIDTH,
    _SELECTED_TILE,
    _selected_latent_mla_decode_b32_k512_block32_weight_reuse_nki,
    _selected_latent_mla_decode_nki,
    selected_latent_mla_decode,
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
    populated_selection = min(logical_length, SELECTED_LATENT_MLA_LONG_WIDTH)
    selected[:, 0, :populated_selection] = torch.arange(populated_selection)
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
        use_b32_weight_reuse=False,
    ):
        del actual_weight, actual_scales
        assert actual_k_cache is k_cache
        assert actual_v_cache is v_cache
        assert actual_block_table is block_table
        assert block_size in SELECTED_LATENT_MLA_BLOCK_SIZES
        assert row_offset in (0, 64)
        assert not use_b32_weight_reuse
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


def _launch_selected_kernel(
    monkeypatch,
    *,
    batch_size: int,
    logical_length: int,
    block_size: int,
    enable_weight_reuse: bool,
):
    attention = _pinned_attention()
    queries, selected, k_cache, v_cache, block_table = _dispatch_inputs(
        batch_size=batch_size,
        logical_length=logical_length,
        block_size=block_size,
        cache_dtype=torch.float8_e4m3fn,
    )
    logical_key_count = block_table.shape[1] * block_size
    selected_width = (
        SELECTED_LATENT_MLA_SHORT_WIDTH
        if logical_key_count in SELECTED_LATENT_MLA_SHORT_CONTEXT_BUCKETS
        else SELECTED_LATENT_MLA_LONG_WIDTH
    )
    selected = selected[..., :selected_width].to(torch.int32)
    launched = []

    def fake_wrap(kernel):
        launched.append(kernel)

        def launch(*args, **kwargs):
            del kwargs
            return torch.zeros_like(args[0])

        return None, launch

    monkeypatch.setattr(sparse_mla_module, "wrap_nki", fake_wrap)
    selected_latent_mla_decode(
        queries,
        k_cache,
        v_cache,
        block_table,
        selected,
        attention.kv_b_proj.weight,
        attention.kv_b_proj.weight_scale_inv,
        block_size=block_size,
        row_offset=attention.kv_b_proj.row_offset,
        use_b32_weight_reuse=enable_weight_reuse,
    )
    assert len(launched) == 1
    return launched[0]


@pytest.mark.parametrize(
    ("batch_size", "logical_length", "block_size"),
    (
        (1, 512, 32),
        (8, 512, 32),
        (32, 128, 32),
        (32, 512, 16),
        (32, 4096, 32),
    ),
)
def test_b32_weight_reuse_specialization_fails_closed_to_generic(
    monkeypatch,
    batch_size: int,
    logical_length: int,
    block_size: int,
) -> None:
    assert (
        _launch_selected_kernel(
            monkeypatch,
            batch_size=batch_size,
            logical_length=logical_length,
            block_size=block_size,
            enable_weight_reuse=True,
        )
        is _selected_latent_mla_decode_nki
    )


def test_b32_weight_reuse_dispatch_requires_exact_contract_and_flag(monkeypatch) -> None:
    assert (
        _launch_selected_kernel(
            monkeypatch,
            batch_size=32,
            logical_length=512,
            block_size=32,
            enable_weight_reuse=False,
        )
        is _selected_latent_mla_decode_nki
    )
    assert (
        _launch_selected_kernel(
            monkeypatch,
            batch_size=32,
            logical_length=512,
            block_size=32,
            enable_weight_reuse=True,
        )
        is _selected_latent_mla_decode_b32_k512_block32_weight_reuse_nki
    )


def test_attention_latches_b32_weight_reuse_flag_and_forwards_it(monkeypatch) -> None:
    monkeypatch.delenv(SELECTED_LATENT_MLA_B32_WEIGHT_REUSE_ENV, raising=False)
    assert not _pinned_attention().enable_selected_latent_mla_b32_weight_reuse

    monkeypatch.delenv(SELECTED_LATENT_MLA_ENV, raising=False)
    monkeypatch.setenv(SELECTED_LATENT_MLA_B32_WEIGHT_REUSE_ENV, "1")
    attention = _pinned_attention()
    assert attention.enable_selected_latent_mla_b32_weight_reuse

    queries, selected, k_cache, v_cache, block_table = _dispatch_inputs(
        batch_size=32,
        logical_length=512,
        block_size=32,
        cache_dtype=torch.float8_e4m3fn,
    )
    monkeypatch.setattr(attention_module, "can_run_kernel", lambda tensor: True)
    assert not attention.should_use_selected_latent_mla(
        queries,
        selected,
        mla_k_cache=k_cache,
        mla_v_cache=v_cache,
        block_table=block_table,
        block_size=32,
        is_decode=True,
    )
    forwarded = []

    def selected_kernel(*args, **kwargs):
        forwarded.append(kwargs.pop("use_b32_weight_reuse"))
        assert kwargs == {"block_size": 32, "row_offset": 0}
        return torch.zeros_like(args[0])

    monkeypatch.setattr(attention_module, "selected_latent_mla_decode", selected_kernel)
    attention.attend_selected_latents(
        queries,
        selected,
        k_cache,
        v_cache,
        block_table,
        32,
    )
    assert forwarded == [True]


def _scaled_block_projection(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    row_offset: int,
    output_start: int,
    output_end: int,
) -> torch.Tensor:
    result = torch.zeros(
        inputs.shape[0], weight.shape[1], dtype=torch.float32
    )
    for output_block in range(4):
        row_start = max(
            output_start,
            output_block * _SELECTED_TILE - row_offset,
        )
        row_end = min(
            output_end,
            (output_block + 1) * _SELECTED_TILE - row_offset,
        )
        if row_start >= row_end:
            continue
        for latent_tile in range(4):
            latent_start = latent_tile * _SELECTED_TILE
            partial = (
                inputs[:, row_start - output_start : row_end - output_start].float()
                @ weight[
                    row_start:row_end,
                    latent_start : latent_start + _SELECTED_TILE,
                ].float()
            )
            result[
                :, latent_start : latent_start + _SELECTED_TILE
            ] += partial * scales[output_block, latent_tile]
    return result


def _scaled_value_projection(
    normalized_latent: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    row_offset: int,
) -> torch.Tensor:
    result = torch.zeros(normalized_latent.shape[0], 256, dtype=torch.float32)
    for output_block in range(4):
        row_start = max(192, output_block * _SELECTED_TILE - row_offset)
        row_end = min(448, (output_block + 1) * _SELECTED_TILE - row_offset)
        if row_start >= row_end:
            continue
        output_slice = slice(row_start - 192, row_end - 192)
        for latent_tile in range(4):
            latent_start = latent_tile * _SELECTED_TILE
            partial = (
                normalized_latent[
                    :, latent_start : latent_start + _SELECTED_TILE
                ].float()
                @ weight[
                    row_start:row_end,
                    latent_start : latent_start + _SELECTED_TILE,
                ].float().transpose(0, 1)
            )
            result[:, output_slice] += partial * scales[
                output_block, latent_tile
            ]
    return result


@pytest.mark.parametrize("row_offset", (0, 64))
def test_b32_weight_reuse_projection_math_matches_serial_b1(row_offset: int) -> None:
    generator = torch.Generator().manual_seed(7103 + row_offset)
    weight = torch.randn(448, 512, generator=generator).to(torch.float8_e4m3fn)
    scales = torch.rand(4, 4, generator=generator, dtype=torch.float32) + 0.25
    q_nope = torch.randn(32, 192, generator=generator).to(torch.bfloat16)
    normalized_latent = torch.randn(32, 512, generator=generator).to(
        torch.bfloat16
    )

    batched_query = _scaled_block_projection(
        q_nope,
        weight,
        scales,
        row_offset=row_offset,
        output_start=0,
        output_end=192,
    )
    serial_query = torch.cat(
        [
            _scaled_block_projection(
                q_nope[index : index + 1],
                weight,
                scales,
                row_offset=row_offset,
                output_start=0,
                output_end=192,
            )
            for index in range(32)
        ]
    )
    torch.testing.assert_close(batched_query, serial_query, atol=2.0e-5, rtol=2.0e-5)

    batched_value = _scaled_value_projection(
        normalized_latent,
        weight,
        scales,
        row_offset=row_offset,
    )
    serial_value = torch.cat(
        [
            _scaled_value_projection(
                normalized_latent[index : index + 1],
                weight,
                scales,
                row_offset=row_offset,
            )
            for index in range(32)
        ]
    )
    torch.testing.assert_close(batched_value, serial_value, atol=2.0e-5, rtol=2.0e-5)


def test_b32_weight_reuse_kernel_has_projection_only_static_dataflow() -> None:
    kernel_source = inspect.getsource(
        _selected_latent_mla_decode_b32_k512_block32_weight_reuse_nki
    )
    generic_source = inspect.getsource(_selected_latent_mla_decode_nki)
    kernel_tree = ast.parse(textwrap.dedent(kernel_source))
    batch_loops = [
        node
        for node in ast.walk(kernel_tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "batch_index"
    ]

    assert _SELECTED_TILE == 128
    assert len(batch_loops) == 1
    request_local_source = ast.get_source_segment(
        textwrap.dedent(kernel_source), batch_loops[0]
    )
    assert request_local_source is not None
    assert "weight[" not in request_local_source
    assert "weight_scale_inv[" not in request_local_source
    assert generic_source.count("src=weight[") == 2
    assert "src=weight[" in ast.get_source_segment(
        textwrap.dedent(generic_source),
        next(
            node
            for node in ast.walk(ast.parse(textwrap.dedent(generic_source)))
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "batch_index"
        ),
    )
    assert kernel_source.count("src=weight[") == 2
    assert kernel_source.count("buffer=nl.shared_hbm") == 1
    assert "buffer=nl.hbm" not in kernel_source
    assert "nl.dynamic_range" not in kernel_source
    assert "register_" not in kernel_source
    assert "hwdge" not in kernel_source
    assert "tensor_copy_dynamic" not in kernel_source
    assert "latent_batch = nl.ndarray(" in kernel_source
    assert (
        "(_SELECTED_TILE, 4, 32), dtype=queries.dtype, buffer=nl.sbuf"
        in kernel_source
    )
    assert "(32, _LATENT_WIDTH), dtype=queries.dtype, buffer=nl.sbuf" not in kernel_source
    assert "(4, _SELECTED_TILE, 32)" not in kernel_source
    assert "(32, _LATENT_WIDTH), dtype=nl.float32" not in kernel_source
    assert "absorbed_tile_bf16 = nl.ndarray(" in kernel_source
    assert "absorbed_tile_transpose_psum = nl.ndarray(" in kernel_source
    assert "normalized_tile_transpose_psum = nl.ndarray(" in kernel_source
    assert "stationary=latent_batch[" in kernel_source
    assert kernel_source.index("absorbed_tile_bf16 = nl.ndarray(") > kernel_source.index(
        "scaled_absorbed_partial = nl.ndarray("
    )
    assert kernel_source.index("absorbed_tile_transpose_psum = nl.ndarray(") > (
        kernel_source.index("absorbed_tile_bf16 = nl.ndarray(")
    )
    assert kernel_source.count(".broadcast(dim=0, size=32)") == 2

    latent_accesses = [
        node
        for node in ast.walk(kernel_tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "latent_batch"
    ]
    assert len(latent_accesses) == 4
    for access in latent_accesses:
        assert isinstance(access.slice, ast.Tuple)
        partition_index = access.slice.elts[0]
        assert isinstance(partition_index, ast.Slice)
        assert isinstance(partition_index.lower, ast.Constant)
        assert partition_index.lower.value == 0
        assert isinstance(partition_index.upper, ast.Name)
        assert partition_index.upper.id == "_SELECTED_TILE"

    assert hashlib.sha256(generic_source.encode()).hexdigest() == (
        "675def4599109331b93fccbb269274e7ca58d6246de983443c1a1e8e5e419530"
    )


def test_b32_weight_reuse_partition_safe_latent_roundtrip() -> None:
    generator = torch.Generator().manual_seed(9021)
    absorbed_rows = torch.randn(32, 512, generator=generator).to(torch.bfloat16)
    normalized_rows = torch.randn(32, 512, generator=generator).to(torch.bfloat16)

    persistent = torch.empty(128, 4, 32, dtype=torch.bfloat16)
    for latent_tile_index in range(4):
        latent_start = latent_tile_index * 128
        latent_end = latent_start + 128
        persistent[:, latent_tile_index, :] = absorbed_rows[
            :, latent_start:latent_end
        ].transpose(0, 1)

    for batch_index in range(32):
        for latent_tile_index in range(4):
            latent_start = latent_tile_index * 128
            latent_end = latent_start + 128
            torch.testing.assert_close(
                persistent[:, latent_tile_index, batch_index],
                absorbed_rows[batch_index, latent_start:latent_end],
                rtol=0,
                atol=0,
            )
            persistent[:, latent_tile_index, batch_index] = normalized_rows[
                batch_index, latent_start:latent_end
            ]

    reconstructed = persistent.permute(2, 1, 0).reshape(32, 512)
    torch.testing.assert_close(reconstructed, normalized_rows, rtol=0, atol=0)
    assert not torch.equal(reconstructed[0], reconstructed[1])
    assert not torch.equal(persistent[:, 0, 0], persistent[:, 1, 0])


def test_b32_weight_reuse_keeps_request_local_attention_ast_unchanged() -> None:
    def batch_loop(function) -> ast.For:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        return next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "batch_index"
        )

    class NormalizeSpecializedLatentRead(ast.NodeTransformer):
        def visit_Subscript(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == "latent_batch":
                return ast.copy_location(
                    ast.Name(id="absorbed_tile", ctx=ast.Load()), node
                )
            return node

    class RemoveGenericLatentLayoutPrep(ast.NodeTransformer):
        @staticmethod
        def is_absorbed_tile_assignment(node) -> bool:
            return (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "absorbed_tile"
            )

        @staticmethod
        def is_absorbed_tile_transpose(node) -> bool:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                return False
            call = node.value
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "dma_transpose":
                return False
            return any(
                keyword.arg == "dst"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "absorbed_tile"
                for keyword in call.keywords
            )

        def visit_For(self, node):
            self.generic_visit(node)
            node.body = [
                statement
                for statement in node.body
                if not self.is_absorbed_tile_assignment(statement)
                and not self.is_absorbed_tile_transpose(statement)
            ]
            return node

    generic_attention = batch_loop(_selected_latent_mla_decode_nki).body[5:20]
    generic_attention = [
        RemoveGenericLatentLayoutPrep().visit(node) for node in generic_attention
    ]
    specialized_attention = batch_loop(
        _selected_latent_mla_decode_b32_k512_block32_weight_reuse_nki
    ).body[:15]
    specialized_attention = [
        NormalizeSpecializedLatentRead().visit(node)
        for node in specialized_attention
    ]

    assert ast.dump(ast.Module(body=generic_attention, type_ignores=[])) == ast.dump(
        ast.Module(body=specialized_attention, type_ignores=[])
    )
