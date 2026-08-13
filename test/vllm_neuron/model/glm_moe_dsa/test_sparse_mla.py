# SPDX-License-Identifier: Apache-2.0
"""Focused numeric contract for the selected-latent MLA decode spike."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm_neuron.model.glm_moe_dsa.sparse_mla import (
    _selected_latent_mla_decode_nki,
    selected_latent_mla_decode,
    validate_selected_latent_mla_decode_contract,
)


_BLOCK_SIZE = 16
_LOGICAL_LENGTH = 4096
_PHYSICAL_BLOCKS = 520
_VALUE_SHARD_WIDTH = 128


def _physical_block_count(logical_length: int) -> int:
    return (logical_length // _BLOCK_SIZE) * 2 + 8


def _cache_poison_value(dtype: torch.dtype) -> float:
    return 240.0 if dtype is torch.float8_e4m3fn else torch.nan


def _fp8_poisoned_cache(cache: torch.Tensor) -> torch.Tensor:
    return torch.full(cache.shape, 240.0, dtype=torch.bfloat16).to(cache.dtype)


def _reference(
    queries: torch.Tensor,
    cache: torch.Tensor,
    selected: torch.Tensor,
    block_table: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    row_offset: int,
) -> torch.Tensor:
    valid = (selected >= 0) & (selected < cache.shape[1])
    safe = selected.clamp(0, cache.shape[1] - 1)
    logical_blocks = torch.div(safe, _BLOCK_SIZE, rounding_mode="floor")
    expanded_table = block_table[:, None, :].expand(-1, selected.shape[1], -1)
    physical_blocks = torch.gather(expanded_table, 2, logical_blocks)
    valid &= (physical_blocks >= 0) & (
        physical_blocks < _physical_block_count(cache.shape[1])
    )
    batch = torch.arange(cache.shape[0], device=cache.device)[:, None, None]
    chosen = cache[batch, safe]
    chosen = torch.where(valid[..., None], chosen, torch.zeros_like(chosen))
    return _selected_attention_reference(
        queries,
        chosen,
        valid,
        weight,
        weight_scale_inv,
        row_offset=row_offset,
    )


def _tiled_online_reference(
    queries: torch.Tensor,
    cache: torch.Tensor,
    selected: torch.Tensor,
    block_table: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    row_offset: int,
) -> torch.Tensor:
    valid = (selected >= 0) & (selected < cache.shape[1])
    safe = selected.clamp(0, cache.shape[1] - 1)
    logical_blocks = torch.div(safe, _BLOCK_SIZE, rounding_mode="floor")
    expanded_table = block_table[:, None, :].expand(-1, selected.shape[1], -1)
    physical_blocks = torch.gather(expanded_table, 2, logical_blocks)
    valid &= (physical_blocks >= 0) & (
        physical_blocks < _physical_block_count(cache.shape[1])
    )
    batch = torch.arange(cache.shape[0], device=cache.device)[:, None, None]
    chosen = cache[batch, safe]
    chosen = torch.where(valid[..., None], chosen, torch.zeros_like(chosen))
    return _selected_attention_tiled_online_reference(
        queries,
        chosen,
        valid,
        weight,
        weight_scale_inv,
        row_offset=row_offset,
    )


def _selected_attention_reference(
    queries: torch.Tensor,
    chosen: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    row_offset: int,
) -> torch.Tensor:
    latent = chosen[..., :512].float()
    rope = chosen[..., 512:].float()
    q_nope = queries[..., :192].float()
    q_rope = queries[..., 192:].float()

    # Match the kernel's scale-after-matmul FP32 accumulation.  A TP64 kv_b
    # shard begins at global row rank * 448, hence row offsets alternate 0/64.
    absorbed_query = torch.zeros(*q_nope.shape[:-1], 512, dtype=torch.float32)
    for output_block in range(4):
        key_start = max(0, output_block * 128 - row_offset)
        key_end = min(192, (output_block + 1) * 128 - row_offset)
        if key_end <= key_start:
            continue
        for latent_block in range(4):
            latent_start = latent_block * 128
            partial = torch.matmul(
                q_nope[..., key_start:key_end],
                weight[
                    key_start:key_end,
                    latent_start : latent_start + 128,
                ].float(),
            )
            absorbed_query[..., latent_start : latent_start + 128].add_(
                partial * weight_scale_inv[output_block, latent_block]
            )
    absorbed_query = absorbed_query.to(queries.dtype).float()
    scores = torch.matmul(absorbed_query, latent.transpose(-1, -2)).add(
        torch.matmul(q_rope, rope.transpose(-1, -2))
    ) * (256**-0.5)
    valid_scores = valid.unsqueeze(-2)
    scores = scores.masked_fill(~valid_scores, torch.finfo(torch.float32).min)
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.where(
        valid_scores,
        probabilities,
        torch.zeros_like(probabilities),
    )
    probabilities /= probabilities.sum(-1, keepdim=True).clamp_min(1.0e-20)
    probabilities_compute = probabilities.to(queries.dtype).float()
    reduced_latent = torch.matmul(probabilities_compute, latent)
    reduced_latent = reduced_latent.to(queries.dtype).float()
    output = torch.zeros(*reduced_latent.shape[:-1], 256, dtype=torch.float32)
    for output_block in range(4):
        value_row_start = max(192, output_block * 128 - row_offset)
        value_row_end = min(448, (output_block + 1) * 128 - row_offset)
        if value_row_end <= value_row_start:
            continue
        value_start = value_row_start - 192
        value_end = value_row_end - 192
        for latent_block in range(4):
            latent_start = latent_block * 128
            partial = torch.matmul(
                reduced_latent[..., latent_start : latent_start + 128],
                weight[
                    value_row_start:value_row_end,
                    latent_start : latent_start + 128,
                ]
                .float()
                .t(),
            )
            output[..., value_start:value_end].add_(
                partial * weight_scale_inv[output_block, latent_block]
            )
    return output.to(queries.dtype)


def _selected_attention_tiled_online_reference(
    queries: torch.Tensor,
    chosen: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    row_offset: int,
) -> torch.Tensor:
    """Mirror the kernel's sequential 128-row online-softmax order."""

    latent = chosen[..., :512].float()
    rope = chosen[..., 512:].float()
    q_nope = queries[..., :192].float()
    q_rope = queries[..., 192:].float()

    absorbed_query = torch.zeros(*q_nope.shape[:-1], 512, dtype=torch.float32)
    for latent_block in range(4):
        latent_start = latent_block * _VALUE_SHARD_WIDTH
        absorbed_tile = torch.zeros(
            *q_nope.shape[:-1], _VALUE_SHARD_WIDTH, dtype=torch.float32
        )
        for output_block in range(4):
            key_start = max(0, output_block * _VALUE_SHARD_WIDTH - row_offset)
            key_end = min(192, (output_block + 1) * _VALUE_SHARD_WIDTH - row_offset)
            if key_end <= key_start:
                continue
            partial = torch.matmul(
                q_nope[..., key_start:key_end],
                weight[
                    key_start:key_end,
                    latent_start : latent_start + _VALUE_SHARD_WIDTH,
                ].float(),
            )
            absorbed_tile.add_(partial * weight_scale_inv[output_block, latent_block])
        absorbed_query[..., latent_start : latent_start + _VALUE_SHARD_WIDTH].copy_(
            absorbed_tile
        )
    absorbed_query = absorbed_query.to(queries.dtype).float()

    running_max = torch.full((*queries.shape[:-1], 1), -9984.0, dtype=torch.float32)
    running_sum = torch.zeros_like(running_max)
    latent_accumulator = torch.zeros(*queries.shape[:-1], 512, dtype=torch.float32)
    for selected_start in range(0, chosen.shape[-2], _VALUE_SHARD_WIDTH):
        selected_end = selected_start + _VALUE_SHARD_WIDTH
        latent_tile = latent[..., selected_start:selected_end, :]
        rope_tile = rope[..., selected_start:selected_end, :]
        valid_tile = valid[..., selected_start:selected_end].unsqueeze(-2)
        scores = torch.matmul(absorbed_query, latent_tile.transpose(-1, -2)).add(
            torch.matmul(q_rope, rope_tile.transpose(-1, -2))
        ) * (256**-0.5)
        valid_float = valid_tile.float()
        scores = scores * valid_float + valid_float * 9984.0 - 9984.0

        tile_max = scores.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(running_max, tile_max)
        old_scale = torch.exp(running_max - new_max)
        probabilities = torch.exp(scores - new_max) * valid_float
        tile_sum = probabilities.sum(dim=-1, keepdim=True)
        running_sum = running_sum * old_scale + tile_sum

        probabilities_compute = probabilities.to(queries.dtype).float()
        weighted_latent = torch.matmul(probabilities_compute, latent_tile)
        latent_accumulator = latent_accumulator * old_scale + weighted_latent
        running_max = new_max

    safe_sum = torch.maximum(running_sum, torch.full_like(running_sum, 1.0e-20))
    normalized_latent = (
        (latent_accumulator * torch.reciprocal(safe_sum)).to(queries.dtype).float()
    )

    output = torch.zeros(*normalized_latent.shape[:-1], 256, dtype=torch.float32)
    for output_block in range(4):
        value_row_start = max(192, output_block * _VALUE_SHARD_WIDTH - row_offset)
        value_row_end = min(448, (output_block + 1) * _VALUE_SHARD_WIDTH - row_offset)
        if value_row_end <= value_row_start:
            continue
        value_start = value_row_start - 192
        value_end = value_row_end - 192
        value_block = torch.zeros(
            *normalized_latent.shape[:-1], value_end - value_start, dtype=torch.float32
        )
        for latent_block in range(4):
            latent_start = latent_block * _VALUE_SHARD_WIDTH
            partial = torch.matmul(
                normalized_latent[
                    ..., latent_start : latent_start + _VALUE_SHARD_WIDTH
                ],
                weight[
                    value_row_start:value_row_end,
                    latent_start : latent_start + _VALUE_SHARD_WIDTH,
                ]
                .float()
                .t(),
            )
            value_block.add_(partial * weight_scale_inv[output_block, latent_block])
        output[..., value_start:value_end].copy_(value_block)
    return output.to(queries.dtype)


def _page_logical_cache(
    logical_cache: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    physical_block_count = _physical_block_count(logical_cache.shape[1])
    physical = torch.full(
        (physical_block_count, 1, _BLOCK_SIZE, 576),
        _cache_poison_value(logical_cache.dtype),
        dtype=torch.bfloat16,
    )
    for request_index in range(logical_cache.shape[0]):
        for logical_block in range(block_table.shape[1]):
            physical_block = int(block_table[request_index, logical_block])
            if 0 <= physical_block < physical_block_count:
                logical_start = logical_block * _BLOCK_SIZE
                physical[physical_block, 0] = logical_cache[
                    request_index,
                    logical_start : logical_start + _BLOCK_SIZE,
                ].to(torch.bfloat16)
    return (
        physical[..., :288].to(logical_cache.dtype).contiguous(),
        physical[..., 288:].to(logical_cache.dtype).contiguous(),
    )


def _gather_paged_selected(
    mla_k_cache: torch.Tensor,
    mla_v_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logical_length = block_table.shape[1] * _BLOCK_SIZE
    physical_block_count = mla_k_cache.shape[0]
    valid = (selected >= 0) & (selected < logical_length)
    safe = selected.clamp(0, logical_length - 1)
    logical_blocks = torch.div(safe, _BLOCK_SIZE, rounding_mode="floor")
    expanded_table = block_table[:, None, :].expand(-1, selected.shape[1], -1)
    physical_blocks = torch.gather(expanded_table, 2, logical_blocks)
    valid &= (physical_blocks >= 0) & (physical_blocks < physical_block_count)
    physical_rows = physical_blocks * _BLOCK_SIZE + torch.remainder(safe, _BLOCK_SIZE)
    safe_rows = physical_rows.clamp(0, physical_block_count * _BLOCK_SIZE - 1)
    physical_cache = torch.cat(
        (mla_k_cache.to(torch.bfloat16), mla_v_cache.to(torch.bfloat16)), dim=-1
    ).reshape(-1, 576)
    chosen = physical_cache[safe_rows]
    return torch.where(valid[..., None], chosen, torch.zeros_like(chosen)), valid


def _poison_unselected_physical_rows(
    mla_k_cache: torch.Tensor,
    mla_v_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dtype = mla_k_cache.dtype
    physical = torch.cat(
        (mla_k_cache.to(torch.bfloat16), mla_v_cache.to(torch.bfloat16)), dim=-1
    )
    logical_length = block_table.shape[1] * _BLOCK_SIZE
    physical_block_count = mla_k_cache.shape[0]
    row_is_selected = torch.zeros(
        physical_block_count * _BLOCK_SIZE,
        dtype=torch.bool,
    )
    valid = (selected >= 0) & (selected < logical_length)
    safe = selected.clamp(0, logical_length - 1)
    logical_blocks = torch.div(safe, _BLOCK_SIZE, rounding_mode="floor")
    expanded_table = block_table[:, None, :].expand(-1, selected.shape[1], -1)
    physical_blocks = torch.gather(expanded_table, 2, logical_blocks)
    valid &= (physical_blocks >= 0) & (physical_blocks < physical_block_count)
    physical_rows = physical_blocks * _BLOCK_SIZE + torch.remainder(safe, _BLOCK_SIZE)
    row_is_selected[physical_rows[valid].to(torch.long)] = True
    physical.reshape(-1, 576)[~row_is_selected] = _cache_poison_value(cache_dtype)
    return (
        physical[..., :288].to(cache_dtype).contiguous(),
        physical[..., 288:].to(cache_dtype).contiguous(),
    )


def _deterministic_inputs(
    *,
    logical_length: int = _LOGICAL_LENGTH,
    cache_dtype: torch.dtype = torch.bfloat16,
):
    assert logical_length % _BLOCK_SIZE == 0
    generator = torch.Generator().manual_seed(11)
    queries = torch.randn(2, 1, 1, 256, dtype=torch.bfloat16, generator=generator)
    logical_cache_storage = torch.randn(
        2,
        logical_length,
        576,
        dtype=torch.bfloat16,
        generator=generator,
    ).to(cache_dtype)
    logical_cache = logical_cache_storage.to(torch.bfloat16)
    logical_block_count = logical_length // _BLOCK_SIZE
    physical_block_count = _physical_block_count(logical_length)
    logical_blocks = torch.arange(logical_block_count, dtype=torch.int32)
    block_table = torch.stack(
        (
            torch.remainder(logical_blocks * 73, logical_block_count) * 2,
            torch.remainder(logical_blocks * 151 + 1, logical_block_count) * 2 + 1,
        )
    )
    block_table[0, 11] = -1
    block_table[1, 23] = physical_block_count + 5
    mla_k_cache, mla_v_cache = _page_logical_cache(logical_cache_storage, block_table)

    raw_values = torch.arange(448 * 512, dtype=torch.int32).reshape(448, 512)
    weight = (((raw_values % 31) - 15).float() / 16).to(torch.float8_e4m3fn)
    weight_scale_inv = torch.tensor(
        [
            [0.25, 0.5, 1.0, 2.0],
            [1.5, 0.75, 0.375, 0.1875],
            [0.125, 0.625, 1.25, 2.5],
            [3.0, 1.75, 0.875, 0.4375],
        ],
        dtype=torch.float32,
    )

    selected = torch.stack(
        (
            torch.arange(2047, -1, -1, dtype=torch.int32),
            torch.arange(
                logical_length - 1,
                logical_length - 2049,
                -1,
                dtype=torch.int32,
            ),
        )
    ).unsqueeze(1)
    selected[0, 0, 17] = 11 * _BLOCK_SIZE + 3
    selected[1, 0, 19] = 23 * _BLOCK_SIZE + 2
    selected[0, 0, 31] = logical_length + 9
    selected[:, 0, 43] = selected[:, 0, 42]
    selected[:, :, -31:] = -1
    return (
        queries,
        logical_cache,
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected,
        weight,
        weight_scale_inv,
    )


def test_kernel_maps_logical_indices_to_physical_sbuf_rows() -> None:
    source = inspect.getsource(_selected_latent_mla_decode_nki)

    assert "flat_block_table.ap(" in source
    assert "vector_offset=block_table_rows" in source
    assert source.count("vector_offset=physical_rows") == 2
    assert source.count("buffer=nl.shared_hbm") == 1
    assert "selected_cache_storage = nl.ndarray" in source
    assert "dtype=mla_k_cache.dtype" in source
    assert "selected_cache = nl.ndarray" in source
    assert "dtype=queries.dtype" in source
    assert "nisa.tensor_copy(dst=selected_cache, src=selected_cache_storage)" in source
    assert "latent_cache" not in source


@pytest.mark.parametrize("logical_length", (4096, 8192))
def test_raw_fp8_cache_matches_bf16_selected_tile_oracle(
    logical_length: int,
) -> None:
    (
        queries,
        logical_cache,
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected,
        weight,
        weight_scale_inv,
    ) = _deterministic_inputs(
        logical_length=logical_length,
        cache_dtype=torch.float8_e4m3fn,
    )
    expected = _tiled_online_reference(
        queries,
        logical_cache,
        selected,
        block_table,
        weight,
        weight_scale_inv,
        row_offset=64,
    )
    poisoned_k, poisoned_v = _poison_unselected_physical_rows(
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected,
    )
    chosen, valid = _gather_paged_selected(
        poisoned_k,
        poisoned_v,
        block_table,
        selected,
    )
    actual = _selected_attention_tiled_online_reference(
        queries,
        chosen.to(queries.dtype),
        valid,
        weight,
        weight_scale_inv,
        row_offset=64,
    )

    assert mla_k_cache.dtype is torch.float8_e4m3fn
    assert mla_v_cache.dtype is torch.float8_e4m3fn
    assert torch.isfinite(poisoned_k).all()
    assert torch.isfinite(poisoned_v).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("cache_dtype", (torch.bfloat16, torch.float8_e4m3fn))
def test_contract_accepts_only_supported_paired_cache_dtype(
    cache_dtype: torch.dtype,
) -> None:
    (
        queries,
        _,
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected,
        weight,
        weight_scale_inv,
    ) = _deterministic_inputs(cache_dtype=cache_dtype)
    validate_selected_latent_mla_decode_contract(
        queries,
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected,
        weight,
        weight_scale_inv,
        block_size=_BLOCK_SIZE,
        row_offset=64,
    )

    with pytest.raises(ValueError, match="identical dtypes"):
        validate_selected_latent_mla_decode_contract(
            queries,
            mla_k_cache,
            mla_v_cache.to(torch.float32),
            block_table,
            selected,
            weight,
            weight_scale_inv,
            block_size=_BLOCK_SIZE,
            row_offset=64,
        )


@pytest.mark.parametrize("row_offset", (0, 64))
def test_block_fp8_reference_reads_only_selected_latent_rows(
    row_offset: int,
) -> None:
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
    expected = _reference(
        queries,
        logical_cache,
        selected,
        block_table,
        weight,
        weight_scale_inv,
        row_offset=row_offset,
    )
    poisoned_k, poisoned_v = _poison_unselected_physical_rows(
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected,
    )
    chosen, valid = _gather_paged_selected(
        poisoned_k,
        poisoned_v,
        block_table,
        selected,
    )
    actual = _selected_attention_reference(
        queries,
        chosen,
        valid,
        weight,
        weight_scale_inv,
        row_offset=row_offset,
    )

    assert weight.dtype is torch.float8_e4m3fn
    assert weight_scale_inv.dtype is torch.float32
    assert torch.all(weight_scale_inv > 0)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)

    request_one_blocks = block_table[1]
    request_one_blocks = request_one_blocks[
        (request_one_blocks >= 0) & (request_one_blocks < _PHYSICAL_BLOCKS)
    ].to(torch.long)
    isolated_k = mla_k_cache.clone()
    isolated_v = mla_v_cache.clone()
    isolated_k[request_one_blocks] = torch.nan
    isolated_v[request_one_blocks] = torch.nan
    isolated_chosen, isolated_valid = _gather_paged_selected(
        isolated_k,
        isolated_v,
        block_table,
        selected,
    )
    isolated = _selected_attention_reference(
        queries,
        isolated_chosen,
        isolated_valid,
        weight,
        weight_scale_inv,
        row_offset=row_offset,
    )
    torch.testing.assert_close(isolated[0, 0, 0], expected[0, 0, 0])

    all_padding = torch.full_like(selected, -1)
    padded_chosen, padded_valid = _gather_paged_selected(
        torch.full_like(mla_k_cache, torch.nan),
        torch.full_like(mla_v_cache, torch.nan),
        block_table,
        all_padding,
    )
    padded = _selected_attention_reference(
        queries,
        padded_chosen,
        padded_valid,
        weight,
        weight_scale_inv,
        row_offset=row_offset,
    )
    torch.testing.assert_close(padded, torch.zeros_like(padded))


class _SelectedLatentMLADecodeProbe(nn.Module):
    """Minimal full-graph wrapper for the standalone selected-MLA kernel."""

    def __init__(self, row_offset: int) -> None:
        super().__init__()
        self.row_offset = row_offset

    def forward(
        self,
        queries: torch.Tensor,
        mla_k_cache: torch.Tensor,
        mla_v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_indices: torch.Tensor,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
    ) -> torch.Tensor:
        return selected_latent_mla_decode(
            queries,
            mla_k_cache,
            mla_v_cache,
            block_table,
            selected_indices,
            weight,
            weight_scale_inv,
            block_size=_BLOCK_SIZE,
            row_offset=self.row_offset,
        )


def _error_metrics(
    actual: torch.Tensor, reference: torch.Tensor
) -> dict[str, float | None]:
    actual_fp32 = actual.float().flatten()
    reference_fp32 = reference.float().flatten()
    absolute_error = (actual_fp32 - reference_fp32).abs()
    denominator = torch.linalg.vector_norm(actual_fp32) * torch.linalg.vector_norm(
        reference_fp32
    )
    cosine = None
    if denominator.item() != 0.0:
        cosine = torch.dot(actual_fp32, reference_fp32).div(denominator).item()
    return {
        "max_abs": absolute_error.max().item(),
        "mean_abs": absolute_error.mean().item(),
        "cosine": cosine,
    }


def _bf16_ordered_bits(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.dtype is torch.bfloat16
    bits = tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    magnitude = bits & 0x7FFF
    return torch.where((bits & 0x8000) != 0, 0x8000 - magnitude, 0x8000 + bits)


def _unravel_index(flat_index: int, shape: torch.Size) -> list[int]:
    index = []
    for dimension in reversed(shape):
        index.append(flat_index % dimension)
        flat_index //= dimension
    return list(reversed(index))


def _print_numeric_diagnostics(
    actual_calls: list[torch.Tensor],
    whole_softmax_reference: torch.Tensor,
    tiled_online_reference: torch.Tensor,
    padded_actual: torch.Tensor,
) -> None:
    actual = actual_calls[0]
    regions: dict[str, object] = {
        "whole_softmax": _error_metrics(actual, whole_softmax_reference),
        "tiled_online": _error_metrics(actual, tiled_online_reference),
    }
    batches = []
    for batch_index in range(actual.shape[0]):
        batch = {
            "batch": batch_index,
            "whole_softmax": _error_metrics(
                actual[batch_index], whole_softmax_reference[batch_index]
            ),
            "tiled_online": _error_metrics(
                actual[batch_index], tiled_online_reference[batch_index]
            ),
            "shards": [],
        }
        for shard_index, start in enumerate(
            range(0, actual.shape[-1], _VALUE_SHARD_WIDTH)
        ):
            end = min(start + _VALUE_SHARD_WIDTH, actual.shape[-1])
            batch["shards"].append(
                {
                    "shard": shard_index,
                    "start": start,
                    "end": end,
                    "whole_softmax": _error_metrics(
                        actual[batch_index, ..., start:end],
                        whole_softmax_reference[batch_index, ..., start:end],
                    ),
                    "tiled_online": _error_metrics(
                        actual[batch_index, ..., start:end],
                        tiled_online_reference[batch_index, ..., start:end],
                    ),
                }
            )
        batches.append(batch)
    regions["batches"] = batches

    ulp_distance = (
        _bf16_ordered_bits(actual) - _bf16_ordered_bits(tiled_online_reference)
    ).abs()
    ulp_values, ulp_counts = torch.unique(ulp_distance, return_counts=True)
    ulp_histogram = {
        str(value): count
        for value, count in zip(ulp_values.tolist(), ulp_counts.tolist(), strict=True)
    }

    absolute_error = (actual.float() - tiled_online_reference.float()).abs()
    close = torch.isclose(actual, tiled_online_reference, rtol=2.0e-2, atol=2.0e-2)
    top_mismatches = []
    for flat_index in torch.argsort(absolute_error.flatten(), descending=True)[
        : min(16, absolute_error.numel())
    ].tolist():
        index = _unravel_index(flat_index, actual.shape)
        coordinate = tuple(index)
        top_mismatches.append(
            {
                "index": index,
                "actual": actual[coordinate].float().item(),
                "reference": tiled_online_reference[coordinate].float().item(),
                "abs_error": absolute_error[coordinate].item(),
                "bf16_ulp": ulp_distance[coordinate].item(),
                "within_tolerance": close[coordinate].item(),
            }
        )

    repeatability = []
    for call_index, repeated in enumerate(actual_calls):
        repeatability.append(
            {
                "call": call_index,
                "exact": torch.equal(repeated, actual),
                **_error_metrics(repeated, actual),
                "batch_exact": [
                    torch.equal(repeated[batch_index], actual[batch_index])
                    for batch_index in range(actual.shape[0])
                ],
            }
        )

    payload = {
        "regions": regions,
        "tiled_acceptance_mismatch_count": torch.count_nonzero(~close).item(),
        "bf16_ulp_histogram": ulp_histogram,
        "top_mismatches": top_mismatches,
        "repeatability": repeatability,
        "all_padding": {
            "exact_zero": torch.equal(padded_actual, torch.zeros_like(padded_actual)),
            "finite": torch.isfinite(padded_actual).all().item(),
            "nonzero_count": torch.count_nonzero(padded_actual).item(),
        },
    }
    print("GLM_STAGE3_SPARSE_MLA_DIAGNOSTIC=" + json.dumps(payload, sort_keys=True))


def test_hardware_probe_uses_pinned_fullgraph_compile_contract() -> None:
    source = inspect.getsource(test_selected_latent_mla_t4096_q1_k2048_neuron)

    assert 'backend="vllm_neuron"' in source
    assert "fullgraph=True" in source
    assert "dynamic=False" in source
    assert 'os.environ["GLM_STAGE3_SPARSE_MLA_COMPILE_DIR"]' in source
    assert 'os.environ.get("GLM_STAGE3_SPARSE_MLA_DIAGNOSTIC") == "1"' in source
    assert 'torch.device("neuron:0")' in source


def test_fp8_hardware_nodes_use_bounded_fullgraph_contract() -> None:
    source = inspect.getsource(_run_fp8_selected_latent_mla_neuron)

    assert 'backend="vllm_neuron"' in source
    assert "fullgraph=True" in source
    assert "dynamic=False" in source
    assert 'os.environ["GLM_STAGE3_SPARSE_MLA_FP8_COMPILE_DIR"]' in source
    assert "cache_dtype=torch.float8_e4m3fn" in source
    assert "row_offset=64" in source
    assert "_fp8_poisoned_cache(mla_k_cache)" in source

    t4096_source = inspect.getsource(test_selected_latent_mla_fp8_t4096_q1_k2048_neuron)
    t8192_source = inspect.getsource(test_selected_latent_mla_fp8_t8192_q1_k2048_neuron)
    assert "GLM_STAGE3_SPARSE_MLA_FP8_T4096_HARDWARE" in t4096_source
    assert "GLM_STAGE3_SPARSE_MLA_FP8_T8192_HARDWARE" in t8192_source


def _run_fp8_selected_latent_mla_neuron(logical_length: int) -> None:
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
    ) = _deterministic_inputs(
        logical_length=logical_length,
        cache_dtype=torch.float8_e4m3fn,
    )
    expected = _tiled_online_reference(
        queries,
        logical_cache,
        selected,
        block_table,
        weight,
        weight_scale_inv,
        row_offset=64,
    )
    poisoned_k, poisoned_v = _poison_unselected_physical_rows(
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected,
    )

    compile_root = Path(os.environ["GLM_STAGE3_SPARSE_MLA_FP8_COMPILE_DIR"])
    module = _SelectedLatentMLADecodeProbe(row_offset=64).eval().to(device)
    compiled = torch.compile(
        module,
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={
            "compiler_workdir": str(compile_root / f"fp8-t{logical_length}-row64")
        },
    )

    all_padding = torch.full_like(selected, -1)
    padded_actual = compiled(
        queries.to(device),
        _fp8_poisoned_cache(mla_k_cache).to(device),
        _fp8_poisoned_cache(mla_v_cache).to(device),
        block_table.to(device),
        all_padding.to(device),
        weight.to(device),
        weight_scale_inv.to(device),
    ).cpu()
    torch.testing.assert_close(padded_actual, torch.zeros_like(padded_actual))

    actual = compiled(
        queries.to(device),
        poisoned_k.to(device),
        poisoned_v.to(device),
        block_table.to(device),
        selected.to(device),
        weight.to(device),
        weight_scale_inv.to(device),
    ).cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)


@pytest.mark.skipif(
    os.environ.get("GLM_STAGE3_SPARSE_MLA_FP8_T4096_HARDWARE") != "1"
    or not Path("/dev/neuron0").exists(),
    reason="requires explicit FP8 T4096 selected-MLA Neuron opt-in",
)
def test_selected_latent_mla_fp8_t4096_q1_k2048_neuron() -> None:
    _run_fp8_selected_latent_mla_neuron(4096)


@pytest.mark.skipif(
    os.environ.get("GLM_STAGE3_SPARSE_MLA_FP8_T8192_HARDWARE") != "1"
    or not Path("/dev/neuron0").exists(),
    reason="requires explicit FP8 T8192 selected-MLA Neuron opt-in",
)
def test_selected_latent_mla_fp8_t8192_q1_k2048_neuron() -> None:
    _run_fp8_selected_latent_mla_neuron(8192)


@pytest.mark.skipif(
    os.environ.get("GLM_STAGE3_SPARSE_MLA_HARDWARE") != "1"
    or not Path("/dev/neuron0").exists(),
    reason="requires explicit GLM_STAGE3_SPARSE_MLA_HARDWARE=1 on Neuron",
)
@pytest.mark.parametrize("row_offset", (0, 64))
def test_selected_latent_mla_t4096_q1_k2048_neuron(row_offset: int):
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
    whole_softmax_reference = _reference(
        queries,
        logical_cache,
        selected,
        block_table,
        weight,
        weight_scale_inv,
        row_offset=row_offset,
    )
    tiled_online_reference = _tiled_online_reference(
        queries,
        logical_cache,
        selected,
        block_table,
        weight,
        weight_scale_inv,
        row_offset=row_offset,
    )
    poisoned_k, poisoned_v = _poison_unselected_physical_rows(
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected,
    )

    compile_root = Path(os.environ["GLM_STAGE3_SPARSE_MLA_COMPILE_DIR"])
    module = _SelectedLatentMLADecodeProbe(row_offset).eval().to(device)
    compiled = torch.compile(
        module,
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={"compiler_workdir": str(compile_root / f"row-offset-{row_offset}")},
    )
    device_inputs = (
        queries.to(device),
        poisoned_k.to(device),
        poisoned_v.to(device),
        block_table.to(device),
        selected.to(device),
        weight.to(device),
        weight_scale_inv.to(device),
    )
    actual = compiled(*device_inputs).cpu()

    assert torch.isfinite(actual).all()
    actual_calls = [actual]
    diagnostic_enabled = os.environ.get("GLM_STAGE3_SPARSE_MLA_DIAGNOSTIC") == "1"
    if diagnostic_enabled:
        actual_calls.extend(compiled(*device_inputs).cpu() for _ in range(4))

    all_padding = torch.full_like(selected, -1)
    padded_actual = compiled(
        queries.to(device),
        torch.full_like(mla_k_cache, torch.nan).to(device),
        torch.full_like(mla_v_cache, torch.nan).to(device),
        block_table.to(device),
        all_padding.to(device),
        weight.to(device),
        weight_scale_inv.to(device),
    ).cpu()
    if diagnostic_enabled:
        _print_numeric_diagnostics(
            actual_calls,
            whole_softmax_reference,
            tiled_online_reference,
            padded_actual,
        )
    torch.testing.assert_close(padded_actual, torch.zeros_like(padded_actual))
    torch.testing.assert_close(actual, tiled_online_reference, rtol=2.0e-2, atol=2.0e-2)
