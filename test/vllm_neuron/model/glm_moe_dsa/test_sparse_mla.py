# SPDX-License-Identifier: Apache-2.0
"""Focused numeric contract for the selected-latent MLA decode spike."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch

from vllm_neuron.model.glm_moe_dsa.sparse_mla import (
    _selected_latent_mla_decode_nki,
    selected_latent_mla_decode,
)


_BLOCK_SIZE = 16
_LOGICAL_LENGTH = 4096
_LOGICAL_BLOCKS = _LOGICAL_LENGTH // _BLOCK_SIZE
_PHYSICAL_BLOCKS = 520


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
    valid &= (physical_blocks >= 0) & (physical_blocks < _PHYSICAL_BLOCKS)
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


def _page_logical_cache(
    logical_cache: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    physical = torch.full(
        (_PHYSICAL_BLOCKS, 1, _BLOCK_SIZE, 576),
        torch.nan,
        dtype=logical_cache.dtype,
    )
    for request_index in range(logical_cache.shape[0]):
        for logical_block in range(_LOGICAL_BLOCKS):
            physical_block = int(block_table[request_index, logical_block])
            if 0 <= physical_block < _PHYSICAL_BLOCKS:
                logical_start = logical_block * _BLOCK_SIZE
                physical[physical_block, 0] = logical_cache[
                    request_index,
                    logical_start : logical_start + _BLOCK_SIZE,
                ]
    return physical[..., :288].contiguous(), physical[..., 288:].contiguous()


def _gather_paged_selected(
    mla_k_cache: torch.Tensor,
    mla_v_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (selected >= 0) & (selected < _LOGICAL_LENGTH)
    safe = selected.clamp(0, _LOGICAL_LENGTH - 1)
    logical_blocks = torch.div(safe, _BLOCK_SIZE, rounding_mode="floor")
    expanded_table = block_table[:, None, :].expand(-1, selected.shape[1], -1)
    physical_blocks = torch.gather(expanded_table, 2, logical_blocks)
    valid &= (physical_blocks >= 0) & (physical_blocks < _PHYSICAL_BLOCKS)
    physical_rows = physical_blocks * _BLOCK_SIZE + torch.remainder(safe, _BLOCK_SIZE)
    safe_rows = physical_rows.clamp(0, _PHYSICAL_BLOCKS * _BLOCK_SIZE - 1)
    physical_cache = torch.cat((mla_k_cache, mla_v_cache), dim=-1).reshape(-1, 576)
    chosen = physical_cache[safe_rows]
    return torch.where(valid[..., None], chosen, torch.zeros_like(chosen)), valid


def _poison_unselected_physical_rows(
    mla_k_cache: torch.Tensor,
    mla_v_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    physical = torch.cat((mla_k_cache, mla_v_cache), dim=-1).clone()
    row_is_selected = torch.zeros(
        _PHYSICAL_BLOCKS * _BLOCK_SIZE,
        dtype=torch.bool,
    )
    valid = (selected >= 0) & (selected < _LOGICAL_LENGTH)
    safe = selected.clamp(0, _LOGICAL_LENGTH - 1)
    logical_blocks = torch.div(safe, _BLOCK_SIZE, rounding_mode="floor")
    expanded_table = block_table[:, None, :].expand(-1, selected.shape[1], -1)
    physical_blocks = torch.gather(expanded_table, 2, logical_blocks)
    valid &= (physical_blocks >= 0) & (physical_blocks < _PHYSICAL_BLOCKS)
    physical_rows = physical_blocks * _BLOCK_SIZE + torch.remainder(safe, _BLOCK_SIZE)
    row_is_selected[physical_rows[valid].to(torch.long)] = True
    physical.reshape(-1, 576)[~row_is_selected] = torch.nan
    return physical[..., :288].contiguous(), physical[..., 288:].contiguous()


def _deterministic_inputs():
    generator = torch.Generator().manual_seed(11)
    queries = torch.randn(2, 1, 1, 256, dtype=torch.bfloat16, generator=generator)
    logical_cache = torch.randn(
        2,
        _LOGICAL_LENGTH,
        576,
        dtype=torch.bfloat16,
        generator=generator,
    )
    logical_blocks = torch.arange(_LOGICAL_BLOCKS, dtype=torch.int32)
    block_table = torch.stack(
        (
            torch.remainder(logical_blocks * 73, _LOGICAL_BLOCKS) * 2,
            torch.remainder(logical_blocks * 151 + 1, _LOGICAL_BLOCKS) * 2 + 1,
        )
    )
    block_table[0, 11] = -1
    block_table[1, 23] = _PHYSICAL_BLOCKS + 5
    mla_k_cache, mla_v_cache = _page_logical_cache(logical_cache, block_table)

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
            torch.arange(4095, 2047, -1, dtype=torch.int32),
        )
    ).unsqueeze(1)
    selected[0, 0, 17] = 11 * _BLOCK_SIZE + 3
    selected[1, 0, 19] = 23 * _BLOCK_SIZE + 2
    selected[0, 0, 31] = _LOGICAL_LENGTH + 9
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
    assert "selected_cache = nl.ndarray" in source
    assert "latent_cache" not in source


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


@pytest.mark.skipif(
    os.environ.get("GLM_STAGE3_SPARSE_MLA_HARDWARE") != "1"
    or not Path("/dev/neuron0").exists(),
    reason="requires explicit GLM_STAGE3_SPARSE_MLA_HARDWARE=1 on Neuron",
)
@pytest.mark.parametrize("row_offset", (0, 64))
def test_selected_latent_mla_t4096_q1_k2048_neuron(row_offset: int):
    device = torch.device("xla")
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
    reference = _reference(
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

    actual = selected_latent_mla_decode(
        queries.to(device),
        poisoned_k.to(device),
        poisoned_v.to(device),
        block_table.to(device),
        selected.to(device),
        weight.to(device),
        weight_scale_inv.to(device),
        block_size=_BLOCK_SIZE,
        row_offset=row_offset,
    ).cpu()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=2.0e-2, atol=2.0e-2)

    all_padding = torch.full_like(selected, -1)
    padded_actual = selected_latent_mla_decode(
        queries.to(device),
        torch.full_like(mla_k_cache, torch.nan).to(device),
        torch.full_like(mla_v_cache, torch.nan).to(device),
        block_table.to(device),
        all_padding.to(device),
        weight.to(device),
        weight_scale_inv.to(device),
        block_size=_BLOCK_SIZE,
        row_offset=row_offset,
    ).cpu()
    torch.testing.assert_close(padded_actual, torch.zeros_like(padded_actual))
