# SPDX-License-Identifier: Apache-2.0
"""Focused numeric contract for the selected-latent MLA decode spike."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vllm_neuron.model.glm_moe_dsa.sparse_mla import selected_latent_mla_decode


def _reference(
    queries: torch.Tensor,
    cache: torch.Tensor,
    selected: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    row_offset: int,
) -> torch.Tensor:
    valid = (selected >= 0) & (selected < cache.shape[1])
    safe = selected.clamp(0, cache.shape[1] - 1)
    batch = torch.arange(cache.shape[0], device=cache.device)[:, None, None]
    chosen = cache[batch, safe]
    chosen = torch.where(valid[..., None], chosen, torch.zeros_like(chosen))
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
    scores = torch.matmul(
        absorbed_query.unsqueeze(-2), latent.transpose(-1, -2)
    ).squeeze(-2).add(
        torch.matmul(q_rope.unsqueeze(-2), rope.transpose(-1, -2)).squeeze(-2)
    ) * (
        256**-0.5
    )
    scores = scores.masked_fill(~valid, torch.finfo(torch.float32).min)
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.where(valid, probabilities, torch.zeros_like(probabilities))
    probabilities /= probabilities.sum(-1, keepdim=True).clamp_min(1.0e-20)
    probabilities_compute = probabilities.to(queries.dtype).float()
    reduced_latent = torch.matmul(probabilities_compute.unsqueeze(-2), latent).squeeze(
        -2
    )
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


def _deterministic_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(11)
    queries = torch.randn(1, 1, 1, 256, dtype=torch.bfloat16, generator=generator)
    cache = torch.randn(1, 4096, 576, dtype=torch.bfloat16, generator=generator)
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

    # Reverse order, an index outside the first 2048 rows, a duplicate, and
    # padding exercise the selected-row contract.
    selected = torch.arange(2047, -1, -1, dtype=torch.int32).view(1, 1, 2048)
    selected[..., 17] = 3001
    selected[..., 23] = selected[..., 22]
    selected[..., -31:] = -1
    return queries, cache, selected, weight, weight_scale_inv


def _poison_unselected(cache: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    poisoned = cache.clone()
    chosen = set(selected[selected >= 0].tolist())
    unselected = torch.tensor(
        [index for index in range(cache.shape[1]) if index not in chosen],
        dtype=torch.long,
    )
    poisoned[:, unselected] = torch.nan
    return poisoned


@pytest.mark.parametrize("row_offset", (0, 64))
def test_block_fp8_reference_reads_only_selected_latent_rows(
    row_offset: int,
) -> None:
    queries, cache, selected, weight, weight_scale_inv = _deterministic_inputs()
    expected = _reference(
        queries,
        cache,
        selected,
        weight,
        weight_scale_inv,
        row_offset=row_offset,
    )
    actual = _reference(
        queries,
        _poison_unselected(cache, selected),
        selected,
        weight,
        weight_scale_inv,
        row_offset=row_offset,
    )

    assert weight.dtype is torch.float8_e4m3fn
    assert weight_scale_inv.dtype is torch.float32
    assert torch.all(weight_scale_inv > 0)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)

    all_padding = torch.full_like(selected, -1)
    padded = _reference(
        queries,
        torch.full_like(cache, torch.nan),
        all_padding,
        weight,
        weight_scale_inv,
        row_offset=row_offset,
    )
    torch.testing.assert_close(padded, torch.zeros_like(padded))


@pytest.mark.skipif(
    not Path("/dev/neuron0").exists(), reason="requires Neuron hardware"
)
@pytest.mark.parametrize("row_offset", (0, 64))
def test_selected_latent_mla_t4096_q1_k2048_neuron(row_offset: int):
    device = torch.device("xla")
    queries, cache, selected, weight, weight_scale_inv = _deterministic_inputs()
    reference = _reference(
        queries,
        cache,
        selected,
        weight,
        weight_scale_inv,
        row_offset=row_offset,
    )
    poisoned = _poison_unselected(cache, selected)

    actual = selected_latent_mla_decode(
        queries.to(device),
        poisoned.to(device),
        selected.to(device),
        weight.to(device),
        weight_scale_inv.to(device),
        row_offset=row_offset,
    ).cpu()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=2.0e-2, atol=2.0e-2)

    all_padding = torch.full_like(selected, -1)
    padded_actual = selected_latent_mla_decode(
        queries.to(device),
        torch.full_like(cache, torch.nan).to(device),
        all_padding.to(device),
        weight.to(device),
        weight_scale_inv.to(device),
        row_offset=row_offset,
    ).cpu()
    torch.testing.assert_close(padded_actual, torch.zeros_like(padded_actual))
