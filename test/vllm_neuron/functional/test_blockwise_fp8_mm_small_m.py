# SPDX-License-Identifier: Apache-2.0
"""Small-token dense FP8 GEMM against CPU references and the prefill kernel."""

from __future__ import annotations

import pytest
import torch
from torch.nn.functional import silu

from vllm_neuron.functional.blockwise_fp8_mm import (
    TILE_SIZE,
    blockwise_fp8_mm,
    blockwise_fp8_mm_torch_oracle,
    dispatch_counters,
    reset_dispatch_counters,
    to_kernel_scale_layout,
)


def _case(tokens: int, rows: int = 256, cols: int = 384):
    generator = torch.Generator().manual_seed(941)
    x = torch.randint(1, 8, (tokens, rows), generator=generator).to(torch.bfloat16)
    weight = torch.randint(1, 8, (rows, cols), generator=generator).to(
        torch.float8_e4m3fn
    )
    # Asymmetric, non-power-of-two scales expose incorrect flattened indexing
    # and inadvertent coarsening to the routed MoE's larger scale blocks.
    scale = torch.tensor(
        [[1.25, 1.75, 2.5], [3.5, 0.625, 0.875]], dtype=torch.float32
    )
    return x, weight, scale


@pytest.mark.parametrize("tokens", [1, 2, 7, 16, 31, 64, 127])
def test_small_m_preserves_each_checkpoint_block(tokens: int) -> None:
    x, weight, scale = _case(tokens)
    prebuilt = to_kernel_scale_layout(scale, weight.shape[0], weight.shape[1])
    reset_dispatch_counters()
    got = blockwise_fp8_mm(x, weight, scale, prebuilt_scale_t=prebuilt)
    expected = blockwise_fp8_mm_torch_oracle(x, weight, scale)
    assert dispatch_counters() == (1, 0)
    assert got.shape == expected.shape == (tokens, weight.shape[1])
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 7, 127])
def test_small_m_matches_the_existing_padded_prefill_route(tokens: int) -> None:
    x, weight, scale = _case(tokens)
    padded = torch.zeros((TILE_SIZE, x.shape[1]), dtype=x.dtype)
    padded[:tokens] = x
    small = blockwise_fp8_mm(x, weight, scale)
    prefill = blockwise_fp8_mm(padded, weight, scale)
    torch.testing.assert_close(small, prefill[:tokens], rtol=0, atol=0)
    assert torch.count_nonzero(prefill[tokens:]) == 0


@pytest.mark.parametrize("tokens,rows", [(1, 2048), (7, 4096), (127, 2048)])
def test_long_k_dma_transpose_preserves_token_and_block_order(tokens, rows):
    # K chunks form DMA rows; their order differs from the token/block order
    # of a conventional MxK tile. Every scale identifies a distinct K chunk.
    generator = torch.Generator().manual_seed(812)
    x = ((torch.arange(tokens * rows).reshape(tokens, rows) % 13) - 6).to(
        torch.bfloat16
    )
    weight = torch.randint(-7, 8, (rows, 128), generator=generator).to(
        torch.float8_e4m3fn
    )
    scale = (torch.arange(1, rows // 128 + 1).reshape(-1, 1) / 8).float()
    got = blockwise_fp8_mm(x, weight, scale)
    expected = blockwise_fp8_mm_torch_oracle(x, weight, scale)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_small_m_signed_fp8_extremes_and_cancellation() -> None:
    generator = torch.Generator().manual_seed(283)
    x = (torch.randn((3, 256), generator=generator) * 3).to(torch.bfloat16)
    raw = torch.randn((256, 256), generator=generator) * 30
    raw[0, 0], raw[1, 128], raw[2, 3] = 240.0, -240.0, 0.001953125
    weight = raw.to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.625, 3.5], [1.75, 0.875]], dtype=torch.float32)
    got = blockwise_fp8_mm(x, weight, scale)
    expected = blockwise_fp8_mm_torch_oracle(x, weight, scale)
    difference = got - expected
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, expected, rtol=3e-2, atol=1e-3)
    assert difference.norm() / expected.norm() < 1e-5
    assert torch.nn.functional.cosine_similarity(
        got.reshape(1, -1), expected.reshape(1, -1)
    ).item() > 0.99999


@pytest.mark.parametrize("tokens", [1, 7])
def test_small_m_mlp_keeps_asymmetric_swiglu_clamps(tokens: int) -> None:
    """Exercise all three projections and a BF16 activation precision boundary."""
    x = torch.ones((tokens, 256), dtype=torch.bfloat16)
    x[:, 128:] = -2
    gate_weight = torch.ones((256, 256), dtype=torch.float8_e4m3fn)
    up_weight = torch.ones_like(gate_weight)
    # Different signs/scale ratios create gate and up values above and below
    # both clamp limits. Gate is capped above only, up is bounded on both sides.
    gate_scale = torch.tensor([[0.5, 0.5], [0.125, 0.5]], dtype=torch.float32)
    up_scale = torch.tensor([[0.5, 0.5], [0.5, 0.125]], dtype=torch.float32)
    down_weight = torch.ones((256, 128), dtype=torch.float8_e4m3fn)
    down_scale = torch.tensor([[1.25], [1.75]], dtype=torch.float32)

    def mlp(mm, activations):
        gate = mm(activations, gate_weight, gate_scale).clamp(max=10)
        up = mm(activations, up_weight, up_scale).clamp(min=-10, max=10)
        activated = (silu(gate) * up).to(torch.bfloat16)
        return mm(activated, down_weight, down_scale)

    got = mlp(blockwise_fp8_mm, x)
    expected = mlp(blockwise_fp8_mm_torch_oracle, x)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    padded = torch.zeros((TILE_SIZE, x.shape[1]), dtype=x.dtype)
    padded[:tokens] = x
    torch.testing.assert_close(got, mlp(blockwise_fp8_mm, padded)[:tokens], rtol=0, atol=0)
