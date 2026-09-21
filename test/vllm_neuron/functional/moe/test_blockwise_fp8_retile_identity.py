# SPDX-License-Identifier: Apache-2.0
"""The routed scale publisher emits the checkpoint's weight and scale grid unchanged.

The fixture grid uses neighbour ratios that are not powers of two inside one
``256``-block, so an identity pass cannot be a power-of-two coincidence.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional.moe.blockwise_fp8_retile import (
    DOWN,
    TILE_SIZE,
    retile_block_scales,
)
from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
    GATE_UP_SCALE_BLOCK,
    to_down_kernel_scale_operand,
)

EXPERTS = 2
#: H and I differ, so a transposed grid cannot satisfy a shape assertion by accident.
ROWS = 256
COLS = 512

#: A plain non-power-of-two neighbour ratio, 2.5.
RETAINED = 1.0e-3
SIBLING = 2.5e-3

#: Values from a real checkpoint: this retained scale, a neighbour tile
#: 2.1818182468414307 times it, and a weight byte at the top of the fp8 range.
RETAINED_SEEN = 3.836495743598789e-4
RATIO_SEEN = 2.1818182468414307
PEAK = 224.0


def _checkpoint_pair(
    retained: float, sibling: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """A weight and its ``[128, 128]`` grid, shaped as the checkpoint ships them."""
    weights = torch.ones((EXPERTS, ROWS, COLS), dtype=torch.float32)
    weights[0, TILE_SIZE:, :TILE_SIZE] = PEAK
    scales = torch.full(
        (EXPERTS, ROWS // TILE_SIZE, COLS // TILE_SIZE), retained, dtype=torch.float32
    )
    scales[0, 1, 0] = sibling
    return weights.to(torch.float8_e4m3fn), scales


def _identity_holds(published: torch.Tensor, original: torch.Tensor) -> int:
    """How many emitted values differ from their input. Zero is identity."""
    return int((published != original).sum())


def test_the_publisher_emits_the_checkpoint_grid_unchanged() -> None:
    """Every emitted scale is its input, at the checkpoint's own granularity."""
    weights, scales = _checkpoint_pair(RETAINED, SIBLING)
    original = scales.clone()

    result = retile_block_scales(weights, scales, DOWN)

    mismatched = _identity_holds(result.published_scales, original)
    assert torch.equal(result.published_scales, original), (
        f"{mismatched} of {original.numel()} emitted scales differ from the "
        f"checkpoint's own"
    )
    assert tuple(result.published_scales.shape) == (
        EXPERTS,
        ROWS // TILE_SIZE,
        COLS // TILE_SIZE,
    ), "the emitted grid is not the checkpoint's own tile grid"
    assert result.published_scales.dtype == torch.float32

    assert result.input_scales_dropped == 0
    assert result.emitted_unsupplied == 0
    assert result.inexact_rescales == 0

    operand = to_down_kernel_scale_operand(result.published_scales[0], ROWS, COLS)
    tiles = (ROWS // GATE_UP_SCALE_BLOCK) * (COLS // GATE_UP_SCALE_BLOCK)
    assert tuple(operand.shape) == (TILE_SIZE, tiles), (
        f"the kernel operand built from the published grid is "
        f"{tuple(operand.shape)}, not the {TILE_SIZE}-partition broadcast of "
        f"{tiles} tiles of {GATE_UP_SCALE_BLOCK}"
    )


def test_the_publisher_emits_the_checkpoint_bytes_unchanged() -> None:
    """Every emitted weight byte is its input byte, compared as bytes."""
    weights, scales = _checkpoint_pair(RETAINED, SIBLING)
    original = weights.clone()

    result = retile_block_scales(weights, scales, DOWN)

    differing = int(
        (result.published_weights.view(torch.uint8) != original.view(torch.uint8)).sum()
    )
    assert result.published_weights.dtype == torch.float8_e4m3fn
    assert torch.equal(
        result.published_weights.view(torch.uint8), original.view(torch.uint8)
    ), f"{differing} of {original.numel()} emitted fp8 bytes differ from the input"


def test_the_publisher_emits_a_real_checkpoint_grid_unchanged() -> None:
    """A real checkpoint's scale grid: published unchanged, even though rescaling
    it would overflow fp8."""
    weights, scales = _checkpoint_pair(RETAINED_SEEN, RETAINED_SEEN * RATIO_SEEN)
    original = scales.clone()
    bound = float(torch.finfo(torch.float8_e4m3fn).max)
    reached = PEAK * RATIO_SEEN
    assert reached > bound, (
        f"rescaling this grid reaches {reached}, which fp8-e4m3 holds, so these are "
        f"not the values that stopped the load"
    )

    result = retile_block_scales(weights, scales, DOWN)

    assert torch.equal(result.published_scales, original)
