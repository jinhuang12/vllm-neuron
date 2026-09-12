# SPDX-License-Identifier: Apache-2.0
"""The routed scale publisher emits the checkpoint's pair, unchanged.

Run: ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 pytest -q -rA -s``

Two identity items and their shared control. The grid carries neighbour ratios that
are not powers of two inside one ``256``-block, and one item uses the values a
whole-model load actually stopped on, so the mapping's return is what reddens here.

The consumer-granularity question this change turns on -- do the kernels the routed
bank now calls READ a scale per ``128`` tile rather than per ``256`` block -- is
already settled positively by
``test_moe_blockwise_fp8.py::test_cte_128_gate_up_matches_the_model_reference_per_expert_block``
and ``::test_cte_128_down_matches_the_model_reference_per_expert_block``: each
compares a kernel fed the checkpoint's own ``[128, 128]`` grid, with distinct scales
inside one ``256`` block, against a reference that dequantises at ``128``. A kernel
reading one scale per ``256`` block cannot match that reference. Their fixtures build
from the checkpoint grid and never call this publisher, so repeating the question
here would add a second answer to a settled one.
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

#: The values a whole-model load stopped on: this retained scale, a neighbour tile
#: 2.1818182468414307 times it, and a weight byte at the top of the fp8 range.
RETAINED_SEEN = 3.836495743598789e-4
RATIO_SEEN = 2.1818182468414307
PEAK = 224.0


def emit(tag: str, value: str) -> None:
    """One reading, on the prefix the launcher greps for."""
    print(f"P114|{tag}|{value}")


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


def _retained_first_tile(
    weights: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The mapping this change removed: keep each block's first tile scale, rescale
    the other three tiles' bytes by their own ratio against it."""
    tiles_per_block = 2
    mapped_scales = scales.clone()
    mapped_weights = weights.to(torch.float32).clone()
    for expert in range(scales.shape[0]):
        for h_block in range(scales.shape[1] // tiles_per_block):
            for i_block in range(scales.shape[2] // tiles_per_block):
                kept = scales[
                    expert, h_block * tiles_per_block, i_block * tiles_per_block
                ]
                for h_off in range(tiles_per_block):
                    for i_off in range(tiles_per_block):
                        h_tile = h_block * tiles_per_block + h_off
                        i_tile = i_block * tiles_per_block + i_off
                        ratio = scales[expert, h_tile, i_tile] / kept
                        mapped_scales[expert, h_tile, i_tile] = kept
                        window = (
                            expert,
                            slice(h_tile * TILE_SIZE, (h_tile + 1) * TILE_SIZE),
                            slice(i_tile * TILE_SIZE, (i_tile + 1) * TILE_SIZE),
                        )
                        mapped_weights[window] = mapped_weights[window] * ratio
    return mapped_weights, mapped_scales


def _identity_holds(published: torch.Tensor, original: torch.Tensor) -> int:
    """How many emitted values differ from their input. Zero is identity."""
    return int((published != original).sum())


def test_the_publisher_emits_the_checkpoint_grid_unchanged() -> None:
    """Every emitted scale is its input, at the checkpoint's own granularity."""
    weights, scales = _checkpoint_pair(RETAINED, SIBLING)
    original = scales.clone()

    result = retile_block_scales(weights, scales, DOWN)

    mismatched = _identity_holds(result.published_scales, original)
    emit(
        "EQUAL_VALUES",
        f"mismatched={mismatched}|elements={original.numel()}"
        f"|grid={tuple(result.published_scales.shape)}"
        f"|dtype={result.published_scales.dtype}",
    )
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

    emit("DROPPED", f"count={result.input_scales_dropped}")
    assert result.input_scales_dropped == 0
    emit(
        "COUNTS",
        f"emitted_unsupplied={result.emitted_unsupplied}"
        f"|inexact_rescales={result.inexact_rescales}",
    )
    assert result.emitted_unsupplied == 0
    assert result.inexact_rescales == 0

    operand = to_down_kernel_scale_operand(result.published_scales[0], ROWS, COLS)
    tiles = (ROWS // GATE_UP_SCALE_BLOCK) * (COLS // GATE_UP_SCALE_BLOCK)
    emit(
        "GRANULARITY",
        f"tile={GATE_UP_SCALE_BLOCK}|operand={tuple(operand.shape)}|tiles={tiles}",
    )
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
    emit("EQUAL_BYTES", f"differing={differing}|bytes={original.numel()}")
    assert result.published_weights.dtype == torch.float8_e4m3fn
    assert torch.equal(
        result.published_weights.view(torch.uint8), original.view(torch.uint8)
    ), f"{differing} of {original.numel()} emitted fp8 bytes differ from the input"


def test_the_grid_a_whole_model_load_stopped_on_is_published_unchanged() -> None:
    """The values that refused a real load: published as they arrived."""
    weights, scales = _checkpoint_pair(RETAINED_SEEN, RETAINED_SEEN * RATIO_SEEN)
    original = scales.clone()
    bound = float(torch.finfo(torch.float8_e4m3fn).max)
    reached = PEAK * RATIO_SEEN
    emit("REGRESSION_PREMISE", f"ratio={RATIO_SEEN}|reached={reached}|fp8_bound={bound}")
    assert reached > bound, (
        f"rescaling this grid reaches {reached}, which fp8-e4m3 holds, so these are "
        f"not the values that stopped the load"
    )

    result = retile_block_scales(weights, scales, DOWN)

    mismatched = _identity_holds(result.published_scales, original)
    emit("REGRESSION_EQUAL_VALUES", f"mismatched={mismatched}")
    assert torch.equal(result.published_scales, original)


def test_restoring_the_mapping_reddens_the_identity_item() -> None:
    """The control: the removed mapping, fed the same grid, breaks identity."""
    weights, scales = _checkpoint_pair(RETAINED, SIBLING)
    original = scales.clone()

    mapped_weights, mapped_scales = _retained_first_tile(weights, scales)

    scale_mismatch = _identity_holds(mapped_scales, original)
    byte_mismatch = int(
        (mapped_weights != weights.to(torch.float32)).sum()
    )
    emit(
        "CONTROL",
        f"scale_mismatch={scale_mismatch}|byte_mismatch={byte_mismatch}",
    )
    assert scale_mismatch > 0, (
        "the mapping stub changed no scale, so it is not the mapping and the "
        "identity items above are not discriminating"
    )
    assert byte_mismatch > 0, (
        "the mapping stub changed no weight byte, so it is a scale swap rather "
        "than a retile"
    )
