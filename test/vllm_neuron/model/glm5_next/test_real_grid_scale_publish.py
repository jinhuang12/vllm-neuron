# SPDX-License-Identifier: Apache-2.0
"""The routed scale publisher, on the real expert bank's own scale grid.

The publisher returns the grid it was handed, so it is bit-exact by construction. What
these tests measure is that it holds over the whole population of one real text layer's
expert bank, at the geometry the published config gives it, on a grid whose own census
reproduces the counts measured on the checkpoint's scale tensors.

The geometry is derived: the expert count and the two extents come from the in-tree
published config, and the block and tile counts follow from them. The scale values are
constructed, from four ratio classes solved so that the fixture reproduces the measured
counts, including how often an alternative rescaling mapping would refuse a block.

One measured count is not reproduced, and is named rather than left out: the real grid
has 33,928 quad maxima that tie the top-left scale, where this fixture has none. A tie
does not enter the refusal predicate, which compares against the fp8 bound, so the
refusal count is unaffected; measuring tie behaviour needs a fifth class.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from vllm_neuron.functional.moe.blockwise_fp8_retile import (
    BLOCK_QUANT_SIZE,
    DOWN,
    GATE_UP,
    TILE_SIZE,
    is_pow2_exact,
    retile_block_scales,
)
import vllm_neuron.functional.moe.blockwise_fp8_retile as _RETILE

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REAL_CONFIG_PATH = FIXTURES_DIR / "hf-config.json"

#: The text layer the scale values were measured on. One routed layer is the whole
#: population: every routed layer of this checkpoint carries the same three extents.
REAL_LAYER = 12

#: The three projections of one expert bank, in the producer's own view ``(E, H, I)``.
#: Gate and up are registered ``[E, I, H]`` and reach the publisher transposed; down
#: is registered ``[E, H, I]`` and reaches it as it is registered.
PROJECTIONS: tuple[tuple[str, int], ...] = ((GATE_UP, 0), (GATE_UP, 1), (DOWN, 0))

#: What the checkpoint's own scale tensors measure over this layer's expert bank.
MEASURED_BLOCKS = 110592
MEASURED_POW2_RATIOS = 134875
MEASURED_RATIOS = 331776
MEASURED_QUAD_POW2_BLOCKS = 11909
MEASURED_RETAINED_POW2_BLOCKS = 258
MEASURED_LOSSLESS_BLOCKS = 0
MEASURED_TOPLEFT_IS_QUAD_MAX = 49319
MEASURED_RESCALE_REFUSALS = 61273

#: A retained scale a whole-model load ran into, which is not a power of two.
RETAINED_SEEN = 3.836495743598789e-4
#: A power-of-two retained scale, for the blocks that carry one.
RETAINED_POW2 = 2.0**-10

#: The fp8 bound, read off the dtype as the publisher's own module reads it.
FP8_DTYPE = torch.float8_e4m3fn
FP8_BOUND = float(torch.finfo(FP8_DTYPE).max)

#: Every tile of the fixture's weight holds a byte at that bound, which the tests read
#: off the weight rather than assume: the refusal predicate reads a window maximum. The
#: weight fills to the bound because the checkpoint's own tiles do, in all 32 sampled,
#: so the refusal count here is the count the real grid would draw.
TILES_PER_BLOCK_AXIS = BLOCK_QUANT_SIZE // TILE_SIZE

#: The order the three sibling tiles are read in, against the block's top-left one.
#: Written out here rather than imported, because the constant belonged to the
#: rescaling mapping this path does not have.
RATIO_ORDER: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1))

#: The four ratio classes the fixture is built from, as three ratios against the
#: retained tile in :data:`RATIO_ORDER`. Powers of two are exact in binary floating
#: point, so a class that must count as a power of two does, and one that must not
#: cannot become one by rounding.
QUAD_POW2_RATIOS = (0.5, 0.5, 0.5)
TWO_POW2_RATIOS = (2.0, 0.5, 1.5)
PLAIN_REFUSING_RATIOS = (1.5, 0.75, 0.75)
PLAIN_KEEPING_RATIOS = (0.75, 0.75, 0.75)


def real_bank_geometry() -> tuple[int, int, int]:
    """The expert count and the two extents of one routed bank, from the config."""
    text = json.loads(REAL_CONFIG_PATH.read_text())["text_config"]
    layer_types = text["layer_types"]
    assert REAL_LAYER < len(layer_types), (
        f"layer {REAL_LAYER} is outside this config's {len(layer_types)}-layer stack"
    )
    assert REAL_LAYER >= int(text["first_k_dense_replace"]), (
        f"layer {REAL_LAYER} is dense in this config, so it carries no expert bank"
    )
    return (
        int(text["n_routed_experts"]),
        int(text["hidden_size"]),
        int(text["moe_intermediate_size"]),
    )


def class_sizes() -> dict[str, int]:
    """How many blocks each ratio class holds, solved from the measured counts.

    Four classes carry all eight measurements at once: the quad-power-of-two class keeps
    its top-left maximum and so never refuses, the two-power-of-two class and one plain
    class refuse, and the last plain class keeps. Every size is arithmetic on the
    measured numbers, so a corrected measurement moves the fixture with it.
    """
    quad_pow2 = MEASURED_QUAD_POW2_BLOCKS
    two_pow2, remainder = divmod(MEASURED_POW2_RATIOS - 3 * quad_pow2, 2)
    assert remainder == 0, (
        f"{MEASURED_POW2_RATIOS} power-of-two ratios do not split into "
        f"{quad_pow2} three-ratio blocks and whole two-ratio blocks"
    )
    plain_refusing = MEASURED_RESCALE_REFUSALS - two_pow2
    plain_keeping = MEASURED_BLOCKS - quad_pow2 - two_pow2 - plain_refusing
    sizes = {
        "quad_pow2": quad_pow2,
        "two_pow2": two_pow2,
        "plain_refusing": plain_refusing,
        "plain_keeping": plain_keeping,
    }
    assert min(sizes.values()) >= 0, f"a class came out negative: {sizes}"
    assert sum(sizes.values()) == MEASURED_BLOCKS, (
        f"the four classes hold {sum(sizes.values())} blocks where the read measured "
        f"{MEASURED_BLOCKS}"
    )
    assert quad_pow2 + plain_keeping == MEASURED_TOPLEFT_IS_QUAD_MAX, (
        f"the two keeping classes hold {quad_pow2 + plain_keeping} blocks where the "
        f"read measured {MEASURED_TOPLEFT_IS_QUAD_MAX} keeping their top-left maximum"
    )
    return sizes


def _class_ratios() -> list[tuple[float, float, float]]:
    """One ratio triple per block, class by class, in a fixed order."""
    sizes = class_sizes()
    return (
        [QUAD_POW2_RATIOS] * sizes["quad_pow2"]
        + [TWO_POW2_RATIOS] * sizes["two_pow2"]
        + [PLAIN_REFUSING_RATIOS] * sizes["plain_refusing"]
        + [PLAIN_KEEPING_RATIOS] * sizes["plain_keeping"]
    )


def build_real_grid() -> dict[str, torch.Tensor]:
    """The three per-projection scale grids of one expert bank, ``(E, H/128, I/128)``.

    The blocks are laid out projection by projection, expert by expert, then row by row
    within the expert, and the ratio classes are handed out in that order. The
    power-of-two retained scales go to the last class rather than the quad-power-of-two
    one, so no block is lossless on both counts, which is what the checkpoint shows and
    the census below measures.
    """
    experts, rows, cols = real_bank_geometry()
    h_blocks, i_blocks = rows // BLOCK_QUANT_SIZE, cols // BLOCK_QUANT_SIZE
    per_projection = experts * h_blocks * i_blocks
    ratios = torch.tensor(_class_ratios(), dtype=torch.float32)
    assert ratios.shape[0] == len(PROJECTIONS) * per_projection, (
        f"{ratios.shape[0]} blocks were built for "
        f"{len(PROJECTIONS)} x {per_projection} slots"
    )

    retained = torch.full((ratios.shape[0],), RETAINED_SEEN, dtype=torch.float32)
    retained[-MEASURED_RETAINED_POW2_BLOCKS:] = RETAINED_POW2

    quad = torch.empty(
        (ratios.shape[0], TILES_PER_BLOCK_AXIS, TILES_PER_BLOCK_AXIS),
        dtype=torch.float32,
    )
    quad[:, 0, 0] = retained
    for index, (h_off, i_off) in enumerate(RATIO_ORDER):
        quad[:, h_off, i_off] = retained * ratios[:, index]

    grids: dict[str, torch.Tensor] = {}
    for position, (projection, half) in enumerate(PROJECTIONS):
        block = quad[position * per_projection : (position + 1) * per_projection]
        grid = (
            block.reshape(experts, h_blocks, i_blocks, TILES_PER_BLOCK_AXIS,
                          TILES_PER_BLOCK_AXIS)
            .permute(0, 1, 3, 2, 4)
            .reshape(experts, rows // TILE_SIZE, cols // TILE_SIZE)
            .contiguous()
        )
        grids[f"{projection}{half}"] = grid
    return grids


def _quads(grid: torch.Tensor) -> torch.Tensor:
    """One grid as ``(blocks, 2, 2)``: the inverse of the layout above."""
    experts, h_tiles, i_tiles = grid.shape
    return (
        grid.reshape(experts, h_tiles // TILES_PER_BLOCK_AXIS, TILES_PER_BLOCK_AXIS,
                     i_tiles // TILES_PER_BLOCK_AXIS, TILES_PER_BLOCK_AXIS)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, TILES_PER_BLOCK_AXIS, TILES_PER_BLOCK_AXIS)
    )


def _census(
    grids: dict[str, torch.Tensor], window_max: torch.Tensor
) -> dict[str, int]:
    """The eight counts measured on the checkpoint, taken over the whole fixture.

    The power-of-two question is asked of the publisher's own module, one value at a
    time, rather than re-derived here from exponent bits. The values reach it as plain
    floats through one flat list, so the loop costs one pass and not one tensor per
    block.

    The refusal question is asked of the weight: its predicate reads the 128-tile window
    maximum, so ``window_max``, measured off the tensor the publisher is handed, is what
    enters it, over all four tiles of the block rather than the three sibling ratios the
    losslessness count uses.
    """
    quads = torch.cat([_quads(grid) for grid in grids.values()])
    retained_column = quads[:, 0, 0]
    ratios = torch.stack(
        [
            quads[:, h_off, i_off] / retained_column
            for h_off, i_off in RATIO_ORDER
        ],
        dim=1,
    )
    ratio_flags = [is_pow2_exact(value) for value in ratios.reshape(-1).tolist()]
    retained_flags = [is_pow2_exact(value) for value in retained_column.tolist()]
    per_block = len(RATIO_ORDER)
    quad_flags = [
        all(ratio_flags[index * per_block : (index + 1) * per_block])
        for index in range(len(retained_flags))
    ]
    quad_max = quads.reshape(quads.shape[0], -1).max(dim=1).values
    # The rescaling mapping's own refusal predicate, written out here: a tile refuses
    # when its rescaled window is not finite, or when ``|weight * ratio|`` passes the fp8
    # bound. Both limbs read the measured window maximum rather than the bound itself, so
    # a weight that could not reach the bound moves this count.
    wanted = window_max * (quads / retained_column[:, None, None])
    flat = wanted.reshape(wanted.shape[0], -1)
    finite = torch.isfinite(flat).all(dim=1)
    worst = flat.max(dim=1).values
    return {
        "blocks": int(quads.shape[0]),
        "ratios": int(ratios.numel()),
        "pow2_ratios": sum(ratio_flags),
        "quad_pow2_blocks": sum(quad_flags),
        "retained_pow2_blocks": sum(retained_flags),
        "lossless_blocks": sum(
            1 for quad, retained in zip(quad_flags, retained_flags) if quad and retained
        ),
        "topleft_is_quad_max": int((quad_max == retained_column).sum()),
        "rescale_refusals": int((~finite | (worst > FP8_BOUND)).sum()),
    }


def _bank_weight(rows: int, cols: int) -> torch.Tensor:
    """One expert's weight, every byte at the fp8 bound.

    One tensor serves every expert, because every byte of it is the same value: the
    publisher reads no weight value, and the refusal predicate reads only the per-tile
    maximum, which this weight makes the bound everywhere.
    :func:`weight_window_maxima` measures that rather than assuming it.
    """
    return torch.full((1, rows, cols), FP8_BOUND, dtype=torch.float32).to(FP8_DTYPE)


def weight_window_maxima(weight: torch.Tensor) -> torch.Tensor:
    """The 128-tile window maxima measured on one published weight, ``(n, 2, 2)``.

    The refusal predicate reads ``|weights[window] * ratio|.max()``, so this reads the
    same window maximum off the tensor the publisher is handed. One published weight
    carries ``rows x cols / 128**2`` windows.
    """
    experts, rows, cols = weight.shape
    maxima = (
        weight.to(torch.float32)
        .abs()
        .reshape(experts, rows // TILE_SIZE, TILE_SIZE, cols // TILE_SIZE, TILE_SIZE)
        .amax(dim=(2, 4))
    )
    return _quads(maxima)


def aligned_to_population(per_weight: torch.Tensor, blocks: int) -> torch.Tensor:
    """One published weight's maxima, aligned to every block of the bank.

    The same tensor is handed to every expert of every projection, so its maxima repeat
    across the population. The tests assert that those maxima are one value, which is
    what makes the repeat sound.
    """
    assert blocks % per_weight.shape[0] == 0, (
        f"{blocks} blocks do not divide into {per_weight.shape[0]} per published weight"
    )
    return per_weight.repeat(blocks // per_weight.shape[0], 1, 1)


def publish_bank(grids: dict[str, torch.Tensor]) -> dict[str, int]:
    """Publish every expert of every projection, and count what came back.

    One loop for all three readings, so the emitted count, the equality and the refusal
    count are read off the same publications. Only the publisher's own error is caught;
    anything else is this file's defect and must surface as one.
    """
    experts, rows, cols = real_bank_geometry()
    weight = _bank_weight(rows, cols)
    counts = {
        "publications": 0,
        "emitted": 0,
        "equal_publications": 0,
        "differing": 0,
        "refused": 0,
        "dropped": 0,
        "unsupplied": 0,
        "weights_returned_as_given": 0,
    }
    for (projection, half), grid in zip(PROJECTIONS, grids.values()):
        for expert in range(experts):
            original = grid[expert : expert + 1]
            try:
                result = retile_block_scales(weight, original, projection, half)
            except _RETILE.BlockwiseFp8RetileError as error:
                counts["refused"] += 1
                continue
            counts["publications"] += 1
            counts["emitted"] += int(result.published_scales.numel())
            # Two readings of the same equality: the whole-tensor one, and the count of
            # differing values, which says how many moved when the first is false.
            counts["equal_publications"] += int(
                torch.equal(result.published_scales, original)
            )
            counts["differing"] += int((result.published_scales != original).sum())
            counts["dropped"] += int(result.input_scales_dropped)
            counts["unsupplied"] += int(result.emitted_unsupplied)
            counts["weights_returned_as_given"] += int(
                result.published_weights is weight
            )
    return counts


def test_the_fixture_reproduces_the_measured_grid_of_the_real_expert_bank() -> None:
    """The fixture's own census reproduces the eight counts measured on the checkpoint.

    Without this the tests below would measure a population nothing connects to the
    checkpoint. The counts are taken with the publisher module's own power-of-two
    predicate and with the rescaling mapping's refusal predicate, so they are read off
    the fixture rather than asserted by its builder.
    """
    _, rows, cols = real_bank_geometry()
    grids = build_real_grid()
    measured = weight_window_maxima(_bank_weight(rows, cols))
    window_max = aligned_to_population(measured, MEASURED_BLOCKS)
    census = _census(grids, window_max)

    # The refusal count is a property of the ratio grid and of the window maxima, so
    # those maxima are read off the weight rather than assumed from the bound.
    assert int(measured.min()) == int(measured.max()) == int(FP8_BOUND), (
        f"the weight's window maxima run {int(measured.min())}..."
        f"{int(measured.max())} where every tile must reach {int(FP8_BOUND)}; the "
        f"refusal count below is about a weight that cannot reach the fp8 bound"
    )

    assert census["blocks"] == MEASURED_BLOCKS, (
        f"the fixture holds {census['blocks']} blocks where the checkpoint measured "
        f"{MEASURED_BLOCKS}; the geometry no longer matches the checkpoint's"
    )
    assert census["ratios"] == MEASURED_RATIOS
    assert census["pow2_ratios"] == MEASURED_POW2_RATIOS
    assert census["quad_pow2_blocks"] == MEASURED_QUAD_POW2_BLOCKS
    assert census["retained_pow2_blocks"] == MEASURED_RETAINED_POW2_BLOCKS
    assert census["lossless_blocks"] == MEASURED_LOSSLESS_BLOCKS, (
        f"{census['lossless_blocks']} blocks are lossless on both counts where the "
        f"checkpoint has {MEASURED_LOSSLESS_BLOCKS}, so this fixture is easier on the "
        f"rescaling mapping than the checkpoint is"
    )
    assert census["topleft_is_quad_max"] == MEASURED_TOPLEFT_IS_QUAD_MAX
    assert census["rescale_refusals"] == MEASURED_RESCALE_REFUSALS, (
        f"the rescaling mapping would refuse {census['rescale_refusals']} of this "
        f"fixture's blocks where it would refuse {MEASURED_RESCALE_REFUSALS} of "
        f"the checkpoint's"
    )


def test_every_checkpoint_tile_is_published_and_none_is_dropped() -> None:
    """One emitted scale per checkpoint tile, over the whole bank, none dropped."""
    experts, rows, cols = real_bank_geometry()
    tiles = len(PROJECTIONS) * experts * (rows // TILE_SIZE) * (cols // TILE_SIZE)
    counts = publish_bank(build_real_grid())

    assert counts["emitted"] == tiles, (
        f"{counts['emitted']} scales were emitted for {tiles} checkpoint tiles, so "
        f"the publisher does not carry one per tile at this geometry"
    )
    assert tiles == MEASURED_BLOCKS * TILES_PER_BLOCK_AXIS**2, (
        f"{tiles} tiles do not decompose into {MEASURED_BLOCKS} blocks of "
        f"{TILES_PER_BLOCK_AXIS**2}, so this geometry is not the measured one"
    )
    assert counts["dropped"] == 0, f"{counts['dropped']} input scales were dropped"
    assert counts["unsupplied"] == 0, (
        f"{counts['unsupplied']} emitted slots were left unsupplied"
    )


def test_every_published_scale_equals_its_checkpoint_input() -> None:
    """Every emitted scale equals the value handed in: the publisher computes nothing."""
    counts = publish_bank(build_real_grid())

    assert counts["refused"] == 0, (
        f"{counts['refused']} publications refused, so N of N is unreadable"
    )
    assert counts["equal_publications"] == counts["publications"], (
        f"{counts['publications'] - counts['equal_publications']} of "
        f"{counts['publications']} published grids are not equal to the grid handed in"
    )
    assert counts["differing"] == 0, (
        f"{counts['differing']} of {counts['emitted']} emitted scales differ from "
        f"their checkpoint input, so a value was computed where the publisher copies"
    )
    assert counts["emitted"] == MEASURED_BLOCKS * TILES_PER_BLOCK_AXIS**2
    assert counts["weights_returned_as_given"] == counts["publications"], (
        f"{counts['publications'] - counts['weights_returned_as_given']} publications "
        f"returned a weight tensor other than the one handed in, so a byte was copied "
        f"where nothing may be"
    )


def test_the_publisher_refuses_nothing_on_the_grid_the_rescaling_mapping_refuses() -> None:
    """The publisher refuses nothing on the blocks the rescaling mapping would refuse."""
    _, rows, cols = real_bank_geometry()
    grids = build_real_grid()
    census = _census(
        grids,
        aligned_to_population(
            weight_window_maxima(_bank_weight(rows, cols)), MEASURED_BLOCKS
        ),
    )
    refused = publish_bank(grids)["refused"]

    assert census["rescale_refusals"] == MEASURED_RESCALE_REFUSALS, (
        "this fixture is not the population the rescaling mapping refuses, so an empty "
        "refusal count below says nothing"
    )
    assert refused == 0, (
        f"the publisher refused {refused} of this bank's publications, where the "
        f"rescaling mapping refuses {census['rescale_refusals']} and this path "
        f"has no such refusal"
    )


def test_the_rescaling_mapping_changes_values_and_refuses_on_the_measured_count(
) -> None:
    """The alternative rescaling mapping changes values and refuses on this fixture.

    That mapping keeps each block's top-left scale and rescales the other three tiles'
    bytes by their ratio against it, refusing when a rescaled tile passes the fp8 bound.
    It is written out here rather than imported, and it must both move values and refuse:
    otherwise the readings above would hold on a fixture that cannot tell the two
    behaviours apart.

    The refusal is counted by asking the predicate of every block, because the mapping
    itself raised on the first offending tile and stopped.
    """
    _, rows, cols = real_bank_geometry()
    grids = build_real_grid()
    quads = torch.cat([_quads(grid) for grid in grids.values()])
    retained = quads[:, 0, 0].unsqueeze(1).unsqueeze(2)
    ratios = quads / retained
    window_max = aligned_to_population(
        weight_window_maxima(_bank_weight(rows, cols)), quads.shape[0]
    )

    mapped_scales = retained.expand_as(quads)
    changed_scales = int((mapped_scales != quads).sum())
    wanted = window_max * ratios
    flat = wanted.reshape(wanted.shape[0], -1)
    rescaled_tiles = int((wanted != window_max).sum())
    refusals = int(
        (~torch.isfinite(flat).all(dim=1) | (flat.max(dim=1).values > FP8_BOUND)).sum()
    )

    assert changed_scales > 0, (
        "the rescaling mapping changed no scale, so the equality above is not "
        "discriminating"
    )
    assert rescaled_tiles > 0, (
        "the rescaling mapping moved no weight tile, so it is a scale swap rather than "
        "a retile"
    )
    assert int(window_max.max()) == int(FP8_BOUND), (
        f"the weight's tiles reach {int(window_max.max())} and not "
        f"{int(FP8_BOUND)}, so this refusal count is about a different weight"
    )
    assert refusals == MEASURED_RESCALE_REFUSALS, (
        f"the rescaling mapping refuses {refusals} times where the checkpoint measured "
        f"{MEASURED_RESCALE_REFUSALS}, so this arm and the fixture disagree about "
        f"the population"
    )
