# SPDX-License-Identifier: Apache-2.0
"""The routed scale publisher, on the real expert bank's own grid.

Run: ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 pytest -q -rA -s``

The publisher is bit-exact by construction: it returns the grid it was handed. This
file measures that on the population the removed mapping could not carry -- one real
text layer's expert bank, at the geometry the published config gives it, with a scale
grid whose own census reproduces the numbers a read of the real checkpoint recorded.

THE GEOMETRY IS DERIVED, NEVER TYPED. The expert count and the two extents come from
the in-tree published config, and the block and tile counts follow from them, so a
config change moves this file's population instead of leaving it asserting an extent
the model no longer has.

THE VALUES ARE CONSTRUCTED, AND THAT IS DISCLOSED. The real checkpoint's own scale
values live on the leased host; a read of them recorded six counts over this layer's
110,592 blocks, and the grid built here reproduces all six, measured by the module's
own bit predicate rather than assumed. What this file proves is therefore: on a grid
that refuses under the removed guard exactly as often as the real one does, the
publisher emits every tile unchanged and refuses nothing.
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

#: The text layer the real-weights scale read measured. One routed layer is the whole
#: population: every routed layer of this checkpoint carries the same three extents.
REAL_LAYER = 12

#: The three projections of one expert bank, in the producer's own view ``(E, H, I)``.
#: Gate and up are registered ``[E, I, H]`` and reach the publisher transposed; down
#: is registered ``[E, H, I]`` and reaches it as it is registered.
PROJECTIONS: tuple[tuple[str, int], ...] = ((GATE_UP, 0), (GATE_UP, 1), (DOWN, 0))

#: WHAT THE REAL CHECKPOINT MEASURED, over this layer's expert bank. Read off the
#: real-weights scale read's own rows, twice, identically:
#: ``increments/real-scales-r2-trn2-1-20260909T171155Z.out``.
MEASURED_BLOCKS = 110592
MEASURED_POW2_RATIOS = 134875
MEASURED_RATIOS = 331776
MEASURED_QUAD_POW2_BLOCKS = 11909
MEASURED_RETAINED_POW2_BLOCKS = 258
MEASURED_LOSSLESS_BLOCKS = 0
MEASURED_TOPLEFT_IS_QUAD_MAX = 49319
MEASURED_REMOVED_GUARD_REFUSALS = 61273

#: The retained scale a whole-model load stopped on -- not a power of two.
RETAINED_SEEN = 3.836495743598789e-4
#: A power-of-two retained scale, for the 258 blocks that carry one.
RETAINED_POW2 = 2.0**-10

#: The fp8 bound, read off the dtype as the publisher's own module reads it.
FP8_DTYPE = torch.float8_e4m3fn
FP8_BOUND = float(torch.finfo(FP8_DTYPE).max)

#: Every tile of the fixture's weight holds a byte at that bound. The real-weights
#: read tested this premise on real bytes rather than assuming it and found it held
#: on 32 of 32 sampled tiles (``C1_TILES_HOLDING_A_BYTE_AT_THE_FP8_BOUND|32 of 32``),
#: which is what makes the removed guard's refusal count a property of the grid.
TILES_PER_BLOCK_AXIS = BLOCK_QUANT_SIZE // TILE_SIZE

#: The order the three sibling tiles are read in, against the block's top-left one.
#: Transliterated rather than imported: the constant lived beside the mapping and was
#: removed with it (``blockwise_fp8_retile.py:116`` in the tree this subject was
#: removed from).
RATIO_ORDER: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1))

#: The four ratio classes the fixture is built from, as
#: ``(ratio against the retained tile) x 3`` in :data:`RATIO_ORDER`. Powers of two are
#: exact in binary floating point, so a class that must count as power-of-two does, and
#: one that must not cannot become one by rounding.
QUAD_POW2_RATIOS = (0.5, 0.5, 0.5)
TWO_POW2_RATIOS = (2.0, 0.5, 1.5)
PLAIN_REFUSING_RATIOS = (1.5, 0.75, 0.75)
PLAIN_KEEPING_RATIOS = (0.75, 0.75, 0.75)


def emit(tag: str, value: str) -> None:
    """One reading, on the prefix the launcher greps for."""
    print(f"REALGRID|{tag}|{value}")


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
    """How many blocks each ratio class holds, SOLVED from the measured counts.

    Four classes carry all six measurements at once. The quad-power-of-two class
    keeps its top-left maximum, so it never refuses; the two-power-of-two class and
    one plain class refuse; the last plain class keeps. Every size below is arithmetic
    on the measured numbers, so a measurement corrected upstream moves the fixture
    rather than leaving it asserting an old population.
    """
    quad_pow2 = MEASURED_QUAD_POW2_BLOCKS
    two_pow2, remainder = divmod(MEASURED_POW2_RATIOS - 3 * quad_pow2, 2)
    assert remainder == 0, (
        f"{MEASURED_POW2_RATIOS} power-of-two ratios do not split into "
        f"{quad_pow2} three-ratio blocks and whole two-ratio blocks"
    )
    plain_refusing = MEASURED_REMOVED_GUARD_REFUSALS - two_pow2
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

    The blocks are laid out projection by projection, expert by expert, then row by
    row within the expert, and the ratio classes are handed out in that order. The
    power-of-two retained scales go to the LAST class, which is not the
    quad-power-of-two one, so no block satisfies both losslessness conjuncts -- which
    is what the real read found, and the census item below measures rather than
    trusts.
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


def _census(grids: dict[str, torch.Tensor]) -> dict[str, int]:
    """The six readings the real-weights read reported, over the whole fixture.

    THE POWER-OF-TWO QUESTION IS ASKED OF THE PUBLISHER'S OWN MODULE, one value at a
    time, rather than re-derived here from exponent bits: the predicate that decides
    losslessness must be the one the design depends on. The values reach it as plain
    floats through one flat list, so the loop costs one pass and not one tensor per
    block.
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
    # The removed guard's own predicate, transliterated from the tree this subject was
    # removed from: a tile refuses when ``|weight * ratio|`` passes the fp8 bound, and
    # every tile of this fixture holds a byte at that bound.
    worst = (ratios * FP8_BOUND).max(dim=1).values
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
        "removed_guard_refusals": int((worst > FP8_BOUND).sum()),
    }


def _bank_weight(rows: int, cols: int) -> torch.Tensor:
    """One expert's weight, every byte at the fp8 bound.

    ONE TENSOR FOR EVERY EXPERT, because every byte of it is the same value: the
    publisher reads no weight value, and the removed guard's predicate reads only the
    per-tile maximum, which this weight makes the bound everywhere.
    """
    return torch.full((1, rows, cols), FP8_BOUND, dtype=torch.float32).to(FP8_DTYPE)


def publish_bank(grids: dict[str, torch.Tensor]) -> dict[str, int]:
    """Publish every expert of every projection, and count what came back.

    ONE LOOP FOR THE THREE READINGS BELOW, so the emitted count, the equality and the
    refusal count are all read off the same publications. Only the publisher's own
    error is caught: anything else is this file's defect and must surface as one.
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
                emit("REFUSED", f"{projection}|{half}|expert={expert}|{error}")
                continue
            counts["publications"] += 1
            counts["emitted"] += int(result.published_scales.numel())
            # TWO READINGS OF THE SAME EQUALITY: the whole-tensor one the acceptance is
            # written in, and the count of differing values, which says HOW MANY moved
            # when the first one is false.
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
    """The fixture's own census equals the real-weights read's six numbers.

    WITHOUT THIS ITEM the four below would measure a population nobody had connected
    to the checkpoint. The counts are measured with the publisher module's own
    power-of-two bit predicate and with the removed guard's own predicate, so they are
    read off the fixture rather than declared by its builder.
    """
    experts, rows, cols = real_bank_geometry()
    grids = build_real_grid()
    census = _census(grids)
    emit(
        "GEOMETRY",
        f"layer={REAL_LAYER}|experts={experts}|rows={rows}|cols={cols}"
        f"|projections={len(PROJECTIONS)}",
    )
    for key in sorted(census):
        emit("CENSUS", f"{key}={census[key]}")

    assert census["blocks"] == MEASURED_BLOCKS, (
        f"the fixture holds {census['blocks']} blocks where the real read measured "
        f"{MEASURED_BLOCKS}; the geometry no longer matches the checkpoint's"
    )
    assert census["ratios"] == MEASURED_RATIOS
    assert census["pow2_ratios"] == MEASURED_POW2_RATIOS
    assert census["quad_pow2_blocks"] == MEASURED_QUAD_POW2_BLOCKS
    assert census["retained_pow2_blocks"] == MEASURED_RETAINED_POW2_BLOCKS
    assert census["lossless_blocks"] == MEASURED_LOSSLESS_BLOCKS, (
        f"{census['lossless_blocks']} blocks satisfy both losslessness conjuncts "
        f"where the real checkpoint has {MEASURED_LOSSLESS_BLOCKS}, so this fixture "
        f"is easier on the removed mapping than the checkpoint is"
    )
    assert census["topleft_is_quad_max"] == MEASURED_TOPLEFT_IS_QUAD_MAX
    assert census["removed_guard_refusals"] == MEASURED_REMOVED_GUARD_REFUSALS, (
        f"the removed guard would refuse {census['removed_guard_refusals']} of this "
        f"fixture's blocks where it would refuse {MEASURED_REMOVED_GUARD_REFUSALS} of "
        f"the checkpoint's"
    )


def test_every_checkpoint_tile_is_published_and_none_is_dropped() -> None:
    """One emitted scale per checkpoint tile, over the whole bank, none dropped."""
    experts, rows, cols = real_bank_geometry()
    tiles = len(PROJECTIONS) * experts * (rows // TILE_SIZE) * (cols // TILE_SIZE)
    counts = publish_bank(build_real_grid())
    emit(
        "EMITTED",
        f"emitted={counts['emitted']}|tiles={tiles}|blocks={MEASURED_BLOCKS}"
        f"|publications={counts['publications']}",
    )
    emit("COUNTS", f"dropped={counts['dropped']}|unsupplied={counts['unsupplied']}")

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
    """Zero arithmetic: every emitted value is its input value, N of N."""
    counts = publish_bank(build_real_grid())
    emit(
        "EQUAL_VALUES",
        f"equal={counts['emitted'] - counts['differing']}|of={counts['emitted']}"
        f"|differing={counts['differing']}"
        f"|equal_publications={counts['equal_publications']}"
        f"|weights_returned_as_given={counts['weights_returned_as_given']}",
    )

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


def test_the_publisher_refuses_nothing_on_the_grid_the_removed_guard_refused() -> None:
    """No refusal, on the same blocks the removed guard refused 61,273 times."""
    grids = build_real_grid()
    census = _census(grids)
    refused = publish_bank(grids)["refused"]
    emit(
        "REFUSALS",
        f"publisher={refused}|removed_guard_would={census['removed_guard_refusals']}",
    )

    assert census["removed_guard_refusals"] == MEASURED_REMOVED_GUARD_REFUSALS, (
        "this fixture is not the population the removed guard refused, so a zero "
        "refusal count below says nothing"
    )
    assert refused == 0, (
        f"the publisher refused {refused} of this bank's publications; the removed "
        f"guard refused {census['removed_guard_refusals']} and this path has no guard"
    )
    emit("IMPORT_ORIGIN", str(_RETILE.__file__))


def test_restoring_the_mapping_breaks_the_equality_and_refuses_on_the_measured_count(
) -> None:
    """The control: the removed mapping and its guard, on the same fixture.

    The mapping kept each block's top-left scale and rescaled the other three tiles'
    bytes by their own ratio against it, refusing when a rescaled byte passed the fp8
    bound (the tree this file's subject was removed from,
    ``blockwise_fp8_retile.py:500-515`` at that commit). It is transliterated here
    because it no longer exists to import, and it must both change values and refuse:
    a control that did neither would leave the four items above passing on a path that
    could not tell the two behaviours apart.
    """
    grids = build_real_grid()
    quads = torch.cat([_quads(grid) for grid in grids.values()])
    retained = quads[:, 0, 0].unsqueeze(1).unsqueeze(2)
    ratios = quads / retained

    mapped_scales = retained.expand_as(quads)
    changed_scales = int((mapped_scales != quads).sum())
    rescaled_bytes = int(((ratios * FP8_BOUND) != FP8_BOUND).sum())
    refusals = int(((ratios * FP8_BOUND).amax(dim=(1, 2)) > FP8_BOUND).sum())
    emit(
        "CONTROL",
        f"changed_scales={changed_scales}|rescaled_bytes={rescaled_bytes}"
        f"|refusals={refusals}",
    )

    assert changed_scales > 0, (
        "the restored mapping changed no scale, so it is not the mapping and the "
        "equality item above is not discriminating"
    )
    assert rescaled_bytes > 0, (
        "the restored mapping rescaled no weight byte, so it is a scale swap rather "
        "than a retile"
    )
    assert refusals == MEASURED_REMOVED_GUARD_REFUSALS, (
        f"the restored guard refuses {refusals} times where the real read measured "
        f"{MEASURED_REMOVED_GUARD_REFUSALS}; the control and the fixture disagree "
        f"about the population"
    )
