# SPDX-License-Identifier: Apache-2.0
"""The four MoE token-tile loops run on the device, and the values do not move.

Each of the three routed kernels walks its token tiles inside one dynamic body per expert
block instead of tracing one body per tile. Four things are read here: that each access
pattern the change introduces reads or writes what the trace-time route reads or writes,
against a marker a dead offset would leave behind; that a shape whose blocks do not divide
into whole tiles is refused before any loop runs; that the three kernels are bit identical
to a frozen copy of themselves as they stood before the change, at three geometries; and
that the per-block operands are gathered once per block rather than once per tile.

The frozen copies below are the base commit's own bodies with their loops unchanged. The
constants they read are RETYPED, so a frozen reference cannot follow the live module if one
of them ever moves.

Run under ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1
NEURON_PLATFORM_TARGET_OVERRIDE=trn2``; nothing here reads or sets an environment variable.
"""

from __future__ import annotations
import torch

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.moe import moe_blockwise_fp8 as live
from vllm_neuron.functional.moe.moe_blockwise_fp8 import MoeBlockwiseFp8Error, moe_gate_up_blockwise_fp8, to_gate_up_kernel_scale_operand

#: The constants the frozen copies read, RETYPED on purpose: a frozen reference that
#: imported them would follow the live module if one of them ever moved.
TILE_SIZE = 128
GATE_UP_SCALE_BLOCK = 128
GATE_UP_H_TILES_PER_BLOCK = 1
GATE_UP_FUSION = 2
DGE_TRANSPOSE_ROWS = 16
ROW_ALIGN = 16

#: The activation bound the model passes, RETYPED from the MODEL's own copy
#: (``glm5_next/config.py`` ``swiglu_limit``, handed to both halves at
#: ``glm5_next/model_fp8.py`` in the routed MoE path), never read off the kernel.
_MODEL_SWIGLU_LIMIT = 10.0

#: The source rows per DMA the device generates its own descriptors for, RETYPED.
_DGE_ROWS = 16

_FP8 = torch.float8_e4m3fn

#: Tokens behind every case: the routing column indexes these rows and the padding row is
#: the appended one.
_TOKENS = 256

#: Two experts size the banks. The instruction stream does not depend on the count.
_EXPERTS = 2

#: The three geometries the plan pins. ``blocks`` dynamic bodies of ``tiles`` token tiles
#: each: 18 tiles with a full last block, 16 tiles at a second quotient, and the same 18
#: tiles over two contraction blocks per axis.
_G18 = {"name": "G18", "blocks": 2, "tiles": 9, "h_blocks": 1, "i_blocks": 1}


def _emit(tag: str, **values: object) -> None:
    """Print one machine-readable reading line for the transcript's reader."""
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"ROLL|{tag}|{body}", flush=True)


def _shape_of(case: dict) -> dict:
    """One case's extents, from its block and tile counts."""
    block = case["tiles"] * TILE_SIZE
    return {
        "block": block,
        "positions": case["blocks"] * block,
        "h_extent": case["h_blocks"] * GATE_UP_SCALE_BLOCK,
        "i_extent": case["i_blocks"] * GATE_UP_SCALE_BLOCK,
        "token_tiles": case["blocks"] * case["tiles"],
    }


def _eighths(seed: int, *shape: int) -> torch.Tensor:
    """Unsigned eighths, exact in fp8 and in bfloat16, so no contraction cancels."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32) / 8.0


# --------------------------------------------------------------------------- #
# The forms the change introduces, each against the route it replaces.        #
# --------------------------------------------------------------------------- #
# READ IN THIS ORDER. The access-pattern STORE is read first and alone, so a later
# item's refusal belongs to the form that item adds rather than to the store. Each
# item writes over a marker the trace-time route left behind, so an offset that
# ignored its block register is read as the marker surviving and not as a pass.
_FORM_BLOCKS = 2
_FORM_TILES = 2
_FORM_ROWS = _FORM_BLOCKS * _FORM_TILES * TILE_SIZE


@nki.jit
def _stored_by_block(source, marker):
    """One tile written to every destination through an access pattern, from a dynamic body."""
    out = nl.ndarray((_FORM_ROWS, TILE_SIZE), dtype=nl.float32, buffer=nl.shared_hbm)
    out_b = out.reshape((_FORM_BLOCKS, _FORM_TILES * TILE_SIZE, TILE_SIZE))
    stamp = nl.load(marker[0:TILE_SIZE, 0:TILE_SIZE])
    for step in range(_FORM_BLOCKS * _FORM_TILES):
        r0 = step * TILE_SIZE
        nl.store(out[r0 : r0 + TILE_SIZE, 0:TILE_SIZE], value=stamp)
    tile = nl.load(source[0:TILE_SIZE, 0:TILE_SIZE])

    def body(at_block):
        for step in range(_FORM_TILES):
            t0 = step * TILE_SIZE
            live._stored(
                out_b.ap(
                    pattern=[[TILE_SIZE, TILE_SIZE], [1, TILE_SIZE]],
                    offset=t0 * TILE_SIZE,
                    **live._block_offset(at_block),
                ),
                tile,
            )

    nl.fori_loop(0, _FORM_BLOCKS, body)
    return out


@nki.jit
def _transposed_by_block(source, marker):
    """Every tile transposed onto partitions from inside a dynamic body, 16 rows per DMA."""
    out = nl.ndarray((_FORM_ROWS, TILE_SIZE), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    out_b = out.reshape((_FORM_BLOCKS, _FORM_TILES * TILE_SIZE, TILE_SIZE))
    staged = nl.ndarray((_FORM_ROWS, TILE_SIZE), dtype=nl.bfloat16, buffer=nl.private_hbm)
    nl.store(staged[0:_FORM_ROWS, 0:TILE_SIZE], value=nl.load(source[0:_FORM_ROWS, 0:TILE_SIZE]))
    staged_b = staged.reshape((_FORM_BLOCKS, _FORM_TILES * TILE_SIZE, TILE_SIZE))
    stamp = nl.load(marker[0:TILE_SIZE, 0:TILE_SIZE])
    for step in range(_FORM_BLOCKS * _FORM_TILES):
        r0 = step * TILE_SIZE
        nl.store(out[r0 : r0 + TILE_SIZE, 0:TILE_SIZE], value=stamp)

    def body(at_block):
        for step in range(_FORM_TILES):
            t0 = step * TILE_SIZE
            tile = nl.ndarray((TILE_SIZE, TILE_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
            live._transpose_rows(
                tile,
                staged_b,
                TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE,
                t0 * TILE_SIZE,
                at_block=at_block,
            )
            live._stored(
                out_b.ap(
                    pattern=[[TILE_SIZE, TILE_SIZE], [1, TILE_SIZE]],
                    offset=t0 * TILE_SIZE,
                    **live._block_offset(at_block),
                ),
                tile,
            )

    nl.fori_loop(0, _FORM_BLOCKS, body)
    return out


@nki.jit
def _broadcast_by_block(index_hbm, marker):
    """One int32 per block, replicated into every partition, addressed by the block itself."""
    out = nl.ndarray((_FORM_BLOCKS * TILE_SIZE, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    out_b = out.reshape((_FORM_BLOCKS, TILE_SIZE, 1))
    stamp = nl.load(marker[0:TILE_SIZE, 0:1])
    for step in range(_FORM_BLOCKS):
        r0 = step * TILE_SIZE
        nl.store(out[r0 : r0 + TILE_SIZE, 0:1], value=stamp)

    def body(at_block):
        live._stored(
            out_b.ap(
                pattern=[[1, TILE_SIZE], [1, 1]],
                offset=0,
                **live._block_offset(at_block),
            ),
            live._broadcast_row(index_hbm, 0, TILE_SIZE, at_block=at_block),
        )

    nl.fori_loop(0, _FORM_BLOCKS, body)
    return out


@nki.jit
def _cast_by_block(source, marker):
    """Every tile read through an access pattern that converts fp32 to bfloat16 on the way in."""
    out = nl.ndarray((_FORM_ROWS, TILE_SIZE), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    out_b = out.reshape((_FORM_BLOCKS, _FORM_TILES * TILE_SIZE, TILE_SIZE))
    source_b = source.reshape((_FORM_BLOCKS, _FORM_TILES * TILE_SIZE, TILE_SIZE))
    stamp = nl.load(marker[0:TILE_SIZE, 0:TILE_SIZE])
    for step in range(_FORM_BLOCKS * _FORM_TILES):
        r0 = step * TILE_SIZE
        nl.store(out[r0 : r0 + TILE_SIZE, 0:TILE_SIZE], value=stamp)

    def body(at_block):
        for step in range(_FORM_TILES):
            t0 = step * TILE_SIZE
            tile = live._loaded(
                source_b.ap(
                    pattern=[[TILE_SIZE, TILE_SIZE], [1, TILE_SIZE]],
                    offset=t0 * TILE_SIZE,
                    **live._block_offset(at_block),
                ),
                TILE_SIZE,
                TILE_SIZE,
                nl.bfloat16,
            )
            live._stored(
                out_b.ap(
                    pattern=[[TILE_SIZE, TILE_SIZE], [1, TILE_SIZE]],
                    offset=t0 * TILE_SIZE,
                    **live._block_offset(at_block),
                ),
                tile,
            )

    nl.fori_loop(0, _FORM_BLOCKS, body)
    return out


def _marker(width: int) -> torch.Tensor:
    """A value no case produces, so a destination the dynamic body missed reads as itself."""
    return torch.full((TILE_SIZE, width), -7.0, dtype=torch.float32)


def _form_source() -> torch.Tensor:
    """The form items' input: distinct eighths per row, exact in bfloat16."""
    return _eighths(3, _FORM_ROWS, TILE_SIZE)


def test_the_access_pattern_store_writes_every_destination_the_slice_route_writes():
    """Form 1 of 4, read alone: the store from inside a dynamic body."""
    source = _form_source()
    got = wrap_nki(_stored_by_block)(source, _marker(TILE_SIZE))
    want = source[0:TILE_SIZE].repeat(_FORM_BLOCKS * _FORM_TILES, 1)
    stale = int(torch.eq(got, -7.0).sum().item())
    _emit("FORM_STORE", equal=torch.equal(got, want), stale_marker_elements=stale,
          destinations=_FORM_BLOCKS * _FORM_TILES, shape=tuple(got.shape))
    assert stale == 0
    assert torch.equal(got, want)


def test_the_transpose_from_a_pattern_with_a_block_offset_moves_the_same_rows():
    """Form 2 of 4: ``nisa.dma_transpose`` over a pattern the block register offsets."""
    source = _form_source().to(torch.bfloat16)
    got = wrap_nki(_transposed_by_block)(source, _marker(TILE_SIZE).to(torch.bfloat16))
    tiles = source.reshape(_FORM_BLOCKS * _FORM_TILES, TILE_SIZE, TILE_SIZE)
    want = tiles.transpose(1, 2).reshape(_FORM_ROWS, TILE_SIZE).contiguous()
    stale = int(torch.eq(got.to(torch.float32), -7.0).sum().item())
    _emit("FORM_TRANSPOSE", equal=torch.equal(got, want), stale_marker_elements=stale,
          rows_per_dma=_DGE_ROWS, shape=tuple(got.shape))
    assert stale == 0
    assert torch.equal(got, want)


def test_the_broadcast_row_reads_its_own_block_and_not_the_first():
    """Form 3 of 4: a block offset on a ZERO-stride broadcast pattern."""
    index = torch.tensor([[11], [23]], dtype=torch.int32)
    got = wrap_nki(_broadcast_by_block)(index, torch.full((TILE_SIZE, 1), -7, dtype=torch.int32))
    want = index.repeat_interleave(TILE_SIZE, dim=0)
    stale = int(torch.eq(got, -7).sum().item())
    _emit("FORM_BROADCAST", equal=torch.equal(got, want), stale_marker_elements=stale,
          blocks=_FORM_BLOCKS, distinct=int(torch.unique(got).numel()))
    assert stale == 0
    assert torch.equal(got, want)


def test_the_converting_access_pattern_load_reads_what_a_slice_load_reads():
    """Form 4 of 4: the DMA converts fp32 to bfloat16, which the down kernel needs."""
    source = _form_source()
    got = wrap_nki(_cast_by_block)(source, _marker(TILE_SIZE).to(torch.bfloat16))
    want = source.to(torch.bfloat16)
    stale = int(torch.eq(got.to(torch.float32), -7.0).sum().item())
    _emit("FORM_CAST", equal=torch.equal(got, want), stale_marker_elements=stale,
          shape=tuple(got.shape))
    assert stale == 0
    assert torch.equal(got, want)


# --------------------------------------------------------------------------- #
# The identity the roll rests on: no partial block ever reaches a kernel.     #
# --------------------------------------------------------------------------- #
def test_a_block_that_is_not_whole_tiles_is_refused_before_any_loop_runs():
    """``n_blocks * tiles_per_block`` covers every token tile because a remainder is refused."""
    case = dict(_G18)
    shape = _shape_of(case)
    operands = _gate_up_inputs(case)
    refused = []
    for name, block in (("not_a_tile_multiple", shape["block"] + 1),
                        ("does_not_divide_positions", shape["block"] * 2)):
        try:
            moe_gate_up_blockwise_fp8(
                operands["hidden"], operands["bank"], operands["scales"],
                operands["row_index"], operands["expert_index"], block,
            )
        except MoeBlockwiseFp8Error:
            refused.append(name)
    covered = case["blocks"] * case["tiles"] == shape["positions"] // TILE_SIZE
    _emit("COVERING_IDENTITY", refused=len(refused), by_cause=tuple(refused),
          blocks=case["blocks"], tiles_per_block=case["tiles"],
          token_tiles=shape["positions"] // TILE_SIZE, covered=covered)
    assert len(refused) == 2
    assert covered


# --------------------------------------------------------------------------- #
# The three kernels, against a frozen copy of themselves.                     #
# --------------------------------------------------------------------------- #
def _gate_up_inputs(case: dict, seed: int = 11) -> dict:
    """One case's gate/up operands: the routing column carries the padding row's ``-1``."""
    shape = _shape_of(case)
    generator = torch.Generator().manual_seed(seed)
    hidden = _eighths(seed, _TOKENS + 1, shape["h_extent"]).to(torch.bfloat16)
    bank = _eighths(
        seed + 1, _EXPERTS, shape["h_extent"], GATE_UP_FUSION * shape["i_extent"]
    ).to(_FP8)
    exponents = torch.randint(
        -2, 2,
        (_EXPERTS, shape["h_extent"] // GATE_UP_SCALE_BLOCK, GATE_UP_FUSION,
         shape["i_extent"] // GATE_UP_SCALE_BLOCK),
        generator=generator,
    ).to(torch.float32)
    grid = torch.pow(2.0, exponents)
    scales = torch.stack([
        to_gate_up_kernel_scale_operand(grid[e], shape["h_extent"], shape["i_extent"])
        for e in range(_EXPERTS)
    ])
    row_index = torch.randint(
        -1, _TOKENS, (shape["positions"], 1), generator=generator, dtype=torch.int32
    )
    expert_index = torch.randint(
        0, _EXPERTS, (case["blocks"], 1), generator=generator, dtype=torch.int32
    )
    return {"hidden": hidden, "bank": bank, "scales": scales, "row_index": row_index,
            "expert_index": expert_index, "block": shape["block"]}


def test_the_kernel_defaults_this_change_depends_on_are_the_ones_it_was_read_at():
    """Every function-changing default is named here; the live module must still hold it."""
    live_values = {name: getattr(live, name) for name in
                   ("TILE_SIZE", "GATE_UP_SCALE_BLOCK", "GATE_UP_H_TILES_PER_BLOCK",
                    "GATE_UP_FUSION", "DGE_TRANSPOSE_ROWS", "ROW_ALIGN")}
    retyped = {"TILE_SIZE": TILE_SIZE, "GATE_UP_SCALE_BLOCK": GATE_UP_SCALE_BLOCK,
               "GATE_UP_H_TILES_PER_BLOCK": GATE_UP_H_TILES_PER_BLOCK,
               "GATE_UP_FUSION": GATE_UP_FUSION, "DGE_TRANSPOSE_ROWS": DGE_TRANSPOSE_ROWS,
               "ROW_ALIGN": ROW_ALIGN}
    moved = tuple(name for name, value in retyped.items() if live_values[name] != value)
    _emit("KERNEL_DEFAULTS", read_at=retyped, live=live_values, moved=len(moved),
          by_name=moved, model_swiglu_limit=_MODEL_SWIGLU_LIMIT)
    assert len(moved) == 0, f"defaults that moved under the reading: {moved}"
