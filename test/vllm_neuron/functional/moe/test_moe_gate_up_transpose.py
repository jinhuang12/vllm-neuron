# SPDX-License-Identifier: Apache-2.0
"""The routed gate/up kernel's DMA transposes, its SBUF tiles, and its results.

A DMA transpose moves 16 two-byte source rows per descriptor -- more rows, or wider
ones, and the host has to build the descriptors instead of the device -- and its
destination must start on a 32-byte line. The allocator packs SBUF tiles back to back
per partition, so a tile's row must itself be a whole number of 32-byte lines.
Both properties are read off the module's own arithmetic. The kernel's output is then
compared against an independent reference implementation at prefill and decode.
"""

from __future__ import annotations

import ast
import pathlib

import torch

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.moe import moe_blockwise_fp8 as mod
from test.vllm_neuron.functional.dma_transpose_census import LINE, census, tile_rows
from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
    moe_gate_up_blockwise_fp8,
    to_gate_up_kernel_scale_operand,
)

#: The kernel geometry the reference implementation below is written against. Spelled
#: out rather than imported, so the reference cannot follow the module it checks.
TILE_SIZE = 128
GATE_UP_SCALE_BLOCK = 128
GATE_UP_H_TILES_PER_BLOCK = 1
GATE_UP_FUSION = 2

_FP8 = torch.float8_e4m3fn

#: The per-rank geometry: hidden width, one expert's intermediate shard, the block the
#: mapping hands the kernel. Two experts size the bank only; the instruction stream does
#: not depend on the expert count.
_HIDDEN = 4096
_INTERMEDIATE = 512
_EXPERTS = 2
_BLOCK = 256

#: ``(tokens, blocks)`` per case: a prefill bucket of 162 token tiles, and decode's 16.
_CASES = ((2048, 81), (1, 8))


# --------------------------------------------------------------------------- #
# An independent reference implementation of the routed gate/up kernel.       #
# --------------------------------------------------------------------------- #
def _gate_up_sbuf(rows: int = TILE_SIZE, cols: int = GATE_UP_SCALE_BLOCK):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)


def _gate_up_psum(rows: int = TILE_SIZE, cols: int = GATE_UP_SCALE_BLOCK):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)


def _row_iota(iota_hbm, rows: int):
    """A column holding ``0 .. rows - 1``, one value per partition."""
    tile = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=iota_hbm.ap(pattern=[[1, rows], [1, 1]], offset=0))
    return tile


def _dense_column(hbm, rows: int, offset: int):
    """A column of ``rows`` consecutive int32 values read from ``offset``."""
    tile = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=hbm.ap(pattern=[[1, rows], [1, 1]], offset=offset))
    return tile


def _broadcast_row(hbm, row: int, rows: int):
    """One int32 row replicated into every partition of a column."""
    tile = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=hbm.ap(pattern=[[0, rows], [1, 1]], offset=row))
    return tile


def _padding_resolved(rows: int, positions, pad_row: int):
    """Routed positions with the negative padding marker moved onto ``pad_row``."""
    negative = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=negative, data=positions, op0=nl.less, operand0=0)
    bumped = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=bumped, data=negative, op0=nl.multiply, operand0=pad_row + 1
    )
    index = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=index, data1=positions, data2=bumped, op=nl.add)
    return index


def _bank_rows(rows: int, expert, iota, stride: int, offset: int, step: int):
    """Bank row indices for one expert: ``expert * stride + offset + iota * step``."""
    base = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=base, data=expert, op0=nl.multiply, operand0=stride)
    nisa.tensor_scalar(dst=base, data=base, op0=nl.add, operand0=offset)
    ramp = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ramp, data=iota, op0=nl.multiply, operand0=step)
    index = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=index, data1=base, data2=ramp, op=nl.add)
    return index


def _gathered(hbm, index, rows: int, width: int, dtype):
    """A ``[rows, width]`` tile gathered from HBM, one indirect row per index."""
    tile = nl.ndarray((rows, width), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=tile,
        src=hbm.ap(
            pattern=[[width, rows], [1, width]],
            offset=0,
            vector_offset=index,
            indirect_dim=0,
        ),
    )
    return tile


@nki.jit
def _reference_gate_up_kernel(
    hidden, weight_bank, scale_bank, row_index, expert_index, iota
):
    """The reference gate/up kernel: one traced body per token tile."""
    positions = row_index.shape[0]
    h_extent = hidden.shape[1]
    n_h_blocks = h_extent // GATE_UP_SCALE_BLOCK
    n_col_blocks = scale_bank.shape[1] // n_h_blocks
    fused_cols = n_col_blocks * GATE_UP_SCALE_BLOCK
    i_extent = fused_cols // GATE_UP_FUSION
    n_i_blocks = i_extent // GATE_UP_SCALE_BLOCK
    block = positions // expert_index.shape[0]
    tiles_per_block = block // TILE_SIZE
    pad_row = hidden.shape[0] - 1

    out = nl.ndarray((positions, fused_cols), dtype=nl.float32, buffer=nl.shared_hbm)
    staged = nl.ndarray((positions, h_extent), dtype=hidden.dtype, buffer=nl.private_hbm)
    ramp = _row_iota(iota, TILE_SIZE)

    for m_tile in range(positions // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
        wanted = _padding_resolved(
            TILE_SIZE, _dense_column(row_index, TILE_SIZE, m0), pad_row
        )
        nl.store(
            staged[m0 : m0 + TILE_SIZE, 0:h_extent],
            value=_gathered(hidden, wanted, TILE_SIZE, h_extent, hidden.dtype),
        )

    for m_tile in range(positions // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
        expert = _broadcast_row(expert_index, m_tile // tiles_per_block, TILE_SIZE)
        scale_sb = _gathered(
            scale_bank,
            _bank_rows(TILE_SIZE, expert, ramp, TILE_SIZE, 0, 1),
            TILE_SIZE,
            n_h_blocks * n_col_blocks,
            nl.float32,
        )
        for i_block in range(n_i_blocks):
            gate_col = i_block * GATE_UP_SCALE_BLOCK
            up_col = i_extent + i_block * GATE_UP_SCALE_BLOCK
            gate_acc = _gate_up_sbuf()
            up_acc = _gate_up_sbuf()
            for h_block in range(n_h_blocks):
                gate_psum = _gate_up_psum()
                up_psum = _gate_up_psum()
                for h_sub in range(GATE_UP_H_TILES_PER_BLOCK):
                    h0 = h_block * GATE_UP_SCALE_BLOCK + h_sub * TILE_SIZE
                    hidden_t = nl.load_transpose2d(
                        staged[m0 : m0 + TILE_SIZE, h0 : h0 + TILE_SIZE]
                    )
                    gate_w = _gathered(
                        weight_bank,
                        _bank_rows(
                            TILE_SIZE,
                            expert,
                            ramp,
                            h_extent * n_col_blocks,
                            h0 * n_col_blocks + i_block,
                            n_col_blocks,
                        ),
                        TILE_SIZE,
                        GATE_UP_SCALE_BLOCK,
                        nl.bfloat16,
                    )
                    up_w = _gathered(
                        weight_bank,
                        _bank_rows(
                            TILE_SIZE,
                            expert,
                            ramp,
                            h_extent * n_col_blocks,
                            h0 * n_col_blocks + n_i_blocks + i_block,
                            n_col_blocks,
                        ),
                        TILE_SIZE,
                        GATE_UP_SCALE_BLOCK,
                        nl.bfloat16,
                    )
                    nisa.nc_matmul(
                        dst=gate_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        stationary=hidden_t,
                        moving=gate_w,
                        accumulate=(h_sub > 0),
                    )
                    nisa.nc_matmul(
                        dst=up_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        stationary=hidden_t,
                        moving=up_w,
                        accumulate=(h_sub > 0),
                    )
                gate_flat = h_block * n_col_blocks + i_block
                up_flat = h_block * n_col_blocks + n_i_blocks + i_block
                if h_block == 0:
                    nisa.tensor_scalar(
                        dst=gate_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        data=gate_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, gate_flat : gate_flat + 1],
                    )
                    nisa.tensor_scalar(
                        dst=up_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        data=up_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, up_flat : up_flat + 1],
                    )
                else:
                    nisa.scalar_tensor_tensor(
                        dst=gate_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        data=gate_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, gate_flat : gate_flat + 1],
                        op1=nl.add,
                        operand1=gate_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                    )
                    nisa.scalar_tensor_tensor(
                        dst=up_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        data=up_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, up_flat : up_flat + 1],
                        op1=nl.add,
                        operand1=up_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                    )
            nl.store(
                out[m0 : m0 + TILE_SIZE, gate_col : gate_col + GATE_UP_SCALE_BLOCK],
                value=gate_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
            )
            nl.store(
                out[m0 : m0 + TILE_SIZE, up_col : up_col + GATE_UP_SCALE_BLOCK],
                value=up_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
            )
    return out


def _fp8_values(seed: int, *shape: int) -> torch.Tensor:
    """Unsigned eighths, exact in fp8, so a 4096-long contraction cannot cancel."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32) / 8.0


def _scale_grid(seed: int) -> torch.Tensor:
    """Powers of two per ``128 x 128`` block: ``[E, H // 128, 2, I // 128]``."""
    generator = torch.Generator().manual_seed(seed)
    shape = (_EXPERTS, _HIDDEN // GATE_UP_SCALE_BLOCK, GATE_UP_FUSION,
             _INTERMEDIATE // GATE_UP_SCALE_BLOCK)
    exponents = torch.randint(-2, 2, shape, generator=generator).to(torch.float32)
    return torch.pow(2.0, exponents)


def _inputs(tokens: int, blocks: int, seed: int = 11) -> dict:
    """One case's seam operands: routed rows include the padding row's ``-1``."""
    generator = torch.Generator().manual_seed(seed)
    hidden = _fp8_values(seed, tokens + 1, _HIDDEN).to(torch.bfloat16)
    bank = _fp8_values(seed + 1, _EXPERTS, _HIDDEN, GATE_UP_FUSION * _INTERMEDIATE).to(_FP8)
    grid = _scale_grid(seed + 2)
    scales = torch.stack([to_gate_up_kernel_scale_operand(grid[e], _HIDDEN, _INTERMEDIATE)
                          for e in range(_EXPERTS)])
    positions = blocks * _BLOCK
    row_index = torch.randint(-1, tokens, (positions, 1), generator=generator,
                              dtype=torch.int32)
    expert_index = torch.randint(0, _EXPERTS, (blocks, 1), generator=generator,
                                 dtype=torch.int32)
    return {"hidden": hidden, "bank": bank, "scales": scales, "row_index": row_index,
            "expert_index": expert_index}


def _reference_gate_up(case: dict) -> torch.Tensor:
    """The reference kernel on the operands the entry point hands the shipped one."""
    n_blocks = int(case["scales"].shape[2])
    iota = torch.arange(TILE_SIZE, dtype=torch.int32).reshape(TILE_SIZE, 1)
    return wrap_nki(_reference_gate_up_kernel)(
        case["hidden"].to(torch.bfloat16),
        case["bank"].reshape(-1, GATE_UP_SCALE_BLOCK),
        case["scales"].to(torch.float32).reshape(-1, n_blocks),
        case["row_index"].to(torch.int32).reshape(-1, 1),
        case["expert_index"].to(torch.int32).reshape(-1, 1),
        iota,
    )


def _assert_bit_identical(tokens: int, blocks: int) -> None:
    """The shipped kernel equals the reference exactly, not within a tolerance."""
    case = _inputs(tokens, blocks)
    got = moe_gate_up_blockwise_fp8(case["hidden"], case["bank"], case["scales"],
                                    case["row_index"], case["expert_index"], _BLOCK)
    expected = _reference_gate_up(case)
    differing = int(torch.ne(got, expected).sum().item())
    assert differing == 0
    assert torch.equal(got, expected)


def test_bit_identical_at_the_prefill_token_tiles():
    """2,048 tokens routed into 81 blocks, which is 162 token tiles."""
    _assert_bit_identical(*_CASES[0])


def test_bit_identical_at_the_decode_token_tiles():
    """One token routed into 8 blocks, which is 16 token tiles."""
    _assert_bit_identical(*_CASES[1])


# --------------------------------------------------------------------------- #
# The transposes and the SBUF tiles, read off the module's source.            #
# --------------------------------------------------------------------------- #
#: The trace-time values the transpose sites are read at: one expert block of 256
#: positions over the prefill position count, hidden 4096, and an intermediate shard 512
#: wide (four scale blocks, eight fused column blocks).
_PREFILL = {"positions": 20736, "h_extent": 4096, "n_h_blocks": 32, "n_col_blocks": 8,
            "i_extent": 512, "n_i_blocks": 4}

#: The decode geometry traces the same widths over 2,048 positions.
_TILE_GEOMETRIES = (("prefill", _PREFILL), ("decode", {**_PREFILL, "positions": 2048}))

#: The names the module supplies to its own size and loop expressions, taken off the
#: module so every expression is read with the arithmetic the kernel will trace. A name
#: the module does not define is left out.
_MODULE_ARITHMETIC = ("DGE_TRANSPOSE_ROWS", "TILE_SIZE", "GATE_UP_SCALE_BLOCK",
                      "GATE_UP_H_TILES_PER_BLOCK", "GATE_UP_FUSION", "ROW_ALIGN", "_padded")

#: The element width the entry point hands the kernel: it casts the routed rows to
#: bfloat16.
_WIDTHS = (2,)

#: The most source rows the device builds its own transpose descriptors for. Spelled out
#: here rather than read off the module, so a module that stepped 32 rows would fail
#: instead of moving the bar with itself.
_DGE_ROWS = 16


def _arithmetic() -> dict:
    """The module's own arithmetic, by name; a module without a name leaves it out."""
    return {name: getattr(mod, name) for name in _MODULE_ARITHMETIC if hasattr(mod, name)}


def _module_source() -> str:
    return pathlib.Path(mod.__file__).read_text(encoding="utf-8")


def _sites(source: str) -> list:
    """Every transpose site of ``source``, read with the module's own arithmetic."""
    return census(source, _PREFILL, _arithmetic(), _WIDTHS)


def _tile_rows(source: str, at: dict) -> list:
    """Every SBUF tile ``source`` declares at ``at``, as ``(line, name, row bytes)``."""
    return tile_rows(source, at, _arithmetic(), _WIDTHS)


def _handed_width(source: str) -> int | None:
    """The element width handed to ``hidden``: 2 when the caller casts to bfloat16."""
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, ast.FunctionDef) or fn.name != "moe_gate_up_blockwise_fp8":
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "to" \
                    and getattr(node.func.value, "id", "") == "hidden_states" and node.args:
                target = node.args[0]
                name = getattr(target, "attr", "")
                return {"bfloat16": 2, "float16": 2, "float32": 4}.get(name)
    return None


def test_every_transpose_moves_sixteen_two_byte_rows_onto_a_readable_line():
    """No transpose starts off a line, steps over 16 rows, or takes wider rows."""
    source = _module_source()
    sites = _sites(source)
    unreadable = tuple(s.line for s in sites if s.misaligned is None)
    misaligned = tuple(s.line for s in sites if s.misaligned)
    shaped = tuple(s.line for s in sites if s.host_shaped(_DGE_ROWS))
    width = _handed_width(source)
    assert sites, "the census read no transpose site"
    assert unreadable == (), f"destinations the arithmetic cannot place: {unreadable}"
    assert shaped == (), f"transposes the device cannot shape: {shaped}"
    assert misaligned == (), f"destinations off a 32-byte line: {misaligned}"
    assert width == 2, f"the seam hands rows of {width} bytes"


def test_every_sbuf_tile_row_is_a_whole_number_of_32_byte_lines():
    """Every SBUF tile the module declares is sizable and is whole 32-byte lines.

    The allocator packs tiles back to back per partition, so one tile whose row is not
    a whole number of lines moves every later tile's base off the line. Both geometries
    are read, because a tile's width can depend on the position count.
    """
    source = _module_source()
    for name, at in _TILE_GEOMETRIES:
        rows = _tile_rows(source, at)
        unreadable = tuple(sorted({(line, tile) for line, tile, size in rows
                                   if size is None}))
        narrow = tuple((line, tile, size) for line, tile, size in rows
                       if size is not None and size % LINE)
        assert unreadable == (), \
            f"tile rows the arithmetic cannot size at {name}: {unreadable}"
        assert narrow == (), \
            f"tile rows at {name} that are not whole 32-byte lines: {narrow}"
