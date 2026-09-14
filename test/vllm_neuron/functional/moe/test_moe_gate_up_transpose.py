# SPDX-License-Identifier: Apache-2.0
"""Every DMA transpose in the routed gate/up kernel moves 16 two-byte source rows.

The kernel transposes each token tile's hidden tiles onto partitions once per token tile,
16 source rows per DMA, and reads them for every intermediate block. Three things are read
here: that every transpose in the module, read at its call site through the helper that
issues the DMAs, lands a readable destination on a 32-byte line, moves 16 source rows per
DMA, and is handed two-byte rows, all from the module's own arithmetic; that every SBUF
tile the module declares has a per-partition row of whole 32-byte lines, so the allocator
packing tiles back to back leaves every tile base on a line; and that the values did not
move, against a frozen copy of the kernel as it stood before the change, exactly rather
than within a tolerance, at the served token-tile counts.

Run under ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1
NEURON_PLATFORM_TARGET_OVERRIDE=trn2``; nothing here reads or sets an environment variable.
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

#: The constants the frozen copy below reads, RETYPED on purpose: a frozen reference
#: that imported them would follow the live module if they ever moved.
TILE_SIZE = 128
GATE_UP_SCALE_BLOCK = 128
GATE_UP_H_TILES_PER_BLOCK = 1
GATE_UP_FUSION = 2

_FP8 = torch.float8_e4m3fn

#: The served per-rank geometry: hidden width, one expert's intermediate shard, the
#: block the mapping hands the kernel. Two experts size the bank only; the instruction
#: stream does not depend on the expert count.
_HIDDEN = 4096
_INTERMEDIATE = 512
_EXPERTS = 2
_BLOCK = 256

#: ``(tokens, blocks)`` per case: the prefill bucket (162 token tiles) and decode (16).
_CASES = ((2048, 81), (1, 8))


def _emit(tag: str, **values: object) -> None:
    """Print one machine-readable reading line for the transcript's reader."""
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"MOE|{tag}|{body}", flush=True)


def _gate_up_sbuf(rows: int = TILE_SIZE, cols: int = GATE_UP_SCALE_BLOCK):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)


def _gate_up_psum(rows: int = TILE_SIZE, cols: int = GATE_UP_SCALE_BLOCK):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)


def _row_iota(iota_hbm, rows: int):
    """Frozen at the base commit: _row_iota."""
    tile = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=iota_hbm.ap(pattern=[[1, rows], [1, 1]], offset=0))
    return tile


def _dense_column(hbm, rows: int, offset: int):
    """Frozen at the base commit: _dense_column."""
    tile = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=hbm.ap(pattern=[[1, rows], [1, 1]], offset=offset))
    return tile


def _broadcast_row(hbm, row: int, rows: int):
    """Frozen at the base commit: _broadcast_row."""
    tile = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=hbm.ap(pattern=[[0, rows], [1, 1]], offset=row))
    return tile


def _padding_resolved(rows: int, positions, pad_row: int):
    """Frozen at the base commit: _padding_resolved."""
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
    """Frozen at the base commit: _bank_rows."""
    base = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=base, data=expert, op0=nl.multiply, operand0=stride)
    nisa.tensor_scalar(dst=base, data=base, op0=nl.add, operand0=offset)
    ramp = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ramp, data=iota, op0=nl.multiply, operand0=step)
    index = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=index, data1=base, data2=ramp, op=nl.add)
    return index


def _gathered(hbm, index, rows: int, width: int, dtype):
    """Frozen at the base commit: _gathered."""
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
def _landed_kernel(hidden, weight_bank, scale_bank, row_index, expert_index, iota):
    """Frozen at the base commit: moe_gate_up_blockwise_fp8_kernel."""
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


def _landed(case: dict) -> torch.Tensor:
    """The frozen kernel on the operands the seam hands the live one."""
    n_blocks = int(case["scales"].shape[2])
    iota = torch.arange(TILE_SIZE, dtype=torch.int32).reshape(TILE_SIZE, 1)
    return wrap_nki(_landed_kernel)(
        case["hidden"].to(torch.bfloat16),
        case["bank"].reshape(-1, GATE_UP_SCALE_BLOCK),
        case["scales"].to(torch.float32).reshape(-1, n_blocks),
        case["row_index"].to(torch.int32).reshape(-1, 1),
        case["expert_index"].to(torch.int32).reshape(-1, 1),
        iota,
    )


def _assert_bit_identical(tokens: int, blocks: int) -> None:
    """The live kernel through the seam equals the frozen one exactly; differences are read."""
    case = _inputs(tokens, blocks)
    got = moe_gate_up_blockwise_fp8(case["hidden"], case["bank"], case["scales"],
                                    case["row_index"], case["expert_index"], _BLOCK)
    want = _landed(case)
    differing = int(torch.ne(got, want).sum().item())
    maxabs = float((got - want).abs().max().item()) if differing else 0.0
    _emit("BIT_IDENTITY", tokens=tokens, blocks=blocks, tiles=blocks * _BLOCK // TILE_SIZE,
          differing=differing, equal=torch.equal(got, want), maxabs=maxabs,
          shape=tuple(got.shape))
    assert differing == 0
    assert torch.equal(got, want)


def test_bit_identical_at_the_prefill_token_tiles():
    """2,048 tokens routed into 81 blocks: 162 token tiles, the served prefill count."""
    _assert_bit_identical(*_CASES[0])


def test_bit_identical_at_the_decode_token_tiles():
    """One token routed into 8 blocks: 16 token tiles, the served decode count."""
    _assert_bit_identical(*_CASES[1])

#: The trace-time values the transpose sites are read at: the served geometry, one expert
#: block of 256 positions, the prefill position count; hidden 4096, the intermediate shard
#: 512 wide (four scale blocks, eight fused column blocks).
_SERVED = {"positions": 20736, "h_extent": 4096, "n_h_blocks": 32, "n_col_blocks": 8,
           "i_extent": 512, "n_i_blocks": 4}

#: The decode geometry traces the same widths over 2,048 positions.
_TILE_GEOMETRIES = (("prefill", _SERVED), ("decode", {**_SERVED, "positions": 2048}))

#: The names the module supplies to its OWN size and loop expressions, read off the module
#: and never retyped, so an expression is read with the arithmetic the kernel will trace.
#: Absent names are left out.
_MODULE_ARITHMETIC = ("DGE_TRANSPOSE_ROWS", "TILE_SIZE", "GATE_UP_SCALE_BLOCK",
                      "GATE_UP_H_TILES_PER_BLOCK", "GATE_UP_FUSION", "ROW_ALIGN", "_padded")

#: The element width the seam hands the kernel: it casts the routed rows to bfloat16.
_WIDTHS = (2,)

#: The source rows per DMA the device generates its own descriptors for, RETYPED: the
#: criterion the module's rows are graded against, never read off the module, so a module
#: that stepped 32 rows is read as stepping 32 and reddens the census.
_DGE_ROWS = 16

#: The SBUF tiles the functions read here declare, as (function, tile name, bytes per partition),
#: RETYPED: a tile one of them drops, renames or gives another row width reddens this item, which is
#: the property it exists for. The list is typed here and never read back off the module it grades.
_OWNED_TILES = (
    ("_affinity_rows", "index", 64),
    ("_affinity_rows", "scaled", 64),
    ("_bank_rows", "base", 64),
    ("_bank_rows", "index", 64),
    ("_bank_rows", "ramp", 64),
    ("_broadcast_row", "tile", 64),
    ("_dense_column", "tile", 64),
    ("_gathered", "tile", 64),
    ("_gathered", "tile", 256),
    ("_gathered", "tile", 512),
    ("_gathered", "tile", 1024),
    ("_gathered", "tile", 8192),
    ("_padding_resolved", "bumped", 64),
    ("_padding_resolved", "index", 64),
    ("_padding_resolved", "negative", 64),
    ("_row_iota", "tile", 64),
    ("moe_gate_up_blockwise_fp8_kernel", "gate_acc", 512),
    ("moe_gate_up_blockwise_fp8_kernel", "tile", 256),
    ("moe_gate_up_blockwise_fp8_kernel", "up_acc", 512),
)

#: Every OTHER tile this shared module declares, in the same shape and RETYPED: the ones the swiglu
#: and down kernels declare, and the staging tiles the rolled token-tile loops add. Their widths are
#: not graded here; counting them by name keeps the module's own tile count EXACT below, so a tile
#: that nobody named -- an extra one anywhere in the module -- reddens this item instead of arriving
#: unread. A change that adds a tile to this module names it here. Both geometries read these rows.
_FOREIGN_TILES = (
    ("_loaded", "tile", 256),
    ("_loaded", "tile", 512),
    ("moe_down_blockwise_fp8_kernel", "acc", 512),
    ("moe_down_blockwise_fp8_kernel", "scaled", 512),
    ("moe_swiglu_transposed_kernel", "bounded", 512),
    ("moe_swiglu_transposed_kernel", "bounded_up", 512),
    ("moe_swiglu_transposed_kernel", "floored", 512),
    ("moe_swiglu_transposed_kernel", "gated", 512),
    ("moe_swiglu_transposed_kernel", "out_sb", 512),
    ("moe_swiglu_transposed_kernel", "silu", 512),
    ("moe_swiglu_transposed_kernel", "squashed", 512),
)

#: The functions whose tiles are owned above, read off that list and not off the module.
_OWNER_FUNCTIONS = tuple(sorted({owner for owner, _name, _size in _OWNED_TILES}))


def _arithmetic() -> dict:
    """The module's own arithmetic, by name; a module without a name leaves it out."""
    return {name: getattr(mod, name) for name in _MODULE_ARITHMETIC if hasattr(mod, name)}


def _module_source() -> str:
    return pathlib.Path(mod.__file__).read_text(encoding="utf-8")


def _sites(source: str) -> list:
    """Every transpose site of ``source`` at the served geometry, read with the module's arithmetic."""
    return census(source, _SERVED, _arithmetic(), _WIDTHS)


def _tile_rows(source: str, at: dict) -> list:
    """Every SBUF tile ``source`` declares at ``at``, as ``(line, name, bytes per partition)``."""
    return tile_rows(source, at, _arithmetic(), _WIDTHS)


def _tiles_by_owner(source: str, rows: list) -> tuple:
    """Every tile row of ``rows`` as ``(top-level function, name, bytes per partition)``, sorted."""
    owner: dict[int, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            # A tile a nested loop body declares belongs to the function the body is written in.
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner[line] = node.name
    found = [(owner.get(line, "<module>"), tile, size) for line, tile, size in rows]
    return tuple(sorted(found, key=lambda one: (one[0], one[1], -1 if one[2] is None else one[2])))


def _handed_width(source: str) -> int | None:
    """The element width the seam hands the kernel's ``hidden``: 2 when it casts to bfloat16."""
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
    """Every transpose is read; none starts off a line, moves rows the device cannot shape, or is handed wider rows."""
    source = _module_source()
    sites = _sites(source)
    unreadable = tuple(s.line for s in sites if s.misaligned is None)
    misaligned = tuple(s.line for s in sites if s.misaligned)
    shaped = tuple(s.line for s in sites if s.host_shaped(_DGE_ROWS))
    width = _handed_width(source)
    _emit("DMA_TRANSPOSE_SITES", count=len(shaped), unreadable=unreadable, misaligned=misaligned,
          step=_DGE_ROWS, width=width, transposes=len(sites), lines=shaped,
          readings=[(s.line, s.source, s.rows, s.width, s.via) for s in sites])
    assert sites, "the census read no transpose site"
    assert unreadable == (), f"destinations the arithmetic cannot place: {unreadable}"
    assert shaped == (), f"transposes the device cannot shape: {shaped}"
    assert misaligned == (), f"destinations off a 32-byte line: {misaligned}"
    assert width == 2, f"the seam hands rows of {width} bytes"


#: The staged form the controls below call, stepping the device's 16 rows as a literal so a
#: control reads the same in a tree whose module has no step constant.
_HELPER = """
def _transpose_rows(dst, src_hbm, row_stride, rows, width, offset):
    for r0 in range(0, rows, 16):
        n = min(16, rows - r0)
        nisa.dma_transpose(dst=dst[:, r0:r0 + n],
                           src=src_hbm.ap(pattern=[[row_stride, n], [1, width]], offset=offset))
"""

#: A body that transposes a whole token tile in one DMA, the form this change retires.
def test_every_sbuf_tile_row_is_a_whole_number_of_32_byte_lines():
    """At the prefill and decode geometries every declared SBUF tile is sized and is whole lines.

    The allocator packs tiles back to back per partition, so one tile whose row is not a whole
    number of lines moves every later tile's base off the line; this reads the tiles the
    transposes above land beside, through the helpers that make them and per caller. The tiles
    these functions own are read by name AND row width; every other tile this shared module
    declares is named and counted, so its whole tile count stays exact.
    """
    source = _module_source()
    for name, at in _TILE_GEOMETRIES:
        rows = _tile_rows(source, at)
        unreadable = tuple(sorted({(line, tile) for line, tile, size in rows if size is None}))
        narrow = tuple((line, tile, size) for line, tile, size in rows
                       if size is not None and size % LINE)
        owned = tuple(one for one in _tiles_by_owner(source, rows) if one[0] in _OWNER_FUNCTIONS)
        want = tuple(sorted(_OWNED_TILES))
        differing = tuple(sorted(set(owned) ^ set(want)))
        whole = len(want) + len(_FOREIGN_TILES)
        _emit("NARROW_TILES", geometry=name, count=len(narrow), unreadable=unreadable,
              tiles=len(rows), width=_WIDTHS[0], lines=tuple(r[0] for r in narrow))
        _emit("OWN_TILES", geometry=name, functions=len(_OWNER_FUNCTIONS), tiles=len(owned),
              foreign=len(_FOREIGN_TILES), total=len(rows), differing=differing)
        assert unreadable == (), f"tile rows the arithmetic cannot size: {unreadable}"
        assert narrow == (), f"tile rows that are not whole 32-byte lines: {narrow}"
        assert owned == want, (f"the tiles these functions declare at {name} moved: {differing} "
                               f"(read {len(owned)}, want {len(want)})")
        assert len(rows) == whole, \
            f"SBUF tiles the census reads at {name}: {len(rows)}, not {whole}"


_PLANTED_WHOLE_TILE = _HELPER + """
def body(staged, h_extent):
    for h_tile in range(n_h_blocks):
        hidden_t = nl.load_transpose2d(staged[0:TILE_SIZE, h_tile * TILE_SIZE:(h_tile + 1) * TILE_SIZE])
"""

#: A caller whose destination slice starts three elements in: 6 bytes at two-byte rows.
_PLANTED_OFF_LINE = _HELPER + """
def body(staged, hidden, h_extent):
    tile = nl.ndarray((TILE_SIZE, TILE_SIZE + 8), dtype=hidden.dtype, buffer=nl.sbuf)
    _transpose_rows(tile[:, 3:3 + TILE_SIZE], staged, h_extent, TILE_SIZE, TILE_SIZE, 0)
"""

#: A caller of a helper that steps 32 rows per DMA, twice the device's shape, the module's own
#: constant notwithstanding: the reader must grade rows against the retyped criterion, not the
#: module's.
_PLANTED_WIDE_STEP = """
def _transpose_rows(dst, src_hbm, row_stride, rows, width, offset):
    for r0 in range(0, rows, 32):
        n = min(32, rows - r0)
        nisa.dma_transpose(dst=dst[:, r0:r0 + n],
                           src=src_hbm.ap(pattern=[[row_stride, n], [1, width]], offset=offset))

def body(staged, hidden):
    tile = nl.ndarray((TILE_SIZE, TILE_SIZE), dtype=hidden.dtype, buffer=nl.sbuf)
    _transpose_rows(tile, staged, h_extent, TILE_SIZE, TILE_SIZE, 0)
"""

#: A caller whose destination the body never declared, which no arithmetic can place.
_PLANTED_UNREADABLE = _HELPER + """
def body(staged, h_extent):
    _transpose_rows(somewhere, staged, h_extent, TILE_SIZE, TILE_SIZE, 0)
"""


def test_control_the_reader_finds_a_planted_whole_tile_transpose():
    """The same reader names the one site that moves a whole tile at once."""
    sites = _sites(_PLANTED_WHOLE_TILE)
    shaped = tuple(s.line for s in sites if s.host_shaped(_DGE_ROWS))
    _emit("CONTROL_READER_FIRES", count=len(shaped), rows=[s.rows for s in sites], lines=shaped)
    assert len(sites) == 1 and len(shaped) == 1


def test_control_the_reader_finds_a_planted_off_line_caller():
    """The same reader names the one caller whose destination starts 6 bytes into a line."""
    sites = _sites(_PLANTED_OFF_LINE)
    misaligned = tuple(s.line for s in sites if s.misaligned)
    _emit("CONTROL_LINE_READER_FIRES", count=len(misaligned), offsets=[s.offsets for s in sites],
          lines=misaligned)
    assert len(sites) == 1 and len(misaligned) == 1


#: Three tiles: a 4-byte row, an 8-byte row made through a helper, and one whole line.
_PLANTED_NARROW = """
def _pair(rows):
    return nl.ndarray((rows, 2), dtype=nl.int32, buffer=nl.sbuf)

def body(hbm):
    column = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
    both = _pair(TILE_SIZE)
    line = nl.ndarray((TILE_SIZE, 8), dtype=nl.int32, buffer=nl.sbuf)
"""

#: One tile sized by a name the arithmetic never has.
_PLANTED_UNSIZED = """
def body(hbm):
    scratch = nl.ndarray((TILE_SIZE, somewhere), dtype=nl.int32, buffer=nl.sbuf)
"""


def test_control_the_tile_census_finds_a_planted_narrow_row():
    """The same reader names the 4-byte tile and the 8-byte one made through a helper, not the line."""
    rows = _tile_rows(_PLANTED_NARROW, _SERVED)
    narrow = tuple((line, tile, size) for line, tile, size in rows
                   if size is not None and size % LINE)
    _emit("CONTROL_TILE_READER_FIRES", count=len(narrow), rows=narrow, tiles=len(rows))
    assert len(rows) == 3 and narrow == ((6, "column", 4), (7, "both", 8))


def test_control_an_unsized_tile_is_never_a_pass():
    """A tile the arithmetic cannot size reads ``None``, which the census refuses."""
    rows = _tile_rows(_PLANTED_UNSIZED, _SERVED)
    _emit("CONTROL_UNSIZED_TILE_FIRES", count=len(rows), rows=rows)
    assert rows == [(3, "scratch", None)]


def test_control_the_reader_finds_a_planted_caller_of_a_wide_stepping_helper():
    """The same reader names the one caller whose helper moves 32 rows per DMA."""
    sites = _sites(_PLANTED_WIDE_STEP)
    shaped = tuple(s.line for s in sites if s.host_shaped(_DGE_ROWS))
    _emit("CONTROL_WIDE_STEP_READER_FIRES", count=len(shaped), rows=[s.rows for s in sites],
          lines=shaped)
    assert len(sites) == 1 and len(shaped) == 1 and sites[0].rows == (32,)


def test_control_an_unreadable_destination_is_never_a_pass():
    """A destination the arithmetic cannot place reads as unreadable, which the census refuses."""
    sites = _sites(_PLANTED_UNREADABLE)
    unreadable = tuple(s.line for s in sites if s.misaligned is None)
    _emit("CONTROL_UNREADABLE_FIRES", count=len(unreadable), lines=unreadable)
    assert len(sites) == 1 and len(unreadable) == 1
