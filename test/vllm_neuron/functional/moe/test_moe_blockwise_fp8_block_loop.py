# SPDX-License-Identifier: Apache-2.0
"""The three routed MoE kernels: their block loops, their refusals, and their results.

Each kernel walks the token tiles of one expert block inside a dynamic loop body, so a
block's operands are gathered once per block and every operand is addressed through an
access pattern that the block register offsets. A launch is sharded over two programs
whose ranges must together cover every block. Results are compared against an
independent unrolled reference implementation of each kernel at several geometries.
"""

from __future__ import annotations

import pytest
import torch

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from vllm_neuron.functional.moe import moe_blockwise_fp8 as live
from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
    MoeBlockwiseFp8Error,
    moe_down_blockwise_fp8,
    moe_gate_up_blockwise_fp8,
    moe_swiglu_transposed,
    to_down_kernel_scale_operand,
    to_gate_up_kernel_scale_operand,
    down_kernel_scale_shape,
    gate_up_kernel_scale_shape,
)

#: The kernel geometry the reference implementations below are written against. Spelled
#: out rather than imported, so a reference cannot follow the module it checks.
TILE_SIZE = 128
GATE_UP_SCALE_BLOCK = 128
GATE_UP_H_TILES_PER_BLOCK = 1
GATE_UP_FUSION = 2
DGE_TRANSPOSE_ROWS = 16
ROW_ALIGN = 16
_SWIGLU_BOUND_COLUMNS = 3

#: The activation bound the model configures: ``glm5_next/config.py``'s
#: ``swiglu_limit``, handed to both halves of the routed MoE path in
#: ``glm5_next/model_fp8.py``. Taken from the model's configuration, not off the kernel.
_MODEL_SWIGLU_LIMIT = 10.0

_FP8 = torch.float8_e4m3fn

#: Tokens behind every case: the routing column indexes these rows, and the padding row
#: is the appended one.
_TOKENS = 256

#: Two experts size the banks. The instruction stream does not depend on the count.
_EXPERTS = 2

#: Three geometries, ``blocks`` dynamic bodies of ``tiles`` token tiles each: 18 tiles
#: with a full last block, 16 tiles at a second quotient, and the same 18 tiles over two
#: contraction blocks per axis.
_G18 = {"name": "G18", "blocks": 2, "tiles": 9, "h_blocks": 1, "i_blocks": 1}
_G16 = {"name": "G16", "blocks": 2, "tiles": 8, "h_blocks": 1, "i_blocks": 1}
_GN = {"name": "GN", "blocks": 2, "tiles": 9, "h_blocks": 2, "i_blocks": 2}
#: Two more geometries for the two-program launch: one block of one tile, which both
#: programs run whole and write identically, and three blocks of three tiles, where the
#: programs share the middle block and the middle token tile.
_G1 = {"name": "G1", "blocks": 1, "tiles": 1, "h_blocks": 1, "i_blocks": 1}
_G3 = {"name": "G3", "blocks": 3, "tiles": 3, "h_blocks": 1, "i_blocks": 1}
_ALL_CASES = (_G18, _G16, _GN, _G1, _G3)


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
# An independent reference implementation of the three routed kernels, with   #
# every token-tile loop written out rather than run from a dynamic body.      #
# --------------------------------------------------------------------------- #
def _padded(width: int) -> int:
    """``width`` rounded up to the next multiple of the SBUF row alignment."""
    return ((width + ROW_ALIGN - 1) // ROW_ALIGN) * ROW_ALIGN


def _column(rows: int):
    """A one-column int32 tile whose row is a whole 32-byte line."""
    return nl.ndarray((rows, ROW_ALIGN), dtype=nl.int32, buffer=nl.sbuf)[:, 0:1]


def _row_iota(iota_hbm, rows: int):
    """A column holding ``0 .. rows - 1``, one value per partition."""
    tile = _column(rows)
    nisa.dma_copy(dst=tile, src=iota_hbm.ap(pattern=[[1, rows], [1, 1]], offset=0))
    return tile


def _dense_column(hbm, rows: int, offset: int):
    """A column of ``rows`` consecutive int32 values read from ``offset``."""
    tile = _column(rows)
    nisa.dma_copy(dst=tile, src=hbm.ap(pattern=[[1, rows], [1, 1]], offset=offset))
    return tile


def _broadcast_row(hbm, row: int, rows: int):
    """One int32 row replicated into every partition of a column."""
    tile = _column(rows)
    nisa.dma_copy(dst=tile, src=hbm.ap(pattern=[[0, rows], [1, 1]], offset=row))
    return tile


def _padding_resolved(rows: int, positions, pad_row: int):
    """Routed positions with the negative padding marker moved onto ``pad_row``."""
    negative = _column(rows)
    nisa.tensor_scalar(dst=negative, data=positions, op0=nl.less, operand0=0)
    bumped = _column(rows)
    nisa.tensor_scalar(
        dst=bumped, data=negative, op0=nl.multiply, operand0=pad_row + 1
    )
    index = _column(rows)
    nisa.tensor_tensor(dst=index, data1=positions, data2=bumped, op=nl.add)
    return index


def _bank_rows(rows: int, expert, iota, stride: int, offset: int, step: int):
    """Bank row indices for one expert: ``expert * stride + offset + iota * step``."""
    base = _column(rows)
    nisa.tensor_scalar(dst=base, data=expert, op0=nl.multiply, operand0=stride)
    nisa.tensor_scalar(dst=base, data=base, op0=nl.add, operand0=offset)
    ramp = _column(rows)
    nisa.tensor_scalar(dst=ramp, data=iota, op0=nl.multiply, operand0=step)
    index = _column(rows)
    nisa.tensor_tensor(dst=index, data1=base, data2=ramp, op=nl.add)
    return index


def _affinity_rows(rows: int, resolved, expert, n_experts: int):
    """Affinity row indices: ``resolved_row * n_experts + expert``."""
    scaled = _column(rows)
    nisa.tensor_scalar(dst=scaled, data=resolved, op0=nl.multiply, operand0=n_experts)
    index = _column(rows)
    nisa.tensor_tensor(dst=index, data1=scaled, data2=expert, op=nl.add)
    return index


def _gathered(hbm, index, rows: int, width: int, dtype):
    """A ``[rows, width]`` tile gathered from HBM, one indirect row per index."""
    tile = nl.ndarray((rows, _padded(width)), dtype=dtype, buffer=nl.sbuf)[:, 0:width]
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


def _transpose_rows(dst, src_hbm, row_stride: int, rows: int, width: int, offset: int):
    """``rows`` source rows transposed onto partitions, 16 rows per DMA."""
    for r0 in range(0, rows, DGE_TRANSPOSE_ROWS):
        n = min(DGE_TRANSPOSE_ROWS, rows - r0)
        nisa.dma_transpose(
            dst=dst[:, r0 : r0 + n],
            src=src_hbm.ap(
                pattern=[[row_stride, n], [1, width]], offset=offset + r0 * row_stride
            ),
        )


def _gate_up_sbuf(rows: int = TILE_SIZE, cols: int = GATE_UP_SCALE_BLOCK):
    """A ``[rows, cols]`` fp32 SBUF tile."""
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)


def _gate_up_psum(rows: int = TILE_SIZE, cols: int = GATE_UP_SCALE_BLOCK):
    """A ``[rows, cols]`` fp32 PSUM tile."""
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)


@nki.jit
def _reference_gate_up_kernel(
    hidden, weight_bank, scale_bank, row_index, expert_index, iota
):
    """The reference gate/up kernel: one written-out body per token tile."""
    positions = row_index.shape[0]
    h_extent = hidden.shape[1]
    n_h_blocks = h_extent // GATE_UP_SCALE_BLOCK
    # One scale column per (H block, fused column block), which the entry point checks
    # against :func:`gate_up_kernel_scale_shape`, so the width over the H blocks is the
    # fused column count in blocks -- and an odd fusion is refused there.
    n_col_blocks = scale_bank.shape[1] // n_h_blocks
    fused_cols = n_col_blocks * GATE_UP_SCALE_BLOCK
    i_extent = fused_cols // GATE_UP_FUSION
    n_i_blocks = i_extent // GATE_UP_SCALE_BLOCK
    # One expert per block: the routing operands' own lengths give the block length.
    block = positions // expert_index.shape[0]
    tiles_per_block = block // TILE_SIZE
    pad_row = hidden.shape[0] - 1

    out = nl.ndarray((positions, fused_cols), dtype=nl.float32, buffer=nl.shared_hbm)
    # The routed rows land here first. Each token tile's hidden tiles are then
    # transposed onto partitions once, 16 source rows per DMA, and read by every
    # intermediate block. Kernel-internal, so ``private_hbm``.
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
        # The expert's own scale operand, one indirect gather of a tiny tensor.
        scale_sb = _gathered(
            scale_bank,
            _bank_rows(TILE_SIZE, expert, ramp, TILE_SIZE, 0, 1),
            TILE_SIZE,
            n_h_blocks * n_col_blocks,
            nl.float32,
        )
        # [H=TILE_SIZE partitions, B=TILE_SIZE free] per hidden tile, in the source dtype.
        hidden_t = []
        for h_tile in range(n_h_blocks * GATE_UP_H_TILES_PER_BLOCK):
            tile = nl.ndarray((TILE_SIZE, TILE_SIZE), dtype=hidden.dtype, buffer=nl.sbuf)
            _transpose_rows(
                tile, staged, h_extent, TILE_SIZE, TILE_SIZE, m0 * h_extent + h_tile * TILE_SIZE
            )
            hidden_t.append(tile)
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
                    hidden_tile = hidden_t[h0 // TILE_SIZE]
                    # [H=TILE_SIZE partitions, I=GATE_UP_SCALE_BLOCK free], each
                    # addressed at ``expert * H * n_col_blocks + (h0 + i) *
                    # n_col_blocks + column_block`` in the bank's own row units.
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
                    # dst = stationary.T @ moving = [B, I]. The accumulate flag is
                    # explicit rather than inferred, so first-write-overwrites is
                    # visible here.
                    nisa.nc_matmul(
                        dst=gate_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        stationary=hidden_tile,
                        moving=gate_w,
                        accumulate=(h_sub > 0),
                    )
                    nisa.nc_matmul(
                        dst=up_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        stationary=hidden_tile,
                        moving=up_w,
                        accumulate=(h_sub > 0),
                    )
                # The same flattening `gate_up_flat_scale_index` returns, written
                # as the block walk that produces it: `h_block * n_col_blocks +
                # (gate_or_up * n_i_blocks + i_block)`.
                gate_flat = h_block * n_col_blocks + i_block
                up_flat = h_block * n_col_blocks + n_i_blocks + i_block
                if h_block == 0:
                    # The first block initialises the accumulator, so there is no
                    # zeroing pass over SBUF.
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


@nki.jit
def _reference_swiglu_kernel(gate_up, bounds):
    """The reference activation kernel: one written-out body per token tile."""
    tokens, fused_cols = gate_up.shape
    i_extent = fused_cols // GATE_UP_FUSION
    # A one-column operand means no bounds, and then no bound instruction is emitted
    # at all: an unset limit removes its own line at trace time.
    bounded_config = bounds.shape[1] == _SWIGLU_BOUND_COLUMNS
    out = nl.ndarray((i_extent, tokens), dtype=nl.float32, buffer=nl.shared_hbm)
    if bounded_config:
        bounds_sb = nl.load(bounds[0:TILE_SIZE, 0:_SWIGLU_BOUND_COLUMNS])

    for m_tile in range(tokens // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
        for i_block in range(i_extent // GATE_UP_SCALE_BLOCK):
            i0 = i_block * GATE_UP_SCALE_BLOCK
            gate = nl.load(
                gate_up[m0 : m0 + TILE_SIZE, i0 : i0 + GATE_UP_SCALE_BLOCK],
                dtype=nl.float32,
            )
            up = nl.load(
                gate_up[
                    m0 : m0 + TILE_SIZE,
                    i_extent + i0 : i_extent + i0 + GATE_UP_SCALE_BLOCK,
                ],
                dtype=nl.float32,
            )
            # The bounds, on the tiles just loaded. ``maximum`` and ``minimum`` are the
            # closed-form pair this package uses for a bound elsewhere
            # (``dsa/causal_fill.py``). An unset limit emits no instruction rather than
            # a bound at infinity, and a configured limit is a per-partition column for
            # the reason :func:`_swiglu_bound_operand` records.
            if bounded_config:
                bounded = _gate_up_sbuf()
                nisa.tensor_scalar(
                    dst=bounded,
                    data=gate,
                    op0=nl.minimum,
                    operand0=bounds_sb[0:TILE_SIZE, 0:1],
                )
                gate = bounded
                # Two calls and not one fused pair: a ``[P, 1]`` column is an
                # ``operand0`` shape in this package and no call passes one as
                # ``operand1``, and a maximum followed by a minimum is the same
                # closed form.
                floored = _gate_up_sbuf()
                nisa.tensor_scalar(
                    dst=floored,
                    data=up,
                    op0=nl.maximum,
                    operand0=bounds_sb[0:TILE_SIZE, 1:2],
                )
                bounded_up = _gate_up_sbuf()
                nisa.tensor_scalar(
                    dst=bounded_up,
                    data=floored,
                    op0=nl.minimum,
                    operand0=bounds_sb[0:TILE_SIZE, 2:3],
                )
                up = bounded_up
            # sigmoid is one activation-engine op on this image, not a composition.
            squashed = _gate_up_sbuf()
            nisa.activation(dst=squashed, data=gate, op=nl.sigmoid)
            # SiLU(gate) = gate * sigmoid(gate). ``1.0`` is the identity scalar the
            # three-operand form needs; see the section comment on tensor_tensor.
            silu = _gate_up_sbuf()
            nisa.scalar_tensor_tensor(
                dst=silu,
                data=gate,
                op0=nl.multiply,
                operand0=1.0,
                op1=nl.multiply,
                operand1=squashed,
            )
            gated = _gate_up_sbuf()
            nisa.scalar_tensor_tensor(
                dst=gated,
                data=silu,
                op0=nl.multiply,
                operand0=1.0,
                op1=nl.multiply,
                operand1=up,
            )
            # [B, I] -> [I, B], the orientation the down projection contracts on.
            transposed = _gate_up_psum()
            nisa.nc_transpose(dst=transposed, data=gated)
            out_sb = _gate_up_sbuf()
            nisa.tensor_copy(dst=out_sb, src=transposed)
            nl.store(
                out[i0 : i0 + GATE_UP_SCALE_BLOCK, m0 : m0 + TILE_SIZE],
                value=out_sb,
            )
    return out


@nki.jit
def _reference_down_kernel(
    intermediate_t,
    weight_bank,
    scale_bank,
    affinity_bank,
    row_index,
    expert_index,
    iota,
):
    """The reference down kernel: one written-out body per token tile."""
    i_extent, positions = intermediate_t.shape
    n_i_blocks = i_extent // GATE_UP_SCALE_BLOCK
    # One scale column per (I block, H block), so the width over the I blocks is H.
    n_h_blocks = scale_bank.shape[1] // n_i_blocks
    h_extent = n_h_blocks * GATE_UP_SCALE_BLOCK
    # The scale operand stacks one ``TILE_SIZE``-tall operand per expert, so its
    # height IS the affinity row stride.
    n_experts = scale_bank.shape[0] // TILE_SIZE
    block = positions // expert_index.shape[0]
    tiles_per_block = block // TILE_SIZE
    # The padding row is the appended one. The seam checks the affinity bank's length
    # against ``T`` for exactly this reason, so the quotient cannot move it silently.
    pad_row = affinity_bank.shape[0] // n_experts - 1

    out = nl.ndarray((positions, h_extent), dtype=nl.float32, buffer=nl.shared_hbm)
    ramp = _row_iota(iota, TILE_SIZE)

    for m_tile in range(positions // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
        expert = _broadcast_row(expert_index, m_tile // tiles_per_block, TILE_SIZE)
        scale_sb = _gathered(
            scale_bank,
            _bank_rows(TILE_SIZE, expert, ramp, TILE_SIZE, 0, 1),
            TILE_SIZE,
            n_i_blocks * n_h_blocks,
            nl.float32,
        )
        # Loaded per token tile, never hoisted out of this loop: a token is a
        # partition, the partition axis serves 128 of them, and a whole-column load
        # would bound this kernel to one tile. The causal-bound kernel re-loads its
        # per-row length column inside its own row loop for the same reason.
        # ``row * E_local + expert``, both on device: the resolved row is the same
        # padding-aware index the gate/up limb gathered its tokens with, and the
        # expert is this block's scalar.
        affinity_sb = _gathered(
            affinity_bank,
            _affinity_rows(
                TILE_SIZE,
                _padding_resolved(
                    TILE_SIZE, _dense_column(row_index, TILE_SIZE, m0), pad_row
                ),
                expert,
                n_experts,
            ),
            TILE_SIZE,
            1,
            nl.float32,
        )
        for h_block in range(n_h_blocks):
            h0 = h_block * GATE_UP_SCALE_BLOCK
            acc = _gate_up_sbuf()
            for i_block in range(n_i_blocks):
                i0 = i_block * GATE_UP_SCALE_BLOCK
                psum = _gate_up_psum()
                inter_tile = nl.load(
                    intermediate_t[i0 : i0 + TILE_SIZE, m0 : m0 + TILE_SIZE],
                    dtype=nl.bfloat16,
                )
                w_tile = _gathered(
                    weight_bank,
                    _bank_rows(
                        TILE_SIZE,
                        expert,
                        ramp,
                        i_extent * n_h_blocks,
                        i0 * n_h_blocks + h_block,
                        n_h_blocks,
                    ),
                    TILE_SIZE,
                    GATE_UP_SCALE_BLOCK,
                    nl.bfloat16,
                )
                nisa.nc_matmul(
                    dst=psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                    stationary=inter_tile,
                    moving=w_tile,
                    accumulate=False,
                )
                flat = i_block * n_h_blocks + h_block
                if i_block == 0:
                    nisa.tensor_scalar(
                        dst=acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        data=psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                    )
                else:
                    nisa.scalar_tensor_tensor(
                        dst=acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        data=psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                        op1=nl.add,
                        operand1=acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                    )
            # The affinity is a per-partition column and the partition axis is the
            # token axis, so one value per token is exactly this operand's shape.
            scaled = _gate_up_sbuf()
            nisa.tensor_scalar(
                dst=scaled[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                data=acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                op0=nl.multiply,
                operand0=affinity_sb[0:TILE_SIZE, 0:1],
            )
            nl.store(
                out[m0 : m0 + TILE_SIZE, h0 : h0 + GATE_UP_SCALE_BLOCK],
                value=scaled[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
            )
    return out


# --------------------------------------------------------------------------- #
# The access patterns a dynamic loop body addresses its operands through.     #
# --------------------------------------------------------------------------- #
# Every destination is pre-filled with a marker, so an offset that ignored its block
# register leaves the marker behind instead of passing on a coincidence.
_PATTERN_BLOCKS = 2
_PATTERN_TILES = 2
_PATTERN_ROWS = _PATTERN_BLOCKS * _PATTERN_TILES * TILE_SIZE


@nki.jit
def _stored_by_block(source, marker):
    """One tile written to every destination through an access pattern."""
    out = nl.ndarray((_PATTERN_ROWS, TILE_SIZE), dtype=nl.float32, buffer=nl.shared_hbm)
    out_b = out.reshape((_PATTERN_BLOCKS, _PATTERN_TILES * TILE_SIZE, TILE_SIZE))
    stamp = nl.load(marker[0:TILE_SIZE, 0:TILE_SIZE])
    for step in range(_PATTERN_BLOCKS * _PATTERN_TILES):
        r0 = step * TILE_SIZE
        nl.store(out[r0 : r0 + TILE_SIZE, 0:TILE_SIZE], value=stamp)
    tile = nl.load(source[0:TILE_SIZE, 0:TILE_SIZE])

    def body(at_block):
        for step in range(_PATTERN_TILES):
            t0 = step * TILE_SIZE
            live._stored(
                out_b.ap(
                    pattern=[[TILE_SIZE, TILE_SIZE], [1, TILE_SIZE]],
                    offset=t0 * TILE_SIZE,
                    scalar_offset=at_block,
                    indirect_dim=0,
                ),
                tile,
            )

    nl.fori_loop(0, _PATTERN_BLOCKS, body)
    return out


@nki.jit
def _transposed_by_block(source, marker):
    """Every tile transposed onto partitions from a dynamic body, 16 rows per DMA."""
    out = nl.ndarray((_PATTERN_ROWS, TILE_SIZE), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    out_b = out.reshape((_PATTERN_BLOCKS, _PATTERN_TILES * TILE_SIZE, TILE_SIZE))
    staged = nl.ndarray((_PATTERN_ROWS, TILE_SIZE), dtype=nl.bfloat16, buffer=nl.private_hbm)
    staged_b = staged.reshape((_PATTERN_BLOCKS, _PATTERN_TILES * TILE_SIZE, TILE_SIZE))
    stamp = nl.load(marker[0:TILE_SIZE, 0:TILE_SIZE])
    # One tile at a time, because a load takes at most TILE_SIZE partitions.
    for step in range(_PATTERN_BLOCKS * _PATTERN_TILES):
        r0 = step * TILE_SIZE
        nl.store(staged[r0 : r0 + TILE_SIZE, 0:TILE_SIZE],
                 value=nl.load(source[r0 : r0 + TILE_SIZE, 0:TILE_SIZE]))
        nl.store(out[r0 : r0 + TILE_SIZE, 0:TILE_SIZE], value=stamp)

    def body(at_block):
        for step in range(_PATTERN_TILES):
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
                    scalar_offset=at_block,
                    indirect_dim=0,
                ),
                tile,
            )

    nl.fori_loop(0, _PATTERN_BLOCKS, body)
    return out


@nki.jit
def _broadcast_by_block(index_hbm, marker):
    """One int32 per block, replicated into every partition, addressed by the block."""
    out = nl.ndarray((_PATTERN_BLOCKS * TILE_SIZE, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    out_b = out.reshape((_PATTERN_BLOCKS, TILE_SIZE, 1))
    stamp = nl.load(marker[0:TILE_SIZE, 0:1])
    for step in range(_PATTERN_BLOCKS):
        r0 = step * TILE_SIZE
        nl.store(out[r0 : r0 + TILE_SIZE, 0:1], value=stamp)

    def body(at_block):
        live._stored(
            out_b.ap(
                pattern=[[1, TILE_SIZE], [1, 1]],
                offset=0,
                scalar_offset=at_block,
                indirect_dim=0,
            ),
            live._broadcast_row(index_hbm, 0, TILE_SIZE, at_block=at_block),
        )

    nl.fori_loop(0, _PATTERN_BLOCKS, body)
    return out


@nki.jit
def _cast_by_block(source, marker):
    """Every tile read through a pattern that converts fp32 to bfloat16 on the way."""
    out = nl.ndarray((_PATTERN_ROWS, TILE_SIZE), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    out_b = out.reshape((_PATTERN_BLOCKS, _PATTERN_TILES * TILE_SIZE, TILE_SIZE))
    source_b = source.reshape((_PATTERN_BLOCKS, _PATTERN_TILES * TILE_SIZE, TILE_SIZE))
    stamp = nl.load(marker[0:TILE_SIZE, 0:TILE_SIZE])
    for step in range(_PATTERN_BLOCKS * _PATTERN_TILES):
        r0 = step * TILE_SIZE
        nl.store(out[r0 : r0 + TILE_SIZE, 0:TILE_SIZE], value=stamp)

    def body(at_block):
        for step in range(_PATTERN_TILES):
            t0 = step * TILE_SIZE
            tile = live._loaded(
                source_b.ap(
                    pattern=[[TILE_SIZE, TILE_SIZE], [1, TILE_SIZE]],
                    offset=t0 * TILE_SIZE,
                    scalar_offset=at_block,
                    indirect_dim=0,
                ),
                TILE_SIZE,
                TILE_SIZE,
                nl.bfloat16,
            )
            live._stored(
                out_b.ap(
                    pattern=[[TILE_SIZE, TILE_SIZE], [1, TILE_SIZE]],
                    offset=t0 * TILE_SIZE,
                    scalar_offset=at_block,
                    indirect_dim=0,
                ),
                tile,
            )

    nl.fori_loop(0, _PATTERN_BLOCKS, body)
    return out


def _marker(width: int) -> torch.Tensor:
    """A value no case produces, so a destination nothing wrote reads as the marker."""
    return torch.full((TILE_SIZE, width), -7.0, dtype=torch.float32)


def _pattern_source() -> torch.Tensor:
    """The input these patterns read: distinct eighths per row, exact in bfloat16."""
    return _eighths(3, _PATTERN_ROWS, TILE_SIZE)


def test_the_access_pattern_store_writes_every_destination_the_slice_route_writes():
    """A block-offset store fills every destination a slice store fills."""
    source = _pattern_source()
    got = wrap_nki(_stored_by_block)(source, _marker(TILE_SIZE))
    expected = source[0:TILE_SIZE].repeat(_PATTERN_BLOCKS * _PATTERN_TILES, 1)
    stale = int(torch.eq(got, -7.0).sum().item())
    assert stale == 0
    assert torch.equal(got, expected)


def test_the_transpose_from_a_pattern_with_a_block_offset_moves_the_same_rows():
    """``nisa.dma_transpose`` over a block-offset pattern transposes every tile."""
    source = _pattern_source().to(torch.bfloat16)
    got = wrap_nki(_transposed_by_block)(source, _marker(TILE_SIZE).to(torch.bfloat16))
    tiles = source.reshape(_PATTERN_BLOCKS * _PATTERN_TILES, TILE_SIZE, TILE_SIZE)
    expected = tiles.transpose(1, 2).reshape(_PATTERN_ROWS, TILE_SIZE).contiguous()
    stale = int(torch.eq(got.to(torch.float32), -7.0).sum().item())
    assert stale == 0
    assert torch.equal(got, expected)


def test_the_broadcast_row_reads_its_own_block_and_not_the_first():
    """A block offset on a zero-stride pattern broadcasts that block's own value."""
    index = torch.tensor([[11], [23]], dtype=torch.int32)
    marker = torch.full((TILE_SIZE, 1), -7, dtype=torch.int32)
    got = wrap_nki(_broadcast_by_block)(index, marker)
    expected = index.repeat_interleave(TILE_SIZE, dim=0)
    stale = int(torch.eq(got, -7).sum().item())
    assert stale == 0
    assert torch.equal(got, expected)


def test_the_converting_access_pattern_load_reads_what_a_slice_load_reads():
    """A block-offset load converts fp32 to bfloat16 on the way in, as down needs."""
    source = _pattern_source()
    got = wrap_nki(_cast_by_block)(source, _marker(TILE_SIZE).to(torch.bfloat16))
    expected = source.to(torch.bfloat16)
    stale = int(torch.eq(got.to(torch.float32), -7.0).sum().item())
    assert stale == 0
    assert torch.equal(got, expected)


# --------------------------------------------------------------------------- #
# No partial block ever reaches a kernel.                                     #
# --------------------------------------------------------------------------- #
def test_a_block_that_is_not_whole_tiles_is_refused_before_any_loop_runs():
    """A partial block is refused, so the block loops cover every token tile."""
    case = dict(_G18)
    shape = _shape_of(case)
    operands = _gate_up_inputs(case)
    declared = (("not_a_tile_multiple", shape["block"] + 1, "not a positive multiple"),
                ("does_not_divide_positions", shape["block"] * 3, "not a multiple of block="),
                ("one_expert_per_block", shape["block"] * 2, "one expert per block"))
    refused = []
    for name, block, reason in declared:
        try:
            moe_gate_up_blockwise_fp8(
                operands["hidden"], operands["bank"], operands["scales"],
                operands["row_index"], operands["expert_index"], block,
            )
        except MoeBlockwiseFp8Error as refusal:
            if reason in str(refusal):
                refused.append(name)
    covered = case["blocks"] * case["tiles"] == shape["positions"] // TILE_SIZE
    assert refused == [name for name, _, _ in declared], f"refused for its own reason: {refused}"
    assert covered


# --------------------------------------------------------------------------- #
# The three kernels, against the reference implementation.                    #
# --------------------------------------------------------------------------- #
def _gate_up_inputs(case: dict, seed: int = 11) -> dict:
    """One case's gate/up operands: the routing column carries the padding ``-1``."""
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


def _iota() -> torch.Tensor:
    """The partition ramp the kernels take as an operand."""
    return torch.arange(TILE_SIZE, dtype=torch.int32).reshape(TILE_SIZE, 1)


def _reference_gate_up(operands: dict, case: dict) -> torch.Tensor:
    """The reference gate/up kernel on the operands the shipped one is handed."""
    shape = _shape_of(case)
    # The module's own column count, not one computed here: it counts a block per
    # contraction block as well, so a hand-built count is right only at one block.
    n_col_blocks = gate_up_kernel_scale_shape(shape["h_extent"], shape["i_extent"])[1]
    return wrap_nki(_reference_gate_up_kernel)(
        operands["hidden"].to(torch.bfloat16),
        operands["bank"].reshape(-1, GATE_UP_SCALE_BLOCK),
        operands["scales"].to(torch.float32).reshape(-1, n_col_blocks),
        operands["row_index"].to(torch.int32).reshape(-1, 1),
        operands["expert_index"].to(torch.int32).reshape(-1, 1),
        _iota(),
    )


@pytest.mark.parametrize("case", _ALL_CASES, ids=[c["name"] for c in _ALL_CASES])
def test_the_gate_up_kernel_matches_an_unrolled_reference(case):
    """The gate/up kernel equals the reference exactly, not within a tolerance."""
    operands = _gate_up_inputs(case)
    got = moe_gate_up_blockwise_fp8(
        operands["hidden"], operands["bank"], operands["scales"],
        operands["row_index"], operands["expert_index"], operands["block"],
    )
    expected = _reference_gate_up(operands, case)
    differing = int(torch.ne(got, expected).sum().item())
    assert differing == 0
    assert torch.equal(got, expected)


def _activation_inputs(case: dict, seed: int = 23) -> torch.Tensor:
    """The pre-activation tensor: ``[B, 2*I]`` fp32 eighths, both halves distinct."""
    shape = _shape_of(case)
    return _eighths(seed, shape["positions"], GATE_UP_FUSION * shape["i_extent"])


def _reference_swiglu(gate_up: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """The reference activation kernel on the operands the shipped one is handed."""
    return wrap_nki(_reference_swiglu_kernel)(gate_up.to(torch.float32), bounds)


def _bounds(limit: float | None) -> torch.Tensor:
    """The bound operand: the gate limit, ``-up``, ``up``, or one neutral column."""
    if limit is None:
        return torch.full((TILE_SIZE, 1), 0.0, dtype=torch.float32)
    return torch.cat([torch.full((TILE_SIZE, 1), value, dtype=torch.float32)
                      for value in (limit, -limit, limit)], dim=1)


@pytest.mark.parametrize("case", _ALL_CASES, ids=[c["name"] for c in _ALL_CASES])
def test_the_activation_kernel_matches_an_unrolled_reference(case):
    """The activation kernel equals the reference exactly, bounded and unbounded."""
    gate_up = _activation_inputs(case)
    for name, limit in (("bounded", _MODEL_SWIGLU_LIMIT), ("unbounded", None)):
        got = moe_swiglu_transposed(gate_up, limit, limit)
        expected = _reference_swiglu(gate_up, _bounds(limit))
        differing = int(torch.ne(got, expected).sum().item())
        assert differing == 0, f"the {name} activation differs from the reference"
        assert torch.equal(got, expected)


def test_the_activation_bound_changes_the_result():
    """Operands above the limit come out different bounded and unbounded."""
    case = _G18
    shape = _shape_of(case)
    over_the_limit = _eighths(31, shape["positions"], GATE_UP_FUSION * shape["i_extent"])
    over_the_limit = over_the_limit + 2.0 * _MODEL_SWIGLU_LIMIT
    bounded = moe_swiglu_transposed(over_the_limit, _MODEL_SWIGLU_LIMIT, _MODEL_SWIGLU_LIMIT)
    unbounded = moe_swiglu_transposed(over_the_limit, None, None)
    differing = int(torch.ne(bounded, unbounded).sum().item())
    assert differing > 0


def _down_inputs(case: dict, seed: int = 41) -> dict:
    """One case's down operands, with the flat affinity bank the mapping emits."""
    shape = _shape_of(case)
    generator = torch.Generator().manual_seed(seed)
    intermediate_t = _eighths(seed, shape["i_extent"], shape["positions"])
    bank = _eighths(seed + 1, _EXPERTS, shape["i_extent"], shape["h_extent"]).to(_FP8)
    exponents = torch.randint(
        -2, 2,
        (_EXPERTS, shape["i_extent"] // GATE_UP_SCALE_BLOCK,
         shape["h_extent"] // GATE_UP_SCALE_BLOCK),
        generator=generator,
    ).to(torch.float32)
    grid = torch.pow(2.0, exponents)
    scales = torch.stack([
        to_down_kernel_scale_operand(grid[e], shape["i_extent"], shape["h_extent"])
        for e in range(_EXPERTS)
    ])
    affinity = _eighths(seed + 2, (_TOKENS + 1) * _EXPERTS, 1)
    affinity[_TOKENS * _EXPERTS :] = 0.0
    row_index = torch.randint(
        -1, _TOKENS, (shape["positions"], 1), generator=generator, dtype=torch.int32
    )
    expert_index = torch.randint(
        0, _EXPERTS, (case["blocks"], 1), generator=generator, dtype=torch.int32
    )
    return {"intermediate_t": intermediate_t, "bank": bank, "scales": scales,
            "affinity": affinity, "row_index": row_index, "expert_index": expert_index,
            "block": shape["block"]}


def _reference_down(operands: dict, case: dict) -> torch.Tensor:
    """The reference down kernel on the operands the shipped one is handed."""
    shape = _shape_of(case)
    expected = down_kernel_scale_shape(shape["i_extent"], shape["h_extent"])
    return wrap_nki(_reference_down_kernel)(
        operands["intermediate_t"].to(torch.float32),
        operands["bank"].reshape(-1, GATE_UP_SCALE_BLOCK),
        operands["scales"].to(torch.float32).reshape(-1, expected[1]),
        operands["affinity"].to(torch.float32).reshape(-1, 1),
        operands["row_index"].to(torch.int32).reshape(-1, 1),
        operands["expert_index"].to(torch.int32).reshape(-1, 1),
        _iota(),
    )


@pytest.mark.parametrize("case", _ALL_CASES, ids=[c["name"] for c in _ALL_CASES])
def test_the_down_kernel_matches_an_unrolled_reference(case):
    """The down kernel equals the reference exactly, not within a tolerance."""
    operands = _down_inputs(case)
    got = moe_down_blockwise_fp8(
        operands["intermediate_t"], operands["bank"], operands["scales"],
        operands["affinity"], operands["row_index"], operands["expert_index"],
        operands["block"], _TOKENS,
    )
    expected = _reference_down(operands, case)
    differing = int(torch.ne(got, expected).sum().item())
    assert differing == 0
    assert torch.equal(got, expected)


# --------------------------------------------------------------------------- #
# The two-program launch: which program walks which blocks.                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("count", (1, 2, 3, 81, 162))
def test_each_program_range_covers_the_blocks_in_equal_trips(count):
    """Two programs run equal trips that cover every block and share at most one."""
    ranges = [live._program_block_range("moe_gate_up_blockwise_fp8", count, 2, prg_id)
              for prg_id in (0, 1)]
    covered = [at for first, end in ranges for at in range(first, end)]
    lengths = [end - first for first, end in ranges]
    assert lengths[0] == lengths[1] == -(-count // 2)
    assert sorted(set(covered)) == list(range(count))
    assert len(covered) - count == count % 2
    if count < 2:
        assert ranges == [(0, count), (0, count)]


@pytest.mark.parametrize("count", (1, 3))
def test_a_one_program_launch_is_refused_before_any_loop_runs(count):
    """The range refuses a one-program launch by kernel name and launch degree."""
    with pytest.raises(
        AssertionError,
        match="moe_gate_up_blockwise_fp8: traced with 1 programs, the kernel wants 2",
    ):
        live._program_block_range("moe_gate_up_blockwise_fp8", count, 1, 0)


def test_a_shipped_kernel_launched_with_one_program_is_refused_by_the_simulator():
    """A real launch at degree one raises rather than running the whole range."""
    gate_up = _activation_inputs(_G1).to(torch.float32)
    with pytest.raises(Exception) as caught:
        wrap_nki(live.moe_swiglu_transposed_kernel)[1](gate_up, _bounds(_MODEL_SWIGLU_LIMIT))
    chain, error = [], caught.value
    while error is not None:
        chain.append(error)
        error = error.__cause__ or error.__context__
    message = " ".join(str(error) for error in chain)
    assertion_error = any(isinstance(error, AssertionError) for error in chain)
    assert assertion_error
    assert "moe_swiglu_transposed: traced with 1 programs, the kernel wants 2" in message


@nki.jit
def _mark_blocks_by_program(blocks):
    """Mark, in this program's own rows, every block its range covers."""
    count = blocks.shape[0]
    out = nl.ndarray((2 * count, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    _ndim, n_prgs, prg_id = get_verified_program_sharding_info(
        "test_mark_blocks_by_program", (0, 1), 2
    )
    first, end = live._program_block_range("test_mark_blocks_by_program", count, n_prgs, prg_id)
    marks = nl.ndarray((count, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=marks, value=0.0)
    for at_block in range(first, end):
        marks[at_block, 0] = 1.0
    nl.store(out[prg_id * count : (prg_id + 1) * count, :], value=marks)
    return out


@pytest.mark.parametrize(
    "count, expected",
    ((2, [1, 0, 0, 1]), (3, [1, 1, 0, 0, 1, 1]), (1, [1, 1])),
    ids=["two-blocks-one-each", "three-blocks-two-each-sharing-the-middle",
         "one-block-run-by-both"],
)
def test_two_programs_mark_the_blocks_their_ranges_cover(count, expected):
    """Inside a two-program launch, each program marks the blocks its range covers.

    Program 0's rows come first, program 1's after, so a block both ranges cover is
    written twice; the reference comparisons at G1 and G3 show both writes agree.
    """
    marks = wrap_nki(_mark_blocks_by_program)[2](torch.zeros(count, 1, dtype=torch.float32))
    got = [int(v) for v in marks.to(torch.float32).flatten().tolist()]
    assert got == expected
