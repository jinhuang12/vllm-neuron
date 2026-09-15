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

import pathlib

import pytest
import torch

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from vllm_neuron.functional.moe import moe_blockwise_fp8 as live
from test.vllm_neuron.functional.dma_transpose_census import census, tile_rows
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

#: The constants the frozen copies read, RETYPED on purpose: a frozen reference that
#: imported them would follow the live module if one of them ever moved.
TILE_SIZE = 128
GATE_UP_SCALE_BLOCK = 128
GATE_UP_H_TILES_PER_BLOCK = 1
GATE_UP_FUSION = 2
DGE_TRANSPOSE_ROWS = 16
ROW_ALIGN = 16
_SWIGLU_BOUND_COLUMNS = 3

#: The activation bound the model passes, RETYPED from the MODEL's own copy
#: (``glm5_next/config.py`` ``swiglu_limit``, handed to both halves at
#: ``glm5_next/model_fp8.py`` in the routed MoE path), never read off the kernel.
_MODEL_SWIGLU_LIMIT = 10.0

#: The source rows per DMA the device generates its own descriptors for, RETYPED.
_DGE_ROWS = 16

#: The destination line the runtime requires, RETYPED.
_LINE = 32

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
_G16 = {"name": "G16", "blocks": 2, "tiles": 8, "h_blocks": 1, "i_blocks": 1}
_GN = {"name": "GN", "blocks": 2, "tiles": 9, "h_blocks": 2, "i_blocks": 2}
#: Two more geometries for the two-program launch: ONE block of one tile, which both
#: programs run whole and write identically, and THREE blocks of three tiles, where the
#: programs share the middle block and the middle token tile.
_G1 = {"name": "G1", "blocks": 1, "tiles": 1, "h_blocks": 1, "i_blocks": 1}
_G3 = {"name": "G3", "blocks": 3, "tiles": 3, "h_blocks": 1, "i_blocks": 1}
_ALL_CASES = (_G18, _G16, _GN, _G1, _G3)


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
# FROZEN AT THE BASE COMMIT: the helpers and the three kernels as they stood   #
# before the roll. Their bodies are the base's own; only the kernel names and   #
# the docstrings differ, so the reference cannot drift with the live module.    #
# --------------------------------------------------------------------------- #
def _padded(width: int) -> int:
    """Frozen at the base commit: _padded."""
    return ((width + ROW_ALIGN - 1) // ROW_ALIGN) * ROW_ALIGN


def _column(rows: int):
    """Frozen at the base commit: _column."""
    return nl.ndarray((rows, ROW_ALIGN), dtype=nl.int32, buffer=nl.sbuf)[:, 0:1]


def _row_iota(iota_hbm, rows: int):
    """Frozen at the base commit: _row_iota."""
    tile = _column(rows)
    nisa.dma_copy(dst=tile, src=iota_hbm.ap(pattern=[[1, rows], [1, 1]], offset=0))
    return tile


def _dense_column(hbm, rows: int, offset: int):
    """Frozen at the base commit: _dense_column."""
    tile = _column(rows)
    nisa.dma_copy(dst=tile, src=hbm.ap(pattern=[[1, rows], [1, 1]], offset=offset))
    return tile


def _broadcast_row(hbm, row: int, rows: int):
    """Frozen at the base commit: _broadcast_row."""
    tile = _column(rows)
    nisa.dma_copy(dst=tile, src=hbm.ap(pattern=[[0, rows], [1, 1]], offset=row))
    return tile


def _padding_resolved(rows: int, positions, pad_row: int):
    """Frozen at the base commit: _padding_resolved."""
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
    """Frozen at the base commit: _bank_rows."""
    base = _column(rows)
    nisa.tensor_scalar(dst=base, data=expert, op0=nl.multiply, operand0=stride)
    nisa.tensor_scalar(dst=base, data=base, op0=nl.add, operand0=offset)
    ramp = _column(rows)
    nisa.tensor_scalar(dst=ramp, data=iota, op0=nl.multiply, operand0=step)
    index = _column(rows)
    nisa.tensor_tensor(dst=index, data1=base, data2=ramp, op=nl.add)
    return index


def _affinity_rows(rows: int, resolved, expert, n_experts: int):
    """Frozen at the base commit: _affinity_rows."""
    scaled = _column(rows)
    nisa.tensor_scalar(dst=scaled, data=resolved, op0=nl.multiply, operand0=n_experts)
    index = _column(rows)
    nisa.tensor_tensor(dst=index, data1=scaled, data2=expert, op=nl.add)
    return index


def _gathered(hbm, index, rows: int, width: int, dtype):
    """Frozen at the base commit: _gathered."""
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
    """Frozen at the base commit: _transpose_rows."""
    for r0 in range(0, rows, DGE_TRANSPOSE_ROWS):
        n = min(DGE_TRANSPOSE_ROWS, rows - r0)
        nisa.dma_transpose(
            dst=dst[:, r0 : r0 + n],
            src=src_hbm.ap(
                pattern=[[row_stride, n], [1, width]], offset=offset + r0 * row_stride
            ),
        )


def _gate_up_sbuf(rows: int = TILE_SIZE, cols: int = GATE_UP_SCALE_BLOCK):
    """Frozen at the base commit: _gate_up_sbuf."""
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)


def _gate_up_psum(rows: int = TILE_SIZE, cols: int = GATE_UP_SCALE_BLOCK):
    """Frozen at the base commit: _gate_up_psum."""
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)


@nki.jit
def _landed_gate_up(
    hidden, weight_bank, scale_bank, row_index, expert_index, iota
):
    """Frozen at the base commit: moe_gate_up_blockwise_fp8_kernel."""
    positions = row_index.shape[0]
    h_extent = hidden.shape[1]
    n_h_blocks = h_extent // GATE_UP_SCALE_BLOCK
    # ONE SCALE COLUMN PER (H BLOCK, FUSED COLUMN BLOCK), which the seam pins against
    # :func:`gate_up_kernel_scale_shape`, so the width over the H blocks IS the fused
    # column count in blocks -- and the fusion is even, which the seam also refuses.
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
def _landed_swiglu(gate_up, bounds):
    """Frozen at the base commit: moe_swiglu_transposed_kernel."""
    tokens, fused_cols = gate_up.shape
    i_extent = fused_cols // GATE_UP_FUSION
    # A ONE-COLUMN OPERAND IS AN UNBOUNDED CONFIGURATION, and then no bound
    # instruction is emitted at all -- the trace-time elision the landed form had
    # when the two limits arrived as ``None``.
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
            # THE BOUNDS, on the tiles just loaded. ``maximum`` and ``minimum`` are
            # the closed-form pair this campaign already uses for a bound
            # (``dsa/causal_fill.py``), and the two-scalar chain is the landed shape
            # of that call. A ``None`` limit removes its own line at trace time, so
            # an unbounded configuration emits no instruction rather than a bound at
            # infinity. A configured limit is a per-partition column for the reason
            # :func:`_swiglu_bound_operand` records.
            if bounded_config:
                bounded = _gate_up_sbuf()
                nisa.tensor_scalar(
                    dst=bounded,
                    data=gate,
                    op0=nl.minimum,
                    operand0=bounds_sb[0:TILE_SIZE, 0:1],
                )
                gate = bounded
                # TWO CALLS AND NOT ONE FUSED PAIR. A ``[P, 1]`` column is the
                # landed form for ``operand0`` in this package and no landed call
                # passes one as ``operand1``; only the compiler's verifier could settle
                # that, and a maximum followed by a minimum is the same closed form.
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
def _landed_down(
    intermediate_t,
    weight_bank,
    scale_bank,
    affinity_bank,
    row_index,
    expert_index,
    iota,
):
    """Frozen at the base commit: moe_down_blockwise_fp8_kernel."""
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
        # LOADED PER TOKEN TILE, never hoisted out of this loop: a token is a
        # PARTITION, the partition axis serves 128 of them, and a whole-column load
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
                    scalar_offset=at_block,
                    indirect_dim=0,
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
    staged_b = staged.reshape((_FORM_BLOCKS, _FORM_TILES * TILE_SIZE, TILE_SIZE))
    stamp = nl.load(marker[0:TILE_SIZE, 0:TILE_SIZE])
    # ONE TILE AT A TIME, because a load takes at most TILE_SIZE partitions.
    for step in range(_FORM_BLOCKS * _FORM_TILES):
        r0 = step * TILE_SIZE
        nl.store(staged[r0 : r0 + TILE_SIZE, 0:TILE_SIZE],
                 value=nl.load(source[r0 : r0 + TILE_SIZE, 0:TILE_SIZE]))
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
                    scalar_offset=at_block,
                    indirect_dim=0,
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
                scalar_offset=at_block,
                indirect_dim=0,
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
    _emit("COVERING_IDENTITY", refused=len(refused), by_cause=tuple(refused),
          blocks=case["blocks"], tiles_per_block=case["tiles"],
          token_tiles=shape["positions"] // TILE_SIZE, covered=covered)
    assert refused == [name for name, _, _ in declared], f"refused for its own reason: {refused}"
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


def _iota() -> torch.Tensor:
    """The partition ramp the kernels take as an operand."""
    return torch.arange(TILE_SIZE, dtype=torch.int32).reshape(TILE_SIZE, 1)


def _frozen_gate_up(operands: dict, case: dict) -> torch.Tensor:
    """The frozen gate/up kernel on the operands the live seam hands the live one."""
    shape = _shape_of(case)
    # THE SEAM'S OWN COLUMN COUNT, not one computed here: it counts a block per
    # contraction block as well, so a hand-built count is right only at one block.
    n_col_blocks = gate_up_kernel_scale_shape(shape["h_extent"], shape["i_extent"])[1]
    return wrap_nki(_landed_gate_up)(
        operands["hidden"].to(torch.bfloat16),
        operands["bank"].reshape(-1, GATE_UP_SCALE_BLOCK),
        operands["scales"].to(torch.float32).reshape(-1, n_col_blocks),
        operands["row_index"].to(torch.int32).reshape(-1, 1),
        operands["expert_index"].to(torch.int32).reshape(-1, 1),
        _iota(),
    )


@pytest.mark.parametrize("case", _ALL_CASES, ids=[c["name"] for c in _ALL_CASES])
def test_the_gate_up_kernel_is_bit_identical_to_the_frozen_copy(case):
    """The staging loop and the projection nest, both rolled, at all three geometries."""
    operands = _gate_up_inputs(case)
    got = moe_gate_up_blockwise_fp8(
        operands["hidden"], operands["bank"], operands["scales"],
        operands["row_index"], operands["expert_index"], operands["block"],
    )
    want = _frozen_gate_up(operands, case)
    differing = int(torch.ne(got, want).sum().item())
    _emit("GATE_UP_IDENTITY", case=case["name"], token_tiles=_shape_of(case)["token_tiles"],
          h_blocks=case["h_blocks"], i_blocks=case["i_blocks"], differing=differing,
          equal=torch.equal(got, want), shape=tuple(got.shape))
    assert differing == 0
    assert torch.equal(got, want)


def _activation_inputs(case: dict, seed: int = 23) -> torch.Tensor:
    """The pre-activation tensor: ``[B, 2*I]`` fp32 eighths, both halves distinct."""
    shape = _shape_of(case)
    return _eighths(seed, shape["positions"], GATE_UP_FUSION * shape["i_extent"])


def _frozen_swiglu(gate_up: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """The frozen activation kernel on the operands the live seam builds."""
    return wrap_nki(_landed_swiglu)(gate_up.to(torch.float32), bounds)


def _bounds(limit: float | None) -> torch.Tensor:
    """The bound operand the seam builds: gate, ``-up``, ``up``, or one neutral column."""
    if limit is None:
        return torch.full((TILE_SIZE, 1), 0.0, dtype=torch.float32)
    return torch.cat([torch.full((TILE_SIZE, 1), value, dtype=torch.float32)
                      for value in (limit, -limit, limit)], dim=1)


@pytest.mark.parametrize("case", _ALL_CASES, ids=[c["name"] for c in _ALL_CASES])
def test_the_activation_kernel_is_bit_identical_to_the_frozen_copy(case):
    """Both bound configurations, and both counts of intermediate block the store offsets by."""
    gate_up = _activation_inputs(case)
    shape = _shape_of(case)
    for name, limit in (("bounded", _MODEL_SWIGLU_LIMIT), ("unbounded", None)):
        got = moe_swiglu_transposed(gate_up, limit, limit)
        want = _frozen_swiglu(gate_up, _bounds(limit))
        differing = int(torch.ne(got, want).sum().item())
        _emit("ACTIVATION_IDENTITY", case=case["name"], bounds=name,
              token_tiles=shape["token_tiles"], i_blocks=case["i_blocks"],
              tokens=shape["positions"], differing=differing,
              equal=torch.equal(got, want), shape=tuple(got.shape))
        assert differing == 0
        assert torch.equal(got, want)


def test_the_activation_bound_the_model_passes_changes_the_result():
    """The failing control for the bound: an unbounded arm must not equal a bounded one."""
    case = _G18
    shape = _shape_of(case)
    over_the_limit = _eighths(31, shape["positions"], GATE_UP_FUSION * shape["i_extent"])
    over_the_limit = over_the_limit + 2.0 * _MODEL_SWIGLU_LIMIT
    bounded = moe_swiglu_transposed(over_the_limit, _MODEL_SWIGLU_LIMIT, _MODEL_SWIGLU_LIMIT)
    unbounded = moe_swiglu_transposed(over_the_limit, None, None)
    differing = int(torch.ne(bounded, unbounded).sum().item())
    _emit("ACTIVATION_BOUND_CONTROL", limit=_MODEL_SWIGLU_LIMIT, differing=differing,
          equal=torch.equal(bounded, unbounded))
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


def _frozen_down(operands: dict, case: dict) -> torch.Tensor:
    """The frozen down kernel on the operands the live seam hands the live one."""
    shape = _shape_of(case)
    expected = down_kernel_scale_shape(shape["i_extent"], shape["h_extent"])
    return wrap_nki(_landed_down)(
        operands["intermediate_t"].to(torch.float32),
        operands["bank"].reshape(-1, GATE_UP_SCALE_BLOCK),
        operands["scales"].to(torch.float32).reshape(-1, expected[1]),
        operands["affinity"].to(torch.float32).reshape(-1, 1),
        operands["row_index"].to(torch.int32).reshape(-1, 1),
        operands["expert_index"].to(torch.int32).reshape(-1, 1),
        _iota(),
    )


@pytest.mark.parametrize("case", _ALL_CASES, ids=[c["name"] for c in _ALL_CASES])
def test_the_down_kernel_is_bit_identical_to_the_frozen_copy(case):
    """The free-axis window and the output store, at one and at two blocks on each axis."""
    operands = _down_inputs(case)
    shape = _shape_of(case)
    got = moe_down_blockwise_fp8(
        operands["intermediate_t"], operands["bank"], operands["scales"],
        operands["affinity"], operands["row_index"], operands["expert_index"],
        operands["block"], _TOKENS,
    )
    want = _frozen_down(operands, case)
    differing = int(torch.ne(got, want).sum().item())
    _emit("DOWN_IDENTITY", case=case["name"], token_tiles=shape["token_tiles"],
          h_blocks=case["h_blocks"], i_blocks=case["i_blocks"],
          positions=shape["positions"], h_extent=shape["h_extent"],
          store_contiguous=shape["h_extent"] == GATE_UP_SCALE_BLOCK,
          differing=differing, equal=torch.equal(got, want), shape=tuple(got.shape))
    assert differing == 0
    assert torch.equal(got, want)


# --------------------------------------------------------------------------- #
# What the roll is for: the per-block operands are gathered once per block.   #
# --------------------------------------------------------------------------- #
def _counting(module, name: str, log: list):
    """Replace ``module.name`` with a wrapper that logs its call, and return the original."""
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        log.append((args, kwargs))
        return original(*args, **kwargs)

    setattr(module, name, wrapper)
    return original


def test_the_expert_operands_are_gathered_once_per_block():
    """The broadcast and the scale gather leave the tile loop, which is the roll's own saving."""
    case = _G18
    shape = _shape_of(case)
    operands = _gate_up_inputs(case)
    broadcasts: list = []
    rows: list = []
    original_broadcast = _counting(live, "_broadcast_row", broadcasts)
    original_rows = _counting(live, "_bank_rows", rows)
    try:
        moe_gate_up_blockwise_fp8(
            operands["hidden"], operands["bank"], operands["scales"],
            operands["row_index"], operands["expert_index"], operands["block"],
        )
    finally:
        setattr(live, "_broadcast_row", original_broadcast)
        setattr(live, "_bank_rows", original_rows)
    scale_gathers = [call for call in rows
                     if call[0][3] == TILE_SIZE and call[0][4] == 0 and call[0][5] == 1]
    token_tiles = shape["positions"] // TILE_SIZE
    _emit("PER_BLOCK_GATHERS", blocks=case["blocks"], token_tiles=token_tiles,
          broadcasts=len(broadcasts), scale_gathers=len(scale_gathers),
          per_tile_before=token_tiles)
    assert len(broadcasts) == case["blocks"]
    assert len(scale_gathers) == case["blocks"]
    assert case["blocks"] < token_tiles


# --------------------------------------------------------------------------- #
# The transposes and the SBUF tiles, through the shared census.               #
# --------------------------------------------------------------------------- #
#: The trace-time values the transpose sites are read at: one case's own extents, so the
#: census reads the arithmetic the rolled body will trace.
_CENSUS_AT = {"positions": 2304, "h_extent": 128, "n_h_blocks": 1, "n_col_blocks": 2,
              "i_extent": 128, "n_i_blocks": 1, "block": 1152, "tiles_per_block": 9,
              "n_blocks": 2, "tokens": 2304, "fused_cols": 256}

#: The names the module supplies to its own size and loop expressions, read off the module.
_MODULE_ARITHMETIC = ("DGE_TRANSPOSE_ROWS", "TILE_SIZE", "GATE_UP_SCALE_BLOCK",
                      "GATE_UP_H_TILES_PER_BLOCK", "GATE_UP_FUSION", "ROW_ALIGN", "_padded")

#: The element width the seam hands the kernel: it casts the routed rows to bfloat16.
_WIDTHS = (2,)


def _arithmetic() -> dict:
    """The module's own arithmetic, by name; a module without a name leaves it out."""
    return {name: getattr(live, name) for name in _MODULE_ARITHMETIC if hasattr(live, name)}


def test_every_transpose_still_moves_sixteen_two_byte_rows_onto_a_readable_line():
    """The rolled body keeps every property the shared census reads, by count and by cause."""
    source = pathlib.Path(live.__file__).read_text(encoding="utf-8")
    sites = census(source, _CENSUS_AT, _arithmetic(), _WIDTHS)
    unreadable = tuple(s.source for s in sites if s.misaligned is None)
    misaligned = tuple(s.source for s in sites if s.misaligned)
    host_shaped = tuple(s.source for s in sites if s.host_shaped(_DGE_ROWS))
    _emit("TRANSPOSE_CENSUS", transposes=len(sites), unreadable=len(unreadable),
          misaligned=len(misaligned), host_shaped=len(host_shaped),
          by_source=tuple(sorted({s.source for s in sites})), step=_DGE_ROWS, line=_LINE)
    assert sites, "the census read no transpose site"
    assert len(unreadable) == 0, f"destinations the arithmetic cannot place: {unreadable}"
    assert len(host_shaped) == 0, f"transposes the device cannot shape: {host_shaped}"
    assert len(misaligned) == 0, f"destinations off a 32-byte line: {misaligned}"


def test_every_sbuf_tile_the_rolled_bodies_declare_keeps_whole_lines():
    """A rolled body allocates the same tiles; every one still ends on a 32-byte line."""
    source = pathlib.Path(live.__file__).read_text(encoding="utf-8")
    tiles = tile_rows(source, _CENSUS_AT, _arithmetic(), _WIDTHS)
    # A tile inside a helper takes its width from a parameter, which these extents cannot
    # place, so the count that cannot be placed is a reading and the placed ones are graded.
    placed = tuple((name, row) for _, name, row in tiles if row is not None)
    unplaced = tuple(name for _, name, row in tiles if row is None)
    ragged = tuple(name for name, row in placed if row % _LINE)
    _emit("SBUF_TILE_LINES", tiles=len(tiles), placed=len(placed), unplaced=len(unplaced),
          ragged=len(ragged), by_name=ragged, line=_LINE)
    assert placed, "the census placed no SBUF tile"
    assert len(ragged) == 0, f"tiles whose partition row is not whole lines: {ragged}"


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


# --------------------------------------------------------------------------- #
# The two-program launch: which program walks which blocks.                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("count", (1, 2, 3, 81, 162))
def test_each_program_range_covers_the_blocks_in_equal_trips(count):
    """Two programs run equal trips that together cover every block and share at most one."""
    ranges = [live._program_block_range("range-item", count, 2, prg_id) for prg_id in (0, 1)]
    covered = [at for first, end in ranges for at in range(first, end)]
    lengths = [end - first for first, end in ranges]
    _emit("PROGRAM_RANGE", blocks=count, ranges=ranges, lengths=lengths)
    assert lengths[0] == lengths[1] == -(-count // 2)
    assert sorted(set(covered)) == list(range(count))
    assert len(covered) - count == count % 2
    if count < 2:
        assert ranges == [(0, count), (0, count)]


@pytest.mark.parametrize("count", (1, 3))
def test_a_one_program_launch_is_refused_before_any_loop_runs(count):
    """A launch of one program cannot pass as a split: the range refuses it by name and degree."""
    with pytest.raises(ValueError, match="control-item: traced with 1 programs, the kernel wants 2"):
        live._program_block_range("control-item", count, 1, 0)


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
    "count, want",
    ((2, [1, 0, 0, 1]), (3, [1, 1, 0, 0, 1, 1]), (1, [1, 1])),
    ids=["two-blocks-one-each", "three-blocks-two-each-sharing-the-middle", "one-block-run-by-both"],
)
def test_two_programs_mark_the_blocks_their_ranges_cover(count, want):
    """Read from inside a two-program launch: program 0's rows first, program 1's after.

    A block in both programs' rows is written twice. The identity items at G1 and G3
    are what prove those two writes carry the same values.
    """
    marks = wrap_nki(_mark_blocks_by_program)[2](torch.zeros(count, 1, dtype=torch.float32))
    got = [int(v) for v in marks.to(torch.float32).flatten().tolist()]
    _emit("PROGRAM_MARKS", blocks=count, marks=got, want=want)
    assert got == want
