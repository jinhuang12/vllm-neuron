# SPDX-License-Identifier: Apache-2.0
"""Sum each token's expert rows of the fused MoE emission, as an NKI kernel.

Per token::

    out[t] = sum_j valid[t, j] * contribution[index[t, j]]      # fp32, j in slot order

The fused expert kernel emits ``[blocks * rows, H]`` fp32 rows sized for the worst
case, so most rows are padding. Gathering each token's ``K`` rows reads at most
``T * K`` rows, where a scatter-add over the emission reads every row and runs as a
serial GpSimd accumulate.

The kernel puts 128 tokens on the partitions and keeps an fp32 accumulator and a
gather buffer per token tile and program's part of a hidden piece (a unit).
Slot 0 gathers every token's row (an invalid slot names a finite row; the caller
points it at row 0), and the scalar engine starts the accumulator as ``row * valid + 0``.
The vector engine adds the later slots, in one device loop per token tile that ends
after the last slot some token of the tile fills. A rank holds a few of the experts,
so a token fills few of its ``K`` slots, and ``combine_slots`` fills them from slot 0,
so the loop of a tile runs only the slots its tokens fill. In a later slot, an invalid
slot gets an index past the last row, so its gather moves no bytes for that token and
leaves the token's previous row in the buffer, which the vector engine weighs by zero.
The tile's first token reads its named row instead, so no gather skips every row,
whatever the slot order. The adds run in slot order from zero, the order of
``token_gather_combine_torch``.

A two-program grid splits each hidden piece's columns across the two PNCs of one LNC2
core. Every program takes every token tile, so all of them run the same loop trips (a
device loop under LNC2 passes a core barrier at every trip).
"""

from __future__ import annotations

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.kernel_assert import kernel_assert

from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: Tokens per tile: the partition count.
TOKEN_TILE = 128

#: Hidden elements per row piece. The programs split each piece's columns, so each of the
#: two programs of an LNC2 core gathers 8 KB per partition, up to 1 MB per gather.
HIDDEN_TILE = 4096

#: SBUF bytes per partition that one batch's accumulators and gather buffers take.
_BATCH_BYTES = 128 * 1024

#: Fixed-address loads and the output stores take the hardware descriptor generator on the
#: sync engine's queue, so the GpSimd engine generates only the gathers' software descriptors.
_HWDGE = nisa.dge_mode.hwdge
_QUEUE = nisa.engine.sync

#: int32 elements per 32-byte line. Small tiles are whole lines wide, so every tile
#: placed after one stays line-aligned.
_LINE = 8


def _lines(columns: int) -> int:
    """Columns of a small tile holding ``columns`` 4-byte entries: whole lines."""
    return -(-columns // _LINE) * _LINE


def _register(src):
    """A register loaded from the ``[1, 1]`` SBUF integer ``src``."""
    register = nisa.register_alloc()
    nisa.register_load(dst=register, src=src)
    return register


def _tile_tables(index_hbm, valid_hbm, t0: int):
    """Tokens ``t0 .. t0 + 127``'s ``(index, weights)``: whole ``[128, lines]`` int32 and
    fp32 tensors whose first ``K`` columns hold the slots, so a device loop reads a column
    by offset."""
    slots = index_hbm.shape[1]
    lines = _lines(slots)
    index = nl.ndarray((TOKEN_TILE, lines), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=index[:, 0:slots],
                  src=index_hbm.ap(pattern=[[slots, TOKEN_TILE], [1, slots]], offset=t0 * slots),
                  dge_mode=_HWDGE, engine=_QUEUE)
    weights = nl.ndarray((TOKEN_TILE, lines), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=weights[:, 0:slots],
                  src=valid_hbm.ap(pattern=[[slots, TOKEN_TILE], [1, slots]], offset=t0 * slots),
                  dge_mode=_HWDGE, engine=_QUEUE)
    return index, weights


def _unit_rows(index, weights, slots: int, piece: int, pieces: int, banked_rows: int):
    """A tile's rows of banked piece ``piece`` (of ``pieces`` per row): a whole ``[128, lines]``
    int32 tensor.

    Column ``j`` holds ``index * pieces + piece`` for slot 0, for a valid slot, and on
    partition 0 (the tile's first token, so a gather moves at least one row); an invalid
    later slot of any other token is past the last row.
    """
    lines = index.shape[1]
    # past = (1 - valid) * banked_rows + piece, then slot 0 and partition 0 keep their rows.
    past = nl.ndarray((TOKEN_TILE, lines), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=past[:, 0:slots], data=weights[:, 0:slots], op0=nl.multiply,
                       operand0=-float(banked_rows), op1=nl.add,
                       operand1=float(banked_rows + piece))
    nisa.memset(dst=past[:, 0:1], value=piece)
    nisa.memset(dst=past[0:1, 0:slots], value=piece)
    rows = nl.ndarray((TOKEN_TILE, lines), dtype=nl.int32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=rows[:, 0:slots], data=index[:, 0:slots], op0=nl.multiply,
                              operand0=float(pieces), op1=nl.add, operand1=past[:, 0:slots])
    return rows


def _slot_groups(table, tiles: int, slots: int):
    """``table`` (``[1, tiles * lines]`` on partition 0) as ``[1, tiles, K]``: per tile, the
    first ``K`` columns of its line group."""
    return table.ap(pattern=[[table.shape[1], 1], [_lines(slots), tiles], [1, slots]],
                    offset=0)


def _tile_ends(counts, scores, tiles: int, slots: int):
    """``[1, tiles]`` int32 on partition 0: per tile, one past the last slot that some token
    of the tile fills, and at least 1.

    ``counts`` holds each tile's tokens per slot, ``scores`` the slot numbers plus one; a
    filled slot scores its number plus one, and the end is the greatest score.
    """
    filled = nl.ndarray((1, tiles * _lines(slots)), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=_slot_groups(filled, tiles, slots),
                              data=_slot_groups(counts, tiles, slots), op0=nl.minimum,
                              operand0=1.0, op1=nl.multiply,
                              operand1=_slot_groups(scores, tiles, slots))
    best = nl.ndarray((1, _lines(tiles)), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=best[:, 0:tiles], data=_slot_groups(filled, tiles, slots),
                       op=nl.maximum, axis=2)
    ends = nl.ndarray((1, _lines(tiles)), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ends[:, 0:tiles], data=best[:, 0:tiles], op0=nl.maximum,
                       operand0=1.0)
    return ends


def _column(dst, table, slot):
    """Copy column ``slot`` (a register) of the whole ``[128, n]`` SBUF ``table`` to ``dst``.

    The scalar engine copies: a dynamic SBUF read runs on the vector or scalar engine,
    and the vector engine carries the adds.
    """
    nisa.tensor_copy(dst=dst, src=table.ap(pattern=[[table.shape[1], TOKEN_TILE], [1, 1]],
                                           offset=0, scalar_offset=slot, indirect_dim=1),
                     engine=nisa.engine.scalar)


def _gather(picked, banked, rows, skip: bool):
    """Gather ``banked`` rows ``rows`` ([128, 1] int32) into ``picked``; ``skip`` drops a row
    past the last one. Software descriptors: a per-partition row offset has no hwdge form."""
    tile_h = banked.shape[1]
    src = banked.ap(pattern=[[tile_h, TOKEN_TILE], [1, tile_h]], vector_offset=rows,
                    indirect_dim=0)
    if skip:
        nisa.dma_copy(dst=picked, src=src, oob_mode=nisa.oob_mode.skip)
    else:
        nisa.dma_copy(dst=picked, src=src)


@nki.jit
def token_gather_combine_kernel(contribution_hbm, index_hbm, valid_hbm):
    """``[R, H]`` fp32 rows, ``[T, K]`` int32 row indices, ``[T, K]`` fp32 0/1 -> ``[T, H]`` fp32.

    Every index below ``R``; every row a slot reads is finite, as a slot weighed zero
    multiplies its token's row in the buffer.
    """
    total_rows, hidden = contribution_hbm.shape
    tokens, slots = index_hbm.shape
    n_prgs = nl.num_programs(axes=0)
    prg_id = nl.program_id(0)
    kernel_assert(hidden % HIDDEN_TILE == 0 and HIDDEN_TILE % n_prgs == 0,
                  "whole hidden pieces, split evenly over the programs")
    # Program p gathers part p of every piece: banked piece k * n_prgs + p.
    tile_h = HIDDEN_TILE // n_prgs
    share = hidden // HIDDEN_TILE
    pieces = share * n_prgs
    # One row part per index, so a gather carries only a tensor-borne offset.
    banked_rows = total_rows * pieces
    banked = contribution_hbm.reshape((banked_rows, tile_h))
    out_hbm = nl.ndarray((tokens, hidden), dtype=nl.float32, buffer=nl.shared_hbm)
    tiles = tokens // TOKEN_TILE
    lines = _lines(slots)
    # An accumulator and a gather buffer per unit.
    batch_tiles = max(1, _BATCH_BYTES // (2 * share * tile_h * 4))
    zero = nl.ndarray((TOKEN_TILE, _LINE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero, value=0.0)
    ones = nl.ndarray((TOKEN_TILE, _LINE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    scores = nl.ndarray((1, batch_tiles * lines), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=scores, pattern=[[0, batch_tiles], [1, lines]], offset=1, channel_multiplier=0)
    for first in range(0, tiles, batch_tiles):
        count = min(batch_tiles, tiles - first)
        counts = nl.ndarray((1, count * lines), dtype=nl.float32, buffer=nl.psum)
        work = []
        for b in range(count):
            tb = first + b
            tables = _tile_tables(index_hbm, valid_hbm, tb * TOKEN_TILE)
            index = tables[0]
            weights = tables[1]
            # Each slot's token count: a product with a ones column, exact for 0/1.
            nisa.nc_matmul(dst=counts[:, b * lines:b * lines + slots], stationary=ones[:, 0:1],
                           moving=weights[:, 0:slots], is_stationary_onezero=True,
                           is_moving_onezero=True, accumulate=False)
            units = []
            for part in range(share):
                piece = part * n_prgs + prg_id
                rows = _unit_rows(index, weights, slots, piece, pieces, banked_rows)
                picked = nl.ndarray((TOKEN_TILE, tile_h), dtype=nl.float32, buffer=nl.sbuf)
                acc = nl.ndarray((TOKEN_TILE, tile_h), dtype=nl.float32, buffer=nl.sbuf)
                _gather(picked, banked, rows[:, 0:1], False)
                # The scalar engine starts the accumulator, so the vector engine carries only
                # the later slots; + 0 turns an empty slot's -0 (a negative row weighed zero)
                # into +0, as torch.
                nisa.activation(dst=acc, op=nl.copy, data=picked, scale=weights[:, 0:1],
                                bias=zero[:, 0:1])
                units.append((tb * TOKEN_TILE * hidden + piece * tile_h, rows, picked, acc))
            work.append((weights, units))
        ends = _tile_ends(counts, scores, count, slots)

        for b in range(count):
            weights = work[b][0]
            units = work[b][1]

            def add_slot(slot):
                """Slot ``slot`` (a register) of every unit of the tile."""
                weight = nl.ndarray((TOKEN_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                _column(weight, weights, slot)
                for unit in units:
                    row = nl.ndarray((TOKEN_TILE, 1), dtype=nl.int32, buffer=nl.sbuf)
                    _column(row, unit[1], slot)
                    _gather(unit[2], banked, row, True)
                    nisa.scalar_tensor_tensor(dst=unit[3], data=unit[2], op0=nl.multiply,
                                              operand0=weight, op1=nl.add, operand1=unit[3])

            nl.fori_loop(1, _register(ends[:, b:b + 1]), add_slot)
            for unit in units:
                nisa.dma_copy(dst=out_hbm.ap(pattern=[[hidden, TOKEN_TILE], [1, tile_h]],
                                             offset=unit[0]), src=unit[3], dge_mode=_HWDGE,
                              engine=_QUEUE)
    return out_hbm


def token_gather_combine_torch(contribution: Tensor, index: Tensor, valid: Tensor) -> Tensor:
    """The same sum in torch, slot by slot; the reference and the small-shape path."""
    combined = torch.zeros(index.shape[0], contribution.shape[1], dtype=torch.float32,
                           device=contribution.device)
    for j in range(index.shape[1]):
        gathered = contribution.index_select(0, index[:, j].to(torch.int64))
        combined = combined + torch.where(valid[:, j:j + 1] > 0, gathered, 0.0)
    return combined


def can_run_token_gather_combine(contribution: Tensor, index: Tensor) -> bool:
    """Whole token tiles, whole hidden tiles, fp32 rows, and a device that runs NKI."""
    return (
        can_run_kernel(contribution)
        and contribution.dtype == torch.float32
        and index.shape[0] % TOKEN_TILE == 0
        and contribution.shape[1] % HIDDEN_TILE == 0
    )


def token_gather_combine(contribution: Tensor, index: Tensor, valid: Tensor) -> Tensor:
    """``[T, H]`` fp32: each token's valid rows of ``contribution`` summed in slot order.

    Args:
        contribution: ``[R, H]`` fp32 rows.
        index: ``[T, K]`` integer row indices; an invalid slot must name a finite row.
        valid: ``[T, K]`` 0/1, which slots count.
    """
    if not can_run_token_gather_combine(contribution, index):
        return token_gather_combine_torch(contribution, index, valid)
    call = wrap_nki(token_gather_combine_kernel)
    if lnc_pair():
        call = call[2]
    return call(
        contribution.contiguous(),
        index.to(torch.int32).contiguous(),
        valid.to(torch.float32).contiguous(),
    )
