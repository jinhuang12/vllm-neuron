# SPDX-License-Identifier: Apache-2.0
"""Sum each token's expert rows of the fused MoE emission, as an NKI kernel.

Per token::

    out[t] = sum_j valid[t, j] * contribution[index[t, j]]      # fp32, j in slot order

The fused expert kernel emits ``[blocks * rows, H]`` fp32 rows sized for the worst
case, so most rows are padding. Gathering each token's ``K`` rows reads at most
``T * K`` rows, where a scatter-add over the emission reads every row and runs as a
serial GpSimd accumulate.

The kernel puts 128 tokens on the partitions and keeps an fp32 accumulator and a
gather buffer per token tile and program's part of a hidden piece (a unit). In a batch
of ``n`` token tiles, partition ``p`` of tile ``b`` holds the batch's token ``p * n + b``, so
each partition's index and weight entries for the whole batch are one contiguous piece:
one load per table per batch, and tile ``b``'s slots are line group ``b`` of the table.
Slot 0 gathers every token's row (an invalid slot names a finite row; the caller
points it at row 0), and the scalar engine starts the accumulator as ``row * valid + 0``.
The vector engine adds the later slots, in one device loop per batch of token tiles
that ends after the last slot some token of the batch fills; each trip gathers one slot
of every unit of the batch, so the batch's gathers are in flight together. A trip reads
its slot's weight of every tile, and its row of every unit, as one strided copy per
table at the loop register's offset. A rank holds
a few of the experts, so a token fills few of its ``K`` slots, and ``combine_slots``
fills them from slot 0, so the loop runs only the slots the batch's tokens fill. In a
later slot, an invalid slot gets an index past the last row, so its gather moves no
bytes for that token and leaves the token's previous row in the buffer, which the
vector engine weighs by zero.
The tile's first token reads its named row instead, so no gather skips every row,
whatever the slot order. The adds run in slot order from zero, the order of
``token_gather_combine_torch``.

A two-program grid splits each hidden piece's columns across the two PNCs of one LNC2
core. Every program takes every token tile, so all of them run the same loop trips (a
device loop under LNC2 passes a core barrier at every trip).

Every SBUF tensor has a fixed address from one ``SbufManager`` stack, and the token counts a
fixed PSUM address, as in the fused expert kernel, whose emission this kernel reads. neuronx-cc
inlines a kernel with fixed addresses after its allocator and one without before it. In one
graph with the expert kernel's device loop, a device loop inlined first splits the graph's block
out of order, and the compile fails (NCC_ICFG001 in a modular-flow subgraph, NCC_IGCA024 in a
one-function graph). With fixed addresses both kernels are inlined in one pass, in graph order.
"""

from __future__ import annotations

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.allocator import SbufManager
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


def _slot_groups(table, tiles: int, slots: int, width: int | None = None,
                 partitions: int | None = None):
    """``table`` (``[P, tiles * lines]``) as ``[partitions, tiles, width]``: per tile, the first
    ``width`` (default ``K``) columns of its line group, on the first ``partitions`` (default all)."""
    if width is None:
        width = slots
    if partitions is None:
        partitions = table.shape[0]
    return table.ap(pattern=[[table.shape[1], partitions], [_lines(slots), tiles], [1, width]],
                    offset=0)


def _batch_tables(sbm, index_hbm, valid_hbm, t0: int, tiles: int):
    """The ``(index, weights)`` of the batch's tokens ``t0 .. t0 + 128 * tiles - 1``: whole
    ``[128, tiles * lines]`` int32 and fp32 tensors from ``sbm``, one load each.

    Partition ``p`` holds tokens ``t0 + p * tiles + b``, tile ``b``'s, whose ``K`` slots fill the
    first columns of line group ``b``; a device loop reads one slot of every tile by one offset.
    """
    slots = index_hbm.shape[1]
    lines = _lines(slots)
    index = sbm.alloc_stack((TOKEN_TILE, tiles * lines), dtype=nl.int32)
    _load_groups(index, index_hbm, t0, tiles)
    weights = sbm.alloc_stack((TOKEN_TILE, tiles * lines), dtype=nl.float32)
    _load_groups(weights, valid_hbm, t0, tiles)
    return index, weights


def _load_groups(table, hbm, t0: int, tiles: int):
    """Load ``hbm`` rows ``t0 + p * tiles + b`` (``[T, K]``) into line group ``b`` of ``table``'s
    partition ``p``: one contiguous piece per partition."""
    slots = hbm.shape[1]
    nisa.dma_copy(dst=_slot_groups(table, tiles, slots),
                  src=hbm.ap(pattern=[[tiles * slots, TOKEN_TILE], [slots, tiles], [1, slots]],
                             offset=t0 * slots),
                  dge_mode=_HWDGE, engine=_QUEUE)


def _unit_rows(sbm, index, weights, tiles: int, slots: int, piece: int, pieces: int,
               banked_rows: int):
    """The batch's rows of banked piece ``piece`` (of ``pieces`` per row): a whole
    ``[128, tiles * lines]`` int32 tensor from ``sbm``, tile ``b``'s in line group ``b``.

    Column ``j`` of a group holds ``index * pieces + piece`` for slot 0, for a valid slot, and
    on partition 0 (the tile's first token, so a gather moves at least one row); an invalid
    later slot of any other token is past the last row.
    """
    lines = index.shape[1] // tiles
    # past = (1 - valid) * banked_rows + piece, then slot 0 and partition 0 keep their rows.
    past = sbm.alloc_stack((TOKEN_TILE, tiles * lines), dtype=nl.int32)
    nisa.tensor_scalar(dst=_slot_groups(past, tiles, slots),
                       data=_slot_groups(weights, tiles, slots), op0=nl.multiply,
                       operand0=-float(banked_rows), op1=nl.add,
                       operand1=float(banked_rows + piece))
    nisa.memset(dst=_slot_groups(past, tiles, slots, width=1), value=piece)
    nisa.memset(dst=_slot_groups(past, tiles, slots, partitions=1), value=piece)
    rows = sbm.alloc_stack((TOKEN_TILE, tiles * lines), dtype=nl.int32)
    nisa.scalar_tensor_tensor(dst=_slot_groups(rows, tiles, slots),
                              data=_slot_groups(index, tiles, slots), op0=nl.multiply,
                              operand0=float(pieces), op1=nl.add,
                              operand1=_slot_groups(past, tiles, slots))
    return rows


def _batch_end(sbm, counts, scores, tiles: int, slots: int):
    """``[1, 1]`` int32 on partition 0 (in a whole line from ``sbm``): one past the last slot
    that some token of the batch's ``tiles`` token tiles fills, and at least 1.

    ``counts`` holds each tile's tokens per slot, ``scores`` the slot numbers plus one; a
    filled slot scores its number plus one, and the end is the greatest score.
    """
    filled = sbm.alloc_stack((1, tiles * _lines(slots)), dtype=nl.float32)
    nisa.scalar_tensor_tensor(dst=_slot_groups(filled, tiles, slots),
                              data=_slot_groups(counts, tiles, slots), op0=nl.minimum,
                              operand0=1.0, op1=nl.multiply,
                              operand1=_slot_groups(scores, tiles, slots))
    best = sbm.alloc_stack((1, _lines(tiles)), dtype=nl.float32)
    nisa.tensor_reduce(dst=best[:, 0:tiles], data=_slot_groups(filled, tiles, slots),
                       op=nl.maximum, axis=2)
    most = sbm.alloc_stack((1, _LINE), dtype=nl.float32)
    nisa.tensor_reduce(dst=most[:, 0:1], data=best[:, 0:tiles], op=nl.maximum, axis=1)
    end = sbm.alloc_stack((1, _LINE), dtype=nl.int32)
    nisa.tensor_scalar(dst=end[:, 0:1], data=most[:, 0:1], op0=nl.maximum, operand0=1.0)
    return end


def _columns(dst, table, tiles: int, slot):
    """Copy column ``slot`` (a register) of each of the ``tiles`` line groups of the whole
    ``[128, tiles * lines]`` SBUF ``table`` to ``dst`` (``[128, tiles]``): one strided copy.

    The scalar engine copies: a dynamic SBUF read runs on the vector or scalar engine,
    and the vector engine carries the adds.
    """
    lines = table.shape[1] // tiles
    nisa.tensor_copy(dst=dst, src=table.ap(pattern=[[table.shape[1], TOKEN_TILE], [lines, tiles]],
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
    sbm = SbufManager(sb_lower_bound=0, sb_upper_bound=nl.tile_size.sbuf_fmax_bytes)
    sbm.open_scope(name="token_gather_combine")
    zero = sbm.alloc_stack((TOKEN_TILE, _LINE), dtype=nl.float32)
    nisa.memset(dst=zero, value=0.0)
    ones = sbm.alloc_stack((TOKEN_TILE, _LINE), dtype=nl.float32)
    nisa.memset(dst=ones, value=1.0)
    scores = sbm.alloc_stack((1, batch_tiles * lines), dtype=nl.float32)
    nisa.iota(dst=scores, pattern=[[0, batch_tiles], [1, lines]], offset=1, channel_multiplier=0)
    for first in range(0, tiles, batch_tiles):
        count = min(batch_tiles, tiles - first)
        # A batch's tensors are freed after its output stores; the next batch reuses them.
        sbm.open_scope(name="batch")
        counts = nl.ndarray((1, count * lines), dtype=nl.float32, buffer=nl.psum, address=(0, 0))
        # The batch's gather buffers and accumulators take the scope's first addresses.
        buffers = []
        for _ in range(count * share):
            picked = sbm.alloc_stack((TOKEN_TILE, tile_h), dtype=nl.float32)
            acc = sbm.alloc_stack((TOKEN_TILE, tile_h), dtype=nl.float32)
            buffers.append((picked, acc))
        t0 = first * TOKEN_TILE
        tables = _batch_tables(sbm, index_hbm, valid_hbm, t0, count)
        index = tables[0]
        weights = tables[1]
        # Per part of the program's hidden share, the batch's rows of its banked piece.
        rows = []
        for part in range(share):
            rows.append(_unit_rows(sbm, index, weights, count, slots, part * n_prgs + prg_id,
                                   pieces, banked_rows))
        for b in range(count):
            # Each slot's token count: a product with a ones column, exact for 0/1.
            nisa.nc_matmul(dst=counts[:, b * lines:b * lines + slots], stationary=ones[:, 0:1],
                           moving=weights[:, b * lines:b * lines + slots],
                           is_stationary_onezero=True, is_moving_onezero=True, accumulate=False)
            for part in range(share):
                picked = buffers[b * share + part][0]
                acc = buffers[b * share + part][1]
                _gather(picked, banked, rows[part][:, b * lines:b * lines + 1], False)
                # The scalar engine starts the accumulator, so the vector engine carries only
                # the later slots; + 0 turns an empty slot's -0 (a negative row weighed zero)
                # into +0, as torch.
                nisa.activation(dst=acc, op=nl.copy, data=picked,
                                scale=weights[:, b * lines:b * lines + 1], bias=zero[:, 0:1])
        end = _batch_end(sbm, counts, scores, count, slots)
        # A trip's slot columns: every tile's weight, every unit's row.
        weight_cols = sbm.alloc_stack((TOKEN_TILE, _lines(count)), dtype=nl.float32)
        row_cols = []
        for _ in range(share):
            row_cols.append(sbm.alloc_stack((TOKEN_TILE, _lines(count)), dtype=nl.int32))

        def add_slot(slot):
            """Slot ``slot`` (a register) of every unit of the batch."""
            _columns(weight_cols[:, 0:count], weights, count, slot)
            for part in range(share):
                _columns(row_cols[part][:, 0:count], rows[part], count, slot)
            for b in range(count):
                for part in range(share):
                    unit = buffers[b * share + part]
                    _gather(unit[0], banked, row_cols[part][:, b:b + 1], True)
                    nisa.scalar_tensor_tensor(dst=unit[1], data=unit[0], op0=nl.multiply,
                                              operand0=weight_cols[:, b:b + 1], op1=nl.add,
                                              operand1=unit[1])

        nl.fori_loop(1, _register(end[:, 0:1]), add_slot)
        for b in range(count):
            for part in range(share):
                # Tile b's partition p is token t0 + p * count + b.
                nisa.dma_copy(dst=out_hbm.ap(pattern=[[count * hidden, TOKEN_TILE], [1, tile_h]],
                                             offset=(t0 + b) * hidden
                                             + (part * n_prgs + prg_id) * tile_h),
                              src=buffers[b * share + part][1], dge_mode=_HWDGE, engine=_QUEUE)
        sbm.close_scope()
    sbm.close_scope()
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
