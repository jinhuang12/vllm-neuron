# SPDX-License-Identifier: Apache-2.0
"""Shape-specialized, fused block-FP8 experts on Trainium2.

BLOCK_M/N/K are compile-time scheduling choices. The inner 128x128 product
and its scale remain fixed by the checkpoint quantization contract.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.allocator import create_auto_alloc_manager

# Routing is counted, and row classes are sized, in row tiles of at most this
# many rows (the largest divisor of the block width that does not exceed it).
_ROW_TILE = 32
# Zero-fill blocks per iteration of the trailing zero loop, which writes the
# empty blocks that the item loops did not reach.
_TRAILING_ZERO_BLOCKS = 4
# Gate/up accumulators the Vector Engine folds in turn. A fold reads the
# accumulator its predecessor wrote, and one fold's latency (about 280 ns at
# 64 rows) exceeds its issue interval (about 200 ns), so a single chain stalls.
_FOLD_CHAINS = 4
# Of every _OFFLOAD_PERIOD accumulators, _OFFLOAD_FOLDS fold as a Scalar
# Engine scale plus a GpSimd add instead of one Vector Engine instruction. At
# 64 rows each engine issues a fold step in about 200 ns (Vector, Scalar) or
# 300 ns (GpSimd), so 3 of 8 balances the Vector Engine against GpSimd.
_OFFLOAD_FOLDS = 3
_OFFLOAD_PERIOD = 8
# DMA descriptors come from the hardware generator: weights, routing reads and
# stores on the Sync engine's queue, zero fills on the Scalar engine's queue so
# they stream beside the weights instead of behind them. Only the per-row
# gathers, which need a vector of row offsets, keep software descriptors.
_HWDGE = nisa.dge_mode.hwdge
_LOAD_QUEUE = nisa.engine.sync
_ZERO_QUEUE = nisa.engine.scalar


def _tile(rows, cols, dtype=nl.float32):
    """Allocate an SBUF tile of shape (rows, cols), backed by at least 8 columns."""
    return nl.ndarray((rows, max(cols, 8)), dtype=dtype, buffer=nl.sbuf)[:, :cols]


def _block_ap(tensor, block, pattern, offset=0):
    """Access ``pattern`` at element ``offset`` of routing block ``block`` (a register)."""
    return tensor.ap(pattern=pattern, offset=offset, scalar_offset=block,
                     indirect_dim=0)


def _weight_panel(weights, expert, panel, first, count):
    """Load contiguous hidden tiles from one packed gate/up or down panel."""
    hidden = weights.shape[3] * 128
    tile = nl.ndarray((128, count, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=tile.reshape((128, count * 128)),
        src=weights.ap(
            pattern=[[hidden, 128], [1, count * 128]],
            offset=panel * 128 * hidden + first * 128,
            scalar_offset=expert, indirect_dim=0,
        ),
        dge_mode=_HWDGE, engine=_LOAD_QUEUE,
    )
    return tile


def _gather_rows(hidden, affinity, row_ids, expert_ids, block, m0, m, x):
    """Transpose the hidden rows [m0, m0+m) of a block into ``x`` [128, H/128, m].

    Returns the routing weight of every row, one [rows, 1] tile per 128 rows.
    An invalid id (-1) reads the zero hidden row and the zero affinity row.
    """
    tokens, h = hidden.shape
    nh = h // 128
    experts = affinity.shape[0] // tokens
    routing = []
    for t0 in range(0, m, 128):
        rows = min(128, m - t0)
        ids = _tile(rows, 1, nl.int32)
        nisa.dma_copy(dst=ids, src=_block_ap(row_ids, block, [[1, rows], [1, 1]], m0 + t0),
                      dge_mode=_HWDGE, engine=_LOAD_QUEUE)
        invalid = _tile(rows, 1, nl.int32)
        nisa.tensor_scalar(dst=invalid, data=ids, op0=nl.less, operand0=0)
        bump = _tile(rows, 1, nl.int32)
        nisa.tensor_scalar(dst=bump, data=invalid, op0=nl.multiply, operand0=tokens)
        resolved = _tile(rows, 1, nl.int32)
        nisa.tensor_tensor(dst=resolved, data1=ids, data2=bump, op=nl.add)
        gathered = nl.ndarray((rows, h), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=gathered, src=hidden.ap(
            pattern=[[h, rows], [1, h]], vector_offset=resolved, indirect_dim=0))
        # Transposes share a PSUM bank; one copy evacuates the whole bank.
        group = max(1, min(nh, nl.tile_size.psum_fmax // rows))
        for hb0 in range(0, nh, group):
            count = min(group, nh - hb0)
            psum = nl.ndarray((128, count, rows), dtype=nl.bfloat16, buffer=nl.psum)
            for j in range(count):
                hb = hb0 + j
                nisa.nc_transpose(dst=psum[:, j, :],
                                  data=gathered[:, hb * 128:(hb + 1) * 128])
            nisa.tensor_copy(dst=x[:, hb0:hb0 + count, t0:t0 + rows], src=psum)
        expert_rows = _tile(rows, 1, nl.int32)
        nisa.dma_copy(dst=expert_rows, src=_block_ap(expert_ids, block, [[0, rows], [1, 1]]),
                      dge_mode=_HWDGE, engine=_LOAD_QUEUE)
        affinity_rows = _tile(rows, 1, nl.int32)
        nisa.tensor_scalar(dst=affinity_rows, data=resolved, op0=nl.multiply, operand0=experts)
        nisa.tensor_tensor(dst=affinity_rows, data1=affinity_rows, data2=expert_rows, op=nl.add)
        route_rows = _tile(rows, 1)
        nisa.dma_copy(dst=route_rows, src=affinity.ap(
            pattern=[[1, rows], [1, 1]], vector_offset=affinity_rows, indirect_dim=0))
        routing.append(route_rows)
    return routing


def _offloaded(index):
    """Whether accumulator ``index`` folds on the Scalar and GpSimd engines.

    Spreads ``_OFFLOAD_FOLDS`` of every ``_OFFLOAD_PERIOD`` accumulators evenly.
    """
    return ((index + 1) * _OFFLOAD_FOLDS // _OFFLOAD_PERIOD
            > index * _OFFLOAD_FOLDS // _OFFLOAD_PERIOD)


def _fold(dst, partial, scale, first, offload):
    """Scale one 128x128 product and add it to ``dst`` in contraction order.

    The Vector Engine fuses both steps. With ``offload`` the Scalar Engine
    scales the product out of PSUM and GpSimd, which cannot read PSUM, adds
    it; every engine rounds the FP32 product and then the FP32 sum, so the
    bits match.
    """
    if offload:
        if first:
            nisa.tensor_scalar(dst=dst, data=partial, op0=nl.multiply, operand0=scale,
                               engine=nisa.engine.scalar)
        else:
            scaled = nl.ndarray(partial.shape, dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=scaled, data=partial, op0=nl.multiply, operand0=scale,
                               engine=nisa.engine.scalar)
            nisa.tensor_tensor(dst=dst, data1=scaled, data2=dst, op=nl.add,
                               engine=nisa.engine.gpsimd)
    elif first:
        nisa.tensor_scalar(dst=dst, data=partial, op0=nl.multiply, operand0=scale,
                           engine=nisa.engine.vector)
    else:
        nisa.scalar_tensor_tensor(dst=dst, data=partial, op0=nl.multiply, operand0=scale,
                                  op1=nl.add, operand1=dst)


def _swiglu(activation, gate_up, clamp, ib, m):
    """Clamp gate/up panel ``ib`` and store its SwiGLU in BF16."""
    ni = activation.shape[1]
    gate, up = _tile(128, m), _tile(128, m)
    nisa.tensor_scalar(dst=gate, data=gate_up[:, ib, :], op0=nl.minimum, operand0=clamp[:, 0:1])
    nisa.tensor_scalar(dst=up, data=gate_up[:, ni + ib, :], op0=nl.maximum, operand0=clamp[:, 1:2])
    nisa.tensor_scalar(dst=up, data=up, op0=nl.minimum, operand0=clamp[:, 2:3])
    sigmoid, silu, gated = _tile(128, m), _tile(128, m), _tile(128, m)
    nisa.activation(dst=sigmoid, data=gate, op=nl.sigmoid)
    nisa.scalar_tensor_tensor(dst=silu, data=gate, op0=nl.multiply, operand0=1.0,
                              op1=nl.multiply, operand1=sigmoid)
    nisa.scalar_tensor_tensor(dst=gated, data=silu, op0=nl.multiply, operand0=1.0,
                              op1=nl.multiply, operand1=up)
    nisa.tensor_copy(dst=activation[:, ib, :], src=gated)


def _compute_rows(hidden, weights, affinity, row_ids, expert_ids, out,
                  clamp, scale, expert, block, m0, m, nstep, kstep):
    """Compute and store rows [m0, m0+m) of one routing block for its expert.

    Every output element keeps the original arithmetic: an FP32 128x128
    product, its block scale, a left fold over the contraction tiles, the same
    SwiGLU instruction sequence with a BF16 activation, and the routing weight.
    Gate/up panels run in groups of (gate, up) pairs, the folds of a group
    interleaved, so each group's SwiGLU overlaps the next group's products.
    BLOCK_K groups the hidden tiles of one gate/up weight load and BLOCK_N the
    hidden output tiles of one down weight load.
    """
    experts, panels, contraction, nh, channels = weights.shape
    h = hidden.shape[1]
    ni = panels // 3
    x = nl.ndarray((128, nh, m), dtype=nl.bfloat16, buffer=nl.sbuf)
    routing = _gather_rows(hidden, affinity, row_ids, expert_ids, block, m0, m, x)

    gate_up = nl.ndarray((128, 2 * ni, m), dtype=nl.float32, buffer=nl.sbuf)
    activation = nl.ndarray((128, ni, m), dtype=nl.bfloat16, buffer=nl.sbuf)
    pairs = max(1, _FOLD_CHAINS // 2)
    for ib0 in range(0, ni, pairs):
        group = []
        for ib in range(ib0, min(ni, ib0 + pairs)):
            group.append(ib)
            group.append(ni + ib)
        for k0 in range(0, nh, kstep):
            nk = min(kstep, nh - k0)
            tiles = []
            for panel in group:
                tiles.append(_weight_panel(weights, expert, panel, k0, nk))
            for k in range(nk):
                hb = k0 + k
                for g in range(len(group)):
                    panel = group[g]
                    partial = nl.ndarray((128, m), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=partial, stationary=tiles[g][:, k, :],
                                   moving=x[:, hb, :], accumulate=False)
                    _fold(gate_up[:, panel, :], partial, scale[:, panel, hb:hb + 1], hb == 0,
                          _offloaded(panel))
        for ib in range(ib0, min(ni, ib0 + pairs)):
            _swiglu(activation, gate_up, clamp, ib, m)

    result = nl.ndarray((128, nh, m), dtype=nl.float32, buffer=nl.sbuf)
    for ib in range(ni):
        for n0 in range(0, nh, nstep):
            nn = min(nstep, nh - n0)
            w = _weight_panel(weights, expert, 2 * ni + ib, n0, nn)
            for n in range(nn):
                hb = n0 + n
                partial = nl.ndarray((128, m), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=partial, stationary=w[:, n, :],
                               moving=activation[:, ib, :], accumulate=False)
                _fold(result[:, hb, :], partial, scale[:, 2 * ni + ib, hb:hb + 1], ib == 0,
                      _offloaded(hb))

    group = max(1, nl.tile_size.psum_fmax // 128)
    for t0 in range(0, m, 128):
        rows = min(128, m - t0)
        output_rows = nl.ndarray((rows, nh, 128), dtype=nl.float32, buffer=nl.sbuf)
        for hb0 in range(0, nh, group):
            count = min(group, nh - hb0)
            psum = nl.ndarray((rows, count, 128), dtype=nl.float32, buffer=nl.psum)
            for j in range(count):
                nisa.nc_transpose(dst=psum[:, j, :], data=result[:, hb0 + j, t0:t0 + rows])
            # The routing weight scales the transposed product straight out of PSUM.
            nisa.tensor_scalar(dst=output_rows[:, hb0:hb0 + count, :], data=psum,
                               op0=nl.multiply, operand0=routing[t0 // 128])
        nisa.dma_copy(dst=_block_ap(out, block, [[h, rows], [1, h]], (m0 + t0) * h),
                      src=output_rows.reshape((rows, h)), dge_mode=_HWDGE, engine=_LOAD_QUEUE)


def _expert_operands(expert_ids, scales, block, panels, nh):
    """Load a block's expert id into a register and its block scales on every partition."""
    expert = _tile(1, 1, nl.int32)
    nisa.dma_copy(dst=expert, src=_block_ap(expert_ids, block, [[1, 1], [1, 1]]), dge_mode=_HWDGE, engine=_LOAD_QUEUE)
    expert_reg = nisa.register_alloc()
    nisa.register_load(dst=expert_reg, src=expert)
    scale = nl.ndarray((128, panels, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=scale.reshape((128, panels * nh)),
        src=scales.ap(pattern=[[0, 128], [1, panels * nh]],
                      scalar_offset=expert_reg, indirect_dim=0),
        dge_mode=_HWDGE, engine=_LOAD_QUEUE,
    )
    return expert_reg, scale


def _dense_schedule(hidden, weights, scales, row_ids, expert_ids, affinity,
                    out, clamp, block_m, nstep, kstep):
    """Retain the original static block schedule and arithmetic."""
    blocks, q = row_ids.shape
    panels, nh = weights.shape[1], weights.shape[3]
    for static_block in range(nl.program_id(0), blocks, nl.num_programs(0)):
        block = nisa.register_alloc(static_block)
        expert_reg, scale = _expert_operands(expert_ids, scales, block, panels, nh)
        for m0 in range(0, q, block_m):
            _compute_rows(hidden, weights, affinity, row_ids, expert_ids, out, clamp,
                          scale, expert_reg, block, m0, min(block_m, q - m0),
                          nstep, kstep)


def _row_step(q, block_m):
    """The largest divisor of the block width ``q`` within ``min(block_m, _ROW_TILE)``."""
    step = min(block_m, _ROW_TILE)
    while q % step:
        step -= 1
    return step


def _row_classes(q, row_step):
    """Row-tile bounds of the compute classes, widest (the whole block) last.

    A routed block runs in the narrowest class that holds its last routed row
    tile. Widths double from two row tiles; one row tile is not its own class,
    since a 128x128 weight load costs the Tensor Engine about as long as the
    product of two tiles' rows.
    """
    tiles = q // row_step
    bounds, width = [], 2
    while width < tiles:
        bounds.append(width)
        width *= 2
    return tuple(bounds) + (tiles,)


def _zero_slots(blocks, experts):
    """Empty blocks to zero beside each compute item.

    With every expert's rows in one block, ``blocks / experts - 1`` empty
    blocks remain per item. Slots past the empty list are skipped; empty
    blocks past the slots go to the trailing zero loop.
    """
    return max(1, -(-blocks // experts) - 1)


def _routing_counts(row_ids, row_step, sbm):
    """Read routing once and return valid rows per row tile, [1, blocks, tiles]."""
    blocks, q = row_ids.shape
    ntiles = q // row_step
    tile_counts = sbm.alloc((1, blocks, ntiles), nl.int32)
    # At most two 64-KiB temporary tiles share partition 0. Both are released
    # before the expert body, including for larger prefill buckets.
    tiles_per_load = max(1, 16384 // row_step)
    for first in range(0, blocks * ntiles, tiles_per_load):
        count = min(tiles_per_load, blocks * ntiles - first)
        sbm.open_scope(name="routing_count_input")
        ids = sbm.alloc((1, max(count * row_step, 8)), nl.int32)[:, :count * row_step]
        valid = sbm.alloc((1, max(count * row_step, 8)), nl.int32)[:, :count * row_step]
        nisa.dma_copy(dst=ids, src=row_ids.ap(
            pattern=[[0, 1], [1, count * row_step]], offset=first * row_step))
        nisa.tensor_scalar(dst=valid, data=ids, op0=nl.greater_equal, operand0=0)
        nisa.tensor_reduce(
            dst=tile_counts.reshape((1, blocks * ntiles))[:, first:first + count],
            data=valid.reshape((1, count, row_step)), op=nl.add, axis=(2,),
        )
        sbm.close_scope()
    return tile_counts


def _worklist(blocks, extra, sbm):
    """Allocate a compacted block list as ``(tile, count)``.

    The tile is int32 in partition 0: flags in columns [0, blocks), and the
    list from column ``blocks`` on, which ``nonzero_with_count`` writes. Source
    and destination are disjoint column ranges of one tile, so no placement can
    alias them. ``count`` is the FP32 list length.
    """
    tile = sbm.alloc((1, max(2 * blocks + 1 + extra, 8)), nl.int32)
    count = sbm.alloc((1, 8), nl.float32)[:, :1]
    return tile, count


def _list_entries(worklist, blocks):
    """The list columns of a worklist tile: the ids, then the out-of-range id ``blocks``."""
    tile = worklist[0]
    return tile[:, blocks:tile.shape[1]]


def _compact(worklist, blocks):
    """Compact the flags; every entry past the count, and its slot, reads ``blocks``."""
    tile, count = worklist
    nisa.nonzero_with_count(dst=tile[:, blocks:2 * blocks + 1], src=tile[:, :blocks],
                            padding_val=blocks)
    nisa.tensor_copy(dst=count, src=tile[:, 2 * blocks:2 * blocks + 1])
    nisa.memset(dst=tile[:, 2 * blocks:tile.shape[1]], value=blocks)


def _select(worklist, blocks, index, width=1):
    """``width`` list entries from register ``index`` on, as an SBUF read."""
    tile = worklist[0]
    return tile.ap(pattern=[[tile.shape[1], 1], [1, width]], offset=blocks,
                   scalar_offset=index, indirect_dim=1)


def _routing_worklists(tile_counts, expert_ids, bounds, sbm, merge_pairs=False, extra=0):
    """Compact routed blocks by row class, adjacent dense pairs, and empty blocks.

    A block's span is its last routed row tile plus one (0 when empty). Class
    ``c`` holds the blocks with ``bounds[c-1] < span <= bounds[c]``; the last
    class is the whole block. With ``merge_pairs``, consecutive whole-block
    entries of one expert are paired greedily and leave that class. Returns
    ``(paired or None, classes, empty)``; every PNC builds the same lists.
    """
    blocks, ntiles = tile_counts.shape[1], tile_counts.shape[2]
    span = sbm.alloc((1, max(blocks, 8)), nl.int32)[:, :blocks]
    sbm.open_scope(name="routing_span")
    position = sbm.alloc((1, blocks, ntiles), nl.int32)
    routed = sbm.alloc((1, blocks, ntiles), nl.int32)
    nisa.iota(dst=position, pattern=[[0, blocks], [1, ntiles]], offset=1)
    nisa.tensor_scalar(dst=routed, data=tile_counts, op0=nl.greater, operand0=0)
    nisa.tensor_tensor(dst=routed, data1=routed, data2=position, op=nl.multiply)
    nisa.tensor_reduce(dst=span, data=routed, op=nl.maximum, axis=(2,))
    sbm.close_scope()

    classes = []
    for _ in range(len(bounds)):
        classes.append(_worklist(blocks, extra, sbm))
    empty = _worklist(blocks, extra, sbm)
    paired = None
    if merge_pairs:
        paired = _worklist(blocks, extra, sbm)
    sbm.open_scope(name="routing_class_flags")
    below = sbm.alloc((1, max(blocks, 8)), nl.int32)[:, :blocks]
    low = 0
    for c in range(len(bounds)):
        flags = classes[c][0][:, :blocks]
        nisa.tensor_scalar(dst=flags, data=span, op0=nl.greater, operand0=low)
        nisa.tensor_scalar(dst=below, data=span, op0=nl.less_equal, operand0=bounds[c])
        nisa.tensor_tensor(dst=flags, data1=flags, data2=below, op=nl.multiply)
        low = bounds[c]
    nisa.tensor_scalar(dst=empty[0][:, :blocks], data=span, op0=nl.equal, operand0=0)

    if merge_pairs:
        # Greedily pair consecutive whole-block entries for the same expert.
        # All PNCs build the same non-overlapping list and keep output positions.
        dense = classes[-1][0][:, :blocks]
        pair_flags = paired[0][:, :blocks]
        nisa.memset(dst=pair_flags, value=0)
        if blocks > 1:
            block_experts = sbm.alloc((1, max(blocks, 8)), nl.int32)[:, :blocks]
            eligible = sbm.alloc((1, max(blocks - 1, 8)), nl.int32)[:, :blocks - 1]
            inverse = sbm.alloc((1, 8), nl.int32)[:, :1]
            members = sbm.alloc((1, max(blocks, 8)), nl.int32)[:, :blocks]
            nisa.dma_copy(dst=block_experts, src=expert_ids.reshape((1, blocks)))
            nisa.tensor_tensor(dst=eligible, data1=block_experts[:, :blocks - 1],
                               data2=block_experts[:, 1:blocks], op=nl.equal)
            nisa.tensor_tensor(dst=eligible, data1=eligible,
                               data2=dense[:, :blocks - 1], op=nl.multiply)
            nisa.tensor_tensor(dst=eligible, data1=eligible,
                               data2=dense[:, 1:blocks], op=nl.multiply)
            nisa.tensor_copy(dst=pair_flags[:, :1], src=eligible[:, :1])
            for position in range(1, blocks - 1):
                nisa.tensor_scalar(dst=inverse,
                                   data=pair_flags[:, position - 1:position],
                                   op0=nl.multiply, operand0=-1,
                                   op1=nl.add, operand1=1)
                nisa.tensor_tensor(dst=pair_flags[:, position:position + 1],
                                   data1=eligible[:, position:position + 1],
                                   data2=inverse, op=nl.multiply)
            nisa.tensor_copy(dst=members[:, :1], src=pair_flags[:, :1])
            nisa.tensor_tensor(dst=members[:, 1:blocks],
                               data1=pair_flags[:, 1:blocks],
                               data2=pair_flags[:, :blocks - 1], op=nl.add)
            nisa.tensor_tensor(dst=dense, data1=dense, data2=members, op=nl.subtract)
    sbm.close_scope()
    if merge_pairs:
        _compact(paired, blocks)
    for c in range(len(bounds)):
        _compact(classes[c], blocks)
    _compact(empty, blocks)
    return paired, classes, empty


def _ceil_div(count, divisor, sbm):
    """ceil(count / divisor) of a small non-negative FP32 count, as int32 [1, 1].

    Float-to-int tensor_copy rounds to nearest even. Adding
    (divisor - 1) / (2 * divisor) moves an exact quotient k below k + 1/2 and
    any remainder above it, so the rounding lands on the ceiling with no tie.
    """
    quotient = sbm.alloc((1, 8), nl.float32)[:, :1]
    result = sbm.alloc((1, 8), nl.int32)[:, :1]
    nisa.tensor_scalar(dst=quotient, data=count, op0=nl.multiply, operand0=1.0 / divisor,
                       op1=nl.add, operand1=(divisor - 1) / (2 * divisor))
    nisa.tensor_copy(dst=result, src=quotient)
    return result


def _phase_bounds(total, program, nprograms, sbm):
    """Round a phase to an equal number of iterations on both cores."""
    sbm.open_scope(name="routing_phase_bounds")
    boundary = _ceil_div(total, nprograms, sbm)
    nisa.tensor_scalar(dst=boundary, data=boundary, op0=nl.multiply,
                       operand0=nprograms, op1=nl.add, operand1=program)
    boundary_reg = nisa.register_alloc()
    nisa.register_load(dst=boundary_reg, src=boundary)
    sbm.close_scope()
    return boundary_reg


def _work_item(worklist, blocks, at_index, sbm):
    """Select one compacted item; duplicate the last item for an odd pair."""
    sbm.open_scope(name="routing_work_item")
    index = sbm.alloc((1, 8), nl.int32)[:, :1]
    last = sbm.alloc((1, 8), nl.float32)[:, :1]
    item = sbm.alloc((1, 8), nl.int32)[:, :1]
    nisa.register_store(dst=index, src=at_index)
    nisa.tensor_scalar(dst=last, data=worklist[1], op0=nl.add, operand0=-1)
    nisa.tensor_scalar(dst=index, data=index, op0=nl.minimum, operand0=last)
    position = nisa.register_alloc()
    nisa.register_load(dst=position, src=index)
    nisa.tensor_copy(dst=item, src=_select(worklist, blocks, position))
    item_reg = nisa.register_alloc()
    nisa.register_load(dst=item_reg, src=item)
    sbm.close_scope()
    return item_reg


def _zero_blocks(out, zeros, empty, first_slot, at_index, slots, sbm):
    """Zero empty-list entries [first + at*slots, +slots), first = ``first_slot``.

    Positions past the list read the out-of-range id, whose DMAs the engine
    skips, so a slot never writes a routed block.
    """
    blocks, q, h = out.shape
    sbm.open_scope(name="zero_blocks")
    index = sbm.alloc((1, 8), nl.int32)[:, :1]
    nisa.register_store(dst=index, src=at_index)
    nisa.tensor_scalar(dst=index, data=index, op0=nl.multiply, operand0=slots,
                       op1=nl.add, operand1=first_slot)
    nisa.tensor_scalar(dst=index, data=index, op0=nl.minimum, operand0=empty[1])
    position = nisa.register_alloc()
    nisa.register_load(dst=position, src=index)
    targets = sbm.alloc((1, max(slots, 8)), nl.int32)[:, :slots]
    nisa.tensor_copy(dst=targets, src=_select(empty, blocks, position, slots))
    for slot in range(slots):
        target = nisa.register_alloc()
        nisa.register_load(dst=target, src=targets[:, slot:slot + 1])
        for start in range(0, q, zeros.shape[0]):
            rows = min(zeros.shape[0], q - start)
            nisa.dma_copy(dst=out.ap(pattern=[[h, rows], [1, h]], offset=start * h,
                                     scalar_offset=target, indirect_dim=0),
                          src=zeros[:rows, :], oob_mode=nisa.oob_mode.skip, dge_mode=_HWDGE,
                          engine=_ZERO_QUEUE)
    sbm.close_scope()


def _block_rows(operands, block, width, block_m, nstep, kstep):
    """Zero rows [width, q) of a block, then compute rows [0, width) in ``block_m`` slices."""
    hidden, weights, scales, row_ids, expert_ids, affinity, clamp, out, zeros = operands
    q, h = out.shape[1], out.shape[2]
    # The zero rows go first: their queue's engine reaches them before its
    # share of the products, and the loop drains them with the stores.
    for start in range(width, q, zeros.shape[0]):
        count = min(zeros.shape[0], q - start)
        nisa.dma_copy(dst=_block_ap(out, block, [[h, count], [1, h]], start * h),
                      src=zeros[:count, :], dge_mode=_HWDGE, engine=_ZERO_QUEUE)
    expert, scale = _expert_operands(expert_ids, scales, block, weights.shape[1],
                                     weights.shape[3])
    for m0 in range(0, width, block_m):
        _compute_rows(hidden, weights, affinity, row_ids, expert_ids, out, clamp,
                      scale, expert, block, m0, min(block_m, width - m0), nstep, kstep)


def _item_loop(operands, worklist, zero_fill, width, block_m, nstep, kstep, sbm):
    """One dynamic loop over a worklist, computing rows [0, width) of each block.

    Each iteration also zeroes ``slots`` entries of the empty list from
    ``first_slot`` on; ``first_slot`` then advances past every slot the loop
    owned, on both programs.
    """
    out = operands[7]
    zeros = operands[8]
    empty, first_slot, slots = zero_fill
    blocks = out.shape[0]
    nprograms, program = nl.num_programs(0), nl.program_id(0)
    boundary = _phase_bounds(worklist[1], program, nprograms, sbm)

    def iteration(at_index):
        _zero_blocks(out, zeros, empty, first_slot, at_index, slots, sbm)
        block = _work_item(worklist, blocks, at_index, sbm)
        _block_rows(operands, block, width, block_m, nstep, kstep)
    nl.fori_loop(program, boundary, iteration, step=nprograms)
    sbm.open_scope(name="zero_slot_advance")
    iterations = _ceil_div(worklist[1], nprograms, sbm)
    nisa.tensor_scalar(dst=iterations, data=iterations, op0=nl.multiply,
                       operand0=nprograms * slots)
    nisa.tensor_tensor(dst=first_slot, data1=first_slot, data2=iterations, op=nl.add)
    sbm.close_scope()


@nki.jit
def moe_fused_fp8_kernel(hidden, weights, scales, row_ids, expert_ids, affinity,
                         bounds, BLOCK_M=128, BLOCK_N=1024, BLOCK_K=4096,
                         SKIP_PADDING=True):
    """Return FP32 [blocks,q,H] contributions from device routing.

    Args:
        hidden: BF16 [T+1,H], final row zero.
        weights: Prepared finite FP8 [E,3*(I/128),128,H/128,128].
        scales: Matching FP32 [E,3*(I/128),H/128].
        row_ids: int32 [blocks,q], token IDs or -1 (holes are allowed).
        expert_ids: int32 [blocks,1], expert for each block.
        affinity: FP32 [(T+1)*E,1], final E entries zero.
        bounds: FP32 [128,3], gate upper, up lower, up upper.
        BLOCK_M/N/K: Compile-time scheduling groups; scaled products stay
            128-granular and accumulate in their original contraction order.
        SKIP_PADDING: Skip empty blocks and rows using device control flow.
            The false setting preserves dense execution for comparisons.

    Notes:
        Padding rows are zero. Routing IDs are read once into row-tile counts;
        each routed block runs once, for the row prefix that holds its last
        routed row tile, in the narrowest of a few compile-time row classes, so
        its expert weights load once. Rows past the prefix and every empty block
        are written with zeros beside the products, from a separate DMA queue;
        every output row has exactly one writer, so no barrier orders them. The
        verified production geometry merges adjacent whole blocks only when
        their expert IDs match; its 512-row moving tile reuses one set of
        weights. Both physical programs take equal iteration counts; an odd
        list duplicates its last pure output write. Single-row decode retains
        its original dense expert schedule. All routing decisions remain on
        device, including zero and skewed routing.
    """
    blocks, q = row_ids.shape
    experts, panels, contraction, nh, channels = weights.shape
    h = hidden.shape[1]
    ni = panels // 3
    kernel_assert(blocks > 0 and q > 0 and experts > 0,
                  "Require positive blocks, row width, and expert count")
    kernel_assert(panels % 3 == 0 and ni > 0 and h == nh * 128,
                  "Packed panels and hidden width disagree")
    kernel_assert(contraction == 128 and channels == 128,
                  "FP8 products must retain the 128x128 checkpoint blocks")
    kernel_assert(scales.shape == (experts, panels, nh),
                  "Scale shape must match the packed weights")
    kernel_assert(1 <= BLOCK_M <= 512, "BLOCK_M must be in [1,512]")
    kernel_assert(BLOCK_N > 0 and BLOCK_N % 128 == 0,
                  "BLOCK_N must be a positive multiple of 128")
    kernel_assert(BLOCK_K > 0 and BLOCK_K % 128 == 0,
                  "BLOCK_K must be a positive multiple of 128")
    nstep, kstep = BLOCK_N // 128, BLOCK_K // 128
    nprograms, program = nl.num_programs(0), nl.program_id(0)
    out = nl.ndarray((blocks, q, h), dtype=nl.float32, buffer=nl.shared_hbm)
    clamp = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=clamp, src=bounds)

    if not SKIP_PADDING or q == 1:
        _dense_schedule(hidden, weights, scales, row_ids, expert_ids, affinity,
                        out, clamp, BLOCK_M, nstep, kstep)
        return out
    kernel_assert(nprograms in (1, 2),
                  "Padding-aware prefill supports one or two core programs")
    # The 512-row merge is enabled only for the geometry verified on Trn2.
    merge_pairs = (h == 4096 and ni * 128 == 512 and q == 256
                   and BLOCK_M == 256 and BLOCK_N == 4096 and BLOCK_K == 4096
                   and nprograms == 2)
    row_step = _row_step(q, BLOCK_M)
    class_bounds = _row_classes(q, row_step)
    slots = _zero_slots(blocks, experts)
    trailing = max(slots, _TRAILING_ZERO_BLOCKS)
    sbm = create_auto_alloc_manager()
    sbm.open_scope(name="routing")
    tile_counts = _routing_counts(row_ids, row_step, sbm)
    paired, classes, empty = _routing_worklists(
        tile_counts, expert_ids, class_bounds, sbm, merge_pairs=merge_pairs,
        extra=trailing)
    zeros = sbm.alloc((min(128, q), h), nl.float32)
    nisa.memset(dst=zeros, value=0.0)
    first_slot = sbm.alloc((1, 8), nl.float32)[:, :1]
    nisa.memset(dst=first_slot, value=0.0)

    operands = (hidden, weights, scales, row_ids, expert_ids, affinity, clamp, out, zeros)
    zero_fill = (empty, first_slot, slots)
    if paired is not None:
        kernel_assert(2 * q <= 512, "Merged moving dimension exceeds 512")
        _item_loop(operands, paired, zero_fill, 2 * q, 2 * q, nstep, kstep, sbm)
    for c in range(len(class_bounds)):
        _item_loop(operands, classes[c], zero_fill, class_bounds[c] * row_step,
                   BLOCK_M, nstep, kstep, sbm)

    sbm.open_scope(name="trailing_zero_bounds")
    remaining = sbm.alloc((1, 8), nl.float32)[:, :1]
    nisa.tensor_tensor(dst=remaining, data1=empty[1], data2=first_slot, op=nl.subtract)
    nisa.tensor_scalar(dst=remaining, data=remaining, op0=nl.maximum, operand0=0.0)
    passes = _ceil_div(remaining, trailing, sbm)
    passes_f32 = sbm.alloc((1, 8), nl.float32)[:, :1]
    nisa.tensor_copy(dst=passes_f32, src=passes)
    zero_boundary = _phase_bounds(passes_f32, program, nprograms, sbm)
    sbm.close_scope()

    def zero_rest(at_index):
        _zero_blocks(out, zeros, empty, first_slot, at_index, trailing, sbm)
    nl.fori_loop(program, zero_boundary, zero_rest, step=nprograms)
    sbm.close_scope()
    return out
