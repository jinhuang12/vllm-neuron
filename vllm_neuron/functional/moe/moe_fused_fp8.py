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
# DMA descriptors come from the hardware generator on the Sync engine's queue,
# which issues nothing else, in program order: an item's weights and scales,
# its zero fills, its row ids, then its stores. Only the per-row gathers, which
# need a vector of row offsets, keep software descriptors.
_HWDGE = nisa.dge_mode.hwdge
_DMA_QUEUE = nisa.engine.sync


def _tile(rows, cols, dtype=nl.float32):
    """Allocate an SBUF tile of shape (rows, cols), backed by at least 8 columns."""
    return nl.ndarray((rows, max(cols, 8)), dtype=dtype, buffer=nl.sbuf)[:, :cols]


def _block_ap(tensor, block, pattern, offset=0):
    """Access ``pattern`` at element ``offset`` of routing block ``block`` (a register)."""
    return tensor.ap(pattern=pattern, offset=offset, scalar_offset=block,
                     indirect_dim=0)


def _experts_row(expert_ids):
    """Load the block experts [blocks, 1] into one SBUF row, [1, max(blocks, 8)] int32."""
    blocks = expert_ids.shape[0]
    row = nl.ndarray((1, max(blocks, 8)), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=row[:, :blocks], src=expert_ids.reshape((1, blocks)))
    return row


def _expert_operands(weights, scales, experts_row, block, kstep, nstep):
    """Load a block's expert weights and block scales, once per block.

    The expert id comes from ``experts_row`` (SBUF [1, >=blocks] int32), at
    register ``block``. Returns ``(gate_up, down, scale)``: FP8 gate/up tiles
    ``[128, 2I/128, nk, 128]`` for each BLOCK_K hidden chunk and down tiles
    ``[128, I/128, nn, 128]`` for each BLOCK_N hidden chunk, one DMA each, so
    every 128x128 stationary is one contiguous row of 128 columns; and every
    scale of the expert on every partition, ``[128, 3I/128, H/128]``, from one
    broadcast DMA.
    """
    experts, panels, contraction, nh, channels = weights.shape
    h, ni = nh * 128, panels // 3
    expert = _tile(1, 1, nl.int32)
    nisa.tensor_copy(dst=expert, src=experts_row.ap(
        pattern=[[experts_row.shape[1], 1], [1, 1]], scalar_offset=block, indirect_dim=1))
    expert_reg = nisa.register_alloc()
    nisa.register_load(dst=expert_reg, src=expert)

    gate_up = []
    for k0 in range(0, nh, kstep):
        nk = min(kstep, nh - k0)
        tile = nl.ndarray((128, 2 * ni, nk, 128), dtype=weights.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile.reshape((128, 2 * ni * nk * 128)), src=weights.ap(
            pattern=[[h, 128], [128 * h, 2 * ni], [1, nk * 128]], offset=k0 * 128,
            scalar_offset=expert_reg, indirect_dim=0), dge_mode=_HWDGE, engine=_DMA_QUEUE)
        gate_up.append(tile)
    scale = nl.ndarray((128, panels, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale.reshape((128, panels * nh)), src=scales.ap(
        pattern=[[0, 128], [1, panels * nh]], scalar_offset=expert_reg, indirect_dim=0),
        dge_mode=_HWDGE, engine=_DMA_QUEUE)
    down = []
    for n0 in range(0, nh, nstep):
        nn = min(nstep, nh - n0)
        tile = nl.ndarray((128, ni, nn, 128), dtype=weights.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile.reshape((128, ni * nn * 128)), src=weights.ap(
            pattern=[[h, 128], [128 * h, ni], [1, nn * 128]],
            offset=2 * ni * 128 * h + n0 * 128,
            scalar_offset=expert_reg, indirect_dim=0), dge_mode=_HWDGE, engine=_DMA_QUEUE)
        down.append(tile)
    return gate_up, down, scale


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
                      dge_mode=_HWDGE, engine=_DMA_QUEUE)
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
                      dge_mode=_HWDGE, engine=_DMA_QUEUE)
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


def _compute_rows(operands, loaded, block, m0, m):
    """Compute and store rows [m0, m0+m) of one routing block for its expert.

    Every output element keeps the original arithmetic: an FP32 128x128
    product, its block scale, a left fold over the contraction tiles, the same
    SwiGLU instruction sequence with a BF16 activation, and the routing weight.
    Gate/up panels run in groups of (gate, up) pairs, the folds of a group
    interleaved, so each group's SwiGLU overlaps the next group's products.
    Every stationary is one contiguous 128-column row of a weight tile: the
    Tensor Engine reads a stationary with one free dimension only.
    """
    hidden, weights, row_ids, expert_ids, affinity, clamp, out = operands
    gate_up_tiles, down_tiles, scale = loaded
    nh, h = weights.shape[3], hidden.shape[1]
    ni = weights.shape[1] // 3
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
        k0 = 0
        for tile in gate_up_tiles:
            for k in range(tile.shape[2]):
                hb = k0 + k
                for panel in group:
                    partial = nl.ndarray((128, m), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=partial, stationary=tile[:, panel, k, :],
                                   moving=x[:, hb, :], accumulate=False)
                    _fold(gate_up[:, panel, :], partial, scale[:, panel, hb:hb + 1], hb == 0,
                          _offloaded(panel))
            k0 += tile.shape[2]
        for ib in range(ib0, min(ni, ib0 + pairs)):
            _swiglu(activation, gate_up, clamp, ib, m)

    result = nl.ndarray((128, nh, m), dtype=nl.float32, buffer=nl.sbuf)
    for ib in range(ni):
        n0 = 0
        for tile in down_tiles:
            for n in range(tile.shape[2]):
                hb = n0 + n
                partial = nl.ndarray((128, m), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=partial, stationary=tile[:, ib, n, :],
                               moving=activation[:, ib, :], accumulate=False)
                _fold(result[:, hb, :], partial, scale[:, 2 * ni + ib, hb:hb + 1], ib == 0,
                      _offloaded(hb))
            n0 += tile.shape[2]

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
                      src=output_rows.reshape((rows, h)), dge_mode=_HWDGE, engine=_DMA_QUEUE)


def _dense_schedule(compute, loads, block_m):
    """Retain the original static block schedule and arithmetic."""
    weights, scales, experts_row, kstep, nstep = loads
    blocks, q = compute[2].shape
    for static_block in range(nl.program_id(0), blocks, nl.num_programs(0)):
        block = nisa.register_alloc(static_block)
        loaded = _expert_operands(weights, scales, experts_row, block, kstep, nstep)
        for m0 in range(0, q, block_m):
            _compute_rows(compute, loaded, block, m0, min(block_m, q - m0))


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


def _routing_spans(row_ids, row_step, sbm):
    """Read routing once and return each block's span, int32 [1, blocks].

    A block's span is its last routed row tile plus one, 0 when it is empty.
    Blocks sit on partitions, up to 128 at a time, so every step works on
    whole rows of ids; one PE transpose per chunk moves the spans to partition 0.
    """
    blocks, q = row_ids.shape
    ntiles = q // row_step
    span = sbm.alloc((1, max(blocks, 8)), nl.int32)[:, :blocks]
    for b0 in range(0, blocks, 128):
        count = min(128, blocks - b0)
        sbm.open_scope(name="routing_spans")
        ids = sbm.alloc((count, max(q, 8)), nl.int32)[:, :q]
        nisa.dma_copy(dst=ids, src=row_ids[b0:b0 + count, :])
        valid = sbm.alloc((count, max(q, 8)), nl.int32)[:, :q]
        nisa.tensor_scalar(dst=valid, data=ids, op0=nl.greater_equal, operand0=0)
        counts = sbm.alloc((count, max(ntiles, 8)), nl.int32)[:, :ntiles]
        nisa.tensor_reduce(dst=counts, data=valid.reshape((count, ntiles, row_step)),
                           op=nl.add, axis=(2,))
        position = sbm.alloc((count, max(ntiles, 8)), nl.int32)[:, :ntiles]
        nisa.iota(dst=position, pattern=[[1, ntiles]], offset=1)
        routed = sbm.alloc((count, max(ntiles, 8)), nl.int32)[:, :ntiles]
        nisa.tensor_scalar(dst=routed, data=counts, op0=nl.greater, operand0=0)
        nisa.tensor_tensor(dst=routed, data1=routed, data2=position, op=nl.multiply)
        last = sbm.alloc((count, 8), nl.float32)[:, :1]
        nisa.tensor_reduce(dst=last, data=routed, op=nl.maximum, axis=(1,))
        row = nl.ndarray((1, count), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=row, data=last)
        nisa.tensor_copy(dst=span[:, b0:b0 + count], src=row)
        sbm.close_scope()
    return span


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


def _routing_worklists(span, experts_row, bounds, sbm, merge_pairs=False, extra=0,
                       empty_extra=0):
    """Compact routed blocks by row class, adjacent dense pairs, and empty blocks.

    ``span`` is ``_routing_spans``'s [1, blocks] row. Class ``c`` holds the
    blocks with ``bounds[c-1] < span <= bounds[c]``; the last class is the
    whole block. With ``merge_pairs``, consecutive whole-block entries of one
    expert are paired greedily and leave that class. Every list holds ``extra``
    padding entries past its ``blocks + 1``, the empty list
    ``max(extra, empty_extra)``. Returns ``(paired or None, classes, empty)``;
    every PNC builds the same lists.
    """
    blocks = span.shape[1]

    classes = []
    for _ in range(len(bounds)):
        classes.append(_worklist(blocks, extra, sbm))
    empty = _worklist(blocks, max(extra, empty_extra), sbm)
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
            block_experts = experts_row[:, :blocks]
            eligible = sbm.alloc((1, max(blocks - 1, 8)), nl.int32)[:, :blocks - 1]
            negated = sbm.alloc((1, max(blocks - 1, 8)), nl.int32)[:, :blocks - 1]
            members = sbm.alloc((1, max(blocks, 8)), nl.int32)[:, :blocks]
            nisa.tensor_tensor(dst=eligible, data1=block_experts[:, :blocks - 1],
                               data2=block_experts[:, 1:blocks], op=nl.equal)
            nisa.tensor_tensor(dst=eligible, data1=eligible,
                               data2=dense[:, :blocks - 1], op=nl.multiply)
            nisa.tensor_tensor(dst=eligible, data1=eligible,
                               data2=dense[:, 1:blocks], op=nl.multiply)
            # pair[p] = eligible[p] * (1 - pair[p - 1]), as one scan:
            # pair[p] = (-eligible[p]) * pair[p - 1] + eligible[p], pair[-1] = 0.
            nisa.tensor_scalar(dst=negated, data=eligible, op0=nl.multiply, operand0=-1)
            nisa.tensor_tensor_scan(dst=pair_flags[:, :blocks - 1], data0=negated,
                                    data1=eligible, initial=0.0, op0=nl.multiply,
                                    op1=nl.add)
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


def _table_rows(blocks, nprograms):
    """Rows of an item table: the item indices a loop over up to ``blocks`` items reaches."""
    return -(-blocks // nprograms) * nprograms


def _table_reach(blocks, nprograms, loops, slots):
    """Empty-list entries that ``loops`` item tables can read, from entry 0.

    A table reads entries up to first_slot + rows * slots. A loop over c items
    advances first_slot by ceil(c / programs) * programs * slots, and the loops
    share at most ``blocks`` items, so first_slot stays at or below
    (blocks + loops * (programs - 1)) * slots.
    """
    return (blocks + loops * (nprograms - 1) + _table_rows(blocks, nprograms)) * slots


def _item_table(worklist, empty, first_slot, slots, blocks, rows, sbm):
    """Tabulate a loop's items and zero fills, int32 [1, rows, 1 + slots].

    Row ``at`` serves iteration ``at``: column 0 is list entry ``at``, or past
    the list its last entry (an odd list's second program repeats the last
    item; later rows never run). Columns 1.. are empty-list entries
    [first + at * slots, +slots), first = ``first_slot``; past that list they
    hold the out-of-range id ``blocks``, whose DMAs the engine skips. One copy
    then gives an iteration its block and its zero-fill targets, so the fills
    issue beside the weights instead of behind engine work queued for the
    products. ``first_slot`` then advances past every row the loop runs.
    """
    tile, count = worklist
    empty_tile = empty[0]
    nprograms = nl.num_programs(0)
    table = sbm.alloc((1, rows, 1 + slots), nl.int32)
    sbm.open_scope(name="item_table")
    items = table[:, :, 0]
    nisa.tensor_copy(dst=items, src=tile[:, blocks:blocks + rows])
    # Entries past the list read ``blocks``; they become the last entry.
    last = _tile(1, 1)
    nisa.tensor_scalar(dst=last, data=count, op0=nl.add, operand0=-1)
    last_index = _tile(1, 1, nl.int32)
    nisa.tensor_copy(dst=last_index, src=last)
    last_reg = nisa.register_alloc()
    nisa.register_load(dst=last_reg, src=last_index)
    shift = _tile(1, 1)
    nisa.tensor_copy(dst=shift, src=_select(worklist, blocks, last_reg))
    nisa.tensor_scalar(dst=shift, data=shift, op0=nl.add, operand0=-blocks)
    padding = sbm.alloc((1, max(rows, 8)), nl.float32)[:, :rows]
    nisa.tensor_scalar(dst=padding, data=items, op0=nl.equal, operand0=blocks,
                       op1=nl.multiply, operand1=shift)
    nisa.tensor_tensor(dst=items, data1=items, data2=padding, op=nl.add)
    # Zero-fill targets: one strided read of the empty list from ``first_slot``.
    first = _tile(1, 1, nl.int32)
    nisa.tensor_copy(dst=first, src=first_slot)
    first_reg = nisa.register_alloc()
    nisa.register_load(dst=first_reg, src=first)
    nisa.tensor_copy(dst=table[:, :, 1:1 + slots], src=empty_tile.ap(
        pattern=[[empty_tile.shape[1], 1], [slots, rows], [1, slots]], offset=blocks,
        scalar_offset=first_reg, indirect_dim=1))
    iterations = _ceil_div(count, nprograms, sbm)
    nisa.tensor_scalar(dst=iterations, data=iterations, op0=nl.multiply,
                       operand0=nprograms * slots)
    nisa.tensor_tensor(dst=first_slot, data1=first_slot, data2=iterations, op=nl.add)
    sbm.close_scope()
    return table


def _zero_rows(out, zeros, block, start, count):
    """Write zeros to rows [start, start+count) of block ``block`` (a register), in one DMA.

    The source repeats the rows of ``zeros``; an out-of-range block is skipped.
    """
    q, h = out.shape[1], out.shape[2]
    rows = min(zeros.shape[0], count)
    while count % rows:
        rows -= 1
    nisa.dma_copy(dst=_block_ap(out, block, [[h, count], [1, h]], start * h),
                  src=zeros.ap(pattern=[[h, rows], [0, count // rows], [1, h]]),
                  oob_mode=nisa.oob_mode.skip, dge_mode=_HWDGE, engine=_DMA_QUEUE)


def _zero_blocks(out, zeros, empty, first_slot, at_index, slots, sbm):
    """Zero empty-list entries [first + at*slots, +slots), first = ``first_slot``.

    Positions past the list read the out-of-range id, whose DMAs the engine
    skips, so a slot never writes a routed block.
    """
    blocks, q = out.shape[0], out.shape[1]
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
        _zero_rows(out, zeros, target, 0, q)
    sbm.close_scope()


def _block_rows(compute, loaded, zeros, block, width, block_m):
    """Zero rows [width, q) of a block, then compute rows [0, width) in ``block_m`` slices."""
    out = compute[6]
    q = out.shape[1]
    if width < q:
        _zero_rows(out, zeros, block, width, q - width)
    for m0 in range(0, width, block_m):
        _compute_rows(compute, loaded, block, m0, min(block_m, width - m0))


def _item_loop(compute, loads, worklist, zero_fill, width, block_m, sbm):
    """One dynamic loop over a worklist, computing rows [0, width) of each block.

    An iteration reads its block and its ``slots`` zero-fill targets from the
    loop's item table (``_item_table``), loads the block's weights, queues the
    zero fills behind them, then computes.
    """
    weights, scales, experts_row, kstep, nstep = loads
    zeros, empty, first_slot, slots = zero_fill
    out = compute[6]
    blocks, q = out.shape[0], out.shape[1]
    nprograms, program = nl.num_programs(0), nl.program_id(0)
    rows = _table_rows(blocks, nprograms)
    table = _item_table(worklist, empty, first_slot, slots, blocks, rows, sbm)
    boundary = _phase_bounds(worklist[1], program, nprograms, sbm)

    def iteration(at_index):
        sbm.open_scope(name="routing_item")
        entry = sbm.alloc((1, max(1 + slots, 8)), nl.int32)[:, :1 + slots]
        nisa.tensor_copy(dst=entry, src=table.ap(
            pattern=[[rows * (1 + slots), 1], [1, 1 + slots]], scalar_offset=at_index,
            indirect_dim=1))
        block = nisa.register_alloc()
        nisa.register_load(dst=block, src=entry[:, 0:1])
        targets = []
        for slot in range(slots):
            target = nisa.register_alloc()
            nisa.register_load(dst=target, src=entry[:, 1 + slot:2 + slot])
            targets.append(target)
        sbm.close_scope()
        loaded = _expert_operands(weights, scales, experts_row, block, kstep, nstep)
        for target in targets:
            _zero_rows(out, zeros, target, 0, q)
        _block_rows(compute, loaded, zeros, block, width, block_m)
    nl.fori_loop(program, boundary, iteration, step=nprograms)


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
        bounds: FP32 [128,3], gate upper, up lower, up upper, the same on
            every row.
        BLOCK_M/N/K: Compile-time scheduling groups; scaled products stay
            128-granular and accumulate in their original contraction order.
        SKIP_PADDING: Skip empty blocks and rows using device control flow.
            The false setting preserves dense execution for comparisons.

    Notes:
        Padding rows are zero. Routing IDs are read once into per-block spans;
        each routed block runs once, for the row prefix that holds its last
        routed row tile, in the narrowest of a few compile-time row classes, so
        its expert weights load once, as FP8 PE stationaries. Rows past the
        prefix and every empty block are written with zeros beside the
        products, queued behind the block's weights; every output row has
        exactly one writer, so no barrier orders them. The verified production
        geometry merges adjacent whole blocks only when their expert IDs match;
        its 512-row moving tile reuses one set of weights. Both physical
        programs take equal iteration counts; an odd list duplicates its last
        pure output write. Single-row decode retains its original dense expert
        schedule. All routing decisions remain on device, including zero and
        skewed routing.
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
    experts_row = _experts_row(expert_ids)
    zeros = nl.ndarray((min(128, q), h), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zeros, value=0.0)
    compute = (hidden, weights, row_ids, expert_ids, affinity, clamp, out)
    loads = (weights, scales, experts_row, kstep, nstep)

    if not SKIP_PADDING or q == 1:
        _dense_schedule(compute, loads, BLOCK_M)
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
    span = _routing_spans(row_ids, row_step, sbm)
    loops = len(class_bounds) + (1 if merge_pairs else 0)
    paired, classes, empty = _routing_worklists(
        span, experts_row, class_bounds, sbm, merge_pairs=merge_pairs, extra=trailing,
        empty_extra=_table_reach(blocks, nprograms, loops, slots) - blocks - 1)
    first_slot = sbm.alloc((1, 8), nl.float32)[:, :1]
    nisa.memset(dst=first_slot, value=0.0)

    zero_fill = (zeros, empty, first_slot, slots)
    if paired is not None:
        kernel_assert(2 * q <= 512, "Merged moving dimension exceeds 512")
        _item_loop(compute, loads, paired, zero_fill, 2 * q, 2 * q, sbm)
    for c in range(len(class_bounds)):
        _item_loop(compute, loads, classes[c], zero_fill, class_bounds[c] * row_step,
                   BLOCK_M, sbm)

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
