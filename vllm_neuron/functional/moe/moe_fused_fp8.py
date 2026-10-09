# SPDX-License-Identifier: Apache-2.0
"""Shape-specialized, fused block-FP8 experts on Trainium2.

BLOCK_M/N/K are compile-time scheduling choices. The 128x128 weight block and
its scale remain fixed by the checkpoint quantization contract.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.kernel_assert import kernel_assert

# Partitions of an SBUF or PSUM tile (nl.tile_size.pmax), and the edge of every
# FP8 weight block: the checkpoint scales 128x128 blocks, and each block is one
# Tensor Engine operand whose contraction rows fill the partitions.
_PMAX = 128
# Moving free size of one Tensor Engine product (nl.tile_size.gemm_moving_fmax),
# also the FP32 width of one PSUM bank (nl.tile_size.psum_fmax).
_MOVING_FMAX = 512
# Most rows of one chunk. A product streams max(rows, 64) Tensor Engine
# cycles, so a chunk of 64 rows costs the Tensor Engine about what one row does.
_CHUNK_ROWS = 64
# Gate/up weights, block scales and routing tables move through the hardware
# descriptor generator on the Sync engine's queue, and the down weights on the
# Scalar Engine's queue: the split moves about 1 TB/s of 4-panel weight DMAs
# per core (MEASURED). Row ids, row gathers and the output scatter use the
# GpSimd engine's software descriptors, the only ones that take a per-row
# index.
_HWDGE = nisa.dge_mode.hwdge
_SWDGE = nisa.dge_mode.swdge
_DMA_QUEUE = nisa.engine.sync
_DOWN_QUEUE = nisa.engine.scalar
# Hidden tiles interleaved in the transposed rows, [128, H/128/2, rows, 2]. A
# Vector Engine multiply runs in its 2x mode only when the innermost dimension
# of every operand is contiguous, and the block scale varies by hidden tile, so
# the tiles must be innermost; the Tensor Engine reads a moving operand of
# stride 2 at the rate of a contiguous one, and a stride of 4 or more slows it.
_INTERLEAVE = 2
# Free elements of one block-scale instruction: long enough that its issue
# overhead is small, short enough that the products it feeds start early.
_SCALE_COLUMNS = 2048
# PSUM banks of down-projection accumulators that one copy evacuates.
_DOWN_BANKS = 2
# Gate/up accumulators, one PSUM bank each, that consecutive products take in
# turn: a pair's gate and up products, and the previous pair's up bank while
# its SwiGLU reads it.
_GATE_UP_BANKS = 3
# Block-scaled moving operands and down stationaries in turn, so the Vector
# Engine scales ahead of the Tensor Engine.
_STAGED = 4
# FP32 output rows of consecutive chunks in turn: one is stored while the next
# is evacuated.
_RESULTS = 2
# Every PSUM evacuation and activation runs on the Scalar Engine; the Vector
# Engine carries the block scales and the SwiGLU arithmetic.
_EVACUATE = nisa.engine.scalar
# Every SBUF tensor has a fixed address from one SbufManager stack. The
# compiler's own placement reuses one address for the buffers of consecutive
# chunks, and each reuse orders a chunk's writes after its neighbour's reads
# (MEASURED: one address for every down stationary, ~40 us per chunk), so the
# chunks would not overlap.


def _tile(sbm, rows, cols, dtype=nl.float32):
    """Allocate an SBUF tile of shape (rows, cols) from ``sbm``, backed by at least 8 columns."""
    return sbm.alloc_stack((rows, max(cols, 8)), dtype=dtype)[:, :cols]


def _register(src):
    """A register loaded from the [1, 1] SBUF integer ``src``."""
    register = nisa.register_alloc()
    nisa.register_load(dst=register, src=src)
    return register


def _entry(sbm, table, offset, index):
    """A register with column ``offset + index`` of the int32 [1, n] ``table``.

    ``index`` is a register; ``table`` is a whole SBUF tensor, so its partition
    stride is its free size.
    """
    # uint32: the hardware's dynamic SBUF reads take an unsigned offset.
    entry = _tile(sbm, 1, 1, nl.uint32)
    _dynamic_read(entry, table, offset, index)
    return _register(entry)


def _dynamic_read(dst, table, offset, index):
    """Copy column ``offset + index`` of every row of ``table`` to ``dst``.

    ``index`` is a dynamic offset: a register or a [1, 1] SBUF integer.

    The Scalar Engine copies: a dynamic SBUF read runs on the Vector or Scalar
    Engine, and the Vector Engine carries the block scales.
    """
    nisa.tensor_copy(dst=dst, src=table.ap(
        pattern=[[table.shape[1], dst.shape[0]], [1, 1]], offset=offset, scalar_offset=index,
        indirect_dim=1), engine=_EVACUATE)


def _slot_buffers(sbm, weights, kstep, nstep, width):
    """SBUF of one chunk's expert weights and rows: ``(gate_up, down, xt, gated)``.

    ``gate_up[c]`` holds the FP8 gate and up tiles, each [128, I/128, nk, 128],
    of the c-th group of at most BLOCK_K/128 hidden tiles, and ``down[c]`` the
    [128, I/128, nn, 128] tile of the c-th group of at most BLOCK_N/128 hidden
    tiles. ``xt`` takes the transposed rows (_transposed) and ``gated`` the
    FP32 SwiGLU output, [128, I/128, width].
    """
    panels, nh = weights.shape[1], weights.shape[3]
    ni = panels // 3
    gate_up, down = [], []
    for k0 in range(0, nh, kstep):
        nk = min(kstep, nh - k0)
        gate = sbm.alloc_stack((_PMAX, ni, nk, _PMAX), dtype=weights.dtype)
        up = sbm.alloc_stack((_PMAX, ni, nk, _PMAX), dtype=weights.dtype)
        gate_up.append((gate, up))
    for n0 in range(0, nh, nstep):
        down.append(sbm.alloc_stack((_PMAX, ni, min(nstep, nh - n0), _PMAX), dtype=weights.dtype))
    xt = sbm.alloc_stack((_PMAX, nh * width), dtype=nl.bfloat16)
    gated = sbm.alloc_stack((_PMAX, ni, width), dtype=nl.float32)
    return gate_up, down, xt, gated


def _scale_buffers(sbm, weights):
    """SBUF of one chunk's block scales: ``(scale, gate_up_scale)``.

    ``scale`` holds the FP32 scales of the expert on every partition,
    ``[128, 3I/128, H/128]``, of which _load_scales fills the down panels, and
    ``gate_up_scale`` the gate/up scales in BF16, ``[128, 2I/128, H/128]``,
    the operand of the Vector Engine's 2x mode.
    """
    panels, nh = weights.shape[1], weights.shape[3]
    scale = sbm.alloc_stack((_PMAX, panels, nh), dtype=nl.float32)
    gate_up_scale = sbm.alloc_stack((_PMAX, 2 * (panels // 3), nh), dtype=nl.bfloat16)
    return scale, gate_up_scale


def _slot(buffers, scales):
    """One chunk's ``(gate_up, down, scale, gate_up_scale, xt, gated)``.

    ``buffers`` is _slot_buffers and ``scales`` is _scale_buffers.
    """
    return buffers[0], buffers[1], scales[0], scales[1], buffers[2], buffers[3]


def _turn_buffers(sbm, width, nh, ni):
    """Buffers that consecutive chunks take in turn.

    Returns ``(moving, stationary, result, gate_up_banks, transpose_bank,
    down_banks, swiglu)``: _STAGED BF16 [128, H/128 * width] block-scaled
    moving operands, _STAGED BF16 down stationaries [128, I/128, part, width]
    (_down_part), _RESULTS FP32 output rows [width, H/128, 128], the eight
    PSUM banks of a chunk (_GATE_UP_BANKS gate/up accumulators, one bank of
    two transpose groups, and two down accumulators of _DOWN_BANKS banks), and
    two sets of FP32 [128, width] SwiGLU temporaries ``(gate, silu, up)``.
    """
    part = _down_part(ni, nh, width)
    moving, stationary, result, swiglu = [], [], [], []
    for _ in range(_STAGED):
        moving.append(sbm.alloc_stack((_PMAX, nh * width), dtype=nl.bfloat16))
        stationary.append(sbm.alloc_stack((_PMAX, ni, part, width), dtype=nl.bfloat16))
    for _ in range(_RESULTS):
        result.append(sbm.alloc_stack((width, nh, _PMAX), dtype=nl.float32))
    for _ in range(2):
        swiglu.append((_tile(sbm, _PMAX, width), _tile(sbm, _PMAX, width),
                       _tile(sbm, _PMAX, width)))
    gate_up_banks, transpose_bank, down_banks = _psum_banks(width)
    return moving, stationary, result, gate_up_banks, transpose_bank, down_banks, swiglu


def _psum_banks(width):
    """The eight PSUM banks of a chunk: ``(gate_up_banks, transpose_bank, down_banks)``.

    Each buffer has a fixed bank: banks 0 .. _GATE_UP_BANKS - 1 accumulate
    gate/up products, the next holds the transposes, and the last two pairs
    the down products. The compiler's own placement rotates every PSUM buffer
    through all eight banks (MEASURED), so a gate/up product would wait for
    the evacuation of a down bank it happens to share.
    """
    bank = nl.tile_size.psum_bank_fmax_bytes
    kernel_assert(_GATE_UP_BANKS + 1 + 2 * _DOWN_BANKS <= nl.tile_size.psum_num_banks,
                  "The chunk's PSUM buffers must fit the PSUM banks")
    gate_up_banks, down_banks = [], []
    for b in range(_GATE_UP_BANKS):
        gate_up_banks.append(nl.ndarray((_PMAX, _MOVING_FMAX), dtype=nl.float32,
                                        buffer=nl.psum, address=(0, b * bank)))
    transpose_bank = nl.ndarray((_PMAX, 2, _MOVING_FMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                address=(0, _GATE_UP_BANKS * bank))
    for d in range(2):
        down_banks.append(nl.ndarray((width, _DOWN_BANKS, _MOVING_FMAX), dtype=nl.float32,
                                     buffer=nl.psum,
                                     address=(0, (_GATE_UP_BANKS + 1 + d * _DOWN_BANKS) * bank)))
    return gate_up_banks, transpose_bank, down_banks


def _down_part(ni, nh, width):
    """Hidden tiles per down part: one block-scale instruction and one output store."""
    return min(nh, max(1, _SCALE_COLUMNS // (ni * width)))


def _load_expert(weights, scales, expert, slot):
    """Load the FP8 weights and block scales of ``expert`` (a dynamic offset) into ``slot``."""
    _load_scales(scales, expert, slot)
    _load_gate_up(weights, expert, slot)
    _load_down(weights, expert, slot)


def _load_scales(scales, expert, slot):
    """Load the block scales of ``expert`` (a dynamic offset) to every partition of ``slot``.

    ``slot`` is _slot. Two broadcast DMAs on the _DMA_QUEUE: the gate/up
    scales, which the DMA rounds to BF16 (the Vector Engine's 2x-mode
    operand), and the FP32 down scales.
    """
    scale, gate_up_scale = slot[2], slot[3]
    panels, nh = scale.shape[1], scale.shape[2]
    gate_up = gate_up_scale.shape[1] * nh
    nisa.dma_copy(dst=gate_up_scale.reshape((_PMAX, gate_up)), src=scales.ap(
        pattern=[[0, _PMAX], [1, gate_up]], scalar_offset=expert, indirect_dim=0),
        dge_mode=_HWDGE, engine=_DMA_QUEUE)
    nisa.dma_copy(dst=scale.ap(pattern=[[panels * nh, _PMAX], [1, panels * nh - gate_up]],
                               offset=gate_up),
                  src=scales.ap(pattern=[[0, _PMAX], [1, panels * nh - gate_up]], offset=gate_up,
                                scalar_offset=expert, indirect_dim=0),
                  dge_mode=_HWDGE, engine=_DMA_QUEUE)


def _load_gate_up(weights, expert, slot):
    """Load the FP8 gate and up tiles of ``expert`` (a dynamic offset) into ``slot``.

    ``slot`` is _slot. Each tile is one DMA on the _DMA_QUEUE, whose rows are
    the 128-column rows of the 128x128 blocks.
    """
    ni = weights.shape[1] // 3
    h0 = 0
    for tiles in slot[0]:
        _load_panels(weights, expert, 0, tiles[0], h0)
        h0 = _load_panels(weights, expert, ni, tiles[1], h0)


def _load_down(weights, expert, slot):
    """Load the FP8 down tiles of ``expert`` into ``slot``, as _load_gate_up, on the _DOWN_QUEUE."""
    ni = weights.shape[1] // 3
    h0 = 0
    for tile in slot[1]:
        h0 = _load_panels(weights, expert, 2 * ni, tile, h0, _DOWN_QUEUE)


def _load_panels(weights, expert, first, tile, h0, queue=_DMA_QUEUE):
    """One DMA of panels ``first ..`` of ``expert``, hidden tiles ``h0 ..``, into ``tile``.

    ``tile`` is FP8 [128, panels, nk, 128]; ``queue`` issues the DMA. Returns
    the hidden tile after it.
    """
    h = weights.shape[3] * _PMAX
    count, nk = tile.shape[1], tile.shape[2]
    nisa.dma_copy(dst=tile.reshape((_PMAX, count * nk * _PMAX)), src=weights.ap(
        pattern=[[h, _PMAX], [_PMAX * h, count], [1, nk * _PMAX]],
        offset=first * _PMAX * h + h0 * _PMAX, scalar_offset=expert, indirect_dim=0),
        dge_mode=_HWDGE, engine=queue)
    return h0 + nk


def _chunk_width(q, block_m):
    """Rows per chunk: the largest divisor of ``q`` within min(BLOCK_M, _CHUNK_ROWS)."""
    width = min(q, block_m, _CHUNK_ROWS)
    while q % width:
        width -= 1
    return width


def _routed_chunks(sbm, row_ids, width, skip_padding):
    """Flag every chunk that holds a routed row: FP32 [1, blocks * q / width].

    Chunk ``c`` of block ``b`` is item ``b * (q / width) + c``, so an item
    times ``width`` is its first row in the flattened [blocks * q] output.
    Blocks sit on partitions, up to 128 at a time, and one PE transpose per
    chunk column moves their flags to partition 0. Without ``skip_padding``
    every chunk is flagged.
    """
    blocks, q = row_ids.shape
    chunks = q // width
    items = blocks * chunks
    routed = sbm.alloc_stack((1, max(items, 8)), dtype=nl.float32)
    if not skip_padding:
        nisa.memset(dst=routed[:, :items], value=1.0)
        return routed
    for b0 in range(0, blocks, _PMAX):
        count = min(_PMAX, blocks - b0)
        sbm.open_scope(name="routed_chunks")
        ids = _tile(sbm, count, q, nl.int32)
        nisa.dma_copy(dst=ids, src=row_ids[b0:b0 + count, :], dge_mode=_HWDGE,
                      engine=_DMA_QUEUE)
        valid = _tile(sbm, count, q)
        nisa.tensor_scalar(dst=valid, data=ids, op0=nl.greater_equal, operand0=0)
        flags = _tile(sbm, count, chunks)
        nisa.tensor_reduce(dst=flags, data=valid.reshape((count, chunks, width)),
                           op=nl.maximum, axis=(2,))
        for c in range(chunks):
            row = nl.ndarray((1, count), dtype=nl.float32, buffer=nl.psum,
                             address=(0, c % nl.tile_size.psum_num_banks
                                      * nl.tile_size.psum_bank_fmax_bytes))
            nisa.nc_transpose(dst=row, data=flags[:, c:c + 1])
            nisa.tensor_copy(dst=routed.ap(pattern=[[routed.shape[1], 1], [chunks, count]],
                                           offset=b0 * chunks + c), src=row)
        sbm.close_scope()
    return routed


def _item_experts(sbm, experts_row, items, chunks):
    """The expert of every item, int32 [1, items]: block ``b``'s expert ``chunks`` times."""
    blocks = items // chunks
    table = sbm.alloc_stack((1, max(items, 8)), dtype=nl.int32)
    nisa.tensor_copy(
        dst=table.ap(pattern=[[table.shape[1], 1], [chunks, blocks], [1, chunks]]),
        src=experts_row.ap(pattern=[[experts_row.shape[1], 1], [1, blocks], [0, chunks]]))
    return table


def _program_list(sbm, routed, experts_row, experts, items, chunks):
    """List the routed chunks of this program's experts: int32 [1, 2 * items + 1].

    Program ``p`` owns the experts ``e`` with ``e % nprograms == p``. Columns
    [0, items) hold the flags of their chunks; ``nonzero_with_count`` writes
    the list of their items, in item order, from column ``items`` on, padded
    with the out-of-range item ``items`` and ended by the count, which the
    kernel does not read (_list_end). Source and destination are disjoint
    column ranges of one tile, so no placement can alias them.
    """
    blocks = items // chunks
    nprograms, program = nl.num_programs(0), nl.program_id(0)
    listing = sbm.alloc_stack((1, max(2 * items + 1, 8)), dtype=nl.int32)
    sbm.open_scope(name="program_list")
    match = sbm.alloc_stack((1, max(blocks, 8)), dtype=nl.float32)
    nisa.tensor_scalar(dst=match[:, :blocks], data=experts_row[:, :blocks], op0=nl.equal,
                       operand0=program)
    for expert in range(program + nprograms, experts, nprograms):
        nisa.scalar_tensor_tensor(dst=match[:, :blocks], data=experts_row[:, :blocks],
                                  op0=nl.equal, operand0=expert + 0.0, op1=nl.add,
                                  operand1=match[:, :blocks])
    # flags[b, c] = routed[b, c] * match[b], the match repeated over the chunks.
    nisa.tensor_tensor(
        dst=listing.ap(pattern=[[listing.shape[1], 1], [chunks, blocks], [1, chunks]]),
        data1=routed.ap(pattern=[[routed.shape[1], 1], [chunks, blocks], [1, chunks]]),
        data2=match.ap(pattern=[[match.shape[1], 1], [1, blocks], [0, chunks]]),
        op=nl.multiply)
    nisa.nonzero_with_count(dst=listing[:, items:2 * items + 1], src=listing[:, :items],
                            padding_val=items)
    sbm.close_scope()
    return listing


def _list_end(sbm, routed, item_experts, items, slots):
    """The end of the dynamic loop, int32 [1, 1]: the longest program list, at least ``slots``.

    Under LNC2 both cores pass a core barrier at every trip of a device loop,
    so every program must run the same number of trips: a core with more
    trips waits at the barrier for ever (MEASURED: HANG_ON_COMPUTE with lists
    of 12 and 24 entries past the slots). Every program counts every
    program's list, the routed items whose expert ``e`` has
    ``e % nprograms == q`` (_program_list), from the same data and with the
    same instructions, so all programs get the same end. The owner is
    ``e & (nprograms - 1)``: the compiler has no integer remainder operator.
    """
    nprograms = nl.num_programs(0)
    kernel_assert(nprograms & (nprograms - 1) == 0, "The program count must be a power of two")
    end = _tile(sbm, 1, 1, nl.int32)
    sbm.open_scope(name="list_end")
    owner = _tile(sbm, 1, items, nl.int32)
    nisa.tensor_scalar(dst=owner, data=item_experts[:, :items], op0=nl.bitwise_and,
                       operand0=nprograms - 1)
    # tagged = (owner + 1) * routed: the owner plus one, or 0 when unrouted.
    tagged = _tile(sbm, 1, items)
    nisa.scalar_tensor_tensor(dst=tagged, data=owner, op0=nl.add, operand0=1.0,
                              op1=nl.multiply, operand1=routed[:, :items])
    lengths = _tile(sbm, 1, nprograms + 1)
    nisa.memset(dst=lengths[:, nprograms:nprograms + 1], value=float(slots))
    matches = _tile(sbm, 1, items)
    for q in range(nprograms):
        nisa.tensor_scalar_reduce(dst=matches, data=tagged, op0=nl.equal, operand0=q + 1.0,
                                  reduce_op=nl.add, reduce_res=lengths[:, q:q + 1])
    nisa.tensor_reduce(dst=end, data=lengths, op=nl.maximum, axis=(1,))
    sbm.close_scope()
    return end


def _gather_buffers(sbm, width, experts):
    """SBUF of one chunk's gather: ``(ids, bump, resolved, starts, affinities, routing)``.

    ``routing`` takes the rows' routing weights.
    """
    ids, bump = _tile(sbm, width, 1, nl.int32), _tile(sbm, width, 1, nl.int32)
    resolved, starts = _tile(sbm, width, 1, nl.int32), _tile(sbm, width, 1, nl.int32)
    affinities = sbm.alloc_stack((width, max(experts, 8)), dtype=nl.float32)
    return ids, bump, resolved, starts, affinities, _tile(sbm, width, 1)


def _gather_rows(compute, item, rows, gather):
    """Gather the hidden rows of ``item`` into ``rows`` and their affinities into ``gather``.

    ``item`` is a dynamic offset, ``rows`` BF16 [width, H] and ``gather``
    _gather_buffers; _down reads the routing weights from its affinities. An
    invalid id (-1) moves to the last hidden row, which is zero, and its zero
    affinities, so its output row is zero. The GpSimd engine runs every step.
    """
    hidden, row_ids, affinity = compute[0], compute[1], compute[2]
    tokens, h = hidden.shape
    width = rows.shape[0]
    experts = affinity.shape[0] // tokens
    ids, bump, resolved, starts = gather[0], gather[1], gather[2], gather[3]
    affinities = gather[4]
    nisa.dma_copy(dst=ids, src=row_ids.ap(pattern=[[1, width], [1, 1]], scalar_offset=item,
                                          indirect_dim=0), dge_mode=_SWDGE)
    nisa.tensor_scalar(dst=bump, data=ids, op0=nl.less, operand0=0, op1=nl.multiply,
                       operand1=tokens, engine=nisa.engine.gpsimd)
    nisa.tensor_tensor(dst=resolved, data1=ids, data2=bump, op=nl.add,
                       engine=nisa.engine.gpsimd)
    nisa.dma_copy(dst=rows, src=hidden.ap(pattern=[[h, width], [1, h]], vector_offset=resolved,
                                          indirect_dim=0), dge_mode=_SWDGE)
    nisa.tensor_scalar(dst=starts, data=resolved, op0=nl.multiply, operand0=experts,
                       engine=nisa.engine.gpsimd)
    # Each row's affinities for every expert; the expert's column is a dynamic read.
    nisa.dma_copy(dst=affinities[:, :experts], src=affinity.ap(
        pattern=[[1, width], [1, experts]], vector_offset=starts, indirect_dim=0),
        dge_mode=_SWDGE)


def _interleave(nh):
    """Hidden tiles interleaved per column of the transposed rows (_INTERLEAVE or 1)."""
    return _INTERLEAVE if nh % _INTERLEAVE == 0 else 1


def _hidden_tile(tensor, hb, width):
    """Hidden tile ``hb`` [128, width] of an interleaved [128, H/128 * width] tile."""
    g = _interleave(tensor.shape[1] // width)
    return tensor.ap(pattern=[[tensor.shape[1], _PMAX], [g, width]],
                     offset=hb // g * width * g + hb % g)


def _transposed(rows, xt, bank):
    """Transpose the rows [width, H] into ``xt``, BF16 [128, H/128 * width], hidden on partitions.

    Hidden tile ``hb`` of row ``t`` sits at column ``(hb // G * width + t) * G
    + hb % G`` for G = _interleave(H/128); _hidden_tile reads one tile. A group
    of transposes fills one half of the PSUM ``bank`` and one copy evacuates
    it, while the next group fills the other half.
    """
    width = rows.shape[0]
    nh = xt.shape[1] // width
    g = _interleave(nh)
    group = max(g, min(nh, _MOVING_FMAX // width) // g * g)
    half = 0
    for hb0 in range(0, nh, group):
        count = min(group, nh - hb0)
        for j in range(count):
            hb = hb0 + j
            nisa.nc_transpose(dst=bank[:, half, j * width:(j + 1) * width],
                              data=rows[:, hb * _PMAX:(hb + 1) * _PMAX])
        nisa.tensor_copy(dst=xt.ap(pattern=[[nh * width, _PMAX], [width * g, count // g],
                                            [1, g], [g, width]], offset=hb0 * width),
                         src=bank[:, half, 0:count * width], engine=_EVACUATE)
        half = 1 - half


def _gate(pair_gate, clamp, temps):
    """SiLU of the gate PSUM ``pair_gate`` clamped from above, into ``temps``' FP32 silu."""
    gate, silu = temps[0], temps[1]
    nisa.tensor_scalar(dst=gate, data=pair_gate, op0=nl.minimum, operand0=clamp[:, 0:1],
                       engine=nisa.engine.vector)
    nisa.activation(dst=silu, data=gate, op=nl.silu)
    return silu


def _swiglu(gated, temps, pair_up, clamp, ib):
    """Store the SwiGLU of tile ``ib``: ``temps``' silu times the clamped up PSUM ``pair_up``."""
    silu, up = temps[1], temps[2]
    nisa.tensor_scalar(dst=up, data=pair_up, op0=nl.maximum, operand0=clamp[:, 1:2],
                       op1=nl.minimum, operand1=clamp[:, 2:3], engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=gated[:, ib, :], data1=silu, data2=up, op=nl.multiply,
                       engine=nisa.engine.vector)


def _gate_up(compute, slot, turns, width):
    """Gate/up products and SwiGLU of one chunk into ``slot``'s FP32 [128, I/128, width].

    ``slot`` is _slot, holding the chunk's expert and transposed rows;
    ``turns`` is _turn_buffers. Each 128x128 block scale multiplies the BF16
    hidden rows of its product, one Vector Engine instruction per panel in its
    2x mode, so the rows and the scales are both BF16, and a whole contraction
    accumulates in one PSUM bank. Every accumulation group has a PSUM bank of
    its own: the Tensor Engine tracks accumulation per PSUM bank, and the
    scheduler may move a group's products past the first product of a group
    that shares the bank. Pair ``ib``'s gate products start once pair
    ``ib - 1``'s gate is read, and its up products once that pair's up is read,
    so three banks keep the Tensor Engine busy while the Vector Engine scales.
    """
    scale = slot[2]
    accumulators = []
    for ib in range(scale.shape[1] // 3):
        _gate_up_pair(compute, slot, turns, width, ib, accumulators)
    _gate_up_last(compute, slot, turns, accumulators)


def _gate_up_pair(compute, slot, turns, width, ib, accumulators):
    """Pair ``ib`` of _gate_up: its gate and up products, and the SwiGLU of pair ``ib - 1``.

    ``accumulators`` lists the PSUM accumulators of the pairs before ``ib``;
    the pair appends its gate and up accumulators.
    """
    clamp = compute[3]
    gate_up_tiles, scale, gate_up_scale, xt, gated = slot[0], slot[2], slot[3], slot[4], slot[5]
    moving_turns, banks, swiglu = turns[0], turns[3], turns[6]
    nh, panels = scale.shape[2], scale.shape[1]
    ni = panels // 3
    g = _interleave(nh)
    for half in range(2):
        panel = ib + half * ni
        moving = moving_turns[(2 * ib + half) % _STAGED]
        nisa.tensor_tensor(
            dst=moving.reshape((_PMAX, nh // g, width, g)),
            data1=xt.reshape((_PMAX, nh // g, width, g)),
            data2=gate_up_scale.ap(pattern=[[2 * ni * nh, _PMAX], [g, nh // g], [0, width],
                                            [1, g]], offset=panel * nh),
            op=nl.multiply, engine=nisa.engine.vector)
        if ib > 0:
            temps = swiglu[(ib - 1) % 2]
            if half == 0:
                _gate(accumulators[2 * ib - 2], clamp, temps)
            else:
                _swiglu(gated, temps, accumulators[2 * ib - 1], clamp, ib - 1)
        bank = banks[(2 * ib + half) % _GATE_UP_BANKS]
        acc = bank[:, 0:width]
        hb = 0
        for tiles in gate_up_tiles:
            tile = tiles[half]
            for k in range(tile.shape[2]):
                nisa.nc_matmul(dst=acc, stationary=tile[:, ib, k, :],
                               moving=_hidden_tile(moving, hb, width), accumulate=hb > 0)
                hb += 1
        accumulators.append(acc)


def _gate_up_last(compute, slot, turns, accumulators):
    """The SwiGLU of _gate_up's last pair, from its ``accumulators``."""
    scale, gated, clamp = slot[2], slot[5], compute[3]
    ni = scale.shape[1] // 3
    temps = turns[6][(ni - 1) % 2]
    _gate(accumulators[2 * ni - 2], clamp, temps)
    _swiglu(gated, temps, accumulators[2 * ni - 1], clamp, ni - 1)


def _down(compute, slot, gather, expert, base, turns, result, width):
    """Down products of ``slot``'s SwiGLU output into ``result``; store its rows to ``base``.

    ``gather`` is the chunk's _gather_buffers, ``expert`` a dynamic offset and
    ``base`` the int32 [width, 1] output rows of the chunk (_store_bases).
    ``result`` is one of _turn_buffers' FP32 output rows. The FP32 SwiGLU output times the
    FP32 scales of a part of the hidden tiles, rounded once to BF16, is the
    stationary of one 128x128 product, so each hidden tile's contraction
    accumulates in one PSUM bank. The weights stream, which leaves the rows on
    the output partitions, so the FP32 [width, 1] routing weights scale them
    as they leave PSUM. One scatter stores the rows.
    """
    _dynamic_read(gather[5], gather[4], 0, expert)
    for index in range(_down_parts(slot)):
        _down_products(slot, gather, turns, result, width, index)
    _store(compute, result, base)


def _down_parts(slot):
    """Down parts of ``slot``: hidden tiles in groups of _down_part."""
    scale, gated = slot[2], slot[5]
    nh, ni = scale.shape[2], scale.shape[1] // 3
    part = _down_part(ni, nh, gated.shape[2])
    return (nh + part - 1) // part


def _down_products(slot, gather, turns, result, width, index):
    """Part ``index`` of _down: block-scale, multiply and evacuate its hidden tiles.

    The routing weights (``gather``'s last tile) are read before the first part.
    """
    down_tiles, scale, gated = slot[1], slot[2], slot[5]
    stationary_turns, down_banks = turns[1], turns[5]
    nh, panels = scale.shape[2], scale.shape[1]
    ni = panels // 3
    first_tile = down_tiles[0]
    nstep = first_tile.shape[2]
    part = _down_part(ni, nh, width)
    routing = gather[5]
    hb0 = index * part
    count = min(part, nh - hb0)
    # Every part but the last evacuates (part / _DOWN_BANKS) groups, in turn.
    evacuation = index * ((part + _DOWN_BANKS - 1) // _DOWN_BANKS)
    stationary = stationary_turns[index % _STAGED]
    nisa.tensor_tensor(
        dst=stationary[:, :, 0:count, :],
        data1=gated.ap(pattern=[[ni * width, _PMAX], [width, ni], [0, count], [1, width]]),
        data2=scale.ap(pattern=[[panels * nh, _PMAX], [nh, ni], [1, count], [0, width]],
                       offset=2 * ni * nh + hb0),
        op=nl.multiply, engine=nisa.engine.vector)
    for b0 in range(0, count, _DOWN_BANKS):
        banks = min(_DOWN_BANKS, count - b0)
        acc = down_banks[evacuation % 2]
        for j in range(banks):
            hb = hb0 + b0 + j
            tile = down_tiles[hb // nstep]
            for ib in range(ni):
                nisa.nc_matmul(dst=acc[:, j, 0:_PMAX], stationary=stationary[:, ib, b0 + j, :],
                               moving=tile[:, ib, hb % nstep, :], accumulate=ib > 0)
        nisa.activation(dst=result[:, hb0 + b0:hb0 + b0 + banks, :], op=nl.copy,
                        data=acc[:, 0:banks, 0:_PMAX], scale=routing)
        evacuation += 1


def _store(compute, result, target):
    """Scatter the FP32 rows ``result`` [width, H/128, 128] to the output rows ``target``.

    ``target`` is int32 [width, 1] (_store_bases); a row past the output is skipped.
    """
    out = compute[4]
    width, h = result.shape[0], result.shape[1] * result.shape[2]
    nisa.dma_copy(dst=out.ap(pattern=[[h, width], [1, h]], vector_offset=target, indirect_dim=0),
                  src=result.reshape((width, h)), oob_mode=nisa.oob_mode.skip, dge_mode=_SWDGE)


def _store_bases(sbm, entries, width):
    """The output rows of each listed item, on the row partitions: int32 [width, n].

    ``entries`` is an integer [1, n] row of items. Row ``t`` of item ``i`` is
    output row ``i * width + t``, so the out-of-range item ``items`` lies past
    the output. A Vector Engine shuffle copies partition 0 to 32 partitions at
    a time.
    """
    count = entries.shape[1]
    copies = _tile(sbm, width, count, entries.dtype)
    for p0 in range(0, width, 32):
        nisa.nc_stream_shuffle(dst=copies[p0:p0 + min(32, width - p0), :], src=entries,
                               shuffle_mask=[0] * 32)
    # FP32: a per-partition operand of a Vector Engine instruction is FP32.
    rows = _tile(sbm, width, 1)
    nisa.iota(dst=rows, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    bases = _tile(sbm, width, count, nl.int32)
    nisa.tensor_scalar(dst=bases, data=copies, op0=nl.multiply, operand0=width, op1=nl.add,
                       operand1=rows)
    return bases


def _slot_items(sbm, listing, items, slots):
    """The item each of list entries 0 .. slots - 1 reads: uint32 [1, slots].

    An entry past the list's end reads the last item, so its loads stay in
    range; its store (the listed entry itself) writes nothing.
    """
    # uint32: the items index dynamic SBUF reads, which take an unsigned offset.
    read = _tile(sbm, 1, slots, nl.uint32)
    nisa.tensor_scalar(dst=read, data=listing[:, items:items + slots], op0=nl.minimum,
                       operand0=items - 1)
    return read


def _slot_expert(experts, item_experts, read, s):
    """Read the expert of slot ``s``'s item into column ``s`` of the uint32 ``experts``."""
    _dynamic_read(experts[:, s:s + 1], item_experts, 0, read[:, s:s + 1])


def _zero_unrouted(sbm, out, routed, width):
    """Write zeros to every chunk without a routed row, one DMA per chunk.

    A routed chunk's target is the out-of-range item, whose DMA the engine
    skips, so every output row has exactly one writer. Program ``p`` takes
    every ``nprograms``-th chunk from chunk ``p``. A chunk is ``width * H``
    contiguous FP32 words, written from 128 partitions.
    """
    items, columns = out.shape[0], out.shape[1] // _PMAX
    nprograms, program = nl.num_programs(0), nl.program_id(0)
    zeros = sbm.alloc_stack((_PMAX, columns), dtype=nl.float32)
    nisa.memset(dst=zeros, value=0.0)
    index = _tile(sbm, 1, items, nl.int32)
    nisa.iota(dst=index, pattern=[[1, items]], offset=0)
    # target = index + routed * (items - index): the chunk itself, or items.
    away = _tile(sbm, 1, items)
    nisa.tensor_scalar(dst=away, data=index, op0=nl.multiply, operand0=-1,
                       op1=nl.add, operand1=items)
    nisa.tensor_tensor(dst=away, data1=away, data2=routed[:, :items], op=nl.multiply)
    targets = _tile(sbm, 1, items, nl.int32)
    nisa.tensor_tensor(dst=targets, data1=away, data2=index, op=nl.add)
    # The two hardware queues take turns, idle once the products are done.
    queues = (_DMA_QUEUE, _DOWN_QUEUE)
    turn = 0
    for item in range(program, items, nprograms):
        nisa.dma_copy(dst=out.ap(pattern=[[columns, _PMAX], [1, columns]],
                                 scalar_offset=_register(targets[:, item:item + 1]),
                                 indirect_dim=0),
                      src=zeros, oob_mode=nisa.oob_mode.skip, dge_mode=_HWDGE,
                      engine=queues[turn % len(queues)])
        turn += 1


@nki.jit
def moe_fused_fp8_kernel(hidden, weights, scales, row_ids, expert_ids, affinity,
                         bounds, BLOCK_M=_PMAX, BLOCK_N=1024, BLOCK_K=4096,
                         SKIP_PADDING=True):
    """Return FP32 [blocks,q,H] contributions from device routing.

    Args:
        hidden: BF16 [T+1,H], final row zero.
        weights: Prepared finite FP8 [E,3*(I/128),128,H/128,128].
        scales: Matching FP32 [E,3*(I/128),H/128].
        row_ids: int32 [blocks,q], token IDs or -1 (holes are allowed).
        expert_ids: int32 [blocks,1], expert for each block, in [0, E).
        affinity: FP32 [(T+1)*E,1], final E entries zero.
        bounds: FP32 [128,3], gate upper, up lower, up upper, the same on
            every row.
        BLOCK_M: Bounds the rows of one chunk, with _CHUNK_ROWS.
        BLOCK_N/K: Hidden columns per down / gate and up weight DMA; every
            product stays one 128x128 scaled block.
        SKIP_PADDING: Compute only the chunks of a block that hold a routed
            row. The false setting computes every row, for comparisons.

    Notes:
        Padding rows are zero. Each 128x128 block scale multiplies one BF16
        operand of its product, rounded once, and every contraction
        accumulates in FP32 in PSUM, so the result is a tolerance-defined FP32
        sum, not a fixed reference order. Program ``p``
        owns the experts ``e`` with ``e % nprograms == p`` and lists their
        routed chunks of at most _CHUNK_ROWS rows. Each program runs E //
        nprograms slots in line: each loads its chunk's expert weights, by a
        dynamic address, and gathers its rows while the previous slot
        computes, so a short list computes a chunk that it stores nowhere.
        Entries past the slots run in one dynamic loop at the end, which every
        program runs up to the longest program list. Chunks without a routed row
        are written with zeros after the products; every output row has
        exactly one writer. All routing decisions remain on device.
    """
    blocks, q = row_ids.shape
    experts, panels, contraction, nh, channels = weights.shape
    h = hidden.shape[1]
    ni = panels // 3
    kernel_assert(blocks > 0 and q > 0 and experts > 0,
                  "Require positive blocks, row width, and expert count")
    kernel_assert(panels % 3 == 0 and ni > 0 and h == nh * _PMAX,
                  "Packed panels and hidden width disagree")
    kernel_assert(contraction == _PMAX and channels == _PMAX,
                  "FP8 products must retain the 128x128 checkpoint blocks")
    kernel_assert(scales.shape == (experts, panels, nh),
                  "Scale shape must match the packed weights")
    kernel_assert(1 <= BLOCK_M <= _MOVING_FMAX, "BLOCK_M must be in [1,512]")
    kernel_assert(BLOCK_N > 0 and BLOCK_N % _PMAX == 0,
                  "BLOCK_N must be a positive multiple of 128")
    kernel_assert(BLOCK_K > 0 and BLOCK_K % _PMAX == 0,
                  "BLOCK_K must be a positive multiple of 128")
    nstep, kstep = BLOCK_N // _PMAX, BLOCK_K // _PMAX
    nprograms = nl.num_programs(0)
    width = _chunk_width(q, BLOCK_M)
    chunks = q // width
    items = blocks * chunks
    out = nl.ndarray((blocks, q, h), dtype=nl.float32, buffer=nl.shared_hbm)
    out_rows = out.reshape((items * width, h))
    sbm = SbufManager(sb_lower_bound=0, sb_upper_bound=nl.tile_size.sbuf_fmax_bytes)
    sbm.open_scope(name="moe_fused_fp8")
    clamp = sbm.alloc_stack((_PMAX, 3), dtype=nl.float32)
    nisa.dma_copy(dst=clamp, src=bounds, dge_mode=_HWDGE, engine=_DMA_QUEUE)
    experts_row = sbm.alloc_stack((1, max(blocks, 8)), dtype=nl.int32)
    nisa.dma_copy(dst=experts_row[:, :blocks], src=expert_ids.reshape((1, blocks)),
                  dge_mode=_HWDGE, engine=_DMA_QUEUE)
    routed = _routed_chunks(sbm, row_ids, width, SKIP_PADDING)
    item_experts = _item_experts(sbm, experts_row, items, chunks)
    listing = _program_list(sbm, routed, experts_row, experts, items, chunks)
    compute = (hidden, row_ids.reshape((items, width)), affinity, clamp, out_rows)

    # The same slot count on every program, so every program starts its dynamic
    # loop at the same entry (_list_end).
    slots = min(experts // nprograms, items)
    end = _list_end(sbm, routed, item_experts, items, slots)
    tokens = hidden.shape[0]
    affinity_experts = affinity.shape[0] // tokens
    # Slot ``s`` uses the expert buffers and rows ``s % 2``, and the scales and
    # gather ``s % 3``. Its rows, scales and gate/up weights load two slots
    # ahead, each as soon as slot ``s - 2`` has read that buffer, and its down
    # weights one slot ahead, after slot ``s - 2``'s down products; it
    # transposes its rows after slot ``s - 1``'s gate/up products, so they
    # are ready before the Vector Engine needs them. Slot ``s``'s down parts
    # interleave with slot ``s + 1``'s gate/up pairs, so one slot's down
    # projection overlaps the next slot's gate/up projection, and no slot's
    # buffer is one its neighbour still reads.
    sbm.open_scope(name="chunks")
    turns = _turn_buffers(sbm, width, nh, ni)
    buffers, scale_sets, rows, gathers = [], [], [], []
    for _ in range(2):
        buffers.append(_slot_buffers(sbm, weights, kstep, nstep, width))
        rows.append(sbm.alloc_stack((width, h), dtype=nl.bfloat16))
    for _ in range(3):
        scale_sets.append(_scale_buffers(sbm, weights))
        gathers.append(_gather_buffers(sbm, width, affinity_experts))
    expert_slots = []
    for s in range(6):
        expert_slots.append(_slot(buffers[s % 2], scale_sets[s % 3]))
    # The slots' items, experts and store rows stay in SBUF: each dynamic
    # address loads its register on the engine that issues it. A slot's
    # expert is read just before its loads, so the first loads wait for one
    # read only.
    if slots > 0:
        read = _slot_items(sbm, listing, items, slots)
        slot_experts = _tile(sbm, 1, slots, nl.uint32)
        bases = _store_bases(sbm, listing[:, items:items + slots], width)
    for s in range(min(2, slots)):
        _slot_expert(slot_experts, item_experts, read, s)
        expert = slot_experts[:, s:s + 1]
        _load_scales(scales, expert, expert_slots[s])
        _load_gate_up(weights, expert, expert_slots[s])
        if s == 0:
            _load_down(weights, expert, expert_slots[0])
        _gather_rows(compute, read[:, s:s + 1], rows[s], gathers[s])
    for s in range(min(2, slots)):
        _transposed(rows[s], expert_slots[s][4], turns[4])
    if slots > 0:
        _gate_up(compute, expert_slots[0], turns, width)
    for s in range(slots):
        if s + 1 < slots:
            _load_down(weights, slot_experts[:, s + 1:s + 2], expert_slots[(s + 1) % 6])
        if s + 2 < slots:
            ahead = expert_slots[(s + 2) % 6]
            _slot_expert(slot_experts, item_experts, read, s + 2)
            _load_scales(scales, slot_experts[:, s + 2:s + 3], ahead)
            _load_gate_up(weights, slot_experts[:, s + 2:s + 3], ahead)
            _gather_rows(compute, read[:, s + 2:s + 3], rows[s % 2], gathers[(s + 2) % 3])
        current, upcoming = expert_slots[s % 6], expert_slots[(s + 1) % 6]
        result = turns[2][s % _RESULTS]
        _dynamic_read(gathers[s % 3][5], gathers[s % 3][4], 0, slot_experts[:, s:s + 1])
        accumulators = []
        parts = _down_parts(current)
        for k in range(max(ni, parts)):
            if s + 1 < slots and k < ni:
                _gate_up_pair(compute, upcoming, turns, width, k, accumulators)
            if k < parts:
                _down_products(current, gathers[s % 3], turns, result, width, k)
        if s + 1 < slots:
            _gate_up_last(compute, upcoming, turns, accumulators)
        _store(compute, result, bases[:, s:s + 1])
        if s + 2 < slots:
            _transposed(rows[s % 2], expert_slots[(s + 2) % 6][4], turns[4])

    # Entries past the slots reuse the first slot's buffers, one at a time, up
    # to the longest program list.
    def later(index):
        # uint32: the entry indexes dynamic SBUF reads, which take an unsigned offset.
        entry = _tile(sbm, 1, 1, nl.uint32)
        _dynamic_read(entry, listing, items, index)
        # An entry past this program's list loads the last item, as a slot does
        # (_slot_items); its store (the listed entry itself) writes nothing.
        read = _tile(sbm, 1, 1, nl.uint32)
        nisa.tensor_scalar(dst=read, data=entry, op0=nl.minimum, operand0=items - 1)
        item = _register(read)
        expert = _entry(sbm, item_experts, 0, item)
        slot = expert_slots[0]
        gate_up_banks, transpose_bank, down_banks = _psum_banks(width)
        loop_turns = (turns[0], turns[1], turns[2], gate_up_banks, transpose_bank, down_banks,
                      turns[6])
        _load_expert(weights, scales, expert, slot)
        _gather_rows(compute, item, rows[0], gathers[0])
        _transposed(rows[0], slot[4], transpose_bank)
        _gate_up(compute, slot, loop_turns, width)
        _down(compute, slot, gathers[0], expert, _store_bases(sbm, entry, width), loop_turns,
              turns[2][0], width)
    nl.fori_loop(slots, _register(end), later)
    # The zero fill reuses the chunks' SBUF.
    sbm.close_scope()
    _zero_unrouted(sbm, out.reshape((items, width * h)), routed, width)
    sbm.close_scope()
    return out
