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


def _tile(rows, cols, dtype=nl.float32):
    """Allocate an SBUF tile of shape (rows, cols), backed by at least 8 columns."""
    return nl.ndarray((rows, max(cols, 8)), dtype=dtype, buffer=nl.sbuf)[:, :cols]


def _transpose(dst, src):
    """Transpose src into dst through a PSUM round-trip."""
    psum = nl.ndarray(dst.shape, dtype=dst.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=psum, data=src)
    nisa.tensor_copy(dst=dst, src=psum)


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
    )
    return tile


def _compute_rows(hidden, weights, affinity, row_ids, expert_ids, out,
                  clamp, scale, expert_reg, block, m0, m, nstep, kstep,
                  static_block=-1, expert_block=None):
    """Preserve the original 128-product, scale, and contraction order.

    Arithmetic buffers retain their original compiler automatic allocation.
    New routing buffers use managed scopes separately from this legacy body.
    """
    blocks, q = row_ids.shape
    experts, panels, contraction, nh, channels = weights.shape
    h = hidden.shape[1]
    ni = panels // 3
    x = nl.ndarray((128, nh, m), dtype=nl.bfloat16, buffer=nl.sbuf)
    routing = []
    for t0 in range(0, m, 128):
        rows = min(128, m - t0)
        ids = _tile(rows, 1, nl.int32)
        if static_block >= 0:
            nisa.dma_copy(dst=ids, src=row_ids.ap(
                pattern=[[1, rows], [1, 1]], offset=static_block * q + m0 + t0))
        else:
            nisa.dma_copy(dst=ids, src=row_ids.ap(
                pattern=[[1, rows], [1, 1]], offset=m0 + t0,
                scalar_offset=block, indirect_dim=0))
        invalid = _tile(rows, 1, nl.int32)
        nisa.tensor_scalar(dst=invalid, data=ids, op0=nl.less, operand0=0)
        bump = _tile(rows, 1, nl.int32)
        nisa.tensor_scalar(dst=bump, data=invalid, op0=nl.multiply, operand0=hidden.shape[0])
        resolved = _tile(rows, 1, nl.int32)
        nisa.tensor_tensor(dst=resolved, data1=ids, data2=bump, op=nl.add)
        gathered = nl.ndarray((rows, h), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=gathered, src=hidden.ap(
            pattern=[[h, rows], [1, h]], vector_offset=resolved, indirect_dim=0))
        for hb in range(nh):
            _transpose(x[:, hb, t0:t0 + rows], gathered[:, hb * 128:(hb + 1) * 128])
        expert_rows = _tile(rows, 1, nl.int32)
        if static_block >= 0:
            nisa.dma_copy(dst=expert_rows, src=expert_ids.ap(
                pattern=[[0, rows], [1, 1]], offset=static_block))
        else:
            nisa.dma_copy(dst=expert_rows, src=expert_ids.ap(
                pattern=[[0, rows], [1, 1]], offset=0,
                scalar_offset=block if expert_block is None else expert_block,
                indirect_dim=0))
        affinity_rows = _tile(rows, 1, nl.int32)
        nisa.tensor_scalar(dst=affinity_rows, data=resolved, op0=nl.multiply, operand0=experts)
        nisa.tensor_tensor(dst=affinity_rows, data1=affinity_rows, data2=expert_rows, op=nl.add)
        route_rows = _tile(rows, 1)
        nisa.dma_copy(dst=route_rows, src=affinity.ap(
            pattern=[[1, rows], [1, 1]], vector_offset=affinity_rows, indirect_dim=0))
        routing.append(route_rows)

    gate_up = nl.ndarray((128, 2 * ni, m), dtype=nl.float32, buffer=nl.sbuf)
    for n0 in range(0, 2 * ni, nstep):
        nn = min(nstep, 2 * ni - n0)
        for k0 in range(0, nh, kstep):
            nk = min(kstep, nh - k0)
            gate_weights = []
            for n in range(nn):
                gate_weights.append(_weight_panel(weights, expert_reg, n0 + n, k0, nk))
            for k in range(nk):
                hb = k0 + k
                for n in range(nn):
                    panel = n0 + n
                    partial = nl.ndarray((128, m), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=partial, stationary=gate_weights[n][:, k, :],
                                   moving=x[:, hb, :], accumulate=False)
                    if hb == 0:
                        nisa.tensor_scalar(dst=gate_up[:, panel, :], data=partial,
                                           op0=nl.multiply, operand0=scale[:, panel, hb:hb + 1])
                    else:
                        nisa.scalar_tensor_tensor(dst=gate_up[:, panel, :], data=partial,
                            op0=nl.multiply, operand0=scale[:, panel, hb:hb + 1],
                            op1=nl.add, operand1=gate_up[:, panel, :])
    activation = nl.ndarray((128, ni, m), dtype=nl.bfloat16, buffer=nl.sbuf)
    for ib in range(ni):
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

    result = nl.ndarray((128, nh, m), dtype=nl.float32, buffer=nl.sbuf)
    for n0 in range(0, nh, nstep):
        nn = min(nstep, nh - n0)
        for k0 in range(0, ni, kstep):
            nk = min(kstep, ni - k0)
            for k in range(nk):
                ib = k0 + k
                w = _weight_panel(weights, expert_reg, 2 * ni + ib, n0, nn)
                for n in range(nn):
                    hb = n0 + n
                    partial = nl.ndarray((128, m), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=partial, stationary=w[:, n, :],
                                   moving=activation[:, ib, :], accumulate=False)
                    if ib == 0:
                        nisa.tensor_scalar(dst=result[:, hb, :], data=partial,
                            op0=nl.multiply, operand0=scale[:, 2 * ni + ib, hb:hb + 1])
                    else:
                        nisa.scalar_tensor_tensor(dst=result[:, hb, :], data=partial,
                            op0=nl.multiply, operand0=scale[:, 2 * ni + ib, hb:hb + 1],
                            op1=nl.add, operand1=result[:, hb, :])
    for t0 in range(0, m, 128):
        rows = min(128, m - t0)
        output_rows = nl.ndarray((rows, h), dtype=nl.float32, buffer=nl.sbuf)
        for hb in range(nh):
            _transpose(output_rows[:, hb * 128:(hb + 1) * 128], result[:, hb, t0:t0 + rows])
        nisa.tensor_scalar(dst=output_rows, data=output_rows, op0=nl.multiply,
                           operand0=routing[t0 // 128])
        if static_block >= 0:
            nisa.dma_copy(dst=out[static_block, m0 + t0:m0 + t0 + rows, :],
                          src=output_rows)
        else:
            nisa.dma_copy(dst=out.ap(
                pattern=[[h, rows], [1, h]], offset=(m0 + t0) * h,
                scalar_offset=block, indirect_dim=0), src=output_rows)


def _routing_counts(row_ids, row_step, sbm):
    """Read routing once; retain small counts and release ID temporaries."""
    blocks, q = row_ids.shape
    ntiles = q // row_step
    tile_counts = sbm.alloc((1, blocks, ntiles), nl.int32)
    block_counts = sbm.alloc((1, max(blocks, 8)), nl.int32)[:, :blocks]
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
    nisa.tensor_reduce(dst=block_counts, data=tile_counts, op=nl.add, axis=(2,))
    return tile_counts, block_counts


def _expert_operands(expert_ids, scales, block, panels, nh, static_block=-1):
    expert = _tile(1, 1, nl.int32)
    if static_block >= 0:
        nisa.dma_copy(dst=expert, src=expert_ids[static_block:static_block + 1, :])
    else:
        nisa.dma_copy(dst=expert, src=expert_ids.ap(
            pattern=[[1, 1], [1, 1]], scalar_offset=block, indirect_dim=0))
    expert_reg = nisa.register_alloc()
    nisa.register_load(dst=expert_reg, src=expert)
    scale = nl.ndarray((128, panels, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=scale.reshape((128, panels * nh)),
        src=scales.ap(pattern=[[0, 128], [1, panels * nh]],
                      scalar_offset=expert_reg, indirect_dim=0),
    )
    return expert_reg, scale


def _dense_schedule(hidden, weights, scales, row_ids, expert_ids, affinity,
                    out, clamp, block_m, nstep, kstep):
    """Retain the original static block schedule and arithmetic."""
    blocks, q = row_ids.shape
    panels, nh = weights.shape[1], weights.shape[3]
    for static_block in range(nl.program_id(0), blocks, nl.num_programs(0)):
        expert_reg, scale = _expert_operands(
            expert_ids, scales, None, panels, nh, static_block=static_block
        )
        for m0 in range(0, q, block_m):
            m = min(block_m, q - m0)
            _compute_rows(hidden, weights, affinity, row_ids, expert_ids,
                          out, clamp, scale, expert_reg, None, m0, m,
                          nstep, kstep, static_block=static_block)


def _phase_bounds(total, program, nprograms, sbm):
    """Round the dense prefix to an equal number of iterations on both cores."""
    sbm.open_scope(name="routing_phase_bounds")
    pairs = sbm.alloc((1, 8), nl.float32)[:, :1]
    boundary = sbm.alloc((1, 8), nl.int32)[:, :1]
    # Float-to-int tensor_copy uses round-to-nearest-even. For P=2, integral
    # counts become k+.25 (even) or k+.75 (odd), yielding ceil(count/2)
    # without a .5 tie. P=1 keeps the exact integral count.
    nisa.tensor_scalar(dst=pairs, data=total, op0=nl.multiply,
                       operand0=1.0 / nprograms, op1=nl.add,
                       operand1=(nprograms - 1) * 0.25)
    nisa.tensor_copy(dst=boundary, src=pairs)
    nisa.tensor_scalar(dst=boundary, data=boundary, op0=nl.multiply,
                       operand0=nprograms, op1=nl.add, operand1=program)
    boundary_reg = nisa.register_alloc()
    nisa.register_load(dst=boundary_reg, src=boundary)
    sbm.close_scope()
    return boundary_reg


def _routing_worklists(tile_counts, block_counts, expert_ids, q, sbm,
                       merge_pairs=False):
    """Compact dense blocks, sparse blocks, and active sparse row tiles."""
    blocks, ntiles = tile_counts.shape[1], tile_counts.shape[2]
    paired, paired_count = None, None
    if merge_pairs:
        paired = sbm.alloc((1, blocks + 1), nl.int32)
        paired_count = sbm.alloc((1, 8), nl.float32)[:, :1]
    dense = sbm.alloc((1, blocks + 1), nl.int32)
    sparse = sbm.alloc((1, blocks + 1), nl.int32)
    tiles = sbm.alloc((1, blocks * ntiles + 1), nl.int32)
    dense_count = sbm.alloc((1, 8), nl.float32)[:, :1]
    sparse_count = sbm.alloc((1, 8), nl.float32)[:, :1]
    tile_count = sbm.alloc((1, 8), nl.float32)[:, :1]
    sbm.open_scope(name="routing_worklist_flags")
    dense_flags = sbm.alloc((1, max(blocks, 8)), nl.int32)[:, :blocks]
    sparse_flags = sbm.alloc((1, max(blocks, 8)), nl.int32)[:, :blocks]
    tile_flags = sbm.alloc((1, blocks, ntiles), nl.int32)
    nisa.tensor_scalar(dst=dense_flags, data=block_counts,
                       op0=nl.greater, operand0=q // 2)
    nisa.tensor_scalar(dst=sparse_flags, data=block_counts,
                       op0=nl.less_equal, operand0=q // 2)
    nisa.tensor_scalar(dst=tile_flags, data=tile_counts,
                       op0=nl.greater, operand0=0)
    nisa.tensor_tensor(dst=tile_flags, data1=tile_flags,
                       data2=sparse_flags.ap(
                           pattern=[[max(blocks, 8), 1], [1, blocks],
                                    [0, ntiles]]),
                       op=nl.multiply)
    if merge_pairs:
        # Greedily pair consecutive dense blocks for the same expert. All PNCs
        # build the same non-overlapping list and keep original output positions.
        pair_flags = sbm.alloc((1, max(blocks, 8)), nl.int32)[:, :blocks]
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
                               data2=dense_flags[:, :blocks - 1], op=nl.multiply)
            nisa.tensor_tensor(dst=eligible, data1=eligible,
                               data2=dense_flags[:, 1:blocks], op=nl.multiply)
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
            nisa.tensor_tensor(dst=dense_flags, data1=dense_flags,
                               data2=members, op=nl.subtract)
        nisa.nonzero_with_count(dst=paired, src=pair_flags)
        nisa.tensor_copy(dst=paired_count, src=paired[:, blocks:blocks + 1])
    nisa.nonzero_with_count(dst=dense, src=dense_flags)
    nisa.nonzero_with_count(dst=sparse, src=sparse_flags)
    nisa.nonzero_with_count(dst=tiles,
                           src=tile_flags.reshape((1, blocks * ntiles)))
    nisa.tensor_copy(dst=dense_count, src=dense[:, blocks:blocks + 1])
    nisa.tensor_copy(dst=sparse_count, src=sparse[:, blocks:blocks + 1])
    nisa.tensor_copy(dst=tile_count,
                     src=tiles[:, blocks * ntiles:blocks * ntiles + 1])
    sbm.close_scope()
    return paired, paired_count, dense, sparse, tiles, dense_count, sparse_count, tile_count


def _work_item(worklist, count, at_index, sbm):
    """Select one compacted item; duplicate the last item for an odd pair."""
    sbm.open_scope(name="routing_work_item")
    index = sbm.alloc((1, 8), nl.int32)[:, :1]
    last = sbm.alloc((1, 8), nl.float32)[:, :1]
    item = sbm.alloc((1, 8), nl.int32)[:, :1]
    nisa.register_store(dst=index, src=at_index)
    nisa.tensor_scalar(dst=last, data=count, op0=nl.add, operand0=-1)
    nisa.tensor_scalar(dst=index, data=index, op0=nl.minimum, operand0=last)
    position = nisa.register_alloc()
    nisa.register_load(dst=position, src=index)
    nisa.tensor_copy(dst=item, src=worklist.ap(
        pattern=[[worklist.shape[1], 1], [1, 1]],
        scalar_offset=position, indirect_dim=1))
    item_reg = nisa.register_alloc()
    nisa.register_load(dst=item_reg, src=item)
    sbm.close_scope()
    return item_reg


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
        SKIP_PADDING: Skip empty blocks and row tiles using device control flow.
            The false setting preserves dense execution for comparisons.

    Notes:
        Padding rows are initialized to zero. Small tiles avoid calculating a
        whole 256-row expert block for an expert that receives only a few tokens.
        Blocks over half full retain wide tiles and their weight reuse. Routing
        IDs are read once into tile counts. Device worklists retain original
        output block positions. The verified production geometry merges adjacent
        dense blocks only when their expert IDs match; its 512-row moving tile
        reuses one set of weights. Remaining dense blocks use the original width,
        and active sparse row tiles run in a separate compact loop. Both physical
        programs take equal iteration counts; an odd list duplicates its last
        pure output write. Sparse blocks are zeroed before a shared-output barrier
        and sparse tile computation, including holes inside a block.
        Single-row decode retains its original dense expert schedule.
        All routing decisions remain on device, including zero and skewed routing.
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
    # Other callers retain the same compact single-block and row-tile schedule.
    merge_pairs = (h == 4096 and ni * 128 == 512 and q == 256
                   and BLOCK_M == 256 and BLOCK_N == 4096 and BLOCK_K == 4096
                   and nprograms == 2)
    row_step = min(BLOCK_M, 32)
    while q % row_step:
        row_step -= 1
    sbm = create_auto_alloc_manager()
    sbm.open_scope(name="routing")
    tile_counts, block_counts = _routing_counts(row_ids, row_step, sbm)
    paired, paired_count, dense, sparse, tiles, dense_count, sparse_count, tile_count = _routing_worklists(
        tile_counts, block_counts, expert_ids, q, sbm,
        merge_pairs=merge_pairs,
    )
    boundary = _phase_bounds(dense_count, program, nprograms, sbm)
    tiles_per_block = q // row_step
    tile_rows = row_ids.reshape((blocks * tiles_per_block, row_step))
    tile_out = out.reshape((blocks * tiles_per_block, row_step, h))
    tile_blocks = sbm.alloc((1, max(blocks * tiles_per_block, 8)), nl.int32)
    nisa.iota(dst=tile_blocks[:, :blocks * tiles_per_block].reshape(
        (1, blocks, tiles_per_block)),
        pattern=[[1, blocks], [0, tiles_per_block]])

    if merge_pairs:
        kernel_assert(2 * q <= 512, "Merged moving dimension exceeds 512")

        def process_paired(at_block):
            block = _work_item(paired, paired_count, at_block, sbm)
            expert_reg, scale = _expert_operands(
                expert_ids, scales, block, panels, nh
            )
            _compute_rows(hidden, weights, affinity, row_ids, expert_ids,
                          out, clamp, scale, expert_reg, block, 0, 2 * q,
                          nstep, kstep)
        pair_boundary = _phase_bounds(paired_count, program, nprograms, sbm)
        nl.fori_loop(program, pair_boundary, process_paired, step=nprograms)

    def process_dense(at_block):
        block = _work_item(dense, dense_count, at_block, sbm)
        expert_reg, scale = _expert_operands(
            expert_ids, scales, block, panels, nh
        )
        for m0 in range(0, q, BLOCK_M):
            m = min(BLOCK_M, q - m0)
            _compute_rows(hidden, weights, affinity, row_ids, expert_ids,
                          out, clamp, scale, expert_reg, block, m0, m,
                          nstep, kstep)
    nl.fori_loop(program, boundary, process_dense, step=nprograms)

    zeros = sbm.alloc((128, h), nl.float32)
    nisa.memset(dst=zeros, value=0.0)

    def zero_sparse(at_block):
        block = _work_item(sparse, sparse_count, at_block, sbm)
        for start in range(0, q, 128):
            rows = min(128, q - start)
            nisa.dma_copy(dst=out.ap(
                pattern=[[h, rows], [1, h]], offset=start * h,
                scalar_offset=block, indirect_dim=0), src=zeros[:rows, :])
    zero_boundary = _phase_bounds(sparse_count, program, nprograms, sbm)
    nl.fori_loop(program, zero_boundary, zero_sparse, step=nprograms)
    if nprograms == 2:
        nisa.core_barrier(data=out, cores=(0, 1))

    def process_sparse(at_tile):
        linear = _work_item(tiles, tile_count, at_tile, sbm)
        sbm.open_scope(name="sparse_tile")
        block_value = sbm.alloc((1, 8), nl.int32)[:, :1]
        nisa.tensor_copy(dst=block_value, src=tile_blocks.ap(
            pattern=[[max(blocks * tiles_per_block, 8), 1], [1, 1]],
            scalar_offset=linear, indirect_dim=1))
        block = nisa.register_alloc()
        nisa.register_load(dst=block, src=block_value)
        expert_reg, scale = _expert_operands(
            expert_ids, scales, block, panels, nh
        )
        _compute_rows(hidden, weights, affinity, tile_rows, expert_ids,
                      tile_out, clamp, scale, expert_reg, linear, 0,
                      row_step, nstep, kstep, expert_block=block)
        sbm.close_scope()
    sparse_boundary = _phase_bounds(tile_count, program, nprograms, sbm)
    nl.fori_loop(program, sparse_boundary, process_sparse, step=nprograms)
    sbm.close_scope()
    return out
