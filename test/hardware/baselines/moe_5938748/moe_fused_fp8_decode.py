# SPDX-License-Identifier: Apache-2.0
"""Single-token GLM expert contributions for the measured production geometry.

The public wrapper selects this entry only for its exact shape/tile contract.
Stable active compaction retains original output slots and scaled-product order.
The original compiler-assigned ``nl.ndarray`` allocations are preserved together
with the verified arithmetic: changing allocator lifetimes would require new
compiler, numerical, and latency gates. The measured kernel has no SBUF spills.
"""

import nki
import nki.isa as nisa
import nki.language as nl

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


def _compute_block(hidden, weights, scales, row_ids, expert_ids, affinity, out, clamp, block, nstep, kstep):
    q = row_ids.shape[1]
    experts = weights.shape[0]
    panels, nh = weights.shape[1], weights.shape[3]
    h, ni, BLOCK_M = hidden.shape[1], panels // 3, 1
    expert = _tile(1, 1, nl.int32)
    nisa.dma_copy(dst=expert, src=expert_ids.ap(pattern=[[1, 1], [1, 1]], scalar_offset=block, indirect_dim=0))
    expert_reg = nisa.register_alloc()
    nisa.register_load(dst=expert_reg, src=expert)
    scale = nl.ndarray((128, panels, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale.reshape((128, panels * nh)), src=scales.ap(pattern=[[0, 128], [1, panels * nh]], scalar_offset=expert_reg, indirect_dim=0))
    for m0 in range(0, q, BLOCK_M):
        m = min(BLOCK_M, q - m0)
        x = nl.ndarray((128, nh, m), dtype=nl.bfloat16, buffer=nl.sbuf)
        routing = []
        for t0 in range(0, m, 128):
            rows = min(128, m - t0)
            ids = _tile(rows, 1, nl.int32)
            nisa.dma_copy(dst=ids, src=row_ids.ap(pattern=[[1, rows], [1, 1]], offset=m0 + t0, scalar_offset=block, indirect_dim=0))
            invalid = _tile(rows, 1, nl.int32)
            nisa.tensor_scalar(dst=invalid, data=ids, op0=nl.less, operand0=0)
            bump = _tile(rows, 1, nl.int32)
            nisa.tensor_scalar(dst=bump, data=invalid, op0=nl.multiply, operand0=hidden.shape[0])
            resolved = _tile(rows, 1, nl.int32)
            nisa.tensor_tensor(dst=resolved, data1=ids, data2=bump, op=nl.add)
            gathered = nl.ndarray((rows, h), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=gathered, src=hidden.ap(pattern=[[h, rows], [1, h]], vector_offset=resolved, indirect_dim=0))
            for hb in range(nh):
                _transpose(x[:, hb, t0:t0 + rows], gathered[:, hb * 128:(hb + 1) * 128])
            expert_rows = _tile(rows, 1, nl.int32)
            nisa.dma_copy(dst=expert_rows, src=expert_ids.ap(pattern=[[0, rows], [1, 1]], scalar_offset=block, indirect_dim=0))
            affinity_rows = _tile(rows, 1, nl.int32)
            nisa.tensor_scalar(dst=affinity_rows, data=resolved, op0=nl.multiply, operand0=experts)
            nisa.tensor_tensor(dst=affinity_rows, data1=affinity_rows, data2=expert_rows, op=nl.add)
            route_rows = _tile(rows, 1)
            nisa.dma_copy(dst=route_rows, src=affinity.ap(pattern=[[1, rows], [1, 1]], vector_offset=affinity_rows, indirect_dim=0))
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
                        nisa.nc_matmul(dst=partial, stationary=gate_weights[n][:, k, :], moving=x[:, hb, :], accumulate=False)
                        if hb == 0:
                            nisa.tensor_scalar(dst=gate_up[:, panel, :], data=partial, op0=nl.multiply, operand0=scale[:, panel, hb:hb + 1])
                        else:
                            nisa.scalar_tensor_tensor(dst=gate_up[:, panel, :], data=partial, op0=nl.multiply, operand0=scale[:, panel, hb:hb + 1], op1=nl.add, operand1=gate_up[:, panel, :])
        activation = nl.ndarray((128, ni, m), dtype=nl.bfloat16, buffer=nl.sbuf)
        for ib in range(ni):
            gate, up = (_tile(128, m), _tile(128, m))
            nisa.tensor_scalar(dst=gate, data=gate_up[:, ib, :], op0=nl.minimum, operand0=clamp[:, 0:1])
            nisa.tensor_scalar(dst=up, data=gate_up[:, ni + ib, :], op0=nl.maximum, operand0=clamp[:, 1:2])
            nisa.tensor_scalar(dst=up, data=up, op0=nl.minimum, operand0=clamp[:, 2:3])
            sigmoid, silu, gated = (_tile(128, m), _tile(128, m), _tile(128, m))
            nisa.activation(dst=sigmoid, data=gate, op=nl.sigmoid)
            nisa.scalar_tensor_tensor(dst=silu, data=gate, op0=nl.multiply, operand0=1.0, op1=nl.multiply, operand1=sigmoid)
            nisa.scalar_tensor_tensor(dst=gated, data=silu, op0=nl.multiply, operand0=1.0, op1=nl.multiply, operand1=up)
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
                        nisa.nc_matmul(dst=partial, stationary=w[:, n, :], moving=activation[:, ib, :], accumulate=False)
                        if ib == 0:
                            nisa.tensor_scalar(dst=result[:, hb, :], data=partial, op0=nl.multiply, operand0=scale[:, 2 * ni + ib, hb:hb + 1])
                        else:
                            nisa.scalar_tensor_tensor(dst=result[:, hb, :], data=partial, op0=nl.multiply, operand0=scale[:, 2 * ni + ib, hb:hb + 1], op1=nl.add, operand1=result[:, hb, :])
        for t0 in range(0, m, 128):
            rows = min(128, m - t0)
            output_rows = nl.ndarray((rows, h), dtype=nl.float32, buffer=nl.sbuf)
            for hb in range(nh):
                _transpose(output_rows[:, hb * 128:(hb + 1) * 128], result[:, hb, t0:t0 + rows])
            nisa.tensor_scalar(dst=output_rows, data=output_rows, op0=nl.multiply, operand0=routing[t0 // 128])
            nisa.dma_copy(dst=out.ap(pattern=[[h, rows], [1, h]], offset=(m0 + t0) * h, scalar_offset=block, indirect_dim=0), src=output_rows)


def _active_order(row_ids):
    blocks = row_ids.shape[0]
    ids = nl.ndarray((1, max(blocks, 8)), nl.int32, buffer=nl.sbuf)[:, :blocks]
    active = nl.ndarray((1, max(blocks, 8)), nl.int32, buffer=nl.sbuf)[:, :blocks]
    inactive = nl.ndarray((1, max(blocks, 8)), nl.int32, buffer=nl.sbuf)[:, :blocks]
    ones = nl.ndarray((1, max(blocks, 8)), nl.int32, buffer=nl.sbuf)[:, :blocks]
    prefix = nl.ndarray((1, max(blocks, 8)), nl.int32, buffer=nl.sbuf)[:, :blocks]
    indices = nl.ndarray((1, max(blocks, 8)), nl.int32, buffer=nl.sbuf)[:, :blocks]
    positions = nl.ndarray((1, max(blocks, 8)), nl.int32, buffer=nl.sbuf)[:, :blocks]
    active_pos = nl.ndarray((1, max(blocks, 8)), nl.int32, buffer=nl.sbuf)[:, :blocks]
    inactive_pos = nl.ndarray((1, max(blocks, 8)), nl.int32, buffer=nl.sbuf)[:, :blocks]
    total = nl.ndarray((1, 8), nl.float32, buffer=nl.sbuf)[:, :1]
    order = nl.ndarray((1, max(blocks, 8)), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ids, src=row_ids.reshape((1, blocks)))
    nisa.tensor_scalar(dst=active, data=ids, op0=nl.greater_equal, operand0=0)
    nisa.tensor_scalar(dst=inactive, data=ids, op0=nl.less, operand0=0)
    nisa.memset(dst=ones, value=1)
    nisa.tensor_tensor_scan(dst=prefix, data0=ones, data1=active,
                            initial=0.0, op0=nl.multiply, op1=nl.add)
    nisa.tensor_reduce(dst=total, data=active, op=nl.add, axis=(1,))
    nisa.iota(dst=indices, pattern=[[1, blocks]])
    nisa.tensor_scalar(dst=active_pos, data=prefix, op0=nl.add, operand0=-1)
    nisa.tensor_tensor(dst=active_pos, data1=active_pos, data2=active, op=nl.multiply)
    nisa.tensor_tensor(dst=inactive_pos, data1=indices, data2=prefix, op=nl.subtract)
    nisa.tensor_scalar(dst=inactive_pos, data=inactive_pos, op0=nl.add, operand0=total)
    nisa.tensor_tensor(dst=inactive_pos, data1=inactive_pos, data2=inactive, op=nl.multiply)
    nisa.tensor_tensor(dst=positions, data1=active_pos, data2=inactive_pos, op=nl.add)
    for destination in range(blocks):
        matches = nl.ndarray((1, max(blocks, 8)), nl.int32, buffer=nl.sbuf)[:, :blocks]
        weighted = nl.ndarray((1, max(blocks, 8)), nl.int32, buffer=nl.sbuf)[:, :blocks]
        nisa.tensor_scalar(dst=matches, data=positions, op0=nl.equal,
                           operand0=destination)
        nisa.tensor_tensor(dst=weighted, data1=matches, data2=indices, op=nl.multiply)
        nisa.tensor_reduce(dst=order[:, destination:destination + 1],
                           data=weighted, op=nl.add, axis=(1,))
    return order, total


@nki.jit
def compact_decode_kernel(hidden, weights, scales, row_ids, expert_ids,
                          affinity, bounds, BLOCK_N=4096, BLOCK_K=4096):
    blocks, q = row_ids.shape
    assert q == 1 and blocks > 0
    assert nl.num_programs(0) in (1, 2)
    assert BLOCK_N > 0 and BLOCK_N % 128 == 0
    assert BLOCK_K > 0 and BLOCK_K % 128 == 0
    program, programs = nl.program_id(0), nl.num_programs(0)
    h = hidden.shape[1]
    out = nl.ndarray((blocks, 1, h), dtype=nl.float32, buffer=nl.shared_hbm)
    clamp = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=clamp, src=bounds)
    order, total = _active_order(row_ids)
    zeros = nl.ndarray((1, h), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zeros, value=0.0)

    # Every permutation position has one zero owner. The same program computes
    # that active position later. The odd partner only adds an identical compute.
    for position in range(program, blocks, programs):
        block = nisa.register_alloc()
        nisa.register_load(dst=block, src=order[:, position:position + 1])
        nisa.dma_copy(dst=out.ap(pattern=[[h, 1], [1, h]],
                                scalar_offset=block, indirect_dim=0), src=zeros)

    pair_count = nl.ndarray((1, 8), nl.float32, buffer=nl.sbuf)[:, :1]
    boundary = nl.ndarray((1, 8), nl.int32, buffer=nl.sbuf)[:, :1]
    last_active = nl.ndarray((1, 8), nl.int32, buffer=nl.sbuf)[:, :1]
    # FP32->int uses RNE. Quarter bias implements ceil(count/2) without ties.
    nisa.tensor_scalar(dst=pair_count, data=total, op0=nl.multiply,
                       operand0=1.0 / programs,
                       op1=nl.add, operand1=(programs - 1) * 0.25)
    nisa.tensor_copy(dst=boundary, src=pair_count)
    nisa.tensor_scalar(dst=boundary, data=boundary, op0=nl.multiply,
                       operand0=programs, op1=nl.add, operand1=program)
    nisa.tensor_scalar(dst=last_active, data=total, op0=nl.add, operand0=-1)
    end = nisa.register_alloc()
    nisa.register_load(dst=end, src=boundary)

    def compute_active(at_position):
        position = nl.ndarray((1, 8), nl.int32, buffer=nl.sbuf)[:, :1]
        nisa.register_store(dst=position, src=at_position)
        nisa.tensor_tensor(dst=position, data1=position, data2=last_active,
                           op=nl.minimum)
        position_reg = nisa.register_alloc()
        nisa.register_load(dst=position_reg, src=position)
        block_value = nl.ndarray((1, 8), nl.int32, buffer=nl.sbuf)[:, :1]
        nisa.tensor_copy(dst=block_value, src=order.ap(
            pattern=[[max(blocks, 8), 1], [1, 1]],
            scalar_offset=position_reg, indirect_dim=1))
        block = nisa.register_alloc()
        nisa.register_load(dst=block, src=block_value)
        _compute_block(hidden, weights, scales, row_ids, expert_ids, affinity,
                       out, clamp, block, BLOCK_N // 128, BLOCK_K // 128)

    # active0 gives end == program: both programs execute zero iterations.
    nl.fori_loop(program, end, compute_active, step=programs)
    return out
