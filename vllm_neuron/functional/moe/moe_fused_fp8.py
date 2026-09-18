# SPDX-License-Identifier: Apache-2.0
"""Shape-specialized, fused block-FP8 experts on Trainium2.

BLOCK_M/N/K are compile-time scheduling choices. The inner 128x128 product
and its scale remain fixed by the checkpoint quantization contract.
"""

import nki
import nki.isa as nisa
import nki.language as nl


def _tile(rows, cols, dtype=nl.float32):
    return nl.ndarray((rows, max(cols, 8)), dtype=dtype, buffer=nl.sbuf)[:, :cols]


def _transpose(dst, src):
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


@nki.jit
def moe_fused_fp8_kernel(hidden, weights, scales, row_ids, expert_ids, affinity,
                         bounds, BLOCK_M=128, BLOCK_N=1024, BLOCK_K=4096):
    """Return FP32 [blocks,q,H] contributions from device routing.

    hidden: BF16 [T+1,H], final row zero.
    weights: prepared finite FP8 [E,3*(I/128),128,H/128,128].
    scales: matching FP32 [E,3*(I/128),H/128].
    row_ids: int32 [blocks,q], token IDs or -1; expert_ids: int32 [blocks,1].
    affinity: FP32 [(T+1)*E,1], final E entries zero.
    bounds: FP32 [128,3], repeated gate upper, up lower, up upper.

    BLOCK_M tiles token rows (1..512). BLOCK_N and BLOCK_K group positive
    multiples of128 output and contraction channels. Each group still uses
    separate 128x128 scaled products, in increasing contraction order.
    q can exceed BLOCK_M. Shape tails need no host routing read.
    """
    blocks, q = row_ids.shape
    experts, panels, contraction, nh, channels = weights.shape
    h = hidden.shape[1]
    ni = panels // 3
    assert blocks > 0 and q > 0 and experts > 0
    assert panels % 3 == 0 and ni > 0 and h == nh * 128
    assert contraction == 128 and channels == 128
    assert scales.shape == (experts, panels, nh)
    assert 1 <= BLOCK_M <= 512
    assert BLOCK_N > 0 and BLOCK_N % 128 == 0
    assert BLOCK_K > 0 and BLOCK_K % 128 == 0
    nstep, kstep = BLOCK_N // 128, BLOCK_K // 128
    nprograms, program = nl.num_programs(0), nl.program_id(0)
    out = nl.ndarray((blocks, q, h), dtype=nl.float32, buffer=nl.shared_hbm)
    clamp = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=clamp, src=bounds)

    for block in range(program, blocks, nprograms):
        expert = _tile(1, 1, nl.int32)
        nisa.dma_copy(dst=expert, src=expert_ids[block:block + 1, :])
        expert_reg = nisa.register_alloc()
        nisa.register_load(dst=expert_reg, src=expert)
        scale = nl.ndarray((128, panels, nh), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=scale.reshape((128, panels * nh)),
            src=scales.ap(pattern=[[0, 128], [1, panels * nh]],
                          scalar_offset=expert_reg, indirect_dim=0),
        )
        for m0 in range(0, q, BLOCK_M):
            m = min(BLOCK_M, q - m0)
            x = nl.ndarray((128, nh, m), dtype=nl.bfloat16, buffer=nl.sbuf)
            routing = []
            for t0 in range(0, m, 128):
                rows = min(128, m - t0)
                ids = _tile(rows, 1, nl.int32)
                nisa.dma_copy(dst=ids, src=row_ids.ap(
                    pattern=[[1, rows], [1, 1]], offset=block * q + m0 + t0))
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
                nisa.dma_copy(dst=expert_rows, src=expert_ids.ap(
                    pattern=[[0, rows], [1, 1]], offset=block))
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
                nisa.dma_copy(dst=out[block, m0 + t0:m0 + t0 + rows, :], src=output_rows)
    return out
