# SPDX-License-Identifier: Apache-2.0
"""Dense blockwise-fp8 GEMM: ``out[M, N] = x[M, K] @ dequantise(weight[K, N])``.

``weight`` is fp8-e4m3 with one fp32 scale per ``128 x 128`` block of ``(K, N)``,
so ``dequantise(weight)[k, n] = weight[k, n] * weight_scale[k // 128, n // 128]``
-- the granularity a checkpoint stores, which the kernel indexes directly instead
of re-tiling. A scale block spans exactly one ``128``-wide contraction tile, so
every ``nc_matmul`` result is multiplied by a single scale before it is added into
an fp32 SBUF accumulator: no partial sum crosses two scales.

fp8 weight tiles are upcast to bf16 on the load DMA. The upcast is bit-exact
(e4m3's 3 significand and 4 exponent bits fit bf16's 7 and 8) and avoids
``perf_mode="double_row"`` fp8 matmul, which NeuronCore-v2 does not support. PSUM,
the accumulator and the returned tensor are fp32; casting the result down is the
caller's choice.

Short matrices with fewer than 128 tokens use an unpadded GEMV-oriented kernel:
the weights are stationary and real tokens form the streamed free dimension.
Whole-token tiles retain the existing prefill kernel and arithmetic.

:func:`blockwise_fp8_mm_torch_oracle` is the torch reference for the same product,
and the fallback taken when no Neuron device or simulator is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.kernel_assert import kernel_assert

from vllm_neuron.functional.moe.blockwise_fp8_retile import TILE_SIZE
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: Scale-block extent this module indexes by: one fp32 scale per ``128 x 128``
#: weight block. Equals ``TILE_SIZE``; ``blockwise_fp8_retile.BLOCK_QUANT_SIZE``
#: (256) is the MoE path's granularity and is never read here.
SCALE_BLOCK_SIZE = TILE_SIZE

#: Contraction tiles per scale block -- ``1`` at this granularity. Derived from the
#: two constants so the pair cannot drift from the quotient.
K_TILES_PER_BLOCK = SCALE_BLOCK_SIZE // TILE_SIZE

#: Tensor Engine free-size bounds from ``nl.tile_size``: 128 stationary, 512 moving
#: (also the fp32 PSUM bank extent). ``SCALE_BLOCK_SIZE`` is the moving extent used
#: here, so it fits both with room to spare.
STATIONARY_FMAX = 128
MOVING_FMAX = 512

__all__ = [
    "SCALE_BLOCK_SIZE",
    "TILE_SIZE",
    "BlockwiseFp8MmError",
    "blockwise_fp8_mm",
    "blockwise_fp8_mm_kernel",
    "blockwise_fp8_mm_small_m_kernel",
    "blockwise_fp8_mm_torch_oracle",
    "can_run_blockwise_fp8_mm",
    "dispatch_counters",
    "flat_scale_index",
    "kernel_identity",
    "kernel_scale_shape",
    "reset_dispatch_counters",
    "scale_grid_shape",
    "to_kernel_scale_layout",
]


class BlockwiseFp8MmError(ValueError):
    """A geometry or layout this module refuses.

    Raised in preference to letting NKI trap at trace time: the message names the
    offending extent, and no extent is silently truncated into a different
    function than the one requested.
    """


@nki.jit
def blockwise_fp8_mm_small_m_kernel(x, weight, weight_scale_t):
    """Multiply a short activation matrix without padding its token dimension.

    Args:
        x: ``[M, K]`` bf16 activations, ``1 <= M < TILE_SIZE``.
        weight: ``[K, N]`` fp8-e4m3 weights, with K and N blocked by 128.
        weight_scale_t: The existing replicated checkpoint scale operand,
            ``[128, (K // 128) * (N // 128)]`` fp32.

    Returns:
        ``[M, N]`` fp32, with each 128-wide contraction partial scaled
        independently before accumulation.

    Notes:
        Weights are stationary and the short activation tile is moving.
        Thus the Tensor Engine's streamed free dimension is M, rather than
        128 output columns or 128 padded tokens. Output columns occupy all
        128 partitions while scaling. Tiles transpose into a contiguous result
        in SBUF, which is stored in one DMA to avoid short strided HBM writes.
        No checkpoint scales are merged, and no activation is quantized.
    """
    m_extent, k_extent = x.shape
    _, n_extent = weight.shape
    kernel_assert(0 < m_extent < TILE_SIZE, "small-M GEMM needs 1 <= M < 128")
    kernel_assert(k_extent % SCALE_BLOCK_SIZE == 0, "K must be blocked by 128")
    kernel_assert(n_extent % SCALE_BLOCK_SIZE == 0, "N must be blocked by 128")
    kernel_assert(weight.shape[0] == k_extent, "activation and weight K must agree")

    n_n_blocks = n_extent // SCALE_BLOCK_SIZE
    n_k_blocks = k_extent // SCALE_BLOCK_SIZE
    out = nl.ndarray((m_extent, n_extent), dtype=nl.float32, buffer=nl.shared_hbm)
    sbm = SbufManager(0, nl.tile_size.sbuf_fmax_bytes, use_auto_alloc=True)
    sbm.open_scope(name="small_m")
    scale_sb = sbm.alloc(weight_scale_t.shape, dtype=nl.float32)
    nisa.dma_copy(dst=scale_sb, src=weight_scale_t)
    result_sb = sbm.alloc((m_extent, n_extent), dtype=nl.float32)
    # Load the whole short activation once. The partition axis is K modulo
    # 128; the free axes select the contraction block and real token.
    # Small per-block HBM loads otherwise spend more time in DMA setup than
    # the Tensor Engine saves by avoiding padded token rows.
    if n_k_blocks % 16 == 0:
        # Reinterpret each checkpoint K chunk as an HBM row. DMA transpose
        # then has a 16-aligned first dimension even when there is one token.
        x_cache = sbm.alloc(
            (TILE_SIZE, m_extent * n_k_blocks), dtype=x.dtype
        )
        nisa.dma_transpose(
            dst=x_cache,
            src=x.ap(
                pattern=[
                    [TILE_SIZE, m_extent * n_k_blocks],
                    [1, TILE_SIZE],
                ]
            ),
        )
    else:
        x_cache = sbm.alloc(
            (TILE_SIZE, n_k_blocks, m_extent), dtype=x.dtype
        )
        nisa.dma_copy(
            dst=x_cache,
            src=x.ap(
                pattern=[
                    [1, TILE_SIZE],
                    [TILE_SIZE, n_k_blocks],
                    [k_extent, m_extent],
                ]
            ),
        )

    for n_block in nl.affine_range(n_n_blocks):
        sbm.open_scope(name="output_block")
        n0 = n_block * SCALE_BLOCK_SIZE
        acc = sbm.alloc((SCALE_BLOCK_SIZE, m_extent), dtype=nl.float32)
        for k_block in nl.affine_range(n_k_blocks):
            sbm.open_scope(name="contraction_block")
            k0 = k_block * SCALE_BLOCK_SIZE
            x_t = sbm.alloc((TILE_SIZE, m_extent), dtype=x.dtype)
            if n_k_blocks % 16 == 0:
                nisa.tensor_copy(
                    dst=x_t,
                    src=x_cache.ap(
                        pattern=[
                            [m_extent * n_k_blocks, TILE_SIZE],
                            [n_k_blocks, m_extent],
                        ],
                        offset=k_block,
                    ),
                )
            else:
                nisa.tensor_copy(dst=x_t, src=x_cache[:, k_block, :])
            w_tile = sbm.alloc(
                (TILE_SIZE, SCALE_BLOCK_SIZE), dtype=nl.bfloat16
            )
            nisa.dma_copy(
                dst=w_tile,
                src=weight[k0:k0 + TILE_SIZE, n0:n0 + SCALE_BLOCK_SIZE],
            )
            partial = nl.ndarray(
                (SCALE_BLOCK_SIZE, m_extent), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(
                dst=partial, stationary=w_tile, moving=x_t, accumulate=False
            )
            flat = k_block * n_n_blocks + n_block
            if k_block == 0:
                nisa.tensor_scalar(
                    dst=acc,
                    data=partial,
                    op0=nl.multiply,
                    operand0=scale_sb[0:SCALE_BLOCK_SIZE, flat:flat + 1],
                )
            else:
                nisa.scalar_tensor_tensor(
                    dst=acc,
                    data=partial,
                    op0=nl.multiply,
                    operand0=scale_sb[0:SCALE_BLOCK_SIZE, flat:flat + 1],
                    op1=nl.add,
                    operand1=acc,
                )
            sbm.close_scope()
        transposed = nl.ndarray(
            (m_extent, SCALE_BLOCK_SIZE), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_transpose(dst=transposed, data=acc)
        nisa.tensor_copy(
            dst=result_sb[:, n0:n0 + SCALE_BLOCK_SIZE], src=transposed
        )
        sbm.close_scope()
    nisa.dma_copy(dst=out, src=result_sb)
    sbm.close_scope()
    return out


#: Token rows per pass of the fused MLP kernel. The gate/up moving operand is
#: ``2 * chunk`` columns (bf16 hi/lo pair) and the down projection holds
#: ``(I // 128) * (H // 128) * chunk`` fp32 per partition in PSUM; 32 keeps
#: both inside the PSUM budget at the dense MLP's two intermediate blocks.
MLP_TOKEN_CHUNK = 32

#: Widest intermediate the fused kernel takes. All three weights stay in SBUF:
#: ``2 * (H // 128) * I + (I // 128) * H`` fp8 bytes per partition, 96 KiB at
#: ``H = 4096, I = 1024`` of the 192 KiB partition. TP=64 needs I = 128 (shared
#: expert) and 256 (dense MLP); a wider per-rank shard keeps the three calls.
MLP_MAX_INTERMEDIATE = 1024

#: fp32 elements in one PSUM bank, per partition. No matmul output may cross one.
_PSUM_BANK_FP32 = 512


def _next_pow2(value):
    """Smallest power of two >= ``value`` (``value >= 1``)."""
    power = 1
    while power < value:
        power *= 2
    return power


def _mlp_projection(weights_sb, x_view, scale_col, ib, col, kb_count, m_rows):
    """One gate or up block: ``[128 (i), m] = W[:, ib-block]^T (x * s)`` in fp32.

    The per-partition block scale ``scale_col[:, col]`` is folded into the
    moving operand as a bf16 ``hi | lo`` pair, laid out ``[p, kb, 2m]``, so all
    ``kb_count`` k-tiles accumulate in one PSUM tile.
    """
    pair = nl.ndarray(
        (TILE_SIZE, kb_count, 2 * m_rows), dtype=nl.bfloat16, buffer=nl.sbuf
    )
    hi = pair[:, :, 0:m_rows]
    lo = pair[:, :, m_rows:2 * m_rows]
    nisa.tensor_scalar(
        dst=hi, data=x_view, op0=nl.multiply, operand0=scale_col[:, col:col + 1]
    )
    nisa.scalar_tensor_tensor(
        dst=lo,
        data=x_view,
        op0=nl.multiply,
        operand0=scale_col[:, col:col + 1],
        op1=nl.subtract,
        operand1=hi,
    )
    acc = nl.ndarray((TILE_SIZE, 2 * m_rows), dtype=nl.float32, buffer=nl.psum)
    for kb in range(kb_count):
        nisa.nc_matmul(
            dst=acc,
            stationary=weights_sb[:, kb, ib * TILE_SIZE:(ib + 1) * TILE_SIZE],
            moving=pair[:, kb, :],
            accumulate=(kb > 0),
        )
    # An engine reads at most one PSUM operand: stage lo in SBUF.
    lo_sum = nl.ndarray((TILE_SIZE, m_rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lo_sum, src=acc[:, m_rows:2 * m_rows])
    total = nl.ndarray((TILE_SIZE, m_rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=total, data1=acc[:, 0:m_rows], data2=lo_sum, op=nl.add)
    return total


@nki.jit
def blockwise_fp8_mlp_small_m_kernel(
    x,
    gate_weight,
    up_weight,
    down_weight,
    gate_scale,
    up_scale,
    down_scale,
    swiglu_limit,
):
    """One fused blockwise-fp8 SwiGLU MLP for a short token matrix.

    ``down(silu(min(gate(x), L)) * clip(up(x), -L, L))`` with the bf16 cast of
    the SwiGLU product that the three-call route performs between projections.

    Args:
        x: ``[M, H]`` bf16, ``1 <= M < 128``; tokens are processed in chunks of
            :data:`MLP_TOKEN_CHUNK` with the weights resident in SBUF.
        gate_weight, up_weight: ``[H, I]`` fp8-e4m3, compute frame.
        down_weight: ``[I, H]`` fp8-e4m3, compute frame.
        gate_scale, up_scale: ``[H // 128, I // 128]`` fp32 public grids.
        down_scale: ``[I // 128, H // 128]`` fp32 public grid.
        swiglu_limit: the checkpoint's clamp bound ``L``.

    Returns:
        ``[M, H]`` fp32.

    Notes:
        Contraction rows are laid out ``k = p * KB + kb`` (``KB = H // 128``),
        so every weight loads with one DMA of ``KB * I`` contiguous bytes per
        partition, and the scale block of row ``k`` is ``p // (128 // KB)``: a
        per-partition scalar. That scalar is folded into the moving activation
        as an exact bf16 ``hi + lo`` pair (``hi = bf16(x * s)``,
        ``lo = bf16(x * s - hi)``, error <= 2**-17 relative), so all 32 k-tiles
        of a projection accumulate in one PSUM tile and the pair costs no PE
        time (``2M <= 64`` moving columns). The down projection's output column
        ``h = p * KB + j`` is chosen by a strided stationary view, so its
        scale is again per partition and each token row leaves in one DMA of
        ``KB`` contiguous fp32 per partition. Runs on one physical core: at
        decode M the whole MLP is ~1.5-3 MB of weight, and splitting it across
        the LNC2 pair would need an SBUF->HBM exchange plus ``core_barrier``
        around the SwiGLU, which costs more than the halved PE time saves.
    """
    m_total, hidden = x.shape
    _, inter = gate_weight.shape
    kernel_assert(0 < m_total < TILE_SIZE, "fused small-M MLP needs 1 <= M < 128")
    kernel_assert(hidden % TILE_SIZE == 0, "H must be blocked by 128")
    kernel_assert(inter % TILE_SIZE == 0, "I must be blocked by 128")
    kernel_assert(inter <= MLP_MAX_INTERMEDIATE, "I too wide for SBUF-resident weights")
    kb_count = hidden // TILE_SIZE
    kernel_assert(
        TILE_SIZE % kb_count == 0, "H // 128 must divide 128 (H <= 16384, pow2)"
    )
    kernel_assert(up_weight.shape[1] == inter, "gate and up must agree")
    kernel_assert(down_weight.shape[0] == inter, "down must be [I, H]")
    kernel_assert(down_weight.shape[1] == hidden, "down must be [I, H]")
    group = TILE_SIZE // kb_count
    n_inter_blocks = inter // TILE_SIZE
    limit = float(swiglu_limit)

    out = nl.ndarray((m_total, hidden), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- Weights: one DMA each, fp8 bytes as stored. ---------------------- #
    # No cast on the DMA: a casting DMA is generated by software DGE on GpSimd,
    # which serialises descriptor generation. The Tensor Engine takes the fp8
    # stationary with a bf16 moving operand directly (exact e4m3 upcast).
    gate_sb = nl.ndarray((TILE_SIZE, kb_count, inter), dtype=gate_weight.dtype, buffer=nl.sbuf)
    up_sb = nl.ndarray((TILE_SIZE, kb_count, inter), dtype=up_weight.dtype, buffer=nl.sbuf)
    down_sb = nl.ndarray(
        (TILE_SIZE, n_inter_blocks, hidden), dtype=down_weight.dtype, buffer=nl.sbuf
    )
    nisa.dma_copy(
        dst=gate_sb.reshape((TILE_SIZE, kb_count * inter)),
        src=gate_weight.reshape((TILE_SIZE, kb_count * inter)),
    )
    nisa.dma_copy(
        dst=up_sb.reshape((TILE_SIZE, kb_count * inter)),
        src=up_weight.reshape((TILE_SIZE, kb_count * inter)),
    )
    # down_sb[p, ib, h] = down[ib * 128 + p, h]: H contiguous bytes per row.
    nisa.dma_copy(
        dst=down_sb,
        src=down_weight.ap(
            pattern=[[hidden, TILE_SIZE], [TILE_SIZE * hidden, n_inter_blocks],
                     [1, hidden]],
        ),
    )

    # ---- Scales: every grid replicated over partitions as stored. -------- #
    # Each grid is one contiguous row per partition (stride-0 partition read),
    # so the DMA has 128 descriptors, not one per scale. A [p, b] mask
    # (b == p // group) then selects each partition's own H-block b: columns
    # [0, nI) gate, [nI, 2nI) up, [2nI, 3nI) down of ``scale_col``.
    n_cols = 3 * n_inter_blocks
    n_grid = kb_count * n_inter_blocks
    grids = nl.ndarray((TILE_SIZE, 3, n_grid), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=grids[:, 0, :], src=gate_scale.ap(pattern=[[0, TILE_SIZE], [1, n_grid]])
    )
    nisa.dma_copy(
        dst=grids[:, 1, :], src=up_scale.ap(pattern=[[0, TILE_SIZE], [1, n_grid]])
    )
    nisa.dma_copy(
        dst=grids[:, 2, :], src=down_scale.ap(pattern=[[0, TILE_SIZE], [1, n_grid]])
    )
    # offset = b * group - p lies in (-group, 0] exactly when b == p // group.
    block_offset = nl.ndarray((TILE_SIZE, kb_count), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(
        dst=block_offset, pattern=[[group, kb_count]], offset=0, channel_multiplier=-1
    )
    not_above = nl.ndarray((TILE_SIZE, kb_count), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=not_above, data=block_offset, op0=nl.less_equal, operand0=0.0
    )
    mask = nl.ndarray((TILE_SIZE, kb_count), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=mask,
        data=block_offset,
        op0=nl.greater,
        operand0=float(-group),
        op1=nl.multiply,
        operand1=not_above,
    )
    picked = nl.ndarray((TILE_SIZE, n_cols, kb_count), dtype=nl.float32, buffer=nl.sbuf)
    mask_wide = mask.ap(
        pattern=[[kb_count, TILE_SIZE], [0, n_inter_blocks], [1, kb_count]]
    )
    # gate and up grids are [H-block, I-block]: (nb, b) sits at b * nI + nb.
    for index in range(2):
        nisa.tensor_tensor(
            dst=picked[:, index * n_inter_blocks:(index + 1) * n_inter_blocks, :],
            data1=grids.ap(
                pattern=[[3 * n_grid, TILE_SIZE], [1, n_inter_blocks],
                         [n_inter_blocks, kb_count]],
                offset=index * n_grid,
            ),
            data2=mask_wide,
            op=nl.multiply,
        )
    # The down grid is [I-block, H-block]: (kb, b) sits at kb * KB + b.
    nisa.tensor_tensor(
        dst=picked[:, 2 * n_inter_blocks:n_cols, :],
        data1=grids.ap(
            pattern=[[3 * n_grid, TILE_SIZE], [kb_count, n_inter_blocks],
                     [1, kb_count]],
            offset=2 * n_grid,
        ),
        data2=mask_wide,
        op=nl.multiply,
    )
    scale_col = nl.ndarray((TILE_SIZE, n_cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=scale_col, op=nl.add, data=picked, axis=(2,))

    # The down output's H-tiles per PSUM tensor: no matmul crosses a bank.
    for m0 in range(0, m_total, MLP_TOKEN_CHUNK):
        m_rows = min(MLP_TOKEN_CHUNK, m_total - m0)
        m_pad = _next_pow2(m_rows)
        j_per_bank = min(kb_count, _PSUM_BANK_FP32 // m_pad)
        n_banks = kb_count // j_per_bank

        # x_t[p, m, kb] = x[m0 + m, p * KB + kb]: KB contiguous bf16 per row.
        x_t = nl.ndarray((TILE_SIZE, m_rows, kb_count), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=x_t,
            src=x.ap(
                pattern=[[kb_count, TILE_SIZE], [hidden, m_rows], [1, kb_count]],
                offset=m0 * hidden,
            ),
        )
        x_view = x_t.ap(
            pattern=[[m_rows * kb_count, TILE_SIZE], [1, kb_count], [kb_count, m_rows]]
        )

        activated = nl.ndarray(
            (TILE_SIZE, n_inter_blocks, m_rows), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        for ib in range(n_inter_blocks):
            gate = _mlp_projection(
                gate_sb, x_view, scale_col, ib, ib, kb_count, m_rows
            )
            up = _mlp_projection(
                up_sb, x_view, scale_col, ib, n_inter_blocks + ib, kb_count, m_rows
            )
            nisa.tensor_scalar(dst=gate, data=gate, op0=nl.minimum, operand0=limit)
            nisa.tensor_scalar(
                dst=up,
                data=up,
                op0=nl.maximum,
                operand0=-limit,
                op1=nl.minimum,
                operand1=limit,
            )
            gated = nl.ndarray((TILE_SIZE, m_rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=gated, op=nl.silu, data=gate)
            nisa.tensor_tensor(
                dst=activated[:, ib, :], data1=gated, data2=up, op=nl.multiply
            )

        # ---- Down: output column h = p * KB + j via a strided stationary. - #
        result = nl.ndarray((TILE_SIZE, m_rows, kb_count), dtype=nl.float32, buffer=nl.sbuf)
        for bank in range(n_banks):
            j0 = bank * j_per_bank
            for ib in range(n_inter_blocks):
                partial = nl.ndarray(
                    (TILE_SIZE, j_per_bank, m_pad), dtype=nl.float32, buffer=nl.psum
                )
                for jj in range(j_per_bank):
                    nisa.nc_matmul(
                        dst=partial[:, jj, 0:m_rows],
                        stationary=down_sb.ap(
                            pattern=[
                                [n_inter_blocks * hidden, TILE_SIZE],
                                [kb_count, TILE_SIZE],
                            ],
                            offset=ib * hidden + j0 + jj,
                        ),
                        moving=activated[:, ib, :],
                        accumulate=False,
                    )
                dst = result.ap(
                    pattern=[
                        [m_rows * kb_count, TILE_SIZE],
                        [1, j_per_bank],
                        [kb_count, m_rows],
                    ],
                    offset=j0,
                )
                col = 2 * n_inter_blocks + ib
                if ib == 0:
                    nisa.tensor_scalar(
                        dst=dst,
                        data=partial[:, :, 0:m_rows],
                        op0=nl.multiply,
                        operand0=scale_col[:, col:col + 1],
                    )
                else:
                    nisa.scalar_tensor_tensor(
                        dst=dst,
                        data=partial[:, :, 0:m_rows],
                        op0=nl.multiply,
                        operand0=scale_col[:, col:col + 1],
                        op1=nl.add,
                        operand1=dst,
                    )
        nisa.dma_copy(
            dst=out.ap(
                pattern=[[kb_count, TILE_SIZE], [hidden, m_rows], [1, kb_count]],
                offset=m0 * hidden,
            ),
            src=result,
        )
    return out


@nki.jit
def blockwise_fp8_mm_kernel(x, weight, weight_scale_t):
    """``out[M, N] = x[M, K] @ dequantise(weight[K, N])``, in NKI.

    Both operands of every ``nc_matmul`` carry the contraction extent on the
    partition axis, which is what the Tensor Engine contracts over, so ``x`` is
    loaded through ``nl.load_transpose2d`` (a DMA-side transpose) rather than
    transposed on chip.

    Args:
        x: ``[M, K]`` activations, bf16. ``M`` is tiled by ``TILE_SIZE`` over the
            PSUM partition axis and ``K`` by ``SCALE_BLOCK_SIZE`` over the
            contraction axis.
        weight: ``[K, N]`` fp8-e4m3 weights, expressed against the
            ``128``-granular block scales the checkpoint itself stores.
        weight_scale_t: ``[TILE_SIZE, (K // 128) * (N // 128)]`` fp32, the operand
            layout :func:`to_kernel_scale_layout` builds: one column per
            ``128 x 128`` block, replicated across the partition axis because
            ``nisa.tensor_scalar`` broadcasts only along the free dimension.

    Returns:
        ``[M, N]`` fp32.
    """
    m_extent, k_extent = x.shape
    _, n_extent = weight.shape
    n_n_blocks = n_extent // SCALE_BLOCK_SIZE
    n_k_blocks = k_extent // SCALE_BLOCK_SIZE

    out = nl.ndarray((m_extent, n_extent), dtype=nl.float32, buffer=nl.shared_hbm)
    # One load: the scale operand is (partitions x blocks) and tiny.
    scale_sb = nl.load(weight_scale_t)

    for m_tile in range(m_extent // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
        for n_block in range(n_n_blocks):
            n0 = n_block * SCALE_BLOCK_SIZE
            # fp32 accumulator over the K blocks, in SBUF: PSUM is reclaimed per
            # block so the block scale can be applied between blocks.
            acc = nl.ndarray(
                (TILE_SIZE, SCALE_BLOCK_SIZE), dtype=nl.float32, buffer=nl.sbuf
            )
            for k_block in range(n_k_blocks):
                psum = nl.ndarray(
                    (TILE_SIZE, SCALE_BLOCK_SIZE), dtype=nl.float32, buffer=nl.psum
                )
                for k_sub in range(K_TILES_PER_BLOCK):
                    k0 = k_block * SCALE_BLOCK_SIZE + k_sub * TILE_SIZE
                    # [K=TILE_SIZE partitions, M=TILE_SIZE free]
                    x_t = nl.load_transpose2d(
                        x[m0 : m0 + TILE_SIZE, k0 : k0 + TILE_SIZE]
                    )
                    # [K=TILE_SIZE partitions, N=SCALE_BLOCK_SIZE free], upcast
                    # from fp8 on the DMA.
                    w_tile = nl.load(
                        weight[k0 : k0 + TILE_SIZE, n0 : n0 + SCALE_BLOCK_SIZE],
                        dtype=nl.bfloat16,
                    )
                    # dst = stationary.T @ moving = [M, N]. The accumulate flag is
                    # explicit rather than inferred by the compiler, so the
                    # first-write-overwrites contract is visible here.
                    nisa.nc_matmul(
                        dst=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        stationary=x_t,
                        moving=w_tile,
                        accumulate=(k_sub > 0),
                    )
                flat = k_block * n_n_blocks + n_block
                if k_block == 0:
                    # First block initialises the accumulator, so no zeroing pass.
                    nisa.tensor_scalar(
                        dst=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        data=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                    )
                else:
                    nisa.scalar_tensor_tensor(
                        dst=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        data=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                        op1=nl.add,
                        operand1=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                    )
            nl.store(
                out[m0 : m0 + TILE_SIZE, n0 : n0 + SCALE_BLOCK_SIZE],
                value=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
            )
    return out


def _require_blocked(rows: int, cols: int, tokens: int) -> None:
    """Check every extent condition the kernel above imposes.

    ``rows`` is ``K`` (the contraction extent), ``cols`` is ``N``, ``tokens`` is
    ``M``.

    Raises:
        BlockwiseFp8MmError: naming every condition that failed.
    """
    problems: list[str] = []
    if tokens <= 0 or (tokens > TILE_SIZE and tokens % TILE_SIZE):
        problems.append(
            f"M={tokens} is not a positive multiple of TILE_SIZE={TILE_SIZE}; "
            f"short matrices with 1 <= M < {TILE_SIZE} use the unpadded "
            f"small-M kernel; larger matrices require whole token tiles"
        )
    if rows <= 0 or rows % SCALE_BLOCK_SIZE:
        problems.append(
            f"K={rows} is not a positive multiple of "
            f"SCALE_BLOCK_SIZE={SCALE_BLOCK_SIZE}; the remainder has no block "
            f"scale"
        )
    if cols <= 0 or cols % SCALE_BLOCK_SIZE:
        problems.append(
            f"N={cols} is not a positive multiple of "
            f"SCALE_BLOCK_SIZE={SCALE_BLOCK_SIZE}; the remainder has no block "
            f"scale"
        )
    if SCALE_BLOCK_SIZE > MOVING_FMAX:
        # Asserted rather than assumed: if either constant moves, the moving free
        # extent has to be re-tiled, not silently exceeded.
        problems.append(
            f"SCALE_BLOCK_SIZE={SCALE_BLOCK_SIZE} exceeds the Tensor Engine "
            f"moving free bound {MOVING_FMAX}"
        )
    if TILE_SIZE > STATIONARY_FMAX:
        problems.append(
            f"TILE_SIZE={TILE_SIZE} exceeds the Tensor Engine stationary free "
            f"bound {STATIONARY_FMAX}"
        )
    if problems:
        raise BlockwiseFp8MmError(
            "blockwise fp8 mm refuses this geometry: " + "; ".join(problems)
        )


def scale_grid_shape(rows: int, cols: int) -> tuple[int, int]:
    """``(K // 128, N // 128)`` -- the shape of the public scale grid.

    This is the shape a caller supplies: one fp32 scale per ``128 x 128`` weight
    block, indexed ``[k_block, n_block]``. The kernel operand is a different
    shape; :func:`to_kernel_scale_layout` is the bridge.
    """
    if rows <= 0 or rows % SCALE_BLOCK_SIZE or cols <= 0 or cols % SCALE_BLOCK_SIZE:
        raise BlockwiseFp8MmError(
            f"weight extent [{rows},{cols}] is not a whole number of "
            f"{SCALE_BLOCK_SIZE}x{SCALE_BLOCK_SIZE} blocks"
        )
    return rows // SCALE_BLOCK_SIZE, cols // SCALE_BLOCK_SIZE


def flat_scale_index(k_block: int, n_block: int, n_n_blocks: int) -> int:
    """The kernel's flat block index: ``k_block`` major, ``n_block`` minor.

    Written once here and mirrored by the kernel's own
    ``flat = k_block * n_n_blocks + n_block``, so a consumer never repeats the
    arithmetic: a transposed flattening leaves every shape valid and no range
    check can see it.
    """
    return k_block * n_n_blocks + n_block


def kernel_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    """``(TILE_SIZE, n_blocks)`` -- the shape the kernel's scale operand needs."""
    n_k_blocks, n_n_blocks = scale_grid_shape(rows, cols)
    return TILE_SIZE, n_k_blocks * n_n_blocks


def to_kernel_scale_layout(weight_scale: Tensor, rows: int, cols: int) -> Tensor:
    """Bridge the public ``[K//128, N//128]`` grid to the kernel's operand.

    Flattens the grid by :func:`flat_scale_index` and replicates each scalar
    across ``TILE_SIZE`` partitions, because ``nisa.tensor_scalar`` broadcasts
    ``operand0`` only along the **free** dimension, from a tile of shape
    ``(data.shape[0], 1)``.

    Returns:
        ``[TILE_SIZE, (K//128) * (N//128)]`` fp32, contiguous.

    Raises:
        BlockwiseFp8MmError: if ``weight_scale`` is not the declared grid shape
            or not fp32. A mis-sized grid can reshape without error onto a
            different block-to-scale assignment, and the two orders are
            indistinguishable by any range check.
    """
    want = scale_grid_shape(rows, cols)
    if tuple(weight_scale.shape) != want:
        raise BlockwiseFp8MmError(
            f"weight_scale has shape {tuple(weight_scale.shape)}, expected "
            f"{want} for a [K={rows}, N={cols}] weight. Refusing to reshape: a "
            f"mis-sized scale grid can flatten onto a different "
            f"block-to-scale assignment without any error."
        )
    if weight_scale.dtype != torch.float32:
        raise BlockwiseFp8MmError(
            f"weight_scale must be fp32, got {weight_scale.dtype}"
        )
    n_k_blocks, n_n_blocks = want
    flat = torch.empty(
        n_k_blocks * n_n_blocks, dtype=torch.float32, device=weight_scale.device
    )
    for k_block in range(n_k_blocks):
        for n_block in range(n_n_blocks):
            flat[flat_scale_index(k_block, n_block, n_n_blocks)] = weight_scale[
                k_block, n_block
            ]
    return flat.unsqueeze(0).expand(TILE_SIZE, flat.numel()).contiguous()


@dataclass
class _DispatchCounters:
    """Which route ran, counted rather than inferred.

    ``nki_dispatch`` counts entries into the ``wrap_nki`` path, ``torch_fallback``
    counts entries into the torch path. Two counters rather than one flag, so
    "the kernel ran" and "the fallback did not run" are independent readings.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: Module level so the counters can be zeroed and read from outside this module.
_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


# Counter stores stay off the traced graph: a store Dynamo traces turns into a
# value guard that fails on the first call after warmup.
@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch."""
    _COUNTERS.nki_dispatch += 1


def can_run_blockwise_fp8_mm(x: Tensor, rows: int, cols: int, tokens: int) -> bool:
    """Is the NKI route available *and* admissible for this geometry?

    Two independent conditions, deliberately not merged: ``can_run_kernel``
    answers "is there a device or a simulator", :func:`_require_blocked` answers
    "does this kernel accept these extents". A geometry the kernel cannot serve
    raises rather than falling back to torch.

    Raises:
        BlockwiseFp8MmError: if the geometry is inadmissible.
    """
    _require_blocked(rows, cols, tokens)
    return can_run_kernel(x)


def _checked_prebuilt_scale(prebuilt: Tensor, rows: int, cols: int) -> Tensor:
    """Accept a caller-supplied kernel scale operand, or refuse it by shape.

    When the operand is built once at load time the bridge does not run, so the
    bridge's own checks do not run either; this replaces them at the call site.

    The check here is the weaker of the two: :func:`kernel_scale_shape` pins
    ``TILE_SIZE`` and the product ``n_k_blocks * n_n_blocks``, so a transposed
    public grid -- ``[16, 8]`` where ``[8, 16]`` was meant -- has the same element
    count and passes. That case is caught where the operand is built instead,
    because :func:`to_kernel_scale_layout` compares the public grid against
    ``(rows, cols)`` before flattening.
    """
    want = kernel_scale_shape(rows, cols)
    if tuple(prebuilt.shape) != want:
        raise BlockwiseFp8MmError(
            f"prebuilt_scale_t has shape {tuple(prebuilt.shape)}, expected "
            f"{want} = kernel_scale_shape(rows={rows}, cols={cols}). Refusing "
            f"to reshape or rebuild: an operand of the wrong extent means the "
            f"caller prepared it for a different weight."
        )
    if prebuilt.dtype != torch.float32:
        raise BlockwiseFp8MmError(
            f"prebuilt_scale_t must be fp32, got {prebuilt.dtype}; the kernel "
            f"reads block scales as fp32 and a narrower dtype computes a "
            f"different function without changing any shape"
        )
    return prebuilt


def blockwise_fp8_mm(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    *,
    prebuilt_scale_t: Tensor | None = None,
) -> Tensor:
    """Dense blockwise-fp8 GEMM: NKI kernel when available, torch otherwise.

    Args:
        x: ``[M, K]`` activations, bf16. Short ``1 <= M < 128`` matrices run
            without padding; larger M must be a positive multiple of 128.
        weight: ``[K, N]`` fp8-e4m3, expressed against ``weight_scale``.
        weight_scale: ``[K//128, N//128]`` fp32, one scale per weight block.
        prebuilt_scale_t: Optional, keyword-only. The kernel operand
            :func:`to_kernel_scale_layout` would have built, already built --
            shape :func:`kernel_scale_shape`, fp32. Supply it and the bridge is
            not entered on this call. ``weight_scale`` is still required either
            way: the torch fallback consumes the public grid, so a prebuilt
            operand cannot stand in for it.

    Returns:
        ``[M, N]`` fp32.

    Raises:
        BlockwiseFp8MmError: on an inadmissible geometry, a mis-shaped scale
            grid, or a prebuilt operand that is not
            ``kernel_scale_shape(rows, cols)`` fp32.
    """
    tokens, rows = x.shape[-2], x.shape[-1]
    if weight.shape[-2] != rows:
        raise BlockwiseFp8MmError(
            f"x has K={rows} but weight has K={weight.shape[-2]}; the "
            f"contraction extents must agree"
        )
    cols = weight.shape[-1]

    if not can_run_blockwise_fp8_mm(x, rows, cols, tokens):
        _count_torch_fallback()
        logger.debug(
            "blockwise_fp8_mm: NKI route unavailable, using the torch path "
            "(oracle / constraint-violation fallback, not the shipped path)"
        )
        return blockwise_fp8_mm_torch_oracle(x, weight, weight_scale)

    _count_nki_dispatch()
    if prebuilt_scale_t is None:
        _count_scale_layout_build()
        scale_t = to_kernel_scale_layout(weight_scale, rows, cols)
    else:
        scale_t = _checked_prebuilt_scale(prebuilt_scale_t, rows, cols)
    kernel = (
        blockwise_fp8_mm_small_m_kernel
        if tokens < TILE_SIZE
        else blockwise_fp8_mm_kernel
    )
    return wrap_nki(kernel)(
        x=x, weight=weight, weight_scale_t=scale_t
    )


@dataclass
class _MlpDispatchCounters:
    """``(nki_dispatch, torch_fallback)`` of the MLP, the contract every family keeps.

    ``fused`` counts fused-kernel calls; ``torch_fallback`` counts MLPs that ran
    with no NKI route at all. The three-call route on kernels (prefill, or a
    wide intermediate) counts in neither: its three calls are
    :func:`blockwise_fp8_mm` dispatches and that family counts them.
    """

    fused: int = 0
    torch_fallback: int = 0


_MLP_COUNTERS = _MlpDispatchCounters()


def reset_mlp_dispatch_counters() -> None:
    """Zero both MLP counters."""
    _MLP_COUNTERS.fused = 0
    _MLP_COUNTERS.torch_fallback = 0


def mlp_dispatch_counters() -> tuple[int, int]:
    """``(fused, torch_fallback)`` since the last reset."""
    return _MLP_COUNTERS.fused, _MLP_COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_mlp_fused() -> None:
    _MLP_COUNTERS.fused += 1


@torch._dynamo.assume_constant_result
def _count_mlp_torch_fallback() -> None:
    _MLP_COUNTERS.torch_fallback += 1


def fused_mlp_admissible(tokens: int, hidden: int, intermediate: int) -> bool:
    """Does :func:`blockwise_fp8_mlp_small_m_kernel` accept this geometry?"""
    if not 0 < tokens < TILE_SIZE:
        return False
    if hidden <= 0 or hidden % TILE_SIZE or intermediate <= 0:
        return False
    if intermediate % TILE_SIZE or TILE_SIZE % (hidden // TILE_SIZE):
        return False
    return intermediate <= MLP_MAX_INTERMEDIATE


def blockwise_fp8_mlp(
    x: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    gate_scale: Tensor,
    up_scale: Tensor,
    down_scale: Tensor,
    *,
    swiglu_limit: float,
    prebuilt_scale_t: tuple[Tensor, Tensor, Tensor] | None = None,
) -> Tensor:
    """The clamped SwiGLU MLP on blockwise fp8 weights, fused when small.

    ``down(silu(min(gate(x), L)) * clip(up(x), -L, L))`` with the SwiGLU product
    cast to ``x.dtype`` before the down projection, exactly as the three-call
    route computes it.

    Args:
        x: ``[M, H]`` bf16.
        gate_weight, up_weight: ``[H, I]`` fp8-e4m3; down_weight: ``[I, H]``.
        gate_scale, up_scale, down_scale: the public ``128``-block grids.
        swiglu_limit: the checkpoint's clamp bound.
        prebuilt_scale_t: the three kernel scale operands, for the three-call
            route only (the fused kernel reads the public grids directly).

    Returns:
        ``[M, H]`` fp32.

    Route: ``1 <= M < 128`` with the NKI route available runs one
    :func:`blockwise_fp8_mlp_small_m_kernel` call and never pads tokens. Any
    other geometry (whole-tile prefill, or ``I > MLP_MAX_INTERMEDIATE``) runs
    the three :func:`blockwise_fp8_mm` calls.
    """
    from torch.nn.functional import silu

    tokens, hidden = int(x.shape[-2]), int(x.shape[-1])
    intermediate = int(gate_weight.shape[-1])
    nki = can_run_kernel(x)
    if nki and fused_mlp_admissible(tokens, hidden, intermediate):
        _count_mlp_fused()
        return wrap_nki(blockwise_fp8_mlp_small_m_kernel)(
            x=x,
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=down_weight,
            gate_scale=gate_scale,
            up_scale=up_scale,
            down_scale=down_scale,
            swiglu_limit=float(swiglu_limit),
        )
    if not nki:
        _count_mlp_torch_fallback()
    gate_t, up_t, down_t = prebuilt_scale_t or (None, None, None)
    gate = blockwise_fp8_mm(x, gate_weight, gate_scale, prebuilt_scale_t=gate_t)
    up = blockwise_fp8_mm(x, up_weight, up_scale, prebuilt_scale_t=up_t)
    gate = gate.clamp(min=None, max=swiglu_limit)
    up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    activated = silu(gate) * up
    return blockwise_fp8_mm(
        activated.to(x.dtype), down_weight, down_scale, prebuilt_scale_t=down_t
    )


def blockwise_fp8_mm_torch_oracle(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
) -> Tensor:
    """Block-dequantise, then matmul -- in torch, in fp32.

    Independent of the kernel in arithmetic order: it dequantises **first** and
    contracts in one fp32 matmul, where the kernel contracts one ``128``-wide
    block at a time and scales between blocks. It also never consults
    :func:`flat_scale_index`, so a transposed flattening in the bridge shows up
    as a numeric disagreement rather than as agreement by construction.

    This is also the fallback for :func:`blockwise_fp8_mm` when the NKI route is
    unavailable.

    Returns:
        ``[M, N]`` fp32.
    """
    rows, cols = weight.shape[-2], weight.shape[-1]
    want = scale_grid_shape(rows, cols)
    if tuple(weight_scale.shape) != want:
        raise BlockwiseFp8MmError(
            f"weight_scale has shape {tuple(weight_scale.shape)}, expected "
            f"{want} for a [K={rows}, N={cols}] weight"
        )
    dequantised = weight.to(torch.float32) * weight_scale.repeat_interleave(
        SCALE_BLOCK_SIZE, 0
    ).repeat_interleave(SCALE_BLOCK_SIZE, 1)
    return x.to(torch.float32) @ dequantised


def kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the NKI kernel, read off the wrapped object."""
    func = getattr(blockwise_fp8_mm_kernel, "func", None)
    target = func if func is not None else blockwise_fp8_mm_kernel
    return target.__module__, target.__qualname__


@dataclass
class _BuildCounters:
    """How many kernel scale operands :func:`blockwise_fp8_mm` built itself.

    Separate from ``_DispatchCounters`` so that its reader keeps returning a
    two-tuple. Only builds at the one site in :func:`blockwise_fp8_mm` are
    counted; a direct call to :func:`to_kernel_scale_layout` from elsewhere is
    not, which is the right population for the question this answers -- whether
    the per-forward path rebuilds the operand.
    """

    scale_layout_builds: int = 0


#: Module level for the same reason ``_COUNTERS`` is: zeroed and read from
#: outside this module.
_BUILD_COUNTERS = _BuildCounters()


def reset_scale_layout_builds() -> None:
    """Zero the build counter."""
    _BUILD_COUNTERS.scale_layout_builds = 0


def scale_layout_builds() -> int:
    """How many operands have been built since the last reset."""
    return _BUILD_COUNTERS.scale_layout_builds


@torch._dynamo.assume_constant_result
def _count_scale_layout_build() -> None:
    """Count one scale-layout build, off the traced graph."""
    _BUILD_COUNTERS.scale_layout_builds += 1
