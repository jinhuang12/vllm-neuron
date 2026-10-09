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


#: Token rows per pass of the fused MLP kernel's narrow path (chunks shorter
#: than :data:`MLP_WIDE_MIN_ROWS`). The gate/up moving operand is ``2 * chunk``
#: columns (bf16 hi/lo pair) and the down projection holds
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

#: Programs of the two-core launch: the two physical cores of an LNC2 core.
MLP_PROGRAMS = 2

#: Calls of at least this many token rows take the wide path: ``x`` loads as
#: stored and is transposed on the Tensor Engine, and the down projection runs
#: with tokens on partitions, so both large DMAs move whole token rows.
#: Below it, the strided ``x`` load and the per-partition down layout cost
#: fewer instructions.
MLP_WIDE_MIN_ROWS = 16

#: Token rows per pass of the wide path, capped so one pass's
#: ``(I // 128) * rows`` fp32 activation half fits one ``nisa.sendrecv``
#: (:data:`_SENDRECV_FP32` per partition).
MLP_WIDE_CHUNK = 64

#: k-tile groups of the wide path's ``hi | lo`` pair: group ``g``'s matmuls
#: run while group ``g + 1``'s pair is formed.
MLP_WIDE_GROUPS = 4

#: Widest ``H // 128`` the wide path is built and tested for (H = 4096). Its
#: transposed ``x`` chunk takes four of the eight PSUM banks, one per k-tile
#: group (``[128, KB / 4, 64]`` bf16 <= 2 KiB each).
MLP_WIDE_MAX_KB = 32

#: bf16 elements per partition in one PSUM bank (2 KiB).
_PSUM_BANK_BF16 = 1024

#: fp32 elements per partition one ``nisa.sendrecv`` moves (1 KiB).
_SENDRECV_FP32 = 256

# DMA modes, from the slice profiles. The default (software DGE) generates
# descriptors on GpSimd at ~0.65 us per DMA and starts packets 2-3 us after
# issue. The weights go through the Sync engine's hardware DGE ring (narrow
# path) or as static descriptors (wide path); x and the output go as static
# descriptors, whose trigger writes also run on Sync. The two scale-grid rows
# use software DGE: GpSimd is idle at kernel start, and this keeps Sync free to
# issue the weight load first.
def _weight_dma(dst, src, static):
    """A weight load: static descriptors on the wide path, else descriptors
    from the Sync engine's hardware DGE ring (slice profiles: each is the
    faster one on its path)."""
    if static:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none)
    else:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def _static_dma(dst, src):
    """x or output: static descriptors, no descriptor generation."""
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none)


def _mlp_wide(m_total, kb_count):
    """Does a call of ``m_total`` rows at ``H = 128 * kb_count`` take the wide path?"""
    return (
        m_total >= MLP_WIDE_MIN_ROWS
        and kb_count % MLP_WIDE_GROUPS == 0
        and kb_count <= MLP_WIDE_MAX_KB
    )


def _next_pow2(value):
    """Smallest power of two >= ``value`` (``value >= 1``)."""
    power = 1
    while power < value:
        power *= 2
    return power


def _mlp_projection(dst, weights_sb, x_view, scale_col, ib, col, kb_count, m_rows, zero):
    """One gate or up block: ``dst[128 (i), m] = W[:, ib-block]^T (x * s)`` in fp32.

    The per-partition block scale ``scale_col[:, col]`` is folded into the
    moving operand as a bf16 ``hi | lo`` pair, laid out ``[p, kb, 2m]``, so all
    ``kb_count`` k-tiles accumulate in one PSUM tile. ``hi = copy(x * s + zero)``
    on the Scalar Engine, with ``zero`` the kernel's table warm-up result
    (all 0.0), so the warm-up is live and its table load stays early.
    """
    pair = nl.ndarray(
        (TILE_SIZE, kb_count, 2 * m_rows), dtype=nl.bfloat16, buffer=nl.sbuf
    )
    hi = pair[:, :, 0:m_rows]
    lo = pair[:, :, m_rows:2 * m_rows]
    nisa.activation(
        dst=hi, op=nl.copy, data=x_view, scale=scale_col[:, col:col + 1], bias=zero
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
    nisa.tensor_tensor(dst=dst, data1=acc[:, 0:m_rows], data2=lo_sum, op=nl.add)


def _wide_projection(dst, weights_sb, x_rows, scale_col, n_inter_blocks, kb_count, m_rows):
    """Every I-block of one gate or up projection for a wide token chunk.

    ``dst[128 (i), ib, m]`` in fp32. ``x_rows`` is ``[m, H]`` bf16 as stored.
    The k-tile ``x_t[p, kb, m] = x[m, p * KB + kb]`` is a Tensor Engine
    transpose of a stride-``KB`` column view (bit-exact, bf16 into PSUM), and
    the ``hi | lo`` pair of :func:`_mlp_projection` is formed from it in PSUM,
    so every product is the narrow path's. The pair and the matmuls run in
    :data:`MLP_WIDE_GROUPS` k-tile groups, so the matmuls of one group overlap
    the pair of the next. Each group of ``x_t`` has a PSUM bank of its own:
    with two groups in one bank, the Scalar Engine's ``hi`` of group ``g + 1``
    waited for the Vector Engine's ``lo`` of group ``g`` (slice profile), and
    the pairs ran one after another. Each I-block accumulates in its own PSUM
    tile: a matmul that starts an accumulation group clears the whole tile's
    group.
    """
    m_pad = _next_pow2(m_rows)
    kb_group = kb_count // MLP_WIDE_GROUPS
    # [p, g, kb, m]: one group per 2 KiB bank (kb_group * m_pad <= 8 * 64 = 512).
    kb_pad = _PSUM_BANK_BF16 // m_pad
    x_t = nl.ndarray(
        (TILE_SIZE, MLP_WIDE_GROUPS, kb_pad, m_pad), dtype=nl.bfloat16, buffer=nl.psum
    )
    for kb in range(kb_count):
        nisa.nc_transpose(
            dst=x_t[:, kb // kb_group, kb % kb_group, 0:m_rows],
            data=x_rows.ap(
                pattern=[[kb_count * TILE_SIZE, m_rows], [kb_count, TILE_SIZE]],
                offset=kb,
            ),
            engine=nisa.engine.tensor,
        )
    for ib in range(n_inter_blocks):
        acc = nl.ndarray((TILE_SIZE, 2 * m_rows), dtype=nl.float32, buffer=nl.psum)
        for g in range(MLP_WIDE_GROUPS):
            k0 = g * kb_group
            pair = nl.ndarray(
                (TILE_SIZE, kb_group, 2 * m_rows), dtype=nl.bfloat16, buffer=nl.sbuf
            )
            hi = pair[:, :, 0:m_rows]
            lo = pair[:, :, m_rows:2 * m_rows]
            nisa.tensor_scalar(
                dst=hi,
                data=x_t[:, g, 0:kb_group, 0:m_rows],
                op0=nl.multiply,
                operand0=scale_col[:, ib:ib + 1],
                engine=nisa.engine.scalar,
            )
            nisa.scalar_tensor_tensor(
                dst=lo,
                data=x_t[:, g, 0:kb_group, 0:m_rows],
                op0=nl.multiply,
                operand0=scale_col[:, ib:ib + 1],
                op1=nl.subtract,
                operand1=hi,
            )
            for kk in range(kb_group):
                nisa.nc_matmul(
                    dst=acc,
                    stationary=weights_sb[:, k0 + kk, ib * TILE_SIZE:(ib + 1) * TILE_SIZE],
                    moving=pair[:, kk, :],
                    accumulate=(k0 + kk > 0),
                )
        lo_sum = nl.ndarray((TILE_SIZE, m_rows), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lo_sum, src=acc[:, m_rows:2 * m_rows])
        nisa.tensor_tensor(dst=dst[:, ib, :], data1=acc[:, 0:m_rows], data2=lo_sum, op=nl.add)


def _wide_down(out, activated, down_sb, down_grid, n_inter_blocks, kb_count,
               h_lo, h_width, out_offset, hidden):
    """``out[m, h_lo + c]`` for one wide chunk: the down projection, tokens on partitions.

    The chunk's ``h_width`` columns are split in two halves of ``half``. Half
    ``t`` runs on Tensor Engine column tile ``t`` (columns ``64 t .. 64 t + 63``):
    ``stationary = activated[:, ib, :]`` (``[128 (i), m]``, ``m <= 64``)
    against 512 fp8 weight columns at a time, into PSUM partitions
    ``64 t + m``. The two tiles run at once, and the result fills all 128
    partitions, so each scale instruction covers both halves and the output
    stores use all 16 DMA queues (``m`` partitions reach only ``m / 8``).
    Column ``c`` of half ``t`` lies in H-block ``(h_lo + t * half + c) / 128``;
    ``scale2[64 t + m, ib, c / 128]`` holds that block's scale.
    """
    m_rows = activated.shape[2]
    half = h_width // 2
    hp = TILE_SIZE // 2
    nb_half = half // TILE_SIZE
    # 128-partition tiles used at rows [64 t, 64 t + m): an instruction that
    # reads two SBUF operands needs them at one base partition.
    scale2 = nl.ndarray((TILE_SIZE, n_inter_blocks, nb_half), dtype=nl.float32, buffer=nl.sbuf)
    for ib in range(n_inter_blocks):
        # The down grid is [I-block, H-block]: (ib, b) at ib * KB + b.
        c0 = ib * kb_count + h_lo // TILE_SIZE
        nisa.tensor_copy(dst=scale2[0:hp, ib, :], src=down_grid[0:hp, c0:c0 + nb_half])
        nisa.tensor_copy(
            dst=scale2[hp:TILE_SIZE, ib, :],
            src=down_grid[hp:TILE_SIZE, c0 + nb_half:c0 + 2 * nb_half],
        )
    res = nl.ndarray((TILE_SIZE, half), dtype=nl.float32, buffer=nl.sbuf)
    # One instruction over both halves when they are contiguous (m == 64).
    n_parts = 2
    if m_rows == hp:
        n_parts = 1
    for n0 in range(0, half, _PSUM_BANK_FP32):
        n_cols = min(_PSUM_BANK_FP32, half - n0)
        for ib in range(n_inter_blocks):
            part = nl.ndarray((TILE_SIZE, n_cols), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=part[0:m_rows, :],
                stationary=activated[:, ib, :],
                moving=down_sb[:, ib, n0:n0 + n_cols],
                accumulate=False,
                tile_position=(0, 0),
                tile_size=(TILE_SIZE, hp),
            )
            nisa.nc_matmul(
                dst=part[hp:hp + m_rows, :],
                stationary=activated[:, ib, :],
                moving=down_sb[:, ib, half + n0:half + n0 + n_cols],
                accumulate=False,
                tile_position=(0, hp),
                tile_size=(TILE_SIZE, hp),
            )
            for q in range(n_cols // TILE_SIZE):
                c0 = n0 + q * TILE_SIZE
                cb = c0 // TILE_SIZE
                for part_i in range(n_parts):
                    r0 = part_i * hp
                    r1 = r0 + m_rows
                    if n_parts == 1:
                        r1 = TILE_SIZE
                    block = part[r0:r1, q * TILE_SIZE:(q + 1) * TILE_SIZE]
                    dst = res[r0:r1, c0:c0 + TILE_SIZE]
                    scale = scale2[r0:r1, ib, cb:cb + 1]
                    if ib > 0:
                        nisa.scalar_tensor_tensor(
                            dst=dst, data=block, op0=nl.multiply, operand0=scale,
                            op1=nl.add, operand1=dst,
                        )
                    elif q % 2 == 0:
                        nisa.tensor_scalar(
                            dst=dst, data=block, op0=nl.multiply, operand0=scale,
                            engine=nisa.engine.scalar,
                        )
                    else:
                        nisa.tensor_scalar(
                            dst=dst, data=block, op0=nl.multiply, operand0=scale,
                            engine=nisa.engine.vector,
                        )
        # Store each 512-column step of both halves as soon as it is scaled.
        _static_dma(
            dst=out.ap(pattern=[[hidden, m_rows], [1, n_cols]], offset=out_offset + n0),
            src=res[0:m_rows, n0:n0 + n_cols],
        )
        _static_dma(
            dst=out.ap(pattern=[[hidden, m_rows], [1, n_cols]], offset=out_offset + half + n0),
            src=res[hp:hp + m_rows, n0:n0 + n_cols],
        )


def _block_mask(kb_count, group, first_block):
    """``[128, KB]`` fp32: 1 where ``b == first_block + p // group``, else 0.

    ``offset = (b - first_block) * group - p`` lies in ``(-group, 0]`` exactly
    when ``b - first_block == p // group``.
    """
    block_offset = nl.ndarray((TILE_SIZE, kb_count), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(
        dst=block_offset,
        pattern=[[group, kb_count]],
        offset=-first_block * group,
        channel_multiplier=-1,
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
    return mask


def _scale_columns(dst, grid_sb, mask, h_major, n_inter_blocks, kb_count):
    """``dst[p, ib]`` = the scale of (H-block ``mask`` picks for ``p``, I-block ``ib``).

    ``grid_sb`` holds one public grid replicated on every partition. A gate or
    up grid is ``[H-block, I-block]`` (``h_major``); the down grid is
    ``[I-block, H-block]``.
    """
    n_grid = kb_count * n_inter_blocks
    if h_major:
        # (b, ib) sits at b * nI + ib.
        view = grid_sb.ap(
            pattern=[[n_grid, TILE_SIZE], [1, n_inter_blocks], [n_inter_blocks, kb_count]]
        )
    else:
        # (ib, b) sits at ib * KB + b.
        view = grid_sb.ap(
            pattern=[[n_grid, TILE_SIZE], [kb_count, n_inter_blocks], [1, kb_count]]
        )
    picked = nl.ndarray(
        (TILE_SIZE, n_inter_blocks, kb_count), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.tensor_tensor(
        dst=picked,
        data1=view,
        data2=mask.ap(
            pattern=[[kb_count, TILE_SIZE], [0, n_inter_blocks], [1, kb_count]]
        ),
        op=nl.multiply,
    )
    nisa.tensor_reduce(dst=dst, op=nl.add, data=picked, axis=(2,))


def _load_grid(grid_hbm, n_grid):
    """One public scale grid replicated on all 128 partitions.

    The grid is read once into partition 0 (one descriptor) and copied to
    every 32-partition quadrant by ``nc_stream_shuffle``. A stride-0 DMA that
    reads the same HBM row for all 128 partitions is far slower and stalls
    the weight transfers queued behind it.
    """
    grid_row = nl.ndarray((1, n_grid), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=grid_row,
        src=grid_hbm.ap(pattern=[[n_grid, 1], [1, n_grid]]),
        dge_mode=nisa.dge_mode.swdge,
    )
    grid_sb = nl.ndarray((TILE_SIZE, n_grid), dtype=nl.float32, buffer=nl.sbuf)
    for quadrant in range(TILE_SIZE // 32):
        nisa.nc_stream_shuffle(
            src=grid_row[0:1, :],
            dst=grid_sb[quadrant * 32:(quadrant + 1) * 32, :],
            shuffle_mask=[0] * 32,
        )
    return grid_sb


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
        x: ``[M, H]`` bf16, ``1 <= M < 128``; tokens are processed in chunks
            with the weights resident in SBUF.
        gate_weight, up_weight: ``[H, I]`` fp8-e4m3, compute frame.
        down_weight: ``[I, H]`` fp8-e4m3, compute frame.
        gate_scale, up_scale: ``[H // 128, I // 128]`` fp32 public grids.
        down_scale: ``[I // 128, H // 128]`` fp32 public grid.
        swiglu_limit: the checkpoint's clamp bound ``L``.

    Returns:
        ``[M, H]`` fp32.

    Launch: one program, or a ``[2]`` grid (:func:`mlp_launch_grid`) that puts
    one program on each physical core of an LNC2 core. With two programs,
    program 0 loads and runs the gate projection and program 1 the up
    projection; each clamps its own result (program 0 also applies ``silu``).
    The two then swap those fp32 ``[128, I // 128, chunk]`` tiles SBUF to SBUF
    (``nisa.sendrecv``, at most 1 KiB per partition), so both hold the same
    activated product. Each program then loads and runs the down projection
    for its own half of the hidden columns and stores that half of ``out``.
    So each core moves half of the weight bytes and runs half of the matmuls;
    no partial sum crosses the cores, and every output element is computed by
    the same instructions as on one core.

    Paths: ``M < MLP_WIDE_MIN_ROWS`` takes the narrow path below; longer calls
    take the wide path (:func:`_wide_projection`, :func:`_wide_down`), which
    computes the same products with whole-row DMAs.

    Notes:
        Contraction rows are laid out ``k = p * KB + kb`` (``KB = H // 128``),
        so every weight loads with one DMA of ``KB * I`` contiguous bytes per
        partition, and the scale block of row ``k`` is ``p // (128 // KB)``: a
        per-partition scalar. That scalar is folded into the moving activation
        as an exact bf16 ``hi + lo`` pair (``hi = bf16(x * s)``,
        ``lo = bf16(x * s - hi)``, error <= 2**-17 relative), so all 32 k-tiles
        of a projection accumulate in one PSUM tile and the pair costs no PE
        time (``2M <= 64`` moving columns). On the narrow path the down
        projection's output column ``h = h_lo + p * JB + j`` (``JB = KB /
        programs``, ``h_lo`` the program's first column) is chosen by a strided
        stationary view, so its scale is again per partition and each token row
        leaves in one DMA of ``JB`` contiguous fp32 per partition.
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
    programs = nl.num_programs(axes=0)
    program = nl.program_id(0)
    kernel_assert(programs in (1, MLP_PROGRAMS), "launch on one program or a [2] grid")
    kernel_assert(kb_count % programs == 0, "H // 128 must split over the programs")
    group = TILE_SIZE // kb_count
    n_inter_blocks = inter // TILE_SIZE
    n_grid = kb_count * n_inter_blocks
    limit = float(swiglu_limit)
    # This program's hidden columns of the down projection: [h_lo, h_lo + h_width).
    jb_count = kb_count // programs
    h_width = hidden // programs
    h_lo = program * h_width
    down_group = TILE_SIZE // jb_count
    # This program's projections: both on one core; gate | up on two.
    run_gate = programs == 1 or program == 0
    run_up = programs == 1 or program == 1
    wide = _mlp_wide(m_total, kb_count)
    chunk = MLP_TOKEN_CHUNK
    if wide:
        chunk = min(MLP_WIDE_CHUNK, _SENDRECV_FP32 // n_inter_blocks)

    out = nl.ndarray((m_total, hidden), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load the Scalar Engine's activation table (Copy and Silu share it) while
    # the DMAs below are in flight, not after the first x tile lands. The
    # result (0.0) is the bias of the narrow path's hi op, so it is not dead.
    zero_in = nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_in, value=0.0)
    zero = nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=zero, op=nl.copy, data=zero_in)

    # x first on a one-pass wide call: its whole-row load uses only the DMA
    # queues of its m partitions, and the transposes wait for it.
    x_early = wide and m_total <= chunk
    if x_early:
        x_rows = nl.ndarray((m_total, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
        _static_dma(dst=x_rows, src=x.ap(pattern=[[hidden, m_total], [1, hidden]]))

    # ---- Weights: one DMA each, fp8 bytes as stored, the longest first. --- #
    # No cast on the DMA: a casting DMA is generated by software DGE on GpSimd,
    # which serialises descriptor generation. The Tensor Engine takes the fp8
    # stationary with a bf16 moving operand directly (exact e4m3 upcast).
    if run_gate:
        gate_sb = nl.ndarray(
            (TILE_SIZE, kb_count, inter), dtype=gate_weight.dtype, buffer=nl.sbuf
        )
        _weight_dma(
            dst=gate_sb.reshape((TILE_SIZE, kb_count * inter)),
            src=gate_weight.reshape((TILE_SIZE, kb_count * inter)),
            static=wide,
        )
    if run_up:
        up_sb = nl.ndarray(
            (TILE_SIZE, kb_count, inter), dtype=up_weight.dtype, buffer=nl.sbuf
        )
        _weight_dma(
            dst=up_sb.reshape((TILE_SIZE, kb_count * inter)),
            src=up_weight.reshape((TILE_SIZE, kb_count * inter)),
            static=wide,
        )
    # down_sb[p, ib, c] = down[ib * 128 + p, h_lo + c]: h_width contiguous bytes.
    down_sb = nl.ndarray(
        (TILE_SIZE, n_inter_blocks, h_width), dtype=down_weight.dtype, buffer=nl.sbuf
    )
    _weight_dma(
        dst=down_sb,
        src=down_weight.ap(
            pattern=[[hidden, TILE_SIZE], [TILE_SIZE * hidden, n_inter_blocks],
                     [1, h_width]],
            offset=h_lo,
        ),
        static=wide,
    )

    # ---- Scales: each grid replicated over partitions as stored. --------- #
    # A [p, b] mask then selects each partition's own H-block: b == p // group
    # for the projections and, on the narrow path, b == h_lo / 128 +
    # p // down_group for this program's down columns. The wide path reads
    # the replicated down grid directly.
    if run_gate:
        gate_grid = _load_grid(gate_scale, n_grid)
    if run_up:
        up_grid = _load_grid(up_scale, n_grid)
    down_grid = _load_grid(down_scale, n_grid)
    proj_mask = _block_mask(kb_count, group, 0)
    if run_gate:
        gate_col = nl.ndarray((TILE_SIZE, n_inter_blocks), dtype=nl.float32, buffer=nl.sbuf)
        _scale_columns(gate_col, gate_grid, proj_mask, True, n_inter_blocks, kb_count)
    if run_up:
        up_col = nl.ndarray((TILE_SIZE, n_inter_blocks), dtype=nl.float32, buffer=nl.sbuf)
        _scale_columns(up_col, up_grid, proj_mask, True, n_inter_blocks, kb_count)
    for m0 in range(0, m_total, chunk):
        m_rows = min(chunk, m_total - m0)
        m_pad = _next_pow2(m_rows)
        j_per_bank = min(jb_count, _PSUM_BANK_FP32 // m_pad)
        n_banks = jb_count // j_per_bank

        if wide and not x_early:
            # x_rows[m, :] = x[m0 + m, :]: whole rows, as stored.
            x_rows = nl.ndarray((m_rows, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
            _static_dma(
                dst=x_rows,
                src=x.ap(pattern=[[hidden, m_rows], [1, hidden]], offset=m0 * hidden),
            )
        elif not wide:
            # x_t[p, m, kb] = x[m0 + m, p * KB + kb]: KB contiguous bf16 per row.
            x_t = nl.ndarray((TILE_SIZE, m_rows, kb_count), dtype=nl.bfloat16, buffer=nl.sbuf)
            _static_dma(
                dst=x_t,
                src=x.ap(
                    pattern=[[kb_count, TILE_SIZE], [hidden, m_rows], [1, kb_count]],
                    offset=m0 * hidden,
                ),
            )
            x_view = x_t.ap(
                pattern=[[m_rows * kb_count, TILE_SIZE], [1, kb_count], [kb_count, m_rows]]
            )

        # ---- gate / up on this program: silu(min(gate, L)), clip(up, -L, L).
        if run_gate:
            gated = nl.ndarray(
                (TILE_SIZE, n_inter_blocks, m_rows), dtype=nl.float32, buffer=nl.sbuf
            )
            gate = nl.ndarray(
                (TILE_SIZE, n_inter_blocks, m_rows), dtype=nl.float32, buffer=nl.sbuf
            )
            if wide:
                _wide_projection(
                    gate, gate_sb, x_rows, gate_col, n_inter_blocks, kb_count, m_rows
                )
            else:
                for ib in range(n_inter_blocks):
                    _mlp_projection(
                        gate[:, ib, :], gate_sb, x_view, gate_col, ib, ib, kb_count,
                        m_rows, zero,
                    )
            nisa.tensor_scalar(dst=gate, data=gate, op0=nl.minimum, operand0=limit)
            nisa.activation(dst=gated, op=nl.silu, data=gate)
        if run_up:
            up = nl.ndarray(
                (TILE_SIZE, n_inter_blocks, m_rows), dtype=nl.float32, buffer=nl.sbuf
            )
            if wide:
                _wide_projection(
                    up, up_sb, x_rows, up_col, n_inter_blocks, kb_count, m_rows
                )
            else:
                for ib in range(n_inter_blocks):
                    _mlp_projection(
                        up[:, ib, :], up_sb, x_view, up_col, ib, ib, kb_count,
                        m_rows, zero,
                    )
            nisa.tensor_scalar(
                dst=up,
                data=up,
                op0=nl.maximum,
                operand0=-limit,
                op1=nl.minimum,
                operand1=limit,
            )
        if programs > 1:
            # Swap the halves core to core: program 0 sends silu(gate) and
            # receives clip(up); program 1 the reverse.
            theirs = nl.ndarray(
                (TILE_SIZE, n_inter_blocks, m_rows), dtype=nl.float32, buffer=nl.sbuf
            )
            mine = gated if run_gate else up
            nisa.sendrecv(
                src=mine.reshape((TILE_SIZE, n_inter_blocks * m_rows)),
                dst=theirs.reshape((TILE_SIZE, n_inter_blocks * m_rows)),
                send_to_rank=1 - program,
                recv_from_rank=1 - program,
                pipe_id=0,
                dma_engine=nisa.dma_engine.gpsimd_dma,
            )
            if run_gate:
                up = theirs
            else:
                gated = theirs
        activated = nl.ndarray(
            (TILE_SIZE, n_inter_blocks, m_rows), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        nisa.tensor_tensor(dst=activated, data1=gated, data2=up, op=nl.multiply)

        if wide:
            # ---- Down, tokens on partitions; each row leaves contiguous. -- #
            _wide_down(
                out, activated, down_sb, down_grid, n_inter_blocks, kb_count, h_lo,
                h_width, m0 * hidden + h_lo, hidden,
            )
        else:
            if m0 == 0:
                # The down scale column, built once and only now: on the
                # Vector Engine it would otherwise sit between the gate/up
                # matmuls and their PSUM evacuation.
                if programs == 1:
                    down_mask = proj_mask
                else:
                    down_mask = _block_mask(kb_count, down_group, h_lo // TILE_SIZE)
                down_col = nl.ndarray(
                    (TILE_SIZE, n_inter_blocks), dtype=nl.float32, buffer=nl.sbuf
                )
                _scale_columns(down_col, down_grid, down_mask, False, n_inter_blocks, kb_count)
            # ---- Down: column h = h_lo + p * JB + j via a strided stationary.
            result = nl.ndarray((TILE_SIZE, m_rows, jb_count), dtype=nl.float32, buffer=nl.sbuf)
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
                                    [n_inter_blocks * h_width, TILE_SIZE],
                                    [jb_count, TILE_SIZE],
                                ],
                                offset=ib * h_width + j0 + jj,
                            ),
                            moving=activated[:, ib, :],
                            accumulate=False,
                        )
                    dst = result.ap(
                        pattern=[
                            [m_rows * jb_count, TILE_SIZE],
                            [1, j_per_bank],
                            [jb_count, m_rows],
                        ],
                        offset=j0,
                    )
                    if ib == 0:
                        nisa.tensor_scalar(
                            dst=dst,
                            data=partial[:, :, 0:m_rows],
                            op0=nl.multiply,
                            operand0=down_col[:, ib:ib + 1],
                        )
                    else:
                        nisa.scalar_tensor_tensor(
                            dst=dst,
                            data=partial[:, :, 0:m_rows],
                            op0=nl.multiply,
                            operand0=down_col[:, ib:ib + 1],
                            op1=nl.add,
                            operand1=dst,
                        )
            _static_dma(
                dst=out.ap(
                    pattern=[[jb_count, TILE_SIZE], [hidden, m_rows], [1, jb_count]],
                    offset=m0 * hidden + h_lo,
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


@dataclass
class _MlpLaunchCounters:
    """Fused-kernel launches by grid: one program, or the ``[2]`` LNC2 grid.

    Kept apart from :class:`_MlpDispatchCounters`, whose reader is the
    two-tuple ``(nki_dispatch, torch_fallback)`` registry contract.
    """

    one_program: int = 0
    two_program: int = 0


_MLP_LAUNCH_COUNTERS = _MlpLaunchCounters()


def reset_mlp_launch_counters() -> None:
    """Zero both launch counters."""
    _MLP_LAUNCH_COUNTERS.one_program = 0
    _MLP_LAUNCH_COUNTERS.two_program = 0


def mlp_launch_counters() -> tuple[int, int]:
    """``(one_program, two_program)`` fused launches since the last reset."""
    return _MLP_LAUNCH_COUNTERS.one_program, _MLP_LAUNCH_COUNTERS.two_program


@torch._dynamo.assume_constant_result
def _count_mlp_launch(programs: int) -> None:
    if programs == MLP_PROGRAMS:
        _MLP_LAUNCH_COUNTERS.two_program += 1
    else:
        _MLP_LAUNCH_COUNTERS.one_program += 1


def mlp_launch_grid(hidden: int) -> tuple[int, ...]:
    """The launch grid of :func:`blockwise_fp8_mlp_small_m_kernel`.

    ``(2,)`` on an LNC2 runtime (``NEURON_LOGICAL_NC_CONFIG=2``: two physical
    cores per logical core) when the ``H // 128`` hidden blocks split into two
    halves, else ``()`` (one program).
    """
    import os

    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        return ()
    if hidden <= 0 or hidden % TILE_SIZE or (hidden // TILE_SIZE) % MLP_PROGRAMS:
        return ()
    return (MLP_PROGRAMS,)


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
    :func:`blockwise_fp8_mlp_small_m_kernel` call and never pads tokens, on the
    launch grid :func:`mlp_launch_grid` picks (both cores of an LNC2 core). Any
    other geometry (whole-tile prefill, or ``I > MLP_MAX_INTERMEDIATE``) runs
    the three :func:`blockwise_fp8_mm` calls.
    """
    from torch.nn.functional import silu

    tokens, hidden = int(x.shape[-2]), int(x.shape[-1])
    intermediate = int(gate_weight.shape[-1])
    nki = can_run_kernel(x)
    if nki and fused_mlp_admissible(tokens, hidden, intermediate):
        _count_mlp_fused()
        grid = mlp_launch_grid(hidden)
        _count_mlp_launch(grid[0] if grid else 1)
        call = wrap_nki(blockwise_fp8_mlp_small_m_kernel)
        if grid:
            call = call[grid]
        return call(
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
