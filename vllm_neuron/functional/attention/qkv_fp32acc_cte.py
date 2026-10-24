# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm and projection at prefill, with the activations in two bf16 passes.

The library ``qkv_cte`` GEMM cut to one configuration: bf16 ``[B, S, H]`` input in HBM,
RMSNorm, ``[H, I]`` bf16 weights, ``[B, S, I]`` bf16 output, no bias or RoPE. Each
program takes a contiguous half of the rows. A block of 128-row tiles is normed and
transposed on the PE array, then multiplied by the weights in 1024-row loads, with each
tile's output in fp32 PSUM banks of 512 columns.

The library rounds the normed activations to bf16 twice. Here ``z = x * gamma`` is exact
in fp32 (two bf16 factors), and it enters the PE array as ``hi = bf16(z)`` and
``lo = bf16(z - hi)``: two passes into the same fp32 PSUM. ``z - hi - lo`` is at most
2^-16 of ``|z|``. The row's ``1 / rms`` scales the fp32 sum, and the output is rounded to
bf16 once.
"""

import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info

_P = 128
_BANK = 512  # fp32 columns per PSUM bank
_ACC_BANKS = 6  # PSUM banks for the outputs; the transposes take the other 2
_MAX_BLOCK_TILES = 4  # 128-row tiles per block, bounded by SBUF
_W_TILES = 8  # 128-row weight tiles per load


def _norm_and_split(x, gamma_sb, eps, rows, hi, lo, rstd):
    """Write ``hi`` and ``lo`` ``[128, H / 128, rows]`` for one tile and its ``rstd`` ``[rows, 1]``."""
    _, size = x.shape
    sq = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=sq[0:rows, 0:1].broadcast(dim=1, size=size), op=nl.square,
                           data=x[0:rows, 0:size], reduce_op=nl.add, reduce_res=rstd[0:rows, 0:1])
    nisa.tensor_scalar(dst=rstd[0:rows, 0:1], data=rstd[0:rows, 0:1], op0=nl.multiply,
                       operand0=1.0 / float(size), op1=nl.add, operand1=float(eps))
    nisa.activation(dst=rstd[0:rows, 0:1], op=nl.rsqrt, data=rstd[0:rows, 0:1])
    for h_tile in range(size // _P):
        xt = nl.ndarray((_P, _P), dtype=x.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=xt[:, 0:rows], data=x[0:rows, h_tile * _P:(h_tile + 1) * _P])
        g = gamma_sb[:, h_tile:h_tile + 1]
        nisa.tensor_scalar(dst=hi[:, h_tile, 0:rows], data=xt[:, 0:rows], op0=nl.multiply, operand0=g)
        nisa.scalar_tensor_tensor(dst=lo[:, h_tile, 0:rows], data=xt[:, 0:rows], op0=nl.multiply,
                                  operand0=g, op1=nl.subtract, operand1=hi[:, h_tile, 0:rows])


def qkv_fp32acc_cte(hidden, qkv_w, norm_w, eps):
    """``[B, S, I]`` bf16: ``rmsnorm(hidden) * norm_w @ qkv_w``, any number of rows.

    Args:
        hidden: ``[B, S, H]`` bf16 in HBM, ``H`` in 128-row tiles.
        qkv_w: ``[H, I]`` bf16, ``I <= 3072``.
        norm_w: ``[1, H]`` bf16 norm gain.
        eps: the norm epsilon.
    """
    _, programs, program = get_verified_program_sharding_info("qkv_fp32acc_cte", (0, 1))
    batch, seq, size = hidden.shape
    rows = batch * seq
    w_size, width = qkv_w.shape
    kernel_assert(w_size == size and size % _P == 0, "H matches the weights in 128-row tiles")
    i_banks = div_ceil(width, _BANK)
    kernel_assert(i_banks <= _ACC_BANKS, "at most 3072 columns")
    h_tiles = size // _P
    block_tiles = max(1, min(_MAX_BLOCK_TILES, _ACC_BANKS // i_banks))

    x_rows = hidden.reshape((rows, size))
    out = nl.ndarray((rows, width), dtype=hidden.dtype, buffer=nl.shared_hbm)
    w_view = qkv_w.reshape_dim(0, [h_tiles, _P])  # [H / 128, 128, I]
    gamma_in = nl.ndarray((_P, h_tiles), dtype=norm_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_in, src=norm_w.ap(pattern=[[1, _P], [_P, h_tiles]], offset=0))
    # A per-partition scalar operand must be fp32; the cast is exact.
    gamma_sb = nl.ndarray((_P, h_tiles), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gamma_sb, src=gamma_in)

    share = div_ceil(rows, programs)
    first = program * share
    last = min(rows, first + share)
    block_rows = block_tiles * _P
    for b0 in range(first, last, block_rows):
        tiles = div_ceil(min(block_rows, last - b0), _P)
        hi = [nl.ndarray((_P, h_tiles, _P), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(tiles)]
        lo = [nl.ndarray((_P, h_tiles, _P), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(tiles)]
        rstd = [nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf) for _ in range(tiles)]
        sizes = [min(_P, last - b0 - t * _P) for t in range(tiles)]
        for t in range(tiles):
            x = nl.ndarray((_P, size), dtype=hidden.dtype, buffer=nl.sbuf)
            r0 = b0 + t * _P
            nisa.dma_copy(dst=x[0:sizes[t], :], src=x_rows[r0:r0 + sizes[t], :])
            _norm_and_split(x, gamma_sb, eps, sizes[t], hi[t], lo[t], rstd[t])

        acc = [[nl.ndarray((_P, _BANK), dtype=nl.float32, buffer=nl.psum) for _ in range(i_banks)]
               for _ in range(tiles)]
        for k0 in range(0, h_tiles, _W_TILES):
            k_tiles = min(_W_TILES, h_tiles - k0)
            w_sb = nl.ndarray((_P, _W_TILES, width), dtype=qkv_w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_sb[:, 0:k_tiles, :], src=w_view.slice(0, k0, k0 + k_tiles).permute([1, 0, 2]))
            for t in range(tiles):
                for j in range(k_tiles):
                    for i in range(i_banks):
                        cols = min(_BANK, width - i * _BANK)
                        for part, act in enumerate((hi[t], lo[t])):
                            nisa.nc_matmul(dst=acc[t][i][0:sizes[t], 0:cols],
                                           stationary=act[:, k0 + j, 0:sizes[t]],
                                           moving=w_sb[:, j, i * _BANK:i * _BANK + cols],
                                           accumulate=(k0 + j > 0 or part > 0))

        for t in range(tiles):
            result = nl.ndarray((_P, width), dtype=hidden.dtype, buffer=nl.sbuf)
            for i in range(i_banks):
                cols = min(_BANK, width - i * _BANK)
                nisa.tensor_scalar(dst=result[0:sizes[t], i * _BANK:i * _BANK + cols],
                                   data=acc[t][i][0:sizes[t], 0:cols], op0=nl.multiply,
                                   operand0=rstd[t][0:sizes[t], 0:1])
            r0 = b0 + t * _P
            nisa.dma_copy(dst=out[r0:r0 + sizes[t], :], src=result[0:sizes[t], :])
    return out.reshape((batch, seq, width))
