# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm and projection at decode, with fp32 sums.

The library ``qkv_tkg`` path cut to one configuration: bf16 ``[B, S, H]`` input in HBM,
RMSNorm, ``[H, I]`` bf16 weights, ``[B, S, I]`` bf16 output in HBM, no bias. The norm is
the library's ``rmsnorm_tkg``. The projection keeps the library's H-column tiling: each
program takes half of H, and the PE array holds up to 4 row groups of tokens at once.

Two sums differ from the library. The row-group partials add in an fp32 SBUF tile, and
at LNC 2 the other program's fp32 partial is added in fp32. The output is rounded to
bf16 once.
"""

import nki.isa as nisa
import nki.language as nl
from nkilib.core.subkernels.rmsnorm_tkg import rmsnorm_tkg
from nkilib.core.utils.allocator import create_auto_alloc_manager, sizeinbytes
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from nkilib.core.utils.tiled_range import TiledRange

_P = 128
_BANK = 512  # fp32 columns per PSUM bank
_PSUM_BANKS = 8
_H_BLOCK_TILES = 16  # 128-row weight tiles per load: 2048 rows of H


def _array_tiling(rows, tiles):
    """``(dim, factor)``: rows per PE row group and the number of groups for ``tiles`` H tiles."""
    dim = 32 if rows <= 32 else (64 if rows <= 64 else 128)
    factor = _P // dim
    if tiles % factor != 0:
        return _P, 1
    return dim, factor


def _project(hidden_sb, qkv_w, acc_sb, rows, h_tiles, sbm):
    """Add ``hidden_sb^T @ qkv_w`` over this program's ``h_tiles`` H tiles into fp32 ``acc_sb``."""
    _, _, width = qkv_w.shape
    sbm.open_scope(name="qkv_fp32acc_tkg_projection")
    # Keep room for the receive buffer, which is allocated after this scope closes.
    free = sbm.get_free_space() - rows * width * sizeinbytes(nl.float32)
    block_bytes = width * _H_BLOCK_TILES * sizeinbytes(qkv_w.dtype)
    n_blocks = div_ceil(h_tiles, _H_BLOCK_TILES)
    slots = max(1, min(n_blocks, free // block_bytes))
    w_sb = sbm.alloc_stack((_P, slots, _H_BLOCK_TILES, width), dtype=qkv_w.dtype, buffer=nl.sbuf)

    psum = [nl.ndarray((_P, _BANK), dtype=nl.float32, buffer=nl.psum)
            for _ in range(div_ceil(width, _BANK))]
    full_dim, full_factor = _array_tiling(rows, _H_BLOCK_TILES)
    for h_block in TiledRange(h_tiles, _H_BLOCK_TILES):
        dim, factor = (full_dim, full_factor) if h_block.size == _H_BLOCK_TILES \
            else _array_tiling(rows, h_block.size)
        hidden_block = hidden_sb.slice(dim=2, start=h_block.start_offset, end=h_block.end_offset)
        w_block = w_sb.select(dim=1, index=h_block.index % slots).slice(dim=1, start=0, end=h_block.size)
        nisa.dma_copy(w_block, qkv_w.slice(dim=1, start=h_block.start_offset, end=h_block.end_offset))
        for group in range(h_block.size // factor):
            for part in range(factor):
                tile = group * factor + part
                for i_tile in TiledRange(width, _BANK):
                    nisa.nc_matmul(
                        psum[i_tile.index][dim * part:dim * part + rows, 0:i_tile.size],
                        hidden_block.select(dim=2, index=tile),
                        w_block.select(dim=1, index=tile).slice(dim=1, start=i_tile.start_offset,
                                                                end=i_tile.end_offset),
                        tile_position=(0, dim * part),
                        tile_size=(_P, dim),
                    )

    # The PE row groups hold separate partials: add them in fp32.
    dim, factor = full_dim, full_factor
    if h_tiles < _H_BLOCK_TILES:
        dim, factor = _array_tiling(rows, h_tiles)
    for i_tile in TiledRange(width, _BANK):
        cols = acc_sb[0:rows, i_tile.start_offset:i_tile.end_offset]
        for part in range(factor):
            nisa.tensor_tensor(cols, cols, psum[i_tile.index][dim * part:dim * part + rows, 0:i_tile.size],
                               op=nl.add)
    sbm.close_scope()


def qkv_fp32acc_tkg(hidden, qkv_w, norm_w, eps):
    """``[B, S, I]`` bf16: ``rmsnorm(hidden) * norm_w @ qkv_w``, for at most 128 rows.

    Args:
        hidden: ``[B, S, H]`` bf16 in HBM, ``B * S <= 128``, ``H`` in 128-row tiles split
            evenly over the programs.
        qkv_w: ``[H, I]`` bf16, ``I <= 4096``.
        norm_w: ``[1, H]`` bf16 norm gain.
        eps: the norm epsilon.
    """
    _, programs, program = get_verified_program_sharding_info("qkv_fp32acc_tkg", (0, 1))
    batch, seq, size = hidden.shape
    rows = batch * seq
    w_size, width = qkv_w.shape
    kernel_assert(w_size == size and size % (_P * programs) == 0,
                  "H matches the weights and splits into 128-row tiles per program")
    kernel_assert(rows <= _P and width <= _PSUM_BANKS * _BANK, "at most 128 rows and 4096 columns")
    h_tiles = size // _P // programs

    sbm = create_auto_alloc_manager()
    sbm.open_scope(name="qkv_fp32acc_tkg")
    normed = sbm.alloc_stack((_P, rows, size // _P), dtype=hidden.dtype, buffer=nl.sbuf)
    normed = rmsnorm_tkg(input=hidden, gamma=norm_w, output=normed, eps=eps, hidden_actual=size, sbm=sbm)
    normed = normed.slice(dim=2, start=program * h_tiles, end=(program + 1) * h_tiles)
    w_shard = qkv_w.reshape_dim(dim=0, shape=(programs, _P, h_tiles)).select(dim=0, index=program)

    out = nl.ndarray((rows, width), dtype=hidden.dtype, buffer=nl.shared_hbm)
    acc = sbm.alloc_stack((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(acc, value=0)
    _project(normed, w_shard, acc, rows, h_tiles, sbm)
    result = sbm.alloc_stack((rows, width), dtype=hidden.dtype, buffer=nl.sbuf)
    if programs > 1:
        sbm.open_scope(name="qkv_fp32acc_tkg_reduce")
        other = sbm.alloc_stack((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(src=acc, dst=other, send_to_rank=1 - program, recv_from_rank=1 - program, pipe_id=0)
        nisa.tensor_tensor(result, acc, other, op=nl.add)
        sbm.close_scope()
    else:
        nisa.tensor_copy(dst=result, src=acc)
    nisa.dma_copy(out[:, :], result)
    sbm.close_scope()
    return out.reshape((batch, seq, width))
