# SPDX-License-Identifier: Apache-2.0
"""Sum each token's expert rows of the fused MoE emission, as an NKI kernel.

Per token::

    out[t] = sum_j valid[t, j] * contribution[index[t, j]]      # fp32, j in slot order

The fused expert kernel emits ``[blocks * rows, H]`` fp32 rows sized for the worst
case, so most rows are padding. Gathering each token's ``K`` rows reads only
``T * K`` rows, where a scatter-add over the emission reads every row and runs as a
serial GpSimd accumulate.

The kernel puts 128 tokens on the partitions. For each hidden tile and each slot, one
indirect DMA gathers the slot's rows, and the vector engine adds them, weighted by
the slot's 0/1 validity, into an fp32 accumulator. The weight is a multiply, so an
invalid slot must still point at a finite row; the caller points it at row 0. A
two-program grid partitions the token tiles across the two PNCs of one LNC2 core.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

#: Tokens per tile: the partition count.
TOKEN_TILE = 128

#: Hidden elements per gathered row piece. 4096 fp32 elements are 16 KB per partition,
#: so one DMA moves 2 MB, and a tile's eight slot buffers and its accumulator fit SBUF
#: together. Measured at T=1024, K=8, H=4096 on one LNC2 core: 0.46 ms, against
#: 0.48 ms at 2048 and 0.63 ms for the torch gather.
HIDDEN_TILE = 4096

#: int32 elements per 32-byte line. Each index tile is one whole line wide and uses
#: one column, so every tile placed after it stays line-aligned.
_LINE = 8


def _index_tile(index_hbm, t0: int, slot: int, slots: int):
    """A ``[128, 1]`` int32 tile holding ``index[t0:t0 + 128, slot]``."""
    tile = nl.ndarray((TOKEN_TILE, _LINE), dtype=nl.int32, buffer=nl.sbuf)[:, 0:1]
    nisa.dma_copy(dst=tile, src=index_hbm.ap(pattern=[[slots, TOKEN_TILE], [1, 1]],
                                             offset=t0 * slots + slot))
    return tile


@nki.jit
def token_gather_combine_kernel(contribution_hbm, index_hbm, valid_hbm):
    """``[R, H]`` fp32 rows, ``[T, K]`` int32 row indices, ``[T, K]`` fp32 0/1 -> ``[T, H]`` fp32."""
    total_rows, hidden = contribution_hbm.shape
    tokens, slots = index_hbm.shape
    tile_h = HIDDEN_TILE
    pieces = hidden // tile_h
    # One row piece per index, so a gather carries only a tensor-borne offset.
    banked = contribution_hbm.reshape((total_rows * pieces, tile_h))
    out_hbm = nl.ndarray((tokens, hidden), dtype=nl.float32, buffer=nl.shared_hbm)

    n_prgs = nl.num_programs(axes=0)
    prg_id = nl.program_id(0)
    for tb in nl.affine_range(prg_id, tokens // TOKEN_TILE, n_prgs):
        t0 = tb * TOKEN_TILE
        valid = nl.ndarray((TOKEN_TILE, slots), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=valid, src=valid_hbm.ap(pattern=[[slots, TOKEN_TILE], [1, slots]],
                                                  offset=t0 * slots))
        rows = []
        for j in range(slots):
            rows.append(_index_tile(index_hbm, t0, j, slots))
        for piece in range(pieces):
            acc = nl.ndarray((TOKEN_TILE, tile_h), dtype=nl.float32, buffer=nl.sbuf)
            for j in range(slots):
                at = nl.ndarray((TOKEN_TILE, _LINE), dtype=nl.int32, buffer=nl.sbuf)[:, 0:1]
                nisa.tensor_scalar(dst=at, data=rows[j], op0=nl.multiply, operand0=pieces,
                                   op1=nl.add, operand1=piece)
                picked = nl.ndarray((TOKEN_TILE, tile_h), dtype=nl.float32, buffer=nl.sbuf)
                src = banked.ap(pattern=[[tile_h, TOKEN_TILE], [1, tile_h]],
                                vector_offset=at, indirect_dim=0)
                # Software descriptors: a per-partition row offset has no hwdge form.
                nisa.dma_copy(dst=picked, src=src)
                if j == 0:
                    nisa.tensor_scalar(dst=acc, data=picked, op0=nl.multiply,
                                       operand0=valid[:, 0:1], engine=nisa.engine.vector)
                else:
                    nisa.scalar_tensor_tensor(dst=acc, data=picked, op0=nl.multiply,
                                              operand0=valid[:, j:j + 1], op1=nl.add,
                                              operand1=acc)
            nisa.dma_copy(dst=out_hbm.ap(pattern=[[hidden, TOKEN_TILE], [1, tile_h]],
                                         offset=t0 * hidden + piece * tile_h),
                          src=acc)
    return out_hbm


def token_gather_combine_torch(contribution: Tensor, index: Tensor, valid: Tensor) -> Tensor:
    """The same sum in torch, slot by slot; the reference and the small-shape path."""
    combined = torch.zeros(index.shape[0], contribution.shape[1], dtype=torch.float32,
                           device=contribution.device)
    for j in range(index.shape[1]):
        gathered = contribution.index_select(0, index[:, j].to(torch.int64))
        combined = combined + torch.where(valid[:, j:j + 1] > 0, gathered, 0.0)
    return combined


def can_run_token_gather_combine(contribution: Tensor, index: Tensor) -> bool:
    """Whole token tiles, whole hidden tiles, fp32 rows, and a device that runs NKI."""
    return (
        can_run_kernel(contribution)
        and contribution.dtype == torch.float32
        and index.shape[0] % TOKEN_TILE == 0
        and contribution.shape[1] % HIDDEN_TILE == 0
    )


def token_gather_combine(contribution: Tensor, index: Tensor, valid: Tensor) -> Tensor:
    """``[T, H]`` fp32: each token's valid rows of ``contribution`` summed in slot order.

    Args:
        contribution: ``[R, H]`` fp32 rows.
        index: ``[T, K]`` integer row indices; an invalid slot must name a finite row.
        valid: ``[T, K]`` 0/1, which slots count.
    """
    if not can_run_token_gather_combine(contribution, index):
        return token_gather_combine_torch(contribution, index, valid)
    call = wrap_nki(token_gather_combine_kernel)
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2" and index.shape[0] >= 2 * TOKEN_TILE:
        call = call[2]
    return call(
        contribution.contiguous(),
        index.to(torch.int32).contiguous(),
        valid.to(torch.float32).contiguous(),
    )
