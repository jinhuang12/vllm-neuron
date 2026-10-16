# SPDX-License-Identifier: Apache-2.0
"""The DSA prefill selection as a keep table: one bit per query and window pool.

The block-shared sparse attention (``functional/attention/mla_block_shared.py``) reads
each query's selection as a bit table over the window's pools, not as a list of ids. This
module builds that table from the indexer's own pool ids, so it holds exactly the pools
the top-k chose: no score is read, so no tie or NaN score can add a pool.

The prefill indexer is row-sharded (``indexer_shard.py``): each rank selects its own
rows and only the pool ids cross ranks. So the table is built where the ids are made,
on each rank's own rows, and all-gathered beside them
(:func:`~vllm_neuron.functional.dsa.indexer_shard.gather_selection_keep`)::

    owning rank:  pool ids [R, k] int32 -> keep_bits [R, P / 8] uint8
    all ranks:    all-gather on dim 0   -> keep_bits [T, P / 8] uint8

The attention unpacks the bits itself; :func:`keep_table_reference` spells the unpack.

Layout, bit-plane major. ``P`` is the window's pool count (a multiple of 8) and
``Q = P / 8`` the bytes of a row. Pool ``p`` covers window rows ``pool_size * p ..`` and
is score column ``p`` of the indexer, so pool order is the score order. Pool ``p`` is bit
``b = p // Q`` of byte ``j = p % Q``, bit 0 the least significant. So bit plane ``b`` of a
row's bytes holds the contiguous pools ``b Q .. (b + 1) Q - 1``. A pool id outside
``0 .. P - 1`` (the sentinel ``-1`` or anything else) sets no bit. The open pool (the
incomplete pool that holds the query's own row, the expansion's tail) is struck by the causal
bound and its pick comes back ``-1``, so its bit is 0.

How the bits are made. Compute engines cannot write a data-dependent address, and a DMA
scatter costs one descriptor per id. So the table is a matmul instead: per query, its
``k`` ids are put on the partitions, ``128`` at a time. Byte ``j`` lies in byte block
``g = j // 128`` at column ``c = j % 128``. The stationary operand is the one-hot of ``g``
scaled by ``2^b`` and the moving operand the one-hot of ``c``; their product summed over
the ids is ``[g, c]`` = the byte ``128 g + c``. A row's ids are distinct, so each byte is a
sum of distinct powers of two at most 255: exact in bf16 operands and the fp32 PSUM.

Distinct ids are the selector's contract after its sentinel: a pool id the selector
returns twice comes back beside a struck value, and ``dsa_causal_sentinel`` turns that
pick, and every NaN or bound-fill pick, into ``-1`` (``model_fp8.py``, the causal-bound
selection). Two equal ids in a row would carry into the next plane's bit.

The entry is a pure function of the ids and reads no knob. It is called only when a reader
consumes the table this layer-chunk; each reader (the block-shared and the dense-masked
attention) gates itself on its own knob.
"""

from __future__ import annotations

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.source_digest import source_digest
from vllm_neuron.utils.neuron_utils import SBUF_PARTITIONS, can_run_kernel

#: Ids one matmul contracts: one per partition.
SLOT_TILE = SBUF_PARTITIONS
#: Bit planes of the table: the bits of one byte.
POOLS_PER_BYTE = 8
#: Bytes of one byte block: the moving operand's columns, one byte each.
BLOCK_BYTES = SBUF_PARTITIONS
#: Byte blocks one matmul's stationary operand covers (its free size).
BLOCKS_PER_MATMUL = SBUF_PARTITIONS
#: Query rows whose one-hot operands one step builds, one instruction per operand. At
#: four, one step's Vector work overlaps the previous step's matmuls; the compiler's
#: schedule is longer at two or eight.
QUERY_STEP = 4

_BLOCK_SHIFT = (BLOCK_BYTES - 1).bit_length()

#: This file's digest as a trace-time int: the compiled-kernel cache keys on the entry's
#: own source and arguments only.
SOURCE_DIGEST = source_digest(__file__)


class SelectionKeepError(ValueError):
    """Raised for a call this module does not serve. There is no fallback here."""


def _sb(shape, dtype):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)


def _slot_codes(ids_t, cols, plane_starts, plane_bytes):
    """Per id (``ids_t`` ``[128, cols]`` int32): ``(block, column, weight)``, each
    ``[128, cols]`` fp32, for the byte ``j = id % Q`` and the plane ``b = id // Q``
    (``Q = plane_bytes``): ``block = j >> 7``, ``column = j & 127``, ``weight = 2^b``.

    ``b`` counts the plane starts ``Q, 2 Q, .. 7 Q`` (``plane_starts`` ``[128, 7]`` int32)
    at or below the id: one compare against all of them and one sum. Vector does that
    arithmetic in fp32, which is exact for every pool id. A negative id has ``b = 0`` and a
    negative byte, so a negative block, which no one-hot column matches; an id at or past
    ``P`` has ``b = 7`` and a byte at or past ``Q``, which is not stored.
    """
    starts = POOLS_PER_BYTE - 1
    at_or_past = _sb((SLOT_TILE, cols, starts), nl.int32)
    nisa.tensor_tensor(dst=at_or_past, data1=_along(ids_t, cols, starts),
                       data2=_across(plane_starts, cols, starts), op=nl.greater_equal,
                       engine=nisa.engine.vector)
    plane_i = _sb((SLOT_TILE, cols), nl.int32)
    nisa.tensor_reduce(dst=plane_i, op=nl.add, data=at_or_past, axis=2)
    # j = id - b Q
    byte_i = _sb((SLOT_TILE, cols), nl.int32)
    nisa.scalar_tensor_tensor(dst=byte_i, data=plane_i, op0=nl.multiply,
                              operand0=-float(plane_bytes), op1=nl.add, operand1=ids_t)
    block_i = _sb((SLOT_TILE, cols), nl.int32)
    nisa.tensor_scalar(dst=block_i, data=byte_i, op0=nl.right_shift, operand0=_BLOCK_SHIFT,
                       engine=nisa.engine.vector)
    column_i = _sb((SLOT_TILE, cols), nl.int32)
    nisa.tensor_scalar(dst=column_i, data=byte_i, op0=nl.bitwise_and, operand0=BLOCK_BYTES - 1,
                       engine=nisa.engine.vector)
    weight_i = _sb((SLOT_TILE, cols), nl.int32)
    nisa.tensor_scalar(dst=weight_i, data=plane_i, op0=nl.left_shift, operand0=1, reverse0=True,
                       engine=nisa.engine.vector)
    codes = []
    for src in (block_i, column_i, weight_i):
        dst = _sb((SLOT_TILE, cols), nl.float32)
        nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.vector)
        codes.append(dst)
    return codes


def _along(codes, cols, width):
    """``codes`` ``[128, cols]`` read as ``[128, cols, width]``: each column ``width`` times."""
    return codes.ap(pattern=[[cols, SLOT_TILE], [1, cols], [0, width]])


def _across(ramp, cols, width):
    """``ramp`` ``[128, width]`` read as ``[128, cols, width]``: the whole ramp ``cols`` times."""
    return ramp.ap(pattern=[[width, SLOT_TILE], [0, cols], [1, width]])


def _keep_bits_rows(pool_ids_hbm, bits_hbm, r0, rows):
    """Keep bits of query rows ``r0 .. r0 + rows - 1`` into ``bits_hbm`` (module docstring).

    The rows go :data:`QUERY_STEP` at a time. A step builds the one-hot operands of all its
    ids at once, one wide instruction per operand (each code repeated along the ramp and
    the ramp along the codes), so each matmul only slices them.
    """
    select_k = pool_ids_hbm.shape[1]
    keep_bytes = bits_hbm.shape[1]
    chunks = select_k // SLOT_TILE
    blocks = (keep_bytes + BLOCK_BYTES - 1) // BLOCK_BYTES
    column_ramp = _sb((SLOT_TILE, BLOCK_BYTES), nl.float32)
    nisa.iota(dst=column_ramp, pattern=[[1, BLOCK_BYTES]], offset=0, channel_multiplier=0)
    plane_starts = _sb((SLOT_TILE, POOLS_PER_BYTE - 1), nl.int32)
    nisa.iota(dst=plane_starts, pattern=[[keep_bytes, POOLS_PER_BYTE - 1]], offset=keep_bytes,
              channel_multiplier=0)
    for q0 in range(0, rows, QUERY_STEP):
        qn = min(QUERY_STEP, rows - q0)
        cols = qn * chunks
        # column q * chunks + t holds the ids t * 128 .. t * 128 + 127 of query r0 + q0 + q
        ids_t = _sb((SLOT_TILE, cols), nl.int32)
        nisa.dma_copy(dst=ids_t, src=pool_ids_hbm.ap(pattern=[[1, SLOT_TILE], [SLOT_TILE, cols]],
                                                     offset=(r0 + q0) * select_k),
                      dge_mode=nisa.dge_mode.none)
        codes = _slot_codes(ids_t, cols, plane_starts, keep_bytes)
        block_f = codes[0]
        column_f = codes[1]
        weight_f = codes[2]
        # column_hot[:, c * 128 + x] = (column of id c == x)
        column_hot = _sb((SLOT_TILE, cols * BLOCK_BYTES), nl.bfloat16)
        nisa.tensor_tensor(dst=column_hot, data1=_along(column_f, cols, BLOCK_BYTES),
                           data2=_across(column_ramp, cols, BLOCK_BYTES), op=nl.equal,
                           engine=nisa.engine.vector)
        for g0 in range(0, blocks, BLOCKS_PER_MATMUL):
            gw = min(BLOCKS_PER_MATMUL, blocks - g0)
            block_ramp = _sb((SLOT_TILE, gw), nl.float32)
            nisa.iota(dst=block_ramp, pattern=[[1, gw]], offset=g0, channel_multiplier=0)
            # block_hot[:, c * gw + g] = weight of id c where its block is g0 + g, else 0. Every
            # one-hot instruction runs on Vector: GpSimd has no float is_equal for two tensors
            # (NCC_IBIR766), and the compiler's schedule prices its float multiply at about
            # 8 ns per element, against about 1 ns on Vector.
            block_eq = _sb((SLOT_TILE, cols * gw), nl.float32)
            nisa.tensor_tensor(dst=block_eq, data1=_along(block_f, cols, gw),
                               data2=_across(block_ramp, cols, gw), op=nl.equal,
                               engine=nisa.engine.vector)
            block_hot = _sb((SLOT_TILE, cols * gw), nl.bfloat16)
            nisa.tensor_tensor(dst=block_hot, data1=block_eq, data2=_along(weight_f, cols, gw),
                               op=nl.multiply, engine=nisa.engine.vector)
            staged = _sb((gw, qn, BLOCK_BYTES), nl.uint8)
            for q in range(qn):
                acc = nl.ndarray((gw, BLOCK_BYTES), dtype=nl.float32, buffer=nl.psum)
                for t in range(chunks):
                    c = q * chunks + t
                    nisa.nc_matmul(dst=acc, stationary=block_hot[:, c * gw:(c + 1) * gw],
                                   moving=column_hot[:, c * BLOCK_BYTES:(c + 1) * BLOCK_BYTES],
                                   accumulate=(t > 0))
                nisa.tensor_copy(dst=staged[:, q, :], src=acc, engine=nisa.engine.scalar)
            _store_blocks(bits_hbm, staged, r0 + q0, qn, g0, gw)


def _store_blocks(bits_hbm, staged, row0, rows, g0, gw):
    """Blocks ``g0 .. g0 + gw - 1`` of rows ``row0 ..`` (``staged`` ``[gw, rows, 128]``) to HBM:
    the whole blocks, then the last block's bytes below the table width."""
    keep_bytes = bits_hbm.shape[1]
    whole = max(0, min(gw, keep_bytes // BLOCK_BYTES - g0))
    if whole > 0:
        nisa.dma_copy(dst=bits_hbm.ap(pattern=[[BLOCK_BYTES, whole], [keep_bytes, rows],
                                               [1, BLOCK_BYTES]],
                                      offset=row0 * keep_bytes + g0 * BLOCK_BYTES),
                      src=staged[0:whole, :, :], dge_mode=nisa.dge_mode.none)
    tail = keep_bytes - (g0 + whole) * BLOCK_BYTES
    if whole < gw and tail > 0:
        nisa.dma_copy(dst=bits_hbm.ap(pattern=[[keep_bytes, 1], [keep_bytes, rows], [1, tail]],
                                      offset=row0 * keep_bytes + (g0 + whole) * BLOCK_BYTES),
                      src=staged[whole:whole + 1, :, 0:tail], dge_mode=nisa.dge_mode.none)


@nki.jit
def selection_keep_kernel(pool_ids_hbm, pools, source_digest=0):
    """Keep bits of a rank's selected rows.

    Args:
        pool_ids_hbm: ``[R, k]`` int32, each row's selected pool ids, distinct, ``-1`` for
            none; ``k`` a multiple of :data:`SLOT_TILE`.
        pools: python int, the window's pools ``P``, a multiple of :data:`POOLS_PER_BYTE`.
        source_digest: python int that only keys the compiled-kernel cache on this file.

    Returns:
        ``keep_bits [R, P / 8]`` uint8 in shared HBM, bit-plane major (module docstring).

    Under a two-program launch each program takes half of the rows (``R`` even).
    """
    rows = pool_ids_hbm.shape[0]
    bits_hbm = nl.ndarray((rows, pools // POOLS_PER_BYTE), dtype=nl.uint8, buffer=nl.shared_hbm)
    share = rows // nl.num_programs(axes=0)
    r0 = nl.program_id(0) * share
    _keep_bits_rows(pool_ids_hbm, bits_hbm, r0, share)
    return bits_hbm


def _require_keep_call(pool_ids: Tensor, pools: int) -> int:
    pools = int(pools)
    if pools < POOLS_PER_BYTE or pools % POOLS_PER_BYTE != 0:
        raise SelectionKeepError(
            f"pools must be a positive multiple of {POOLS_PER_BYTE}; got {pools}"
        )
    if pool_ids.ndim != 2 or pool_ids.dtype != torch.int32:
        raise SelectionKeepError(
            f"pool_ids must be [rows, select_k] int32; got {tuple(pool_ids.shape)} {pool_ids.dtype}"
        )
    rows, select_k = (int(d) for d in pool_ids.shape)
    if rows < 1 or select_k < SLOT_TILE or select_k % SLOT_TILE != 0:
        raise SelectionKeepError(
            f"pool_ids must hold at least one row and a multiple of {SLOT_TILE} ids per row; "
            f"got [{rows}, {select_k}]"
        )
    return pools


def selection_keep_reference(pool_ids: Tensor, pools: int) -> Tensor:
    """The torch spelling of :func:`selection_keep_kernel`: ``keep_bits`` on any device."""
    pools = int(pools)
    rows = int(pool_ids.shape[0])
    ids = pool_ids.to(torch.int64)
    inside = (ids >= 0) & (ids < pools)
    plain = torch.zeros(rows, pools + 1, dtype=torch.int32, device=pool_ids.device)
    plain = plain.scatter(1, torch.where(inside, ids, torch.full_like(ids, pools)),
                          torch.ones_like(ids, dtype=torch.int32))
    # [rows, plane b, byte j]: pool b Q + j -> bit b of byte j
    planes = plain[:, :pools].reshape(rows, POOLS_PER_BYTE, pools // POOLS_PER_BYTE)
    weights = 2 ** torch.arange(POOLS_PER_BYTE, dtype=torch.int32, device=pool_ids.device)
    bits = (planes * weights.view(1, POOLS_PER_BYTE, 1)).sum(dim=1, dtype=torch.int32)
    return bits.to(torch.uint8)


def keep_table_reference(keep_bits: Tensor) -> Tensor:
    """``keep_bits [S, P / 8]`` as one byte per pool: ``[S, P]`` uint8, 1 where selected.

    The torch spelling of the layout (module docstring): what a consumer's unpack must
    produce. Plane ``b`` of the bytes fills columns ``b P / 8 .. (b + 1) P / 8 - 1``.
    """
    rows = int(keep_bits.shape[0])
    shifts = torch.arange(POOLS_PER_BYTE, dtype=torch.int32, device=keep_bits.device)
    plain = (keep_bits.to(torch.int32).unsqueeze(1) >> shifts.view(1, POOLS_PER_BYTE, 1)) & 1
    return plain.reshape(rows, -1).to(torch.uint8)


def dsa_selection_keep(pool_ids: Tensor, pools: int) -> Tensor:
    """Keep bits of a rank's own selected rows.

    Args:
        pool_ids: ``[R, k]`` int32, the rows' selected pool ids as the causal-bound
            selection returns them (distinct, ``-1`` for none, any order); ``k`` a multiple
            of :data:`SLOT_TILE`.
        pools: the window's pools ``P`` (pages x page size / pool size), a multiple of
            :data:`POOLS_PER_BYTE`.

    Returns:
        ``keep_bits``: ``[R, P / 8]`` uint8, bit-plane major (module docstring). The NKI
        kernel serves the call where kernels run; elsewhere the torch reference.

    Raises:
        SelectionKeepError: for a malformed call.
    """
    pools = _require_keep_call(pool_ids, pools)
    if not can_run_kernel(pool_ids):
        return selection_keep_reference(pool_ids, pools)
    call = wrap_nki(selection_keep_kernel)
    if lnc_pair() and int(pool_ids.shape[0]) % 2 == 0:
        call = call[2]
    return call(pool_ids.contiguous(), pools, SOURCE_DIGEST)
