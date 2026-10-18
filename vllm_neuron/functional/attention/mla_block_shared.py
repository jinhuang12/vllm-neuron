# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA prefill attention with one shared gather per block of neighbouring queries.

The sparse kernel (``mla_sparse.py``) gathers each query's selected rows one query at a
time, and those gathers are serial on the software DMA queue. Neighbouring queries select
largely the same rows, so this kernel gathers once per block of ``block_queries``
consecutive queries: the union of the rows its queries select. Each query keeps its own
selection as a per-row predicate ahead of the softmax::

    scores[q, u] = q_lift[q] . window[union[b, u]]   where member[q, u], else _UNSELECTED_SCORE
    out[q]       = softmax(scale * scores[q]) @ window[union[b]]

``union`` is ``[blocks, union_cap]`` int32: block ``b``'s window rows, ``-1`` past its
last one; ``member`` is ``[S, union_cap]`` uint8: 1 where query ``s`` selects union
position ``u`` of its block. A pad position reads window row 0 and no query selects it,
so it adds exactly 0.0 to every sum.

The union producer (``mla_block_shared_union``) builds both on the device from the
chunk's keep bits: one bit per query and pool, set where the query's top-k pool ids name
the pool (exact; built from the ids and gathered across ranks before the call). Query
``s`` sits at window row ``query_start + s``; a pool is ``pool_size`` rows; the 128
queries of block ``b`` own the page of rows ``b0 .. b0 + 127`` (``b0 = query_start + 128
b``), pools from ``pb = b0 / pool_size``:

* keep: ``keep_bits`` ``[S, P / 8]`` uint8, bit-plane major: pool ``p`` is bit
  ``p // (P / 8)`` of byte ``p % (P / 8)``, bit 0 the least significant. Only complete
  pools at or below a query are set; the query's open pool is not (the member adds it);
* union: positions ``0 .. 127`` the page rows ``b0 + j``; from 128, ``pool_size`` rows
  per shared pool (a pool below ``pb`` that some query of the block keeps), ascending,
  at most ``union_cap / pool_size - 128 / pool_size`` pools; ``-1`` after;
* member: page row ``j`` for query ``q`` of the block where ``j <= q`` and its pool is
  ``q``'s open pool (or past it) or a complete pool ``q`` keeps; a shared row where
  ``q`` keeps its pool;
* totals: per block, the shared pools (uncut) and an overflow flag, 1 where they are
  more than the cap holds and the union is cut. The producer reports it and decides
  nothing.

Per block the union is read a superblock (:data:`SUPERBLOCK_ROWS` positions) at a time:
the window rows are translated to bank rows through the block table, gathered by one
software-queue DMA transpose with the latent on the partitions (MM1's moving operand,
the sparse kernel's gather form) and put back on the key partitions by PE transposes
(MM2's moving operand). Each 128-query tile of the block keeps a running row maximum, a
running row sum and an fp32 accumulator over the superblocks (an online softmax): a
superblock's scores leave PSUM through the predicate (an unselected position reads
_UNSELECTED_SCORE, so its probability is exactly 0.0), are exponentiated against the new
running maximum, put on the key partitions by fp32 PE transposes, split bf16 hi/lo and
contracted with the rows. The accumulator is rescaled by ``exp(scale * (old max - new
max))`` before the superblock's sum is added, and divided by the row sum once, at the end.

A program runs its blocks two at a time, one superblock of each in flight: the PE takes
one block's MM1 and turns while the Vector and Scalar engines take the other's select and
exp. Each block's own steps and their order do not change, so its output does not either.

Arithmetic against the sparse low-precision body: the same MM1 products into fp32 PSUM
and the same hi/lo split against the stored cache, but the probabilities are split before
they are normalised, the sums run over the union in its order, and the maximum is a
running one. The answer is the sparse kernel's up to the order and the rounding of these
fp32 steps: value-equal, not bit-equal.

Non-finite rows. MM1 keeps a non-finite row, so a query that selects it gets a non-finite
score, as on the gather path; the predicate drops it for every other query. MM2 contracts
every union row whatever its probability, and ``0 * NaN`` is NaN, so each superblock's
MM2 operand has its non-finite elements set to 0.0 first.

A block whose shared pools do not fit its cap has a cut union (its overflow flag in
``totals`` is 1); this kernel still writes rows for it from what fits. The caller runs
``mla_block_shared_exact`` right after it on every call: a kernel of its own, with fixed
addresses and the device loops, which rewrites those blocks' rows by today's per-query
gather. This kernel holds no device loop: in one graph a fixed-address kernel with a
device loop (the served MoE combine) before a device loop without fixed addresses does
not compile.

Both entries refuse a call unless ``VLLM_NEURON_MLA_BLOCK_SHARED`` (``vllm_neuron/envs.py``)
is set; the DSA prefill calls them through ``model/glm5_next/dsa_block_shared.py``.
``mla_block_shared_serves`` and ``mla_block_shared_union_cap`` answer its trace-time
questions and read no knob.
"""

from __future__ import annotations

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron import envs
from vllm_neuron.functional.attention import mla_dense_window as _dw
from vllm_neuron.functional.attention import mla_sparse as _ms
from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.source_digest import source_digest

#: Partition extent: query rows are tiled to it, union rows are chunked to it, the
#: latent is tiled to it.
ROW_TILE = 128
KEY_CHUNK = 128
LATENT_TILE = 128
#: Moving free-axis extent of one matmul: MM1 scores 512 union rows per instruction.
MOVING_MAX = 512
#: The widest latent one fp32 PSUM bank holds for the MM2 result row (2 KB).
LATENT_MAX = 512
#: PE transposes of 128 fp32 columns that fill one PSUM bank.
_TRANSPOSES_PER_BANK = MOVING_MAX // ROW_TILE
#: PSUM bytes per partition of one bank.
_PSUM_BANK_BYTES = 2048
#: Union positions per superblock: the rows one gather, one pass of MM1, the softmax and
#: MM2 cover before the running statistics are folded. One gather takes them all: 16
#: chunks of 128 rows, 8,192 elements per partition at latent 512, half the 15,872 that one
#: indirect DMA may write per partition. Per partition a superblock holds its rows twice
#: (latent on partitions and keys on partitions: 2 x 16 KiB at latent 512 in bf16) and its
#: guard mask (8 KiB), and each query tile adds its predicate (2 KiB), scores (8 KiB) and
#: one bank of hi/lo halves (2 KiB); a program has :data:`_PAIR` of them in flight.
SUPERBLOCK_ROWS = 2048
#: The score an unselected position reads: a power of two so far below any score that
#: ``exp(scale * (_UNSELECTED_SCORE - max))`` is exactly 0.0.
_UNSELECTED_SCORE = -(2.0 ** 30)
#: The running maximum starts here: a query that has selected no row yet exponentiates
#: every row to 0.0, and one that selects none returns zeros.
_SELECTED_MAX_FLOOR = _UNSELECTED_SCORE / 2
#: The row-sum floor of the final division.
_TOTAL_FLOOR = 1.0e-30

#: This file's digest as a trace-time int: the compiled-kernel cache keys on the entry's
#: own source and arguments only.
SOURCE_DIGEST = source_digest(__file__)

_sb = _dw._sb
_col = _dw._col


class MlaBlockSharedError(ValueError):
    """Raised for a call this kernel does not serve. There is no fallback here."""


#: Blocks a program runs together. Each superblock of the pair is emitted stage-major:
#: both gathers, then per block MM1 and its turns, then per block the softmax and MM2. So
#: while the Vector and Scalar engines take one block's select and exp, the PE has the
#: other block's MM1 and turns, and the PE stop of a lone block (MM1 done, the exp not yet)
#: has work in it.
_PAIR = 2

# Positions in the lists the stages return (the kernel front end resolves plain names and
# integer subscripts, not tuple targets).
_PART_SCORES = 0   # [128, superblock] fp32: the selected scores of one query tile
_PART_MAX = 1      # [128, superblock / 512] fp32: their maximum per 512 positions
_ST_Q = 0          # per tile: the queries with the latent on the partitions
_ST_MAX = 1        # per tile: the running maximum
_ST_SUM = 2        # per tile: the running sum
_ST_ACC = 3        # per tile: the unnormalised output
_ST_ROW = 4        # per tile: its first query row


def _union_offsets(union_hbm, bank_rows, block, u0, n_chunks, page_size):
    """Bank rows of union positions ``u0 .. u0 + 128 * n_chunks - 1`` of ``block``, chunk-major.

    Element ``(p, c)`` of the ``[128, n_chunks]`` uint32 result is position ``u0 + c * 128
    + p``: the gather's offset order (``mla_sparse._lowp_offsets``). A ``-1`` pad reads
    window row 0. Window row ``r`` is row ``r & (page_size - 1)`` of page ``r >>
    log2(page_size)``; ``bank_rows`` (``mla_sparse._lowp_bank_table``) holds each page's
    first bank row on every partition. Every row is far below 2**24, so the fp32
    arithmetic of the clamp is exact. The page numbers, the gathered first rows and the
    places in the pages are three column ranges of one uint32 tile, so ``nc_n_gather``'s
    destination never overlaps the indices it reads.
    """
    cap = union_hbm.shape[1]
    width = _ms._aligned(n_chunks)
    raw = nl.ndarray((KEY_CHUNK, width), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=raw[:, 0:n_chunks],
                  src=union_hbm.ap(pattern=[[1, KEY_CHUNK], [KEY_CHUNK, n_chunks]],
                                   offset=block * cap + u0),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    offsets = nl.ndarray((KEY_CHUNK, width), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=offsets[:, 0:n_chunks], data=raw[:, 0:n_chunks], op0=nl.maximum,
                       operand0=0.0, engine=nisa.engine.vector)
    lookup = nl.ndarray((KEY_CHUNK, 3 * width), dtype=nl.uint32, buffer=nl.sbuf)
    page_of = lookup[:, 0:n_chunks]
    first_row = lookup[:, width:width + n_chunks]
    in_page = lookup[:, 2 * width:2 * width + n_chunks]
    nisa.tensor_scalar(dst=page_of, data=offsets[:, 0:n_chunks], op0=nl.right_shift,
                       operand0=_ms._log2_exact(page_size), engine=nisa.engine.vector)
    nisa.tensor_scalar(dst=in_page, data=offsets[:, 0:n_chunks], op0=nl.bitwise_and,
                       operand0=page_size - 1, engine=nisa.engine.vector)
    nisa.nc_n_gather(dst=first_row, data=bank_rows, indices=page_of)
    nisa.tensor_tensor(dst=offsets[:, 0:n_chunks], data1=first_row, data2=in_page, op=nl.add,
                       engine=nisa.engine.gpsimd)
    return offsets


def _gather_rows(bank_hbm, offsets, n_chunks):
    """``rows_t`` ``[128, 1, L / 128, n_chunks * 128]``: the bank rows ``offsets`` names.

    One software-queue DMA transpose gathers every row with the latent on the partitions
    (``rows_t[l, 0, li, k]`` = latent element ``li * 128 + l`` of the row at position
    ``k``), the sparse kernel's gather. It is MM1's moving operand, the row as stored.
    """
    latent = bank_hbm.shape[1]
    n_lat = latent // LATENT_TILE
    n_rows = n_chunks * KEY_CHUNK
    rows_t = nl.ndarray((LATENT_TILE, 1, n_lat, n_rows), dtype=bank_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_transpose(
        dst=rows_t,
        src=bank_hbm.ap(pattern=[[latent, n_rows], [0, 1], [LATENT_TILE, n_lat], [1, LATENT_TILE]],
                        vector_offset=offsets[:, 0:n_chunks], indirect_dim=0),
        dge_mode=nisa.dge_mode.swdge)
    return rows_t


def _turn_rows(rows_t):
    """``rows`` ``[128, n_chunks, L]``: ``rows_t`` with the keys on the partitions, finite.

    Position ``c * 128 + p`` sits on partition ``p`` of chunk ``c``. PE transposes build it,
    as many chunks as fill one PSUM bank copied out at once on the scalar engine. Then every
    non-finite element of those chunks is set to 0.0, bank by bank, so MM2 can start on the
    first chunks: ``rows`` feeds MM2 only.
    """
    n_lat = rows_t.shape[2]
    latent = n_lat * LATENT_TILE
    n_chunks = rows_t.shape[3] // KEY_CHUNK
    kv_dtype = rows_t.dtype
    rows = _sb((KEY_CHUNK, n_chunks, latent), kv_dtype)
    finite = _sb((KEY_CHUNK, n_chunks, latent), nl.uint8)
    # A 2-byte chunk on the key partitions is 2 L bytes per partition.
    per_bank = _PSUM_BANK_BYTES // (2 * latent)
    for c0 in range(0, n_chunks, per_bank):
        nc = min(per_bank, n_chunks - c0)
        t_ps = nl.ndarray((KEY_CHUNK, per_bank, latent), dtype=kv_dtype, buffer=nl.psum)
        for j in range(nc):
            k0 = (c0 + j) * KEY_CHUNK
            for li in range(n_lat):
                nisa.nc_transpose(dst=t_ps[:, j, li * LATENT_TILE:(li + 1) * LATENT_TILE],
                                  data=rows_t[:, 0, li, k0:k0 + KEY_CHUNK])
        got = rows[:, c0:c0 + nc, :]
        nisa.tensor_copy(dst=got, src=t_ps[:, 0:nc, :], engine=nisa.engine.scalar)
        # x * 0 is 0 for a finite x and NaN for an infinite or NaN one.
        nisa.tensor_scalar(dst=finite[:, c0:c0 + nc, :], data=got, op0=nl.multiply, operand0=0.0,
                           op1=nl.equal, operand1=0.0, engine=nisa.engine.vector)
        nisa.select_reduce(dst=got, predicate=finite[:, c0:c0 + nc, :], on_true=got,
                           on_false=0.0)
    return rows


def _union_scores(q_t, rows_t, t0):
    """``[128, 512]`` fp32 PSUM: ``q . row`` for union positions ``t0 .. t0 + 511`` of the
    superblock, one matmul per latent tile."""
    n_lat = q_t.shape[1]
    s_ps = nl.ndarray((ROW_TILE, MOVING_MAX), dtype=nl.float32, buffer=nl.psum)
    for li in range(n_lat):
        nisa.nc_matmul(dst=s_ps, stationary=q_t[:, li, :],
                       moving=rows_t[:, 0, li, t0:t0 + MOVING_MAX], accumulate=(li > 0))
    return s_ps


def _tile_scores(member_hbm, q_t, rows_t, r0, u0):
    """MM1 of query rows ``r0 .. r0 + 127`` over one superblock (union positions ``u0 ..``).

    ``q_t`` is the tile's queries with the latent on the partitions
    (``mla_dense_window._query_columns``). Returns ``[scores, block_max]`` (the
    ``_PART_*`` positions): the scores where the tile's member bits select the position,
    ``_UNSELECTED_SCORE`` elsewhere, and their maximum per 512 positions.
    """
    cap = member_hbm.shape[1]
    width = rows_t.shape[3]
    n_blocks = width // MOVING_MAX
    pred = _sb((ROW_TILE, width), nl.uint8)
    nisa.dma_copy(dst=pred, src=member_hbm.ap(pattern=[[cap, ROW_TILE], [1, width]],
                                              offset=r0 * cap + u0))
    scores = _sb((ROW_TILE, width), nl.float32)
    block_max = _sb((ROW_TILE, n_blocks), nl.float32)
    for b in range(n_blocks):
        t0 = b * MOVING_MAX
        s_ps = _union_scores(q_t, rows_t, t0)
        nisa.select_reduce(dst=scores[:, t0:t0 + MOVING_MAX], predicate=pred[:, t0:t0 + MOVING_MAX],
                           on_true=s_ps, on_false=_UNSELECTED_SCORE,
                           reduce_res=block_max[:, b:b + 1],
                           reduce_cmd=nisa.reduce_cmd.reset_reduce, reduce_op=nl.maximum)
    part = []
    part.append(scores)
    part.append(block_max)
    return part


def _tile_fold(part, rows, state, softmax_scale):
    """Fold one superblock's ``part`` (:func:`_tile_scores`) into one tile's ``state``.

    ``rows`` is the superblock with the keys on the partitions (:func:`_turn_rows`).
    ``state`` holds the tile's running statistics (the ``_ST_*`` positions), updated in
    place: the raw-score maximum and the sum of ``exp(scale * (score - run_max))``
    (``[128, 1]`` fp32), and the unnormalised output against that maximum (``[128, L]``
    fp32).
    """
    scores = part[_PART_SCORES]
    block_max = part[_PART_MAX]
    run_max = state[_ST_MAX]
    run_sum = state[_ST_SUM]
    acc = state[_ST_ACC]
    latent = rows.shape[2]
    kv_dtype = rows.dtype
    n_blocks = scores.shape[1] // MOVING_MAX

    # ---- the running maximum, the rescale of what came before, the exps and their sum --
    new_max = _col(ROW_TILE, nl.float32)
    nisa.tensor_reduce(dst=new_max, op=nl.maximum, data=block_max, axis=1)
    nisa.tensor_tensor(dst=new_max, data1=new_max, data2=run_max, op=nl.maximum,
                       engine=nisa.engine.vector)
    neg_top = _col(ROW_TILE, nl.float32)
    nisa.tensor_scalar(dst=neg_top, data=new_max, op0=nl.multiply, operand0=-softmax_scale,
                       engine=nisa.engine.vector)
    alpha = _col(ROW_TILE, nl.float32)
    nisa.activation(dst=alpha, op=nl.exp, data=run_max, bias=neg_top, scale=softmax_scale)
    nisa.tensor_copy(dst=run_max, src=new_max, engine=nisa.engine.vector)
    block_sum = _col(ROW_TILE, nl.float32)
    nisa.activation(dst=scores, op=nl.exp, data=scores, bias=neg_top, scale=softmax_scale,
                    reduce_op=nl.add, reduce_res=block_sum,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    nisa.scalar_tensor_tensor(dst=run_sum, data=run_sum, op0=nl.multiply, operand0=alpha,
                              op1=nl.add, operand1=block_sum)

    # ---- p onto the key partitions, hi/lo, MM2 over the superblock ----------------------
    pv_ps = nl.ndarray((ROW_TILE, latent), dtype=nl.float32, buffer=nl.psum)
    for b in range(n_blocks):
        t0 = b * MOVING_MAX
        pt_ps = nl.ndarray((KEY_CHUNK, _TRANSPOSES_PER_BANK * ROW_TILE), dtype=nl.float32,
                           buffer=nl.psum)
        for ck in range(_TRANSPOSES_PER_BANK):
            k0 = t0 + ck * KEY_CHUNK
            nisa.nc_transpose(dst=pt_ps[:, ck * ROW_TILE:(ck + 1) * ROW_TILE],
                              data=scores[:, k0:k0 + KEY_CHUNK])
        # hi = bf16(p) on the scalar engine, lo = bf16(p - hi) on the Vector engine: the
        # difference is exact in fp32.
        p_hi = _sb((KEY_CHUNK, _TRANSPOSES_PER_BANK * ROW_TILE), kv_dtype)
        p_lo = _sb((KEY_CHUNK, _TRANSPOSES_PER_BANK * ROW_TILE), kv_dtype)
        nisa.tensor_copy(dst=p_hi, src=pt_ps, engine=nisa.engine.scalar)
        nisa.tensor_tensor(dst=p_lo, data1=pt_ps, data2=p_hi, op=nl.subtract,
                           engine=nisa.engine.vector)
        for ck in range(_TRANSPOSES_PER_BANK):
            c0 = ck * ROW_TILE
            chunk = t0 // KEY_CHUNK + ck
            nisa.nc_matmul(dst=pv_ps, stationary=p_hi[:, c0:c0 + ROW_TILE],
                           moving=rows[:, chunk, :], accumulate=(chunk > 0))
            nisa.nc_matmul(dst=pv_ps, stationary=p_lo[:, c0:c0 + ROW_TILE],
                           moving=rows[:, chunk, :], accumulate=True)
    nisa.scalar_tensor_tensor(dst=acc, data=acc, op0=nl.multiply, operand0=alpha, op1=nl.add,
                              operand1=pv_ps)


def _store_tile(out_hbm, acc, run_sum, r0):
    """``acc / max(run_sum, _TOTAL_FLOOR)`` to output rows ``r0 .. r0 + 127``."""
    latent = acc.shape[1]
    total = _col(ROW_TILE, nl.float32)
    nisa.tensor_scalar(dst=total, data=run_sum, op0=nl.maximum, operand0=_TOTAL_FLOOR,
                       engine=nisa.engine.vector)
    recip = _col(ROW_TILE, nl.float32)
    nisa.reciprocal(dst=recip, data=total)
    out_sb = _sb((ROW_TILE, latent), nl.float32)
    nisa.tensor_scalar(dst=out_sb, data=acc, op0=nl.multiply, operand0=recip,
                       engine=nisa.engine.vector)
    nisa.dma_copy(dst=out_hbm.ap(pattern=[[latent, ROW_TILE], [1, latent]], offset=r0 * latent),
                  src=out_sb)


def _tile_state(q_hbm, r0):
    """A query tile's state (the ``_ST_*`` positions) before its first superblock."""
    latent = q_hbm.shape[2]
    state = []
    state.append(_dw._query_columns(q_hbm, ROW_TILE, latent, r0 * latent, latent))
    run_max = _col(ROW_TILE, nl.float32)
    nisa.memset(dst=run_max, value=_SELECTED_MAX_FLOOR)
    state.append(run_max)
    run_sum = _col(ROW_TILE, nl.float32)
    nisa.memset(dst=run_sum, value=0.0)
    state.append(run_sum)
    acc = _sb((ROW_TILE, latent), nl.float32)
    nisa.memset(dst=acc, value=0.0)
    state.append(acc)
    state.append(r0)
    return state


def _block_shared_body(q_hbm, bank_hbm, table_hbm, union_hbm, member_hbm, softmax_scale,
                       page_size, block_queries, union_cap, out_hbm):
    """Every block of this program over every superblock of its union, :data:`_PAIR` at a time.

    The blocks are dealt round-robin over the programs; ``program_id`` is a trace-time int
    (the kernel is traced once per program), so each program keeps only its own blocks,
    and runs them in groups of :data:`_PAIR` (a last group may hold one). Per superblock a
    group emits its stages block by block: the gathers, then MM1 and the turns, then the
    softmax and MM2. A block's tiles share each superblock's gather and transposes; the
    last superblock may be shorter, by whole 512-position blocks. Each block folds its
    superblocks in the same order and by the same steps as when it runs alone.
    """
    seq = q_hbm.shape[0]
    n_blocks = seq // block_queries
    tiles = block_queries // ROW_TILE
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    bank_rows = _ms._lowp_bank_table(table_hbm, page_size)

    for first in range(prg, n_blocks, _PAIR * n_prgs):
        n_group = min(_PAIR, (n_blocks - first + n_prgs - 1) // n_prgs)
        states = []
        for gi in range(n_group):
            blk_states = []
            for ti in range(tiles):
                blk_states.append(_tile_state(q_hbm, (first + gi * n_prgs) * block_queries
                                              + ti * ROW_TILE))
            states.append(blk_states)
        for u0 in range(0, union_cap, SUPERBLOCK_ROWS):
            n_chunks = min(SUPERBLOCK_ROWS, union_cap - u0) // KEY_CHUNK
            rows_t = []
            for gi in range(n_group):
                offsets = _union_offsets(union_hbm, bank_rows, first + gi * n_prgs, u0, n_chunks,
                                         page_size)
                rows_t.append(_gather_rows(bank_hbm, offsets, n_chunks))
            parts = []
            rows = []
            for gi in range(n_group):
                blk_parts = []
                for ti in range(tiles):
                    blk_parts.append(_tile_scores(member_hbm, states[gi][ti][_ST_Q], rows_t[gi],
                                                  states[gi][ti][_ST_ROW], u0))
                parts.append(blk_parts)
                rows.append(_turn_rows(rows_t[gi]))
            for gi in range(n_group):
                for ti in range(tiles):
                    _tile_fold(parts[gi][ti], rows[gi], states[gi][ti], softmax_scale)
        for gi in range(n_group):
            for ti in range(tiles):
                _store_tile(out_hbm, states[gi][ti][_ST_ACC], states[gi][ti][_ST_SUM],
                            states[gi][ti][_ST_ROW])


@nki.jit
def mla_block_shared_kernel(q_hbm, bank_hbm, table_hbm, union_hbm, member_hbm, softmax_scale,
                            page_size, block_queries, union_cap, source_digest=0,
                            staging_digest=0):
    """Attention of each query over the union rows its mask selects, one gather per block.

    Args:
        q_hbm: ``[S, 1, L]`` bf16 or fp16, the absorbed queries; ``S`` a multiple of
            ``block_queries``, ``L`` a multiple of 128 at most :data:`LATENT_MAX`.
        bank_hbm: ``[slots, L]``, the whole latent bank, the dtype of ``q_hbm``. It already
            holds this step's rows (the sparse kernel's bank form).
        table_hbm: ``[pages, 1]`` int32, the window's pages in order (``-1`` pads read bank
            page 0, as the sparse kernel's do).
        union_hbm: ``[S / block_queries, union_cap]`` int32: per block, the window rows its
            queries select, ``-1`` past the last.
        member_hbm: ``[S, union_cap]`` uint8: 1 where query ``s`` selects position ``u`` of
            its block's union, 0 elsewhere (0 on every pad position).
        softmax_scale: python float applied to the scores before the softmax.
        page_size: python int, rows per page, a power of two.
        block_queries: python int, queries per block, a multiple of 128.
        union_cap: python int, union positions per block, a multiple of :data:`MOVING_MAX`.
        source_digest, staging_digest: python ints that only key the compiled-kernel cache
            on this file and on ``mla_sparse.py`` (the bank-table helpers).

    Returns:
        ``[S, 1, L]`` fp32 in shared HBM: ``softmax(scale * q . k) @ k`` over the rows the
        query selects; a query that selects no row returns zeros.

    Under a two-program launch the blocks are dealt round-robin between the programs, and
    each program gathers its own blocks' unions.
    """
    seq, heads, latent = q_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _block_shared_body(q_hbm, bank_hbm, table_hbm, union_hbm, member_hbm, softmax_scale,
                       page_size, block_queries, union_cap, out_hbm)
    return out_hbm


def _require_knob() -> None:
    if not envs.VLLM_NEURON_MLA_BLOCK_SHARED:
        raise MlaBlockSharedError(
            "the block-shared MLA kernel is off: set VLLM_NEURON_MLA_BLOCK_SHARED=1 to call it"
        )


def _require_call(q_lift: Tensor, c_kv: Tensor, union_rows: Tensor, member: Tensor,
                  softmax_scale: float, block_table_row: Tensor, page_size: int,
                  block_queries: int) -> None:
    _require_knob()
    if not softmax_scale > 0:
        raise MlaBlockSharedError(f"softmax_scale must be positive; got {softmax_scale}")
    if (q_lift.ndim != 3 or c_kv.ndim != 2 or union_rows.ndim != 2 or member.ndim != 2
            or block_table_row.ndim != 2):
        raise MlaBlockSharedError(
            f"q_lift must be [seq, 1, latent], c_kv [slots, latent], union_rows [blocks, cap], "
            f"member [seq, cap] and block_table_row [pages, 1]; got {tuple(q_lift.shape)}, "
            f"{tuple(c_kv.shape)}, {tuple(union_rows.shape)}, {tuple(member.shape)} and "
            f"{tuple(block_table_row.shape)}"
        )
    seq, heads, latent = (int(d) for d in q_lift.shape)
    if block_queries < ROW_TILE or block_queries % ROW_TILE != 0:
        raise MlaBlockSharedError(
            f"a block is whole {ROW_TILE}-query tiles; got block_queries {block_queries}"
        )
    if heads != 1 or seq < block_queries or seq % block_queries != 0:
        raise MlaBlockSharedError(
            f"queries come in whole {block_queries}-query blocks of one head; got seq {seq}, "
            f"heads {heads}"
        )
    if latent % LATENT_TILE != 0 or latent > LATENT_MAX or int(c_kv.shape[1]) != latent:
        raise MlaBlockSharedError(
            f"the latent must be a multiple of {LATENT_TILE} up to {LATENT_MAX}, the same in "
            f"q_lift and c_kv; got {latent} and {int(c_kv.shape[1])}"
        )
    if q_lift.dtype not in (torch.bfloat16, torch.float16) or c_kv.dtype != q_lift.dtype:
        raise MlaBlockSharedError(
            f"q_lift and c_kv must share a 2-byte float dtype; got {q_lift.dtype} and {c_kv.dtype}"
        )
    cap = int(union_rows.shape[1])
    if (union_rows.dtype != torch.int32 or int(union_rows.shape[0]) != seq // block_queries
            or cap < MOVING_MAX or cap % MOVING_MAX != 0):
        raise MlaBlockSharedError(
            f"union_rows must be [seq / block_queries, cap] = [{seq // block_queries}, cap] int32 "
            f"with cap a multiple of {MOVING_MAX}; got {tuple(union_rows.shape)} {union_rows.dtype}"
        )
    if member.dtype != torch.uint8 or tuple(member.shape) != (seq, cap):
        raise MlaBlockSharedError(
            f"member must be [seq, cap] = [{seq}, {cap}] uint8; got {tuple(member.shape)} "
            f"{member.dtype}"
        )
    page = int(page_size)
    if page < 1 or page & (page - 1) != 0 or int(c_kv.shape[0]) % page != 0:
        raise MlaBlockSharedError(
            f"the page size must be a power of two that divides the bank's {int(c_kv.shape[0])} "
            f"rows; got {page_size}"
        )


def mla_block_shared_attention(q_lift: Tensor, c_kv: Tensor, union_rows: Tensor,
                               member: Tensor, softmax_scale: float, block_table_row: Tensor,
                               page_size: int = KEY_CHUNK,
                               block_queries: int = ROW_TILE) -> Tensor:
    """Sparse MLA prefill attention with one shared gather per block of queries.

    The operands are ``mla_sparse.mla_sparse_attention``'s paged bank-form call, with the
    selection given per block as its union and a row mask over it instead of index rows.

    Args:
        q_lift: ``[S, 1, L]`` bf16/fp16, the absorbed query; ``S`` a multiple of
            ``block_queries``.
        c_kv: ``[slots, L]`` the whole latent bank, same dtype, already holding this
            step's rows.
        union_rows: ``[S / block_queries, cap]`` int32, each block's selected window rows,
            ``-1`` pads; ``cap`` a multiple of :data:`MOVING_MAX`.
        member: ``[S, cap]`` uint8, 1 on each union position the query attends.
        softmax_scale: positive.
        block_table_row: ``[pages, 1]`` int32, the window's pages in order, ``-1`` pads.
        page_size: rows per page, a power of two.
        block_queries: queries per block, a multiple of 128.

    Returns:
        ``[S, 1, L]`` fp32, the sparse kernel's answer up to the order and rounding of its
        fp32 sums (a value-equal class, not bit-equal). A block with a cut union gets rows
        from what fits; ``mla_block_shared_exact`` rewrites them.

    Raises:
        MlaBlockSharedError: while ``VLLM_NEURON_MLA_BLOCK_SHARED`` is unset, or for a
            malformed call or a geometry the kernel does not serve.
    """
    _require_call(q_lift, c_kv, union_rows, member, softmax_scale, block_table_row, page_size,
                  block_queries)
    call = wrap_nki(mla_block_shared_kernel)
    blocks = int(q_lift.shape[0]) // block_queries
    # Two programs for an even block count only, as the exact kernel that follows launches.
    if lnc_pair() and blocks >= 2 and blocks % 2 == 0:
        call = call[2]
    return call(
        q_lift.contiguous(),
        c_kv.contiguous(),
        block_table_row.contiguous().to(torch.int32),
        union_rows.contiguous(),
        member.contiguous(),
        float(softmax_scale),
        int(page_size),
        int(block_queries),
        int(union_rows.shape[1]),
        SOURCE_DIGEST,
        int(_ms.SOURCE_DIGEST),
    )


# ---- the union producer -------------------------------------------------------------------

#: GpSimd cores of ``nonzero_with_count``: core ``k`` reads and writes partition
#: ``_GROUP_STRIDE * k`` only, so a block's pools are listed in ``_GROUPS`` segments.
_GROUPS = 8
_GROUP_STRIDE = ROW_TILE // _GROUPS
#: Bits per keep byte: the bit planes of the keep table.
_PLANES = 8
#: The unpacked keep byte of a set bit: bit 3, which read as fp8 e4m3 is 2^-6 (its
#: smallest normal value), so the count matmuls read the uint8 keep tile as fp8 with no
#: cast; a block's count, at most 128 x 2^-6, is exact in fp32.
_KEEP_SET_BIT = 3
#: The most indices of one keep-bit gather. A gather's cost is set by the keep bits it
#: reads, not by its index count (the compiler's schedule: about 25 us per 32,768 pools
#: whether it gathers 32 bits or 2,048), so the gathers are as wide as they compile.
_GATHER_MAX = 8192
#: Pools one ``nc_n_gather`` reads from: its access pattern counts elements in 16 bits,
#: and a source of more than one row fails the GpSimd buffer-load check.
_GATHER_SPAN = 32768
#: ``nonzero_with_count``'s pad: far enough below every segment's first pool that pad plus
#: that pool stays negative, and exact in fp32.
_NO_POOL = -(1 << 24)

# Positions in the list :func:`_union_constants` returns.
_UC_LEAD = 0       # [128, groups, 128] fp8: (m == 16 k), the count matmuls' stationary
_UC_UPPER = 1      # [128, 128] fp32: (m > p), the exclusive prefix's moving operand
_UC_POOL_AT = 2    # [128, 512] fp32: the pool of count element (16 k, n), k seg + n
_UC_FILL = 3       # [128, list width / 128] int32 -1: the pool list's first value
_UC_PAGE_ROW = 4   # [128, 1] int32: page position j on partition j
_UC_PAGE_POOL = 5  # [128, page pools] int32: page pool jj
_UC_OPEN = 6       # [128, page pools] uint8: page pool jj is q's open pool or past it
_UC_CAUSAL = 7     # [128, page pools, pool] uint8: page position j <= q


def _list_width(list_cap, seg):
    """Columns of a block's HBM pool list: the cap plus one segment of spill, in whole
    partitions (a segment past the cap writes into the spill)."""
    return (list_cap + seg + ROW_TILE - 1) // ROW_TILE * ROW_TILE


def _union_constants(pools, pool_size, list_width):
    """The producer's per-program constants, by the ``_UC_*`` positions."""
    seg = pools // _GROUPS
    page_pools = ROW_TILE // pool_size
    out = []
    lead_i = _sb((ROW_TILE, _GROUPS, ROW_TILE), nl.int32)
    nisa.iota(dst=lead_i, pattern=[[-_GROUP_STRIDE, _GROUPS], [1, ROW_TILE]], offset=0,
              channel_multiplier=0)
    lead = _sb((ROW_TILE, _GROUPS, ROW_TILE), nl.float8_e4m3)
    nisa.tensor_scalar(dst=lead, data=lead_i, op0=nl.equal, operand0=0, engine=nisa.engine.vector)
    out.append(lead)
    upper_i = _sb((ROW_TILE, ROW_TILE), nl.int32)
    nisa.iota(dst=upper_i, pattern=[[1, ROW_TILE]], offset=0, channel_multiplier=-1)
    upper = _sb((ROW_TILE, ROW_TILE), nl.float32)
    nisa.tensor_scalar(dst=upper, data=upper_i, op0=nl.greater, operand0=0,
                       engine=nisa.engine.vector)
    out.append(upper)
    pool_at = _sb((ROW_TILE, MOVING_MAX), nl.float32)
    nisa.iota(dst=pool_at, pattern=[[1, MOVING_MAX]], offset=0,
              channel_multiplier=seg // _GROUP_STRIDE)
    out.append(pool_at)
    fill = _sb((ROW_TILE, list_width // ROW_TILE), nl.int32)
    nisa.memset(dst=fill, value=-1)
    out.append(fill)
    page_row = _col(ROW_TILE, nl.int32)
    nisa.iota(dst=page_row, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    out.append(page_row)
    page_pool = _sb((ROW_TILE, page_pools), nl.int32)
    nisa.iota(dst=page_pool, pattern=[[1, page_pools]], offset=0, channel_multiplier=0)
    out.append(page_pool)
    # pool jj is open or past for query q when its last row is after q: pool (jj + 1) - 1 - q > 0
    last_past = _sb((ROW_TILE, page_pools), nl.int32)
    nisa.iota(dst=last_past, pattern=[[pool_size, page_pools]], offset=pool_size - 1,
              channel_multiplier=-1)
    open_pool = _sb((ROW_TILE, page_pools), nl.uint8)
    nisa.tensor_scalar(dst=open_pool, data=last_past, op0=nl.greater, operand0=0,
                       engine=nisa.engine.vector)
    out.append(open_pool)
    back = _sb((ROW_TILE, page_pools, pool_size), nl.int32)
    nisa.iota(dst=back, pattern=[[-pool_size, page_pools], [-1, pool_size]], offset=0,
              channel_multiplier=1)
    causal = _sb((ROW_TILE, page_pools, pool_size), nl.uint8)
    nisa.tensor_scalar(dst=causal, data=back, op0=nl.greater_equal, operand0=0,
                       engine=nisa.engine.vector)
    out.append(causal)
    return out


def _unpack_keep(keep_hbm, r0, keep):
    """``keep[q, b w + j]`` = bit ``b`` of ``keep_bits[r0 + q, j]`` moved to bit
    :data:`_KEEP_SET_BIT` (uint8 8 or 0) on the query partitions, ``w`` the table's bytes
    per row: one load, then one integer Vector instruction per bit plane (a shift, then a
    mask), each writing one contiguous run of pools."""
    width = keep_hbm.shape[1]
    packed = _sb((ROW_TILE, width), nl.uint8)
    nisa.dma_copy(dst=packed, src=keep_hbm.ap(pattern=[[width, ROW_TILE], [1, width]],
                                              offset=r0 * width),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    for b in range(_PLANES):
        shift = nl.right_shift if b >= _KEEP_SET_BIT else nl.left_shift
        nisa.tensor_scalar(dst=keep[:, b * width:(b + 1) * width], data=packed, op0=shift,
                           operand0=abs(b - _KEEP_SET_BIT), op1=nl.bitwise_and,
                           operand1=1 << _KEEP_SET_BIT, engine=nisa.engine.vector)


def _shared_bits(keep, consts, pb):
    """``[128, seg]`` fp32: on partition ``16 k``, positive where some query keeps pool
    ``k seg + n`` and that pool is below the page (``pb``, every partition), 0.0 elsewhere.

    ``keep`` is the fp8 view of the unpacked keep tile (2^-6 or 0.0). Per 512 pools, one
    matmul per segment with a one-column stationary sums the keep values over the queries
    onto the segment's partition (the count times 2^-6, exact); the minimum with the
    below-the-page bit (0.0 or 1.0) keeps it positive only below the page.
    """
    pools = keep.shape[1]
    seg = pools // _GROUPS
    shared = _sb((ROW_TILE, seg), nl.float32)
    for n0 in range(0, seg, MOVING_MAX):
        nw = min(MOVING_MAX, seg - n0)
        count_ps = nl.ndarray((ROW_TILE, MOVING_MAX), dtype=nl.float32, buffer=nl.psum)
        for k in range(_GROUPS):
            nisa.nc_matmul(dst=count_ps[:, 0:nw], stationary=consts[_UC_LEAD][:, k, :],
                           moving=keep[:, k * seg + n0:k * seg + n0 + nw], accumulate=(k > 0))
        below = _sb((ROW_TILE, MOVING_MAX), nl.float32)
        nisa.tensor_scalar(dst=below[:, 0:nw], data=consts[_UC_POOL_AT][:, 0:nw], op0=nl.add,
                           operand0=float(n0), op1=nl.less, operand1=pb,
                           engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=shared[:, n0:n0 + nw], data1=count_ps[:, 0:nw],
                           data2=below[:, 0:nw], op=nl.minimum, engine=nisa.engine.vector)
    return shared


def _compact(shared, consts, list_cap, list_hbm):
    """The pools ``shared`` marks to ``list_hbm`` ``[1, width]`` int32, ascending, ``-1`` after.

    ``nonzero_with_count`` lists each segment's marked positions and their count; an
    exclusive prefix of the eight counts (one fp32 matmul with a strictly upper triangular
    moving operand) places each segment; segment ``k`` (its positions plus its first pool,
    ``_NO_POOL`` pads clamped to ``-1``) is written at ``min(prefix_k, list_cap)`` over the
    ``-1`` tail of the one before, in order. Positions below ``list_cap`` are exact whatever
    the total: a list past the cap spills into the columns after it.

    Returns the ``[1, 128]`` int32 prefix: column 127 holds the number of marked pools.
    """
    seg = shared.shape[1]
    width = list_hbm.shape[1]
    lists = _sb((ROW_TILE, seg + 1), nl.int32)
    # the prefix reads every partition's count; nonzero_with_count writes those of 16 k only
    nisa.memset(dst=lists[:, seg:seg + 1], value=0)
    nisa.nonzero_with_count(dst=lists, src=shared, index_offset=0, padding_val=_NO_POOL)
    first = _col(ROW_TILE, nl.float32)
    nisa.iota(dst=first, pattern=[[0, 1]], offset=0, channel_multiplier=seg // _GROUP_STRIDE)
    cnt = _col(ROW_TILE, nl.float32)
    nisa.tensor_copy(dst=cnt, src=lists[:, seg:seg + 1], engine=nisa.engine.vector)
    pre_ps = nl.ndarray((1, ROW_TILE), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=pre_ps, stationary=cnt, moving=consts[_UC_UPPER])
    prefix = _sb((1, ROW_TILE), nl.int32)
    nisa.tensor_copy(dst=prefix, src=pre_ps, engine=nisa.engine.vector)
    start = _sb((1, ROW_TILE), nl.int32)
    nisa.tensor_scalar(dst=start, data=prefix, op0=nl.minimum, operand0=list_cap,
                       engine=nisa.engine.vector)
    nisa.tensor_scalar(dst=lists[:, 0:seg], data=lists[:, 0:seg], op0=nl.add, operand0=first,
                       op1=nl.maximum, operand1=-1, engine=nisa.engine.vector)
    nisa.dma_copy(dst=list_hbm.ap(pattern=[[width // ROW_TILE, ROW_TILE], [1, width // ROW_TILE]],
                                  offset=0),
                  src=consts[_UC_FILL])
    for k in range(_GROUPS):
        p = k * _GROUP_STRIDE
        nisa.dma_copy(dst=list_hbm.ap(pattern=[[width, 1], [1, seg]], offset=0,
                                      scalar_offset=start[0:1, p:p + 1], indirect_dim=1),
                      src=lists[p:p + 1, 0:seg])
    return prefix


def _union_block(keep_hbm, consts, start_f, blk, union_cap, pool_size, outs):
    """Block ``blk``'s union rows, member bits and totals into ``outs``."""
    pools = keep_hbm.shape[1] * _PLANES
    seg = pools // _GROUPS
    page_pools = ROW_TILE // pool_size
    list_cap = union_cap // pool_size - page_pools
    per_part = list_cap // page_pools
    r0 = blk * ROW_TILE
    union_hbm = outs[0]
    member_hbm = outs[1]
    totals_hbm = outs[2]

    # ---- the block's first row and first page pool on every partition (exact in fp32) ----
    b0 = _col(ROW_TILE, nl.float32)
    nisa.tensor_scalar(dst=b0, data=start_f, op0=nl.add, operand0=float(r0),
                       engine=nisa.engine.vector)
    pb = _col(ROW_TILE, nl.float32)
    nisa.tensor_scalar(dst=pb, data=b0, op0=nl.multiply, operand0=1.0 / pool_size,
                       engine=nisa.engine.vector)

    # ---- keep bits, the shared pools, their list ------------------------------------------
    # Per block the gathers read the keep bits at the list's positions and then at the
    # page's pools: `ext` indices in `n_chunks` chunks of at most `width`, one gather per
    # chunk and span of _GATHER_SPAN pools. Their destination is columns of the keep bits'
    # own tile, so no placement can overlap a gather's source.
    ext = list_cap + page_pools
    n_chunks = (ext + _GATHER_MAX - 1) // _GATHER_MAX
    width = (ext + n_chunks - 1) // n_chunks
    keep_got = _sb((ROW_TILE, pools + width), nl.uint8)
    keep = keep_got[:, 0:pools]
    got = keep_got[:, pools:pools + width]
    _unpack_keep(keep_hbm, r0, keep)
    shared = _shared_bits(keep_got.view(nl.float8_e4m3)[:, 0:pools], consts, pb)
    list_hbm = nl.ndarray((1, _list_width(list_cap, seg)), dtype=nl.int32, buffer=nl.private_hbm)
    prefix = _compact(shared, consts, list_cap, list_hbm)

    # ---- union rows: the page, then pool_size rows per listed pool, -1 after ---------------
    page_rows = _col(ROW_TILE, nl.int32)
    nisa.tensor_scalar(dst=page_rows, data=consts[_UC_PAGE_ROW], op0=nl.add, operand0=b0,
                       engine=nisa.engine.vector)
    nisa.dma_copy(dst=union_hbm.ap(pattern=[[1, ROW_TILE], [1, 1]], offset=blk * union_cap),
                  src=page_rows)
    codes = _sb((page_pools, per_part), nl.int32)
    nisa.dma_copy(dst=codes, src=list_hbm.ap(pattern=[[per_part, page_pools], [1, per_part]],
                                             offset=0))
    rows = _sb((page_pools, per_part, pool_size), nl.int32)
    for r in range(pool_size):
        nisa.tensor_scalar(dst=rows[:, :, r], data=codes, op0=nl.multiply, operand0=pool_size,
                           op1=nl.add, operand1=r, engine=nisa.engine.vector)
    # a -1 code gives pool_size * -1 + r < 0
    nisa.tensor_scalar(dst=rows, data=rows, op0=nl.maximum, operand0=-1, engine=nisa.engine.vector)
    nisa.dma_copy(dst=union_hbm.ap(pattern=[[per_part * pool_size, page_pools],
                                            [1, per_part * pool_size]],
                                   offset=blk * union_cap + ROW_TILE),
                  src=rows)

    # ---- member bits: the list's pools, then the page's ------------------------------------
    page_keep = _sb((ROW_TILE, page_pools), nl.uint8)
    for g in range(n_chunks):
        g0 = g * width
        gw = min(width, ext - g0)
        lw = max(0, min(gw, list_cap - g0))
        lst = _sb((ROW_TILE, width), nl.int32)
        if lw > 0:
            nisa.dma_copy(dst=lst[:, 0:lw],
                          src=list_hbm.ap(pattern=[[0, ROW_TILE], [1, lw]], offset=g0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        if lw < gw:
            j0 = g0 + lw - list_cap
            nisa.tensor_scalar(dst=lst[:, lw:gw], data=consts[_UC_PAGE_POOL][:, j0:j0 + gw - lw],
                               op0=nl.add, operand0=pb, engine=nisa.engine.vector)
        # bits[q, i] = keep[q, lst[i]]: per span the index is clamped into it and the bit
        # counts only where the clamp left it unchanged (a -1 pad counts in no span); the
        # minimum of the gathered byte (0 or 8) and that inside bit (0 or 1) is 0 or 1
        bits = _sb((ROW_TILE, width), nl.uint8)
        for s0 in range(0, pools, _GATHER_SPAN):
            sw = min(_GATHER_SPAN, pools - s0)
            rel = lst
            if s0 > 0:
                rel = _sb((ROW_TILE, width), nl.int32)
                nisa.tensor_scalar(dst=rel[:, 0:gw], data=lst[:, 0:gw], op0=nl.subtract,
                                   operand0=s0, engine=nisa.engine.vector)
            idx = _sb((ROW_TILE, width), nl.uint32)
            nisa.tensor_scalar(dst=idx[:, 0:gw], data=rel[:, 0:gw], op0=nl.maximum, operand0=0,
                               op1=nl.minimum, operand1=sw - 1, engine=nisa.engine.vector)
            inside = _sb((ROW_TILE, width), nl.uint8)
            nisa.tensor_tensor(dst=inside[:, 0:gw], data1=rel[:, 0:gw], data2=idx[:, 0:gw],
                               op=nl.equal, engine=nisa.engine.vector)
            nisa.nc_n_gather(dst=got[:, 0:gw], data=keep[:, s0:s0 + sw], indices=idx[:, 0:gw])
            if s0 == 0:
                nisa.tensor_tensor(dst=bits[:, 0:gw], data1=got[:, 0:gw], data2=inside[:, 0:gw],
                                   op=nl.minimum, engine=nisa.engine.vector)
            else:
                nisa.tensor_tensor(dst=inside[:, 0:gw], data1=got[:, 0:gw], data2=inside[:, 0:gw],
                                   op=nl.minimum, engine=nisa.engine.vector)
                nisa.tensor_tensor(dst=bits[:, 0:gw], data1=bits[:, 0:gw], data2=inside[:, 0:gw],
                                   op=nl.maximum, engine=nisa.engine.vector)
        if lw > 0:
            mem_rows = _sb((ROW_TILE, width, pool_size), nl.uint8)
            nisa.tensor_copy(dst=mem_rows[:, 0:lw, :],
                             src=bits[:, 0:lw].expand_dim(2).broadcast(2, pool_size),
                             engine=nisa.engine.scalar)
            nisa.dma_copy(dst=member_hbm.ap(pattern=[[union_cap, ROW_TILE], [1, lw * pool_size]],
                                            offset=r0 * union_cap + ROW_TILE + g0 * pool_size),
                          src=mem_rows[:, 0:lw, :], dge_mode=nisa.dge_mode.hwdge,
                          engine=nisa.engine.scalar)
        if lw < gw:
            j0 = g0 + lw - list_cap
            nisa.tensor_copy(dst=page_keep[:, j0:j0 + gw - lw], src=bits[:, lw:gw],
                             engine=nisa.engine.vector)

    # ---- the page: member bits ---------------------------------------------------------------
    # open pool or kept pool; a kept bit past the open pool is under the causal bound
    attend = _sb((ROW_TILE, page_pools), nl.uint8)
    nisa.tensor_tensor(dst=attend, data1=page_keep, data2=consts[_UC_OPEN], op=nl.maximum,
                       engine=nisa.engine.vector)
    page_member = _sb((ROW_TILE, page_pools, pool_size), nl.uint8)
    nisa.tensor_tensor(dst=page_member, data1=consts[_UC_CAUSAL],
                       data2=attend.expand_dim(2).broadcast(2, pool_size), op=nl.minimum,
                       engine=nisa.engine.vector)
    nisa.dma_copy(dst=member_hbm.ap(pattern=[[union_cap, ROW_TILE], [1, ROW_TILE]],
                                    offset=r0 * union_cap),
                  src=page_member)

    # ---- totals: the block's shared pools and its overflow flag -----------------------------
    totals = _sb((1, 2), nl.int32)
    nisa.tensor_copy(dst=totals[:, 0:1], src=prefix[:, ROW_TILE - 1:ROW_TILE],
                     engine=nisa.engine.vector)
    nisa.tensor_scalar(dst=totals[:, 1:2], data=prefix[:, ROW_TILE - 1:ROW_TILE], op0=nl.greater,
                       operand0=list_cap, engine=nisa.engine.vector)
    nisa.dma_copy(dst=totals_hbm.ap(pattern=[[2, 1], [1, 2]], offset=2 * blk), src=totals)


@nki.jit
def mla_block_shared_union_kernel(keep_hbm, start_hbm, union_cap, pool_size, source_digest=0):
    """Each 128-query block's union rows and member bits from the chunk's keep bits.

    Args:
        keep_hbm: ``[S, P / 8]`` uint8, the keep bits in the module docstring's layout;
            ``S`` a multiple of 128, ``P`` a multiple of 128.
        start_hbm: ``[1, 1]`` int32, the window row of query 0, a multiple of 128.
        union_cap: python int, union positions per block, a multiple of
            :data:`MOVING_MAX`.
        pool_size: python int, rows per pool, a power of two that divides 128.
        source_digest: python int that only keys the compiled-kernel cache on this file.

    Returns:
        ``union [S / 128, union_cap]`` int32, ``member [S, union_cap]`` uint8 and
        ``totals [S / 128, 2]`` int32, in shared HBM (the module docstring states each).

    Under a two-program launch the blocks are dealt round-robin between the programs.
    """
    seqs = keep_hbm.shape[0]
    pools = keep_hbm.shape[1] * _PLANES
    n_blocks = seqs // ROW_TILE
    page_pools = ROW_TILE // pool_size
    outs = []
    outs.append(nl.ndarray((n_blocks, union_cap), dtype=nl.int32, buffer=nl.shared_hbm))
    outs.append(nl.ndarray((seqs, union_cap), dtype=nl.uint8, buffer=nl.shared_hbm))
    outs.append(nl.ndarray((n_blocks, 2), dtype=nl.int32, buffer=nl.shared_hbm))
    list_width = _list_width(union_cap // pool_size - page_pools, pools // _GROUPS)
    consts = _union_constants(pools, pool_size, list_width)
    start_i = _col(ROW_TILE, nl.int32)
    nisa.dma_copy(dst=start_i, src=start_hbm.ap(pattern=[[0, ROW_TILE], [1, 1]], offset=0))
    start_f = _col(ROW_TILE, nl.float32)
    nisa.tensor_copy(dst=start_f, src=start_i, engine=nisa.engine.vector)
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    for blk in range(prg, n_blocks, n_prgs):
        _union_block(keep_hbm, consts, start_f, blk, union_cap, pool_size, outs)
    return outs[0], outs[1], outs[2]


def _require_union_call(keep_bits: Tensor, query_start: Tensor, union_cap: int,
                        pool_size: int) -> None:
    _require_knob()
    if keep_bits.ndim != 2 or keep_bits.dtype != torch.uint8:
        raise MlaBlockSharedError(
            f"keep_bits must be [seq, pools / {_PLANES}] uint8; got {tuple(keep_bits.shape)} "
            f"{keep_bits.dtype}"
        )
    seq, width = (int(d) for d in keep_bits.shape)
    pools = width * _PLANES
    if seq < ROW_TILE or seq % ROW_TILE != 0 or pools < ROW_TILE or pools % ROW_TILE != 0:
        raise MlaBlockSharedError(
            f"keep_bits must be whole {ROW_TILE}-query blocks over a multiple of {ROW_TILE} pools; "
            f"got [{seq}, {width}] bytes ({pools} pools)"
        )
    if tuple(query_start.shape) != (1, 1) or query_start.dtype != torch.int32:
        raise MlaBlockSharedError(
            f"query_start must be [1, 1] int32; got {tuple(query_start.shape)} {query_start.dtype}"
        )
    pool = int(pool_size)
    if pool < 1 or pool & (pool - 1) != 0 or ROW_TILE % pool != 0:
        raise MlaBlockSharedError(
            f"pool_size must be a power of two that divides {ROW_TILE}; got {pool_size}"
        )
    if int(union_cap) <= ROW_TILE or int(union_cap) % MOVING_MAX != 0:
        raise MlaBlockSharedError(
            f"union_cap must be a multiple of {MOVING_MAX} over the block's {ROW_TILE}-row page; "
            f"got {union_cap}"
        )


def mla_block_shared_union(keep_bits: Tensor, query_start: Tensor, union_cap: int,
                           pool_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """Each 128-query block's union and member bits for :func:`mla_block_shared_attention`.

    Args:
        keep_bits: ``[S, P / 8]`` uint8, the chunk's keep bits (the module docstring's
            layout) for ``S`` queries (a multiple of 128) over ``P`` pools (a multiple of
            128).
        query_start: ``[1, 1]`` int32, the window row of query 0, a multiple of 128.
        union_cap: union positions per block, a multiple of :data:`MOVING_MAX`.
        pool_size: rows per pool, a power of two that divides 128.

    Returns:
        ``(union_rows, member, totals)``: ``[S / 128, union_cap]`` int32, ``[S, union_cap]``
        uint8 and ``[S / 128, 2]`` int32 (the module docstring states each). A block whose
        overflow flag (``totals[b, 1]``) is 1 is not served by its union.

    Raises:
        MlaBlockSharedError: while ``VLLM_NEURON_MLA_BLOCK_SHARED`` is unset, or for a
            malformed call.
    """
    _require_union_call(keep_bits, query_start, union_cap, pool_size)
    call = wrap_nki(mla_block_shared_union_kernel)
    if lnc_pair() and int(keep_bits.shape[0]) >= 2 * ROW_TILE:
        call = call[2]
    return call(
        keep_bits.contiguous(),
        query_start.contiguous(),
        int(union_cap),
        int(pool_size),
        SOURCE_DIGEST,
    )


# ---- call-site questions ------------------------------------------------------------------

#: Window pages -> the static union cap in rows of that served window bucket. Sized from
#: the recorded top-k selections of the served prompts: each cap holds the largest block
#: union seen at its bucket, in whole superblocks (the 1,032-page bucket takes the
#: 1,024-page cap). A block past it raises its overflow flag.
_UNION_CAPS = {264: 20480, 520: 30720, 1024: 40960, 1032: 40960, 2048: 55296}


def mla_block_shared_union_cap(window_pages: int) -> int | None:
    """The static union cap, in rows, of a window of ``window_pages`` pages, or None for a
    window with no cap (not served)."""
    return _UNION_CAPS.get(int(window_pages))


def mla_block_shared_serves(seq: int, heads: int, latent: int, window_pages: int,
                            page_size: int, pool_size: int, q_dtype: torch.dtype,
                            kv_dtype: torch.dtype) -> bool:
    """Whether the two entries serve a NoPE paged call of these trace-time values: the
    checks both entries make on them, and a union cap for the window. Reads no knob."""
    page = int(page_size)
    pool = int(pool_size)
    return (int(seq) >= ROW_TILE and int(seq) % ROW_TILE == 0 and int(heads) == 1
            and int(latent) % LATENT_TILE == 0 and 0 < int(latent) <= LATENT_MAX
            and page >= 1 and page & (page - 1) == 0
            and pool >= 1 and pool & (pool - 1) == 0 and ROW_TILE % pool == 0
            and q_dtype in (torch.bfloat16, torch.float16) and kv_dtype == q_dtype
            and mla_block_shared_union_cap(window_pages) is not None)
