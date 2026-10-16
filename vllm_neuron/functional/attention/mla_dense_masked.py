# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA prefill attention as one dense pass over the window, the selection as a mask.

The sparse kernel (``mla_sparse.py``) gathers each query's selected rows one query at a
time, and those gathers are serial on the software DMA queue. This kernel reads every
window row once per program instead, scores 128 queries per matmul against it, and
keeps each query's selection as a per-row predicate ahead of the softmax::

    scores[q, w] = q_lift[q] . window[w]      where w % R < count[q, w // R], else _UNSELECTED_SCORE
    out[q]       = softmax(scale * scores[q]) @ window

The window is read through the block table straight from the latent bank, a superblock
of pages at a time: its rows on the key partitions (MM2's moving operand) and, by PE
transposes, with the latent on the partitions (MM1's). Each program owns a set of
128-query tiles and keeps, per tile, a running row maximum, a running row sum and an
fp32 accumulator over the superblocks (an online softmax). A superblock's scores are
taken out of PSUM by the predicate (an unselected row reads _UNSELECTED_SCORE, so its
probability is exactly 0.0), exponentiated against the new running maximum, put on
the key partitions by fp32 PE transposes, split bf16 hi/lo and contracted with the
rows. The accumulator is rescaled by ``exp(scale * (old max - new max))`` before the
superblock's sum is added, and divided by the row sum once, at the end.

Arithmetic against the sparse low-precision body: the same MM1 products into fp32 PSUM
and the same hi/lo split against the stored cache, but the probabilities are split
before they are normalised, the sums run over the window in row order, and the
maximum is a running one. The answer is the sparse kernel's up to the order and the
rounding of these fp32 steps: value-equal, not bit-equal.

Non-finite rows. MM1 keeps a non-finite row, so a query that selects it gets a
non-finite score, as on the gather path; the predicate drops it for every other query.
MM2 contracts every row of the window whatever its probability, and ``0 * NaN`` is NaN,
so a non-finite row would reach every query of the tile. Each superblock's MM2 operand
therefore has its non-finite elements set to 0.0 first.

The selection is the indexer's keep table: ``keep_bits`` ``[S, P / 8]`` uint8, one bit
per query and ``R``-row pool of the window (``R`` the indexer's pool size, ``P`` the
window's pools), bit-plane major: pool ``p`` is bit ``p // (P / 8)`` of byte
``p % (P / 8)``, bit 0 the least significant, so each bit plane of a row's bytes holds a
contiguous run of pools. A set bit is a complete pool the query's top-k chose. One bit
per pool is the exact selection because the indexer selects whole pools; the open pool
(the incomplete one holding the query's own row) is never set, because
``dsa_index_expand`` appends its rows up to the query's position after the selected
pools. So with ``pos`` the query's window row (``query_start`` plus its row in the
chunk), the query reads ``count[q, p]`` leading rows of pool ``p``: ``R`` for a set bit,
``(pos + 1) % R`` for its open pool ``(pos + 1) // R``, and 0 elsewhere.

Each 128-query tile loads its rows of the table once. Per superblock, one integer Vector
instruction per bit plane the superblock's pools fall in moves that plane's bit to the
value ``R`` (the counts of the kept pools), three more add the open pool's count, and one
Vector compare reads each count through a stride-0 view against its ``R`` rows' ranks in
the pool.

The entry, :func:`mla_dense_masked_attention`, takes the keep table and the window row of
the chunk's first query, and it refuses a call unless ``VLLM_NEURON_MLA_DENSE_MASKED``
(``vllm_neuron/envs.py``) is set. The sparse attention seam
(``mla_sparse.mla_sparse_attention``) calls it for a paged NoPE call whose caller passes
the keep table and whose bank already holds the step's rows, wherever
:func:`dense_masked_serves` says so; every other call keeps the gather.
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

#: Partition extent: query rows are tiled to it, window rows are chunked to it (one
#: page per chunk), the latent is tiled to it.
ROW_TILE = 128
KEY_CHUNK = 128
LATENT_TILE = 128
#: Moving free-axis extent of one matmul: MM1 scores 512 window rows per instruction.
MOVING_MAX = 512
#: The widest latent one fp32 PSUM bank holds for the MM2 result row (2 KB).
LATENT_MAX = 512
#: PE transposes of 128 fp32 columns that fill one PSUM bank.
_TRANSPOSES_PER_BANK = MOVING_MAX // ROW_TILE
#: PSUM bytes per partition of one bank.
_PSUM_BANK_BYTES = 2048
#: Pages per superblock: the window rows one pass of MM1, the softmax and MM2 covers
#: before the running statistics are folded. Per partition a superblock holds its rows
#: twice (keys on partitions and transposed: 2 x 16 x 1 KiB at latent 512 in bf16) and
#: its guard mask (8 KiB), and each query tile adds its pool counts and open-pool counts
#: (0.5 KiB each at 4-row pools), predicate (2 KiB), scores (8 KiB) and one bank of hi/lo
#: halves (2 KiB). Each program also holds, per query tile, its rows of the keep table
#: (``P / 8`` bytes: 8 KiB at 2,048 pages) and its accumulators (2 KiB), and once its pool
#: ranks and pool numbers (2 KiB each): about 100 KiB at 2,048 pages with four tiles per
#: program, below the 224 KiB of a partition, which leaves the compiler room to
#: double-buffer the superblocks.
SUPERBLOCK_PAGES = 16
#: The score an unselected row reads: a power of two so far below any score that
#: ``exp(scale * (_UNSELECTED_SCORE - max))`` is exactly 0.0.
_UNSELECTED_SCORE = -(2.0 ** 30)
#: The running maximum starts here: a query that has selected no row yet
#: exponentiates every row to 0.0, and one that selects none returns zeros.
_SELECTED_MAX_FLOOR = _UNSELECTED_SCORE / 2
#: The row-sum floor of the final division.
_TOTAL_FLOOR = 1.0e-30
#: Bit planes of a keep-table byte.
_KEEP_PLANES = 8

#: This file's digest as a trace-time int: the compiled-kernel cache keys on the entry's
#: own source and arguments only.
SOURCE_DIGEST = source_digest(__file__)

_sb = _dw._sb
_col = _dw._col


class MlaDenseMaskedError(ValueError):
    """Raised for a call this kernel does not serve. There is no fallback here."""


# Positions in the list :func:`_superblock_rows` returns (the kernel front end resolves
# plain names and integer subscripts, not tuple targets).
_SB_ROWS = 0    # the superblock, keys on partitions: MM2's moving operand
_SB_T = 1       # the superblock transposed, latent on partitions: MM1's moving operand


def _superblock_rows(banked, bank_page, page0, pages, latent):
    """Window pages ``page0 .. page0 + pages - 1`` in SBUF twice: ``[rows, rows_t]``.

    ``banked`` is the bank viewed as ``[bank pages, page, L]`` and ``bank_page`` the
    ``[1, window pages]`` int32 tile of each window page's bank page
    (``mla_sparse._bank_pieces``, a -1 pad reading page 0). ``rows`` is ``[128, pages,
    L]`` (page ``j``'s row ``p`` on partition ``p``) and ``rows_t`` ``[128, L / 128,
    pages * 128]``, built by PE transposes, as many pages as fill one PSUM bank copied
    out at once on the scalar engine. Then every non-finite element of ``rows`` is set
    to 0.0: ``rows`` feeds MM2 only, ``rows_t`` keeps the row as stored for MM1.
    """
    n_lat = latent // LATENT_TILE
    kv_dtype = banked.dtype
    rows = _sb((KEY_CHUNK, pages, latent), kv_dtype)
    for j in range(pages):
        column = page0 + j
        nisa.dma_copy(dst=rows[:, j, :],
                      src=banked.ap(pattern=[[latent, KEY_CHUNK], [1, latent]], offset=0,
                                    scalar_offset=bank_page[:, column:column + 1],
                                    indirect_dim=0))
    rows_t = _sb((LATENT_TILE, n_lat, pages * KEY_CHUNK), kv_dtype)
    # A 2-byte page transposed is L / 128 blocks of 128 x 128: 2 L bytes per partition.
    per_bank = _PSUM_BANK_BYTES // (2 * latent)
    for c0 in range(0, pages, per_bank):
        nc = min(per_bank, pages - c0)
        t_ps = nl.ndarray((LATENT_TILE, n_lat, per_bank * KEY_CHUNK), dtype=kv_dtype,
                          buffer=nl.psum)
        for j in range(nc):
            for li in range(n_lat):
                nisa.nc_transpose(dst=t_ps[:, li, j * KEY_CHUNK:(j + 1) * KEY_CHUNK],
                                  data=rows[:, c0 + j, li * LATENT_TILE:(li + 1) * LATENT_TILE])
        nisa.tensor_copy(dst=rows_t[:, :, c0 * KEY_CHUNK:(c0 + nc) * KEY_CHUNK],
                         src=t_ps[:, :, 0:nc * KEY_CHUNK], engine=nisa.engine.scalar)
    # x * 0 is 0 for a finite x and NaN for an infinite or NaN one.
    finite = _sb((KEY_CHUNK, pages, latent), nl.uint8)
    nisa.tensor_scalar(dst=finite, data=rows, op0=nl.multiply, operand0=0.0, op1=nl.equal,
                       operand1=0.0, engine=nisa.engine.vector)
    nisa.select_reduce(dst=rows, predicate=finite, on_true=rows, on_false=0.0)
    ops = []
    ops.append(rows)
    ops.append(rows_t)
    return ops


# Positions in the list :func:`_tile_selection` returns.
_SEL_KEEP = 0         # [128, P / 8] uint8: the tile's rows of the keep table
_SEL_OPEN = 1         # [128, 1] int32: each query's open pool, (pos + 1) // R
_SEL_OPEN_COUNT = 2   # [128, 1] int32: the rows of it the query reads, (pos + 1) % R


def _pool_ranks(pools, pool_rows):
    """``[128, pools, pool_rows]`` uint8: each row's rank in its pool, ``0 .. pool_rows - 1``."""
    rank = _sb((ROW_TILE, pools, pool_rows), nl.uint8)
    nisa.iota(dst=rank, pattern=[[0, pools], [1, pool_rows]], offset=0)
    return rank


def _pool_numbers(pools):
    """``[128, pools]`` int32: ``0 .. pools - 1`` along the free axis, on every partition."""
    numbers = _sb((ROW_TILE, pools), nl.int32)
    nisa.iota(dst=numbers, pattern=[[1, pools]], offset=0, channel_multiplier=0)
    return numbers


def _tile_selection(keep_hbm, start, r0, pool_rows):
    """Query rows ``r0 .. r0 + 127``'s selection: ``[keep, open, open_count]`` (``_SEL_*``).

    ``start`` is ``query_start`` on every partition (``[128, 1]`` int32). The tile's rows of
    the keep table come in one DMA. Query ``r0 + p`` sits at window row ``pos = start +
    r0 + p``: its open pool is ``(pos + 1) // R`` and it reads ``(pos + 1) % R`` rows of it,
    none when its own row closes a pool.
    """
    width = keep_hbm.shape[1]
    keep = _sb((ROW_TILE, width), nl.uint8)
    nisa.dma_copy(dst=keep, src=keep_hbm.ap(pattern=[[width, ROW_TILE], [1, width]],
                                            offset=r0 * width))
    after = _col(ROW_TILE, nl.int32)
    nisa.iota(dst=after, pattern=[[0, 1]], offset=r0 + 1, channel_multiplier=1)
    nisa.tensor_tensor(dst=after, data1=after, data2=start, op=nl.add, engine=nisa.engine.vector)
    open_pool = _col(ROW_TILE, nl.int32)
    nisa.tensor_scalar(dst=open_pool, data=after, op0=nl.right_shift,
                       operand0=_ms._log2_exact(pool_rows), engine=nisa.engine.vector)
    open_count = _col(ROW_TILE, nl.int32)
    nisa.tensor_scalar(dst=open_count, data=after, op0=nl.bitwise_and, operand0=pool_rows - 1,
                       engine=nisa.engine.vector)
    sel = []
    sel.append(keep)
    sel.append(open_pool)
    sel.append(open_count)
    return sel


def _selected_rows(sel, rank, numbers, pools, w0, width):
    """``[128, width]`` uint8: 1 on window rows ``w0 .. w0 + width - 1`` the tile's queries select.

    ``sel`` is :func:`_tile_selection`'s list and ``pools`` the window's pool count ``P``.
    Row ``w`` is selected where its rank in its pool is below the query's count of that
    pool (the module docstring). Each bit plane the superblock's pools fall in is one
    integer Vector instruction: a shift that moves the plane's bit to the bit of value
    ``R``, then a mask. The open pool's count goes in by a compare of the pool numbers with
    each query's open pool, scaled by its count, and an add. Then one Vector compare reads
    each count through a stride-0 view, once per row of its pool.
    """
    pool_rows = rank.shape[2]
    set_bit = _ms._log2_exact(pool_rows)
    plane = pools // _KEEP_PLANES
    n = width // pool_rows
    p0 = w0 // pool_rows
    counts = _sb((ROW_TILE, n), nl.uint8)
    for b in range(p0 // plane, (p0 + n - 1) // plane + 1):
        lo = max(p0, b * plane)
        hi = min(p0 + n, (b + 1) * plane)
        shift = nl.right_shift if b >= set_bit else nl.left_shift
        nisa.tensor_scalar(dst=counts[:, lo - p0:hi - p0],
                           data=sel[_SEL_KEEP][:, lo - b * plane:hi - b * plane], op0=shift,
                           operand0=abs(b - set_bit), op1=nl.bitwise_and, operand1=pool_rows,
                           engine=nisa.engine.vector)
    open_here = _col(ROW_TILE, nl.int32)
    nisa.tensor_scalar(dst=open_here, data=sel[_SEL_OPEN], op0=nl.subtract, operand0=p0,
                       engine=nisa.engine.vector)
    opened = _sb((ROW_TILE, n), nl.uint8)
    nisa.tensor_scalar(dst=opened, data=numbers[:, 0:n], op0=nl.equal, operand0=open_here,
                       op1=nl.multiply, operand1=sel[_SEL_OPEN_COUNT], engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=counts, data1=counts, data2=opened, op=nl.add, engine=nisa.engine.vector)
    pred = _sb((ROW_TILE, n, pool_rows), nl.uint8)
    nisa.tensor_tensor(dst=pred, data1=counts.expand_dim(2).broadcast(2, pool_rows),
                       data2=rank[:, 0:n, :], op=nl.greater, engine=nisa.engine.vector)
    return pred.reshape((ROW_TILE, width))


def _superblock_tile(sel, rank, numbers, pools, q_t, rows_t, rows, run_max, run_sum, acc,
                     softmax_scale, w0):
    """Fold window rows ``w0 ..`` (one superblock) into one 128-query tile.

    ``sel`` is the tile's :func:`_tile_selection`; ``rank`` and ``numbers`` are
    :func:`_pool_ranks` and :func:`_pool_numbers` for the longest superblock, and ``pools``
    the window's pool count. ``q_t`` is the tile's queries with the latent on the
    partitions (``mla_dense_window._query_columns``); ``run_max``, ``run_sum`` (``[128, 1]``
    fp32) and ``acc`` (``[128, L]`` fp32) are the tile's running statistics, updated in
    place: the raw-score maximum, the sum of ``exp(scale * (score - run_max))`` and the
    unnormalised output against that maximum.
    """
    width = rows_t.shape[2]
    latent = rows.shape[2]
    kv_dtype = rows.dtype
    n_blocks = width // MOVING_MAX

    # ---- MM1 per 512 rows, the selected scores and their block maxima ------------------
    pred = _selected_rows(sel, rank, numbers, pools, w0, width)
    scores = _sb((ROW_TILE, width), nl.float32)
    block_max = _sb((ROW_TILE, n_blocks), nl.float32)
    for b in range(n_blocks):
        t0 = b * MOVING_MAX
        s_ps = _dw._window_scores(q_t, rows_t, ROW_TILE, t0, MOVING_MAX)
        nisa.select_reduce(dst=scores[:, t0:t0 + MOVING_MAX], predicate=pred[:, t0:t0 + MOVING_MAX],
                           on_true=s_ps, on_false=_UNSELECTED_SCORE,
                           reduce_res=block_max[:, b:b + 1],
                           reduce_cmd=nisa.reduce_cmd.reset_reduce, reduce_op=nl.maximum)

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


def _dense_masked_body(q_hbm, bank_hbm, table_hbm, keep_hbm, start_hbm, softmax_scale,
                       page_size, pool_rows, superblock_pages, out_hbm):
    """Every 128-query tile of this program over every superblock of the window.

    The tiles are dealt round-robin over the programs; ``program_id`` is a trace-time int
    (the kernel is traced once per program), so each program keeps only its own tiles.
    The superblock loop is outermost, so each program reads each window row once
    whatever its tile count; the last superblock may be shorter, by whole 512-row blocks.
    """
    seq, latent = q_hbm.shape[0], q_hbm.shape[2]
    pages = table_hbm.shape[0]
    pools = pages * page_size // pool_rows
    n_tiles = seq // ROW_TILE
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    banked = bank_hbm.reshape((bank_hbm.shape[0] // page_size, page_size, latent))
    bank_page = _ms._bank_pieces(table_hbm, 1)
    rank = _pool_ranks(superblock_pages * page_size // pool_rows, pool_rows)
    numbers = _pool_numbers(superblock_pages * page_size // pool_rows)
    start = _col(ROW_TILE, nl.int32)
    nisa.dma_copy(dst=start, src=start_hbm.ap(pattern=[[0, ROW_TILE], [1, 1]], offset=0))

    q_cols = []
    sels = []
    run_max = []
    run_sum = []
    acc = []
    starts = []
    mine = 0
    for qt in range(prg, n_tiles, n_prgs):
        r0 = qt * ROW_TILE
        q_cols.append(_dw._query_columns(q_hbm, ROW_TILE, latent, r0 * latent, latent))
        sels.append(_tile_selection(keep_hbm, start, r0, pool_rows))
        m = _col(ROW_TILE, nl.float32)
        nisa.memset(dst=m, value=_SELECTED_MAX_FLOOR)
        run_max.append(m)
        s = _col(ROW_TILE, nl.float32)
        nisa.memset(dst=s, value=0.0)
        run_sum.append(s)
        a = _sb((ROW_TILE, latent), nl.float32)
        nisa.memset(dst=a, value=0.0)
        acc.append(a)
        starts.append(r0)
        mine += 1

    for page0 in range(0, pages, superblock_pages):
        n_pages = min(superblock_pages, pages - page0)
        ops = _superblock_rows(banked, bank_page, page0, n_pages, latent)
        for j in range(mine):
            _superblock_tile(sels[j], rank, numbers, pools, q_cols[j], ops[_SB_T], ops[_SB_ROWS],
                             run_max[j], run_sum[j], acc[j], softmax_scale, page0 * page_size)
    for j in range(mine):
        _store_tile(out_hbm, acc[j], run_sum[j], starts[j])


@nki.jit
def mla_dense_masked_kernel(q_hbm, bank_hbm, table_hbm, keep_hbm, start_hbm, softmax_scale,
                            page_size, pool_rows, superblock_pages, source_digest=0,
                            staging_digest=0):
    """Attention of each query over the window rows its keep bits select, in one dense pass.

    Args:
        q_hbm: ``[S, 1, L]`` bf16 or fp16, the absorbed queries; ``S`` a multiple of 128,
            ``L`` a multiple of 128 at most :data:`LATENT_MAX`.
        bank_hbm: ``[slots, L]``, the whole latent bank, the dtype of ``q_hbm``. It already
            holds this step's rows (the sparse kernel's bank form).
        table_hbm: ``[pages, 1]`` int32, the window's pages in order (``-1`` pads read
            bank page 0, as the sparse kernel's do).
        keep_hbm: ``[S, P / 8]`` uint8, the keep table (the module docstring), ``P = pages *
            page_size / pool_rows``.
        start_hbm: ``[1, 1]`` int32, the window row of query 0.
        softmax_scale: python float applied to the scores before the softmax.
        page_size: python int, rows per page: :data:`KEY_CHUNK`.
        pool_rows: python int, rows per pool, a divisor of ``page_size``.
        superblock_pages: python int, pages per superblock (:data:`SUPERBLOCK_PAGES`), a
            multiple of ``MOVING_MAX // page_size``.
        source_digest, staging_digest: python ints that only key the compiled-kernel cache
            on this file and on ``mla_sparse.py`` (the bank-page helper).

    Returns:
        ``[S, 1, L]`` fp32 in shared HBM: ``softmax(scale * q . k) @ k`` over the rows the
        query selects; a query that selects no row returns zeros.

    Under a two-program launch the 128-query tiles are dealt round-robin between the
    programs, and each program reads the whole window.
    """
    seq, heads, latent = q_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _dense_masked_body(q_hbm, bank_hbm, table_hbm, keep_hbm, start_hbm, softmax_scale,
                       page_size, pool_rows, superblock_pages, out_hbm)
    return out_hbm


def _geometry_refusal(seq: int, heads: int, latent: int, kv_latent: int, bank_rows: int,
                      pages: int, q_dtype: torch.dtype, kv_dtype: torch.dtype, pool_rows: int,
                      page_size: int, superblock_pages: int) -> str | None:
    """Why the kernel does not serve this geometry, or None where it does."""
    if heads != 1 or seq < ROW_TILE or seq % ROW_TILE != 0:
        return f"queries come in whole {ROW_TILE}-row tiles of one head; got seq {seq}, heads {heads}"
    if latent % LATENT_TILE != 0 or latent > LATENT_MAX or kv_latent != latent:
        return (f"the latent must be a multiple of {LATENT_TILE} up to {LATENT_MAX}, the same in "
                f"q_lift and c_kv; got {latent} and {kv_latent}")
    if q_dtype not in (torch.bfloat16, torch.float16) or kv_dtype != q_dtype:
        return f"q_lift and c_kv must share a 2-byte float dtype; got {q_dtype} and {kv_dtype}"
    blocks = MOVING_MAX // KEY_CHUNK
    if page_size != KEY_CHUNK or pages % blocks != 0:
        return (f"a page is one {KEY_CHUNK}-row chunk and the window whole {MOVING_MAX}-row "
                f"blocks; got page_size {page_size} and {pages} pages")
    if pool_rows <= 0 or page_size % pool_rows != 0:
        return f"a pool must divide the {page_size}-row page; got {pool_rows} rows per pool"
    if pages * page_size // pool_rows % _KEEP_PLANES != 0:
        return (f"the keep table needs whole bytes of {_KEEP_PLANES} pools; got "
                f"{pages * page_size // pool_rows} pools")
    if superblock_pages < blocks or superblock_pages % blocks != 0:
        return f"a superblock is whole {MOVING_MAX}-row blocks; got {superblock_pages} pages"
    if bank_rows % KEY_CHUNK != 0:
        return f"the bank must be whole pages of {KEY_CHUNK} rows; got {bank_rows} rows"
    return None


def dense_masked_serves(seq: int, heads: int, latent: int, kv_latent: int, bank_rows: int,
                        pages: int, page_size: int, pool_rows: int, q_dtype: torch.dtype,
                        kv_dtype: torch.dtype) -> bool:
    """True where the sparse attention seam (``mla_sparse.mla_sparse_attention``) hands a
    paged NoPE call with a keep table to :func:`mla_dense_masked_attention`.

    Three conditions, each read at trace time: ``VLLM_NEURON_MLA_DENSE_MASKED`` is set; the
    window is at most ``VLLM_NEURON_MLA_DENSE_MASKED_MAX_PAGES`` pages (``vllm_neuron/envs.py``
    states where the dense pass beats the gather); and the kernel serves the geometry.
    """
    return (
        envs.VLLM_NEURON_MLA_DENSE_MASKED
        and int(pages) <= envs.VLLM_NEURON_MLA_DENSE_MASKED_MAX_PAGES
        and _geometry_refusal(int(seq), int(heads), int(latent), int(kv_latent), int(bank_rows),
                              int(pages), q_dtype, kv_dtype, int(pool_rows), int(page_size),
                              SUPERBLOCK_PAGES) is None
    )


def _require_call(q_lift: Tensor, c_kv: Tensor, keep_bits: Tensor, query_start: Tensor,
                  softmax_scale: float, block_table_row: Tensor, pool_rows: int, page_size: int,
                  superblock_pages: int) -> None:
    if not envs.VLLM_NEURON_MLA_DENSE_MASKED:
        raise MlaDenseMaskedError(
            "the dense-masked MLA kernel is off: set VLLM_NEURON_MLA_DENSE_MASKED=1 to call it"
        )
    if not softmax_scale > 0:
        raise MlaDenseMaskedError(f"softmax_scale must be positive; got {softmax_scale}")
    if q_lift.ndim != 3 or c_kv.ndim != 2 or keep_bits.ndim != 2 or block_table_row.ndim != 2:
        raise MlaDenseMaskedError(
            f"q_lift must be [seq, 1, latent], c_kv [slots, latent], keep_bits [seq, pools / "
            f"{_KEEP_PLANES}] and block_table_row [pages, 1]; got {tuple(q_lift.shape)}, "
            f"{tuple(c_kv.shape)}, {tuple(keep_bits.shape)} and {tuple(block_table_row.shape)}"
        )
    seq, heads, latent = (int(d) for d in q_lift.shape)
    pages = int(block_table_row.shape[0])
    refusal = _geometry_refusal(seq, heads, latent, int(c_kv.shape[1]), int(c_kv.shape[0]), pages,
                                q_lift.dtype, c_kv.dtype, int(pool_rows), int(page_size),
                                int(superblock_pages))
    if refusal is not None:
        raise MlaDenseMaskedError(refusal)
    width = pages * int(page_size) // int(pool_rows) // _KEEP_PLANES
    if keep_bits.dtype != torch.uint8 or tuple(keep_bits.shape) != (seq, width):
        raise MlaDenseMaskedError(
            f"keep_bits must be [seq, pools / {_KEEP_PLANES}] = [{seq}, {width}] uint8; got "
            f"{tuple(keep_bits.shape)} {keep_bits.dtype}"
        )
    if tuple(query_start.shape) != (1, 1) or query_start.dtype != torch.int32:
        raise MlaDenseMaskedError(
            f"query_start must be [1, 1] int32; got {tuple(query_start.shape)} {query_start.dtype}"
        )


def mla_dense_masked_attention(q_lift: Tensor, c_kv: Tensor, keep_bits: Tensor,
                               query_start: Tensor, softmax_scale: float,
                               block_table_row: Tensor, pool_rows: int,
                               page_size: int = KEY_CHUNK,
                               superblock_pages: int = SUPERBLOCK_PAGES) -> Tensor:
    """Sparse MLA prefill attention over each query's selection as a dense masked pass.

    The operands are ``mla_sparse.mla_sparse_attention``'s paged bank-form call, with the
    selection as the indexer's keep table (the module docstring) and the indexer's pool
    size.

    Args:
        q_lift: ``[S, 1, L]`` bf16/fp16, the absorbed query; ``S`` a multiple of 128.
        c_kv: ``[slots, L]`` the whole latent bank, same dtype, already holding this
            step's rows.
        keep_bits: ``[S, P / 8]`` uint8, the keep table, ``P = pages * page_size /
            pool_rows``.
        query_start: ``[1, 1]`` int32, the window row of query 0 (the sparse call's
            ``write_offset``); read on the device.
        softmax_scale: positive.
        block_table_row: ``[pages, 1]`` int32, the window's pages in order, ``-1`` pads.
        pool_rows: rows per pool, a divisor of ``page_size``.
        page_size: rows per page, :data:`KEY_CHUNK`.
        superblock_pages: pages per superblock.

    Returns:
        ``[S, 1, L]`` fp32, the sparse kernel's answer up to the order and rounding of its
        fp32 sums (a value-equal class, not bit-equal).

    Raises:
        MlaDenseMaskedError: while ``VLLM_NEURON_MLA_DENSE_MASKED`` is unset, or for a
            malformed call or a geometry the kernel does not serve.
    """
    _require_call(q_lift, c_kv, keep_bits, query_start, softmax_scale, block_table_row,
                  pool_rows, page_size, superblock_pages)
    call = wrap_nki(mla_dense_masked_kernel)
    if lnc_pair() and int(q_lift.shape[0]) >= 2 * ROW_TILE:
        call = call[2]
    return call(
        q_lift.contiguous(),
        c_kv.contiguous(),
        block_table_row.contiguous().to(torch.int32),
        keep_bits.contiguous(),
        query_start.contiguous(),
        float(softmax_scale),
        int(page_size),
        int(pool_rows),
        int(superblock_pages),
        SOURCE_DIGEST,
        int(_ms.SOURCE_DIGEST),
    )
