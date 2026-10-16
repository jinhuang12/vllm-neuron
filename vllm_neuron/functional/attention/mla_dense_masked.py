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

The selection is a pool mask: one uint8 per ``R``-row pool of the window (``R`` the
indexer's pool size), the count of the pool's leading rows the query selects.
:func:`selection_pool_counts` builds it from the index rows the sparse call takes, and
the kernel reads it through a stride-0 view, each count against its ``R`` rows' ranks
in the pool. One byte per pool is the exact selection because the indexer selects whole
pools. ``dsa_index_expand`` lays each query's index row out in groups of ``R``
columns: a selected pool's ``R`` rows, or ``R`` times ``-1``, then the open pool (the
one holding the query's own position) as its leading rows up to that position, then
``-1``. So every group names the leading rows of one pool, and its count of
non-negative columns says which. A row mask holds the same bits at ``R`` times the
bytes, and it costs its producer one scattered element per selected row.

The entry, :func:`mla_dense_masked_attention`, takes that pool mask, and it refuses a
call unless ``VLLM_NEURON_MLA_DENSE_MASKED`` (``vllm_neuron/envs.py``) is set. The sparse
attention seam (``mla_sparse.mla_sparse_attention``) builds the mask and calls the entry
for a paged NoPE call whose bank already holds the step's rows, wherever
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
#: its guard mask (8 KiB), and each query tile adds its pool counts (0.5 KiB at 4-row
#: pools), predicate (2 KiB), scores (8 KiB) and one bank of hi/lo halves (2 KiB): about
#: 57 KiB, with each program's accumulators (2 KiB per tile) and pool ranks (2 KiB), far
#: below the 224 KiB of a partition, which leaves the compiler room to double-buffer the
#: superblocks.
SUPERBLOCK_PAGES = 16
#: The score an unselected row reads: a power of two so far below any score that
#: ``exp(scale * (_UNSELECTED_SCORE - max))`` is exactly 0.0.
_UNSELECTED_SCORE = -(2.0 ** 30)
#: The running maximum starts here: a query that has selected no row yet
#: exponentiates every row to 0.0, and one that selects none returns zeros.
_SELECTED_MAX_FLOOR = _UNSELECTED_SCORE / 2
#: The row-sum floor of the final division.
_TOTAL_FLOOR = 1.0e-30
#: Index groups per scatter of the pool-mask producer: it scatters the largest whole
#: multiple of this many groups first, then the rest. One scatter over a group count off
#: that multiple (544 at the served index width) had its group axis tiled to 1,024 in
#: the compiled graph, twice the DMA instances: the compiler's estimate was 2.7 ms per
#: call against 1.5 ms for the same groups scattered as 512, then 32.
_SCATTER_GROUPS = 128

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


def _pool_ranks(pools, pool_rows):
    """``[128, pools, pool_rows]`` uint8: each row's rank in its pool, ``0 .. pool_rows - 1``."""
    rank = _sb((ROW_TILE, pools, pool_rows), nl.uint8)
    nisa.iota(dst=rank, pattern=[[0, pools], [1, pool_rows]], offset=0)
    return rank


def _selected_rows(pool_hbm, rank, r0, w0, width):
    """``[128, width]`` uint8: 1 on window rows ``w0 .. w0 + width - 1`` query ``r0 + p`` selects.

    Row ``w`` is selected where its rank in its pool is below the pool's count. The counts
    of the superblock's pools come in one DMA, then one Vector compare reads each through a
    stride-0 view, once per row of its pool. The pool mask's trailing column (the
    producer's trash) is never read.
    """
    pool_rows = rank.shape[2]
    stride = pool_hbm.shape[1]
    pools = width // pool_rows
    counts = _sb((ROW_TILE, pools), nl.uint8)
    nisa.dma_copy(dst=counts, src=pool_hbm.ap(pattern=[[stride, ROW_TILE], [1, pools]],
                                              offset=r0 * stride + w0 // pool_rows))
    pred = _sb((ROW_TILE, pools, pool_rows), nl.uint8)
    nisa.tensor_tensor(dst=pred, data1=counts.expand_dim(2).broadcast(2, pool_rows),
                       data2=rank[:, 0:pools, :], op=nl.greater, engine=nisa.engine.vector)
    return pred.reshape((ROW_TILE, width))


def _superblock_tile(pool_hbm, rank, q_t, rows_t, rows, run_max, run_sum, acc, softmax_scale,
                     r0, w0):
    """Fold window rows ``w0 ..`` (one superblock) into query rows ``r0 .. r0 + 127``.

    ``rank`` is :func:`_pool_ranks` for the longest superblock. ``q_t`` is the tile's
    queries with the latent on the partitions (``mla_dense_window._query_columns``);
    ``run_max``, ``run_sum`` (``[128, 1]`` fp32) and ``acc`` (``[128, L]`` fp32) are the
    tile's running statistics, updated in place: the raw-score maximum, the sum of
    ``exp(scale * (score - run_max))`` and the unnormalised output against that maximum.
    """
    width = rows_t.shape[2]
    latent = rows.shape[2]
    kv_dtype = rows.dtype
    n_blocks = width // MOVING_MAX

    # ---- MM1 per 512 rows, the selected scores and their block maxima ------------------
    pred = _selected_rows(pool_hbm, rank, r0, w0, width)
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


def _dense_masked_body(q_hbm, bank_hbm, table_hbm, pool_hbm, softmax_scale, page_size,
                       pool_rows, superblock_pages, out_hbm):
    """Every 128-query tile of this program over every superblock of the window.

    The tiles are dealt round-robin over the programs; ``program_id`` is a trace-time int
    (the kernel is traced once per program), so each program keeps only its own tiles.
    The superblock loop is outermost, so each program reads each window row once
    whatever its tile count; the last superblock may be shorter, by whole 512-row blocks.
    """
    seq, latent = q_hbm.shape[0], q_hbm.shape[2]
    pages = table_hbm.shape[0]
    n_tiles = seq // ROW_TILE
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    banked = bank_hbm.reshape((bank_hbm.shape[0] // page_size, page_size, latent))
    bank_page = _ms._bank_pieces(table_hbm, 1)
    rank = _pool_ranks(superblock_pages * page_size // pool_rows, pool_rows)

    q_cols = []
    run_max = []
    run_sum = []
    acc = []
    starts = []
    mine = 0
    for qt in range(prg, n_tiles, n_prgs):
        r0 = qt * ROW_TILE
        q_cols.append(_dw._query_columns(q_hbm, ROW_TILE, latent, r0 * latent, latent))
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
            _superblock_tile(pool_hbm, rank, q_cols[j], ops[_SB_T], ops[_SB_ROWS], run_max[j],
                             run_sum[j], acc[j], softmax_scale, starts[j], page0 * page_size)
    for j in range(mine):
        _store_tile(out_hbm, acc[j], run_sum[j], starts[j])


@nki.jit
def mla_dense_masked_kernel(q_hbm, bank_hbm, table_hbm, pool_hbm, softmax_scale, page_size,
                            pool_rows, superblock_pages, source_digest=0, staging_digest=0):
    """Attention of each query over the window rows its pool mask selects, in one dense pass.

    Args:
        q_hbm: ``[S, 1, L]`` bf16 or fp16, the absorbed queries; ``S`` a multiple of 128,
            ``L`` a multiple of 128 at most :data:`LATENT_MAX`.
        bank_hbm: ``[slots, L]``, the whole latent bank, the dtype of ``q_hbm``. It already
            holds this step's rows (the sparse kernel's bank form).
        table_hbm: ``[pages, 1]`` int32, the window's pages in order (``-1`` pads read
            bank page 0, as the sparse kernel's do).
        pool_hbm: ``[S, pools + 1]`` uint8, ``pools = pages * page_size / pool_rows``
            (:func:`selection_pool_counts`): for query ``s`` and window pool ``g`` the
            count of the pool's leading rows the query selects; the last column is never
            read.
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
    _dense_masked_body(q_hbm, bank_hbm, table_hbm, pool_hbm, softmax_scale, page_size,
                       pool_rows, superblock_pages, out_hbm)
    return out_hbm


def _scatter_pool_counts(mask: Tensor, part: Tensor, pool_rows: int, pools: int) -> Tensor:
    """``mask`` with the counts of ``part``'s index groups scattered in (:func:`selection_pool_counts`)."""
    seq = int(part.shape[0])
    groups = part.reshape(seq, int(part.shape[1]) // pool_rows, pool_rows)
    counts = (groups >= 0).to(torch.int32).sum(dim=2, dtype=torch.int32).to(torch.uint8)
    first = groups[:, :, 0]
    pool = torch.where(first >= 0, first // pool_rows, torch.full_like(first, pools))
    return mask.scatter(1, pool.long(), counts)


def selection_pool_counts(indices: Tensor, pool_rows: int, pools: int) -> Tensor:
    """The pool mask of index rows in ``dsa_index_expand``'s layout: ``[S, pools + 1]`` uint8.

    Each group of ``pool_rows`` columns of ``indices`` names the leading rows of one window
    pool, or no row (the module docstring). The group's count of non-negative columns is
    scattered to its pool's column, the pool of its first index; a group of ``-1`` sends
    its count, 0, to the trailing trash column instead, so no pool's column is written by
    an empty group. This rests on the layout's two facts: the pools of one row are
    distinct, and each index is a window row, below ``pools * pool_rows``. The groups go
    in two scatters (:data:`_SCATTER_GROUPS`); they write distinct pools, so the order
    does not change the mask.

    Args:
        indices: ``[S, width]`` int32 window rows, ``-1`` for none; ``width`` a multiple
            of ``pool_rows``.
        pool_rows: rows per pool, the indexer's pool size.
        pools: the window's pools, ``pages * page_size / pool_rows``.

    Returns:
        ``[S, pools + 1]`` uint8: each selected pool's count (1 to ``pool_rows``), 0
        elsewhere and in the trash column.

    Raises:
        MlaDenseMaskedError: for index rows that are not whole groups of ``pool_rows``
            int32 columns.
    """
    if indices.ndim != 2 or indices.dtype != torch.int32 or int(pool_rows) <= 0 \
            or int(indices.shape[1]) % int(pool_rows) != 0:
        raise MlaDenseMaskedError(
            f"indices must be [seq, width] int32 with width a multiple of the pool size; got "
            f"{tuple(indices.shape)} {indices.dtype} and pool size {pool_rows}"
        )
    pool_rows, pools = int(pool_rows), int(pools)
    width = int(indices.shape[1])
    head = width // pool_rows // _SCATTER_GROUPS * _SCATTER_GROUPS * pool_rows
    mask = torch.zeros(int(indices.shape[0]), pools + 1, dtype=torch.uint8, device=indices.device)
    for start, stop in ((0, head), (head, width)):
        if stop > start:
            mask = _scatter_pool_counts(mask, indices[:, start:stop], pool_rows, pools)
    return mask


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
    if superblock_pages < blocks or superblock_pages % blocks != 0:
        return f"a superblock is whole {MOVING_MAX}-row blocks; got {superblock_pages} pages"
    if bank_rows % KEY_CHUNK != 0:
        return f"the bank must be whole pages of {KEY_CHUNK} rows; got {bank_rows} rows"
    return None


def dense_masked_serves(seq: int, heads: int, latent: int, kv_latent: int, bank_rows: int,
                        pages: int, page_size: int, pool_rows: int, index_cols: int,
                        q_dtype: torch.dtype, kv_dtype: torch.dtype) -> bool:
    """True where the sparse attention seam (``mla_sparse.mla_sparse_attention``) hands a
    paged NoPE call to :func:`mla_dense_masked_attention`.

    Three conditions, each read at trace time: ``VLLM_NEURON_MLA_DENSE_MASKED`` is set; the
    window is at most ``VLLM_NEURON_MLA_DENSE_MASKED_MAX_PAGES`` pages (``vllm_neuron/envs.py``
    states where the dense pass beats the gather); and the kernel serves the geometry, with
    index rows in whole groups of ``pool_rows`` columns (:func:`selection_pool_counts`).
    """
    return (
        envs.VLLM_NEURON_MLA_DENSE_MASKED
        and int(pages) <= envs.VLLM_NEURON_MLA_DENSE_MASKED_MAX_PAGES
        and int(pool_rows) > 0
        and int(index_cols) % int(pool_rows) == 0
        and _geometry_refusal(int(seq), int(heads), int(latent), int(kv_latent), int(bank_rows),
                              int(pages), q_dtype, kv_dtype, int(pool_rows), int(page_size),
                              SUPERBLOCK_PAGES) is None
    )


def _require_call(q_lift: Tensor, c_kv: Tensor, pool_mask: Tensor, softmax_scale: float,
                  block_table_row: Tensor, pool_rows: int, page_size: int,
                  superblock_pages: int) -> None:
    if not envs.VLLM_NEURON_MLA_DENSE_MASKED:
        raise MlaDenseMaskedError(
            "the dense-masked MLA kernel is off: set VLLM_NEURON_MLA_DENSE_MASKED=1 to call it"
        )
    if not softmax_scale > 0:
        raise MlaDenseMaskedError(f"softmax_scale must be positive; got {softmax_scale}")
    if q_lift.ndim != 3 or c_kv.ndim != 2 or pool_mask.ndim != 2 or block_table_row.ndim != 2:
        raise MlaDenseMaskedError(
            f"q_lift must be [seq, 1, latent], c_kv [slots, latent], pool_mask [seq, pools + 1] "
            f"and block_table_row [pages, 1]; got {tuple(q_lift.shape)}, {tuple(c_kv.shape)}, "
            f"{tuple(pool_mask.shape)} and {tuple(block_table_row.shape)}"
        )
    seq, heads, latent = (int(d) for d in q_lift.shape)
    pages = int(block_table_row.shape[0])
    refusal = _geometry_refusal(seq, heads, latent, int(c_kv.shape[1]), int(c_kv.shape[0]), pages,
                                q_lift.dtype, c_kv.dtype, int(pool_rows), int(page_size),
                                int(superblock_pages))
    if refusal is not None:
        raise MlaDenseMaskedError(refusal)
    pools = pages * int(page_size) // int(pool_rows)
    if pool_mask.dtype != torch.uint8 or tuple(pool_mask.shape) != (seq, pools + 1):
        raise MlaDenseMaskedError(
            f"pool_mask must be [seq, pools + 1] = [{seq}, {pools + 1}] uint8 "
            f"(selection_pool_counts); got {tuple(pool_mask.shape)} {pool_mask.dtype}"
        )


def mla_dense_masked_attention(q_lift: Tensor, c_kv: Tensor, pool_mask: Tensor,
                               softmax_scale: float, block_table_row: Tensor, pool_rows: int,
                               page_size: int = KEY_CHUNK,
                               superblock_pages: int = SUPERBLOCK_PAGES) -> Tensor:
    """Sparse MLA prefill attention over each query's selection as a dense masked pass.

    The operands are ``mla_sparse.mla_sparse_attention``'s paged bank-form call, with the
    selection as its pool mask (:func:`selection_pool_counts` of the index rows) and the
    indexer's pool size.

    Args:
        q_lift: ``[S, 1, L]`` bf16/fp16, the absorbed query; ``S`` a multiple of 128.
        c_kv: ``[slots, L]`` the whole latent bank, same dtype, already holding this
            step's rows.
        pool_mask: ``[S, pools + 1]`` uint8, :func:`selection_pool_counts`.
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
    _require_call(q_lift, c_kv, pool_mask, softmax_scale, block_table_row, pool_rows, page_size,
                  superblock_pages)
    call = wrap_nki(mla_dense_masked_kernel)
    if lnc_pair() and int(q_lift.shape[0]) >= 2 * ROW_TILE:
        call = call[2]
    return call(
        q_lift.contiguous(),
        c_kv.contiguous(),
        block_table_row.contiguous().to(torch.int32),
        pool_mask.contiguous(),
        float(softmax_scale),
        int(page_size),
        int(pool_rows),
        int(superblock_pages),
        SOURCE_DIGEST,
        int(_ms.SOURCE_DIGEST),
    )
