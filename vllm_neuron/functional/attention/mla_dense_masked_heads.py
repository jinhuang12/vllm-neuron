# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA prefill attention of two query heads in one dense masked pass over the window.

``mla_dense_masked.py`` attends one query head per call. All the query heads of a rank read
the same latent bank, page table, keep table and query start: the latent is one KV head that
every query head shares, and the keep table holds one bit per query and pool whatever the
head. A rank that holds two query heads would call that kernel twice and do twice the work
that does not depend on the head. This kernel attends both heads in one call and does that
work once:

==========================================================  ===================  ==============
work                                                        two one-head calls   this kernel
==========================================================  ===================  ==============
page loads, transposes and finite guard of each superblock  twice                once
selection predicate of each (128-query tile, superblock)    twice                once
MM1, select and maximum, exp, probability transposes, MM2   once per head        once per head
==========================================================  ===================  ==============

The work of one head is the one-head kernel's, instruction for instruction: this file calls
that file's helpers for every step of an item, on the same operands. So head ``h`` of the
output equals, bit for bit, the one-head kernel's output for ``q_lift[:, h:h + 1]`` with the
same bank, table, keep table, query start and ``VLLM_NEURON_MLA_DENSE_MASKED_BF16_PROBS``
form. The module docstring of ``mla_dense_masked.py`` states that arithmetic, the keep-table
layout and the handling of non-finite rows; all of it holds here per head.

Order. As in the one-head kernel, the 128-query tiles are dealt round-robin over the programs,
and each program runs its items superblock by superblock. Here an item is one (tile, head)
against one superblock, and the items of a superblock run tile by tile with the heads
innermost, so each tile's predicate serves its heads on consecutive items. Each step issues
MM1 of the next item ahead of the probabilities and MM2 of the current one (the Tensor Engine
has MM1 to run while the softmax of the item before is on the Vector and scalar engines),
builds the predicate of a tile two items before its first head needs it, and readies one
piece of the next superblock: a superblock is readied in as many pieces as it has items.

SBUF per partition, against the one-head kernel at the same window and query count: per tile,
one more fp32 accumulator (2 KiB at latent 512) and one more set of query columns (1 KiB at
latent 512 in bf16). The superblocks, the keep-table rows and the items in flight do not grow:
at 2,048 pages and four tiles per program, about 12 KiB more than the one-head kernel's
estimate (``mla_dense_masked.SUPERBLOCK_PAGES``).

The entry, :func:`mla_dense_masked_heads_attention`, takes the one-head entry's arguments with
``q_lift`` ``[S, 2, L]`` and returns ``[S, 2, L]``. :func:`dense_masked_heads_serves` is the
one-head predicate (``mla_dense_masked.dense_masked_serves``) with two heads. Both read the
one-head kernel's knobs at trace time (``vllm_neuron/envs.py``): ``VLLM_NEURON_MLA_DENSE_MASKED``
(the route; the entry refuses a call while it is off), ``VLLM_NEURON_MLA_DENSE_MASKED_MAX_PAGES``
(the largest window the route takes) and ``VLLM_NEURON_MLA_DENSE_MASKED_BF16_PROBS`` (the form
of MM2).
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from torch import Tensor

from vllm_neuron import envs
from vllm_neuron.functional.attention import mla_dense_masked as _dm
from vllm_neuron.functional.attention import mla_dense_window as _dw
from vllm_neuron.functional.attention import mla_sparse as _ms
from vllm_neuron.functional.attention.mla_dense_masked import MlaDenseMaskedError
from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.source_digest import source_digest

#: Query heads per call. One head is the one-head kernel's (``mla_dense_masked.py``). Each
#: head past it adds an accumulator and query columns per tile (the module docstring); no
#: wider form is built.
HEADS = 2
#: The one-head kernel's tile extents and its pages per superblock: this kernel runs its
#: items, so it runs them at its extents.
ROW_TILE = _dm.ROW_TILE
KEY_CHUNK = _dm.KEY_CHUNK
SUPERBLOCK_PAGES = _dm.SUPERBLOCK_PAGES

#: The code of this file and of the two files whose helpers its kernel calls, as a trace-time
#: int: the compiled-kernel cache keys on the entry's own source and arguments only.
SOURCE_DIGEST = source_digest(__file__, _dm.__file__, _dw.__file__)

_sb = _dw._sb
_col = _dw._col


def _item_scores(preds, q_cols, sbs, tops, run_max0, softmax_scale, heads, per_sb, item):
    """Item ``item``'s ``mla_dense_masked._tile_scores``: its tile's predicate and its head's
    query columns, against the maximum that tile and head reached one superblock before.

    Item ``item`` is (tile ``item % per_sb // heads``, head ``item % heads``) against superblock
    ``item // per_sb``; its tile's predicate is ``preds[item // heads]`` and its state (query
    columns, statistics, accumulator) is at index ``item % per_sb``.
    """
    state = item % per_sb
    run_max = tops[item - per_sb][_dm._TOP_MAX] if item >= per_sb else run_max0[state]
    return _dm._tile_scores(preds[item // heads], q_cols[state], sbs[item // per_sb][_dm._SB_T],
                            run_max, softmax_scale)


def _store_head_tile(out_hbm, acc, run_sum, r0, head):
    """``mla_dense_masked._store_tile`` for one head: ``acc / max(run_sum, floor)`` to rows
    ``r0 .. r0 + 127`` of head ``head`` of ``out_hbm`` (``[S, heads, L]``).

    The arithmetic is the one-head store's, instruction for instruction; only the DMA's row
    stride (``heads * L``) and offset name the head.
    """
    heads = out_hbm.shape[1]
    latent = acc.shape[1]
    total = _col(ROW_TILE, nl.float32)
    nisa.tensor_scalar(dst=total, data=run_sum, op0=nl.maximum, operand0=_dm._TOTAL_FLOOR,
                       engine=nisa.engine.vector)
    recip = _col(ROW_TILE, nl.float32)
    nisa.reciprocal(dst=recip, data=total)
    out_sb = _sb((ROW_TILE, latent), nl.float32)
    nisa.tensor_scalar(dst=out_sb, data=acc, op0=nl.multiply, operand0=recip,
                       engine=nisa.engine.vector)
    nisa.dma_copy(dst=out_hbm.ap(pattern=[[heads * latent, ROW_TILE], [1, latent]],
                                 offset=(r0 * heads + head) * latent),
                  src=out_sb)


def _dense_masked_heads_body(q_hbm, bank_hbm, table_hbm, keep_hbm, start_hbm, softmax_scale,
                             page_size, pool_rows, superblock_pages, bf16_probabilities,
                             out_hbm):
    """Every (128-query tile, head) of this program over every superblock of the window, pipelined.

    ``mla_dense_masked._dense_masked_body`` with the heads inside the tiles: the program keeps
    its tiles (round-robin over the programs; ``program_id`` is a trace-time int), and per
    tile its keep-table rows and first row; per tile and head, at index ``tile * heads +
    head``, the query columns, the running maximum and sum and the accumulator. A superblock
    holds ``per_sb = tiles * heads`` items in that index order, and item ``k`` is the one-head
    kernel's item ``k // heads`` for head ``k % heads``: the same predicate, superblock and
    statistics chain. The schedule is the one-head kernel's with ``per_sb`` items per
    superblock, plus one rule: a tile's predicate is built at the step two items before its
    first head (``_dense_masked_body``'s docstring gives the rest).
    """
    seq = q_hbm.shape[0]
    heads = q_hbm.shape[1]
    latent = q_hbm.shape[2]
    pages = table_hbm.shape[0]
    pools = pages * page_size // pool_rows
    n_tiles = seq // ROW_TILE
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    banked = bank_hbm.reshape((bank_hbm.shape[0] // page_size, page_size, latent))
    bank_page = _ms._bank_pieces(table_hbm, 1)
    rank = _dm._pool_ranks(superblock_pages * page_size // pool_rows, pool_rows)
    numbers = _dm._pool_numbers(superblock_pages * page_size // pool_rows)
    start = _col(ROW_TILE, nl.int32)
    nisa.dma_copy(dst=start, src=start_hbm.ap(pattern=[[0, ROW_TILE], [1, 1]], offset=0))

    sels = []
    starts = []
    q_cols = []
    run_max0 = []
    run_sum = []
    acc = []
    mine = 0
    for qt in range(prg, n_tiles, n_prgs):
        r0 = qt * ROW_TILE
        sels.append(_dm._tile_selection(keep_hbm, start, r0, pool_rows))
        starts.append(r0)
        for head in range(heads):
            q_cols.append(_dw._query_columns(q_hbm, ROW_TILE, heads * latent,
                                             (r0 * heads + head) * latent, latent))
            m = _col(ROW_TILE, nl.float32)
            nisa.memset(dst=m, value=_dm._SELECTED_MAX_FLOOR)
            run_max0.append(m)
            s = _col(ROW_TILE, nl.float32)
            nisa.memset(dst=s, value=0.0)
            run_sum.append(s)
            a = _sb((ROW_TILE, latent), nl.float32)
            nisa.memset(dst=a, value=0.0)
            acc.append(a)
        mine += 1

    per_sb = mine * heads
    n_sb = -(-pages // superblock_pages)
    n_items = n_sb * per_sb
    sbs = []
    preds = []
    tops = []
    sbs.append(_dm._superblock_load(banked, bank_page, 0,
                                    _dm._superblock_pages(pages, superblock_pages, 0), latent))
    _dm._prep_piece(sbs, pages, superblock_pages, 0, 0, 1)
    if n_sb > 1:
        sbs.append(_dm._superblock_load(banked, bank_page, superblock_pages,
                                        _dm._superblock_pages(pages, superblock_pages, 1), latent))
    for item in range(min(2, n_items)):
        if item % heads == 0:
            preds.append(_dm._item_pred(sels, rank, numbers, pools, page_size, pages,
                                        superblock_pages, mine, item // heads))
    tops.append(_item_scores(preds, q_cols, sbs, tops, run_max0, softmax_scale, heads, per_sb, 0))
    _dm._tile_exp(tops[0], run_sum[0], softmax_scale)

    for k in range(n_items):
        sb = k // per_sb
        i = k - sb * per_sb
        has_next = sb + 1 < n_sb
        if i == 0 and sb > 0 and has_next:
            sbs.append(_dm._superblock_load(banked, bank_page, (sb + 1) * superblock_pages,
                                            _dm._superblock_pages(pages, superblock_pages, sb + 1),
                                            latent))
        if i == per_sb - 1 and has_next:
            _dm._prep_piece(sbs, pages, superblock_pages, sb + 1, i, per_sb)
        if k + 1 < n_items:
            tops.append(_item_scores(preds, q_cols, sbs, tops, run_max0, softmax_scale, heads,
                                     per_sb, k + 1))
        split = _dm._tile_probabilities(tops[k][_dm._TOP_SCORES], q_hbm.dtype, bf16_probabilities)
        if i < per_sb - 1 and has_next:
            _dm._prep_piece(sbs, pages, superblock_pages, sb + 1, i, per_sb)
        if k + 1 < n_items:
            _dm._tile_exp(tops[k + 1], run_sum[(k + 1) % per_sb], softmax_scale)
        _dm._tile_values(split, sbs[sb][_dm._SB_ROWS], acc[i], tops[k][_dm._TOP_ALPHA],
                         bf16_probabilities)
        if k + 2 < n_items and (k + 2) % heads == 0:
            preds.append(_dm._item_pred(sels, rank, numbers, pools, page_size, pages,
                                        superblock_pages, mine, (k + 2) // heads))
    for state in range(per_sb):
        _store_head_tile(out_hbm, acc[state], run_sum[state], starts[state // heads],
                         state % heads)


@nki.jit
def mla_dense_masked_heads_kernel(q_hbm, bank_hbm, table_hbm, keep_hbm, start_hbm,
                                  softmax_scale, page_size, pool_rows, superblock_pages,
                                  bf16_probabilities=0, source_digest=0, staging_digest=0):
    """Attention of each query of each head over the window rows its keep bits select, in one
    dense pass that both heads share.

    Args:
        q_hbm: ``[S, heads, L]`` bf16 or fp16, the absorbed queries of :data:`HEADS` heads;
            ``S`` a multiple of 128, ``L`` a multiple of 128 at most
            ``mla_dense_masked.LATENT_MAX``.
        bank_hbm, table_hbm, keep_hbm, start_hbm, softmax_scale, page_size, pool_rows,
        superblock_pages, bf16_probabilities: as ``mla_dense_masked.mla_dense_masked_kernel``
            takes them; the keep table and the query start serve every head.
        source_digest, staging_digest: python ints that only key the compiled-kernel cache on
            this file and the two whose helpers it calls (:data:`SOURCE_DIGEST`) and on
            ``mla_sparse.py`` (the bank-page helper).

    Returns:
        ``[S, heads, L]`` fp32 in shared HBM; head ``h`` is the one-head kernel's output for
        ``q_hbm[:, h:h + 1]``, bit for bit.

    Under a two-program launch the 128-query tiles are dealt round-robin between the programs,
    and each program reads the whole window once for both heads.
    """
    seq, heads, latent = q_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _dense_masked_heads_body(q_hbm, bank_hbm, table_hbm, keep_hbm, start_hbm, softmax_scale,
                             page_size, pool_rows, superblock_pages, bf16_probabilities, out_hbm)
    return out_hbm


def _geometry_refusal(seq: int, heads: int, latent: int, kv_latent: int, bank_rows: int,
                      pages: int, q_dtype: torch.dtype, kv_dtype: torch.dtype, pool_rows: int,
                      page_size: int, superblock_pages: int) -> str | None:
    """Why the kernel does not serve this geometry, or None where it does: :data:`HEADS` heads,
    and every other condition the one-head kernel's (``mla_dense_masked._geometry_refusal``)."""
    if heads != HEADS:
        return (f"this kernel attends {HEADS} query heads per call (one head is "
                f"mla_dense_masked's); got {heads}")
    if seq < ROW_TILE or seq % ROW_TILE != 0:
        return f"queries come in whole {ROW_TILE}-row tiles; got seq {seq}"
    return _dm._geometry_refusal(seq, 1, latent, kv_latent, bank_rows, pages, q_dtype, kv_dtype,
                                 pool_rows, page_size, superblock_pages)


def dense_masked_heads_serves(seq: int, heads: int, latent: int, kv_latent: int, bank_rows: int,
                              pages: int, page_size: int, pool_rows: int, q_dtype: torch.dtype,
                              kv_dtype: torch.dtype) -> bool:
    """True where a paged NoPE call with a keep table and ``heads`` query heads may take
    :func:`mla_dense_masked_heads_attention`: ``mla_dense_masked.dense_masked_serves``'s
    arguments and conditions, with :data:`HEADS` heads in place of one.

    Three conditions, each read at trace time: ``VLLM_NEURON_MLA_DENSE_MASKED`` is on; the
    window is at most ``VLLM_NEURON_MLA_DENSE_MASKED_MAX_PAGES`` pages; and the kernel serves
    the geometry.
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
    """The one-head entry's checks (``mla_dense_masked._require_call``) with :data:`HEADS` heads."""
    if not envs.VLLM_NEURON_MLA_DENSE_MASKED:
        raise MlaDenseMaskedError(
            "the dense-masked MLA kernels are off: set VLLM_NEURON_MLA_DENSE_MASKED=1 to call them"
        )
    if not softmax_scale > 0:
        raise MlaDenseMaskedError(f"softmax_scale must be positive; got {softmax_scale}")
    if q_lift.ndim != 3 or c_kv.ndim != 2 or keep_bits.ndim != 2 or block_table_row.ndim != 2:
        raise MlaDenseMaskedError(
            f"q_lift must be [seq, {HEADS}, latent], c_kv [slots, latent], keep_bits [seq, "
            f"pools / {_dm._KEEP_PLANES}] and block_table_row [pages, 1]; got "
            f"{tuple(q_lift.shape)}, {tuple(c_kv.shape)}, {tuple(keep_bits.shape)} and "
            f"{tuple(block_table_row.shape)}"
        )
    seq, heads, latent = (int(d) for d in q_lift.shape)
    pages = int(block_table_row.shape[0])
    refusal = _geometry_refusal(seq, heads, latent, int(c_kv.shape[1]), int(c_kv.shape[0]), pages,
                                q_lift.dtype, c_kv.dtype, int(pool_rows), int(page_size),
                                int(superblock_pages))
    if refusal is not None:
        raise MlaDenseMaskedError(refusal)
    width = pages * int(page_size) // int(pool_rows) // _dm._KEEP_PLANES
    if keep_bits.dtype != torch.uint8 or tuple(keep_bits.shape) != (seq, width):
        raise MlaDenseMaskedError(
            f"keep_bits must be [seq, pools / {_dm._KEEP_PLANES}] = [{seq}, {width}] uint8; got "
            f"{tuple(keep_bits.shape)} {keep_bits.dtype}"
        )
    if tuple(query_start.shape) != (1, 1) or query_start.dtype != torch.int32:
        raise MlaDenseMaskedError(
            f"query_start must be [1, 1] int32; got {tuple(query_start.shape)} {query_start.dtype}"
        )


def mla_dense_masked_heads_attention(q_lift: Tensor, c_kv: Tensor, keep_bits: Tensor,
                                     query_start: Tensor, softmax_scale: float,
                                     block_table_row: Tensor, pool_rows: int,
                                     page_size: int = KEY_CHUNK,
                                     superblock_pages: int = SUPERBLOCK_PAGES) -> Tensor:
    """Sparse MLA prefill attention of :data:`HEADS` query heads over each query's selection, as
    one dense masked pass.

    ``mla_dense_masked.mla_dense_masked_attention``'s call with two heads: the heads share the
    bank, the window, the keep table and the query start.

    Args:
        q_lift: ``[S, 2, L]`` bf16/fp16, the absorbed queries of two heads; ``S`` a multiple
            of 128.
        c_kv: ``[slots, L]`` the whole latent bank, same dtype, already holding this step's
            rows.
        keep_bits: ``[S, P / 8]`` uint8, the keep table, ``P = pages * page_size /
            pool_rows``; one table for both heads.
        query_start: ``[1, 1]`` int32, the window row of query 0; read on the device.
        softmax_scale: positive.
        block_table_row: ``[pages, 1]`` int32, the window's pages in order, ``-1`` pads.
        pool_rows: rows per pool, a divisor of ``page_size``.
        page_size: rows per page, :data:`KEY_CHUNK`.
        superblock_pages: pages per superblock.

    ``VLLM_NEURON_MLA_DENSE_MASKED_BF16_PROBS`` is read here at trace time, as the one-head
    entry reads it.

    Returns:
        ``[S, 2, L]`` fp32. Head ``h`` is, bit for bit, what
        ``mla_dense_masked.mla_dense_masked_attention`` returns for ``q_lift[:, h:h + 1]`` on
        the other operands.

    Raises:
        MlaDenseMaskedError: while ``VLLM_NEURON_MLA_DENSE_MASKED`` is off, or for a malformed
            call or a geometry the kernel does not serve.
    """
    _require_call(q_lift, c_kv, keep_bits, query_start, softmax_scale, block_table_row,
                  pool_rows, page_size, superblock_pages)
    call = wrap_nki(mla_dense_masked_heads_kernel)
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
        int(envs.VLLM_NEURON_MLA_DENSE_MASKED_BF16_PROBS),
        SOURCE_DIGEST,
        int(_ms.SOURCE_DIGEST),
    )
