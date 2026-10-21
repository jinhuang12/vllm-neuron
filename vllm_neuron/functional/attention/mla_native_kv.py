# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA prefill attention over native K/V pages, and the kernel that writes those pages.

With one attention head per rank the expanded keys and values of a token take as many
bytes as its latent row: ``key_width + value_width`` against ``latent_width`` elements
(256 + 256 against 512 for a 64-head model on 64 ranks). Attending them directly needs no
query lift and no value projection, and both attention matmuls run at the head width in
place of the latent width::

    key[t]   = latent[t] @ w_key                      (stored once, at the cache write)
    value[t] = finite(latent[t] @ w_value)            (non-finite elements stored as 0.0)
    scores[q, w] = query[q] . key[w]      where w % R < count[q, w // R], else _UNSELECTED_SCORE
    out[q]       = softmax(scale * scores[q]) @ value

which equals the latent form ``softmax(scale * (query @ w_key^T) . latent) @ latent @ w_value``
in exact arithmetic; ``R`` is ``pool_rows`` and ``count[q, p]`` the number of rows query ``q``
keeps in pool ``p``. The selection (the keep table, the causal cut at each query's open pool)
and the softmax are ``mla_dense_masked``'s: without ``engine_balance`` this kernel runs its
stages unchanged, and only the page layout, the operand widths and the guard's place differ.

Page layout (the contract between the two kernels). A KV page holds ``page_size`` tokens in
``page_size * (key_width + value_width)`` elements of the cache dtype, keys first:

* keys, contraction-major: element ``p * (key_width // 128) * page_size + t * page_size + r``
  holds key column ``t * 128 + p`` of the page's token ``r``. One DMA per page puts it on the
  128 partitions as MM1's moving operand (the key columns on the partitions), so no
  transpose of the keys runs in the attention kernel.
* then values, key-major: element ``key_width * page_size + r * value_width + d`` holds value
  column ``d`` of token ``r``: MM2's moving operand (the tokens on the partitions).

Each half is ``key_width * 2`` or ``value_width * 2`` contiguous bytes per partition per page
(512 B at width 256 in bf16). At ``key_width + value_width == latent_width`` a KV page has the
bytes of a latent page, so a cache of the latent cache's shape holds it.

Rounding. The write kernel accumulates both projections in fp32 on the Tensor Engine and
rounds each key and value element to the cache dtype once. The attention arithmetic is
``mla_dense_masked``'s with bf16 probabilities: fp32 scores from the stored query and keys,
the online softmax in fp32, MM2 over ``bf16(p)`` and the stored values into fp32. So the
rounding points are the stored keys and values (one rounding each) and the probabilities;
the latent form rounds the lifted query, the latent and the 512-wide attention output
instead. Neither form is the other's up to order: a bf16-K/V class.

Non-finite rows. A non-finite latent element makes its token's whole key and value rows
non-finite. The key row is stored as computed, so a query that selects the token gets a
non-finite score and the predicate drops it for every other query, as in the latent form.
MM2 contracts every token of a superblock whatever its probability, and ``0 * NaN`` is NaN,
so the write kernel stores every non-finite value element as 0.0: the attention kernel needs
no guard of its own, and a non-finite token reaches exactly the queries that select it.

Engine balance. With ``engine_balance`` the attention kernel makes three changes, in this
file's ``_balanced_*`` helpers: the Scalar engine expands each pool count to its rows
(``relu(count - rank)``, one activation per rank in the pool) in place of one Vector compare
over every row; the four 512-row selects of a superblock keep their running maximum in the
Vector accumulator and read it out once; the exp writes ``bf16(p)`` itself and the Tensor
Engine transposes it at the 2-byte rate. The selected rows, the maxima, the row sum and the
one rounding of each probability are those of the form without it.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from torch import Tensor

from vllm_neuron.functional.attention import mla_dense_masked as _dm
from vllm_neuron.functional.attention import mla_dense_window as _dw
from vllm_neuron.functional.attention import mla_sparse as _ms
from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.source_digest import source_digest

#: Partition extent: query and token rows are tiled to it, the key and latent columns are
#: tiled to it for the matmul contractions. A page is one token tile (``mla_dense_masked``).
ROW_TILE = _dm.ROW_TILE
KEY_TILE = _dm.LATENT_TILE
PAGE_ROWS = _dm.KEY_CHUNK
#: The Tensor Engine and PSUM limits (``mla_dense_window`` owns them): one matmul's moving
#: free-axis extent (MM1 scores 512 tokens per instruction), the widest fp32 row one PSUM
#: bank holds (the MM2 result row and each projection's), and the PSUM bytes per partition
#: of one bank.
MOVING_MAX = _dm.MOVING_MAX
WIDTH_MAX = _dm.LATENT_MAX
_PSUM_BANK_BYTES = _dm._PSUM_BANK_BYTES
#: Pages per superblock of the attention kernel: ``mla_dense_masked``'s. Per partition a
#: superblock holds its keys and its values once each (16 x 512 B + 16 x 512 B at width 256
#: in bf16), and two superblocks are held at once (the one in use and the next, loading).
SUPERBLOCK_PAGES = _dm.SUPERBLOCK_PAGES

#: This file's digest, and those of the modules whose helpers its kernels trace, as
#: trace-time ints: the compiled-kernel cache keys on an entry's own source and arguments
#: only, so a helper edit must reach the key through an argument.
SOURCE_DIGEST = source_digest(__file__)
MASKED_DIGEST = int(_dm.SOURCE_DIGEST)
WINDOW_DIGEST = int(_dw.SOURCE_DIGEST)
STAGING_DIGEST = int(_ms.SOURCE_DIGEST)

_sb = _dw._sb
_col = _dw._col


class MlaNativeKvError(ValueError):
    """Raised for a call these kernels do not serve. There is no fallback here."""


# ---------------------------------------------------------------------------------------
# The write kernel
# ---------------------------------------------------------------------------------------


def _page_columns(page_ids_hbm, pages_in_pool):
    """``[1, n]`` int32: each page id clamped into ``0 .. pages_in_pool - 1``.

    The ids are device data the checks cannot read. An out-of-range dynamic offset drops
    the whole DMA tile on the device without an error, so each id is clamped onto the pool
    first; a caller passes ids of pool pages. The row is padded to whole 32-byte lines
    (``mla_sparse._bank_pieces``'s rule).
    """
    count = page_ids_hbm.shape[0]
    ids = _ms._sbuf_i32(1, _ms._aligned(count))[:, 0:count]
    nisa.dma_copy(dst=ids, src=page_ids_hbm.ap(pattern=[[count, 1], [1, count]], offset=0))
    nisa.tensor_scalar(dst=ids, data=ids, op0=nl.maximum, operand0=0, op1=nl.minimum,
                       operand1=pages_in_pool - 1, engine=nisa.engine.vector)
    return ids


def _weight_tiles(w_hbm):
    """``[128, L / 128, width]``: ``w_hbm`` (``[L, width]``) with its rows on the partitions,
    latent tile ``li`` holding rows ``li * 128 ..``: the contraction-side operand."""
    latent, width = w_hbm.shape
    n_lat = latent // KEY_TILE
    w_sb = _sb((KEY_TILE, n_lat, width), w_hbm.dtype)
    nisa.dma_copy(dst=w_sb, src=w_hbm.ap(pattern=[[width, KEY_TILE], [KEY_TILE * width, n_lat],
                                                  [1, width]], offset=0))
    return w_sb


def _latent_columns(latent_hbm, row0, pages):
    """Tokens ``row0 .. row0 + pages * 128 - 1`` of ``latent_hbm`` (``[T, L]``) transposed:
    ``[128, L / 128, pages * 128]``, the latent columns on the partitions.

    One DMA puts the tokens on the partitions, a page per free-axis slot, and PE transposes
    (data movement only: a non-finite element stays in its own token's column) fill one
    PSUM bank at a time, copied out on the scalar engine.
    """
    latent = latent_hbm.shape[1]
    n_lat = latent // KEY_TILE
    rows = _sb((PAGE_ROWS, pages, latent), latent_hbm.dtype)
    nisa.dma_copy(dst=rows, src=latent_hbm.ap(pattern=[[latent, PAGE_ROWS],
                                                       [PAGE_ROWS * latent, pages], [1, latent]],
                                              offset=row0 * latent))
    cols = _sb((KEY_TILE, n_lat, pages * PAGE_ROWS), latent_hbm.dtype)
    # A 2-byte page transposed is L / 128 blocks of 128 x 128: 2 L bytes per partition.
    per_bank = _PSUM_BANK_BYTES // (2 * latent)
    for c0 in range(0, pages, per_bank):
        nc = min(per_bank, pages - c0)
        t_ps = nl.ndarray((KEY_TILE, n_lat, per_bank * PAGE_ROWS), dtype=latent_hbm.dtype,
                          buffer=nl.psum)
        for j in range(nc):
            for li in range(n_lat):
                nisa.nc_transpose(dst=t_ps[:, li, j * PAGE_ROWS:(j + 1) * PAGE_ROWS],
                                  data=rows[:, c0 + j, li * KEY_TILE:(li + 1) * KEY_TILE])
        nisa.tensor_copy(dst=cols[:, :, c0 * PAGE_ROWS:(c0 + nc) * PAGE_ROWS],
                         src=t_ps[:, :, 0:nc * PAGE_ROWS], engine=nisa.engine.scalar)
    return cols


def _project_keys(cols, w_key_sb, kv_dtype):
    """``[128, key_width / 128, n]`` in ``kv_dtype``: the keys of the ``n`` tokens of ``cols``
    with the key columns on the partitions, key tile ``t`` holding columns ``t * 128 ..``.

    Per key tile one fp32 PSUM bank of up to 512 tokens takes the accumulating matmuls over
    the latent tiles (``w_key`` stationary, the tokens moving), and one copy rounds it.
    """
    n_lat, tokens = cols.shape[1], cols.shape[2]
    n_key = w_key_sb.shape[2] // KEY_TILE
    keys = _sb((KEY_TILE, n_key, tokens), kv_dtype)
    for t in range(n_key):
        k_ps = nl.ndarray((KEY_TILE, MOVING_MAX), dtype=nl.float32, buffer=nl.psum)
        for li in range(n_lat):
            nisa.nc_matmul(dst=k_ps[:, 0:tokens],
                           stationary=w_key_sb[:, li, t * KEY_TILE:(t + 1) * KEY_TILE],
                           moving=cols[:, li, :], accumulate=(li > 0))
        nisa.tensor_copy(dst=keys[:, t, :], src=k_ps[:, 0:tokens], engine=nisa.engine.scalar)
    return keys


def _project_values(cols, w_value_sb, kv_dtype):
    """``[128, pages, value_width]`` in ``kv_dtype``: the values of the tokens of ``cols``, a
    page's tokens on the partitions, every non-finite element stored as 0.0.

    Per page one fp32 PSUM bank takes the accumulating matmuls over the latent tiles (the
    page's tokens stationary, ``w_value`` moving), and one copy rounds it. Then the
    finite test runs on the rounded values (an fp32 value past the dtype's range rounds to
    infinity, and the stored value is the one MM2 reads).
    """
    n_lat = cols.shape[1]
    pages = cols.shape[2] // PAGE_ROWS
    value_width = w_value_sb.shape[2]
    values = _sb((PAGE_ROWS, pages, value_width), kv_dtype)
    for j in range(pages):
        v_ps = nl.ndarray((PAGE_ROWS, value_width), dtype=nl.float32, buffer=nl.psum)
        for li in range(n_lat):
            nisa.nc_matmul(dst=v_ps, stationary=cols[:, li, j * PAGE_ROWS:(j + 1) * PAGE_ROWS],
                           moving=w_value_sb[:, li, :], accumulate=(li > 0))
        nisa.tensor_copy(dst=values[:, j, :], src=v_ps, engine=nisa.engine.vector)
    # x * 0 is 0 for a finite x and NaN for an infinite or NaN one.
    finite = _sb((PAGE_ROWS, pages, value_width), nl.uint8)
    nisa.tensor_scalar(dst=finite, data=values, op0=nl.multiply, operand0=0.0, op1=nl.equal,
                       operand1=0.0, engine=nisa.engine.vector)
    nisa.select_reduce(dst=values, predicate=finite, on_true=values, on_false=0.0)
    return values


def _store_pages(kv_flat, ids, keys, values, page0, key_width, page_size):
    """Pages ``page0 ..`` of the chunk to their pool pages: per page one DMA of its keys and
    one of its values, each to the page ``ids[:, page0 + j]`` names (the layout of the
    module docstring)."""
    n_key = keys.shape[1]
    value_width = values.shape[2]
    for j in range(values.shape[1]):
        column = page0 + j
        nisa.dma_copy(dst=kv_flat.ap(pattern=[[n_key * page_size, KEY_TILE], [page_size, n_key],
                                              [1, page_size]], offset=0,
                                     scalar_offset=ids[:, column:column + 1], indirect_dim=0),
                      src=keys[:, :, j * page_size:(j + 1) * page_size])
        nisa.dma_copy(dst=kv_flat.ap(pattern=[[value_width, page_size], [1, value_width]],
                                     offset=key_width * page_size,
                                     scalar_offset=ids[:, column:column + 1], indirect_dim=0),
                      src=values[:, j, :])


@nki.jit
def mla_native_kv_write_kernel(latent_hbm, w_key_hbm, w_value_hbm, page_ids_hbm, kv_hbm,
                               page_size, source_digest=0, staging_digest=0):
    """Project a chunk's latent rows to keys and values and store them as KV pages, in place.

    Args:
        latent_hbm: ``[T, L]`` bf16 or fp16, the chunk's latent rows; ``T`` whole pages,
            ``L`` a multiple of 128 up to :data:`WIDTH_MAX`.
        w_key_hbm: ``[L, key_width]``, the latent-to-key projection (``key = latent @
            w_key``), the dtype of ``latent_hbm``.
        w_value_hbm: ``[L, value_width]``, the latent-to-value projection.
        page_ids_hbm: ``[T / page_size, 1]`` int32, the pool page of each ``page_size``
            tokens of the chunk, in order.
        kv_hbm: ``[pool pages, page_size, key_width + value_width]``, the KV pages (the
            module docstring), the dtype of ``latent_hbm``; written in place.
        page_size: python int, tokens per page: :data:`PAGE_ROWS`.
        source_digest, staging_digest: python ints that only key the compiled-kernel cache
            on this file and on ``mla_sparse.py`` (the int32 tile helpers).

    Returns:
        ``kv_hbm``, the chunk's pages written whole: every token of a page comes from the
        chunk, so a chunk starts on a page boundary.

    Under a two-program launch the chunk's 512-token blocks are dealt round-robin between
    the programs; each program loads both projections.
    """
    tokens = latent_hbm.shape[0]
    key_width = w_key_hbm.shape[1]
    pool_pages = kv_hbm.shape[0]
    kv_flat = kv_hbm.reshape((pool_pages, page_size * (key_width + w_value_hbm.shape[1])))
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    ids = _page_columns(page_ids_hbm, pool_pages)
    w_key_sb = _weight_tiles(w_key_hbm)
    w_value_sb = _weight_tiles(w_value_hbm)
    block_pages = MOVING_MAX // page_size
    chunk_pages = tokens // page_size
    n_blocks = -(-chunk_pages // block_pages)
    for blk in range(prg, n_blocks, n_prgs):
        page0 = blk * block_pages
        pages = min(block_pages, chunk_pages - page0)
        cols = _latent_columns(latent_hbm, page0 * page_size, pages)
        keys = _project_keys(cols, w_key_sb, kv_hbm.dtype)
        values = _project_values(cols, w_value_sb, kv_hbm.dtype)
        _store_pages(kv_flat, ids, keys, values, page0, key_width, page_size)
    return kv_hbm


# ---------------------------------------------------------------------------------------
# The attention kernel
# ---------------------------------------------------------------------------------------

# Positions in the list :func:`_superblock_load` returns (the kernel front end resolves
# plain names and integer subscripts, not tuple targets).
_SB_VALUES = 0   # [128, pages, value_width]: tokens on partitions, MM2's moving operand
_SB_KEYS = 1     # [128, key_width / 128, pages * 128]: key columns on partitions, MM1's


def _superblock_load(kv_flat, bank_page, page0, pages, key_width, value_width, page_size,
                     kv_dtype):
    """Window pages ``page0 .. page0 + pages - 1`` into SBUF: ``[values, keys]`` (``_SB_*``).

    ``kv_flat`` is the KV pages viewed as ``[pool pages, page elements]`` and ``bank_page``
    the ``[1, window pages]`` int32 tile of each window page's pool page
    (``mla_sparse._bank_pieces``, a -1 pad reading page 0). Per page one DMA moves its keys
    and one its values, each ``2 * width`` contiguous bytes per partition.
    """
    n_key = key_width // KEY_TILE
    values = _sb((PAGE_ROWS, pages, value_width), kv_dtype)
    keys = _sb((KEY_TILE, n_key, pages * page_size), kv_dtype)
    for j in range(pages):
        column = page0 + j
        nisa.dma_copy(dst=keys[:, :, j * page_size:(j + 1) * page_size],
                      src=kv_flat.ap(pattern=[[n_key * page_size, KEY_TILE], [page_size, n_key],
                                              [1, page_size]], offset=0,
                                     scalar_offset=bank_page[:, column:column + 1],
                                     indirect_dim=0))
        nisa.dma_copy(dst=values[:, j, :],
                      src=kv_flat.ap(pattern=[[value_width, page_size], [1, value_width]],
                                     offset=key_width * page_size,
                                     scalar_offset=bank_page[:, column:column + 1],
                                     indirect_dim=0))
    ops = []
    ops.append(values)
    ops.append(keys)
    return ops


def _balanced_selected_rows(sel, numbers, pools, pool_rows, w0, width):
    """``[128, width]`` uint8, not 0 exactly on the window rows ``w0 .. w0 + width - 1`` the
    tile's queries select: ``mla_dense_masked._selected_rows`` with the last step on the
    scalar engine.

    The pool counts are built as there (one integer Vector instruction per bit plane, the
    causal cut before each query's open pool, the open pool's count added). Then the scalar
    engine writes the rows of rank ``r`` of every pool as ``relu(count - r)``, one activation
    per rank: ``count - r`` on a selected row (exact small integers), 0 elsewhere.
    """
    set_bit = _ms._log2_exact(pool_rows)
    plane = pools // _dm._KEEP_PLANES
    n = width // pool_rows
    p0 = w0 // pool_rows
    counts = _sb((ROW_TILE, n), nl.uint8)
    for b in range(p0 // plane, (p0 + n - 1) // plane + 1):
        lo = max(p0, b * plane)
        hi = min(p0 + n, (b + 1) * plane)
        shift = nl.right_shift if b >= set_bit else nl.left_shift
        nisa.tensor_scalar(dst=counts[:, lo - p0:hi - p0],
                           data=sel[_dm._SEL_KEEP][:, lo - b * plane:hi - b * plane], op0=shift,
                           operand0=abs(b - set_bit), op1=nl.bitwise_and, operand1=pool_rows,
                           engine=nisa.engine.vector)
    open_here = _col(ROW_TILE, nl.float32)
    nisa.tensor_scalar(dst=open_here, data=sel[_dm._SEL_OPEN], op0=nl.subtract,
                       operand0=float(p0), engine=nisa.engine.vector)
    nisa.scalar_tensor_tensor(dst=counts, data=numbers[:, 0:n], op0=nl.less, operand0=open_here,
                              op1=nl.multiply, operand1=counts)
    opened = _sb((ROW_TILE, n), nl.uint8)
    nisa.tensor_scalar(dst=opened, data=numbers[:, 0:n], op0=nl.equal, operand0=open_here,
                       op1=nl.multiply, operand1=sel[_dm._SEL_OPEN_COUNT],
                       engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=counts, data1=counts, data2=opened, op=nl.add,
                       engine=nisa.engine.vector)
    pred = _sb((ROW_TILE, n, pool_rows), nl.uint8)
    for r in range(pool_rows):
        nisa.activation(dst=pred[:, :, r:r + 1], op=nl.relu, data=counts.expand_dim(2),
                        bias=float(-r), scale=1.0)
    return pred.reshape((ROW_TILE, width))


def _balanced_tile_scores(pred, q_t, keys, run_max, softmax_scale):
    """``mla_dense_masked._tile_scores`` with one maximum read-out per item.

    The 512-token selects share the Vector accumulator (the first resets it, the others add
    to it, no other accumulating Vector instruction runs between them) and the last one
    reads the item's maximum out, in place of a read-out per block and a reduce. Returns
    the ``mla_dense_masked._TOP_*`` list.
    """
    width = keys.shape[2]
    n_blocks = width // MOVING_MAX
    scores = _sb((ROW_TILE, width), nl.float32)
    new_max = _col(ROW_TILE, nl.float32)
    for b in range(n_blocks):
        t0 = b * MOVING_MAX
        s_ps = _dw._window_scores(q_t, keys, ROW_TILE, t0, MOVING_MAX)
        nisa.select_reduce(dst=scores[:, t0:t0 + MOVING_MAX], predicate=pred[:, t0:t0 + MOVING_MAX],
                           on_true=s_ps, on_false=_dm._UNSELECTED_SCORE,
                           reduce_res=new_max if b == n_blocks - 1 else None,
                           reduce_cmd=(nisa.reduce_cmd.reset_reduce if b == 0
                                       else nisa.reduce_cmd.reduce),
                           reduce_op=nl.maximum)
    nisa.tensor_tensor(dst=new_max, data1=new_max, data2=run_max, op=nl.maximum,
                       engine=nisa.engine.vector)
    neg_top = _col(ROW_TILE, nl.float32)
    nisa.tensor_scalar(dst=neg_top, data=new_max, op0=nl.multiply, operand0=-softmax_scale,
                       engine=nisa.engine.vector)
    alpha = _col(ROW_TILE, nl.float32)
    nisa.activation(dst=alpha, op=nl.exp, data=run_max, bias=neg_top, scale=softmax_scale)
    top = []
    top.append(scores)
    top.append(neg_top)
    top.append(alpha)
    top.append(new_max)
    return top


def _balanced_tile_exp(top, run_sum, softmax_scale, probs_dtype):
    """``mla_dense_masked._tile_exp`` with the exp writing ``p`` in ``probs_dtype`` to a new
    tile, returned: one rounding of each probability at the output, while the sum is the
    scalar engine's reduction of the fp32 values before that rounding."""
    probs = _sb(top[_dm._TOP_SCORES].shape, probs_dtype)
    block_sum = _col(ROW_TILE, nl.float32)
    nisa.activation(dst=probs, op=nl.exp, data=top[_dm._TOP_SCORES], bias=top[_dm._TOP_NEG],
                    scale=softmax_scale, reduce_op=nl.add, reduce_res=block_sum,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    nisa.scalar_tensor_tensor(dst=run_sum, data=run_sum, op0=nl.multiply,
                              operand0=top[_dm._TOP_ALPHA], op1=nl.add, operand1=block_sum)
    return probs


def _balanced_tile_probabilities(probs):
    """The item's 2-byte probabilities (``[128 queries, width]``) on the key partitions:
    PE transposes at the 2-byte rate, as many 128-column chunks as fill one PSUM bank,
    each bank copied out on the scalar engine. Returns the ``mla_dense_masked._P_*`` list
    (hi alone)."""
    width = probs.shape[1]
    split = []
    split.append(_sb((PAGE_ROWS, width), probs.dtype))
    # A 2-byte transpose of 128 columns takes 256 bytes of a PSUM bank per partition.
    per_bank = _PSUM_BANK_BYTES // (2 * ROW_TILE) * ROW_TILE
    for c0 in range(0, width, per_bank):
        nc = min(per_bank, width - c0)
        pt_ps = nl.ndarray((PAGE_ROWS, per_bank), dtype=probs.dtype, buffer=nl.psum)
        for k0 in range(0, nc, PAGE_ROWS):
            nisa.nc_transpose(dst=pt_ps[:, k0:k0 + ROW_TILE],
                              data=probs[:, c0 + k0:c0 + k0 + PAGE_ROWS])
        nisa.tensor_copy(dst=split[_dm._P_HI][:, c0:c0 + nc], src=pt_ps[:, 0:nc],
                         engine=nisa.engine.scalar)
    return split


def _item_pred(sels, rank, numbers, pools, pool_rows, page_size, pages, superblock_pages,
               mine, item, engine_balance):
    """Item ``item``'s selected rows: tile ``item % mine`` over superblock ``item // mine``."""
    sb = item // mine
    width = _dm._superblock_pages(pages, superblock_pages, sb) * page_size
    w0 = sb * superblock_pages * page_size
    if engine_balance:
        return _balanced_selected_rows(sels[item % mine], numbers, pools, pool_rows, w0, width)
    return _dm._selected_rows(sels[item % mine], rank, numbers, pools, w0, width)


def _item_scores(preds, q_cols, sbs, tops, run_max0, softmax_scale, mine, item, engine_balance):
    """Item ``item``'s MM1 and maximum, against the maximum its tile reached one item before."""
    j = item % mine
    run_max = tops[item - mine][_dm._TOP_MAX] if item >= mine else run_max0[j]
    keys = sbs[item // mine][_SB_KEYS]
    if engine_balance:
        return _balanced_tile_scores(preds[item], q_cols[j], keys, run_max, softmax_scale)
    return _dm._tile_scores(preds[item], q_cols[j], keys, run_max, softmax_scale)


def _item_exp(top, run_sum, softmax_scale, kv_dtype, engine_balance):
    """Item's exps and row sum; returns the probabilities MM2's transposes read."""
    if engine_balance:
        return _balanced_tile_exp(top, run_sum, softmax_scale, kv_dtype)
    _dm._tile_exp(top, run_sum, softmax_scale)
    return top[_dm._TOP_SCORES]


def _item_probabilities(probs, kv_dtype, engine_balance):
    """The item's probabilities on the key partitions in the cache dtype (``_P_*`` list)."""
    if engine_balance:
        return _balanced_tile_probabilities(probs)
    return _dm._tile_probabilities(probs, kv_dtype, True)


def _native_body(q_hbm, kv_hbm, table_hbm, keep_hbm, start_hbm, softmax_scale, page_size,
                 pool_rows, superblock_pages, engine_balance, out_hbm):
    """Every 128-query tile of this program over every superblock of the window, pipelined.

    ``mla_dense_masked._dense_masked_body``'s schedule (the items, their stages and their
    program order) with the superblock read from the KV pages: the keys arrive with their
    columns on the partitions and the values arrive finite, so a superblock needs no
    transposes and no guard once loaded. The next superblock is loaded at the start of a
    superblock, two items ahead of the MM1 that reads it.
    """
    seq, key_width = q_hbm.shape[0], q_hbm.shape[2]
    value_width = kv_hbm.shape[2] - key_width
    kv_dtype = kv_hbm.dtype
    pages = table_hbm.shape[0]
    pools = pages * page_size // pool_rows
    n_tiles = seq // ROW_TILE
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    pool_pages = kv_hbm.shape[0]
    kv_flat = kv_hbm.reshape((pool_pages, page_size * (key_width + value_width)))
    bank_page = _ms._bank_pieces(table_hbm, 1)
    # The engine-balanced form expands the pool counts on the scalar engine, without ranks.
    rank = None if engine_balance else _dm._pool_ranks(superblock_pages * page_size // pool_rows,
                                                       pool_rows)
    numbers = _dm._pool_numbers(superblock_pages * page_size // pool_rows)
    start = _col(ROW_TILE, nl.int32)
    nisa.dma_copy(dst=start, src=start_hbm.ap(pattern=[[0, ROW_TILE], [1, 1]], offset=0))

    q_cols = []
    sels = []
    run_max0 = []
    run_sum = []
    acc = []
    starts = []
    mine = 0
    for qt in range(prg, n_tiles, n_prgs):
        r0 = qt * ROW_TILE
        q_cols.append(_dw._query_columns(q_hbm, ROW_TILE, key_width, r0 * key_width, key_width))
        sels.append(_dm._tile_selection(keep_hbm, start, r0, pool_rows))
        m = _col(ROW_TILE, nl.float32)
        nisa.memset(dst=m, value=_dm._SELECTED_MAX_FLOOR)
        run_max0.append(m)
        s = _col(ROW_TILE, nl.float32)
        nisa.memset(dst=s, value=0.0)
        run_sum.append(s)
        a = _sb((ROW_TILE, value_width), nl.float32)
        nisa.memset(dst=a, value=0.0)
        acc.append(a)
        starts.append(r0)
        mine += 1

    n_sb = -(-pages // superblock_pages)
    n_items = n_sb * mine
    sbs = []
    preds = []
    tops = []
    probs = []
    sbs.append(_superblock_load(kv_flat, bank_page, 0,
                                _dm._superblock_pages(pages, superblock_pages, 0), key_width,
                                value_width, page_size, kv_dtype))
    if n_sb > 1:
        sbs.append(_superblock_load(kv_flat, bank_page, superblock_pages,
                                    _dm._superblock_pages(pages, superblock_pages, 1), key_width,
                                    value_width, page_size, kv_dtype))
    for item in range(min(2, n_items)):
        preds.append(_item_pred(sels, rank, numbers, pools, pool_rows, page_size, pages,
                                superblock_pages, mine, item, engine_balance))
    tops.append(_item_scores(preds, q_cols, sbs, tops, run_max0, softmax_scale, mine, 0,
                             engine_balance))
    probs.append(_item_exp(tops[0], run_sum[0], softmax_scale, kv_dtype, engine_balance))

    for k in range(n_items):
        sb = k // mine
        j = k - sb * mine
        if j == 0 and sb > 0 and sb + 1 < n_sb:
            sbs.append(_superblock_load(kv_flat, bank_page, (sb + 1) * superblock_pages,
                                        _dm._superblock_pages(pages, superblock_pages, sb + 1),
                                        key_width, value_width, page_size, kv_dtype))
        if k + 1 < n_items:
            tops.append(_item_scores(preds, q_cols, sbs, tops, run_max0, softmax_scale, mine,
                                     k + 1, engine_balance))
        split = _item_probabilities(probs[k], kv_dtype, engine_balance)
        if k + 1 < n_items:
            probs.append(_item_exp(tops[k + 1], run_sum[(k + 1) % mine], softmax_scale, kv_dtype,
                                   engine_balance))
        _dm._tile_values(split, sbs[sb][_SB_VALUES], acc[j], tops[k][_dm._TOP_ALPHA], True)
        if k + 2 < n_items:
            preds.append(_item_pred(sels, rank, numbers, pools, pool_rows, page_size, pages,
                                    superblock_pages, mine, k + 2, engine_balance))
    for j in range(mine):
        _dm._store_tile(out_hbm, acc[j], run_sum[j], starts[j])


@nki.jit
def mla_native_kv_attention_kernel(q_hbm, kv_hbm, table_hbm, keep_hbm, start_hbm, softmax_scale,
                                   page_size, pool_rows, superblock_pages, engine_balance=0,
                                   source_digest=0, masked_digest=0, window_digest=0,
                                   staging_digest=0):
    """Attention of each query over the window tokens its keep bits select, from KV pages.

    Args:
        q_hbm: ``[S, 1, key_width]`` bf16 or fp16, the queries (not lifted); ``S`` a multiple
            of 128, ``key_width`` a multiple of 128 up to :data:`WIDTH_MAX`.
        kv_hbm: ``[pool pages, page_size, key_width + value_width]``, the KV pages (the
            module docstring), the dtype of ``q_hbm``; ``value_width`` a multiple of 128 up to
            :data:`WIDTH_MAX`. They already hold this step's tokens.
        table_hbm: ``[pages, 1]`` int32, the window's pages in order (``-1`` pads read pool
            page 0).
        keep_hbm: ``[S, P / 8]`` uint8, the keep table (``mla_dense_masked``), ``P = pages *
            page_size / pool_rows``.
        start_hbm: ``[1, 1]`` int32, the window row of query 0.
        softmax_scale: python float applied to the scores before the softmax.
        page_size: python int, tokens per page: :data:`PAGE_ROWS`.
        pool_rows: python int, tokens per pool, a divisor of ``page_size``.
        superblock_pages: python int, pages per superblock, a multiple of ``MOVING_MAX //
            page_size``.
        engine_balance: python int, 1 for the engine-balanced stages (the module docstring),
            0 for ``mla_dense_masked``'s; a kernel argument, so each form has its own
            compiled-kernel cache entry.
        source_digest, masked_digest, window_digest, staging_digest: python ints that only key
            the compiled-kernel cache on this file and on the helper modules it traces.

    Returns:
        ``[S, 1, value_width]`` fp32 in shared HBM: ``softmax(scale * q . key) @ value`` over the
        tokens the query selects, with bf16 probabilities; a query that selects no token
        returns zeros.

    Under a two-program launch the 128-query tiles are dealt round-robin between the
    programs, and each program reads the whole window.
    """
    seq, heads, key_width = q_hbm.shape
    out_hbm = nl.ndarray((seq, heads, kv_hbm.shape[2] - key_width), dtype=nl.float32,
                         buffer=nl.shared_hbm)
    _native_body(q_hbm, kv_hbm, table_hbm, keep_hbm, start_hbm, softmax_scale, page_size,
                 pool_rows, superblock_pages, engine_balance, out_hbm)
    return out_hbm


# ---------------------------------------------------------------------------------------
# Host entries
# ---------------------------------------------------------------------------------------


def _width_refusal(name: str, width: int) -> str | None:
    if width <= 0 or width % KEY_TILE != 0 or width > WIDTH_MAX:
        return f"{name} must be a multiple of {KEY_TILE} up to {WIDTH_MAX}; got {width}"
    return None


def _kv_refusal(kv_pages: Tensor, key_width: int, value_width: int, page_size: int,
                dtype: torch.dtype) -> str | None:
    if page_size != PAGE_ROWS:
        return f"a page is one {PAGE_ROWS}-token tile; got page_size {page_size}"
    if dtype not in (torch.bfloat16, torch.float16):
        return f"the operands must share a 2-byte float dtype; got {dtype}"
    want = (page_size, key_width + value_width)
    if (kv_pages.ndim != 3 or tuple(kv_pages.shape[1:]) != want or kv_pages.dtype != dtype
            or not kv_pages.is_contiguous()):
        return (f"kv_pages must be a contiguous [pool pages, {want[0]}, {want[1]}] {dtype} tensor; "
                f"got {tuple(kv_pages.shape)} {kv_pages.dtype}")
    return None


def mla_native_kv_write(latent: Tensor, w_key: Tensor, w_value: Tensor, page_ids: Tensor,
                        kv_pages: Tensor, page_size: int = PAGE_ROWS) -> Tensor:
    """Store a chunk's keys and values as KV pages: ``key = latent @ w_key`` and ``value =
    latent @ w_value`` in fp32, each rounded once to the cache dtype, every non-finite value
    element stored as 0.0 (the module docstring).

    Args:
        latent: ``[T, L]`` bf16/fp16, the chunk's latent rows; ``T`` whole pages.
        w_key: ``[L, key_width]``, same dtype.
        w_value: ``[L, value_width]``, same dtype.
        page_ids: ``[T / page_size]`` or ``[T / page_size, 1]`` int32, the distinct pool
            pages of the chunk's pages in order, each clamped onto the pool on the device.
        kv_pages: ``[pool pages, page_size, key_width + value_width]``, written in place.
        page_size: tokens per page, :data:`PAGE_ROWS`.

    Returns:
        The written KV pages: in a traced graph the kernel's result, which the backend hands
        to every later operation on the same graph input.

    Raises:
        MlaNativeKvError: for a malformed call or a geometry the kernel does not serve.
    """
    if latent.ndim != 2 or w_key.ndim != 2 or w_value.ndim != 2:
        raise MlaNativeKvError(
            f"latent must be [T, L], w_key [L, key_width] and w_value [L, value_width]; got "
            f"{tuple(latent.shape)}, {tuple(w_key.shape)} and {tuple(w_value.shape)}")
    tokens, latent_width = (int(d) for d in latent.shape)
    key_width, value_width = int(w_key.shape[1]), int(w_value.shape[1])
    for name, width in (("the latent", latent_width), ("key_width", key_width),
                        ("value_width", value_width)):
        refusal = _width_refusal(name, width)
        if refusal is not None:
            raise MlaNativeKvError(refusal)
    if int(w_key.shape[0]) != latent_width or int(w_value.shape[0]) != latent_width:
        raise MlaNativeKvError(
            f"w_key and w_value must have {latent_width} rows; got {tuple(w_key.shape)} and "
            f"{tuple(w_value.shape)}")
    if w_key.dtype != latent.dtype or w_value.dtype != latent.dtype:
        raise MlaNativeKvError(
            f"latent, w_key and w_value must share a dtype; got {latent.dtype}, {w_key.dtype} "
            f"and {w_value.dtype}")
    refusal = _kv_refusal(kv_pages, key_width, value_width, int(page_size), latent.dtype)
    if refusal is not None:
        raise MlaNativeKvError(refusal)
    if tokens <= 0 or tokens % int(page_size) != 0:
        raise MlaNativeKvError(f"the chunk must be whole {page_size}-token pages; got {tokens}")
    chunk_pages = tokens // int(page_size)
    if page_ids.dtype != torch.int32 or int(page_ids.numel()) != chunk_pages:
        raise MlaNativeKvError(
            f"page_ids must be {chunk_pages} int32 page ids; got {tuple(page_ids.shape)} "
            f"{page_ids.dtype}")
    call = wrap_nki(mla_native_kv_write_kernel)
    if lnc_pair() and chunk_pages * int(page_size) > MOVING_MAX:
        call = call[2]
    return call(latent.contiguous(), w_key.contiguous(), w_value.contiguous(),
                page_ids.reshape(chunk_pages, 1).contiguous(), kv_pages, int(page_size),
                SOURCE_DIGEST, STAGING_DIGEST)


def mla_native_kv_attention(query: Tensor, kv_pages: Tensor, keep_bits: Tensor,
                            query_start: Tensor, softmax_scale: float, block_table_row: Tensor,
                            pool_rows: int, page_size: int = PAGE_ROWS,
                            superblock_pages: int = SUPERBLOCK_PAGES,
                            engine_balance: bool = False) -> Tensor:
    """Sparse MLA prefill attention over each query's selection, from KV pages.

    The operands are ``mla_dense_masked.mla_dense_masked_attention``'s with the query in
    place of the lifted query and the KV pages in place of the latent bank.

    Args:
        query: ``[S, 1, key_width]`` bf16/fp16, one head; ``S`` a multiple of 128.
        kv_pages: ``[pool pages, page_size, key_width + value_width]``, same dtype, already
            holding this step's tokens (:func:`mla_native_kv_write`).
        keep_bits: ``[S, P / 8]`` uint8, the keep table, ``P = pages * page_size /
            pool_rows``.
        query_start: ``[1, 1]`` int32, the window row of query 0; read on the device.
        softmax_scale: positive.
        block_table_row: ``[pages, 1]`` int32, the window's pages in order, ``-1`` pads.
        pool_rows: tokens per pool, a divisor of ``page_size``.
        page_size: tokens per page, :data:`PAGE_ROWS`.
        superblock_pages: pages per superblock.
        engine_balance: the engine-balanced stages (the module docstring); off by default.

    Returns:
        ``[S, 1, value_width]`` fp32, a bf16-K/V class answer (the module docstring).

    Raises:
        MlaNativeKvError: for a malformed call or a geometry the kernel does not serve.
    """
    if not softmax_scale > 0:
        raise MlaNativeKvError(f"softmax_scale must be positive; got {softmax_scale}")
    if query.ndim != 3 or keep_bits.ndim != 2 or block_table_row.ndim != 2:
        raise MlaNativeKvError(
            f"query must be [seq, 1, key_width], keep_bits [seq, pools / 8] and block_table_row "
            f"[pages, 1]; got {tuple(query.shape)}, {tuple(keep_bits.shape)} and "
            f"{tuple(block_table_row.shape)}")
    seq, heads, key_width = (int(d) for d in query.shape)
    pages = int(block_table_row.shape[0])
    if heads != 1:
        raise MlaNativeKvError(f"one head per call; got {heads}")
    value_width = int(kv_pages.shape[-1]) - key_width if kv_pages.ndim == 3 else 0
    for name, width in (("key_width", key_width), ("value_width", value_width)):
        refusal = _width_refusal(name, width)
        if refusal is not None:
            raise MlaNativeKvError(refusal)
    refusal = _kv_refusal(kv_pages, key_width, value_width, int(page_size), query.dtype)
    if refusal is None:
        # The selection geometry is the dense-masked kernel's; its latent checks see the
        # key width on both sides.
        refusal = _dm._geometry_refusal(seq, heads, key_width, key_width,
                                        int(kv_pages.shape[0]) * int(page_size), pages,
                                        query.dtype, kv_pages.dtype, int(pool_rows),
                                        int(page_size), int(superblock_pages))
    if refusal is not None:
        raise MlaNativeKvError(refusal)
    width = pages * int(page_size) // int(pool_rows) // _dm._KEEP_PLANES
    if keep_bits.dtype != torch.uint8 or tuple(keep_bits.shape) != (seq, width):
        raise MlaNativeKvError(
            f"keep_bits must be [seq, pools / 8] = [{seq}, {width}] uint8; got "
            f"{tuple(keep_bits.shape)} {keep_bits.dtype}")
    if tuple(query_start.shape) != (1, 1) or query_start.dtype != torch.int32:
        raise MlaNativeKvError(
            f"query_start must be [1, 1] int32; got {tuple(query_start.shape)} "
            f"{query_start.dtype}")
    call = wrap_nki(mla_native_kv_attention_kernel)
    if lnc_pair() and seq >= 2 * ROW_TILE:
        call = call[2]
    return call(
        query.contiguous(),
        kv_pages,
        block_table_row.contiguous().to(torch.int32),
        keep_bits.contiguous(),
        query_start.contiguous(),
        float(softmax_scale),
        int(page_size),
        int(pool_rows),
        int(superblock_pages),
        int(bool(engine_balance)),
        SOURCE_DIGEST,
        MASKED_DIGEST,
        WINDOW_DIGEST,
        STAGING_DIGEST,
    )
