# SPDX-License-Identifier: Apache-2.0
"""The DSA layer's prefill attention on the keep table: the block-shared gather, where the knob
routes it, and the table's other reader, the dense-masked pass.

Today each query of a selecting prefill chunk gathers its own selected rows
(``mla_sparse.mla_sparse_attention``): 1,024 queries, 2,048 rows each, one indirect
gather per query. The block-shared kernel (``functional/attention/mla_block_shared.py``)
gathers once per block of 128 neighbouring queries, the union of their selections, and
masks each query to its own rows. Each query's selection is unchanged, so the result is
value-equal to today's.

The kernel reads the selection as a keep table, one bit per query and window pool. The
indexer builds it from its own top-k ids, on the rank that selected those rows, and
all-gathers it beside the ids (``functional/dsa/selection_keep.py``,
``functional/dsa/indexer_shard.py``). A block whose shared pools overflow the union cap
gets today's per-query gather, from ``topk_indices``, in a second kernel
(``functional/attention/mla_block_shared_exact.py``). It runs after the block-shared
kernel on every call and rewrites in place the rows of each such block and of the block
that shares its round; nothing here branches on a device value. The dense-masked pass
(``functional/attention/mla_dense_masked.py``) reads the same table as its mask, through
the sparse attention seam.

Routing, at trace time, per prefill chunk, in this order (:func:`prefill_keep_table_route`):

1. a window the masked-window kernel serves keeps it (``masked_window_serves``), and the
   chunk makes no keep table;
2. with ``VLLM_NEURON_MLA_BLOCK_SHARED`` on, a window of at least
   ``VLLM_NEURON_MLA_BLOCK_SHARED_MIN_PAGES`` pages that the kernel serves
   (``mla_block_shared_serves``) takes the block-shared kernel;
3. a window the dense-masked pass serves (``dense_masked_serves``: its knob, its page cap and
   its geometry) takes that pass;
4. every other window keeps today's kernel, and the chunk makes no keep table.

Routes 2 and 3 both need the indexer's candidates to be the window's pools, because the
table's columns are the indexer's score columns and both readers take them as the window's
pools in order. The caller adds its own seam's conditions: no DCP, a full-width call, the
selecting regime. One answer decides both sides: the prefill forward asks once, the indexer
makes the keep table once for a chunk that either route attends, and ``attend`` hands it to
that route. Every knob is read at trace time, so a change needs a new compile.
"""

from __future__ import annotations

import enum

import torch
from torch import Tensor

from vllm_neuron import envs
from vllm_neuron.functional.attention.mla_block_shared import (
    mla_block_shared_attention,
    mla_block_shared_serves,
    mla_block_shared_union,
    mla_block_shared_union_cap,
)
from vllm_neuron.functional.attention.mla_block_shared_exact import mla_block_shared_exact
from vllm_neuron.functional.attention.mla_dense_masked import dense_masked_serves
from vllm_neuron.functional.attention.mla_dense_window import masked_window_serves


class KeepTableRoute(enum.Enum):
    """The attention that reads a prefill chunk's keep table."""

    #: ``attend_block_shared``: the union gather per 128-query block.
    BLOCK_SHARED = "block_shared"
    #: ``mla_sparse_attention`` with the table: its seam hands the call to the dense pass.
    DENSE_MASKED = "dense_masked"


def prefill_keep_table_route(*, rows: int, heads: int, latent: int, kv_latent: int,
                             bank_rows: int, window_pages: int, page_size: int,
                             pool_size: int, candidates: int, index_cols: int,
                             q_dtype: torch.dtype,
                             kv_dtype: torch.dtype) -> KeepTableRoute | None:
    """The route that reads this prefill chunk's keep table, or None for a chunk that needs none.

    The arguments are the operands the attention seam will see: the query rows it attends,
    the heads and latent width of ``q_lift``, the latent width and row count of the bank, the
    window's pages and page size, the indexer's pool size and its candidate pool count, the
    width of the expanded selection, and the query and cache dtypes. Every input is a python
    value or a dtype, so the answer is fixed when the graph is traced. ``candidates`` must be
    the window's pools for either route. The dense-masked test is the one its seam makes
    (``mla_sparse_attention``), which the full-width bank call passes: a NoPE paged call with
    the step's rows in the bank.
    """
    window_rows = int(window_pages) * int(page_size)
    if int(candidates) * int(pool_size) != window_rows:
        return None
    if masked_window_serves(int(rows), int(heads), int(latent), window_rows, int(pool_size),
                            int(index_cols), q_dtype, kv_dtype):
        return None
    if (
        envs.VLLM_NEURON_MLA_BLOCK_SHARED
        and int(window_pages) >= envs.VLLM_NEURON_MLA_BLOCK_SHARED_MIN_PAGES
        and mla_block_shared_serves(int(rows), int(heads), int(latent), int(window_pages),
                                    int(page_size), int(pool_size), q_dtype, kv_dtype)
    ):
        return KeepTableRoute.BLOCK_SHARED
    if dense_masked_serves(int(rows), int(heads), int(latent), int(kv_latent), int(bank_rows),
                           int(window_pages), int(page_size), int(pool_size), q_dtype,
                           kv_dtype):
        return KeepTableRoute.DENSE_MASKED
    return None


def attend_block_shared(q_lift: Tensor, c_kv: Tensor, keep_bits: Tensor, topk_indices: Tensor,
                        softmax_scale: float, *, block_table_row: Tensor, query_start: Tensor,
                        page_size: int, pool_size: int) -> Tensor:
    """The chunk's attention on the block-shared kernel: ``[S, 1, latent]`` float32.

    Three kernels run, in this order, on every call: the union build, the block-shared
    attention over every block, and the exact kernel. The exact kernel reads the union
    build's overflow flags and takes the blocks in rounds of one block per program; for
    each round that holds a flagged block it rewrites the round's rows in place with
    today's per-query gather. A call with no flag keeps the block-shared rows. Every
    call launches all three, so the graph does not depend on a device value.

    Args:
        q_lift: ``[S, 1, latent]``, the absorbed queries of the chunk.
        c_kv: ``[slots, latent]``, the whole latent bank; it already holds this step's rows.
        keep_bits: ``[S, P / 8]`` uint8, the chunk's keep table from the indexer.
        topk_indices: ``[S, columns]`` int32, the expanded selection; the exact kernel
            reads it only for a block whose union overflows its cap.
        softmax_scale: the attention's scale.
        block_table_row: ``[pages, 1]`` int32, the window's pages.
        query_start: ``[1, 1]`` int32, the window row of the chunk's first query.
        page_size: rows per page.
        pool_size: rows per pool of the indexer.

    The union cap is the kernel's for the window (``mla_block_shared_union_cap``):
    :func:`prefill_keep_table_route` has already proven that one exists.
    """
    union_cap = mla_block_shared_union_cap(int(block_table_row.shape[0]))
    union_rows, member, totals = mla_block_shared_union(keep_bits, query_start, union_cap,
                                                        int(pool_size))
    out = mla_block_shared_attention(q_lift, c_kv, union_rows, member, softmax_scale,
                                     block_table_row, page_size=int(page_size))
    return mla_block_shared_exact(q_lift, c_kv, topk_indices, totals, out, softmax_scale,
                                  block_table_row, page_size=int(page_size))
