# SPDX-License-Identifier: Apache-2.0
"""The DSA layer's prefill attention on the keep table: which chunks make one, and its reader.

Each query of a selecting prefill chunk gathers its own selected rows on the sparse path
(``mla_sparse.mla_sparse_attention``): 1,024 queries, one indirect gather per query. The
dense-masked pass (``functional/attention/mla_dense_masked.py``) reads the whole window
once instead, with each query's selection as a mask. It reads the selection as a keep
table, one bit per query and window pool. The indexer builds the table from its own top-k
ids, on the rank that selected those rows, and all-gathers it beside the ids
(``functional/dsa/selection_keep.py``, ``functional/dsa/indexer_shard.py``). The table
reaches the pass through the sparse attention seam.

Routing, at trace time, per prefill chunk, in this order (:func:`prefill_keep_table_route`):

1. a window the masked-window kernel serves keeps it (``masked_window_serves``), and the
   chunk makes no keep table;
2. a window the dense-masked pass serves (``dense_masked_serves``: its knob, its page cap and
   its geometry) takes that pass;
3. every other window keeps the gather, and the chunk makes no keep table.

Route 2 needs the indexer's candidates to be the window's pools, because the table's
columns are the indexer's score columns and the pass takes them as the window's pools in
order. The caller adds its own seam's conditions: no DCP, a full-width call, the selecting
regime. One answer decides both sides: the prefill forward asks once, the indexer makes the
keep table once for a chunk that the pass attends, and ``attend`` hands it to the pass.
Every knob is read at trace time, so a change needs a new compile.
"""

from __future__ import annotations

import enum

import torch

from vllm_neuron.functional.attention.mla_dense_masked import dense_masked_serves
from vllm_neuron.functional.attention.mla_dense_window import masked_window_serves


class KeepTableRoute(enum.Enum):
    """The attention that reads a prefill chunk's keep table."""

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
    the window's pools. The dense-masked test is the one its seam makes
    (``mla_sparse_attention``), which the full-width bank call passes: a NoPE paged call with
    the step's rows in the bank.
    """
    window_rows = int(window_pages) * int(page_size)
    if int(candidates) * int(pool_size) != window_rows:
        return None
    if masked_window_serves(int(rows), int(heads), int(latent), window_rows, int(pool_size),
                            int(index_cols), q_dtype, kv_dtype):
        return None
    if dense_masked_serves(int(rows), int(heads), int(latent), int(kv_latent), int(bank_rows),
                           int(window_pages), int(page_size), int(pool_size), q_dtype,
                           kv_dtype):
        return KeepTableRoute.DENSE_MASKED
    return None
