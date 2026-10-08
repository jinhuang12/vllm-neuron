# SPDX-License-Identifier: Apache-2.0
"""The DSA layer's prefill short regime: when selection keeps every token, attend densely.

The indexer keeps the ``select_k = index_topk // index_kpool`` highest-scoring complete
pools and appends the open tail. A query row whose causal length is at most
``select_k * index_kpool + index_kpool - 1`` (2051 tokens on this checkpoint) has at
most ``select_k`` complete pools, so all of them are kept and its selected set is its
whole causal prefix. ``decode_bypass.py`` states the bound for decode; this module
applies the same bound to a prefill chunk.

What the short regime replaces, per DSA layer and chunk: the indexer's query side
(``wq_b``, the Hadamard rotation of ``T x 32`` rows, ``weights_proj``), its scoring,
causal bound, top-k, sentinel order and expansion, and the sparse kernel's per-query
gathers of 2176 index columns -- for one dense pass over the window
(``functional/attention/mla_dense_window.py``). The indexer's write stage (key
projection, pooling, the pooled-key store and the ring) still runs, because later
chunks and decode read it.

The decision is a trace-time constant: a prefill graph is captured once per bucket,
and the chunk's start and end positions travel as tensors so the graph is not pinned
to one prompt. The bound a graph can prove is the one ``decode_bypass.selection_bound``
already takes, ``min(max_seq_len, window rows)``: every row this graph serves sits
inside the window its block table names, and the runner refuses a prompt longer than
that window. It is therefore decided per chunk from the context the chunk's graph can
hold (its cached prefix bucket plus the chunk), never from ``max_model_len`` alone.

Kill switch: ``VLLM_NEURON_MLA_DENSE_WINDOW=0`` (``vllm_neuron/envs.py``) restores the
sparse path. It is read at trace time, so a change needs a new compile.
"""

from __future__ import annotations

import torch
from torch import Tensor

from vllm_neuron import envs
from vllm_neuron.functional.attention.mla_dense_window import (
    dense_window_serves,
    mla_dense_window_attention,
)
from vllm_neuron.functional.dsa.decode_bypass import (
    selection_bound,
    selection_is_a_no_op,
)


def prefill_takes_dense_window(*, is_decode: bool, max_seq_len: int, window_rows: int,
                               index_topk: int, index_kpool: int,
                               hidden_dtype: torch.dtype, cache_dtype: torch.dtype,
                               heads: int, latent: int) -> bool:
    """True when this prefill chunk's selection is a no-op and the dense kernel serves it.

    False when the switch is off, on a decode step, when the bound this graph can prove
    exceeds the identity bound, or when the dense kernel does not serve the geometry (a
    2-byte query and cache of one dtype, a 128-multiple latent, a window of at most
    ``MAX_KEY_ROWS`` rows). ``heads`` is an upper bound on the heads per rank. Every
    input is a python value or a dtype, so the answer is fixed when the graph is traced.
    """
    if not envs.VLLM_NEURON_MLA_DENSE_WINDOW or is_decode:
        return False
    bound = selection_bound(int(max_seq_len), int(window_rows))
    if not selection_is_a_no_op(bound, int(index_topk), int(index_kpool)):
        return False
    # The kernel stages the whole window, as the sparse kernel does, so the window and
    # not the bound sizes its working set.
    return dense_window_serves(hidden_dtype, cache_dtype, int(heads), int(latent),
                               int(window_rows))


def attend_dense_window(q_lift: Tensor, c_kv: Tensor, seq_lens: Tensor,
                        softmax_scale: float, *, block_table_row: Tensor,
                        written: Tensor, write_offset: Tensor, page_size: int,
                        active_rows: int | None) -> Tensor:
    """The short regime's attention: ``[tokens, H, L]`` float32, as the sparse call returns.

    ``active_rows`` is the sparse branch's own contract: only that prefix of query rows is
    attended and the rest are zero rows. The kernel writes those rows itself, so a padded
    chunk adds no op here and keeps the kernel's two-program launch.
    """
    return mla_dense_window_attention(
        q_lift, c_kv, seq_lens, float(softmax_scale),
        block_table_row=block_table_row, written=written, write_offset=write_offset,
        page_size=int(page_size), active_rows=active_rows,
    )
