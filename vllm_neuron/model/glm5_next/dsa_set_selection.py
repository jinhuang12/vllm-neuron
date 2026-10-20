# SPDX-License-Identifier: Apache-2.0
"""The DSA layer's prefill selection as a set, where only the masked-window attention reads it.

The indexer's selection ends in two steps that fix an order within each row: the top-k
kernel's final sort of its ``select_k`` winners, highest first, and the sentinel order
(``functional/dsa/sentinel_order.py``), which moves the row's ``-1`` sentinels behind its
real pool ids. Neither adds or removes an id: the sort ranks the winners the stages chose,
and the sentinel order is a stable partition. The gather attention (``mla_sparse.py``) and
the DCP merge read the order. The masked-window attention
(``mla_dense_window.mla_masked_window_attention``) does not: it reads the last column of
each pool's group of ``dsa_index_expand`` columns, counts each group's pool id on the PE with
exact integer sums, counts nothing for a ``-1`` group, and finds the open pool from the
row's largest index. Its output is a function of each row's set of pool ids.

So where the masked-window kernel is the selection's one reader, both steps are skipped:
the top-k runs with ``ordered=False`` and the sentinel order is not called. The decision is
the attention seam's own (``mla_dense_window.masked_window_serves``), asked with the operands
the seam will be handed, so a selection that the gather reads keeps its order;
``Glm5NextMLAAttention.attend`` asks the seam again with the real operands and refuses a
set that the seam would not hand to the masked-window kernel. A forward that collects the
layer streams (``VLLM_NEURON_DUMP_LAYER_STREAMS``) keeps the order too, because the dump
hands the index rows on as the indexer emitted them.

Kill switch: ``VLLM_NEURON_DSA_SET_SELECTION=0`` (``vllm_neuron/envs.py``) keeps the ordered
selection on every route. It is read at trace time, so a change needs a new compile.
"""

from __future__ import annotations

import torch

from vllm_neuron import envs
from vllm_neuron.functional.attention.mla_dense_window import masked_window_serves


def prefill_selects_a_set(*, rows: int, heads: int, latent: int, window_rows: int,
                          pool_size: int, index_cols: int, q_dtype: torch.dtype,
                          kv_dtype: torch.dtype) -> bool:
    """True when this prefill chunk's selection may come back as a set.

    The arguments are the operands the attention seam (``mla_sparse_attention``) will see:
    the query rows it attends, the heads and latent width of ``q_lift``, the window that the
    block table names, the indexer's pool size, the width of the expanded selection, and the
    query and cache dtypes. Every input is a python value or a dtype, so the answer is fixed
    when the graph is traced. False when the switch is off.
    """
    if not envs.VLLM_NEURON_DSA_SET_SELECTION:
        return False
    return masked_window_serves(int(rows), int(heads), int(latent), int(window_rows),
                                int(pool_size), int(index_cols), q_dtype, kv_dtype)
