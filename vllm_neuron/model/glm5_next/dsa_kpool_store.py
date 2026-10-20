# SPDX-License-Identifier: Apache-2.0
"""The DSA prefill key pooling writes its pooled keys into the pool cache itself.

The prefill indexer pools a window at every position of the chunk and writes the pools that
complete into the layer's pool cache, every other position into the cache's trash row. The
scatter route has the pooling kernel store the pooled keys, and the graph loads them back and
scatters them with ``index_copy_``. The store route hands the cache and each position's row to
``kpool_hadamard.dsa_kpool_hadamard_store``, whose kernel writes each pooled row straight to
its row by indirect DMA, so the pooled keys never go to HBM and back. The written rows are the
same bits.

The store route is the window route's kernel with the write added, so it runs only where the
window route does (``dsa_kpool_window``).

Kill switch: ``VLLM_NEURON_DSA_KPOOL_STORE=0`` (``vllm_neuron/envs.py``) keeps the scatter
route. It is read at trace time, so a change needs a new compile.
"""

from __future__ import annotations

from vllm_neuron import envs
from vllm_neuron.model.glm5_next import dsa_kpool_window


def prefill_stores_pools() -> bool:
    """Whether the prefill key pooling takes the store route
    (``envs.VLLM_NEURON_DSA_KPOOL_STORE`` on the window route)."""
    return bool(envs.VLLM_NEURON_DSA_KPOOL_STORE) and dsa_kpool_window.prefill_reads_windows()
