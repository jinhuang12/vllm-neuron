# SPDX-License-Identifier: Apache-2.0
"""The DSA prefill key pooling reads its sliding windows from the keys in place.

The prefill indexer pools a window at every position of the chunk: position ``p``'s pool is
the ``index_kpool`` keys and gate scores ending at ``p``, clamped at row 0
(``kpool_hadamard.sliding_windows``). The gather route builds those windows as copies,
``key[windows]`` and ``gate_score[windows]``, and hands them to ``dsa_kpool_hadamard``. The
window route hands it ``key`` and ``gate_score`` with ``windows=True``, and its kernel reads
each window in place into the tiles the gather route's kernel loads, so the pooled keys are
the same bits and the gather is gone.

Kill switch: ``VLLM_NEURON_DSA_KPOOL_WINDOW_READ=0`` (``vllm_neuron/envs.py``) keeps the
gather route. It is read at trace time, so a change needs a new compile.
"""

from __future__ import annotations

from vllm_neuron import envs


def prefill_reads_windows() -> bool:
    """Whether the prefill key pooling takes the window route
    (``envs.VLLM_NEURON_DSA_KPOOL_WINDOW_READ``)."""
    return bool(envs.VLLM_NEURON_DSA_KPOOL_WINDOW_READ)
