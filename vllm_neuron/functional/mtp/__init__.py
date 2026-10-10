# SPDX-License-Identifier: Apache-2.0
"""The MTP draft iteration's tail as authored NKI kernels.

* :mod:`.tail_in` (K1'): embedding gather, position-0 mask, ``enorm``/``hnorm``,
  concatenation and the row-sharded ``eh_proj`` GEMV -> this rank's slice of
  ``layer_input``.
* :mod:`.tail_out` (K2): residual add, ``shared_head.norm``, the vocab-shard logits
  GEMV and the local ``(max, argmax)`` pair the draft-token gather consumes.

Each module exposes ``<name>_kernel`` (the ``nki.jit`` kernel), ``<name>`` (the
dispatching entry point) and ``<name>_torch`` (the traced head's arithmetic, the CPU
route and the tests' reference), plus dispatch counters the tests read.
"""
