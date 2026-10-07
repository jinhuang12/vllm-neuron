# SPDX-License-Identifier: Apache-2.0
"""Small fused kernels that replace the XLA glue between the decode-path NKI kernels.

Each module here takes one run of elementwise and small-matmul torch ops that the
decode profile charges to "compiler ops" and does it in one NKI launch, on the
operands' served dtypes, so the per-step casts and transposes the torch spelling
implies are not issued at all. Every entry point has an ``*_admits`` predicate; a
call site keeps its 0a08ff4 torch route whenever the predicate says no, and
``VLLM_NEURON_GLUE_FUSED=0`` says no everywhere.
"""

import os

#: The switch for every kernel in ``functional/glue``; ``0`` keeps 0a08ff4's routes.
GLUE_FUSED_ENV = "VLLM_NEURON_GLUE_FUSED"


def glue_fused_enabled() -> bool:
    """``VLLM_NEURON_GLUE_FUSED`` is not ``0``."""
    return os.environ.get(GLUE_FUSED_ENV, "1") != "0"
