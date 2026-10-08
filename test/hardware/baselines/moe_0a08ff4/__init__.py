# SPDX-License-Identifier: Apache-2.0
"""The routed-expert decode kernel exactly as of commit 0a08ff4, for before/after.

``expert_decode.py`` and ``fused_fp8_pack.py`` are
``git show 0a08ff4:vllm_neuron/functional/moe/<name>``, unedited (the kernel's
one relative import, ``.fused_fp8_pack``, resolves to the copy next to it).
``pipeline.py`` is the 0a08ff4 entry point ``fused_fp8.fused_fp8_decode_experts``
without the seam's dispatch counter, so a test can count the new route alone.

Load with :func:`load` (the directory is not on ``sys.path``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

NAME = "moe_0a08ff4"


def load(path: str | Path | None = None):
    """Import this directory as package ``moe_0a08ff4`` and return its pipeline."""
    root = Path(path) if path is not None else Path(__file__).resolve().parent
    if NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            NAME, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[NAME] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{NAME}.pipeline")
