# SPDX-License-Identifier: Apache-2.0
"""MoE decode modules exactly as of commit 5938748, for before/after comparisons.

Every ``*.py`` here except this file and ``pipeline.py`` is
``git show 5938748:vllm_neuron/functional/moe/<name>`` with one edit: the two
``from vllm_neuron.functional.moe.rmsnorm_router_topk_tkg import`` lines in
``router.py`` became relative imports, so the snapshot router uses the snapshot
substrate. Shared helpers outside ``vllm_neuron/functional/moe`` are imported
from the tree. ``pipeline.py`` holds the 5938748 model-side composition.

Load with :func:`load` (the directory is not on ``sys.path``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

NAME = "moe_5938748"


def load(path: str | Path | None = None):
    """Import this directory as package ``moe_5938748`` and return its pipeline."""
    root = Path(path) if path is not None else Path(__file__).resolve().parent
    if NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            NAME, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[NAME] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{NAME}.pipeline")
