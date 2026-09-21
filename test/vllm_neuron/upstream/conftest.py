"""Collection rules for the test files vendored from upstream vLLM.

Every file here is a copy of an upstream test, taken ahead of the code it
exercises, and is never edited -- ``test/OVERLAY.md`` records each file's origin
and sha256 so an edit cannot pass unnoticed. Two rules follow from that:

* A file whose upstream target is not installed is not collected. Importing it
  would be a collection error, and adding an import guard to the body would
  break its recorded digest.
* Whatever does collect is skipped. These tests assert upstream behaviour that
  this platform does not implement, so running them proves nothing.
"""

from __future__ import annotations

import importlib.util

import pytest

#: Upstream module each vendored file imports at module scope and the installed
#: vLLM may not provide. A file is left uncollected while its module is absent.
UPSTREAM_TARGETS = {
    "test_flashinfer_mla_sparse_sm90.py": (
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90",
    ),
    "test_glm5next.py": ("vllm.transformers_utils.processors.glm5next",),
    "test_kpool_tail_slot_mapping.py": ("vllm.v1.kv_cache_layout",),
    "test_sparse_indexer_decode_seq_lens.py": (
        "vllm.models.glm5next",
        "vllm.model_executor.layers.sparse_attn_indexer_kpool",
    ),
}

QUARANTINED = pytest.mark.quarantined
SKIP = pytest.mark.skip(
    reason="vendored from upstream vLLM ahead of the code it exercises"
)


def _installed(module: str) -> bool:
    """Whether ``module`` can be located without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


collect_ignore = [
    name
    for name, modules in UPSTREAM_TARGETS.items()
    if not all(_installed(module) for module in modules)
]


def pytest_itemcollected(item: pytest.Item) -> None:
    """Mark and skip every item collected from this directory."""
    item.add_marker(QUARANTINED)
    item.add_marker(SKIP)
