"""Collection-wide quarantine for the vendored upstream test files.

Every file in this directory is vendored **verbatim** from an upstream vLLM pull
request, ahead of the code it exercises. ``test/OVERLAY.md`` pins each file's
origin PR, upstream path and sha256 at vendoring time, so a later rebase can
prove whether the upstream original moved.

Quarantine gates **selection**, never **import**, and that distinction is the
whole point of applying the marker here rather than inside the files:

1. **The vendored bodies stay byte-identical to upstream.** A module-level
   ``pytestmark`` would mean editing a vendored body -- exactly the drift the
   sha256 rows in ``OVERLAY.md`` exist to detect. Re-targeting a vendored file's
   imports belongs to the increment that lands its target, together with the
   removal of its quarantine.
2. **A module-level skip would make the collected count vacuous.**
   ``pytest.skip(allow_module_level=True)`` or ``importorskip`` gates the
   *import*, so the module contributes **zero** collected functions. The
   measurement these files are vendored under is that each module imports in
   full and yields its own test-function count while selecting nothing -- a
   count that only exists if the import really happens.

``pytest_itemcollected`` fires per item as collection produces it, which is
strictly before ``-m`` expressions are evaluated against the collected set. So
every item here carries ``quarantined`` by the time any selection happens,
independently of plugin hook ordering, while the modules themselves are imported
in full.

The marker is registered once, in ``pyproject.toml``; its meaning lives in
``test/OVERLAY.md``.
"""

from __future__ import annotations

import pytest

#: Applied to every item collected from this directory. Vendored-ahead-of-target
#: tests are collected so their imports are proved, and skipped so they cannot
#: report green over code that does not exist yet.
QUARANTINED = pytest.mark.quarantined


def pytest_itemcollected(item: pytest.Item) -> None:
    """Mark every item collected under ``test/vllm_neuron/upstream/``."""
    item.add_marker(QUARANTINED)
