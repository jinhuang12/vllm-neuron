# SPDX-License-Identifier: Apache-2.0
"""The 5938748 decode snapshots run on the 5938748 chunked-recurrence helpers.

``decode_state.py`` and ``gate_clamp.py`` in ``test/hardware/baselines/kda_5938748``
import emit helpers by name from ``vllm_neuron.functional.kda.chunked_recurrence``.
``load_baseline`` binds that name to the snapshot of that file while the two files
execute. Each case replaces every name a snapshot imports with a sentinel in the
live module, so the live module differs in exactly the names that matter. Then it
checks that the snapshot holds the snapshot module's object for each name. The
names come from the snapshots' own import statements.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
BASELINE_DIR = REPO / "test" / "hardware" / "baselines" / "kda_5938748"

#: The snapshots that import from the chunked-recurrence module.
IMPORTERS = ("decode_state.py", "gate_clamp.py")


def _loader():
    spec = importlib.util.spec_from_file_location(
        "_kda_5938748_loader_for_helpers", BASELINE_DIR / "loader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOADER = _loader()


def _imported_names(filename: str) -> tuple[str, ...]:
    """The names ``filename`` imports from the chunked-recurrence module."""
    tree = ast.parse((BASELINE_DIR / filename).read_text())
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == LOADER.CHUNKED_MODULE:
            names.extend(alias.name for alias in node.names)
    return tuple(names)


@pytest.fixture
def restored_modules():
    """Put back every ``sys.modules`` entry the case replaces.

    Another file's loaded baseline stays registered under its own names.
    """
    saved = dict(sys.modules)
    yield
    for name, module in saved.items():
        if sys.modules.get(name) is not module:
            sys.modules[name] = module


@pytest.mark.parametrize("filename", IMPORTERS)
def test_the_snapshot_takes_the_snapshot_helpers_when_the_live_module_differs(
    filename, monkeypatch, restored_modules
):
    names = _imported_names(filename)
    assert names, f"{filename} imports nothing from {LOADER.CHUNKED_MODULE}"
    live = importlib.import_module(LOADER.CHUNKED_MODULE)
    sentinel = object()
    for name in names:
        monkeypatch.setattr(live, name, sentinel)

    baseline = LOADER.load_baseline(BASELINE_DIR)

    module = getattr(baseline, Path(filename).stem)
    for name in names:
        assert getattr(module, name) is not sentinel, (
            f"{filename} took {name} from the live module"
        )
    snapshot = baseline.chunked_recurrence
    assert Path(inspect.getsourcefile(snapshot)) == BASELINE_DIR / "chunked_recurrence.py"
    for name in names:
        assert getattr(module, name) is getattr(snapshot, name), (
            f"{filename}: {name} is not the snapshot's"
        )
    assert sys.modules[LOADER.CHUNKED_MODULE] is live
