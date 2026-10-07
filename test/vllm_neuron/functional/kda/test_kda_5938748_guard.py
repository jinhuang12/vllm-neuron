# SPDX-License-Identifier: Apache-2.0
"""The 5938748 decode baseline's guard on the live ``chunked_recurrence``.

The snapshot ``decode_state.py`` and ``gate_clamp.py`` in
``test/hardware/baselines/kda_5938748`` import six names from the live
``chunked_recurrence`` (``loader.IMPORTED_HELPERS``). The loader refuses to load when any
of the six is defined differently from 5938748, and loads when only the rest of the file
changed -- this tree's prefill dispatch to ``chunked_lnc2`` is such a change.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from vllm_neuron.functional.kda import chunked_recurrence as chunked

BASELINE_DIR = (Path(__file__).resolve().parents[4]
                / "test" / "hardware" / "baselines" / "kda_5938748")


def _loader():
    spec = importlib.util.spec_from_file_location(
        "_kda_5938748_guard_loader", BASELINE_DIR / "loader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOADER = _loader()


def test_the_guard_covers_every_name_the_snapshots_import():
    import ast
    imported = set()
    for name in ("decode_state.py", "gate_clamp.py"):
        tree = ast.parse((BASELINE_DIR / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == (
                    "vllm_neuron.functional.kda.chunked_recurrence"):
                imported.update(alias.name for alias in node.names)
    assert imported == set(LOADER.IMPORTED_HELPERS)


def test_the_baseline_loads_against_a_live_file_changed_outside_the_helpers():
    live = hashlib.sha256(Path(chunked.__file__).read_bytes()).hexdigest()
    assert live != LOADER.SNAPSHOT_SHA256["chunked_recurrence.py"]
    LOADER.load_baseline(BASELINE_DIR)


@pytest.mark.parametrize("old,new,name", [
    ("    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)",
     "    return nl.ndarray((rows, cols), dtype=nl.bfloat16, buffer=nl.sbuf)", "_sbuf"),
    ("L2_NORM_EPS = 1e-6", "L2_NORM_EPS = 1e-5", "L2_NORM_EPS"),
    ("    nisa.nc_transpose(dst=ps, data=src)",
     "    nisa.nc_transpose(dst=ps, data=src)\n    nisa.tensor_copy(dst=ps, src=ps)",
     "_emit_transpose"),
])
def test_the_baseline_refuses_a_changed_helper(monkeypatch, tmp_path, old, new, name):
    text = Path(chunked.__file__).read_text()
    assert text.count(old) == 1
    changed = tmp_path / "chunked_recurrence.py"
    changed.write_text(text.replace(old, new))
    monkeypatch.setattr(chunked, "__file__", str(changed))
    with pytest.raises(ValueError, match=name):
        LOADER.load_baseline(BASELINE_DIR)
