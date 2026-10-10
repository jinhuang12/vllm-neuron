# SPDX-License-Identifier: Apache-2.0
"""``mla_decode.py`` exactly as commit ea04c81 shipped it, importable beside HEAD.

The one-row decode attention before the T-row form: :func:`load` reads the file with
``git show ea04c81:<path>``, checks it against the pinned blob id and imports it from a
temporary directory outside the worktree. The T-row tests pin ``T = 1`` to it bit for
bit, so the speculative form cannot move the served single-token step. The layout is
``dsa_5938748``'s.
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import tempfile
import types

COMMIT = "ea04c81"

#: module name -> (repository path, git blob id at :data:`COMMIT`).
SOURCES: dict[str, tuple[str, str]] = {
    "mla_decode": ("vllm_neuron/functional/attention/mla_decode.py",
                   "011fc1b7f5f0b5b52c5d8f374b51c303c9cb6ca1"),
}

PACKAGE = "mla_decode_ea04c81_snapshot"
_LOADED: types.SimpleNamespace | None = None


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_repo_root()), *args], check=True,
                          capture_output=True, text=True).stdout


def load() -> types.SimpleNamespace:
    """``SimpleNamespace(mla_decode=...)``. Loaded once per process."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    where = pathlib.Path(tempfile.mkdtemp(prefix="mla_decode_ea04c81_"))
    package = where / PACKAGE
    package.mkdir()
    (package / "__init__.py").write_text('"""Commit ea04c81, read by git show."""\n')
    for name, (path, blob) in SOURCES.items():
        found = _git("rev-parse", f"{COMMIT}:{path}").strip()
        if found != blob:
            raise RuntimeError(f"{COMMIT}:{path} is blob {found}, pinned {blob}")
        (package / f"{name}.py").write_text(_git("show", f"{COMMIT}:{path}"))
    sys.path.insert(0, str(where))
    modules = {name: importlib.import_module(f"{PACKAGE}.{name}") for name in SOURCES}
    _LOADED = types.SimpleNamespace(commit=COMMIT, directory=str(package), **modules)
    return _LOADED
