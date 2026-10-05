# SPDX-License-Identifier: Apache-2.0
"""The DSA/MLA decode code exactly as commit 5938748 shipped it, importable beside HEAD.

:func:`load` reads each file below with ``git show 5938748:<path>``, checks it against
the pinned blob id, rewrites the attention-kernel imports to point at the snapshot's own
copies, and imports the result from a temporary directory outside the worktree. Every
other import (configs, loaders, the DSA kernels this change does not touch) resolves to
the live tree.

The snapshot is read rather than committed so the baseline is the commit itself and not
a hand-made copy of it; the blob ids make a silently different baseline impossible.
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import tempfile
import types

COMMIT = "5938748"

#: module name -> (repository path, git blob id at :data:`COMMIT`).
SOURCES: dict[str, tuple[str, str]] = {
    "mla_projections": ("vllm_neuron/functional/attention/mla_projections.py",
                        "357ae9b70f5e00412a9b6be78b13934274932322"),
    "mla_sparse": ("vllm_neuron/functional/attention/mla_sparse.py",
                   "1dd9443bb66c5966483ef7a35335488cb87a87ea"),
    "mla_absorb": ("vllm_neuron/functional/attention/mla_absorb.py",
                   "9b2d700cefdbee5c9428d2b02b47e45529799e6a"),
    "model_fp8": ("vllm_neuron/model/glm5_next/model_fp8.py",
                  "73eebf97fa09e00e850a8e04d8a6af3785ab2b61"),
}

#: Live module paths the snapshot's own copies replace inside every snapshot file.
_REWRITES = {
    "vllm_neuron.functional.attention.mla_projections": "mla_projections",
    "vllm_neuron.functional.attention.mla_sparse": "mla_sparse",
    "vllm_neuron.functional.attention.mla_absorb": "mla_absorb",
}

PACKAGE = "dsa_5938748_snapshot"
_LOADED: types.SimpleNamespace | None = None


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_repo_root()), *args], check=True,
                          capture_output=True, text=True).stdout


def load() -> types.SimpleNamespace:
    """``SimpleNamespace(mla_projections=..., mla_sparse=..., mla_absorb=..., model_fp8=...)``.

    Loaded once per process; later calls return the same modules.
    """
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    where = pathlib.Path(tempfile.mkdtemp(prefix="dsa_5938748_"))
    package = where / PACKAGE
    package.mkdir()
    (package / "__init__.py").write_text('"""Commit 5938748, read by git show."""\n')
    for name, (path, blob) in SOURCES.items():
        found = _git("rev-parse", f"{COMMIT}:{path}").strip()
        if found != blob:
            raise RuntimeError(f"{COMMIT}:{path} is blob {found}, pinned {blob}")
        text = _git("show", f"{COMMIT}:{path}")
        for live, local in _REWRITES.items():
            text = text.replace(live, f"{PACKAGE}.{local}")
        (package / f"{name}.py").write_text(text)
    sys.path.insert(0, str(where))
    modules = {name: importlib.import_module(f"{PACKAGE}.{name}") for name in SOURCES}
    _LOADED = types.SimpleNamespace(commit=COMMIT, directory=str(package), **modules)
    return _LOADED
