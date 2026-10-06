# SPDX-License-Identifier: Apache-2.0
"""The DSA/MLA decode code exactly as commit 75090b9 (wt/dsa) shipped it, beside HEAD.

:func:`load` reads each file below with ``git show 75090b9:<path>``, checks it against
the pinned blob id, rewrites the attention-kernel imports to point at the snapshot's own
copies, and imports the result from a temporary directory outside the worktree. Every
other import (configs, loaders, the DSA kernels) resolves to the live tree, where this
change adds ``decode_batch.py`` and edits none of them.

The snapshot is read rather than committed so the baseline is the commit itself and not
a hand-made copy of it; the blob ids make a silently different baseline impossible. The
layout is ``dsa_5938748``'s.
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import tempfile
import types

COMMIT = "75090b9"

#: module name -> (repository path, git blob id at :data:`COMMIT`).
SOURCES: dict[str, tuple[str, str]] = {
    "mla_sparse": ("vllm_neuron/functional/attention/mla_sparse.py",
                   "1dd9443bb66c5966483ef7a35335488cb87a87ea"),
    "mla_decode": ("vllm_neuron/functional/attention/mla_decode.py",
                   "011fc1b7f5f0b5b52c5d8f374b51c303c9cb6ca1"),
    "model_fp8": ("vllm_neuron/model/glm5_next/model_fp8.py",
                  "4bd6f3ca9b1044142d712be6f4fc7898bf86e806"),
}

#: Live module paths the snapshot's own copies replace inside every snapshot file.
_REWRITES = {
    "vllm_neuron.functional.attention.mla_sparse": "mla_sparse",
    "vllm_neuron.functional.attention.mla_decode": "mla_decode",
}

PACKAGE = "dsa_75090b9_snapshot"
_LOADED: types.SimpleNamespace | None = None


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_repo_root()), *args], check=True,
                          capture_output=True, text=True).stdout


def load() -> types.SimpleNamespace:
    """``SimpleNamespace(mla_sparse=..., mla_decode=..., model_fp8=...)``.

    Loaded once per process; later calls return the same modules.
    """
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    where = pathlib.Path(tempfile.mkdtemp(prefix="dsa_75090b9_"))
    package = where / PACKAGE
    package.mkdir()
    (package / "__init__.py").write_text('"""Commit 75090b9, read by git show."""\n')
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
