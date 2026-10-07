# SPDX-License-Identifier: Apache-2.0
"""The two prefill kernels exactly as commit 342e93e shipped them, beside HEAD.

:func:`load` reads each file below with ``git show 342e93e:<path>``, checks it against
the pinned blob id, and imports the result from a temporary directory outside the
worktree as a module of its own. Neither file imports the other or any other DSA / KDA
module, so no import is rewritten; every other import (``nki``, ``libtorch_neuronx_lite``,
``vllm_neuron.utils``) resolves to the live tree.

What it serves (the prefill entry points this change makes two-core):

* ``chunked_recurrence.kda_intra_chunk`` / ``kda_inter_chunk`` -- the KDA chunked
  recurrence, stages 1 to 3 and 4 to 5, one dispatch each per head per prefill call;
* ``kpool_hadamard.dsa_kpool_hadamard`` -- the DSA indexer key pooling and Hadamard
  rotation of a prefill (``Glm5NextDSAIndexer.pool_window``), and ``dsa_hadamard128``,
  the query rotation the same leg runs.

The snapshot is read rather than committed so the baseline is the commit itself and not a
hand-made copy of it; the blob ids make a silently different baseline impossible. The
layout is ``dsa_0a08ff4``'s.
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import tempfile
import types

COMMIT = "342e93e"

#: module name -> (repository path, git blob id at :data:`COMMIT`).
SOURCES: dict[str, tuple[str, str]] = {
    "chunked_recurrence": ("vllm_neuron/functional/kda/chunked_recurrence.py",
                           "54c833fbc5e188430102334db4fa4cfa45bcc076"),
    "kpool_hadamard": ("vllm_neuron/functional/dsa/kpool_hadamard.py",
                       "e3b1f94378ef31ae806863ea14a650c428e61d1b"),
}

PACKAGE = "prefill_cores_342e93e_snapshot"

_LOADED: types.SimpleNamespace | None = None


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_repo_root()), *args], check=True,
                          capture_output=True, text=True).stdout


def load() -> types.SimpleNamespace:
    """The snapshot modules by name (``chunked_recurrence``, ``kpool_hadamard``).

    Loaded once per process; later calls return the same modules.
    """
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    where = pathlib.Path(tempfile.mkdtemp(prefix="prefill_cores_342e93e_"))
    package = where / PACKAGE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Commit 342e93e, read by git show."""\n')
    for name, (path, blob) in SOURCES.items():
        found = _git("rev-parse", f"{COMMIT}:{path}").strip()
        if found != blob:
            raise RuntimeError(f"{COMMIT}:{path} is blob {found}, pinned {blob}")
        (package / f"{name}.py").write_text(_git("show", f"{COMMIT}:{path}"))
    sys.path.insert(0, str(where))
    modules = {name: importlib.import_module(f"{PACKAGE}.{name}") for name in SOURCES}
    _LOADED = types.SimpleNamespace(commit=COMMIT, directory=str(package), **modules)
    return _LOADED
