# SPDX-License-Identifier: Apache-2.0
"""The decode glue's code exactly as commit 0a08ff4 shipped it, beside HEAD.

:func:`load` reads each file below with ``git show 0a08ff4:<path>``, checks it against
the pinned blob id, rewrites the imports of the other snapshot files to point at the
snapshot's own copies, and imports the result from a temporary directory outside the
worktree. Every other import (configs, loaders, the kernels this branch does not edit)
resolves to the live tree.

The files are the ones this branch edits: the model file (call sites only) and the
kernel modules whose prologue or epilogue it changes. The layout is ``dsa_75090b9``'s.
The module names carry the snapshot package, so the compile cache, which keys a graph by
the kernel's module-qualified name, never serves a snapshot kernel's NEFF to the live
tree or the reverse.
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import tempfile
import types

COMMIT = "0a08ff4"

#: module name -> (repository path, git blob id at :data:`COMMIT`).
SOURCES: dict[str, tuple[str, str]] = {
    "model_fp8": ("vllm_neuron/model/glm5_next/model_fp8.py",
                  "6d21beaea7f956cd4db27d1d22d6a8bd647de774"),
    "hyper_connection": ("vllm_neuron/functional/mhc/hyper_connection.py",
                         "4f8d198da0acff1660d66e0e5c21996c5073024b"),
    "sinkhorn": ("vllm_neuron/functional/mhc/sinkhorn.py",
                 "5c04e67a6f141108cf92bb6350c1ba3dd0a7e4ae"),
    "fused_decode": ("vllm_neuron/functional/kda/fused_decode.py",
                     "1f0715e4a312768671d05048e77838d571093605"),
    "mla_projections": ("vllm_neuron/functional/attention/mla_projections.py",
                        "a4193389dbdc1b7352621c70f35f41f01a19f844"),
}

#: Live module paths the snapshot's own copies replace inside every snapshot file.
_REWRITES = {
    "vllm_neuron.functional.mhc.hyper_connection": "hyper_connection",
    "vllm_neuron.functional.mhc.sinkhorn": "sinkhorn",
    "vllm_neuron.functional.kda.fused_decode": "fused_decode",
    "vllm_neuron.functional.attention.mla_projections": "mla_projections",
}

PACKAGE = "glue_0a08ff4_snapshot"
_LOADED: types.SimpleNamespace | None = None


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_repo_root()), *args], check=True,
                          capture_output=True, text=True).stdout


def load() -> types.SimpleNamespace:
    """``SimpleNamespace(model_fp8=..., hyper_connection=..., sinkhorn=..., ...)``.

    Loaded once per process; later calls return the same modules.
    """
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    where = pathlib.Path(tempfile.mkdtemp(prefix="glue_0a08ff4_"))
    package = where / PACKAGE
    package.mkdir()
    (package / "__init__.py").write_text('"""Commit 0a08ff4, read by git show."""\n')
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
