# SPDX-License-Identifier: Apache-2.0
"""The DSA/MLA decode code exactly as commit 75090b9 shipped it, beside HEAD.

The three files beside this one are byte copies of ``git show 75090b9:<path>`` for the
paths in :data:`SOURCES`. :func:`load` checks each copy against the pinned git blob id
(computed from the bytes, as ``git hash-object`` does, so neither the 75090b9 object
nor a ``.git`` directory is needed), rewrites the attention-kernel imports to point at
the snapshot's own copies, and imports the result from a temporary directory outside the
worktree. Every other import (configs, loaders, the DSA kernels) resolves to the live
tree, where this change adds ``decode_batch.py`` and edits none of them.

75090b9 is not an ancestor of this branch (it lives on the
``baseline-dsa-75090b9`` tag), so the copies are committed: a clone of the branch alone,
or an exported tree, holds the baseline. The blob ids make a silently different baseline
impossible. The layout is ``dsa_5938748``'s.
"""

from __future__ import annotations

import hashlib
import importlib
import pathlib
import sys
import tempfile
import types

COMMIT = "75090b9"

#: module name -> (repository path, git blob id at :data:`COMMIT`). The byte copy of
#: each is ``<module name>.py`` beside this file.
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
HERE = pathlib.Path(__file__).resolve().parent
_LOADED: types.SimpleNamespace | None = None


def _blob_id(data: bytes) -> str:
    """The git blob id of ``data``: sha1 over ``blob <size>\\0`` and the bytes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


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
    (package / "__init__.py").write_text('"""Commit 75090b9, from the committed copies."""\n')
    for name, (path, blob) in SOURCES.items():
        copy = HERE / f"{name}.py"
        data = copy.read_bytes()
        found = _blob_id(data)
        if found != blob:
            raise RuntimeError(f"{copy} is blob {found}, not {COMMIT}:{path} ({blob})")
        text = data.decode()
        for live, local in _REWRITES.items():
            text = text.replace(live, f"{PACKAGE}.{local}")
        (package / f"{name}.py").write_text(text)
    sys.path.insert(0, str(where))
    modules = {name: importlib.import_module(f"{PACKAGE}.{name}") for name in SOURCES}
    _LOADED = types.SimpleNamespace(commit=COMMIT, directory=str(package), **modules)
    return _LOADED
