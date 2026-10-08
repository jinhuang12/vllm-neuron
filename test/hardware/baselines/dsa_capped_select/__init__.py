# SPDX-License-Identifier: Apache-2.0
"""The DSA decode indexer chain exactly as commit e3f38f8 (b17526a + dsa8k) shipped it.

:func:`load` reads each file below with ``git show e3f38f8:<path>``, checks it against the
pinned blob id, rewrites the imports between these files to point at the snapshot's own
copies, and imports the result from a temporary directory outside the worktree. Every
other import (configs, loaders, attention kernels, the DSA kernels this change does not
edit) resolves to the live tree. The layout is ``dsa_four_kernel_select``'s.

What it serves: :func:`chain` is e3f38f8's decode indexer chain for ``B`` requests from
the indexer's projections on -- the query rotation (``dsa_hadamard128``) and then
``Glm5NextDSAIndexer.forward_requests`` with those projections handed in: the ring step,
the batched scores, the one selection kernel (``decode_select``), and the two bank writes.

At e3f38f8 the score stage refuses more than ``decode_batch.MAX_CANDIDATES`` (16384)
candidates and the selection sends more than ``decode_select.MAX_SELECT_CANDIDATES``
(16384) to its torch oracle; this snapshot keeps both, so a caller sees what e3f38f8 does.
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import tempfile
import types

COMMIT = "e3f38f8"

#: module name -> (repository path, git blob id at :data:`COMMIT`).
SOURCES: dict[str, tuple[str, str]] = {
    "kpool_hadamard": ("vllm_neuron/functional/dsa/kpool_hadamard.py",
                       "e3b1f94378ef31ae806863ea14a650c428e61d1b"),
    "decode_batch": ("vllm_neuron/functional/dsa/decode_batch.py",
                     "34c970ff3613dc5746229dcea10d3f7bf6ca6696"),
    "decode_select": ("vllm_neuron/functional/dsa/decode_select.py",
                      "971fe17fe515fa34f039db19493dcd95df68996f"),
    "model_fp8": ("vllm_neuron/model/glm5_next/model_fp8.py",
                  "9f0f372d12177ba88320fd87326528528d6ad13e"),
    "bucket_utils": ("vllm_neuron/utils/bucket_utils.py",
                     "8a7d01c0cce30e92bd3bfac03599523d76ee2413"),
}

PACKAGE = "dsa_capped_select_snapshot"

#: Live module paths the snapshot's own copies replace inside every snapshot file. Each
#: is a full dotted module path, so no rewrite is a prefix of a module kept live.
_REWRITES = {
    **{f"vllm_neuron.functional.dsa.{name}": f"{PACKAGE}.{name}"
       for name in ("kpool_hadamard", "decode_batch", "decode_select")},
    "vllm_neuron.functional.dsa import kpool_hadamard":
        f"{PACKAGE} import kpool_hadamard",
}

_LOADED: types.SimpleNamespace | None = None


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_repo_root()), *args], check=True,
                          capture_output=True, text=True).stdout


def load() -> types.SimpleNamespace:
    """The snapshot modules by name (``kpool_hadamard``, ``decode_batch``,
    ``decode_select``, ``model_fp8``, ``bucket_utils``). Loaded once per process."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    where = pathlib.Path(tempfile.mkdtemp(prefix="dsa_capped_select_"))
    package = where / PACKAGE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Commit e3f38f8, read by git show."""\n')
    for name, (path, blob) in SOURCES.items():
        found = _git("rev-parse", f"{COMMIT}:{path}").strip()
        if found != blob:
            raise RuntimeError(f"{COMMIT}:{path} is blob {found}, pinned {blob}")
        text = _git("show", f"{COMMIT}:{path}")
        for live, local in _REWRITES.items():
            text = text.replace(live, local)
        (package / f"{name}.py").write_text(text)
    sys.path.insert(0, str(where))
    modules = {name: importlib.import_module(f"{PACKAGE}.{name}") for name in SOURCES}
    _LOADED = types.SimpleNamespace(commit=COMMIT, directory=str(package), **modules)
    return _LOADED


def load_or_skip() -> types.SimpleNamespace:
    """:func:`load`, or skip the calling test in a clone without commit :data:`COMMIT` (a
    shallow clone, or a history that never had it): the comparison reads that commit."""
    import pytest

    try:
        return load()
    except subprocess.CalledProcessError:
        pytest.skip(f"commit {COMMIT} is not in this clone; the base comparison needs it")


def chain(base, indexer, query_rows, key, weights, gate_score, pool_bank, tail_bank, slots,
          seq_lens, position, *, max_seq_len: int):
    """e3f38f8's decode indexer chain for ``B`` requests: ``[B, width]`` int32 or None.

    ``base`` is :func:`load`'s namespace and ``indexer`` a
    ``base.model_fp8.Glm5NextDSAIndexer``. ``query_rows`` is the query projection before
    its rotation, ``[B * heads, head_dim]`` bf16; ``key``, ``weights`` and ``gate_score``
    are ``project_stage``'s other three outputs. The banks are written in place.
    """
    batch = int(key.shape[0])
    query = base.kpool_hadamard.dsa_hadamard128(query_rows).reshape(
        batch, indexer.index_n_heads, indexer.index_head_dim)
    return indexer.forward_requests(
        key.new_zeros((batch, 1)), None, pool_bank, tail_bank, slots, seq_lens, position,
        max_seq_len=int(max_seq_len), indices_wanted=True,
        projected=(query, key, weights, gate_score))
