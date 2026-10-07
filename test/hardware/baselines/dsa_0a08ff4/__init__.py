# SPDX-License-Identifier: Apache-2.0
"""The DSA decode indexer chain exactly as commit 0a08ff4 shipped it, beside HEAD.

:func:`load` reads each file below with ``git show 0a08ff4:<path>``, checks it against
the pinned blob id, rewrites the imports between these files to point at the snapshot's
own copies, and imports the result from a temporary directory outside the worktree.
Every other import (configs, loaders, attention kernels, the DSA kernels this change does
not edit) resolves to the live tree.

The snapshot is read rather than committed so the baseline is the commit itself and not a
hand-made copy of it; the blob ids make a silently different baseline impossible. The
layout is ``dsa_75090b9``'s.

What it serves: :func:`chain` is 0a08ff4's decode indexer chain for ``B`` requests from
the indexer's projections on -- the query rotation (``dsa_hadamard128``, the first kernel
of ``project_stage``'s query side) and then ``Glm5NextDSAIndexer.forward_requests`` with
those projections handed in: the ring step, the batched scores, top-k, sentinel, sentinel
order and expand, and the two bank writes.
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import tempfile
import types

COMMIT = "0a08ff4"

#: module name -> (repository path, git blob id at :data:`COMMIT`). A name with a slash
#: is a file inside a snapshot subpackage.
SOURCES: dict[str, tuple[str, str]] = {
    "kpool_hadamard": ("vllm_neuron/functional/dsa/kpool_hadamard.py",
                       "dc13e33ae7de774ec42b0a6bd401677ed7d7797e"),
    "decode_batch": ("vllm_neuron/functional/dsa/decode_batch.py",
                     "8b834c0bbd910e0783d84d261a745596c0dac49b"),
    "causal_bound": ("vllm_neuron/functional/dsa/causal_bound.py",
                     "925ca7dfc8e99466e42161299ccc5d2b92c617a3"),
    "sentinel_order": ("vllm_neuron/functional/dsa/sentinel_order.py",
                       "2f6f90e9bfee18948a6c4af54c06193cf61a8494"),
    "index_expand": ("vllm_neuron/functional/dsa/index_expand.py",
                     "846eb36aba46c112e737c5908e733ef5d75f0dec"),
    "topk_select": ("vllm_neuron/functional/dsa/topk_select.py",
                    "86b257726b220da32612251c1dc17e8cb3bcbed1"),
    "decode_tail_update": ("vllm_neuron/functional/dsa/decode_tail_update.py",
                           "b1c07db22c9d5c61c5fdeb4d61712f5708137c97"),
    "rotational_topk/__init__": (
        "vllm_neuron/functional/vendored_kernels/rotational_topk/__init__.py",
        "0ff5bf469665f7d0d664f7f09008bca59c652c09"),
    "rotational_topk/rotational_topk": (
        "vllm_neuron/functional/vendored_kernels/rotational_topk/rotational_topk.py",
        "f0e7cd85f02d54d9c455a1292c84471c9c5404ee"),
    "rotational_topk/rotational_topk_utils": (
        "vllm_neuron/functional/vendored_kernels/rotational_topk/rotational_topk_utils.py",
        "6414f3e203b9e51a941014d149bd6311cd42683c"),
    "rotational_topk/cascaded_max_utils": (
        "vllm_neuron/functional/vendored_kernels/rotational_topk/cascaded_max_utils.py",
        "9b28d1331810d7400518b04b5ec1f5601b748bd7"),
    "model_fp8": ("vllm_neuron/model/glm5_next/model_fp8.py",
                  "6d21beaea7f956cd4db27d1d22d6a8bd647de774"),
}

PACKAGE = "dsa_0a08ff4_snapshot"

#: Live module paths the snapshot's own copies replace inside every snapshot file. Each
#: is a full dotted module path, so no rewrite is a prefix of a module this change keeps
#: live (``decode_bypass``, ``causal_fill`` and the rest stay the live tree's).
_REWRITES = {
    "vllm_neuron.functional.vendored_kernels.rotational_topk":
        f"{PACKAGE}.rotational_topk",
    **{f"vllm_neuron.functional.dsa.{name}": f"{PACKAGE}.{name}"
       for name in ("kpool_hadamard", "decode_batch", "causal_bound", "sentinel_order",
                    "index_expand", "topk_select", "decode_tail_update")},
}

_LOADED: types.SimpleNamespace | None = None


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_repo_root()), *args], check=True,
                          capture_output=True, text=True).stdout


def load() -> types.SimpleNamespace:
    """The snapshot modules by name (``kpool_hadamard``, ``decode_batch``, ...,
    ``model_fp8``), plus :func:`chain` bound to them.

    Loaded once per process; later calls return the same modules.
    """
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    where = pathlib.Path(tempfile.mkdtemp(prefix="dsa_0a08ff4_"))
    package = where / PACKAGE
    (package / "rotational_topk").mkdir(parents=True)
    (package / "__init__.py").write_text('"""Commit 0a08ff4, read by git show."""\n')
    for name, (path, blob) in SOURCES.items():
        found = _git("rev-parse", f"{COMMIT}:{path}").strip()
        if found != blob:
            raise RuntimeError(f"{COMMIT}:{path} is blob {found}, pinned {blob}")
        text = _git("show", f"{COMMIT}:{path}")
        for live, local in _REWRITES.items():
            text = text.replace(live, local)
        (package / f"{name}.py").write_text(text)
    sys.path.insert(0, str(where))
    # A subpackage file is bound as ``<subpackage>_<module>``; the subpackage itself by name.
    modules = {}
    for name in SOURCES:
        dotted = name.replace("/__init__", "").replace("/", ".")
        modules[dotted.replace(".", "_")] = importlib.import_module(f"{PACKAGE}.{dotted}")
    _LOADED = types.SimpleNamespace(commit=COMMIT, directory=str(package), **modules)
    return _LOADED


def chain(base, indexer, query_rows, key, weights, gate_score, pool_bank, tail_bank, slots,
          seq_lens, position, *, max_seq_len: int):
    """0a08ff4's decode indexer chain for ``B`` requests: ``[B, width]`` int32 or None.

    ``base`` is :func:`load`'s namespace and ``indexer`` a ``base.model_fp8.Glm5NextDSAIndexer``.
    ``query_rows`` is the query projection before its rotation, ``[B * heads, head_dim]``
    bf16, which ``project_stage`` hands to ``dsa_hadamard128``; ``key``, ``weights`` and
    ``gate_score`` are ``project_stage``'s other three outputs. The banks are written in
    place, as ``forward_requests`` writes them.
    """
    batch = int(key.shape[0])
    query = base.kpool_hadamard.dsa_hadamard128(query_rows).reshape(
        batch, indexer.index_n_heads, indexer.index_head_dim)
    return indexer.forward_requests(
        key.new_zeros((batch, 1)), None, pool_bank, tail_bank, slots, seq_lens, position,
        max_seq_len=int(max_seq_len), indices_wanted=True,
        projected=(query, key, weights, gate_score))
