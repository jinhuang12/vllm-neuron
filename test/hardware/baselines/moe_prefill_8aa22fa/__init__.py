# SPDX-License-Identifier: Apache-2.0
"""The MoE prefill call site exactly as commit 8aa22fa shipped it, beside HEAD.

:func:`load` reads ``model_fp8.py`` with ``git show 8aa22fa:<path>``, checks it against
the pinned blob id and imports it from a temporary directory outside the worktree, as
package ``moe_prefill_8aa22fa_snapshot``. Every import inside it (configs, loaders and
all kernel modules) resolves to the live tree: this branch edits no existing kernel
module, only ``model_fp8.py``'s ``Glm5NextRoutedExperts.route_tokens`` (one added branch),
so the snapshot's MoE block runs 8aa22fa's route (``route_tokens`` -> the fused RMSNorm +
router + noaux_tc kernel on the pre-norm activations) on the same kernels HEAD has.
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import tempfile
import types

COMMIT = "8aa22fa"

#: module name -> (repository path, git blob id at :data:`COMMIT`).
SOURCES: dict[str, tuple[str, str]] = {
    "model_fp8": ("vllm_neuron/model/glm5_next/model_fp8.py",
                  "6d21beaea7f956cd4db27d1d22d6a8bd647de774"),
}

PACKAGE = "moe_prefill_8aa22fa_snapshot"
_LOADED: types.SimpleNamespace | None = None


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_repo_root()), *args], check=True,
                          capture_output=True, text=True).stdout


def load() -> types.SimpleNamespace:
    """``SimpleNamespace(commit=..., directory=..., model_fp8=<module>)``, once per process."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    where = pathlib.Path(tempfile.mkdtemp(prefix="moe_prefill_8aa22fa_"))
    package = where / PACKAGE
    package.mkdir()
    (package / "__init__.py").write_text('"""Commit 8aa22fa, read by git show."""\n')
    for name, (path, blob) in SOURCES.items():
        found = _git("rev-parse", f"{COMMIT}:{path}").strip()
        if found != blob:
            raise RuntimeError(f"{COMMIT}:{path} is blob {found}, pinned {blob}")
        (package / f"{name}.py").write_text(_git("show", f"{COMMIT}:{path}"))
    sys.path.insert(0, str(where))
    modules = {name: importlib.import_module(f"{PACKAGE}.{name}") for name in SOURCES}
    _LOADED = types.SimpleNamespace(commit=COMMIT, directory=str(package), **modules)
    return _LOADED
