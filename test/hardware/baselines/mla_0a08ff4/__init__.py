# SPDX-License-Identifier: Apache-2.0
"""The MLA decode kernel exactly as commit 0a08ff4 shipped it, importable beside HEAD.

``mla_decode.py`` next to this file is a byte copy taken with
``git show 0a08ff4:vllm_neuron/functional/attention/mla_decode.py``. :func:`load` checks
it against :data:`SNAPSHOT_SHA256` (and, where git can see the commit, against the
blob id at 0a08ff4), so an edit to the copy fails the load instead of moving the
baseline. Its imports (``wrap_nki``, ``values_are_readable``) resolve to the live tree;
neither changes in this branch.

The copy keys its own compiled-kernel cache: its ``SOURCE_DIGEST`` is the digest of
its own bytes, which are 0a08ff4's, so it never shares a cache entry with the live
kernel file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

COMMIT = "0a08ff4"
PATH = "vllm_neuron/functional/attention/mla_decode.py"
BLOB = "011fc1b7f5f0b5b52c5d8f374b51c303c9cb6ca1"
SNAPSHOT_SHA256 = "552383dd0b1a9b0c92e9d8c6ea0006fc5873371a8010d89ccc9e967369d6676d"
HERE = Path(__file__).resolve().parent
MODULE = "mla_decode_0a08ff4_snapshot"

_LOADED: ModuleType | None = None


def _git_blob() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(HERE), "rev-parse", f"{COMMIT}:{PATH}"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out


def load() -> ModuleType:
    """The 0a08ff4 ``mla_decode`` module, loaded once per process."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    path = HERE / "mla_decode.py"
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != SNAPSHOT_SHA256:
        raise ValueError(f"baseline snapshot {path} hashes to {got}, not the {COMMIT} copy "
                         f"{SNAPSHOT_SHA256}")
    blob = _git_blob()
    if blob is not None and blob != BLOB:
        raise ValueError(f"{COMMIT}:{PATH} is blob {blob}, pinned {BLOB}")
    spec = importlib.util.spec_from_file_location(MODULE, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load baseline snapshot {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module
    spec.loader.exec_module(module)
    _LOADED = module
    return module
