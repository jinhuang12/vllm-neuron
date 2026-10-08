# SPDX-License-Identifier: Apache-2.0
"""``vllm_neuron/functional/attention/mla_sparse.py`` exactly as commit 8aa22fa shipped it.

``mla_sparse.py`` beside this file is ``git show 8aa22fa:<path>`` committed verbatim, and
:data:`BLOB` is that path's git blob id at the commit, so :func:`load` can refuse a copy
that drifted. The module is imported from its file under its own name, beside the live
tree's ``vllm_neuron.functional.attention.mla_sparse``; both resolve their other imports
(``nki``, ``wrap_nki``, ``neuron_utils``) to the live tree, which this change leaves alone.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import sys

COMMIT = "8aa22fa"
PATH = "vllm_neuron/functional/attention/mla_sparse.py"
#: ``git rev-parse 8aa22fa:vllm_neuron/functional/attention/mla_sparse.py``.
BLOB = "1dd9443bb66c5966483ef7a35335488cb87a87ea"
MODULE_NAME = "mla_sparse_8aa22fa_baseline"
FILE = pathlib.Path(__file__).with_name("mla_sparse.py")


def blob_id(path: pathlib.Path) -> str:
    """The git blob id of ``path``'s content, computed without git."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def load():
    """Import the 8aa22fa kernel module, once per process, after checking its blob id."""
    cached = sys.modules.get(MODULE_NAME)
    if cached is not None:
        return cached
    found = blob_id(FILE)
    if found != BLOB:
        raise ValueError(f"{FILE} is not {COMMIT}:{PATH}: blob {found}, expected {BLOB}")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, FILE)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
