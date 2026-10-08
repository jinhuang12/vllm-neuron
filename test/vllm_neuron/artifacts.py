# SPDX-License-Identifier: Apache-2.0
"""Host artefacts the tests read that are not in the repository.

Each artefact has one knob in ``vllm_neuron/envs.py``, with its default there, and
this module is the only reader of that knob under ``test/``. A test that needs an
artefact the host does not have skips, naming the path it resolved and the knob
that sets it; nothing substitutes stand-in data for it.

* :func:`checkpoint_dir` / :func:`require_checkpoint`: the served GLM-5.3-Flash
  checkpoint, ``VLLM_NEURON_GLM5NEXT_CHECKPOINT_DIR``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm_neuron import envs

#: The ``vllm_neuron.envs`` knob naming the checkpoint directory.
CHECKPOINT_KNOB = "VLLM_NEURON_GLM5NEXT_CHECKPOINT_DIR"
#: The file that makes a directory a sharded safetensors checkpoint.
CHECKPOINT_INDEX = "model.safetensors.index.json"


def checkpoint_dir() -> Path:
    """The checkpoint directory the knob names; not checked for existence."""
    return Path(envs.VLLM_NEURON_GLM5NEXT_CHECKPOINT_DIR)


def require_checkpoint() -> Path:
    """:func:`checkpoint_dir`, or skip the calling test when it holds no index.

    The skip message carries the index path that was looked for and the knob, so
    a run that lacks the checkpoint says so by name in its summary (``-rs``).
    """
    root = checkpoint_dir()
    index = root / CHECKPOINT_INDEX
    if not index.is_file():
        pytest.skip(
            f"GLM-5.3-Flash checkpoint not found: {index} does not exist "
            f"({CHECKPOINT_KNOB}={root}); set {CHECKPOINT_KNOB} to the directory "
            f"that holds {CHECKPOINT_INDEX}"
        )
    return root
