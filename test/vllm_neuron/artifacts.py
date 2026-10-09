# SPDX-License-Identifier: Apache-2.0
"""Host artefacts the tests read that are not in the repository.

Each kind of artefact has one knob, declared in ``vllm_neuron/envs.py`` and resolved
by ``vllm_neuron/_artifact_paths.py``, and this module is the only reader of those
knobs under ``test/``. A test that needs an artefact the host does not have skips,
naming the path it resolved and the knob that sets it; nothing substitutes stand-in
data for it.

* :func:`checkpoint_dir` / :func:`require_checkpoint`: the served GLM-5.3-Flash
  checkpoint, ``VLLM_NEURON_GLM5NEXT_CHECKPOINT_DIR``.
* :func:`campaign_path`: the performance campaign's worktrees and records
  (calibration, reports, device profiles), ``VLLM_NEURON_GLM5NEXT_CAMPAIGN_DIR``.

Importing this module imports neither torch, vLLM nor ``vllm_neuron/__init__.py``,
and ``pytest`` only to skip, so the scripts under ``test/hardware``, ``test/perf``
and ``test/kernel_ledger`` read their default paths here too and start fast.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_PATHS = "vllm_neuron._artifact_paths"


def _load_paths() -> ModuleType:
    """``vllm_neuron._artifact_paths``, without running ``vllm_neuron/__init__.py``.

    It is the module ``import vllm_neuron._artifact_paths`` would give: the file in
    the ``vllm_neuron`` package that ``sys.path`` finds, registered under its own
    name, so ``vllm_neuron.envs`` resolves the knobs with the same functions.
    """
    if _PATHS in sys.modules:
        return sys.modules[_PATHS]
    if "vllm_neuron" in sys.modules:
        return importlib.import_module(_PATHS)
    package = importlib.util.find_spec("vllm_neuron")
    if package is None or not package.submodule_search_locations:
        raise ModuleNotFoundError("the vllm_neuron package is not on sys.path", name="vllm_neuron")
    source = Path(next(iter(package.submodule_search_locations)), "_artifact_paths.py")
    spec = importlib.util.spec_from_file_location(_PATHS, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PATHS] = module
    spec.loader.exec_module(module)
    return module


_paths = _load_paths()

#: The knob naming the checkpoint directory.
CHECKPOINT_KNOB = _paths.CHECKPOINT_KNOB
#: The knob naming the campaign directory.
CAMPAIGN_KNOB = _paths.CAMPAIGN_KNOB
#: The file that makes a directory a sharded safetensors checkpoint.
CHECKPOINT_INDEX = "model.safetensors.index.json"


def checkpoint_dir() -> Path:
    """The checkpoint directory the knob names; not checked for existence."""
    return Path(_paths.checkpoint_dir())


def require_checkpoint() -> Path:
    """:func:`checkpoint_dir`, or skip the calling test when it holds no index.

    The skip message carries the index path that was looked for and the knob, so
    a run that lacks the checkpoint says so by name in its summary (``-rs``).
    """
    root = checkpoint_dir()
    index = root / CHECKPOINT_INDEX
    if not index.is_file():
        import pytest

        pytest.skip(
            f"GLM-5.3-Flash checkpoint not found: {index} does not exist "
            f"({CHECKPOINT_KNOB}={root}); set {CHECKPOINT_KNOB} to the directory "
            f"that holds {CHECKPOINT_INDEX}"
        )
    return root


def campaign_path(*parts: str) -> Path:
    """``parts`` below the campaign directory the knob names; not checked for existence."""
    return Path(_paths.campaign_dir(), *parts)
