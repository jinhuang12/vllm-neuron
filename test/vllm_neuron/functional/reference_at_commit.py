# SPDX-License-Identifier: Apache-2.0
"""One source file of an older commit, loaded as a module: the reference of a bit-equality test.

The file comes from ``git show <commit>:<path>`` in the checkout that holds the imported
``vllm_neuron`` package, and is imported from a temporary directory under a module name of
its own, so the tree itself is never edited. A checkout without that commit (a shallow
clone, an exported tree) has no reference: a test that needs one carries
:func:`needs_reference`, an explicit ``skipif`` that names the missing commit and file.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

import vllm_neuron

#: The checkout the imported ``vllm_neuron`` package lives in.
REPO_ROOT = Path(vllm_neuron.__file__).resolve().parents[1]


def has_reference(commit: str, path: str) -> bool:
    """True when ``path`` at ``commit`` is in this checkout."""
    probe = subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{commit}:{path}"],
                           capture_output=True)
    return probe.returncode == 0


def needs_reference(commit: str, path: str) -> pytest.MarkDecorator:
    """A ``skipif`` for a test whose reference is ``path`` at ``commit``, with the reason."""
    return pytest.mark.skipif(
        not has_reference(commit, path),
        reason=f"{commit}:{path} is not in this checkout, so the bit-equality reference is absent")


def load_reference(commit: str, path: str, name: str) -> ModuleType:
    """``path`` as it was at ``commit``, imported as the module ``name``.

    Raises ``FileNotFoundError`` when the checkout does not hold it; a test marked with
    :func:`needs_reference` is skipped before it gets here.
    """
    shown = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{commit}:{path}"],
                           capture_output=True)
    if shown.returncode != 0:
        raise FileNotFoundError(f"{commit}:{path}: {shown.stderr.decode().strip()}")
    file = Path(tempfile.mkdtemp(prefix=f"{name}_")) / f"{name}.py"
    file.write_bytes(shown.stdout)
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
