# SPDX-License-Identifier: Apache-2.0
"""The capture-site tests pass after the ``meta`` forward test, in one process.

``test_meta_forward.py`` replaces the vendor's NKI dispatch while it drives a captured step
on ``meta``. A module imported for the first time inside that window builds its import-time
kernel table out of the replacement, and the replacement hands its outputs back on ``meta``.
If that table outlives the test, the capture-site tests that drive the same forward on the
host afterwards fail on a device mismatch, and they fail only in that order and only in one
process. A run of either file alone cannot show it, so this runs the two in that order in a
fresh interpreter and requires every test of both to pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as tiny_e2e

# Device-free, like the two files it runs. Not ``forked``: it starts its own interpreter for
# them, so nothing in this process decides the outcome.
pytestmark = [pytest.mark.fast]

#: The repository root, where ``python -m pytest`` imports this checkout's packages from.
REPO = Path(__file__).resolve().parents[5]

#: The test that patches the dispatch, then the tests that run the host forward after it.
FIRST = "test/vllm_neuron/model/glm5_next/test_meta_forward.py"
AFTER = "test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_capture_sites.py"


def _test_ids(path: str) -> list[str]:
    """The test functions ``path`` defines, read from its source, as pytest node ids."""
    source = (REPO / path).read_text()
    return [f"{path}::{name}" for name in re.findall(r"^def (test_\w+)\(", source, re.M)]


def test_the_capture_sites_pass_when_the_meta_forward_runs_first_in_the_same_process():
    tiny_e2e._require_cpu_mode()
    expected = _test_ids(FIRST) + _test_ids(AFTER)
    assert len(_test_ids(FIRST)) >= 1 and len(_test_ids(AFTER)) >= 2, (
        f"found {expected}; this order test needs the meta forward and the capture sites "
        f"it guards to still define their tests"
    )
    # ``-p no:randomly`` keeps the order given here if a shuffling plugin is installed. The
    # environment is inherited, so the child runs on the same CPU backend as this test.
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider",
         "-p", "no:randomly", FIRST, AFTER],
        cwd=REPO, capture_output=True, text=True, timeout=3600, check=False,
    )
    tail = "\n".join((run.stdout + run.stderr).splitlines()[-40:])
    passed = set(re.findall(r"^PASSED (\S+)", run.stdout, re.M))
    # A parametrized test passes as ``name[case]``; each case of it has to be among them.
    missing = [
        node for node in expected
        if not any(seen == node or seen.startswith(f"{node}[") for seen in passed)
    ]
    assert run.returncode == 0 and not missing, (
        f"run in this order in one process, {missing} did not pass (exit "
        f"{run.returncode}). A module the meta forward imported kept its patched dispatch "
        f"if the failure is a device mismatch.\n{tail}"
    )
