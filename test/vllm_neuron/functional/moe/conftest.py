# SPDX-License-Identifier: Apache-2.0
"""Enable the NKI CPU simulator for every MoE kernel test in this directory.

``vllm_neuron.utils.neuron_utils.can_run_kernel`` answers
``os.environ.get("NKI_SIMULATOR") == "1"`` in CPU mode, and ``wrap_nki`` reads
the same variable at call time. ``test/conftest.py`` pins
``VLLM_NEURON_CPU_MODE=1`` but not ``NKI_SIMULATOR``. Without it every seam that
asserts an NKI dispatch reads 0 and takes its torch path; that was the state of
this directory at 5938748 when the documented command ran without the variable.

A private ``MonkeyPatch`` keeps the change per test, survives a case's own
``monkeypatch.undo()``, and still lets a case that sets ``NKI_SIMULATOR=0``
itself (the torch-fallback cases) win inside its body.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _nki_simulator_on():
    # A private MonkeyPatch, not the test's own: some cases call
    # ``monkeypatch.undo()`` mid-body, which would otherwise drop the simulator.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("NKI_SIMULATOR", "1")
        yield
