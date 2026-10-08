# SPDX-License-Identifier: Apache-2.0
"""Run this directory's NKI kernels on the CPU simulator, every kernel selected.

The root conftest selects CPU mode, where a kernel dispatch refuses unless
``NKI_SIMULATOR=1`` is set. Every test here asserts that its kernel dispatched, so the
simulator is this directory's default; a test that sets ``NKI_SIMULATOR=0`` itself
still wins inside its own body.

The kernel tests bound each kernel at every phase and row count, so the glue switch is
``all`` here (821274e's routes) rather than the served default; the switch's own tests
(``test_selector*.py``) and the kill-switch tests set it inside their bodies. The value
the run started with is kept in :data:`SERVED_VALUE` first: ``test_served_value.py``
checks the routes under it.
"""

from __future__ import annotations

import os

import pytest

#: ``VLLM_NEURON_GLUE_FUSED`` as this run started (None: unset, which means ``1``).
SERVED_VALUE = os.environ.get("VLLM_NEURON_GLUE_FUSED")


@pytest.fixture(autouse=True)
def _nki_simulator_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NKI_SIMULATOR", "1")


@pytest.fixture(autouse=True)
def _every_glue_kernel_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_NEURON_GLUE_FUSED", "all")
