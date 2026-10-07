# SPDX-License-Identifier: Apache-2.0
"""Run this directory's NKI kernels on the CPU simulator.

The root conftest selects CPU mode, where a kernel dispatch refuses unless
``NKI_SIMULATOR=1`` is set. Every test here asserts that its kernel dispatched, so the
simulator is this directory's default; a test that sets ``NKI_SIMULATOR=0`` itself
still wins inside its own body.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _nki_simulator_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NKI_SIMULATOR", "1")
