# SPDX-License-Identifier: Apache-2.0
"""Enable the NKI CPU simulator for every mHC kernel test.

Why this file exists: ``vllm_neuron.utils.neuron_utils.can_run_kernel`` in CPU
mode answers ``os.environ.get("NKI_SIMULATOR") == "1"``, and
``libtorch_neuronx_lite``'s ``wrap_nki`` reads the same variable at call time.
``test/conftest.py`` pins ``VLLM_NEURON_CPU_MODE=1`` but not ``NKI_SIMULATOR``.
Without it the gate correctly reports "no device and no simulator", the combine
takes its torch path, and every case that asserts an NKI dispatch reads 0. That
was the baseline failure at 5938748 ("dispatch counter read 0, expected 1"): the
gate was right and the test shape was right; the tests never enabled the
simulator they assert on.

The fixture uses ``monkeypatch`` so a test that sets ``NKI_SIMULATOR=0`` itself
(the torch-fallback cases) still wins inside its own body.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _nki_simulator_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NKI_SIMULATOR", "1")
