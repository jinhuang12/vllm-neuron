# SPDX-License-Identifier: Apache-2.0
"""Run every test in this directory on the NKI CPU simulator.

``test/conftest.py`` pins ``VLLM_NEURON_CPU_MODE=1`` but leaves the simulator to
the invocation, and every kernel here needs it: without ``NKI_SIMULATOR=1``
``can_run_kernel`` reads False and the gate clamp, which has no torch route,
cannot run at all. A bare ``pytest test/vllm_neuron/functional/kda`` therefore
used to fail 54 of 71 tests on environment alone.

The variables are set for this package only and restored afterwards, so a wider
run does not leak the simulator into directories that expect it off. A value the
invocation supplied is kept.
"""

from __future__ import annotations

import os

import pytest

SIMULATOR_ENV = {"NKI_SIMULATOR": "1", "NKI_PRECISE_FP": "1"}


@pytest.fixture(scope="package", autouse=True)
def _kda_simulator_env():
    with pytest.MonkeyPatch.context() as patch:
        for name, value in SIMULATOR_ENV.items():
            if os.environ.get(name) is None:
                patch.setenv(name, value)
        yield
