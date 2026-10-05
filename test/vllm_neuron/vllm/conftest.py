"""Pin the NKI simulator for this directory before any kernel module is imported.

The tiny GLM-5.3-Flash root used by the runner tests here reads ``NKI_SIMULATOR`` and
``NKI_PRECISE_FP`` at import time. The root ``test/conftest.py`` pins only the CPU
mode, so a plain ``pytest test/vllm_neuron/vllm`` would otherwise refuse those tests.
A value the caller set is kept.
"""

from __future__ import annotations

import os

SIMULATOR_ENV = {"NKI_SIMULATOR": "1", "NKI_PRECISE_FP": "1"}


def pytest_configure(config) -> None:
    for name, value in SIMULATOR_ENV.items():
        os.environ.setdefault(name, value)
