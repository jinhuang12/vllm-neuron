"""Root conftest for the fork's ``test/`` overlay.

This file sets **no** environment variable and defines **no** env-setting
fixture, by design. ``FP8_CLAMP_MAX`` resolves at *import* time, so
``NEURON_PLATFORM_TARGET_OVERRIDE`` has to be in the process environment before
pytest imports anything under test. A fixture or ``monkeypatch.setenv`` runs
after that import and would silently leave the wrong clamp pinned, so the
environment belongs in the invocation and nowhere else.

What this file does instead is *assert* that the invocation pinned the
environment, and fail loudly before collection if it did not. That turns a
silent wrong-clamp run into a hard, readable stop.
"""

from __future__ import annotations

import os

import pytest

# Exactly the values the declared CPU-mode acceptance commands carry in the
# invocation. This is an assertion about the caller, never an assignment.
REQUIRED_ENV = {
    "VLLM_NEURON_CPU_MODE": "1",
    "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
}


def pytest_configure(config: pytest.Config) -> None:
    """Stop before collection if the acceptance environment is not pinned."""
    wrong = {
        name: os.environ.get(name)
        for name, want in REQUIRED_ENV.items()
        if os.environ.get(name) != want
    }
    if wrong:
        detail = ", ".join(
            f"{name}={value!r} (want {REQUIRED_ENV[name]!r})"
            for name, value in sorted(wrong.items())
        )
        raise pytest.UsageError(
            "CPU-mode acceptance environment is not pinned: "
            f"{detail}. Set these on the command line -- FP8_CLAMP_MAX "
            "resolves at import time, so a fixture or monkeypatch.setenv is "
            "too late and is forbidden as the mechanism."
        )
