"""Root conftest for the fork's ``test/`` overlay.

``FP8_CLAMP_MAX`` resolves at *import* time
(``vllm_neuron/utils/dtype_utils.py:41``, from ``:22-37``), so
``NEURON_PLATFORM_TARGET_OVERRIDE`` has to be in the **process environment**
before pytest imports anything under test. A fixture or ``monkeypatch.setenv``
runs after that import and would silently leave the wrong clamp pinned, so it is
still forbidden as the mechanism.

WHAT THIS FILE DOES. It pins the two variables in ``pytest_configure``, which
runs before collection and therefore before any test module -- and so before
``dtype_utils`` -- is imported. When a variable is unset it is DEFAULTED here;
when it is set the caller's value is kept; and one explicit contradiction is
refused. Every run then prints what it resolved, so the pinning is visible in
the transcript rather than assumed.

WHY IT NO LONGER DEMANDS THE VALUES FROM THE CALLER. It used to raise
``pytest.UsageError`` unless the environment already carried exactly
``VLLM_NEURON_CPU_MODE=1`` and ``NEURON_PLATFORM_TARGET_OVERRIDE=trn2``. That
refused the repository's own documented invocations -- ``pytest test/unit -v
--timeout=300`` and ``pytest test/vllm_neuron/functional/ -v --timeout=60``
(``docs/model-dev/cpu-development.md:63``, ``:69``;
``docs/model-dev/nki_cpu_simulator.md:85``, ``:88``) set no platform override,
so they collected zero tests and stopped. Worse, it refused
``NEURON_PLATFORM_TARGET_OVERRIDE=trn3`` outright, and ``dtype_utils.py:34-37``
returns the 448.0 clamp only for a ``trn3`` target -- so on a trn3 machine the
only way to run this suite at all was to pin the wrong 240.0 clamp. Defaulting
serves the same purpose the old gate served, which is that no run reaches a test
with the clamp unpinned, without refusing the callers it exists to serve.
"""

from __future__ import annotations

import os

import pytest

#: The value each variable takes when the invocation does not set it. These are
#: the values the campaign's own declared acceptance commands carry, so a
#: defaulted run and a fully-pinned campaign run resolve the same clamp.
DEFAULTED_ENV = {
    "VLLM_NEURON_CPU_MODE": "1",
    "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
}

#: The two platform families ``dtype_utils.py:34-37`` distinguishes: a ``trn3``
#: target pins 448.0 and everything else pins 240.0. Recorded for the reader and
#: for the header below. An unrecognised target is NOT refused -- the clamp has a
#: defined answer for it (the 240.0 ``else`` arm), and refusing it here would
#: reintroduce the defect this file was repaired for.
ACCEPTED_TARGET_FAMILIES = ("trn2", "trn3")

#: The one explicit contradiction that is still refused. CPU mode is what selects
#: the whole simulator path this overlay tests; a caller who sets it to something
#: other than ``1`` has asked for a run this tree cannot give, and defaulting
#: over that request would hide the disagreement instead of reporting it.
CPU_MODE = "VLLM_NEURON_CPU_MODE"

#: Filled by :func:`pytest_configure`, read by :func:`pytest_report_header`.
_RESOLUTION: dict[str, str] = {}


def pytest_configure(config: pytest.Config) -> None:
    """Pin the acceptance environment before collection imports anything.

    Refuses exactly one thing: ``VLLM_NEURON_CPU_MODE`` explicitly set to a
    value other than ``1``.
    """
    supplied = os.environ.get(CPU_MODE)
    if supplied is not None and supplied != "1":
        raise pytest.UsageError(
            f"{CPU_MODE}={supplied!r} contradicts this test tree: the overlay "
            f"runs on the CPU simulator path and needs {CPU_MODE}=1. Unset it "
            f"to take the default, or set it to '1'."
        )

    for name, default in DEFAULTED_ENV.items():
        current = os.environ.get(name)
        if current is None:
            os.environ[name] = default
            _RESOLUTION[name] = f"{default} (DEFAULTED by test/conftest.py)"
        else:
            _RESOLUTION[name] = f"{current} (from the invocation)"

    target = os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"]
    family = next(
        (f for f in ACCEPTED_TARGET_FAMILIES if target.startswith(f)), "other"
    )
    _RESOLUTION["_clamp"] = (
        f"{'448.0' if target.startswith('trn3') else '240.0'} "
        f"(target family {family}, dtype_utils.py:34-37)"
    )


def pytest_report_header() -> list[str]:
    """Print what the run resolved, so the pinning is a reading and not a claim.

    This does NOT import ``dtype_utils``: importing it here would pin the clamp
    inside the pytest process and pull the platform plugin in as a side effect.
    The clamp line below is computed from the same branch condition
    ``dtype_utils.py:34-37`` uses; the value itself is measured in a fresh child
    by ``test/unit/test_fp8_clamp_pinning.py``.
    """
    if not _RESOLUTION:
        return []
    return [
        "overlay environment pinned by test/conftest.py:",
        *(
            f"  {name}={value}"
            for name, value in sorted(_RESOLUTION.items())
            if not name.startswith("_")
        ),
        f"  expected FP8_CLAMP_MAX={_RESOLUTION['_clamp']}",
    ]
