"""Root conftest for the ``test/`` overlay: pin the CPU-simulator environment.

``FP8_CLAMP_MAX`` in ``vllm_neuron/utils/dtype_utils.py`` resolves at *import*
time from ``NEURON_PLATFORM_TARGET_OVERRIDE``, so that variable has to be in the
process environment before pytest imports anything under test. A fixture or
``monkeypatch.setenv`` runs too late and would leave the wrong clamp pinned, so
the two variables are set in ``pytest_configure`` instead, which runs before
collection.

An unset variable is defaulted here; a value the caller supplied is kept, so the
documented per-directory invocations in ``docs/model-dev/`` work without pinning
anything, including on a trn3 host where the clamp is 448.0 rather than 240.0.
Only one contradiction is refused: ``VLLM_NEURON_CPU_MODE`` set to something
other than ``1``. The run header then prints what was resolved and where each
value came from.
"""

from __future__ import annotations

import os

import pytest

#: The value each variable takes when the invocation does not set it, chosen so a
#: defaulted run and a fully pinned run resolve the same clamp.
DEFAULTED_ENV = {
    "VLLM_NEURON_CPU_MODE": "1",
    "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
}

#: The two platform families ``dtype_utils`` distinguishes: a ``trn3`` target pins
#: 448.0 and everything else pins 240.0. Named for the header below. An
#: unrecognised target is not refused, because the clamp has a defined answer for
#: it -- the 240.0 branch.
ACCEPTED_TARGET_FAMILIES = ("trn2", "trn3")

#: The one contradiction that is refused. CPU mode selects the simulator path this
#: overlay tests, so defaulting over a caller who asked for something else would
#: hide the disagreement instead of reporting it.
CPU_MODE = "VLLM_NEURON_CPU_MODE"

#: Whether each variable in :data:`DEFAULTED_ENV` came from the invocation or from
#: this file. Both cases leave the identical value in the environment, so the
#: origin has to be recorded to stay readable. Printed only -- nothing here reads
#: it to decide behaviour.
SUPPLIED = "supplied"
DEFAULTED = "defaulted"
RESOLUTION_ORIGIN: dict[str, str] = {}

#: Filled by :func:`pytest_configure`, read by :func:`pytest_report_header`.
_RESOLUTION: dict[str, str] = {}


def pytest_configure(config: pytest.Config) -> None:
    """Pin the CPU-simulator environment before collection imports anything."""
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
            RESOLUTION_ORIGIN[name] = DEFAULTED
        else:
            _RESOLUTION[name] = f"{current} (from the invocation)"
            RESOLUTION_ORIGIN[name] = SUPPLIED

    target = os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"]
    family = next(
        (f for f in ACCEPTED_TARGET_FAMILIES if target.startswith(f)), "other"
    )
    _RESOLUTION["_clamp"] = (
        f"{'448.0' if target.startswith('trn3') else '240.0'} "
        f"(target family {family}, dtype_utils.py:34-37)"
    )


def pytest_report_header() -> list[str]:
    """Print what the run resolved, so the pinning is visible in the transcript.

    This does not import ``dtype_utils``: importing it here would pin the clamp
    inside the pytest process and pull the platform plugin in as a side effect.
    The clamp line is computed from the same branch condition ``dtype_utils``
    uses; the value itself is measured in a fresh child process by
    ``test/unit/test_fp8_clamp_pinning.py``.
    """
    if not _RESOLUTION:
        return []
    origins = ", ".join(
        f"{name}={RESOLUTION_ORIGIN[name]}" for name in sorted(RESOLUTION_ORIGIN)
    )
    return [
        "overlay environment pinned by test/conftest.py:",
        *(
            f"  {name}={value}"
            for name, value in sorted(_RESOLUTION.items())
            if not name.startswith("_")
        ),
        f"  expected FP8_CLAMP_MAX={_RESOLUTION['_clamp']}",
        f"  resolution origin: {origins}",
    ]
