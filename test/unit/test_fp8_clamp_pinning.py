"""``FP8_CLAMP_MAX`` is resolved once, at import time, from the platform target.

``vllm_neuron/utils/dtype_utils.py`` resolves ``FP8_CLAMP_MAX`` at module import
from ``_resolve_fp8_clamp_max()``: a ``trn3`` target gives 448.0 and any other
target gives 240.0. Because the value is fixed by that import, neither a fixture
nor ``monkeypatch.setenv`` can change it, so every reading below is taken in a
fresh subprocess running this same interpreter, with the target set explicitly in
the child's own copy of the environment.

Covered here: the clamp each target resolves, that the constant does not follow a
target change made after import, and ``test/conftest.py``'s pre-collection
handling of the two variables this test tree needs. No reading claims the host is
trn3 or validates trn3 hardware, and no target is exported to a shell where it
could point a compiling run at the wrong architecture.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

OVERRIDE = "NEURON_PLATFORM_TARGET_OVERRIDE"
CPU_MODE = "VLLM_NEURON_CPU_MODE"

# What test/conftest.py supplies when the invocation sets no override. An
# invocation may still supply its own, so this is not what the parent carries.
DEFAULTED_OVERRIDE = "trn2"

E4M3_MAX = 240.0  # dtype_utils.py:18 -- trn2, e4m3 with inf
E4M3FN_MAX = 448.0  # dtype_utils.py:19 -- trn3 / finite-FP8 CPU

#: The clamp each target family resolves, keyed by the target the child is given.
CLAMP_BY_TARGET = {"trn2": E4M3_MAX, "trn3": E4M3FN_MAX}

#: Prints exactly one machine-readable line, so the parent compares a parsed
#: value exactly instead of scanning a log.
PROBE = (
    "from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX;"
    "print('FP8_CLAMP_MAX=%r' % (FP8_CLAMP_MAX,))"
)
READING = re.compile(r"^FP8_CLAMP_MAX=(.+)$", re.MULTILINE)

#: The header ``test/conftest.py`` prints, and the two notes it tags a resolved
#: variable with. Read out of a child pytest's own output, so the mechanism is
#: measured where it runs instead of being re-implemented here.
CONFTEST_HEADER = "overlay environment pinned by test/conftest.py:"
DEFAULTED_NOTE = "(DEFAULTED by test/conftest.py)"
INVOCATION_NOTE = "(from the invocation)"

#: pytest's exit code for a ``pytest.UsageError`` (``ExitCode.USAGE_ERROR``).
USAGE_ERROR = 4


def _import_time_probe(flip_to: str) -> str:
    """Source for the child that separates import-time from per-access resolution."""
    # The third reading is the one that shows the flip took effect: the vendor's
    # get_platform_target reads the environment on every call and caches nothing
    # (libtorch_neuronx_lite/compile/platform.py:85-86), so a resolver still
    # returning the first value would mean the environment never changed.
    return (
        "import os;"
        "from vllm_neuron.utils import dtype_utils as d;"
        "first = d.FP8_CLAMP_MAX;"
        f"os.environ[{OVERRIDE!r}] = {flip_to!r};"
        "second = d.FP8_CLAMP_MAX;"
        "third = d._resolve_fp8_clamp_max();"
        "print('FIRST=%r' % (first,));"
        "print('SECOND=%r' % (second,));"
        "print('THIRD=%r' % (third,))"
    )


def _base_env() -> dict[str, str]:
    """A copy of this process's environment, minus inherited pytest options."""
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    return env


def _read_clamp_max(env: dict[str, str], cwd: str) -> float:
    """Import ``FP8_CLAMP_MAX`` in a fresh child; return the value it resolved."""
    done = subprocess.run(  # noqa: S603 - fixed argv, same interpreter
        [sys.executable, "-c", PROBE],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = done.stdout + done.stderr
    assert done.returncode == 0, f"probe child exited {done.returncode}:\n{out}"
    found = READING.search(done.stdout)
    assert found, f"probe printed no FP8_CLAMP_MAX line:\n{out}"
    return float(found.group(1))


def _read_across_a_live_flip(
    env: dict[str, str], cwd: str, flip_to: str
) -> tuple[float, float, float]:
    """Return ``(constant, constant_after_flip, resolver_after_flip)`` from one child."""
    done = subprocess.run(  # noqa: S603 - fixed argv, same interpreter
        [sys.executable, "-c", _import_time_probe(flip_to)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = done.stdout + done.stderr
    assert done.returncode == 0, f"probe child exited {done.returncode}:\n{out}"
    values = []
    for label in ("FIRST", "SECOND", "THIRD"):
        found = re.search(rf"^{label}=(.+)$", done.stdout, re.MULTILINE)
        assert found, f"probe printed no {label} line:\n{out}"
        values.append(float(found.group(1)))
    return values[0], values[1], values[2]


def _collect_only(env: dict[str, str], cwd: str) -> subprocess.CompletedProcess[str]:
    """Collect this file in a child pytest, so ``test/conftest.py`` runs for real."""
    # --collect-only imports the module and prints the session header but runs no
    # test, so the caller cannot recurse. The header carries the resolution, so
    # no -q and no --no-header.
    return subprocess.run(  # noqa: S603 - fixed argv, same interpreter
        [
            sys.executable,
            "-m",
            "pytest",
            "test/unit/test_fp8_clamp_pinning.py",
            "--collect-only",
            "-p",
            "no:cacheprovider",
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.fast
def test_each_platform_target_resolves_its_own_clamp(
    pytestconfig: pytest.Config,
) -> None:
    """A trn2 child resolves 240.0 and a trn3 child resolves 448.0."""
    root = str(pytestconfig.rootpath)

    measured: dict[str, float] = {}
    for target in CLAMP_BY_TARGET:
        env = _base_env()
        env[CPU_MODE] = "1"
        env[OVERRIDE] = target
        measured[target] = _read_clamp_max(env, root)

    assert measured == CLAMP_BY_TARGET, (
        f"each target must resolve its own clamp exactly "
        f"(dtype_utils.py:18-19,34-37); expected {CLAMP_BY_TARGET!r}, the "
        f"children resolved {measured!r}"
    )


@pytest.mark.fast
def test_the_clamp_is_resolved_at_import_and_not_on_every_access(
    pytestconfig: pytest.Config,
) -> None:
    """The constant keeps its import-time value after the target changes in-process."""
    env = _base_env()
    env[CPU_MODE] = "1"
    env[OVERRIDE] = "trn2"

    constant, after_flip, resolver = _read_across_a_live_flip(
        env, str(pytestconfig.rootpath), flip_to="trn3"
    )

    # The resolver reading comes first: it is what shows the flip took effect, so
    # an unmoved constant means import-time resolution rather than an environment
    # change that never happened.
    assert resolver == E4M3FN_MAX, (
        f"_resolve_fp8_clamp_max() called after the flip to trn3 must return "
        f"{E4M3FN_MAX!r} (dtype_utils.py:34-35); it returned {resolver!r}, so the "
        f"environment change did not take effect"
    )
    assert after_flip == constant == E4M3_MAX, (
        f"FP8_CLAMP_MAX resolves once at import (dtype_utils.py:40-41), so it must "
        f"still read {E4M3_MAX!r} after the target changed in-process; it read "
        f"{after_flip!r}"
    )


@pytest.mark.fast
def test_conftest_defaults_the_two_variables_and_refuses_cpu_mode_zero(
    pytestconfig: pytest.Config,
) -> None:
    """Unset variables are defaulted, a supplied trn3 is kept, CPU mode 0 is refused."""
    root = str(pytestconfig.rootpath)

    # Neither variable set: both are defaulted, and the run collects.
    env = _base_env()
    env.pop(CPU_MODE, None)
    env.pop(OVERRIDE, None)
    unset = _collect_only(env, root)
    assert unset.returncode == 0, (
        f"a run with neither variable set must collect:\n{unset.stdout}{unset.stderr}"
    )
    assert "collected" in unset.stdout, (
        f"nothing was collected, so nothing was pinned:\n{unset.stdout}"
    )
    assert CONFTEST_HEADER in unset.stdout, (
        f"the conftest printed no resolution:\n{unset.stdout}"
    )
    assert f"{CPU_MODE}=1 {DEFAULTED_NOTE}" in unset.stdout, unset.stdout
    assert f"{OVERRIDE}={DEFAULTED_OVERRIDE} {DEFAULTED_NOTE}" in unset.stdout, unset.stdout
    assert f"expected FP8_CLAMP_MAX={E4M3_MAX}" in unset.stdout, unset.stdout

    # A supplied trn3 is kept, and the run is allowed: refusing it would leave a
    # trn3 host no way to run this tree with the clamp its own target resolves.
    env = _base_env()
    env[CPU_MODE] = "1"
    env[OVERRIDE] = "trn3"
    supplied = _collect_only(env, root)
    assert supplied.returncode == 0, (
        f"a trn3 invocation must be allowed, not refused:\n"
        f"{supplied.stdout}{supplied.stderr}"
    )
    assert f"{OVERRIDE}=trn3 {INVOCATION_NOTE}" in supplied.stdout, supplied.stdout
    assert f"expected FP8_CLAMP_MAX={E4M3FN_MAX}" in supplied.stdout, supplied.stdout

    # The one contradiction that is refused.
    env = _base_env()
    env[CPU_MODE] = "0"
    refused = _collect_only(env, root)
    said = refused.stdout + refused.stderr
    assert refused.returncode == USAGE_ERROR, (
        f"{CPU_MODE}=0 must be refused as a usage error (exit {USAGE_ERROR}); "
        f"the child exited {refused.returncode}:\n{said}"
    )
    assert f"{CPU_MODE}='0' contradicts this test tree" in said, said
    assert CONFTEST_HEADER not in said, (
        f"the refusal happens before the resolution, so no resolution should be "
        f"printed:\n{said}"
    )
