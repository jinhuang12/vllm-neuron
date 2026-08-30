"""``FP8_CLAMP_MAX`` is pinned by the environment at import time -- measured.

``vllm_neuron/utils/dtype_utils.py`` resolves ``FP8_CLAMP_MAX`` **once, at module
import** (``:41``, with the file's own note at ``:40``), from
``_resolve_fp8_clamp_max()`` (``:22-37``). The two literals it can return are the
pin's own: ``_FP8_E4M3_MAX = 240.0`` and ``_FP8_E4M3FN_MAX = 448.0`` (``:18-19``,
with the trn2/trn3 rationale in the comment above them).

Because the value resolves at *import*, no fixture and no ``monkeypatch.setenv``
can measure it -- both run after the import that already froze it, and
``test/conftest.py`` forbids them by name. Each case therefore needs its own
**fresh interpreter**, so every reading below is taken in a ``subprocess`` child
running this same interpreter (the campaign venv, not some other python).

Two readings, one per environment; both children carry
``VLLM_NEURON_CPU_MODE=1``:

* ``NEURON_PLATFORM_TARGET_OVERRIDE=trn2`` -> ``FP8_CLAMP_MAX == 240.0`` exactly.
* ``NEURON_PLATFORM_TARGET_OVERRIDE=trn3`` -> ``FP8_CLAMP_MAX == 448.0`` exactly.

The second reading is what makes the first falsifiable: it asserts the value
genuinely *moves* with the environment, so a passing first reading is a pin and
not a coincidence.

Both readings travel the branch at ``:34-37`` -- the ``trn3`` arm (``:34-35``)
and the ``else`` arm (``:36-37``). What these two readings do **not** exercise
is the bare-CPU fallback at ``:27-32``: reaching it needs a host with no NRT,
so nothing here claims that ``FP8_CLAMP_MAX`` resolves to 448.0 with the
override unset. Neither reading claims the host is trn3, and neither validates
trn3 hardware.

**"Fresh import" is not "clean environment".** ``subprocess`` inherits
``os.environ``, and ``test/conftest.py`` guarantees the *parent* carries
``NEURON_PLATFORM_TARGET_OVERRIDE=trn2`` before collection. The ``trn3``
child's environment is therefore **constructed** -- the variable is set to
``trn3`` over the inherited ``trn2``, never assumed -- and the inherited value
that is replaced is asserted here so the change is attributable rather than
silent. ``trn3`` lives only in that one constructed child dict: it is never
exported to a shell and never reaches a compiling run, which would point the
compiler at the wrong architecture.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

OVERRIDE = "NEURON_PLATFORM_TARGET_OVERRIDE"
CPU_MODE = "VLLM_NEURON_CPU_MODE"

#: The parent's pinned override, guaranteed by ``test/conftest.py``'s gate.
INHERITED_OVERRIDE = "trn2"

E4M3_MAX = 240.0  # dtype_utils.py:18 -- trn2, e4m3 with inf
E4M3FN_MAX = 448.0  # dtype_utils.py:19 -- trn3 / finite-FP8 CPU

#: Prints exactly one machine-readable line, so the parent compares a parsed
#: value exactly instead of eyeballing a log.
PROBE = (
    "from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX;"
    "print('FP8_CLAMP_MAX=%r' % (FP8_CLAMP_MAX,))"
)
READING = re.compile(r"^FP8_CLAMP_MAX=(.+)$", re.MULTILINE)


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


@pytest.mark.fast
def test_override_trn2_pins_clamp_to_240(pytestconfig: pytest.Config) -> None:
    env = _base_env()
    env[CPU_MODE] = "1"
    env[OVERRIDE] = "trn2"

    value = _read_clamp_max(env, str(pytestconfig.rootpath))
    assert value == E4M3_MAX, (
        f"{OVERRIDE}=trn2 must pin FP8_CLAMP_MAX to {E4M3_MAX!r} exactly "
        f"(dtype_utils.py:18); the child resolved {value!r}"
    )


@pytest.mark.fast
def test_override_trn3_resolves_clamp_to_448(pytestconfig: pytest.Config) -> None:
    # The parent is guaranteed to carry trn2, so record what this child replaces.
    assert os.environ.get(OVERRIDE) == INHERITED_OVERRIDE, (
        f"expected the inherited {OVERRIDE}=={INHERITED_OVERRIDE!r} that "
        f"test/conftest.py pins; found {os.environ.get(OVERRIDE)!r}"
    )

    env = _base_env()
    env[CPU_MODE] = "1"
    replaced = env.get(OVERRIDE)
    env[OVERRIDE] = "trn3"
    assert replaced == INHERITED_OVERRIDE, f"replaced {replaced!r}, not the inherited value"
    assert env[OVERRIDE] == "trn3", f"{OVERRIDE} is {env[OVERRIDE]!r} in the child env"

    value = _read_clamp_max(env, str(pytestconfig.rootpath))
    assert value == E4M3FN_MAX, (
        f"{OVERRIDE}=trn3 (with {CPU_MODE}=1) must resolve FP8_CLAMP_MAX to "
        f"{E4M3FN_MAX!r} exactly (dtype_utils.py:19,34-35); the child resolved {value!r}"
    )
