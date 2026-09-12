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

Four arms. Every child carries ``VLLM_NEURON_CPU_MODE=1`` except the one arm that
deliberately contradicts it:

1. ``NEURON_PLATFORM_TARGET_OVERRIDE=trn2`` -> ``FP8_CLAMP_MAX == 240.0`` exactly.
2. The **pair**, in one arm: a ``trn2`` child and a ``trn3`` child, ``240.0`` and
   ``448.0``, asserted to differ. This is what makes arm 1 falsifiable -- the value
   genuinely *moves* with the environment, so a passing arm 1 is a pin and not a
   coincidence.
3. The **import-time** reading: one child reads the constant, changes the target in
   its own live process, reads the constant again, then calls the resolver directly.
   The constant must not move; the resolver must.
4. The **mechanism** that pins the parent: ``test/conftest.py``'s pre-collection
   check, exercised in a child pytest -- defaulted, supplied, and refused -- plus the
   origin it records, read both from the live plugin module and from each child's
   header. A supplied ``trn2`` and a defaulted ``trn2`` leave the identical value
   behind, so the origin is the only thing that separates them.

The readings travel the branch at ``:34-37`` -- the ``trn3`` arm (``:34-35``)
and the ``else`` arm (``:36-37``). What none of them exercises is the bare-CPU
fallback at ``:27-32``: reaching it needs a host with no NRT, so nothing here
claims that ``FP8_CLAMP_MAX`` resolves to 448.0 with the override unset. No
reading claims the host is trn3, and none validates trn3 hardware.

**"Fresh import" is not "clean environment".** ``subprocess`` inherits
``os.environ``, and ``test/conftest.py`` resolves the *parent*'s
``NEURON_PLATFORM_TARGET_OVERRIDE`` before collection. Every child's environment
is therefore **constructed** -- each target is set explicitly in the child dict,
never assumed from the parent -- and what the parent resolved is RECORDED rather
than demanded. Each target lives only in its own constructed child dict: none is
exported to a shell and none reaches a compiling run, which would point the
compiler at the wrong architecture.

REPAIRED BY ``inc-glm53f-014``'s R2 ROUND, for finding
``B07-M1-014-import-time-coverage-unmeasured``. The review's point was that the two
original readings gave the same answer whether the clamp resolves once at import
or on every access, so the file claimed a coverage it did not measure. Three things
changed:

* **The import-time reading now exists** (arm 3). One child imports the module,
  reads the constant, changes the target **in its own live process**, reads the
  constant again, and then calls the resolver directly. Import-time resolution
  gives ``second == first`` while the resolver returns the other value; per-access
  resolution would give ``second != first``. That is a reading whose value differs
  between the two designs, which is what the finding asked for.
* **Arm 2 no longer demands ``trn2`` of its parent.** It reads a ``trn2`` child and
  a ``trn3`` child in the same arm and asserts the two differ, so the pin is
  falsified by a pair the arm builds rather than by an inherited value. The old form
  asserted the parent carried ``trn2`` and therefore could not run at all on a trn3
  machine -- which blocked ``inc-glm53f-001``'s trn3 reading.
* **The conftest mechanism is now asserted** (arm 4). The finding's second half
  recorded that nothing in the tree exercised ``test/conftest.py``'s pre-collection
  check. It is exercised here, in its post-``-001`` form: the two variables are
  DEFAULTED when unset, a caller's value is kept, and one explicit contradiction is
  refused.
* **The recorded origin is asserted too**, on the lead's design call N7. Defaulting
  the variables removed the only way to tell a supplied ``trn2`` from a defaulted
  ``trn2`` -- the environment reads the same either way -- so the conftest now records
  which it was and arm 4 asserts the pair comes back different for the same value.
  Without that reading the design's D2 invocation rule is unfalsifiable.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

OVERRIDE = "NEURON_PLATFORM_TARGET_OVERRIDE"
CPU_MODE = "VLLM_NEURON_CPU_MODE"

#: What ``test/conftest.py`` DEFAULTS the override to when the invocation sets none
#: (``test/conftest.py:47-49``). It is not what the parent necessarily carries: an
#: invocation may supply its own, and arm 2 records the parent's value rather than
#: demanding this one.
DEFAULTED_OVERRIDE = "trn2"

E4M3_MAX = 240.0  # dtype_utils.py:18 -- trn2, e4m3 with inf
E4M3FN_MAX = 448.0  # dtype_utils.py:19 -- trn3 / finite-FP8 CPU

#: The clamp each target family pins, keyed by the target the child is given. Both
#: are read in arm 2, in one arm, so "the pin moves" is a comparison and not two
#: assertions in two arms that could each be true for the wrong reason.
CLAMP_BY_TARGET = {"trn2": E4M3_MAX, "trn3": E4M3FN_MAX}

#: Prints exactly one machine-readable line, so the parent compares a parsed
#: value exactly instead of eyeballing a log.
PROBE = (
    "from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX;"
    "print('FP8_CLAMP_MAX=%r' % (FP8_CLAMP_MAX,))"
)
READING = re.compile(r"^FP8_CLAMP_MAX=(.+)$", re.MULTILINE)

#: The header ``test/conftest.py:132`` prints, and the two notes it tags a resolved
#: variable with (``:101``, ``:104``). Arm 4 reads these out of a child pytest's own
#: output, so the mechanism is measured where it runs rather than re-implemented here.
CONFTEST_HEADER = "overlay environment pinned by test/conftest.py:"
DEFAULTED_NOTE = "(DEFAULTED by test/conftest.py)"
INVOCATION_NOTE = "(from the invocation)"
ORIGIN_LINE = "resolution origin: "

#: pytest's exit code for a ``pytest.UsageError`` (``ExitCode.USAGE_ERROR``).
USAGE_ERROR = 4


def _import_time_probe(flip_to: str) -> str:
    """Source for the child that separates import-time from per-access resolution.

    The child reads ``FP8_CLAMP_MAX``, sets ``NEURON_PLATFORM_TARGET_OVERRIDE`` to
    ``flip_to`` in its **own live process**, reads the constant a second time, and
    then calls ``_resolve_fp8_clamp_max()`` directly. The third reading is the
    non-vacuity control: the vendor's ``get_platform_target`` reads the environment
    on every call and caches nothing
    (``libtorch_neuronx_lite/compile/platform.py:85-86``), so a resolver that still
    returned the first value would mean the flip never took effect and the second
    reading proved nothing.
    """
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


def _origin_line(text: str) -> str:
    """The recorded-origin line out of a session header, without its label."""
    found = re.search(rf"^\s*{re.escape(ORIGIN_LINE)}(.+)$", text, re.MULTILINE)
    assert found, f"no {ORIGIN_LINE!r} line in:\n{text}"
    return found.group(1).strip()


def _root_conftest(config: pytest.Config):
    """The live ``test/conftest.py`` plugin module, so its record is read at source.

    Reading the origin out of the module the running session actually loaded is what
    makes it a fact rather than a re-implementation: a copy of the logic here would
    pass even if the conftest stopped recording anything.
    """
    for _name, plugin in config.pluginmanager.list_name_plugin():
        path = getattr(plugin, "__file__", "") or ""
        if path.replace(os.sep, "/").endswith("/test/conftest.py"):
            return plugin
    raise AssertionError(
        "the root test/conftest.py is not among this session's registered plugins, "
        "so nothing pinned the overlay environment"
    )


def _collect_only(env: dict[str, str], cwd: str) -> subprocess.CompletedProcess[str]:
    """Collect this file in a child pytest, so ``test/conftest.py`` runs for real.

    ``--collect-only`` imports the module and prints the session header but runs no
    test, so the arm reading this cannot recurse. The header is what carries the
    resolution, so no ``-q`` and no ``--no-header``.
    """
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
    """The pair in one arm: trn2 and trn3 must not agree.

    The parent's resolved target is RECORDED, not demanded. The old form asserted
    the parent carried ``trn2`` and so reddened this whole file on a trn3 machine,
    which is the reading ``inc-glm53f-001`` needed and could not take. Both children
    are built here, so the arm's claim -- that the clamp moves with the target -- is
    settled by a comparison it makes itself and does not depend on what invoked it.
    """
    root = str(pytestconfig.rootpath)
    parent_resolved = os.environ.get(OVERRIDE)
    print(f"PARENT_RESOLVED_TARGET={parent_resolved!r}")

    measured: dict[str, float] = {}
    for target in CLAMP_BY_TARGET:
        env = _base_env()
        env[CPU_MODE] = "1"
        env[OVERRIDE] = target
        assert env[OVERRIDE] == target, f"{OVERRIDE} is {env[OVERRIDE]!r} in the child env"
        measured[target] = _read_clamp_max(env, root)
    print(f"CLAMP_BY_TARGET_MEASURED={measured!r}")

    assert measured == CLAMP_BY_TARGET, (
        f"each target must pin its own clamp exactly (dtype_utils.py:18-19,34-37); "
        f"expected {CLAMP_BY_TARGET!r}, the children resolved {measured!r}"
    )
    assert measured["trn2"] != measured["trn3"], (
        "the two targets resolved the same clamp, so nothing here shows the pin "
        f"moves with the environment: {measured!r}"
    )


@pytest.mark.fast
def test_the_clamp_is_resolved_at_import_and_not_on_every_access(
    pytestconfig: pytest.Config,
) -> None:
    """Three readings from ONE child, whose values differ between the two designs.

    ``dtype_utils.py:41`` resolves the clamp once at import; the file says so at
    ``:40``. Nothing in this tree used to measure it -- both original arms gave the
    same answer either way. Here one child reads the constant, changes the target in
    its own process, and reads the constant again:

    * import-time resolution -> the second reading equals the first;
    * per-access resolution -> the second reading equals the OTHER target's clamp.

    The resolver is then called directly, after the same flip. It must return the
    other value, which is what proves the flip took effect; without that control a
    second reading that did not move would be equally consistent with an environment
    change that never happened.
    """
    env = _base_env()
    env[CPU_MODE] = "1"
    env[OVERRIDE] = "trn2"

    constant, after_flip, resolver = _read_across_a_live_flip(
        env, str(pytestconfig.rootpath), flip_to="trn3"
    )
    print(
        f"IMPORT_TIME_CONSTANT={constant!r} "
        f"CONSTANT_AFTER_LIVE_FLIP={after_flip!r} "
        f"RESOLVER_AFTER_LIVE_FLIP={resolver!r}"
    )

    # The control first: the flip was real, so the second reading means something.
    assert resolver == E4M3FN_MAX, (
        f"_resolve_fp8_clamp_max() called after the live flip to trn3 must return "
        f"{E4M3FN_MAX!r} (dtype_utils.py:34-35); it returned {resolver!r}, so the "
        f"environment change did not take effect and this arm proves nothing"
    )
    assert resolver != constant, (
        f"the resolver returned the import-time value {constant!r} after the flip, "
        f"so the two designs are indistinguishable in this child"
    )

    # The reading that separates import-time from per-access resolution.
    assert after_flip == constant == E4M3_MAX, (
        f"FP8_CLAMP_MAX resolves ONCE at import (dtype_utils.py:40-41), so it must "
        f"still read {E4M3_MAX!r} after the target changed in-process; it read "
        f"{after_flip!r}. A value of {E4M3FN_MAX!r} here would mean the clamp is "
        f"resolved per access, and every fixture-based approach this file rejects "
        f"would have worked after all"
    )


@pytest.mark.fast
def test_conftest_defaults_the_two_variables_and_refuses_one_contradiction(
    pytestconfig: pytest.Config,
) -> None:
    """The mechanism that pins the parent, exercised in a child pytest.

    Nothing in this tree asserted ``test/conftest.py``'s pre-collection check. Three
    child collections cover the three things it does (``test/conftest.py:83-105``):
    default an unset variable, keep a supplied one, and refuse ``VLLM_NEURON_CPU_MODE``
    set to anything but ``1``. The middle case is the one the pre-``inc-glm53f-001``
    gate refused outright, so it is asserted to SUCCEED.

    AND IT ASSERTS THE RECORDED ORIGIN, not just the value. Once the two variables are
    defaulted, ``trn2`` in the environment no longer says whether the invocation pinned
    it or the conftest supplied it, so the design's D2 invocation rule could not be
    falsified by reading the value. The conftest records the origin
    (``test/conftest.py`` ``RESOLUTION_ORIGIN``); this arm reads it out of the live
    plugin module for its own session, and reads it again from each child's header,
    where the two cases must come back DIFFERENT for the same value.
    """
    root = str(pytestconfig.rootpath)

    # 0. This session's own origin, read from the module that actually pinned it.
    conftest = _root_conftest(pytestconfig)
    recorded = dict(conftest.RESOLUTION_ORIGIN)
    print(f"PARENT_RESOLUTION_ORIGIN={recorded!r}")
    assert set(recorded) == set(conftest.DEFAULTED_ENV), (
        f"the conftest recorded an origin for {sorted(recorded)}, but it resolves "
        f"{sorted(conftest.DEFAULTED_ENV)}; a variable with no recorded origin cannot "
        f"be checked against D2's invocation rule"
    )
    assert set(recorded.values()) <= {conftest.SUPPLIED, conftest.DEFAULTED}, recorded

    # The record and what the run reports must be one fact, not two.
    parent_header = "\n".join(conftest.pytest_report_header())
    parent_origin = _origin_line(parent_header)
    print(f"PARENT_ORIGIN_LINE={parent_origin!r}")
    for name, origin in sorted(recorded.items()):
        assert f"{name}={origin}" in parent_origin, (
            f"{name} is recorded as {origin!r} but the header says {parent_origin!r}"
        )
        note = INVOCATION_NOTE if origin == conftest.SUPPLIED else DEFAULTED_NOTE
        assert f"  {name}={os.environ[name]} {note}" in parent_header, (
            f"{name} is recorded as {origin!r}, so the header must tag its value "
            f"{note}; the header is:\n{parent_header}"
        )

    # 1. Neither variable set: both are defaulted, and the run collects.
    env = _base_env()
    env.pop(CPU_MODE, None)
    env.pop(OVERRIDE, None)
    unset = _collect_only(env, root)
    print(f"CONFTEST_UNSET_EXIT={unset.returncode}")
    assert unset.returncode == 0, f"a run with neither variable set must collect:\n{unset.stdout}{unset.stderr}"
    assert "collected" in unset.stdout, f"nothing was collected, so nothing was pinned:\n{unset.stdout}"
    assert CONFTEST_HEADER in unset.stdout, f"the conftest printed no resolution:\n{unset.stdout}"
    assert f"{CPU_MODE}=1 {DEFAULTED_NOTE}" in unset.stdout, unset.stdout
    assert f"{OVERRIDE}={DEFAULTED_OVERRIDE} {DEFAULTED_NOTE}" in unset.stdout, unset.stdout
    assert f"expected FP8_CLAMP_MAX={E4M3_MAX}" in unset.stdout, unset.stdout
    unset_origin = _origin_line(unset.stdout)
    print(f"CONFTEST_UNSET_ORIGIN={unset_origin!r}")
    assert unset_origin == f"{OVERRIDE}={conftest.DEFAULTED}, {CPU_MODE}={conftest.DEFAULTED}", unset_origin

    # 2. A supplied trn3 is KEPT, and the run is allowed. The old gate refused this.
    env = _base_env()
    env[CPU_MODE] = "1"
    env[OVERRIDE] = "trn3"
    supplied = _collect_only(env, root)
    print(f"CONFTEST_SUPPLIED_TRN3_EXIT={supplied.returncode}")
    assert supplied.returncode == 0, (
        f"a trn3 invocation must be allowed, not refused:\n{supplied.stdout}{supplied.stderr}"
    )
    assert f"{OVERRIDE}=trn3 {INVOCATION_NOTE}" in supplied.stdout, supplied.stdout
    assert f"expected FP8_CLAMP_MAX={E4M3FN_MAX}" in supplied.stdout, supplied.stdout
    supplied_origin = _origin_line(supplied.stdout)
    print(f"CONFTEST_SUPPLIED_ORIGIN={supplied_origin!r}")
    assert supplied_origin == f"{OVERRIDE}={conftest.SUPPLIED}, {CPU_MODE}={conftest.SUPPLIED}", supplied_origin

    # 2b. THE READING D2 NEEDS: same resolved value, different recorded origin. A
    # child that supplies trn2 and a child that lets the conftest default trn2 leave
    # the identical environment behind, so only the origin separates them.
    env = _base_env()
    env[CPU_MODE] = "1"
    env[OVERRIDE] = DEFAULTED_OVERRIDE
    same_value_supplied = _collect_only(env, root)
    print(f"CONFTEST_SUPPLIED_TRN2_EXIT={same_value_supplied.returncode}")
    assert same_value_supplied.returncode == 0, (
        f"{same_value_supplied.stdout}{same_value_supplied.stderr}"
    )
    supplied_trn2_origin = _origin_line(same_value_supplied.stdout)
    print(f"CONFTEST_SUPPLIED_TRN2_ORIGIN={supplied_trn2_origin!r}")
    assert f"{OVERRIDE}={DEFAULTED_OVERRIDE} {INVOCATION_NOTE}" in same_value_supplied.stdout, (
        same_value_supplied.stdout
    )
    assert f"{OVERRIDE}={conftest.SUPPLIED}" in supplied_trn2_origin, supplied_trn2_origin
    assert supplied_trn2_origin != unset_origin, (
        f"a supplied trn2 and a defaulted trn2 recorded the same origin "
        f"{supplied_trn2_origin!r}, so nothing distinguishes them and D2's invocation "
        f"rule cannot be falsified by any reading in this tree"
    )
    # And the value really is identical in both, which is what makes the pair the point.
    assert f"expected FP8_CLAMP_MAX={E4M3_MAX}" in same_value_supplied.stdout, (
        same_value_supplied.stdout
    )

    # 3. The one contradiction that is still refused.
    env = _base_env()
    env[CPU_MODE] = "0"
    refused = _collect_only(env, root)
    said = refused.stdout + refused.stderr
    print(f"CONFTEST_CPU_MODE_0_EXIT={refused.returncode}")
    assert refused.returncode == USAGE_ERROR, (
        f"{CPU_MODE}=0 must be refused as a usage error (exit {USAGE_ERROR}); "
        f"the child exited {refused.returncode}:\n{said}"
    )
    assert f"{CPU_MODE}='0' contradicts this test tree" in said, said
    assert CONFTEST_HEADER not in said, (
        f"the refusal happened before the resolution, so no resolution should be "
        f"printed:\n{said}"
    )
