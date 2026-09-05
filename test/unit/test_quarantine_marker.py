"""The quarantine mechanism, measured on a population that can reach it.

``test/vllm_neuron/upstream/conftest.py`` marks every item collected from the
vendored directory ``quarantined``, and that marker gates **selection**, never
**import**. The vendored files themselves therefore cannot measure it: all four
fail at module level against absent upstream targets, before any marker is
evaluated. This test measures the mechanism on a fixture package that imports
cleanly, against the **shipped** conftest -- the copy's sha256 is asserted equal
to the shipped file's, so the two cannot drift apart.

Two inner readings, each a subprocess run with the **same interpreter** as this
test, so the instrument is the campaign venv and not some other python:
``-m 'not quarantined'`` -> 5 deselected / 0 selected / 0 failed, **exit 5**
(``NO_TESTS_COLLECTED``, *not* 0); ``--collect-only -q`` -> 5 collected /
0 errors, **exit 0**, the import half the marker does not touch.

The five items are 2 plain plus one three-way parametrized test and every body
raises, so a selected item is a loud failure: this cannot pass vacuously.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys

import pytest

SHIPPED_CONFTEST = "test/vllm_neuron/upstream/conftest.py"
EXPECTED_ITEMS = 5

#: 2 plain + 1 three-way parametrized = 5 collected items.
FIXTURE_MODULE = '''\
import pytest


def test_plain_one() -> None:
    raise AssertionError("must never run: selection is gated")


def test_plain_two() -> None:
    raise AssertionError("must never run: selection is gated")


@pytest.mark.parametrize("case", [0, 1, 2])
def test_parametrized(case: int) -> None:
    raise AssertionError("must never run: selection is gated")
'''

#: The fixture is its own rootdir, so the inner runs never depend on the repo's
#: configuration being discovered from a temp directory.
FIXTURE_INI = """\
[pytest]
markers =
    fast: cheap, device-free
    forked: must run in its own process
    quarantined: vendored ahead of its target; selection is gated
"""


def _run(*args: str, cwd: str) -> subprocess.CompletedProcess[str]:
    """Same interpreter, no inherited pytest options, output captured."""
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    return subprocess.run(  # noqa: S603 - fixed argv, same interpreter
        [sys.executable, "-m", "pytest", *args],
        cwd=cwd, env=env, capture_output=True, text=True, check=False,
    )


@pytest.mark.fast
def test_quarantine_marker_gates_selection_but_not_import(
    pytestconfig: pytest.Config, tmp_path
) -> None:
    shipped = pytestconfig.rootpath / SHIPPED_CONFTEST
    assert shipped.is_file(), f"{SHIPPED_CONFTEST} is missing"

    fixture = tmp_path / "quarantined_fixture"
    fixture.mkdir()
    shutil.copy2(shipped, fixture / "conftest.py")
    (fixture / "test_fixture_bodies.py").write_text(FIXTURE_MODULE, encoding="utf-8")
    (fixture / "pytest.ini").write_text(FIXTURE_INI, encoding="utf-8")

    copied = hashlib.sha256((fixture / "conftest.py").read_bytes()).hexdigest()
    original = hashlib.sha256(shipped.read_bytes()).hexdigest()
    assert copied == original, (
        "the fixture conftest is not the shipped one, so this test would prove "
        f"nothing about the tree: copied {copied}, shipped {original}"
    )

    deselect = _run(str(fixture), "-q", "-m", "not quarantined", cwd=str(tmp_path))
    out = deselect.stdout + deselect.stderr
    assert deselect.returncode == 5, (
        f"'not quarantined' must exit 5 (NO_TESTS_COLLECTED), not "
        f"{deselect.returncode}:\n{out}"
    )
    assert re.search(rf"\b{EXPECTED_ITEMS} deselected\b", out), (
        f"expected {EXPECTED_ITEMS} deselected items:\n{out}"
    )
    assert re.search(r"\b\d+ (passed|failed|error|errors|skipped)\b", out) is None, (
        f"an item was selected, failed or errored, so nothing was gated:\n{out}"
    )

    collect = _run(str(fixture), "--collect-only", "-q", cwd=str(tmp_path))
    out = collect.stdout + collect.stderr
    assert collect.returncode == 0, (
        f"--collect-only must exit 0, not {collect.returncode}:\n{out}"
    )
    assert re.search(rf"\b{EXPECTED_ITEMS} tests? collected\b", out), (
        f"expected {EXPECTED_ITEMS} collected items:\n{out}"
    )
    assert re.search(r"\b\d+ (error|errors)\b", out) is None, (
        f"collection must report 0 errors:\n{out}"
    )
