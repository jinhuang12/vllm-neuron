"""The quarantine rules in the vendored-test conftest, run on a live fixture.

``test/vllm_neuron/upstream/conftest.py`` marks every item it collects
``quarantined`` and skips it. The vendored files cannot measure that themselves:
the four whose upstream target is absent are never collected. So this runs the
shipped conftest over a throwaway package that imports cleanly, in a child
process using the same interpreter.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import pytest

SHIPPED_CONFTEST = "test/vllm_neuron/upstream/conftest.py"
EXPECTED_ITEMS = 5

#: 2 plain + 1 three-way parametrized = 5 collected items. Every body raises, so
#: an item that is not gated fails loudly instead of passing vacuously.
FIXTURE_MODULE = '''\
import pytest


def test_plain_one() -> None:
    raise AssertionError("must never run: this item should be skipped")


def test_plain_two() -> None:
    raise AssertionError("must never run: this item should be skipped")


@pytest.mark.parametrize("case", [0, 1, 2])
def test_parametrized(case: int) -> None:
    raise AssertionError("must never run: this item should be skipped")
'''

#: The fixture is its own rootdir, so the child runs never depend on the repo's
#: configuration being discovered from a temporary directory.
FIXTURE_INI = """\
[pytest]
markers =
    fast: cheap, device-free
    forked: must run in its own process
    quarantined: vendored from upstream ahead of the code it exercises
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
def test_vendored_items_are_collected_marked_and_skipped(
    pytestconfig: pytest.Config, tmp_path
) -> None:
    """Items import and collect, then skip; ``-m 'not quarantined'`` drops them."""
    shipped = pytestconfig.rootpath / SHIPPED_CONFTEST
    assert shipped.is_file(), f"{SHIPPED_CONFTEST} is missing"

    fixture = tmp_path / "quarantined_fixture"
    fixture.mkdir()
    shutil.copy2(shipped, fixture / "conftest.py")
    (fixture / "test_fixture_bodies.py").write_text(FIXTURE_MODULE, encoding="utf-8")
    (fixture / "pytest.ini").write_text(FIXTURE_INI, encoding="utf-8")

    plain = _run(str(fixture), "-q", cwd=str(tmp_path))
    out = plain.stdout + plain.stderr
    assert plain.returncode == 0, f"a plain run must exit 0, not {plain.returncode}:\n{out}"
    assert re.search(rf"\b{EXPECTED_ITEMS} skipped\b", out), (
        f"expected {EXPECTED_ITEMS} skipped items:\n{out}"
    )
    assert re.search(r"\b\d+ (passed|failed|error|errors)\b", out) is None, (
        f"an item ran or failed to import, so nothing was gated:\n{out}"
    )

    deselect = _run(str(fixture), "-q", "-m", "not quarantined", cwd=str(tmp_path))
    out = deselect.stdout + deselect.stderr
    assert deselect.returncode == 5, (
        f"'not quarantined' must exit 5 (no tests collected), not "
        f"{deselect.returncode}:\n{out}"
    )
    assert re.search(rf"\b{EXPECTED_ITEMS} deselected\b", out), (
        f"expected {EXPECTED_ITEMS} deselected items:\n{out}"
    )
