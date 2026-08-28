"""Self-test for the ``test/`` overlay: the tree matches the shipped config.

Two claims, both read from the *effective* pytest configuration rather than from
a re-parse of ``pyproject.toml``, so this fails if the config that actually runs
ever drifts from the tree on disk:

1. ``testpaths`` is exactly the pair the pin ships, and both entries resolve to
   directories that exist.
2. The three overlay markers are registered, so ``@pytest.mark.*`` in this tree
   raises no ``PytestUnknownMarkWarning``.
"""

from __future__ import annotations

import pytest

EXPECTED_TESTPATHS = ("test/unit", "test/vllm_neuron")
EXPECTED_MARKERS = ("fast", "forked", "quarantined")


def _registered_marker_names(config: pytest.Config) -> set[str]:
    """Leading name of every ``markers`` ini entry (``"name: description"``)."""
    return {entry.split(":", 1)[0].split("(", 1)[0].strip() for entry in config.getini("markers")}


@pytest.mark.fast
def test_layout(pytestconfig: pytest.Config) -> None:
    testpaths = tuple(pytestconfig.getini("testpaths"))
    assert testpaths == EXPECTED_TESTPATHS, (
        f"testpaths drifted: {testpaths} != {EXPECTED_TESTPATHS}"
    )

    rootpath = pytestconfig.rootpath
    missing = [entry for entry in testpaths if not (rootpath / entry).is_dir()]
    assert not missing, f"testpaths entries do not exist as directories: {missing}"

    registered = _registered_marker_names(pytestconfig)
    absent = [name for name in EXPECTED_MARKERS if name not in registered]
    assert not absent, f"markers not registered in pyproject.toml: {absent}"
