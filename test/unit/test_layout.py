"""Self-test for the ``test/`` overlay: the tree matches the shipped config.

Two claims, both read from the *effective* pytest configuration rather than from
a re-parse of ``pyproject.toml``, so this fails if the config that actually runs
ever drifts from the tree on disk:

1. ``testpaths`` is exactly the pair the pin ships, and both entries resolve to
   directories that exist.
2. The three overlay markers are registered, so ``@pytest.mark.*`` in this tree
   raises no ``PytestUnknownMarkWarning``.

A third claim covers the vendored half of the overlay:

3. **Overlay completeness.** Every vendored file under
   ``test/vllm_neuron/upstream/`` has exactly one register row in
   ``test/OVERLAY.md``, and the sha256 that row records equals the sha256 of the
   file on disk. This is what makes "vendored verbatim" checkable instead of
   asserted: it fails both ways round -- a vendored file nobody registered, and a
   registered file whose body was edited after vendoring.
"""

from __future__ import annotations

import hashlib
import re

import pytest

EXPECTED_TESTPATHS = ("test/unit", "test/vllm_neuron")
EXPECTED_MARKERS = ("fast", "forked", "quarantined")

#: Where vendored upstream files live, relative to the repo root.
UPSTREAM_DIR = "test/vllm_neuron/upstream"
#: The provenance register that must account for every one of them.
OVERLAY_REGISTER = "test/OVERLAY.md"
#: A 64-hex-digit sha256, wherever it appears in a register row.
_SHA256_RE = re.compile(r"\b([0-9a-f]{64})\b")


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


def _register_rows(register_text: str) -> dict[str, str]:
    """Map vendored filename -> recorded sha256, from ``OVERLAY.md``'s table.

    Keyed on the *basename* so the row can spell the path however it likes; the
    sha256 is found positionally-free, so a later column re-order cannot quietly
    turn this check into a no-op.
    """
    rows: dict[str, str] = {}
    for line in register_text.splitlines():
        if not line.lstrip().startswith("|") or UPSTREAM_DIR not in line:
            continue
        recorded = _SHA256_RE.search(line)
        paths = re.findall(rf"{re.escape(UPSTREAM_DIR)}/([A-Za-z0-9_.-]+\.py)", line)
        if recorded is None or len(paths) != 1:
            continue
        rows[paths[0]] = recorded.group(1)
    return rows


@pytest.mark.fast
def test_overlay_register_accounts_for_every_vendored_file(
    pytestconfig: pytest.Config,
) -> None:
    """Every vendored file is registered, and its body still hashes to the row."""
    rootpath = pytestconfig.rootpath
    upstream = rootpath / UPSTREAM_DIR
    register = rootpath / OVERLAY_REGISTER
    assert register.is_file(), f"{OVERLAY_REGISTER} is missing"

    if not upstream.is_dir():
        pytest.skip(f"{UPSTREAM_DIR} does not exist yet; nothing is vendored")

    # conftest.py carries the quarantine mechanism and is fork-authored, so it is
    # not a vendored file and is deliberately not expected in the register.
    vendored = sorted(path.name for path in upstream.glob("test_*.py"))
    assert vendored, f"{UPSTREAM_DIR} exists but holds no vendored test files"

    rows = _register_rows(register.read_text(encoding="utf-8"))

    unregistered = [name for name in vendored if name not in rows]
    assert not unregistered, (
        f"vendored with no {OVERLAY_REGISTER} row (origin PR, path, sha256): {unregistered}"
    )

    orphaned = sorted(name for name in rows if name not in vendored)
    assert not orphaned, (
        f"{OVERLAY_REGISTER} registers files absent from {UPSTREAM_DIR}: {orphaned}"
    )

    drifted = []
    for name in vendored:
        digest = hashlib.sha256((upstream / name).read_bytes()).hexdigest()
        if digest != rows[name]:
            drifted.append(f"{name}: on disk {digest}, registered {rows[name]}")
    assert not drifted, (
        "vendored bodies no longer match their sha256 at vendoring -- either the "
        f"body was edited or the row is stale: {drifted}"
    )
