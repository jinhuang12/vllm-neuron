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

    The **subject** of a row is read from its first cell only. Other cells
    legitimately mention paths under the vendored directory -- the quarantine
    column names the ``conftest.py`` that applies the marker -- so scanning the
    whole line for "the" path is ambiguous, and treating that ambiguity as
    "unparseable" silently drops the row instead of checking it.

    The sha256 is still matched anywhere in the row, so re-ordering the later
    columns cannot turn this check into a no-op.
    """
    rows: dict[str, str] = {}
    path_re = re.compile(rf"{re.escape(UPSTREAM_DIR)}/([A-Za-z0-9_.-]+\.py)")
    for line in register_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or UPSTREAM_DIR not in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        subject = path_re.search(cells[0])
        recorded = _SHA256_RE.search(stripped)
        if subject is None or recorded is None:
            continue
        rows[subject.group(1)] = recorded.group(1)
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
