"""Self-tests for the ``test/`` overlay: the tree matches the shipped config.

Both claims are read from the *effective* pytest configuration rather than from a
re-parse of ``pyproject.toml``, so they fail if the config that actually runs
drifts from the tree on disk. The second test checks the vendored-file register
in ``test/OVERLAY.md`` against the bytes on disk, which is what makes "vendored
from upstream unchanged" checkable instead of merely asserted.
"""

from __future__ import annotations

import hashlib
import re
from typing import NamedTuple

import pytest

EXPECTED_TESTPATHS = ("test/unit", "test/vllm_neuron")
EXPECTED_MARKERS = ("fast", "forked", "quarantined")

#: Where files vendored from upstream live, relative to the repo root.
UPSTREAM_DIR = "test/vllm_neuron/upstream"
#: The register that must account for every one of them.
OVERLAY_REGISTER = "test/OVERLAY.md"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
#: The register's column count: the vendored path as subject, then origin PR,
#: origin path, sha256 when copied, sha256 on disk, adoption, un-skip condition.
#: Read by POSITION, so a shape change fails here instead of silently skipping a
#: check.
REGISTER_CELLS = 7
ADOPTION_DOMAIN = ("VERBATIM", "ADAPTED")
#: What an adapted row carries instead of an on-disk digest. Hashing a file this
#: repository edits pins nothing: the digest is written in the same change as the
#: file it describes.
NO_ONDISK_DIGEST = "n/a"


def _registered_marker_names(config: pytest.Config) -> set[str]:
    """Leading name of every ``markers`` ini entry (``"name: description"``)."""
    return {entry.split(":", 1)[0].split("(", 1)[0].strip() for entry in config.getini("markers")}


@pytest.mark.fast
def test_layout(pytestconfig: pytest.Config) -> None:
    """``testpaths`` and the overlay's markers match what the tree needs."""
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


class RegisterRow(NamedTuple):
    """One register row, read by column position rather than by pattern search."""

    origin_sha256: str
    ondisk_sha256: str
    adoption: str
    unskip: str


def _cell(raw: str) -> str:
    """A table cell's value: markdown emphasis and code quoting are not content."""
    return raw.replace("**", "").replace("`", "").strip()


def _register_rows(register_text: str) -> tuple[dict[str, RegisterRow], list[str]]:
    """Map vendored filename to its row, plus every row this parser cannot read.

    A row's subject is its first cell. Later cells legitimately hold other paths
    and other digests, so neither the path nor the sha256 can be found by
    scanning the whole line: each row carries two digests, and taking the first
    one would check the wrong column. Rows that do not match the column shape are
    returned rather than dropped, because a check that silently skips what it
    cannot parse reports a clean register while checking nothing.
    """
    rows: dict[str, RegisterRow] = {}
    malformed: list[str] = []
    subject_re = re.compile(rf"^{re.escape(UPSTREAM_DIR)}/([A-Za-z0-9_.-]+\.py)$")
    for line in register_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [_cell(cell) for cell in stripped.strip("|").split("|")]
        subject = subject_re.match(cells[0]) if cells else None
        if subject is None:
            continue
        name = subject.group(1)
        if len(cells) != REGISTER_CELLS:
            malformed.append(f"{name}: {len(cells)} cells, expected {REGISTER_CELLS}")
            continue
        if name in rows:
            malformed.append(f"{name}: registered more than once")
            continue
        rows[name] = RegisterRow(*cells[3:7])
    return rows, malformed


@pytest.mark.fast
def test_overlay_register_matches_the_vendored_files(
    pytestconfig: pytest.Config,
) -> None:
    """Every vendored file has one register row and still hashes to it.

    This fails every way round: a vendored file nobody registered, a registered
    file whose body was edited after copying, a row for a file that is gone, a
    row whose cell count no longer matches the column shape, a blank un-skip
    cell, and a row claiming byte-identity its two digests contradict.
    """
    rootpath = pytestconfig.rootpath
    upstream = rootpath / UPSTREAM_DIR
    register = rootpath / OVERLAY_REGISTER
    assert register.is_file(), f"{OVERLAY_REGISTER} is missing"

    if not upstream.is_dir():
        pytest.skip(f"{UPSTREAM_DIR} does not exist; nothing is vendored")

    # conftest.py carries the collection rules and is fork-authored, so it is not
    # a vendored file and is deliberately not expected in the register.
    vendored = sorted(path.name for path in upstream.glob("test_*.py"))
    assert vendored, f"{UPSTREAM_DIR} exists but holds no vendored test files"

    rows, malformed = _register_rows(register.read_text(encoding="utf-8"))
    assert not malformed, (
        f"{OVERLAY_REGISTER} rows do not match the column shape, so they were not "
        f"checked at all: {malformed}"
    )

    unregistered = [name for name in vendored if name not in rows]
    assert not unregistered, f"vendored with no {OVERLAY_REGISTER} row: {unregistered}"

    orphaned = sorted(name for name in rows if name not in vendored)
    assert not orphaned, (
        f"{OVERLAY_REGISTER} registers files absent from {UPSTREAM_DIR}: {orphaned}"
    )

    # Only the unchanged copies are hashed. An adapted body is fork-authored, so
    # its on-disk digest would be written alongside the edit it claims to detect.
    drifted = []
    for name in vendored:
        if rows[name].adoption != "VERBATIM":
            continue
        digest = hashlib.sha256((upstream / name).read_bytes()).hexdigest()
        if digest != rows[name].ondisk_sha256:
            drifted.append(
                f"{name}: on disk {digest}, registered {rows[name].ondisk_sha256}"
            )
    assert not drifted, (
        "an unchanged copy no longer matches its registered sha256 -- either the "
        f"body was edited or the row is stale: {drifted}"
    )

    bad_digest = sorted(
        f"{name}: {row.origin_sha256!r} / {row.ondisk_sha256!r}"
        for name, row in rows.items()
        if not _SHA256_RE.match(row.origin_sha256)
        or not (
            _SHA256_RE.match(row.ondisk_sha256)
            if row.adoption == "VERBATIM"
            else row.ondisk_sha256 == NO_ONDISK_DIGEST
        )
    )
    assert not bad_digest, (
        f"the origin cell must be a bare sha256, and the on-disk cell a sha256 for "
        f"VERBATIM or {NO_ONDISK_DIGEST!r} for ADAPTED: {bad_digest}"
    )

    bad_adoption = sorted(
        f"{name}: {row.adoption!r}"
        for name, row in rows.items()
        if row.adoption not in ADOPTION_DOMAIN
    )
    assert not bad_adoption, f"adoption outside {ADOPTION_DOMAIN}: {bad_adoption}"

    # A blank cell is the failure this check exists for: it makes "nobody decided
    # when this file stops being skipped" indistinguishable from "never".
    blank = sorted(name for name, row in rows.items() if not row.unskip)
    assert not blank, f"register rows carry an empty un-skip condition: {blank}"

    contradictory = sorted(
        name
        for name, row in rows.items()
        if row.adoption == "VERBATIM" and row.origin_sha256 != row.ondisk_sha256
    )
    assert not contradictory, (
        f"row claims VERBATIM but its two digests differ: {contradictory}"
    )
