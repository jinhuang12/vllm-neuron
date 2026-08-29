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
   ``test/OVERLAY.md``; the **on-disk** sha256 that row records equals the sha256
   of the file on disk; and every row carries a **non-empty** un-quarantine
   disposition. This is what makes "vendored verbatim" checkable instead of
   asserted: it fails every way round -- a vendored file nobody registered, a
   registered file whose body was edited after vendoring, a register row for a
   file that is gone, a row left with a blank disposition, and a row whose cell
   count no longer matches the declared column shape.
"""

from __future__ import annotations

import hashlib
import re
from typing import NamedTuple

import pytest

EXPECTED_TESTPATHS = ("test/unit", "test/vllm_neuron")
EXPECTED_MARKERS = ("fast", "forked", "quarantined")

#: Where vendored upstream files live, relative to the repo root.
UPSTREAM_DIR = "test/vllm_neuron/upstream"
#: The provenance register that must account for every one of them.
OVERLAY_REGISTER = "test/OVERLAY.md"
#: A 64-hex-digit sha256.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
#: The register's declared column shape: the vendored path as subject, then the
#: six columns -- origin PR, origin path, origin sha256, on-disk sha256,
#: adoption disposition, un-quarantine disposition. Read by POSITION, so a shape
#: change fails loudly here instead of silently degrading a check.
REGISTER_CELLS = 7
#: The two dispositions' declared domains.
ADOPTION_DOMAIN = ("VERBATIM", "ADAPTED")
_PERMANENT = "PERMANENT"
#: An increment id, in either the full or the short spelling the plan uses.
_INCREMENT_ID_RE = re.compile(r"\binc-[a-z0-9]+-\d{3}\b|(?<![\w-])-\d{3}\b")


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


class RegisterRow(NamedTuple):
    """One register row, read by column position rather than by pattern search."""

    origin_sha256: str
    ondisk_sha256: str
    adoption: str
    unquarantine: str


def _cell(raw: str) -> str:
    """A table cell's value: markdown emphasis and code quoting are not content."""
    return raw.replace("**", "").replace("`", "").strip()


def _register_rows(register_text: str) -> tuple[dict[str, RegisterRow], list[str]]:
    """Map vendored filename -> its row, plus every row this parser CANNOT read.

    A row's **subject** is its first cell, and a candidate row is one whose first
    cell is a path under the vendored directory. Later cells legitimately mention
    other paths and other digests, so neither "the path" nor "the sha256" can be
    found by scanning the whole line -- the register carries two digests per row
    on purpose, and picking "the first one" would check the wrong column.

    Malformed candidates are **returned, never skipped.** A check that drops the
    rows it cannot parse fails open, which is the worse direction: it reports a
    clean register while checking nothing.
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
            malformed.append(f"{name}: {len(cells)} cells, want {REGISTER_CELLS}")
            continue
        if name in rows:
            malformed.append(f"{name}: registered more than once")
            continue
        rows[name] = RegisterRow(*cells[3:7])
    return rows, malformed


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

    rows, malformed = _register_rows(register.read_text(encoding="utf-8"))
    assert not malformed, (
        f"{OVERLAY_REGISTER} rows do not match the declared column shape, so they "
        f"were not checked at all: {malformed}"
    )

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
        if digest != rows[name].ondisk_sha256:
            drifted.append(
                f"{name}: on disk {digest}, registered {rows[name].ondisk_sha256}"
            )
    assert not drifted, (
        "vendored bodies no longer match their registered on-disk sha256 -- either "
        f"the body was edited or the row is stale: {drifted}"
    )


@pytest.mark.fast
def test_overlay_register_declares_a_disposition_for_every_vendored_file(
    pytestconfig: pytest.Config,
) -> None:
    """No row may carry an empty un-quarantine disposition.

    An empty cell is the failure this check exists for: it is how "we have not
    decided who un-quarantines this file" becomes indistinguishable from "this
    file is permanently quarantined and here is the native coverage instead".
    Both dispositions are checked against their declared domains, and a
    ``VERBATIM`` row's two digests must agree -- if they do not, the row is
    claiming byte-identity it does not have.
    """
    rootpath = pytestconfig.rootpath
    upstream = rootpath / UPSTREAM_DIR
    register = rootpath / OVERLAY_REGISTER
    assert register.is_file(), f"{OVERLAY_REGISTER} is missing"

    if not upstream.is_dir():
        pytest.skip(f"{UPSTREAM_DIR} does not exist yet; nothing is vendored")

    rows, malformed = _register_rows(register.read_text(encoding="utf-8"))
    assert not malformed, f"{OVERLAY_REGISTER} rows are unreadable: {malformed}"
    assert rows, f"{OVERLAY_REGISTER} holds no register row for {UPSTREAM_DIR}"

    blank = sorted(name for name, row in rows.items() if not row.unquarantine)
    assert not blank, (
        "register rows carry an EMPTY un-quarantine disposition; the domain is "
        f"an increment id or the literal {_PERMANENT}: {blank}"
    )

    off_domain = sorted(
        f"{name}: {row.unquarantine!r}"
        for name, row in rows.items()
        if row.unquarantine != _PERMANENT
        and not _INCREMENT_ID_RE.search(row.unquarantine)
    )
    assert not off_domain, (
        "un-quarantine disposition is neither an increment id nor "
        f"{_PERMANENT}: {off_domain}"
    )

    bad_adoption = sorted(
        f"{name}: {row.adoption!r}"
        for name, row in rows.items()
        if row.adoption not in ADOPTION_DOMAIN
    )
    assert not bad_adoption, (
        f"adoption disposition outside {ADOPTION_DOMAIN}: {bad_adoption}"
    )

    bad_digest = sorted(
        f"{name}: {row.origin_sha256!r} / {row.ondisk_sha256!r}"
        for name, row in rows.items()
        if not (
            _SHA256_RE.match(row.origin_sha256) and _SHA256_RE.match(row.ondisk_sha256)
        )
    )
    assert not bad_digest, f"a digest cell is not a bare sha256: {bad_digest}"

    contradictory = sorted(
        name
        for name, row in rows.items()
        if row.adoption == "VERBATIM" and row.origin_sha256 != row.ondisk_sha256
    )
    assert not contradictory, (
        "row claims VERBATIM but its origin and on-disk digests differ: "
        f"{contradictory}"
    )
