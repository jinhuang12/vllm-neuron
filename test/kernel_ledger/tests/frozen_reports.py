# SPDX-License-Identifier: Apache-2.0
"""The reports the tests read: a frozen copy, never the live ``readers.micro.REPORTS_DIR``.

``fixtures/reports/`` holds byte copies (taken 2026-10-06) of the team's records that the
tests pin: the microbenchmark files, ``dsa_kernels.json``, ``moe-t.md`` (a cited source)
and the gate runs up to and including the wt/host merge (594d425). The gate keeps writing
``gate_*.json`` into the live directory, and the ledger picks a gate run by tree and by
write order, so a test that read the live directory changed its answer after every gate.

Read one file: ``FIXTURE_REPORTS / name``. Anything that orders gate runs (``run``,
``read_gates``, ``resolve_tip``) takes the ``reports_dir`` fixture (``conftest.py``)
instead: ``read_gates`` orders gate runs by mtime and a git checkout does not keep
mtimes, so ``copy_reports`` sets them in the order the gate wrote the files.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from test.kernel_ledger.readers.micro import REPORTS_DIR

FIXTURE_REPORTS = Path(__file__).resolve().parents[1] / "fixtures" / "reports"

#: The gate runs in the fixture, oldest first (their write order in the live directory).
GATE_ORDER = ("gate_baseline.json", "gate_kv.json", "gate_tip-a9f86ba.json", "gate_host.json")


def copy_reports(dest: Path) -> Path:
    """Copy the fixture into ``dest``; the gate runs get mtimes in ``GATE_ORDER``."""
    for f in sorted(FIXTURE_REPORTS.iterdir()):
        shutil.copyfile(f, dest / f.name)
    assert sorted(p.name for p in dest.glob("gate_*.json")) == sorted(GATE_ORDER)
    for i, name in enumerate(GATE_ORDER):
        t = 1_759_700_000 + 600 * i
        os.utime(dest / name, (t, t))
    return dest


def frozen(path: Path) -> Path:
    """A file of the live reports directory -> its frozen copy; any other path is unchanged."""
    path = Path(path)
    return FIXTURE_REPORTS / path.name if path.parent == REPORTS_DIR else path
