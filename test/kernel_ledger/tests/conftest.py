# SPDX-License-Identifier: Apache-2.0
"""``reports_dir``: the frozen reports (``frozen_reports.py``), gate runs in write order."""

from __future__ import annotations

from pathlib import Path

import pytest

from test.kernel_ledger.tests.frozen_reports import copy_reports


@pytest.fixture(scope="session")
def reports_dir(tmp_path_factory) -> Path:
    return copy_reports(tmp_path_factory.mktemp("reports"))
