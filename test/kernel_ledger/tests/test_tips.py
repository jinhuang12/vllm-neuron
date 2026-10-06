# SPDX-License-Identifier: Apache-2.0
"""``--tip``: which wave-1 kernels a tip runs, and which gate measured it."""

from __future__ import annotations

import pytest

from test.kernel_ledger.models.glm53f.configs import BASELINE, CURRENT
from test.kernel_ledger.models.glm53f.tips import resolve_tip
from test.kernel_ledger.readers.gate import read_gates
from test.kernel_ledger.readers.micro import REPORTS_DIR


@pytest.fixture(scope="module")
def gates():
    return read_gates(REPORTS_DIR)


def test_5938748_is_the_baseline_measured_by_gate_baseline(gates):
    tip = resolve_tip("5938748", gates)
    assert tip.kernel_set == BASELINE
    assert tip.gate.file.endswith("gate_baseline.json")
    assert tip.merged == ()


def test_current_is_every_wave1_kernel_against_the_latest_gate(gates):
    tip = resolve_tip("current", gates)
    assert tip.kernel_set == CURRENT
    assert tip.gate.file.endswith("gate_host.json")
    # the latest gate's tree runs 5938748 kernels except the on-device sampler
    assert tip.gate_tip.kernel_set.after == frozenset({"host"})


def test_merge_tip_runs_the_merged_families_and_its_merge_gate(gates):
    tip = resolve_tip("594d425", gates)
    assert tip.kernel_set.after == frozenset({"host"})
    assert tip.merged == ("kv", "host")
    assert tip.gate.file.endswith("gate_host.json")


def test_kv_merge_tip_keeps_5938748_kernels_and_its_own_launch(gates):
    tip = resolve_tip("a9f86ba", gates)
    assert tip.kernel_set.after == frozenset()
    assert tip.merged == ("kv",)
    assert tip.gate.file.endswith("gate_tip-a9f86ba.json")


def test_unknown_tip_is_refused(gates):
    with pytest.raises(ValueError):
        resolve_tip("notasha", gates)
