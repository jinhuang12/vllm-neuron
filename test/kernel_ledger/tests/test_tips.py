# SPDX-License-Identifier: Apache-2.0
"""``--tip``: which wave-1 kernels a tip runs, and which gate measured it.

The gate files are copied (mtimes kept) into a temporary directory, so new gate runs
in ``reports/`` do not move the "latest gate" these tests pin.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

from test.kernel_ledger.models.glm53f.configs import BASELINE, CURRENT
from test.kernel_ledger.models.glm53f.tips import resolve_tip
from test.kernel_ledger.readers.gate import read_gates
from test.kernel_ledger.readers.micro import REPORTS_DIR

#: The gate runs up to and including the wt/host merge (594d425).
UP_TO_HOST = ("gate_baseline.json", "gate_kv.json", "gate_tip-a9f86ba.json", "gate_host.json")
MHC_CANDIDATE = "f083375ac3622f5288dfab695fdaea533c59a9c0"  # wt/mhc rebased on 594d425


@pytest.fixture(scope="module")
def gate_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("gates")
    for name in UP_TO_HOST:
        shutil.copy2(REPORTS_DIR / name, d / name)
    return d


@pytest.fixture(scope="module")
def gates(gate_dir):
    return read_gates(gate_dir)


def test_5938748_is_the_baseline_measured_by_gate_baseline(gates):
    tip = resolve_tip("5938748", gates)
    assert tip.kernel_set == BASELINE
    assert tip.gate.file.endswith("gate_baseline.json")
    assert tip.branches == ()


def test_current_is_every_wave1_kernel_against_the_latest_gate(gates):
    tip = resolve_tip("current", gates)
    assert tip.kernel_set == CURRENT
    assert tip.gate.file.endswith("gate_host.json")
    # the latest gate's tree runs 5938748 kernels except the on-device sampler
    assert tip.gate_tip.kernel_set.after == frozenset({"host"})


def test_merge_tip_runs_the_merged_families_and_its_merge_gate(gates):
    tip = resolve_tip("594d425", gates)
    assert tip.kernel_set.after == frozenset({"host"})
    assert tip.branches == ("kv", "host")
    assert tip.gate.file.endswith("gate_host.json")


def test_kv_merge_tip_keeps_5938748_kernels_and_its_own_launch(gates):
    tip = resolve_tip("a9f86ba", gates)
    assert tip.kernel_set.after == frozenset()
    assert tip.branches == ("kv",)
    assert tip.gate.file.endswith("gate_tip-a9f86ba.json")


def test_candidate_tree_runs_its_own_family_whatever_the_verdict(gate_dir, tmp_path):
    for f in gate_dir.iterdir():
        shutil.copy2(f, tmp_path / f.name)
    cand = tmp_path / "gate_mhc.json"
    cand.write_text(json.dumps({"name": "mhc", "verdict": "BLOCKED", "gate_sha": MHC_CANDIDATE,
                                "after": {"device_step_ms": {"mean": 60.0}, "environment": {"head": MHC_CANDIDATE}}}))
    newest = max(os.stat(f).st_mtime for f in gate_dir.iterdir()) + 60
    os.utime(cand, (newest, newest))
    gates = read_gates(tmp_path)
    tip = resolve_tip(MHC_CANDIDATE[:7], gates)
    assert tip.kernel_set.after == frozenset({"host", "mhc"})
    assert tip.branches == ("kv", "host", "mhc")
    assert tip.gate.file == str(cand)
    # the merge tip below it does not hold the candidate
    assert "mhc" not in resolve_tip("594d425", gates).kernel_set.after
    assert resolve_tip("current", gates).gate_tip.kernel_set.after == frozenset({"host", "mhc"})


def test_unknown_tip_is_refused(gates):
    with pytest.raises(ValueError):
        resolve_tip("notasha", gates)


def test_a_non_merged_candidate_does_not_count_below_its_own_tree(gate_dir, tmp_path):
    # ruling: "after" only for MERGE verdicts whose candidate is an ancestor; a REJECTed
    # candidate counts only for its own gated tree
    for f in gate_dir.iterdir():
        shutil.copy2(f, tmp_path / f.name)
    rej = tmp_path / "gate_dsa.json"
    host_sha = resolve_tip("594d425", read_gates(gate_dir)).gate.gate_sha  # 2fd8161, an ancestor of 594d425
    rej.write_text(json.dumps({"name": "dsa", "verdict": "REJECT", "gate_sha": host_sha,
                               "after": {"device_step_ms": {"mean": 70.0}, "environment": {"head": host_sha}}}))
    gates = read_gates(tmp_path)
    assert "dsa" not in resolve_tip("594d425", gates).kernel_set.after
    assert "dsa" in resolve_tip(host_sha, gates).kernel_set.after


def test_a_gated_tree_runs_every_candidate_in_it_whatever_the_verdict(gate_dir, tmp_path):
    # team-lead ruling, round 2: a sha with a gate record of any verdict uses the kernels of the
    # tree that gate measured; the MERGE-only ancestry rule is for "current" and ungated shas
    for f in gate_dir.iterdir():
        shutil.copy2(f, tmp_path / f.name)
    host_sha = resolve_tip("594d425", read_gates(gate_dir)).gate.gate_sha  # 2fd8161, below f083375
    (tmp_path / "gate_dsa.json").write_text(json.dumps(
        {"name": "dsa", "verdict": "REJECT", "gate_sha": host_sha,
         "after": {"device_step_ms": {"mean": 70.0}, "environment": {"head": host_sha}}}))
    (tmp_path / "gate_mhc.json").write_text(json.dumps(
        {"name": "mhc", "verdict": "BLOCKED", "gate_sha": MHC_CANDIDATE,
         "after": {"device_step_ms": {"mean": 60.0}, "environment": {"head": MHC_CANDIDATE}}}))
    gates = read_gates(tmp_path)
    assert resolve_tip(MHC_CANDIDATE[:7], gates).kernel_set.after == frozenset({"host", "mhc", "dsa"})
    assert resolve_tip("594d425", gates).kernel_set.after == frozenset({"host"})  # no gate record of its own
