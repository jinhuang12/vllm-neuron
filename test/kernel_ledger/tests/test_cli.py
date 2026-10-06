# SPDX-License-Identifier: Apache-2.0
"""The command line: the acceptance invocations (as restated by team-lead) and the shape emitter."""

from __future__ import annotations

import json
import re
import shutil

import pytest

from test.kernel_ledger.cli import main, run
from test.kernel_ledger.models.glm53f.configs import BS1_CTX1K, BS64_CTX8K


def _bucket_row(text, bucket):
    m = re.search(rf"^{re.escape(bucket)}\s+([\d.]+)\s+([\d.]+)", text, re.M)
    assert m, bucket
    return float(m.group(1))


@pytest.fixture(scope="module")
def base_run(tmp_path_factory, reports_dir):
    path = tmp_path_factory.mktemp("j") / "base.json"
    return run("5938748", BS1_CTX1K, reports_dir, json_path=path), json.loads(path.read_text())


def test_5938748_prints_every_bucket(base_run):
    out, _ = base_run
    for b in ("mHC", "KDA", "DSA/MLA", "MoE", "dense", "lm_head", "collectives"):
        assert _bucket_row(out, b) > 0


def test_5938748_reconciliation_verdicts_follow_the_ruling(base_run):
    out, doc = base_run
    verdicts = {r["bucket"]: r["verdict"] for r in doc["reconciliation"]["rows"]}
    assert verdicts == {"mHC": "PASS", "KDA": "PASS", "DSA/MLA": "PASS", "lm_head": "PASS",
                        "MoE": "FAIL", "dense": "FAIL", "collectives": "FAIL"}
    for r in doc["reconciliation"]["rows"]:
        assert (r["verdict"] == "FAIL") == r["cause"].startswith("scope:")
        assert abs(r["delta_pct"]) <= 15.0 or r["verdict"] == "FAIL"
    assert "PASS 4 of 7; every verdict as expected" in out
    assert out.count("FAIL cause: scope:") == 3


def test_kda_verdict_prints_its_scope_range(base_run):
    # da-2 round 1 #1: the KDA PASS depends on how much KDA-layer glue is in scope
    out, doc = base_run
    kda = next(r for r in doc["reconciliation"]["rows"] if r["bucket"] == "KDA")
    assert (round(kda["reference_ms"], 2), round(kda["delta_pct"], 1), kda["verdict"]) == (12.3, -10.4, "PASS")
    alts = {round(a["reference_ms"], 2): a["verdict"] for a in kda["alternatives"]}
    assert alts == {12.47: "PASS", 13.45: "FAIL"}
    assert all(abs(a["delta_pct"] - (kda["ledger_ms"] / a["reference_ms"] - 1) * 100) < 1e-9
               for a in kda["alternatives"])
    assert re.search(r"alternative reference: 13\.45 ms .*: -18\.1% FAIL", out)
    assert "scope-dependent verdict: KDA (PASS at 12.30, 12.47 ms; FAIL at 13.45 ms)" in out
    assert "un-modeled (lands in the residual): 0.975 ms KDA-layer glue (ACT, DVE) (DECODE_BREAKDOWN.md)" in out
    assert kda["unmodeled_ms"] == pytest.approx(0.975)
    # the residual lines name it
    assert re.search(r"residual holds the listed un-modeled in-model terms: 0\.975 ms \(KDA: KDA-layer glue", out)


def test_5938748_residual_against_both_steps_with_labels(base_run):
    out, _ = base_run
    assert re.search(r"breakdown, full host \(DECODE_BREAKDOWN.md\): step 77.36 ms -> residual \d+\.\d\d ms vs "
                     r"calibrated \(\d+\.\d%\); 16\.\d\d ms vs raw", out)
    assert re.search(r"gate baseline, CPU split \(gate_baseline.json, tree 5938748\): step 82.57 ms", out)


def test_sum_line_is_calibrated_and_the_raw_sum_is_shown(base_run):
    out, doc = base_run
    m = re.search(r"^sum of measured kernels \+ collectives, calibrated \(PREDICTOR, not a test\): ([\d.]+) ms", out, re.M)
    r = re.search(r"^sum of measured kernels \+ collectives, raw benchmark medians: ([\d.]+) ms", out, re.M)
    assert m and r
    assert float(m.group(1)) == pytest.approx(doc["calibrated_ms"], abs=0.006)
    assert float(r.group(1)) == pytest.approx(doc["raw_ms"], abs=0.006)
    assert "scope gap raw - calibrated" in out


def test_current_uses_after_medians_and_prints_the_predicted_step(reports_dir):
    out = run("current", BS1_CTX1K, reports_dir)
    assert "every wave-1 kernel" in out
    assert re.search(r"^mhc_pre_attn_sinkhorn .* 23\.30 .* after ", out, re.M)
    assert re.search(r"^latest gate run: gate_\S+\.json \(tree [0-9a-f]{7}[^)]*\): device step [\d.]+ ms", out, re.M)
    assert re.search(r"^ledger at tree [0-9a-f]{7}: calibrated [\d.]+ ms, raw [\d.]+ ms", out, re.M)
    assert "predicted device step for current" in out


def test_current_predicts_with_the_calibrated_sum_plus_a_residual(reports_dir):
    out = run("current", BS1_CTX1K, reports_dir)
    s = re.search(r"^sum of measured kernels \+ collectives, calibrated \(PREDICTOR, not a test\): ([\d.]+) ms", out, re.M)
    for label in ("the latest gate run's residual", "the 5938748 residual vs 77.36 ms (breakdown, full host)",
                  "the 5938748 residual vs 82.57 ms (gate baseline, CPU split)"):
        m = re.search(rf"^  ([\d.]+) ms with {re.escape(label)}[^:]*: ([\d.]+) ms", out, re.M)
        assert m, label
        assert abs(float(m.group(1)) - float(s.group(1)) - float(m.group(2))) < 0.011  # 2-decimal rounding


def test_merged_tip_shows_the_measured_profile_next_to_the_predictor(reports_dir):
    out = run("594d425", BS1_CTX1K, reports_dir)
    assert "Measured in-model per bucket (gate_host.json profile, tree 2fd8161)" in out
    m = re.search(r"^mHC\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)", out.split("Measured in-model")[1], re.M)
    assert m and float(m.group(3)) == pytest.approx(16.766, abs=0.001)


def test_bs64_prints_a_roofline_row_for_every_node_with_the_expert_term(tmp_path, reports_dir):
    path = tmp_path / "l.json"
    out = run("current", BS64_CTX8K, reports_dir, json_path=path)
    doc = json.loads(path.read_text())
    table = out.split("Per bucket")[0]
    for row in doc["rows"]:
        assert re.search(rf"^{re.escape(row['node'])}\s", table, re.M), row["node"]
        assert row["roofline_us"] > 0
    assert "E[distinct local]=15.03 of 18" in out
    assert "PARTIAL: 7 of 15 measured units" in out and "no gate run serves bs=64 ctx=8192" in out
    assert "+ mHC batch files: mhc_micro_mid.json, mhc_micro_large.json" in out


def test_emit_shapes_writes_the_file(tmp_path, capsys, reports_dir):
    path = tmp_path / "ledger_shapes.json"
    assert main(["--emit-shapes", str(path), "--tip", "5938748", "--reports-dir", str(reports_dir)]) == 0
    doc = json.loads(path.read_text())
    assert [p["point"]["bs"] for p in doc["points"]] == [1, 64]
    assert f"wrote {path}" in capsys.readouterr().out


def test_reports_dir_without_a_family_file_leaves_its_units_unmeasured(tmp_path, reports_dir):
    for f in reports_dir.glob("*.json"):
        if f.name != "kda_micro.json":
            shutil.copy2(f, tmp_path / f.name)
    out = run("5938748", BS1_CTX1K, reports_dir=tmp_path)
    assert f"reports: {tmp_path} (5 of 6 microbenchmark files; missing: kda_micro.json)" in out
    assert re.search(r"^kda_step\s+KDA\s+missing\s", out, re.M)
    assert "PARTIAL: 14 of 15 measured units" in out
    assert "the sum is partial" in out


def test_not_wired_note_only_when_its_family_is_merged(reports_dir):
    assert "not wired" in run("current", BS1_CTX1K, reports_dir).splitlines()[1]
    head = run("594d425", BS1_CTX1K, reports_dir).splitlines()[1]
    assert "for host;" in head and "not wired" not in head
