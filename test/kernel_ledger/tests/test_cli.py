# SPDX-License-Identifier: Apache-2.0
"""The command line: the three acceptance invocations and the shape emitter."""

from __future__ import annotations

import json
import re
import shutil

from test.kernel_ledger.cli import main, run
from test.kernel_ledger.models.glm53f.configs import BS1_CTX1K, BS64_CTX8K
from test.kernel_ledger.readers.micro import REPORTS_DIR


def _bucket_row(text, bucket):
    m = re.search(rf"^{re.escape(bucket)}\s+([\d.]+)\s+([\d.]+)", text, re.M)
    assert m, bucket
    return float(m.group(1))


def test_5938748_prints_buckets_reconciliation_and_the_residual_against_77_36():
    out = run("5938748", BS1_CTX1K)
    for b in ("mHC", "KDA", "DSA/MLA", "MoE", "dense", "lm_head", "collectives"):
        assert _bucket_row(out, b) > 0
    assert "Reconciliation with DECODE_BREAKDOWN.md" in out
    assert re.search(r"DECODE_BREAKDOWN.md \(quiet host\): step 77.36 ms -> residual 16\.\d\d ms", out)
    assert "gate_baseline.json (tree 5938748): step 82.57 ms" in out


def test_current_uses_after_medians_and_prints_the_predicted_step():
    out = run("current", BS1_CTX1K)
    assert "every wave-1 kernel" in out
    assert re.search(r"^mhc_pre_attn_sinkhorn .* 23\.30 .* after ", out, re.M)
    assert "predicted device step" in out
    assert "latest gate run: gate_host.json" in out


def test_bs64_prints_a_roofline_row_for_every_node_with_the_expert_term(tmp_path):
    path = tmp_path / "l.json"
    out = run("current", BS64_CTX8K, json_path=path)
    doc = json.loads(path.read_text())
    table = out.split("Per bucket")[0]
    for row in doc["rows"]:
        assert re.search(rf"^{re.escape(row['node'])}\s", table, re.M), row["node"]
        assert row["roofline_us"] > 0
    assert "E[distinct local]=15.03 of 18" in out
    assert "PARTIAL" in out and "no gate run serves bs=64 ctx=8192" in out


def test_emit_shapes_writes_the_file(tmp_path, capsys):
    path = tmp_path / "ledger_shapes.json"
    assert main(["--emit-shapes", str(path), "--tip", "5938748"]) == 0
    doc = json.loads(path.read_text())
    assert [p["point"]["bs"] for p in doc["points"]] == [1, 64]
    assert f"wrote {path}" in capsys.readouterr().out


def test_reports_dir_without_a_family_file_leaves_its_units_unmeasured(tmp_path):
    for f in REPORTS_DIR.glob("*.json"):
        if f.name != "kda_micro.json":
            shutil.copy2(f, tmp_path / f.name)
    out = run("5938748", BS1_CTX1K, reports_dir=tmp_path)
    assert f"reports: {tmp_path} (5 of 6 microbenchmark files; missing: kda_micro.json)" in out
    assert re.search(r"^kda_step\s+KDA\s+missing\s", out, re.M)
    assert "PARTIAL: 14 of 15 measured units" in out
    assert "the sum is partial" in out
