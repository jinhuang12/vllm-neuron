# SPDX-License-Identifier: Apache-2.0
"""Readers of the team's microbenchmark and gate JSONs (the real files are the fixtures)."""

from __future__ import annotations

import json

import pytest

from test.kernel_ledger.readers.gate import latest_gate, read_gate_step, read_gates
from test.kernel_ledger.readers.micro import (
    config_name,
    load_micro_results,
    parse_carrier,
)
from test.kernel_ledger.tests.frozen_reports import FIXTURE_REPORTS


@pytest.fixture(scope="module")
def results():
    return load_micro_results(FIXTURE_REPORTS)


def _get(results, kernel, variant, **record):
    key = config_name(kernel, record)
    hit = results.get((key, variant))
    assert hit is not None, f"no {variant} result for {key}; have {sorted(k for k, _ in results if kernel in k)}"
    return hit


def test_config_name_is_order_independent():
    assert config_name("k", {"B": 1, "H": 2}) == config_name("k", {"H": 2, "B": 1}) == "k|B=1,H=2"


def test_parse_carrier_strips_the_layout_note():
    assert parse_carrier("bfloat16 [B, 3, 384] (SD)", 4) == {"dtype": "bfloat16", "shape": [4, 3, 384]}
    assert parse_carrier("float32 [B, 1, 128, 128]", 1) == {"dtype": "float32", "shape": [1, 1, 128, 128]}


def test_mhc_reader_splits_sinkhorn_and_combine(results):
    rec = dict(B=1, hidden=4096, streams=4, iters=20)
    assert _get(results, "mhc_sinkhorn_tkg", "before", **rec).latency_us == pytest.approx(101.1024375)
    assert _get(results, "mhc_sinkhorn_tkg", "after", **rec).latency_us == pytest.approx(23.3014375)
    comb = dict(B=1, hidden=4096, streams=4)
    assert _get(results, "mhc_combine_tkg", "before", **comb).latency_us == pytest.approx(87.571125)
    assert _get(results, "mhc_combine_tkg", "before", **comb).iterations == 50


def test_kda_reader_uses_the_34_layer_graph(results):
    rec = dict(B=1, heads_per_rank=1, head_dim=128, conv_taps=4, conv_channels=384)
    before = _get(results, "kda_decode_tkg", "before", **rec)
    assert before.latency_us == pytest.approx(324.09398529411766)
    assert before.p90_us == pytest.approx(326.7649705882353)
    assert _get(results, "kda_decode_tkg", "after", **rec).latency_us == pytest.approx(32.72491176470588)
    assert before.record["conv_carrier"] == {"dtype": "bfloat16", "shape": [1, 3, 384]}


def test_dsa_reader_keys_each_variant_by_its_window(results):
    common = dict(batch=1, ctx=1024, max_seq_len=4096)
    served = _get(results, "dsa_mla_layer_tkg", "before", window_rows=4096, **common)
    assert served.latency_us == pytest.approx(1160.477409090909)
    # two cases time the after code at window 2048: the bypass case is the value (the dsa.md
    # headline); the bypass_vs_default_window case repeats it and is kept as a repeat only
    after = _get(results, "dsa_mla_layer_tkg", "after", window_rows=2048, **common)
    assert after.latency_us == pytest.approx(286.63827272727275)
    assert after.sources == ("dsa_micro.json#layer/bypass/B1",)
    assert after.repeats == (("dsa_micro.json#layer/bypass_vs_default_window/B1", pytest.approx(301.07877272727274)),)
    bucketed = _get(results, "dsa_mla_layer_tkg", "before", window_rows=2048, **common)
    assert bucketed.latency_us == pytest.approx(1112.3295)


def test_moe_reader_keeps_the_hit_curve(results):
    shp = dict(H=4096, E_global=288, top_k=8)
    assert _get(results, "rmsnorm_router_topk_tkg", "before", T=1, **shp).latency_us == pytest.approx(78.88364285714286)
    experts = _get(results, "moe_experts_tkg", "before", T=1, E_local=18, I_local=512, **shp)
    assert experts.points == ((0, pytest.approx(131.85414285714288)), (1, pytest.approx(229.0562142857143)),
                              (2, pytest.approx(228.39635714285714)))
    t64 = _get(results, "moe_experts_tkg", "after", T=64, E_local=18, I_local=512, **shp)
    assert t64.points == ((15, pytest.approx(874.7079285714286)),)


def test_dense_reader(results):
    assert _get(results, "shared_expert_tkg", "before", M=1, H=4096, I=128).latency_us == pytest.approx(112.98884375)
    assert _get(results, "dense_mlp_tkg", "after", M=1, H=4096, I=256).latency_us == pytest.approx(30.2616875)
    lm = dict(M=1, H=4096, vocab=154880, shard_rows=2420)
    assert _get(results, "lm_head", "before", **lm).latency_us == pytest.approx(2196.95)
    assert _get(results, "lm_head", "after", **lm).latency_us == pytest.approx(37.1406875)
    assert _get(results, "rmsnorm_tkg", "before", M=1, H=4096).latency_us == pytest.approx(5.3186875)


def test_sampler_reader_marks_the_host_path(results):
    rec = dict(B=1, vocab=154880, logits_dtype="bfloat16", mode="all_greedy")
    before = _get(results, "sampling", "before", **rec)
    after = _get(results, "sampling", "after", **rec)
    assert (before.location, before.latency_us) == ("host", pytest.approx(335.8415))
    assert (after.location, after.latency_us) == ("device", pytest.approx(146.31900000000002))


def test_missing_reports_dir_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_micro_results(tmp_path / "nope")


def test_gate_reader_handles_both_schemas():
    base = read_gate_step(FIXTURE_REPORTS / "gate_baseline.json")
    assert base.device_step_ms == pytest.approx(82.56535485714285)
    assert base.head.startswith("5938748")
    host = read_gate_step(FIXTURE_REPORTS / "gate_host.json")
    assert host.device_step_ms == pytest.approx(74.119792)
    assert (host.name, host.verdict) == ("host", "MERGE")
    assert host.gate_sha.startswith("2fd8161")


def test_latest_gate_is_the_newest_file(tmp_path):
    for name, step, mtime in (("gate_a.json", 80.0, 100), ("gate_b.json", 70.0, 200)):
        p = tmp_path / name
        p.write_text(json.dumps({"label": name, "device_step_ms": {"mean": step}}))
        import os

        os.utime(p, (mtime, mtime))
    gates = read_gates(tmp_path)
    assert latest_gate(gates).device_step_ms == 70.0


@pytest.mark.parametrize("name", ["gate_baseline.json", "gate_kv.json"])
def test_gate_reader_keeps_the_profile_buckets(name):
    g = read_gate_step(FIXTURE_REPORTS / name)
    d = json.loads((FIXTURE_REPORTS / name).read_text())
    raw = (d.get("after") or d)["device_step_ms"]["buckets_ms"]
    assert g.buckets_ms == raw
    # attribute_decode.py splits the whole step: the buckets add up to the mean step
    assert sum(g.buckets_ms.values()) == pytest.approx(g.device_step_ms, rel=1e-4)


def test_gate_profile_values():
    base = read_gate_step(FIXTURE_REPORTS / "gate_baseline.json")
    kv = read_gate_step(FIXTURE_REPORTS / "gate_kv.json")
    assert base.buckets_ms["mhc/hyper_connection.py"] == pytest.approx(9.658, abs=1e-3)
    assert kv.buckets_ms["wait: collective"] == pytest.approx(2.9534285714285713)


def test_cases_of_one_config_are_averaged_unless_marked_as_repeats():
    from test.kernel_ledger.readers.micro import _Collector
    rec = dict(B=1, hidden=4096, streams=4, iters=20)
    c = _Collector()
    c.add("mhc_sinkhorn_tkg", "after", rec, 10.0, p90=None, iterations=5, source="a#1")
    c.add("mhc_sinkhorn_tkg", "after", rec, 20.0, p90=None, iterations=5, source="b#1")
    c.add("mhc_sinkhorn_tkg", "after", rec, 99.0, p90=None, iterations=5, source="c#1", repeat=True)
    (hit,) = c.results().values()
    assert hit.latency_us == 15.0 and hit.sources == ("a#1", "b#1") and hit.repeats == (("c#1", 99.0),)


def test_mhc_batch_files_are_read(results):
    rec = dict(hidden=4096, streams=4, iters=20)
    for b, f in ((8, "mhc_micro_mid.json"), (32, "mhc_micro_mid.json"), (64, "mhc_micro_large.json")):
        hit = _get(results, "mhc_sinkhorn_tkg", "after", B=b, **rec)
        assert hit.sources == (f"{f}#B{b}/sinkhorn",)
    assert _get(results, "mhc_sinkhorn_tkg", "before", B=64, **rec).latency_us == pytest.approx(124.18975)
