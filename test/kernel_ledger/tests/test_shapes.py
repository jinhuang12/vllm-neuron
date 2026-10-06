# SPDX-License-Identifier: Apache-2.0
"""Emitted per-kernel shapes match the shapes the hardware benchmarks recorded.

The expected values are read straight from the benchmark JSONs (not via the ledger's
readers), field by field, at the decode batches the ledger uses (1 and 4; T=64 for the
experts). The other batches and fields are not checked here.
"""

from __future__ import annotations

import json

import pytest

from test.kernel_ledger.models.glm53f.configs import BS1_CTX1K, BS64_CTX8K, DecodePoint
from test.kernel_ledger.models.glm53f.shapes import emit_shapes
from test.kernel_ledger.readers.micro import REPORTS_DIR


def _json(name):
    return json.loads((REPORTS_DIR / name).read_text())


def _entries(point, kernel, kernel_set=None):
    doc = emit_shapes([point])
    (pt,) = doc["points"]
    return [e for e in pt["entries"] if e["kernel"] == kernel and e.get("kernel_set") in (None, kernel_set)]


def _one(point, kernel, kernel_set=None):
    found = _entries(point, kernel, kernel_set)
    assert found, f"no {kernel} entry"
    shapes = {json.dumps(e["shape"], sort_keys=True) for e in found}
    assert len(shapes) == 1, f"{kernel}: entries disagree: {shapes}"
    return found[0]["shape"]


@pytest.mark.parametrize("fname,b", [("mhc_micro.json", 1), ("mhc_micro.json", 4), ("mhc_micro_mid.json", 8),
                                     ("mhc_micro_mid.json", 16), ("mhc_micro_mid.json", 32),
                                     ("mhc_micro_large.json", 64), ("mhc_micro_large.json", 128)])
def test_mhc_shapes_match_mhc_micro(fname, b):
    case = next(c for c in _json(fname)["cases"] if c["B"] == b)
    pt = DecodePoint(bs=b, ctx=1024, max_model_len=4096)
    for kernel in ("mhc_sinkhorn_tkg", "mhc_combine_tkg"):
        shape = _one(pt, kernel)
        for k in ("B", "hidden", "streams"):
            assert shape[k] == case[k]
    assert _one(pt, "mhc_sinkhorn_tkg")["iters"] == case["iters"]


@pytest.mark.parametrize("b", [1, 4])
def test_kda_shapes_match_kda_micro(b):
    d = _json("kda_micro.json")
    s = d["shapes"]
    shape = _one(DecodePoint(bs=b, ctx=1024, max_model_len=4096), "kda_decode_tkg")
    for k in ("heads_per_rank", "head_dim", "conv_taps", "conv_channels"):
        assert shape[k] == s[k]
    assert any(c["B"] == b for c in d["cases"])
    # "bfloat16 [B, 3, 384] (SD)", "float32 [B, 1, 128, 128]"
    conv = s["conv_carrier"].split("[")[1].split("]")[0].replace("B", str(b))
    rec = s["recurrent_carrier"].split("[")[1].split("]")[0].replace("B", str(b))
    assert shape["conv_carrier"] == {"dtype": s["conv_carrier"].split()[0], "shape": [int(x) for x in conv.split(",")]}
    assert shape["recurrent_carrier"] == {"dtype": s["recurrent_carrier"].split()[0],
                                          "shape": [int(x) for x in rec.split(",")]}


@pytest.mark.parametrize("b", [1, 4])
def test_dsa_layer_shapes_match_dsa_micro(b):
    case = next(c for c in _json("dsa_micro.json")["cases"]
                if c["table"] == "layer" and c["case"] == "bypass_vs_default_window" and c["batch"] == b) \
        if b == 1 else next(c for c in _json("dsa_micro.json")["cases"]
                            if c["table"] == "layer" and c["case"] == "bypass" and c["batch"] == b)
    pt = DecodePoint(bs=b, ctx=case["ctx"], max_model_len=case["max_seq_len"])
    for kernel_set, variant in (("5938748", "before"), ("current", "after")):
        shape = _one(pt, "dsa_mla_layer_tkg", kernel_set)
        assert (shape["batch"], shape["ctx"], shape["max_seq_len"]) == (case["batch"], case["ctx"], case["max_seq_len"])
        if b == 1 or kernel_set == "current":
            assert shape["window_rows"] == case["window_rows"][variant]


def test_dsa_kernel_table_shapes_match_dsa_kernels():
    cases = _json("dsa_kernels.json")["cases"]
    shapes = {e["shape"]["case"]: e["shape"] for e in _entries(BS1_CTX1K, "dsa_projection")}
    projections = [c for c in cases if c["table"] == "projection" and c["rows"] == 1]
    assert {c["case"] for c in projections} == set(shapes)
    for c in projections:
        assert shapes[c["case"]] == {"case": c["case"], "in": c["in"], "out": c["out"], "rows": 1, "weight": c["weight"]}
    att = next(c for c in cases if c["table"] == "attention" and c["case"] == "dense" and c["batch"] == 1)
    # the attention table times the wave-1 DSA kernels: the "current" window
    (shape,) = [e["shape"] for e in _entries(BS1_CTX1K, "dsa_attention", "current")]
    assert shape == {k: att[k] for k in ("case", "ctx", "window_rows", "indices", "batch")}
    sel = next(c for c in cases if c["table"] == "attention" and c["case"] == "selected" and c["batch"] == 1)
    (shape,) = [e["shape"] for e in _entries(DecodePoint(bs=1, ctx=4096, max_model_len=4096), "dsa_attention",
                                             "current")]
    assert shape == {k: sel[k] for k in ("case", "ctx", "window_rows", "indices", "batch")}


@pytest.mark.parametrize("t", [1, 4, 64])
def test_moe_shapes_match_moe_micro(t):
    d = _json("moe-t_micro.json")
    s = d["shapes"]
    pt = DecodePoint(bs=t, ctx=1024, max_model_len=4096)
    router = _one(pt, "rmsnorm_router_topk_tkg")
    experts = _one(pt, "moe_experts_tkg")
    assert any(c["family"] == "router" and c["T"] == t for c in d["cases"])
    assert any(c["family"] == "experts" and c["T"] == t for c in d["cases"])
    for k in ("H", "E_global", "top_k"):
        assert router[k] == s[k] and experts[k] == s[k]
    assert (experts["E_local"], experts["I_local"]) == (s["E_local"], s["I_local"])
    assert router["T"] == experts["T"] == t


def test_moe_expected_distinct_experts_at_64_tokens_is_the_benchmarked_scenario():
    d = _json("moe-t_micro.json")
    case = next(c for c in d["cases"] if c["family"] == "experts" and c["T"] == 64)
    experts = _one(DecodePoint(bs=64, ctx=8192, max_model_len=8192), "moe_experts_tkg")
    assert round(experts["expected_distinct_local_experts"]) == case["distinct_local_experts_per_layer"][0]


@pytest.mark.parametrize("b", [1, 4])
def test_dense_shapes_match_dense_micro(b):
    cases = {c["case"]: c for c in _json("dense_micro.json")["cases"]}
    pt = DecodePoint(bs=b, ctx=1024, max_model_len=4096)
    shared = cases[f"shared_mlp_b{b}"]
    assert _one(pt, "shared_expert_tkg") == {k: shared[k] for k in ("M", "H", "I")}
    if b == 1:
        for case, kernel, keys in (("dense_mlp_b1", "dense_mlp_tkg", ("M", "H", "I")),
                                   ("lm_head_b1", "lm_head", ("M", "H", "vocab", "shard_rows")),
                                   ("norm_b1", "rmsnorm_tkg", ("M", "H"))):
            assert _one(pt, kernel) == {k: cases[case][k] for k in keys}


@pytest.mark.parametrize("b", [1, 4, 64])
def test_sampler_shapes_match_host_sampler_micro(b):
    case = next(c for c in _json("host_sampler_micro.json")["cases"] if c["B"] == b and c["mode"] == "all_greedy")
    shape = _one(DecodePoint(bs=b, ctx=1024, max_model_len=4096), "sampling")
    assert shape == {k: case[k] for k in ("B", "vocab", "logits_dtype", "mode")}


def test_emit_shapes_covers_both_operating_points_and_every_measured_kernel():
    doc = emit_shapes([BS1_CTX1K, BS64_CTX8K])
    assert [p["point"]["bs"] for p in doc["points"]] == [1, 64]
    kernels = {e["kernel"] for e in doc["points"][1]["entries"]}
    assert {"mhc_sinkhorn_tkg", "mhc_combine_tkg", "kda_decode_tkg", "dsa_mla_layer_tkg", "rmsnorm_tkg",
            "dense_mlp_tkg", "shared_expert_tkg", "rmsnorm_router_topk_tkg", "moe_experts_tkg", "lm_head",
            "sampling", "dsa_projection", "dsa_attention"} <= kernels
    for e in doc["points"][0]["entries"]:
        assert e["layer_count"] >= 1 and e["benchmark"].endswith(".json")
