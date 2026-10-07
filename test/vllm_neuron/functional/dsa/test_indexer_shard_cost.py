# SPDX-License-Identifier: Apache-2.0
"""The arithmetic of the query-sharded selection's cost model (``indexer_shard_cost.py``).

These pin what the report's numbers rest on: the baseline is worker-3's calibrated model
unchanged, degree 1 changes nothing, the top-k fit reproduces its two measured points, the
all-gather is the calibrated all-reduce's own half, and the committed report JSON is what
the model computes today. They read worker-3's directory and never write to it.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from test.vllm_neuron.functional.dsa import indexer_shard_cost as cost_model

REPORT_JSON = "/home/ubuntu/glm53f-wt5/reports/indexer_shard.json"

pytestmark = pytest.mark.skipif(not cost_model.calibration_available(),
                                reason="worker-3's calibrated prefill model is not on this host")


@pytest.fixture(scope="module")
def cost():
    return cost_model.IndexerCost()


def test_the_baseline_is_the_calibrated_table(cost):
    """prefill_calibrated.md's context table, bs=1, TP=64: 688.69 s / 422.83 s at 262,144,
    15.09 s / 6841 ms at 8,192, and the p1 anchor 1966.8 ms."""
    assert cost.ttft(262144, "asbuilt")["before_ms"] == pytest.approx(688_690, abs=10)
    assert cost.ttft(262144, "withbranches")["before_ms"] == pytest.approx(422_830, abs=10)
    assert cost.ttft(8192, "asbuilt")["before_ms"] == pytest.approx(15_090, abs=10)
    assert cost.ttft(8192, "withbranches")["before_ms"] == pytest.approx(6_841, abs=1)
    assert cost.ttft(1024, "asbuilt")["before_ms"] == pytest.approx(1_966.8, abs=0.1)


@pytest.mark.parametrize("cands", [1024, 2048, 16384, 65536])
def test_degree_one_changes_nothing(cost, cands):
    one = cost.query_sharded(cands, 1)
    assert one["compute_after_ms"] == pytest.approx(one["compute_before_ms"], rel=1e-12)
    assert one["comm_added_ms"] == one["glue_added_ms"] == 0.0
    assert one["net_saved_ms"] == pytest.approx(0.0, abs=1e-9)


def test_the_topk_fit_reproduces_both_measured_points(cost):
    for cands in (1024, 2048):
        fit = cost.layers * cost.topk_ms_layer(cost_model.CHUNK, cands)
        assert fit == pytest.approx(cost.m.idx_value("rotational_topk", cands), rel=1e-9)
    # The slope is the clock: one Mcycle at 1.4 GHz is 0.714 ms. A fixed part remains.
    assert cost.beta == pytest.approx(1 / 1.4, rel=0.03)
    assert cost.alpha > 0


def test_the_all_gather_is_half_the_calibrated_all_reduce():
    import sys

    sys.path.insert(0, cost_model.PLANNER_DIR)
    try:
        import entitlement_formulas as ef
    finally:
        sys.path.remove(cost_model.PLANNER_DIR)
    cal = json.load(open(cost_model.CAL_CONSTANTS))
    hw = ef.HW(**{k: v for k, v in cal.items() if not k.startswith("_")})
    for tp in (8, 16, 64):
        for nbytes in (1 << 21, 1 << 23):
            assert 2 * cost_model.allgather_ms(tp, nbytes) == pytest.approx(
                ef.allreduce_ms(hw, tp, nbytes), rel=1e-12)
    # T = 1024 rows of 512 fp32 ids: 2 MiB per layer, about 38 us at TP = 64.
    assert cost_model.allgather_ms(64, 1024 * 512 * 4) == pytest.approx(0.0382, abs=5e-4)


def test_small_tiles_cost_a_whole_tile_in_the_score_and_bound(cost):
    """d = 8 (128 rows) and d = 64 (16 rows) both run one row tile: equal score and bound."""
    for cands in (2048, 65536):
        d8, d64 = cost.query_sharded(cands, 8), cost.query_sharded(cands, 64)
        for bid in ("score_gemm", "causal_bound", "sentinel_order"):
            assert d8["sharded_ms"][bid] == pytest.approx(d64["sharded_ms"][bid], rel=1e-12)
        # Only the top-k and the collective tell them apart; the top-k wins for d = 64.
        assert d64["sharded_ms"]["rotational_topk"] < d8["sharded_ms"]["rotational_topk"]
        assert d64["net_saved_ms"] > d8["net_saved_ms"]


def test_the_saving_grows_with_context_and_beats_its_comm(cost):
    nets = []
    for context in cost_model.CONTEXTS.values():
        q = cost.query_sharded(context // 4, 64)
        assert q["compute_saved_ms"] > 10 * (q["comm_added_ms"] + q["glue_added_ms"])
        nets.append(q["net_saved_ms"])
    assert nets == sorted(nets)


def test_the_ttft_after_is_before_less_the_per_chunk_saving(cost):
    for prompt in cost_model.TTFT_PROMPTS.values():
        row = cost.ttft(prompt, "asbuilt")
        assert row["n_chunks"] == math.ceil(prompt / cost_model.CHUNK)
        assert row["after_ms"] == pytest.approx(
            row["before_ms"] - row["n_chunks"] * row["saved_per_chunk_ms"], rel=1e-12)


def test_the_committed_report_json_is_what_the_model_computes():
    if not os.path.exists(REPORT_JSON):
        pytest.skip(f"{REPORT_JSON} is not written yet")
    fresh = json.loads(json.dumps(cost_model.report(), sort_keys=True, default=float))
    assert json.load(open(REPORT_JSON)) == fresh
