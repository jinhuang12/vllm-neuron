# SPDX-License-Identifier: Apache-2.0
"""Measurement binding and rollup of the GLM-5.3-Flash decode ledger."""

from __future__ import annotations

import pytest

from test.kernel_ledger.engine.collectives import ConstantCollectiveModel
from test.kernel_ledger.models.glm53f.configs import BS1_CTX1K, BS64_CTX8K, BASELINE, CURRENT, KernelSet
from test.kernel_ledger.models.glm53f.ledger import build_ledger, interpolate_points
from test.kernel_ledger.readers.micro import load_micro_results
from test.kernel_ledger.tests.frozen_reports import FIXTURE_REPORTS

AR_US = 17.5 + 16384 / 100e9 * 1e6  # one fp32 [1, 4096] all-reduce


@pytest.fixture(scope="module")
def results():
    return load_micro_results(FIXTURE_REPORTS)


@pytest.fixture(scope="module")
def base(results):
    return build_ledger(BS1_CTX1K, BASELINE, results)


@pytest.fixture(scope="module")
def cur(results):
    return build_ledger(BS1_CTX1K, CURRENT, results)


def test_interpolation_is_linear_inside_and_nearest_within_half_a_hit():
    pts = ((0, 100.0), (1, 200.0), (2, 200.0))
    assert interpolate_points(pts, 0.5) == pytest.approx(150.0)
    assert interpolate_points(pts, 2.0) == pytest.approx(200.0)
    assert interpolate_points(((15, 874.7),), 15.03) == pytest.approx(874.7)
    assert interpolate_points(((15, 874.7),), 16.0) is None
    assert interpolate_points(pts, 3.0) is None


def test_a_repeat_case_is_shown_but_not_used(cur):
    # team-lead round 2: no duplicate averaging for DSA; the repeat stays visible in the note
    row = cur.row("mla_sparse")
    assert row.source == "dsa_micro.json#layer/bypass/B1"
    assert row.measured_us == pytest.approx(286.63827272727275)
    assert "repeat not used: dsa_micro.json#layer/bypass_vs_default_window/B1 301.08 us" in row.note


def test_a_row_from_several_cases_says_it_is_their_mean():
    from test.kernel_ledger.models.glm53f.ledger import _sources
    from test.kernel_ledger.readers.emf import KernelResult
    hit = KernelResult(config_name="x", kernel_api="k", latency_us=1.0, test_name="a", sources=("a#1", "b#1"))
    assert _sources(hit) == "mean of 2 cases: a#1 + b#1"


def test_kernel_set_variants():
    assert BASELINE.variant("mhc_sinkhorn_tkg") == "before"
    assert CURRENT.variant("mhc_sinkhorn_tkg") == "after"
    # the NKI RMSNorm is measured but not wired (dense.md): current keeps the compiler lowering
    assert CURRENT.variant("rmsnorm_tkg") == "before"
    host_only = KernelSet("594d425", frozenset({"host"}))
    assert (host_only.variant("sampling"), host_only.variant("lm_head")) == ("after", "before")


def test_mhc_bucket_is_ninety_sinkhorns_plus_ninety_combines(base):
    assert base.buckets()["mHC"].measured_ms == pytest.approx(90 * (101.1024375 + 87.571125) / 1e3)


def test_kda_and_dsa_rows_take_their_kernel_set_window(base, cur):
    assert base.row("kda_step").measured_us == pytest.approx(324.09398529411766)
    assert base.row("mla_sparse").measured_us == pytest.approx(1160.477409090909)  # served 4096-row window
    assert cur.row("mla_sparse").measured_us == pytest.approx(286.63827272727275)  # the bypass case; its repeat is not used
    assert cur.row("mla_sparse").layer_count == 11


def test_experts_are_read_at_the_expected_distinct_local_experts(base, cur):
    row = base.row("moe_experts")
    assert row.note.startswith("E[distinct local]=0.50")
    assert row.measured_us == pytest.approx((131.85414285714288 + 229.0562142857143) / 2)
    assert cur.row("moe_experts").measured_us == pytest.approx((35.41585714285714 + 58.88735714285714) / 2)


def test_norms_keep_the_5938748_lowering_in_current(cur):
    row = cur.row("attn_norm")
    assert (row.variant, row.measured_us) == ("before", pytest.approx(5.3186875))
    assert "not wired" in row.note


def test_5938748_sampler_is_host_time_outside_the_device_step(base):
    row = base.row("sampler")
    assert (row.location, row.measured_us) == ("host", pytest.approx(335.8415))
    assert base.host_ms == pytest.approx(0.3358415)
    assert base.buckets()["sampler"].measured_ms == 0.0


def test_current_sampler_runs_on_device(cur):
    assert cur.buckets()["sampler"].measured_ms == pytest.approx(0.146319)
    assert cur.host_ms == 0.0


def test_logit_gather_runs_only_with_the_vocab_parallel_lm_head(base, cur):
    assert base.row("lm_head_gather").kind == "inactive"
    assert base.buckets()["collectives"].measured_ms == pytest.approx(90 * AR_US / 1e3)
    gather = cur.row("lm_head_gather")
    assert gather.measured_us == pytest.approx(17.5 + 64 * 2420 * 2 / 100e9 * 1e6)
    assert cur.buckets()["collectives"].measured_ms == pytest.approx((90 * AR_US + gather.measured_us) / 1e3)


def test_all_reduce_rows_use_the_constant_collective_model(base):
    row = base.row("ar_attn")
    assert (row.kind, row.layer_count, row.bytes) == ("collective", 45, 16384)
    assert row.measured_us == pytest.approx(ConstantCollectiveModel().lookup_us("all_reduce", None, 16384))


def test_fused_members_keep_their_roofline_and_carry_no_time(base):
    for name in ("mla_x_proj", "mla_q_b", "idx_q_proj", "idx_x_proj", "dsa_indexer", "mla_o_proj"):
        row = base.row(name)
        assert row.kind == "fused" and row.measured_us is None and row.source == "in mla_sparse"
    assert base.row("mla_x_proj").roofline_us > 2.0
    assert base.row("dsa_indexer").roofline_us == 2.0  # bypass at ctx 1024: zero work, the floor


def test_unmeasured_compiler_ops_are_roofline_only(base):
    row = base.row("mhc_pre_attn_fn")
    assert (row.kind, row.measured_us, row.layer_count) == ("unmeasured", None, 45)
    assert base.unmeasured_roofline_ms == pytest.approx(
        sum(r.roofline_ms for r in base.rows if r.kind in ("unmeasured", "fused")))


def test_rollup_sums_buckets_and_states_the_residual(base):
    b = base.buckets()
    assert base.measured_ms == pytest.approx(sum(t.measured_ms for t in b.values()))
    assert base.measured_ms == pytest.approx(base.kernels_ms + base.collectives_ms)
    assert base.residual_ms(77.36) == pytest.approx(77.36 - base.measured_ms)
    assert base.complete and base.missing == []


def test_bs1_baseline_bucket_totals(base):
    b = base.buckets()
    assert b["KDA"].measured_ms == pytest.approx(34 * 324.09398529411766 / 1e3)
    assert b["MoE"].measured_ms == pytest.approx(42 * (78.88364285714286 + (131.85414285714288 + 229.0562142857143) / 2) / 1e3)
    assert b["dense"].measured_ms == pytest.approx((42 * 112.98884375 + 3 * 144.19803125 + 91 * 5.3186875) / 1e3, rel=1e-4)
    assert b["lm_head"].measured_ms == pytest.approx(2.19695)


def test_bs64_point_is_roofline_for_every_node_with_the_expert_term(results):
    led = build_ledger(BS64_CTX8K, CURRENT, results)
    assert all(r.roofline_us >= 2.0 or r.kind in ("collective", "inactive") for r in led.rows)
    experts = led.row("moe_experts")
    assert experts.note.startswith("E[distinct local]=15.03")
    assert experts.measured_us == pytest.approx(874.7079285714286)  # the 15-distinct benchmark, T=64
    assert led.row("mla_sparse").measured_us is None and led.row("mla_sparse").kind == "missing"
    assert not led.complete and "mla_sparse" in led.missing
    assert led.row("dsa_indexer").note.startswith("selected")
    # mhc_micro_large.json has B=64: the four mHC rows are measured
    for n in ("mhc_pre_attn_sinkhorn", "mhc_post_attn", "mhc_pre_mlp_sinkhorn", "mhc_post_mlp"):
        assert led.row(n).kind == "measured" and led.row(n).source.startswith("mhc_micro_large.json#B64/"), n


def test_roofline_total_equals_the_ported_engine_rollup(cur):
    from test.kernel_ledger.engine.graph import compute_rollup
    from test.kernel_ledger.models.glm53f.decode import build_glm53f_decode_graph

    rollup = compute_rollup(build_glm53f_decode_graph(BS1_CTX1K), collective_lookup=ConstantCollectiveModel())
    assert cur.roofline_ms == pytest.approx(rollup.total_latency_us / 1e3)
    assert cur.collectives_ms == pytest.approx(rollup.collective_latency_us / 1e3)
