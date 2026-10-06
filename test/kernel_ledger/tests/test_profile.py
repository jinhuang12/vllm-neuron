# SPDX-License-Identifier: Apache-2.0
"""Gate profile buckets (attribute_decode.py source files) mapped onto ledger buckets."""

from __future__ import annotations

import pytest

from test.kernel_ledger.models.glm53f.profile import profile_buckets
from test.kernel_ledger.readers.gate import read_gate_step
from test.kernel_ledger.readers.micro import REPORTS_DIR


@pytest.fixture(scope="module")
def base():
    return read_gate_step(REPORTS_DIR / "gate_baseline.json")


@pytest.fixture(scope="module")
def kv():
    return read_gate_step(REPORTS_DIR / "gate_kv.json")


def test_baseline_profile_maps_every_named_kernel(base):
    p = profile_buckets(base.buckets_ms)
    b = base.buckets_ms
    assert p.unmapped == []
    assert p.by_bucket["mHC"] == pytest.approx(b["mhc/hyper_connection.py"] + b["mhc/sinkhorn.py"])
    assert p.by_bucket["KDA"] == pytest.approx(
        b["nkilib/experimental/conv/depthwise_conv1d.py"] + b["kda/chunked_recurrence.py"] + b["kda/gate_clamp.py"]
        + b["kda/decode_state.py"])
    assert p.by_bucket["collectives"] == pytest.approx(b["wait: collective"])
    assert p.by_bucket["dense"] == pytest.approx(b["blockwise_fp8_mm.py"])
    assert "lm_head" not in p.by_bucket  # the lm_head GEMV is an unnamed compiler op


def test_baseline_profile_matches_the_breakdown_buckets(base):
    p = profile_buckets(base.buckets_ms)
    # same code as DECODE_BREAKDOWN.md, other CPU placement: mHC 16.61, KDA 5.47, DSA 8.86, MoE 3.51 + 0.803
    assert p.by_bucket["mHC"] == pytest.approx(16.61, rel=0.02)
    assert p.by_bucket["KDA"] == pytest.approx(5.47, rel=0.02)
    assert p.by_bucket["DSA/MLA"] == pytest.approx(8.86, rel=0.02)
    assert p.by_bucket["MoE"] == pytest.approx(3.51 + 0.803, rel=0.06)


def test_profile_buckets_and_residual_add_up_to_the_step(base, kv):
    for g in (base, kv):
        p = profile_buckets(g.buckets_ms)
        assert sum(p.by_bucket.values()) + p.residual_ms == pytest.approx(g.device_step_ms, rel=1e-4)
        assert p.residual_parts["compiler ops (unnamed XLA)"] > 10.0


def test_kv_changes_no_kernel_bucket(base, kv):
    pb, pk = profile_buckets(base.buckets_ms), profile_buckets(kv.buckets_ms)
    assert set(pb.by_bucket) == set(pk.by_bucket)
    for bucket in pb.by_bucket:
        assert pk.by_bucket[bucket] == pytest.approx(pb.by_bucket[bucket], rel=0.05), bucket


def test_an_unknown_kernel_source_is_listed_and_kept_in_the_residual():
    p = profile_buckets({"mhc/sinkhorn.py": 1.0, "new/kernel.py": 2.0, "wait: semaphore": 3.0})
    assert p.by_bucket == {"mHC": 1.0}
    assert p.unmapped == ["new/kernel.py"]
    assert p.residual_ms == pytest.approx(5.0)
