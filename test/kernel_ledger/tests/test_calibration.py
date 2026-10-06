# SPDX-License-Identifier: Apache-2.0
"""Calibrated PREDICTOR: k_bucket = in-model (gate profile, else the breakdown reference) / micro before."""

from __future__ import annotations

import pytest

from test.kernel_ledger.models.glm53f.calibration import calibrate, calibrated_buckets, calibrated_ms
from test.kernel_ledger.models.glm53f.configs import BS1_CTX1K, BASELINE, CURRENT
from test.kernel_ledger.models.glm53f.ledger import build_ledger
from test.kernel_ledger.models.glm53f.profile import profile_buckets
from test.kernel_ledger.models.glm53f.references import REFERENCES
from test.kernel_ledger.readers.gate import read_gate_step
from test.kernel_ledger.readers.micro import REPORTS_DIR, load_micro_results


@pytest.fixture(scope="module")
def results():
    return load_micro_results(REPORTS_DIR)


@pytest.fixture(scope="module")
def base(results):
    return build_ledger(BS1_CTX1K, BASELINE, results)


@pytest.fixture(scope="module")
def cur(results):
    return build_ledger(BS1_CTX1K, CURRENT, results)


@pytest.fixture(scope="module")
def gate():
    return read_gate_step(REPORTS_DIR / "gate_baseline.json")


@pytest.fixture(scope="module")
def cal(base, gate):
    return calibrate(base, gate)


def test_k_is_the_gate_profile_over_the_micro_before_sum(base, gate, cal):
    prof = profile_buckets(gate.buckets_ms)
    b = base.buckets()
    for bucket in ("mHC", "KDA", "DSA/MLA", "MoE", "dense", "collectives"):
        assert cal.k[bucket] == pytest.approx(prof.by_bucket[bucket] / b[bucket].measured_ms), bucket
        assert cal.source[bucket].startswith("gate_baseline.json buckets_ms")


def test_lm_head_falls_back_to_the_breakdown_reference(base, cal):
    assert cal.k["lm_head"] == pytest.approx(REFERENCES["lm_head"].reference_ms / base.buckets()["lm_head"].measured_ms)
    assert cal.source["lm_head"].startswith("fallback")


def test_without_a_gate_profile_every_bucket_falls_back(base):
    cal = calibrate(base, None)
    assert cal.k["KDA"] == pytest.approx(REFERENCES["KDA"].reference_ms / base.buckets()["KDA"].measured_ms)
    assert all(s.startswith("fallback") for s in cal.source.values())


def test_calibrated_5938748_is_the_in_model_time(base, gate, cal):
    got = calibrated_buckets(base, cal)
    prof = profile_buckets(gate.buckets_ms)
    assert got["mHC"] == pytest.approx(prof.by_bucket["mHC"])
    assert got["lm_head"] == pytest.approx(REFERENCES["lm_head"].reference_ms)
    assert calibrated_ms(base, cal) == pytest.approx(sum(got.values()))


def test_calibrated_current_is_k_times_after(cur, cal):
    got = calibrated_buckets(cur, cal)
    b = cur.buckets()
    assert got["KDA"] == pytest.approx(cal.k["KDA"] * b["KDA"].measured_ms)
    assert got["collectives"] == pytest.approx(cal.k["collectives"] * b["collectives"].measured_ms)
    assert got["sampler"] == pytest.approx(b["sampler"].measured_ms)  # no in-model number: k = 1


def test_calibration_needs_the_5938748_ledger(cur, gate):
    with pytest.raises(ValueError):
        calibrate(cur, gate)
