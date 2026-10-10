# SPDX-License-Identifier: Apache-2.0
"""Every reference number is cited by a snippet that is still in its source file."""

from __future__ import annotations

import pytest

from test.kernel_ledger.models.glm53f.decode import BUCKETS
from test.kernel_ledger.models.glm53f.references import RECONCILED, REFERENCES, STEP_5938748
from test.kernel_ledger.tests.frozen_reports import frozen

_TERMS = [STEP_5938748] + [t for r in REFERENCES.values() for t in r.reference + r.engine_active + r.evidence
                            + tuple(x for a in r.alternatives for x in a.terms) + r.unmodeled]


@pytest.mark.parametrize("term", _TERMS, ids=lambda t: t.label[:40])
def test_cited_snippet_is_in_the_source(term):
    # a file of the live reports directory is read from its frozen copy
    assert term.snippet in frozen(term.file).read_text(), f"{term.file.name}: {term.snippet!r}"


def test_every_reconciled_bucket_is_a_ledger_bucket_with_a_reference():
    assert set(RECONCILED) <= set(BUCKETS)
    assert set(REFERENCES) == set(RECONCILED)
    assert all(REFERENCES[b].reference_ms > 0 for b in RECONCILED)


def test_expected_failures_carry_their_cause():
    # mHC, KDA, DSA, lm_head must pass; MoE, dense, collectives fail with the scope cause
    assert {b for b, r in REFERENCES.items() if r.expect == "FAIL"} == {"MoE", "dense", "collectives"}
    for r in REFERENCES.values():
        assert (r.expect == "FAIL") == bool(r.cause)


def test_reference_sums():
    assert REFERENCES["mHC"].reference_ms == pytest.approx(16.61)
    assert REFERENCES["KDA"].reference_ms == pytest.approx(12.3)  # the breakdown's as-built KDA bound
    assert REFERENCES["KDA"].engine_active_ms == pytest.approx(5.47)
    assert REFERENCES["DSA/MLA"].reference_ms == pytest.approx(13.794)
    assert REFERENCES["MoE"].reference_ms == pytest.approx(1.7766 + 3.982 + 3.17)
    assert REFERENCES["dense"].reference_ms == pytest.approx(1.916)
    assert REFERENCES["lm_head"].reference_ms == pytest.approx(2.161)
    assert REFERENCES["collectives"].reference_ms == pytest.approx(2.90)


def test_kda_is_judged_on_the_breakdown_bound_and_keeps_every_other_reading():
    # PASS/FAIL against the 12.3 ms bound; no reference is dropped;
    # the 0.975 ms KDA-layer glue is an un-modeled term that lands in the residual
    kda = REFERENCES["KDA"]
    assert [round(a.ms, 3) for a in kda.alternatives] == [12.472, 13.447]
    assert [(t.ms, t.label) for t in kda.unmodeled] == [(0.975, "KDA-layer glue (ACT, DVE)")]
    assert all(not r.alternatives and not r.unmodeled for b, r in REFERENCES.items() if b != "KDA")
