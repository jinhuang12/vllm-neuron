# SPDX-License-Identifier: Apache-2.0
"""Every reference number is cited by a snippet that is still in its source file."""

from __future__ import annotations

import pytest

from test.kernel_ledger.models.glm53f.decode import BUCKETS
from test.kernel_ledger.models.glm53f.references import RECONCILED, REFERENCES, STEP_5938748

_TERMS = [STEP_5938748] + [t for r in REFERENCES.values() for t in r.reference + r.engine_active + r.evidence]


@pytest.mark.parametrize("term", _TERMS, ids=lambda t: t.label[:40])
def test_cited_snippet_is_in_the_source(term):
    assert term.snippet in term.file.read_text(), f"{term.file.name}: {term.snippet!r}"


def test_every_reconciled_bucket_is_a_ledger_bucket_with_a_reference():
    assert set(RECONCILED) <= set(BUCKETS)
    assert set(REFERENCES) == set(RECONCILED)
    assert all(REFERENCES[b].reference_ms > 0 for b in RECONCILED)


def test_expected_failures_carry_their_cause():
    # team-lead ruling: mHC, KDA, DSA, lm_head must pass; MoE, dense, collectives fail with the scope cause
    assert {b for b, r in REFERENCES.items() if r.expect == "FAIL"} == {"MoE", "dense", "collectives"}
    for r in REFERENCES.values():
        assert (r.expect == "FAIL") == bool(r.cause)


def test_reference_sums():
    assert REFERENCES["mHC"].reference_ms == pytest.approx(16.61)
    assert REFERENCES["KDA"].reference_ms == pytest.approx(11.832 + 0.64)
    assert REFERENCES["KDA"].engine_active_ms == pytest.approx(5.47)
    assert REFERENCES["DSA/MLA"].reference_ms == pytest.approx(13.794)
    assert REFERENCES["MoE"].reference_ms == pytest.approx(1.7766 + 3.982 + 3.17)
    assert REFERENCES["dense"].reference_ms == pytest.approx(1.916)
    assert REFERENCES["lm_head"].reference_ms == pytest.approx(2.161)
    assert REFERENCES["collectives"].reference_ms == pytest.approx(2.90)
