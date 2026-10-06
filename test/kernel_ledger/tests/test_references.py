# SPDX-License-Identifier: Apache-2.0
"""Every reference number is cited by a snippet that is still in its source file."""

from __future__ import annotations

import pytest

from test.kernel_ledger.models.glm53f.decode import BUCKETS
from test.kernel_ledger.models.glm53f.references import RECONCILED, REFERENCES, STEP_5938748

_TERMS = [STEP_5938748] + [t for r in REFERENCES.values() for t in r.as_built + r.in_model_wall]


@pytest.mark.parametrize("term", _TERMS, ids=lambda t: t.label[:40])
def test_cited_snippet_is_in_the_source(term):
    assert term.snippet in term.file.read_text(), f"{term.file.name}: {term.snippet!r}"


def test_every_reconciled_bucket_is_a_ledger_bucket_with_an_as_built_number():
    assert set(RECONCILED) <= set(BUCKETS)
    assert set(REFERENCES) == set(RECONCILED)
    assert all(REFERENCES[b].as_built_ms > 0 for b in RECONCILED)


def test_reference_sums():
    assert REFERENCES["mHC"].as_built_ms == pytest.approx(16.61)
    assert REFERENCES["KDA"].as_built_ms == pytest.approx(5.47)
    assert REFERENCES["KDA"].in_model_wall_ms == pytest.approx(11.832 + 0.64 + 0.975)
    assert REFERENCES["DSA/MLA"].in_model_wall_ms == pytest.approx(13.794)
    assert REFERENCES["MoE"].in_model_wall_ms == pytest.approx(1.7766 + 3.982 + 3.17)
    assert REFERENCES["lm_head"].in_model_wall_ms == pytest.approx(2.161)
    assert REFERENCES["collectives"].as_built_ms == pytest.approx(2.90)
