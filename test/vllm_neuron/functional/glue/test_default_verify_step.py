# SPDX-License-Identifier: Apache-2.0
"""The default fuses the mHC pre-mix at the speculative verify step, and nothing else there.

Under ``VLLM_NEURON_GLUE_FUSED`` unset or ``1`` (``envs.DEFAULT_GLUE_FUSED_SPEC``) the
verify step takes the fused mhc_pre at both mHC sites of every layer. The mHC combine
keeps its fp32 operands there (mhc_post is not selected), and the KDA layer, which calls
the verify step decode, keeps its torch routes. ``0`` fuses nothing there, and the six
prefill rules alone keep the torch route there.

The verify step's row counts and phase come from the selector's own derivation, as the
mHC layer reads them (``glue.verify_rows``, ``glue.phase_of_rows``): at batch size 1 with
3 drafts, one 4-row call. The expected row counts are written out.
"""

from __future__ import annotations

import pytest

from test.vllm_neuron.functional.glue import glue_case
from vllm_neuron.functional import glue
from vllm_neuron.utils.bucket_utils import get_default_num_seqs_buckets

#: The served draft count.
DRAFTS = 3
#: ``max_num_seqs`` of the batch-1 line and of the ``SERVED_MAX_NUM_SEQS`` line, and the
#: verify step's row counts there at 3 drafts, written out: each decode bucket
#: (1, 2, 4, ..., max_num_seqs) times 4.
VERIFY_ROWS = {1: {4}, glue_case.SERVED_MAX_NUM_SEQS: {4, 8, 16, 32, 64, 128, 256}}


@pytest.fixture(params=("unset", "1"))
def default(request, monkeypatch):
    """The switch unset, then set to ``1``: both are the default."""
    if request.param == "unset":
        monkeypatch.delenv(glue.GLUE_FUSED_ENV, raising=False)
    else:
        monkeypatch.setenv(glue.GLUE_FUSED_ENV, "1")
    return request.param


def _verify_calls(max_num_seqs: int) -> list[tuple[int, str]]:
    """``(rows, phase)`` of each verify-step call at an mHC site of a server with
    ``max_num_seqs`` requests and 3 drafts, as the mHC layer derives them."""
    rows = glue.verify_rows(get_default_num_seqs_buckets(max_num_seqs), DRAFTS)
    assert rows == VERIFY_ROWS[max_num_seqs]
    return [(r, glue.phase_of_rows(r, max_num_seqs, rows)) for r in sorted(rows)]


def test_bs1_with_3_drafts_verifies_one_4_row_call():
    assert _verify_calls(1) == [(4, "verify")]


@pytest.mark.parametrize("max_num_seqs", sorted(VERIFY_ROWS))
def test_the_default_fuses_mhc_pre_at_the_verify_step(default, max_num_seqs):
    for rows, phase in _verify_calls(max_num_seqs):
        assert phase == "verify"
        assert glue.glue_selected("mhc_pre", rows, phase), rows


@pytest.mark.parametrize("max_num_seqs", sorted(VERIFY_ROWS))
def test_the_default_fuses_nothing_else_at_the_verify_step(default, max_num_seqs):
    """mhc_post keeps the fp32 combine at the verify step; the KDA kernels, whose layer
    calls the verify step decode, keep their torch routes."""
    for rows, phase in _verify_calls(max_num_seqs):
        assert not glue.glue_selected("mhc_post", rows, phase), rows
        for kernel in ("kda_projections", "kda_output"):
            assert not glue.glue_selected(kernel, rows, "decode"), (kernel, rows)


def test_zero_fuses_nothing_at_the_verify_step(monkeypatch):
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, "0")
    for max_num_seqs in sorted(VERIFY_ROWS):
        for rows, phase in _verify_calls(max_num_seqs):
            for kernel in glue.KERNELS:
                assert not glue.glue_selected(kernel, rows, phase), (kernel, rows)
                assert not glue.glue_selected(kernel, rows, "decode"), (kernel, rows)


def test_the_prefill_rules_alone_keep_the_torch_route_at_the_verify_step(monkeypatch):
    """The value that turns the verify fusion off keeps the prefill kernels."""
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, glue_case.PREFILL_RULES)
    for rows, phase in _verify_calls(1):
        for kernel in glue.KERNELS:
            assert not glue.glue_selected(kernel, rows, phase), (kernel, rows)
    assert glue.glue_selected("mhc_pre", 1024, "prefill")
    assert glue.glue_selected("mhc_post", 1024, "prefill")


@pytest.mark.parametrize("max_num_seqs", sorted(VERIFY_ROWS))
@pytest.mark.parametrize("prefill", (glue_case.SERVED_PREFILL_BUCKETS,
                                     (glue_case.UNCAPPED_PREFILL_CHUNK,)))
def test_the_default_is_kept_on_the_served_speculative_buckets(default, max_num_seqs,
                                                               prefill):
    """An mHC layer on these buckets with 3 drafts keeps the default at construction: no
    prefill bucket has a decode batch's or a verify step's row count."""
    rows = glue.verify_rows(get_default_num_seqs_buckets(max_num_seqs), DRAFTS)
    glue.require_rows_tell_phase(("mhc_pre", "mhc_post"), max_num_seqs, prefill,
                                 verify=rows)
