# SPDX-License-Identifier: Apache-2.0
"""``VLLM_NEURON_GLUE_FUSED``: which fused glue kernel serves which (phase, row count).

The contract (``vllm_neuron/functional/glue/__init__.py``):

* ``0``: nothing fused, every site keeps 0a08ff4's torch route (the kill switch).
* ``1`` or unset: ``envs.DEFAULT_GLUE_FUSED_SPEC``, the subset that won in-graph on the
  device, at prefill only.
* ``all``: every kernel at every phase and row count (821274e's behaviour).
* otherwise a comma list of ``kernel[:phase][@rows]`` rules.
"""

from __future__ import annotations

import pytest

from test.vllm_neuron.functional.glue import glue_case
from vllm_neuron import envs
from vllm_neuron.functional import glue

KERNELS = ("mhc_pre", "kda_projections", "kda_output", "mhc_post")
TOKENS = (1, 2, 8, 64, 65, 128, 1024)


def _table(spec: str) -> set:
    """``{(kernel, phase, tokens)}`` the spec selects over a grid of calls."""
    sel = glue.glue_selection(spec)
    return {(k, p, t) for k in KERNELS for p in glue.PHASES for t in TOKENS
            if sel.selects(k, t, p)}


def test_kernel_names_are_the_four_sites():
    assert glue.KERNELS == KERNELS
    assert glue.PHASES == ("prefill", "decode")


def test_zero_selects_nothing():
    assert _table("0") == set()


def test_all_selects_every_kernel_everywhere():
    assert _table("all") == {(k, p, t) for k in KERNELS for p in glue.PHASES
                             for t in TOKENS}


def test_the_switch_is_registered_in_envs(monkeypatch):
    assert glue.GLUE_FUSED_ENV in dir(envs)
    monkeypatch.delenv(glue.GLUE_FUSED_ENV, raising=False)
    assert envs.VLLM_NEURON_GLUE_FUSED == "1"
    assert not envs.is_set(glue.GLUE_FUSED_ENV)
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, " mhc_pre:decode ")
    assert envs.VLLM_NEURON_GLUE_FUSED == "mhc_pre:decode"


def test_one_and_unset_are_the_default_spec(monkeypatch):
    default = envs.DEFAULT_GLUE_FUSED_SPEC
    assert _table("1") == _table(default)
    monkeypatch.delenv(glue.GLUE_FUSED_ENV, raising=False)
    assert glue.glue_selection() == glue.glue_selection(default)
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, "1")
    assert glue.glue_selection() == glue.glue_selection(default)


def test_default_is_neither_nothing_nor_everything():
    default = _table(envs.DEFAULT_GLUE_FUSED_SPEC)
    assert default and default != _table("all")


#: Row counts either side of every bound the default holds.
DEFAULT_GRID = (1, 2, 8, 16, 32, 63, 64, 65, 127, 128, 129, 512, 1023, 1024, 1025, 2048)


def test_default_is_the_measured_prefill_subset():
    """``1`` is mhc_pre and mhc_post at 128- and 1024-row prefills, and nothing
    else: the buckets where the device A/B measured a win
    (``envs.DEFAULT_GLUE_FUSED_SPEC``'s docstring). A row count between or beyond them,
    such as the 512 rows where mhc_post lost, keeps the torch route. The table is
    written out, not derived, so a rule added to or removed from the default without
    changing this table fails here."""
    sel = glue.glue_selection("1")
    got = {(k, p, t) for k in KERNELS for p in glue.PHASES for t in DEFAULT_GRID
           if sel.selects(k, t, p)}
    assert got == {("mhc_pre", "prefill", 128), ("mhc_pre", "prefill", 1024),
                   ("mhc_post", "prefill", 128), ("mhc_post", "prefill", 1024)}


def test_default_decode_takes_the_zero_route_at_every_row_count():
    """Under ``1`` no kernel is selected at decode, at any row count, so every decode
    graph is the graph ``0`` traces."""
    one, zero = glue.glue_selection("1"), glue.glue_selection("0")
    for rows in DEFAULT_GRID:
        for kernel in KERNELS:
            assert not one.selects(kernel, rows, "decode"), (kernel, rows)
            assert one.selects(kernel, rows, "decode") == zero.selects(kernel, rows,
                                                                       "decode")


def test_default_selects_nothing_where_the_phase_is_not_known():
    """Every rule of ``1`` names a phase, so a call of unknown phase (an mHC layer built
    without the runner's buckets) keeps the torch route."""
    sel = glue.glue_selection("1")
    assert not any(sel.selects(k, t) for k in KERNELS for t in DEFAULT_GRID)


def test_an_mhc_layer_follows_the_default_on_the_serving_buckets_only():
    """``1`` routes the mHC kernels by phase at 128 rows. A server whose decode batches
    reach 128 rows and that has a 128-row prefill bucket cannot tell the two apart at the
    mHC sites, so the layer refuses ``1`` there; on the serving lines' buckets (decode
    batches of 1 or up to ``SERVED_MAX_NUM_SEQS``, one 1024-row prefill bucket) it holds."""
    with pytest.raises(ValueError, match="'1'"):
        glue.require_rows_tell_phase(("mhc_pre", "mhc_post"), DECODE_ROWS, PREFILL_ROWS,
                                     spec="1")
    for max_decode_rows in (1, glue_case.SERVED_MAX_NUM_SEQS):
        glue.require_rows_tell_phase(("mhc_pre", "mhc_post"), max_decode_rows,
                                     glue_case.SERVED_PREFILL_BUCKETS, spec="1")


def test_the_environment_is_read_at_each_call(monkeypatch):
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, "0")
    assert not glue.glue_selected("mhc_post", 1024, "prefill")
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, "all")
    assert glue.glue_selected("mhc_post", 1024, "prefill")
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, "mhc_pre:decode")
    assert glue.glue_selected("mhc_pre", 1, "decode")
    assert not glue.glue_selected("mhc_post", 1, "decode")


def test_a_bare_kernel_name_selects_both_phases():
    assert _table("kda_output") == {("kda_output", p, t) for p in glue.PHASES
                                    for t in TOKENS}


def test_a_phase_restricts_the_rule():
    assert _table("mhc_pre:prefill,kda_projections:all") == (
        {("mhc_pre", "prefill", t) for t in TOKENS}
        | {("kda_projections", p, t) for p in glue.PHASES for t in TOKENS})


def test_token_ranges_restrict_the_rule():
    got = _table("mhc_post:decode@64,mhc_pre:decode@2-64,kda_output@65-,"
                 "kda_projections:prefill@-8")
    assert got == (
        {("mhc_post", "decode", 64)}
        | {("mhc_pre", "decode", t) for t in (2, 8, 64)}
        | {("kda_output", p, t) for p in glue.PHASES for t in (65, 128, 1024)}
        | {("kda_projections", "prefill", t) for t in (1, 2, 8)})


def test_rules_for_one_kernel_add_up_and_whitespace_is_ignored():
    assert _table(" mhc_post:decode@1 , mhc_post:prefill ") == (
        {("mhc_post", "decode", 1)} | {("mhc_post", "prefill", t) for t in TOKENS})


@pytest.mark.parametrize("spec", (
    "mhc_pree", "mhc_pre:prefil", "mhc_pre@x", "mhc_pre@5-2", "mhc_pre@0",
    "mhc_pre:decode:1", "kda_output,,mhc_post", "", "2", "ALL", "hyper_connection",
    "mhc_pre@1-2-3", "mhc_pre@-",
))
def test_malformed_or_unknown_rules_are_refused_by_name(spec):
    with pytest.raises(ValueError, match=glue.GLUE_FUSED_ENV):
        glue.glue_selection(spec)


def test_a_bad_environment_value_is_refused_at_the_call_site(monkeypatch):
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, "mhc_pre:prefil")
    with pytest.raises(ValueError, match="prefil"):
        glue.glue_selected("mhc_pre", 1, "decode")


def test_an_unknown_kernel_or_phase_at_the_call_site_is_refused():
    with pytest.raises(ValueError, match="mhc_pree"):
        glue.glue_selected("mhc_pree", 1, "decode")
    with pytest.raises(ValueError, match="verify"):
        glue.glue_selected("mhc_pre", 1, "verify")


@pytest.mark.parametrize("tokens", TOKENS)
def test_a_call_of_unknown_phase_is_selected_only_by_a_rule_without_one(tokens):
    sel = glue.glue_selection("mhc_pre:prefill,mhc_pre:decode,kda_output")
    assert not sel.selects("mhc_pre", tokens)
    assert sel.selects("mhc_pre", tokens, "prefill") and sel.selects("mhc_pre", tokens,
                                                                      "decode")
    assert sel.selects("kda_output", tokens)
    assert glue.glue_selection("all").selects("mhc_post", tokens)


#: A site that tells the phase by row count: decode batches of up to this many rows, and
#: a prefill bucket of fewer rows (so the two look alike there) next to a larger one.
DECODE_ROWS = 155
PREFILL_ROWS = (128, 1024)


def test_a_value_the_row_count_cannot_follow_is_refused_by_name():
    with pytest.raises(ValueError, match=rf"{glue.GLUE_FUSED_ENV}=.*mhc_pre at 128 rows"):
        glue.require_rows_tell_phase(("mhc_pre", "mhc_post"), DECODE_ROWS, PREFILL_ROWS,
                                     spec="mhc_pre:prefill@128")
    with pytest.raises(ValueError, match="mhc_post at 128 rows"):
        glue.require_rows_tell_phase(("mhc_pre", "mhc_post"), DECODE_ROWS, PREFILL_ROWS,
                                     spec="mhc_post:decode")


@pytest.mark.parametrize("spec", ("0", "all", "mhc_pre", "mhc_post:decode@129-",
                                  "mhc_post:prefill@1024", "kda_output:prefill"))
def test_a_value_the_row_count_can_follow_is_kept(spec):
    glue.require_rows_tell_phase(("mhc_pre", "mhc_post"), DECODE_ROWS, PREFILL_ROWS,
                                 spec=spec)


def test_a_prefill_bucket_above_every_decode_batch_is_told_apart_by_its_rows():
    glue.require_rows_tell_phase(("mhc_pre", "mhc_post"), PREFILL_ROWS[0] - 1,
                                 PREFILL_ROWS, spec="mhc_pre:prefill@128")


def test_the_refusal_reads_the_switch_when_no_value_is_given(monkeypatch):
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, "mhc_pre:decode")
    with pytest.raises(ValueError, match="'mhc_pre:decode'"):
        glue.require_rows_tell_phase(("mhc_pre",), DECODE_ROWS, PREFILL_ROWS)


def test_the_dma_transpose_switch_is_registered_in_envs(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE", raising=False)
    assert envs.VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE is True
    monkeypatch.setenv("VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE", "0")
    assert envs.VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE is False
