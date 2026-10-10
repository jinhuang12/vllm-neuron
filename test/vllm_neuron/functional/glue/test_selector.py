# SPDX-License-Identifier: Apache-2.0
"""``VLLM_NEURON_GLUE_FUSED``: which fused glue kernel serves which (phase, row count).

The contract (``vllm_neuron/functional/glue/__init__.py``):

* ``0``: nothing fused, every site keeps 0a08ff4's torch route (the kill switch).
* ``1`` or unset: ``envs.DEFAULT_GLUE_FUSED_SPEC``, the subset that won on the device: at
  prefill, and mhc_pre at the speculative verify step.
* ``all``: every kernel at every phase and row count (821274e's behaviour).
* otherwise a comma list of ``kernel[:phase][@rows]`` rules.
"""

from __future__ import annotations

import pytest

from test.vllm_neuron.functional.glue import glue_case
from vllm_neuron import envs
from vllm_neuron.functional import glue
from vllm_neuron.utils.bucket_utils import get_default_num_seqs_buckets

KERNELS = ("mhc_pre", "kda_projections", "kda_output", "mhc_post")
TOKENS = (1, 2, 8, 64, 65, 128, 1024)


def _table(spec: str) -> set:
    """``{(kernel, phase, tokens)}`` the spec selects over a grid of calls."""
    sel = glue.glue_selection(spec)
    return {(k, p, t) for k in KERNELS for p in glue.PHASES for t in TOKENS
            if sel.selects(k, t, p)}


def test_kernel_names_are_the_four_sites():
    assert glue.KERNELS == KERNELS
    assert glue.PHASES == ("prefill", "decode", "verify")


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
DEFAULT_GRID = (1, 2, 3, 4, 5, 8, 16, 32, 63, 64, 65, 127, 128, 129, 512, 1023, 1024, 1025,
                2047, 2048, 2049, 4096)


def test_default_is_the_measured_prefill_subset_and_mhc_pre_at_verify():
    """``1`` is mhc_pre and mhc_post at 128-, 1024- and 2048-row prefills, and mhc_pre at
    every verify step, and nothing else: where the device A/B measured a win
    (``envs.DEFAULT_GLUE_FUSED_SPEC``'s docstring). A prefill row count between or
    beyond them, such as the 512 rows where mhc_post lost, 1025 to 2047 rows, or a
    4096-row chunk, keeps the torch route. The table is written out, not derived, so a
    rule added to or removed from the default without changing this table fails here."""
    sel = glue.glue_selection("1")
    got = {(k, p, t) for k in KERNELS for p in glue.PHASES for t in DEFAULT_GRID
           if sel.selects(k, t, p)}
    assert got == ({("mhc_pre", "prefill", 128), ("mhc_pre", "prefill", 1024),
                    ("mhc_pre", "prefill", 2048), ("mhc_post", "prefill", 128),
                    ("mhc_post", "prefill", 1024), ("mhc_post", "prefill", 2048)}
                   | {("mhc_pre", "verify", t) for t in DEFAULT_GRID})


#: The measured prefill subset, written out: the route every prefill and decode call
#: takes with and without the verify rule.
PREFILL_DEFAULT = ("mhc_pre:prefill@128,mhc_pre:prefill@1024,mhc_pre:prefill@2048,"
                   "mhc_post:prefill@128,mhc_post:prefill@1024,mhc_post:prefill@2048")
#: The prefill subset and mhc_pre at every verify step.
VERIFY_VALUE = PREFILL_DEFAULT + ",mhc_pre:verify"


def test_one_is_the_prefill_subset_and_mhc_pre_at_verify():
    """The default is the verify value: every test of that value holds for ``1``."""
    assert glue.glue_selection("1") == glue.glue_selection(VERIFY_VALUE)


def test_the_verify_rule_moves_no_prefill_or_decode_call():
    """The verify rule changes the route of verify-step calls only: at every row count a
    prefill, decode or unknown-phase call takes the route it took before the rule."""
    one, before = glue.glue_selection(VERIFY_VALUE), glue.glue_selection(PREFILL_DEFAULT)
    for rows in range(1, 4097):
        for kernel in KERNELS:
            for phase in ("prefill", "decode", None):
                assert one.selects(kernel, rows, phase) == before.selects(
                    kernel, rows, phase), (kernel, rows, phase)
            assert one.selects(kernel, rows, "verify") == (kernel == "mhc_pre"), (
                kernel, rows)


#: The draft counts and the largest decode buckets the verify derivation is tested at.
DRAFT_KS = (1, 2, 3)
MAX_SEQS = (1, 2)


@pytest.mark.parametrize("k", DRAFT_KS)
@pytest.mark.parametrize("max_seqs", MAX_SEQS)
def test_the_verify_step_is_each_decode_bucket_times_one_plus_k(k, max_seqs):
    buckets = get_default_num_seqs_buckets(max_seqs)
    rows = glue.verify_rows(buckets, k)
    assert rows == {b * (1 + k) for b in buckets}
    for verify in rows:
        assert glue.phase_of_rows(verify, max_seqs, rows) == "verify"
    for bucket in set(buckets) - rows:
        assert glue.phase_of_rows(bucket, max_seqs, rows) == "decode"
    for bucket in glue_case.SERVED_PREFILL_BUCKETS:
        assert glue.phase_of_rows(bucket, max_seqs, rows) == "prefill"
    glue.require_rows_tell_phase(("mhc_pre", "mhc_post"), max_seqs,
                                 glue_case.SERVED_PREFILL_BUCKETS, spec=VERIFY_VALUE,
                                 verify=rows)


def test_a_decode_batch_of_a_verify_row_count_is_the_verify_step():
    """k=1 at decode buckets [1, 2]: one request's verify step and the one-row-per-request
    decode step of two requests (which a speculative server also runs) both have 2 rows.
    The row count cannot tell them apart, so the site calls 2 rows verify, and that
    decode batch takes the verify route."""
    rows = glue.verify_rows([1, 2], 1)
    assert rows == {2, 4}
    assert glue.phase_of_rows(2, 2, rows) == "verify"
    assert glue.phase_of_rows(1, 2, rows) == "decode"


def test_without_speculation_there_is_no_verify_step():
    """k=0 (no speculative config, or the shadow draft) has no verify rows, so every call
    keeps the phase it had before the verify rule: a 4-row call at a bs=1 site is a
    prefill call, which the verify value does not fuse at 4 rows."""
    assert glue.verify_rows(get_default_num_seqs_buckets(64), 0) == frozenset()
    assert glue.phase_of_rows(4, 1) == "prefill"
    assert glue.phase_of_rows(1, 1) == "decode"
    assert not glue.glue_selection(VERIFY_VALUE).selects("mhc_pre", 4, "prefill")


@pytest.mark.parametrize("draft_k, buckets", ((-1, [1]), (1, [0, 1])))
def test_a_negative_k_or_an_empty_bucket_is_refused_by_name(draft_k, buckets):
    with pytest.raises(ValueError, match=glue.GLUE_FUSED_ENV):
        glue.verify_rows(buckets, draft_k)


def test_a_verify_rule_that_leaves_a_verify_row_count_out_is_refused_by_name():
    """``mhc_pre:verify@4`` names the 4-row verify step of bs=1 with 3 drafts. With 2
    drafts the step has 3 rows, and at decode buckets [1, 2] with 3 drafts it also has
    8: a row count with no entry would keep the torch route without a word."""
    sites, prefill = ("mhc_pre", "mhc_post"), glue_case.SERVED_PREFILL_BUCKETS
    with pytest.raises(ValueError, match=r"'mhc_pre:verify@4'.*\[3\] rows"):
        glue.require_rows_tell_phase(sites, 1, prefill, spec="mhc_pre:verify@4",
                                     verify=glue.verify_rows([1], 2))
    with pytest.raises(ValueError, match=r"\[8\] rows"):
        glue.require_rows_tell_phase(sites, 2, prefill, spec="mhc_pre:verify@4",
                                     verify=glue.verify_rows([1, 2], 3))
    for spec, buckets, k in (("mhc_pre:verify@4", [1], 3),
                             ("mhc_pre:verify@4,mhc_pre:verify@8", [1, 2], 3),
                             ("mhc_pre:verify", [1, 2], 2),
                             ("mhc_pre:prefill@4", [1], 2)):
        glue.require_rows_tell_phase(sites, max(buckets), prefill, spec=spec,
                                     verify=glue.verify_rows(buckets, k))


def test_a_prefill_bucket_of_a_verify_row_count_refuses_a_phase_split():
    """32 requests with 3 drafts verify 128 rows, the row count of a 128-row prefill
    bucket: the verify value fuses mhc_post there at prefill and not at verify."""
    rows = glue.verify_rows(get_default_num_seqs_buckets(32), 3)
    assert 128 in rows
    with pytest.raises(ValueError, match="mhc_post at 128 rows.*a verify step"):
        glue.require_rows_tell_phase(("mhc_pre", "mhc_post"), 32, PREFILL_ROWS,
                                     spec=VERIFY_VALUE, verify=rows)
    glue.require_rows_tell_phase(("mhc_pre", "mhc_post"), 32, PREFILL_ROWS, spec="all",
                                 verify=rows)


#: The device A/B's verify-step rule, ``mhc_pre:prefill@4`` (on a tree that called the
#: bs=1 verify step's 4 rows prefill), next to this tree's prefill entries. The A/B ran it
#: next to the entries before 10935e6's 2048-row ones, which no decode or verify call
#: reaches.
AB_VALUE = PREFILL_DEFAULT + ",mhc_pre:prefill@4"


def test_the_verify_value_routes_bs1_with_3_drafts_as_the_device_ab_did():
    """At bs=1 with 3 drafts every call takes the route the device A/B measured, so the
    A/B measures the verify value's graph."""
    ab, one = glue.glue_selection(AB_VALUE), glue.glue_selection(VERIFY_VALUE)
    verify = glue.verify_rows([1], 3)
    for rows in range(1, 4097):
        before, now = glue.phase_of_rows(rows, 1), glue.phase_of_rows(rows, 1, verify)
        for kernel in KERNELS:
            assert ab.selects(kernel, rows, before) == one.selects(kernel, rows, now), (
                kernel, rows)


def test_the_selector_does_not_import_the_model_package():
    """Nothing under ``vllm_neuron/functional/`` imports ``vllm_neuron.model``: the mHC
    layer passes the draft count in (:func:`glue.verify_draft_k`)."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(glue))
    names = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    names += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
              for alias in node.names]
    assert not [n for n in names if n and n.startswith("vllm_neuron.model")], names


def test_a_verify_rule_names_an_mhc_kernel_only():
    """Only the mHC sites tell the verify step; the KDA layer calls it decode."""
    assert glue.VERIFY_KERNELS == ("mhc_pre", "mhc_post")
    for kernel in ("kda_projections", "kda_output"):
        with pytest.raises(ValueError, match=rf"{kernel} has no verify phase"):
            glue.glue_selection(f"{kernel}:verify")
    assert glue.glue_selection("mhc_post:verify@8").selects("mhc_post", 8, "verify")


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
    "mhc_pre@1-2-3", "mhc_pre@-", "kda_output:verify", "kda_projections:verify@4",
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
    with pytest.raises(ValueError, match="draft"):
        glue.glue_selected("mhc_pre", 1, "draft")


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
