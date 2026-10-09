# SPDX-License-Identifier: Apache-2.0
"""An mHC layer tells the glue switch the step's phase from the runner's buckets.

``Glm5NextHyperConnection`` sees only ``[T, S, H]`` streams. It takes its largest decode
batch from ``neuron_config.num_seqs_buckets`` (``max_decode_rows``), and under
speculative method "mtp" the verify step's row counts, each bucket times ``1 + k``
(``verify_rows``, ``k`` read at construction). A verify row count is the verify step, any
other call of at most ``max_decode_rows`` rows a decode step, a larger one a prefill
chunk. A layer built without the buckets knows no phase, so only a rule without a phase
selects its kernels. At construction it refuses a switch value that routes an mHC kernel
by phase at the row count of a prefill bucket a decode batch or the verify step can also
have, and a ``kernel:verify`` rule that leaves a verify row count out.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from test.vllm_neuron.functional.glue import glue_case
from test.vllm_neuron.functional.glue import test_mhc_pre as pre_case
from vllm_neuron.functional import glue
from vllm_neuron.functional.glue import mhc_pre
from vllm_neuron.functional.mhc import hyper_connection as combine
from vllm_neuron.model.glm5_next import model_fp8, mtp
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.bucket_utils import get_default_num_seqs_buckets

#: A decode batch of exactly the row count a 128-row prefill rule names: one token tile.
ROWS = mhc_pre.MHC_PRE_TOKEN_TILE
#: A server whose decode batches outgrow that row count: more concurrent requests than
#: ``ROWS``.
MANY_SEQS = 2 * ROWS
#: The phase-split value these tests route by: mhc_pre and mhc_post at prefill only.
PREFILL_ONLY = "mhc_pre:prefill,mhc_post:prefill"
#: The verify value these tests route by: mhc_pre at the verify step only.
VERIFY_ONLY = "mhc_pre:verify"


def _neuron_config(max_num_seqs: int, prefill=glue_case.SERVED_PREFILL_BUCKETS):
    return NeuronConfig(num_seqs_buckets=get_default_num_seqs_buckets(max_num_seqs),
                        num_batched_tokens_buckets=list(prefill))


def _site(neuron_config):
    site, cfg = pre_case._site(model_fp8, glue_case._Source(seed=4),
                               neuron_config=neuron_config)
    return site, cfg


def _routes(site, cfg, rows: int, monkeypatch) -> tuple[tuple[int, int], torch.dtype]:
    """mhc_pre's counters and the dtype mhc_post hands the combine, for one call."""
    streams = glue_case.streams_input(cfg, rows)
    mhc_pre.reset_dispatch_counters()
    post_mix, comb_mix, layer_input = site.mhc_pre(streams)
    seen = []
    real = combine.hyper_connection_combine

    def spy(x, residual, post_layer_mix, comb_res_mix):
        seen.append(residual.dtype)
        return real(x, residual, post_layer_mix, comb_res_mix)

    monkeypatch.setattr(combine, "hyper_connection_combine", spy)
    site.mhc_post(layer_input, streams, post_mix, comb_mix)
    monkeypatch.setattr(combine, "hyper_connection_combine", real)
    return mhc_pre.dispatch_counters(), seen[0]


def test_the_largest_decode_bucket_bounds_the_decode_rows():
    cfg = glue_case.text_config()
    for max_num_seqs in (1, glue_case.SERVED_MAX_NUM_SEQS, MANY_SEQS):
        site = model_fp8.Glm5NextHyperConnection(cfg,
                                                 neuron_config=_neuron_config(max_num_seqs))
        assert site.max_decode_rows == max_num_seqs
    assert model_fp8.Glm5NextHyperConnection(cfg).max_decode_rows is None
    assert model_fp8.Glm5NextHyperConnection(
        cfg, neuron_config=NeuronConfig()).max_decode_rows is None


def test_a_decode_batch_past_a_prefill_row_count_keeps_the_decode_route(monkeypatch):
    """At ``MANY_SEQS`` concurrent requests a ``ROWS``-row call is a decode step."""
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, PREFILL_ONLY)
    site, cfg = _site(_neuron_config(MANY_SEQS))
    assert _routes(site, cfg, ROWS, monkeypatch) == ((0, 1), torch.float32)
    site, cfg = _site(_neuron_config(glue_case.SERVED_MAX_NUM_SEQS))
    assert _routes(site, cfg, ROWS, monkeypatch) == ((1, 0), torch.bfloat16)


#: A chunk between the default's 1024- and 2048-row rules.
BETWEEN_ROWS = 1536


def test_the_default_fuses_the_uncapped_prefill_chunk(monkeypatch):
    """Under ``1`` (``envs.DEFAULT_GLUE_FUSED_SPEC``) a site on the uncapped line's buckets
    (decode batches up to ``SERVED_MAX_NUM_SEQS``, 2048-row prefill chunks) takes both
    fused kernels for a 2048-row chunk: the fused mhc_pre, and the bf16 operands into the
    combine. A chunk of a row count the default does not name keeps the torch route."""
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, "1")
    chunk = glue_case.UNCAPPED_PREFILL_CHUNK
    site, cfg = _site(_neuron_config(glue_case.SERVED_MAX_NUM_SEQS, prefill=(chunk,)))
    assert _routes(site, cfg, chunk, monkeypatch) == ((1, 0), torch.bfloat16)
    assert _routes(site, cfg, BETWEEN_ROWS, monkeypatch) == ((0, 1), torch.float32)


def test_without_the_runner_buckets_only_a_rule_without_a_phase_selects(monkeypatch):
    site, cfg = _site(NeuronConfig())
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, PREFILL_ONLY)
    assert _routes(site, cfg, ROWS, monkeypatch) == ((0, 1), torch.float32)
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, "mhc_pre,mhc_post")
    assert _routes(site, cfg, ROWS, monkeypatch) == ((1, 0), torch.bfloat16)


def test_a_prefill_bucket_a_decode_batch_can_have_refuses_a_phase_split(monkeypatch):
    cfg = glue_case.text_config()
    ambiguous = _neuron_config(MANY_SEQS, prefill=(ROWS, *glue_case.SERVED_PREFILL_BUCKETS))
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, PREFILL_ONLY)
    with pytest.raises(ValueError, match=glue.GLUE_FUSED_ENV):
        model_fp8.Glm5NextHyperConnection(cfg, neuron_config=ambiguous)
    for value in ("0", "all", "mhc_pre,mhc_post"):
        monkeypatch.setenv(glue.GLUE_FUSED_ENV, value)
        model_fp8.Glm5NextHyperConnection(cfg, neuron_config=ambiguous)


def _speculative(k: int):
    """The worker's config context with speculative method "mtp" and ``k`` drafts."""
    config = VllmConfig()
    config.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=k)
    return set_current_vllm_config(config, check_compile=False)


@pytest.mark.parametrize("k", (1, 2, 3))
def test_the_verify_step_has_the_draft_count_only_under_mtp(k, monkeypatch):
    """``glue.verify_draft_k`` passes the model's draft count under speculative method
    "mtp", and is 0 under another method, with no config, or for the shadow draft's knob
    alone (the shadow draft runs no verify step)."""
    monkeypatch.delenv(mtp.SHADOW_DRAFT_ENV, raising=False)
    with _speculative(k):
        assert glue.verify_draft_k(mtp.shadow_draft_k()) == k
    assert glue.verify_draft_k(k) == 0
    config = VllmConfig()
    config.speculative_config = SimpleNamespace(method="eagle", num_speculative_tokens=k)
    with set_current_vllm_config(config, check_compile=False):
        assert glue.verify_draft_k(k) == 0
    monkeypatch.setenv(mtp.SHADOW_DRAFT_ENV, str(k))
    assert mtp.shadow_draft_k() == k
    assert glue.verify_draft_k(mtp.shadow_draft_k()) == 0


@pytest.mark.parametrize("k", (1, 2, 3))
@pytest.mark.parametrize("max_num_seqs", (1, 2))
def test_the_layer_derives_the_verify_rows_from_the_speculative_config(k, max_num_seqs,
                                                                      monkeypatch):
    """Built inside the config context, the layer calls each decode bucket times ``1 + k``
    rows verify, and under ``mhc_pre:verify`` its mhc_pre site fuses them and its mhc_post
    site does not. One-row decode calls and the prefill bucket keep their routes."""
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, VERIFY_ONLY)
    monkeypatch.delenv(mtp.SHADOW_DRAFT_ENV, raising=False)
    with _speculative(k):
        site, cfg = _site(_neuron_config(max_num_seqs))
    buckets = get_default_num_seqs_buckets(max_num_seqs)
    assert site.verify_rows == {b * (1 + k) for b in buckets}
    for rows in sorted(site.verify_rows):
        assert site._glue_phase(rows) == "verify"
        streams = glue_case.streams_input(cfg, rows)
        assert mhc_pre.mhc_pre_admits(streams, site.fn, site.hc_scale, site.hc_base,
                                      phase=site._glue_phase(rows))
        assert not glue.glue_selected("mhc_post", rows, site._glue_phase(rows))
    assert site._glue_phase(1) == "decode"
    one = glue_case.streams_input(cfg, 1)
    assert not mhc_pre.mhc_pre_admits(one, site.fn, site.hc_scale, site.hc_base,
                                      phase=site._glue_phase(1))
    for rows in glue_case.SERVED_PREFILL_BUCKETS:
        assert site._glue_phase(rows) == "prefill"


def test_the_verify_step_takes_the_fused_mhc_pre(monkeypatch):
    """bs=1 with 3 drafts under ``mhc_pre:verify``: the 4-row verify call runs the fused
    mhc_pre (and the fp32 combine, mhc_post is not selected); the one-row decode call keeps
    the torch route. The layer reads ``k`` at construction; the calls run outside the
    config context, as the forward does."""
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, VERIFY_ONLY)
    monkeypatch.delenv(mtp.SHADOW_DRAFT_ENV, raising=False)
    with _speculative(3):
        site, cfg = _site(_neuron_config(1))
    assert _routes(site, cfg, 4, monkeypatch) == ((1, 0), torch.float32)
    assert _routes(site, cfg, 1, monkeypatch) == ((0, 1), torch.float32)


def test_under_the_default_the_verify_step_takes_the_fused_mhc_pre(monkeypatch):
    """The switch unset (``1``), bs=1 with 3 drafts: the 4-row verify call runs the fused
    mhc_pre and the fp32 combine; the one-row decode call keeps the torch route."""
    monkeypatch.delenv(glue.GLUE_FUSED_ENV, raising=False)
    monkeypatch.delenv(mtp.SHADOW_DRAFT_ENV, raising=False)
    with _speculative(3):
        site, cfg = _site(_neuron_config(1))
    assert _routes(site, cfg, 4, monkeypatch) == ((1, 0), torch.float32)
    assert _routes(site, cfg, 1, monkeypatch) == ((0, 1), torch.float32)


def test_a_decode_batch_of_a_verify_row_count_takes_the_verify_route(monkeypatch):
    """k=1 at decode buckets [1, 2]: a 2-row call is one request's verify step or the
    one-row-per-request decode step of two requests; the row count cannot tell them apart,
    so both take the verify route, the fused mhc_pre."""
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, VERIFY_ONLY)
    monkeypatch.delenv(mtp.SHADOW_DRAFT_ENV, raising=False)
    with _speculative(1):
        site, cfg = _site(_neuron_config(2))
    assert site.verify_rows == {2, 4}
    assert _routes(site, cfg, 2, monkeypatch) == ((1, 0), torch.float32)


def test_without_speculation_a_verify_shaped_call_keeps_its_old_phase(monkeypatch):
    """No speculative config, or the shadow draft's knob alone (the shadow draft does not
    verify): no verify rows, so a bs=1 layer's 4-row call is prefill and keeps the torch
    route under ``mhc_pre:verify``, as before the verify rule."""
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, VERIFY_ONLY)
    monkeypatch.delenv(mtp.SHADOW_DRAFT_ENV, raising=False)
    site, cfg = _site(_neuron_config(1))
    assert site.verify_rows == frozenset()
    assert site._glue_phase(4) == "prefill"
    assert _routes(site, cfg, 4, monkeypatch) == ((0, 1), torch.float32)
    monkeypatch.setenv(mtp.SHADOW_DRAFT_ENV, "3")
    with set_current_vllm_config(VllmConfig(), check_compile=False):
        shadow = model_fp8.Glm5NextHyperConnection(glue_case.text_config(),
                                                   neuron_config=_neuron_config(1))
    assert shadow.verify_rows == frozenset()


def test_the_layer_refuses_a_verify_rule_without_its_verify_rows(monkeypatch):
    """``mhc_pre:verify@4`` at 2 drafts (3-row verify step) is refused at construction,
    naming the switch and the row count; at 3 drafts it is kept."""
    monkeypatch.delenv(mtp.SHADOW_DRAFT_ENV, raising=False)
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, "mhc_pre:verify@4")
    cfg = glue_case.text_config()
    with _speculative(2), pytest.raises(ValueError, match=r"'mhc_pre:verify@4'.*\[3\] rows"):
        model_fp8.Glm5NextHyperConnection(cfg, neuron_config=_neuron_config(1))
    with _speculative(3):
        site = model_fp8.Glm5NextHyperConnection(cfg, neuron_config=_neuron_config(1))
    assert site.verify_rows == {4}


def test_the_layer_reads_k_through_the_heads_reader(monkeypatch):
    """``k`` comes from ``mtp.shadow_draft_k`` (contract C1): a knob that disagrees with
    the speculative config is refused by that reader, at construction."""
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, VERIFY_ONLY)
    monkeypatch.setenv(mtp.SHADOW_DRAFT_ENV, "2")
    with _speculative(3), pytest.raises(ValueError, match="num_speculative_tokens=3"):
        model_fp8.Glm5NextHyperConnection(glue_case.text_config(),
                                          neuron_config=_neuron_config(1))
