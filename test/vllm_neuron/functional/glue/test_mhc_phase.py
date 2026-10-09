# SPDX-License-Identifier: Apache-2.0
"""An mHC layer tells the glue switch the step's phase from the runner's buckets.

``Glm5NextHyperConnection`` sees only ``[T, S, H]`` streams. It takes its largest decode
batch from ``neuron_config.num_seqs_buckets`` (``max_decode_rows``): a call of at most
that many rows is a decode step, a larger one a prefill chunk. A layer built without the
buckets knows no phase, so only a rule without a phase selects its kernels. At
construction it refuses a switch value that routes an mHC kernel by phase at the row
count of a prefill bucket a decode batch can also have.
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.functional.glue import glue_case
from test.vllm_neuron.functional.glue import test_mhc_pre as pre_case
from vllm_neuron.functional import glue
from vllm_neuron.functional.glue import mhc_pre
from vllm_neuron.functional.mhc import hyper_connection as combine
from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.bucket_utils import get_default_num_seqs_buckets

#: A decode batch of exactly the row count a 128-row prefill rule names: one token tile.
ROWS = mhc_pre.MHC_PRE_TOKEN_TILE
#: A server whose decode batches outgrow that row count: more concurrent requests than
#: ``ROWS``.
MANY_SEQS = 2 * ROWS
#: The phase-split value these tests route by: mhc_pre and mhc_post at prefill only.
PREFILL_ONLY = "mhc_pre:prefill,mhc_post:prefill"


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
