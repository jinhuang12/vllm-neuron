# SPDX-License-Identifier: Apache-2.0
"""Each fused site follows ``VLLM_NEURON_GLUE_FUSED`` per kernel, phase and row count.

A deselected site takes the 0a08ff4 torch route (counted as a decline, as the kill switch
is) and returns exactly what 0a08ff4 returns; a selected one dispatches its kernel. The
KDA sites pass the phase the layer hands them; an mHC site tells it by row count against
its config's largest decode bucket (``glue_case.served_decode_buckets``).
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.functional.glue import glue_case
from test.vllm_neuron.functional.glue import test_kda_output as output_case
from test.vllm_neuron.functional.glue import test_kda_projections as projection_case
from test.vllm_neuron.functional.glue import test_mhc_post_bf16 as post_case
from test.vllm_neuron.functional.glue import test_mhc_pre as pre_case
from vllm_neuron.functional import glue
from vllm_neuron.functional.glue import kda_output, kda_projections, mhc_pre
from vllm_neuron.functional.mhc import hyper_connection as combine

sites = pre_case.sites
pair = projection_case.pair
post_sites = post_case.sites


def _spec(monkeypatch, value: str) -> None:
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, value)


@pytest.mark.parametrize(("spec", "batch", "served"), (
    ("mhc_pre:prefill", 4, False),  # 4 rows: a decode batch
    ("mhc_pre:decode@4", 4, True),
    ("mhc_pre:decode@4", 1, False),
    ("mhc_pre:decode", 64, True),
    ("kda_output,mhc_post", 1, False),
    ("all", 1, True),
))
def test_mhc_pre_site_follows_the_switch(sites, spec, batch, served, monkeypatch):
    live, old, cfg = sites
    _spec(monkeypatch, spec)
    streams = glue_case.streams_input(cfg, batch)
    mhc_pre.reset_dispatch_counters()
    got = live.mhc_pre(streams)
    assert mhc_pre.dispatch_counters() == ((1, 0) if served else (0, 1))
    if not served:
        for a, b in zip(got, old.mhc_pre(streams)):
            assert torch.equal(a, b)


@pytest.mark.parametrize(("spec", "phase", "served"), (
    ("kda_projections:decode", "decode", True),
    ("kda_projections:decode", "prefill", False),
    ("kda_projections:prefill", "decode", False),
    ("kda_projections:prefill@-4", "prefill", True),
    ("kda_projections:prefill@5-", "prefill", False),
    ("mhc_pre,kda_output,mhc_post", "decode", False),
))
def test_kda_projections_follow_the_phase_they_are_handed(pair, spec, phase, served,
                                                          monkeypatch):
    live, _, cfg = pair
    _spec(monkeypatch, spec)
    hidden = projection_case._hidden(cfg, 4)
    kda_projections.reset_dispatch_counters()
    got = kda_projections.kda_projections(hidden, live, phase=phase)
    assert kda_projections.dispatch_counters() == ((1, 0) if served else (0, 1))
    if not served:
        for a, b in zip(got, kda_projections.kda_projections_torch(hidden, live)):
            assert torch.equal(a, b)


@pytest.mark.parametrize(("spec", "batch", "served"), (
    ("kda_projections:prefill", 1, False),
    ("kda_projections:decode", 1, True),
    ("kda_projections:decode", 4, True),  # the batched decode leg
    ("kda_projections:decode@2-", 1, False),
))
def test_kda_attention_forward_hands_the_projections_its_phase(pair, spec, batch, served,
                                                               monkeypatch):
    live, _, cfg = pair
    _spec(monkeypatch, spec)
    hidden = projection_case._hidden(cfg, batch)
    carriers = projection_case._carriers(live, batch)
    kda_projections.reset_dispatch_counters()
    live(hidden, **{k: v for k, v in carriers.items() if k != "banks"})
    assert kda_projections.dispatch_counters() == ((1, 0) if served else (0, 1))


@pytest.mark.parametrize(("spec", "batch", "phase", "served"), (
    ("kda_output@2-", 1, "decode", False),
    ("kda_output@2-", 4, "decode", True),
    ("kda_output:prefill", 4, "decode", False),
    ("kda_output:prefill", 4, "prefill", True),
    ("kda_output:decode", 64, "decode", True),
    ("kda_output:decode", 4, None, False),  # a phase rule needs the phase
))
def test_kda_output_site_follows_the_switch(pair, spec, batch, phase, served,
                                            monkeypatch):
    live, _, _ = pair
    _spec(monkeypatch, spec)
    core, gate, _ = output_case._operands(batch)
    kda_output.reset_dispatch_counters()
    got = kda_output.kda_gated_projection(core, gate, live, phase=phase)
    assert kda_output.dispatch_counters() == ((1, 0) if served else (0, 1))
    if not served:
        assert torch.equal(got, kda_output.kda_gated_projection_torch(core, gate, live))


@pytest.mark.parametrize(("spec", "batch", "bf16"), (
    ("mhc_post:prefill", 4, False),
    ("mhc_post:prefill", 128, True),  # 128 rows: more than any decode batch
    ("mhc_post:decode@4", 4, True),
    ("mhc_post:decode@4", 64, False),
    ("mhc_pre,kda_projections,kda_output", 4, False),
    ("0", 4, False),
    ("all", 4, True),
))
def test_mhc_post_widens_its_operands_only_when_deselected(post_sites, spec, batch, bf16,
                                                           monkeypatch):
    live, old, cfg = post_sites
    _spec(monkeypatch, spec)
    x, streams, post, comb = post_case._operands(cfg, batch)
    seen = {}
    real = combine.hyper_connection_combine

    def spy(x, residual, post_layer_mix, comb_res_mix):
        seen.update(x=x.dtype, residual=residual.dtype)
        return real(x, residual, post_layer_mix, comb_res_mix)

    monkeypatch.setattr(combine, "hyper_connection_combine", spy)
    got = live.mhc_post(x, streams, post, comb)
    want = torch.bfloat16 if bf16 else torch.float32
    assert seen == {"x": want, "residual": want}
    monkeypatch.setattr(combine, "hyper_connection_combine", real)
    assert torch.equal(got, old.mhc_post(x, streams, post, comb))
