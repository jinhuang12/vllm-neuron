# SPDX-License-Identifier: Apache-2.0
"""The tiny root's glue counters at decode, under the switch value of this run.

The runner's decode capture entry point (``extract_decode_graphs``) drives the tiny
fixture root (three layers, two mHC sites each) through one 1-request decode step, with
the capture backend stood in by a call to the model
(``test_tiny_glm5next_capture_sites.py``'s harness, imported). This reads ``mhc_pre``'s
counters ``(dispatched, took the torch route)``, how many of its launches also returned
the feed-forward RMSNorm (every feed-forward site's when mhc_pre serves, the dense layers'
as well as the MoE layer's), and the dtype the mHC combine was handed (bf16 when
``mhc_post`` is selected, fp32 otherwise). The root's mHC sites are bound from a config
that carries the harness's buckets, as a served load binds them, so they see the step's
phase.

The tiny root reaches no KDA glue site (``kda_projections`` and ``kda_output`` stay at
``(0, 0)``); those two follow the switch at their sites in ``test_served_value.py``.

The value is the one the run started with (``conftest.SERVED_VALUE``, unset meaning
``1``), so run this directory under ``0``, ``1``, ``all`` and a per-kernel value. For
those three named values the expected table is written out below rather than derived from
:func:`glue.glue_selection`, so a change to what ``1`` selects at these rows fails here
until the table is changed with it.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from test.vllm_neuron.functional.glue import conftest
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_capture_sites as capture
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny
from vllm_neuron.functional import glue
from vllm_neuron.functional.mhc import hyper_connection as combine
from vllm_neuron.model.neuron_config import NeuronConfig

#: mHC sites the tiny root runs per forward: two per layer, three layers.
SITES = tiny.MHC_SITES_PER_LAYER * tiny.STACK_LAYERS
#: Feed-forward mHC sites per forward, one per layer. The first ``tiny.STACK_FIRST_K_DENSE``
#: layers are dense and the rest MoE, and every one hands its site the norm.
FFN_SITES = tiny.STACK_LAYERS
#: Rows each capture hands the mHC sites.
ROWS = {"prefill": capture.PREFILL_BUCKET, "decode": capture.DECODE_BATCH}
#: ``{value: {phase: (mhc_pre counters, combine dtype)}}``, written out.
RECORDED = {
    "0": {"prefill": ((0, SITES), torch.float32), "decode": ((0, SITES), torch.float32)},
    "1": {"prefill": ((SITES, 0), torch.bfloat16), "decode": ((0, SITES), torch.float32)},
    "all": {"prefill": ((SITES, 0), torch.bfloat16),
            "decode": ((SITES, 0), torch.bfloat16)},
}


@pytest.fixture(autouse=True)
def _served_value(monkeypatch):
    if conftest.SERVED_VALUE is None:
        monkeypatch.delenv(glue.GLUE_FUSED_ENV, raising=False)
    else:
        monkeypatch.setenv(glue.GLUE_FUSED_ENV, conftest.SERVED_VALUE)


def _expected(phase: str) -> tuple:
    value = "1" if conftest.SERVED_VALUE is None else conftest.SERVED_VALUE.strip()
    if value in RECORDED:
        return RECORDED[value][phase]
    sel = glue.glue_selection(value)
    rows = ROWS[phase]
    pre = (SITES, 0) if sel.selects("mhc_pre", rows, phase) else (0, SITES)
    return pre, torch.bfloat16 if sel.selects("mhc_post", rows, phase) else torch.float32


def _root_with_the_runner_buckets():
    """The capture harness's root, its mHC sites bound as a served load binds them.

    A served load binds each layer's sites from the root's text config, whose neuron
    config carries the runner's buckets (the sites read the step's phase off them). The
    fixture binds them from a config without any, so they are bound again here from one
    with this harness's buckets: one decode request, and the prefill bucket.
    """
    root, _ = capture._bound_root()
    cfg = dataclasses.replace(root.text_config, neuron_config=NeuronConfig(
        num_seqs_buckets=[capture.DECODE_BATCH],
        num_batched_tokens_buckets=[capture.PREFILL_BUCKET]))
    for layer in root.model.layers:
        assert layer.bind_hyper_connection_sites(cfg, torch.device("cpu")) == (
            tiny.MHC_SITES_PER_LAYER)
    return root


def test_the_recorded_tables_are_what_the_selector_says():
    """The written-out tables and the selector agree at the tiny root's rows."""
    for value, phases in RECORDED.items():
        sel = glue.glue_selection(value)
        for phase, ((dispatched, _), dtype) in phases.items():
            assert sel.selects("mhc_pre", ROWS[phase], phase) == (dispatched == SITES), (
                value, phase)
            assert sel.selects("mhc_post", ROWS[phase], phase) == (
                dtype == torch.bfloat16), (value, phase)


@pytest.mark.parametrize("phase", ("decode",))
def test_the_tiny_root_takes_the_routes_the_value_selects(phase, monkeypatch):
    from vllm_neuron.functional.glue import kda_output, kda_projections, mhc_pre

    root = _root_with_the_runner_buckets()
    runner = capture._runner(root)
    backend = capture._StandInBackend(runner)
    runner.capture_backend_model = backend
    handed = []
    real = combine.hyper_connection_combine

    def spy(x, residual, post_layer_mix, comb_res_mix):
        handed.append((x.dtype, residual.dtype, int(residual.shape[0])))
        return real(x, residual, post_layer_mix, comb_res_mix)

    monkeypatch.setattr(combine, "hyper_connection_combine", spy)
    for module in (mhc_pre, kda_projections, kda_output):
        module.reset_dispatch_counters()
    runner.extract_decode_graphs(capture.DECODE_BATCH)
    assert len(backend.seen) == 1
    want_pre, want_dtype = _expected(phase)
    assert mhc_pre.dispatch_counters() == want_pre
    assert mhc_pre.normed_dispatches() == (FFN_SITES if want_pre[0] else 0)
    assert kda_projections.dispatch_counters() == (0, 0)
    assert kda_output.dispatch_counters() == (0, 0)
    assert handed == [(want_dtype, want_dtype, ROWS[phase])] * SITES
