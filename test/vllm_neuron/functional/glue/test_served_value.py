# SPDX-License-Identifier: Apache-2.0
"""Every site follows the switch value this run started with (the value a server gets).

This directory's conftest pins ``all`` for the kernel tests. It first saves the value
the process was started with (``conftest.SERVED_VALUE``; unset means ``1``), and this
module puts that value back. Then it checks, for each site, at decode and at prefill and
at the row counts the served lines use, that the site dispatches its kernel exactly where
the value selects it, and takes the counted torch route everywhere else. Run the
directory under ``0``, ``1``, ``all`` and a per-kernel value to check each one.
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.functional.glue import conftest
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

#: Decode buckets (B) and the 128-row prefill block; every kernel serves these sizes.
ROWS = (1, 4, 64, 128)


def _mhc_phase(rows: int) -> str:
    """The phase an mHC site sees: decode up to the fixture's largest decode bucket."""
    return "decode" if rows <= max(glue_case.served_decode_buckets()) else "prefill"


@pytest.fixture(autouse=True)
def _served_value(monkeypatch):
    if conftest.SERVED_VALUE is None:
        monkeypatch.delenv(glue.GLUE_FUSED_ENV, raising=False)
    else:
        monkeypatch.setenv(glue.GLUE_FUSED_ENV, conftest.SERVED_VALUE)


def _selection():
    return glue.glue_selection(conftest.SERVED_VALUE)


def _route(module) -> str:
    dispatched, declined = module.dispatch_counters()
    assert dispatched + declined == 1, (dispatched, declined)
    return "kernel" if dispatched else "torch"


def test_the_served_value_parses():
    """A bad value fails here first, by name, rather than inside a kernel test."""
    assert isinstance(_selection(), glue.GlueSelection)


@pytest.mark.parametrize("rows", ROWS)
def test_mhc_pre_follows_the_served_value(sites, rows):
    live, _, cfg = sites
    mhc_pre.reset_dispatch_counters()
    live.mhc_pre(glue_case.streams_input(cfg, rows))
    want = "kernel" if _selection().selects("mhc_pre", rows, _mhc_phase(rows)) else "torch"
    assert _route(mhc_pre) == want


@pytest.mark.parametrize("phase", glue.PHASES)
@pytest.mark.parametrize("rows", ROWS)
def test_kda_projections_follow_the_served_value(pair, rows, phase):
    live, _, cfg = pair
    kda_projections.reset_dispatch_counters()
    kda_projections.kda_projections(projection_case._hidden(cfg, rows), live, phase=phase)
    want = "kernel" if _selection().selects("kda_projections", rows, phase) else "torch"
    assert _route(kda_projections) == want


@pytest.mark.parametrize("phase", glue.PHASES)
@pytest.mark.parametrize("rows", ROWS)
def test_kda_output_follows_the_served_value(pair, rows, phase):
    live, _, _ = pair
    core, gate, _ = output_case._operands(rows)
    kda_output.reset_dispatch_counters()
    kda_output.kda_gated_projection(core, gate, live, phase=phase)
    want = "kernel" if _selection().selects("kda_output", rows, phase) else "torch"
    assert _route(kda_output) == want


@pytest.mark.parametrize("rows", ROWS)
def test_mhc_post_follows_the_served_value(post_sites, rows, monkeypatch):
    live, _, cfg = post_sites
    x, streams, post, comb = post_case._operands(cfg, rows)
    seen = {}
    real = combine.hyper_connection_combine

    def spy(x, residual, post_layer_mix, comb_res_mix):
        seen.update(x=x.dtype, residual=residual.dtype)
        return real(x, residual, post_layer_mix, comb_res_mix)

    monkeypatch.setattr(combine, "hyper_connection_combine", spy)
    live.mhc_post(x, streams, post, comb)
    bf16 = _selection().selects("mhc_post", rows, _mhc_phase(rows))
    want = torch.bfloat16 if bf16 else torch.float32
    assert seen == {"x": want, "residual": want}
