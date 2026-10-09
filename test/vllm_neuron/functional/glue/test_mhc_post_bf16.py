# SPDX-License-Identifier: Apache-2.0
"""``mhc_post`` on bf16 streams against 0a08ff4's ``mhc_post``, decode and prefill rows.

0a08ff4 casts ``x`` and the streams to fp32, runs the combine kernel in fp32 and
casts the fp32 result back to the streams' bf16. This tree hands the combine the bf16
operands: the kernel upcasts them on chip (bf16 -> fp32 is exact), does the same fp32
products and adds in the same order, and rounds the last add to bf16 once, which is
the cast 0a08ff4 does after the kernel. So the result is asserted bit for bit, in both
of the kernel's layouts (hidden on the partitions up to 32 tokens, tokens above) and
under one and two programs: at the decode batches B in {1, 4, 33, 64} and at the
2048-row chunk of the uncapped prefill line (16 token tiles).
"""

from __future__ import annotations

import pytest
import torch

from test.hardware.baselines.glue_0a08ff4 import load as load_0a08ff4
from test.vllm_neuron.functional.glue import glue_case
from vllm_neuron.functional.mhc import hyper_connection as combine
from vllm_neuron.model.glm5_next import model_fp8

BATCHES = (1, 4, 33, 64)
#: The decode batches and the uncapped line's prefill chunk.
ROWS = BATCHES + (glue_case.UNCAPPED_PREFILL_CHUNK,)


@pytest.fixture(scope="module")
def sites():
    cfg = glue_case.text_config()
    return (model_fp8.Glm5NextHyperConnection(cfg, neuron_config=cfg.neuron_config),
            load_0a08ff4().model_fp8.Glm5NextHyperConnection(
                cfg, neuron_config=cfg.neuron_config), cfg)


def _operands(cfg, batch):
    gen = torch.Generator().manual_seed(100 + batch)
    streams = glue_case.streams_input(cfg, batch)
    x = (torch.randn(batch, int(cfg.hidden_size), generator=gen) * 0.3).to(torch.bfloat16)
    post = torch.sigmoid(torch.randn(batch, 4, 1, generator=gen)) * 2.0
    comb = torch.softmax(torch.randn(batch, 4, 4, generator=gen), dim=-1)
    return x, streams, post, comb


@pytest.mark.parametrize("lnc", ("1", "2"))
@pytest.mark.parametrize("batch", ROWS)
def test_bf16_combine_matches_0a08ff4_bit_for_bit(sites, batch, lnc, monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    live, old, cfg = sites
    x, streams, post, comb = _operands(cfg, batch)
    combine.reset_dispatch_counters()
    got = live.mhc_post(x, streams, post, comb)
    assert combine.dispatch_counters() == (1, 0)
    want = old.mhc_post(x, streams, post, comb)
    assert got.dtype == want.dtype == torch.bfloat16
    assert torch.equal(got, want)


def test_the_kernel_returns_the_streams_dtype():
    gen = torch.Generator().manual_seed(7)
    streams = torch.randn(2, 4, 512, generator=gen).to(torch.bfloat16)
    x = torch.randn(2, 512, generator=gen).to(torch.bfloat16)
    post = torch.rand(2, 4, 1, generator=gen)
    comb = torch.rand(2, 4, 4, generator=gen)
    out = combine.hyper_connection_combine(x, streams, post, comb)
    assert out.dtype == torch.bfloat16
    want = combine.hyper_connection_combine(x.float(), streams.float(), post, comb)
    assert want.dtype == torch.float32
    assert torch.equal(out, want.to(torch.bfloat16))


def test_mhc_post_hands_the_kernel_the_bf16_operands(sites, monkeypatch):
    """No fp32 copies of ``x`` or the streams are made before the kernel."""
    live, _, cfg = sites
    x, streams, post, comb = _operands(cfg, 4)
    seen = {}
    real = combine.hyper_connection_combine

    def spy(x, residual, post_layer_mix, comb_res_mix):
        seen.update(x=x.dtype, residual=residual.dtype)
        return real(x, residual, post_layer_mix, comb_res_mix)

    monkeypatch.setattr(combine, "hyper_connection_combine", spy)
    live.mhc_post(x, streams, post, comb)
    assert seen == {"x": torch.bfloat16, "residual": torch.bfloat16}
