# SPDX-License-Identifier: Apache-2.0
"""KDA decode projections through the fused kernel against 0a08ff4, at B in {1, 4, 64}.

Two levels. The kernel's six outputs against 0a08ff4's torch expressions on checkpoint
layer 4's rank-0 weights; and the whole ``Glm5NextKDAAttention.forward`` of this tree
(which calls the kernel at both decode call sites: one token, and the concurrent
``_fused_decode_requests``) against the 0a08ff4 snapshot's, on the same carriers.

Tolerance, and why. The first-stage products are bf16 x bf16, exact in the kernel's
fp32 accumulator, so ``q, k, v, beta`` and the two bottlenecks differ from the fp32
reference only by summation order over 4096 terms; the second stage is an fp32
matmul, as in the reference. Asserted: ``|d| <= PROJ_RTOL * max|ref|`` per output.

Through the layer, the kernel stores ``q, k, v`` into the bf16 conv-state bank, so a
1e-7 change can move one value across a bf16 rounding boundary (one step, 2^-8
relative), and that value then runs through the conv, the norms and the recurrence.
The layer also runs ``_gated_output`` on ``functional/glue/kda_output.py``, whose
two-pass bf16 projection is within 2^-18 of fp32 per product (about 4e-6 relative), so
about 0.3% of the bf16 outputs round one step the other way.
So the layer is asserted at the scale of one flipped bf16 rounding, not bit for bit:
the bf16 conv bank within one bf16 step per element, in at most ``CONV_MAX_FLIPS`` of
them; the fp32 recurrent bank to ``STATE_ATOL``; the bf16 output to ``OUTPUT_REL_L2``
and ``|d| <= OUTPUT_MAX_ABS * max|ref|`` (one bf16 step of the largest element), with
at most ``OUTPUT_MAX_FLIPS`` elements changed. Observed in the simulator (B = 1 / 4 /
64): output rel-L2 1.7e-4 / 1.0e-4 / 1.1e-4, max |d| / max|ref| 2.0e-3 / 2.2e-3 /
2.8e-3, flips 0.32% / 0.28% / 0.24%; conv flips 0 / 0.012% / 0.005%; recurrent max |d|
7e-8 / 9e-8 / 1.4e-7.

fp32 operands (the CPU fixtures' dtype; the served checkpoint is bf16) take the kernel
on fp32 tiles, and are asserted to the same ``PROJ_RTOL`` against the same expressions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from test.hardware.baselines.glue_0a08ff4 import load as load_0a08ff4
from test.vllm_neuron.functional.glue import glue_case
from vllm_neuron.functional.glue import kda_projections as fused
from vllm_neuron.model.glm5_next import model_fp8

BATCHES = (1, 4, 64)
PROJ_RTOL = 2e-6
OUTPUT_REL_L2 = 5e-4
OUTPUT_MAX_ABS = 2.0 ** -7
OUTPUT_MAX_FLIPS = 0.01
CONV_MAX_FLIPS = 1e-3
STATE_ATOL = 1e-6


def _attention(model):
    cfg = glue_case.text_config()
    attn = model.Glm5NextKDAAttention(cfg, glue_case.TP_WORLD)
    prefix = f"model.language_model.layers.{glue_case.KDA_LAYER}."
    glue_case._kda_weights(attn, glue_case._Source(seed=4), prefix, int(cfg.hidden_size))
    return attn, cfg


@pytest.fixture(scope="module")
def pair():
    live, cfg = _attention(model_fp8)
    old, _ = _attention(load_0a08ff4().model_fp8)
    return live, old, cfg


def _hidden(cfg, batch):
    gen = torch.Generator().manual_seed(50 + batch)
    return (torch.randn(batch, int(cfg.hidden_size), generator=gen)).to(torch.bfloat16)


def _bf16_step(*values: torch.Tensor) -> torch.Tensor:
    """One bf16 step at the larger magnitude of each element pair (2^-7 of its binade).

    The larger of the two, because a one-step rounding flip can cross a binade
    boundary, where the lower binade's step is half the flip.
    """
    mag = torch.stack([v.float().abs() for v in values]).amax(0).clamp_min(2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7.0)


@pytest.mark.parametrize("dma", ("1", "0"))
@pytest.mark.parametrize("lnc", ("1", "2"))
@pytest.mark.parametrize("batch", BATCHES)
def test_kernel_matches_the_0a08ff4_expressions(pair, batch, lnc, dma, monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    monkeypatch.setenv(fused.DMA_TRANSPOSE_ENV, dma)
    live, _, cfg = pair
    hidden = _hidden(cfg, batch)
    fused.reset_dispatch_counters()
    got = fused.kda_projections(hidden, live)
    assert fused.dispatch_counters() == (1, 0)
    want = fused.kda_projections_torch(hidden, live)
    names = ("q_in", "k_in", "v_in", "raw_gate", "raw_beta", "out_gate")
    for name, a, b in zip(names, got, want):
        assert a.shape == b.shape and a.dtype == b.dtype == torch.float32, name
        bound = PROJ_RTOL * float(b.abs().max())
        assert float((a - b).abs().max()) <= bound, (name, float((a - b).abs().max()), bound)


WEIGHT_NAMES = ("q_proj_weight", "k_proj_weight", "v_proj_weight", "b_proj_weight",
                "f_a_proj_weight", "f_b_proj_weight", "g_a_proj_weight", "g_b_proj_weight")


def _widened(t: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """``t`` as fp32 values bf16 cannot hold: a 2^-8 relative jitter on each element."""
    t = t.float()
    return t + torch.randn(t.shape, generator=gen) * float(t.abs().mean()) * 2.0 ** -8


@pytest.mark.parametrize("dma", ("1", "0"))
@pytest.mark.parametrize("lnc", ("1", "2"))
@pytest.mark.parametrize("batch", (1, 64))
@pytest.mark.parametrize("x_dtype,w_dtype", (
    (torch.float32, torch.float32), (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16)))
def test_fp32_operands_match_the_0a08ff4_expressions(pair, batch, lnc, dma, x_dtype, w_dtype,
                                                     monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    monkeypatch.setenv(fused.DMA_TRANSPOSE_ENV, dma)
    live, _, cfg = pair
    gen = torch.Generator().manual_seed(90 + batch)
    attn = SimpleNamespace(**{
        name: (_widened(getattr(live, name), gen) if w_dtype == torch.float32
               else getattr(live, name)) for name in WEIGHT_NAMES})
    hidden = _hidden(cfg, batch)
    hidden = _widened(hidden, gen) if x_dtype == torch.float32 else hidden
    fused.reset_dispatch_counters()
    got = fused.kda_projections(hidden, attn)
    assert fused.dispatch_counters() == (1, 0)
    want = fused.kda_projections_torch(hidden, attn)
    for name, a, b in zip(("q_in", "k_in", "v_in", "raw_gate", "raw_beta", "out_gate"),
                          got, want):
        assert a.shape == b.shape and a.dtype == b.dtype == torch.float32, name
        bound = PROJ_RTOL * float(b.abs().max())
        assert float((a - b).abs().max()) <= bound, (name, float((a - b).abs().max()), bound)


def _carriers(attn, batch):
    case = SimpleNamespace(layer=SimpleNamespace(self_attn=attn))
    return glue_case.kda_carriers(case, batch)


@pytest.mark.parametrize("batch", BATCHES)
def test_attention_forward_matches_0a08ff4(pair, batch):
    live, old, cfg = pair
    hidden = _hidden(cfg, batch)
    mine, theirs = _carriers(live, batch), _carriers(old, batch)
    keywords = {k: v for k, v in mine.items() if k != "banks"}
    fused.reset_dispatch_counters()
    got = live(hidden, **keywords)
    assert fused.dispatch_counters() == (1, 0)
    want = old(hidden, **{k: v for k, v in theirs.items() if k != "banks"})
    assert got.dtype == want.dtype == torch.bfloat16
    diff = (got.float() - want.float()).abs()
    assert float(diff.norm() / want.float().norm()) <= OUTPUT_REL_L2
    assert float(diff.max()) <= OUTPUT_MAX_ABS * float(want.float().abs().max())
    assert float((diff > 0).float().mean()) <= OUTPUT_MAX_FLIPS
    (conv, rec), (want_conv, want_rec) = mine["banks"], theirs["banks"]
    assert conv.dtype == torch.bfloat16 and rec.dtype == torch.float32
    conv_diff = (conv.float() - want_conv.float()).abs()
    assert bool((conv_diff <= _bf16_step(want_conv, conv)).all())
    assert float((conv_diff > 0).float().mean()) <= CONV_MAX_FLIPS
    torch.testing.assert_close(rec, want_rec, atol=STATE_ATOL, rtol=0)


def test_the_kill_switch_restores_0a08ff4_bit_for_bit(pair, monkeypatch):
    live, old, cfg = pair
    monkeypatch.setenv("VLLM_NEURON_GLUE_FUSED", "0")
    hidden = _hidden(cfg, 4)
    mine, theirs = _carriers(live, 4), _carriers(old, 4)
    fused.reset_dispatch_counters()
    got = live(hidden, **{k: v for k, v in mine.items() if k != "banks"})
    assert fused.dispatch_counters() == (0, 1)
    want = old(hidden, **{k: v for k, v in theirs.items() if k != "banks"})
    assert torch.equal(got, want)
    for a, b in zip(mine["banks"], theirs["banks"]):
        assert torch.equal(a, b)
