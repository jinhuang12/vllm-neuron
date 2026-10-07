# SPDX-License-Identifier: Apache-2.0
"""KDA gated output norm + ``o_proj`` through the fused kernel against 0a08ff4.

0a08ff4's ``_gated_output`` normalises ``core`` over the head extent, applies the gain
and ``sigmoid(out_gate)``, and multiplies by ``o_proj.float().t()`` in fp32. The kernel
does the same fp32 elementwise steps in the same order and the projection on the
tensor engine as two bf16 passes, ``g = hi + lo`` (``hi = bf16(g)``,
``lo = bf16(g - hi)``): each product is exact in fp32, and ``g - hi - lo`` is at most
2^-18 of ``|g|``. So the fp32 result is asserted to ``ATTN_RTOL * max|ref|`` against
the 0a08ff4 expression (an absolute bound at the output's scale: an output that
cancels to ~1e-5 carries the error of its ~1e-2 terms), and the layer's bf16 output
(the cast after the reduction) to that bound plus one bf16 step of each element, with
at most ``OUTPUT_MAX_FLIPS`` of the elements changed (observed 0.2-0.3%).

An fp32 ``o_proj`` (the CPU fixtures' dtype; the served checkpoint is bf16) takes one
fp32 pass instead of the two bf16 ones, asserted to the same ``ATTN_RTOL``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from test.hardware.baselines.glue_0a08ff4 import load as load_0a08ff4
from test.vllm_neuron.functional.glue import glue_case
from vllm_neuron.functional.glue import kda_output as fused
from vllm_neuron.model.glm5_next import model_fp8

BATCHES = (1, 4, 64)
ATTN_RTOL = 1e-5
OUTPUT_MAX_FLIPS = 0.01


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


def _operands(batch):
    gen = torch.Generator().manual_seed(70 + batch)
    core = torch.randn(batch, 128, generator=gen) * 0.2
    gate = torch.randn(batch, 128, generator=gen) * 2.0
    hidden = torch.randn(batch, 4096, generator=gen).to(torch.bfloat16)
    return core, gate, hidden


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
def test_kernel_matches_the_0a08ff4_expression(pair, batch, lnc, dma, monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    monkeypatch.setenv(fused.DMA_TRANSPOSE_ENV, dma)
    live, _, _ = pair
    core, gate, _ = _operands(batch)
    fused.reset_dispatch_counters()
    got = fused.kda_gated_projection(core, gate, live)
    assert fused.dispatch_counters() == (1, 0)
    want = fused.kda_gated_projection_torch(core, gate, live)
    assert got.shape == want.shape == (batch, 4096) and got.dtype == torch.float32
    bound = ATTN_RTOL * float(want.abs().max())
    assert float((got - want).abs().max()) <= bound


@pytest.mark.parametrize("dma", ("1", "0"))
@pytest.mark.parametrize("lnc", ("1", "2"))
@pytest.mark.parametrize("batch", (1, 64))
@pytest.mark.parametrize("gain_dtype", (torch.bfloat16, torch.float32))
def test_fp32_o_proj_matches_the_0a08ff4_expression(pair, batch, lnc, dma, gain_dtype,
                                                    monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    monkeypatch.setenv(fused.DMA_TRANSPOSE_ENV, dma)
    live, _, _ = pair
    gen = torch.Generator().manual_seed(80 + batch)
    o_proj = live.o_proj_weight.float()
    # fp32 values bf16 cannot hold: a 2^-8 relative jitter on each element.
    o_proj = o_proj + torch.randn(o_proj.shape, generator=gen) * float(
        o_proj.abs().mean()) * 2.0 ** -8
    attn = SimpleNamespace(
        o_proj_weight=o_proj, o_norm_weight=live.o_norm_weight.to(gain_dtype),
        num_kv_heads_per_rank=live.num_kv_heads_per_rank, head_dim=live.head_dim,
        rms_norm_eps=live.rms_norm_eps)
    core, gate, _ = _operands(batch)
    fused.reset_dispatch_counters()
    got = fused.kda_gated_projection(core, gate, attn)
    assert fused.dispatch_counters() == (1, 0)
    want = fused.kda_gated_projection_torch(core, gate, attn)
    assert got.shape == want.shape == (batch, 4096) and got.dtype == torch.float32
    bound = ATTN_RTOL * float(want.abs().max())
    assert float((got - want).abs().max()) <= bound


@pytest.mark.parametrize("batch", BATCHES)
def test_gated_output_matches_0a08ff4(pair, batch):
    live, old, _ = pair
    core, gate, hidden = _operands(batch)
    fused.reset_dispatch_counters()
    got = live._gated_output(core, gate, hidden)
    assert fused.dispatch_counters() == (1, 0)
    want = old._gated_output(core, gate, hidden)
    assert got.dtype == want.dtype == torch.bfloat16
    diff = (got.float() - want.float()).abs()
    bound = _bf16_step(want, got) + ATTN_RTOL * float(want.float().abs().max())
    assert bool((diff <= bound).all()), float(diff.max())
    assert float((diff > 0).float().mean()) <= OUTPUT_MAX_FLIPS


def test_the_kill_switch_restores_0a08ff4_bit_for_bit(pair, monkeypatch):
    live, old, _ = pair
    monkeypatch.setenv("VLLM_NEURON_GLUE_FUSED", "0")
    core, gate, hidden = _operands(4)
    fused.reset_dispatch_counters()
    got = live._gated_output(core, gate, hidden)
    assert fused.dispatch_counters() == (0, 1)
    assert torch.equal(got, old._gated_output(core, gate, hidden))
