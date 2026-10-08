# SPDX-License-Identifier: Apache-2.0
"""``mhc_pre`` through the fused kernel against 0a08ff4's ``mhc_pre``, at B in {1, 4, 64}.

Both sides are ``Glm5NextHyperConnection.mhc_pre`` on the same bf16 streams and the same
site weights (checkpoint layer 4's attention site when the checkpoint is present): this
tree's, which runs ``functional/glue/mhc_pre.py`` (the mixes, the gates, the softmax and
the pre-weighted collapse in one launch) before the unchanged Sinkhorn, and the 0a08ff4
snapshot's, which runs that region as torch ops. The Sinkhorn kernel is the same file on
both sides.

Tolerance, and why. The fused kernel multiplies bf16 by bf16 on the tensor engine, which
is exact in its fp32 accumulator, so the mixes and the square sum differ from the fp32
reference only by summation order (16384 terms): observed |d mix| ~1e-6 relative. The
gates are smooth (|d sigmoid| <= |dz| / 4), so ``post_mix`` and ``comb_mix`` are asserted
to ``POST_ATOL`` / ``COMB_ATOL``. ``layer_input`` is bf16 on both sides: a pre weight
that moved by ~1e-7 can move a product across a bf16 rounding boundary, so an element
may differ by one bf16 step of its own magnitude and no more, in at most
``LAYER_INPUT_MAX_FLIPS`` of the elements.

fp32 operands (the CPU fixtures' dtype; the served checkpoint holds ``fn`` and the
streams in bf16) run the same kernel on fp32 tiles: an fp32 ``layer_input`` is asserted
to ``FP32_INPUT_RTOL * max|ref|`` (the pre weights' ~1e-7 relative change, and the
four-stream sum's order), a bf16 one to one bf16 step plus that same absolute term: the
four-stream sum can cancel (observed: outputs of ~1e-6 from terms of ~1, off by 3e-7 =
1e-7 of max|ref|), and a cancelled output carries the error of its terms, not of its
own magnitude.
"""

from __future__ import annotations

import os

import pytest
import torch

from test.hardware.baselines.glue_0a08ff4 import load as load_0a08ff4
from test.vllm_neuron.functional.glue import glue_case
from vllm_neuron.functional import glue
from vllm_neuron.functional.glue import mhc_pre as fused
from vllm_neuron.model.glm5_next import model_fp8

BATCHES = (1, 4, 64)
POST_ATOL = 2e-5
COMB_ATOL = 2e-5
#: Share of ``layer_input`` elements allowed one bf16 step away from 0a08ff4.
LAYER_INPUT_MAX_FLIPS = 0.01
#: fp32 ``layer_input``: ``|d| <= FP32_INPUT_RTOL * max|ref|``.
FP32_INPUT_RTOL = 1e-5


def _site(model, src, fn_dtype=torch.bfloat16, neuron_config=None):
    """An mHC site with checkpoint layer 4's attention-site weights.

    ``neuron_config`` defaults to the config's own (the serving line's buckets).
    """
    cfg = glue_case.text_config()
    site = model.Glm5NextHyperConnection(
        cfg, neuron_config=cfg.neuron_config if neuron_config is None else neuron_config)
    prefix = f"model.language_model.layers.{glue_case.KDA_LAYER}."
    streams, hidden = int(cfg.hc_mult), int(cfg.hidden_size)
    mix = (2 + streams) * streams
    site.fn.data = src.get(f"{prefix}hc_attn_fn", (mix, streams * hidden), fn_dtype,
                           scale=(streams * hidden) ** -0.5)
    site.hc_scale.data = src.get(f"{prefix}hc_attn_scale", (3,), torch.float32,
                                 scale=0.1, offset=1.0)
    site.hc_base.data = src.get(f"{prefix}hc_attn_base", (mix,), torch.float32, scale=0.5)
    return site, cfg


@pytest.fixture(scope="module")
def sites():
    base = load_0a08ff4().model_fp8
    live, cfg = _site(model_fp8, glue_case._Source(seed=4))
    old, _ = _site(base, glue_case._Source(seed=4))
    return live, old, cfg


@pytest.fixture(scope="module")
def fp32_fn_sites():
    """A site whose ``fn`` is fp32 values that bf16 cannot hold (random, no checkpoint)."""
    base = load_0a08ff4().model_fp8
    live, cfg = _site(model_fp8, glue_case._Source(seed=5, use_checkpoint=False),
                      torch.float32)
    old, _ = _site(base, glue_case._Source(seed=5, use_checkpoint=False), torch.float32)
    return live, old, cfg


def _bf16_step(*values: torch.Tensor) -> torch.Tensor:
    """One bf16 step at the larger magnitude of each element pair (2^-7 of its binade).

    The larger of the two, because a one-step rounding flip can cross a binade
    boundary, where the lower binade's step is half the flip.
    """
    mag = torch.stack([v.float().abs() for v in values]).amax(0).clamp_min(2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7.0)


@pytest.mark.parametrize("batch", BATCHES)
def test_fused_mhc_pre_matches_0a08ff4(sites, batch):
    live, old, cfg = sites
    streams = glue_case.streams_input(cfg, batch)
    fused.reset_dispatch_counters()
    post, comb, layer_input = live.mhc_pre(streams)
    assert fused.dispatch_counters() == (1, 0), "the fused kernel did not serve mhc_pre"
    want_post, want_comb, want_input = old.mhc_pre(streams)

    assert post.shape == want_post.shape and post.dtype == want_post.dtype
    assert comb.shape == want_comb.shape and comb.dtype == want_comb.dtype
    assert layer_input.shape == want_input.shape and layer_input.dtype == want_input.dtype
    torch.testing.assert_close(post, want_post, atol=POST_ATOL, rtol=0)
    torch.testing.assert_close(comb, want_comb, atol=COMB_ATOL, rtol=0)
    diff = (layer_input.float() - want_input.float()).abs()
    assert bool((diff <= _bf16_step(want_input, layer_input)).all()), float(diff.max())
    flips = float((diff > 0).float().mean())
    assert flips <= LAYER_INPUT_MAX_FLIPS, flips


@pytest.mark.parametrize("batch", (1, 64))
def test_two_programs_agree_with_one(sites, batch, monkeypatch):
    live, _, cfg = sites
    streams = glue_case.streams_input(cfg, batch)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "1")
    one = live.mhc_pre(streams)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    fused.reset_dispatch_counters()
    two = live.mhc_pre(streams)
    assert fused.dispatch_counters() == (1, 0)
    assert fused.launch_programs() == 2
    torch.testing.assert_close(two[0], one[0], atol=POST_ATOL, rtol=0)
    torch.testing.assert_close(two[1], one[1], atol=COMB_ATOL, rtol=0)
    diff = (two[2].float() - one[2].float()).abs()
    assert bool((diff <= _bf16_step(one[2], two[2])).all())


def test_the_kill_switch_restores_0a08ff4_bit_for_bit(sites, monkeypatch):
    live, old, cfg = sites
    streams = glue_case.streams_input(cfg, 4)
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, "0")
    fused.reset_dispatch_counters()
    got = live.mhc_pre(streams)
    assert fused.dispatch_counters() == (0, 1)
    want = old.mhc_pre(streams)
    for a, b in zip(got, want):
        assert torch.equal(a, b)


def test_prefill_token_counts_keep_the_0a08ff4_route(sites):
    live, old, cfg = sites
    streams = glue_case.streams_input(cfg, fused.MHC_PRE_MAX_TOKENS + 1)
    fused.reset_dispatch_counters()
    got = live.mhc_pre(streams)
    assert fused.dispatch_counters() == (0, 1)
    want = old.mhc_pre(streams)
    for a, b in zip(got, want):
        assert torch.equal(a, b)


@pytest.mark.parametrize("fn_dtype,streams_dtype", (
    (torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16),
    (torch.float32, torch.float32)))
@pytest.mark.parametrize("batch", (1, 64))
def test_fp32_operands_match_0a08ff4(sites, fp32_fn_sites, batch, fn_dtype, streams_dtype):
    """fp32 ``fn`` or streams (the CPU fixtures' dtypes) take the kernel too."""
    live, old, cfg = fp32_fn_sites if fn_dtype == torch.float32 else sites
    assert live.fn.dtype == fn_dtype
    streams = glue_case.streams_input(cfg, batch).to(streams_dtype)
    fused.reset_dispatch_counters()
    post, comb, layer_input = live.mhc_pre(streams)
    assert fused.dispatch_counters() == (1, 0), "the fused kernel did not serve mhc_pre"
    want_post, want_comb, want_input = old.mhc_pre(streams)
    assert layer_input.dtype == want_input.dtype == streams_dtype
    torch.testing.assert_close(post, want_post, atol=POST_ATOL, rtol=0)
    torch.testing.assert_close(comb, want_comb, atol=COMB_ATOL, rtol=0)
    diff = (layer_input.float() - want_input.float()).abs()
    if streams_dtype == torch.float32:
        bound = FP32_INPUT_RTOL * float(want_input.abs().max())
        assert float(diff.max()) <= bound, (float(diff.max()), bound)
    else:
        bound = (_bf16_step(want_input, layer_input)
                 + FP32_INPUT_RTOL * float(want_input.float().abs().max()))
        assert bool((diff <= bound).all()), float(diff.max())
        assert float((diff > 0).float().mean()) <= LAYER_INPUT_MAX_FLIPS


def test_kernel_mixes_match_a_float64_reference(sites):
    """The kernel's own outputs against float64, before any rounding to bf16."""
    live, _, cfg = sites
    streams = glue_case.streams_input(cfg, 4)
    post, comb, layer_input = fused.mhc_pre_fused(
        streams, live.fn, live.hc_scale, live.hc_base, rms_eps=live.rms_eps,
        hc_eps=live.hc_eps, post_mult=live.post_mult_value)
    flat = streams.reshape(4, -1).double()
    mixes = flat @ live.fn.double().t()
    mixes = mixes * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + live.rms_eps)
    s, b = live.hc_scale.double(), live.hc_base.double()
    pre = torch.sigmoid(mixes[:, :4] * s[0] + b[:4]) + live.hc_eps
    want_post = torch.sigmoid(mixes[:, 4:8] * s[1] + b[4:8]) * live.post_mult_value
    logits = mixes[:, 8:].reshape(4, 4, 4) * s[2] + b[8:].reshape(1, 4, 4)
    want_comb = torch.softmax(logits, dim=-1) + live.hc_eps
    want_input = (pre.unsqueeze(-1) * streams.double()).sum(dim=1)
    assert tuple(post.shape) == (4, 4, 1) and tuple(comb.shape) == (4, 4, 4)
    torch.testing.assert_close(post.double().reshape(4, 4), want_post, atol=1e-5, rtol=0)
    torch.testing.assert_close(comb.double(), want_comb, atol=1e-5, rtol=0)
    diff = (layer_input.double() - want_input).abs()
    assert bool((diff <= _bf16_step(want_input, layer_input)).all())


def test_launch_programs_follow_the_lnc(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert fused.launch_programs() == 2
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert fused.launch_programs() == 1
    assert os.environ.get("NEURON_LOGICAL_NC_CONFIG") is None
