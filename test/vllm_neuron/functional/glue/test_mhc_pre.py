# SPDX-License-Identifier: Apache-2.0
"""``mhc_pre`` through the fused kernel against 0a08ff4's ``mhc_pre``, at both mHC sites.

Both sides are ``Glm5NextHyperConnection.mhc_pre`` on the same bf16 streams and the same
site weights (checkpoint layer 4's attention or feed-forward site when the checkpoint is
present): this tree's, which runs ``functional/glue/mhc_pre.py`` (the mixes, the gates,
the softmax and the pre-weighted collapse in one launch) before the unchanged Sinkhorn,
and the 0a08ff4 snapshot's, which runs that region as torch ops. The Sinkhorn kernel is
the same file on both sides. Row counts: the decode batches B in {1, 4, 64}, one token
tile (:data:`~vllm_neuron.functional.glue.mhc_pre.MHC_PRE_TOKEN_TILE`), the served
1024-row prefill chunk, and 200 rows (no bucket: a whole tile, then a partial one).

The feed-forward norm. At the feed-forward site the kernel also returns the sub-block's
RMSNorm of ``layer_input`` (``mhc_pre_normed``). Against ``Glm5NextModel._rms_norm`` of
the kernel's own ``layer_input`` the two differ only in the square sum's order (4096
exact bf16 squares, summed in fp32) and the rsqrt: an fp32 value off by a few ulp,
so after the rounding to bf16 an element is equal or one bf16 step away, in at most
``LAYER_INPUT_MAX_FLIPS`` of the elements. Against a norm of another ``layer_input``
``b`` (the torch chain: ``_rms_norm`` of 0a08ff4's; or the other program count's), each
element is ``a * r * g`` against ``b * r' * g``, with ``r`` and ``r'`` the rows' rsqrt
factors, so before rounding it differs by at most ``|a - b| * r' * |g|`` plus the factors'
relative difference (``FP32_INPUT_RTOL``; a few flipped elements in a row of 4096 move
its mean square by far less), and the two roundings add one bf16 step:
:func:`_norm_bound`.

Tolerance, and why. The fused kernel multiplies bf16 by bf16 on the tensor engine, which
is exact in its fp32 accumulator, so the mixes and the square sum differ from the fp32
reference only by summation order (16384 terms): observed |d mix| ~1e-6 relative. The
gates are smooth (|d sigmoid| <= |dz| / 4), so ``post_mix`` and ``comb_mix`` are asserted
to ``POST_ATOL`` / ``COMB_ATOL``. ``layer_input`` is bf16 on both sides: a pre weight
that moved by ~1e-7 can move a product across a bf16 rounding boundary, so an element
may differ by one bf16 step of its own magnitude, in at most ``LAYER_INPUT_MAX_FLIPS``
of the elements, plus ``FP32_INPUT_RTOL * max|ref|`` where the four-stream sum cancels
(below): at 1024 rows (4M elements) such outputs occur (observed: -1.13e-6 against
-1.31e-6 from terms of ~1, 24 steps of their own tiny magnitude).

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
#: One token tile, the served prefill chunk, and two tiles of which the second is
#: partial (72 rows: also a partial 32-row transpose chunk).
TILED_ROWS = (fused.MHC_PRE_TOKEN_TILE, glue_case.SERVED_PREFILL_BUCKETS[0], 200)
POST_ATOL = 2e-5
COMB_ATOL = 2e-5
#: Share of ``layer_input`` elements allowed one bf16 step away from 0a08ff4.
LAYER_INPUT_MAX_FLIPS = 0.01
#: fp32 ``layer_input``: ``|d| <= FP32_INPUT_RTOL * max|ref|``.
FP32_INPUT_RTOL = 1e-5


def _site(model, src, fn_dtype=torch.bfloat16, neuron_config=None, kind="attn"):
    """An mHC site with checkpoint layer 4's attention-site (``kind="ffn"``: FFN) weights.

    ``neuron_config`` defaults to the config's own (the serving line's buckets).
    """
    cfg = glue_case.text_config()
    site = model.Glm5NextHyperConnection(
        cfg, neuron_config=cfg.neuron_config if neuron_config is None else neuron_config)
    prefix = f"model.language_model.layers.{glue_case.KDA_LAYER}."
    streams, hidden = int(cfg.hc_mult), int(cfg.hidden_size)
    mix = (2 + streams) * streams
    site.fn.data = src.get(f"{prefix}hc_{kind}_fn", (mix, streams * hidden), fn_dtype,
                           scale=(streams * hidden) ** -0.5)
    site.hc_scale.data = src.get(f"{prefix}hc_{kind}_scale", (3,), torch.float32,
                                 scale=0.1, offset=1.0)
    site.hc_base.data = src.get(f"{prefix}hc_{kind}_base", (mix,), torch.float32,
                                scale=0.5)
    return site, cfg


@pytest.fixture(scope="module")
def sites():
    base = load_0a08ff4().model_fp8
    live, cfg = _site(model_fp8, glue_case._Source(seed=4))
    old, _ = _site(base, glue_case._Source(seed=4))
    return live, old, cfg


@pytest.fixture(scope="module")
def ffn_sites():
    """The feed-forward site, its norm's gain and epsilon, and ``_rms_norm`` to check it."""
    base = load_0a08ff4().model_fp8
    live, cfg = _site(model_fp8, glue_case._Source(seed=6), kind="ffn")
    old, _ = _site(base, glue_case._Source(seed=6), kind="ffn")
    hidden = int(cfg.hidden_size)
    prefix = f"model.language_model.layers.{glue_case.KDA_LAYER}."
    gain = glue_case._Source(seed=6).get(f"{prefix}post_attention_layernorm.weight",
                                         (hidden,), torch.bfloat16, scale=0.05, offset=1.0)
    owner = glue_case._ffn_owner(model_fp8, cfg)
    return live, old, cfg, gain, float(cfg.rms_norm_eps), owner._rms_norm


def _assert_mixes_match(got, want):
    """``post_mix``, ``comb_mix`` and a bf16 ``layer_input`` at this module's tolerances."""
    post, comb, layer_input = got
    want_post, want_comb, want_input = want
    assert post.shape == want_post.shape and post.dtype == want_post.dtype
    assert comb.shape == want_comb.shape and comb.dtype == want_comb.dtype
    assert layer_input.shape == want_input.shape and layer_input.dtype == want_input.dtype
    torch.testing.assert_close(post, want_post, atol=POST_ATOL, rtol=0)
    torch.testing.assert_close(comb, want_comb, atol=COMB_ATOL, rtol=0)
    _assert_input_match(layer_input, want_input)


def _assert_input_match(got: torch.Tensor, want: torch.Tensor) -> None:
    """A bf16 ``layer_input``: one step, plus the cancellation term; few flips."""
    diff = (got.float() - want.float()).abs()
    bound = _bf16_step(want, got) + FP32_INPUT_RTOL * float(want.float().abs().max())
    assert bool((diff <= bound).all()), float((diff - bound).max())
    flips = float((diff > 0).float().mean())
    assert flips <= LAYER_INPUT_MAX_FLIPS, flips


def _norm_bound(normed, other, layer_input, other_input, gain, eps):
    """Elementwise bound on ``|normed - other|``, two norms of two ``layer_input``s.

    ``|a - b| * r_b * |g|`` (``r_b`` the rsqrt factor of ``b``'s row), the factors'
    relative difference on the value, and one bf16 step for the two roundings.
    """
    b = other_input.float()
    r_b = torch.rsqrt(b.square().mean(dim=-1, keepdim=True) + eps)
    spread = (layer_input.float() - b).abs() * r_b * gain.float().abs()
    return spread + FP32_INPUT_RTOL * other.float().abs() + _bf16_step(other, normed)


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


@pytest.mark.parametrize("batch", BATCHES + TILED_ROWS)
def test_fused_mhc_pre_matches_0a08ff4(sites, batch):
    live, old, cfg = sites
    streams = glue_case.streams_input(cfg, batch)
    fused.reset_dispatch_counters()
    got = live.mhc_pre(streams)
    assert fused.dispatch_counters() == (1, 0), "the fused kernel did not serve mhc_pre"
    assert fused.normed_dispatches() == 0
    _assert_mixes_match(got, old.mhc_pre(streams))


@pytest.mark.parametrize("rows", (1,) + TILED_ROWS)
def test_the_ffn_site_kernel_returns_the_ffn_norm(ffn_sites, rows):
    live, old, cfg, gain, eps, rms_norm = ffn_sites
    streams = glue_case.streams_input(cfg, rows)
    fused.reset_dispatch_counters()
    post, comb, layer_input, normed = live.mhc_pre_normed(streams, gain, eps)
    assert fused.dispatch_counters() == (1, 0) and fused.normed_dispatches() == 1
    want = old.mhc_pre(streams)
    _assert_mixes_match((post, comb, layer_input), want)

    assert normed.shape == layer_input.shape and normed.dtype == layer_input.dtype
    own = rms_norm(layer_input, gain)
    diff = (normed.float() - own.float()).abs()
    assert bool((diff <= _bf16_step(own, normed)).all()), float(diff.max())
    assert float((diff > 0).float().mean()) <= LAYER_INPUT_MAX_FLIPS
    chain = rms_norm(want[2], gain)
    bound = _norm_bound(normed, chain, layer_input, want[2], gain, eps)
    diff = (normed.float() - chain.float()).abs()
    assert bool((diff <= bound).all()), float((diff - bound).max())


def test_a_gain_the_kernel_cannot_take_is_declined(ffn_sites):
    live, _, cfg, gain, _, _ = ffn_sites
    streams = glue_case.streams_input(cfg, 4)
    args = (streams, live.fn, live.hc_scale, live.hc_base)
    fused.reset_dispatch_counters()
    assert fused.mhc_pre_admits(*args, norm_gain=gain)
    assert not fused.mhc_pre_admits(*args, norm_gain=gain[: gain.numel() // 2])
    assert not fused.mhc_pre_admits(*args, norm_gain=gain.to(torch.float16))
    assert fused.dispatch_counters() == (0, 2)


def test_a_gain_without_its_epsilon_is_refused(ffn_sites):
    live, _, cfg, gain, _, _ = ffn_sites
    fused.reset_dispatch_counters()
    with pytest.raises(ValueError, match="norm_eps"):
        fused.mhc_pre_fused(glue_case.streams_input(cfg, 4), live.fn, live.hc_scale,
                            live.hc_base, rms_eps=live.rms_eps, hc_eps=live.hc_eps,
                            post_mult=live.post_mult_value, norm_gain=gain)
    assert fused.dispatch_counters() == (0, 0) and fused.normed_dispatches() == 0


def test_a_launch_that_raises_counts_nothing(ffn_sites, monkeypatch):
    """The counters are the evidence of what served, so a failed launch is not one."""
    live, _, cfg, gain, eps, _ = ffn_sites

    def raising(**_operands):
        raise RuntimeError("kernel launch failed")

    tables = {programs: raising for programs in (1, 2)}
    monkeypatch.setattr(fused, "_KERNELS", tables)
    monkeypatch.setattr(fused, "_NORM_KERNELS", tables)
    fused.reset_dispatch_counters()
    for norm in ({}, {"norm_gain": gain, "norm_eps": eps}):
        with pytest.raises(RuntimeError, match="kernel launch failed"):
            fused.mhc_pre_fused(glue_case.streams_input(cfg, 4), live.fn, live.hc_scale,
                                live.hc_base, rms_eps=live.rms_eps, hc_eps=live.hc_eps,
                                post_mult=live.post_mult_value, **norm)
    assert fused.dispatch_counters() == (0, 0) and fused.normed_dispatches() == 0


def test_the_torch_route_leaves_the_norm_to_the_sub_block(ffn_sites, monkeypatch):
    live, old, cfg, gain, eps, _ = ffn_sites
    streams = glue_case.streams_input(cfg, 4)
    monkeypatch.setenv(glue.GLUE_FUSED_ENV, "0")
    fused.reset_dispatch_counters()
    *got, normed = live.mhc_pre_normed(streams, gain, eps)
    assert normed is None and fused.dispatch_counters() == (0, 1)
    for a, b in zip(got, old.mhc_pre(streams)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("batch", (1, 64) + TILED_ROWS)
def test_two_programs_agree_with_one(ffn_sites, batch, monkeypatch):
    """Both sites' kernels (the feed-forward site's, with the norm) under LNC2."""
    live, _, cfg, gain, eps, _ = ffn_sites
    streams = glue_case.streams_input(cfg, batch)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "1")
    one = live.mhc_pre_normed(streams, gain, eps)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    fused.reset_dispatch_counters()
    two = live.mhc_pre_normed(streams, gain, eps)
    plain = live.mhc_pre(streams)
    assert fused.dispatch_counters() == (2, 0) and fused.normed_dispatches() == 1
    assert fused.launch_programs() == 2
    for got in (two[:3], plain):
        _assert_mixes_match(got, one[:3])
    bound = _norm_bound(two[3], one[3], two[2], one[2], gain, eps)
    assert bool(((two[3].float() - one[3].float()).abs() <= bound).all())


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
    post, comb, layer_input, normed = fused.mhc_pre_fused(
        streams, live.fn, live.hc_scale, live.hc_base, rms_eps=live.rms_eps,
        hc_eps=live.hc_eps, post_mult=live.post_mult_value)
    assert normed is None
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
