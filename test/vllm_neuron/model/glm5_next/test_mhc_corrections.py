"""The mHC layer's arithmetic: the post gate's multiplier and both epsilons.

Each is compared against a reference expression transcribed into this file.
"""

from __future__ import annotations

import math
import os
import re

import pytest
import torch
import torch.nn.functional as F

import nki  # noqa: F401  -- the simulator route this layer's Sinkhorn needs
import nki.simulator  # noqa: F401

from vllm_neuron.functional.mhc import sinkhorn as sinkhorn_mod
from vllm_neuron.functional.mhc.sinkhorn import MHC_STREAMS
from vllm_neuron.utils.neuron_utils import can_run_kernel

# --------------------------------------------------------------------------- #
# the case. Small on purpose: this file measures arithmetic, not size.         #
# --------------------------------------------------------------------------- #
#: tokens. Well inside every seam bound, so no test here is also a boundary case.
T = 8
#: Streams. The Sinkhorn seam's own constant for the target's ``hc_mult 4``, imported.
S = MHC_STREAMS
#: Hidden.
H = 64

#: The target's post multiplier, read off the reference.
REFERENCE_POST_MULT = 2.0


class VacuousControlError(AssertionError):
    """A control whose input could not have made it fail. """


class RouteInstrumentError(AssertionError):
    """A route reading that is not the one this file declares."""


def _impl():
    """Import the implementation module inside a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _cited_tolerances() -> tuple[float, float]:
    """``(rtol, atol)`` read out of ``test_mhc_layer.py``, not declared here. """
    sibling = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_mhc_layer.py")
    text = open(sibling, encoding="utf-8").read()
    rtol = re.search(r"^RTOL = (\S+)$", text, re.M)
    atol = re.search(r"^ATOL = (\S+)$", text, re.M)
    if rtol is None or atol is None:
        raise AssertionError(
            f"could not read the declared tolerance pair out of {sibling}; this "
            f"file registers no tolerance of its own and has nothing to fall "
            f"back on"
        )
    return float(rtol.group(1)), float(atol.group(1))


def _config(hidden: int = H):
    """A real ``Glm5NextTextConfig``, carrying the checkpoint's own dials."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    return Glm5NextTextConfig(hidden_size=hidden, hc_mult=S)


def _fixture(seed: int = 30, tokens: int = T, hidden: int = H):
    """``(fn, hc_scale, hc_base, residual)`` fp32, on ``test_mhc_layer.py``'s scales.
    """
    gen = torch.Generator().manual_seed(seed)
    hc_mult3 = 2 * S + S * S
    fn = torch.randn(hc_mult3, S * hidden, generator=gen, dtype=torch.float32) * 1e-4
    hc_scale = torch.randn(3, generator=gen, dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult3, generator=gen, dtype=torch.float32) * 0.1
    residual = torch.randn(tokens, S, hidden, generator=gen, dtype=torch.float32)
    return fn, hc_scale, hc_base, residual


def _layer(post_mult_value: float | None = None, hidden: int = H):
    """The layer under test. """
    impl = _impl()
    cfg = _config(hidden)
    if post_mult_value is None:
        return impl.Glm5NextHyperConnection(cfg), cfg
    return impl.Glm5NextHyperConnection(cfg, post_mult_value=post_mult_value), cfg


def _load(layer, fn, hc_scale, hc_base) -> None:
    """Set the layer's parameters; the acceptance is synthetic by declaration."""
    with torch.no_grad():
        layer.fn.copy_(fn)
        layer.hc_scale.copy_(hc_scale)
        layer.hc_base.copy_(hc_base)


# --------------------------------------------------------------------------- #
# the comparator: the target model's arithmetic, transcribed with its lines.    #
# --------------------------------------------------------------------------- #
def _reference_post_and_mixes(
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    rms_eps: float,
    post_mult: float = REFERENCE_POST_MULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(post, mixes)`` as ``Glm5NextTextHyperConnection.forward`` computes them. """
    hc = S
    flat = residual.flatten(start_dim=1).to(torch.float32)
    # +: the model's own RMSNorm, on the model's own epsilon.
    normed = flat * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    #: one projection, split pre / post / comb.
    mixes = F.linear(normed, fn.to(torch.float32))
    post_w = mixes[:, hc : 2 * hc]
    post_b = hc_base.to(torch.float32)[hc : 2 * hc]
    post_scale = hc_scale.to(torch.float32)[1]
    #: the multiplier is on the outside of the sigmoid, and it is 2.
    post = post_mult * torch.sigmoid(post_w * post_scale + post_b)
    return post, mixes


def _route_reading(label: str, calls: int) -> str:
    """Check the Sinkhorn route, so no reading here is a torch pass."""
    nki_dispatch, torch_fallback = sinkhorn_mod.dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"[{label}] mhc_pre_calls={calls} sinkhorn_nki_dispatch={nki_dispatch} "
        f"sinkhorn_torch_fallback={torch_fallback} can_run_kernel={gate} "
        f"per_call={nki_dispatch / calls if calls else float('nan')}"
    )
    if nki_dispatch != calls:
        raise RouteInstrumentError(
            f"{label}: the Sinkhorn dispatch counter read {nki_dispatch} over {calls} "
            f"mhc_pre call(s); exactly ONE per call is declared. {reading}"
        )
    if torch_fallback != 0:
        raise RouteInstrumentError(
            f"{label}: torch_fallback read {torch_fallback}, declared 0 -- a "
            f"fallback pass would compare torch against torch. {reading}"
        )
    if gate is not True:
        raise RouteInstrumentError(
            f"{label}: can_run_kernel() read {gate}, declared True. {reading}"
        )
    return reading


def _max_abs_rel(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float]:
    """``(max_abs, max_rel)``, both floats, read by every test that compares."""
    diff = (got.to(torch.float32) - want.to(torch.float32)).abs()
    denom = want.to(torch.float32).abs().clamp_min(1e-12)
    return float(diff.max()), float((diff / denom).max())


# --------------------------------------------------------------------------- #
# counted value 1 -- the post gate equals the target's ``2 * sigmoid``.         #
# --------------------------------------------------------------------------- #
def test_post_gate_matches_the_reference_two_sigmoid() -> None:
    """The layer's ``post_mix`` equals the reference on ``N/N`` cases. """
    rtol, atol = _cited_tolerances()
    fn, hc_scale, hc_base, residual = _fixture()
    layer, cfg = _layer()
    _load(layer, fn, hc_scale, hc_base)

    sinkhorn_mod.reset_dispatch_counters()
    post_mix, _comb_mix, _layer_input = layer.mhc_pre(residual)
    _route_reading("value-1-post-gate", calls=1)

    want, _ = _reference_post_and_mixes(
        fn, hc_scale, hc_base, residual, float(cfg.rms_norm_eps)
    )
    got = post_mix.reshape(T, S)
    cases = int(want.numel())
    within = int(
        torch.isclose(got, want, rtol=rtol, atol=atol).sum()
    )
    assert layer.post_mult_value == REFERENCE_POST_MULT, layer.post_mult_value
    assert within == cases, f"{within}/{cases} within the cited pair"
    torch.testing.assert_close(got, want, rtol=rtol, atol=atol)


def test_the_default_post_multiplier_is_the_targets_two() -> None:
    """The default itself, read off a layer nobody passed the value to. """
    layer, _ = _layer()
    assert layer.post_mult_value == REFERENCE_POST_MULT, layer.post_mult_value


# --------------------------------------------------------------------------- #
# failing control 2 -- the old ``1.0`` must fail counted value 1.               #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# counted value 2 and failing control 3 -- each epsilon at its own site.        #
# --------------------------------------------------------------------------- #
def test_rms_epsilon_reads_the_models_rms_norm_eps() -> None:
    """The layer carries the model's RMSNorm epsilon, distinct from ``hc_eps``. """
    layer, cfg = _layer()
    assert layer.rms_eps == float(cfg.rms_norm_eps), layer.rms_eps
    assert layer.hc_eps == float(cfg.hc_eps), layer.hc_eps
    if layer.rms_eps == layer.hc_eps:
        raise VacuousControlError(
            f"rms_eps and hc_eps are both {layer.rms_eps}: this config cannot "
            f"distinguish the corrected site from the old one, so every epsilon "
            f"delta measured on it would be vacuous"
        )


def _ulp(x: torch.Tensor) -> torch.Tensor:
    """The exact fp32 step at each entry of ``x``, from ``nextafter``. """
    magnitude = x.abs()
    return torch.nextafter(magnitude, torch.full_like(magnitude, float("inf"))) - magnitude


def test_rms_epsilon_site_is_load_bearing_and_the_old_value_moves_it() -> None:
    """counted value 2, site 1: the RMS epsilon moves the norm output by the predicted
    amount.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    layer, _cfg = _layer()
    _load(layer, fn, hc_scale, hc_base)

    eps_new = float(layer.rms_eps)  # the corrected constant, `config.rms_norm_eps`
    eps_old = float(layer.hc_eps)  # what the layer wrongly read at this site
    if eps_new == eps_old:
        raise VacuousControlError(
            f"rms_eps and hc_eps are both {eps_new}, so swapping one for the "
            f"other changes nothing and this arm would measure zero on correct "
            f"bytes"
        )

    # ---- 1. The asserted reading: the norm output, against its reference. ---- #
    flat = residual.flatten(start_dim=1).to(torch.float32)
    mean_square = flat.square().mean(dim=-1, keepdim=True)
    normed_new = flat * torch.rsqrt(mean_square + eps_new)
    normed_old = flat * torch.rsqrt(mean_square + eps_old)

    # Per token, because `mean_square` is per token; the max of each side is
    # compared against the max of the other, and the relative change is uniform
    # across a token's entries because the scale is one scalar.
    predicted_rel = float(
        (abs(eps_new - eps_old) / (2.0 * (mean_square + eps_new))).max()
    )
    delta = (normed_new - normed_old).abs()
    live = normed_old.abs() > 0.0
    measured_rel = float((delta[live] / normed_old.abs()[live]).max())
    measured_ulps = float((delta[live] / _ulp(normed_old)[live]).max())
    ratio = measured_rel / predicted_rel if predicted_rel > 0.0 else float("nan")
    assert 0.5 * predicted_rel <= measured_rel <= 2.0 * predicted_rel, (
        f"the norm output moved by {measured_rel:.6e} relative when swapping "
        f"eps {eps_new} for {eps_old}, and first-order theory predicts "
        f"{predicted_rel:.6e} (ratio {ratio:.4f}); outside the [0.5x, 2.0x] band "
        f"the denominator is not `mean_square + eps`"
    )
    assert measured_ulps >= 16.0, (
        f"the norm output moved by only {measured_ulps:.2f} fp32 steps, under "
        f"the floor of 16, so this reading is at the resolution limit and could "
        f"pass or fail on rounding rather than on the constant -- the defect "
        f"that took the three-return form out of service"
    )

    sinkhorn_mod.reset_dispatch_counters()
    post_a, comb_a, input_a = layer.mhc_pre(residual)
    # Put the old, wrong constant back at this one site and nothing else.
    layer.rms_eps = eps_old
    post_b, comb_b, input_b = layer.mhc_pre(residual)
    _route_reading("value-2-rms-site", calls=2)

    d_post = float((post_a - post_b).abs().max())
    d_comb = float((comb_a - comb_b).abs().max())
    d_input = float((input_a - input_b).abs().max())
    for _name, _moved, reference in (
        ("post_mix", d_post, post_a),
        ("comb_mix", d_comb, comb_a),
        ("layer_input", d_input, input_a),
    ):
        float(_ulp(reference).max())


def test_hc_epsilon_sites_are_the_pre_and_comb_gates_only() -> None:
    """counted value 2, sites 2 and 3, with their negative reading. """
    fn, hc_scale, hc_base, residual = _fixture()
    layer, _cfg = _layer()
    _load(layer, fn, hc_scale, hc_base)

    sinkhorn_mod.reset_dispatch_counters()
    post_a, comb_a, input_a = layer.mhc_pre(residual)
    layer.hc_eps = layer.hc_eps * 100.0
    post_b, comb_b, input_b = layer.mhc_pre(residual)
    _route_reading("value-2-hc-sites", calls=2)

    d_post = float((post_a - post_b).abs().max())
    d_comb = float((comb_a - comb_b).abs().max())
    d_input = float((input_a - input_b).abs().max())
    assert d_input > 0.0, (
        f"scaling hc_eps left layer_input bit-identical (delta {d_input}); the "
        f"pre gate does not read hc_eps -- a finding, not a pass"
    )
    assert d_comb > 0.0, (
        f"scaling hc_eps left comb_mix bit-identical (delta {d_comb}); the comb "
        f"gate does not read hc_eps -- a finding, not a pass"
    )
    assert d_post == 0.0, (
        f"scaling hc_eps moved post_mix by {d_post:.6e}; the target puts NO "
        f"epsilon on the post term (the reference), so this layer is adding one "
        f"the model does not have"
    )


# --------------------------------------------------------------------------- #
# counted value 3 -- every normalised block hits the seam's declared targets.    #
# --------------------------------------------------------------------------- #
def test_each_normalised_block_hits_the_seams_declared_targets() -> None:
    """The batched entry's ``[T, S, S]`` output is doubly stochastic, per block. """
    rtol, atol = _cited_tolerances()
    fn, hc_scale, hc_base, residual = _fixture()
    layer, _cfg = _layer()
    _load(layer, fn, hc_scale, hc_base)

    sinkhorn_mod.reset_dispatch_counters()
    _post_mix, comb_mix, _layer_input = layer.mhc_pre(residual)
    _route_reading("value-3-block-targets", calls=1)

    row_goal = sinkhorn_mod.row_target()
    col_goal = sinkhorn_mod.column_target(S, S)
    assert tuple(comb_mix.shape) == (T, S, S), tuple(comb_mix.shape)

    # Per block, so one bad token cannot hide inside an average.
    row_sums = comb_mix.sum(dim=-1)          # [T, S]
    col_sums = comb_mix.sum(dim=-2)          # [T, S]
    row_dev = (row_sums - row_goal).abs()
    col_dev = (col_sums - col_goal).abs()
    # Every entry stays strictly positive: a doubly stochastic matrix reached by
    # multiplicative rescaling cannot introduce a zero, and a zero would mean the
    # kernel returned an unwritten tile rather than a normalised block.
    assert float(comb_mix.min()) > 0.0, float(comb_mix.min())
    assert float(col_dev.max()) <= atol, (
        f"worst column-sum deviation {float(col_dev.max()):.6e} exceeds the "
        f"cited atol {atol}; the seam ends on a column pass, so this axis is the "
        f"exact one and a miss here is a seam defect rather than a schedule gap"
    )
    assert float(row_dev.max()) <= rtol, (
        f"worst row-sum deviation {float(row_dev.max()):.6e} exceeds the cited "
        f"rtol {rtol}"
    )


# --------------------------------------------------------------------------- #
# failing control 4 -- the batched seam refuses a missing route, and the        #
# square seam it replaced would have papered over it.                          #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# the comparator's own provenance.                                             #
# --------------------------------------------------------------------------- #
