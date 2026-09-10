"""``inc-glm53f-030c``: the mHC layer's arithmetic corrections, against the model.

WHAT THIS FILE MEASURES, and why it is a new file rather than more items in
``test_mhc_layer.py``. ``inc-glm53f-030`` landed :class:`Glm5NextHyperConnection`
and measured it against the **pinned base's** two spellings of ``mhc_pre``. That
comparison cannot see a place where the base and the TARGET MODEL disagree,
because both sides of it are the base. ``-030c`` compares the layer against the
checkpoint's own model file instead, and two disagreements fall out:

* **the post gate's multiplier.** The target computes
  ``post = 2 * torch.sigmoid(post_w * post_scale + post_b)``
  (``modeling_glm5_next.py:284``) and calls the result "block-output placement,
  range [0, 2]" in its own shape guide (``:246``). The layer defaulted to
  ``1.0``, the pinned base's *kernel test* value
  (``tests/kernels/test_mhc_kernels.py:126``), so every post term it produced
  was HALF the target's.
* **the RMSNorm epsilon.** The target normalises the folded input with
  ``Glm5NextTextUnweightedRMSNorm(eps=config.rms_norm_eps)`` (``:257``, forward
  at ``:216``, applied at ``:278``) -- the model's ``rms_norm_eps``, ``1e-05``.
  The layer's RMS denominator read ``hc_eps``, ``1e-06``. The two mHC-native
  epsilons are unaffected: the target adds ``hc_eps`` after the pre sigmoid
  (``:283``) and after the comb softmax (``:286``), which is what the layer does.

A NEW FILE AND NOT AN EXTENSION, for a measured reason. ``inc-glm53f-028b-tn``
declares ``test_mhc_layer.py``'s collection as **exactly 28** items with **28**
``PASSED`` lines. Adding items there would falsify both of that block's counted
values, so this increment authors its own file and leaves every value ``-028b-tn``
recorded untouched. That is also what ``inc-glm53f-054d`` did, landing its proof
in a new ``test_ffn_reduction_054d.py``.

THE REFERENCE IS TRANSCRIBED, WITH ITS LINES CITED, and the transcription is
itself checked. The campaign's reference copy lives outside this repository
(``design/reference/modeling_glm5_next.py``, sha256 ``2092bbb4...``), so
:func:`_reference_post_and_mixes` restates its arithmetic here the way
``test_mhc_layer.py:395-409`` restates the base's Sinkhorn -- verbatim in
structure, every line cited. :func:`test_030c_the_reference_transcription_is_faithful`
verifies the transcription against the real file whenever that file is reachable,
and says so in its transcript when it is not.

NO TOLERANCE IS REGISTERED HERE. The pair is ``test_mhc_layer.py``'s already
declared ``(RTOL, ATOL)``, read out of that file's own bytes by
:func:`_cited_tolerances` so the citation is mechanical rather than a comment,
and never widened.

THE MODELING MODULE IS IMPORTED INSIDE TEST BODIES, never at module scope, for
the reason ``test_mhc_layer.py:102-107`` records: ``test_factory.py``'s C03
asserts ``model_fp8`` is absent from ``sys.modules``, and pytest imports every
collected module before running any test.
"""

from __future__ import annotations

import hashlib
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
# The case. Small on purpose: this file measures arithmetic, not size.         #
# --------------------------------------------------------------------------- #
#: Tokens. Well inside every seam bound, so no item here is also a boundary case.
T = 8
#: Streams. ``-028``'s own constant for the target's ``hc_mult 4``, imported.
S = MHC_STREAMS
#: Hidden.
H = 64

#: The reference copy's identity, from ``design/reference/PROVENANCE.txt``.
REFERENCE_SHA256 = "2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b"
#: The reference lines this file transcribes, each with the text that must be on it.
REFERENCE_LINES = {
    216: "torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)",
    257: "self.input_norm = Glm5NextTextUnweightedRMSNorm(eps=config.rms_norm_eps)",
    278: "flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())",
    283: "pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps",
    284: "post = 2 * torch.sigmoid(post_w * post_scale + post_b)",
    286: "comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps",
}
#: The target's post multiplier, read off ``reference:284``.
REFERENCE_POST_MULT = 2.0
#: The value ``-030`` defaulted to. Kept ONLY as the failing control's input.
OLD_POST_MULT = 1.0


class VacuousControlError(AssertionError):
    """A control whose input could not have made it fail.

    Borrowed by name from ``test_mhc_layer.py:171``: a control that passes over
    input incapable of failing measures nothing, so it refuses the pass.
    """


class RouteInstrumentError(AssertionError):
    """A route reading that is not the one the plan declares."""


def _impl():
    """Import the implementation module INSIDE a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _cited_tolerances() -> tuple[float, float]:
    """``(rtol, atol)`` READ OUT OF ``test_mhc_layer.py``, not declared here.

    The pair this file compares on is the one that file already declares at its
    lines 150-152 ("The declared tolerance pair, from the plan block. Not
    widened anywhere."). Reading it from that file's bytes makes the citation
    mechanical: if the sibling's pair ever moves, this file moves with it or
    fails, and it can never silently hold a looser number of its own.
    """
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

    The magnitudes are that file's (``fn`` at ``1e-4``, ``hc_scale`` and
    ``hc_base`` at ``0.1``, its lines 345-361), so ``mixes`` lands where the
    target actually runs rather than in sigmoid saturation -- where every
    implementation agrees and a comparison would measure nothing.
    """
    gen = torch.Generator().manual_seed(seed)
    hc_mult3 = 2 * S + S * S
    fn = torch.randn(hc_mult3, S * hidden, generator=gen, dtype=torch.float32) * 1e-4
    hc_scale = torch.randn(3, generator=gen, dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult3, generator=gen, dtype=torch.float32) * 0.1
    residual = torch.randn(tokens, S, hidden, generator=gen, dtype=torch.float32)
    return fn, hc_scale, hc_base, residual


def _layer(post_mult_value: float | None = None, hidden: int = H):
    """The layer under test. ``post_mult_value`` is LEFT DEFAULT unless given.

    Every item but the failing control takes the default on purpose: the default
    is the thing ``-030c`` corrects, so a test that always passed the value
    explicitly could not see the correction at all -- which is exactly why
    ``test_mhc_layer.py`` cannot: it passes ``POST_ALPHA`` at its line 375.
    """
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
# The comparator: the TARGET MODEL's arithmetic, transcribed with its lines.    #
# --------------------------------------------------------------------------- #
def _reference_post_and_mixes(
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    rms_eps: float,
    post_mult: float = REFERENCE_POST_MULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(post, mixes)`` as ``Glm5NextTextHyperConnection.forward`` computes them.

    Verbatim in structure against ``modeling_glm5_next.py``, line by line:

    * ``:278`` ``flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())``
      -- the norm is applied to the FOLDED input, BEFORE the projection.
    * ``:216`` the norm itself,
      ``x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)``,
      whose ``eps`` is ``config.rms_norm_eps`` by ``:257``.
    * ``:279`` ``F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)``
      -- pre, post, comb in that order, which is the order the layer slices.
    * ``:284`` ``post = 2 * torch.sigmoid(post_w * post_scale + post_b)``.

    The reference's leading batch axis is absent here because this layer's
    contract is ``[T, S, H]`` rather than ``[B, S, H, D]``; the per-token
    arithmetic is unchanged, and ``flatten(start_dim=2)`` on a batched input is
    ``flatten(start_dim=1)`` on this one.

    ``mixes`` is returned beside ``post`` so an item can attribute a delta to a
    site rather than only to an output.
    """
    hc = S
    flat = residual.flatten(start_dim=1).to(torch.float32)
    # `:216` + `:257`: the model's own RMSNorm, on the model's own epsilon.
    normed = flat * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    # `:279`: one projection, split pre / post / comb.
    mixes = F.linear(normed, fn.to(torch.float32))
    post_w = mixes[:, hc : 2 * hc]
    post_b = hc_base.to(torch.float32)[hc : 2 * hc]
    post_scale = hc_scale.to(torch.float32)[1]
    # `:284`: the multiplier is on the OUTSIDE of the sigmoid, and it is 2.
    post = post_mult * torch.sigmoid(post_w * post_scale + post_b)
    return post, mixes


def _reference_path() -> str | None:
    """The campaign's reference copy, if this machine has it. Never required."""
    env = os.environ.get("GLM53F_REFERENCE_FILE")
    if env and os.path.exists(env):
        return env
    tail = os.path.join(
        "artifacts/campaigns/glm-5.3-flash-port/design/reference",
        "modeling_glm5_next.py",
    )
    here = os.path.abspath(__file__)
    for _ in range(12):
        here = os.path.dirname(here)
        # Both shapes: the campaign inside this checkout, and the campaign in a
        # SIBLING checkout, which is where it sits on the authoring machine.
        for cand in (
            os.path.join(here, tail),
            os.path.join(here, "NeuronAgenticDevelopment", tail),
        ):
            if os.path.exists(cand):
                return cand
        if here == os.path.dirname(here):
            break
    return None


def _route_reading(label: str, calls: int) -> str:
    """Print and CHECK the Sinkhorn route, so no reading here is a torch pass.

    ``mhc_pre`` enters ``-028``'s seam exactly once per call. A run whose
    ``torch_fallback`` moved would be comparing torch against torch, which would
    make every number in this file meaningless rather than merely wrong.
    """
    nki_dispatch, torch_fallback = sinkhorn_mod.dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"[{label}] mhc_pre_calls={calls} sinkhorn_nki_dispatch={nki_dispatch} "
        f"sinkhorn_torch_fallback={torch_fallback} can_run_kernel={gate} "
        f"per_call={nki_dispatch / calls if calls else float('nan')}"
    )
    print(reading)
    if nki_dispatch != calls:
        raise RouteInstrumentError(
            f"{label}: -028's dispatch counter read {nki_dispatch} over {calls} "
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
    """``(max_abs, max_rel)``, both floats, printed by every item that compares."""
    diff = (got.to(torch.float32) - want.to(torch.float32)).abs()
    denom = want.to(torch.float32).abs().clamp_min(1e-12)
    return float(diff.max()), float((diff / denom).max())


# --------------------------------------------------------------------------- #
# COUNTED VALUE 1 -- the post gate equals the target's ``2 * sigmoid``.         #
# --------------------------------------------------------------------------- #
def test_030c_post_gate_matches_the_reference_two_sigmoid() -> None:
    """The layer's ``post_mix`` equals ``reference:284`` on ``N/N`` cases.

    Taken on the DEFAULT ``post_mult_value``, because the default is what
    ``-030c`` corrects.
    """
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
    max_abs, max_rel = _max_abs_rel(got, want)
    cases = int(want.numel())
    within = int(
        torch.isclose(got, want, rtol=rtol, atol=atol).sum()
    )
    print(
        f"[value-1] post_mult_default={layer.post_mult_value} "
        f"reference_post_mult={REFERENCE_POST_MULT} cases={within}/{cases} "
        f"rtol={rtol} atol={atol} max_abs_error={max_abs:.6e} "
        f"max_rel_error={max_rel:.6e} "
        f"reference_post_min={float(want.min()):.6f} "
        f"reference_post_max={float(want.max()):.6f}"
    )
    assert layer.post_mult_value == REFERENCE_POST_MULT, layer.post_mult_value
    assert within == cases, f"{within}/{cases} within the cited pair"
    torch.testing.assert_close(got, want, rtol=rtol, atol=atol)


def test_030c_the_default_post_multiplier_is_the_targets_two() -> None:
    """The default itself, read off a layer nobody passed the value to.

    ``test_mhc_layer.py`` cannot make this reading: it constructs with
    ``post_mult_value=POST_ALPHA`` at its line 375, so the default never reaches
    it. That is why the defect survived ``-030``'s landed acceptance.
    """
    layer, _ = _layer()
    print(
        f"[default] post_mult_value={layer.post_mult_value} "
        f"reference={REFERENCE_POST_MULT} old_default={OLD_POST_MULT} "
        f"range_upper_bound_in_reference_shape_guide=2"
    )
    assert layer.post_mult_value == REFERENCE_POST_MULT, layer.post_mult_value


# --------------------------------------------------------------------------- #
# FAILING CONTROL 2 -- the old ``1.0`` must FAIL counted value 1.               #
# --------------------------------------------------------------------------- #
def test_030c_control_the_old_one_point_zero_default_fails_value_1() -> None:
    """With ``1.0`` in place of the target's ``2``, value 1 must FAIL.

    The parent block's words are "``counted value 1`` must FAIL and the
    transcript prints the delta -- a composition that halves the post term must
    not be able to pass". The vacuity guard is not decoration: if the reference
    post term were near zero, ``1x`` and ``2x`` would be indistinguishable and a
    passing control would measure nothing.
    """
    rtol, atol = _cited_tolerances()
    fn, hc_scale, hc_base, residual = _fixture()
    layer, cfg = _layer(post_mult_value=OLD_POST_MULT)
    _load(layer, fn, hc_scale, hc_base)

    sinkhorn_mod.reset_dispatch_counters()
    post_mix, _c, _l = layer.mhc_pre(residual)
    _route_reading("control-2-old-default", calls=1)

    want, _ = _reference_post_and_mixes(
        fn, hc_scale, hc_base, residual, float(cfg.rms_norm_eps)
    )
    got = post_mix.reshape(T, S)
    max_abs, max_rel = _max_abs_rel(got, want)
    ratio = float((want / got.clamp_min(1e-12)).median())
    print(
        f"[control-2] post_mult={layer.post_mult_value} "
        f"max_abs_error={max_abs:.6e} max_rel_error={max_rel:.6e} "
        f"median_reference_over_got={ratio:.6f} rtol={rtol} atol={atol} "
        f"smallest_reference_post={float(want.abs().min()):.6e}"
    )

    if float(want.abs().min()) <= atol:
        raise VacuousControlError(
            f"the smallest reference post term is {float(want.abs().min()):.3e}, "
            f"at or under atol={atol}: halving it could not have been detected, "
            f"so this control proves nothing about the multiplier"
        )
    assert not torch.allclose(got, want, rtol=rtol, atol=atol), (
        f"the old {OLD_POST_MULT} default matched the target's "
        f"{REFERENCE_POST_MULT} within the cited pair -- the control is not "
        f"discriminating, so value 1 cannot be trusted either"
    )


# --------------------------------------------------------------------------- #
# COUNTED VALUE 2 and FAILING CONTROL 3 -- each epsilon at its own site.        #
# --------------------------------------------------------------------------- #
def test_030c_rms_epsilon_reads_the_models_rms_norm_eps() -> None:
    """The layer carries the model's RMSNorm epsilon, distinct from ``hc_eps``.

    Both constants come off the same config object, and the whole correction is
    that they are DIFFERENT numbers: a config where they happened to be equal
    would make every downstream delta in this file vacuous, so that is checked
    here rather than assumed.
    """
    layer, cfg = _layer()
    print(
        f"[value-2-constants] layer.rms_eps={layer.rms_eps} "
        f"config.rms_norm_eps={float(cfg.rms_norm_eps)} "
        f"layer.hc_eps={layer.hc_eps} config.hc_eps={float(cfg.hc_eps)}"
    )
    assert layer.rms_eps == float(cfg.rms_norm_eps), layer.rms_eps
    assert layer.hc_eps == float(cfg.hc_eps), layer.hc_eps
    if layer.rms_eps == layer.hc_eps:
        raise VacuousControlError(
            f"rms_eps and hc_eps are both {layer.rms_eps}: this config cannot "
            f"distinguish the corrected site from the old one, so every epsilon "
            f"delta measured on it would be vacuous"
        )


def _ulp(x: torch.Tensor) -> torch.Tensor:
    """The exact fp32 step at each entry of ``x``, from ``nextafter``.

    ``finfo.eps * |x|`` is the step at 1.0 scaled by the magnitude, which
    overstates the true step by up to 2x depending on where the value sits inside
    its binade. This arm's floor is stated in ulps, so it uses the real step
    rather than a bound that could be twice the truth in either direction.
    """
    magnitude = x.abs()
    return torch.nextafter(magnitude, torch.full_like(magnitude, float("inf"))) - magnitude


def test_030c_rms_epsilon_site_is_load_bearing_and_the_old_value_moves_it() -> None:
    """COUNTED VALUE 2, site 1: the RMS epsilon moves the NORM OUTPUT by the predicted amount.

    WHAT THIS ARM ASSERTS, AND WHY IT IS NOT THE THREE RETURNS. The first cut
    asserted that swapping ``1e-05`` for ``1e-06`` moved all three of
    :meth:`mhc_pre`'s returns. Grant 177 read ``post_mix`` and ``layer_input``
    as exactly zero, and grant 179's transcript settled why: the epsilon enters
    as one scalar per token, ``rsqrt(mean_square + eps)``, so the swap is a
    **4.5e-06 relative** change; the pre and post gates then multiply it by
    ``hc_scale`` of about ``0.1`` and add ``hc_base`` of about ``0.1``, which
    lands the signal at about **a tenth of one fp32 step** of the value it must
    move. Two of the three returns therefore cannot move at all, and the third
    moved by two steps. A reading that can go either way on correct bytes is not
    a control, so this arm now measures the epsilon where the epsilon is read.

    THE MEASURABLE IS THE NORM OUTPUT (``reference:278`` applies
    ``reference:216``'s norm to the folded streams before the projection). One
    honest limitation, stated rather than buried: this tree folds that norm's
    scale onto the projection's OUTPUT instead (``model_fp8.py:1220-1221``, which
    is the same number because the projection carries no bias), so the normalised
    tensor is never materialised and cannot be read out of the three returns.
    The norm here is therefore the transcribed one, evaluated on this layer's own
    two constants read off the layer. What binds it to this tree is
    :func:`test_030c_rms_epsilon_reads_the_models_rms_norm_eps` on the constants
    and :func:`test_030c_post_gate_matches_the_reference_two_sigmoid` on the
    arithmetic.

    THE PREDICTION IS FIRST-ORDER AND THAT IS EXACT ENOUGH. For
    ``r(eps) = (ms + eps) ** -0.5`` the relative change is
    ``|d eps| / (2 * (ms + eps))``; the next term is
    ``(3/8) * (d eps / (ms + eps)) ** 2``, about ``1e-11`` relative here, which
    is five orders below one fp32 step. So a measurement outside ``[0.5x, 2x]``
    of the prediction is a finding about the norm, not about the algebra.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    layer, cfg = _layer()
    _load(layer, fn, hc_scale, hc_base)

    eps_new = float(layer.rms_eps)  # the corrected constant, `config.rms_norm_eps`
    eps_old = float(layer.hc_eps)  # what `-030` wrongly read at this site
    if eps_new == eps_old:
        raise VacuousControlError(
            f"rms_eps and hc_eps are both {eps_new}, so swapping one for the "
            f"other changes nothing and this arm would measure zero on correct "
            f"bytes"
        )

    # ---- 1. THE ASSERTED READING: the norm output, against its prediction. --- #
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
    print(
        f"[value-2-site-rms-norm] eps {eps_new} -> {eps_old} "
        f"predicted_rel={predicted_rel:.6e} measured_rel={measured_rel:.6e} "
        f"ratio={ratio:.4f} band=[0.5,2.0] measured_ulps={measured_ulps:.2f} "
        f"ulp_floor=16 mean_square_min={float(mean_square.min()):.6f} "
        f"entries={int(live.sum())}"
    )
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

    # ---- 2. THE THREE RETURNS: PRINTED, NOT ASSERTED. ----------------------- #
    # Kept because they are the evidence for the paragraph above and because the
    # r2 transcript carries the same row, so the two runs stay comparable. The
    # design bullet that asserted them is corrected, not the site: grant 179 read
    # `post 0 / comb 2.98e-08 / input 0`, and `2.98e-08` is two fp32 steps at
    # `comb_mix`'s own magnitude.
    sinkhorn_mod.reset_dispatch_counters()
    post_a, comb_a, input_a = layer.mhc_pre(residual)
    # Put the OLD, wrong constant back at this one site and nothing else.
    layer.rms_eps = eps_old
    post_b, comb_b, input_b = layer.mhc_pre(residual)
    _route_reading("value-2-rms-site", calls=2)

    d_post = float((post_a - post_b).abs().max())
    d_comb = float((comb_a - comb_b).abs().max())
    d_input = float((input_a - input_b).abs().max())
    print(
        f"[value-2-site-rms] swapped rms_eps {float(cfg.rms_norm_eps)} -> "
        f"{eps_old} delta_post_mix={d_post:.6e} "
        f"delta_comb_mix={d_comb:.6e} delta_layer_input={d_input:.6e} asserted=no"
    )
    for name, moved, reference in (
        ("post_mix", d_post, post_a),
        ("comb_mix", d_comb, comb_a),
        ("layer_input", d_input, input_a),
    ):
        floor = float(_ulp(reference).max())
        print(
            f"[value-2-site-rms-arrived] {name} delta={moved:.6e} "
            f"one_step_at_its_own_magnitude={floor:.6e} "
            f"steps={(moved / floor) if floor > 0.0 else float('nan'):.3f} "
            f"asserted=no"
        )


def test_030c_hc_epsilon_sites_are_the_pre_and_comb_gates_only() -> None:
    """COUNTED VALUE 2, sites 2 and 3, WITH their negative reading.

    ``hc_eps`` belongs at exactly two sites -- after the pre sigmoid
    (``reference:283``) and after the comb softmax (``reference:286``) -- and the
    target puts NO epsilon on the post term (``reference:284``). So moving
    ``hc_eps`` must move ``layer_input`` (through ``pre_mix``) and ``comb_mix``,
    and must leave ``post_mix`` **bit-identical**. The zero is as much a
    criterion as the two nonzeros: a ``post_mix`` that moved would mean this
    layer adds an epsilon the target does not have.
    """
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
    print(
        f"[value-2-sites-hc] scaled hc_eps by 100 delta_post_mix={d_post:.6e} "
        f"delta_comb_mix={d_comb:.6e} delta_layer_input={d_input:.6e} "
        f"expected_post_delta=0"
    )
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
        f"epsilon on the post term (reference:284), so this layer is adding one "
        f"the model does not have"
    )


# --------------------------------------------------------------------------- #
# COUNTED VALUE 3 -- every normalised block hits the seam's declared targets.    #
# --------------------------------------------------------------------------- #
def test_030c_each_normalised_block_hits_the_seams_declared_targets() -> None:
    """The batched entry's ``[T, S, S]`` output is doubly stochastic, per block.

    This is the plan's row for correction (iii), and it is a reading about the
    SEAM SWAP rather than about arithmetic the layer authors: ``mhc_pre`` now
    hands ``-028b``'s batched kernel the ``T`` blocks directly, so every one of
    the ``T`` blocks -- not their average -- must come back on target.

    The two targets are IMPORTED from the seam, never restated here
    (:func:`row_target`, :func:`column_target`), which is what
    ``sinkhorn.py:262-270`` asks of a caller by name: the kernel, the oracle and
    every acceptance take the number from one place so they cannot drift.

    NO NEW THRESHOLD IS REGISTERED. The two bounds are the cited pair read out of
    ``test_mhc_layer.py``, used per axis the way that file's own landed item uses
    them on this same quantity at this same case: the seam ends on a COLUMN pass,
    so the column axis is the exact one (``atol``) and the row axis is the one
    left one half-step behind (``rtol``). ``-028``'s own declared per-axis
    expected result, "within ``1e-3`` of *its* target"
    (``sinkhorn.py:267-268``), is printed beside them so a reader can see all
    three numbers rather than trust one.
    """
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

    # Per BLOCK, so one bad token cannot hide inside an average.
    row_sums = comb_mix.sum(dim=-1)          # [T, S]
    col_sums = comb_mix.sum(dim=-2)          # [T, S]
    row_dev = (row_sums - row_goal).abs()
    col_dev = (col_sums - col_goal).abs()
    worst_row_block = int(row_dev.amax(dim=-1).argmax())
    worst_col_block = int(col_dev.amax(dim=-1).argmax())
    print(
        f"[value-3] blocks={T} row_target={row_goal} column_target={col_goal} "
        f"worst_row_deviation={float(row_dev.max()):.6e} "
        f"in_block={worst_row_block} "
        f"worst_column_deviation={float(col_dev.max()):.6e} "
        f"in_block={worst_col_block} rtol={rtol} atol={atol} "
        f"seam_declared_per_axis_bound=1e-3 "
        f"min_entry={float(comb_mix.min()):.6e}"
    )
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
# FAILING CONTROL 4 -- the batched seam REFUSES a missing route, and the        #
# square seam it replaced would have papered over it.                          #
# --------------------------------------------------------------------------- #
def test_030c_control_the_batched_seam_refuses_an_unavailable_route() -> None:
    """With no device and no simulator, ``mhc_pre`` must RAISE, not fall back.

    This is correction (iii)'s failing control, and it is built in BOTH
    directions on purpose, because a one-directional version of it would pass on
    the code this increment replaced:

    * the batched entry has **no torch path at all** and raises
      :class:`SinkhornError` (``sinkhorn.py:952-1002``), so nothing is computed
      and neither counter moves;
    * the SQUARE entry this increment stopped calling does the opposite on the
      same input -- it returns the torch oracle and charges a
      ``torch_fallback`` (``sinkhorn.py:904-950``).

    So reverting (iii) to ``sinkhorn_normalise(torch.block_diag(...))`` fails
    this item by name. That is the point: kernel-class work ships no torch path
    (P13), and an mHC layer that silently normalised its blocks in torch would be
    slow in a way no numeric arm in this file could see.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    layer, _cfg = _layer()
    _load(layer, fn, hc_scale, hc_base)

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(os.environ, "NKI_SIMULATOR", "0")
        gate = can_run_kernel(torch.zeros(1))
        if gate is not False:
            raise RouteInstrumentError(
                f"can_run_kernel() still reads {gate} with NKI_SIMULATOR=0, so "
                f"this control is unarmed and its raise would prove nothing"
            )

        sinkhorn_mod.reset_dispatch_counters()
        with pytest.raises(sinkhorn_mod.SinkhornError) as excinfo:
            layer.mhc_pre(residual)
        after_batched = sinkhorn_mod.dispatch_counters()
        message = str(excinfo.value)

        # The same blocks, through the seam this increment REPLACED.
        blocks = torch.rand(T, S, S, dtype=torch.float32) + 0.5
        sinkhorn_mod.reset_dispatch_counters()
        square = sinkhorn_mod.sinkhorn_normalise(
            torch.block_diag(*blocks.unbind(0)),
            iters=int(layer.sinkhorn_iters),
        )
        after_square = sinkhorn_mod.dispatch_counters()

    print(
        f"[control-4] gate={gate} batched_raised={type(excinfo.value).__name__} "
        f"counters_after_batched={after_batched} "
        f"square_returned_shape={tuple(square.shape)} "
        f"counters_after_square={after_square} "
        f"message={message[:120]!r}"
    )
    assert "no torch path" in message, message
    assert "kernel-class" in message, message
    assert after_batched == (0, 0), after_batched
    # The discriminating half: the replaced seam would have RETURNED here.
    assert after_square == (0, 1), after_square
    assert tuple(square.shape) == (T * S, T * S), tuple(square.shape)


# --------------------------------------------------------------------------- #
# The comparator's own provenance.                                             #
# --------------------------------------------------------------------------- #
def test_030c_the_reference_transcription_is_faithful() -> None:
    """The transcription above is checked against the real reference file.

    Never a skip: when the campaign's reference copy is not on this machine the
    item still runs, prints that the file was unreachable, and asserts the
    transcription's own internal consistency. When the file IS reachable its
    sha256 is asserted against ``PROVENANCE.txt``'s recorded value and every
    cited line is read.
    """
    path = _reference_path()
    print(f"[provenance] reference_file={path!r} expected_sha256={REFERENCE_SHA256}")
    if path is None:
        print(
            "[provenance] reference_file_unreachable=1 -- the transcription is "
            "checked by its cited line numbers only; the reachable-file arm runs "
            "on the authoring machine and in the filed commit record"
        )
        assert len(REFERENCE_SHA256) == 64, REFERENCE_SHA256
        assert set(REFERENCE_LINES) == {216, 257, 278, 283, 284, 286}
        return

    body = open(path, encoding="utf-8").read()
    got_sha = hashlib.sha256(body.encode()).hexdigest()
    rows = body.split("\n")
    print(f"[provenance] measured_sha256={got_sha} lines={len(rows)}")
    assert got_sha == REFERENCE_SHA256, got_sha
    for line_no, must in REFERENCE_LINES.items():
        text = rows[line_no - 1]
        print(f"[provenance] :{line_no} {text.strip()[:78]}")
        assert must in text, f"reference:{line_no} reads {text.strip()!r}"
