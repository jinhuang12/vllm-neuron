# SPDX-License-Identifier: Apache-2.0
"""The mHC layer orchestration: Sinkhorn normalisation, the mix, and the combine.

Numeric agreement against a torch reference, the stream count the config declares,
and the dispatch counts the kernel route is supposed to make.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

import nki
import nki.simulator

from vllm_neuron.functional.mhc import hyper_connection as combine_mod
from vllm_neuron.functional.mhc import sinkhorn as sinkhorn_mod
from vllm_neuron.functional.mhc.hyper_connection import HyperConnectionError
from vllm_neuron.functional.mhc.sinkhorn import (
    MHC_STREAMS,
    MOVING_FMAX,
    PARTITION_MAX,
    SinkhornError,
    column_target,
    row_target,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# --------------------------------------------------------------------------- #
# the declared tiny case.                                                     #
# --------------------------------------------------------------------------- #
#: tokens. Re-grounded: the ceiling used to be the square
#: embedding's, ``T * S <= MOVING_FMAX``. Since the declared combine kernel no
#: upper extent bound is left on this path at all; see
#: :func:`test_mhc_layer_serves_above_the_combines_old_ceiling`. ``mhc_pre`` enters
#: the batched
#: entry, which carries no token bound, so the only ceiling left is the combine
#: kernel's ``T <= PARTITION_MAX`` = ``128``. ``8`` sits far inside it, so the
#: declared case is not also a boundary case. The boundary is a separate arm.
T = 8
#: Streams. ``MHC_STREAMS`` is the Sinkhorn seam's named constant for the target's
S = MHC_STREAMS
#: Hidden. Small: the layer's projection is ``[hc_mult3, S*H]`` and the
#: acceptance measures arithmetic, not size.
H = 64

RTOL = 1e-2
ATOL = 1e-5

#: The checkpoint's own mHC dials (``glm5_next/config.py``).
HC_SINKHORN_ITERS = 20
HC_EPS = 1e-06
#: ``hc_post_mult_value``. The base's own kernel test sets
#: ``hc_post_alpha = 1.0`` (``tests/kernels/test_mhc_kernels.py``); no fork
#: config field carries it, so the layer takes it as an argument and this is the
#: value the reference and the layer are both given.
POST_ALPHA = 1.0

_SINKHORN_FILE = os.path.realpath(sinkhorn_mod.__file__)
_COMBINE_FILE = os.path.realpath(combine_mod.__file__)


class RouteInstrumentError(AssertionError):
    """A route reading that is not what this file declares."""


class VacuousControlError(AssertionError):
    """A control whose input could not have made it fail. """


def _impl():
    """Import the implementation module inside a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


# --------------------------------------------------------------------------- #
# instrument 5, attributed across both seams.                                  #
# --------------------------------------------------------------------------- #
class _AttributedSimulatorCounter:
    """Counts ``nki.simulator.simulate_kernel`` entries, attributed per seam. """

    def __init__(self) -> None:
        self.total = 0
        self.sinkhorn = 0
        self.combine = 0
        self.elsewhere = 0
        self._real = None

    def __enter__(self) -> "_AttributedSimulatorCounter":
        self._real = nki.simulator.simulate_kernel
        real = self._real

        def counting(*args, **kwargs):
            self.total += 1
            frame = sys._getframe(1)
            hit = None
            while frame is not None:
                path = os.path.realpath(frame.f_code.co_filename)
                if path == _SINKHORN_FILE:
                    hit = "sinkhorn"
                    break
                if path == _COMBINE_FILE:
                    hit = "combine"
                    break
                frame = frame.f_back
            if hit == "sinkhorn":
                self.sinkhorn += 1
            elif hit == "combine":
                self.combine += 1
            else:
                self.elsewhere += 1
            return real(*args, **kwargs)

        nki.simulator.simulate_kernel = counting
        return self

    def __exit__(self, *exc_info) -> None:
        nki.simulator.simulate_kernel = self._real


def _reset_both() -> None:
    """Zero both seams' counters. The start of every declared case (section 4b)."""
    sinkhorn_mod.reset_dispatch_counters()
    combine_mod.reset_dispatch_counters()


def _read_both() -> tuple[tuple[int, int], tuple[int, int]]:
    """``((sink_nki, sink_fallback), (comb_nki, comb_fallback))`` since the reset."""
    return sinkhorn_mod.dispatch_counters(), combine_mod.dispatch_counters()


def _assert_route(
    sim: _AttributedSimulatorCounter, calls: int, label: str
) -> str:
    """Read every route instrument and return the reading."""
    (sink_nki, sink_fb), (comb_nki, comb_fb) = _read_both()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"[{label}] layer_calls={calls} "
        f"sinkhorn_nki_dispatch={sink_nki} sinkhorn_torch_fallback={sink_fb} "
        f"combine_nki_dispatch={comb_nki} combine_torch_fallback={comb_fb} "
        f"can_run_kernel={gate} simulate_kernel_total={sim.total} "
        f"simulate_kernel_through_028_seam={sim.sinkhorn} "
        f"simulate_kernel_through_029_seam={sim.combine} "
        f"simulate_kernel_elsewhere={sim.elsewhere} "
        f"per_layer_call_sinkhorn={sink_nki / calls if calls else float('nan')} "
        f"per_layer_call_combine={comb_nki / calls if calls else float('nan')}"
    )

    if sink_nki != calls:
        raise RouteInstrumentError(
            f"{label}: the Sinkhorn seam's dispatch counter read {sink_nki} over {calls} "
            f"layer call(s); this file declares exactly ONE per layer call, so "
            f"{calls} was expected. {reading}"
        )
    if comb_nki != calls:
        raise RouteInstrumentError(
            f"{label}: the combine seam's dispatch counter read {comb_nki} over {calls} "
            f"layer call(s); this file declares exactly ONE per layer call, so "
            f"{calls} was expected. {reading}"
        )
    if sink_fb != 0 or comb_fb != 0:
        raise RouteInstrumentError(
            f"{label}: torch-fallback counters read sinkhorn={sink_fb} "
            f"combine={comb_fb}, declared exactly 0 on both -- a fallback pass "
            f"would compare torch against torch. {reading}"
        )
    if gate is not True:
        raise RouteInstrumentError(
            f"{label}: can_run_kernel() read {gate}, declared True. {reading}"
        )
    if sim.sinkhorn != calls or sim.combine != calls:
        raise RouteInstrumentError(
            f"{label}: the vendor's simulator entry point attributed "
            f"{sim.sinkhorn} entries to the Sinkhorn seam and {sim.combine} to "
            f"the combine's; {calls} each was expected. A total that is right while "
            f"the split is wrong means one kernel ran twice. {reading}"
        )
    if sim.elsewhere != 0:
        raise RouteInstrumentError(
            f"{label}: {sim.elsewhere} simulator entries came from neither "
            f"seam. {reading}"
        )
    return reading


# --------------------------------------------------------------------------- #
# the fixture, and the sub-block the layer wraps.                              #
# --------------------------------------------------------------------------- #
def _sublayer(hidden_states: torch.Tensor) -> torch.Tensor:
    """The wrapped sub-block, stood in for by a deterministic elementwise map. """
    return torch.tanh(hidden_states) * 1.5


def _fixture(seed: int = 30, tokens: int = T, hidden: int = H):
    """``(fn, hc_scale, hc_base, residual)`` in fp32, on the base's own scales. """
    gen = torch.Generator().manual_seed(seed)
    hc_mult3 = 2 * S + S * S
    fn = torch.randn(hc_mult3, S * hidden, generator=gen, dtype=torch.float32) * 1e-4
    hc_scale = torch.randn(3, generator=gen, dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult3, generator=gen, dtype=torch.float32) * 0.1
    residual = torch.randn(
        tokens, S, hidden, generator=gen, dtype=torch.float32
    )
    return fn, hc_scale, hc_base, residual


def _layer(hidden: int = H, iters: int = HC_SINKHORN_ITERS):
    """Build the layer under test, sized from a real ``Glm5NextTextConfig``."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    impl = _impl()
    text_config = Glm5NextTextConfig(
        hidden_size=hidden,
        hc_mult=S,
        hc_sinkhorn_iters=iters,
        hc_eps=HC_EPS,
    )
    return impl.Glm5NextHyperConnection(text_config, post_mult_value=POST_ALPHA)


def _load(layer, fn, hc_scale, hc_base) -> None:
    """Set the layer's parameters. """
    with torch.no_grad():
        layer.fn.copy_(fn)
        layer.hc_scale.copy_(hc_scale)
        layer.hc_base.copy_(hc_base)


# --------------------------------------------------------------------------- #
# the comparator: the pinned base's mHC, in both of its spellings.              #
# --------------------------------------------------------------------------- #
def _sinkhorn_normalize_tilelang(
    x: torch.Tensor, repeat: int, eps: float
) -> torch.Tensor:
    """``sinkhorn_normalize_ref``, ``tests/kernels/test_mhc_kernels.py``. """
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def _reference_pre_torch(fn, hc_scale, hc_base, residual, eps, alpha, repeat):
    """``mhc_pre_torch``, ``vllm/model_executor/kernels/mhc/torch.py``. """
    tokens, hc_mult, hidden = (int(v) for v in residual.shape)
    x = residual.reshape(tokens, hc_mult * hidden).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden) + eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
        + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = torch.sigmoid(post_logits) * alpha

    comb_logits = mixes[:, 2 * hc_mult :].reshape(
        tokens, hc_mult, hc_mult
    ) * hc_scale[2] + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + eps)

    layer_input = torch.sum(
        pre_mix.unsqueeze(-1) * residual.to(torch.float32), dim=1
    )
    return post_mix.reshape(tokens, hc_mult, 1), comb_mix, layer_input


def _reference_pre_tilelang(fn, hc_scale, hc_base, residual, eps, alpha, repeat):
    """``mhc_pre_ref``, ``tests/kernels/test_mhc_kernels.py``. """
    hc_mult = int(residual.shape[-2])
    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = (
        residual_flat @ fn.T
        * (sqrsum.unsqueeze(-1) / fn.shape[-1] + eps).rsqrt()
    )
    scale = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * scale + hc_base

    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + eps
    post_mix = (mixes[:, hc_mult : 2 * hc_mult].sigmoid() * alpha).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
    res_mix = _sinkhorn_normalize_tilelang(res_mix, repeat=repeat, eps=eps)
    layer_input = (residual.float() * pre_mix).sum(-2)
    return post_mix, res_mix, layer_input


def _reference_post_torch(x, residual, post_layer_mix, comb_res_mix):
    """``mhc_post_torch``'s ``einsum`` spelling, kept fp32."""
    mixed = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return mixed + post


def _reference_post_tilelang(x, residual, post_layer_mix, comb_res_mix):
    """``mhc_post_ref``'s ``bmm(comb.mT, residual)`` spelling, kept fp32. """
    term2 = torch.bmm(comb_res_mix.mT.to(torch.float32), residual.float())
    return x.float().unsqueeze(-2) * post_layer_mix.to(torch.float32) + term2


def _reference_layer(fn, hc_scale, hc_base, residual, spelling: str):
    """One full reference layer call: pre, then the sub-block, then post. """
    if spelling == "torch":
        pre_fn, post_fn = _reference_pre_torch, _reference_post_torch
    elif spelling == "tilelang":
        pre_fn, post_fn = _reference_pre_tilelang, _reference_post_tilelang
    else:  # pragma: no cover - guards a typo in this file, not a code path
        raise ValueError(f"unknown spelling {spelling!r}")
    post_mix, comb_mix, layer_input = pre_fn(
        fn, hc_scale, hc_base, residual, HC_EPS, POST_ALPHA, HC_SINKHORN_ITERS
    )
    x = _sublayer(layer_input)
    return post_fn(x, residual, post_mix, comb_mix), post_mix, comb_mix, layer_input, x


def _errors(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float]:
    """``(max_abs_error, max_rel_error)`` as plain floats."""
    got32 = got.to(torch.float32)
    want32 = want.to(torch.float32)
    diff = (got32 - want32).abs()
    max_abs = float(diff.max())
    denom = want32.abs()
    max_rel = float((diff / torch.where(denom > 0, denom, torch.ones_like(denom))).max())
    return max_abs, max_rel


# --------------------------------------------------------------------------- #
# the declared acceptance -- 1/1 tiny case, both checks in one test.         #
# --------------------------------------------------------------------------- #
def test_mhc_layer_output_matches_the_torch_reference_layer_tiny_case() -> None:
    """Both halves: the numbers and the two per-layer-call counters."""
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    want, _, _want_comb, _want_layer_input, _ = _reference_layer(
        fn, hc_scale, hc_base, residual, "torch"
    )

    _reset_both()
    with _AttributedSimulatorCounter() as sim:
        got = layer.forward(residual, _sublayer)
    reading = _assert_route(sim, calls=1, label="acceptance-tiny")

    assert tuple(got.shape) == (T, S, H), tuple(got.shape)
    assert got.dtype is torch.float32, got.dtype

    # Non-vacuity: an expected tensor sitting at zero would make any tolerance
    # pass and prove nothing.
    if float(want.abs().max()) < 1e-3:
        raise VacuousControlError(
            f"the expected tensor's largest magnitude is "
            f"{float(want.abs().max()):.6e}; the comparison would be vacuous"
        )

    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)
    assert "sinkhorn_nki_dispatch=1" in reading
    assert "combine_nki_dispatch=1" in reading


def test_mhc_layer_the_two_upstream_spellings_agree() -> None:
    """The comparator does not rest on one reading of one upstream file. """
    fn, hc_scale, hc_base, residual = _fixture()
    a, _, a_comb, a_input, _ = _reference_layer(
        fn, hc_scale, hc_base, residual, "torch"
    )
    b, _, b_comb, b_input, _ = _reference_layer(
        fn, hc_scale, hc_base, residual, "tilelang"
    )
    _comb_abs, _ = _errors(a_comb, b_comb)
    _in_abs, _ = _errors(a_input, b_input)
    torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(a_comb, b_comb, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(a_input, b_input, rtol=RTOL, atol=ATOL)


def test_mhc_layer_counters_read_one_per_layer_call_across_two_calls() -> None:
    """Once per layer call is a rate, so it is read at two rates."""
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    _reset_both()
    with _AttributedSimulatorCounter() as sim:
        layer.forward(residual, _sublayer)
        mid = _read_both()
        layer.forward(residual, _sublayer)
    assert mid == ((1, 0), (1, 0)), mid
    _assert_route(sim, calls=2, label="per-call-two")


# --------------------------------------------------------------------------- #
# the composition controls -- per-token independence, re-grounded.              #
# --------------------------------------------------------------------------- #
def test_mhc_layer_tokens_are_independent_of_each_other() -> None:
    """Perturb one token; every other token's output must be bit-identical. """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    base_out = layer.forward(residual, _sublayer)

    perturbed = residual.clone()
    perturbed[0] = perturbed[0] + 0.75
    moved_out = layer.forward(perturbed, _sublayer)

    token0_delta = float((moved_out[0] - base_out[0]).abs().max())
    others_delta = float((moved_out[1:] - base_out[1:]).abs().max())
    if token0_delta == 0.0:
        raise VacuousControlError(
            "perturbing token 0 changed token 0's output by exactly 0.0, so "
            "this control could not have detected cross-token leakage"
        )
    assert others_delta == 0.0, (
        f"perturbing token 0 moved other tokens by {others_delta:.6e}; the "
        f"per-token composition is leaking across tokens, which is exactly "
        f"what a flat [T*S, S] reshape into the Sinkhorn seam would do"
    )


def test_mhc_layer_off_block_entries_stay_zero() -> None:
    """The embedding's zeros stay zero, and each block hits the base's targets. """
    from vllm_neuron.functional.mhc.sinkhorn import sinkhorn_normalise

    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    # Reproduce the layer's own embedding input through its own pre block, then
    # normalise the same embedding again so the raw [T*S, T*S] result is visible
    # here. The layer's helper is used for the extraction so the extraction
    # itself is the one under test.
    _reset_both()
    _post_mix, comb_mix, _ = layer.mhc_pre(residual)
    assert sinkhorn_mod.dispatch_counters() == (1, 0), (
        sinkhorn_mod.dispatch_counters()
    )

    comb_start = _embedding_input(layer, residual)
    raw = sinkhorn_normalise(
        torch.block_diag(*comb_start.unbind(0)), iters=HC_SINKHORN_ITERS
    )
    side = T * S
    mask = torch.ones(side, side, dtype=torch.bool)
    for t in range(T):
        mask[t * S : (t + 1) * S, t * S : (t + 1) * S] = False
    off_block_max = float(raw[mask].abs().max())

    row_dev = float((comb_mix.sum(dim=-1) - row_target()).abs().max())
    col_dev = float((comb_mix.sum(dim=-2) - column_target(S, S)).abs().max())
    assert off_block_max == 0.0, off_block_max
    assert tuple(comb_mix.shape) == (T, S, S)
    # The base's own Sinkhorn leaves the last-applied axis exact and the other
    # near-exact, and the seam ends on a column pass too, so the column reading is
    # the tight one. Both are read; neither number is a declared criterion.
    assert col_dev < 1e-5, col_dev
    assert row_dev < 1e-2, row_dev


def _embedding_input(layer, residual: torch.Tensor) -> torch.Tensor:
    """The ``[T, S, S]`` the layer hands to ``block_diag``, recomputed here. """
    tokens, streams, hidden = (int(v) for v in residual.shape)
    flat = residual.reshape(tokens, streams * hidden).to(torch.float32)
    mixes = flat @ layer.fn.to(torch.float32).t()
    sqrsum = flat.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / float(streams * hidden) + layer.hc_eps)
    scale = layer.hc_scale.to(torch.float32)
    base = layer.hc_base.to(torch.float32)
    comb_logits = mixes[:, 2 * streams :].reshape(
        tokens, streams, streams
    ) * scale[2] + base[2 * streams :].reshape(1, streams, streams)
    return torch.softmax(comb_logits, dim=-1) + layer.hc_eps


def test_mhc_layer_the_mixing_convention_is_not_transposed() -> None:
    """The layer against a transposed-mix reference must be loudly wrong. """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    got = layer.forward(residual, _sublayer)
    _, post_mix, comb_mix, _, x = _reference_layer(
        fn, hc_scale, hc_base, residual, "torch"
    )
    asymmetry = float((comb_mix - comb_mix.mT).abs().max())
    # The same reference, with only the mix transposed. Nothing else moves, so a
    # difference can only be the i/j convention.
    wrong = _reference_post_torch(
        x, residual, post_mix, comb_mix.mT.contiguous()
    )
    if asymmetry < 1e-3:
        raise VacuousControlError(
            f"the fixture's mixing matrix is symmetric to {asymmetry:.6e}, so a "
            f"transposed reading would be invisible and this control measures "
            f"nothing"
        )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, wrong, rtol=RTOL, atol=ATOL)


def test_mhc_layer_numeric_comparison_is_armed() -> None:
    """Perturb one residual element; the comparison must move far outside. """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    want, _, _, _, _ = _reference_layer(fn, hc_scale, hc_base, residual, "torch")
    perturbed = residual.clone()
    perturbed[0, 0, 0] = perturbed[0, 0, 0] + 4.0
    got = layer.forward(perturbed, _sublayer)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_mhc_layer_folds_the_streams_as_the_base_does() -> None:
    """``layer_input`` is the pre-mix fold, and it is checked on its own. """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    _reset_both()
    _, _, layer_input = layer.mhc_pre(residual)
    _, _, _, want_input, _ = _reference_layer(
        fn, hc_scale, hc_base, residual, "torch"
    )
    assert tuple(layer_input.shape) == (T, H)
    torch.testing.assert_close(layer_input, want_input, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# route controls -- so every zero above is a measurement.                       #
# --------------------------------------------------------------------------- #


def test_mhc_layer_f1_numeric_arm_alone_cannot_discriminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-grounded. """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)
    want, _, _, _, _ = _reference_layer(fn, hc_scale, hc_base, residual, "torch")

    # The kernel route, kept for the comparison the fallback must be indistinguishable from.
    _reset_both()
    post_mix, comb_mix, layer_input = layer.mhc_pre(residual)
    x = _sublayer(layer_input)
    kernel_out = layer.mhc_post(x, residual, post_mix, comb_mix)
    on_route = _read_both()

    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")

    _reset_both()
    with pytest.raises(SinkhornError) as excinfo:
        layer.forward(residual, _sublayer)
    refused = _read_both()

    _reset_both()
    fallback_out = layer.mhc_post(x, residual, post_mix, comb_mix)
    fell_back = _read_both()

    assert on_route == ((1, 0), (1, 0)), on_route
    assert refused == ((0, 0), (0, 0)), refused
    assert fell_back == ((0, 0), (0, 1)), fell_back
    # The point of the arm: both comparisons pass with no combine kernel dispatched.
    torch.testing.assert_close(fallback_out, kernel_out, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(fallback_out, want, rtol=RTOL, atol=ATOL)


def test_mhc_layer_a_pure_torch_layer_reads_zero_on_both_seams() -> None:
    """The declared falsifier, measured: "a pure-torch layer produces 0 on both". """
    fn, hc_scale, hc_base, residual = _fixture()
    _reset_both()
    with _AttributedSimulatorCounter() as sim:
        out, _, _, _, _ = _reference_layer(
            fn, hc_scale, hc_base, residual, "torch"
        )
    readings = _read_both()
    assert readings == ((0, 0), (0, 0)), readings
    assert sim.total == 0, sim.total
    assert tuple(out.shape) == (T, S, H)


def test_mhc_layer_the_two_seam_counters_are_independent() -> None:
    """Resetting one seam's counter must not touch the other's. """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    _reset_both()
    layer.forward(residual, _sublayer)
    assert _read_both() == ((1, 0), (1, 0)), _read_both()

    sinkhorn_mod.reset_dispatch_counters()
    after = _read_both()
    assert after == ((0, 0), (1, 0)), after

    combine_mod.reset_dispatch_counters()
    after2 = _read_both()
    assert after2 == ((0, 0), (0, 0)), after2


# --------------------------------------------------------------------------- #
# the token extents -- what the layer serves, and where the refusal now sits.   #
# the Sinkhorn's row axis is tiled inside the kernel, so the                     #
# refusal moved from the row axis to the column axis and these cases were        #
# re-grounded onto what the code does now rather than deleted.                   #
# --------------------------------------------------------------------------- #
def test_mhc_layer_runs_at_the_old_row_axis_ceiling() -> None:
    """``T = PARTITION_MAX // S`` runs, and both counters still read one each. """
    ceiling = PARTITION_MAX // S
    fn, hc_scale, hc_base, residual = _fixture(tokens=ceiling)
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    _reset_both()
    with _AttributedSimulatorCounter() as sim:
        got = layer.forward(residual, _sublayer)
    _assert_route(sim, calls=1, label=f"ceiling-T{ceiling}")
    want, _, _, _, _ = _reference_layer(fn, hc_scale, hc_base, residual, "torch")
    assert tuple(got.shape) == (ceiling, S, H)
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("tokens", [PARTITION_MAX // S + 1, 2 * (PARTITION_MAX // S)])
def test_mhc_layer_serves_above_the_old_ceiling(tokens: int) -> None:
    """Token counts that used to be refused are served, and match the oracle. """
    fn, hc_scale, hc_base, residual = _fixture(tokens=tokens)
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    _reset_both()
    with _AttributedSimulatorCounter() as sim:
        got = layer.forward(residual, _sublayer)
    _assert_route(sim, calls=1, label=f"above-old-ceiling-T{tokens}")

    want, _, _, _, _ = _reference_layer(fn, hc_scale, hc_base, residual, "torch")
    assert tuple(got.shape) == (tokens, S, H)
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("tokens", [PARTITION_MAX + 1, 200])
def test_mhc_layer_serves_above_the_combines_old_ceiling(tokens: int) -> None:
    """129 and 200 tokens are served and match the oracle. """
    fn, hc_scale, hc_base, residual = _fixture(tokens=tokens)
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    _reset_both()
    with _AttributedSimulatorCounter() as sim:
        got = layer.forward(residual, _sublayer)
    _assert_route(sim, calls=1, label=f"above-combine-ceiling-T{tokens}")
    _read_both()

    want, _, _, _, _ = _reference_layer(fn, hc_scale, hc_base, residual, "torch")
    assert tuple(got.shape) == (tokens, S, H)
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# layer-level refusals -- the agreements neither seam can see.                  #
# --------------------------------------------------------------------------- #
def test_mhc_layer_refuses_a_stream_count_that_contradicts_its_config() -> None:
    """A wrong ``S`` would mis-slice ``mixes`` rather than fail, so it is refused. """
    impl = _impl()
    layer = _layer()
    residual = torch.randn(T, S + 1, H, dtype=torch.float32)
    with pytest.raises(impl.Glm5NextHyperConnectionError) as excinfo:
        layer.forward(residual, _sublayer)
    assert f"S={S + 1}" in str(excinfo.value)
    assert f"hc_mult={S}" in str(excinfo.value)


def test_mhc_layer_refuses_a_hidden_extent_that_contradicts_its_config() -> None:
    """A wrong ``H`` would make ``fn`` unmultipliable; it is named instead."""
    impl = _impl()
    layer = _layer()
    residual = torch.randn(T, S, H + 8, dtype=torch.float32)
    with pytest.raises(impl.Glm5NextHyperConnectionError) as excinfo:
        layer.forward(residual, _sublayer)
    assert f"H={H + 8}" in str(excinfo.value)


def test_mhc_layer_refuses_a_non_3d_residual() -> None:
    """``[T, S, H]`` is the rank the whole operation is defined on."""
    impl = _impl()
    layer = _layer()
    with pytest.raises(impl.Glm5NextHyperConnectionError) as excinfo:
        layer.forward(torch.randn(T, H, dtype=torch.float32), _sublayer)
    assert "3-D" in str(excinfo.value)


def test_mhc_layer_refuses_a_bad_sublayer() -> None:
    """A non-callable sub-block, and one that changes the shape, both named. """
    impl = _impl()
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    with pytest.raises(impl.Glm5NextHyperConnectionError) as not_callable:
        layer.forward(residual, object())
    assert "callable" in str(not_callable.value)

    with pytest.raises(impl.Glm5NextHyperConnectionError) as wrong_shape:
        layer.forward(residual, lambda t: t[:, :4])
    assert "expected the [T, H] shape" in str(wrong_shape.value)


def test_mhc_layer_refuses_a_non_positive_iteration_count() -> None:
    """The configuration refusal, at construction rather than at the seam."""
    impl = _impl()
    with pytest.raises(impl.Glm5NextHyperConnectionError) as excinfo:
        _layer(iters=0)
    assert "must be positive" in str(excinfo.value)


def test_mhc_layer_neuron_config_overrides_win() -> None:
    """``NeuronConfig``'s two mHC overrides take precedence when set. """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
    from vllm_neuron.model.neuron_config import NeuronConfig

    impl = _impl()
    text_config = Glm5NextTextConfig(
        hidden_size=H, hc_mult=S, hc_sinkhorn_iters=20, hc_eps=1e-06
    )
    neuron_config = NeuronConfig(mhc_sinkhorn_iters=7, mhc_eps=1e-04)
    layer = impl.Glm5NextHyperConnection(text_config, neuron_config)
    assert layer.sinkhorn_iters == 7
    assert layer.hc_eps == 1e-04

    plain = impl.Glm5NextHyperConnection(text_config, None)
    assert (plain.sinkhorn_iters, plain.hc_eps) == (20, 1e-06)


def test_mhc_layer_stream_count_matches_the_target_hc_mult() -> None:
    """``S`` is the checkpoint's ``hc_mult``, imported from the seam, not chosen. """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    default_hc_mult = int(Glm5NextTextConfig().hc_mult)
    assert S == MHC_STREAMS == default_hc_mult == 4
    assert int(Glm5NextTextConfig().hc_sinkhorn_iters) == HC_SINKHORN_ITERS
    assert float(Glm5NextTextConfig().hc_eps) == HC_EPS


def test_mhc_layer_parameters_are_real_and_sized_from_the_config() -> None:
    """The three parameters exist, are allocated, and carry the base's names. """
    layer = _layer()
    names = {name for name, _ in layer.named_parameters()}
    assert names == {"fn", "hc_scale", "hc_base"}
    assert layer.hc_mult3 == 2 * S + S * S
    assert tuple(layer.fn.shape) == (layer.hc_mult3, S * H)
    assert tuple(layer.hc_scale.shape) == (3,)
    assert tuple(layer.hc_base.shape) == (layer.hc_mult3,)


def test_mhc_layer_combine_seam_refusals_reach_the_caller() -> None:
    """The combine seam's own refusal is not swallowed by this layer either. """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)
    post_mix, comb_mix, _ = layer.mhc_pre(residual)
    with pytest.raises(HyperConnectionError) as excinfo:
        layer.mhc_post(
            torch.randn(T, H + 8, dtype=torch.float32), residual, post_mix, comb_mix
        )
    assert "expected [T, H]" in str(excinfo.value)
