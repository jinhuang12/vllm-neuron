# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for ``inc-glm53f-030`` -- WP8, the mHC layer orchestration.

The declared acceptance (increment plan revision 35,
``16c1ed71f5872b3af81ab59bdba1dd2ef999cd5daa6ecdba98d2b99fe28b54fc``, L958),
verbatim on both of the things it declares:

    "the mHC layer's output matches a torch reference layer with
    ``assert_close(rtol=1e-2, atol=1e-5)`` in 1/1 tiny case, **and** the seam
    counters show the Sinkhorn and combine kernels were each entered **exactly
    once per layer call** (silent-fallback guard, as ``-027``)."

Tier N, so the command is the harness this block inherits from ``-025``
(plan L913, with revision 33's ``-p no:cacheprovider``)::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    python -m pytest test/vllm_neuron/model/glm5_next/test_mhc_layer.py \\
      --timeout 60 -v -s -p no:cacheprovider

THE ROUTE PREDICATE (D13 form R-2, plan L959). This increment authors NO seam:
it is the layer wiring that selects and feeds ``-028``'s Sinkhorn kernel and
``-029``'s combine kernel. So the predicate is the R-2 form -- simulator
dispatches counted on the F1 chain (``wrap_nki`` -> ``NKIHOPCaller`` -> HOP ->
``DispatchKey.CPU`` -> ``nki.simulator.simulate_kernel``) through **two** seams
this increment does not own. Five instruments per declared case, each reported
as a number:

1. ``-028``'s seam dispatch counter -- ``nki_dispatch == 1`` per layer call;
2. ``-029``'s seam dispatch counter -- ``nki_dispatch == 1`` per layer call;
3. both modules' torch-fallback counters -- exactly ``0``;
4. ``can_run_kernel()`` -- ``True``;
5. real ``nki.simulator.simulate_kernel`` entries -- ``2`` per layer call, and
   **ATTRIBUTED PER SEAM**, because a total of two proves nothing on its own: it
   is satisfied by two entries into one kernel. The Python frame chain at each
   entry is walked for each seam's own file, so the reading distinguishes "two
   kernels ran" from "one kernel ran twice". This is ``-027``'s attribution
   instrument (``test_moe_path.py:271-311``) generalised from one seam to two,
   which is what having two counted seams requires.

Instruments 1-4's counted values ARE the acceptance bullet's counter clause;
they are cited here, not restated as extra criteria. Instrument 5 is the
vendor's own entry point, so a bug in this campaign's bookkeeping cannot fake
it.

F1: WHY THE NUMERIC ARM ALONE IS A FALSE GREEN WAITING TO HAPPEN. Both seams
fall back to a torch oracle that computes the same function when
``can_run_kernel()`` is false, so the numeric comparison below passes on the
fallback path too -- :func:`test_mhc_layer_f1_numeric_arm_alone_cannot_discriminate`
MEASURES that it does. The plan says it plainly: "A pure-torch layer produces
``0`` on both and therefore cannot pass."

WHAT THE COMPARATOR IS, AND WHY THERE ARE TWO OF THEM. A torch reference layer
authored in this file from the pinned base's own mHC, read at tag ``v0.24.0``
because the campaign's target base is the 0.24 line. The base states the
operation in **two independent spellings**, and both are transcribed here:

* :func:`_reference_pre_torch` / :func:`_reference_post_torch` --
  ``mhc_pre_torch`` / ``mhc_post_torch``, ``vllm/model_executor/kernels/mhc/torch.py``
  (the plain-torch backend; the post half is the ``einsum`` spelling);
* :func:`_reference_pre_tilelang` / :func:`_reference_post_tilelang` --
  ``mhc_pre_ref`` / ``mhc_post_ref``, ``tests/kernels/test_mhc_kernels.py``
  (the TileLang-repo reference; the post half is the ``bmm(comb.mT, res)``
  spelling).

:func:`test_mhc_layer_the_two_upstream_spellings_agree` requires them to agree,
so the comparator does not rest on one reading of one file. This is the
discipline ``-029`` set (``evidence-029.md`` section 2.1).

THE COMPOSITION THE LAYER HAD TO AUTHOR, AND THE CONTROL THAT GUARDS IT.
``-028``'s seam normalises ONE ``[M, N]`` matrix; the target needs ``T``
independent ``[S, S]`` ones; the predicate above declares ONE Sinkhorn dispatch
per layer call. The layer reconciles those with a **block-diagonal embedding**
(``model_fp8.py``, the ``Glm5NextHyperConnection`` docstring). The embedding is
not asserted, it is guarded from two sides:

* :func:`test_mhc_layer_tokens_are_independent_of_each_other` perturbs ONE
  token and requires every OTHER token's output to be **bit-identical**. The
  rejected flat ``[T*S, S]`` reshape lets ``-028``'s column pass sum across
  tokens, so it fails this arm loudly -- which is why the arm exists.
* :func:`test_mhc_layer_off_block_entries_stay_zero` reads the off-block maximum
  of the kernel's own output and requires exactly ``0.0``, and reads each
  token's row and column sums against the base's own targets.

TWO DIVERGENCES FROM THE BASE THAT THIS INCREMENT CANNOT REMOVE, and both are
inside ``-028``'s LANDED kernel, so both are measured rather than repaired: the
base adds ``hc_sinkhorn_eps`` to every Sinkhorn denominator while ``-028`` adds
an inert ``1e-30``, and the two iteration schedules differ by a leading
half-step. Sinkhorn-Knopp has one fixed point, so the gap is small at the
target's ``20`` iterations -- and how small is a number the acceptance reports
rather than a claim it makes.

D1.2: the item count reported for this file is pytest ITEMS from a dedicated
``--collect-only -q`` run, not a count declared here.

D15 / test-layout rule 3: no fixture in this file sets ``NKI_SIMULATOR`` or
``NEURON_PLATFORM_TARGET_OVERRIDE`` -- both resolve at import time and belong in
the process invocation. The two route CONTROLS below flip ``NKI_SIMULATOR``
inside a single test body and restore it, which is a measurement of the gate,
not a fixture.

THE MODELING MODULE IS IMPORTED INSIDE TEST BODIES, never at module scope.
``test_factory.py``'s C03 asserts ``model_fp8`` is absent from ``sys.modules``,
pytest imports every collected module before running any test, and this file
sorts after ``test_factory.py`` -- so a module-level import here would break a
landed assertion. ``-023`` and ``-013`` both record this; :func:`_impl` is the
form they use.
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
    PARTITION_MAX,
    SinkhornError,
    column_target,
    row_target,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# --------------------------------------------------------------------------- #
# The declared tiny case.                                                     #
# --------------------------------------------------------------------------- #
#: Tokens. Chosen inside the ceiling the block-diagonal embedding implies --
#: ``T * S <= PARTITION_MAX`` -- with room to spare, so the declared case is not
#: also a boundary case. The boundary itself is a separate arm.
T = 8
#: Streams. ``MHC_STREAMS`` is ``-028``'s named constant for the target's
#: ``hc_mult 4``; it is imported rather than restated, which is what
#: ``sinkhorn.py:136-139`` asks of this increment by name.
S = MHC_STREAMS
#: Hidden. Small: the layer's projection is ``[hc_mult3, S*H]`` and the
#: acceptance measures arithmetic, not size.
H = 64

#: The declared tolerance pair, from the plan block. Not widened anywhere.
RTOL = 1e-2
ATOL = 1e-5

#: The checkpoint's own mHC dials (``config.py:151-154``).
HC_SINKHORN_ITERS = 20
HC_EPS = 1e-06
#: ``hc_post_mult_value``. The base's own kernel test sets
#: ``hc_post_alpha = 1.0`` (``tests/kernels/test_mhc_kernels.py:126``); no fork
#: config field carries it, so the layer takes it as an argument and this is the
#: value the reference and the layer are both given.
POST_ALPHA = 1.0

_SINKHORN_FILE = os.path.realpath(sinkhorn_mod.__file__)
_COMBINE_FILE = os.path.realpath(combine_mod.__file__)


class RouteInstrumentError(AssertionError):
    """A route reading that is not what the plan declares."""


class VacuousControlError(AssertionError):
    """A control whose input could not have made it fail.

    A zero over vacuous input measures nothing, so the control refuses to
    report a pass it did not earn.
    """


def _impl():
    """Import the implementation module INSIDE a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


# --------------------------------------------------------------------------- #
# Instrument 5, attributed across BOTH seams.                                  #
# --------------------------------------------------------------------------- #
class _AttributedSimulatorCounter:
    """Counts ``nki.simulator.simulate_kernel`` entries, attributed per seam.

    ``total`` is the raw count at the vendor's own entry point, independent of
    every counter this campaign wrote. ``sinkhorn`` and ``combine`` are the
    subsets whose Python frame chain contains a frame executing in the
    respective seam file; ``elsewhere`` is the remainder.

    Why the split matters here and did not for ``-027``: this layer calls TWO
    seams, so ``total == 2`` is satisfied by one kernel running twice. Only the
    per-seam attribution says the Sinkhorn ran once AND the combine ran once.

    The frame walk reaches across the ``wrap_nki`` -> HOP -> ``DispatchKey.CPU``
    hop because CPython links each new Python frame to the interpreter's current
    top frame regardless of intervening C++ frames, so the calling Python frame
    stays reachable through ``f_back``. That is not argued here: it is measured,
    because :func:`test_mhc_layer_attribution_control_separates_the_two_seams`
    requires each seam's count to read ``1`` while the other reads ``0`` on
    single dispatches through each one in turn.
    """

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
    """Read every route instrument and return the reading for the transcript.

    Args:
        sim: the attributed simulator counter, entered around the layer calls.
        calls: how many LAYER CALLS happened inside the read window. The plan
            declares the counts PER LAYER CALL, so the per-case expectation is
            that value times this multiplicity -- the conversion ``-033``'s
            block states (per-call value x the case's own call multiplicity),
            recorded with the case rather than assumed to be one.
        label: the case name, printed with the reading.

    The certifying component of each conjunct is named in the failure message
    (D1.4): conjuncts 1-3 are ``-028``'s and ``-029``'s module-level counters,
    conjunct 4 is ``vllm_neuron.utils.neuron_utils.can_run_kernel``, conjunct 5
    is ``nki.simulator.simulate_kernel`` itself.
    """
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
    print(reading)

    if sink_nki != calls:
        raise RouteInstrumentError(
            f"{label}: -028's seam dispatch counter read {sink_nki} over {calls} "
            f"layer call(s); the plan declares exactly ONE per layer call, so "
            f"{calls} was expected. {reading}"
        )
    if comb_nki != calls:
        raise RouteInstrumentError(
            f"{label}: -029's seam dispatch counter read {comb_nki} over {calls} "
            f"layer call(s); the plan declares exactly ONE per layer call, so "
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
            f"{sim.sinkhorn} entries to -028's seam and {sim.combine} to "
            f"-029's; {calls} each was expected. A total that is right while "
            f"the split is wrong means one kernel ran twice. {reading}"
        )
    if sim.elsewhere != 0:
        raise RouteInstrumentError(
            f"{label}: {sim.elsewhere} simulator entries came from neither "
            f"seam. {reading}"
        )
    return reading


# --------------------------------------------------------------------------- #
# The fixture, and the sub-block the layer wraps.                              #
# --------------------------------------------------------------------------- #
def _sublayer(hidden_states: torch.Tensor) -> torch.Tensor:
    """The wrapped sub-block, stood in for by a deterministic elementwise map.

    ``[T, H] -> [T, H]``, nonlinear so ``x`` is not a scaled copy of
    ``layer_input``, and identical on both sides of every comparison. What the
    real sub-block is (attention, MoE) is another D14 section's business; this
    increment measures the mixing around it.
    """
    return torch.tanh(hidden_states) * 1.5


def _fixture(seed: int = 30, tokens: int = T, hidden: int = H):
    """``(fn, hc_scale, hc_base, residual)`` in fp32, on the base's own scales.

    The magnitudes are the base's own kernel test's (``fn`` at ``1e-4``,
    ``hc_scale`` and ``hc_base`` at ``0.1``), so ``mixes`` lands in the regime
    the target actually runs in rather than in sigmoid saturation, where every
    implementation agrees and the comparison would measure nothing.
    """
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
    """Set the layer's parameters. The acceptance is synthetic by declaration.

    The weight map declares the ``multi_hyper_connections`` family absent
    (``weight_loaders_fp8.py:82-86``), so there is no loader on this route and
    the test supplies the tensors -- which is the lead's ruling for this
    increment, not a shortcut taken here.
    """
    with torch.no_grad():
        layer.fn.copy_(fn)
        layer.hc_scale.copy_(hc_scale)
        layer.hc_base.copy_(hc_base)


# --------------------------------------------------------------------------- #
# The comparator: the pinned base's mHC, in BOTH of its spellings.              #
# --------------------------------------------------------------------------- #
def _sinkhorn_normalize_tilelang(
    x: torch.Tensor, repeat: int, eps: float
) -> torch.Tensor:
    """``sinkhorn_normalize_ref``, ``tests/kernels/test_mhc_kernels.py:18-24``.

    Verbatim in structure: ``softmax(-1) + eps``, ONE column pass, then
    ``repeat - 1`` row/column pairs. ``dim=-1`` sums over ``j``, ``dim=-2`` over
    ``i``.
    """
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def _reference_pre_torch(fn, hc_scale, hc_base, residual, eps, alpha, repeat):
    """``mhc_pre_torch``, ``vllm/model_executor/kernels/mhc/torch.py``.

    Kept in fp32 throughout rather than the base's bf16, for ``-028``'s and
    ``-029``'s recorded reason: the declared ``atol`` is ``1e-5`` and bf16's ~3
    decimal digits cannot express that difference, so a bf16 comparator would
    put quantisation noise between the two sides of the check. The base's own
    looser ``atol=5e-2`` is an artefact of its output dtype.

    The one epsilon: the base's signature takes ``rms_eps``, ``hc_pre_eps`` and
    ``hc_sinkhorn_eps`` separately and its own test sets all three to ``1e-6``
    (``test_mhc_kernels.py:121``), which is the fork's single ``hc_eps``.
    """
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
    """``mhc_pre_ref``, ``tests/kernels/test_mhc_kernels.py:27-68``.

    The SECOND, independent spelling: it divides ``sqrsum`` by ``fn.shape[-1]``
    where the first divides by ``hc_mult * hidden_size``, and it concatenates
    ``hc_scale`` into a length-``hc_mult3`` vector where the first multiplies
    per head slice. Same arithmetic, written differently -- which is what makes
    their agreement worth asserting.
    """
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
    """``mhc_post_ref``'s ``bmm(comb.mT, residual)`` spelling, kept fp32.

    The ``.mT`` is the whole content of the ``i``/``j`` convention: a
    implementation reading ``comb[j, i]`` would be transposed, and carrying both
    spellings is how the convention is corroborated rather than remembered.
    """
    term2 = torch.bmm(comb_res_mix.mT.to(torch.float32), residual.float())
    return x.float().unsqueeze(-2) * post_layer_mix.to(torch.float32) + term2


def _reference_layer(fn, hc_scale, hc_base, residual, spelling: str):
    """One full reference layer call: pre, then the sub-block, then post.

    Returns:
        ``(out, post_mix, comb_mix, layer_input, x)``.
    """
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
    """``(max_abs_error, max_rel_error)`` as plain floats, for the transcript."""
    got32 = got.to(torch.float32)
    want32 = want.to(torch.float32)
    diff = (got32 - want32).abs()
    max_abs = float(diff.max())
    denom = want32.abs()
    max_rel = float((diff / torch.where(denom > 0, denom, torch.ones_like(denom))).max())
    return max_abs, max_rel


# --------------------------------------------------------------------------- #
# THE DECLARED ACCEPTANCE -- 1/1 tiny case, both conjuncts in one test.         #
# --------------------------------------------------------------------------- #
def test_mhc_layer_output_matches_the_torch_reference_layer_tiny_case() -> None:
    """Plan L958, both halves: the numbers AND the two per-layer-call counters.

    One layer call, so the per-case totals equal the per-call values: call
    multiplicity ``1``. Recorded with the case, per ``-033``'s convention.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    want, _, want_comb, want_layer_input, _ = _reference_layer(
        fn, hc_scale, hc_base, residual, "torch"
    )

    _reset_both()
    with _AttributedSimulatorCounter() as sim:
        got = layer.forward(residual, _sublayer)
    reading = _assert_route(sim, calls=1, label="acceptance-tiny")
    print(
        "[acceptance-tiny] case_call_multiplicity=1 per_case_total_sinkhorn=1 "
        "per_case_total_combine=1 per_layer_call_declared=1 each"
    )

    assert tuple(got.shape) == (T, S, H), tuple(got.shape)
    assert got.dtype is torch.float32, got.dtype

    max_abs, max_rel = _errors(got, want)
    print(
        f"[acceptance-tiny] T={T} S={S} H={H} rtol={RTOL} atol={ATOL} "
        f"max_abs_error={max_abs:.6e} max_rel_error={max_rel:.6e}"
    )
    # Non-vacuity: an expected tensor sitting at zero would make any tolerance
    # pass and prove nothing.
    print(
        f"[acceptance-tiny] want_absmin={float(want.abs().min()):.6e} "
        f"want_absmax={float(want.abs().max()):.6e} "
        f"comb_absmin={float(want_comb.abs().min()):.6e} "
        f"layer_input_absmax={float(want_layer_input.abs().max()):.6e}"
    )
    if float(want.abs().max()) < 1e-3:
        raise VacuousControlError(
            f"the expected tensor's largest magnitude is "
            f"{float(want.abs().max()):.6e}; the comparison would be vacuous"
        )

    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)
    assert "sinkhorn_nki_dispatch=1" in reading
    assert "combine_nki_dispatch=1" in reading


def test_mhc_layer_the_two_upstream_spellings_agree() -> None:
    """The comparator does not rest on one reading of one upstream file.

    Both spellings are transcribed from tag ``v0.24.0``. If they disagreed, the
    acceptance above would be measuring this file's misreading rather than the
    layer.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    a, _, a_comb, a_input, _ = _reference_layer(
        fn, hc_scale, hc_base, residual, "torch"
    )
    b, _, b_comb, b_input, _ = _reference_layer(
        fn, hc_scale, hc_base, residual, "tilelang"
    )
    out_abs, out_rel = _errors(a, b)
    comb_abs, _ = _errors(a_comb, b_comb)
    in_abs, _ = _errors(a_input, b_input)
    print(
        f"[spellings] out_max_abs={out_abs:.6e} out_max_rel={out_rel:.6e} "
        f"comb_max_abs={comb_abs:.6e} layer_input_max_abs={in_abs:.6e}"
    )
    torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(a_comb, b_comb, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(a_input, b_input, rtol=RTOL, atol=ATOL)


def test_mhc_layer_counters_read_one_per_layer_call_across_two_calls() -> None:
    """"Exactly once per LAYER CALL" is a rate, so it is read at two rates.

    Two consecutive calls in one read window must move each counter by exactly
    two, and the intermediate read after the first call must be exactly one.
    A counter that latched, or one that counted per token, would separate here.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    _reset_both()
    with _AttributedSimulatorCounter() as sim:
        layer.forward(residual, _sublayer)
        mid = _read_both()
        layer.forward(residual, _sublayer)
    print(f"[per-call] after_first_call={mid}")
    assert mid == ((1, 0), (1, 0)), mid
    _assert_route(sim, calls=2, label="per-call-two")
    print(
        "[per-call] case_call_multiplicity=2 per_case_total_sinkhorn=2 "
        "per_case_total_combine=2 per_layer_call=1 each"
    )


# --------------------------------------------------------------------------- #
# THE COMPOSITION CONTROLS -- the block-diagonal embedding, guarded.            #
# --------------------------------------------------------------------------- #
def test_mhc_layer_tokens_are_independent_of_each_other() -> None:
    """Perturb ONE token; every OTHER token's output must be BIT-IDENTICAL.

    This is the arm that guards the layer's composition. ``-028``'s seam
    normalises one matrix, and the rejected way to feed it ``T`` per-token
    matrices -- a flat ``[T*S, S]`` reshape -- makes its column pass sum ACROSS
    tokens, so every token's result would move when one token's input moved.
    ``probe-030-composition-algebra.out`` measures that error at up to
    ``4.68e-01``; this arm is the same defect stated as a property of the
    landed layer.

    Bit-identity is the right bar, not a tolerance: the untouched tokens'
    inputs are unchanged, so a correct per-token composition recomputes exactly
    the same floats.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    base_out = layer.forward(residual, _sublayer)

    perturbed = residual.clone()
    perturbed[0] = perturbed[0] + 0.75
    moved_out = layer.forward(perturbed, _sublayer)

    token0_delta = float((moved_out[0] - base_out[0]).abs().max())
    others_delta = float((moved_out[1:] - base_out[1:]).abs().max())
    print(
        f"[token-independence] token0_max_abs_change={token0_delta:.6e} "
        f"other_tokens_max_abs_change={others_delta:.6e} "
        f"bit_identical_elsewhere={others_delta == 0.0}"
    )
    if token0_delta == 0.0:
        raise VacuousControlError(
            "perturbing token 0 changed token 0's output by exactly 0.0, so "
            "this control could not have detected cross-token leakage"
        )
    assert others_delta == 0.0, (
        f"perturbing token 0 moved other tokens by {others_delta:.6e}; the "
        f"per-token composition is leaking across tokens, which is exactly "
        f"what a flat [T*S, S] reshape into -028's seam would do"
    )


def test_mhc_layer_off_block_entries_stay_zero() -> None:
    """The embedding's zeros stay zero, and each block hits the base's targets.

    Read off the kernel's OWN output rather than argued: the block-diagonal
    embedding is only equivalent to ``T`` independent problems if multiplicative
    rescaling leaves the off-block zeros at zero, so the off-block maximum is
    measured, and each token's row and column sums are read against
    ``row_target()`` and ``column_target(S, S)`` -- ``-028``'s own two numbers,
    imported rather than restated.
    """
    from vllm_neuron.functional.mhc.sinkhorn import sinkhorn_normalise

    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    # Reproduce the layer's own embedding input through its own pre block, then
    # normalise the same embedding again so the raw [T*S, T*S] result is visible
    # here. The layer's helper is used for the extraction so the extraction
    # itself is the one under test.
    _reset_both()
    post_mix, comb_mix, _ = layer.mhc_pre(residual)
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
    print(
        f"[embedding] off_block_max={off_block_max:.6e} "
        f"row_target={row_target()} column_target={column_target(S, S)} "
        f"worst_row_sum_deviation={row_dev:.6e} "
        f"worst_col_sum_deviation={col_dev:.6e} "
        f"extracted_shape={tuple(comb_mix.shape)} "
        f"post_mix_shape={tuple(post_mix.shape)}"
    )
    assert off_block_max == 0.0, off_block_max
    assert tuple(comb_mix.shape) == (T, S, S)
    # The base's own Sinkhorn leaves the LAST-applied axis exact and the other
    # near-exact, and `-028` ends on a column pass too, so the column reading is
    # the tight one. Both are read; neither number is a declared criterion.
    assert col_dev < 1e-5, col_dev
    assert row_dev < 1e-2, row_dev


def _embedding_input(layer, residual: torch.Tensor) -> torch.Tensor:
    """The ``[T, S, S]`` the layer hands to ``block_diag``, recomputed here.

    Deliberately recomputed in this file from the base's spelling rather than
    exposed by the layer: a helper on the module would put the module on both
    sides of the embedding check above.
    """
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
    """The layer against a TRANSPOSED-mix reference must be loudly wrong.

    ``comb_res_mix[t, i, j]`` weights input stream ``i`` into output ``j``. A
    transposed reading is the single most likely error in this whole operation,
    and it is invisible to any symmetric fixture -- so the fixture's asymmetry
    is measured here too, otherwise this arm could pass on a mix that has no
    ``i``/``j`` distinction to get wrong.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    got = layer.forward(residual, _sublayer)
    _, post_mix, comb_mix, _, x = _reference_layer(
        fn, hc_scale, hc_base, residual, "torch"
    )
    asymmetry = float((comb_mix - comb_mix.mT).abs().max())
    # The SAME reference, with only the mix transposed. Nothing else moves, so a
    # difference can only be the i/j convention.
    wrong = _reference_post_torch(
        x, residual, post_mix, comb_mix.mT.contiguous()
    )
    max_abs, max_rel = _errors(got, wrong)
    print(
        f"[transpose] fixture_max_asymmetry={asymmetry:.6e} "
        f"vs_transposed_reference max_abs={max_abs:.6e} max_rel={max_rel:.6e}"
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
    """Perturb one residual element; the comparison must move far outside.

    A tolerance loose enough to pass anything measures nothing, so it is shown
    failing on input it should fail on.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    want, _, _, _, _ = _reference_layer(fn, hc_scale, hc_base, residual, "torch")
    perturbed = residual.clone()
    perturbed[0, 0, 0] = perturbed[0, 0, 0] + 4.0
    got = layer.forward(perturbed, _sublayer)
    max_abs, max_rel = _errors(got, want)
    print(
        f"[armed] perturbed_one_element max_abs_error={max_abs:.6e} "
        f"max_rel_error={max_rel:.6e} against rtol={RTOL} atol={ATOL}"
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_mhc_layer_folds_the_streams_as_the_base_does() -> None:
    """``layer_input`` is the pre-mix fold, and it is checked on its own.

    Without this arm the ``pre_mix`` head is only measured through the sub-block
    and the combine, where a scaling error could partly cancel.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    _reset_both()
    _, _, layer_input = layer.mhc_pre(residual)
    _, _, _, want_input, _ = _reference_layer(
        fn, hc_scale, hc_base, residual, "torch"
    )
    max_abs, max_rel = _errors(layer_input, want_input)
    print(
        f"[layer-input] shape={tuple(layer_input.shape)} "
        f"max_abs_error={max_abs:.6e} max_rel_error={max_rel:.6e} "
        f"absmax={float(want_input.abs().max()):.6e}"
    )
    assert tuple(layer_input.shape) == (T, H)
    torch.testing.assert_close(layer_input, want_input, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# ROUTE CONTROLS -- so every zero above is a measurement.                       #
# --------------------------------------------------------------------------- #
def test_mhc_layer_route_control_fallback_counters_discriminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the simulator disabled BOTH seams take the torch path, and it is COUNTED.

    This is what makes ``torch_fallback == 0`` on both seams a measurement
    rather than a default: each counter is shown reading ``(0, 1)`` through the
    real gate rather than a mock.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0, so this control is unarmed"
    )

    _reset_both()
    with _AttributedSimulatorCounter() as sim:
        out = layer.forward(residual, _sublayer)
    readings = _read_both()
    print(
        f"[route-control] sinkhorn={readings[0]} combine={readings[1]} "
        f"simulate_kernel_total={sim.total}"
    )
    assert readings == ((0, 1), (0, 1)), readings
    assert sim.total == 0, sim.total
    assert tuple(out.shape) == (T, S, H)


def test_mhc_layer_f1_numeric_arm_alone_cannot_discriminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The torch-fallback layer passes the NUMERIC arm, which is why counters decide.

    F1's hazard stated as a measurement: with no kernel running at all, the
    declared tolerance comparison still succeeds, because both seams' fallbacks
    compute the same function. So a green numeric arm is not evidence a kernel
    ran, and the plan's counter clause is the only thing that is.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)
    want, _, _, _, _ = _reference_layer(fn, hc_scale, hc_base, residual, "torch")

    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    _reset_both()
    got = layer.forward(residual, _sublayer)
    readings = _read_both()
    max_abs, max_rel = _errors(got, want)
    print(
        f"[f1] no_kernel_ran sinkhorn={readings[0]} combine={readings[1]} "
        f"max_abs_error={max_abs:.6e} max_rel_error={max_rel:.6e}"
    )
    assert readings == ((0, 1), (0, 1)), readings
    # The point of the arm: this comparison PASSES with zero kernels dispatched.
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_mhc_layer_route_control_simulator_is_load_bearing() -> None:
    """Both NKI chains RAISE without the simulator rather than computing torch.

    Recorded because it forecloses the F1 false green BELOW this repository's
    seams: if the HOP degraded silently, no counter of this campaign's could
    detect it.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    saved = os.environ.get("NKI_SIMULATOR")
    os.environ["NKI_SIMULATOR"] = "0"
    try:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from vllm_neuron.functional.mhc.hyper_connection import (
            hyper_connection_kernel,
        )
        from vllm_neuron.functional.mhc.sinkhorn import sinkhorn_kernel

        with pytest.raises(RuntimeError) as sink_exc:
            wrap_nki(sinkhorn_kernel)(
                affinity=torch.rand(S, S, dtype=torch.float32) + 0.1,
                iters=HC_SINKHORN_ITERS,
            )
        with pytest.raises(RuntimeError) as comb_exc:
            wrap_nki(hyper_connection_kernel)(
                x=torch.rand(T, H, dtype=torch.float32),
                residual=residual,
                post_layer_mix=torch.rand(T, S, 1, dtype=torch.float32),
                comb_res_mix=torch.rand(T, S, S, dtype=torch.float32),
            )
    finally:
        if saved is None:
            os.environ.pop("NKI_SIMULATOR", None)
        else:
            os.environ["NKI_SIMULATOR"] = saved

    print(f"[route-control] sinkhorn_off_raise={str(sink_exc.value)[:120]!r}")
    print(f"[route-control] combine_off_raise={str(comb_exc.value)[:120]!r}")
    assert "simulator" in str(sink_exc.value).lower()
    assert "simulator" in str(comb_exc.value).lower()


def test_mhc_layer_a_pure_torch_layer_reads_zero_on_both_seams() -> None:
    """The plan's falsifier, measured: "a pure-torch layer produces 0 on both".

    The reference layer in this file IS a pure-torch mHC layer. Run with both
    counters zeroed, it must leave them at zero -- so the declared ``1`` on each
    seam is a reading a torch implementation cannot produce.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    _reset_both()
    with _AttributedSimulatorCounter() as sim:
        out, _, _, _, _ = _reference_layer(
            fn, hc_scale, hc_base, residual, "torch"
        )
    readings = _read_both()
    print(
        f"[falsifier] pure_torch_layer sinkhorn={readings[0]} "
        f"combine={readings[1]} simulate_kernel_total={sim.total}"
    )
    assert readings == ((0, 0), (0, 0)), readings
    assert sim.total == 0, sim.total
    assert tuple(out.shape) == (T, S, H)


def test_mhc_layer_attribution_control_separates_the_two_seams() -> None:
    """One dispatch through each seam in turn; the attribution must not blur them.

    Without this arm, ``through_028 == 1 and through_029 == 1`` in the
    acceptance could be a frame walk that matches anything. Here each seam is
    driven ALONE and the other's count must read ``0`` while the total reads
    ``1``. One instrument, two opposite readings on real input.
    """
    from vllm_neuron.functional.mhc.hyper_connection import (
        hyper_connection_combine,
    )
    from vllm_neuron.functional.mhc.sinkhorn import sinkhorn_normalise

    affinity = torch.rand(S, S, dtype=torch.float32) + 0.1
    _reset_both()
    with _AttributedSimulatorCounter() as sink_only:
        sinkhorn_normalise(affinity, iters=HC_SINKHORN_ITERS)
    print(
        f"[attribution] sinkhorn_alone total={sink_only.total} "
        f"through_028={sink_only.sinkhorn} through_029={sink_only.combine} "
        f"elsewhere={sink_only.elsewhere}"
    )
    assert (sink_only.total, sink_only.sinkhorn, sink_only.combine) == (1, 1, 0)

    _, _, _, residual = _fixture()
    _reset_both()
    with _AttributedSimulatorCounter() as comb_only:
        hyper_connection_combine(
            x=torch.rand(T, H, dtype=torch.float32),
            residual=residual,
            post_layer_mix=torch.rand(T, S, 1, dtype=torch.float32),
            comb_res_mix=torch.rand(T, S, S, dtype=torch.float32),
        )
    print(
        f"[attribution] combine_alone total={comb_only.total} "
        f"through_028={comb_only.sinkhorn} through_029={comb_only.combine} "
        f"elsewhere={comb_only.elsewhere}"
    )
    assert (comb_only.total, comb_only.sinkhorn, comb_only.combine) == (1, 0, 1)


def test_mhc_layer_the_two_seam_counters_are_independent() -> None:
    """Resetting one seam's counter must not touch the other's.

    ``-030`` reads two numbers, so the two seams must not share one counter
    object. Both source modules declare this on their own side
    (``sinkhorn.py:362-367``, ``hyper_connection.py:328-335``); this is the
    reading from the consumer that depends on it.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    _reset_both()
    layer.forward(residual, _sublayer)
    assert _read_both() == ((1, 0), (1, 0)), _read_both()

    sinkhorn_mod.reset_dispatch_counters()
    after = _read_both()
    print(f"[independence] after_resetting_only_sinkhorn={after}")
    assert after == ((0, 0), (1, 0)), after

    combine_mod.reset_dispatch_counters()
    after2 = _read_both()
    print(f"[independence] after_resetting_only_combine={after2}")
    assert after2 == ((0, 0), (0, 0)), after2


# --------------------------------------------------------------------------- #
# THE TOKEN CEILING -- measured at the boundary, refusal allowed to propagate.  #
# --------------------------------------------------------------------------- #
def test_mhc_layer_runs_at_the_embedding_token_ceiling() -> None:
    """``T = PARTITION_MAX // S`` runs, and both counters still read one each.

    The boundary is measured rather than assumed, so the refusal arm below is
    known to be a refusal of what is genuinely out of range and not of
    something that never worked.
    """
    ceiling = PARTITION_MAX // S
    fn, hc_scale, hc_base, residual = _fixture(tokens=ceiling)
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    _reset_both()
    with _AttributedSimulatorCounter() as sim:
        got = layer.forward(residual, _sublayer)
    _assert_route(sim, calls=1, label=f"ceiling-T{ceiling}")
    want, _, _, _, _ = _reference_layer(fn, hc_scale, hc_base, residual, "torch")
    max_abs, max_rel = _errors(got, want)
    print(
        f"[ceiling] T={ceiling} sinkhorn_M=N={ceiling * S} "
        f"PARTITION_MAX={PARTITION_MAX} max_abs_error={max_abs:.6e} "
        f"max_rel_error={max_rel:.6e}"
    )
    assert tuple(got.shape) == (ceiling, S, H)
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_mhc_layer_above_the_ceiling_the_seam_refusal_propagates() -> None:
    """One token past the ceiling: ``SinkhornError`` reaches the caller unchanged.

    No pad, no tile, no torch fallback -- the lead ruled that the tiling policy
    is a design revision's and P13 forbids the fallback outright. The refusal's
    own message is captured, and both counters are read AFTER it to show a
    refused call charges neither seam.
    """
    over = PARTITION_MAX // S + 1
    fn, hc_scale, hc_base, residual = _fixture(tokens=over)
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    _reset_both()
    with pytest.raises(SinkhornError) as excinfo:
        layer.forward(residual, _sublayer)
    after = _read_both()
    message = str(excinfo.value)
    print(f"[ceiling-refusal] T={over} sinkhorn_M={over * S} message={message!r}")
    print(f"[ceiling-refusal] counters_after_refusal={after}")
    assert f"M={over * S}" in message, message
    assert f"PARTITION_MAX={PARTITION_MAX}" in message, message
    assert after == ((0, 0), (0, 0)), after


@pytest.mark.parametrize(
    "tokens,needle",
    [
        (PARTITION_MAX // S + 1, f"exceeds PARTITION_MAX={PARTITION_MAX}"),
        (PARTITION_MAX, "exceeds PARTITION_MAX"),
    ],
)
def test_mhc_layer_refusal_names_the_offending_extent(tokens, needle) -> None:
    """The refusal names ``M`` and the bound, at two token counts above it.

    A refusal a caller cannot act on is barely better than a trap, so the
    message content is asserted rather than only the exception type.
    """
    fn, hc_scale, hc_base, residual = _fixture(tokens=tokens)
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)
    with pytest.raises(SinkhornError) as excinfo:
        layer.forward(residual, _sublayer)
    print(f"[refusal] T={tokens} message={str(excinfo.value)!r}")
    assert needle in str(excinfo.value)


# --------------------------------------------------------------------------- #
# LAYER-LEVEL REFUSALS -- the agreements neither seam can see.                  #
# --------------------------------------------------------------------------- #
def test_mhc_layer_refuses_a_stream_count_that_contradicts_its_config() -> None:
    """A wrong ``S`` would MIS-SLICE ``mixes`` rather than fail, so it is refused.

    The seams read their extents off the tensors they are handed, so neither can
    tell that the stream count disagrees with the config this layer was built
    from. That is why this check is here and not a restatement of theirs.
    """
    impl = _impl()
    layer = _layer()
    residual = torch.randn(T, S + 1, H, dtype=torch.float32)
    with pytest.raises(impl.Glm5NextHyperConnectionError) as excinfo:
        layer.forward(residual, _sublayer)
    print(f"[refusal] stream_mismatch={str(excinfo.value)!r}")
    assert f"S={S + 1}" in str(excinfo.value)
    assert f"hc_mult={S}" in str(excinfo.value)


def test_mhc_layer_refuses_a_hidden_extent_that_contradicts_its_config() -> None:
    """A wrong ``H`` would make ``fn`` unmultipliable; it is named instead."""
    impl = _impl()
    layer = _layer()
    residual = torch.randn(T, S, H + 8, dtype=torch.float32)
    with pytest.raises(impl.Glm5NextHyperConnectionError) as excinfo:
        layer.forward(residual, _sublayer)
    print(f"[refusal] hidden_mismatch={str(excinfo.value)!r}")
    assert f"H={H + 8}" in str(excinfo.value)


def test_mhc_layer_refuses_a_non_3d_residual() -> None:
    """``[T, S, H]`` is the rank the whole operation is defined on."""
    impl = _impl()
    layer = _layer()
    with pytest.raises(impl.Glm5NextHyperConnectionError) as excinfo:
        layer.forward(torch.randn(T, H, dtype=torch.float32), _sublayer)
    print(f"[refusal] rank={str(excinfo.value)!r}")
    assert "3-D" in str(excinfo.value)


def test_mhc_layer_refuses_a_bad_sublayer() -> None:
    """A non-callable sub-block, and one that changes the shape, both named.

    The combine reads ``x`` and the streams on the same token and hidden
    extents, so a sub-block that reshapes would otherwise surface as a
    ``HyperConnectionError`` about a tensor the caller never passed.
    """
    impl = _impl()
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)

    with pytest.raises(impl.Glm5NextHyperConnectionError) as not_callable:
        layer.forward(residual, object())
    print(f"[refusal] sublayer_not_callable={str(not_callable.value)!r}")
    assert "callable" in str(not_callable.value)

    with pytest.raises(impl.Glm5NextHyperConnectionError) as wrong_shape:
        layer.forward(residual, lambda t: t[:, :4])
    print(f"[refusal] sublayer_wrong_shape={str(wrong_shape.value)!r}")
    assert "expected the [T, H] shape" in str(wrong_shape.value)


def test_mhc_layer_refuses_a_non_positive_iteration_count() -> None:
    """The configuration refusal, at construction rather than at the seam."""
    impl = _impl()
    with pytest.raises(impl.Glm5NextHyperConnectionError) as excinfo:
        _layer(iters=0)
    print(f"[refusal] iters={str(excinfo.value)!r}")
    assert "must be positive" in str(excinfo.value)


def test_mhc_layer_neuron_config_overrides_win() -> None:
    """``NeuronConfig``'s two mHC overrides take precedence when set.

    ``-013``'s section note for this class states that contract, so it is
    measured rather than left to the reader.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
    from vllm_neuron.model.neuron_config import NeuronConfig

    impl = _impl()
    text_config = Glm5NextTextConfig(
        hidden_size=H, hc_mult=S, hc_sinkhorn_iters=20, hc_eps=1e-06
    )
    neuron_config = NeuronConfig(mhc_sinkhorn_iters=7, mhc_eps=1e-04)
    layer = impl.Glm5NextHyperConnection(text_config, neuron_config)
    print(
        f"[overrides] sinkhorn_iters={layer.sinkhorn_iters} hc_eps={layer.hc_eps} "
        f"(config said 20 / 1e-06)"
    )
    assert layer.sinkhorn_iters == 7
    assert layer.hc_eps == 1e-04

    plain = impl.Glm5NextHyperConnection(text_config, None)
    assert (plain.sinkhorn_iters, plain.hc_eps) == (20, 1e-06)


def test_mhc_layer_stream_count_matches_the_target_hc_mult() -> None:
    """``S`` is the checkpoint's ``hc_mult``, imported from ``-028``, not chosen.

    Three readings of one number must agree: this file's ``S``, ``-028``'s
    ``MHC_STREAMS``, and the checkpoint config's ``hc_mult`` default.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    default_hc_mult = int(Glm5NextTextConfig().hc_mult)
    print(
        f"[dials] S={S} MHC_STREAMS={MHC_STREAMS} "
        f"config_hc_mult={default_hc_mult} "
        f"config_iters={int(Glm5NextTextConfig().hc_sinkhorn_iters)} "
        f"config_hc_eps={float(Glm5NextTextConfig().hc_eps)} "
        f"PARTITION_MAX={PARTITION_MAX} ceiling_T={PARTITION_MAX // S}"
    )
    assert S == MHC_STREAMS == default_hc_mult == 4
    assert int(Glm5NextTextConfig().hc_sinkhorn_iters) == HC_SINKHORN_ITERS
    assert float(Glm5NextTextConfig().hc_eps) == HC_EPS


def test_mhc_layer_parameters_are_real_and_sized_from_the_config() -> None:
    """The three parameters exist, are allocated, and carry the base's names.

    Ordinary ``nn.Parameter`` rather than ``_declare_parameters``' reservation is
    the lead's ruling for this section: the weight map declares this family
    absent, so there is no map name to reserve and the acceptance sets these
    tensors itself.
    """
    layer = _layer()
    names = {name for name, _ in layer.named_parameters()}
    print(
        f"[params] names={sorted(names)} fn={tuple(layer.fn.shape)} "
        f"hc_scale={tuple(layer.hc_scale.shape)} "
        f"hc_base={tuple(layer.hc_base.shape)} hc_mult3={layer.hc_mult3}"
    )
    assert names == {"fn", "hc_scale", "hc_base"}
    assert layer.hc_mult3 == 2 * S + S * S
    assert tuple(layer.fn.shape) == (layer.hc_mult3, S * H)
    assert tuple(layer.hc_scale.shape) == (3,)
    assert tuple(layer.hc_base.shape) == (layer.hc_mult3,)


def test_mhc_layer_combine_seam_refusals_reach_the_caller() -> None:
    """``-029``'s own refusal is not swallowed by this layer either.

    Driven by handing :meth:`mhc_post` a mismatched ``x``, which is the one
    combine argument the layer does not build itself.
    """
    fn, hc_scale, hc_base, residual = _fixture()
    layer = _layer()
    _load(layer, fn, hc_scale, hc_base)
    post_mix, comb_mix, _ = layer.mhc_pre(residual)
    with pytest.raises(HyperConnectionError) as excinfo:
        layer.mhc_post(
            torch.randn(T, H + 8, dtype=torch.float32), residual, post_mix, comb_mix
        )
    print(f"[refusal] combine_seam={str(excinfo.value)!r}")
    assert "expected [T, H]" in str(excinfo.value)
