# SPDX-License-Identifier: Apache-2.0
"""Acceptance for `inc-glm53f-084` -- the KDA gate in the reference's own form.

**Two items, one per declared case, and no ``parametrize`` decorator in this file**
(D1.2). Each item names the component whose behaviour it certifies (D1.4). The
whole file is the selection, so the collected count is 2.

Run on the Tier N harness -- the NKI simulator on the host CPU, no device and no
lease. The recorded form of that harness is ``evidence-025.md:408-412``::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/kda/test_gate_clamp.py \
        -q -s --timeout 60 -p no:cacheprovider

THE COMPARATOR IS THE BLOCK'S OWN AND IS CITED, NOT CHOSEN HERE. The block's
Acceptance bullet (increment plan rev 60, line 549) reads
``assert_close(rtol=1e-2, atol=1e-5)`` over **2/2** declared cases with the worst
error reported as a number, and it states that the pair is copied byte-for-byte
from ``inc-glm53f-046``'s Acceptance bullet (plan line 663) so that no new
tolerance value is minted (P9). This file therefore calls
``torch.testing.assert_close`` with exactly that pair and mints nothing.

THE REFERENCE IS AUTHORED HERE, IN TORCH, BECAUSE THE BLOCK SAYS SO. The
Acceptance bullet asks for "a torch reference of the same function authored in the
test". ``_reference`` below is that function and nothing else is: the module under
test carries no torch oracle any more, so there is only one spelling of the
reference and the two cannot drift apart.

WHY THE PRE-REWRITE GATE FAILS THESE TWO ITEMS. The landed
``inc-glm53f-037`` gate computed ``clamp(-exp(A_log) * softplus(g + bias),
min=-5.0)``, which is a DIFFERENT function from the reference's
``-5.0 * sigmoid(exp(A_log) * (g + bias))``. The two disagree most at
``g = ln 4``, and they disagree at ``g = 0`` for every ``A_log`` except a narrow
coincidence band near ``A_log = 1.283``; case 2 therefore fixes ``A_log = 0``,
away from that band, so the reading cannot be an accident. The screen note
``pin-feasibility-note-lap-0903b.md`` S2 holds the closed-form gap and every
reference line, and nothing of it is restated here.

WHICH READING CLOSES A TORCH-FALLBACK PASS. Two, and both are needed: the seam's
``torch_fallback`` counter must read exactly ``0``, and ``can_run_gate_clamp`` must
be ``True`` so that a ``0`` cannot mean "the seam was never entered". The
``nki_dispatch`` equality against ``1`` per case then fixes that the NKI route ran
once (route predicate D13 form R-1, per the block's own bullet). No item lets the
seam reach a torch path, because the module has none.
"""

from __future__ import annotations

import math

import torch

from vllm_neuron.functional.kda.gate_clamp import (
    can_run_gate_clamp,
    gate_clamp_dispatch_counters,
    gate_clamp_kernel_identity,
    kda_gate_clamp,
    reset_gate_clamp_dispatch_counters,
)

#: The block's comparator, quoted from its Acceptance bullet at increment plan
#: line 549: ``assert_close(rtol=1e-2, atol=1e-5)``. Registered before measurement
#: and not touched after (P9).
RTOL = 1e-2
ATOL = 1e-5

#: The gate's lower bound. THIS FILE CARRIES THE VALUE AS A LITERAL AND CITES ITS
#: SOURCE RATHER THAN IMPORTING IT, for two reasons. The bound is now a REQUIRED
#: argument of the seam with no module default, so a caller has to supply it; and
#: the block forbids ``functional/kda/`` from importing ``model/glm5_next/``, which
#: this test mirrors so that the test cannot be the thing that creates the
#: dependency. The source is ``"gate_lower_bound": -5.0`` at
#: ``vllm_neuron/model/glm5_next/config.py:157``, inside the ``linear_attn_config``
#: field of ``Glm5NextTextConfig``.
GATE_LOWER_BOUND = -5.0

#: Case 1's geometry. Both axes pass through a transpose inside the kernel, and 8
#: tokens by 16 channels is small while leaving the two axes UNEQUAL, so an
#: orientation error cannot hide in a square tile.
TOKENS = 8
KDIM = 16

#: Case 1's gate input scale. At this scale the sigmoid spans from near-saturated
#: to near-linear across the tile, which is what "mixed-magnitude" has to mean for
#: a saturating function.
GATE_SCALE = 5.0

#: Case 1's decay exponent. NON-ZERO on purpose: ``exp(A_log)`` scales the
#: pre-activation, so a kernel that dropped the scale entirely would still pass at
#: ``A_log = 0`` where the scale is exactly ``1.0``. Case 2 fixes ``A_log = 0`` for
#: a different reason of its own, so between them both readings are covered.
CASE1_A_LOG = 0.35

SEED = 20260903

#: Case 2's per-channel pre-activation ladder, 16 values for 16 key channels.
#: ``0.0`` is first because the block names ``g = 0`` explicitly, and the ladder is
#: signed and symmetric in magnitude so that "both signs" is a property of the data
#: rather than of a claim. ``+-1.3862944`` is ``+-ln 4``, the point where the
#: pre-rewrite gate and the reference are furthest apart, so the ladder passes
#: through the worst case rather than around it.
CASE2_BASE = (
    0.0,
    0.5,
    -0.5,
    1.0,
    -1.0,
    1.3862944,
    -1.3862944,
    3.0,
    -3.0,
    6.0,
    -6.0,
    9.0,
    -9.0,
    12.0,
    -15.0,
    15.0,
)

#: Case 2's per-token scale ladder, 8 values for 8 tokens. The last entry is
#: negative, which flips the whole signed ladder and doubles the sign coverage; the
#: largest product is ``2.0 * 15.0 = 30.0``, deep into saturation on both sides.
CASE2_ROW_SCALE = (1.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0, -1.0)


def _reference(
    g: torch.Tensor,
    a_log: torch.Tensor,
    lower: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """The reference function, in torch: ``lower * sigmoid(exp(A_log) * (g + bias))``.

    This is the transformers v5.16.1 ``Glm5NextTextForgetGate`` limb the campaign
    declares as its correctness reference; the screen note
    ``pin-feasibility-note-lap-0903b.md`` S2 holds the source lines. The
    unreachable softplus limb is not written here either, for the same reason the
    kernel does not implement it: the bound is never ``None`` for this checkpoint.

    P13 note: this is the acceptance's REFERENCE, not a route. It lives in the test
    file, so it cannot be reached from the module under test at all.
    """
    pre = g.to(torch.float32)
    if bias is not None:
        pre = pre + bias.reshape(1, -1).to(torch.float32)
    decay_rate = torch.exp(a_log.reshape(()).to(torch.float32))
    return lower * torch.sigmoid(decay_rate * pre)


def _case1_inputs():
    """Case 1: a random gate tile, a per-channel bias, a non-zero ``A_log``.

    THE BIAS IS WHAT MAKES THE TWO INTERNAL TRANSPOSES FALSIFIABLE. It carries a
    different value per key channel, so a kernel that mixed up its axes would apply
    channel biases to the wrong channels and the difference would exceed the
    tolerance by orders of magnitude. A zero or scalar bias would let an
    orientation error pass unseen.
    """
    gen = torch.Generator().manual_seed(SEED)
    g = torch.randn((TOKENS, KDIM), generator=gen, dtype=torch.float32) * GATE_SCALE
    bias = torch.randn((KDIM,), generator=gen, dtype=torch.float32) * 0.5
    a_log = torch.tensor(CASE1_A_LOG, dtype=torch.float32)
    return g, a_log, bias


def _case2_inputs():
    """Case 2: an explicit saturation ladder, no bias, ``A_log = 0``.

    NO BIAS, so every entry of the tile is exactly the declared product and the
    claim "``g = 0`` is in the data" is checkable by reading the tile rather than
    by trusting the construction. ``A_log = 0`` makes ``exp(A_log)`` exactly
    ``1.0``, which keeps the ladder's numbers the pre-activation's own numbers, and
    it sits far from the ``A_log = 1.283`` coincidence band where the pre-rewrite
    gate happens to agree with the reference at ``g = 0``.
    """
    base = torch.tensor(CASE2_BASE, dtype=torch.float32)
    scale = torch.tensor(CASE2_ROW_SCALE, dtype=torch.float32)
    g = scale.reshape(-1, 1) * base.reshape(1, -1)
    a_log = torch.zeros((), dtype=torch.float32)
    return g, a_log


def _one_call(g, a_log, bias):
    """Reset, make exactly one seam call, read the counters. Returns both."""
    reset_gate_clamp_dispatch_counters()
    got = kda_gate_clamp(g, a_log, bias=bias, lower=GATE_LOWER_BOUND)
    return got, gate_clamp_dispatch_counters()


def _report(item: str, certifies: str) -> None:
    """Name what this item certifies, and name the kernel it actually ran.

    THE IDENTITY IS PRINTED, NOT ASSERTED, because the block declares two items and
    a third would change the declared count. Printing it inside each item still
    puts the reading in the transcript of the declared acceptance command, which is
    where a reviewer looks, and the landed ``test_chunked_recurrence`` prints its
    own kernel identity the same way.
    """
    module, qualname = gate_clamp_kernel_identity()
    print(
        f"\nGATE|{item}|certifies={certifies}"
        f"|kernel_identity=({module}, {qualname})",
        flush=True,
    )


def _worst(got: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    """``(worst_abs_diff, worst_ratio_of_diff_to_its_own_allowance)``.

    The second number is the one that says how close the case came to failing:
    ``assert_close`` allows ``atol + rtol*|reference|`` per element, so the ratio of
    the difference to that per-element allowance is comparable across elements of
    very different magnitude, and its maximum is the real margin. A value below
    ``1.0`` is a pass with that much room; ``0.5`` means the worst element used half
    its budget.
    """
    diff = (got - reference).abs()
    allowance = ATOL + RTOL * reference.abs()
    return diff.max().item(), (diff / allowance).max().item()


def test_the_gate_matches_the_reference_on_a_mixed_magnitude_case():
    """Case 1 of 2. Certifying component: the ``wrap_nki`` seam in ``gate_clamp``.

    A random pre-activation tile, a per-channel bias and a non-zero ``A_log``,
    compared against the reference function at the block's comparator. This is the
    item that exercises the ``exp(A_log)`` scale, the per-channel bias and both
    internal transposes together.
    """
    _report("mixed_magnitude", "kda_gate_clamp (the wrap_nki seam this block rewrites)")
    g, a_log, bias = _case1_inputs()

    assert can_run_gate_clamp(g, TOKENS, KDIM) is True, (
        "the NKI route must be available on the Tier N harness; a False here "
        "would mean the seam could not be entered and the counter readings below "
        "would be vacuous"
    )

    reference = _reference(g, a_log, GATE_LOWER_BOUND, bias)
    got, (nki_dispatch, torch_fallback) = _one_call(g, a_log, bias)

    assert tuple(got.shape) == (TOKENS, KDIM), (
        f"the result must keep the [T, D] boundary orientation "
        f"{(TOKENS, KDIM)}, got {tuple(got.shape)}"
    )

    worst_abs, worst_ratio = _worst(got, reference)
    print(
        f"GATE_CASE|mixed_magnitude|dispatch={nki_dispatch}|declared=1"
        f"|fallback={torch_fallback}"
        f"|a_log={CASE1_A_LOG}|decay_rate={math.exp(CASE1_A_LOG):.6f}"
        f"|worst_abs_diff={worst_abs:.6e}"
        f"|worst_allowance_ratio={worst_ratio:.6f}"
        f"|comparator=rtol{RTOL:.0e},atol{ATOL:.0e}"
        f"|reference_min={reference.min().item():.6f}"
        f"|reference_max={reference.max().item():.6f}"
        f"|got_min={got.min().item():.6f}"
        f"|got_max={got.max().item():.6f}"
        f"|elements={got.numel()}",
        flush=True,
    )

    torch.testing.assert_close(got, reference, rtol=RTOL, atol=ATOL)

    assert nki_dispatch == 1, (
        f"one call must make exactly one dispatch; read {nki_dispatch}"
    )
    assert torch_fallback == 0, (
        f"the module carries no torch route for this kernel-class op, so this "
        f"counter can only be 0; read {torch_fallback}"
    )


def test_the_gate_matches_the_reference_through_saturation_and_at_zero():
    """Case 2 of 2. Certifying component: the same ``wrap_nki`` seam.

    The sigmoid saturates toward the bound as the pre-activation grows, so the port
    and the reference must agree at large ``|g|`` in both signs and at ``g = 0``.
    The ladder is asserted to contain those points before the comparison is made,
    so the case cannot silently stop covering what it claims to cover.
    """
    _report("saturation", "kda_gate_clamp (the wrap_nki seam this block rewrites)")
    g, a_log = _case2_inputs()

    zeros = int((g == 0.0).sum())
    assert zeros == len(CASE2_ROW_SCALE), (
        f"the ladder must put g = 0 in every token row -- {len(CASE2_ROW_SCALE)} "
        f"entries -- because the block names that point; found {zeros}"
    )
    assert g.max().item() >= 20.0 and g.min().item() <= -20.0, (
        f"the ladder must reach deep saturation in BOTH signs; it spans "
        f"[{g.min().item()}, {g.max().item()}]"
    )

    assert can_run_gate_clamp(g, TOKENS, KDIM) is True, (
        "the NKI route must be available on the Tier N harness; a False here "
        "would mean the seam could not be entered and the counter readings below "
        "would be vacuous"
    )

    reference = _reference(g, a_log, GATE_LOWER_BOUND, None)
    got, (nki_dispatch, torch_fallback) = _one_call(g, a_log, None)

    assert tuple(got.shape) == (TOKENS, KDIM), (
        f"the result must keep the [T, D] boundary orientation "
        f"{(TOKENS, KDIM)}, got {tuple(got.shape)}"
    )

    at_zero = got[g == 0.0]
    saturated_low = got[g <= -20.0]
    saturated_high = got[g >= 20.0]
    worst_abs, worst_ratio = _worst(got, reference)
    print(
        f"GATE_CASE|saturation|dispatch={nki_dispatch}|declared=1"
        f"|fallback={torch_fallback}"
        f"|a_log=0.0|decay_rate=1.000000"
        f"|worst_abs_diff={worst_abs:.6e}"
        f"|worst_allowance_ratio={worst_ratio:.6f}"
        f"|comparator=rtol{RTOL:.0e},atol{ATOL:.0e}"
        f"|g_span=[{g.min().item():.6f},{g.max().item():.6f}]"
        f"|got_at_g0_worst={(at_zero - 0.5 * GATE_LOWER_BOUND).abs().max().item():.6e}"
        f"|got_at_g0_first={at_zero.reshape(-1)[0].item():.9g}"
        f"|reference_at_g0={0.5 * GATE_LOWER_BOUND!r}"
        f"|got_saturated_low_max_abs={saturated_low.abs().max().item():.6e}"
        f"|got_saturated_high_worst_from_bound="
        f"{(saturated_high - GATE_LOWER_BOUND).abs().max().item():.6e}"
        f"|nonfinite={int((~torch.isfinite(got)).sum())}"
        f"|elements={got.numel()}",
        flush=True,
    )

    assert int((~torch.isfinite(got)).sum()) == 0, (
        "saturation must not produce a non-finite value; the sigmoid is bounded "
        "in (0, 1) at every input, so any NaN or inf is the kernel's"
    )

    torch.testing.assert_close(got, reference, rtol=RTOL, atol=ATOL)

    assert nki_dispatch == 1, (
        f"one call must make exactly one dispatch; read {nki_dispatch}"
    )
    assert torch_fallback == 0, (
        f"the module carries no torch route for this kernel-class op, so this "
        f"counter can only be 0; read {torch_fallback}"
    )
