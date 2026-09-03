# SPDX-License-Identifier: Apache-2.0
"""Acceptance for `inc-glm53f-037` -- the KDA gate activation with its lower bound.

**Three items, one per declared case, and no ``parametrize`` decorator in this
file** (D1.2). Each item names the component whose behaviour it certifies (D1.4).
The whole file is the selection, so the collected count is 3.

Run on the Tier N harness -- the NKI simulator on the host CPU, no device and no
lease::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/kda/test_gate_clamp.py \
        -q -s --timeout 60 -p no:cacheprovider

THE COMPARATOR IS THE BLOCK'S OWN AND IS NOT THE KDA ``assert_close`` PAIR. The
block's Acceptance bullet says **max abs diff <= 1e-5** over 1/1 tiny case, and
gives its own ground: "single-op scope, so single-op tolerance". So this file
computes a maximum absolute difference and compares it to ``1e-5`` directly.
Substituting the sibling increments' ``assert_close(rtol, atol)`` pair would be a
different criterion, so it is not used here (P9).

WHY THE TWO BOUNDARY ITEMS ASSERT EXACT EQUALITY AND NOT THE TOLERANCE. The
distance from ``-5.0`` to the next float32 below it is one ULP, ``2**-21``, which is
``4.768372e-07`` -- about **21 times SMALLER** than the ``1e-5`` tolerance. A kernel
that dropped the bound entirely would therefore still pass a ``1e-5`` comparison on
the ``-5.0-eps`` input, and the boundary reading would be vacuous. The block asks
that those inputs "produce the clamped value", which is an exact claim, so these
two items assert bit equality with ``-5.0``.

HOW THE BOUNDARY INPUTS REACH THE BOUND EXACTLY. Three choices, each forced:
``A_log = 0`` so ``-exp(A)`` is exactly ``-1.0`` (``exp(0)`` is exact); no bias, so
nothing is added; and ``beta = 5.0``, which puts ``beta*g = 25`` above upstream's
``threshold = 20`` and so inside the branch where the activation is exactly the
identity. With the DEFAULT ``beta = 1.0`` the same ``g = 5.0`` gives
``-5.006715297698975``, not ``-5.0``, so no exact boundary exists at the default and
the reading would test softplus's rounding instead of the bound. ``beta`` does not
change what the bound does -- it only controls how the input arrives at it -- and
the default ``beta`` is exercised by the tolerance item below.

WHICH READING CLOSES A TORCH-FALLBACK PASS. Two, and both are needed: the seam's
``torch_fallback`` counter must read exactly ``0``, and ``can_run_gate_clamp`` must
be ``True`` so that a ``0`` cannot mean "the seam was never entered". The
``nki_dispatch`` equality against ``1`` per case then fixes that the NKI route ran
once. The module's torch oracle is this file's REFERENCE, not a route: no item lets
the seam reach it.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional.kda.gate_clamp import (
    GATE_SOFTPLUS_BETA,
    KDA_GATE_LOWER_BOUND,
    can_run_gate_clamp,
    gate_clamp_dispatch_counters,
    kda_gate_clamp,
    kda_gate_clamp_torch_oracle,
    reset_gate_clamp_dispatch_counters,
)

#: The block's comparator, quoted from its Acceptance bullet: "max abs diff
#: <= 1e-5". Registered before measurement and not touched after (P9).
MAX_ABS_DIFF = 1e-5

#: One ULP at ``5.0`` in float32, exactly. ``5.0`` lies in the binade ``[4, 8)``,
#: where the unbiased exponent is ``2``, and float32 carries 23 fraction bits, so
#: the spacing of representable numbers there is ``2**(2-23) = 2**-21``. Both
#: ``5.0 + EPS`` and ``-5.0 - EPS`` are therefore exactly representable and
#: distinct from ``5.0`` and ``-5.0``. This is the SMALLEST admissible epsilon,
#: which makes it the hardest boundary case available at this dtype.
EPS = 2.0**-21

#: The tiny case's geometry. Both axes pass through a transpose inside the kernel,
#: and 8 tokens by 16 channels is small enough to be a "tiny case" while leaving
#: the two axes UNEQUAL, so an orientation error cannot hide in a square tile.
TOKENS = 8
KDIM = 16

#: Gate input scale. Chosen by measurement, not taste: at this scale the bound
#: actually bites on a fifth of the elements, so the item reads the composition
#: rather than the activation alone. Each item reports its own clamped count.
GATE_SCALE = 5.0

#: The boundary beta. See the module docstring for why the default cannot serve an
#: exact boundary.
BOUNDARY_BETA = 5.0

SEED = 20260905


def _tolerance_inputs():
    """The tiny case: a random gate tile and a per-channel bias.

    THE BIAS IS WHAT MAKES THE TWO INTERNAL TRANSPOSES FALSIFIABLE. It carries a
    different value per key channel, so a kernel that mixed up its axes would apply
    channel biases to the wrong channels and the difference would exceed the
    tolerance by orders of magnitude. A zero or scalar bias would let an
    orientation error pass unseen.
    """
    gen = torch.Generator().manual_seed(SEED)
    g = torch.randn((TOKENS, KDIM), generator=gen, dtype=torch.float32) * GATE_SCALE
    bias = torch.randn((KDIM,), generator=gen, dtype=torch.float32) * 0.5
    a_log = torch.zeros((), dtype=torch.float32)
    return g, a_log, bias


def _one_call(g, a_log, bias, beta):
    """Reset, make exactly one seam call, read the counters. Returns both."""
    reset_gate_clamp_dispatch_counters()
    got = kda_gate_clamp(g, a_log, bias, beta=beta)
    return got, gate_clamp_dispatch_counters()


def _report(item: str, certifies: str) -> None:
    print(f"\nGATE|{item}|certifies={certifies}", flush=True)


def _run_boundary_case(case: str, gate_value: float) -> None:
    """One boundary case: every element enters at ``gate_value`` after activation.

    ``gate_value`` is the value the ACTIVATION produces, so the input is its
    negation: ``-exp(0) * softplus_5(g) = -g`` for ``g`` in the linear branch.
    """
    g = torch.full((TOKENS, KDIM), -gate_value, dtype=torch.float32)
    a_log = torch.zeros((), dtype=torch.float32)

    assert can_run_gate_clamp(g, TOKENS, KDIM, BOUNDARY_BETA) is True, (
        "the NKI route must be available on the Tier N harness; a False here "
        "would mean the seam took the torch fallback"
    )

    unclamped = kda_gate_clamp_torch_oracle(
        g, a_log, None, beta=BOUNDARY_BETA, lower=float("-inf")
    )
    assert unclamped.min().item() == gate_value, (
        f"the boundary construction must put the ACTIVATION exactly at "
        f"{gate_value!r} before the bound is applied; it produced "
        f"{unclamped.min().item()!r}, so this case would test rounding rather "
        f"than the bound"
    )

    got, (nki_dispatch, torch_fallback) = _one_call(g, a_log, None, BOUNDARY_BETA)

    assert tuple(got.shape) == (TOKENS, KDIM), (
        f"the result must keep the [T, D] boundary orientation "
        f"{(TOKENS, KDIM)}, got {tuple(got.shape)}"
    )
    exactly_bound = int((got == KDA_GATE_LOWER_BOUND).sum())
    assert exactly_bound == got.numel(), (
        f"every element must come back exactly at the bound "
        f"{KDA_GATE_LOWER_BOUND!r}; {got.numel() - exactly_bound} of "
        f"{got.numel()} did not, worst {got.min().item()!r}"
    )

    assert nki_dispatch == 1, (
        f"one call must make exactly one dispatch; read {nki_dispatch}"
    )
    assert torch_fallback == 0

    print(
        f"GATE_CASE|{case}|dispatch={nki_dispatch}|declared=1"
        f"|fallback={torch_fallback}"
        f"|beta={BOUNDARY_BETA}"
        f"|activation_before_bound={unclamped.min().item():.9g}"
        f"|activation_is_below_bound={unclamped.min().item() < KDA_GATE_LOWER_BOUND}"
        f"|elements_exactly_at_bound={exactly_bound}/{got.numel()}"
        f"|worst_abs_diff_from_bound="
        f"{(got - KDA_GATE_LOWER_BOUND).abs().max().item():.3e}"
        f"|eps={EPS!r}",
        flush=True,
    )


def test_the_gate_and_its_bound_match_the_upstream_reference():
    """The tiny case. Certifying component: the ``wrap_nki`` seam in ``gate_clamp``.

    Compares the kernel against upstream's threshold form of the activation
    composed with the port's lower bound, at the block's own comparator. This is
    the item that exercises the DEFAULT ``beta = 1.0``, the transcendental branch of
    softplus, the per-channel bias, and both internal transposes.
    """
    _report("tolerance", "kda_gate_clamp (the wrap_nki seam this block authors)")
    g, a_log, bias = _tolerance_inputs()

    assert can_run_gate_clamp(g, TOKENS, KDIM, GATE_SOFTPLUS_BETA) is True, (
        "the NKI route must be available on the Tier N harness; a False here "
        "would mean the seam took the torch fallback"
    )

    reference = kda_gate_clamp_torch_oracle(g, a_log, bias, beta=GATE_SOFTPLUS_BETA)
    got, (nki_dispatch, torch_fallback) = _one_call(g, a_log, bias, GATE_SOFTPLUS_BETA)

    assert tuple(got.shape) == (TOKENS, KDIM), (
        f"the result must keep the [T, D] boundary orientation "
        f"{(TOKENS, KDIM)}, got {tuple(got.shape)}"
    )

    diff = (got - reference).abs()
    worst = diff.max().item()
    assert worst <= MAX_ABS_DIFF, (
        f"max abs diff {worst:.6e} exceeds the block's comparator "
        f"{MAX_ABS_DIFF:.0e}"
    )

    assert nki_dispatch == 1, (
        f"one call must make exactly one dispatch; read {nki_dispatch}"
    )
    assert torch_fallback == 0

    unbounded = kda_gate_clamp_torch_oracle(
        g, a_log, bias, beta=GATE_SOFTPLUS_BETA, lower=float("-inf")
    )
    bit = int((unbounded < KDA_GATE_LOWER_BOUND).sum())
    linear = int(((g + bias) * GATE_SOFTPLUS_BETA > 20.0).sum())
    print(
        f"GATE_CASE|tolerance|dispatch={nki_dispatch}|declared=1"
        f"|fallback={torch_fallback}"
        f"|beta={GATE_SOFTPLUS_BETA}"
        f"|max_abs_diff={worst:.6e}"
        f"|comparator={MAX_ABS_DIFF:.0e}"
        f"|margin_ratio={MAX_ABS_DIFF / worst if worst else float('inf'):.2f}"
        f"|elements_over_comparator={int((diff > MAX_ABS_DIFF).sum())}"
        f"|bound_bites_on={bit}/{g.numel()}"
        f"|linear_branch_elements={linear}"
        f"|reference_min={reference.min().item():.6f}"
        f"|reference_max={reference.max().item():.6f}"
        f"|activation_min_unbounded={unbounded.min().item():.6f}",
        flush=True,
    )


def test_an_activation_exactly_at_the_lower_bound_is_returned_unchanged():
    """Boundary 1 of 2. Certifying component: the same ``wrap_nki`` seam.

    An input whose activation lands exactly on ``-5.0`` must come back as ``-5.0``.
    This is the inclusive side of the bound: ``min=`` keeps the endpoint, and a
    kernel using a strict comparison would move it.
    """
    _report(
        "boundary_at_bound",
        "kda_gate_clamp (the wrap_nki seam this block authors)",
    )
    _run_boundary_case("boundary_at_bound", KDA_GATE_LOWER_BOUND)


def test_an_activation_one_ulp_below_the_lower_bound_is_returned_clamped():
    """Boundary 2 of 2. Certifying component: the same ``wrap_nki`` seam.

    An input whose activation lands one float32 ULP BELOW ``-5.0`` must come back
    at exactly ``-5.0``. This is the item a kernel without the bound fails, and it
    fails on exact equality rather than on the tolerance -- one ULP here is 21
    times smaller than the tolerance, so the tolerance could not see it.
    """
    _report(
        "boundary_below_bound",
        "kda_gate_clamp (the wrap_nki seam this block authors)",
    )
    _run_boundary_case("boundary_below_bound", KDA_GATE_LOWER_BOUND - EPS)
