# SPDX-License-Identifier: Apache-2.0
"""Tests for the KDA gate kernel.

Each case compares :func:`kda_gate_clamp` against a torch reference of the same
function, authored here so the module under test contributes only one side of the
comparison, and checks that the NKI path is the one that ran.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional.kda.gate_clamp import (
    can_run_gate_clamp,
    gate_clamp_dispatch_counters,
    kda_gate_clamp,
    reset_gate_clamp_dispatch_counters,
)

RTOL = 1e-2
ATOL = 1e-5

#: The checkpoint's ``gate_lower_bound``, from ``Glm5NextTextConfig``'s
#: ``linear_attn_config``. Written as a literal rather than imported, because
#: ``functional/kda/`` does not depend on ``model/glm5_next/`` and this test must
#: not be what creates that dependency.
GATE_LOWER_BOUND = -5.0

#: Case 1's geometry. Both axes pass through a transpose inside the kernel, and the
#: two extents are unequal so an orientation error cannot hide in a square tile.
TOKENS = 8
KDIM = 16

#: Case 1's gate input scale. At this scale the sigmoid spans from near-saturated
#: to near-linear across the tile.
GATE_SCALE = 5.0

#: Case 1's decay exponent, non-zero so that a kernel dropping ``exp(a_log)``
#: entirely cannot pass: at ``a_log = 0`` the scale is exactly 1.
CASE1_A_LOG = 0.35

SEED = 20260903

#: Case 2's per-channel pre-activation ladder, one value per key channel. Signed and
#: symmetric in magnitude, and it includes ``0.0`` and ``+-ln 4``.
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

#: Case 2's per-token scale ladder. The negative last entry flips the signed ladder,
#: and the largest product ``2.0 * 15.0`` is deep into saturation on both sides.
CASE2_ROW_SCALE = (1.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0, -1.0)


def _reference(
    g: torch.Tensor,
    a_log: torch.Tensor,
    lower: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``lower * sigmoid(exp(a_log) * (g + bias))`` in torch.

    The module under test carries no torch path, so this is the only spelling of
    the reference and the two cannot drift apart.
    """
    pre = g.to(torch.float32)
    if bias is not None:
        pre = pre + bias.reshape(1, -1).to(torch.float32)
    decay_rate = torch.exp(a_log.reshape(()).to(torch.float32))
    return lower * torch.sigmoid(decay_rate * pre)


def _case1_inputs():
    """A random gate tile, a per-channel bias, and a non-zero ``a_log``.

    The bias carries a different value per key channel, which is what makes the
    kernel's two internal transposes falsifiable: a kernel that mixed up its axes
    would bias the wrong channels and miss by orders of magnitude. A zero or scalar
    bias would let an orientation error pass unseen.
    """
    gen = torch.Generator().manual_seed(SEED)
    g = torch.randn((TOKENS, KDIM), generator=gen, dtype=torch.float32) * GATE_SCALE
    bias = torch.randn((KDIM,), generator=gen, dtype=torch.float32) * 0.5
    a_log = torch.tensor(CASE1_A_LOG, dtype=torch.float32)
    return g, a_log, bias


def _case2_inputs():
    """A saturation ladder, no bias, and ``a_log = 0``.

    With no bias every entry of the tile is exactly the declared product, and
    ``a_log = 0`` makes ``exp(a_log)`` exactly 1, so the ladder's numbers are the
    pre-activation's own numbers.
    """
    base = torch.tensor(CASE2_BASE, dtype=torch.float32)
    scale = torch.tensor(CASE2_ROW_SCALE, dtype=torch.float32)
    g = scale.reshape(-1, 1) * base.reshape(1, -1)
    a_log = torch.zeros((), dtype=torch.float32)
    return g, a_log


def _one_call(g, a_log, bias):
    """Reset the counters, make exactly one call, and return the result and counters."""
    reset_gate_clamp_dispatch_counters()
    got = kda_gate_clamp(g, a_log, bias=bias, lower=GATE_LOWER_BOUND)
    return got, gate_clamp_dispatch_counters()


def test_the_gate_matches_the_reference_on_a_mixed_magnitude_case():
    """A random tile, a per-channel bias and a non-zero ``a_log`` match the reference."""
    g, a_log, bias = _case1_inputs()

    assert can_run_gate_clamp(g, TOKENS, KDIM) is True, (
        "the NKI path must be available; a False here would mean the kernel was "
        "never entered and the counter readings below would be vacuous"
    )

    reference = _reference(g, a_log, GATE_LOWER_BOUND, bias)
    got, (nki_dispatch, torch_fallback) = _one_call(g, a_log, bias)

    assert tuple(got.shape) == (TOKENS, KDIM), (
        f"the result must keep the [T, D] boundary orientation "
        f"{(TOKENS, KDIM)}, got {tuple(got.shape)}"
    )

    torch.testing.assert_close(got, reference, rtol=RTOL, atol=ATOL)

    assert nki_dispatch == 1, (
        f"one call must make exactly one dispatch; read {nki_dispatch}"
    )
    assert torch_fallback == 0, (
        f"the module carries no torch path, so this counter can only be 0; read "
        f"{torch_fallback}"
    )


def test_the_gate_matches_the_reference_through_saturation_and_at_zero():
    """The gate matches the reference at ``g = 0`` and deep into saturation both ways.

    The sigmoid saturates toward the bound as the pre-activation grows, so the
    result must stay finite and still agree at large ``|g|`` in both signs.
    """
    g, a_log = _case2_inputs()

    assert can_run_gate_clamp(g, TOKENS, KDIM) is True, (
        "the NKI path must be available; a False here would mean the kernel was "
        "never entered and the counter readings below would be vacuous"
    )

    reference = _reference(g, a_log, GATE_LOWER_BOUND, None)
    got, (nki_dispatch, torch_fallback) = _one_call(g, a_log, None)

    assert tuple(got.shape) == (TOKENS, KDIM), (
        f"the result must keep the [T, D] boundary orientation "
        f"{(TOKENS, KDIM)}, got {tuple(got.shape)}"
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
        f"the module carries no torch path, so this counter can only be 0; read "
        f"{torch_fallback}"
    )
