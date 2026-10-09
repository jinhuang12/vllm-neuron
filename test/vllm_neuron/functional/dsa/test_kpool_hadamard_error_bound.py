# SPDX-License-Identifier: Apache-2.0
"""How far each DSA Hadamard kernel may be from the exact result, and the tests that hold it there.

The tolerance is derived, not fitted: it is the worst case of the kernels' own steps. Each
fp32 add, multiply and PSUM accumulation is taken as faithfully rounded, so its relative
error is below ``EPS``. The Scalar Engine's ``exp`` and reciprocal are not that accurate;
their worst relative errors are measured on trn2 (``SCALAR_ENGINE_EXP_ERROR``,
``SCALAR_ENGINE_RECIPROCAL_ERROR``). With ``gamma(n) = n EPS / (1 - n EPS)``, per pool and
channel, over the ``P`` slots ``j``: ``z_j = score_j + ape_j``, ``w = softmax(z)`` and
``p = sum_j w_j k_j``.

1. Shift. ``z_j - max z`` comes from two rounded sums and is rounded itself, so it is off by
   at most ``tau = 2 EPS Z (2 + EPS)``, with ``Z = max_j |z_j|``.
2. Weights. ``exp`` of the shift is then off by a factor within ``1 +- eta``, with
   ``eta = e^tau (1 + exp_error) - 1``. A factor common to all slots cancels in the softmax,
   so the pooled value moves by at most ``2 eta / (1 - eta) * S``, with ``S = sum_j w_j |k_j|``.
3. Sums. The two ``P``-term sums add ``gamma(P) (1 + eta) / (1 - eta) * S``. The reciprocal
   of the weight sum and the multiply by it scale the result by a factor within ``1 +- mu``,
   with ``mu = (1 + reciprocal_error) (1 + EPS) / (1 - gamma(P - 1)) - 1``. In all:
   ``|p^ - p| <= F = (2 eta + gamma(P) (1 + eta)) / (1 - eta) * S * (1 + mu) + mu |p|``.
4. Rotation. ``H_128`` is all ``+-1``, so its 128 products are exact; their accumulation in any
   order adds ``gamma(127) * sum_i |p^_i|``, and the scale with its fp32 rounding ``gamma(2)``:
   ``E = sigma (1 + gamma(2)) (sum_i F_i + gamma(127) sum_i (|p_i| + F_i)) + gamma(2) |y|``.
5. Output. A bf16 result is rounded to nearest once more: half a bf16 ulp at ``|y| + E``.

The rotation kernel alone is steps 4 and 5 with ``F = 0`` and ``p = x``. f3a833f's kernels
take the same steps with a correctly rounded reciprocal and a 7-level butterfly in place of
the 127 accumulations, so the same bound covers them.

The two measured errors hold over the ranges they were measured on, which the bound
checks: every fp32 shift from ``EXP_LOW`` to 0, and every weight sum from 1 to
``RECIPROCAL_HIGH``. The same sweep found ``exp(0) == 1`` and no ``exp`` result above 1, so a
sum of ``P`` weights is in ``[1, P]``.

On the CPU simulator the engine functions are numpy's, so the tests here check the
kernels' arithmetic: its order, its fp32 intermediates, its single rounding. The hardware
benchmark (``test/hardware/benchmark_dsa_hadamard.py``) applies the same bound, with the
engine errors it measures in the same run, to the device's results.
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.functional.dsa.test_kpool_hadamard import (
    LNC2,
    _inputs,
    _pools_reaching_every_pool_branch,
    _rows,
    _rows_reaching_every_rotation_branch,
)
from vllm_neuron.functional.dsa import kpool_hadamard
from vllm_neuron.functional.dsa.kpool_hadamard import (
    HADAMARD_SCALE,
    dsa_hadamard128,
    dsa_kpool_hadamard,
    hadamard_matrix,
    kpool_hadamard_dispatch_counters,
    reset_kpool_hadamard_dispatch_counters,
)

EPS = torch.finfo(torch.float32).eps
"""The relative error of one faithfully rounded fp32 step: under one ulp at 1."""

SCALAR_ENGINE_EXP_ERROR = 1.2e-5
"""Measured on trn2: the largest ``|exp(x) - e^x| / e^x`` over every fp32 ``x`` from ``EXP_LOW``
to 0 (``engine_checks`` in the hardware benchmark), 1.1348e-5 at -4.75, rounded up."""

SCALAR_ENGINE_RECIPROCAL_ERROR = 1.2e-5
"""Measured on trn2: the largest ``|r(x) - 1/x| * x`` over every fp32 ``x`` from 1 to 4,
1.1984e-5, rounded up."""

EXP_LOW = -64.0
"""The lowest ``exp`` input the measurement covers."""

RECIPROCAL_HIGH = 4
"""The largest weight sum the measurement covers: one weight of at most 1 for each of 4 slots."""


def gamma(n: int) -> float:
    """The relative error bound of ``n`` faithfully rounded fp32 steps in a row."""
    return n * EPS / (1 - n * EPS)


def ulp(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """One ulp of ``dtype`` at each ``|value|``: ``eps * 2**(e-1)`` on ``[2**(e-1), 2**e)``."""
    info = torch.finfo(dtype)
    _, exponent = torch.frexp(value.abs().clamp(min=info.tiny))
    return torch.ldexp(torch.full_like(value, info.eps), exponent - 1)


def exact_rotation(x: torch.Tensor) -> torch.Tensor:
    """The rotation of ``x [rows, 128]`` in fp64, unrounded: the value a kernel rounds."""
    hadamard = hadamard_matrix(int(x.shape[-1]), dtype=torch.float64)
    return (x.double() @ hadamard.t()) * HADAMARD_SCALE


def exact_pooling(
    slot_k: torch.Tensor, slot_score: torch.Tensor, ape: torch.Tensor
) -> torch.Tensor:
    """The fused pooling of ``[pools, slots, 128]`` keys and scores in fp64, unrounded."""
    weights = torch.softmax(slot_score.double() + ape.double().unsqueeze(0), dim=1)
    return exact_rotation((weights * slot_k.double()).sum(dim=1))


def _rotated_bound(
    magnitude: torch.Tensor, error: torch.Tensor, exact: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Steps 4 and 5: the bound on each output, from the rotated rows' ``|p|``, ``F`` and ``y``."""
    terms = int(magnitude.shape[-1])
    accumulated = gamma(terms - 1) * (magnitude + error).sum(-1, keepdim=True)
    spread = error.sum(-1, keepdim=True) + accumulated
    bound = HADAMARD_SCALE * (1 + gamma(2)) * spread + gamma(2) * exact.abs()
    if out_dtype == torch.float32:
        return bound  # the output is the scale's own fp32 rounding, inside gamma(2)
    return bound + ulp(exact.abs() + bound, out_dtype) / 2


def rotation_error_bound(x: torch.Tensor) -> torch.Tensor:
    """The largest ``|out - exact_rotation(x)|`` the rotation kernel may return, per element."""
    magnitude = x.double().abs()
    return _rotated_bound(magnitude, torch.zeros_like(magnitude), exact_rotation(x), x.dtype)


def pooling_error_bound(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    exp_error: float = SCALAR_ENGINE_EXP_ERROR,
    reciprocal_error: float = SCALAR_ENGINE_RECIPROCAL_ERROR,
) -> torch.Tensor:
    """The largest ``|out - exact_pooling(...)|`` the pooling kernel may return, per element.

    Raises ``ValueError`` for inputs outside the ranges ``exp_error`` and ``reciprocal_error``
    were measured on.
    """
    pool_size = int(slot_k.shape[1])
    if pool_size > RECIPROCAL_HIGH:
        raise ValueError(
            f"{pool_size} slots: weight sums above {RECIPROCAL_HIGH} were not measured"
        )
    z = slot_score.double() + ape.double().unsqueeze(0)
    largest = z.abs().amax(dim=1)
    tau = 2 * EPS * largest * (2 + EPS)
    lowest_shift = (z - z.amax(dim=1, keepdim=True)).amin(dim=1) - tau
    if bool((lowest_shift < EXP_LOW).any()):
        raise ValueError(f"a shifted score below {EXP_LOW}: exp was not measured there")
    eta = torch.exp(tau) * (1 + exp_error) - 1
    mu = (1 + reciprocal_error) * (1 + EPS) / (1 - gamma(pool_size - 1)) - 1
    weights = torch.softmax(z, dim=1)
    keys = slot_k.double()
    spread = (weights * keys.abs()).sum(dim=1)
    pooled = (weights * keys).sum(dim=1)
    moved = (2 * eta + gamma(pool_size) * (1 + eta)) / (1 - eta) * spread
    error = moved * (1 + mu) + mu * pooled.abs()
    return _rotated_bound(pooled.abs(), error, exact_rotation(pooled), slot_k.dtype)


def error_over_bound(out: torch.Tensor, exact: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
    """``|out - exact| / bound`` per element: at most 1 wherever ``out`` is within its bound."""
    return (out.double() - exact).abs() / bound.clamp(min=torch.finfo(torch.float64).tiny)


# ---------------------------------------------------------------------------------------------
# The kernels against the bound
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lnc", [None, LNC2], ids=["grid1", "lnc2"])
@pytest.mark.parametrize("score_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_pooling_is_within_its_derived_bound(
    monkeypatch: pytest.MonkeyPatch, lnc, score_dtype: torch.dtype
) -> None:
    """Every pooled and rotated output, on every tiling branch, is within the bound."""
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    pools = _pools_reaching_every_pool_branch(kpool_hadamard._programs(2))
    slot_k, slot_score, ape = _inputs(pools, seed=501, score_dtype=score_dtype)
    reset_kpool_hadamard_dispatch_counters()
    got = dsa_kpool_hadamard(slot_k, slot_score, ape)
    assert kpool_hadamard_dispatch_counters() == (1, 0)
    ratio = error_over_bound(
        got, exact_pooling(slot_k, slot_score, ape), pooling_error_bound(slot_k, slot_score, ape)
    )
    assert float(ratio.max()) <= 1.0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_rotation_is_within_its_derived_bound(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    """Every rotated output under LNC2, on every tiling branch, is within the bound."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", LNC2)
    x = _rows(_rows_reaching_every_rotation_branch(kpool_hadamard._programs(2)), dtype, seed=502)
    reset_kpool_hadamard_dispatch_counters()
    got = dsa_hadamard128(x)
    assert kpool_hadamard_dispatch_counters() == (1, 0)
    ratio = error_over_bound(got, exact_rotation(x), rotation_error_bound(x))
    assert float(ratio.max()) <= 1.0


# ---------------------------------------------------------------------------------------------
# The bound itself
# ---------------------------------------------------------------------------------------------


def _served_pool_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 keys and scores, the served dtypes, for checks on the bound alone."""
    return _inputs(1024, seed=503, score_dtype=torch.bfloat16)


def test_the_bound_admits_the_correctly_rounded_pooling() -> None:
    """The exact result rounded once to bf16, the best any kernel can return, is within it."""
    slot_k, slot_score, ape = _served_pool_inputs()
    exact = exact_pooling(slot_k, slot_score, ape)
    error_bound = pooling_error_bound(slot_k, slot_score, ape)
    ratio = error_over_bound(exact.to(torch.bfloat16), exact, error_bound)
    assert float(ratio.max()) <= 1.0


def test_the_bound_rejects_one_ulp_off_wherever_the_value_is_at_least_its_row_rms() -> None:
    """One bf16 ulp further from the exact value than the rounded result fails the bound at
    every element as large as its row's rms, so the bound is tighter than one ulp there."""
    slot_k, slot_score, ape = _served_pool_inputs()
    exact = exact_pooling(slot_k, slot_score, ape)
    rounded = exact.to(torch.bfloat16).double()
    away = torch.where(rounded == exact, exact.sign(), (rounded - exact).sign())
    off = (rounded + away * ulp(rounded, torch.bfloat16)).to(torch.bfloat16)
    ratio = error_over_bound(off, exact, pooling_error_bound(slot_k, slot_score, ape))
    large = exact.abs() >= exact.pow(2).mean(dim=-1, keepdim=True).sqrt()
    assert bool(large.any())
    assert bool((ratio[large] > 1.0).all())


def test_the_bound_refuses_shifts_below_the_measured_exp_range() -> None:
    """A score more than ``-EXP_LOW`` below its pool's max leaves the measured ``exp`` range."""
    slot_k, slot_score, ape = _inputs(4, seed=504)
    slot_score[0, 0, 0] = EXP_LOW * 2
    with pytest.raises(ValueError, match="exp was not measured"):
        pooling_error_bound(slot_k, slot_score, ape)


def test_the_bound_refuses_more_slots_than_the_measured_reciprocal_range() -> None:
    """More than ``RECIPROCAL_HIGH`` slots put the weight sum past the measured reciprocal."""
    slot_k, slot_score, ape = _inputs(4, seed=505, pool_size=RECIPROCAL_HIGH + 1)
    with pytest.raises(ValueError, match="were not measured"):
        pooling_error_bound(slot_k, slot_score, ape)
