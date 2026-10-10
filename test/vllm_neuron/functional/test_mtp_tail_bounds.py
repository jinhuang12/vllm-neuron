# SPDX-License-Identifier: Apache-2.0
"""The DERIVED error bounds the MTP tail kernels are held to (no chosen tolerances).

Both kernels compute an RMSNorm in fp32 from bf16 operands, round it to bf16, and
multiply the bf16 rows by a bf16 weight on the tensor engine with fp32 accumulation,
rounding the result to bf16. Every term below is a property of that arithmetic:

* :func:`bf16_step`: one bf16 unit in the last place at an element's magnitude.
* :func:`norm_rtol`: the kernel's fp32 norm value differs from the exact one by at
  most ``(H + 8) * 2**-24`` relative -- ``H`` fp32 additions of exact bf16 squares in
  some order (the sequential worst case), the GpSimd ``rsqrt`` (a few ulp) and two
  fp32 multiplies. After the bf16 rounding the kernel's element equals the exact
  value's bf16 rounding unless the exact value lies within that relative distance of
  a rounding midpoint (:func:`near_midpoint`); such an element may land one bf16 step
  away. The reference's own fp32 route has the same property, so it is held to the
  same bound.
* :func:`gemv_bound`: for ``out = bf16(sum_j w_j x_j)`` over bf16 ``w`` and ``x``
  (products exact in fp32), ``|out - exact| <= own + flip + acc`` with ``own`` half a
  bf16 step of ``out``, ``flip`` the one-step flips the eligible ``x_j`` may carry
  (``sum_j |w_j| step(x_j)`` over the near-midpoint ``j``) and ``acc`` the
  accumulation order's worst case, ``K * 2**-24 * sum_j |w_j x_j|`` over the ``K``
  contraction terms.
"""

from __future__ import annotations

import torch


def bf16_step(x: torch.Tensor) -> torch.Tensor:
    """One bf16 unit in the last place at each element's magnitude (7 stored bits)."""
    mag = x.detach().abs().double().clamp_min(2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7.0)


def norm_rtol(hidden: int) -> float:
    """Relative error of a kernel's fp32 RMSNorm value before its bf16 rounding."""
    return (hidden + 8) * 2.0 ** -24


def rms64(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """The exact (float64) RMSNorm of bf16 ``x`` with bf16 ``gain``."""
    x = x.double()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * gain.double()


def near_midpoint(exact: torch.Tensor, hidden: int) -> torch.Tensor:
    """Elements whose exact value is within ``norm_rtol`` of a bf16 rounding midpoint."""
    rounded = exact.to(torch.bfloat16).double()
    step = bf16_step(rounded)
    return (exact - rounded).abs() >= step / 2 - norm_rtol(hidden) * exact.abs()


def assert_normed_within_one_flip(got: torch.Tensor, exact: torch.Tensor, hidden: int,
                                  what: str) -> None:
    """``got`` (bf16) equals ``bf16(exact)`` except, by one step, where eligible."""
    rounded = exact.to(torch.bfloat16)
    diff = (got.double() - rounded.double()).abs()
    allowed = near_midpoint(exact, hidden).double() * bf16_step(rounded)
    assert bool((diff <= allowed).all()), (
        f"{what}: {int((diff > allowed).sum())} elements differ from the exact norm's "
        f"rounding where the derived term allows no flip; max |diff| {float(diff.max()):.3e}")


def gemv_bound(out: torch.Tensor, x_exact: torch.Tensor, weight: torch.Tensor,
               hidden: int) -> torch.Tensor:
    """``own + flip + acc`` per element of ``out = bf16(bf16(x_exact) @ weight.T)``.

    ``x_exact`` ``[B, K]`` fp64 is the exact pre-rounding value of the rows (the
    norms' outputs); ``weight`` ``[N, K]`` bf16; ``hidden`` the norms' length (their
    relative error); the accumulation term counts ``K`` contraction terms.
    """
    x_bf = x_exact.to(torch.bfloat16).double()
    w_abs = weight.double().abs()
    flip = (near_midpoint(x_exact, hidden).double() * bf16_step(x_bf)) @ w_abs.t()
    acc = x_exact.shape[-1] * 2.0 ** -24 * (x_bf.abs() @ w_abs.t())
    return bf16_step(out) / 2 + flip + acc


def gemv_bound_rounded_rows(exact: torch.Tensor, x_bf: torch.Tensor, weight: torch.Tensor
                            ) -> torch.Tensor:
    """``own + acc`` for ``bf16(x_bf @ weight.T)`` against its exact value (no flips).

    ``exact`` ``[B, N]`` fp64 is ``x_bf @ weight.T``; the kernel rounds a value within
    ``acc`` of it, so ``own`` is half a bf16 step at the magnitude ``|exact| + acc``
    (the binade the rounded value can at most reach).
    """
    w_abs = weight.double().abs()
    acc = x_bf.shape[-1] * 2.0 ** -24 * (x_bf.double().abs() @ w_abs.t())
    return bf16_step(exact.abs() + acc) / 2 + acc
