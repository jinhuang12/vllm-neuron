# SPDX-License-Identifier: Apache-2.0
"""Fused NKI RMSNorm against the 5938748 torch RMSNorm at decode rows.

The 5938748 path is the body every decoder norm site had at that commit
(``Glm5NextModel._rms_norm``, both layer ``_input_norm`` methods)::

    x = h.to(fp32); var = x.pow(2).mean(-1, keepdim=True)
    out = (x * rsqrt(var + eps) * gain.to(fp32)).to(h.dtype)

Tolerance: one bf16 rounding step. Both paths compute in fp32 and differ only in
the order of the 4096-term sum of squares (fp32, ~1e-7 relative), so an output
can land on the other side of a bf16 rounding boundary:
``|got - ref| <= 2**-7 * |ref|`` (one bf16 ulp) elementwise, and at most 1% of
elements differ.
"""

from __future__ import annotations

import importlib

import pytest
import torch

norm = importlib.import_module("vllm_neuron.functional.norm")

HIDDEN = 4096
EPS = 1e-5


@pytest.fixture(autouse=True)
def _simulator(monkeypatch, tmp_path):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    # The NKI driver writes compile artifacts into the working directory.
    monkeypatch.chdir(tmp_path)


def path_5938748(hidden_states, gain, eps):
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    normed = normed * gain.to(torch.float32)
    return normed.to(hidden_states.dtype)


def _operands(tokens, seed):
    g = torch.Generator().manual_seed(seed)
    # Decode residual streams: per-row scale varies by orders of magnitude.
    row_scale = torch.logspace(-1, 2, tokens).reshape(tokens, 1)
    x = (torch.randn((tokens, HIDDEN), generator=g) * row_scale).to(torch.bfloat16)
    gain = (torch.rand((HIDDEN,), generator=g) * 2 + 0.05).to(torch.bfloat16)
    return x, gain


def _assert_one_rounding(got, ref):
    assert got.shape == ref.shape and got.dtype == ref.dtype
    diff = (got.float() - ref.float()).abs()
    assert (diff <= 2.0 ** -7 * ref.float().abs() + 1e-30).all(), diff.max()
    assert (diff > 0).float().mean().item() <= 0.01


@pytest.mark.parametrize("tokens", [1, 4, 64, 128])
def test_fused_norm_matches_the_5938748_path(tokens):
    x, gain = _operands(tokens, seed=tokens)
    ref = path_5938748(x, gain, EPS)
    norm.reset_norm_dispatch_counters()
    got = norm.rms_norm(x, gain, EPS)
    assert norm.norm_dispatch_counters() == (1, 0)
    _assert_one_rounding(got, ref)


def test_fused_norm_accepts_a_2d_gain():
    x, gain = _operands(2, seed=9)
    ref = path_5938748(x, gain.reshape(1, HIDDEN), EPS)
    got = norm.rms_norm(x, gain.reshape(1, HIDDEN), EPS)
    _assert_one_rounding(got, ref)


def test_eps_reaches_the_kernel():
    # An all-zero row normalises to exactly zero; a tiny row is eps-dominated.
    x = torch.zeros((2, HIDDEN), dtype=torch.bfloat16)
    x[1] = 1e-4
    gain = torch.ones((HIDDEN,), dtype=torch.bfloat16)
    for eps in (1e-5, 1e-2):
        got = norm.rms_norm(x, gain, eps)
        ref = path_5938748(x, gain, eps)
        assert torch.equal(got[0], torch.zeros_like(got[0]))
        _assert_one_rounding(got[1:], ref[1:])


def test_long_prefill_keeps_the_torch_path():
    x, gain = _operands(129, seed=4)
    norm.reset_norm_dispatch_counters()
    got = norm.rms_norm(x, gain, EPS)
    assert norm.norm_dispatch_counters() == (0, 1)
    assert torch.equal(got, path_5938748(x, gain, EPS))
