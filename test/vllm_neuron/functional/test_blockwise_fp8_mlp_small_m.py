# SPDX-License-Identifier: Apache-2.0
"""Fused small-M blockwise-fp8 SwiGLU MLP against the 5938748 three-call path.

The 5938748 path is the shared expert / dense MLP as built at that commit: three
``blockwise_fp8_mm`` calls (the snapshot in ``test/hardware/baselines``) with the
clamp, ``silu``, product and bf16 cast in torch between them. The fused kernel
must agree with it at B in {1, 4} (and at chunked B) on the real decode shapes:
hidden 4096, shared-expert intermediate 128 per rank (32 padded to one block),
dense-MLP intermediate 256 per rank (192 padded to two blocks).

Tolerance: ``|got - ref| <= 2e-3 * max|ref|`` elementwise. The kernel folds the
per-block fp32 scale into the activation as an exact-to-2**-17 bf16 hi/lo pair,
so gate/up agree with the fp32 reference to ~1e-5 relative; the remaining
difference is the bf16 rounding of the SwiGLU product (one bf16 ulp, 2**-8, on
a few elements whose fp32 value sits on a rounding boundary).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.nn.functional import silu

# The package re-exports the function under the module's name, so import the
# module itself.
bw = importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")

BASELINE_DIR = (
    Path(__file__).resolve().parents[2] / "hardware" / "baselines" / "dense_5938748"
)
HIDDEN = 4096
SWIGLU_LIMIT = 10.0
#: ``|got - ref| <= TOLERANCE * max|ref|``, see module docstring.
TOLERANCE = 2e-3


@pytest.fixture(autouse=True)
def _simulator(monkeypatch, tmp_path):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    # The NKI driver writes compile artifacts into the working directory.
    monkeypatch.chdir(tmp_path)


def _baseline():
    name = "_dense_5938748_blockwise_fp8_mm"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, BASELINE_DIR / "blockwise_fp8_mm.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def path_5938748(x, gate_w, up_w, down_w, gate_s, up_s, down_s, limit):
    """``Glm5NextSharedExperts.shared_expert_mm`` at 5938748, for M < 128."""
    base = _baseline()
    gate = base.blockwise_fp8_mm(x, gate_w, gate_s)
    up = base.blockwise_fp8_mm(x, up_w, up_s)
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    activated = silu(gate) * up
    return base.blockwise_fp8_mm(activated.to(x.dtype), down_w, down_s)


def _operands(tokens: int, intermediate: int, seed: int):
    """Decode-like operands: unit activations, fp8 weights, per-block scales.

    The scales are distinct per block and not powers of two, so a scale applied
    to the wrong block (or the prefill kernel's coarser grid) cannot agree. They
    are sized so that gate and up leave the ``[-L, L]`` box on some elements,
    which exercises both clamps.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((tokens, HIDDEN), generator=g).to(torch.bfloat16)

    def weight(rows, cols):
        raw = torch.randn((rows, cols), generator=g) * 64
        return raw.clamp(-240, 240).to(torch.float8_e4m3fn)

    def scale(rows, cols, magnitude):
        grid = torch.rand((rows // 128, cols // 128), generator=g) + 0.37
        return (grid * magnitude).to(torch.float32)

    gate_w = weight(HIDDEN, intermediate)
    up_w = weight(HIDDEN, intermediate)
    down_w = weight(intermediate, HIDDEN)
    gate_s = scale(HIDDEN, intermediate, 3e-3)
    up_s = scale(HIDDEN, intermediate, 3e-3)
    down_s = scale(intermediate, HIDDEN, 1e-3)
    return x, gate_w, up_w, down_w, gate_s, up_s, down_s


def _assert_agrees(got, ref):
    assert got.shape == ref.shape
    assert got.dtype == torch.float32
    assert torch.isfinite(got).all()
    bound = TOLERANCE * ref.abs().max().item()
    worst = (got - ref).abs().max().item()
    assert worst <= bound, f"max|got-ref|={worst:.3e} > {bound:.3e}"


@pytest.mark.parametrize("intermediate", [128, 256], ids=["shared", "dense"])
@pytest.mark.parametrize("tokens", [1, 4])
def test_fused_mlp_matches_the_5938748_path(tokens, intermediate):
    ops = _operands(tokens, intermediate, seed=100 + tokens + intermediate)
    ref = path_5938748(*ops, SWIGLU_LIMIT)
    bw.reset_mlp_dispatch_counters()
    bw.reset_dispatch_counters()
    got = bw.blockwise_fp8_mlp(*ops, swiglu_limit=SWIGLU_LIMIT)
    # The fused kernel ran once; neither the torch fallback nor the three-call
    # path was entered.
    assert bw.mlp_dispatch_counters() == (1, 0)
    assert bw.dispatch_counters() == (0, 0)
    _assert_agrees(got, ref)
    # Both clamps were active somewhere, so the test exercises them.
    x, gate_w, up_w, _, gate_s, up_s, _ = ops
    gate = bw.blockwise_fp8_mm_torch_oracle(x, gate_w, gate_s)
    up = bw.blockwise_fp8_mm_torch_oracle(x, up_w, up_s)
    assert (gate > SWIGLU_LIMIT).any() and (up.abs() > SWIGLU_LIMIT).any()


@pytest.mark.parametrize("tokens", [33, 64, 127])
def test_fused_mlp_chunks_tokens_without_padding(tokens):
    ops = _operands(tokens, 128, seed=7 + tokens)
    ref = path_5938748(*ops, SWIGLU_LIMIT)
    bw.reset_mlp_dispatch_counters()
    got = bw.blockwise_fp8_mlp(*ops, swiglu_limit=SWIGLU_LIMIT)
    assert bw.mlp_dispatch_counters() == (1, 0)
    _assert_agrees(got, ref)


def test_fused_mlp_rows_are_independent():
    # Token rows must not mix: row r of a 4-row call equals the 1-row call on r.
    # Not bitwise: the simulator's fp32 matmul may sum in a shape-dependent
    # order. A mixing bug moves rows by O(max|row|), far above this bound.
    ops = _operands(4, 256, seed=55)
    x, *rest = ops
    batch = bw.blockwise_fp8_mlp(x, *rest, swiglu_limit=SWIGLU_LIMIT)
    for row in range(4):
        one = bw.blockwise_fp8_mlp(x[row:row + 1], *rest, swiglu_limit=SWIGLU_LIMIT)
        _assert_agrees(batch[row:row + 1], one)


def test_full_tiles_take_the_three_call_path():
    # M >= 128 is prefill; the fused small-M kernel refuses it and the
    # dispatcher keeps the 5938748 three-call arithmetic.
    ops = _operands(128, 128, seed=3)
    bw.reset_mlp_dispatch_counters()
    bw.reset_dispatch_counters()
    got = bw.blockwise_fp8_mlp(*ops, swiglu_limit=SWIGLU_LIMIT)
    # Three blockwise_fp8_mm kernel dispatches; not a torch fallback of the MLP.
    assert bw.mlp_dispatch_counters() == (0, 0)
    assert bw.dispatch_counters() == (3, 0)
    ref = path_5938748(*ops, SWIGLU_LIMIT)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_intermediates_that_overflow_sbuf_take_the_three_call_path():
    # The fused kernel keeps all three weights in SBUF (96 B per partition per
    # intermediate column at H=4096); a wide per-rank intermediate (low TP) does
    # not fit, so the dispatcher must keep the three-call route.
    assert bw.fused_mlp_admissible(1, HIDDEN, bw.MLP_MAX_INTERMEDIATE)
    assert not bw.fused_mlp_admissible(1, HIDDEN, bw.MLP_MAX_INTERMEDIATE + 128)
    ops = _operands(1, bw.MLP_MAX_INTERMEDIATE + 128, seed=11)
    bw.reset_mlp_dispatch_counters()
    bw.reset_dispatch_counters()
    got = bw.blockwise_fp8_mlp(*ops, swiglu_limit=SWIGLU_LIMIT)
    assert bw.mlp_dispatch_counters() == (0, 0)
    assert bw.dispatch_counters() == (3, 0)
    _assert_agrees(got, path_5938748(*ops, SWIGLU_LIMIT))


def test_without_nki_the_mlp_counts_one_torch_fallback(monkeypatch):
    # The (nki_dispatch, torch_fallback) contract the model's seam registry reads.
    monkeypatch.delenv("NKI_SIMULATOR")
    ops = _operands(1, 128, seed=12)
    bw.reset_mlp_dispatch_counters()
    bw.reset_dispatch_counters()
    got = bw.blockwise_fp8_mlp(*ops, swiglu_limit=SWIGLU_LIMIT)
    assert bw.mlp_dispatch_counters() == (0, 1)
    assert bw.dispatch_counters() == (0, 3)
    _assert_agrees(got, path_5938748(*ops, SWIGLU_LIMIT))
