# SPDX-License-Identifier: Apache-2.0
"""Tests for ``attention_decode`` when more than one new token is decoded at once.

The oracle is written in this file from the documented contract and calls nothing in
the module except the mask builder production itself calls, so it cannot agree with
the implementation for the wrong reason. Each comparison also witnesses which path
the call took: ``attention_decode`` chooses between the NKI kernel and a torch
implementation from the environment and the shapes, so a green comparison against the
torch path would say nothing about the kernel.

The mask convention, read in sequential order: a prior slot at position ``p`` is
valid when ``p < min(pos_ids)``, and the last ``s_active`` slots carry the causal
staircase, slot ``k`` visible to query ``j`` when ``k <= j``. At the shapes below the
kernel resizes its cache block length to 1, so sequential order and the block-KV
layout the module consumes are the same thing here; at a longer context the two would
differ and the oracle would have to borrow the module's layout code.
"""

import math

import pytest
import torch

from vllm_neuron.functional.attention import attention_decode as AD
from vllm_neuron.functional.attention.attention_decode_mask import (
    gen_attention_decode_mask,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

DTYPE = torch.bfloat16
B = 1
# A multiple of 256: the kernel runs on the module's two-core grid and asserts that
# H / 128 is even, so H = 128 is admitted by the router and then refused by the
# kernel. Measured: 128 refused, 256 and 512 ran.
H = 256
D_HEAD = 128            # even and <= 128, per the module's docstring
Q_HEADS = 1             # derived: W_qkv.shape[1] // D_HEAD - 2 * KV_HEADS
KV_HEADS = 1            # inferred by the module from a 3D value cache
BLOCK_LEN = 128
NUM_BLOCKS_TOTAL = 2
BLOCK_IN_USE = 1        # block 1 of 2, so a gather that ignores the table reads wrong
NUM_BLOCKS_PER_SEQ = 1
S_CTX = NUM_BLOCKS_PER_SEQ * BLOCK_LEN
MIN_POS = 99            # prior slots 0..98 valid, 99..(S_CTX - s_active - 1) masked
S_ACTIVE_CASES = (1, 2, 4)      # inside the module's documented bound of 8

# The module's own tolerance for this kernel; this file does not widen it.
RTOL = 1e-2
ATOL = 1e-5

# The module globals the dispatch resolves at call time, captured at import so the
# spies can recognise them by OBJECT IDENTITY. Recognising by __name__ would be
# fragile: a jitted kernel object need not carry a usable one.
_KERNEL_FN = AD._torch_compatible_attention_block_tkg_kernel
_KERNEL_DCP_FN = AD._torch_compatible_attention_block_tkg_kernel_dcp

# The real callables, captured at IMPORT. A spy must call these and not whatever is
# bound to the attribute when it is installed: one test installs spies twice, and a
# spy that read the live attribute would wrap the previous spy, so one dispatch would
# add to two counters and every total would be wrong.
_REAL_WRAP_NKI = AD.wrap_nki
_REAL_TORCH_IMPL = AD._torch_attention_decode_impl


def _logical_mask(s_active):
    """The mask in plain sequential order: ``[S_CTX, s_active]``, 1 means visible."""
    mask = torch.zeros(S_CTX, s_active, dtype=torch.float32)
    n_prior = S_CTX - s_active
    for pos in range(n_prior):
        if pos < MIN_POS:
            mask[pos, :] = 1.0
    for k in range(s_active):
        for j in range(s_active):
            mask[n_prior + k, j] = 1.0 if k <= j else 0.0
    return mask


def _build_inputs(s_active):
    """Every tensor the call needs, at the shapes above, from a fixed seed.

    The mask is built here once and reused by every leg that consumes these inputs.
    ``gen_attention_decode_mask`` itself chooses between a kernel and a torch build
    from the same environment gate the kernel-versus-torch test flips, so a mask built
    inside a flipped region would differ between the two legs.

    The value cache is drawn from [0.5, 1.5] rather than from a signed distribution:
    attention output is a weighted average of value rows, so with signed values an
    output element can cancel toward zero, and there ``atol`` compares rounding noise
    instead of the quantity under test.
    """
    gen = torch.Generator().manual_seed(1000 + s_active)
    x = torch.randn(B, s_active, H, generator=gen).to(DTYPE)
    w_qkv = (
        torch.randn(H, D_HEAD * (Q_HEADS + 2 * KV_HEADS), generator=gen) / math.sqrt(H)
    ).to(DTYPE)
    k_cache = torch.randn(NUM_BLOCKS_TOTAL, BLOCK_LEN, D_HEAD, generator=gen).to(DTYPE)
    v_cache = (
        torch.rand(NUM_BLOCKS_TOTAL, BLOCK_LEN, D_HEAD, generator=gen) + 0.5
    ).to(DTYPE)
    table = torch.tensor([[BLOCK_IN_USE]], dtype=torch.int32)
    pos_ids = torch.arange(
        MIN_POS, MIN_POS + s_active, dtype=torch.float32
    ).view(1, B * s_active)
    mask = gen_attention_decode_mask(
        pos_ids=pos_ids,
        bs=B,
        q_head=Q_HEADS,
        s_active=s_active,
        s_prior=S_CTX,
        block_len=BLOCK_LEN,
    )
    return {
        "X": x,
        "W_qkv": w_qkv,
        "K_cache": k_cache,
        "V_cache": v_cache,
        "active_blocks_table": table,
        "pos_ids": pos_ids,
        "attention_mask": mask,
    }


def _call(inp):
    """The one entry point under test, called exactly as production calls it."""
    return AD.attention_decode(
        X=inp["X"],
        W_qkv=inp["W_qkv"],
        active_blocks_table=inp["active_blocks_table"],
        K_cache=inp["K_cache"],
        V_cache=inp["V_cache"],
        attention_mask=inp["attention_mask"],
    )


def _oracle(inp, s_active):
    """Attention computed from the documented contract, in this file, from scratch.

    Projection uses the bfloat16 weights, because that is what any implementation
    would do with them. Everything after it is float32: a reference that reproduced
    the implementation's own rounding would not be a reference.
    """
    x = inp["X"]
    qkv = (x.reshape(B * s_active, H) @ inp["W_qkv"]).reshape(B, s_active, -1)
    q_end = Q_HEADS * D_HEAD
    k_end = q_end + KV_HEADS * D_HEAD
    q = qkv[..., :q_end].reshape(B, s_active, Q_HEADS, D_HEAD).transpose(1, 2)
    k = qkv[..., q_end:k_end].reshape(B, s_active, KV_HEADS, D_HEAD).transpose(1, 2)
    v = qkv[..., k_end:].reshape(B, s_active, KV_HEADS, D_HEAD).transpose(1, 2)

    # The gathered window is the blocks the table names, concatenated in table order,
    # with the new tokens written over the last s_active slots.
    k_win = inp["K_cache"][BLOCK_IN_USE].to(torch.float32).clone().view(1, 1, S_CTX, D_HEAD)
    v_win = inp["V_cache"][BLOCK_IN_USE].to(torch.float32).clone().view(1, 1, S_CTX, D_HEAD)
    k_win[:, :, -s_active:, :] = k.to(torch.float32)
    v_win[:, :, -s_active:, :] = v.to(torch.float32)

    scale = D_HEAD**-0.5
    scores = (q.to(torch.float32) @ k_win.transpose(-2, -1)) * scale
    keep = _logical_mask(s_active).transpose(0, 1).view(1, 1, s_active, S_CTX)
    scores = scores.masked_fill(keep == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = weights @ v_win                       # [B, Q_HEADS, s_active, D_HEAD]
    # Without W_out the module returns the transposed layout.
    return out.transpose(-2, -1).to(DTYPE)


class _Spy:
    """Dispatch counts for one call or one group of calls."""

    def __init__(self):
        self.kernel = 0
        self.kernel_dcp = 0
        self.fallback = 0
        self.unrecognised = 0

    def __repr__(self):
        return (
            f"kernel={self.kernel} kernel_dcp={self.kernel_dcp} "
            f"fallback={self.fallback} unrecognised={self.unrecognised}"
        )


def _install_spies(monkeypatch):
    """Count dispatches out of the seam without replacing anything that computes.

    The kernel side wraps ``wrap_nki``, records which function object was passed and
    hands back the real wrapper; substituting a python stand-in for a jitted kernel
    would change what actually runs. Anything ``wrap_nki`` is handed that this file
    does not recognise is counted separately, so an unrecognised dispatch cannot hide
    inside a passing total.
    """
    spy = _Spy()

    def wrap(fn, *args, **kwargs):
        if fn is _KERNEL_FN:
            spy.kernel += 1
        elif fn is _KERNEL_DCP_FN:
            spy.kernel_dcp += 1
        else:
            spy.unrecognised += 1
        return _REAL_WRAP_NKI(fn, *args, **kwargs)

    def fallback(*args, **kwargs):
        spy.fallback += 1
        return _REAL_TORCH_IMPL(*args, **kwargs)

    monkeypatch.setattr(AD, "wrap_nki", wrap)
    monkeypatch.setattr(AD, "_torch_attention_decode_impl", fallback)
    return spy


def test_the_kernel_gate_follows_the_environment_at_call_time(monkeypatch):
    """``can_run_kernel`` reads the disable variable on every call, not at import."""
    assert can_run_kernel(torch.zeros(1)) is True

    monkeypatch.setenv("VLLM_NEURON_DISABLE_NKI_KERNELS", "1")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not follow the environment at call time"
    )

    monkeypatch.setenv("VLLM_NEURON_DISABLE_NKI_KERNELS", "0")
    assert can_run_kernel(torch.zeros(1)) is True, "the flip did not reverse"


def test_the_router_admits_these_shapes_and_refuses_attention_data_parallelism():
    """``_can_use_attention_block_kernel`` answers True here and False at dp of 2.

    None of the router's refusing conditions fires at these shapes: the hidden
    dimension is a multiple of 128, the head dimension is even, the cache is 3D so the
    key/value head count infers to one and the per-head-table rule is vacuous.
    """
    inp = _build_inputs(1)
    gate = AD._can_use_attention_block_kernel(
        X=inp["X"],
        V_cache=inp["V_cache"],
        active_blocks_table=inp["active_blocks_table"],
        attention_dp=1,
    )
    assert can_run_kernel(inp["X"]) is True
    assert gate is True

    refused = AD._can_use_attention_block_kernel(
        X=inp["X"],
        V_cache=inp["V_cache"],
        active_blocks_table=inp["active_blocks_table"],
        attention_dp=2,
    )
    assert refused is False, "the router admitted a case it must refuse"


@pytest.mark.parametrize("s_active", S_ACTIVE_CASES)
def test_the_generated_mask_matches_the_causal_convention(s_active):
    """The generated mask equals the mask derived in this file, element for element.

    Exact equality is the right bound: the mask holds only zeros and ones.
    """
    inp = _build_inputs(s_active)
    generated = inp["attention_mask"][:, 0, 0, :].to(torch.float32)
    expected = _logical_mask(s_active)
    # Arithmetic, not a recorded number: MIN_POS valid prior slots per query, plus one
    # staircase entry for each pair with k <= j.
    expected_sum = MIN_POS * s_active + s_active * (s_active + 1) // 2
    assert float(expected.sum()) == expected_sum
    differing = int((generated != expected).sum())
    assert differing == 0, (
        f"{differing} of {generated.numel()} mask elements differ from the convention"
    )


@pytest.mark.parametrize("s_active", S_ACTIVE_CASES)
def test_the_kernel_matches_an_independent_oracle(s_active, monkeypatch):
    """The kernel result matches the in-file oracle at the module's tolerance.

    The spies ride along so each case also witnesses that the number it compared came
    out of the kernel rather than out of the torch fallback.
    """
    spy = _install_spies(monkeypatch)
    inp = _build_inputs(s_active)
    got = _call(inp)[0]
    expected = _oracle(inp, s_active)
    assert spy.kernel == 1, f"this case did not reach the kernel: {spy}"
    assert spy.fallback == 0
    assert spy.kernel_dcp == 0
    assert spy.unrecognised == 0
    assert tuple(got.shape) == (B, Q_HEADS, D_HEAD, s_active)
    torch.testing.assert_close(
        got.to(torch.float32), expected.to(torch.float32), rtol=RTOL, atol=ATOL
    )


def test_disabling_kernels_routes_the_call_through_the_torch_path(monkeypatch):
    """With kernels disabled the same call takes the module's torch implementation."""
    spy = _install_spies(monkeypatch)
    inp = _build_inputs(1)
    monkeypatch.setenv("VLLM_NEURON_DISABLE_NKI_KERNELS", "1")
    out = _call(inp)[0]
    assert spy.fallback == 1, f"the torch path was not taken: {spy}"
    assert spy.kernel == 0
    assert spy.unrecognised == 0
    assert tuple(out.shape) == (B, Q_HEADS, D_HEAD, 1)


def test_the_kernel_and_the_torch_path_agree_bit_for_bit(monkeypatch):
    """At one new token the two paths agree to a maximum absolute difference of zero.

    The module's docstring promises it for a mask built by
    ``gen_attention_decode_mask``. Both legs consume the same tensors, mask included;
    only the environment gate moves between them, and the spies witness which path
    each leg took.
    """
    inp = _build_inputs(1)

    spy_kernel_leg = _install_spies(monkeypatch)
    kernel_out = _call(inp)[0]
    assert spy_kernel_leg.kernel == 1 and spy_kernel_leg.fallback == 0

    monkeypatch.setenv("VLLM_NEURON_DISABLE_NKI_KERNELS", "1")
    spy_torch_leg = _install_spies(monkeypatch)
    torch_out = _call(inp)[0]
    assert spy_torch_leg.fallback == 1 and spy_torch_leg.kernel == 0

    a = kernel_out.to(torch.float32)
    b = torch_out.to(torch.float32)
    max_abs = float((a - b).abs().max())
    assert max_abs == 0.0, (
        f"kernel and torch paths disagree: max abs diff {max_abs:.6e} over "
        f"{int((a != b).sum())} of {a.numel()} elements"
    )
