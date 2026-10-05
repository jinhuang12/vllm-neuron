# SPDX-License-Identifier: Apache-2.0
"""The token-axis routed-expert decode kernel against the 5938748 expert path.

Old side: the 5938748 packed branch of ``block_quant_expert_mm`` (local gather,
``build_blockwise_mapping``, ``fused_fp8_experts`` -- the compact kernel at
T=1 under LNC=2, the general kernel at T=4 -- and the fp32 token-gather
combine), from the snapshot under ``test/hardware/baselines/moe_5938748``.
New side: ``fused_fp8.fused_fp8_decode_experts`` on the global router
output, one launch, no mapping and no combine.

Declared tolerances, on the fp32 combined output ``[T, H]``:
``|new - old| <= FP32_ATOL_REL * max|old| + FP32_RTOL * |old|`` with
``FP32_RTOL = 1e-5`` and ``FP32_ATOL_REL = 1e-6``. The two paths take the same
fp32 products and the same bf16 SwiGLU activation; they sum the per-tile scaled
partials, the two I halves and the experts in different orders, so they differ
by fp32 round-off (measured at T=1 and T=4: max 1.1e-7 of max|old|). On the bf16
output the bound is one bf16 rounding step (``rtol=2**-7``) plus the same floor
(measured: all equal at T=1, one element one step apart at T=4). The torch
oracle sums in einsum order and uses torch's sigmoid; it is held to
``ORACLE_RTOL = 1e-3`` and ``ORACLE_ATOL_REL = 1e-4`` (measured 2.3e-5).
"""

from __future__ import annotations

import functools

import pytest
import torch

from vllm_neuron.functional.moe import expert_decode
from vllm_neuron.functional.moe.expert_decode import EXPERT_DECODE_MAX_TOKENS
from vllm_neuron.functional.moe.fused_fp8 import (
    fused_dispatch_counters,
    fused_fp8_decode_experts,
    reset_fused_dispatch_counters,
)
from vllm_neuron.functional.moe.moe_blockwise_fp8 import _swiglu_bound_operand

from .decode_fixtures import (
    EXPERTS,
    LOCAL_EXPERTS,
    RANK,
    SWIGLU_LIMIT,
    SimulatorCounter,
    baseline,
    decode_hidden,
    packed_expert_bank,
    routed_affinities,
)

FP32_RTOL, FP32_ATOL_REL = 1e-5, 1e-6
ORACLE_RTOL, ORACLE_ATOL_REL = 1e-3, 1e-4
BF16_RTOL = 2.0 ** -7
NEW_KERNEL = "expert_decode_kernel"

#: Hit patterns on this rank's 18 experts. T=4 shares expert 11 between tokens
#: 0 and 1 and expert 3 between tokens 0 and 3, and leaves token 2 with no local
#: expert, so the distinct-expert count (4) is below the (token, expert) count (6).
HITS = {
    1: [[3, 11]],
    4: [[3, 11], [11], [], [0, 3, 17]],
}


@functools.lru_cache(maxsize=None)
def _bank():
    return packed_expert_bank()


def _bounds():
    return _swiglu_bound_operand(SWIGLU_LIMIT, SWIGLU_LIMIT, torch.device("cpu"))


def _new(x, aff, bank, *, rank=RANK, programs=2, out_dtype=torch.float32):
    """The decode route, asserting one fused-seam NKI entry ran the new kernel."""
    reset_fused_dispatch_counters()
    with SimulatorCounter() as sim:
        out = fused_fp8_decode_experts(
            x, aff, bank, _bounds(), rank, programs=programs, out_dtype=out_dtype,
        )
    assert fused_dispatch_counters() == (1, 0), "the NKI route was not taken"
    assert sim.kernels == [NEW_KERNEL], f"simulated {sim.kernels}"
    return out


def _old(x, aff, bank, out_dtype):
    return baseline().routed_experts(
        x, aff, bank.weights, bank.scales, RANK, swiglu_limit=SWIGLU_LIMIT,
        out_dtype=out_dtype,
    )


def _assert_close_fp32(new, old, rtol=FP32_RTOL, atol_rel=FP32_ATOL_REL):
    assert new.shape == old.shape and new.dtype == torch.float32
    bound = atol_rel * old.abs().max() + rtol * old.abs()
    excess = (new - old).abs() - bound
    assert bool((excess <= 0).all()), (
        f"max |new-old| {(new - old).abs().max():.3e} exceeds the bound by "
        f"{excess.max():.3e} (max|old| {old.abs().max():.3e})"
    )


@pytest.mark.parametrize("tokens", [1, 4])
def test_expert_decode_matches_5938748_at_decode_shapes(tokens, monkeypatch):
    """18 local experts, H=4096, I=512 per rank, the router's global [T, 288]."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")  # 5938748's T=1 route
    bank = _bank()
    x = decode_hidden(tokens)
    aff = routed_affinities(HITS[tokens])
    old = _old(x, aff, bank, torch.float32)
    new = _new(x, aff, bank)
    _assert_close_fp32(new, old)
    # Tokens with no local expert are exactly zero on both sides.
    idle = torch.tensor([not hits for hits in HITS[tokens]])
    assert torch.count_nonzero(new[idle]) == 0 and torch.count_nonzero(old[idle]) == 0
    assert bool((new[~idle] != 0).any(-1).all())


@pytest.mark.parametrize("tokens", [1, 4])
def test_expert_decode_bf16_output_matches_5938748(tokens, monkeypatch):
    """The model's dtype: bf16 out, as ``block_quant_expert_mm`` returns."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    bank = _bank()
    x = decode_hidden(tokens, seed=41)
    aff = routed_affinities(HITS[tokens], seed=43)
    old = _old(x, aff, bank, None)
    new = _new(x, aff, bank, out_dtype=torch.bfloat16)
    assert new.dtype == torch.bfloat16 and new.shape == old.shape
    torch.testing.assert_close(
        new.float(), old.float(), rtol=BF16_RTOL,
        atol=FP32_ATOL_REL * float(old.float().abs().max()),
    )


def test_expert_decode_no_local_hit_is_exact_zero():
    bank = _bank()
    x = decode_hidden(4)
    aff = routed_affinities([[], [], [], []])
    for programs in (1, 2):
        out = _new(x, aff, bank, programs=programs)
        assert torch.equal(out, torch.zeros_like(out))


def test_expert_decode_one_and_two_programs_agree():
    """LNC2 splits each expert's I across the two cores; one core does all of I."""
    bank = _bank()
    x = decode_hidden(4, seed=47)
    aff = routed_affinities(HITS[4], seed=53)
    one = _new(x, aff, bank, programs=1)
    two = _new(x, aff, bank, programs=2)
    torch.testing.assert_close(two, one, rtol=1e-5, atol=1e-5 * float(one.abs().max()))


def test_expert_decode_reads_the_rank_slice_it_is_given():
    """The rank is an operand: a tensor rank and an int rank select the same slice,
    and another rank's slice of the same router output gives another answer."""
    bank = _bank()
    x = decode_hidden(1)
    aff = routed_affinities(HITS[1])
    as_int = _new(x, aff, bank, rank=RANK)
    as_tensor = _new(x, aff, bank, rank=torch.tensor([RANK], dtype=torch.int64))
    assert torch.equal(as_int, as_tensor)
    other = _new(x, aff, bank, rank=RANK + 1)
    assert torch.equal(other, torch.zeros_like(other))  # no hit on group 6


def test_expert_decode_clamps_a_rank_outside_the_groups():
    """A rank operand past the last group reads the last group, never past the
    router output (shape inference runs the kernel on ones-filled operands)."""
    bank, x, aff = _small_case(2, [[1], [0, 3]])
    last = fused_fp8_decode_experts(x, aff, bank, _bounds(), 1, out_dtype=torch.float32)
    beyond = fused_fp8_decode_experts(x, aff, bank, _bounds(), 7, out_dtype=torch.float32)
    assert torch.equal(beyond, last) and bool(last.abs().max() > 0)


def _small_case(tokens, hits, seed=61):
    """H=256, I=256, 4 local experts in 2 groups: fast, every hit pattern."""
    bank = packed_expert_bank(experts=4, hidden=256, intermediate=256, seed=seed)
    x = decode_hidden(tokens, hidden=256, seed=seed + 1)
    aff = routed_affinities(hits, experts=8, local_experts=4, rank=1, top_k=4,
                            seed=seed + 2)
    return bank, x, aff


@pytest.mark.parametrize("programs", [1, 2])
@pytest.mark.parametrize(
    "hits",
    [[[0]], [[3]], [[0, 1, 2, 3]], [[1], [1], [1]], [[0, 2], [], [3], [2, 1]]],
    ids=["first", "last", "all", "shared", "mixed"],
)
def test_expert_decode_small_shapes_match_the_torch_oracle(hits, programs):
    bank, x, aff = _small_case(len(hits), hits)
    reset_fused_dispatch_counters()
    with SimulatorCounter() as sim:
        out = fused_fp8_decode_experts(x, aff, bank, _bounds(), 1,
                                       programs=programs, out_dtype=torch.float32)
    assert fused_dispatch_counters() == (1, 0)
    assert sim.kernels == [NEW_KERNEL]
    ref = expert_decode.expert_decode_torch_oracle(
        x, aff, bank, _bounds(), 1, out_dtype=torch.float32)
    _assert_close_fp32(out, ref, ORACLE_RTOL, ORACLE_ATOL_REL)


def test_expert_decode_token_axis_at_64_rows():
    """T=64 in one launch; each row equals the same row computed alone."""
    gen = torch.Generator().manual_seed(67)
    hits = [sorted(torch.randperm(4, generator=gen)[: int(n)].tolist())
            for n in torch.randint(0, 3, (EXPERT_DECODE_MAX_TOKENS,), generator=gen)]
    bank, x, aff = _small_case(EXPERT_DECODE_MAX_TOKENS, hits)
    reset_fused_dispatch_counters()
    batch = fused_fp8_decode_experts(x, aff, bank, _bounds(), 1, programs=2,
                                     out_dtype=torch.float32)
    ref = expert_decode.expert_decode_torch_oracle(
        x, aff, bank, _bounds(), 1, out_dtype=torch.float32)
    _assert_close_fp32(batch, ref, ORACLE_RTOL, ORACLE_ATOL_REL)
    for t in (0, 17, 63):
        alone = fused_fp8_decode_experts(x[t:t + 1], aff[t:t + 1], bank, _bounds(), 1,
                                         programs=2, out_dtype=torch.float32)
        torch.testing.assert_close(alone[0], batch[t], rtol=1e-6, atol=1e-6)
    assert fused_dispatch_counters() == (4, 0)


def test_expert_decode_refuses_without_a_device(monkeypatch):
    """Like ``fused_fp8_experts`` the seam has no torch fallback: it refuses by name."""
    bank = _bank()
    x = decode_hidden(1)
    aff = routed_affinities(HITS[1])
    monkeypatch.setenv("NKI_SIMULATOR", "0")
    reset_fused_dispatch_counters()
    with pytest.raises(ValueError, match="fused_fp8_decode_experts serves"):
        fused_fp8_decode_experts(x, aff, bank, _bounds(), RANK)
    assert fused_dispatch_counters() == (0, 0)


def test_expert_decode_torch_oracle_matches_5938748(monkeypatch):
    """The reference the small-shape and T=64 cases use agrees with 5938748."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    bank = _bank()
    x = decode_hidden(4)
    aff = routed_affinities(HITS[4])
    old = _old(x, aff, bank, torch.float32)
    ref = expert_decode.expert_decode_torch_oracle(x, aff, bank, _bounds(), RANK,
                                                   out_dtype=torch.float32)
    _assert_close_fp32(ref, old, ORACLE_RTOL, ORACLE_ATOL_REL)


def test_expert_decode_admits_only_its_envelope():
    bank = _bank()
    x = decode_hidden(1)
    aff = routed_affinities(HITS[1])
    assert expert_decode.can_run_expert_decode(x, aff, bank)
    wide = torch.zeros(EXPERT_DECODE_MAX_TOKENS + 1, x.shape[1], dtype=x.dtype)
    wide_aff = torch.zeros(EXPERT_DECODE_MAX_TOKENS + 1, EXPERTS)
    assert not expert_decode.can_run_expert_decode(wide, wide_aff, bank)
    # The router width must be a whole number of this bank's expert groups.
    assert not expert_decode.can_run_expert_decode(x, aff[:, : EXPERTS - 1], bank)
    assert not expert_decode.can_run_expert_decode(x.float(), aff, bank)
    assert LOCAL_EXPERTS == bank.weights.shape[0]
