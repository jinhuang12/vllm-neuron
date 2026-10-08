# SPDX-License-Identifier: Apache-2.0
"""The routed-expert decode kernel at B in {1, 4, 64} against the 0a08ff4 kernel.

Old side: the 0a08ff4 ``expert_decode_kernel`` and its entry point, copied
unedited under ``test/hardware/baselines/moe_0a08ff4`` (``git show 0a08ff4:...``).
New side: ``fused_fp8.fused_fp8_decode_experts`` of this tree.

Shapes: one rank's bank at EP=16, TP=64 -- 18 local experts, H=4096, I=512 per
rank, the router's global ``[T, 288]`` output, rank 5. Routing: seeded tables from
``decode_fixtures.batch_routing_table`` (Zipf-skewed popularity, served-like: some
local experts get no token, a few get many, tokens carry 0..k local picks), plus
a stress table at T=64 whose hottest expert takes more tokens than one group.

Declared tolerance, on the fp32 output ``[T, H]``:
``|new - old| <= FP32_ATOL_REL * max|old| + FP32_RTOL * |old|`` with
``FP32_RTOL = 1e-5``, ``FP32_ATOL_REL = 1e-6``. Both kernels take the same fp32
128x128 products, the same per-block scale multiply and hidden-block reduce, the
same SwiGLU sequence and bf16 activation; the grouped path (T > 16) gathers
each expert's token columns and gate weights with one-hot PE products (exact
copies) and adds each column into its token's accumulator with a one-hot PE
product (one nonzero term: exact), in ascending expert order, so the bound
only allows for fp32 round-off. On the bf16 output: one bf16 step
(``rtol = 2**-7``) plus the same floor. Tokens with no local expert are exactly
zero on both sides.
"""

from __future__ import annotations

import functools

import pytest
import torch

from vllm_neuron.functional.moe import expert_decode
from vllm_neuron.functional.moe.fused_fp8 import (
    fused_dispatch_counters,
    fused_fp8_decode_experts,
    reset_fused_dispatch_counters,
)
from vllm_neuron.functional.moe.moe_blockwise_fp8 import _swiglu_bound_operand

from .decode_fixtures import (
    RANK,
    SWIGLU_LIMIT,
    SimulatorCounter,
    baseline_0a08ff4,
    batch_routing_table,
    decode_hidden,
    hit_histogram,
    packed_expert_bank,
    routed_affinities,
)

FP32_RTOL, FP32_ATOL_REL = 1e-5, 1e-6
ORACLE_RTOL, ORACLE_ATOL_REL = 1e-3, 1e-4
BF16_RTOL = 2.0 ** -7
NEW_KERNEL = "expert_decode_kernel"

#: Seed of the routing table per token count (the benchmark uses the same ones).
TABLE_SEED = {1: 1, 4: 4, 64: 3}


@functools.lru_cache(maxsize=None)
def _bank():
    return packed_expert_bank()


def _bounds():
    return _swiglu_bound_operand(SWIGLU_LIMIT, SWIGLU_LIMIT, torch.device("cpu"))


def _table(tokens):
    hits = batch_routing_table(tokens, TABLE_SEED[tokens])
    if tokens == 1 and not hits[0]:
        hits = [[7]]  # a B=1 table with no local pick would compare zeros only
    return hits


def _stress_table():
    """T=64: expert 4 takes 40 tokens (more than one group), 6 experts none."""
    hits = [[] for _ in range(64)]
    for t in range(40):
        hits[t].append(4)
    for t, e in enumerate([0, 2, 3, 9, 11, 13, 14, 15, 16, 17] * 3):
        if e not in hits[t + 20]:
            hits[t + 20].append(e)
    hits[63] = [0, 3, 4, 9, 11, 16, 17]  # one token with seven local picks
    return [sorted(row) for row in hits]


def _new(x, aff, bank, *, programs=2, out_dtype=torch.float32, rank=RANK):
    reset_fused_dispatch_counters()
    with SimulatorCounter() as sim:
        out = fused_fp8_decode_experts(x, aff, bank, _bounds(), rank,
                                       programs=programs, out_dtype=out_dtype)
    assert fused_dispatch_counters() == (1, 0), "the NKI route was not taken"
    assert sim.kernels == [NEW_KERNEL], f"simulated {sim.kernels}"
    return out


def _old(x, aff, bank, *, programs=2, out_dtype=torch.float32, rank=RANK):
    return baseline_0a08ff4().decode_experts(x, aff, bank, _bounds(), rank,
                                             programs=programs, out_dtype=out_dtype)


def _assert_close_fp32(new, old, rtol=FP32_RTOL, atol_rel=FP32_ATOL_REL):
    assert new.shape == old.shape and new.dtype == torch.float32
    bound = atol_rel * old.abs().max() + rtol * old.abs()
    excess = (new - old).abs() - bound
    assert bool((excess <= 0).all()), (
        f"max |new-old| {(new - old).abs().max():.3e} exceeds the bound by "
        f"{excess.max():.3e} (max|old| {old.abs().max():.3e})"
    )


def _idle(hits):
    return torch.tensor([not row for row in hits])


def test_routing_tables_are_realistic():
    """The tables the comparisons use: skewed, with empty and crowded experts."""
    for tokens in (4, 64):
        summary = hit_histogram(_table(tokens))
        assert summary["token_expert_pairs"] > 0
    big = hit_histogram(_table(64))
    assert 0 < big["distinct_local_experts"] < 18  # some experts get no token
    assert big["max_tokens_on_one_expert"] >= 5  # and some get many
    assert big["tokens_with_no_local_expert"] > 0
    stress = hit_histogram(_stress_table())
    assert stress["max_tokens_on_one_expert"] > expert_decode.GROUP_TOKENS


@pytest.mark.parametrize("tokens", [1, 4, 64])
def test_matches_0a08ff4_fp32_at_decode_shapes(tokens):
    bank = _bank()
    hits = _table(tokens)
    x = decode_hidden(tokens, seed=100 + tokens)
    aff = routed_affinities(hits, seed=200 + tokens)
    old = _old(x, aff, bank)
    new = _new(x, aff, bank)
    _assert_close_fp32(new, old)
    idle = _idle(hits)
    assert torch.count_nonzero(new[idle]) == 0 and torch.count_nonzero(old[idle]) == 0
    assert bool((new[~idle] != 0).any(-1).all())


@pytest.mark.parametrize("tokens", [1, 4, 64])
def test_matches_0a08ff4_bf16_at_decode_shapes(tokens):
    bank = _bank()
    hits = _table(tokens)
    x = decode_hidden(tokens, seed=300 + tokens)
    aff = routed_affinities(hits, seed=400 + tokens)
    old = _old(x, aff, bank, out_dtype=torch.bfloat16)
    new = _new(x, aff, bank, out_dtype=torch.bfloat16)
    assert new.dtype == torch.bfloat16 and new.shape == old.shape
    torch.testing.assert_close(new.float(), old.float(), rtol=BF16_RTOL,
                               atol=FP32_ATOL_REL * float(old.float().abs().max()))


def test_matches_0a08ff4_when_one_expert_spans_several_groups():
    """T=64, expert 4 takes 40 tokens: more than one token group of the kernel."""
    bank = _bank()
    hits = _stress_table()
    x = decode_hidden(64, seed=501)
    aff = routed_affinities(hits, seed=503)
    old = _old(x, aff, bank)
    new = _new(x, aff, bank)
    _assert_close_fp32(new, old)
    idle = _idle(hits)
    assert torch.count_nonzero(new[idle]) == 0


def test_batch_one_and_two_programs_agree():
    """LNC2 splits each expert's I across the cores; one core does all of I."""
    bank = _bank()
    hits = _table(64)
    x = decode_hidden(64, seed=601)
    aff = routed_affinities(hits, seed=603)
    one = _new(x, aff, bank, programs=1)
    two = _new(x, aff, bank, programs=2)
    torch.testing.assert_close(two, one, rtol=1e-5, atol=1e-5 * float(one.abs().max()))


def test_batch_no_local_hit_is_exact_zero():
    bank = _bank()
    x = decode_hidden(64, seed=701)
    aff = routed_affinities([[] for _ in range(64)], seed=703)
    for programs in (1, 2):
        out = _new(x, aff, bank, programs=programs)
        assert torch.equal(out, torch.zeros_like(out))


def _small_case(hits, seed=61):
    """H=256, I=256, 4 local experts in 2 groups."""
    bank = packed_expert_bank(experts=4, hidden=256, intermediate=256, seed=seed)
    x = decode_hidden(len(hits), hidden=256, seed=seed + 1)
    aff = routed_affinities(hits, experts=8, local_experts=4, rank=1, top_k=4,
                            seed=seed + 2)
    return bank, x, aff


def _small_patterns():
    gen = torch.Generator().manual_seed(67)
    mixed = [sorted(torch.randperm(4, generator=gen)[: int(n)].tolist())
             for n in torch.randint(0, 3, (48,), generator=gen)]
    return {
        "all_on_one_expert_64": [[2]] * 64,
        "every_expert_every_token_32": [[0, 1, 2, 3]] * 32,
        "mixed_48": mixed,
        "first_and_last_token_only_17": [[1]] + [[]] * 15 + [[3]],
    }


@pytest.mark.parametrize("programs", [1, 2])
@pytest.mark.parametrize("name", list(_small_patterns()))
def test_grouped_shapes_match_the_torch_oracle(name, programs):
    hits = _small_patterns()[name]
    bank, x, aff = _small_case(hits)
    out = _new(x, aff, bank, programs=programs, rank=1)
    ref = expert_decode.expert_decode_torch_oracle(x, aff, bank, _bounds(), 1,
                                                   out_dtype=torch.float32)
    _assert_close_fp32(out, ref, ORACLE_RTOL, ORACLE_ATOL_REL)
    idle = _idle(hits)
    assert torch.count_nonzero(out[idle]) == 0


def test_decode_plan_groups_tokens_above_the_token_axis_limit():
    """T <= 16 streams every token through each visited expert (the 0a08ff4
    arithmetic); above that each expert sees only its own tokens, in groups."""
    for tokens in (1, 2, 4, 8, 16):
        assert expert_decode.decode_plan(tokens) == ("token_axis", tokens)
    for tokens in (17, 32, 64):
        assert expert_decode.decode_plan(tokens) == ("grouped", expert_decode.GROUP_TOKENS)
    assert expert_decode.GROUP_TOKENS < 64
