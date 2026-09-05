# SPDX-License-Identifier: Apache-2.0
"""Acceptance for `inc-glm53f-062` -- token-generation attention when more than one
new token is decoded at once.

WHAT THIS FILE MEASURES, in one sentence: that `attention_decode` computes the same
attention as an independent torch oracle at `s_active` of 1, 2 and 4, and that the
call really reaches the NKI kernel instead of quietly taking the module's torch
fallback.

WHY THE SECOND HALF OF THAT SENTENCE IS NOT OPTIONAL. `attention_decode` chooses
between a kernel and a torch implementation on its own, from the environment and the
shapes. A test that only compared numbers would pass just as green if the kernel
never ran, and the whole point of the increment is a verified multi-token KERNEL. So
the route is measured with the same seriousness as the numbers: call spies on the
three module globals the dispatch resolves at call time, a firing control that shows
those spies can read the other way, and a zero for the fallback that owns a control
proving the zero is a reading rather than an empty loop.

WHY THE ORACLE IS WRITTEN HERE AND BORROWS NOTHING. The module imported below is used
for exactly two things: the entry point under test, and the mask builder that
production itself calls (`attention_decode.py` L1543-1553 builds the mask this same
way). Every number the oracle computes -- the projection split, the gather, the mask
semantics, the scores, the softmax, the value matmul, the output layout -- is derived
in this file from the documented contract. An oracle that called the module's helpers
would be a second spelling of the implementation and would agree with it for the
wrong reason.

THE MASK CONVENTION, DERIVED AND THEN CHECKED. `gen_attention_decode_mask` returns
`[s_prior, bs, q_head, s_active]` in the kernel's block-KV layout. Read in sequential
order the rule is:

    a prior slot at position p is valid when p < min(pos_ids)
    the last s_active slots carry the causal staircase: slot k is visible to query j
    when k <= j

This file builds that mask itself and asserts it equals the generated one exactly, so
the oracle rests on a checked reading of the convention and not on a belief about it.

WHY THE LAYOUT QUESTION DOES NOT BITE AT THIS GEOMETRY, and why that is derived
rather than lucky. The kernel resizes the cache block length internally, and at the
declared shapes it resizes to 1. At block length 1 the sequential-position-to-linear-
index map collapses to the identity, so "sequential order" and "the layout the module
consumes" are the same thing here. That is asserted over every position, with the
module's own helper printed beside this file's independent derivation. A larger
context would make the map a real permutation and force the oracle to borrow the
module's layout code -- which is exactly the independence this file exists to keep.
The narrowness is deliberate and is recorded as a scope limit, not hidden.

WHY H IS 256 AND NOT 128. The docstring at `attention_decode.py` L1361 says the
hidden dimension must be a multiple of 128, and the kernel router at L1211 checks
only that, but the kernel runs on the module's own two-core grid (`wrapped[2]`, L1664)
and asserts `H1 % num_shards == 0` with `H1 = H / 128`. H = 128 is therefore admitted
by the gate and refused by the kernel. Measured, not inferred: H = 128 refused,
H = 256 and H = 512 ran. The gate/kernel disagreement is recorded as debt against
whoever next owns a change to that module; this file simply declares a shape the
kernel accepts.

WHY THE VALUE CACHE IS POSITIVE. The registered tolerance is `rtol=1e-2, atol=1e-5`
and this file does not author it. The operand DISTRIBUTION is this file's decision and
it matters: attention output is a weighted average of value rows, so with signed
values an output element can cancel toward zero, and there `atol` is comparing
rounding noise instead of the quantity under test. Values drawn in [0.5, 1.5] make
every output element a convex combination of positive numbers, so `rtol` always has
something to bite on. That is a numerically honest test rather than a tolerance
quietly widened to fit.

PARAMETRIZE IS USED HERE ON PURPOSE, unlike `test_mla_absorb.py` next door. That file
avoids it because its plan counts exactly three items for three conjuncts. This
increment's registered pair is "3/3 at s_active in {1, 2, 4}", so three separate
collected items per case IS the reading, and a loop inside one item would collapse
three verdicts into one.

NO SOURCE CHANGE. This file adds no production code and touches none. The kernel it
exercises already exists, so the increment is non-kernel-class by design ruling and
P13 is untouched.
"""

import math
import os

import pytest
import torch

from vllm_neuron.functional.attention import attention_decode as AD
from vllm_neuron.functional.attention.attention_decode_mask import (
    gen_attention_decode_mask,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# ----------------------------------------------------------------------------------
# THE DECLARED GEOMETRY. Stamped in predictions-062-shapes-r2.txt before this file
# existed, so none of it was chosen after seeing a result.
# ----------------------------------------------------------------------------------
DTYPE = torch.bfloat16
B = 1
H = 256                 # a multiple of 256, because H/128 must be even for the LNC-2 grid
D_HEAD = 128            # even and <= 128 (docstring L1362)
Q_HEADS = 1             # derived: W_qkv.shape[1] // D_HEAD - 2 * KV_HEADS = 3 - 2
KV_HEADS = 1            # inferred by the module from a 3D value cache (L1364-1367)
BLOCK_LEN = 128
NUM_BLOCKS_TOTAL = 2
BLOCK_IN_USE = 1        # block 1 of 2, so a gather that ignores the table reads wrong
NUM_BLOCKS_PER_SEQ = 1
S_CTX = NUM_BLOCKS_PER_SEQ * BLOCK_LEN      # the module's own derivation, L1535
MIN_POS = 99            # prior slots 0..98 valid, 99..(S_CTX - s_active - 1) masked
S_ACTIVE_CASES = (1, 2, 4)                  # inside the documented bound of 8 (L1358)
P_MAX = 128

# The registered tolerance. This file does not author it and must not move it.
RTOL = 1e-2
ATOL = 1e-5

# The module globals the dispatch resolves at call time. Captured at import so the
# spies can recognise them by OBJECT IDENTITY. Recognising by __name__ would be
# fragile: a jitted kernel object need not carry a usable one.
_KERNEL_FN = AD._torch_compatible_attention_block_tkg_kernel
_KERNEL_DCP_FN = AD._torch_compatible_attention_block_tkg_kernel_dcp

# The real callables, captured at IMPORT. A spy must call these and not whatever is
# bound to the attribute when it is installed: the bit-identity item installs spies
# twice in one test, and a spy that read the live attribute would wrap the previous
# spy, so one dispatch would increment two counters and every total would be wrong.
_REAL_WRAP_NKI = AD.wrap_nki
_REAL_TORCH_IMPL = AD._torch_attention_decode_impl


# ----------------------------------------------------------------------------------
# THE INDEPENDENT DERIVATIONS. Everything below is written from the documented
# contract. Nothing here calls into the module.
# ----------------------------------------------------------------------------------
def _my_resized_block_len(block_len, bs, q_head, s_active, s_prior, p_max=P_MAX):
    """The block-length resize the kernel applies internally, derived here.

    Batch sharding needs an even batch; sequence sharding needs a prior length of at
    least two partitions and only when batch sharding is off. The reduced length is
    the greatest common divisor of the block length and the number of shard-sized
    chunks in the bucket.
    """
    lnc = 2
    batch_sharded = (bs % lnc == 0) and (
        bs * q_head * s_active >= p_max or s_prior <= 2 * p_max
    )
    sprior_sharded = (not batch_sharded) and s_prior >= lnc * p_max
    n_prgs = lnc if sprior_sharded else 1
    bucket = (s_prior // block_len) * block_len
    min_multiple = n_prgs * p_max
    if bucket % min_multiple != 0:
        return block_len
    return math.gcd(block_len, bucket // min_multiple)


def _my_seq_to_linear(pos, block_len, p_max=P_MAX):
    """Sequential position to linear index in the kernel's partition-major layout."""
    fold = pos // (p_max * block_len)
    within = pos % (p_max * block_len)
    partition = within // block_len
    blk_off = within % block_len
    return (fold * block_len + blk_off) * p_max + partition


def _logical_mask(s_active):
    """The mask in plain sequential order: `[S_CTX, s_active]`, 1 means visible."""
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
    """Every tensor the call needs, at the declared shapes, from a fixed seed.

    The mask is built HERE, once, and reused by every leg that consumes these inputs.
    That placement is deliberate: `gen_attention_decode_mask` itself chooses between a
    kernel and a torch build from the same environment gate the bit-identity test
    flips, so a mask built inside a flipped region would differ between the two legs
    and "identical inputs" would be false while the test still ran green.
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

    Projection uses the declared bfloat16 weights, because that is what any
    implementation would do with them. Everything after it is float32: a reference
    that reproduced the implementation's own rounding would not be a reference.
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
    # Without W_out the module returns the transposed layout (L736).
    return out.transpose(-2, -1).to(DTYPE)


# ----------------------------------------------------------------------------------
# THE SPIES. They observe the dispatch and change nothing about it.
# ----------------------------------------------------------------------------------
class _Spy:
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
    """Count dispatches out of seam, without replacing anything that computes.

    The kernel side wraps `wrap_nki`, which is what the module calls on the kernel
    function at L1608 and L1663, records which function object was passed, and hands
    back the real wrapper. Substituting a python stand-in for a jitted kernel would
    change what actually runs, so nothing here does that. The fallback side is a plain
    function and is wrapped directly. Anything `wrap_nki` is handed that this file
    does not recognise is counted too, and that count must read zero -- an
    unrecognised dispatch must not be able to hide inside a passing total.
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


# ----------------------------------------------------------------------------------
# 1. THE CONTROL THAT EVERY ROUTE READING RESTS ON.
# ----------------------------------------------------------------------------------
def test_the_env_gate_is_read_at_call_time(monkeypatch):
    """Flipping the environment must change `can_run_kernel()` on the NEXT call.

    Two later items force the torch path by setting this variable. That only proves
    anything if the gate is read when the function is called rather than captured when
    the module was imported, so it is measured here first instead of assumed. The
    variable resolves through `vllm_neuron/envs.py`'s module `__getattr__` on every
    access -- but that is a claim about code, and this is the measurement.
    """
    before = can_run_kernel(torch.zeros(1))
    print(f"  GATE_WITH_THE_PINNED_ENVIRONMENT={before}"
          f" (NKI_SIMULATOR={os.environ.get('NKI_SIMULATOR')!r},"
          f" DISABLE={os.environ.get('VLLM_NEURON_DISABLE_NKI_KERNELS')!r})")
    assert before is True

    monkeypatch.setenv("VLLM_NEURON_DISABLE_NKI_KERNELS", "1")
    during = can_run_kernel(torch.zeros(1))
    print(f"  GATE_WITH_KERNELS_DISABLED={during}")
    assert during is False, "the gate did not follow the environment at call time"

    monkeypatch.setenv("VLLM_NEURON_DISABLE_NKI_KERNELS", "0")
    after = can_run_kernel(torch.zeros(1))
    print(f"  GATE_RESTORED={after}")
    print(f"[gate] pinned={before} disabled={during} restored={after}")
    assert after is True, "the flip did not reverse, so the reading is not repeatable"


# ----------------------------------------------------------------------------------
# 2. THE ROUTER ADMITS THE DECLARED SHAPES.
# ----------------------------------------------------------------------------------
def test_the_declared_shapes_pass_the_kernel_router():
    """`_can_use_attention_block_kernel` must answer True at the declared shapes.

    It has six refusing conditions. None of them fires here: the hidden dimension is
    a multiple of 128, the head dimension is even, the cache is 3D so the key/value
    head count infers to one and the per-head-table rule is vacuous, and attention
    data parallelism is one. The reading is paired with a control that makes the same
    gate say no, so True is a discrimination and not a constant.
    """
    inp = _build_inputs(1)
    gate = AD._can_use_attention_block_kernel(
        X=inp["X"],
        V_cache=inp["V_cache"],
        active_blocks_table=inp["active_blocks_table"],
        attention_dp=1,
    )
    print(f"  CAN_RUN_KERNEL={can_run_kernel(inp['X'])}  ROUTER_SAYS_KERNEL={gate}")
    print(f"  H={H} H_MOD_128={H % 128} H1={H // 128} H1_IS_EVEN={(H // 128) % 2 == 0}")
    print(f"  D_HEAD={D_HEAD} IS_EVEN={D_HEAD % 2 == 0}"
          f"  V_CACHE_DIM={inp['V_cache'].dim()}")
    assert can_run_kernel(inp["X"]) is True
    assert gate is True

    refused = AD._can_use_attention_block_kernel(
        X=inp["X"],
        V_cache=inp["V_cache"],
        active_blocks_table=inp["active_blocks_table"],
        attention_dp=2,
    )
    print(f"  CONTROL_THE_SAME_GATE_WITH_DP_2={refused}")
    print(f"[router] gate={gate} can_run_kernel={can_run_kernel(inp['X'])}"
          f" control_dp2={refused} H={H} d_head={D_HEAD}")
    assert refused is False, "the gate answered True for a case it must refuse"


# ----------------------------------------------------------------------------------
# 3. THE LAYOUT THE ORACLE'S INDEPENDENCE DEPENDS ON.
# ----------------------------------------------------------------------------------
def test_the_block_kv_layout_is_the_identity_at_this_geometry():
    """At the declared shapes the mask layout map is the identity, over every slot.

    This is what lets the oracle reason in plain sequential order. The value is
    derived in this file and the module's own helper is printed beside it: if the two
    ever disagree, the derivation is wrong and this test says so rather than the
    oracle failing later for an unrelated-looking reason.
    """
    from vllm_neuron.functional.attention.attention_decode_mask import (
        _resize_block_len as _module_resize,
    )

    for s_active in S_ACTIVE_CASES:
        mine = _my_resized_block_len(BLOCK_LEN, B, Q_HEADS, s_active, S_CTX)
        theirs = _module_resize(BLOCK_LEN, B, Q_HEADS, s_active, S_CTX)
        moved = [p for p in range(S_CTX) if _my_seq_to_linear(p, mine) != p]
        print(f"  S_ACTIVE={s_active} RESIZED mine={mine} module={theirs}"
              f"  BLOCKS_AFTER_RESIZE={S_CTX // mine}"
              f"  POSITIONS_THAT_MOVE={len(moved)} of {S_CTX}")
        print(f"[layout] s_active={s_active} resized={mine} module_resized={theirs}"
              f" moved={len(moved)} population={S_CTX}")
        assert mine == theirs, "my derivation of the resize disagrees with the module"
        assert mine == 1
        assert moved == [], "the layout is not the identity, so the oracle cannot"

    # The map must not be the identity in general, or the check above proves nothing.
    moved_elsewhere = [p for p in range(1024) if _my_seq_to_linear(p, 4) != p]
    print(f"  CONTROL_AT_BLOCK_LENGTH_4_POSITIONS_THAT_MOVE={len(moved_elsewhere)}"
          f" of 1024")
    assert moved_elsewhere, "the map is the identity everywhere, so it measures nothing"


# ----------------------------------------------------------------------------------
# 4-6. THE MASK CONVENTION, CHECKED RATHER THAN BELIEVED.
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("s_active", S_ACTIVE_CASES)
def test_the_generated_mask_equals_the_independent_logical_mask(s_active):
    """The mask production builds must equal the mask this file derives, exactly.

    Exact equality is the right bound: the mask holds only zeros and ones. The
    expected total is arithmetic, not a recorded number -- `MIN_POS` valid prior slots
    per query plus one staircase entry for each pair with k <= j.
    """
    inp = _build_inputs(s_active)
    generated = inp["attention_mask"][:, 0, 0, :].to(torch.float32)
    mine = _logical_mask(s_active)
    expected_sum = MIN_POS * s_active + s_active * (s_active + 1) // 2
    differing = int((generated != mine).sum())
    print(f"  MASK_SHAPE={tuple(inp['attention_mask'].shape)}"
          f"  SUM_GENERATED={float(generated.sum())}"
          f"  SUM_MINE={float(mine.sum())}  SUM_BY_ARITHMETIC={expected_sum}")
    print(f"  DIFFERING_ELEMENTS={differing} of {generated.numel()}")
    print(f"[mask] s_active={s_active} differing={differing}"
          f" population={generated.numel()} sum={float(generated.sum())}"
          f" expected_sum={expected_sum}")
    assert float(mine.sum()) == expected_sum
    assert differing == 0
    # A zero difference must be a comparison that can also read non-zero.
    perturbed = mine.clone()
    perturbed[0, 0] = 1.0 - perturbed[0, 0]
    print(f"  CONTROL_ONE_FLIPPED_ELEMENT_IS_SEEN="
          f"{int((generated != perturbed).sum())}")
    assert int((generated != perturbed).sum()) == 1


# ----------------------------------------------------------------------------------
# 7-9. THE REGISTERED PAIR: THREE CASES, THREE ORACLE COMPARISONS.
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("s_active", S_ACTIVE_CASES)
def test_the_kernel_matches_the_oracle(s_active, monkeypatch):
    """The kernel result must match the in-file oracle at the registered tolerance.

    The spies ride along so each case also witnesses that the number it compared came
    out of the kernel. A green comparison against a torch fallback would be the exact
    failure this increment exists to rule out.
    """
    spy = _install_spies(monkeypatch)
    inp = _build_inputs(s_active)
    got = _call(inp)[0]
    want = _oracle(inp, s_active)
    print(f"  S_ACTIVE={s_active} SPIES[{spy}]")
    print(f"  SHAPES got={tuple(got.shape)} want={tuple(want.shape)}"
          f" dtype={got.dtype}")
    assert spy.kernel == 1, f"this case did not reach the kernel: {spy}"
    assert spy.fallback == 0
    assert spy.kernel_dcp == 0
    assert spy.unrecognised == 0
    assert tuple(got.shape) == (B, Q_HEADS, D_HEAD, s_active)

    diff = (got.to(torch.float32) - want.to(torch.float32)).abs()
    print(f"  MAX_ABS_DIFF={float(diff.max()):.6e}"
          f"  MEAN_ABS_DIFF={float(diff.mean()):.6e}"
          f"  OUT_RANGE=[{float(got.to(torch.float32).min()):.5f},"
          f" {float(got.to(torch.float32).max()):.5f}]")
    print(f"  SMALLEST_ABS_OUTPUT={float(got.to(torch.float32).abs().min()):.5f}"
          f" (kept away from zero on purpose, so rtol has something to bite on)")
    print(f"[oracle] s_active={s_active} max_abs_diff={float(diff.max()):.6e}"
          f" mean_abs_diff={float(diff.mean()):.6e} population={diff.numel()}"
          f" kernel_calls={spy.kernel} fallback_calls={spy.fallback}"
          f" rtol={RTOL} atol={ATOL}")
    torch.testing.assert_close(
        got.to(torch.float32), want.to(torch.float32), rtol=RTOL, atol=ATOL
    )

    # The comparison must be able to fail, or a pass says nothing.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            got.to(torch.float32),
            want.to(torch.float32) + 1.0,
            rtol=RTOL,
            atol=ATOL,
        )
    print("  CONTROL_A_SHIFTED_ORACLE_IS_REJECTED=yes")


# ----------------------------------------------------------------------------------
# 10-11. THE ROUTE PREDICATE AND ITS FIRING CONTROL.
# ----------------------------------------------------------------------------------
def test_the_spies_count_one_kernel_entry_per_case(monkeypatch):
    """Across the three declared cases: three kernel entries and no torch entries.

    The population is named beside the count -- three calls, one per declared
    `s_active` -- so the total cannot be read as a bare number.
    """
    spy = _install_spies(monkeypatch)
    for s_active in S_ACTIVE_CASES:
        _call(_build_inputs(s_active))
    print(f"  POPULATION=3 calls, one per s_active in {S_ACTIVE_CASES}")
    print(f"  SPIES[{spy}]")
    print(f"[route] population={len(S_ACTIVE_CASES)} kernel={spy.kernel}"
          f" fallback={spy.fallback} dcp={spy.kernel_dcp}"
          f" unrecognised={spy.unrecognised}")
    assert spy.kernel == len(S_ACTIVE_CASES)
    assert spy.fallback == 0
    assert spy.kernel_dcp == 0
    assert spy.unrecognised == 0


def test_the_firing_control_with_kernels_disabled(monkeypatch):
    """With kernels disabled the same call must take the torch path instead.

    This is what makes the zero above a reading. Without it, a fallback count of zero
    would be equally consistent with a spy that never fires at all.
    """
    spy = _install_spies(monkeypatch)
    inp = _build_inputs(1)
    monkeypatch.setenv("VLLM_NEURON_DISABLE_NKI_KERNELS", "1")
    out = _call(inp)[0]
    print(f"  SPIES_WITH_KERNELS_DISABLED[{spy}]  OUT_SHAPE={tuple(out.shape)}")
    print(f"[firing-control] population=1 fallback={spy.fallback} kernel={spy.kernel}"
          f" unrecognised={spy.unrecognised}")
    assert spy.fallback == 1, f"the torch path was not taken: {spy}"
    assert spy.kernel == 0
    assert spy.unrecognised == 0


# ----------------------------------------------------------------------------------
# 12. THE MODULE'S OWN PROMISE, MADE FALSIFIABLE.
# ----------------------------------------------------------------------------------
def test_bit_identity_kernel_versus_the_module_torch_path(monkeypatch):
    """At one new token the two paths must agree to a maximum absolute difference of
    exactly zero.

    The module promises it at `attention_decode.py` L1745-1746: with the mask built by
    `gen_attention_decode_mask`, "kernel and torch paths produce identical numerics".
    This turns that sentence into a measurement. The inputs are built once and both
    legs consume the same tensors, including the same mask; only the environment gate
    moves between them, and the spies witness which path each leg actually took.

    A non-zero reading is a contradiction to report with the value recorded. It is not
    a reason to loosen this bound, and this file does not carry a looser one.
    """
    inp = _build_inputs(1)

    spy_kernel_leg = _install_spies(monkeypatch)
    kernel_out = _call(inp)[0]
    print(f"  LEG_A_SPIES[{spy_kernel_leg}]")
    assert spy_kernel_leg.kernel == 1 and spy_kernel_leg.fallback == 0

    monkeypatch.setenv("VLLM_NEURON_DISABLE_NKI_KERNELS", "1")
    spy_torch_leg = _install_spies(monkeypatch)
    torch_out = _call(inp)[0]
    print(f"  LEG_B_SPIES[{spy_torch_leg}]")
    assert spy_torch_leg.fallback == 1 and spy_torch_leg.kernel == 0

    a = kernel_out.to(torch.float32)
    b = torch_out.to(torch.float32)
    diff = (a - b).abs()
    max_abs = float(diff.max())
    differing = int((a != b).sum())
    print(f"  SHAPES kernel={tuple(kernel_out.shape)} torch={tuple(torch_out.shape)}")
    print(f"  MAX_ABS_DIFF={max_abs:.6e}  DIFFERING_ELEMENTS={differing}"
          f" of {a.numel()}")
    print(f"  KERNEL_RANGE=[{float(a.min()):.6f}, {float(a.max()):.6f}]"
          f"  TORCH_RANGE=[{float(b.min()):.6f}, {float(b.max()):.6f}]")

    # The zero this arm hopes for owns a control: the same comparison on tensors that
    # really differ must read non-zero.
    nudged = b.clone()
    nudged.view(-1)[0] += 1.0
    print(f"  CONTROL_A_NUDGED_TENSOR_READS"
          f"={float((a - nudged).abs().max()):.6e}")
    print(f"[bit-identity] s_active=1 max_abs_diff={max_abs:.6e}"
          f" differing={differing} population={a.numel()}"
          f" kernel_leg={spy_kernel_leg.kernel} torch_leg={spy_torch_leg.fallback}"
          f" control={float((a - nudged).abs().max()):.6e}")
    assert float((a - nudged).abs().max()) > 0.0

    assert max_abs == 0.0, (
        f"kernel and torch paths disagree: max abs diff {max_abs:.6e} over "
        f"{differing} differing elements. Recorded as evidence_contradicts_design; "
        f"the bound is not loosened."
    )
