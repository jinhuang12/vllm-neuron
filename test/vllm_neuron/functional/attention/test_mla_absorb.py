# SPDX-License-Identifier: Apache-2.0
"""Acceptance for `inc-glm53f-097` -- the MLA absorb kernel.

THREE tests, one per counted conjunct of the increment plan's `inc-glm53f-097`
Acceptance bullet, and NO `parametrize` decorator in this file: the plan requires
exactly 3 collected items, and a parametrized case would collect as several items
for one conjunct, so the count would stop meaning what it says. Each conjunct's
several cases are therefore a loop INSIDE its item, and each item prints its counted
value. Conjunct 3 asserts the absence of `parametrize` by reading this file's own
source, so the count cannot drift without a test going red.

THIS FILE ANSWERS "DOES THE KERNEL COMPUTE THE PER-HEAD BATCHED MATMUL?" AND
NOTHING ELSE. Whether anything is wired to the seam is `inc-glm53f-042`'s question,
asked in its own file, and so is where `W_UK` and `W_UV` come from. Keeping them
apart is deliberate: in one file either increment's counted predicate could be
satisfied by the other increment's items.

WHY THE OPERANDS ARE SCALED THE WAY THEY ARE, because it is a numerical decision and
not a cosmetic one. The tolerance is the plan's -- `rtol=1e-2, atol=1e-5` -- and
this file does not author it. But the operand DISTRIBUTION is this file's, and it
matters: with unit-variance operands an output element is a sum of `K` products, so
it has standard deviation `sqrt(K)` -- about 16 at `K = 256` and 23 at `K = 512`.
The float32 summation-order difference between the kernel's PSUM accumulation and
the oracle's `einsum` then scales with that magnitude, and near a cancellation zero
it can exceed `atol` while `rtol` still has nothing to bite on. So `w` is scaled by
`1/sqrt(K)`, which is how a projection weight is actually initialised and which puts
output elements at unit scale -- the scale a post-normalisation activation really
has in this model. That is a more faithful test than the unscaled version AND a
numerically honest one, rather than a tolerance quietly widened to fit.

Run it with the Tier N harness -- the NKI simulator on a host CPU, no device::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/attention/test_mla_absorb.py \
        -q -s --timeout 600 -p no:cacheprovider
"""

from __future__ import annotations

import ast
import inspect
import math

import pytest
import torch

from vllm_neuron.functional.attention import mla_absorb as MA

#: The head count this checkpoint declares for the MLA attention half. Both
#: production shapes carry it, so it is stated once.
DECLARED_HEADS = 64

#: The three sequence extents the plan names. `128` is a whole sequence tile;
#: `1` and `7` are RAGGED against it, and one of them is the decode step, which is
#: the length production spends almost all of its time at. A file that measured
#: only `128` could not fail if the ragged bound were broken.
SEQ_CASES = (1, 7, 128)

#: absorb-in: `query [S, H, 256] @ W_UK [H, 256, 512]` -> `q_lift [S, H, 512]`.
ABSORB_IN_K, ABSORB_IN_N = 256, 512

#: absorb-out: the sparse seam's `[S, H, 512]` @ `W_UV [H, 512, 256]` -> the width
#: `project_output` consumes.
ABSORB_OUT_K, ABSORB_OUT_N = 512, 256

#: THE TOLERANCE IS READ FROM THE PLAN AND NOT AUTHORED HERE. This increment's
#: boundary states it authors no criterion or tolerance; both numbers are the
#: `inc-glm53f-097` Acceptance bullet's, verbatim, and the argument order the bullet
#: fixes is kernel first, oracle second.
RTOL = 1e-2
ATOL = 1e-5


#: Shapes that are RAGGED against the kernel's tile extents, so the `min` bounds in
#: its three inner tiling loops -- and the head loop -- are load-bearing instead of
#: inert.
#:
#: WHY THEY EXIST, and it is a known defect in this campaign rather than a
#: precaution. BOTH production shapes divide exactly on the contraction and output
#: axes: `K` is 256 or 512, whole multiples of the 128-wide contraction tile, and
#: `N` is 512 or 256 against a 512-wide output tile. So the three declared cases
#: CANNOT FAIL if the ragged contraction or output tail is broken, and the sibling
#: increment `-039a` shipped with exactly that hole -- review finding
#: `B37-M1-ragged-tail-uncovered-by-acceptance`, where a dropped ragged tail
#: returned never-written memory and its suite still read all green. Repeating a
#: finding this campaign has already paid for would be the avoidable kind.
#:
#: WHY THEY FOLD INTO CONJUNCT 1 AND ARE NOT A FOURTH TEST. The plan pins this file
#: at exactly three collected items. These use the same oracle, the same tolerance
#: and the same comparison as conjunct 1, and they are counted SEPARATELY from the
#: three declared shapes, so conjunct 1's `3/3` reading does not move either.
#:
#: The extents are DERIVED from the kernel module's own tile constants and never
#: retyped, so if a tile width changes these cases stay ragged instead of silently
#: becoming exact. Each axis is covered both below its tile and past it, because the
#: two fail differently: below the tile the loop runs once with a short bound, past
#: it the loop runs again with a short final bound. The head count is varied down to
#: 1 as well -- `H` is not tiled, but a head loop that ran the wrong number of times
#: would be invisible at the single declared `H = 64`.
def _ragged_cases() -> tuple[tuple[str, int, int, int, int], ...]:
    k, n, s = MA.CONTRACTION_TILE, MA.OUTPUT_TILE, MA.SEQUENCE_TILE
    return (
        #  label                    S        H   K         N
        ("k_below_tile",            2,       3,  63,       n),
        ("k_one_past_tile",         2,       3,  k + 1,    n),
        ("k_ragged_wide",           2,       2,  k + 72,   n),
        ("n_below_tile",            2,       3,  k,        300),
        ("n_one_past_tile",         2,       2,  k,        n + 1),
        ("n_ragged_wide",           2,       2,  k,        n + 88),
        ("s_one_past_tile",         s + 1,   2,  k,        n),
        ("single_head",             5,       1,  k + 13,   n + 29),
        ("all_axes_ragged",         s + 7,   3,  k + 72,   n + 88),
        ("all_axes_below_tile",     7,       2,  13,       29),
    )


RAGGED_CASES = _ragged_cases()


def ragged_axes(seq: int, kdim: int, ndim: int) -> tuple[str, ...]:
    """Which of the three tiled axes this shape leaves a short final tile on."""
    return tuple(
        axis
        for axis, extent, tile in (
            ("S", seq, MA.SEQUENCE_TILE),
            ("K", kdim, MA.CONTRACTION_TILE),
            ("N", ndim, MA.OUTPUT_TILE),
        )
        if extent % tile != 0
    )


def _operands(seq: int, heads: int, kdim: int, ndim: int, seed: int):
    """bf16 `x [S, H, K]` and `w [H, K, N]`, with `w` scaled by ``1/sqrt(K)``.

    The seed is per case, so a failure names one reproducible case rather than
    depending on how many cases ran before it. The module docstring says why `w` is
    scaled.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(seq, heads, kdim, generator=g).to(torch.bfloat16)
    w = (torch.randn(heads, kdim, ndim, generator=g) / math.sqrt(kdim)).to(
        torch.bfloat16
    )
    return x, w


def _one_case(
    label: str, seq: int, kdim: int, ndim: int, seed: int, heads: int = DECLARED_HEADS
) -> float:
    """Run one shape through the seam and compare it to the fp32 oracle.

    THE ORACLE IS GIVEN THE SAME bf16 TENSORS the kernel got, upcast -- not a
    separately drawn fp32 pair. Otherwise the comparison would measure the
    difference between two random draws rather than the kernel.

    HEADROOM IS PRINTED, not just the verdict. At the worst element this prints the
    difference, the allowance `atol + rtol*|expected|` that element actually got,
    and the ratio between them. That matters for reading this test honestly: the
    output dtype is bf16, so most of the difference below is the OUTPUT CAST rather
    than kernel error -- bf16 spacing at magnitude 4 is 0.031, so a half-spacing of
    0.0156 is expected and is not a defect. The ratio is what says whether the
    comparison still has room to detect a real error, and a reviewer should not have
    to derive it from a bare max.

    Returns the measured max absolute difference so the caller can print a value
    instead of only a verdict.
    """
    x, w = _operands(seq, heads, kdim, ndim, seed)
    got = MA.mla_absorb(x, w)
    expected = MA.mla_absorb_torch_oracle(x, w)

    assert tuple(got.shape) == (seq, heads, ndim), (
        f"{label}: seam returned {tuple(got.shape)}, expected "
        f"{(seq, heads, ndim)}"
    )
    assert got.dtype == x.dtype, (
        f"{label}: the plan declares the output dtype is x.dtype "
        f"({x.dtype}); got {got.dtype}"
    )

    # The values are compared in float32. The seam's own output dtype is asserted
    # separately, just above, so widening here loses no reading -- and comparing a
    # bf16 result against a bf16-rounded oracle would hide the seam rounding badly,
    # which is the one thing the oracle exists to catch.
    got_f32 = got.to(torch.float32)
    err = (got_f32 - expected).abs()
    flat = int(err.argmax())
    diff = err.flatten()[flat].item()
    at_worst = expected.flatten()[flat].abs().item()
    allowance = ATOL + RTOL * at_worst
    ratio = diff / allowance if allowance > 0 else float("inf")

    torch.testing.assert_close(got_f32, expected, rtol=RTOL, atol=ATOL)
    print(
        f"    {label}: S={seq} H={heads} K={kdim} N={ndim} "
        f"ragged={','.join(ragged_axes(seq, kdim, ndim)) or 'none'} "
        f"max_abs_diff={diff:.8f} allowance_there={allowance:.8f} "
        f"used={ratio:.3f}_of_allowance"
    )
    return diff


def test_absorb_in_matches_the_fp32_oracle():
    """CONJUNCT 1 -- absorb-in at every declared S, and the kernel is ours.

    Counted reading: 3 shapes agree with the fp32 oracle, and the seam dispatched
    to NKI exactly once per call.

    THE KERNEL IDENTITY IS FOLDED IN HERE rather than made a fourth item, because
    the plan pins this file at three collected items. It does not move this item's
    3/3: it asserts that the kernel these three cases exercised is authored in this
    fork's own module and not imported from the substrate, which is the difference
    between this increment having done its work and having called someone else's.
    """
    module, qualname = MA.mla_absorb_kernel_identity()
    print(f"  KERNEL_IDENTITY module={module} qualname={qualname}")
    assert module == "vllm_neuron.functional.attention.mla_absorb", (
        f"the kernel under test is defined in {module}, not in this fork's "
        f"mla_absorb module -- an imported substrate kernel would report exactly "
        f"this way, which is why the identity is unwrapped and checked"
    )
    assert qualname == "mla_absorb_kernel"

    MA.reset_mla_absorb_dispatch_counters()
    agreed = 0
    for seq in SEQ_CASES:
        _one_case("absorb_in", seq, ABSORB_IN_K, ABSORB_IN_N, seed=9700 + seq)
        agreed += 1

    nki_dispatch, torch_fallback = MA.mla_absorb_dispatch_counters()
    print(f"  CONJUNCT1_SHAPES_AGREEING={agreed} of {len(SEQ_CASES)}")
    print(f"  CONJUNCT1_NKI_DISPATCH={nki_dispatch} "
          f"TORCH_FALLBACK={torch_fallback}")
    assert agreed == 3
    assert nki_dispatch == 3, (
        f"the plan's route predicate expects exactly one NKI dispatch per call, so "
        f"3 calls read 3; got {nki_dispatch}"
    )
    assert torch_fallback == 0, (
        f"this module has no torch absorb route, so this counter can only read 0 "
        f"(P13); got {torch_fallback}"
    )

    # THE RAGGED SET, counted separately so the declared 3/3 above cannot move. The
    # counter is re-read from a fresh reset for the same reason: the plan's route
    # predicate says conjunct 1 reads 3, and it must keep reading 3 no matter how
    # many extra shapes this item exercises.
    MA.reset_mla_absorb_dispatch_counters()
    ragged_agreed = 0
    covered: set[str] = set()
    for label, seq, heads, kdim, ndim in RAGGED_CASES:
        _one_case(f"ragged/{label}", seq, kdim, ndim,
                  seed=9750 + ragged_agreed, heads=heads)
        covered.update(ragged_axes(seq, kdim, ndim))
        ragged_agreed += 1

    ragged_dispatch, ragged_fallback = MA.mla_absorb_dispatch_counters()
    print(f"  CONJUNCT1_RAGGED_SHAPES_AGREEING={ragged_agreed} of "
          f"{len(RAGGED_CASES)}  (counted separately from the declared 3)")
    print(f"  CONJUNCT1_RAGGED_AXES_COVERED={sorted(covered)}")
    print(f"  CONJUNCT1_RAGGED_NKI_DISPATCH={ragged_dispatch} "
          f"TORCH_FALLBACK={ragged_fallback}")
    assert ragged_agreed == len(RAGGED_CASES)
    assert ragged_dispatch == len(RAGGED_CASES)
    assert ragged_fallback == 0
    # All three tiled axes must actually be left ragged by this set. Without this,
    # deleting a case would quietly shrink the coverage instead of reddening a test,
    # which is how the sibling increment's ragged hole survived review in the first
    # place.
    assert covered == {"S", "K", "N"}, (
        f"the ragged set must leave a short final tile on every tiled axis; it "
        f"covers {sorted(covered)}"
    )


def test_absorb_out_matches_the_fp32_oracle():
    """CONJUNCT 2 -- absorb-out at every declared S, and the oracle can fail.

    Counted reading: 3 shapes agree with the fp32 oracle, and the seam dispatched
    to NKI exactly once per call.

    THE VACUITY GUARD IS FOLDED IN HERE, for the same reason the identity check sits
    in conjunct 1: three items are all this file may collect. Without it, conjunct 1
    and conjunct 2 agreeing six times over would ALSO be the reading produced by a
    comparison incapable of disagreeing, and the two cases would be
    indistinguishable. It does not move this item's 3/3.
    """
    MA.reset_mla_absorb_dispatch_counters()
    agreed = 0
    last_pair = None
    for seq in SEQ_CASES:
        _one_case("absorb_out", seq, ABSORB_OUT_K, ABSORB_OUT_N, seed=9800 + seq)
        agreed += 1
        last_pair = seq

    nki_dispatch, torch_fallback = MA.mla_absorb_dispatch_counters()
    print(f"  CONJUNCT2_SHAPES_AGREEING={agreed} of {len(SEQ_CASES)}")
    print(f"  CONJUNCT2_NKI_DISPATCH={nki_dispatch} "
          f"TORCH_FALLBACK={torch_fallback}")
    assert agreed == 3
    assert nki_dispatch == 3
    assert torch_fallback == 0

    # The guard: perturb ONE element of the oracle and require the same comparison
    # to reject it. The perturbation is 1.0 against unit-scale outputs, so it is far
    # outside rtol=1e-2 and the rejection is not a borderline reading.
    x, w = _operands(
        last_pair, DECLARED_HEADS, ABSORB_OUT_K, ABSORB_OUT_N, seed=9800 + last_pair
    )
    got = MA.mla_absorb(x, w).to(torch.float32)
    perturbed = MA.mla_absorb_torch_oracle(x, w)
    perturbed[0, 0, 0] += 1.0
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, perturbed, rtol=RTOL, atol=ATOL)
    print("  CONJUNCT2_THE_COMPARISON_CAN_REJECT=True "
          "(one oracle element moved by 1.0 against unit-scale outputs)")


def test_the_gate_refuses_by_name_and_never_dispatches():
    """CONJUNCT 3 -- four inadmissible geometries, each refused by name.

    Counted reading: 4 refusals raise `MlaAbsorbError`, and the dispatch counter
    reads 0 -- a refusal that had already incremented the counter would be a
    silently half-executed call.

    TWO DISCIPLINE CHECKS ARE FOLDED IN, and neither moves the counted 4/4. The
    first reads this file's own source and asserts no `parametrize`, so the
    plan-declared item count cannot drift silently. The second reads the seam
    module's source and asserts it carries no torch matmul outside the one function
    named as the oracle -- that is P13, a run-wide prohibition, so checking it is
    never out of scope for a file that would otherwise be the only thing standing
    between a torch fallback and a review gate.
    """
    MA.reset_mla_absorb_dispatch_counters()
    heads, kdim, ndim = DECLARED_HEADS, ABSORB_IN_K, ABSORB_IN_N
    good_x, good_w = _operands(4, heads, kdim, ndim, seed=9900)

    cases = (
        ("x_is_rank_2", good_x[:, 0, :], good_w),
        ("w_is_rank_2", good_x, good_w[0]),
        ("head_count_mismatch", good_x, good_w[: heads - 1]),
        ("contraction_mismatch", good_x, good_w[:, : kdim - 1, :]),
    )

    refused = 0
    for label, bad_x, bad_w in cases:
        with pytest.raises(MA.MlaAbsorbError) as excinfo:
            MA.mla_absorb(bad_x, bad_w)
        message = str(excinfo.value)
        assert message, f"{label}: refused with an empty message"
        print(f"    {label}: MlaAbsorbError -- {message.splitlines()[0]}")
        refused += 1

    nki_dispatch, torch_fallback = MA.mla_absorb_dispatch_counters()
    print(f"  CONJUNCT3_REFUSALS={refused} of {len(cases)}")
    print(f"  CONJUNCT3_NKI_DISPATCH={nki_dispatch} "
          f"TORCH_FALLBACK={torch_fallback}")
    assert refused == 4
    assert nki_dispatch == 0, (
        f"every gate check runs before the counter moves, so a refused call must "
        f"leave it at 0; got {nki_dispatch}"
    )
    assert torch_fallback == 0

    # THE ZERO ABOVE OWNS A FIRING CONTROL, and it runs AFTER the counted reading is
    # already asserted, so it cannot move it. A counter that was simply broken --
    # never incrementing at all -- would produce the same 0 as a gate that correctly
    # refuses before dispatching, and those two are the opposite of each other. So
    # the counter is reset and one ADMISSIBLE call is made: it must read 1.
    MA.reset_mla_absorb_dispatch_counters()
    MA.mla_absorb(good_x, good_w)
    control_dispatch, control_fallback = MA.mla_absorb_dispatch_counters()
    print(f"  CONJUNCT3_COUNTER_FIRES_ON_AN_ADMISSIBLE_CALL={control_dispatch} "
          f"TORCH_FALLBACK={control_fallback}")
    assert control_dispatch == 1, (
        "the zero above is only a reading if this same counter can be shown to "
        "move; it did not, so the gate reading proves nothing"
    )
    # And the torch-fallback zero is a reading for the same reason: it is one field
    # of a counter pair whose sibling has now been shown to record.
    assert control_fallback == 0

    # Folded check one: this file collects exactly the items the plan declares.
    own_module = inspect.getmodule(
        test_the_gate_refuses_by_name_and_never_dispatches
    )
    own_tree = ast.parse(inspect.getsource(own_module))
    parametrize_uses = sum(
        1
        for node in ast.walk(own_tree)
        if isinstance(node, ast.Attribute) and node.attr == "parametrize"
    )
    test_functions = sorted(
        node.name
        for node in own_tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )
    print(f"  FOLDED_TEST_FUNCTIONS={test_functions}")
    print(f"  FOLDED_PARAMETRIZE_USES={parametrize_uses}")
    assert parametrize_uses == 0, (
        "the plan pins this file at three collected items; a parametrized case "
        "collects as several, so the declared count would stop meaning what it says"
    )
    assert len(test_functions) == 3

    # THE ZERO ABOVE OWNS A FIRING CONTROL. The same predicate is run over a snippet
    # that DOES parametrize, and must report 1 -- otherwise a screen that can only
    # ever return 0 would read identically to a file that genuinely has none.
    planted = ast.parse(
        "import pytest\n"
        "@pytest.mark.parametrize('n', [1, 2])\n"
        "def test_planted(n):\n"
        "    pass\n"
    )
    planted_uses = sum(
        1
        for node in ast.walk(planted)
        if isinstance(node, ast.Attribute) and node.attr == "parametrize"
    )
    print(f"  FOLDED_PARAMETRIZE_SCREEN_FIRES_ON_A_PLANTED_CASE={planted_uses}")
    assert planted_uses == 1

    # Folded check two: P13. The seam module carries no torch matmul except inside
    # the single function named as the CPU oracle. The screen walks the AST rather
    # than grepping text, so a mention in a docstring or a comment -- and the module
    # has several, explaining why torch is absent -- cannot make it fire.
    seam_tree = ast.parse(inspect.getsource(MA))
    oracle_name = "mla_absorb_torch_oracle"
    offenders = []
    for node in seam_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == oracle_name:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.MatMult):
                offenders.append(f"{getattr(node, 'name', '?')}:@")
            if isinstance(sub, ast.Attribute) and sub.attr in (
                "bmm", "einsum", "matmul", "mm", "baddbmm",
            ):
                offenders.append(f"{getattr(node, 'name', '?')}:{sub.attr}")
    print(f"  FOLDED_P13_TORCH_MATMUL_OUTSIDE_THE_ORACLE={len(offenders)} "
          f"{offenders}")
    assert offenders == [], (
        f"P13: kernel-class work must not carry a torch matmul path; found "
        f"{offenders} outside {oracle_name}"
    )

    # And the P13 screen is shown to be capable of firing, on the oracle it
    # deliberately excludes -- otherwise a zero above would also be what a screen
    # that never fires produces.
    oracle_hits = [
        sub.attr
        for node in seam_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == oracle_name
        for sub in ast.walk(node)
        if isinstance(sub, ast.Attribute) and sub.attr == "einsum"
    ]
    print(f"  FOLDED_P13_SCREEN_FIRES_ON_THE_ORACLE={oracle_hits}")
    assert oracle_hits == ["einsum"]
