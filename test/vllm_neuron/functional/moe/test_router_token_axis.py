# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ``inc-glm53f-088`` -- the router seam serves ANY token extent.

What the increment does, in one sentence: the ``noaux_tc`` seam used to REFUSE any
token count that was not a multiple of 256, and it now pads the token axis at each
entry point to that entry's own tile multiple, runs the kernel, and slices the
outputs back.

Six declared conjuncts, six collected items, one item each and no
``parametrize`` (plan D1.2). Every item names the component it certifies (D1.4).

    1. tiny extents      -- T = 1 and T = 8, BOTH entry points, against the
                            shipped torch oracle at the declared comparator.
    2. each own target   -- one non-multiple T above 256, with each entry's pad
                            target read from ITS OWN tiling constant.
    3. the pad is unseen -- the first T rows are BIT-IDENTICAL between
                            pad-to-own-multiple and pad-to-next-higher-multiple.
    4. counted zero      -- 0 authored stages reduce across the token axis.
                            Item 3 is this zero's control (plan D1.5).
    5. route predicate   -- the seam's own counters, read from this module.
    6. narrow narrowing  -- the three other named raises still fire, 3/3.

Tier N, so the command is D1's Tier N template and every env var in it is
load-bearing:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    python -m pytest test/vllm_neuron/functional/moe/test_router_token_axis.py \\
      --timeout 60 -v -s

THE COMPARATOR IS NOT NEW (P9). ``assert_close(rtol=1e-2, atol=1e-5)`` and index
SET equality are the ``inc-glm53f-032`` seam's own declared readings, restated
here at new extents. Nothing is widened and no new tolerance is introduced.

WHY THE FIXTURE IS REBUILT HERE rather than imported. The conditioned ladder is
``test/vllm_neuron/model/glm5_next/test_router.py``'s
(``build_designed_logits``), and that module hardcodes ``T = 256`` and takes no
token count. This file needs arbitrary extents, and it lives in a different test
package -- a cross-package test-to-test import would make this file's collection
depend on that one's. The construction below is the same one, with the token
count as an argument, and it reads the seam's own constants rather than copies of
them.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import nki
import nki.simulator
import pytest
import torch

from vllm_neuron.functional.moe import router as seam
from vllm_neuron.functional.moe.router import (
    NOAUX_TC_DENOM_EPS,
    NOAUX_TC_K,
    NOAUX_TC_TILE,
    NoauxTcRouterError,
    noaux_tc_correct,
    noaux_tc_correct_torch_oracle,
    noaux_tc_dispatch_counters,
    noaux_tc_rmsnorm_router_topk,
    reset_noaux_tc_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# ---------------------------------------------------------------------------
# Declared values. The checkpoint's or the seam's; none is chosen here.
# ---------------------------------------------------------------------------

#: The pin's own ``n_routed_experts`` (``glm5_next/config.py:185``).
DECLARED_E = 288

#: The seam's declared gate-weight tolerances, order named inline per D3.
RTOL, ATOL = 1e-2, 1e-5

#: The checkpoint's own routing hyperparameters
#: (``glm5_next/config.py:143,147,148``).
DECLARED_TOP_K = 8
DECLARED_NORM_TOPK_PROB = True
DECLARED_ROUTED_SCALING_FACTOR = 2.5

#: The smallest hidden extent the substrate's unconditional ``H % 256 == 0``
#: assert admits (``rmsnorm_router_topk_tkg.py:159``).
TINY_H = 256

#: The fused entry's pad multiple, read from the seam rather than written as 256.
FUSED_MULTIPLE = seam._NOAUX_TC_T_MULTIPLE

# Fixture conditioning, carried from ``inc-glm53f-032``'s own construction.
WINNER_HI, WINNER_LO = 0.785, 0.750
LOSER_HI, LOSER_LO = 0.700, 0.100
BIAS_AMP = 0.040
FIXTURE_SEED = 88


def build_logits(tokens: int, seed: int = FIXTURE_SEED):
    """The ``-032`` conditioned ladder at an arbitrary token count.

    The CORRECTED score is what is constructed, because ``noaux_tc`` selects on
    ``sigmoid(logits) + bias``: the top-8 sit in ``[0.750, 0.785]`` and the rest
    in ``[0.100, 0.700]``, so the 8/9 boundary gap is 0.050 and no row's
    selection turns on round-off. Conditioning is a per-ROW property -- each row
    is its own permutation of one ladder -- so it holds at every ``tokens``.
    """
    gen = torch.Generator().manual_seed(seed)
    w_top = (WINNER_HI - WINNER_LO) / (NOAUX_TC_K - 1)
    w_lo = (LOSER_HI - LOSER_LO) / (DECLARED_E - NOAUX_TC_K - 1)

    ladder = torch.empty(DECLARED_E, dtype=torch.float32)
    for rank in range(DECLARED_E):
        if rank < NOAUX_TC_K:
            ladder[rank] = WINNER_HI - rank * w_top
        else:
            ladder[rank] = LOSER_HI - (rank - NOAUX_TC_K) * w_lo

    choice = torch.empty(tokens, DECLARED_E, dtype=torch.float32)
    for token in range(tokens):
        choice[token, torch.randperm(DECLARED_E, generator=gen)] = ladder

    bias = (
        (torch.rand(1, DECLARED_E, generator=gen) - 0.5) * 2.0 * BIAS_AMP
    ).to(torch.float32)
    scores = (choice - bias).clamp(1e-4, 1.0 - 1e-4)
    logits = torch.log(scores / (1.0 - scores))
    return logits.contiguous(), bias.contiguous()


def build_hidden(tokens: int, seed: int = FIXTURE_SEED):
    """A ``[1, T, TINY_H]`` hidden-states fixture for the fused entry."""
    gen = torch.Generator().manual_seed(seed)
    hidden_states = (
        torch.randn(1, tokens, TINY_H, generator=gen) * 0.5
    ).to(torch.bfloat16)
    gamma = torch.ones(1, TINY_H, dtype=torch.bfloat16)
    router_weights = (
        torch.randn(TINY_H, DECLARED_E, generator=gen) * 0.1
    ).to(torch.bfloat16)
    _, bias = build_logits(tokens, seed)
    return hidden_states, gamma, router_weights, bias


# ---------------------------------------------------------------------------
# The route instruments. Same four readings as ``inc-glm53f-032``'s test module,
# rebuilt here for the reason given in the module docstring.
# ---------------------------------------------------------------------------


class RouteInstrumentError(AssertionError):
    """A route reading that contradicts the declared predicate."""


class _SimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls for the duration.

    The INDEPENDENT instrument: the other readings are this campaign's own
    bookkeeping, and a counter this campaign increments cannot by itself prove a
    vendor kernel ran.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._real = None

    def __enter__(self) -> "_SimulatorCounter":
        self._real = nki.simulator.simulate_kernel
        real = self._real

        def counting(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        nki.simulator.simulate_kernel = counting
        return self

    def __exit__(self, *exc_info) -> None:
        nki.simulator.simulate_kernel = self._real


def _assert_route(sim: _SimulatorCounter, expected: int, label: str) -> str:
    """Read all four route instruments and return the reading for the transcript."""
    nki_dispatch, torch_fallback = noaux_tc_dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"[{label}] nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
        f"can_run_kernel={gate} simulate_kernel_calls={sim.calls}"
    )
    print(reading)
    if nki_dispatch != expected:
        raise RouteInstrumentError(
            f"{label}: seam dispatch counter read {nki_dispatch}, declared "
            f"{expected}. {reading}"
        )
    if torch_fallback != 0:
        raise RouteInstrumentError(
            f"{label}: torch-fallback counter read {torch_fallback}, declared 0. "
            f"A torch fallback for kernel-class work is a P13 violation, not a "
            f"degraded pass. {reading}"
        )
    if gate is not True:
        raise RouteInstrumentError(
            f"{label}: can_run_kernel() read {gate}, declared True. {reading}"
        )
    if sim.calls != expected:
        raise RouteInstrumentError(
            f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, "
            f"declared {expected}. A numeric pass without a simulator call is the "
            f"F1 false green. {reading}"
        )
    return reading


def set_equal_rows(got_index: torch.Tensor, want_index: torch.Tensor) -> int:
    """Rows whose selected expert index SETS are equal.

    Sets, not sequences: upstream selects with ``sorted=False`` and
    ``nisa.max8`` emits descending, so only the sets are comparable.
    """
    got = torch.sort(got_index.to(torch.int64), dim=-1)[0]
    want = torch.sort(want_index.to(torch.int64), dim=-1)[0]
    return int((got == want).all(dim=-1).sum())


# ---------------------------------------------------------------------------
# The two arms, one per entry point. Helpers, so each conjunct below is one
# collected item that reads them rather than a parametrised family.
# ---------------------------------------------------------------------------


def _correct_arm(tokens: int, label: str) -> None:
    """Entry 1 -- ``noaux_tc_correct`` at ``tokens``, against its shipped oracle.

    The oracle reads the SAME logits the kernel read, so this arm measures the
    authored stage and not a torch recomputation of anything.
    """
    logits, bias = build_logits(tokens)
    want_index, want_affinities = noaux_tc_correct_torch_oracle(
        logits, bias, DECLARED_NORM_TOPK_PROB, DECLARED_ROUTED_SCALING_FACTOR
    )

    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim:
        got_index, got_affinities = noaux_tc_correct(
            logits,
            bias,
            top_k=DECLARED_TOP_K,
            norm_topk_prob=DECLARED_NORM_TOPK_PROB,
            routed_scaling_factor=DECLARED_ROUTED_SCALING_FACTOR,
        )
    _assert_route(sim, 1, label)

    got = got_affinities.to(torch.float32)
    want = want_affinities.to(torch.float32)
    assert tuple(got_index.shape) == (tokens, NOAUX_TC_K)
    assert got.shape == want.shape == (tokens, DECLARED_E)

    equal_rows = set_equal_rows(got_index, want_index)
    print(
        f"[{label}] T={tokens} pad_target="
        f"{seam._noaux_tc_pad_target(tokens, NOAUX_TC_TILE)} "
        f"set_equal_rows={equal_rows}/{tokens} "
        f"max_abs_error={float((got - want).abs().max()):.6e}"
    )
    assert equal_rows == tokens
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def _fused_arm(tokens: int, label: str) -> None:
    """Entry 2 -- the fused seam at ``tokens``, against its shipped oracle.

    The reference reads the logits the KERNEL returned, the form
    ``test_router.py::test_fused_seam_matches_the_reference_on_its_own_logits``
    established: it isolates the authored correction from the substrate's bf16
    matmul, whose near-tie flips would otherwise be charged to this increment.
    """
    hidden_states, gamma, router_weights, bias = build_hidden(tokens)

    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim:
        logits, got_index, got_affinities, substrate_index = (
            noaux_tc_rmsnorm_router_topk(
                hidden_states=hidden_states,
                gamma=gamma,
                router_weights=router_weights,
                correction_bias=bias,
                top_k=DECLARED_TOP_K,
                norm_topk_prob=DECLARED_NORM_TOPK_PROB,
                routed_scaling_factor=DECLARED_ROUTED_SCALING_FACTOR,
            )
        )
    _assert_route(sim, 1, label)

    # ALL FOUR outputs are sliced back, so all four are asserted at the caller's
    # extent. `substrate_index` included: it is the non-vacuity control, and a
    # control left at the padded length would be compared row-for-row against a
    # shorter selection.
    assert tuple(logits.shape) == (tokens, DECLARED_E)
    assert tuple(got_index.shape) == (tokens, NOAUX_TC_K)
    assert tuple(got_affinities.shape) == (tokens, DECLARED_E)
    assert tuple(substrate_index.shape) == (tokens, NOAUX_TC_K)
    assert bool(torch.isfinite(logits).all())

    want_index, want_affinities = noaux_tc_correct_torch_oracle(
        logits, bias, DECLARED_NORM_TOPK_PROB, DECLARED_ROUTED_SCALING_FACTOR
    )
    got = got_affinities.to(torch.float32)
    want = want_affinities.to(torch.float32)
    equal_rows = set_equal_rows(got_index, want_index)
    print(
        f"[{label}] T={tokens} pad_target="
        f"{seam._noaux_tc_pad_target(tokens, FUSED_MULTIPLE)} "
        f"set_equal_rows={equal_rows}/{tokens} "
        f"max_abs_error={float((got - want).abs().max()):.6e}"
    )
    assert equal_rows == tokens
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


# ===========================================================================
# CONJUNCT 1 -- the extents the old clause refused outright.
# ===========================================================================


def test_tiny_token_extents_match_the_oracle_on_both_entries() -> None:
    """``T = 1`` and ``T = 8``, both entry points, at the declared comparator.

    These are the extents a single-token decode step and a small batch actually
    present, and both were a named refusal before this increment. Both entries
    are read because they pad to DIFFERENT targets and a change that served one
    would leave the other refusing.
    """
    for tokens in (1, 8):
        _correct_arm(tokens, f"tiny-correct-T{tokens}")
        _fused_arm(tokens, f"tiny-fused-T{tokens}")


# ===========================================================================
# CONJUNCT 2 -- each entry's pad target is its own, read from its own constant.
# ===========================================================================


def test_a_non_multiple_extent_above_256_pads_to_each_entrys_own_target() -> None:
    """``T = 300``: a non-multiple ABOVE the old 256, and the two targets differ.

    The two numbers are recorded beside the constant each is read from, because
    the whole content of this conjunct is that they are NOT interchangeable:

        entry 1 (no grid, one program owns every token)
            multiple = NOAUX_TC_TILE            -> 300 pads to 384
        entry 2 (`[2]` grid, the stage shards the tokens)
            multiple = _NOAUX_TC_T_MULTIPLE     -> 300 pads to 512

    Padding entry 2 to 384 instead would hand each core 192 rows, and
    ``192 // NOAUX_TC_TILE`` is 1 tile covering 128 of them -- 64 rows per core
    silently uncomputed behind a green route reading.
    """
    tokens = 300
    correct_target = seam._noaux_tc_pad_target(tokens, NOAUX_TC_TILE)
    fused_target = seam._noaux_tc_pad_target(tokens, FUSED_MULTIPLE)
    print(
        f"[pad-targets] T={tokens} "
        f"NOAUX_TC_TILE={NOAUX_TC_TILE} -> correct_pad={correct_target} "
        f"_NOAUX_TC_T_MULTIPLE={FUSED_MULTIPLE} -> fused_pad={fused_target} "
        f"per_core_rows={fused_target // 2} "
        f"tiles_per_core={(fused_target // 2) // NOAUX_TC_TILE}"
    )
    assert tokens % NOAUX_TC_TILE != 0 and tokens % FUSED_MULTIPLE != 0
    assert correct_target == 384 and correct_target % NOAUX_TC_TILE == 0
    assert fused_target == 512 and fused_target % FUSED_MULTIPLE == 0
    assert correct_target != fused_target
    # The fused half must itself be a whole tile. This is the inequality that
    # makes the two constants non-interchangeable.
    assert (fused_target // 2) % NOAUX_TC_TILE == 0
    assert (correct_target // 2) % NOAUX_TC_TILE != 0

    _correct_arm(tokens, f"nonmultiple-correct-T{tokens}")
    _fused_arm(tokens, f"nonmultiple-fused-T{tokens}")


# ===========================================================================
# CONJUNCT 3 -- the pad is invisible. This is also conjunct 4's control.
# ===========================================================================


def test_the_pad_is_invisible_to_the_real_rows(monkeypatch) -> None:
    """The first ``T`` rows are BIT-IDENTICAL under two different pad lengths.

    ``T = 100`` pads to 128 on its own multiple. The second run is forced one
    whole multiple higher, to 256, by patching the pad target -- nothing else
    changes, same fixture, same entry point. If any stage reduced across the
    token axis, 156 pad rows instead of 28 would move the real rows.

    Bit-identical is the reading, not "within tolerance": max abs diff exactly
    0.0 on the weights and exact equality on the indices. A tolerance here would
    let a real coupling hide under it.
    """
    tokens = 100
    own = seam._noaux_tc_pad_target(tokens, NOAUX_TC_TILE)
    logits, bias = build_logits(tokens)

    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim:
        index_own, aff_own = noaux_tc_correct(
            logits,
            bias,
            top_k=DECLARED_TOP_K,
            norm_topk_prob=DECLARED_NORM_TOPK_PROB,
            routed_scaling_factor=DECLARED_ROUTED_SCALING_FACTOR,
        )
    _assert_route(sim, 1, f"pad-own-{own}")

    real_target = seam._noaux_tc_pad_target

    def one_multiple_higher(num_tokens: int, multiple: int) -> int:
        return real_target(num_tokens, multiple) + multiple

    monkeypatch.setattr(seam, "_noaux_tc_pad_target", one_multiple_higher)
    higher = one_multiple_higher(tokens, NOAUX_TC_TILE)

    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim:
        index_hi, aff_hi = noaux_tc_correct(
            logits,
            bias,
            top_k=DECLARED_TOP_K,
            norm_topk_prob=DECLARED_NORM_TOPK_PROB,
            routed_scaling_factor=DECLARED_ROUTED_SCALING_FACTOR,
        )
    _assert_route(sim, 1, f"pad-higher-{higher}")

    max_abs = float((aff_own.to(torch.float32) - aff_hi.to(torch.float32)).abs().max())
    index_equal = int((index_own == index_hi).all(dim=-1).sum())
    print(
        f"[pad-invisible] T={tokens} own_pad={own} higher_pad={higher} "
        f"pad_rows={own - tokens} vs {higher - tokens} "
        f"max_abs_diff={max_abs} identical_index_rows={index_equal}/{tokens}"
    )
    assert higher == own + NOAUX_TC_TILE
    assert tuple(aff_own.shape) == tuple(aff_hi.shape) == (tokens, DECLARED_E)
    assert max_abs == 0.0
    assert index_equal == tokens


# ===========================================================================
# CONJUNCT 4 -- the counted zero. Its control is the conjunct above.
# ===========================================================================

#: Members that reduce a NAMED axis. On this seam ``axis=1`` is the expert (free)
#: axis and ``axis=0`` would be the token (partition) axis.
_AXIS_REDUCERS = {"sum", "mean", "max", "min", "prod", "all", "any"}
#: Members that contract the PARTITION axis by construction -- a matmul against
#: the token axis is a token-axis reduction whether or not it names one.
_PARTITION_CONTRACTORS = {"nc_matmul", "dot", "matmul"}
#: Per-partition ISA members: they reduce the FREE axis within one partition and
#: are named here so the census cannot be accused of overlooking them.
_PER_PARTITION = {"max8", "nc_find_index8"}

_AUTHORED_STAGES = (
    "_noaux_tc_stage",
    "_noaux_tc_correct_nki",
    "_noaux_tc_rmsnorm_router_topk_nki",
)


def test_no_authored_stage_reduces_across_the_token_axis() -> None:
    """COUNTED ZERO: 0 of the authored stages reduce across the token axis.

    A census of the seam's own source, parsed rather than read by eye: every call
    in the three authored stages is classified, and the population is printed
    BEFORE the zero so an empty parse cannot pass as clean.

    THE ZERO'S CONTROL IS ``test_the_pad_is_invisible_to_the_real_rows`` above
    (plan D1.5). That item MOVES this property: a token-axis reduction would make
    the real rows depend on how many pad rows follow them, and it would fail
    while this census still parsed cleanly. Neither reading stands alone -- this
    one says what the source does, that one says what the kernel computed.
    """
    tree = ast.parse(Path(inspect.getsourcefile(seam)).read_text(encoding="utf-8"))
    bodies = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in _AUTHORED_STAGES
    }
    assert sorted(bodies) == sorted(_AUTHORED_STAGES), (
        f"census population is wrong: parsed {sorted(bodies)}, declared "
        f"{sorted(_AUTHORED_STAGES)}"
    )

    axis_reducers: list[str] = []
    token_axis: list[str] = []
    per_partition: list[str] = []
    for name, node in bodies.items():
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            member = getattr(call.func, "attr", getattr(call.func, "id", ""))
            axis = next(
                (kw.value for kw in call.keywords if kw.arg == "axis"), None
            )
            axis_value = axis.value if isinstance(axis, ast.Constant) else None
            where = f"{name}:{member}(axis={axis_value})"
            if member in _PER_PARTITION:
                per_partition.append(f"{name}:{member}")
            elif member in _PARTITION_CONTRACTORS:
                token_axis.append(where)
            elif member in _AXIS_REDUCERS:
                axis_reducers.append(where)
                if axis_value != 1:
                    token_axis.append(where)

    print(
        f"[token-axis-census] stages={len(bodies)} "
        f"axis_reducers={len(axis_reducers)} {axis_reducers} "
        f"per_partition_members={len(per_partition)} {per_partition} "
        f"partition_contractors_in_authored_source=0 "
        f"TOKEN_AXIS_REDUCTIONS={len(token_axis)} {token_axis}"
    )
    # Population first: a zero over an empty population is not a measurement.
    assert len(axis_reducers) >= 1, "no reduction was found at all; census vacuous"
    assert len(per_partition) >= 1, "no ISA top-K member found; wrong source parsed"
    assert len(token_axis) == 0


# ===========================================================================
# CONJUNCT 5 -- the route predicate, in this module's own counted values.
# ===========================================================================


def test_the_route_predicate_reads_its_declared_counted_values() -> None:
    """D13 form R-2: the seam's counters, read from THIS module, at a padded extent.

    ``T = 8`` needs a pad on both entries, so this is the route reading at exactly
    the extents the increment adds. Two entries, one dispatch each, and the
    counters accumulate to ``(2, 0)`` across them -- a fallback would show as a
    non-zero second value while every numeric arm above still passed.
    """
    reset_noaux_tc_counters()
    assert noaux_tc_dispatch_counters() == (0, 0)

    logits, bias = build_logits(8)
    with _SimulatorCounter() as sim_one:
        noaux_tc_correct(logits, bias)
    after_correct = noaux_tc_dispatch_counters()

    hidden_states, gamma, router_weights, fused_bias = build_hidden(8)
    with _SimulatorCounter() as sim_two:
        noaux_tc_rmsnorm_router_topk(
            hidden_states=hidden_states,
            gamma=gamma,
            router_weights=router_weights,
            correction_bias=fused_bias,
        )
    after_fused = noaux_tc_dispatch_counters()

    print(
        f"[route-predicate] after_reset=(0, 0) after_correct={after_correct} "
        f"after_fused={after_fused} can_run_kernel={can_run_kernel(torch.zeros(1))} "
        f"simulate_kernel_calls={sim_one.calls + sim_two.calls}"
    )
    assert after_correct == (1, 0)
    assert after_fused == (2, 0)
    assert sim_one.calls == 1 and sim_two.calls == 1


# ===========================================================================
# CONJUNCT 6 -- the narrowing is narrow.
# ===========================================================================


def test_the_narrowing_is_narrow_the_other_raises_still_fire() -> None:
    """3/3 of the seam's other named refusals still raise, unchanged.

    Removing one clause from a function of four is one edit away from removing
    the function's warrant, so the three surviving clauses are read here rather
    than assumed. ``NOAUX_TC_DENOM_EPS`` is imported and asserted alongside them:
    it is the one numeric constant a "just make the extents work" edit could have
    rounded, and the fork's output is compared against upstream's.
    """
    cases = [
        (DECLARED_E, 4, "top_k must be exactly 8"),
        (4, DECLARED_TOP_K, "E must be >= 8"),
        (513, DECLARED_TOP_K, "E must be <= 512"),
    ]
    fired = 0
    for experts, top_k, needle in cases:
        logits = torch.zeros(256, experts, dtype=torch.float32)
        bias = torch.zeros(1, experts, dtype=torch.float32)
        reset_noaux_tc_counters()
        with pytest.raises(NoauxTcRouterError) as excinfo:
            noaux_tc_correct(logits, bias, top_k=top_k)
        message = str(excinfo.value)
        print(f"[still-refused] E={experts} k={top_k} -> {message[:100]}")
        assert needle in message
        # A refusal is not a fallback: neither counter moves.
        assert noaux_tc_dispatch_counters() == (0, 0)
        fired += 1

    print(f"[still-refused] raises_fired={fired}/{len(cases)} "
          f"NOAUX_TC_DENOM_EPS={NOAUX_TC_DENOM_EPS}")
    assert fired == 3
    assert NOAUX_TC_DENOM_EPS == 1e-20
