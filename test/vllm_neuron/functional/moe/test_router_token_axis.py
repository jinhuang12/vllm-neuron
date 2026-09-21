# SPDX-License-Identifier: Apache-2.0
"""Token-extent coverage for the two ``noaux_tc`` router entry points.

Each entry point pads the token axis to its own tile multiple, runs the NKI
stage and slices the outputs back, so any token count is served. The logit
fixture is rebuilt here instead of imported from
``test/vllm_neuron/model/glm5_next/test_router.py``, which fixes ``T = 256``
and takes no token count; the construction below is that one with the token
count as an argument.
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
    noaux_tc_rmsnorm_router_topk_torch_oracle,
    reset_noaux_tc_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# ---------------------------------------------------------------------------
# Fixture and comparison constants.
# ---------------------------------------------------------------------------

#: ``n_routed_experts`` from ``glm5_next/config.py:185``.
NUM_EXPERTS = 288

#: Tolerances for the bf16 gate-weight comparison.
RTOL, ATOL = 1e-2, 1e-5

#: Routing hyperparameters from ``glm5_next/config.py:143,147,148``.
TOP_K = 8
NORM_TOPK_PROB = True
ROUTED_SCALING_FACTOR = 2.5

#: The smallest hidden extent the substrate's unconditional ``H % 256 == 0``
#: assert admits (``rmsnorm_router_topk_tkg.py:159``).
TINY_H = 256

#: The fused entry's pad multiple, read from the module rather than written as 256.
FUSED_MULTIPLE = seam._NOAUX_TC_T_MULTIPLE

# Logit conditioning, carried over from the router module's own fixture.
WINNER_HI, WINNER_LO = 0.785, 0.750
LOSER_HI, LOSER_LO = 0.700, 0.100
BIAS_AMP = 0.040
FIXTURE_SEED = 88


def build_logits(tokens: int, seed: int = FIXTURE_SEED):
    """Logits whose corrected scores form a well-separated ladder.

    ``noaux_tc`` selects on ``sigmoid(logits) + bias``, so the ladder is built in
    that space: the top 8 sit in ``[0.750, 0.785]`` and the rest in
    ``[0.100, 0.700]``, which leaves a 0.050 gap at the 8/9 boundary and keeps no
    row's selection on the edge of round-off. The conditioning is a per-row
    property -- each row is its own permutation of one ladder -- so it holds at
    every ``tokens``.
    """
    gen = torch.Generator().manual_seed(seed)
    w_top = (WINNER_HI - WINNER_LO) / (NOAUX_TC_K - 1)
    w_lo = (LOSER_HI - LOSER_LO) / (NUM_EXPERTS - NOAUX_TC_K - 1)

    ladder = torch.empty(NUM_EXPERTS, dtype=torch.float32)
    for rank in range(NUM_EXPERTS):
        if rank < NOAUX_TC_K:
            ladder[rank] = WINNER_HI - rank * w_top
        else:
            ladder[rank] = LOSER_HI - (rank - NOAUX_TC_K) * w_lo

    choice = torch.empty(tokens, NUM_EXPERTS, dtype=torch.float32)
    for token in range(tokens):
        choice[token, torch.randperm(NUM_EXPERTS, generator=gen)] = ladder

    bias = (
        (torch.rand(1, NUM_EXPERTS, generator=gen) - 0.5) * 2.0 * BIAS_AMP
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
        torch.randn(TINY_H, NUM_EXPERTS, generator=gen) * 0.1
    ).to(torch.bfloat16)
    _, bias = build_logits(tokens, seed)
    return hidden_states, gamma, router_weights, bias


# ---------------------------------------------------------------------------
# Route checks shared by both entry points.
# ---------------------------------------------------------------------------


class RouteCheckError(AssertionError):
    """A dispatch, fallback or simulator count that disagrees with the route."""


class _SimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls while active.

    The router's dispatch counters are bookkeeping inside the module under test;
    this count is the independent evidence that a kernel body ran at all.
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


def _assert_route(sim: _SimulatorCounter, expected: int, label: str) -> None:
    """The kernel route was taken ``expected`` times and nothing fell back."""
    nki_dispatch, torch_fallback = noaux_tc_dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"[{label}] nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
        f"can_run_kernel={gate} simulate_kernel_calls={sim.calls}"
    )
    if nki_dispatch != expected:
        raise RouteCheckError(
            f"{label}: dispatch counter read {nki_dispatch}, expected "
            f"{expected}. {reading}"
        )
    if torch_fallback != 0:
        raise RouteCheckError(
            f"{label}: torch-fallback counter read {torch_fallback}, expected 0, "
            f"so the kernel path was not the one taken. {reading}"
        )
    if gate is not True:
        raise RouteCheckError(
            f"{label}: can_run_kernel() read {gate}, expected True. {reading}"
        )
    if sim.calls != expected:
        raise RouteCheckError(
            f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, "
            f"expected {expected}; a numeric comparison that never entered the "
            f"simulator measures nothing. {reading}"
        )


def set_equal_rows(got_index: torch.Tensor, expected_index: torch.Tensor) -> int:
    """Rows whose selected expert index SETS are equal.

    Sets, not sequences: upstream selects with ``sorted=False`` and ``nisa.max8``
    emits descending, so only the sets are comparable.
    """
    got = torch.sort(got_index.to(torch.int64), dim=-1)[0]
    expected = torch.sort(expected_index.to(torch.int64), dim=-1)[0]
    return int((got == expected).all(dim=-1).sum())


# ---------------------------------------------------------------------------
# One helper per entry point.
# ---------------------------------------------------------------------------


def _check_correct_entry(tokens: int, label: str) -> None:
    """``noaux_tc_correct`` at ``tokens``, against its shipped torch reference.

    The reference reads the same logits the kernel read, so this measures the NKI
    stage and not a torch recomputation of the fixture.
    """
    logits, bias = build_logits(tokens)
    expected_index, expected_affinities = noaux_tc_correct_torch_oracle(
        logits, bias, NORM_TOPK_PROB, ROUTED_SCALING_FACTOR
    )

    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim:
        got_index, got_affinities = noaux_tc_correct(
            logits,
            bias,
            top_k=TOP_K,
            norm_topk_prob=NORM_TOPK_PROB,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
        )
    _assert_route(sim, 1, label)

    got = got_affinities.to(torch.float32)
    expected = expected_affinities.to(torch.float32)
    assert tuple(got_index.shape) == (tokens, NOAUX_TC_K)
    assert got.shape == expected.shape == (tokens, NUM_EXPERTS)
    assert set_equal_rows(got_index, expected_index) == tokens
    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)


def _check_fused_entry(tokens: int, label: str) -> None:
    """The fused entry at ``tokens``, against the independent torch reference.

    The reference is built from the same hidden states, gamma, router weights and
    bias the kernel received, never from the kernel's own returned logits. The
    pad repeats the last real row, so an entry that handed back the padded tail
    would return T copies of one row and still match a reference recomputed from
    those same rows; a reference built from the inputs knows what row 0 holds.
    """
    hidden_states, gamma, router_weights, bias = build_hidden(tokens)
    # The entry point's own default, read rather than copied, so the reference
    # normalises with exactly the epsilon the kernel used.
    seam_eps = inspect.signature(
        noaux_tc_rmsnorm_router_topk
    ).parameters["eps"].default

    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim:
        logits, got_index, got_affinities, substrate_index = (
            noaux_tc_rmsnorm_router_topk(
                hidden_states=hidden_states,
                gamma=gamma,
                router_weights=router_weights,
                correction_bias=bias,
                top_k=TOP_K,
                eps=seam_eps,
                norm_topk_prob=NORM_TOPK_PROB,
                routed_scaling_factor=ROUTED_SCALING_FACTOR,
            )
        )
    _assert_route(sim, 1, label)

    # All four outputs are sliced back, so all four are read at the caller's
    # extent -- `substrate_index` included, which would otherwise stay at the
    # padded length and be compared row-for-row against a shorter selection.
    assert tuple(logits.shape) == (tokens, NUM_EXPERTS)
    assert tuple(got_index.shape) == (tokens, NOAUX_TC_K)
    assert tuple(got_affinities.shape) == (tokens, NUM_EXPERTS)
    assert tuple(substrate_index.shape) == (tokens, NOAUX_TC_K)
    assert bool(torch.isfinite(logits).all())

    expected_logits, expected_index, expected_affinities, _ = (
        noaux_tc_rmsnorm_router_topk_torch_oracle(
            hidden_states,
            gamma,
            router_weights,
            bias,
            seam_eps,
            NORM_TOPK_PROB,
            ROUTED_SCALING_FACTOR,
        )
    )
    got_logits32 = logits.to(torch.float32)
    expected_logits32 = expected_logits.to(torch.float32)
    got_affinities32 = got_affinities.to(torch.float32)
    expected_affinities32 = expected_affinities.to(torch.float32)

    # An all-zero reference would pass every comparison below.
    assert float(expected_logits32.abs().max()) > 0.0
    # Row identity: a tail slice, or any other permutation of the token axis,
    # fails here instead of passing.
    assert set_equal_rows(got_index, expected_index) == tokens
    torch.testing.assert_close(
        got_logits32, expected_logits32, rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(
        got_affinities32, expected_affinities32, rtol=RTOL, atol=ATOL
    )
    # `substrate_index` is top-K on the RAW logits, where this fixture builds no
    # boundary gap at all, so only its shape is read here: an index comparison
    # there would turn on tie behaviour nothing in this file fixes.


def test_tiny_token_extents_match_the_torch_reference_on_both_entries() -> None:
    """``T = 1`` and ``T = 8`` match the torch reference on both entry points."""
    for tokens in (1, 8):
        _check_correct_entry(tokens, f"tiny-correct-T{tokens}")
        _check_fused_entry(tokens, f"tiny-fused-T{tokens}")


def test_a_non_multiple_token_extent_pads_to_each_entrys_own_target() -> None:
    """``T = 300`` pads to 384 on the correct-only entry and to 512 on the fused."""
    # The two targets are not interchangeable. `noaux_tc_correct` launches with no
    # grid, so one whole 128-row tile (`NOAUX_TC_TILE`) is enough and 300 rounds
    # to 384. The fused entry launches `[2]` and the stage splits the tokens
    # across the two cores, so each core needs a whole tile and the padded extent
    # must be a multiple of 256 (`_NOAUX_TC_T_MULTIPLE`), which rounds 300 to 512.
    # Padding the fused entry to 384 would give each core 192 rows, of which one
    # tile covers 128, leaving 64 rows per core uncomputed.
    tokens = 300
    correct_target = seam._noaux_tc_pad_target(tokens, NOAUX_TC_TILE)
    fused_target = seam._noaux_tc_pad_target(tokens, FUSED_MULTIPLE)
    assert tokens % NOAUX_TC_TILE != 0 and tokens % FUSED_MULTIPLE != 0
    assert correct_target == 384 and correct_target % NOAUX_TC_TILE == 0
    assert fused_target == 512 and fused_target % FUSED_MULTIPLE == 0
    assert correct_target != fused_target
    # The fused half must itself be a whole tile; the other target's half is not.
    assert (fused_target // 2) % NOAUX_TC_TILE == 0
    assert (correct_target // 2) % NOAUX_TC_TILE != 0

    _check_correct_entry(tokens, f"nonmultiple-correct-T{tokens}")
    _check_fused_entry(tokens, f"nonmultiple-fused-T{tokens}")


def _pad_differential(monkeypatch, entry: str, tokens: int) -> None:
    """One entry, one ``T``, two pad lengths: the first ``T`` rows must not move.

    The same call is made twice with nothing changed but the pad length, the
    second forced one whole multiple higher by patching the pad-target helper. If
    any stage reduced across the token axis, a different number of pad rows would
    move the real rows. Bit-identity is the reading rather than a tolerance --
    exactly 0.0 on every float output, exact equality on every index output --
    and every sliced output is compared, not just one, so a coupling in the
    logits or in the substrate selection cannot hide.
    """
    multiple = NOAUX_TC_TILE if entry == "correct" else FUSED_MULTIPLE
    real_target = seam._noaux_tc_pad_target
    own = real_target(tokens, multiple)
    higher = own + multiple

    if entry == "correct":
        logits, bias = build_logits(tokens)

        def once():
            reset_noaux_tc_counters()
            with _SimulatorCounter() as sim:
                index, aff = noaux_tc_correct(
                    logits,
                    bias,
                    top_k=TOP_K,
                    norm_topk_prob=NORM_TOPK_PROB,
                    routed_scaling_factor=ROUTED_SCALING_FACTOR,
                )
            return sim, (index,), (aff,)
    else:
        hidden_states, gamma, router_weights, bias = build_hidden(tokens)
        seam_eps = inspect.signature(
            noaux_tc_rmsnorm_router_topk
        ).parameters["eps"].default

        def once():
            reset_noaux_tc_counters()
            with _SimulatorCounter() as sim:
                lg, index, aff, sub = noaux_tc_rmsnorm_router_topk(
                    hidden_states=hidden_states,
                    gamma=gamma,
                    router_weights=router_weights,
                    correction_bias=bias,
                    top_k=TOP_K,
                    eps=seam_eps,
                    norm_topk_prob=NORM_TOPK_PROB,
                    routed_scaling_factor=ROUTED_SCALING_FACTOR,
                )
            return sim, (index, sub), (lg, aff)

    sim, ints_own, floats_own = once()
    _assert_route(sim, 1, f"pad-own-{entry}-T{tokens}-{own}")

    def one_multiple_higher(num_tokens: int, mult: int) -> int:
        return real_target(num_tokens, mult) + mult

    monkeypatch.setattr(seam, "_noaux_tc_pad_target", one_multiple_higher)
    try:
        sim, ints_hi, floats_hi = once()
    finally:
        # Undone here rather than at teardown, so the next token count in the
        # caller's loop measures its own pad target and not a doubly-patched one.
        monkeypatch.undo()
    _assert_route(sim, 1, f"pad-higher-{entry}-T{tokens}-{higher}")

    max_abs = max(
        float((a.to(torch.float32) - b.to(torch.float32)).abs().max())
        for a, b in zip(floats_own, floats_hi)
    )
    index_equal = min(
        int((a == b).all(dim=-1).sum()) for a, b in zip(ints_own, ints_hi)
    )
    assert higher == own + multiple
    assert one_multiple_higher(tokens, multiple) == higher
    for a, b in zip(floats_own + ints_own, floats_hi + ints_hi):
        assert tuple(a.shape) == tuple(b.shape)
        assert a.shape[0] == tokens, (
            f"{entry} at T={tokens} returned {a.shape[0]} rows, not the caller's "
            f"extent, so the slice back is wrong"
        )
    assert max_abs == 0.0
    assert index_equal == tokens


def test_the_pad_is_invisible_to_the_real_rows(monkeypatch) -> None:
    """The first ``T`` rows are bit-identical under two different pad lengths."""
    for tokens in (1, 8, 100, 300):
        _pad_differential(monkeypatch, "correct", tokens)
    # The fused entry is read at T = 300, where the two pad lengths are 512 and
    # 768: its `[2]` grid then gives 256 rows per core against 384, so the two
    # calls shard the real tokens differently. The grid-free entry only appends
    # rows, so a longer pad there cannot move a shard boundary at all.
    _pad_differential(monkeypatch, "fused", 300)


#: Members that reduce a named axis. In these stages ``axis=1`` is the expert
#: (free) axis and ``axis=0`` would be the token (partition) axis.
_AXIS_REDUCERS = {"sum", "mean", "max", "min", "prod", "all", "any"}
#: Members that contract the partition axis by construction: a matmul against the
#: token axis reduces that axis whether or not it names one.
_PARTITION_CONTRACTORS = {"nc_matmul", "dot", "matmul"}
#: Per-partition ISA members: they reduce the free axis within one partition, so
#: they are classified apart from the reducers above rather than overlooked.
_PER_PARTITION = {"max8", "nc_find_index8"}

_STAGE_FUNCTIONS = (
    "_noaux_tc_stage",
    "_noaux_tc_correct_nki",
    "_noaux_tc_rmsnorm_router_topk_nki",
)


def test_no_stage_reduces_across_the_token_axis() -> None:
    """No call in the NKI stages reduces or contracts the token axis."""
    tree = ast.parse(Path(inspect.getsourcefile(seam)).read_text(encoding="utf-8"))
    bodies = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in _STAGE_FUNCTIONS
    }
    assert sorted(bodies) == sorted(_STAGE_FUNCTIONS), (
        f"parsed {sorted(bodies)}, expected {sorted(_STAGE_FUNCTIONS)}"
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

    # An empty parse would find nothing to classify and read as clean, so what
    # the scan did find is asserted before the zero.
    assert len(axis_reducers) >= 1, "no reduction was found at all"
    assert len(per_partition) >= 1, "no ISA top-K member found; wrong source parsed"
    assert token_axis == [], f"these calls reduce across the token axis: {token_axis}"


def test_the_seam_dispatches_to_the_kernel_and_does_not_fall_back() -> None:
    """Each entry point counts one kernel dispatch and no torch fallback."""
    # T = 8 pads on both entry points, so the counters are read at a padded extent.
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

    # The counters accumulate across the two calls, so the second reads (2, 0).
    # A fallback would show as a non-zero second value.
    assert after_correct == (1, 0)
    assert after_fused == (2, 0)
    assert sim_one.calls == 1 and sim_two.calls == 1


def test_the_other_named_refusals_still_raise() -> None:
    """The router's three other named refusals still raise ``NoauxTcRouterError``."""
    cases = [
        (NUM_EXPERTS, 4, "top_k must be exactly 8"),
        (4, TOP_K, "E must be >= 8"),
        (513, TOP_K, "E must be <= 512"),
    ]
    fired = 0
    for experts, top_k, needle in cases:
        logits = torch.zeros(256, experts, dtype=torch.float32)
        bias = torch.zeros(1, experts, dtype=torch.float32)
        reset_noaux_tc_counters()
        with pytest.raises(NoauxTcRouterError) as excinfo:
            noaux_tc_correct(logits, bias, top_k=top_k)
        assert needle in str(excinfo.value)
        # A refusal is not a fallback: neither counter moves.
        assert noaux_tc_dispatch_counters() == (0, 0)
        fired += 1

    assert fired == 3
    # The one numeric constant a token-extent change could have rounded; it comes
    # from the reference implementation.
    assert NOAUX_TC_DENOM_EPS == 1e-20
