# SPDX-License-Identifier: Apache-2.0
"""The router: top-8 sigmoid scoring with a correction bias and group selection.

Numeric agreement against a torch reference at several token extents, the weight
layout the kernel consumes, and the refusals a bad expert count raises.
"""

import importlib
import inspect
import os
import sys

import pytest
import torch
import torch.nn as nn

import nki
import nki.isa as nisa
import nki.language as nl
import nki.simulator
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

# The substrate's own sharding query -- the same call the vendor router and the
# repaired `noaux_tc` stage both use. Imported here so the in-kernel reading
# below asks the question the shipped code asks.
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from vllm_neuron.functional.moe.router import (
    NOAUX_TC_DENOM_EPS,
    NOAUX_TC_K,
    NOAUX_TC_TILE,
    NoauxTcRouterError,
    _noaux_tc_correct_nki,
    _noaux_tc_shard_range,
    noaux_tc_correct,
    noaux_tc_correct_torch_oracle,
    noaux_tc_dispatch_counters,
    noaux_tc_rmsnorm_router_topk,
    noaux_tc_rmsnorm_router_topk_torch_oracle,
    reset_noaux_tc_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

_ROUTER_MODULE = "vllm_neuron.functional.moe.router"

# ---------------------------------------------------------------------------
# Declared values. Every one is the declared or the checkpoint's; none is chosen
# here, and none is widened anywhere below.
# ---------------------------------------------------------------------------

#: The declared fixture extents. ``T = 256`` because the
#: substrate's admission gate refuses ``T % 256 != 0`` and does so silently
#: (``rmsnorm_router_topk_tkg.py``); ``E = 288`` is the pin's own
#: ``n_routed_experts`` (``glm5_next/config.py``).
DECLARED_T = 256
DECLARED_E = 288

#: The declared gate-weight tolerances, order named inline.
RTOL, ATOL = 1e-2, 1e-5

#: The declared index comparison: set equality on every row.
DECLARED_ROWS = DECLARED_T

#: The checkpoint's own routing hyperparameters (``glm5_next/config.py,147,148``).
DECLARED_TOP_K = 8
DECLARED_NORM_TOPK_PROB = True
DECLARED_ROUTED_SCALING_FACTOR = 2.5

#: A hidden extent satisfying the substrate's unconditional ``H % 256 == 0``
TINY_H = 256

# ---------------------------------------------------------------------------
# The conditioned fixture. These constants are fixture values, not plan values:
# this file declares T, E, both tolerances and the set-equality form, and declares
# nothing about how the fixture is built.
# ---------------------------------------------------------------------------

WINNER_HI, WINNER_LO = 0.785, 0.750
LOSER_HI, LOSER_LO = 0.700, 0.100
BIAS_AMP = 0.040
FIXTURE_SEED = 32

#: ``WINNER_LO - LOSER_HI``. Asserted, not assumed.
BOUNDARY_MARGIN_FLOOR = 0.045

MOVED_ROWS_FLOOR = 64


def build_designed_logits(seed: int = FIXTURE_SEED):
    """Build the corrected ladder first, then derive the raw logits from it. """
    gen = torch.Generator().manual_seed(seed)
    w_top = (WINNER_HI - WINNER_LO) / (NOAUX_TC_K - 1)
    w_lo = (LOSER_HI - LOSER_LO) / (DECLARED_E - NOAUX_TC_K - 1)

    ladder = torch.empty(DECLARED_E, dtype=torch.float32)
    for rank in range(DECLARED_E):
        if rank < NOAUX_TC_K:
            ladder[rank] = WINNER_HI - rank * w_top
        else:
            ladder[rank] = LOSER_HI - (rank - NOAUX_TC_K) * w_lo

    choice = torch.empty(DECLARED_T, DECLARED_E, dtype=torch.float32)
    for token in range(DECLARED_T):
        perm = torch.randperm(DECLARED_E, generator=gen)
        choice[token, perm] = ladder

    bias = (
        (torch.rand(1, DECLARED_E, generator=gen) - 0.5) * 2.0 * BIAS_AMP
    ).to(torch.float32)
    scores = (choice - bias).clamp(1e-4, 1.0 - 1e-4)
    logits = torch.log(scores / (1.0 - scores))
    return logits.contiguous(), bias.contiguous()


def build_hidden_states(seed: int = FIXTURE_SEED, hidden: int = TINY_H):
    """A hidden-states fixture for the fused seam, at the declared extents."""
    gen = torch.Generator().manual_seed(seed)
    hidden_states = (
        torch.randn(1, DECLARED_T, hidden, generator=gen) * 0.5
    ).to(torch.bfloat16)
    gamma = torch.ones(1, hidden, dtype=torch.bfloat16)
    router_weights = (
        torch.randn(hidden, DECLARED_E, generator=gen) * 0.1
    ).to(torch.bfloat16)
    _, bias = build_designed_logits(seed)
    return hidden_states, gamma, router_weights, bias


# ---------------------------------------------------------------------------
# The four route instruments.
# ---------------------------------------------------------------------------


class RouteInstrumentError(AssertionError):
    """A route reading that contradicts the declared predicate. """


class _SimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls for the duration. """

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


def _assert_route(sim: _SimulatorCounter, expected_dispatches: int, label: str) -> str:
    """Read all four route instruments and return the reading."""
    nki_dispatch, torch_fallback = noaux_tc_dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"[{label}] nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
        f"can_run_kernel={gate} simulate_kernel_calls={sim.calls}"
    )
    if nki_dispatch != expected_dispatches:
        raise RouteInstrumentError(
            f"{label}: seam dispatch counter read {nki_dispatch}, declared "
            f"{expected_dispatches}. {reading}"
        )
    if torch_fallback != 0:
        raise RouteInstrumentError(
            f"{label}: torch-fallback counter read {torch_fallback}, declared 0. "
            f"A torch fallback for work the kernel owns is a route failure, not a "
            f"degraded pass. {reading}"
        )
    if gate is not True:
        raise RouteInstrumentError(
            f"{label}: can_run_kernel() read {gate}, declared True. {reading}"
        )
    if sim.calls != expected_dispatches:
        raise RouteInstrumentError(
            f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, "
            f"declared {expected_dispatches}. A numeric pass without a simulator "
            f"call is the vacuous pass this arm screens for. {reading}"
        )
    return reading


# ---------------------------------------------------------------------------
# Reference helpers.
# ---------------------------------------------------------------------------


def verbatim_upstream_stage(logits: torch.Tensor, bias: torch.Tensor):
    """``transformers`` 5.16.1 ``Glm5NextTextTopkRouter.forward`` :161-182, verbatim.
    """
    num_group = 1
    topk_group = num_group  # forced at n_group == 1, not a chosen default
    num_experts = logits.shape[-1]
    scores = logits.to(torch.float32).sigmoid()
    scores_for_choice = scores + bias.to(torch.float32).reshape(-1)
    group_scores = (
        scores_for_choice.view(-1, num_group, num_experts // num_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, num_group, num_experts // num_group)
        .reshape(-1, num_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )
    topk_indices = torch.topk(
        scores_for_choice, k=NOAUX_TC_K, dim=-1, sorted=False
    )[1]
    topk_weights = scores.gather(1, topk_indices)
    denominator = topk_weights.sum(dim=-1, keepdim=True) + NOAUX_TC_DENOM_EPS
    topk_weights = topk_weights / denominator
    return topk_indices, topk_weights * DECLARED_ROUTED_SCALING_FACTOR


def set_equal_rows(got_index: torch.Tensor, expected_index: torch.Tensor) -> int:
    """Rows whose selected expert index sets are equal. """
    got = torch.sort(got_index.to(torch.int64), dim=-1)[0]
    want = torch.sort(expected_index.to(torch.int64), dim=-1)[0]
    return int((got == want).all(dim=-1).sum())


def rows_with_k_distinct(index: torch.Tensor) -> int:
    """Rows selecting ``K`` distinct experts. """
    return sum(
        1 for row in index.to(torch.int64) if len(set(row.tolist())) == NOAUX_TC_K
    )


class ReferenceShapeError(AssertionError):
    """A reference tensor whose shape does not match its scatter index. """


def scatter_reference(
    affinities_like: torch.Tensor, index: torch.Tensor, weights_tk: torch.Tensor
) -> torch.Tensor:
    """Scatter a ``[T, K]`` weight block into a dense ``[T, E]`` tensor. """
    if tuple(weights_tk.shape) != tuple(index.shape):
        raise ReferenceShapeError(
            f"scatter_reference expects a [T, K] weight block matching its index "
            f"{tuple(index.shape)}, got {tuple(weights_tk.shape)}. A dense "
            f"[T, E] reference must be compared directly, not scattered again."
        )
    want = torch.zeros_like(affinities_like)
    want.scatter_(1, index.to(torch.int64), weights_tk)
    return want


# ===========================================================================
# fixture conditioning -- preconditions on the declared arms below.
# ===========================================================================


def test_fixture_conditioning_is_measured_not_assumed() -> None:
    """The declared index arm can only turn on the 8/9 gap. Measure it."""
    logits, bias = build_designed_logits()
    assert tuple(logits.shape) == (DECLARED_T, DECLARED_E)
    assert tuple(bias.shape) == (1, DECLARED_E)

    choice = logits.sigmoid() + bias.reshape(-1)
    descending, _ = torch.sort(choice, dim=-1, descending=True)
    boundary = float(
        (descending[:, NOAUX_TC_K - 1] - descending[:, NOAUX_TC_K]).min()
    )
    ascending, _ = torch.sort(choice, dim=-1)
    pairwise = float((ascending[:, 1:] - ascending[:, :-1]).min())
    assert boundary > BOUNDARY_MARGIN_FLOOR, (
        f"boundary margin {boundary:.6e} is below the floor "
        f"{BOUNDARY_MARGIN_FLOOR}; the declared set equality would then turn on "
        f"fp32 round-off rather than on the implementation"
    )
    assert pairwise > 0.0, "two corrected scores tie exactly; conditioning lost"

    # The weight arm's own conditioning: no cancellation anywhere.
    scores = logits.sigmoid()
    index = torch.topk(choice, k=NOAUX_TC_K, dim=-1, sorted=False)[1]
    gathered = scores.gather(1, index)
    denominator = gathered.sum(dim=-1, keepdim=True)
    assert bool((gathered > 0).all()), "a gathered weight is non-positive"
    assert float(denominator.min()) > 1.0, "the L1 denominator is near zero"

    # Every expert is exercised, so E = 288 is not decorative.
    distinct_selected = int(torch.unique(index).numel())
    assert distinct_selected == DECLARED_E


def test_correction_moves_the_selection() -> None:
    """the control: the conditioned fixture must not condition the correction away."""
    logits, bias = build_designed_logits()
    scores = logits.sigmoid()
    uncorrected = torch.topk(scores, k=NOAUX_TC_K, dim=-1, sorted=False)[1]
    corrected = torch.topk(
        scores + bias.reshape(-1), k=NOAUX_TC_K, dim=-1, sorted=False
    )[1]
    moved = DECLARED_T - set_equal_rows(uncorrected, corrected)
    assert moved >= MOVED_ROWS_FLOOR, (
        f"only {moved} rows differ; an implementation that ignored the "
        f"correction bias entirely would pass the declared index arm"
    )


def test_oracle_equals_verbatim_upstream_at_n_group_1() -> None:
    """The reference's provenance, measured against upstream rather than argued."""
    logits, bias = build_designed_logits()
    reduced_index, reduced_affinities = noaux_tc_correct_torch_oracle(
        logits, bias, DECLARED_NORM_TOPK_PROB, DECLARED_ROUTED_SCALING_FACTOR
    )
    upstream_index, upstream_weights = verbatim_upstream_stage(logits, bias)

    equal_rows = set_equal_rows(reduced_index, upstream_index)
    # `upstream_weights` is a [T, K] block, so the scatter is the right tool here
    # -- this is the one call site the helper was written for.
    upstream_scattered = scatter_reference(
        reduced_affinities, upstream_index, upstream_weights
    )
    max_abs = float((reduced_affinities - upstream_scattered).abs().max())
    assert equal_rows == DECLARED_ROWS
    assert max_abs == 0.0, (
        "the reduced reference and the verbatim upstream stage disagree; the "
        "n_group == 1 identity does not hold and the reduction is unsound"
    )


def test_pinned_fixture_config_carries_n_group_1() -> None:
    """The reduction's precondition, read from the pinned bytes. """
    import json

    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "config.json")
    with open(fixture) as handle:
        text_config = json.load(handle)["text_config"]
    assert text_config.get("n_group") == 1, (
        "the reduced reference omits upstream's group-routing stage, which is "
        "an identity only at n_group == 1"
    )
    assert text_config.get("topk_method") == "noaux_tc"
    assert text_config.get("scoring_func") == "sigmoid"
    assert text_config.get("n_routed_experts") == DECLARED_E


# ===========================================================================
# the declared arms.
# ===========================================================================


def test_declared_index_sets_match_the_noaux_tc_reference_exactly() -> None:
    """declared arm 1: index sets match exactly, 256/256 rows, set equality."""
    logits, bias = build_designed_logits()
    expected_index, _ = noaux_tc_correct_torch_oracle(
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
    _assert_route(sim, 1, "declared-index-arm")

    assert tuple(got_index.shape) == (DECLARED_T, NOAUX_TC_K)
    assert tuple(got_affinities.shape) == (DECLARED_T, DECLARED_E)

    distinct = rows_with_k_distinct(got_index)
    equal_rows = set_equal_rows(got_index, expected_index)
    nonzero = (got_affinities != 0).sum(dim=1)
    assert distinct == DECLARED_ROWS, (
        f"{DECLARED_T - distinct} rows select fewer than {NOAUX_TC_K} distinct "
        f"experts; the selection collapsed"
    )
    assert equal_rows == DECLARED_ROWS
    assert int(nonzero.min()) == NOAUX_TC_K
    assert int(nonzero.max()) == NOAUX_TC_K


def test_kernel_affinity_columns_are_its_own_emitted_indices() -> None:
    """The kernel's two outputs must agree with each other. """
    logits, bias = build_designed_logits()
    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim:
        got_index, got_affinities = noaux_tc_correct(
            logits,
            bias,
            top_k=DECLARED_TOP_K,
            norm_topk_prob=DECLARED_NORM_TOPK_PROB,
            routed_scaling_factor=DECLARED_ROUTED_SCALING_FACTOR,
        )
    _assert_route(sim, 1, "affinity-index-agreement")

    nonzero_columns = [
        sorted(torch.nonzero(row, as_tuple=True)[0].tolist())
        for row in got_affinities
    ]
    emitted = [sorted(row.tolist()) for row in got_index.to(torch.int64)]
    agreeing = sum(1 for a, b in zip(nonzero_columns, emitted) if a == b)
    assert agreeing == DECLARED_ROWS, (
        "the kernel's affinity nonzeros and its own emitted expert_index "
        "disagree; the mask inside the authored stage is not the one that "
        "produced the indices"
    )

    # the control: the comparison must be able to fail. Shift one row's index set by one
    # column and the row must stop agreeing.
    shifted = [sorted(((e + 1) % DECLARED_E) for e in emitted[0])]
    assert nonzero_columns[0] != shifted[0], "the control is a no-op"


def test_declared_gate_weights_match_within_declared_tolerances() -> None:
    """declared arm 2: gate weights at ``assert_close(rtol=1e-2, atol=1e-5)``. """
    logits, bias = build_designed_logits()
    expected_index, expected_affinities = noaux_tc_correct_torch_oracle(
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
    _assert_route(sim, 1, "declared-weight-arm")

    got = got_affinities.to(torch.float32)
    want = expected_affinities.to(torch.float32)
    assert got.shape == want.shape == (DECLARED_T, DECLARED_E)
    # The comparison is only about weights if the two sides agree on where the
    # weights go, so the selection is re-read here rather than assumed from the
    # arm above.
    assert set_equal_rows(got_index, expected_index) == DECLARED_ROWS
    # The declared comparator, with both tolerances named in order.
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)

    # The row sum is the routed scaling factor by construction, so a lost or
    # doubled normalisation shows here even if every element passed the
    # tolerance individually.
    row_sums = got.sum(dim=1)
    torch.testing.assert_close(
        row_sums,
        torch.full_like(row_sums, DECLARED_ROUTED_SCALING_FACTOR),
        rtol=RTOL,
        atol=ATOL,
    )


# ===========================================================================
# route controls -- what makes the counted zeros measurements (the control).
# ===========================================================================


def test_dispatch_counters_are_module_level_state_reachable_from_elsewhere() -> None:
    """The counters must be resettable and readable across a module boundary. """
    foreign = importlib.import_module(_ROUTER_MODULE)
    assert foreign is sys.modules[_ROUTER_MODULE]
    assert foreign.noaux_tc_dispatch_counters is noaux_tc_dispatch_counters
    assert foreign.reset_noaux_tc_counters is reset_noaux_tc_counters

    logits, bias = build_designed_logits()
    foreign.reset_noaux_tc_counters()
    assert noaux_tc_dispatch_counters() == (0, 0)

    with _SimulatorCounter() as sim_one:
        foreign.noaux_tc_correct(logits, bias)
    after_one = noaux_tc_dispatch_counters()
    with _SimulatorCounter() as sim_two:
        foreign.noaux_tc_correct(logits, bias)
    after_two = foreign.noaux_tc_dispatch_counters()
    assert after_one == (1, 0)
    assert after_two == (2, 0)
    assert sim_one.calls == 1 and sim_two.calls == 1


# ===========================================================================
# named refusals -- the substrate's silent rules, made loud on this seam.
# ===========================================================================


@pytest.mark.parametrize(
    "tokens, experts, top_k, needle",
    [
        (DECLARED_T, 513, DECLARED_TOP_K, "E must be <= 512"),
        (DECLARED_T, 4, DECLARED_TOP_K, "E must be >= 8"),
        (DECLARED_T, DECLARED_E, 4, "top_k must be exactly 8"),
        (DECLARED_T, DECLARED_E, 9, "top_k must be exactly 8"),
    ],
)
def test_refused_extents_raise_by_name(tokens, experts, top_k, needle) -> None:
    """Each refusal is a named error carrying the extent, never a silent False. """
    logits = torch.zeros(tokens, experts, dtype=torch.float32)
    bias = torch.zeros(1, experts, dtype=torch.float32)
    reset_noaux_tc_counters()
    with pytest.raises(NoauxTcRouterError) as excinfo:
        noaux_tc_correct(logits, bias, top_k=top_k)
    message = str(excinfo.value)
    assert needle in message
    # A refusal is not a fallback: neither counter moves.
    assert noaux_tc_dispatch_counters() == (0, 0)


def logits_at_token_extent(tokens: int):
    """The conditioned fixture, re-indexed to ``tokens`` rows. """
    logits, bias = build_designed_logits()
    return logits[torch.arange(tokens) % DECLARED_T].contiguous(), bias


def _assert_admitted_token_extent(tokens: int) -> None:
    """The whole reading for a token extent this seam used to refuse. """
    logits, bias = logits_at_token_extent(tokens)
    expected_index, expected_affinities = noaux_tc_correct_torch_oracle(
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
    _assert_route(sim, 1, f"admitted-T{tokens}")

    got = got_affinities.to(torch.float32)
    want = expected_affinities.to(torch.float32)
    assert tuple(got_index.shape) == (tokens, NOAUX_TC_K)
    assert got.shape == want.shape == (tokens, DECLARED_E)

    equal_rows = set_equal_rows(got_index, expected_index)
    distinct = rows_with_k_distinct(got_index)
    assert equal_rows == tokens
    assert distinct == tokens
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_token_extent_128_is_admitted_and_matches_the_reference() -> None:
    """``T = 128`` was a named refusal here and now runs. """
    _assert_admitted_token_extent(128)


def test_token_extent_384_is_admitted_and_matches_the_reference() -> None:
    """``T = 384`` -- re-pin 2, above the old multiple. """
    _assert_admitted_token_extent(384)


def test_correction_bias_shape_is_refused_by_name() -> None:
    logits, _ = build_designed_logits()
    for bad in (
        torch.zeros(1, DECLARED_E + 1),
        torch.zeros(2, DECLARED_E),
        torch.zeros(DECLARED_E - 1),
    ):
        with pytest.raises(NoauxTcRouterError) as excinfo:
            noaux_tc_correct(logits, bad)
        assert "correction_bias must be" in str(excinfo.value)


def test_wrong_hidden_extent_raises_on_both_routes(monkeypatch) -> None:
    """``H`` needs no acceptance value, and this is why: it is loud on both routes. """
    bad_hidden = 384  # 384 % 256 == 128
    hidden_states, gamma, router_weights, bias = build_hidden_states(
        hidden=bad_hidden
    )
    for simulator in ("1", "0"):
        monkeypatch.setitem(os.environ, "NKI_SIMULATOR", simulator)
        reset_noaux_tc_counters()
        with pytest.raises(AssertionError) as excinfo:
            noaux_tc_rmsnorm_router_topk(
                hidden_states=hidden_states,
                gamma=gamma,
                router_weights=router_weights,
                correction_bias=bias,
                top_k=DECLARED_TOP_K,
            )
        message = str(excinfo.value)
        assert "divisible by 256" in message
        assert noaux_tc_dispatch_counters() == (0, 0)


# ===========================================================================
# the fused seam -- supplementary coverage at the same declared tolerances.
# ===========================================================================


def test_fused_seam_matches_the_reference_on_its_own_logits() -> None:
    """One dispatch for RMSNorm + router matmul + the authored correction. """
    hidden_states, gamma, router_weights, bias = build_hidden_states()

    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim:
        logits, expert_index, expert_affinities, substrate_index = (
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
    _assert_route(sim, 1, "fused-seam")

    assert tuple(logits.shape) == (DECLARED_T, DECLARED_E)
    assert tuple(expert_index.shape) == (DECLARED_T, NOAUX_TC_K)
    assert tuple(expert_affinities.shape) == (DECLARED_T, DECLARED_E)
    assert tuple(substrate_index.shape) == (DECLARED_T, NOAUX_TC_K)
    assert bool(torch.isfinite(logits).all())

    expected_index, expected_affinities = noaux_tc_correct_torch_oracle(
        logits, bias, DECLARED_NORM_TOPK_PROB, DECLARED_ROUTED_SCALING_FACTOR
    )
    equal_rows = set_equal_rows(expert_index, expected_index)
    got = expert_affinities.to(torch.float32)
    want = expected_affinities.to(torch.float32)
    assert equal_rows == DECLARED_ROWS
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_kernel_corrected_selection_differs_from_substrate_selection() -> None:
    """The non-vacuity control read off the kernel's own two outputs. """
    hidden_states, gamma, router_weights, bias = build_hidden_states()
    # Amplify the bias so the correction is unambiguously visible against a
    # matmul-produced logit spread this fixture does not design. Same seam, same
    # tolerances; only the fixture's bias scale changes.
    strong_bias = bias * 20.0

    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim:
        logits, expert_index, _affinities, substrate_index = (
            noaux_tc_rmsnorm_router_topk(
                hidden_states=hidden_states,
                gamma=gamma,
                router_weights=router_weights,
                correction_bias=strong_bias,
                top_k=DECLARED_TOP_K,
                norm_topk_prob=DECLARED_NORM_TOPK_PROB,
                routed_scaling_factor=DECLARED_ROUTED_SCALING_FACTOR,
            )
        )
    _assert_route(sim, 1, "non-vacuity-in-kernel")

    moved = DECLARED_T - set_equal_rows(expert_index, substrate_index)
    assert moved > 0, (
        "the corrected selection equals the substrate's uncorrected selection on "
        "every row, so the authored correction is not affecting the dispatch"
    )
    # And it is still the reference's selection, on the kernel's own logits.
    expected_index, _ = noaux_tc_correct_torch_oracle(
        logits, strong_bias, DECLARED_NORM_TOPK_PROB, DECLARED_ROUTED_SCALING_FACTOR
    )
    assert set_equal_rows(expert_index, expected_index) == DECLARED_ROWS


def test_model_call_site_routes_through_the_seam() -> None:
    """``Glm5NextRoutedExperts.route_tokens`` reaches the kernel, once."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextRoutedExperts

    text_config = Glm5NextTextConfig(hidden_size=TINY_H)
    bank = Glm5NextRoutedExperts(text_config, world_size=1)

    hidden_states, gamma, router_weights, bias = build_hidden_states()
    bank.router_weight = nn.Parameter(router_weights, requires_grad=False)
    bank.router_bias = nn.Parameter(bias.reshape(-1), requires_grad=False)

    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim:
        logits, expert_index, expert_affinities = bank.route_tokens(
            hidden_states, gamma, text_config
        )
    _assert_route(sim, 1, "model-call-site")

    assert tuple(logits.shape) == (DECLARED_T, DECLARED_E)
    assert tuple(expert_index.shape) == (DECLARED_T, NOAUX_TC_K)
    assert tuple(expert_affinities.shape) == (DECLARED_T, DECLARED_E)

    expected_index, expected_affinities = noaux_tc_correct_torch_oracle(
        logits,
        bank.router_bias.detach(),
        bool(text_config.norm_topk_prob),
        float(text_config.routed_scaling_factor),
    )
    got = expert_affinities.to(torch.float32)
    want = expected_affinities.to(torch.float32)
    assert set_equal_rows(expert_index, expected_index) == DECLARED_ROWS
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_model_call_site_passes_the_checkpoints_own_hyperparameters() -> None:
    """The call site must not hardcode what the config declares."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    text_config = Glm5NextTextConfig()
    assert int(text_config.num_experts_per_tok) == DECLARED_TOP_K
    assert bool(text_config.norm_topk_prob) is DECLARED_NORM_TOPK_PROB
    assert (
        float(text_config.routed_scaling_factor) == DECLARED_ROUTED_SCALING_FACTOR
    )
    assert int(text_config.n_routed_experts) == DECLARED_E
    assert text_config.topk_method == "noaux_tc"
    assert text_config.scoring_func == "sigmoid"


def test_shard_range_partitions_the_declared_token_extent() -> None:
    """The shard arithmetic, read as plain python at the declared extent. """
    for n_prgs in (1, 2):
        covered: list[int] = []
        ranges = []
        for prg_id in range(n_prgs):
            t_offset, t_local = _noaux_tc_shard_range(DECLARED_T, n_prgs, prg_id)
            ranges.append((t_offset, t_local))
            covered.extend(range(t_offset, t_offset + t_local))
        assert sorted(covered) == list(range(DECLARED_T)), (
            f"at n_prgs={n_prgs} the cores' ranges {ranges} do not cover "
            f"0..{DECLARED_T - 1} exactly once"
        )
        assert len(covered) == len(set(covered)), (
            f"at n_prgs={n_prgs} two cores own the same token rows"
        )
        for _offset, t_local in ranges:
            assert t_local % NOAUX_TC_TILE == 0, (
                f"a core owns {t_local} tokens, which is not a whole number of "
                f"{NOAUX_TC_TILE}-row tiles, so its loop would drop rows. The "
                f"multiple-of-256 admission clause is what rules this out."
            )
    # And the two-core split is not the one-core split: the reading discriminates.
    assert _noaux_tc_shard_range(DECLARED_T, 1, 0) == (0, DECLARED_T)
    assert _noaux_tc_shard_range(DECLARED_T, 2, 0) == (0, DECLARED_T // 2)
    assert _noaux_tc_shard_range(DECLARED_T, 2, 1) == (DECLARED_T // 2,
                                                       DECLARED_T // 2)

    # The vendor's tail rule, read rather than trusted. The producer gives the
    # remainder of an uneven split to the second core (`router_topk.py`).
    # No extent this stage can receive is uneven -- the admission clause forces a
    # multiple of 256 -- so this reading guards the helper against agreeing with
    # the producer only by luck of the admitted extents.
    assert _noaux_tc_shard_range(7, 2, 0) == (0, 3)
    assert _noaux_tc_shard_range(7, 2, 1) == (3, 4), (
        "the odd token goes to the second core in the vendor's split, so it must "
        "go there here too; a floor split on both cores would drop it"
    )
    covered_uneven: list[int] = []
    for prg in (0, 1):
        offset, local = _noaux_tc_shard_range(7, 2, prg)
        covered_uneven.extend(range(offset, offset + local))
    assert sorted(covered_uneven) == list(range(7))


def test_fused_seam_stage_takes_a_distinct_token_range_per_core(monkeypatch) -> None:
    """A per-core token range, read from inside the kernel."""
    import vllm_neuron.functional.moe.router as router_module

    seen: list[tuple[int, int, int, int, int]] = []
    real_shard_range = router_module._noaux_tc_shard_range

    def recording(num_tokens, n_prgs, prg_id):
        t_offset, t_local = real_shard_range(num_tokens, n_prgs, prg_id)
        seen.append(
            (int(num_tokens), int(n_prgs), int(prg_id), int(t_offset), int(t_local))
        )
        return t_offset, t_local

    monkeypatch.setattr(router_module, "_noaux_tc_shard_range", recording)

    hidden_states, gamma, router_weights, bias = build_hidden_states()
    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim:
        _logits, expert_index, expert_affinities, _substrate_index = (
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
    _assert_route(sim, 1, "grid-aware-fused")
    fused_calls = sorted(seen)

    if not fused_calls:
        raise RouteInstrumentError(
            "the authored stage never asked which tokens this core owns, so it "
            "is not grid-aware"
        )
    program_ids = sorted({c[2] for c in fused_calls})
    ranges = sorted({(c[3], c[4]) for c in fused_calls})
    n_prgs_seen = sorted({c[1] for c in fused_calls})
    assert n_prgs_seen == [2], (
        f"the stage read n_prgs={n_prgs_seen}, but the fused seam is launched on "
        f"a [2] grid, so each program must see 2"
    )
    assert program_ids == [0, 1], (
        f"the stage saw program ids {program_ids}; a two-core launch must give "
        f"one call per core with distinct identifiers"
    )
    assert ranges == [(0, DECLARED_T // 2), (DECLARED_T // 2, DECLARED_T // 2)], (
        f"the per-core ranges were {ranges}, not the two halves of the token "
        f"extent; a core that reads outside its own half is reading logits the "
        f"other core wrote"
    )
    covered: list[int] = []
    for _t, _n, _p, offset, local in fused_calls:
        covered.extend(range(offset, offset + local))
    assert sorted(covered) == list(range(DECLARED_T)), (
        "the two cores' ranges do not partition the token extent"
    )

    # The opposite reading, same instrument: no grid, one program, whole extent.
    seen.clear()
    logits, standalone_bias = build_designed_logits()
    reset_noaux_tc_counters()
    noaux_tc_correct(
        logits,
        standalone_bias,
        top_k=DECLARED_TOP_K,
        norm_topk_prob=DECLARED_NORM_TOPK_PROB,
        routed_scaling_factor=DECLARED_ROUTED_SCALING_FACTOR,
    )
    standalone_calls = sorted(seen)
    assert standalone_calls == [(DECLARED_T, 1, 0, 0, DECLARED_T)], (
        f"the standalone entry point is launched with no grid, so it must own "
        f"the whole extent at offset 0; it read {standalone_calls}"
    )
    assert standalone_calls != fused_calls, (
        "the sharded and unsharded readings are identical, so this instrument "
        "is reporting a constant rather than measuring the launch"
    )

    # The union of the two cores' writes is still the complete output.
    assert tuple(expert_index.shape) == (DECLARED_T, NOAUX_TC_K)
    assert tuple(expert_affinities.shape) == (DECLARED_T, DECLARED_E)
    nonzero_per_row = (expert_affinities != 0).sum(dim=-1)
    assert int(nonzero_per_row.min()) == NOAUX_TC_K, (
        "some token row carries fewer than K gate weights, so no core wrote it "
        "-- the shard leaves a gap"
    )
    assert int(nonzero_per_row.max()) == NOAUX_TC_K


@nki.jit
def _read_per_core_token_range(src):
    """Write this program's id, program count and token range into its own row. """
    t_total, _ = src.shape
    out = nl.ndarray((4, 4), dtype=nl.float32, buffer=nl.shared_hbm)
    _ndim, n_prgs, prg_id = get_verified_program_sharding_info(
        "test_read_per_core_token_range", (0, 1), 2
    )
    t_offset, t_local = _noaux_tc_shard_range(t_total, n_prgs, prg_id)
    row = nl.ndarray((1, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=row, value=0.0)
    row[0, 0] = prg_id
    row[0, 1] = n_prgs
    row[0, 2] = t_offset
    row[0, 3] = t_local
    nl.store(out[prg_id : prg_id + 1, :], value=row)
    return out


@nki.jit
def _write_only_this_cores_tokens(src):
    """Write ``program_id + 1`` into this program's own token rows and no others. """
    t_total, e_total = src.shape
    out = nl.ndarray((t_total, e_total), dtype=nl.float32, buffer=nl.shared_hbm)
    _ndim, n_prgs, prg_id = get_verified_program_sharding_info(
        "test_write_only_this_cores_tokens", (0, 1), 2
    )
    t_offset, t_local = _noaux_tc_shard_range(t_total, n_prgs, prg_id)
    for t_tile in range(t_local // NOAUX_TC_TILE):
        lo = t_offset + t_tile * NOAUX_TC_TILE
        val = nl.ndarray((NOAUX_TC_TILE, e_total), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=val, value=0.0)
        nisa.tensor_scalar(dst=val, data=val, op0=nl.add, operand0=prg_id + 1)
        nl.store(out[lo : lo + NOAUX_TC_TILE, :], value=val)
    return out


def test_per_core_token_range_is_read_from_inside_a_two_program_kernel() -> None:
    """The finding's evidence bar, taken literally: the range, read in-kernel. """
    src = torch.zeros(DECLARED_T, 8, dtype=torch.float32)
    rows = wrap_nki(_read_per_core_token_range)[2](src=src).to(torch.float32).tolist()
    # The buffer has four slots and the launch has two programs, so the two
    # unused slots stay at their zero fill. `n_prgs` can never be 0 in a row a
    # program actually wrote, which is what makes it the liveness marker.
    live = [row for row in rows if row[1] != 0.0]
    program_ids = sorted({int(row[0]) for row in live})
    n_prgs_seen = sorted({int(row[1]) for row in live})
    ranges = sorted({(int(row[2]), int(row[3])) for row in live})
    assert len(live) == 2, (
        f"{len(live)} programs wrote a row on a [2] launch; the identity is not "
        f"readable, so this instrument cannot answer the finding"
    )
    assert program_ids == [0, 1]
    assert n_prgs_seen == [2]
    assert ranges == [(0, DECLARED_T // 2), (DECLARED_T // 2, DECLARED_T // 2)], (
        f"the two programs read token ranges {ranges}, not the two halves"
    )

    union = wrap_nki(_write_only_this_cores_tokens)[2](src=src).to(torch.float32)
    half = DECLARED_T // 2
    top = sorted({float(v) for v in union[:half].flatten().tolist()})
    bottom = sorted({float(v) for v in union[half:].flatten().tolist()})
    unwritten = int((union == 0).sum())
    assert top == [1.0], (
        f"the first half carries {top}; program 0 must own it alone"
    )
    assert bottom == [2.0], (
        f"the second half carries {bottom}; program 1 must own it alone"
    )
    assert unwritten == 0, (
        f"{unwritten} elements were left unwritten, so the two shards do not "
        f"cover the token extent"
    )


def test_fused_seam_logits_match_an_independent_torch_reference() -> None:
    """The normalisation and the matmul, compared."""
    hidden_states, gamma, router_weights, bias = build_hidden_states()
    seam_eps = inspect.signature(
        noaux_tc_rmsnorm_router_topk
    ).parameters["eps"].default

    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim:
        logits, expert_index, expert_affinities, _substrate_index = (
            noaux_tc_rmsnorm_router_topk(
                hidden_states=hidden_states,
                gamma=gamma,
                router_weights=router_weights,
                correction_bias=bias,
                top_k=DECLARED_TOP_K,
                eps=seam_eps,
                norm_topk_prob=DECLARED_NORM_TOPK_PROB,
                routed_scaling_factor=DECLARED_ROUTED_SCALING_FACTOR,
            )
        )
    _assert_route(sim, 1, "independent-reference")

    # Independent: built from the inputs, never from the kernel's own output.
    expected_logits, expected_index, expected_affinities, _expected_sub = (
        noaux_tc_rmsnorm_router_topk_torch_oracle(
            hidden_states,
            gamma,
            router_weights,
            bias,
            seam_eps,
            DECLARED_NORM_TOPK_PROB,
            DECLARED_ROUTED_SCALING_FACTOR,
        )
    )
    got = logits.to(torch.float32)
    want = expected_logits.to(torch.float32)
    equal_rows = set_equal_rows(expert_index, expected_index)
    # The reference must not be vacuous: an all-zero reference would pass.
    assert float(want.abs().max()) > 0.0
    assert equal_rows == DECLARED_ROWS, (
        "the kernel's selection differs from a reference built independently "
        "from the same inputs, so a substrate stage is wrong"
    )
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        expert_affinities.to(torch.float32),
        expected_affinities.to(torch.float32),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "perturbation",
    ["eps", "gamma"],
)
def test_independent_reference_arm_fails_when_the_normalisation_moves(
    perturbation: str,
) -> None:
    """The non-vacuity control for the arm above, on the two inputs it guards. """
    hidden_states, gamma, router_weights, bias = build_hidden_states()
    seam_eps = inspect.signature(
        noaux_tc_rmsnorm_router_topk
    ).parameters["eps"].default

    reset_noaux_tc_counters()
    logits, _index, _affinities, _sub = noaux_tc_rmsnorm_router_topk(
        hidden_states=hidden_states,
        gamma=gamma,
        router_weights=router_weights,
        correction_bias=bias,
        top_k=DECLARED_TOP_K,
        eps=seam_eps,
        norm_topk_prob=DECLARED_NORM_TOPK_PROB,
        routed_scaling_factor=DECLARED_ROUTED_SCALING_FACTOR,
    )

    if perturbation == "eps":
        ref_eps, ref_gamma = 1e-5, gamma
    else:
        ref_eps = seam_eps
        ref_gamma = (gamma.to(torch.float32) * 1.05).to(gamma.dtype)

    expected_logits = noaux_tc_rmsnorm_router_topk_torch_oracle(
        hidden_states,
        ref_gamma,
        router_weights,
        bias,
        ref_eps,
        DECLARED_NORM_TOPK_PROB,
        DECLARED_ROUTED_SCALING_FACTOR,
    )[0]
    got = logits.to(torch.float32)
    want = expected_logits.to(torch.float32)
    allowed = ATOL + RTOL * want.abs()
    outside = int(((got - want).abs() > allowed).sum())
    assert outside > 0, (
        f"perturbing the {perturbation} left every element inside the declared "
        f"tolerance, so the independent-reference arm above cannot see a wrong "
        f"normalisation and is vacuous"
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_fused_torch_fallback_executes_and_is_the_cpu_oracle(monkeypatch) -> None:
    """The fused seam's torch fallback runs, and agrees with the kernel. """
    hidden_states, gamma, router_weights, bias = build_hidden_states()
    seam_eps = inspect.signature(
        noaux_tc_rmsnorm_router_topk
    ).parameters["eps"].default

    # The kernel result first, with the gate on.
    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim_on:
        kernel_logits, kernel_index, kernel_affinities, _sub = (
            noaux_tc_rmsnorm_router_topk(
                hidden_states=hidden_states,
                gamma=gamma,
                router_weights=router_weights,
                correction_bias=bias,
                top_k=DECLARED_TOP_K,
                eps=seam_eps,
                norm_topk_prob=DECLARED_NORM_TOPK_PROB,
                routed_scaling_factor=DECLARED_ROUTED_SCALING_FACTOR,
            )
        )
    _assert_route(sim_on, 1, "fallback-arm-kernel-half")

    # Now the same call with the gate off, which must take the fused fallback.
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not move, so this control is unarmed"
    )
    reset_noaux_tc_counters()
    with _SimulatorCounter() as sim_off:
        fb_logits, fb_index, fb_affinities, fb_sub = noaux_tc_rmsnorm_router_topk(
            hidden_states=hidden_states,
            gamma=gamma,
            router_weights=router_weights,
            correction_bias=bias,
            top_k=DECLARED_TOP_K,
            eps=seam_eps,
            norm_topk_prob=DECLARED_NORM_TOPK_PROB,
            routed_scaling_factor=DECLARED_ROUTED_SCALING_FACTOR,
        )
    readings = noaux_tc_dispatch_counters()
    assert readings == (0, 1), (
        f"the fused seam read {readings}; with the gate off it must take the "
        f"torch fallback exactly once and dispatch no kernel"
    )
    assert sim_off.calls == 0
    assert tuple(fb_logits.shape) == (DECLARED_T, DECLARED_E)
    assert tuple(fb_index.shape) == (DECLARED_T, NOAUX_TC_K)
    assert tuple(fb_affinities.shape) == (DECLARED_T, DECLARED_E)
    assert tuple(fb_sub.shape) == (DECLARED_T, NOAUX_TC_K)
    # The fallback is the CPU oracle, so it must agree with the kernel.
    assert set_equal_rows(fb_index, kernel_index) == DECLARED_ROWS
    torch.testing.assert_close(
        fb_logits.to(torch.float32),
        kernel_logits.to(torch.float32),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        fb_affinities.to(torch.float32),
        kernel_affinities.to(torch.float32),
        rtol=RTOL,
        atol=ATOL,
    )
