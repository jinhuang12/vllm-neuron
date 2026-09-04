# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ``inc-glm53f-032`` -- WP7 router: top-8 sigmoid with
``noaux_tc``.

The declared acceptance (increment plan revision 32,
``bfb7199ef72039a66dca7bfefdf47c116cc733ac44aabe1e74ea38dea4d9c1e4``, L924),
verbatim on the two things it declares:

    "for a ``[256, 288]`` logits fixture -- the selected expert **index sets**
    match a torch ``noaux_tc`` reference **exactly (256/256 rows, set
    equality)** -- an index comparison, not a tolerance comparison, because a
    top-k selection is discrete; and the gate **weights** match at
    ``assert_close(rtol=1e-2, atol=1e-5)``."

Tier N, so the command is D1's Tier N template and every env var in it is
load-bearing:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    python -m pytest test/vllm_neuron/model/glm5_next/test_router.py \\
      --timeout 60 -v -s

THE ROUTE PREDICATE (D13 form R-1, plan L925). Three instruments per declared
case, and all three are read: this module's own seam dispatch counter reads
exactly ``1``, its torch-fallback counter reads exactly ``0``, and
``can_run_kernel()`` reads ``True``. A fourth, independent instrument counts
real ``nki.simulator.simulate_kernel`` entries, because the first three are this
campaign's own bookkeeping and the fourth is the vendor's. A pure-torch
implementation reads ``0`` on the first and the fourth and cannot pass.

WHY THE ORACLE READS THE KERNEL'S OWN LOGITS. The declared index comparison is a
SET EQUALITY -- discrete, with no tolerance to absorb anything. So the kernel and
its reference must consume BYTE-IDENTICAL logits. Handing the reference a torch
recomputation of the router matmul instead would measure the substrate's bf16
matmul precision rather than this increment's numerics, and one flipped near-tie
would fail the arm for a reason that is not the implementation's. The declared
arms therefore run on the logits-input seam, where the fixture IS the ``[256,
288]`` logits tensor the plan names; the fused seam is covered separately, at
the SAME declared tolerances, with its reference reading the logits the kernel
returned.

THE REFERENCE'S OWN PROVENANCE IS MEASURED, NOT ASSERTED.
``test_oracle_equals_verbatim_upstream_at_n_group_1`` runs the VERBATIM
``transformers`` 5.16.1 ``Glm5NextTextTopkRouter`` stage (:161-183), group
routing included, against this module's reduced reference. The reduction is
legitimate only because ``n_group == 1``, which is read from the campaign's own
pinned ``fixtures/config.json`` rather than assumed.

FIXTURE CONDITIONING (the plan's carry #7, and ``-025``'s lesson one level up).
``-025`` attempt 1 died to catastrophic cancellation in a signed random fixture.
The hazard here is different and discrete: a TIE AT THE SELECTION BOUNDARY. If a
row's 8th and 9th largest corrected scores sit within fp32 round-off, torch's
``topk`` and the kernel's ``nisa.max8`` may pick different experts and the set
equality fails on the fixture, not on the code. The first draft of this fixture
measured a minimum pairwise gap of exactly ``0.0`` -- it passed, but on luck.
The fixture below removes the luck by constructing the CORRECTED ladder first
and deriving the raw scores from it, which fixes the boundary margin by design
at ``0.05`` (about 4.2e+05 x fp32 eps) independently of the bias, and
``test_fixture_conditioning_is_measured_not_assumed`` asserts that margin so a
future edit that breaks the conditioning fails loudly instead of flaking. The
weight arm is conditioned in the same construction: every gathered weight lies
in ``[0.71, 0.83]`` and the L1 denominator is a sum of eight such positives, so
no subtraction of near-equal magnitudes occurs anywhere.

NON-VACUITY (D1.5). A fixture conditioned against ties must not also condition
away the CORRECTION. Two controls, both measured:
``test_correction_moves_the_selection`` reads how many rows the corrected
selection differs from the uncorrected one on (a fixture with zero would let an
implementation that ignores the bias pass the index arm), and
``test_kernel_corrected_selection_differs_from_substrate_selection`` reads the
same difference off the FUSED kernel's own two outputs rather than off a torch
recomputation.

SELF-CONSISTENCY, SEPARATE FROM THE ORACLE.
``test_kernel_affinity_columns_are_its_own_emitted_indices`` compares the
kernel's two outputs against EACH OTHER and involves no reference at all. It was
added after attempt 1, whose failure showed that "the index sets are right" and
"each row has 8 nonzeros" together still do not say the nonzeros sit on the
emitted indices (``increments/investigation-032.md`` §6).

REFERENCE ASSEMBLY IS GUARDED BY SHAPE. A dense ``[T, E]`` reference is compared
directly; only a ``[T, K]`` per-slot weight block goes through
``scatter_reference``, which now REFUSES a mismatched shape by name
(``ReferenceShapeError``). Attempt 1 fed a dense tensor to that scatter and
``Tensor.scatter_``'s permissive shape rule accepted it silently, producing a
mis-assembled reference that failed three arms while the implementation was
correct.
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
    # The module's fused torch reference. Repair round 1 of `inc-glm53f-032`
    # imports it, because finding `M-B20-3` is that NO test did: it is the only
    # reference here that computes the normalisation and the router matmul
    # INDEPENDENTLY of the kernel, and it shipped with no execution coverage.
    noaux_tc_rmsnorm_router_topk_torch_oracle,
    reset_noaux_tc_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

_ROUTER_MODULE = "vllm_neuron.functional.moe.router"

# ---------------------------------------------------------------------------
# Declared values. Every one is the plan's or the checkpoint's; none is chosen
# here, and none is widened anywhere below.
# ---------------------------------------------------------------------------

#: The plan's declared fixture extents (L924). ``T = 256`` because the
#: substrate's admission gate refuses ``T % 256 != 0`` and does so SILENTLY
#: (``rmsnorm_router_topk_tkg.py:209``); ``E = 288`` is the pin's own
#: ``n_routed_experts`` (``glm5_next/config.py:185``).
DECLARED_T = 256
DECLARED_E = 288

#: The plan's declared gate-weight tolerances (L924), order named inline per D3.
RTOL, ATOL = 1e-2, 1e-5

#: The plan's declared index comparison: set equality on every row.
DECLARED_ROWS = DECLARED_T

#: The checkpoint's own routing hyperparameters (``glm5_next/config.py:143,147,148``).
DECLARED_TOP_K = 8
DECLARED_NORM_TOPK_PROB = True
DECLARED_ROUTED_SCALING_FACTOR = 2.5

#: A hidden extent satisfying the substrate's unconditional ``H % 256 == 0``
#: assert (``rmsnorm_router_topk_tkg.py:159``; the ``% 512`` variant at ``:168``
#: applies under MX only, and this seam runs at ``QuantizationType.NONE``).
#: 256 is the value ``increments/probe-032-simulator-route.py`` ARM E actually
#: ran the composed kernel at, so it is measured ground rather than a guess.
#: ``H`` is deliberately NOT a declared acceptance extent: the substrate's
#: ``_validate_inputs`` runs unconditionally at ``:79``, BEFORE the ``:94``
#: admission branch, so a wrong ``H`` raises on the NKI route and on the torch
#: route alike -- there is no silent fall-through on ``H`` to guard against.
#: ``test_wrong_hidden_extent_raises_on_both_routes`` measures that.
TINY_H = 256

# ---------------------------------------------------------------------------
# The conditioned fixture. These constants are FIXTURE values, not plan values:
# the plan declares T, E, both tolerances and the set-equality form, and declares
# nothing about how the fixture is built.
# ---------------------------------------------------------------------------

WINNER_HI, WINNER_LO = 0.785, 0.750
LOSER_HI, LOSER_LO = 0.700, 0.100
BIAS_AMP = 0.040
FIXTURE_SEED = 32

#: ``WINNER_LO - LOSER_HI``. Asserted, not assumed.
BOUNDARY_MARGIN_FLOOR = 0.045

#: Measured at ``FIXTURE_SEED`` by ``increments/probe-032-fixture-conditioning.py``:
#: 111 of 256 rows. A floor rather than the exact number, because the number is a
#: property of the seed and the FLOOR is the property the control needs.
MOVED_ROWS_FLOOR = 64


def build_designed_logits(seed: int = FIXTURE_SEED):
    """Build the CORRECTED ladder first, then derive the raw logits from it.

    ``noaux_tc`` selects on ``choice = sigmoid(logits) + bias``, so ``choice`` is
    the tensor whose 8/9 gap decides the declared index arm -- and it is the one
    this construction fixes::

        choice[rank <  8] in [0.750, 0.785]     (the winners)
        choice[rank >= 8] in [0.100, 0.700]     (the losers)
        boundary gap      == 0.050, INDEPENDENT of the bias

    The raw scores are then ``choice - bias`` and the logits are
    ``logit(scores)``. Because ``BIAS_AMP`` exceeds half the boundary gap,
    subtracting the bias reorders experts ACROSS the boundary, so the
    uncorrected top-8 differs from the corrected top-8 on most rows -- the
    non-vacuity control -- while the corrected margin is untouched, because
    ``choice`` is what was constructed.
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
    """A hidden-states fixture for the FUSED seam, at the declared extents."""
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
    """A route reading that contradicts the declared predicate.

    A named type rather than a bare ``assert``, so a failing transcript names
    what failed. Plan D13: "An acceptance whose route predicate did not fire has
    NOT passed, even if its numeric comparison did."
    """


class _SimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls for the duration.

    The INDEPENDENT instrument: the other three readings are this campaign's own
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


def _assert_route(sim: _SimulatorCounter, expected_dispatches: int, label: str) -> str:
    """Read all four route instruments and return the reading for the transcript."""
    nki_dispatch, torch_fallback = noaux_tc_dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"[{label}] nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
        f"can_run_kernel={gate} simulate_kernel_calls={sim.calls}"
    )
    print(reading)
    if nki_dispatch != expected_dispatches:
        raise RouteInstrumentError(
            f"{label}: seam dispatch counter read {nki_dispatch}, declared "
            f"{expected_dispatches}. {reading}"
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
    if sim.calls != expected_dispatches:
        raise RouteInstrumentError(
            f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, "
            f"declared {expected_dispatches}. A numeric pass without a simulator "
            f"call is the F1 false green. {reading}"
        )
    return reading


# ---------------------------------------------------------------------------
# Reference helpers.
# ---------------------------------------------------------------------------


def verbatim_upstream_stage(logits: torch.Tensor, bias: torch.Tensor):
    """``transformers`` 5.16.1 ``Glm5NextTextTopkRouter.forward`` :161-182, verbatim.

    Group routing INCLUDED, so the reduction this module ships is measured
    against upstream rather than argued for.

    THE TWO GROUP VALUES HAVE DIFFERENT WARRANTS, and neither is assumed.
    ``n_group == 1`` is READ from the campaign's own pinned
    ``fixtures/config.json`` (see
    ``test_pinned_fixture_config_carries_n_group_1``). ``topk_group`` is ABSENT
    from that fixture -- measured, not overlooked -- and at ``n_group == 1`` its
    value is FORCED: upstream selects ``topk_group`` of ``n_group`` groups
    (:170), so the only value in range is 1, and it is derived below rather than
    written as a literal so that the derivation is what a reader checks.
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


def set_equal_rows(got_index: torch.Tensor, want_index: torch.Tensor) -> int:
    """Rows whose selected expert index SETS are equal.

    Sets, not sequences: upstream selects with ``sorted=False`` (:177) and
    ``nisa.max8`` emits descending, so the two orders differ by construction and
    only the sets are comparable. This is the plan's declared form.
    """
    got = torch.sort(got_index.to(torch.int64), dim=-1)[0]
    want = torch.sort(want_index.to(torch.int64), dim=-1)[0]
    return int((got == want).all(dim=-1).sum())


def rows_with_k_distinct(index: torch.Tensor) -> int:
    """Rows selecting ``K`` DISTINCT experts.

    ``nisa.nc_find_index8`` documents "the first occurrence of each value", so a
    tie could in principle collapse two slots onto one expert. This counts
    rather than trusts.
    """
    return sum(
        1 for row in index.to(torch.int64) if len(set(row.tolist())) == NOAUX_TC_K
    )


class ReferenceShapeError(AssertionError):
    """A reference tensor whose shape does not match its scatter index.

    This exists because ``Tensor.scatter_(1, index, src)`` is PERMISSIVE in
    exactly the direction that hides a defect: it reads ``src[i][j]`` only at the
    positions ``index`` names, so an oversized ``src`` is silently accepted and
    only its first ``K`` columns are ever read. Attempt 1 of this increment fed
    the oracle's already-scattered ``[T, E]`` affinity tensor here as if it were
    a ``[T, K]`` weight block; the call quietly scattered ``affinities[:, 0:8]``
    onto the selected columns and produced a reference that was zero on most
    selected cells. The three declared arms then failed against a
    mis-assembled reference while the implementation itself was correct
    (``increments/investigation-032.md`` §4).

    Removing the three bad call sites fixes today. This refusal fixes the class.
    """


def scatter_reference(
    affinities_like: torch.Tensor, index: torch.Tensor, weights_tk: torch.Tensor
) -> torch.Tensor:
    """Scatter a ``[T, K]`` weight block into a dense ``[T, E]`` tensor.

    ``weights_tk`` must be ``[T, K]``, the same shape as ``index``. Use this ONLY
    for a reference that reports its weights per selected slot (the verbatim
    upstream stage). A reference that already returns a dense ``[T, E]`` tensor --
    such as ``noaux_tc_correct_torch_oracle`` -- is compared DIRECTLY and never
    passed through here.
    """
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
# Fixture conditioning -- preconditions on the declared arms below.
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
    eps = torch.finfo(torch.float32).eps
    print(
        f"[fixture] boundary_margin={boundary:.6e} ({boundary / eps:.4g} x fp32_eps) "
        f"min_pairwise={pairwise:.6e} fp32_eps={eps:.6e}"
    )
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
    print(
        f"[fixture] gathered_min={float(gathered.min()):.6f} "
        f"gathered_max={float(gathered.max()):.6f} "
        f"denominator_min={float(denominator.min()):.6f}"
    )
    assert bool((gathered > 0).all()), "a gathered weight is non-positive"
    assert float(denominator.min()) > 1.0, "the L1 denominator is near zero"

    # Every expert is exercised, so E = 288 is not decorative.
    distinct_selected = int(torch.unique(index).numel())
    print(f"[fixture] distinct_experts_ever_selected={distinct_selected}/{DECLARED_E}")
    assert distinct_selected == DECLARED_E


def test_correction_moves_the_selection() -> None:
    """D1.5: the conditioned fixture must not condition the CORRECTION away."""
    logits, bias = build_designed_logits()
    scores = logits.sigmoid()
    uncorrected = torch.topk(scores, k=NOAUX_TC_K, dim=-1, sorted=False)[1]
    corrected = torch.topk(
        scores + bias.reshape(-1), k=NOAUX_TC_K, dim=-1, sorted=False
    )[1]
    moved = DECLARED_T - set_equal_rows(uncorrected, corrected)
    print(f"[non-vacuity] rows_where_correction_moves_selection={moved}/{DECLARED_T}")
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
    # `upstream_weights` IS a [T, K] block, so the scatter is the right tool here
    # -- this is the one call site the helper was written for.
    upstream_scattered = scatter_reference(
        reduced_affinities, upstream_index, upstream_weights
    )
    max_abs = float((reduced_affinities - upstream_scattered).abs().max())
    print(
        f"[oracle-provenance] set_equal_rows={equal_rows}/{DECLARED_T} "
        f"weight_max_abs_diff={max_abs:.6e}"
    )
    assert equal_rows == DECLARED_ROWS
    assert max_abs == 0.0, (
        "the reduced reference and the verbatim upstream stage disagree; the "
        "n_group == 1 identity does not hold and the reduction is unsound"
    )


def test_pinned_fixture_config_carries_n_group_1() -> None:
    """The reduction's precondition, read from the pinned bytes.

    ``topk_group`` is absent from the pinned fixture. That absence is recorded
    here rather than papered over: at ``n_group == 1`` the value is forced to 1,
    so its absence costs the reduction nothing. If a future pin ever carries
    ``n_group > 1``, this test fails and the reduction must be revisited -- which
    is a design question for the lead, not a build-time widening.
    """
    import json

    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "config.json")
    with open(fixture) as handle:
        text_config = json.load(handle)["text_config"]
    print(
        f"[pin] n_group={text_config.get('n_group')!r} "
        f"topk_group={text_config.get('topk_group', '<ABSENT>')!r} "
        f"topk_method={text_config.get('topk_method')!r} "
        f"scoring_func={text_config.get('scoring_func')!r} "
        f"n_routed_experts={text_config.get('n_routed_experts')!r}"
    )
    assert text_config.get("n_group") == 1, (
        "the reduced reference omits upstream's group-routing stage, which is "
        "an identity only at n_group == 1"
    )
    assert text_config.get("topk_method") == "noaux_tc"
    assert text_config.get("scoring_func") == "sigmoid"
    assert text_config.get("n_routed_experts") == DECLARED_E


# ===========================================================================
# THE DECLARED ARMS.
# ===========================================================================


def test_declared_index_sets_match_the_noaux_tc_reference_exactly() -> None:
    """DECLARED ARM 1: index sets match exactly, 256/256 rows, set equality."""
    logits, bias = build_designed_logits()
    want_index, _ = noaux_tc_correct_torch_oracle(
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
    equal_rows = set_equal_rows(got_index, want_index)
    nonzero = (got_affinities != 0).sum(dim=1)
    print(
        f"[declared-index-arm] set_equal_rows={equal_rows}/{DECLARED_T} "
        f"rows_with_{NOAUX_TC_K}_distinct={distinct}/{DECLARED_T} "
        f"nonzero_per_row_min={int(nonzero.min())} max={int(nonzero.max())}"
    )
    assert distinct == DECLARED_ROWS, (
        f"{DECLARED_T - distinct} rows select fewer than {NOAUX_TC_K} distinct "
        f"experts; the selection collapsed"
    )
    assert equal_rows == DECLARED_ROWS
    assert int(nonzero.min()) == NOAUX_TC_K
    assert int(nonzero.max()) == NOAUX_TC_K


def test_kernel_affinity_columns_are_its_own_emitted_indices() -> None:
    """The kernel's two outputs must agree with EACH OTHER.

    THE GAP THIS CLOSES, named. Attempt 1 asserted that the index sets match the
    reference and that each row carries exactly 8 nonzero affinities -- but never
    that the 8 nonzero COLUMNS are the 8 experts the kernel itself emitted in
    `expert_index`. That agreement was assumed. Both outputs come from the same
    mask inside the authored stage, so a mask defect could in principle move the
    weights while leaving the indices right, and every attempt-1 arm would still
    have read green on the index side.

    This is a self-consistency reading, not a reference comparison: it involves
    no oracle at all, which is exactly why it catches a class the oracle
    comparisons cannot isolate.
    """
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
    print(
        f"[affinity-index-agreement] rows_where_nonzero_columns_equal_index="
        f"{agreeing}/{DECLARED_T}"
    )
    assert agreeing == DECLARED_ROWS, (
        "the kernel's affinity nonzeros and its own emitted expert_index "
        "disagree; the mask inside the authored stage is not the one that "
        "produced the indices"
    )

    # D1.5: the comparison must be able to fail. Shift one row's index set by one
    # column and the row must stop agreeing.
    shifted = [sorted(((e + 1) % DECLARED_E) for e in emitted[0])]
    print(
        f"[affinity-index-agreement] control shifted_row_agrees="
        f"{nonzero_columns[0] == shifted[0]}"
    )
    assert nonzero_columns[0] != shifted[0], "the control is a no-op"


def test_declared_gate_weights_match_within_declared_tolerances() -> None:
    """DECLARED ARM 2: gate weights at ``assert_close(rtol=1e-2, atol=1e-5)``.

    Both sides are DENSE ``[T, E]`` tensors and are compared as they stand.
    ``noaux_tc_correct_torch_oracle`` returns an already-scattered affinity
    tensor (``router.py:1535-1537``); passing it through a second scatter is what
    broke attempt 1 (``increments/investigation-032.md`` §4).
    """
    logits, bias = build_designed_logits()
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
    _assert_route(sim, 1, "declared-weight-arm")

    got = got_affinities.to(torch.float32)
    want = want_affinities.to(torch.float32)
    assert got.shape == want.shape == (DECLARED_T, DECLARED_E)
    # The comparison is only about weights if the two sides agree on WHERE the
    # weights go, so the selection is re-read here rather than assumed from the
    # arm above.
    assert set_equal_rows(got_index, want_index) == DECLARED_ROWS
    diff = (got - want).abs()
    relative = diff / (want.abs() + 1e-30)
    print(
        f"[declared-weight-arm] max_abs_error={float(diff.max()):.6e} "
        f"max_rel_error_on_selected="
        f"{float(relative[want != 0].max()):.6e} "
        f"declared rtol={RTOL} atol={ATOL}"
    )
    # The declared comparator, with both tolerances named in order (D3).
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)

    # The row sum is the routed scaling factor by construction, so a lost or
    # doubled normalisation shows here even if every element passed the
    # tolerance individually.
    row_sums = got.sum(dim=1)
    print(
        f"[declared-weight-arm] row_sum_min={float(row_sums.min()):.6f} "
        f"max={float(row_sums.max()):.6f} declared={DECLARED_ROUTED_SCALING_FACTOR}"
    )
    torch.testing.assert_close(
        row_sums,
        torch.full_like(row_sums, DECLARED_ROUTED_SCALING_FACTOR),
        rtol=RTOL,
        atol=ATOL,
    )


# ===========================================================================
# Route controls -- what makes the counted zeros measurements (D1.5).
# ===========================================================================


def test_route_control_fallback_counter_discriminates(monkeypatch) -> None:
    """With the simulator off, the readings must INVERT: (0, 1), sim 0.

    ``can_run_kernel`` re-reads ``os.environ`` on every call
    (``neuron_utils.py:16-23``), so this control arms the gate rather than
    describing it. This is not the D2 fixture ban: D2 forbids setting the
    acceptance ENVIRONMENT in a fixture because ``FP8_CLAMP_MAX`` resolves at
    import time. This control deliberately flips a live gate inside one test to
    prove it moves, and restores it.
    """
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not move, so this control is unarmed"
    )

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
    readings = noaux_tc_dispatch_counters()
    print(
        f"[control-fallback] counters={readings} simulate_kernel_calls={sim.calls}"
    )
    assert readings == (0, 1)
    assert sim.calls == 0
    # The fallback is still the CORRECT function -- it is the CPU oracle.
    want_index, _ = noaux_tc_correct_torch_oracle(
        logits, bias, DECLARED_NORM_TOPK_PROB, DECLARED_ROUTED_SCALING_FACTOR
    )
    assert set_equal_rows(got_index, want_index) == DECLARED_ROWS
    assert tuple(got_affinities.shape) == (DECLARED_T, DECLARED_E)


def test_route_control_simulator_is_load_bearing(monkeypatch) -> None:
    """Below the seam, the chain must REFUSE rather than quietly compute torch."""
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    logits, bias = build_designed_logits()
    with _SimulatorCounter() as sim:
        with pytest.raises(RuntimeError) as excinfo:
            wrap_nki(_noaux_tc_correct_nki)(
                router_logits=logits,
                correction_bias=bias,
                norm_topk_prob=DECLARED_NORM_TOPK_PROB,
                routed_scaling_factor=DECLARED_ROUTED_SCALING_FACTOR,
            )
    message = str(excinfo.value)
    print(f"[control-simulator] simulate_kernel_calls={sim.calls} msg={message[:160]}")
    assert "simulator" in message.lower()
    assert sim.calls == 0


def test_dispatch_counters_are_module_level_state_reachable_from_elsewhere() -> None:
    """The counters must be resettable and readable across a module boundary.

    The ``-025``/``-026`` precedent: a sibling increment counts this seam from
    its own test module (form R-2), so the counter's identity is a contract.
    """
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
    print(
        f"[cross-module] after_reset=(0, 0) after_one_call={after_one} "
        f"after_two_calls={after_two} sim_calls={sim_one.calls + sim_two.calls}"
    )
    assert after_one == (1, 0)
    assert after_two == (2, 0)
    assert sim_one.calls == 1 and sim_two.calls == 1


# ===========================================================================
# Named refusals -- the substrate's silent rules, made loud on this seam.
# ===========================================================================


@pytest.mark.parametrize(
    "tokens, experts, top_k, needle",
    [
        (128, DECLARED_E, DECLARED_TOP_K, "T must be a multiple of 256"),
        (384, DECLARED_E, DECLARED_TOP_K, "T must be a multiple of 256"),
        (DECLARED_T, 513, DECLARED_TOP_K, "E must be <= 512"),
        (DECLARED_T, 4, DECLARED_TOP_K, "E must be >= 8"),
        (DECLARED_T, DECLARED_E, 4, "top_k must be exactly 8"),
        (DECLARED_T, DECLARED_E, 9, "top_k must be exactly 8"),
    ],
)
def test_refused_extents_raise_by_name(tokens, experts, top_k, needle) -> None:
    """Each refusal is a named error carrying the extent, never a silent False.

    The ``T % 256`` row is the load-bearing one. The substrate answers that same
    question by RETURNING FALSE (``rmsnorm_router_topk_tkg.py:209``) and its
    caller then computes torch -- so a ``T = 128`` fixture would have run torch
    against a torch oracle and passed green with zero kernel dispatches. On this
    seam it raises, so that outcome is unreachable.
    """
    logits = torch.zeros(tokens, experts, dtype=torch.float32)
    bias = torch.zeros(1, experts, dtype=torch.float32)
    reset_noaux_tc_counters()
    with pytest.raises(NoauxTcRouterError) as excinfo:
        noaux_tc_correct(logits, bias, top_k=top_k)
    message = str(excinfo.value)
    print(f"[refusal] T={tokens} E={experts} k={top_k} -> {message[:120]}")
    assert needle in message
    # A refusal is not a fallback: neither counter moves.
    assert noaux_tc_dispatch_counters() == (0, 0)


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
    """``H`` needs no acceptance value, and this is why: it is loud on BOTH routes.

    The substrate's ``_validate_inputs`` is called unconditionally at
    ``rmsnorm_router_topk_tkg.py:79``, BEFORE the ``:94`` admission branch, and
    this seam calls it in the same position. So a wrong ``H`` raises whether the
    simulator is on or off -- there is no false-green path on ``H`` for an
    acceptance value to guard.
    """
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
        print(f"[H-loud] NKI_SIMULATOR={simulator} -> {message[:100]}")
        assert "divisible by 256" in message
        assert noaux_tc_dispatch_counters() == (0, 0)


# ===========================================================================
# The FUSED seam -- supplementary coverage at the SAME declared tolerances.
# ===========================================================================


def test_fused_seam_matches_the_reference_on_its_own_logits() -> None:
    """One dispatch for RMSNorm + router matmul + the authored correction.

    The reference reads the logits the KERNEL returned, which is what isolates
    the authored ``noaux_tc`` stage from the substrate's bf16 matmul. Both
    tolerances are the declared ones, unchanged.
    """
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

    want_index, want_affinities = noaux_tc_correct_torch_oracle(
        logits, bias, DECLARED_NORM_TOPK_PROB, DECLARED_ROUTED_SCALING_FACTOR
    )
    equal_rows = set_equal_rows(expert_index, want_index)
    got = expert_affinities.to(torch.float32)
    want = want_affinities.to(torch.float32)
    print(
        f"[fused-seam] set_equal_rows={equal_rows}/{DECLARED_T} "
        f"max_abs_error={float((got - want).abs().max()):.6e} "
        f"logits_absmax={float(logits.abs().max()):.6e}"
    )
    assert equal_rows == DECLARED_ROWS
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_kernel_corrected_selection_differs_from_substrate_selection() -> None:
    """The non-vacuity control read off the kernel's OWN two outputs.

    ``substrate_index`` is nkilib's uncorrected top-8 on the raw logits;
    ``expert_index`` is the authored corrected selection. They must differ, or
    the correction is not in the dispatch.
    """
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
    print(
        f"[non-vacuity-in-kernel] rows_differing_from_substrate_selection="
        f"{moved}/{DECLARED_T} bias_absmax={float(strong_bias.abs().max()):.6e}"
    )
    assert moved > 0, (
        "the corrected selection equals the substrate's uncorrected selection on "
        "every row, so the authored correction is not affecting the dispatch"
    )
    # And it is still the reference's selection, on the kernel's own logits.
    want_index, _ = noaux_tc_correct_torch_oracle(
        logits, strong_bias, DECLARED_NORM_TOPK_PROB, DECLARED_ROUTED_SCALING_FACTOR
    )
    assert set_equal_rows(expert_index, want_index) == DECLARED_ROWS


# ===========================================================================
# The model call site (D14 section: ``Glm5NextMoEBlock`` router call site).
# ===========================================================================


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

    want_index, want_affinities = noaux_tc_correct_torch_oracle(
        logits,
        bank.router_bias.detach(),
        bool(text_config.norm_topk_prob),
        float(text_config.routed_scaling_factor),
    )
    got = expert_affinities.to(torch.float32)
    want = want_affinities.to(torch.float32)
    print(
        f"[model-call-site] set_equal_rows="
        f"{set_equal_rows(expert_index, want_index)}/{DECLARED_T} "
        f"max_abs_error={float((got - want).abs().max()):.6e} "
        f"top_k={text_config.num_experts_per_tok} "
        f"norm_topk_prob={text_config.norm_topk_prob} "
        f"routed_scaling_factor={text_config.routed_scaling_factor}"
    )
    assert set_equal_rows(expert_index, want_index) == DECLARED_ROWS
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_model_call_site_passes_the_checkpoints_own_hyperparameters() -> None:
    """The call site must not hardcode what the config declares."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    text_config = Glm5NextTextConfig()
    print(
        f"[config] num_experts_per_tok={text_config.num_experts_per_tok} "
        f"norm_topk_prob={text_config.norm_topk_prob} "
        f"routed_scaling_factor={text_config.routed_scaling_factor} "
        f"n_routed_experts={text_config.n_routed_experts} "
        f"topk_method={text_config.topk_method!r} "
        f"scoring_func={text_config.scoring_func!r}"
    )
    assert int(text_config.num_experts_per_tok) == DECLARED_TOP_K
    assert bool(text_config.norm_topk_prob) is DECLARED_NORM_TOPK_PROB
    assert (
        float(text_config.routed_scaling_factor) == DECLARED_ROUTED_SCALING_FACTOR
    )
    assert int(text_config.n_routed_experts) == DECLARED_E
    assert text_config.topk_method == "noaux_tc"
    assert text_config.scoring_func == "sigmoid"


def test_pin_router_gate_is_untouched() -> None:
    """The pin's dead ``_can_use_kernel`` short circuit stays in place.

    This increment brings its OWN gate rather than removing the pin's
    unconditional ``return False`` ("TODO: Remove this after debugging
    compilation issue on TRN3"). Removing it would change landed pin behaviour
    for every existing ``router()`` caller, which is a design question and not
    this increment's. Asserted here so a later hand cannot quietly take it out
    under this increment's name.
    """
    import ast
    import inspect
    import textwrap

    from vllm_neuron.functional.moe import router as router_module

    source = textwrap.dedent(inspect.getsource(router_module._can_use_kernel))
    function = ast.parse(source).body[0]
    body = list(function.body)
    # Drop the docstring, if any, by NODE TYPE rather than by string matching:
    # this gate's docstring contains the words "Returns" and "if", so a textual
    # scan for the first statement would read the prose.
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    first = body[0]
    rendered = ast.dump(first)
    print(f"[pin-gate] first executable node: {rendered}")
    assert isinstance(first, ast.Return), (
        f"the pin's dead short circuit is gone; first node is {type(first).__name__}"
    )
    assert isinstance(first.value, ast.Constant) and first.value.value is False, (
        "the pin's unconditional `return False` was changed under this "
        "increment's name; that is a design decision for the lead"
    )


# ===========================================================================
# REPAIR ROUND 1 (batch R6). Two findings, five arms.
#
# M-B20-1 -- the authored `noaux_tc` stage looped over ALL tokens on BOTH
# logical cores while the vendor router had written only its own half, with
# nothing ordering the two. The repair shards the stage's token loop by program
# identifier, through the vendor's OWN sharding call. The arms below read the
# per-core token range the SHIPPED stage asked for, which is the demonstration
# the finding requires; a numeric arm cannot supply it, because the CPU simulator
# serialises the two programs and so hides the hazard entirely.
#
# M-B20-3 -- every arm that ran the fused seam built its reference from the
# logits the kernel itself returned, so the normalisation and the router matmul
# were never compared with anything, and the module's own fused torch reference
# was imported by no test. The arms below compare against that reference, built
# independently from the same inputs, and execute the fallback path.
#
# NOT touched here: the multiple-of-256 refusal (`M-B20-2`) is the `-088`
# token-policy question and belongs to the lead. No tolerance, extent or
# comparator moves in this round.
# ===========================================================================


def test_shard_range_partitions_the_declared_token_extent() -> None:
    """The shard arithmetic, read as plain python at the declared extent.

    ``_noaux_tc_shard_range`` is the function the authored stage uses to decide
    which token rows this core owns. Here it is exercised directly, so the
    per-core reading in the next arm has an exact expectation rather than a
    self-consistent one.

    The property that matters is a PARTITION: the cores' ranges must cover every
    token exactly once, with no overlap and no gap. Overlap would put two cores'
    writes on the same rows -- the defect in a different shape -- and a gap would
    leave rows nobody computed.
    """
    for n_prgs in (1, 2):
        covered: list[int] = []
        ranges = []
        for prg_id in range(n_prgs):
            t_offset, t_local = _noaux_tc_shard_range(DECLARED_T, n_prgs, prg_id)
            ranges.append((t_offset, t_local))
            covered.extend(range(t_offset, t_offset + t_local))
        print(
            f"[shard-range] n_prgs={n_prgs} ranges={ranges} "
            f"covered={len(covered)} distinct={len(set(covered))} "
            f"tiles_per_core={[t // NOAUX_TC_TILE for _o, t in ranges]}"
        )
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

    # THE VENDOR'S TAIL RULE, read rather than trusted. The producer gives the
    # remainder of an uneven split to the SECOND core (`router_topk.py:221-224`).
    # No extent this stage can receive is uneven -- the admission clause forces a
    # multiple of 256 -- so this reading guards the helper against agreeing with
    # the producer only by luck of the admitted extents.
    uneven = [(_noaux_tc_shard_range(7, 2, prg), prg) for prg in (0, 1)]
    print(f"[shard-range] uneven_extent_7_ranges={uneven}")
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
    """THE M-B20-1 DEMONSTRATION: a per-core token range, read from the kernel.

    The finding's evidence bar is "a per-core token range recorded from inside
    the kernel, or a hardware run", and it rules out a repeat of a numeric arm
    because the simulator serialises the two programs and hides the hazard. This
    arm takes the first route: it records every call the SHIPPED stage makes to
    its own shard function while a real two-core launch runs, and reads the
    ranges back.

    Two opposite readings from one instrument, so neither is a constant:

    * the FUSED seam, launched on the ``[2]`` grid, must call the shard function
      once per program, with distinct program identifiers, and receive two
      distinct ranges that partition the token extent;
    * the STANDALONE entry point, launched with no grid, must call it once and
      receive the whole extent at offset 0.

    What this does NOT claim: it is not a hardware run, and it does not observe
    the two cores executing at the same time. It settles the question the finding
    actually raises -- whether the authored stage is grid-aware, and whether each
    core confines itself to the rows whose logits it produced.
    """
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
    print(f"[grid-aware-fused] shard_calls={fused_calls}")

    if not fused_calls:
        raise RouteInstrumentError(
            "the authored stage never asked which tokens this core owns, so it "
            "is not grid-aware and finding M-B20-1 is unrepaired"
        )
    program_ids = sorted({c[2] for c in fused_calls})
    ranges = sorted({(c[3], c[4]) for c in fused_calls})
    n_prgs_seen = sorted({c[1] for c in fused_calls})
    print(
        f"[grid-aware-fused] program_ids={program_ids} n_prgs={n_prgs_seen} "
        f"per_core_ranges={ranges}"
    )
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

    # THE OPPOSITE READING, same instrument: no grid, one program, whole extent.
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
    print(f"[grid-aware-standalone] shard_calls={standalone_calls}")
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
    print(
        f"[grid-aware-union] nonzero_per_row_min={int(nonzero_per_row.min())} "
        f"max={int(nonzero_per_row.max())} rows={int(nonzero_per_row.numel())}"
    )
    assert int(nonzero_per_row.min()) == NOAUX_TC_K, (
        "some token row carries fewer than K gate weights, so no core wrote it "
        "-- the shard leaves a gap"
    )
    assert int(nonzero_per_row.max()) == NOAUX_TC_K


@nki.jit
def _read_per_core_token_range(src):
    """Write this program's id, program count and token range into its own row.

    The row INDEX is the program id, so two programs writing this buffer leave
    two distinct rows and the reading cannot be one program counted twice. The
    token range comes from the SHIPPED ``_noaux_tc_shard_range``, so this kernel
    measures the shipped arithmetic rather than a copy of it.
    """
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
    """Write ``program_id + 1`` into this program's OWN token rows and no others.

    A shard is only correct if the two cores' writes still cover every row. This
    kernel makes the coverage readable: any element left at 0 is a row no core
    wrote, and a row carrying both marker values would be a row two cores wrote.
    """
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
    """The finding's evidence bar, taken literally: the range, read in-kernel.

    The arm above reads the shard decision from the host side, by watching the
    shipped stage ask for it. This arm reads it from INSIDE a kernel body on a
    real ``[2]`` launch: each program stores its own id, the program count and
    its token range into its own row of an output buffer, and the rows come back
    to the host as numbers.

    The second half then reads the CONSEQUENCE rather than the decision. Each
    program marks only its own token rows, and the union must leave no row
    unwritten and no row marked by both -- which is what makes the landed
    declared readings survive a shard that halves each core's work.

    What this does NOT settle: the simulator runs the two programs one after the
    other, so nothing here observes them overlapping in time. The finding is
    about which rows each core touches, and that is what is measured.
    """
    src = torch.zeros(DECLARED_T, 8, dtype=torch.float32)
    rows = wrap_nki(_read_per_core_token_range)[2](src=src).to(torch.float32).tolist()
    print("[in-kernel-range] raw rows (prg_id, n_prgs, t_offset, t_local):")
    for slot, row in enumerate(rows):
        print(f"[in-kernel-range]   slot {slot}: {row}")
    # The buffer has four slots and the launch has two programs, so the two
    # unused slots stay at their zero fill. `n_prgs` can never be 0 in a row a
    # program actually wrote, which is what makes it the liveness marker.
    live = [row for row in rows if row[1] != 0.0]
    program_ids = sorted({int(row[0]) for row in live})
    n_prgs_seen = sorted({int(row[1]) for row in live})
    ranges = sorted({(int(row[2]), int(row[3])) for row in live})
    print(
        f"[in-kernel-range] live_rows={len(live)} program_ids={program_ids} "
        f"n_prgs={n_prgs_seen} per_core_ranges={ranges}"
    )
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
    print(
        f"[in-kernel-union] top_half_values={top} bottom_half_values={bottom} "
        f"unwritten_elements={unwritten}/{union.numel()}"
    )
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
    """THE M-B20-3 DEMONSTRATION: the normalisation and the matmul, compared.

    Every landed fused-seam arm builds its reference from the logits the kernel
    returned, which isolates the authored correction but leaves the two substrate
    stages measured against nothing. This arm builds the whole reference from the
    SAME hidden states, gamma and router weights the kernel was given, using the
    module's own fused torch reference, and compares the logits.

    The tolerances are the landed declared ones, taken from this file's own
    constants and not restated. Nothing here moves a declared value.
    """
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

    # INDEPENDENT: built from the inputs, never from the kernel's own output.
    want_logits, want_index, want_affinities, _want_sub = (
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
    want = want_logits.to(torch.float32)
    allowed = ATOL + RTOL * want.abs()
    outside = int(((got - want).abs() > allowed).sum())
    equal_rows = set_equal_rows(expert_index, want_index)
    print(
        f"[independent-reference] eps={seam_eps} "
        f"logits_max_abs_error={float((got - want).abs().max()):.6e} "
        f"elements_outside_tolerance={outside}/{got.numel()} "
        f"set_equal_rows={equal_rows}/{DECLARED_T} "
        f"affinities_max_abs_error="
        f"{float((expert_affinities.to(torch.float32) - want_affinities.to(torch.float32)).abs().max()):.6e} "
        f"declared rtol={RTOL} atol={ATOL}"
    )
    # The reference must not be vacuous: an all-zero reference would pass.
    assert float(want.abs().max()) > 0.0
    assert equal_rows == DECLARED_ROWS, (
        "the kernel's selection differs from a reference built independently "
        "from the same inputs, so a substrate stage is wrong"
    )
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        expert_affinities.to(torch.float32),
        want_affinities.to(torch.float32),
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
    """The non-vacuity control for the arm above, on the two inputs it guards.

    The previous arm is only worth having if it can FAIL. The defect that
    motivated the finding was a wrong normalisation epsilon that every landed arm
    read as green, because both sides consumed the same wrong logits. Here the
    reference is rebuilt with a perturbed epsilon, and separately with a perturbed
    gamma, and the comparison must go outside the declared tolerance.

    The perturbations are this file's own choices, not declared values: the
    epsilon moves from the seam's default to the value the pinned config actually
    carries, and gamma is scaled by 5 per cent.
    """
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

    want_logits = noaux_tc_rmsnorm_router_topk_torch_oracle(
        hidden_states,
        ref_gamma,
        router_weights,
        bias,
        ref_eps,
        DECLARED_NORM_TOPK_PROB,
        DECLARED_ROUTED_SCALING_FACTOR,
    )[0]
    got = logits.to(torch.float32)
    want = want_logits.to(torch.float32)
    allowed = ATOL + RTOL * want.abs()
    outside = int(((got - want).abs() > allowed).sum())
    print(
        f"[non-vacuity-{perturbation}] perturbed_eps={ref_eps} "
        f"gamma_absmax={float(ref_gamma.to(torch.float32).abs().max()):.6f} "
        f"max_abs_error={float((got - want).abs().max()):.6e} "
        f"elements_outside_tolerance={outside}/{got.numel()} "
        f"declared rtol={RTOL} atol={ATOL}"
    )
    assert outside > 0, (
        f"perturbing the {perturbation} left every element inside the declared "
        f"tolerance, so the independent-reference arm above cannot see a wrong "
        f"normalisation and is vacuous"
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_fused_torch_fallback_executes_and_is_the_cpu_oracle(monkeypatch) -> None:
    """The fused seam's torch fallback runs, and agrees with the kernel.

    Finding ``M-B20-3`` records that ``noaux_tc_rmsnorm_router_topk_torch_oracle``
    shipped with no execution coverage at all -- nothing imported it. It is the
    fallback the seam takes when ``can_run_kernel`` is false, so this arm flips
    that live gate, checks the fallback counter moved and the simulator did not
    run, and then checks the fallback is the CORRECT function by comparing it
    against the kernel's own result at the declared tolerances.

    Same discipline as ``test_route_control_fallback_counter_discriminates``,
    one level up: that arm covers the standalone entry point, this one the fused
    seam.
    """
    hidden_states, gamma, router_weights, bias = build_hidden_states()
    seam_eps = inspect.signature(
        noaux_tc_rmsnorm_router_topk
    ).parameters["eps"].default

    # The kernel result first, with the gate ON.
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

    # Now the same call with the gate OFF, which must take the fused fallback.
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
    print(
        f"[fused-fallback] counters={readings} "
        f"simulate_kernel_calls={sim_off.calls} "
        f"logits_shape={tuple(fb_logits.shape)} "
        f"index_shape={tuple(fb_index.shape)} "
        f"affinities_shape={tuple(fb_affinities.shape)} "
        f"substrate_index_shape={tuple(fb_sub.shape)} "
        f"logits_max_abs_error_vs_kernel="
        f"{float((fb_logits.to(torch.float32) - kernel_logits.to(torch.float32)).abs().max()):.6e}"
    )
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
