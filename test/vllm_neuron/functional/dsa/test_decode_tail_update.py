# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for ``inc-glm53f-049`` -- the decode-step k-pool tail update.

WHAT THIS FILE MEASURES, and the shape of the argument it makes:

A stepped state update is correct only if ``k`` steps land where a FULL RECOMPUTE from the same
``k`` tokens lands. That equivalence IS the measurement, and it has two halves, both compared on
every declared case:

* the compressed keys of every pool that ended during the ``k`` steps, in order, and
* the ring itself after all ``k`` steps.

THE RECOMPUTE DOES NOT STEP. It is ``decode_tail_recompute``, which works from the token stream:
the members of a pool that ends at position ``p`` are the tokens at ``p - pool_size + 1 .. p``,
taken from the decode window where they fall inside it and from the seeded ring where they do not.
It never consults a stepped ring, so an error in the ring's addressing cannot hide by appearing on
both sides of the comparison. It also dispatches no kernel and touches no counter -- see the two
readings under "the reference is produced outside the seam".

EVERY DECLARED CASE COMPLETES AT LEAST ONE POOL. All three start at position ``pool_size - 1``,
which is the realistic decode entry (a prefill seeds the ring, then the first decode token finishes
the prompt's trailing pool) and is also what stops the ``k = 1`` case from being vacuous on the
arithmetic. A ``k = 1`` case that completed nothing would compare an empty list of pooled keys and
would measure the ring copy alone.

ONE ITEM PER COUNTED CONJUNCT and no ``parametrize`` (plan section 6, rules 4b and 6), so a failure
names the reading that failed instead of a parameter id. Counters are reset at the START of each
declared case and read at its END (section 4b's per-case convention).

EVERY COUNTED ZERO HAS A CONTROL THAT FIRES (plan D1.5). This file counts three zeros -- the
NKI-dispatch counter across the recompute, the torch-fallback counter across the declared cases, and
the pooled-key count on a non-completing step -- and each has a companion item that makes the same
reading move.

THE ENVIRONMENT IS PINNED IN THE INVOCATION, NEVER IN A FIXTURE (plan D2): this file is run under
``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2``,
with ``-s`` so the ``_emit`` lines reach the transcript. Nothing here reads or sets an environment
variable.

A NOTE ON WHAT THE SIMULATOR PROVES. ``NKI_SIMULATOR=1`` does not run the MLIR verifier, so a pass
here is evidence about VALUES and never about COMPILABILITY. Compilability is settled separately, by
this increment's capture leg with the verifier on.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.dsa.decode_tail_update import (
    DEFAULT_POOL_SIZE,
    TAIL_HALVES,
    DecodeTailUpdateError,
    _compress_pool_torch,
    can_run_dsa_decode_tail_update,
    completes_pool,
    decode_tail_dispatch_counters,
    decode_tail_kernel_identity,
    decode_tail_recompute,
    dsa_decode_tail_update,
    reset_decode_tail_dispatch_counters,
    slot_of,
)
from vllm_neuron.functional.dsa.kpool_hadamard import (
    HADAMARD_SCALE,
    INDEX_HEAD_DIM,
    hadamard_matrix,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

POOL_SIZE = DEFAULT_POOL_SIZE
"""``index_kpool`` on the target checkpoint (``fixtures/hf-config.json``), read from the module."""

START_POSITION = POOL_SIZE - 1
"""Every declared case's first position, so every case completes a pool on its first step."""

RTOL = 1e-2
ATOL = 1e-5
"""The Acceptance bullet's tolerance for the stepped-versus-recomputed equivalence, verbatim."""

DECLARED_K = (1, 4, 16)
"""The Acceptance bullet's step counts, verbatim. 3 cases, 1 + 4 + 16 = 21 steps in total."""

EXPECTED_POOLS = {1: 1, 4: 1, 16: 4}
"""Pools completed per case, from ``START_POSITION``. Derived and then asserted, not assumed:
positions ``3 .. 3+k-1``, of which those with ``p % 4 == 3`` complete -- k=1: {3}; k=4: {3};
k=16: {3, 7, 11, 15}."""


def _inputs(k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One case's four input tensors: ``(tail0, keys, scores, ape)``.

    The ring and the tokens are bf16, which is upstream's own assert (``:651``, ``:658-659``), and
    ``ape`` is fp32 (``:661``). ``tail0`` is RANDOM rather than zero: a zero-seeded ring would let a
    kernel that read the wrong ring row still pass, because every wrong row would hold the same
    value.
    """
    gen = torch.Generator().manual_seed(seed)

    def bf16(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=gen, dtype=torch.float32).to(torch.bfloat16)

    tail0 = bf16(TAIL_HALVES, POOL_SIZE, INDEX_HEAD_DIM)
    keys = bf16(k, INDEX_HEAD_DIM)
    scores = bf16(k, INDEX_HEAD_DIM)
    ape = torch.randn((POOL_SIZE, INDEX_HEAD_DIM), generator=gen, dtype=torch.float32)
    return tail0, keys, scores, ape


def _stepped(
    tail0: torch.Tensor,
    keys: torch.Tensor,
    scores: torch.Tensor,
    ape: torch.Tensor,
    start_position: int = START_POSITION,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Run the seam once per token, threading the ring. Returns ``(pooled_list, tail_k)``.

    This is the ONLY path in this file that calls the seam, so every dispatch a case counts came
    from here.
    """
    tail = tail0
    pooled: list[torch.Tensor] = []
    for step in range(int(keys.shape[0])):
        out, tail = dsa_decode_tail_update(
            tail, keys[step : step + 1], scores[step : step + 1], ape, start_position + step
        )
        if out is not None:
            pooled.append(out)
    return pooled, tail


def _emit(tag: str, **values: object) -> None:
    """Print one MACHINE-READABLE reading line, for the driver to re-check independently.

    The landed pattern (``test_ragged_pack.py``, ``test_kpool_hadamard.py``): the test asserts, and
    then PRINTS the value it asserted on, so the driver that owns the transcript can check the same
    number without trusting the assertion. That is D1.3's read-and-record.
    """
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"[{tag}] {body}")


def _diff(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, int]:
    """``(max_abs_diff, differing_elements)`` between two tensors, both as plain python numbers."""
    a, b = got.float(), expected.float()
    return (a - b).abs().max().item(), int((a != b).sum().item())


def _worst(got: list[torch.Tensor], expected: list[torch.Tensor]) -> tuple[float, int]:
    """The worst ``(max_abs_diff, differing_elements)`` over a list of pooled keys."""
    worst_abs, worst_n = 0.0, 0
    for g, e in zip(got, expected, strict=True):
        max_abs, differing = _diff(g, e)
        worst_abs = max(worst_abs, max_abs)
        worst_n = max(worst_n, differing)
    return worst_abs, worst_n


def _equivalence_case(k: int, seed: int) -> None:
    """One declared case, both halves. Asserts, emits, and is called by exactly one item per ``k``.

    The recompute runs BEFORE the reset, so the reset cannot be credited with hiding a dispatch the
    reference made: the counters are zeroed after the reference has already run.
    """
    tail0, keys, scores, ape = _inputs(k, seed=seed)
    want_pooled, want_tail = decode_tail_recompute(tail0, keys, scores, ape, START_POSITION)
    reset_decode_tail_dispatch_counters()
    got_pooled, got_tail = _stepped(tail0, keys, scores, ape)

    pool_abs, pool_n = _worst(got_pooled, want_pooled)
    tail_abs, tail_n = _diff(got_tail, want_tail)
    _emit(
        "acceptance",
        k=k,
        steps=k,
        pools_completed=len(got_pooled),
        pools_expected=EXPECTED_POOLS[k],
        pooled_max_abs_diff=f"{pool_abs:.6e}",
        pooled_differing_elements=pool_n,
        pooled_population=f"{len(got_pooled)} pooled keys of {INDEX_HEAD_DIM} channels each",
        tail_max_abs_diff=f"{tail_abs:.6e}",
        tail_differing_elements=tail_n,
        tail_population=f"{TAIL_HALVES}x{POOL_SIZE}x{INDEX_HEAD_DIM} ring elements",
        rtol=RTOL,
        atol=ATOL,
    )
    assert len(got_pooled) == EXPECTED_POOLS[k]
    assert len(want_pooled) == EXPECTED_POOLS[k]
    for g, e in zip(got_pooled, want_pooled, strict=True):
        assert tuple(g.shape) == (1, INDEX_HEAD_DIM)
        torch.testing.assert_close(g.float(), e.float(), rtol=RTOL, atol=ATOL)
    assert tuple(got_tail.shape) == (TAIL_HALVES, POOL_SIZE, INDEX_HEAD_DIM)
    torch.testing.assert_close(got_tail.float(), want_tail.float(), rtol=RTOL, atol=ATOL)
    # The ring half is a COPY on both sides, so it is exactly equal and the tolerance above is not
    # doing the work there. Said out loud, because a tolerance that covers an exact reading hides
    # how much slack the arithmetic half is actually using.
    assert torch.equal(got_tail, want_tail)


def _route_case(k: int, seed: int) -> None:
    """One declared case's route predicate: exactly ``k`` dispatches, zero torch fallbacks."""
    tail0, keys, scores, ape = _inputs(k, seed=seed)
    reset_decode_tail_dispatch_counters()
    gate = can_run_dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape)
    _stepped(tail0, keys, scores, ape)
    nki, fallback = decode_tail_dispatch_counters()
    _emit(
        "route-predicate",
        k=k,
        steps=k,
        nki_dispatch=nki,
        torch_fallback=fallback,
        can_run_kernel=gate,
        population=f"{k} seam calls, one per decode token",
    )
    assert (nki, fallback) == (k, 0)
    assert gate is True


# ---------------------------------------------------------------------------------------------
# The gate. Nothing below means anything if the NKI route is not the route taken.
# ---------------------------------------------------------------------------------------------


def test_gate_can_run_kernel_is_true() -> None:
    """``can_run_kernel()`` is True, so every case below takes the NKI route and not the fallback."""
    assert can_run_kernel() is True


def test_gate_admits_the_declared_input() -> None:
    """The gate admits the declared shapes and dtypes."""
    tail0, keys, scores, ape = _inputs(1, seed=1)
    assert can_run_dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape) is True


def test_gate_refuses_an_fp32_ring() -> None:
    """CONTROL that the gate can read False: an fp32 ring is outside the supported dtypes.

    Without this, "the gate returned True" would be a reading from an instrument that might always
    return True.
    """
    tail0, keys, scores, ape = _inputs(1, seed=2)
    assert can_run_dsa_decode_tail_update(tail0.float(), keys[0:1], scores[0:1], ape) is False


# ---------------------------------------------------------------------------------------------
# The ring's addressing, checked against the reference's own pointer arithmetic.
# ---------------------------------------------------------------------------------------------


def test_ring_row_equals_the_pool_slot_over_the_declared_position_range() -> None:
    """The module's ``pool_slot`` IS upstream's ``phys``, for every position in range.

    Upstream computes ``phys = (pool_logical_start + pool_slot) % POOL_SIZE`` with
    ``pool_logical_start = position - slot`` (``:522``, ``:527``). The module drops the modulo and
    writes ``pool_slot`` directly, on the argument that ``position - slot`` is a multiple of
    ``pool_size``. That argument is ASSERTED here over 64 positions rather than believed from the
    docstring: this item recomputes upstream's expression and compares.
    """
    checked = 0
    for position in range(64):
        slot = slot_of(position, POOL_SIZE)
        pool_logical_start = position - slot
        for pool_slot in range(POOL_SIZE):
            phys = (pool_logical_start + pool_slot) % POOL_SIZE
            assert phys == pool_slot
            checked += 1
    _emit("acceptance", case="ring_addressing", identities_checked=checked,
          population="64 positions x 4 pool members")
    assert checked == 64 * POOL_SIZE


def test_completes_pool_is_the_last_slot_of_each_pool() -> None:
    """``completes_pool`` fires on exactly one position in four, and it is the last of each pool."""
    firing = [p for p in range(64) if completes_pool(p, POOL_SIZE)]
    _emit("acceptance", case="completion_positions", firing=len(firing),
          population="64 consecutive positions")
    assert firing == list(range(POOL_SIZE - 1, 64, POOL_SIZE))
    assert len(firing) == 64 // POOL_SIZE


# ---------------------------------------------------------------------------------------------
# The declared cases: 3/3 over k, two counted conjuncts each, one item apiece.
# ---------------------------------------------------------------------------------------------


def test_stepped_matches_full_recompute_k_1() -> None:
    """CASE 1/3, ``k = 1``: one decode token, which completes the prefill-seeded prompt tail."""
    _equivalence_case(1, seed=301)


def test_route_k_1() -> None:
    """CASE 1/3 route: exactly 1 NKI dispatch and zero torch fallbacks."""
    _route_case(1, seed=301)


def test_stepped_matches_full_recompute_k_4() -> None:
    """CASE 2/3, ``k = 4``: one completion on the first step, then three non-completing steps.

    The three that complete nothing are the ones the reference's own measured defect was about
    (``:502-507``): a stash gated on pool-granular validity would drop all three.
    """
    _equivalence_case(4, seed=302)


def test_route_k_4() -> None:
    """CASE 2/3 route: exactly 4 NKI dispatches and zero torch fallbacks."""
    _route_case(4, seed=302)


def test_stepped_matches_full_recompute_k_16() -> None:
    """CASE 3/3, ``k = 16``: four completions, so the ring wraps four times.

    This is the case that reads the ring as a RING: every row is written four times and read as a
    pool member three times between writes.
    """
    _equivalence_case(16, seed=303)


def test_route_k_16() -> None:
    """CASE 3/3 route: exactly 16 NKI dispatches and zero torch fallbacks."""
    _route_case(16, seed=303)


def test_dispatch_total_over_the_declared_case_set_is_21() -> None:
    """The declared total: ``1 + 4 + 16`` dispatches counted in ONE reset window.

    The per-case items each read their own window; this item reads the SUM the route predicate
    declares, so the two readings cannot disagree without one of them failing.
    """
    reset_decode_tail_dispatch_counters()
    for k, seed in zip(DECLARED_K, (301, 302, 303), strict=True):
        tail0, keys, scores, ape = _inputs(k, seed=seed)
        _stepped(tail0, keys, scores, ape)
    nki, fallback = decode_tail_dispatch_counters()
    _emit("route-predicate", case="declared_total", nki_dispatch=nki, torch_fallback=fallback,
          population="21 seam calls across the 3 declared cases")
    assert (nki, fallback) == (sum(DECLARED_K), 0)
    assert sum(DECLARED_K) == 21


# ---------------------------------------------------------------------------------------------
# The reference is produced OUTSIDE the seam -- and the controls that prove both counters move.
# ---------------------------------------------------------------------------------------------


def test_the_full_recompute_reference_dispatches_zero_nki_kernels() -> None:
    """``decode_tail_recompute`` runs NO kernel and touches NEITHER counter.

    Both halves matter. Zero dispatches is what stops the equivalence from being the kernel
    agreeing with itself. Zero FALLBACKS is what makes the declared cases' ``torch_fallback == 0``
    readable at all: the reference is torch, so a reference that raised the fallback counter would
    put a nonzero reading on every case that compares against it. The module splits the two roles
    for exactly this reason.
    """
    tail0, keys, scores, ape = _inputs(16, seed=401)
    reset_decode_tail_dispatch_counters()
    decode_tail_recompute(tail0, keys, scores, ape, START_POSITION)
    nki, fallback = decode_tail_dispatch_counters()
    _emit("control", case="reference_outside_the_seam", nki_dispatch=nki, torch_fallback=fallback,
          population="1 recompute over 16 tokens")
    assert (nki, fallback) == (0, 0)


def test_control_the_dispatch_counter_does_move_on_the_seam() -> None:
    """CONTROL for the zero above: the same counter reads 1 after one seam call.

    A zero from a counter that cannot move measures nothing. This item makes it move.
    """
    tail0, keys, scores, ape = _inputs(1, seed=401)
    reset_decode_tail_dispatch_counters()
    _stepped(tail0, keys, scores, ape)
    nki = decode_tail_dispatch_counters()[0]
    _emit("control", case="dispatch_counter_moves", nki_dispatch=nki, population="1 seam call")
    assert nki == 1


def test_control_the_fallback_counter_does_move_on_an_unadmitted_dtype() -> None:
    """CONTROL for the declared cases' ``torch_fallback == 0``: an fp32 ring takes the fallback.

    fp32 is outside the module's supported dtypes, so this exercises the REAL gate rather than a
    monkeypatched one.
    """
    tail0, keys, scores, ape = _inputs(1, seed=402)
    reset_decode_tail_dispatch_counters()
    gate = can_run_dsa_decode_tail_update(tail0.float(), keys[0:1], scores[0:1], ape)
    dsa_decode_tail_update(tail0.float(), keys[0:1].float(), scores[0:1].float(), ape,
                           START_POSITION)
    nki, fallback = decode_tail_dispatch_counters()
    _emit("control", case="unadmitted_dtype", dtype="torch.float32", can_run_kernel=gate,
          nki_dispatch=nki, torch_fallback=fallback, population="1 seam call")
    assert (nki, fallback) == (0, 1)
    assert gate is False


# ---------------------------------------------------------------------------------------------
# The certifying component: identity derived THROUGH the seam (D13.1).
# ---------------------------------------------------------------------------------------------


def test_kernel_identity_is_none_before_any_dispatch() -> None:
    """Before a dispatch the identity is ``None`` -- the reading that separates "none ran"."""
    reset_decode_tail_dispatch_counters()
    assert decode_tail_kernel_identity() is None


def test_kernel_identity_names_this_modules_kernel_after_a_dispatch() -> None:
    """After a step the identity is THIS module's kernel, read through the seam and not the import
    list."""
    tail0, keys, scores, ape = _inputs(1, seed=403)
    reset_decode_tail_dispatch_counters()
    _stepped(tail0, keys, scores, ape)
    module, qualname = decode_tail_kernel_identity()
    _emit("acceptance", case="kernel_identity", module=module, qualname=qualname)
    assert module.endswith("decode_tail_update")
    assert qualname == "_decode_tail_update_nki"


# ---------------------------------------------------------------------------------------------
# The stash: on EVERY token, and not only on the ones that complete a pool.
# ---------------------------------------------------------------------------------------------


def test_a_non_completing_step_returns_no_pooled_key() -> None:
    """A step at slot 0 of 4 completes nothing, so the seam returns ``None`` and not a zero row.

    ``None`` rather than zeros, so a caller cannot mistake a legitimate all-zero pooled key for
    "no pool ended".
    """
    tail0, keys, scores, ape = _inputs(1, seed=404)
    reset_decode_tail_dispatch_counters()
    pooled, _ = dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape, 0)
    nki, fallback = decode_tail_dispatch_counters()
    _emit("acceptance", case="non_completing_step", position=0, pooled_is_none=pooled is None,
          nki_dispatch=nki, torch_fallback=fallback)
    assert pooled is None
    # CONTROL for that None: the same seam at position 3 returns a tensor, so the reading
    # discriminates rather than always being None.
    completing, _ = dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape, POOL_SIZE - 1)
    assert completing is not None
    assert tuple(completing.shape) == (1, INDEX_HEAD_DIM)
    assert (nki, fallback) == (1, 0)


def test_the_stash_happens_on_a_step_that_completes_nothing() -> None:
    """A non-completing step still writes the token to its ring row, and leaves the others alone.

    THIS IS THE READING UPSTREAM'S OWN COMMENT SAYS IT GOT WRONG ONCE (``:502-507``): gating the
    stash on pool-granular validity dropped every intra-pool token, so a decode-built pool
    compressed stale prefill entries. Both halves of the ring are checked, and so is every row the
    step must NOT touch.
    """
    tail0, keys, scores, ape = _inputs(1, seed=405)
    reset_decode_tail_dispatch_counters()
    _, new_tail = dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape, 0)
    untouched = [row for row in range(POOL_SIZE) if row != 0]
    _emit("acceptance", case="stash_without_completion", position=0, slot=0,
          rows_required_unchanged=len(untouched) * TAIL_HALVES,
          population=f"{TAIL_HALVES}x{POOL_SIZE} ring rows")
    assert torch.equal(new_tail[0, 0], keys[0])
    assert torch.equal(new_tail[1, 0], scores[0])
    for row in untouched:
        assert torch.equal(new_tail[0, row], tail0[0, row])
        assert torch.equal(new_tail[1, row], tail0[1, row])


def test_the_stash_happens_on_a_completing_step_too() -> None:
    """A completing step ALSO stashes, because the token belongs to future pools as well.

    Upstream: "then leaves this token for future pools" (``:597-598``). A kernel that stashed only
    on non-completing steps would pass every equivalence case whose ``k`` never re-read the
    completing slot, so this reading is taken directly.
    """
    tail0, keys, scores, ape = _inputs(1, seed=406)
    slot = POOL_SIZE - 1
    reset_decode_tail_dispatch_counters()
    pooled, new_tail = dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape, slot)
    _emit("acceptance", case="stash_with_completion", position=slot, slot=slot,
          pooled_is_none=pooled is None)
    assert pooled is not None
    assert torch.equal(new_tail[0, slot], keys[0])
    assert torch.equal(new_tail[1, slot], scores[0])


def test_input_ring_is_not_mutated_by_a_step() -> None:
    """The seam RETURNS a new ring and does not write its argument.

    The ring is state the caller threads from step to step, and the equivalence cases hold the
    original ``tail0`` while they run. If a step mutated it, the recompute -- which reads ``tail0``
    for the pre-decode members -- would be reading a ring the steps had already advanced, and the
    comparison would be against a moving reference.
    """
    tail0, keys, scores, ape = _inputs(4, seed=407)
    before = tail0.clone()
    reset_decode_tail_dispatch_counters()
    _stepped(tail0, keys, scores, ape)
    _emit("acceptance", case="argument_not_mutated",
          differing_elements=int((tail0 != before).sum().item()),
          population=f"{TAIL_HALVES}x{POOL_SIZE}x{INDEX_HEAD_DIM} ring elements")
    assert torch.equal(tail0, before)


# ---------------------------------------------------------------------------------------------
# Discriminating cases: two wrong kernels that the declared cases must NOT pass.
# ---------------------------------------------------------------------------------------------


def test_completion_reads_the_current_token_and_not_the_stale_ring_row() -> None:
    """The pool's own slot takes the CURRENT token, not whatever the ring still holds there.

    Upstream selects on ``is_current`` for both the score and the key (``:533``, ``:564``). A kernel
    that read every member from the ring would differ only in the completing slot -- and only when
    the ring's stale row differs from the current token, which is why ``tail0`` is random here and
    the stale row is made deliberately far away.
    """
    tail0, keys, scores, ape = _inputs(1, seed=408)
    slot = POOL_SIZE - 1
    stale = tail0.clone()
    stale[0, slot] = 8.0
    stale[1, slot] = -8.0
    reset_decode_tail_dispatch_counters()
    want, _ = decode_tail_recompute(stale, keys, scores, ape, slot)
    got, _ = _stepped(stale, keys, scores, ape, start_position=slot)
    max_abs, differing = _diff(got[0], want[0])
    _emit("acceptance", case="current_token_not_stale_row", max_abs_diff=f"{max_abs:.6e}",
          differing_elements=differing, rtol=RTOL, atol=ATOL,
          population=f"1 pooled key of {INDEX_HEAD_DIM} channels")
    torch.testing.assert_close(got[0].float(), want[0].float(), rtol=RTOL, atol=ATOL)


def test_control_the_stale_row_variant_is_far_outside_the_tolerance() -> None:
    """CONTROL for the item above: the wrong kernel's answer is nowhere near the right one.

    Without this, the item above would pass even if reading the stale row happened to give the same
    numbers. Here the stale-row composition is built explicitly and its gap from the correct one is
    measured against ten times the acceptance tolerance.
    """
    tail0, keys, scores, ape = _inputs(1, seed=408)
    slot = POOL_SIZE - 1
    stale = tail0.clone()
    stale[0, slot] = 8.0
    stale[1, slot] = -8.0
    correct = _compress_pool_torch(
        torch.cat([stale[0, :slot], keys[0:1]]),
        torch.cat([stale[1, :slot], scores[0:1]]),
        ape,
    )
    wrong = _compress_pool_torch(stale[0], stale[1], ape)
    gap = (correct.float() - wrong.float()).abs().max().item()
    threshold = 10.0 * (ATOL + RTOL * correct.float().abs().max().item())
    _emit("control", case="stale_row_variant", gap=f"{gap:.6e}", threshold=f"{threshold:.6e}")
    assert gap > threshold


def test_per_slot_channel_softmax_is_distinguishable_from_a_whole_vector_softmax() -> None:
    """A softmax over the 128 channels instead of the 4 slots is a DIFFERENT kernel, and this
    file's inputs can tell them apart.

    The same discriminator ``-047`` carries, taken again here because this module has its own
    softmax and a wrong axis in it would be invisible to that increment's tests.
    """
    tail0, keys, scores, ape = _inputs(1, seed=409)
    slot = POOL_SIZE - 1
    pool_key = torch.cat([tail0[0, :slot], keys[0:1]])
    pool_score = torch.cat([tail0[1, :slot], scores[0:1]])
    correct = _compress_pool_torch(pool_key, pool_score, ape)
    biased = torch.softmax(pool_score.float() + ape.float(), dim=-1)
    wrong_pooled = (biased * pool_key.float()).sum(dim=0, keepdim=True)
    wrong = (wrong_pooled @ hadamard_matrix(INDEX_HEAD_DIM).t() * HADAMARD_SCALE).to(tail0.dtype)
    gap = (correct.float() - wrong.float()).abs().max().item()
    threshold = 10.0 * (ATOL + RTOL * correct.float().abs().max().item())
    _emit("control", case="wrong_axis_softmax", gap=f"{gap:.6e}", threshold=f"{threshold:.6e}")
    assert gap > threshold


# ---------------------------------------------------------------------------------------------
# Malformed calls are refused rather than silently reshaped.
# ---------------------------------------------------------------------------------------------


def test_malformed_key_shape_is_refused() -> None:
    """A key that is not ``[1, head_dim]`` raises instead of being broadcast or squeezed."""
    tail0, keys, scores, ape = _inputs(2, seed=410)
    with pytest.raises(DecodeTailUpdateError):
        dsa_decode_tail_update(tail0, keys, scores[0:1], ape, START_POSITION)


def test_negative_position_is_refused() -> None:
    """A negative position raises rather than being treated as a padded entry.

    Upstream carries padded entries and screens them with ``pos_valid`` (``:518-521``). This module
    takes one real token per call, so a negative position is a caller error and is refused at the
    boundary instead of being silently skipped.
    """
    tail0, keys, scores, ape = _inputs(1, seed=411)
    with pytest.raises(DecodeTailUpdateError):
        dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape, -1)
