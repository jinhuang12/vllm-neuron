# SPDX-License-Identifier: Apache-2.0
"""Tests for the decode-step k-pool tail update.

A stepped state update is correct only if ``k`` steps land where a full recompute from
the same ``k`` tokens lands, so that equivalence is the measurement, in both halves:
the compressed keys of every pool that ended during the ``k`` steps, in order, and the
ring itself after all ``k`` steps.

The reference, ``decode_tail_recompute``, does not step. It works from the token
stream -- the members of a pool that ends at position ``p`` are the tokens at
``p - pool_size + 1 .. p``, taken from the decode window where they fall inside it and
from the seeded ring where they do not -- so an error in the ring's addressing cannot
hide by appearing on both sides of the comparison.

Every case starts at position ``pool_size - 1``, the realistic decode entry (a prefill
seeds the ring, then the first decode token finishes the prompt's trailing pool), so
even the ``k = 1`` case completes a pool instead of comparing an empty list.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.dsa.decode_tail_update import (
    DEFAULT_POOL_SIZE,
    TAIL_HALVES,
    DecodeTailUpdateError,
    can_run_dsa_decode_tail_update,
    completes_pool,
    decode_tail_dispatch_counters,
    decode_tail_kernel_identity,
    decode_tail_recompute,
    dsa_decode_tail_update,
    reset_decode_tail_dispatch_counters,
    slot_of,
)
from vllm_neuron.functional.dsa.kpool_hadamard import INDEX_HEAD_DIM
from vllm_neuron.utils.neuron_utils import can_run_kernel

POOL_SIZE = DEFAULT_POOL_SIZE
"""``index_kpool`` on the target checkpoint, read from the module."""

START_POSITION = POOL_SIZE - 1
"""Every case's first position, so every case completes a pool on its first step."""

RTOL = 1e-2
ATOL = 1e-5
"""Tolerance for the stepped-versus-recomputed equivalence."""

EXPECTED_POOLS = {1: 1, 4: 1, 16: 4}
"""Pools completed per case, from ``START_POSITION``: positions ``3 .. 3+k-1``, of
which those with ``p % 4 == 3`` complete -- k=1: {3}; k=4: {3}; k=16: {3, 7, 11, 15}."""


def _inputs(k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One case's four input tensors: ``(tail0, keys, scores, ape)``.

    The ring and the tokens are bf16 and ``ape`` is fp32, which is what upstream
    asserts. ``tail0`` is random rather than zero: a zero-seeded ring would let a
    kernel that read the wrong ring row still pass, because every wrong row would hold
    the same value.
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

    This is the only path in this file that calls the seam, so every dispatch a case
    counts came from here.
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


def _assert_equivalent_to_recompute(k: int, seed: int) -> None:
    """Both halves of one case: the pooled keys in order, then the ring.

    The recompute runs before the reset, so the counters are zeroed after the reference
    has already run and cannot be credited with hiding a dispatch the reference made.
    """
    tail0, keys, scores, ape = _inputs(k, seed=seed)
    expected_pooled, expected_tail = decode_tail_recompute(
        tail0, keys, scores, ape, START_POSITION
    )
    reset_decode_tail_dispatch_counters()
    got_pooled, got_tail = _stepped(tail0, keys, scores, ape)

    assert len(got_pooled) == EXPECTED_POOLS[k]
    assert len(expected_pooled) == EXPECTED_POOLS[k]
    for got, expected in zip(got_pooled, expected_pooled, strict=True):
        assert tuple(got.shape) == (1, INDEX_HEAD_DIM)
        torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)
    assert tuple(got_tail.shape) == (TAIL_HALVES, POOL_SIZE, INDEX_HEAD_DIM)
    torch.testing.assert_close(got_tail.float(), expected_tail.float(), rtol=RTOL, atol=ATOL)
    # The ring half is a copy on both sides, so it is exactly equal and the tolerance
    # above is not doing the work there.
    assert torch.equal(got_tail, expected_tail)


def _assert_one_dispatch_per_token(k: int, seed: int) -> None:
    """One case's route: exactly ``k`` dispatches, zero torch fallbacks."""
    tail0, keys, scores, ape = _inputs(k, seed=seed)
    reset_decode_tail_dispatch_counters()
    gate = can_run_dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape)
    _stepped(tail0, keys, scores, ape)
    nki, fallback = decode_tail_dispatch_counters()
    assert (nki, fallback) == (k, 0)
    assert gate is True


# ---------------------------------------------------------------------------------------------
# The gate.
# ---------------------------------------------------------------------------------------------


def test_gate_can_run_kernel_is_true() -> None:
    """``can_run_kernel()`` is True, so every case below takes the NKI route."""
    assert can_run_kernel() is True


def test_gate_admits_the_declared_input() -> None:
    """The gate admits the supported shapes and dtypes."""
    tail0, keys, scores, ape = _inputs(1, seed=1)
    assert can_run_dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape) is True


# ---------------------------------------------------------------------------------------------
# The ring's addressing, checked against upstream's own pointer arithmetic.
# ---------------------------------------------------------------------------------------------


def test_ring_row_equals_the_pool_slot_over_the_declared_position_range() -> None:
    """The module's ``pool_slot`` is upstream's ``phys``, for every position in range.

    Upstream computes ``phys = (pool_logical_start + pool_slot) % POOL_SIZE`` with
    ``pool_logical_start = position - slot``. The module drops the modulo and writes
    ``pool_slot`` directly, on the argument that ``position - slot`` is a multiple of
    ``pool_size``; this recomputes upstream's expression over 64 positions and compares.
    """
    checked = 0
    for position in range(64):
        slot = slot_of(position, POOL_SIZE)
        pool_logical_start = position - slot
        for pool_slot in range(POOL_SIZE):
            phys = (pool_logical_start + pool_slot) % POOL_SIZE
            assert phys == pool_slot
            checked += 1
    assert checked == 64 * POOL_SIZE


def test_completes_pool_is_the_last_slot_of_each_pool() -> None:
    """``completes_pool`` fires on exactly one position in four, the last of each pool."""
    firing = [p for p in range(64) if completes_pool(p, POOL_SIZE)]
    assert firing == list(range(POOL_SIZE - 1, 64, POOL_SIZE))
    assert len(firing) == 64 // POOL_SIZE


# ---------------------------------------------------------------------------------------------
# Stepping equals a full recompute, and the route each case takes.
# ---------------------------------------------------------------------------------------------


def test_stepped_matches_full_recompute_for_one_step() -> None:
    """One decode token, which completes the prefill-seeded prompt tail."""
    _assert_equivalent_to_recompute(1, seed=301)


def test_one_step_dispatches_the_kernel_once() -> None:
    """One decode token is exactly 1 NKI dispatch and zero torch fallbacks."""
    _assert_one_dispatch_per_token(1, seed=301)


def test_stepped_matches_full_recompute_for_four_steps() -> None:
    """One completion on the first step, then three steps that complete nothing.

    The three that complete nothing are what a stash gated on pool-granular validity
    would drop.
    """
    _assert_equivalent_to_recompute(4, seed=302)


def test_four_steps_dispatch_the_kernel_once_per_token() -> None:
    """Four decode tokens are exactly 4 NKI dispatches and zero torch fallbacks."""
    _assert_one_dispatch_per_token(4, seed=302)


def test_stepped_matches_full_recompute_for_sixteen_steps() -> None:
    """Four completions, so the ring wraps four times.

    This is the case that reads the ring as a ring: every row is written four times and
    read as a pool member three times between writes.
    """
    _assert_equivalent_to_recompute(16, seed=303)


def test_sixteen_steps_dispatch_the_kernel_once_per_token() -> None:
    """Sixteen decode tokens are exactly 16 NKI dispatches and zero torch fallbacks."""
    _assert_one_dispatch_per_token(16, seed=303)


def test_an_fp32_ring_is_refused_by_the_gate_and_served_by_torch() -> None:
    """fp32 is outside the supported dtypes, so the call takes the torch path instead."""
    tail0, keys, scores, ape = _inputs(1, seed=402)
    reset_decode_tail_dispatch_counters()
    gate = can_run_dsa_decode_tail_update(tail0.float(), keys[0:1], scores[0:1], ape)
    dsa_decode_tail_update(tail0.float(), keys[0:1].float(), scores[0:1].float(), ape,
                           START_POSITION)
    nki, fallback = decode_tail_dispatch_counters()
    assert (nki, fallback) == (0, 1)
    assert gate is False


# ---------------------------------------------------------------------------------------------
# The kernel identity, derived through the seam rather than from the import list.
# ---------------------------------------------------------------------------------------------


def test_kernel_identity_is_none_before_any_dispatch() -> None:
    """Before a dispatch the identity is ``None``, which separates "none ran"."""
    reset_decode_tail_dispatch_counters()
    assert decode_tail_kernel_identity() is None


def test_kernel_identity_names_this_modules_kernel_after_a_dispatch() -> None:
    """After a step the identity names this module's own kernel."""
    tail0, keys, scores, ape = _inputs(1, seed=403)
    reset_decode_tail_dispatch_counters()
    _stepped(tail0, keys, scores, ape)
    module, qualname = decode_tail_kernel_identity()
    assert module.endswith("decode_tail_update")
    assert qualname == "_decode_tail_update_nki"


# ---------------------------------------------------------------------------------------------
# The stash: on every token, and not only on the ones that complete a pool.
# ---------------------------------------------------------------------------------------------


def test_a_non_completing_step_returns_no_pooled_key() -> None:
    """A step at slot 0 of 4 completes nothing, so the seam returns ``None``.

    ``None`` rather than zeros, so a caller cannot mistake a legitimate all-zero pooled
    key for "no pool ended". The same seam at position 3 returns a tensor, so the
    reading discriminates.
    """
    tail0, keys, scores, ape = _inputs(1, seed=404)
    reset_decode_tail_dispatch_counters()
    pooled, _ = dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape, 0)
    nki, fallback = decode_tail_dispatch_counters()
    assert pooled is None
    completing, _ = dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape, POOL_SIZE - 1)
    assert completing is not None
    assert tuple(completing.shape) == (1, INDEX_HEAD_DIM)
    assert (nki, fallback) == (1, 0)


def test_the_stash_happens_on_a_step_that_completes_nothing() -> None:
    """A non-completing step still writes the token to its ring row, and leaves the others alone.

    Gating the stash on pool-granular validity drops every intra-pool token, so a
    decode-built pool compresses stale prefill entries. Both halves of the ring are
    checked, and so is every row the step must not touch.
    """
    tail0, keys, scores, ape = _inputs(1, seed=405)
    reset_decode_tail_dispatch_counters()
    _, new_tail = dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape, 0)
    untouched = [row for row in range(POOL_SIZE) if row != 0]
    assert torch.equal(new_tail[0, 0], keys[0])
    assert torch.equal(new_tail[1, 0], scores[0])
    for row in untouched:
        assert torch.equal(new_tail[0, row], tail0[0, row])
        assert torch.equal(new_tail[1, row], tail0[1, row])


def test_the_stash_happens_on_a_completing_step_too() -> None:
    """A completing step also stashes, because the token belongs to future pools as well.

    A kernel that stashed only on non-completing steps would pass every equivalence
    case whose ``k`` never re-read the completing slot, so this reading is taken
    directly.
    """
    tail0, keys, scores, ape = _inputs(1, seed=406)
    slot = POOL_SIZE - 1
    reset_decode_tail_dispatch_counters()
    pooled, new_tail = dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape, slot)
    assert pooled is not None
    assert torch.equal(new_tail[0, slot], keys[0])
    assert torch.equal(new_tail[1, slot], scores[0])


def test_input_ring_is_not_mutated_by_a_step() -> None:
    """The seam returns a new ring and does not write its argument.

    The ring is state the caller threads from step to step, and the equivalence cases
    hold the original ``tail0`` while they run. If a step mutated it, the recompute --
    which reads ``tail0`` for the pre-decode members -- would be comparing against a
    moving reference.
    """
    tail0, keys, scores, ape = _inputs(4, seed=407)
    before = tail0.clone()
    reset_decode_tail_dispatch_counters()
    _stepped(tail0, keys, scores, ape)
    assert torch.equal(tail0, before)


def test_completion_reads_the_current_token_and_not_the_stale_ring_row() -> None:
    """The pool's own slot takes the current token, not whatever the ring still holds there.

    Upstream selects on ``is_current`` for both the score and the key. A kernel that
    read every member from the ring would differ only in the completing slot, and only
    when the ring's stale row differs from the current token, which is why ``tail0`` is
    random here and the stale row is made deliberately far away.
    """
    tail0, keys, scores, ape = _inputs(1, seed=408)
    slot = POOL_SIZE - 1
    stale = tail0.clone()
    stale[0, slot] = 8.0
    stale[1, slot] = -8.0
    reset_decode_tail_dispatch_counters()
    expected, _ = decode_tail_recompute(stale, keys, scores, ape, slot)
    got, _ = _stepped(stale, keys, scores, ape, start_position=slot)
    torch.testing.assert_close(got[0].float(), expected[0].float(), rtol=RTOL, atol=ATOL)


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

    Upstream carries padded entries and screens them with ``pos_valid``. This module
    takes one real token per call, so a negative position is a caller error and is
    refused at the boundary instead of being silently skipped.
    """
    tail0, keys, scores, ape = _inputs(1, seed=411)
    with pytest.raises(DecodeTailUpdateError):
        dsa_decode_tail_update(tail0, keys[0:1], scores[0:1], ape, -1)
