# SPDX-License-Identifier: Apache-2.0
"""The ring step with ``T`` tokens per request: the verify step's form.

A speculative verify step hands the indexer ``T = 1 + k`` tokens per request at
positions ``start .. start + T - 1`` in one call. ``dsa_decode_ring_rows`` advances each
request's ring by all ``T`` and compresses every pool a row of the step closes, reading the
step's own earlier tokens off its operands rather than the ring. Its reference is the path
it must equal: ``T`` sequential one-token ring steps, each writing its ring back before the
next one reads. At the served ring depth (``pool_size`` rows) that path is
``decode_batch.dsa_decode_ring_step`` and the comparison is bit for bit; at the deeper ring
a speculative config allocates it is this kernel's own one-token form, checked against a
token-stream reference that steps nothing.

The deeper ring is what makes rollback sound: a rejected token at position ``r`` was
stashed at ring row ``r % R``, and the next step's pools still need the ``pool_size - 1``
tokens before the first accepted one. With ``R >= T + pool_size - 2`` no rejected row
aliases a needed one, so the next real tokens overwrite the rejected rows before anything
reads them. The rollback test runs that scenario against the sequential path that never saw
the rejected tokens.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.functional.dsa import decode_tail_update as TU
from vllm_neuron.utils.neuron_utils import can_run_kernel

POOL = 4
HEAD_DIM = 128
DEEP = 8  # the ring depth a config with k <= 5 draft tokens allocates
ROWS = (1, 2, 4, 6)
BATCHES = (1, 4)
# Row 0's position per request: every residue mod 8, and the window's last pool.
STARTS = (4095, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107)


def _bf16(gen, *shape):
    return torch.randn(shape, generator=gen).to(torch.bfloat16)


def _case(batch, rows, depth, *, seed, starts=STARTS, slots_total=11):
    gen = torch.Generator().manual_seed(seed)
    bank = _bf16(gen, slots_total, 2, depth, HEAD_DIM)
    slots = torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32)
    key = _bf16(gen, batch * rows, HEAD_DIM)
    score = _bf16(gen, batch * rows, HEAD_DIM)
    ape = torch.randn(POOL, HEAD_DIM, generator=gen)
    position = torch.tensor(list(starts)[:batch], dtype=torch.int32)
    return bank, slots, key, score, ape, position


def _sequential(step, bank, slots, key, score, ape, position, rows):
    """``rows`` one-token steps per request, each ring written back before the next."""
    batch = int(slots.shape[0])
    live = bank.clone()
    pooled = torch.empty(batch * rows, HEAD_DIM, dtype=bank.dtype)
    for t in range(rows):
        picked = torch.arange(batch) * rows + t
        one_pooled, rings = step(live, slots, key[picked], score[picked], ape, position + t)
        pooled[picked] = one_pooled
        live[slots.long()] = rings
    return pooled, live[slots.long()]


def _closing(position, rows):
    """``[B * rows]`` bool: which rows close a pool."""
    pos = (position.long()[:, None] + torch.arange(rows)[None, :]).reshape(-1)
    return torch.remainder(pos, POOL) == POOL - 1


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    TU.reset_decode_tail_dispatch_counters()


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("rows", (1, 2))
def test_rows_equal_sequential_ring_steps_at_the_served_depth_bit_for_bit(rows, batch):
    bank, slots, key, score, ape, position = _case(batch, rows, POOL, seed=100 + rows + batch)
    want_pooled, want_rings = _sequential(DB.dsa_decode_ring_step, bank, slots, key, score,
                                          ape, position, rows)
    pooled, rings = TU.dsa_decode_ring_rows(bank, slots, key, score, ape, position)
    assert TU.decode_tail_dispatch_counters() == (1, 0)
    assert pooled.shape == (batch * rows, HEAD_DIM) and pooled.dtype == torch.bfloat16
    assert rings.shape == (batch, 2, POOL, HEAD_DIM) and rings.dtype == torch.bfloat16
    assert torch.equal(rings, want_rings)
    closing = _closing(position, rows)
    assert bool(closing.any())
    assert torch.equal(pooled[closing], want_pooled[closing])


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("rows", ROWS)
def test_rows_equal_sequential_one_token_steps_at_the_deep_ring_bit_for_bit(rows, batch):
    bank, slots, key, score, ape, position = _case(batch, rows, DEEP, seed=200 + rows + batch)
    want_pooled, want_rings = _sequential(TU.dsa_decode_ring_rows, bank, slots, key, score,
                                          ape, position, rows)
    TU.reset_decode_tail_dispatch_counters()
    pooled, rings = TU.dsa_decode_ring_rows(bank, slots, key, score, ape, position)
    assert TU.decode_tail_dispatch_counters() == (1, 0)
    assert rings.shape == (batch, 2, DEEP, HEAD_DIM)
    assert torch.equal(rings, want_rings)
    closing = _closing(position, rows)
    assert torch.equal(pooled[closing], want_pooled[closing])
    # And the token-stream reference, which steps no ring at all.
    oracle_pooled, oracle_rings = TU.dsa_decode_ring_rows_torch_oracle(
        bank, slots, key, score, ape, position)
    assert torch.equal(rings, oracle_rings)
    torch.testing.assert_close(pooled[closing].float(), oracle_pooled[closing].float(),
                               rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("accepted", (1, 2, 3))
def test_rollback_leaves_no_rejected_state_the_next_step_reads(accepted):
    """Step 1 proposes ``T`` rows, ``accepted`` of them stand; step 2 proposes ``T`` more.

    After step 2 the rings' readable rows (the ``pool_size - 1`` positions before the next
    step) and every pool step 2 closes must equal the path that only ever saw the accepted
    tokens, at every residue of the start position.
    """
    rows = 4
    bank, slots, key, score, ape, position = _case(8, rows, DEEP, seed=300 + accepted,
                                                   starts=STARTS[1:])
    gen = torch.Generator().manual_seed(400 + accepted)
    key2, score2 = _bf16(gen, 8 * rows, HEAD_DIM), _bf16(gen, 8 * rows, HEAD_DIM)
    # The speculative path: step 1 with every proposed row, then step 2 from the first
    # rejected position, rings written back in between.
    live = bank.clone()
    _, rings1 = TU.dsa_decode_ring_rows(live, slots, key, score, ape, position)
    live[slots.long()] = rings1
    pooled2, rings2 = TU.dsa_decode_ring_rows(live, slots, key2, score2, ape,
                                              position + accepted)
    # The honest path: only the accepted tokens, then step 2's tokens one at a time.
    picked = (torch.arange(8)[:, None] * rows + torch.arange(accepted)[None, :]).reshape(-1)
    honest = bank.clone()
    _, rings_a = _sequential(TU.dsa_decode_ring_rows, honest, slots, key[picked],
                             score[picked], ape, position, accepted)
    honest[slots.long()] = rings_a
    want_pooled2, want_rings2 = _sequential(TU.dsa_decode_ring_rows, honest, slots, key2,
                                            score2, ape, position + accepted, rows)
    closing = _closing(position + accepted, rows)
    assert torch.equal(pooled2[closing], want_pooled2[closing])
    last = position.long() + accepted + rows  # the next step's first position
    readable = torch.remainder(last[:, None] - torch.arange(1, POOL)[None, :], DEEP)
    for b in range(8):
        assert torch.equal(rings2[b, :, readable[b]], want_rings2[b, :, readable[b]]), b


def test_rows_read_only_the_named_slots():
    bank, slots, key, score, ape, position = _case(4, 4, DEEP, seed=17)
    clean = TU.dsa_decode_ring_rows(bank, slots, key, score, ape, position)
    poisoned = torch.full_like(bank, 3.0e4)
    poisoned[slots.long()] = bank[slots.long()]
    dirty = TU.dsa_decode_ring_rows(poisoned, slots, key, score, ape, position)
    assert torch.equal(clean[0], dirty[0]) and torch.equal(clean[1], dirty[1])


def test_refusals_name_the_rows_and_the_depth():
    bank, slots, key, score, ape, position = _case(4, 3, POOL, seed=5)
    with pytest.raises(TU.DecodeTailUpdateError, match="rows"):
        TU.dsa_decode_ring_rows(bank, slots, key, score, ape, position)
    bank, slots, key, score, ape, position = _case(4, 2, POOL, seed=6)
    with pytest.raises(TU.DecodeTailUpdateError, match="whole number of rows"):
        TU.dsa_decode_ring_rows(bank, slots, key[:7], score[:7], ape, position)
    with pytest.raises(TU.DecodeTailUpdateError, match="depth"):
        TU.dsa_decode_ring_rows(bank[:, :, :3], slots, key, score, ape, position)
    with pytest.raises(TU.DecodeTailUpdateError, match="ape"):
        TU.dsa_decode_ring_rows(bank, slots, key, score, ape[:3], position)
