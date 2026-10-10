# SPDX-License-Identifier: Apache-2.0
"""The batched decode kernels of the DSA indexer against the one-request kernels.

``decode_batch`` serves ``B`` decode requests in one launch per stage: the ring step
(``dsa_decode_ring_step``) and the candidate scores with the causal bound applied
(``dsa_decode_scores``). Each request reads its own slot of the two side-cache banks,
so the reference here is the one-request chain run once per request on that request's own
slot view: ``dsa_decode_tail_update_at`` for the ring, and the pool write, the
candidate read, ``dsa_score_gemm`` and ``dsa_causal_bound`` for the scores.

What is exact and what is not:

* The ring step runs the one-request kernel's own instruction sequence, one request per
  partition, so the advanced ring and every completed pool are compared bit for bit.
* The scores contract the same bf16 products in fp32, but the head sum and the PE
  arrangement differ from the one-request kernel (candidates ride the partitions here,
  tokens ride them there). The simulator's matmul is numpy's, whose rounding depends on
  the operand shapes, so the two agree to fp32 rounding, not bit for bit:
  ``SCORE_RTOL``. Bounded columns hold ``BOUND_FILL`` exactly in both.
* A request scored alone and the same request scored inside a batch run the same
  instructions on the same data, so those agree bit for bit.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL, dsa_causal_bound
from vllm_neuron.functional.dsa.decode_tail_update import dsa_decode_tail_update_at
from vllm_neuron.functional.dsa.score_gemm import dsa_score_gemm
from vllm_neuron.utils.neuron_utils import can_run_kernel

POOL = 4
HEAD_DIM = 128
#: fp32 rounding of a 128-term bf16 dot product and a 32-term head sum, in two orders.
#: Scores here are O(1-10) and a few fp32 ulps of the partial sums is ~1e-6; a wrong
#: key, head or weight moves a score by O(1).
SCORE_RTOL = 1e-5
SCORE_ATOL = 1e-5


def _bf16(gen, *shape, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(torch.bfloat16)


def _ring_case(batch: int, seed: int, slots_total: int = 11):
    gen = torch.Generator().manual_seed(seed)
    bank = _bf16(gen, slots_total, 2, POOL, HEAD_DIM)
    slots = torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32)
    key = _bf16(gen, batch, HEAD_DIM)
    score = _bf16(gen, batch, HEAD_DIM)
    ape = torch.randn(POOL, HEAD_DIM, generator=gen)
    # Every ring slot appears from B=4 on, and the first request completes a pool.
    position = torch.tensor([4095, 2101, 2102, 2103, 2104, 2105, 2106, 2107][:batch],
                            dtype=torch.int32)
    return bank, slots, key, score, ape, position


def _one_request_rings(bank, slots, key, score, ape, position):
    pooled, rings = [], []
    for b in range(int(slots.shape[0])):
        one_pooled, one_ring = dsa_decode_tail_update_at(
            bank[int(slots[b])].clone(), key[b:b + 1], score[b:b + 1], ape,
            position[b].to(torch.int64))
        pooled.append(one_pooled)
        rings.append(one_ring)
    return torch.cat(pooled), torch.stack(rings)


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    DB.reset_decode_batch_dispatch_counters()


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_ring_step_equals_the_one_request_kernel_bit_for_bit(batch):
    bank, slots, key, score, ape, position = _ring_case(batch, seed=300 + batch)
    want_pooled, want_rings = _one_request_rings(bank, slots, key, score, ape, position)
    pooled, rings = DB.dsa_decode_ring_step(bank, slots, key, score, ape, position)
    assert DB.decode_batch_dispatch_counters() == (1, 0)
    assert DB.decode_batch_route_counts()[0] == 1
    assert rings.shape == (batch, 2, POOL, HEAD_DIM) and rings.dtype == torch.bfloat16
    assert torch.equal(rings, want_rings)
    completes = (position % POOL) == POOL - 1
    assert bool(completes.any())
    assert torch.equal(pooled[completes], want_pooled[completes])


def test_ring_step_reads_only_the_named_slots():
    bank, slots, key, score, ape, position = _ring_case(4, seed=17)
    clean = DB.dsa_decode_ring_step(bank, slots, key, score, ape, position)
    poisoned = torch.full_like(bank, 3.0e4)
    poisoned[slots.long()] = bank[slots.long()]
    dirty = DB.dsa_decode_ring_step(poisoned, slots, key, score, ape, position)
    assert torch.equal(clean[0], dirty[0]) and torch.equal(clean[1], dirty[1])


@pytest.mark.parametrize("batch", [2, 5, 8])
def test_ring_step_two_programs_split_the_requests_and_equal_one(monkeypatch, batch):
    """Both cores of an LNC2 core advance half the rings each, bit for bit as one."""
    bank, slots, key, score, ape, position = _ring_case(batch, seed=500 + batch)
    one = DB.dsa_decode_ring_step(bank, slots, key, score, ape, position)
    DB.reset_decode_batch_dispatch_counters()
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    two = DB.dsa_decode_ring_step(bank, slots, key, score, ape, position)
    assert DB.decode_batch_route_counts() == (1, 0, 1, 0)
    assert torch.equal(one[0], two[0]) and torch.equal(one[1], two[1])


def test_the_kernel_key_covers_the_butterfly_the_kernels_call():
    """Both kernels' cache key is a digest over this module and ``kpool_hadamard.py``,
    whose butterfly the ring step calls: an edit to either re-keys the kernels."""
    import hashlib
    from pathlib import Path

    from vllm_neuron.functional.dsa import kpool_hadamard as KH
    data = Path(DB.__file__).read_bytes() + Path(KH.__file__).read_bytes()
    assert DB.SOURCE_DIGEST == int(hashlib.sha256(data).hexdigest()[:7], 16)


def test_ring_step_torch_oracle_agrees():
    bank, slots, key, score, ape, position = _ring_case(4, seed=23)
    pooled, rings = DB.dsa_decode_ring_step(bank, slots, key, score, ape, position)
    want_pooled, want_rings = DB.dsa_decode_ring_step_torch_oracle(
        bank, slots, key, score, ape, position)
    assert torch.equal(rings, want_rings)
    completes = (position % POOL) == POOL - 1
    torch.testing.assert_close(pooled[completes].float(), want_pooled[completes].float(),
                               rtol=1e-2, atol=1e-2)


def _one_slot_calls():
    """The ring step and the scores of request 0, its bank narrowed by ``slot_of``."""
    ring = _ring_case(1, seed=29)
    score = _score_case(1, seed=31, candidates=300, heads=4)

    def calls(ring_bank, score_bank, ring_slots, score_slots):
        bank, _slots, key, gate, ape, position = ring
        query, weights, _bank, _s, seq_lens, score_pos, pooled = score
        pooled_out, rings = DB.dsa_decode_ring_step(ring_bank, ring_slots, key, gate, ape,
                                                    position)
        scores = DB.dsa_decode_scores(query, weights, score_bank, score_slots, seq_lens,
                                      score_pos, pooled, candidates=300, pool_size=POOL)
        return pooled_out, rings, scores

    whole = (ring[0], score[2], ring[1], score[3])
    narrowed = (ring[0][ring[1].long()], score[2][score[3].long()],
                torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32))
    return calls, whole, narrowed


def test_a_one_slot_bank_gives_what_its_slot_gives_in_the_whole_bank():
    """A one-request carrier's view is a one-slot bank; the kernels read it statically."""
    calls, whole, narrowed = _one_slot_calls()
    want = calls(*whole)
    got = calls(*narrowed)
    for a, b in zip(got, want):
        assert torch.equal(a, b)


def test_a_one_slot_bank_traces():
    """Tracing in CPU simulation runs each kernel on ones (slot 1), past a one-slot bank."""
    calls, _whole, narrowed = _one_slot_calls()
    want = calls(*narrowed)
    torch._dynamo.reset()
    got = torch.compile(calls, backend="eager", fullgraph=True)(*narrowed)
    for a, b in zip(got, want):
        assert torch.equal(a, b)


# --- scores ------------------------------------------------------------------------


def _score_case(batch: int, seed: int, *, candidates: int, heads: int = 32,
                lengths=None, slots_total: int = 10):
    gen = torch.Generator().manual_seed(seed)
    rows = candidates + 1
    bank = _bf16(gen, slots_total, rows, HEAD_DIM, scale=0.5)
    slots = torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32)
    if lengths is None:
        lengths = [candidates * POOL - 3 * b - 1 for b in range(batch)]
    seq_lens = torch.tensor(lengths[:batch], dtype=torch.int32)
    position = seq_lens - 1
    query = _bf16(gen, batch, heads, HEAD_DIM)
    weights = torch.randn(batch, heads, generator=gen) * heads ** -0.5
    pooled = _bf16(gen, batch, HEAD_DIM, scale=0.5)
    return query, weights, bank, slots, seq_lens, position, pooled


def _per_request_scores(query, weights, bank, slots, seq_lens, position, pooled, candidates):
    """The one-request chain on one request's own slot: write, read, score, bound."""
    out = []
    for b in range(int(slots.shape[0])):
        store = bank[int(slots[b])].clone()
        pos = int(position[b])
        if pos % POOL == POOL - 1:
            store[pos // POOL] = pooled[b]
        scores = dsa_score_gemm(query[b:b + 1], store[:candidates].contiguous(),
                                weights[b:b + 1])
        out.append(dsa_causal_bound(scores, seq_lens[b:b + 1].reshape(1, 1), POOL))
    return torch.cat(out)


def _assert_scores_agree(got, want):
    filled = want == BOUND_FILL
    assert torch.equal(got == BOUND_FILL, filled)
    torch.testing.assert_close(got[~filled], want[~filled], rtol=SCORE_RTOL, atol=SCORE_ATOL)


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_scores_equal_the_per_request_chain(batch):
    candidates = 1024
    lengths = [2100, 3000, 4096, 4001, 2303, 3999, 2052, 4095]
    case = _score_case(batch, seed=40 + batch, candidates=candidates, lengths=lengths)
    want = _per_request_scores(*case, candidates)
    got = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert DB.decode_batch_dispatch_counters() == (1, 0)
    assert DB.decode_batch_route_counts()[1] == 1
    assert got.shape == (batch, candidates) and got.dtype == torch.float32
    _assert_scores_agree(got, want)


def test_a_request_scores_the_same_alone_and_in_a_batch():
    candidates = 512
    case = _score_case(4, seed=5, candidates=candidates,
                       lengths=[2047, 1100, 2048, 1503])
    together = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    for b in range(4):
        alone = DB.dsa_decode_scores(*(t[b:b + 1] for t in case[:2]), case[2],
                                     *(t[b:b + 1] for t in case[3:]),
                                     candidates=candidates, pool_size=POOL)
        assert torch.equal(alone, together[b:b + 1])


def test_scores_never_read_another_slot_or_an_unwritten_pool():
    """Poison every bank row a request may not see; the scores do not move."""
    candidates = 512
    case = _score_case(4, seed=11, candidates=candidates, lengths=[1300, 2048, 1801, 1025])
    query, weights, bank, slots, seq_lens, position, pooled = case
    clean = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    poisoned = torch.full_like(bank, 3.0e4)
    for b in range(4):
        own = int(slots[b])
        complete = int(seq_lens[b]) // POOL
        poisoned[own, :complete] = bank[own, :complete]
    dirty = DB.dsa_decode_scores(query, weights, poisoned, slots, seq_lens, position, pooled,
                                 candidates=candidates, pool_size=POOL)
    pos = position.long()
    completes = (pos % POOL) == POOL - 1
    # The pool this step completes comes from ``pooled``, never from the bank.
    assert bool(completes.any())
    assert torch.equal(clean, dirty)


def test_scores_take_this_steps_pool_from_pooled_not_the_bank():
    candidates = 512
    case = _score_case(2, seed=12, candidates=candidates, lengths=[1200, 2000])
    query, weights, bank, slots, seq_lens, position, pooled = case
    assert all(int(p) % POOL == POOL - 1 for p in position)
    got = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    stale = bank.clone()
    for b in range(2):
        stale[int(slots[b]), int(position[b]) // POOL] = 7.0
    again = DB.dsa_decode_scores(query, weights, stale, slots, seq_lens, position, pooled,
                                 candidates=candidates, pool_size=POOL)
    assert torch.equal(got, again)


@pytest.mark.parametrize("candidates,heads", [(34, 4), (300, 4), (128, 1)])
def test_scores_serve_ragged_candidate_counts_and_few_heads(candidates, heads):
    lengths = [candidates * POOL, candidates * POOL - 2, 9]
    case = _score_case(3, seed=candidates, candidates=candidates, heads=heads,
                       lengths=lengths)
    want = _per_request_scores(*case, candidates)
    got = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert DB.decode_batch_dispatch_counters() == (1, 0)
    _assert_scores_agree(got, want)


def test_scores_torch_oracle_agrees():
    candidates = 256
    case = _score_case(3, seed=77, candidates=candidates, lengths=[1023, 600, 7])
    got = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    _assert_scores_agree(got, DB.dsa_decode_scores_torch_oracle(
        *case, candidates=candidates, pool_size=POOL))


def test_two_programs_split_the_requests_and_agree_with_one(monkeypatch):
    candidates = 256
    case = _score_case(5, seed=88, candidates=candidates, lengths=[1023, 600, 7, 1024, 512])
    one = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    two = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert DB.decode_batch_route_counts()[2] == 1
    assert torch.equal(one, two)


# --- refusals ----------------------------------------------------------------------


def test_refusals_name_the_operand():
    case = _score_case(2, seed=1, candidates=128)
    query, weights, bank, slots, seq_lens, position, pooled = case
    with pytest.raises(DB.DecodeBatchError, match="candidates"):
        DB.dsa_decode_scores(*case, candidates=bank.shape[1], pool_size=POOL)
    with pytest.raises(DB.DecodeBatchError, match="weights"):
        DB.dsa_decode_scores(query, weights[:, :3], bank, slots, seq_lens, position, pooled,
                             candidates=128, pool_size=POOL)
    with pytest.raises(DB.DecodeBatchError, match="slots"):
        DB.dsa_decode_scores(query, weights, bank, slots[:1], seq_lens, position, pooled,
                             candidates=128, pool_size=POOL)
    with pytest.raises(DB.DecodeBatchError, match="pool_size"):
        DB.dsa_decode_scores(*case, candidates=128, pool_size=3)
    ring = _ring_case(2, seed=2)
    with pytest.raises(DB.DecodeBatchError, match="tail"):
        DB.dsa_decode_ring_step(ring[0][:, :1], *ring[1:])


def test_pool_destinations_write_completions_and_trash_otherwise():
    slots = torch.tensor([3, 0, 5], dtype=torch.int32)
    position = torch.tensor([7, 9, 4095], dtype=torch.int32)
    slot_index, row = DB.decode_pool_destinations(slots, position, rows=1025, pool_size=POOL)
    assert slot_index.tolist() == [3, 0, 5]
    assert row.tolist() == [1, 1024, 1023]
