# SPDX-License-Identifier: Apache-2.0
"""The DSA decode indexer under decode context parallelism (DCP), at ``T = 1``.

Under DCP a group of ``CP`` ranks holds one sequence's pooled-key bank interleaved by
128-token block: block ``P // 128`` lives on rank ``(P // 128) % CP`` at local block
``(P // 128) // CP`` (the runner's prefill slot split, ``neuron_model_runner.py``
``slot_mapping.view(S // (W * I), W, I)[:, R, :]``; ``cp_rank = tp_rank % CP``). Each rank
scores and selects over its own pools; trn2-2's item 3 merges the ``CP`` local top-``k``
lists into the global one. What this file pins, against the CP = 1 answer:

* **Operands** (``decode_batch.dcp_decode_operands``): the owned length, the position (the
  token's local index on its owner, the owned length elsewhere) and the owner flag, against
  a token-by-token count of the interleave.
* **Scores over the local shard**: ``dsa_decode_scores`` on a rank's bank with those
  operands gives the CP = 1 scores of the pools the rank owns, at their local columns; the
  step's own pool stands in on its owner only.
* **The owner gate** (``decode_pool_destinations(..., owner=)``): a step that closes a pool
  writes it on the owner, at its local row; every other rank writes its trash row.
* **The local top-k** (``decode_select.dsa_decode_select_pools``): the selected local pool
  ids in ascending order (ties at the ``k``-th value to the lowest local id) and their
  scores beside them, rows short of ``k`` pools padded with ``-1`` and
  :data:`decode_select.POOL_PAD_VALUE`; bit for bit against the torch oracle, and the
  expanded tokens equal to ``dsa_decode_select``'s.
* **The merge**: a reference merge of the ``CP`` local lists under the global tie rule
  ``g = (p // 32) * 32 * CP + cp_rank * 32 + (p % 32)`` gives the CP = 1 list exactly, values
  and ids, at CP in {2, 4, 8}.
* **The ownership remap** (``dcp_global_pool_ids`` / ``dcp_owned_pool_rows``): a round trip,
  and the union of the ranks' expanded rows equal to the CP = 1 token set.
* **The device compile** of the new kernel at served shapes: every access ordered
  (``sbuf_order``) and the dependency checker's one-pass and loop-carried verdicts CLEAN.
"""

from __future__ import annotations

import glob
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import pytest
import torch

from test.vllm_neuron.functional.dsa.dsa_decode_case import decode_config
from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.functional.dsa import decode_select as DS
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL, SENTINEL
from vllm_neuron.functional.dsa.index_expand import _dsa_index_expand_torch, index_expand_width
from vllm_neuron.utils.neuron_utils import can_run_kernel

_CFG = decode_config()
POOL = int(_CFG.index_kpool)
#: Pools the selection keeps: ``index_topk`` tokens in whole pools.
K = int(_CFG.index_topk) // POOL
#: ``hybrid_kv_block_size`` of the serve recipes: the DCP interleave unit.
BLOCK = 128
#: Pools per interleaved block.
BLOCK_POOLS = BLOCK // POOL
HEADS = int(_CFG.index_n_heads)
DIM = int(_CFG.index_head_dim)
CPS = (2, 4, 8)


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    DB.reset_decode_batch_dispatch_counters()


def _select_launches():
    """``(selection launches, torch fallbacks)`` in ``decode_batch``'s family."""
    return (DB.decode_batch_route_counts()[3], DB.decode_batch_dispatch_counters()[1])


# ---------------------------------------------------------------------------------------
# The interleave, spelled token by token (the reference every DCP helper is checked against)
# ---------------------------------------------------------------------------------------


def _owner_of(token: int, cp: int) -> int:
    return (token // BLOCK) % cp


def _local_of(token: int, cp: int) -> int:
    """A token's row in its owner's local store."""
    return (token // (BLOCK * cp)) * BLOCK + token % BLOCK


def _global_token(local: int, rank: int, cp: int) -> int:
    return ((local // BLOCK) * cp + rank) * BLOCK + local % BLOCK


def _global_pool(local_pool: int, rank: int, cp: int) -> int:
    """The interface's global pool id of local pool ``p`` on ``rank``."""
    block, within = divmod(local_pool, BLOCK_POOLS)
    return block * BLOCK_POOLS * cp + rank * BLOCK_POOLS + within


def _reference_operands(position: int, rank: int, cp: int) -> tuple[int, int, int]:
    """``(seq_len, position, owner)`` of one request on ``rank``, counted token by token."""
    owned = sum(1 for t in range(position + 1) if _owner_of(t, cp) == rank)
    if _owner_of(position, cp) == rank:
        return owned, _local_of(position, cp), 1
    return owned, owned, 0


def _local_candidates(ctx: int, cp: int) -> int:
    """The complete pools of the rank holding the most of a ``ctx``-token sequence."""
    if ctx % (BLOCK * cp) == 0:
        return ctx // (cp * POOL)
    return max(sum(1 for t in range(ctx) if _owner_of(t, cp) == r) for r in range(cp)) // POOL


# ---------------------------------------------------------------------------------------
# Operands
# ---------------------------------------------------------------------------------------

_POSITIONS = [0, 1, 3, 4, 126, 127, 128, 129, 255, 256, 383, 511, 512, 1023, 1024, 4095,
              4096, 5000, 8191, 12345]


@pytest.mark.parametrize("cp", (1,) + CPS)
def test_operands_count_the_interleave(cp):
    position = torch.tensor(_POSITIONS, dtype=torch.int64)
    for rank in range(cp):
        ops = DB.dcp_decode_operands(position, cp_rank=rank, cp_size=cp, block_size=BLOCK)
        want = [_reference_operands(p, rank, cp) for p in _POSITIONS]
        assert ops.seq_lens.tolist() == [w[0] for w in want], (cp, rank)
        assert ops.position.tolist() == [w[1] for w in want], (cp, rank)
        assert ops.owner.tolist() == [w[2] for w in want], (cp, rank)
        assert ops.owner.dtype == torch.int32
        assert tuple(ops.seq_lens.shape) == tuple(ops.position.shape) == (len(_POSITIONS),)


@pytest.mark.parametrize("cp", CPS)
def test_exactly_one_rank_owns_the_token_and_non_owners_never_stand_in(cp):
    position = torch.tensor(_POSITIONS, dtype=torch.int64)
    ranks = [DB.dcp_decode_operands(position, cp_rank=r, cp_size=cp, block_size=BLOCK)
             for r in range(cp)]
    assert torch.equal(sum(o.owner for o in ranks), torch.ones(len(_POSITIONS), dtype=torch.int32))
    # The owned lengths add up to the global length.
    assert torch.equal(sum(o.seq_lens for o in ranks), position + 1)
    for ops in ranks:
        other = ops.owner == 0
        # A non-owner's position is its owned length, a whole number of blocks: it never
        # closes a pool, so the score kernel's stand-in (position % pool == pool - 1) never
        # fires there.
        assert torch.all(ops.position[other] % BLOCK == 0)
        assert torch.all(ops.position[other] == ops.seq_lens[other])
        assert torch.all(ops.position[~other] == ops.seq_lens[~other] - 1)


def test_cp_one_is_the_identity():
    position = torch.tensor(_POSITIONS, dtype=torch.int64)
    ops = DB.dcp_decode_operands(position, cp_rank=0, cp_size=1, block_size=BLOCK)
    assert torch.equal(ops.position, position)
    assert torch.equal(ops.seq_lens, position + 1)
    assert torch.equal(ops.owner, torch.ones(len(_POSITIONS), dtype=torch.int32))


@pytest.mark.parametrize("ctx", [2048, 2051, 4100, 8192, 8319, 65536, 131072, 1048576])
@pytest.mark.parametrize("cp", (1,) + CPS)
def test_local_candidates_hold_every_owned_pool(ctx, cp):
    got = DB.dcp_local_candidates(ctx, cp_size=cp, block_size=BLOCK, pool_size=POOL)
    assert got == _local_candidates(ctx, cp)
    if cp == 1:
        assert got == ctx // POOL            # the CP = 1 candidate axis


def test_operand_refusals():
    position = torch.tensor([5], dtype=torch.int64)
    for kwargs in ({"cp_rank": 2, "cp_size": 2, "block_size": BLOCK},
                   {"cp_rank": -1, "cp_size": 2, "block_size": BLOCK},
                   {"cp_rank": 0, "cp_size": 0, "block_size": BLOCK},
                   {"cp_rank": 0, "cp_size": 2, "block_size": 0}):
        with pytest.raises(DB.DecodeBatchError):
            DB.dcp_decode_operands(position, **kwargs)
    with pytest.raises(DB.DecodeBatchError):
        DB.dcp_decode_operands(position.reshape(1, 1), cp_rank=0, cp_size=2, block_size=BLOCK)
    with pytest.raises(DB.DecodeBatchError):
        DB.dcp_local_candidates(4096, cp_size=2, block_size=126, pool_size=POOL)


# ---------------------------------------------------------------------------------------
# Scores over the local shard
# ---------------------------------------------------------------------------------------


def _global_case(lengths, ctx, *, seed):
    """One decode step at CP = 1: banks over the global pool axis, one token per request."""
    gen = torch.Generator().manual_seed(seed)
    batch = len(lengths)
    candidates = ctx // POOL
    rows = candidates + 1
    slots = batch + 2
    return {
        "query": (torch.randn(batch, HEADS, DIM, generator=gen) * 0.5).to(torch.bfloat16),
        "weights": torch.randn(batch, HEADS, generator=gen) * (DIM ** -0.5),
        "bank": (torch.randn(slots, rows, DIM, generator=gen) * 0.5).to(torch.bfloat16),
        "slots": torch.randperm(slots, generator=gen)[:batch].to(torch.int32),
        "pooled": (torch.randn(batch, DIM, generator=gen) * 0.5).to(torch.bfloat16),
        "position": torch.tensor([n - 1 for n in lengths], dtype=torch.int64),
        "candidates": candidates,
    }


def _local_bank(bank, rank, cp, local_candidates):
    """Rank ``rank``'s bank: local pool ``p`` holds global pool ``g(p)``; zeros past the
    global axis; one trash row."""
    out = torch.zeros(bank.shape[0], local_candidates + 1, bank.shape[2], dtype=bank.dtype)
    for p in range(local_candidates):
        g = _global_pool(p, rank, cp)
        if g < bank.shape[1] - 1:
            out[:, p] = bank[:, g]
    return out


def _to_local_columns(scores, rank, cp, local_candidates):
    """The CP = 1 scores of the pools ``rank`` owns, at their local columns; ``BOUND_FILL``
    past the global axis."""
    out = torch.full((scores.shape[0], local_candidates), BOUND_FILL, dtype=scores.dtype)
    for p in range(local_candidates):
        g = _global_pool(p, rank, cp)
        if g < scores.shape[1]:
            out[:, p] = scores[:, g]
    return out


def _scores(case, bank, seq_lens, position, candidates):
    return DB.dsa_decode_scores(case["query"], case["weights"], bank, case["slots"], seq_lens,
                                position, case["pooled"], candidates=candidates,
                                pool_size=POOL)


@pytest.mark.parametrize("cp", CPS)
def test_local_scores_are_the_cp1_scores_of_the_owned_pools(cp):
    # Lengths that close a pool (position % 4 == 3), sit on a block edge, and neither.
    lengths = [4096, 4093, 2944, 2945, 3200, 129]
    ctx = 4096
    case = _global_case(lengths, ctx, seed=cp)
    position = case["position"]
    want = _scores(case, case["bank"], position + 1, position, case["candidates"])
    local_c = DB.dcp_local_candidates(ctx, cp_size=cp, block_size=BLOCK, pool_size=POOL)
    for rank in range(cp):
        ops = DB.dcp_decode_operands(position, cp_rank=rank, cp_size=cp, block_size=BLOCK)
        bank = _local_bank(case["bank"], rank, cp, local_c)
        got = _scores(case, bank, ops.seq_lens, ops.position, local_c)
        assert torch.equal(got, _to_local_columns(want, rank, cp, local_c)), (cp, rank)
    assert DB.decode_batch_dispatch_counters()[1] == 0


def test_the_stand_in_lands_on_the_owner_only():
    """A step that closes a pool: the owner scores the ring's pooled key at the pool's local
    column (its bank row there is stale), every other rank leaves that pool out."""
    cp, ctx = 2, 1024
    lengths = [516]                       # position 515: closes pool 128, block 4 -> rank 0
    case = _global_case(lengths, ctx, seed=7)
    position = case["position"]
    local_c = DB.dcp_local_candidates(ctx, cp_size=cp, block_size=BLOCK, pool_size=POOL)
    stale = case["bank"].clone()
    stale[:, 128] = 0                    # the store row the step has not written yet
    want = _scores({**case, "bank": stale}, stale, position + 1, position, case["candidates"])
    for rank in range(cp):
        ops = DB.dcp_decode_operands(position, cp_rank=rank, cp_size=cp, block_size=BLOCK)
        got = _scores(case, _local_bank(stale, rank, cp, local_c), ops.seq_lens, ops.position,
                      local_c)
        assert torch.equal(got, _to_local_columns(want, rank, cp, local_c))
    owner_col = 128 // (BLOCK_POOLS * cp) * BLOCK_POOLS + 128 % BLOCK_POOLS
    assert want[0, 128] > BOUND_FILL and owner_col == 64


# ---------------------------------------------------------------------------------------
# The owner gate on the pool write
# ---------------------------------------------------------------------------------------


def _destinations(slots, position, rows, owner=None):
    if owner is None:
        return DB.decode_pool_destinations(slots, position, rows=rows, pool_size=POOL)
    return DB.decode_pool_destinations(slots, position, rows=rows, pool_size=POOL, owner=owner)


@pytest.mark.parametrize("cp", CPS)
def test_only_the_owner_writes_the_closed_pool(cp):
    position = torch.tensor(_POSITIONS + [131, 135, 1027], dtype=torch.int64)
    slots = torch.arange(len(position), dtype=torch.int32)
    rows = _local_candidates(16384, cp) + 1
    trash = rows - 1
    for rank in range(cp):
        ops = DB.dcp_decode_operands(position, cp_rank=rank, cp_size=cp, block_size=BLOCK)
        slot, row = _destinations(slots, ops.position, rows, owner=ops.owner)
        assert torch.equal(slot, slots.to(torch.int64))
        for b, p in enumerate(position.tolist()):
            closes = p % POOL == POOL - 1
            if _owner_of(p, cp) == rank and closes:
                assert int(row[b]) == _local_of(p, cp) // POOL, (cp, rank, p)
            else:
                assert int(row[b]) == trash, (cp, rank, p)


def test_the_gate_holds_for_a_global_position_too():
    """RED without the gate: today's call with the global position sends a non-owner's
    closing step to a real row of its bank (``position // pool``); with ``owner`` it lands
    on the trash row whatever the position."""
    cp, rows = 2, 1025
    position = torch.tensor([131], dtype=torch.int64)     # block 1 -> rank 1; closes pool 32
    slots = torch.zeros(1, dtype=torch.int32)
    rank0 = DB.dcp_decode_operands(position, cp_rank=0, cp_size=cp, block_size=BLOCK)
    assert int(rank0.owner[0]) == 0
    _, ungated = _destinations(slots, position, rows)
    assert int(ungated[0]) == 32                           # a real row on the non-owner
    _, gated = _destinations(slots, position, rows, owner=rank0.owner)
    assert int(gated[0]) == rows - 1


def test_without_owner_the_destinations_are_todays():
    position = torch.tensor(_POSITIONS, dtype=torch.int64)
    slots = torch.arange(len(_POSITIONS), dtype=torch.int32)
    slot, row = _destinations(slots, position, 4097)
    at = position % POOL == POOL - 1
    assert torch.equal(row, torch.where(at, position // POOL, torch.full_like(position, 4096)))
    assert slot.dtype == row.dtype == torch.int64
    everyone = torch.ones(len(_POSITIONS), dtype=torch.int32)
    assert torch.equal(_destinations(slots, position, 4097, owner=everyone)[1], row)


def test_owner_refusals():
    position = torch.tensor([3, 7], dtype=torch.int64)
    slots = torch.zeros(2, dtype=torch.int32)
    with pytest.raises(DB.DecodeBatchError):
        _destinations(slots, position, 9, owner=torch.ones(3, dtype=torch.int32))
    with pytest.raises(DB.DecodeBatchError):
        _destinations(slots, position, 9, owner=torch.ones(2, dtype=torch.float32))


# ---------------------------------------------------------------------------------------
# The local top-k: values and pool ids before the expansion
# ---------------------------------------------------------------------------------------


def _bounded(lengths, candidates, *, seed, levels=None):
    """Scores as the score kernel leaves them: ``BOUND_FILL`` past each row's complete
    pools; ``levels`` quantises them, which makes ties."""
    gen = torch.Generator().manual_seed(seed)
    scores = torch.randn(len(lengths), candidates, generator=gen)
    if levels is not None:
        scores = torch.randint(0, levels, (len(lengths), candidates), generator=gen).float()
    cols = torch.arange(candidates)[None, :]
    complete = torch.tensor(lengths)[:, None] // POOL
    return scores.masked_fill(cols >= complete, BOUND_FILL).contiguous()


def _lens(lengths):
    return torch.tensor(lengths, dtype=torch.int32)


def _pools(bounded, lengths, k=K):
    got = DS.dsa_decode_select_pools(bounded, _lens(lengths), select_k=k, pool_size=POOL)
    assert _select_launches() == (1, 0)
    DB.reset_decode_batch_dispatch_counters()
    return got


def _pools_oracle(bounded, lengths, k=K):
    return DS.dsa_decode_select_pools_torch_oracle(bounded, _lens(lengths), select_k=k,
                                                   pool_size=POOL)


def _check_layout(values, ids, bounded):
    """The interface's layout, independent of the oracle."""
    k = int(values.shape[1])
    assert values.dtype == torch.float32 and ids.dtype == torch.int32
    width = int(bounded.shape[1])
    for b in range(int(values.shape[0])):
        real = int((ids[b] >= 0).sum())
        assert torch.all(ids[b, :real] >= 0) and torch.all(ids[b, real:] == SENTINEL)
        assert torch.all(values[b, real:] == DS.POOL_PAD_VALUE)
        assert torch.all(ids[b, :real] < width)
        assert torch.equal(values[b, :real], bounded[b, ids[b, :real].long()])
        assert real == min(k, int((bounded[b] > DS.BOUND_FILL_MARK).sum()))
        assert torch.all(ids[b, :real][:-1] < ids[b, :real][1:])   # ascending local ids
        if real < int((bounded[b] > DS.BOUND_FILL_MARK).sum()):
            # The k largest, ties at the k-th value to the lowest ids: every pool left out
            # is below the k-th value, or equal to it at a higher id than every pool kept
            # at that value.
            kth = values[b, :real].min()
            out = torch.ones(width, dtype=torch.bool)
            out[ids[b, :real].long()] = False
            left = bounded[b][out]
            assert torch.all(left <= kth)
            tied_in = ids[b, :real][values[b, :real] == kth]
            tied_out = torch.nonzero(out & (bounded[b] == kth)).flatten()
            assert tied_out.numel() == 0 or int(tied_out.min()) > int(tied_in.max())


@pytest.mark.parametrize("lengths, candidates", [
    ([8190], 2048),
    ([8190, 8000, 2100, 2051], 2048),
    ([8191 - 37 * b for b in range(19)], 2048),
    ([2900, 2300], 700),
    ([2053], 513),
    ([300, 2047, 2048, 2049], 2048),       # rows with fewer than k complete pools
    ([1500, 600, 4], 512),                 # an axis of exactly k pools: nothing to drop
    ([1000, 2000, 9], 500),                # an axis shorter than k: every pool, then pads
    ([60000, 65536], 16384),               # the widest whole-row axis
])
def test_local_topk_equals_the_oracle(lengths, candidates):
    bounded = _bounded(lengths, candidates, seed=len(lengths) + candidates)
    tokens, values, ids = _pools(bounded, lengths)
    want_tokens, want_values, want_ids = _pools_oracle(bounded, lengths)
    assert tuple(values.shape) == tuple(ids.shape) == (len(lengths), K)
    assert tuple(tokens.shape) == (len(lengths), index_expand_width(K, POOL))
    assert torch.equal(values, want_values)
    assert torch.equal(ids, want_ids)
    assert torch.equal(tokens, want_tokens)
    _check_layout(values, ids, bounded)


@pytest.mark.parametrize("levels", [2, 7, 50])
def test_local_topk_ties_take_the_lowest_local_id(levels):
    lengths = [8190, 6000, 4096, 2300, 8191, 2052]
    bounded = _bounded(lengths, 2048, seed=levels, levels=levels)
    _tokens, values, ids = _pools(bounded, lengths)
    _t, want_values, want_ids = _pools_oracle(bounded, lengths)
    assert torch.equal(values, want_values) and torch.equal(ids, want_ids)
    _check_layout(values, ids, bounded)


def test_signed_zeros_tie():
    gen = torch.Generator().manual_seed(11)
    pick = torch.randint(0, 4, (2, 2048), generator=gen)
    bounded = torch.tensor([-0.0, 0.0, 1.0, -1.0])[pick].contiguous()
    _tokens, values, ids = _pools(bounded, [8190, 8190])
    _t, want_values, want_ids = _pools_oracle(bounded, [8190, 8190])
    assert torch.equal(ids, want_ids)
    assert torch.equal(values, want_values)
    _check_layout(values, ids, bounded)


def test_extreme_finite_scores():
    gen = torch.Generator().manual_seed(12)
    mags = torch.tensor([3.0e38, 1.0e-38, 1.0, 1.0e20, 1.0e-20])
    pick = torch.randint(0, 5, (3, 2048), generator=gen)
    sign = torch.where(torch.rand(3, 2048, generator=gen) < 0.5, -1.0, 1.0)
    bounded = (mags[pick] * sign * torch.rand(3, 2048, generator=gen)).contiguous()
    bounded[2, 1000:] = BOUND_FILL
    lengths = [8190, 8190, 4000]
    _tokens, values, ids = _pools(bounded, lengths)
    _t, want_values, want_ids = _pools_oracle(bounded, lengths)
    assert torch.equal(values, want_values) and torch.equal(ids, want_ids)


def test_the_tokens_are_the_selection_kernels():
    """Where both serve (``C > k``), the expanded tokens are ``dsa_decode_select``'s."""
    lengths = [8191 - 61 * b for b in range(6)]
    bounded = _bounded(lengths, 2048, seed=21)
    tokens, _values, _ids = _pools(bounded, lengths)
    want = DS.dsa_decode_select(bounded, _lens(lengths), select_k=K, pool_size=POOL)
    assert torch.equal(tokens, want)


@pytest.mark.parametrize("batch", [2, 5, 64])
def test_two_programs_equal_one(monkeypatch, batch):
    lengths = [8191 - 37 * b for b in range(batch)]
    bounded = _bounded(lengths, 1024, seed=batch)
    one = _pools(bounded, lengths)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert DS.decode_select_programs(batch) == 2
    two = _pools(bounded, lengths)
    for a, b in zip(one, two):
        assert torch.equal(a, b)


@pytest.mark.parametrize("k", [8, 128, 200, 256])
def test_other_k(k):
    lengths = [900, 3000, 40]
    bounded = _bounded(lengths, 700, seed=k)
    got = _pools(bounded, lengths, k=k)
    want = _pools_oracle(bounded, lengths, k=k)
    for a, b in zip(got, want):
        assert torch.equal(a, b)


def test_local_topk_refusals():
    lengths = [8000]
    bounded = _bounded(lengths, 2048, seed=1)
    for k in (0, -1, 2.0, True):
        with pytest.raises(DS.DecodeSelectError):
            DS.dsa_decode_select_pools(bounded, _lens(lengths), select_k=k, pool_size=POOL)
    with pytest.raises(DS.DecodeSelectError):
        DS.dsa_decode_select_pools(bounded, _lens([1, 2]), select_k=K, pool_size=POOL)
    with pytest.raises(DS.DecodeSelectError):
        DS.dsa_decode_select_pools(bounded, _lens(lengths), select_k=K, pool_size=3)
    with pytest.raises(DS.DecodeSelectError):      # past the served whole-row width
        DS.dsa_decode_select_pools(torch.zeros(1, DS.MAX_DECODE_INDEX_CANDIDATES + 1),
                                   _lens(lengths), select_k=K, pool_size=POOL)


def test_the_pools_kernel_is_keyed_by_this_files_digest():
    seen = []
    real = DS.wrap_nki

    def spy(kernel):
        call = real(kernel)

        class _Spy:
            def __getitem__(self, grid):
                inner = call[grid]
                return lambda *a: (seen.append(a), inner(*a))[1]

            def __call__(self, *a):
                seen.append(a)
                return call(*a)
        return _Spy()

    DS.wrap_nki = spy
    try:
        _pools(_bounded([8000, 3000], 2048, seed=1), [8000, 3000])
    finally:
        DS.wrap_nki = real
    assert seen and seen[-1][-1] == DS.SOURCE_DIGEST


# ---------------------------------------------------------------------------------------
# The merge: CP local lists against the CP = 1 list
# ---------------------------------------------------------------------------------------


def _reference_merge(lists, cp, *, seed=0):
    """trn2-2's item-3 rule on the ``CP`` local lists: higher value first, equal values to
    the lower global pool id; the first ``K``. Pads (``-1``) carry the lowest value.

    Each local list is shuffled first: the rule reads values and global ids only, never the
    order a list comes in. The result is returned in the CP = 1 list's order (ascending
    global id, pads last) so that it compares bit for bit."""
    gen = torch.Generator().manual_seed(seed)
    shuffled = []
    for values, ids in lists:
        order = torch.stack([torch.randperm(int(ids.shape[1]), generator=gen)
                             for _ in range(int(ids.shape[0]))])
        shuffled.append((torch.gather(values, 1, order), torch.gather(ids, 1, order)))
    values = torch.cat([v for v, _ in shuffled], dim=1)
    gids = torch.cat([DS.dcp_global_pool_ids(i, cp_rank=r, cp_size=cp, block_pools=BLOCK_POOLS)
                      for r, (_, i) in enumerate(shuffled)], dim=1)
    by_id = torch.sort(gids.to(torch.int64), dim=1, stable=True)
    values = torch.gather(values, 1, by_id.indices)
    by_value = torch.sort(values, dim=1, descending=True, stable=True)
    gids = torch.gather(by_id.values, 1, by_value.indices)[:, :K]
    values = by_value.values[:, :K]
    last = torch.where(gids >= 0, gids, torch.full_like(gids, torch.iinfo(torch.int64).max))
    order = torch.sort(last, dim=1, stable=True).indices
    return torch.gather(values, 1, order), torch.gather(gids, 1, order).to(torch.int32)


def _rank_lists(bounded, lengths, cp, *, local_bounded=None):
    position = torch.tensor([n - 1 for n in lengths], dtype=torch.int64)
    ctx = int(bounded.shape[1]) * POOL
    local_c = DB.dcp_local_candidates(ctx, cp_size=cp, block_size=BLOCK, pool_size=POOL)
    lists = []
    for rank in range(cp):
        ops = DB.dcp_decode_operands(position, cp_rank=rank, cp_size=cp, block_size=BLOCK)
        mine = (_to_local_columns(bounded, rank, cp, local_c) if local_bounded is None
                else local_bounded[rank])
        _tokens, values, ids = _pools(mine, ops.seq_lens.tolist())
        lists.append((values, ids))
    return lists


@pytest.mark.parametrize("cp", CPS)
@pytest.mark.parametrize("levels", [None, 3])
def test_the_merge_of_the_local_lists_is_the_cp1_list(cp, levels):
    # Rows past k pools, rows short of k (padding), block and pool edges, on 2048 pools.
    lengths = [8192, 8190, 8000, 4600, 2051, 1900, 700, 129]
    bounded = _bounded(lengths, 2048, seed=cp * 10 + (levels or 0), levels=levels)
    _t, want_values, want_ids = _pools(bounded, lengths)
    got_values, got_ids = _reference_merge(_rank_lists(bounded, lengths, cp), cp)
    assert torch.equal(got_ids, want_ids)
    assert torch.equal(got_values, want_values)


@pytest.mark.parametrize("cp", CPS)
def test_the_chain_merges_to_the_cp1_list(cp):
    """Scores on each rank's own bank (with the DCP operands), its local top-k, the merge:
    the CP = 1 chain's list."""
    lengths = [4096, 4093, 3201, 2944, 2051, 600]
    ctx = 4096
    case = _global_case(lengths, ctx, seed=40 + cp)
    position = case["position"]
    bounded = _scores(case, case["bank"], position + 1, position, case["candidates"])
    _t, want_values, want_ids = _pools(bounded, lengths)
    local_c = DB.dcp_local_candidates(ctx, cp_size=cp, block_size=BLOCK, pool_size=POOL)
    local = []
    for rank in range(cp):
        ops = DB.dcp_decode_operands(position, cp_rank=rank, cp_size=cp, block_size=BLOCK)
        local.append(_scores(case, _local_bank(case["bank"], rank, cp, local_c), ops.seq_lens,
                             ops.position, local_c))
    got_values, got_ids = _reference_merge(_rank_lists(bounded, lengths, cp,
                                                       local_bounded=local), cp)
    assert torch.equal(got_ids, want_ids)
    assert torch.equal(got_values, want_values)


# ---------------------------------------------------------------------------------------
# The ownership remap after the merge
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("cp", (1,) + CPS)
def test_the_remap_round_trips(cp):
    gen = torch.Generator().manual_seed(cp)
    local = torch.randint(0, 4096, (3, K), generator=gen).to(torch.int32)
    local[:, -7:] = SENTINEL
    for rank in range(cp):
        gids = DS.dcp_global_pool_ids(local, cp_rank=rank, cp_size=cp, block_pools=BLOCK_POOLS)
        assert gids.dtype == torch.int32
        want = torch.tensor([[_global_pool(p, rank, cp) if p >= 0 else SENTINEL for p in row]
                             for row in local.tolist()], dtype=torch.int32)
        assert torch.equal(gids, want)
        for other in range(cp):
            rows = DS.dcp_owned_pool_rows(gids, cp_rank=other, cp_size=cp,
                                          block_pools=BLOCK_POOLS)
            assert rows.dtype == torch.int32
            if other == rank:
                assert torch.equal(rows, local)
            else:
                assert torch.all(rows == SENTINEL)


@pytest.mark.parametrize("cp", CPS)
def test_the_ranks_rows_expand_to_the_cp1_tokens(cp):
    """After the merge each rank keeps the ids it owns, as local rows, and expands them with
    its own length: the union of the ranks' tokens, mapped back to global positions, is the
    CP = 1 selection's token set (tail included)."""
    lengths = [8192, 8190, 5003, 2051, 1031]
    bounded = _bounded(lengths, 2048, seed=99)
    tokens_cp1, _v, ids_cp1 = _pools(bounded, lengths)
    position = torch.tensor([n - 1 for n in lengths], dtype=torch.int64)
    union = [set() for _ in lengths]
    for rank in range(cp):
        ops = DB.dcp_decode_operands(position, cp_rank=rank, cp_size=cp, block_size=BLOCK)
        rows = DS.dcp_owned_pool_rows(ids_cp1, cp_rank=rank, cp_size=cp, block_pools=BLOCK_POOLS)
        local_tokens = _dsa_index_expand_torch(rows, ops.seq_lens.to(torch.int32), POOL)
        for b, row in enumerate(local_tokens.tolist()):
            union[b] |= {_global_token(t, rank, cp) for t in row if t >= 0}
    assert union == [set(t for t in row if t >= 0) for row in tokens_cp1.tolist()]


def test_remap_refusals():
    ids = torch.zeros(1, K, dtype=torch.int32)
    with pytest.raises(DS.DecodeSelectError):
        DS.dcp_owned_pool_rows(ids, cp_rank=2, cp_size=2, block_pools=BLOCK_POOLS)
    with pytest.raises(DS.DecodeSelectError):
        DS.dcp_global_pool_ids(ids, cp_rank=0, cp_size=0, block_pools=BLOCK_POOLS)
    with pytest.raises(DS.DecodeSelectError):
        DS.dcp_global_pool_ids(ids.float(), cp_rank=0, cp_size=2, block_pools=BLOCK_POOLS)


# ---------------------------------------------------------------------------------------
# The device compile of the new kernel: ordering check and dependency verdicts
# ---------------------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "VLLM_NEURON_CPU_COMPILE",
         "NEURON_LIBTORCH_CPU_MODE", "NEURON_LIBTORCH_REMOTE_CACHE")
_DEVICE_NODES = bool(glob.glob("/dev/neuron*"))
_BWRAP = shutil.which("bwrap")
_NO_DEVICE = (() if not _DEVICE_NODES else
              (_BWRAP, "--dev-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--"))
_needs_compiler = pytest.mark.skipif(
    (_DEVICE_NODES and _BWRAP is None)
    or (shutil.which("neuronx-cc") is None
        and not pathlib.Path(sys.executable).with_name("neuronx-cc").exists()),
    reason="neuronx-cc is not installed, or this host has /dev/neuron* and no bwrap to hide "
           "them from the compile")
#: ``(B, candidates)``: one request at the local axes of a 128k context at CP = 2 (the
#: widest served axis) and of an 8k context at CP = 8 (shorter than ``k``); a full tile
#: (two compaction groups) at the widest axis; the two-program batches.
COMPILE_SHAPES = ((1, 16384), (1, 256), (16, 16384), (4, 4096), (64, 1024), (64, 2048))


def _depcheck(dump: str) -> tuple[str, str]:
    """The dependency checker's one-pass and loop-carried VERDICT lines on one dump, run as
    its command line runs it (``python -m test.kernel_depcheck.depcheck DUMP``)."""
    done = subprocess.run([sys.executable, "-m", "test.kernel_depcheck.depcheck", dump],
                          cwd=str(_ROOT), capture_output=True, text=True, timeout=900,
                          check=False, env={**os.environ, "PYTHONPATH": str(_ROOT)})
    assert done.returncode == 0, (dump, done.stdout[-2000:], done.stderr[-2000:])
    lines = done.stdout.splitlines()
    one_pass = [line for line in lines if line.startswith("VERDICT: ")]
    loop_carried = [line for line in lines if line.startswith("VERDICT (loop-carried): ")]
    assert len(one_pass) == 1 and len(loop_carried) == 1, done.stdout[-2000:]
    return one_pass[0], loop_carried[0]


def _compile_pools(batch: int, candidates: int):
    from test.vllm_neuron.functional.dsa import sbuf_order

    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    programs = 2 if batch >= 2 else 1
    with tempfile.TemporaryDirectory(prefix="dsa_dcp_select_") as art:
        environment.update(
            NEURON_PLATFORM_TARGET_OVERRIDE="trn2", NKI_COMPILE_CACHE_URL=f"{art}/nki",
            PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(_ROOT),
            PATH=f"{pathlib.Path(sys.executable).parent}:{environment.get('PATH', '')}")
        done = subprocess.run(
            [*_NO_DEVICE, sys.executable, str(_HERE), art, str(batch), str(candidates),
             str(programs)],
            cwd=art, env=environment, capture_output=True, text=True, timeout=1500,
            check=False)
        assert done.returncode == 0, (batch, candidates, done.stdout[-2000:],
                                      done.stderr[-3000:])
        dumps = sorted(glob.glob(f"{art}/nc0*/sg00/bir_debug.*after-lower_sync*.json")
                       or glob.glob(f"{art}/sg00/bir_debug.*after-lower_sync*.json"))
        assert len(dumps) == programs, (batch, candidates, dumps)
        out = []
        for dump in dumps:
            out.append((pathlib.Path(dump).relative_to(art).parts[0],
                        sbuf_order.check_dump(dump), *_depcheck(dump)))
        return out


@_needs_compiler
@pytest.mark.parametrize("batch, candidates", COMPILE_SHAPES,
                         ids=[f"B{b}-C{c}" for b, c in COMPILE_SHAPES])
def test_the_pools_kernel_compiles_with_every_access_ordered(batch, candidates):
    for core, order, one_pass, loop_carried in _compile_pools(batch, candidates):
        label = f"pools B={batch} candidates={candidates} {core}"
        print(f"{label}: {order.summary()}")
        print(f"{label}: {one_pass}")
        print(f"{label}: {loop_carried}")
        assert order.clean, (label, order.summary(), order.unsync[:4], order.library_alias[:4],
                             order.stepped_dma[:4], order.early_waits[:4])
        assert one_pass.startswith("VERDICT: CLEAN"), (label, one_pass)
        assert loop_carried.startswith("VERDICT (loop-carried): CLEAN"), (label, loop_carried)


def _compile_child(art: str, batch: int, candidates: int, programs: int):
    """Child process: compile the pools kernel to a NEFF with the ``lower_sync`` dump in
    ``art``, on the arguments its host side passes (``dsa_decode_select_pools``)."""
    import inspect
    from dataclasses import replace

    import nki.language as nl
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import compile_bir_to_neff
    from nki.framework.compiled import CompileKernel
    from nki.language.buffers import shared_hbm
    from nki.language.tensor import NkiTensor

    func = DS.dsa_decode_select_pools_kernel.func
    span = DS.select_span(candidates, K)
    arguments = {"bounded_hbm": ((batch, candidates), nl.float32),
                 "lens_hbm": ((batch, 1), nl.int32),
                 "select_k": K, "pool_size": POOL,
                 "out_cols": int(index_expand_width(K, POOL)),
                 "fold_cap": DS.select_fold_cap(span), "span": span,
                 "source_digest": DS.SOURCE_DIGEST}
    inputs = {}
    for name in inspect.signature(func).parameters:
        value = arguments[name]
        inputs[name] = (NkiTensor(name=name, shape=value[0], dtype=value[1], storage=None,
                                  buffer=shared_hbm) if isinstance(value, tuple) else value)
    compiled = CompileKernel(func=func, lnc=programs, target="trn2", artifacts_dir=art)
    opts = replace(compiled._compile_opts(), output_path=os.path.join(art, "kernel.neff"),
                   neuronx_cc_args=("--internal-print-after=lower_sync",))
    bir = compile_to_bir(func, frontend=ParserFrontend(enable_backend_opt=False),
                         inputs=inputs, compile_opts=opts, enable_cache=False)
    compile_bir_to_neff(opts, bir, input_arrays=[],
                        argument_names=[s.name for s in bir.descriptor.input_specs],
                        output_arg_names=[s.name for s in bir.descriptor.output_specs])


if __name__ == "__main__":
    _compile_child(sys.argv[1], *map(int, sys.argv[2:5]))
