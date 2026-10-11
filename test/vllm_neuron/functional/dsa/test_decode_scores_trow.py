# SPDX-License-Identifier: Apache-2.0
"""Bounded candidate scores for ``T`` query rows per request: the verify step's form.

A speculative verify step scores ``T = 1 + k`` rows per request in one call. Row ``t``
sits at position ``start + t``; it must see every pool complete at that position,
including the ones this very step closes (rows ``t' <= t`` with ``(start + t') % pool
== pool - 1``), which no bank write has landed yet. ``dsa_decode_scores_rows`` stands
every such pool in from the ring step's ``pooled`` rows and bounds each row at its own
length. The reference is the path it must equal: ``T`` sequential one-row
``decode_batch.dsa_decode_scores`` calls, each step's pool written into the bank before
the next one scores. Same matmul shapes, so the comparison is bit for bit.

The kernel also takes the query as projected and rotates it on chip
(``rotate_query=True``): the reference is the same kernel handed the query that
``dsa_hadamard128`` rotates, bit for bit, with no rotation kernel launched.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.functional.dsa import decode_trow as TR
from vllm_neuron.functional.dsa import kpool_hadamard as KH
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL
from vllm_neuron.utils.neuron_utils import can_run_kernel

POOL = 4
HEAD_DIM = 128
HEADS = 32
ROWS = (1, 2, 4, 6)
BATCHES = (1, 4)
#: (candidates, row 0's positions): the 4096 and 8192 windows, every residue mod 4 and a
#: step that reaches the window's last pool.
CASES = {1024: (4090, 2100, 2101, 2102), 2048: (8186, 3000, 2052, 4000)}


def _bf16(gen, *shape, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(torch.bfloat16)


def _case(batch, rows, candidates, *, seed, heads=HEADS, slots_total=10):
    gen = torch.Generator().manual_seed(seed)
    bank = _bf16(gen, slots_total, candidates + 1, HEAD_DIM, scale=0.5)
    slots = torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32)
    position = torch.tensor(list(CASES[candidates])[:batch], dtype=torch.int32)
    query = _bf16(gen, batch * rows, heads, HEAD_DIM)
    weights = torch.randn(batch * rows, heads, generator=gen) * heads ** -0.5
    pooled = _bf16(gen, batch * rows, HEAD_DIM, scale=0.5)
    return query, weights, bank, slots, position, pooled


def _sequential(query, weights, bank, slots, position, pooled, candidates):
    """``rows`` one-row score calls per request, each pool written before the next."""
    batch = int(slots.shape[0])
    rows = int(query.shape[0]) // batch
    live = bank.clone()
    out = torch.empty(batch * rows, candidates, dtype=torch.float32)
    for t in range(rows):
        picked = torch.arange(batch) * rows + t
        pos = position + t
        out[picked] = DB.dsa_decode_scores(
            query[picked], weights[picked], live, slots, pos + 1, pos, pooled[picked],
            candidates=candidates, pool_size=POOL)
        slot_index, row = DB.decode_pool_destinations(slots, pos, rows=int(bank.shape[1]),
                                                      pool_size=POOL)
        live.index_put_((slot_index, row), pooled[picked])
    return out


def _run(case, candidates, monkeypatch, lnc=None, **options):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    TR.reset_decode_trow_dispatch_counters()
    out = TR.dsa_decode_scores_rows(*case, candidates=candidates, pool_size=POOL, **options)
    assert TR.decode_trow_dispatch_counters() == (1, 0)
    return out


def _rotated(query):
    """``query`` as ``dsa_hadamard128`` rotates it: the form the kernel takes by default."""
    total, heads, head_dim = query.shape
    flat = query.reshape(total * heads, head_dim).contiguous()
    return KH.dsa_hadamard128(flat).reshape(total, heads, head_dim)


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"


@pytest.mark.parametrize("candidates", sorted(CASES))
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("rows", ROWS)
def test_rows_equal_sequential_one_row_scores_bit_for_bit(rows, batch, candidates,
                                                          monkeypatch):
    case = _case(batch, rows, candidates, seed=rows + 10 * batch + candidates)
    want = _sequential(*case, candidates)
    got = _run(case, candidates, monkeypatch)
    assert got.shape == (batch * rows, candidates) and got.dtype == torch.float32
    assert torch.equal(got, want)
    # The bound moves with the row: a later row sees at least as many pools.
    filled = (got == BOUND_FILL).sum(dim=1).reshape(batch, rows)
    assert bool((filled[:, 1:] <= filled[:, :-1]).all())


def test_the_steps_own_pools_are_read_from_pooled_not_the_bank(monkeypatch):
    rows, candidates = 6, 1024
    case = list(_case(4, rows, candidates, seed=77))
    clean = _run(tuple(case), candidates, monkeypatch)
    query, weights, bank, slots, position, pooled = case
    poisoned = bank.clone()
    for b in range(4):
        for t in range(rows):
            pos = int(position[b]) + t
            if pos % POOL == POOL - 1:
                poisoned[int(slots[b]), pos // POOL] = 3.0e4
    case[2] = poisoned
    assert torch.equal(clean, _run(tuple(case), candidates, monkeypatch))


def test_rows_read_only_the_named_slots(monkeypatch):
    case = list(_case(4, 4, 1024, seed=78))
    clean = _run(tuple(case), 1024, monkeypatch)
    bank, slots = case[2], case[3]
    poisoned = torch.full_like(bank, 3.0e4)
    poisoned[slots.long()] = bank[slots.long()]
    case[2] = poisoned
    assert torch.equal(clean, _run(tuple(case), 1024, monkeypatch))


def test_two_programs_agree_with_one(monkeypatch):
    case = _case(4, 4, 1024, seed=79)
    one = _run(case, 1024, monkeypatch)
    two = _run(case, 1024, monkeypatch, lnc="2")
    assert TR.decode_trow_route_counts()[1] == 1
    assert torch.equal(one, two)


def test_few_heads_and_a_ragged_candidate_axis(monkeypatch):
    case = _case(2, 3, 1024, seed=80, heads=4)
    query, weights, bank, slots, position, pooled = case
    candidates = 300
    case = (query, weights, bank[:, :candidates + 1].contiguous(), slots,
            torch.tensor([1190, 1100], dtype=torch.int32), pooled)
    want = _sequential(*case, candidates)
    assert torch.equal(_run(case, candidates, monkeypatch), want)


@pytest.mark.parametrize("candidates", sorted(CASES))
@pytest.mark.parametrize("batch, rows", [(1, 1), (1, 4), (4, 1), (4, 4)])
def test_a_projected_query_is_rotated_on_chip_as_the_rotation_kernel_rotates_it(
        batch, rows, candidates, monkeypatch):
    case = _case(batch, rows, candidates, seed=90 + rows + 10 * batch + candidates)
    want = _run((_rotated(case[0]),) + case[1:], candidates, monkeypatch)
    KH.reset_kpool_hadamard_dispatch_counters()
    got = _run(case, candidates, monkeypatch, rotate_query=True)
    assert KH.kpool_hadamard_dispatch_counters() == (0, 0)
    assert torch.equal(got, want)


def test_the_on_chip_rotation_splits_over_an_lnc2_pair(monkeypatch):
    case = _case(4, 4, 1024, seed=91)
    one = _run(case, 1024, monkeypatch, rotate_query=True)
    two = _run(case, 1024, monkeypatch, lnc="2", rotate_query=True)
    assert TR.decode_trow_route_counts()[1] == 1
    assert torch.equal(one, two)


@pytest.mark.parametrize("batch, rows", [(1, 4), (4, 1)])
def test_the_oracle_rotates_a_projected_query_as_the_rotation_kernel_s_torch_path(
        batch, rows):
    case = _case(batch, rows, 1024, seed=92 + rows + 10 * batch)
    total, heads, head_dim = case[0].shape
    rotated = (case[0].float().reshape(-1, head_dim) @ KH.hadamard_matrix(head_dim).t()
               * KH.HADAMARD_SCALE).to(torch.bfloat16).reshape(total, heads, head_dim)
    want = TR.dsa_decode_scores_rows_torch_oracle(
        rotated, *case[1:], candidates=1024, pool_size=POOL)
    got = TR.dsa_decode_scores_rows_torch_oracle(
        *case, candidates=1024, pool_size=POOL, rotate_query=True)
    assert torch.equal(got, want)


def test_a_query_the_rotation_does_not_fit_is_refused_by_name():
    query, weights, bank, slots, position, pooled = _case(1, 2, 1024, seed=93)
    narrow = (query[:, :, :64].contiguous(), weights, bank[:, :, :64].contiguous(), slots,
              position, pooled[:, :64].contiguous())
    with pytest.raises(TR.DecodeTrowError, match="128"):
        TR.dsa_decode_scores_rows(*narrow, candidates=1024, pool_size=POOL,
                                  rotate_query=True)


def test_the_kernel_key_covers_the_rotation_constants_the_kernel_reads():
    """The kernel's cache key is a digest over this module and ``kpool_hadamard.py``,
    whose rotation size and scale the on-chip rotation reads: an edit to either re-keys
    the kernel."""
    from vllm_neuron.source_digest import source_digest

    assert TR.SOURCE_DIGEST == source_digest(TR.__file__, KH.__file__)


def test_refusals_name_the_operand():
    query, weights, bank, slots, position, pooled = _case(4, 2, 1024, seed=81)
    call = TR.dsa_decode_scores_rows
    with pytest.raises(TR.DecodeTrowError, match="whole number of rows"):
        call(query[:7], weights[:7], bank, slots, position, pooled[:7], candidates=1024,
             pool_size=POOL)
    with pytest.raises(TR.DecodeTrowError, match="pooled"):
        call(query, weights, bank, slots, position, pooled[:4], candidates=1024,
             pool_size=POOL)
    with pytest.raises(TR.DecodeTrowError, match="candidates"):
        call(query, weights, bank, slots, position, pooled, candidates=1025, pool_size=POOL)
    with pytest.raises(TR.DecodeTrowError, match="position"):
        call(query, weights, bank, slots, position[:2], pooled, candidates=1024,
             pool_size=POOL)
    with pytest.raises(TR.DecodeTrowError, match="window"):
        call(query, weights, bank, slots, position + 8, pooled, candidates=1024,
             pool_size=POOL)
