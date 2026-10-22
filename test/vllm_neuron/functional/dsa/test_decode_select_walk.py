# SPDX-License-Identifier: Apache-2.0
"""The DSA decode pools selection on a rank axis wider than one compaction row.

Under decode context parallelism (DCP) each rank selects its local top-``k`` (values and
pool ids) and the group merges the ranks' lists. ``decode_select.dsa_decode_select_pools``
gives that list on a rank axis of up to ``WHOLE_ROW_COLUMNS`` (16384) pools;
``decode_select_walk.dsa_decode_select_pools_walk`` gives it on a wider one, up to
``MAX_DECODE_POOLS_WALK_CANDIDATES`` (65536), with the chunk walk. What this file pins:

* **The walk kernel** (NKI simulator) against its torch oracle, ids equal and values bit
  for bit, at 16385, 32768, 40000 and 65536 candidates, one and four rows; two programs
  give the one-program lists; ties at the ``k``-th value across a chunk boundary go to the
  lowest ids; rows with fewer than ``k`` real scores list every one, then pads; every pick
  in one chunk; signed zeros keep their bits; extreme finite scores; other ``k``. The
  oracle is ``decode_select``'s pools oracle, and the lists keep the interface's layout
  independent of either.
* **The chain at DCP 4 and 8**: each rank's walk over the pools it owns, then the merge
  kernel on every rank: the union of the ranks' merged pools is the CP = 1 selection over
  the global axis, and no pool is named by two ranks.
* **The host**: refusals before any launch, each naming its bound and the file; the start
  check (one request per step: the walk's bound; more: the whole-row bound) and the
  runner's start check under DCP at 1048576.
* **The route**: the indexer's DCP branch takes the walk past one compaction row and the
  whole-row entry up to it, and hands the merge the oracle's lists.
* **The device compile** at the a8 and a4 shapes: the NEFF builds, every access is ordered
  (``sbuf_order``), the dependency checker's verdicts are CLEAN, no SB spill, and each
  core's SBUF allocation, every location live at once, is within ``SBUF_BYTES_PER_PARTITION``
  and outside the runtime's reservations (all printed).
"""

from __future__ import annotations

import dataclasses
import functools
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch

import vllm_neuron.model.glm5_next.model_fp8 as model_fp8
from test.vllm_neuron.functional.dsa.dsa_decode_case import decode_config
from vllm_neuron.functional.dsa import dcp_candidate_merge as DCM
from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.functional.dsa import decode_select as DS
from vllm_neuron.functional.dsa import decode_select_walk as DSW
from vllm_neuron.functional.dsa import kpool_hadamard as KH
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL, BOUND_FILL_MARK, SENTINEL
from vllm_neuron.source_digest import source_digest
from vllm_neuron.utils import dcp_ownership as OWN
from vllm_neuron.utils.neuron_utils import SBUF_BYTES_PER_PARTITION, can_run_kernel

_CFG = decode_config()
POOL = int(_CFG.index_kpool)
#: Pools the selection keeps: ``index_topk`` tokens in whole pools.
K = int(_CFG.index_topk) // POOL
#: ``hybrid_kv_block_size`` of the serve recipes: the DCP interleave unit.
BLOCK = 128
PPB = BLOCK // POOL
ROW = DS.WHOLE_ROW_COLUMNS
#: The walk's axes: one pool past one row, two whole rows (a8 at 1M), a partial last chunk,
#: four whole rows (a4 at 1M).
WIDTHS = (ROW + 1, 2 * ROW, 40000, 4 * ROW)


@pytest.fixture(autouse=True)
def _kernels_on(monkeypatch):
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    DB.reset_decode_batch_dispatch_counters()
    DSW.reset_walk_dispatch_count()


def _select_launches():
    """``(selection launches, torch fallbacks)`` in ``decode_batch``'s family."""
    return (DB.decode_batch_route_counts()[3], DB.decode_batch_dispatch_counters()[1])


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


def _bits(values):
    return values.view(torch.int32)


def _walk(bounded, k=K):
    """One walk launch on the kernel, counted in both families; ``(values, ids)``."""
    DB.reset_decode_batch_dispatch_counters()
    DSW.reset_walk_dispatch_count()
    values, ids = DSW.dsa_decode_select_pools_walk(bounded, select_k=k)
    assert _select_launches() == (1, 0)
    assert DSW.walk_dispatch_count() == 1
    return values, ids


def _check_layout(values, ids, bounded):
    """The interface's layout, independent of the oracle: ascending real ids, then pads;
    each value the bounded score at its id bit for bit; the ``k`` largest real scores, ties
    at the ``k``-th value to the lowest ids."""
    k = int(values.shape[1])
    assert values.dtype == torch.float32 and ids.dtype == torch.int32
    width = int(bounded.shape[1])
    for b in range(int(values.shape[0])):
        real_scores = bounded[b] > BOUND_FILL_MARK
        real = int((ids[b] >= 0).sum())
        assert real == min(k, int(real_scores.sum()))
        assert torch.all(ids[b, :real] >= 0) and torch.all(ids[b, real:] == SENTINEL)
        assert torch.all(ids[b, :real] < width)
        assert torch.all(_bits(values[b, real:]) == _bits(torch.tensor(DS.POOL_PAD_VALUE)))
        assert torch.equal(_bits(values[b, :real]), _bits(bounded[b, ids[b, :real].long()]))
        assert torch.all(ids[b, :real][:-1] < ids[b, :real][1:])
        if real and real < int(real_scores.sum()):
            kth = values[b, :real].min()
            out = real_scores.clone()
            out[ids[b, :real].long()] = False
            assert torch.all(bounded[b][out] <= kth)
            tied_in = ids[b, :real][values[b, :real] == kth]
            tied_out = torch.nonzero(out & (bounded[b] == kth)).flatten()
            assert tied_out.numel() == 0 or int(tied_out.min()) > int(tied_in.max())


def _assert_equals_the_oracle(bounded, values, ids, k=K):
    want_values, want_ids = DSW.dsa_decode_select_pools_walk_torch_oracle(bounded, select_k=k)
    assert tuple(values.shape) == tuple(ids.shape) == (int(bounded.shape[0]), k)
    assert torch.equal(ids, want_ids)
    assert torch.equal(_bits(values), _bits(want_values))
    lens = torch.full((int(bounded.shape[0]),), int(bounded.shape[1]) * POOL, dtype=torch.int32)
    _t, head_values, head_ids = DS.dsa_decode_select_pools_torch_oracle(
        bounded, lens, select_k=k, pool_size=POOL)
    assert torch.equal(head_ids, want_ids)
    assert torch.equal(_bits(head_values), _bits(want_values))
    _check_layout(values, ids, bounded)


# ---------------------------------------------------------------------------------------
# The walk kernel against the oracle
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("candidates", WIDTHS)
def test_the_walk_equals_the_oracle(candidates, rows):
    # One row, or a verify step's four rows (lengths n .. n + 3): every pool complete but the
    # last one or two.
    lengths = [4 * candidates - 2 - rows + 1 + t for t in range(rows)]
    bounded = _bounded(lengths, candidates, seed=candidates + rows)
    values, ids = _walk(bounded)
    _assert_equals_the_oracle(bounded, values, ids)


@pytest.mark.parametrize("batch, candidates", [(2, 2 * ROW), (4, 40000), (4, 4 * ROW)])
def test_two_programs_give_the_one_program_lists(monkeypatch, batch, candidates):
    lengths = [4 * candidates - 1 - 37 * b for b in range(batch)]
    bounded = _bounded(lengths, candidates, seed=batch * candidates)
    one = _walk(bounded)
    assert DB.decode_batch_route_counts()[2] == 0
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert DS.decode_select_programs(batch) == 2
    two = _walk(bounded)
    assert DB.decode_batch_route_counts()[2] == 1
    assert torch.equal(one[1], two[1]) and torch.equal(_bits(one[0]), _bits(two[0]))
    _assert_equals_the_oracle(bounded, *two)


@pytest.mark.parametrize("candidates", [2 * ROW, 40000, 4 * ROW])
def test_ties_at_the_kth_value_across_a_chunk_boundary_take_the_lowest_ids(candidates):
    gen = torch.Generator().manual_seed(candidates)
    bounded = -1.0 - torch.rand(2, candidates, generator=gen)
    for row, edge in ((0, ROW), (1, (DS.row_chunks(candidates) - 1) * ROW)):
        # 300 scores of 5.0 in the row's first chunk, then 400 ties at 2.0 around a chunk
        # edge: the k-th value is 2.0, and the 212 lowest tied ids cross the edge.
        top = torch.randperm(ROW - 1000, generator=gen)[:300]
        bounded[row, top] = 5.0
        bounded[row, edge - 184:edge + 216] = 2.0
    values, ids = _walk(bounded.contiguous())
    _assert_equals_the_oracle(bounded, values, ids)
    for row, edge in ((0, ROW), (1, (DS.row_chunks(candidates) - 1) * ROW)):
        tied = ids[row][values[row] == 2.0]
        assert tied.tolist() == list(range(edge - 184, edge + 28)), row


@pytest.mark.parametrize("levels", [2, 9])
def test_quantised_scores_tie_to_the_lowest_ids(levels):
    lengths = [4 * 40000 - 2, 4 * 30000, 4 * 16390 + 1]
    bounded = _bounded(lengths, 40000, seed=levels, levels=levels)
    values, ids = _walk(bounded)
    _assert_equals_the_oracle(bounded, values, ids)


def test_rows_with_fewer_than_k_real_scores_list_every_one_then_pads():
    candidates = 40000
    gen = torch.Generator().manual_seed(7)
    bounded = torch.full((4, candidates), BOUND_FILL)
    bounded[0, :300] = torch.randn(300, generator=gen)            # a short request: 300 pools
    spread = torch.randperm(candidates, generator=gen)[:450]      # 450 real scores, all chunks
    bounded[1, spread] = torch.randn(450, generator=gen)
    exact = torch.randperm(candidates, generator=gen)[:K]         # exactly k real scores
    bounded[2, exact] = torch.randn(K, generator=gen)
    # Row 3: no real score (a rank that owns none of the request's blocks).
    values, ids = _walk(bounded)
    _assert_equals_the_oracle(bounded, values, ids)
    assert [int((r >= 0).sum()) for r in ids] == [300, 450, K, 0]
    assert sorted(ids[1][:450].tolist()) == sorted(spread.tolist())
    assert torch.all(ids[3] == SENTINEL)


@pytest.mark.parametrize("candidates", [2 * ROW, 4 * ROW])
def test_every_pick_in_one_chunk(candidates):
    gen = torch.Generator().manual_seed(candidates + 3)
    chunks = DS.row_chunks(candidates)
    bounded = torch.randn(3, candidates, generator=gen) - 10.0
    for row, chunk in enumerate((0, chunks - 1, 1)):
        bounded[row, chunk * ROW:(chunk + 1) * ROW] += 20.0
    values, ids = _walk(bounded.contiguous())
    _assert_equals_the_oracle(bounded, values, ids)
    for row, chunk in enumerate((0, chunks - 1, 1)):
        assert torch.all(ids[row] // ROW == chunk), row


def test_signed_zeros_are_one_value_and_keep_their_bits():
    candidates = 2 * ROW
    gen = torch.Generator().manual_seed(11)
    bounded = torch.tensor([-1.0, -2.0])[torch.randint(0, 2, (2, candidates), generator=gen)]
    for row in range(2):
        bounded[row, torch.randperm(candidates, generator=gen)[:300]] = 1.0
        # Zeros of both signs: 100 in chunk 0, 400 in chunk 1; the k-th value is zero and
        # its 212 lowest ties take both chunks.
        low = torch.randperm(ROW, generator=gen)[:100]
        high = ROW + torch.randperm(ROW, generator=gen)[:400]
        for cols in (low, high):
            cols = cols[bounded[row, cols] != 1.0]
            sign = torch.where(torch.rand(cols.numel(), generator=gen) < 0.5, -1.0, 1.0)
            bounded[row, cols] = 0.0 * sign
    values, ids = _walk(bounded.contiguous())
    _assert_equals_the_oracle(bounded, values, ids)
    zeros = values == 0.0
    assert bool(torch.any(_bits(values[zeros]) == _bits(torch.tensor(-0.0))))
    assert bool(torch.any(_bits(values[zeros]) == 0))
    assert bool(torch.any((ids >= ROW) & zeros)) and bool(torch.any((ids < ROW) & zeros))


def test_extreme_finite_scores():
    candidates = 40000
    gen = torch.Generator().manual_seed(12)
    mags = torch.tensor([3.0e38, 1.0e-38, 1.0, 1.0e20, 1.0e-20])
    pick = torch.randint(0, 5, (3, candidates), generator=gen)
    sign = torch.where(torch.rand(3, candidates, generator=gen) < 0.5, -1.0, 1.0)
    bounded = (mags[pick] * sign * torch.rand(3, candidates, generator=gen)).contiguous()
    bounded[2, 20000:] = BOUND_FILL
    values, ids = _walk(bounded)
    _assert_equals_the_oracle(bounded, values, ids)


@pytest.mark.parametrize("k", [8, 128, 256])
def test_other_k(k):
    lengths = [4 * 20000 - 1, 4 * 17000, 4 * 900]
    bounded = _bounded(lengths, 20000, seed=k)
    values, ids = _walk(bounded, k=k)
    _assert_equals_the_oracle(bounded, values, ids, k=k)


def test_the_walk_kernel_is_keyed_by_both_files_digest(monkeypatch):
    assert DSW.SOURCE_DIGEST == source_digest(DSW.__file__, DS.__file__)
    seen = []
    real = DSW.wrap_nki

    def spy(kernel):
        call = real(kernel)

        class _Spy:
            def __getitem__(self, grid):
                inner = call[grid]
                return lambda *a: (seen.append((kernel, a)), inner(*a))[1]

            def __call__(self, *a):
                seen.append((kernel, a))
                return call(*a)
        return _Spy()

    monkeypatch.setattr(DSW, "wrap_nki", spy)
    _walk(_bounded([4 * 20000], 20000, seed=1))
    assert len(seen) == 1
    kernel, arguments = seen[0]
    assert kernel is DSW.dsa_decode_select_pools_walk_kernel
    assert arguments[1:] == (K, DS.select_fold_cap(DS.mask_span(20000)), DSW.SOURCE_DIGEST)


# ---------------------------------------------------------------------------------------
# The host: refusals and the start checks
# ---------------------------------------------------------------------------------------


def test_the_walk_bound_is_four_whole_rows():
    assert DSW.POOLS_WALK_ROW_CHUNKS == 4 == DS.DEVICE_CHECKED_ROW_CHUNKS
    assert DSW.MAX_DECODE_POOLS_WALK_CANDIDATES == 4 * ROW == 65536
    assert DS.MAX_DECODE_POOLS_CANDIDATES == ROW


@pytest.mark.parametrize("bounded, select_k, match", [
    (torch.zeros(1, ROW), K,
     r"16384-wide candidate axis is one compaction row \(<= WHOLE_ROW_COLUMNS=16384\).*"
     r"dsa_decode_select_pools.*decode_select_walk\.py"),
    (torch.zeros(1, 4 * ROW + 1), K,
     r"65537-wide candidate axis is past MAX_DECODE_POOLS_WALK_CANDIDATES=65536.*"
     r"decode_select_walk\.py"),
    (torch.zeros(1, 2 * ROW, dtype=torch.bfloat16), K, r"float32.*decode_select_walk\.py"),
    (torch.zeros(2 * ROW), K, r"\[R, C\].*decode_select_walk\.py"),
    (torch.zeros(0, 2 * ROW), K, r"at least one row.*decode_select_walk\.py"),
    (torch.zeros(1, 2 * ROW), 0, r"select_k.*decode_select_walk\.py"),
    (torch.zeros(1, 2 * ROW), -1, r"select_k.*decode_select_walk\.py"),
    (torch.zeros(1, 2 * ROW), 2.0, r"select_k.*decode_select_walk\.py"),
    (torch.zeros(1, 2 * ROW), True, r"select_k.*decode_select_walk\.py"),
    (torch.zeros(1, 2 * ROW), ROW + 1,
     r"select_k.*WHOLE_ROW_COLUMNS=16384.*decode_select_walk\.py"),
], ids=["one-row", "past-four-rows", "bf16", "1d", "no-rows", "k0", "k-1", "k-float",
        "k-bool", "k-past-a-row"])
def test_the_walk_refuses_before_any_launch_naming_the_bound_and_the_file(bounded, select_k,
                                                                          match):
    with pytest.raises(DS.DecodeSelectError, match=match):
        DSW.dsa_decode_select_pools_walk(bounded, select_k=select_k)
    assert _select_launches() == (0, 0) and DSW.walk_dispatch_count() == 0


def test_with_no_kernel_the_walk_is_its_oracle(monkeypatch):
    bounded = _bounded([4 * 20000 - 3, 4 * 300], 20000, seed=5)
    monkeypatch.setattr(DSW, "can_run_kernel", lambda *a: False)
    values, ids = DSW.dsa_decode_select_pools_walk(bounded, select_k=K)
    assert DB.decode_batch_dispatch_counters() == (0, 1) and DSW.walk_dispatch_count() == 0
    want_values, want_ids = DSW.dsa_decode_select_pools_walk_torch_oracle(bounded, select_k=K)
    assert torch.equal(ids, want_ids) and torch.equal(_bits(values), _bits(want_values))


def test_the_start_check_at_one_request_per_step_takes_the_walk_bound():
    DSW.check_decode_pools_walk_context(4 * 65536 + 3, POOL, 1)
    with pytest.raises(DS.DecodeSelectError,
                       match=r"needs 65537 decode index candidates \(index_kpool=4\).*"
                             r"MAX_DECODE_POOLS_WALK_CANDIDATES=65536 \(max_model_len <= "
                             r"262147\).*decode_select_walk\.py"):
        DSW.check_decode_pools_walk_context(4 * 65537, POOL, 1)


def test_the_start_check_batching_requests_keeps_the_whole_row_bound():
    DSW.check_decode_pools_walk_context(4 * ROW + 3, POOL, 2)
    with pytest.raises(DS.DecodeSelectError,
                       match=r"needs 16385 decode index candidates.*max_num_seqs=2.*"
                             r"MAX_DECODE_POOLS_CANDIDATES=16384.*decode_select\.py.*"
                             r"set max_num_seqs=1"):
        DSW.check_decode_pools_walk_context(4 * (ROW + 1), POOL, 2)


def test_the_start_check_reads_nothing_without_an_indexer():
    DSW.check_decode_pools_walk_context(1 << 40, None, 1)
    DSW.check_decode_pools_walk_context(1 << 40, None, 8)


def _vllm_config(max_model_len, dcp, max_num_seqs):
    return SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_model_len),
                           parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
                           cache_config=SimpleNamespace(block_size=BLOCK),
                           scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs))


@pytest.mark.parametrize("dcp", [4, 8])
def test_the_runner_admits_1m_under_dcp_at_one_sequence(dcp):
    from vllm_neuron.vllm.worker.neuron_model_runner import check_dcp_decode_index_context

    check_dcp_decode_index_context(_vllm_config(1048576, dcp, 1), POOL)


@pytest.mark.parametrize("max_num_seqs", [1, 4])
def test_the_runner_admits_the_served_dcp2_form(max_num_seqs):
    from vllm_neuron.vllm.worker.neuron_model_runner import check_dcp_decode_index_context

    check_dcp_decode_index_context(_vllm_config(131072, 2, max_num_seqs), POOL)


@pytest.mark.parametrize("dcp, max_num_seqs, per_rank, match", [
    (2, 1, 524288, r"MAX_DECODE_POOLS_WALK_CANDIDATES=65536.*decode_select_walk\.py"),
    (8, 2, 131072, r"MAX_DECODE_POOLS_CANDIDATES=16384.*decode_select\.py"),
])
def test_the_runner_refuses_naming_both_lengths_and_the_bound(dcp, max_num_seqs, per_rank,
                                                               match):
    from vllm_neuron.vllm.worker.neuron_model_runner import check_dcp_decode_index_context

    with pytest.raises(DS.DecodeSelectError,
                       match=rf"decode_context_parallel_size={dcp} leaves each rank "
                             rf"{per_rank} of max_model_len=1048576 tokens.*{match}"):
        check_dcp_decode_index_context(_vllm_config(1048576, dcp, max_num_seqs), POOL)


# ---------------------------------------------------------------------------------------
# The chain at DCP 4 and 8: each rank's walk, then the merge kernel
# ---------------------------------------------------------------------------------------


def _local_columns(scores, rank, cp, local_candidates):
    """The CP = 1 scores of the pools ``rank`` owns, at their local columns; ``BOUND_FILL``
    past the global axis."""
    gids = OWN.pool_global_index(torch.arange(local_candidates), rank, block_size=BLOCK,
                                 dcp_size=cp, index_kpool=POOL)
    out = torch.full((int(scores.shape[0]), local_candidates), BOUND_FILL)
    inside = gids < int(scores.shape[1])
    out[:, inside] = scores[:, gids[inside]]
    return out.contiguous()


@pytest.mark.parametrize("cp, local_candidates, rows, levels", [
    (4, 4 * ROW, 1, None),   # a4 at 1M: four chunks per rank
    (4, 4 * ROW, 4, 5),
    (8, 2 * ROW, 1, 5),      # a8 at 1M: two chunks per rank
    (8, 2 * ROW, 4, None),
])
def test_the_ranks_walks_merge_to_the_cp1_selection(cp, local_candidates, rows, levels):
    width = cp * local_candidates
    lengths = [4 * width - 1 - 3 * t for t in range(rows)]
    if rows > 1:
        lengths[-1] = 4 * (width // 3) + 2        # a row whose later blocks are empty
    bounded = _bounded(lengths, width, seed=cp * 100 + rows, levels=levels)
    lens = torch.tensor(lengths, dtype=torch.int32)
    _t, _v, want = DS.dsa_decode_select_pools_torch_oracle(bounded, lens, select_k=K,
                                                           pool_size=POOL)
    lists = [_walk(_local_columns(bounded, rank, cp, local_candidates)) for rank in range(cp)]
    values = torch.stack([v for v, _ in lists])
    ids = torch.stack([i for _, i in lists])
    union = [set() for _ in lengths]
    for rank in range(cp):
        DCM.reset_dcp_merge_select_dispatch_counters()
        merged = DCM.dsa_dcp_merge_select(values, ids, cp_rank=rank, pools_per_block=PPB)
        assert DCM.dcp_merge_select_dispatch_counters()[1] == 0
        for r, row in enumerate(merged.tolist()):
            mine = torch.tensor([p for p in row if p >= 0], dtype=torch.int64)
            got = set(OWN.pool_global_index(mine, rank, block_size=BLOCK, dcp_size=cp,
                                            index_kpool=POOL).tolist())
            assert not (got & union[r]), (cp, rank, r)
            union[r] |= got
    for r, got in enumerate(union):
        expected = {int(p) for p in want[r].tolist() if p >= 0}
        assert got == expected, (cp, rows, r, sorted(got ^ expected)[:8])


# ---------------------------------------------------------------------------------------
# The route: the indexer's DCP branch
# ---------------------------------------------------------------------------------------


class _OneRankGroup:
    """A DCP group whose all-gather returns this rank's tensor once per rank."""

    def __init__(self, world: int):
        self.world_size = int(world)
        self.rank_in_group = 0

    def all_gather(self, local: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0, dim
        return torch.cat([local] * self.world_size, dim=0)


@functools.lru_cache(maxsize=1)
def _indexer():
    cfg = decode_config(hidden_size=512, q_lora_rank=256)
    ix = model_fp8.Glm5NextDSAIndexer(cfg)
    ape = (torch.randn(int(cfg.index_kpool), int(cfg.index_head_dim),
                       generator=torch.Generator().manual_seed(5)) * 0.1).to(torch.bfloat16)
    ix.index_kpool_compress_ape = torch.nn.Parameter(ape, requires_grad=False)
    return ix


def _route_operands(local_candidates, cp, *, seed):
    """One request whose rank-0 share fills the rank's window of ``local_candidates`` pools:
    the global length ends three tokens short of the group's last block."""
    ix = _indexer()
    heads, dim = int(ix.index_n_heads), int(ix.index_head_dim)
    gen = torch.Generator().manual_seed(seed)
    window = local_candidates * POOL
    length = cp * window - 3
    return {
        "query_rows": torch.randn(heads, dim, generator=gen).to(torch.bfloat16),
        "key": torch.randn(1, dim, generator=gen).to(torch.bfloat16),
        "weights": torch.randn(1, heads, generator=gen) * (dim ** -0.5) * (heads ** -0.5),
        "gate": torch.randn(1, dim, generator=gen).to(torch.bfloat16),
        "pool_bank": (torch.randn(3, local_candidates + 1, dim, generator=gen) * 0.5
                      ).to(torch.bfloat16),
        "tail_bank": (torch.randn(3, 2, POOL, dim, generator=gen) * 0.5).to(torch.bfloat16),
        "slots": torch.tensor([1], dtype=torch.int32),
        "seq_lens": torch.tensor([length], dtype=torch.int32),
        "position": torch.tensor([length - 1], dtype=torch.int64),
        "window": window,
    }


def _dcp_step(ops, cp, monkeypatch):
    ix = _indexer()
    monkeypatch.setattr(ix, "dcp_size", cp)
    monkeypatch.setattr(ix, ix.SHARD_RANK_ATTR, torch.zeros(1, dtype=torch.int32),
                        raising=False)
    monkeypatch.setattr(model_fp8, "_resolve_dcp_kv_group", lambda: (_OneRankGroup(cp), 0))
    query = KH.dsa_hadamard128(ops["query_rows"]).reshape(1, ix.index_n_heads,
                                                          ix.index_head_dim)
    return ix.forward_requests(
        ops["key"].new_zeros((1, 1)), None, ops["pool_bank"].clone(), ops["tail_bank"].clone(),
        ops["slots"], ops["seq_lens"], ops["position"], max_seq_len=ops["window"],
        indices_wanted=True, projected=(query, ops["key"], ops["weights"], ops["gate"]),
        page_size=BLOCK)


@pytest.mark.parametrize("local_candidates, entry", [(ROW + 32, "walk"), (ROW, "pools")])
def test_the_dcp_branch_takes_the_walk_past_one_row(monkeypatch, local_candidates, entry):
    cp = 2
    ops = _route_operands(local_candidates, cp, seed=local_candidates)
    calls = []
    real_walk, real_pools = DSW.dsa_decode_select_pools_walk, DS.dsa_decode_select_pools

    def walk(bounded, **kw):
        calls.append(("walk", tuple(bounded.shape)))
        return real_walk(bounded, **kw)

    def pools(bounded, seq_lens, **kw):
        calls.append(("pools", tuple(bounded.shape)))
        return real_pools(bounded, seq_lens, **kw)

    monkeypatch.setattr(DSW, "dsa_decode_select_pools_walk", walk)
    monkeypatch.setattr(DS, "dsa_decode_select_pools", pools)
    got = _dcp_step(ops, cp, monkeypatch)
    assert calls == [(entry, (1, local_candidates))]
    assert DSW.walk_dispatch_count() == (1 if entry == "walk" else 0)
    assert DB.decode_batch_dispatch_counters()[1] == 0

    # The same step with the oracle's lists in the selection's place: the same tokens.
    def walk_oracle(bounded, **kw):
        return DSW.dsa_decode_select_pools_walk_torch_oracle(bounded, **kw)

    def pools_oracle(bounded, seq_lens, **kw):
        return DS.dsa_decode_select_pools_torch_oracle(bounded, seq_lens, **kw)

    monkeypatch.setattr(DSW, "dsa_decode_select_pools_walk", walk_oracle)
    monkeypatch.setattr(DS, "dsa_decode_select_pools", pools_oracle)
    want = _dcp_step(ops, cp, monkeypatch)
    assert torch.equal(got, want)
    # The one-rank group hands the merge this rank's list once per rank, so the merge keeps
    # k / cp of this rank's pools; the request's last block is on rank 1 (no tail here).
    assert int((got >= 0).sum()) == K * POOL // cp


# ---------------------------------------------------------------------------------------
# The device compile: ordering check, dependency verdicts, SBUF allocation
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
#: ``(R, candidates)``: a8 at 1M (two chunks per rank) and a4 at 1M (four), one row (a
#: decode or draft step) and a verify step's four rows on both programs.
COMPILE_SHAPES = ((1, 2 * ROW), (4, 2 * ROW), (1, 4 * ROW), (4, 4 * ROW))
#: The SB allocator's spill count in the compiler log.
_SB_SPILLS = re.compile(r"SB spills = (\d+) tensors")


def _depcheck(dump: str) -> tuple[str, str]:
    """The dependency checker's one-pass and loop-carried VERDICT lines on one dump, run as
    its command line runs it (``python -m test.kernel_depcheck.depcheck DUMP``)."""
    done = subprocess.run([sys.executable, "-m", "test.kernel_depcheck.depcheck", dump],
                          cwd=str(_ROOT), capture_output=True, text=True, timeout=900,
                          check=False, env={**os.environ, "PYTHONPATH": str(_ROOT),
                                            "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0, (dump, done.stdout[-2000:], done.stderr[-2000:])
    lines = done.stdout.splitlines()
    one_pass = [line for line in lines if line.startswith("VERDICT: ")]
    loop_carried = [line for line in lines if line.startswith("VERDICT (loop-carried): ")]
    assert len(one_pass) == 1 and len(loop_carried) == 1, done.stdout[-2000:]
    return one_pass[0], loop_carried[0]


#: The runtime's two SBUF reservations as the dump names them (``neuron_utils``
#: ``SBUF_DYNAMIC_DMA_SCRATCH_BYTES`` and ``SBUF_EVAL_ACCEL_RESERVED_BYTES``).
_SB_RESERVED = ("DynamicDMAScratchLoc", "EvalAccelReservedLoc")


@dataclasses.dataclass(frozen=True)
class _SbufAllocation:
    """One core's SB allocation, in bytes per partition (a location's ``dims[1]``)."""

    #: Every location's bytes, the compiler's own included: the peak if all were live at once.
    total: int
    #: The addresses at least one location covers (live ranges that do not meet share them).
    covered: int
    #: Locations that meet a reservation (``(addr, end, name)``).
    in_reserved: list[tuple[int, int, str]]
    #: The three widest locations, ``(bytes, addr, name)``.
    widest: list[tuple[int, int, str]]


def _sbuf_allocation(dump: str) -> _SbufAllocation:
    """One core's SB allocation from its ``lower_sync`` dump: every SB memory location but
    the two reservations, each ``addr`` to ``addr + dims[1]``."""
    with open(dump) as f:
        function = json.load(f)["functions"][0]
    sb = [(loc["addr"], loc["addr"] + loc["dims"][1], loc["name"])
          for alloc in function["allocations"] for loc in alloc.get("memorylocations", [])
          if loc["type"] == "SB"]
    reserved = [(start, end) for start, end, name in sb if name in _SB_RESERVED]
    assert len(reserved) == len(_SB_RESERVED), (dump, reserved)
    tiles = sorted(loc for loc in sb if loc[2] not in _SB_RESERVED)
    covered, run_start, run_end = 0, None, None
    for start, end, _name in tiles:
        if run_end is None or start > run_end:
            covered += 0 if run_end is None else run_end - run_start
            run_start, run_end = start, end
        else:
            run_end = max(run_end, end)
    covered += 0 if run_end is None else run_end - run_start
    return _SbufAllocation(
        total=sum(end - start for start, end, _name in tiles), covered=covered,
        in_reserved=[loc for loc in tiles
                     if any(loc[0] < r_end and r_start < loc[1] for r_start, r_end in reserved)],
        widest=sorted(((end - start, start, name) for start, end, name in tiles),
                      reverse=True)[:3])


def _sb_spills(art: str) -> list[int]:
    """The SB allocator's spill counts, one per allocation pass, over the build's logs."""
    return [int(n) for log in sorted(glob.glob(f"{art}/**/log-neuron-cc*.txt", recursive=True))
            for n in _SB_SPILLS.findall(pathlib.Path(log).read_text(errors="replace"))]


def _compile_walk(batch: int, candidates: int):
    from test.vllm_neuron.functional.dsa import sbuf_order

    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    programs = 2 if batch >= 2 else 1
    with tempfile.TemporaryDirectory(prefix="dsa_dcp_walk_") as art:
        environment.update(
            NEURON_PLATFORM_TARGET_OVERRIDE="trn2", NKI_COMPILE_CACHE_URL=f"{art}/nki",
            PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(_ROOT),
            PATH=f"{pathlib.Path(sys.executable).parent}:{environment.get('PATH', '')}")
        done = subprocess.run(
            [*_NO_DEVICE, sys.executable, "-B", str(_HERE), art, str(batch), str(candidates),
             str(programs)],
            cwd=art, env=environment, capture_output=True, text=True, timeout=2400,
            check=False)
        assert done.returncode == 0, (batch, candidates, done.stdout[-2000:],
                                      done.stderr[-3000:])
        assert pathlib.Path(art, "kernel.neff").is_file(), sorted(os.listdir(art))
        dumps = sorted(glob.glob(f"{art}/nc0*/sg00/bir_debug.*after-lower_sync*.json")
                       or glob.glob(f"{art}/sg00/bir_debug.*after-lower_sync*.json"))
        assert len(dumps) == programs, (batch, candidates, dumps)
        out = []
        for dump in dumps:
            out.append((pathlib.Path(dump).relative_to(art).parts[0], _sbuf_allocation(dump),
                        sbuf_order.check_dump(dump), *_depcheck(dump)))
        return out, _sb_spills(art)


@_needs_compiler
@pytest.mark.parametrize("batch, candidates", COMPILE_SHAPES,
                         ids=[f"R{b}-C{c}" for b, c in COMPILE_SHAPES])
def test_the_walk_kernel_compiles_with_every_access_ordered(batch, candidates):
    builds, spills = _compile_walk(batch, candidates)
    label = f"walk R={batch} candidates={candidates}"
    print(f"{label}: SB allocator spills {spills}")
    assert spills and max(spills) == 0, (label, spills)
    for core, sbuf, order, one_pass, loop_carried in builds:
        bound = SBUF_BYTES_PER_PARTITION
        print(f"{label} {core}: SBUF allocation {sbuf.total} B per partition (every location "
              f"live at once; {sbuf.covered} B of addresses covered), bound "
              f"SBUF_BYTES_PER_PARTITION {bound}, max(value - bound) {sbuf.total - bound}; "
              f"locations in a reservation {len(sbuf.in_reserved)}; widest locations "
              f"(B per partition, addr, name) {sbuf.widest}")
        assert not sbuf.in_reserved, (label, core, sbuf.in_reserved[:4])
        assert sbuf.total <= bound, (label, core, sbuf.total, bound)
        print(f"{label} {core}: {order.summary()}")
        print(f"{label} {core}: {one_pass}")
        print(f"{label} {core}: {loop_carried}")
        assert order.clean, (label, core, order.summary(), order.unsync[:4],
                             order.library_alias[:4], order.stepped_dma[:4],
                             order.early_waits[:4])
        assert one_pass.startswith("VERDICT: CLEAN"), (label, core, one_pass)
        assert loop_carried.startswith("VERDICT (loop-carried): CLEAN"), (label, core,
                                                                         loop_carried)


def _compile_child(art: str, batch: int, candidates: int, programs: int):
    """Child process: compile the walk kernel to a NEFF with the ``lower_sync`` dump in
    ``art``, on the arguments its host side passes (``dsa_decode_select_pools_walk``)."""
    import inspect
    from dataclasses import replace

    import nki.language as nl
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import compile_bir_to_neff
    from nki.framework.compiled import CompileKernel
    from nki.language.buffers import shared_hbm
    from nki.language.tensor import NkiTensor

    import vllm_neuron

    print(f"module={pathlib.Path(vllm_neuron.__file__).resolve()}", flush=True)
    func = DSW.dsa_decode_select_pools_walk_kernel.func
    arguments = {"bounded_hbm": ((batch, candidates), nl.float32),
                 "select_k": K,
                 "fold_cap": DS.select_fold_cap(DS.mask_span(candidates)),
                 "source_digest": DSW.SOURCE_DIGEST}
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
