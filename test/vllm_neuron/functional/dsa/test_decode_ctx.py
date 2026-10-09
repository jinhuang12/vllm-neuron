# SPDX-License-Identifier: Apache-2.0
"""The DSA decode indexer chain at any context length: no candidate ceiling, both cores.

e3f38f8 served the decode indexer's device kernels up to a candidate axis of 16384 pools
(``ctx`` 65536 at ``index_kpool`` 4) and no further: ``dsa_decode_scores`` refused a wider
axis by name (``candidates above 16384 are not served``) and ``dsa_decode_select`` sent it
to its torch oracle, a sort the trn2 compiler refuses. The model's own ceiling is
``max_position_embeddings`` (1048576 tokens, 262144 pools). The selection keeps
``index_topk`` = 2048 tokens; that is the model's top-k and stays.

What this file pins, per stage of the chain:

* **scores** -- any candidate axis, on the kernel, equal to the torch oracle to fp32
  rounding (the bounded columns exactly); at one request both cores split the candidates
  and give the one-program result bit for bit.
* **ring step** -- at one request both cores share the work and give the one-program
  result bit for bit.
* **selection** -- any candidate axis, on the kernel, equal to the torch oracle bit for
  bit (set, order, tie rule, layout), including axes whose compaction runs in several
  segments and merges them, ties that straddle a segment boundary, and rows with fewer
  than ``k`` complete pools.
* **the chain** -- ``forward_requests`` at contexts past e3f38f8's ceiling selects
  exactly what the torch oracle selects from the kernel's own scores, with no torch
  fallback; at e3f38f8's served contexts it selects e3f38f8's sets.
* **the call site's candidate axis** -- ``Glm5NextMLAAttention._forward_requests`` hands
  the indexer the graph's bound (``min(max_model_len, window)``, the decode bucket)
  rather than ``max_model_len``, and the chain selects the same sets on that narrower
  axis as on the ``max_model_len`` one for every row the bucket holds, the row whose
  length is the bucket included.
"""

from __future__ import annotations

import functools

import pytest
import torch

from test.hardware.baselines.dsa_capped_select import chain as chain_base
from test.hardware.baselines.dsa_capped_select import load_or_skip as load_base
from test.vllm_neuron.functional.dsa.dsa_decode_case import decode_config
from vllm_neuron import envs
from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.functional.dsa import decode_select as DS
from vllm_neuron.functional.dsa import kpool_hadamard as KH
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL
from vllm_neuron.functional.dsa.decode_bypass import bypass_max_context
from vllm_neuron.functional.dsa.index_expand import index_expand_width
from vllm_neuron.utils.neuron_utils import can_run_kernel

_CFG = decode_config()
POOL = int(_CFG.index_kpool)
#: Pools the selection keeps: the model's top-k tokens in whole pools.
K = int(_CFG.index_topk) // POOL
HEAD_DIM = int(_CFG.index_head_dim)
#: The base revision's widest served candidate axis (its ``decode_batch.MAX_CANDIDATES``,
#: pinned by ``test_the_base_ceiling_is_the_widest_whole_row``), which is also the widest
#: row the selection compacts whole.
BASE_CEILING = DS.WHOLE_ROW_COLUMNS
#: fp32 rounding of a 128-term dot product and a 32-term head sum in two orders.
SCORE_RTOL = 1e-5
SCORE_ATOL = 1e-5


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    DB.reset_decode_batch_dispatch_counters()


def _bf16(gen, *shape, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(torch.bfloat16)


def _lens(lengths):
    return torch.tensor(lengths, dtype=torch.int32)


def _selected_sets(indices: torch.Tensor) -> list[set]:
    return [set(int(v) for v in row.tolist() if v >= 0) for row in indices]


# ---------------------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------------------


def _score_case(batch, seed, *, candidates, lengths, heads=32, slots_total=None):
    gen = torch.Generator().manual_seed(seed)
    slots_total = slots_total or batch + 2
    bank = _bf16(gen, slots_total, candidates + 1, HEAD_DIM, scale=0.5)
    slots = torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32)
    seq_lens = torch.tensor(lengths, dtype=torch.int32)
    position = seq_lens - 1
    query = _bf16(gen, batch, heads, HEAD_DIM)
    weights = torch.randn(batch, heads, generator=gen) * heads ** -0.5
    pooled = _bf16(gen, batch, HEAD_DIM, scale=0.5)
    return query, weights, bank, slots, seq_lens, position, pooled


def _assert_scores_agree(got, want):
    filled = want == BOUND_FILL
    assert torch.equal(got == BOUND_FILL, filled)
    torch.testing.assert_close(got[~filled], want[~filled], rtol=SCORE_RTOL, atol=SCORE_ATOL)


@pytest.mark.parametrize("batch, candidates, lengths", [
    # One pool past e3f38f8's ceiling (ctx 65540); this step closes pool 16384.
    (1, BASE_CEILING + 1, [4 * (BASE_CEILING + 1)]),
    # Two requests, a ragged last tile, one row far shorter than the axis.
    (2, 20000, [80000, 9001]),
])
def test_scores_serve_a_candidate_axis_past_the_base_ceiling(batch, candidates, lengths):
    case = _score_case(batch, seed=candidates + batch, candidates=candidates,
                       lengths=lengths)
    got = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert DB.decode_batch_dispatch_counters() == (1, 0)
    assert got.shape == (batch, candidates) and got.dtype == torch.float32
    _assert_scores_agree(got, DB.dsa_decode_scores_torch_oracle(
        *case, candidates=candidates, pool_size=POOL))


@pytest.mark.parametrize("candidates, length", [(2048, 8191), (8192, 32767), (300, 1200)])
def test_scores_of_one_request_split_over_both_cores_equal_one_program(
        monkeypatch, candidates, length):
    """At ``B = 1`` both cores of an LNC2 core score half the candidate blocks each."""
    case = _score_case(1, seed=candidates, candidates=candidates, lengths=[length])
    one = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    DB.reset_decode_batch_dispatch_counters()
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    two = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert DB.decode_batch_route_counts() == (0, 1, 1, 0)
    assert torch.equal(one, two)


def _ported_lengths(candidates: int) -> list[int]:
    """origin/wt3/uncap 453a5eb's lengths: the whole axis, a pool closed mid-axis, a short
    one, a ragged tail. A multiple of the pool closes a pool this step, so the stand-in
    lands at column ``length / 4 - 1`` -- past the first block for the long requests."""
    full = candidates * POOL
    return [full, (candidates // 2 + 7) * POOL, 9, full - 2]


def _lnc(monkeypatch, lnc):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)


@pytest.mark.parametrize("lnc", [None, "2"])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("candidates", [512, 2048, BASE_CEILING])
def test_scores_equal_the_base_bit_for_bit_where_it_served(monkeypatch, batch, candidates,
                                                          lnc):
    """453a5eb's equality claim, against e3f38f8's kernel: blocks score each tile with the
    same instructions, on one program or two (B = 1: two halves of the blocks)."""
    _lnc(monkeypatch, lnc)
    base = load_base().decode_batch
    case = _score_case(batch, seed=candidates + batch, candidates=candidates,
                       lengths=_ported_lengths(candidates)[:batch], slots_total=5)
    want = base.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    got = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert DB.decode_batch_dispatch_counters() == (1, 0)
    tiles = -(-candidates // DB.PARTITIONS)
    assert DB.decode_batch_route_counts()[2] == int(lnc == "2" and (batch >= 2 or tiles >= 2))
    assert torch.equal(got, want)


@pytest.mark.parametrize("lnc", [None, "2"])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("candidates", [BASE_CEILING + 1, 2 * BASE_CEILING,
                                        4 * BASE_CEILING])
def test_scores_past_the_ceiling_match_the_oracle_and_the_stand_in(monkeypatch, batch,
                                                                    candidates, lnc):
    """453a5eb's oracle claim at 16385, 32768 and 65536 candidates (the inline walk, the
    device loop, both), and this step's pool at its own column past the first block."""
    _lnc(monkeypatch, lnc)
    case = _score_case(batch, seed=candidates + 3 * batch, candidates=candidates,
                       lengths=_ported_lengths(candidates)[:batch], slots_total=5)
    got = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert DB.decode_batch_dispatch_counters() == (1, 0)
    _assert_scores_agree(got, DB.dsa_decode_scores_torch_oracle(
        *case, candidates=candidates, pool_size=POOL))
    query, weights, _bank, _slots, seq_lens, _position, pooled = case
    for b in range(batch):
        if int(seq_lens[b]) % POOL:
            continue
        column = int(seq_lens[b]) // POOL - 1
        own = (query[b].float() @ pooled[b].float()).clamp(min=0.0)
        torch.testing.assert_close(got[b, column], (own * weights[b].float()).sum(),
                                   rtol=SCORE_RTOL, atol=SCORE_ATOL)


@pytest.mark.parametrize("batch, candidates, lengths, programs, one_slot, unroll", [
    # Five uniform blocks (one group of four in the loop, one inline) and a one-row tail
    # block; this step closes pool 20480, in the tail.
    (1, 20481, [4 * 20481], 1, False, None),
    # Two requests on two programs, nine uniform blocks (two groups, one inline), a 25-tile
    # tail with a ragged tile.
    (2, 40000, [160000, 9001], 2, False, None),
    # One request on both cores, eight uniform blocks each (two groups; ctx 262144).
    (1, 65536, [262144], 2, False, None),
    # The one-request carrier's one-slot bank (static slot): one group, two inline, a
    # ragged tail.
    (1, 24700, [98800], 1, True, None),
    # Groups of two: four requests on two programs, two requests in each loop body, five
    # uniform blocks (two groups, one inline) and a ragged tail.
    (4, 5 * 4096 + 300, [4 * (5 * 4096 + 300), 4 * 9000, 4 * 4096, 4 * 20500 - 2], 2, False,
     2),
])
def test_the_device_loop_over_score_blocks_equals_the_inline_walk(
        monkeypatch, batch, candidates, lengths, programs, one_slot, unroll):
    """Past ``UNROLL_BLOCKS`` uniform blocks a program walks whole groups of them in one
    device loop over every request (registers place each block); it scores exactly what
    the inline walk scores."""
    case = _score_case(batch, seed=candidates + 3 * batch, candidates=candidates,
                       lengths=lengths, slots_total=1 if one_slot else None)
    if programs == 2:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    if unroll is not None:
        monkeypatch.setattr(DB, "UNROLL_BLOCKS", unroll)
    tiles = -(-candidates // DB.PARTITIONS)
    block = DB.score_blocks(batch, candidates, programs)
    per_program = (tiles // block) // (programs if batch < programs else 1)
    assert per_program > DB.UNROLL_BLOCKS, "the case must take the loop"
    looped = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    monkeypatch.setattr(DB, "UNROLL_BLOCKS", 1 << 20)
    inline = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert DB.decode_batch_dispatch_counters() == (2, 0)
    assert torch.equal(looped, inline)
    _assert_scores_agree(looped, DB.dsa_decode_scores_torch_oracle(
        *case, candidates=candidates, pool_size=POOL))


def test_every_axis_the_base_served_scores_inline():
    """``UNROLL_BLOCKS`` keeps e3f38f8's widest axis (16384 candidates, 128 tiles) inline
    on one program, so the device loop runs only past e3f38f8's ceiling."""
    block = DB.score_blocks(4, BASE_CEILING, 2)
    assert (-(-BASE_CEILING // DB.PARTITIONS)) // block <= DB.UNROLL_BLOCKS


@pytest.mark.parametrize("batch, candidates, unroll, lnc", [
    (4, BASE_CEILING, 1, "2"),     # groups of one, two requests in each loop body
    (4, BASE_CEILING, 2, None),    # groups of two, four requests in each loop body
    (1, BASE_CEILING, 0, None),    # one request, every block in the loop
    (4, 12288 + 100, 2, "2"),         # one group, one uniform block inline, a ragged tail
])
def test_the_device_loop_scores_the_base_bit_for_bit_where_it_served(
        monkeypatch, batch, candidates, unroll, lnc):
    """The loop's blocks score each tile with e3f38f8's instructions too: forced at axes
    e3f38f8 served, it gives e3f38f8's scores bit for bit."""
    _lnc(monkeypatch, lnc)
    base = load_base().decode_batch
    case = _score_case(batch, seed=candidates + 5 * batch, candidates=candidates,
                       lengths=_ported_lengths(candidates)[:batch], slots_total=5)
    want = base.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    monkeypatch.setattr(DB, "UNROLL_BLOCKS", unroll)
    got = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert torch.equal(got, want)


@pytest.mark.parametrize("batch, candidates, lengths", [
    (4, 2048, [8192, 8000, 4001, 2100]),   # one uniform block per request
    (1, 300, [1200]),                      # no uniform block: the tail alone
])
def test_the_device_loop_from_the_first_block_equals_the_inline_walk(
        monkeypatch, batch, candidates, lengths):
    """``UNROLL_BLOCKS = 0`` takes the loop at every axis with a uniform block."""
    case = _score_case(batch, seed=candidates + batch, candidates=candidates,
                       lengths=lengths)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    inline = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    monkeypatch.setattr(DB, "UNROLL_BLOCKS", 0)
    looped = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert torch.equal(looped, inline)


def test_ring_step_of_one_request_splits_over_both_cores_and_equals_one(monkeypatch):
    gen = torch.Generator().manual_seed(9)
    bank = _bf16(gen, 3, 2, POOL, HEAD_DIM)
    slots = torch.tensor([2], dtype=torch.int32)
    key, score = _bf16(gen, 1, HEAD_DIM), _bf16(gen, 1, HEAD_DIM)
    ape = torch.randn(POOL, HEAD_DIM, generator=gen)
    for pos in (4095, 4093):  # closes a pool, and does not
        position = torch.tensor([pos], dtype=torch.int32)
        DB.reset_decode_batch_dispatch_counters()
        one = DB.dsa_decode_ring_step(bank, slots, key, score, ape, position)
        DB.reset_decode_batch_dispatch_counters()
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
        two = DB.dsa_decode_ring_step(bank, slots, key, score, ape, position)
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG")
        assert DB.decode_batch_route_counts() == (1, 0, 1, 0)
        assert torch.equal(one[1], two[1])
        if pos % POOL == POOL - 1:
            assert torch.equal(one[0], two[0])


# ---------------------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------------------


def _bounded(lengths, candidates, *, seed, levels=None):
    gen = torch.Generator().manual_seed(seed)
    scores = torch.randn(len(lengths), candidates, generator=gen)
    if levels is not None:
        scores = torch.randint(0, levels, (len(lengths), candidates), generator=gen).float()
    cols = torch.arange(candidates)[None, :]
    complete = torch.tensor(lengths)[:, None] // POOL
    return scores.masked_fill(cols >= complete, BOUND_FILL).contiguous()


def _select(bounded, lengths):
    got = DS.dsa_decode_select(bounded, _lens(lengths), select_k=K, pool_size=POOL)
    assert (DB.decode_batch_route_counts()[3], DB.decode_batch_dispatch_counters()[1]) \
        == (1, 0), "the selection must run on the kernel, not the torch oracle"
    return got


def _oracle(bounded, lengths):
    return DS.dsa_decode_select_torch_oracle(bounded, _lens(lengths), select_k=K,
                                             pool_size=POOL)


@pytest.mark.parametrize("lengths, candidates", [
    ([4 * (BASE_CEILING + 1)], BASE_CEILING + 1),
    ([131071, 70000, 4096 * 4 - 1], 32768),
    ([4 * 65536 - 2], 65536),
    # Nine segments: one more than a merge takes, so a merged list takes the first slot.
    ([4 * 65537 - 1, 4 * 40000], 65537),
    # max_position_embeddings: 1048576 tokens, 262144 pools.
    ([1048575], 262144),
    ([4 * 20000 - 1 - 3 * b for b in range(40)], 20000),
])
def test_selection_serves_a_candidate_axis_past_the_base_ceiling(lengths, candidates):
    bounded = _bounded(lengths, candidates, seed=len(lengths) + candidates)
    got = _select(bounded, lengths)
    assert got.dtype == torch.int32
    assert tuple(got.shape) == (len(lengths), index_expand_width(K, POOL))
    assert torch.equal(got, _oracle(bounded, lengths))


@pytest.mark.parametrize("levels", [2, 9])
def test_selection_ties_across_segments_take_the_lowest_index(levels):
    """With scores on a few levels the ``k``-th value ties across the whole row, so the
    tie quota spans every chunk and every compaction segment."""
    lengths = [4 * 40000 - 1, 4 * 33000, 4 * 21000 + 2]
    bounded = _bounded(lengths, 40000, seed=levels, levels=levels)
    assert torch.equal(_select(bounded, lengths), _oracle(bounded, lengths))


def test_selection_rows_with_fewer_pools_than_k_on_a_wide_axis():
    lengths = [300, 2047, 2049, 4 * 30000 - 1]
    bounded = _bounded(lengths, 30000, seed=3)
    got = _select(bounded, lengths)
    assert torch.equal(got, _oracle(bounded, lengths))
    for row, n in zip(_selected_sets(got), lengths):
        assert len(row) == min(n // POOL, K) * POOL + n % POOL


def test_selected_pools_concentrated_in_one_segment():
    """Every one of the ``k`` largest scores in the last segment: one segment list holds
    all ``k`` and the merge takes nothing from the others."""
    lengths = [4 * 50000 - 1, 4 * 50000 - 1]
    bounded = _bounded(lengths, 50000, seed=5)
    bounded[0, -K:] += 100.0
    bounded[1, :K] += 100.0
    assert torch.equal(_select(bounded, lengths), _oracle(bounded, lengths))


def test_the_rows_the_base_served_are_compacted_whole():
    """Up to e3f38f8's widest axis (16384 candidates) a mask row is compacted whole, as
    e3f38f8 did: two segments and a merge cost 663 -> 873 us per layer at B = 64, ctx
    65536 on the device (worker-47's J4). Past it the row is cut in segments."""
    assert DS.segments(BASE_CEILING) == (1, BASE_CEILING)
    count, width = DS.segments(BASE_CEILING + 1)
    assert count > 1 and width <= DS.SEGMENT_COLUMNS


@pytest.mark.parametrize("batch, candidates", [(1, 4096), (1, BASE_CEILING),
                                               (4, BASE_CEILING)])
def test_selection_equals_the_base_where_it_served(monkeypatch, batch, candidates):
    """Bit for bit e3f38f8's selection kernel (its whole-row compaction) on e3f38f8's
    served axes, on two programs at ``B >= 2``."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    lengths = [4 * candidates - 1 - 3001 * b for b in range(batch)]
    bounded = _bounded(lengths, candidates, seed=7 * batch + candidates)
    want = load_base().decode_select.dsa_decode_select(bounded, _lens(lengths),
                                                          select_k=K, pool_size=POOL)
    assert torch.equal(_select(bounded, lengths), want)


def test_the_fold_cap_is_small_on_a_short_axis_and_every_partition_past_it():
    """The fold cap the device measured fastest: :data:`SMALL_AXIS_FOLD` partitions up to
    :data:`SMALL_AXIS_COLUMNS` candidates, every partition past it."""
    short = DS.SMALL_AXIS_COLUMNS
    assert [DS.select_fold_cap(w) for w in (K + 1, short)] == [DS.SMALL_AXIS_FOLD] * 2
    wide = (short + 1, 2 * short, DS.WHOLE_ROW_COLUMNS, DS.MAX_SELECT_CANDIDATES)
    assert [DS.select_fold_cap(w) for w in wide] == [DB.PARTITIONS] * len(wide)


def test_the_base_ceiling_is_the_widest_whole_row():
    assert load_base().decode_batch.MAX_CANDIDATES == BASE_CEILING


@pytest.mark.parametrize("cap", [16, 32, 64, 128])
@pytest.mark.parametrize("batch, candidates", [(1, 2048), (1, 20000), (4, 8192)])
def test_every_fold_cap_selects_the_oracle_indices(monkeypatch, batch, candidates, cap):
    """How many partitions a request's scores spread over (``select_fold_cap``) is a cost
    choice: every cap selects the oracle's indices bit for bit, on one core or two."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    monkeypatch.setattr(DS, "select_fold_cap", lambda width: cap)
    lengths = [4 * candidates - 2 - 4001 * b for b in range(batch)]
    bounded = _bounded(lengths, candidates, seed=cap + batch * candidates)
    assert torch.equal(_select(bounded, lengths), _oracle(bounded, lengths))


@pytest.mark.parametrize("batch, candidates", [(2, 2048), (3, 20000), (1, 20000)])
def test_selection_two_programs_equal_one_on_a_wide_axis(monkeypatch, batch, candidates):
    lengths = [4 * candidates - 1 - 5 * b for b in range(batch)]
    bounded = _bounded(lengths, candidates, seed=batch * candidates)
    one = _select(bounded, lengths)
    DB.reset_decode_batch_dispatch_counters()
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    two = _select(bounded, lengths)
    assert torch.equal(one, two)


# ---------------------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------------------


def _chain_operands(cfg, ctx: int, batch: int, seed: int) -> dict:
    """One layer's decode operands at the runner's bank shape: ``ctx // pool + 1`` rows
    per slot (``_glm5next_side_caches``), lengths ``ctx - b % 4``."""
    gen = torch.Generator().manual_seed(seed)
    heads, dim, pool = int(cfg.index_n_heads), int(cfg.index_head_dim), int(cfg.index_kpool)
    lengths = [ctx - b % 4 for b in range(batch)]
    slots_total = batch + 2
    rows = ctx // pool + 1
    return {
        "query_rows": torch.randn(batch * heads, dim, generator=gen).to(torch.bfloat16),
        "key": torch.randn(batch, dim, generator=gen).to(torch.bfloat16),
        "weights": torch.randn(batch, heads, generator=gen) * (dim ** -0.5) * (heads ** -0.5),
        "gate": torch.randn(batch, dim, generator=gen).to(torch.bfloat16),
        "pool_bank": (torch.randn(slots_total, rows, dim, generator=gen) * 0.5
                      ).to(torch.bfloat16),
        "tail_bank": (torch.randn(slots_total, 2, pool, dim, generator=gen) * 0.5
                      ).to(torch.bfloat16),
        "slots": torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32),
        "seq_lens": torch.tensor(lengths, dtype=torch.int32),
        "position": torch.tensor([n - 1 for n in lengths], dtype=torch.int64),
    }


@functools.lru_cache(maxsize=None)
def _indexers():
    import vllm_neuron.model.glm5_next.model_fp8 as live_model
    cfg = decode_config(hidden_size=512, q_lora_rank=256)
    ape = (torch.randn(int(cfg.index_kpool), int(cfg.index_head_dim),
                       generator=torch.Generator().manual_seed(5)) * 0.1).to(torch.bfloat16)
    out = []
    for module in (load_base().model_fp8, live_model):
        ix = module.Glm5NextDSAIndexer(cfg)
        ix.index_kpool_compress_ape = torch.nn.Parameter(ape.clone(), requires_grad=False)
        out.append(ix)
    return cfg, out[0], out[1]


def _copy(ops):
    return {n: (v.clone() if torch.is_tensor(v) else v) for n, v in ops.items()}


def _live_chain(indexer, ops, ctx):
    batch = int(ops["key"].shape[0])
    query = KH.dsa_hadamard128(ops["query_rows"]).reshape(
        batch, indexer.index_n_heads, indexer.index_head_dim)
    return indexer.forward_requests(
        ops["key"].new_zeros((batch, 1)), None, ops["pool_bank"], ops["tail_bank"],
        ops["slots"], ops["seq_lens"], ops["position"], max_seq_len=int(ctx),
        indices_wanted=True, projected=(query, ops["key"], ops["weights"], ops["gate"]))


def _oracle_chain_from_kernel_scores(indexer, ops, ctx):
    """The torch selection on the scores the kernels compute: the reference set."""
    batch = int(ops["key"].shape[0])
    query = KH.dsa_hadamard128(ops["query_rows"]).reshape(
        batch, indexer.index_n_heads, indexer.index_head_dim)
    pooled, _rings = DB.dsa_decode_ring_step(
        ops["tail_bank"], ops["slots"], ops["key"], ops["gate"],
        indexer.index_kpool_compress_ape.to(torch.float32), ops["position"])
    bounded = DB.dsa_decode_scores(
        query, ops["weights"], ops["pool_bank"], ops["slots"], ops["seq_lens"],
        ops["position"], pooled, candidates=int(ctx) // POOL, pool_size=POOL)
    return DS.dsa_decode_select_torch_oracle(bounded, ops["seq_lens"], select_k=K,
                                             pool_size=POOL)


@pytest.mark.parametrize("batch, ctx", [
    (1, 4 * (BASE_CEILING + 1)),  # the shape e3f38f8 refused
    (1, 131072),
    (3, 4 * 20000),
])
def test_the_chain_past_the_base_ceiling_selects_the_oracle_sets(batch, ctx):
    cfg, _before, after = _indexers()
    ops = _chain_operands(cfg, ctx, batch, seed=ctx + batch)
    want = _oracle_chain_from_kernel_scores(after, _copy(ops), ctx)
    DB.reset_decode_batch_dispatch_counters()
    got = _live_chain(after, ops, ctx)
    # ring, scores, selection: three kernel launches and no torch route.
    assert DB.decode_batch_dispatch_counters() == (3, 0)
    assert torch.equal(got, want)


@pytest.mark.parametrize("batch, ctx", [(1, 16384), (4, 32768), (64, 8192)])
def test_the_chain_selects_the_base_sets_where_the_base_served(batch, ctx):
    cfg, before, after = _indexers()
    ops = _chain_operands(cfg, ctx, batch, seed=1000 * batch + ctx)
    mine, theirs = _copy(ops), _copy(ops)
    base = load_base()
    want = chain_base(base, before, theirs["query_rows"], theirs["key"], theirs["weights"],
                         theirs["gate"], theirs["pool_bank"], theirs["tail_bank"],
                         theirs["slots"], theirs["seq_lens"], theirs["position"],
                         max_seq_len=ctx)
    got = _live_chain(after, mine, ctx)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert _selected_sets(got) == _selected_sets(want)
    # The layout is a function of the set (ascending pool order, -1, the tail), so the
    # attention reads the very tensor e3f38f8 handed it.
    assert torch.equal(got, want)
    assert torch.equal(mine["pool_bank"], theirs["pool_bank"])
    assert torch.equal(mine["tail_bank"], theirs["tail_bank"])


# ---------------------------------------------------------------------------------------
# the call site's candidate axis: the decode bucket, not max_model_len
# ---------------------------------------------------------------------------------------

#: The serving point of the acceptance list: buckets [2048, 8192, 32768] at 32768.
MAX_MODEL_LEN = 32768


def _with_lengths(ops, lengths):
    ops["seq_lens"] = torch.tensor(lengths, dtype=torch.int32)
    ops["position"] = torch.tensor([n - 1 for n in lengths], dtype=torch.int64)
    return ops


@pytest.mark.parametrize("bucket", [8192, 16384])
def test_a_bucket_wide_candidate_axis_selects_the_max_model_len_sets(bucket):
    """Pools past ``bucket // 4`` are complete for no row of length ``<= bucket``, so
    scoring and selecting over ``bucket // 4`` candidates instead of ``max_model_len //
    4`` changes nothing: the boundary row (length = bucket, which closes the bucket's
    last pool this step), a row one short of it, one inside its last pool, and a short
    row."""
    cfg, _before, after = _indexers()
    lengths = [bucket, bucket - 1, bucket - 6, 3001]
    ops = _with_lengths(_chain_operands(cfg, MAX_MODEL_LEN, len(lengths), seed=bucket),
                        lengths)
    wide, narrow = _copy(ops), _copy(ops)
    want = _live_chain(after, wide, MAX_MODEL_LEN)
    DB.reset_decode_batch_dispatch_counters()
    got = _live_chain(after, narrow, bucket)
    assert DB.decode_batch_dispatch_counters() == (3, 0)
    assert torch.equal(torch.sort(got, dim=-1).values, torch.sort(want, dim=-1).values)
    assert torch.equal(got, want)
    assert torch.equal(narrow["pool_bank"], wide["pool_bank"])
    assert torch.equal(narrow["tail_bank"], wide["tail_bank"])


class _Handed(Exception):
    """Raised by the spy once it has the call site's arguments: nothing past it runs."""


@functools.lru_cache(maxsize=1)
def _attention():
    import vllm_neuron.model.glm5_next.model_fp8 as live_model
    from test.vllm_neuron.functional.dsa.dsa_decode_case import build_attention
    return build_attention(live_model, decode_config(hidden_size=512, q_lora_rank=256))


@pytest.mark.parametrize("window, handed", [
    (8192, 8192),             # a decode bucket below max_model_len: the bucket
    (MAX_MODEL_LEN, MAX_MODEL_LEN),   # the max_model_len fallback bucket
    (2048, 2048),             # inside the bypass: the bound, as e3f38f8 already did
])
def test_the_decode_call_site_hands_the_indexer_its_window_bound(monkeypatch, window,
                                                                  handed):
    from test.vllm_neuron.functional.dsa.dsa_decode_case import PAGE
    module = _attention()
    cfg = decode_config(hidden_size=512, q_lora_rank=256)
    gen = torch.Generator().manual_seed(window)
    lengths = [window - 1, min(window - 1, 1500)]
    batch = len(lengths)
    rows = MAX_MODEL_LEN // POOL + 1
    pool_bank = torch.zeros(batch, rows, HEAD_DIM, dtype=torch.bfloat16)
    tail_bank = torch.zeros(batch, 2, POOL, HEAD_DIM, dtype=torch.bfloat16)
    position = torch.tensor([n - 1 for n in lengths], dtype=torch.int64)
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        raise _Handed

    monkeypatch.setattr(module.indexer, "forward_requests", spy)
    with pytest.raises(_Handed):
        module.forward(
            (torch.randn(batch, int(cfg.hidden_size), generator=gen) * 0.5
             ).to(torch.bfloat16),
            latent_cache=torch.zeros(PAGE, 1, int(cfg.kv_lora_rank), dtype=torch.bfloat16),
            pool_cache=tuple(pool_bank[b] for b in range(batch)),
            seq_lens=torch.tensor(lengths, dtype=torch.int32),
            start_position=position.clone(),
            softmax_scale=1.0,
            max_seq_len=MAX_MODEL_LEN,
            page_size=PAGE,
            block_table_row=torch.full((window // PAGE, batch), -1, dtype=torch.int32),
            latent_slots=torch.zeros(batch, dtype=torch.int64),
            tail=tuple(tail_bank[b] for b in range(batch)),
            position=position,
        )
    assert seen["max_seq_len"] == handed
    assert seen["indices_wanted"] == (handed > bypass_max_context(cfg.index_topk,
                                                                   cfg.index_kpool))


# ---------------------------------------------------------------------------------------
# the launch grid: both cores only under an LNC2 launch
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("programs_of", [
    lambda work: DB._programs(work, 1),
    lambda work: DB._programs(1, work),
    DS.decode_select_programs,
    KH.hadamard128_programs,
], ids=["ring-and-scores-by-requests", "ring-and-scores-by-units", "selection", "rotation"])
@pytest.mark.parametrize("lnc", [None, 1, 2])
def test_the_launch_grid_follows_the_logical_core_config(monkeypatch, programs_of, lnc):
    # The entry, not a module attribute: ``envs`` resolves names lazily, so an attribute
    # set here would outlive the test and shadow the variable for every later test.
    monkeypatch.setitem(envs.environment_variables, "NEURON_LOGICAL_NC_CONFIG", lambda: lnc)
    assert programs_of(1) == 1
    assert programs_of(2) == (2 if lnc == 2 else 1)
