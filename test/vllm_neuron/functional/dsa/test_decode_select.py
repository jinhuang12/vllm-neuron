# SPDX-License-Identifier: Apache-2.0
"""``dsa_decode_select``: the decode selection stage in one kernel, against 0a08ff4.

0a08ff4's ``forward_requests`` selects with four kernels: the rotational top-k, the
causal sentinel, the sentinel order and the index expand. This tree runs one kernel
there. What it must keep, per request row:

* The **set** of selected token indices. Pinned three ways: against the torch oracle
  bit for bit (which also pins the order and the tie rule), against 0a08ff4's four
  kernels on the same bounded scores, and through the whole decode indexer chain --
  the query rotation and ``forward_requests`` -- against 0a08ff4's chain at
  ``B in {1, 4, 64}`` and ``ctx in {2051, 8192}``.
* The **tie rule**. Where values equal the ``k``-th largest value, the lowest candidate
  index wins. 0a08ff4 pinned nothing there (``_canonical_sentinel_order``'s docstring
  says the selector "pins nothing about the order among equal values"), so the sets can
  differ only on a tie at the cut; no seed below has one, and the tests with crafted
  ties compare against the oracle, which spells the rule as a stable sort.
* The **layout**: ``[B, index_expand_width(k, pool)]`` int32, real tokens first,
  ``-1`` after them, the tail tokens at columns ``[k * pool, k * pool + pool - 1)``.
  The one change: real pools are in ascending pool order, where 0a08ff4 had them in
  descending score order. The order among real tokens is not part of the contract
  (``mla_decode`` masks ``-1`` anywhere and sums the selected rows), the set is.
* The **bypass**: at ``ctx <= 2051`` there is nothing to select, the chain emits the
  causal fill and dispatches no selection.
"""

from __future__ import annotations

import functools

import pytest
import torch

from test.hardware.baselines.dsa_four_kernel_select import chain as chain_0a08ff4
from test.hardware.baselines.dsa_four_kernel_select import load as load_0a08ff4
from test.vllm_neuron.functional.dsa.dsa_decode_case import decode_config
from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.functional.dsa import decode_select as DS
from vllm_neuron.functional.dsa import kpool_hadamard as KH
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL
from vllm_neuron.functional.dsa.index_expand import index_expand_width
from vllm_neuron.utils.neuron_utils import can_run_kernel

K = 512
POOL = 4


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    DB.reset_decode_batch_dispatch_counters()


def _select_counts():
    """``(select launches, torch fallbacks)``: the selection counts in ``decode_batch``'s
    family, the batched decode indexer's."""
    return (DB.decode_batch_route_counts()[3], DB.decode_batch_dispatch_counters()[1])


def _bounded(lengths, candidates, *, seed, levels=None):
    """Scores as the score kernel leaves them: ``BOUND_FILL`` past each row's complete
    pools. ``levels`` quantises the scores to that many values, which makes ties."""
    gen = torch.Generator().manual_seed(seed)
    scores = torch.randn(len(lengths), candidates, generator=gen)
    if levels is not None:
        scores = torch.randint(0, levels, (len(lengths), candidates), generator=gen).float()
    cols = torch.arange(candidates)[None, :]
    complete = torch.tensor(lengths)[:, None] // POOL
    return scores.masked_fill(cols >= complete, BOUND_FILL).contiguous()


def _lens(lengths):
    return torch.tensor(lengths, dtype=torch.int32)


def _selected_sets(indices: torch.Tensor) -> list[set]:
    return [set(int(v) for v in row.tolist() if v >= 0) for row in indices]


def _oracle(bounded, lengths):
    return DS.dsa_decode_select_torch_oracle(bounded, _lens(lengths), select_k=K,
                                             pool_size=POOL)


def _kernel(bounded, lengths):
    got = DS.dsa_decode_select(bounded, _lens(lengths), select_k=K, pool_size=POOL)
    assert _select_counts() == (1, 0)
    return got


# ---------------------------------------------------------------------------------------
# The kernel against the oracle: values, order, layout, bit for bit
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("lengths, candidates", [
    ([8190], 2048),
    ([8190, 8000, 2100, 2051], 2048),
    ([8191 - b % 4 for b in range(64)], 2048),
    ([2052, 3000, 5000], 2048),
    ([2100 + 97 * b for b in range(19)], 2048),
    ([2900, 2300], 700),
    ([2400 - 13 * b for b in range(33)], 600),
    ([2053], 513),
])
def test_kernel_equals_the_oracle(lengths, candidates):
    bounded = _bounded(lengths, candidates, seed=len(lengths) + candidates)
    want = _oracle(bounded, lengths)
    got = _kernel(bounded, lengths)
    assert got.dtype == torch.int32
    assert tuple(got.shape) == (len(lengths), index_expand_width(K, POOL))
    assert torch.equal(got, want)


def test_rows_with_fewer_pools_than_k_select_every_pool():
    """A row whose complete pools number fewer than ``k`` selects all of them; the
    ``BOUND_FILL`` columns it must also count to reach ``k`` become ``-1``."""
    lengths = [300, 2047, 2048, 2049, 4100]
    bounded = _bounded(lengths, 2048, seed=3)
    got = _kernel(bounded, lengths)
    assert torch.equal(got, _oracle(bounded, lengths))
    for row, n in zip(_selected_sets(got), lengths):
        pools = min(n // POOL, K)
        assert len(row) == pools * POOL + n % POOL


@pytest.mark.parametrize("levels", [2, 7, 50])
def test_ties_at_the_cut_take_the_lowest_candidate_index(levels):
    """The tie rule. With scores on a few levels nearly every row has a tie at the
    ``k``-th value; the oracle is a stable descending sort, lowest index first."""
    lengths = [8190, 6000, 4096, 2300, 8191, 8189, 7000, 2052]
    bounded = _bounded(lengths, 2048, seed=levels, levels=levels)
    got = _kernel(bounded, lengths)
    assert torch.equal(got, _oracle(bounded, lengths))


def test_signed_zeros_are_one_value():
    """``-0.0`` and ``+0.0`` compare equal, as they do in the score order, so a cut at
    zero takes the lowest-indexed zeros of either sign."""
    gen = torch.Generator().manual_seed(11)
    lengths = [8190, 8190]
    pick = torch.randint(0, 4, (2, 2048), generator=gen)
    values = torch.tensor([-0.0, 0.0, 1.0, -1.0])
    bounded = values[pick].contiguous()
    got = _kernel(bounded, lengths)
    assert torch.equal(got, _oracle(bounded, lengths))


def test_extreme_finite_scores():
    """The bit search walks the whole float range: huge, tiny and negative scores."""
    gen = torch.Generator().manual_seed(12)
    lengths = [8190, 8190, 4000]
    mags = torch.tensor([3.0e38, 1.0e-38, 1.0, 1.0e20, 1.0e-20])
    pick = torch.randint(0, 5, (3, 2048), generator=gen)
    sign = torch.where(torch.rand(3, 2048, generator=gen) < 0.5, -1.0, 1.0)
    bounded = (mags[pick] * sign * torch.rand(3, 2048, generator=gen)).contiguous()
    bounded[2, 1000:] = BOUND_FILL
    got = _kernel(bounded, lengths)
    assert torch.equal(got, _oracle(bounded, lengths))


@pytest.mark.parametrize("batch", [2, 4, 19, 64])
def test_two_programs_split_the_requests_and_equal_one(monkeypatch, batch):
    lengths = [8191 - 37 * b for b in range(batch)]
    bounded = _bounded(lengths, 2048, seed=batch)
    one = _kernel(bounded, lengths)
    DB.reset_decode_batch_dispatch_counters()
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert DS.decode_select_programs(batch) == 2
    two = _kernel(bounded, lengths)
    assert torch.equal(one, two)


def test_one_request_runs_on_one_program(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert DS.decode_select_programs(1) == 1


def test_the_kernel_is_keyed_by_this_files_digest():
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
        lengths = [8000, 3000]
        DS.dsa_decode_select(_bounded(lengths, 2048, seed=1), _lens(lengths), select_k=K,
                             pool_size=POOL)
    finally:
        DS.wrap_nki = real
    assert seen and seen[-1][-1] == DS.SOURCE_DIGEST


def test_refusals():
    lengths = [8000]
    bounded = _bounded(lengths, 2048, seed=1)
    with pytest.raises(DS.DecodeSelectError):
        DS.dsa_decode_select(bounded, _lens(lengths), select_k=2048, pool_size=POOL)
    with pytest.raises(DS.DecodeSelectError):
        DS.dsa_decode_select(bounded, _lens([1, 2]), select_k=K, pool_size=POOL)
    with pytest.raises(DS.DecodeSelectError):
        DS.dsa_decode_select(bounded, _lens(lengths), select_k=K, pool_size=3)
    with pytest.raises(DS.DecodeSelectError):
        DS.dsa_decode_select(bounded[0], _lens(lengths), select_k=K, pool_size=POOL)


# ---------------------------------------------------------------------------------------
# The kernel against 0a08ff4's four selection kernels, on the same bounded scores
# ---------------------------------------------------------------------------------------


def _select_0a08ff4(bounded, lengths):
    base = load_0a08ff4()
    values, ids = base.topk_select.dsa_topk_select(bounded, K)
    sentinel = base.causal_bound.dsa_causal_sentinel(values, ids.to(torch.int32),
                                                     int(bounded.shape[1]))
    ordered = base.sentinel_order.dsa_sentinel_order(sentinel)
    return base.index_expand.dsa_index_expand(ordered, _lens(lengths), POOL)


@pytest.mark.parametrize("batch", [1, 4, 64])
def test_selected_sets_equal_the_0a08ff4_selection(batch):
    lengths = [8191 - (b % 4) - 61 * (b // 4) for b in range(batch)]
    bounded = _bounded(lengths, 2048, seed=100 + batch)
    want = _select_0a08ff4(bounded, lengths)
    got = _kernel(bounded, lengths)
    assert got.shape == want.shape
    assert _selected_sets(got) == _selected_sets(want)
    # Same layout: the real tokens fill the first columns of both, the tail columns agree.
    hist = K * POOL
    assert torch.equal((got[:, :hist] >= 0).sum(1), (want[:, :hist] >= 0).sum(1))
    assert torch.equal(got[:, hist:], want[:, hist:])


# ---------------------------------------------------------------------------------------
# The whole decode indexer chain against 0a08ff4's
# ---------------------------------------------------------------------------------------


def _chain_operands(cfg, ctx: int, batch: int, seed: int) -> dict:
    """One layer's decode operands (the hardware benchmark's), lengths ``ctx - b % 4``."""
    gen = torch.Generator().manual_seed(seed)
    heads, dim, pool = int(cfg.index_n_heads), int(cfg.index_head_dim), int(cfg.index_kpool)
    lengths = [ctx - b % 4 for b in range(batch)]
    slots_total = batch + 3
    rows = -(-(ctx // pool + 1) // 128) * 128
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
    for module in (load_0a08ff4().model_fp8, live_model):
        ix = module.Glm5NextDSAIndexer(cfg)
        ix.index_kpool_compress_ape = torch.nn.Parameter(ape.clone(), requires_grad=False)
        out.append(ix)
    return cfg, out[0], out[1]


def _live_chain(indexer, ops, ctx):
    batch = int(ops["key"].shape[0])
    query = KH.dsa_hadamard128(ops["query_rows"]).reshape(
        batch, indexer.index_n_heads, indexer.index_head_dim)
    return indexer.forward_requests(
        ops["key"].new_zeros((batch, 1)), None, ops["pool_bank"], ops["tail_bank"],
        ops["slots"], ops["seq_lens"], ops["position"], max_seq_len=int(ctx),
        indices_wanted=True, projected=(query, ops["key"], ops["weights"], ops["gate"]))


@pytest.mark.parametrize("ctx", [2051, 8192])
@pytest.mark.parametrize("batch", [1, 4, 64])
def test_the_chain_selects_the_0a08ff4_sets(batch, ctx):
    cfg, before, after = _indexers()
    ops = _chain_operands(cfg, ctx, batch, seed=1000 * batch + ctx)
    mine = {n: (v.clone() if torch.is_tensor(v) else v) for n, v in ops.items()}
    theirs = {n: (v.clone() if torch.is_tensor(v) else v) for n, v in ops.items()}
    want = chain_0a08ff4(load_0a08ff4(), before, theirs["query_rows"], theirs["key"],
                         theirs["weights"], theirs["gate"], theirs["pool_bank"],
                         theirs["tail_bank"], theirs["slots"], theirs["seq_lens"],
                         theirs["position"], max_seq_len=ctx)
    got = _live_chain(after, mine, ctx)
    selects = ctx // int(cfg.index_kpool) > after.select_k()
    assert _select_counts() == ((1, 0) if selects else (0, 0))
    assert got.shape == want.shape and got.dtype == want.dtype
    if selects:
        assert _selected_sets(got) == _selected_sets(want)
    else:
        # The bypass: no selection, the causal fill, column for column.
        assert torch.equal(got, want)
    assert torch.equal(mine["pool_bank"], theirs["pool_bank"])
    assert torch.equal(mine["tail_bank"], theirs["tail_bank"])
