# SPDX-License-Identifier: Apache-2.0
"""``mla_decode_masked_attention``: a selecting decode step's attention by pool mask,
selected on chip from the rows' bounded pool scores.

The served per-rank shape is TP=64 of GLM-5.3-Flash: one head, latent 512, page 128,
pool 4, 512 pools selected, bf16 query and bank. The window is the selecting decode
bucket's 4096 rows (32 pages; 1024 candidate pools), where a verify step's 4 rows gather
``4 * 2176`` rows between them in the index form. What the masked form must keep:

* **The token set.** The module docstring's rule (a selected pool's tokens and the
  incomplete pool's tokens, up to the row's own position) is the set
  ``dsa_decode_select`` lists, at every position of the window, with tied scores and
  pools the bound masks. The kernel selects the pools itself, as
  ``dsa_decode_select_mask_torch_oracle`` does; scores quantised to a few levels put
  ties at every row's cut.
* **The attention.** Against ``mla_decode_attention_torch_oracle`` on the index form of
  the same selection, at relative L2 :data:`REL` (the bound ``test_mla_decode_split.py``
  holds the key-split kernel to), and against the served selected kernel at the same
  bound. The cases put the step's rows across a pool boundary and across the programs'
  halves of the window, on a pool their first row closes (selected for one row and not
  for the next), and on the window's last pool; one program and both.

Tolerance. The products are bf16 with fp32 sums and the value sum takes a bf16 hi/lo
split of the probabilities, as in the selected kernel; each program sums its half of the
keys under its own max, and the merge rescales each once.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.attention import mla_decode as MD
from vllm_neuron.functional.attention import mla_decode_masked as MM
from vllm_neuron.functional.dsa import decode_select as DS
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL
from vllm_neuron.functional.dsa.decode_bypass import bypass_max_context
from vllm_neuron.utils.neuron_utils import can_run_kernel

REL = 2e-5
LATENT, PAGE, POOL, K = 512, 128, 4, 512
PAGES = 32
WINDOW = PAGES * PAGE
CANDIDATES = WINDOW // POOL
SCALE = 0.0625


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    MM.reset_mla_decode_masked_dispatch_counters()


def _lnc(monkeypatch, lnc):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)


def _bounded(lengths, *, seed, levels=None, force=()):
    """Scores as the score kernel leaves them: ``BOUND_FILL`` past each row's complete
    pools. ``levels`` quantises them (ties); ``force`` is ``(row, pool, score)``."""
    gen = torch.Generator().manual_seed(seed)
    scores = torch.randn(len(lengths), CANDIDATES, generator=gen)
    if levels is not None:
        scores = torch.randint(0, levels, (len(lengths), CANDIDATES), generator=gen).float()
    for row, pool, score in force:
        scores[row, pool] = score
    cols = torch.arange(CANDIDATES)[None, :]
    complete = torch.tensor(lengths)[:, None] // POOL
    return scores.masked_fill(cols >= complete, BOUND_FILL).contiguous()


def _rule_rows(mask, lengths):
    """The module docstring's rule as a ``[rows, window]`` bool matrix."""
    cols = torch.arange(WINDOW)[None, :]
    pos = torch.tensor(lengths)[:, None] - 1
    selected = mask.repeat_interleave(POOL, dim=1) > 0
    in_open = cols >= ((pos + 1) & -POOL)
    return (selected | in_open) & (cols <= pos)


def _index_rows(indices):
    """The listed tokens as a ``[rows, window]`` bool matrix, and each row's count."""
    real = indices >= 0
    rows = torch.zeros(indices.shape[0], WINDOW + 1, dtype=torch.bool)
    rows.scatter_(1, torch.where(real, indices.long(), WINDOW), True)
    return rows[:, :WINDOW], real.sum(dim=1)


@pytest.mark.parametrize("levels", [None, 3])
def test_the_rule_is_the_index_set_at_every_selecting_position(levels):
    first = bypass_max_context(K * POOL, POOL)
    lengths = list(range(first + 1, WINDOW + 1))
    bounded = _bounded(lengths, seed=3 if levels is None else 4, levels=levels)
    seq = torch.tensor(lengths, dtype=torch.int32)
    mask = DS.dsa_decode_select_mask_torch_oracle(bounded, select_k=K)
    listed, counts = _index_rows(DS.dsa_decode_select_torch_oracle(
        bounded, seq, select_k=K, pool_size=POOL))
    assert torch.equal(_rule_rows(mask, lengths), listed)
    # Every listed token once: the set is the whole selection.
    assert torch.equal(listed.sum(dim=1), counts)


def _case(batch, rows, starts, *, seed, force=(), levels=None):
    """Operands at the served shape: ``batch`` requests of ``rows`` rows from ``starts``,
    disjoint pages, the rows' bounded pool scores, and their selection in both forms
    (pool mask and token indices)."""
    gen = torch.Generator().manual_seed(seed)
    bank_pages = batch * PAGES + 5
    bank = (torch.randn(bank_pages * PAGE, LATENT, generator=gen) * 0.5).to(torch.bfloat16)
    q = torch.randn(batch * rows, 1, LATENT, generator=gen).to(torch.bfloat16)
    written = torch.randn(batch * rows, LATENT, generator=gen).to(torch.bfloat16)
    table = torch.randperm(bank_pages, generator=gen)[:batch * PAGES].reshape(batch, PAGES)
    table = table.to(torch.int32)
    pos = torch.tensor(starts, dtype=torch.int32)
    for b in range(batch):
        table[b, (int(pos[b]) + rows - 1) // PAGE + 1:] = -1
    lengths = [int(pos[b]) + t + 1 for b in range(batch) for t in range(rows)]
    bounded = _bounded(lengths, seed=seed, levels=levels, force=force)
    mask = DS.dsa_decode_select_mask_torch_oracle(bounded, select_k=K)
    topk = DS.dsa_decode_select_torch_oracle(bounded, torch.tensor(lengths, dtype=torch.int32),
                                             select_k=K, pool_size=POOL)
    return (q, bank, table, pos, written), bounded, mask, topk


def _check(got, want, label):
    rel = float((got.double() - want.double()).norm() / want.double().norm())
    assert rel <= REL, f"{label}: relative L2 {rel:.3e} > {REL}"
    torch.testing.assert_close(got, want, rtol=REL * 10, atol=REL * float(want.abs().max()))


# Row 0 at 3003 closes pool 750 (tokens 3000 .. 3003): unselected for row 0, selected for
# row 1, whose own columns then include row 0.
CLOSE = ((0, 750, -1.0e6), (1, 750, 1.0e6))
CASES = [
    # (batch, rows, starts, lnc, force)
    (1, 4, [2050], "2", ()),        # the tail sits before the start
    (1, 4, [2047], "2", ()),        # the rows cross the programs' halves of the window
    (1, 4, [3001], "2", ()),        # the rows cross a pool boundary
    (1, 4, [3003], "2", CLOSE),     # the first row closes a pool
    (1, 4, [4092], "2", ()),        # the window's last pool
    (1, 2, [2101], "2", ()),
    (1, 1, [3003], "2", ((0, 750, -1.0e6),)),  # a row whose own pool is not selected
    (2, 4, [2600, 4091], "2", ()),
    (1, 4, [3003], None, CLOSE),
    (1, 4, [4092], None, ()),
]


def _attend(args, bounded):
    return MM.mla_decode_masked_attention(*args, SCALE, PAGE, bounded, POOL, select_k=K)


@pytest.mark.parametrize("batch, rows, starts, lnc, force", CASES)
def test_the_kernel_attends_the_index_form(batch, rows, starts, lnc, force, monkeypatch):
    _lnc(monkeypatch, lnc)
    args, bounded, _mask, topk = _case(batch, rows, starts, seed=17 + sum(starts) + rows,
                                       force=force)
    got = _attend(args, bounded)
    assert MM.mla_decode_masked_dispatch_counters() == (1, 0)
    assert MM.mla_decode_masked_two_program_count() == (1 if lnc == "2" else 0)
    want = MD.mla_decode_attention_torch_oracle(*args, SCALE, PAGE, topk)
    _check(got, want, f"B={batch} T={rows} starts={starts} lnc={lnc} vs oracle")


def test_a_forced_pool_reaches_the_rows_it_was_forced_for():
    # The CLOSE case's premise: row 0 does not select pool 750, row 1 does.
    _args, _bounded_rows, mask, _topk = _case(1, 4, [3003], seed=17 + 3003 + 4, force=CLOSE)
    assert float(mask[0, 750]) == 0.0 and float(mask[1, 750]) == 1.0


@pytest.mark.parametrize("starts, lnc", [([3001], "2"), ([2600], "2"), ([4092], None)])
def test_tied_scores_select_as_the_index_selection_does(starts, lnc, monkeypatch):
    # Three score levels: every row's cut falls inside a run of ties, which the kernel
    # breaks as the selection does (the lowest candidates first).
    _lnc(monkeypatch, lnc)
    args, bounded, mask, topk = _case(1, 4, starts, seed=41 + starts[0], levels=3)
    kth = bounded.sort(dim=1, descending=True).values[:, K - 1:K]
    above, ties = (bounded > kth).sum(dim=1), (bounded == kth).sum(dim=1)
    assert bool((above + ties > K).all()), "the cut must split a run of ties in every row"
    got = _attend(args, bounded)
    want = MD.mla_decode_attention_torch_oracle(*args, SCALE, PAGE, topk)
    _check(got, want, f"ties starts={starts} lnc={lnc} vs oracle")


@pytest.mark.parametrize("starts", [[3001], [4092]])
def test_the_kernel_agrees_with_the_served_selected_kernel(starts, monkeypatch):
    _lnc(monkeypatch, "2")
    args, bounded, _mask, topk = _case(1, 4, starts, seed=29 + starts[0])
    got = _attend(args, bounded)
    MD.reset_mla_decode_dispatch_counters()
    want = MD.mla_decode_attention(*args, SCALE, PAGE, topk_indices=topk)
    assert MD.mla_decode_route_counts()[:2] == (0, 1)
    _check(got, want, f"starts={starts} vs the selected kernel")


def test_the_served_shape_is_served_on_one_program_and_on_two(monkeypatch):
    for lnc in ("2", None):
        _lnc(monkeypatch, lnc)
        assert MM.mla_decode_masked_serves(heads=1, latent=LATENT, page=PAGE, window=WINDOW,
                                           rows=4, pool=POOL)


@pytest.mark.parametrize("field, value", [
    ("heads", 2), ("latent", 384), ("page", 64), ("window", WINDOW + PAGE),
    ("rows", 33), ("pool", 3), ("pool", 256),
])
def test_a_geometry_outside_the_kernel_is_not_served(field, value, monkeypatch):
    _lnc(monkeypatch, "2")
    geometry = dict(heads=1, latent=LATENT, page=PAGE, window=WINDOW, rows=4, pool=POOL)
    geometry[field] = value
    assert not MM.mla_decode_masked_serves(**geometry)


def test_the_kernel_key_covers_the_selection_the_kernel_runs():
    """The kernel's cache key is a digest over this module, ``mla_decode.py`` and
    ``decode_select.py``, whose helpers the body runs: an edit to any re-keys the
    kernel."""
    from vllm_neuron.source_digest import source_digest

    assert MM.SOURCE_DIGEST == source_digest(MM.__file__, MD.__file__, DS.__file__)


def test_more_rows_than_one_selection_tile_are_not_served(monkeypatch):
    # A two-page window leaves the moving operand room for 33 rows, but the kernel
    # selects a request's rows in one tile of the selection.
    _lnc(monkeypatch, "2")
    short = dict(heads=1, latent=LATENT, page=PAGE, window=2 * PAGE, pool=POOL)
    tile = DS.tile_requests(2 * PAGE // POOL)
    assert MM.mla_decode_masked_serves(rows=tile, **short)
    assert not MM.mla_decode_masked_serves(rows=tile + 1, **short)


def test_refusals_name_the_operand(monkeypatch):
    _lnc(monkeypatch, "2")
    args, bounded, _mask, _topk = _case(1, 4, [3001], seed=5)
    for wrong, name in ((bounded[:, :-1], "pool_scores"), (bounded[:3], "pool_scores"),
                        (bounded.to(torch.bfloat16), "pool_scores")):
        with pytest.raises(MD.MlaDecodeAttentionError, match=name):
            _attend(args, wrong)
    for select_k in (0, CANDIDATES):
        with pytest.raises(MD.MlaDecodeAttentionError, match="select_k"):
            MM.mla_decode_masked_attention(*args, SCALE, PAGE, bounded, POOL,
                                           select_k=select_k)
    with pytest.raises(MD.MlaDecodeAttentionError, match="pool_size"):
        MM.mla_decode_masked_attention(*args, SCALE, PAGE, bounded, 3, select_k=K)
    q, bank, table, pos, written = args
    two_heads = torch.cat([q, q], dim=1)
    with pytest.raises(MD.MlaDecodeAttentionError, match="heads=2"):
        _attend((two_heads, bank, table, pos, written), bounded)
    assert MM.mla_decode_masked_dispatch_counters() == (0, 0)


# ---------------------------------------------------------------------------------------
# The one-row key-split form at a width of an odd chunk count
# ---------------------------------------------------------------------------------------

#: The served one-row selection: 512 pools of 4 tokens and one 128-row tail chunk.
ONE_ROW_WIDTH = 2176


def _one_row(start, *, seed, width=ONE_ROW_WIDTH):
    """One request, one row at ``start``: the operands and its token indices, cut to
    ``width`` columns."""
    args, _bounded_rows, _mask, topk = _case(1, 1, [start], seed=seed)
    assert int(topk.shape[1]) == ONE_ROW_WIDTH
    return args, topk[:, :width].contiguous()


@pytest.mark.parametrize("start, width", [
    (2049, ONE_ROW_WIDTH),   # the first selecting position: fewer complete pools than k
    (3001, ONE_ROW_WIDTH),
    (4095, ONE_ROW_WIDTH),   # the window's last row
    (3001, 9 * 128),         # 9 chunks: the selected pools' first tokens only
])
def test_the_padded_split_attends_one_row_as_the_oracle(start, width, monkeypatch):
    _lnc(monkeypatch, "2")
    args, topk = _one_row(start, seed=53 + start, width=width)
    assert (width // 128) % 2 == 1
    assert not MD._split_serves(1, LATENT, PAGE, width, 2)
    got = MM.mla_decode_padded_split_attention(*args, SCALE, PAGE, topk)
    assert MM.mla_decode_padded_split_count() == 1
    assert MM.mla_decode_masked_dispatch_counters() == (0, 0)
    want = MD.mla_decode_attention_torch_oracle(*args, SCALE, PAGE, topk)
    _check(got, want, f"padded split start={start} width={width} vs oracle")
    # The kernel mla_decode picks at this width: the general selected kernel.
    MD.reset_mla_decode_dispatch_counters()
    general = MD.mla_decode_attention(*args, SCALE, PAGE, topk_indices=topk)
    assert MD.mla_decode_route_counts()[:2] == (0, 1) and MD.mla_decode_split_counts()[0] == 0
    _check(got, general, f"padded split start={start} width={width} vs the general kernel")


def test_the_padded_split_serves_an_odd_chunk_count_on_an_lnc2_pair_only(monkeypatch):
    def serves(**changed):
        geometry = dict(heads=1, latent=LATENT, page=PAGE, width=ONE_ROW_WIDTH)
        geometry.update(changed)
        return MM.mla_decode_padded_split_serves(**geometry)

    _lnc(monkeypatch, "2")
    assert serves() and serves(width=9 * 128)
    # An even chunk count is mla_decode's key-split kernel's.
    assert MD._split_serves(1, LATENT, PAGE, 2048, 2) and not serves(width=2048)
    assert not serves(heads=2) and not serves(latent=384)
    # One program splits nothing: mla_decode's key-split kernel serves every width there.
    _lnc(monkeypatch, None)
    assert not serves()


def test_the_padded_split_refuses_what_it_does_not_serve(monkeypatch):
    _lnc(monkeypatch, "2")
    args, topk = _one_row(3001, seed=5)
    with pytest.raises(MD.MlaDecodeAttentionError, match="width"):
        MM.mla_decode_padded_split_attention(*args, SCALE, PAGE, topk[:, :2048].contiguous())
    rows4, _bounded_rows, _mask, topk4 = _case(1, 4, [3001], seed=5)
    with pytest.raises(MD.MlaDecodeAttentionError, match="one query row"):
        MM.mla_decode_padded_split_attention(*rows4, SCALE, PAGE, topk4)
    with pytest.raises(MD.MlaDecodeAttentionError, match="multiple of 128"):
        MM.mla_decode_padded_split_attention(*args, SCALE, PAGE, topk[:, :100].contiguous())
    assert MM.mla_decode_padded_split_count() == 0
