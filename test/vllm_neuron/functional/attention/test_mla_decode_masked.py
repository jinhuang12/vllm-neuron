# SPDX-License-Identifier: Apache-2.0
"""``mla_decode_masked_attention``: a selecting decode step's attention by pool mask.

The served per-rank shape is TP=64 of GLM-5.3-Flash: one head, latent 512, page 128,
pool 4, 512 pools selected, bf16 query and bank. The window is the selecting decode
bucket's 4096 rows (32 pages; 1024 candidate pools), where a verify step's 4 rows gather
``4 * 2176`` rows between them in the index form. What the masked form must keep:

* **The token set.** The module docstring's rule (a selected pool's tokens and the
  incomplete pool's tokens, up to the row's own position) is the set
  ``dsa_decode_select`` lists, at every position of the window, with tied scores and
  pools the bound masks.
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


def _case(batch, rows, starts, *, seed, force=()):
    """Operands at the served shape: ``batch`` requests of ``rows`` rows from ``starts``,
    disjoint pages, and one selection in both forms (pool mask and token indices)."""
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
    bounded = _bounded(lengths, seed=seed, force=force)
    mask = DS.dsa_decode_select_mask_torch_oracle(bounded, select_k=K)
    topk = DS.dsa_decode_select_torch_oracle(bounded, torch.tensor(lengths, dtype=torch.int32),
                                             select_k=K, pool_size=POOL)
    return (q, bank, table, pos, written), mask, topk


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


@pytest.mark.parametrize("batch, rows, starts, lnc, force", CASES)
def test_the_kernel_attends_the_index_form(batch, rows, starts, lnc, force, monkeypatch):
    _lnc(monkeypatch, lnc)
    args, mask, topk = _case(batch, rows, starts, seed=17 + sum(starts) + rows, force=force)
    got = MM.mla_decode_masked_attention(*args, SCALE, PAGE, mask, POOL)
    assert MM.mla_decode_masked_dispatch_counters() == (1, 0)
    assert MM.mla_decode_masked_two_program_count() == (1 if lnc == "2" else 0)
    want = MD.mla_decode_attention_torch_oracle(*args, SCALE, PAGE, topk)
    _check(got, want, f"B={batch} T={rows} starts={starts} lnc={lnc} vs oracle")


def test_a_forced_pool_reaches_the_rows_it_was_forced_for():
    # The CLOSE case's premise: row 0 does not select pool 750, row 1 does.
    _args, mask, _topk = _case(1, 4, [3003], seed=17 + 3003 + 4, force=CLOSE)
    assert float(mask[0, 750]) == 0.0 and float(mask[1, 750]) == 1.0


@pytest.mark.parametrize("starts", [[3001], [4092]])
def test_the_kernel_agrees_with_the_served_selected_kernel(starts, monkeypatch):
    _lnc(monkeypatch, "2")
    args, mask, topk = _case(1, 4, starts, seed=29 + starts[0])
    got = MM.mla_decode_masked_attention(*args, SCALE, PAGE, mask, POOL)
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


def test_refusals_name_the_operand(monkeypatch):
    _lnc(monkeypatch, "2")
    args, mask, _topk = _case(1, 4, [3001], seed=5)
    for wrong, name in ((mask[:, :-1], "pool_mask"), (mask[:3], "pool_mask"),
                        (mask.to(torch.bfloat16), "pool_mask")):
        with pytest.raises(MD.MlaDecodeAttentionError, match=name):
            MM.mla_decode_masked_attention(*args, SCALE, PAGE, wrong, POOL)
    with pytest.raises(MD.MlaDecodeAttentionError, match="pool_size"):
        MM.mla_decode_masked_attention(*args, SCALE, PAGE, mask, 3)
    q, bank, table, pos, written = args
    two_heads = torch.cat([q, q], dim=1)
    with pytest.raises(MD.MlaDecodeAttentionError, match="heads=2"):
        MM.mla_decode_masked_attention(two_heads, bank, table, pos, written, SCALE, PAGE,
                                       mask, POOL)
    assert MM.mla_decode_masked_dispatch_counters() == (0, 0)
