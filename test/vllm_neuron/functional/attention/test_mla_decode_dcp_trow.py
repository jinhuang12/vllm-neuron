# SPDX-License-Identifier: Apache-2.0
"""DCP mode of ``mla_decode`` with ``T`` query rows per request: the verify step under DCP.

A speculative verify step hands the decode attention ``T = 1 + k`` rows per request at
positions ``p .. p + T - 1``. Under decode context parallelism (DCP) each of ``CP`` ranks
holds every ``CP``-th 128-token block of a sequence's latent KV, so the rows of one step can
lie in two ranks' blocks. :func:`~vllm_neuron.functional.attention.mla_decode.
mla_decode_partial_attention` takes the step's rows with per-row ownership operands --
``position [B * T]``, each row's owned length on the rank, and ``owner [B * T]``, whether
the rank holds the row (:func:`~vllm_neuron.functional.attention.mla_decode.dcp_step_rows`
builds both from ``utils/dcp_ownership.py``) -- and returns the rank's ``partial
[Hq, B * T, L]`` and ``lse [Hq, B * T]``. The llama3 LSE merge of the ``CP`` ranks' pairs is
the CP=1 answer row by row.

What is checked, on the CPU simulator, against the fp32 CP=1 torch oracle
:func:`~vllm_neuron.functional.attention.mla_decode.mla_decode_attention_torch_oracle`:

* CP in {2, 4} x T in {1, 2, 4} x B in {1, 2}, dense and selected, on the LNC2 grid (both
  programs split every row's keys, one request included); request 0's rows cross a
  128-token block boundary (for ``T = 1`` its row starts a block);
* per (CP, T): a step whose rows all lie in block 0, so every other rank holds nothing
  and returns the empty partial (0, ``EMPTY_LSE``) exactly, and a step that crosses from
  block 0 into block 1, so its new owner holds no row before the step: the step's rows
  before that owner's first row have no column there;
* the bank holds garbage at every row of the step, so a row read from the bank instead of
  ``written`` fails; the owner gate's RED half (every rank counting every own row);
* the served per-rank shape at DCP 2 (2176 selected rows padded to 2304, latent 512);
* one program as well as two; the production NKI merge ``dcp_lse_merge`` at ``R = B * T``;
* ``T = 1`` through the entry is commit b5b516b3's partial kernel bit for bit;
* the refusals name the operand; the NKI front end accepts both T-row kernels.

Tolerance: the form ``test_mla_decode_trow.py`` holds the T-row kernel to against the same
oracle, ``assert_close(rtol=2e-4, atol=2e-5 * max|oracle|)``. The merge reference is a
local fp32 copy of the llama3 formula, as ``test_mla_decode_dcp.py`` uses. Every case
prints one ``DCP_TROW|`` line with its largest absolute error and the bound (run with
``-s`` to see them).
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import pytest
import torch

from test.vllm_neuron.functional.reference_at_commit import load_reference, needs_reference
from vllm_neuron.functional.attention import mla_decode as MD
from vllm_neuron.functional.attention.dcp_merge import dcp_lse_merge
from vllm_neuron.utils.dcp_ownership import token_local_row, token_owner

LATENT, PAGE = 512, 128
SCALE = 0.0625
#: The tolerance form of ``test_mla_decode_trow.py`` against the oracle.
RTOL, ATOL = 2e-4, 2e-5
#: What the bank holds at every row of the step: a row read from the bank instead of
#: ``written`` scores about 64 * sum(q) and swamps the softmax.
POISON = 64.0
#: The commit whose partial kernels a one-row step must still launch, bit for bit, and its
#: mla_decode.py (test/vllm_neuron/functional/reference_at_commit.py loads it).
BASE, BASE_PATH = "b5b516b3", "vllm_neuron/functional/attention/mla_decode.py"
_ROOT = pathlib.Path(__file__).resolve().parents[4]


@dataclass
class Step:
    """A CP=1 verify step: ``B`` requests x ``T`` rows, request-major."""

    q: torch.Tensor           # [B * T, Hq, L]
    bank: torch.Tensor        # [slots, L]
    table: torch.Tensor       # [B, pages] the CP=1 block table
    start: torch.Tensor       # [B] int32, row 0's position
    written: torch.Tensor     # [B * T, L]
    topk: torch.Tensor | None  # [B * T, width] CP=1 window rows, -1 = no token
    rows: int


@dataclass
class Shard:
    """Rank ``rank``'s operands for ``step``."""

    table: torch.Tensor       # [B, local_pages]
    position: torch.Tensor    # [B * T] int32, each row's owned length
    owner: torch.Tensor       # [B * T] int32, 1 where this rank holds the row
    topk: torch.Tensor | None  # [B * T, width] local rows, -1 = none or another rank's


def _slot(table_row, position):
    return int(table_row[position // PAGE]) * PAGE + position % PAGE


def global_step(batch, rows, heads, *, pages, starts, seed, selected=False, width=512):
    """A CP=1 step of ``batch`` requests x ``rows`` rows over ``pages``-page windows.

    Each request owns its pages. In selected mode row ``t``'s ids are causal (positions up
    to ``start + t``), 15 % of them -1, and name every row of the step up to ``t`` on
    purpose, row 0 twice. The bank holds :data:`POISON` at every row of the step.
    """
    gen = torch.Generator().manual_seed(seed)
    bank_pages = batch * pages + 2
    bank = torch.randn(bank_pages * PAGE, LATENT, generator=gen).to(torch.bfloat16)
    q = torch.randn(batch * rows, heads, LATENT, generator=gen).to(torch.bfloat16)
    table = torch.randperm(bank_pages, generator=gen)[:batch * pages].reshape(
        batch, pages).to(torch.int32)
    start = torch.as_tensor(starts, dtype=torch.int32)
    assert int(start.max()) + rows <= pages * PAGE
    for b in range(batch):
        table[b, (int(start[b]) + rows - 1) // PAGE + 1:] = -1
    written = torch.randn(batch * rows, LATENT, generator=gen).to(torch.bfloat16)
    for b in range(batch):
        for t in range(rows):
            bank[_slot(table[b], int(start[b]) + t)] = POISON
    topk = None
    if selected:
        topk = torch.full((batch * rows, width), -1, dtype=torch.int32)
        for b in range(batch):
            for t in range(rows):
                live = int(start[b]) + t + 1
                pick = torch.randint(0, live, (width,), generator=gen)
                keep = torch.rand(width, generator=gen) < 0.85
                row = torch.where(keep, pick, torch.full_like(pick, -1)).to(torch.int32)
                for j in range(t + 1):
                    row[3 + j] = int(start[b]) + j
                row[width - 5] = int(start[b])
                topk[b * rows + t] = row
    return Step(q, bank, table, start, written, topk, rows)


def _local_pages(pages, cp):
    """Every rank's window: ceil(pages / CP) pages, even, so both programs split it."""
    local = -(-pages // cp)
    return local + local % 2


def shard(step: Step, cp: int, rank: int) -> Shard:
    """Rank ``rank``'s view of ``step`` (page = block = 128 rows: rank ``k % CP`` holds
    block ``k`` as its local block ``k // CP``)."""
    batch, pages = (int(d) for d in step.table.shape)
    table = torch.full((batch, _local_pages(pages, cp)), -1, dtype=torch.int32)
    for j in range(rank, pages, cp):
        table[:, j // cp] = step.table[:, j]
    position, owner = MD.dcp_step_rows(step.start, rank, rows=step.rows, block_size=PAGE,
                                       dcp_size=cp)
    topk = None
    if step.topk is not None:
        ids = step.topk.to(torch.int64)
        safe = ids.clamp(min=0)
        held = (ids >= 0) & (token_owner(safe, block_size=PAGE, dcp_size=cp) == rank)
        local = token_local_row(safe, block_size=PAGE, dcp_size=cp)
        topk = torch.where(held, local, torch.full_like(ids, -1)).to(torch.int32)
    return Shard(table, position, owner, topk)


def _grid(monkeypatch, lnc):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)


def run_ranks(step: Step, cp: int, monkeypatch, lnc="2", owner_override=None,
              module=MD):
    """Every rank's (partial, lse), stacked ``[CP, Hq, B * T, ...]``; one NKI launch each."""
    _grid(monkeypatch, lnc)
    partials, lses = [], []
    batch, heads = int(step.table.shape[0]), int(step.q.shape[1])
    for rank in range(cp):
        part = shard(step, cp, rank)
        owner = part.owner if owner_override is None else owner_override
        module.reset_mla_decode_dispatch_counters()
        partial, lse = module.mla_decode_partial_attention(
            step.q, step.bank, part.table, part.position, step.written, owner, SCALE, PAGE,
            topk_indices=part.topk)
        assert module.mla_decode_dispatch_counters() == (1, 0)
        assert module.mla_decode_partial_counts() == (1, 1)
        if module is MD:
            two = 1 if lnc == "2" else 0
            assert MD.mla_decode_partial_row_counts() == (
                (1, two) if step.rows > 1 else (0, 0))
            assert MD.mla_decode_split_counts() == (1, two)
        assert partial.dtype == torch.float32
        assert tuple(partial.shape) == (heads, batch * step.rows, LATENT)
        assert lse.dtype == torch.float32 and tuple(lse.shape) == (heads, batch * step.rows)
        partials.append(partial)
        lses.append(lse)
    return torch.stack(partials), torch.stack(lses)


def lse_merge(partials, lses):
    """The llama3 merge, head-major, fp32 (``test_mla_decode_dcp.py``'s local copy)."""
    merged_lse = torch.logsumexp(lses, dim=0)
    return (torch.exp(lses - merged_lse).unsqueeze(-1) * partials).sum(0)


def oracle(step: Step) -> torch.Tensor:
    """The CP=1 fp32 oracle, head-major ``[Hq, B * T, L]``."""
    return MD.mla_decode_attention_torch_oracle(
        step.q, step.bank, step.table, step.start, step.written, SCALE, PAGE,
        step.topk).permute(1, 0, 2)


def check(got, want, label):
    """``test_mla_decode_trow.py``'s tolerance form; prints the observed error."""
    peak = float(want.abs().max())
    err = float((got - want).abs().max())
    print(f"DCP_TROW|{label}|max_abs_err={err:.3e}|atol={ATOL * peak:.3e}|"
          f"max_rel_err={err / peak:.3e}", flush=True)
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL * peak)


# ---- the operands ---------------------------------------------------------------------------

def test_step_rows_are_each_rows_owned_length_and_owner():
    # Block 1 of a CP=2 group is rank 1's; the step's rows 254 .. 257 cross into block 2,
    # rank 0's local block 1.
    start = torch.tensor([254, 3], dtype=torch.int32)
    position, owner = MD.dcp_step_rows(start, 0, rows=4, block_size=PAGE, dcp_size=2)
    assert position.dtype == torch.int32 and owner.dtype == torch.int32
    assert position.tolist() == [128, 128, 128, 129, 3, 4, 5, 6]
    assert owner.tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
    position, owner = MD.dcp_step_rows(start, torch.tensor([1]), rows=4, block_size=PAGE,
                                       dcp_size=2)
    assert position.tolist() == [126, 127, 128, 128, 0, 0, 0, 0]
    assert owner.tolist() == [1, 1, 0, 0, 0, 0, 0, 0]


def test_the_entry_publishes_its_t_row_capability():
    assert MD.DCP_SPECULATIVE_ROWS is True


# ---- CP partials merge to the CP=1 answer ---------------------------------------------------

CPS = (2, 4)
ROWS = (1, 2, 4)
BATCHES = (1, 2)
GRID = [(cp, rows, batch, selected) for cp in CPS for rows in ROWS for batch in BATCHES
        for selected in (False, True)]


def _crossing(block, rows):
    """Row 0's position for a step whose rows cross into ``block`` (``T = 1``: its row
    starts the block)."""
    return block * PAGE - rows // 2


@pytest.mark.parametrize("cp,rows,batch,selected", GRID)
def test_t_row_partials_merge_to_the_cp1_oracle(cp, rows, batch, selected, monkeypatch):
    # Dense: the 2048-row bypass window (16 pages); selected: a 4096-row window, 512
    # selected rows. Request 0 crosses into block 9 (dense) / 25 (selected); request 1
    # ends at the window's last row.
    pages = 32 if selected else 16
    starts = [_crossing(25 if selected else 9, rows), pages * PAGE - rows][:batch]
    step = global_step(batch, rows, cp, pages=pages, starts=starts,
                       seed=100 * cp + 10 * rows + batch + 7 * selected, selected=selected)
    partials, lses = run_ranks(step, cp, monkeypatch)
    check(lse_merge(partials, lses), oracle(step),
          f"grid cp={cp} T={rows} B={batch} {'selected' if selected else 'dense'}")


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("cp", CPS)
def test_a_step_in_block_0_leaves_every_other_rank_empty(cp, rows, selected, monkeypatch):
    step = global_step(1, rows, cp, pages=8, starts=[3], seed=300 + 10 * cp + rows,
                       selected=selected, width=256)
    partials, lses = run_ranks(step, cp, monkeypatch)
    for rank in range(1, cp):
        assert torch.count_nonzero(partials[rank]) == 0
        assert torch.equal(lses[rank], torch.full_like(lses[rank], MD.EMPTY_LSE))
    check(lse_merge(partials, lses), oracle(step),
          f"empty cp={cp} T={rows} {'selected' if selected else 'dense'}")


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("cp", CPS)
def test_a_step_into_block_1_has_rows_with_no_column_on_its_new_owner(cp, rows, selected,
                                                                      monkeypatch):
    # Rows 128 - T/2 .. : rank 1 holds none of the context before the step, so the step's
    # rows before rank 1's first row attend nothing there (at CP = 4 ranks 2 and 3 hold
    # nothing at all).
    start = _crossing(1, rows)
    step = global_step(1, rows, cp, pages=8, starts=[start], seed=400 + 10 * cp + rows,
                       selected=selected, width=256)
    partials, lses = run_ranks(step, cp, monkeypatch)
    before = PAGE - start
    assert torch.count_nonzero(partials[1][:, :before]) == 0
    assert torch.equal(lses[1][:, :before], torch.full_like(lses[1][:, :before],
                                                            MD.EMPTY_LSE))
    assert bool((lses[1][:, before:] > MD.EMPTY_LSE / 2).all())
    check(lse_merge(partials, lses), oracle(step),
          f"new-owner cp={cp} T={rows} {'selected' if selected else 'dense'}")


@pytest.mark.parametrize("cp", CPS)
@pytest.mark.parametrize("selected", [False, True])
def test_one_program_merges_to_the_oracle_too(selected, cp, monkeypatch):
    step = global_step(2, 4, cp, pages=16, starts=[_crossing(5, 4), 1500], seed=500 + cp,
                       selected=selected, width=512)
    partials, lses = run_ranks(step, cp, monkeypatch, lnc=None)
    check(lse_merge(partials, lses), oracle(step),
          f"one-program cp={cp} {'selected' if selected else 'dense'}")


def test_the_served_dcp2_verify_step(monkeypatch):
    # GLM-5.3-Flash at DCP 2: 2 gathered heads, latent 512, 2176 selected rows padded to
    # 2304 (model_fp8.py _dcp_decode pads to 2 * page_size), MTP k = 3, a 16384-row window,
    # rows crossing into block 100.
    width = 2304
    step = global_step(1, 4, 2, pages=128, starts=[_crossing(100, 4)], seed=601,
                       selected=True, width=width)
    step.topk[:, 2176:] = -1
    partials, lses = run_ranks(step, 2, monkeypatch)
    check(lse_merge(partials, lses), oracle(step), "served dcp2 T=4 selected 2304")


@pytest.mark.parametrize("selected", [False, True])
def test_the_production_merge_takes_the_rows(selected, monkeypatch):
    # dcp_lse_merge (NKI) per rank's own heads, [CP, H, B * T, L] -> [B * T, H, L] bf16.
    cp, heads = 2, 1
    step = global_step(2, 4, cp * heads, pages=16, starts=[_crossing(5, 4), 2000],
                       seed=701, selected=selected)
    partials, lses = run_ranks(step, cp, monkeypatch)
    want = oracle(step)
    for rank in range(cp):
        mine = slice(rank * heads, (rank + 1) * heads)
        merged = dcp_lse_merge(partials[:, mine].contiguous(), lses[:, mine].contiguous())
        assert merged.dtype == torch.bfloat16
        assert tuple(merged.shape) == (2 * 4, heads, LATENT)
        got = merged.float().permute(1, 0, 2)
        peak = float(want[mine].abs().max())
        torch.testing.assert_close(got, want[mine], rtol=1e-2, atol=1e-2 * peak)


def test_only_each_rows_holder_counts_it(monkeypatch):
    # The owner gate (dense): GREEN with each rank's own owner rows; RED with every rank
    # counting every row of the step as its own. The entry refuses that owner (positions
    # would not advance by it), so the RED half launches the dense T-row kernel directly.
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    step = global_step(1, 4, 2, pages=16, starts=[_crossing(5, 4)], seed=801)
    good, good_lse = run_ranks(step, 2, monkeypatch)
    want = oracle(step)
    check(lse_merge(good, good_lse), want, "owner gate GREEN")
    everyone = torch.ones(4, dtype=torch.int32)
    bad, bad_lse = [], []
    for rank in range(2):
        part = shard(step, 2, rank)
        call = wrap_nki(MD.mla_decode_dense_split_rows_partial_kernel)[2]
        partial, lse = call(step.q, step.bank, part.table, part.position, step.written,
                            everyone, SCALE, PAGE, MD.SOURCE_DIGEST)
        bad.append(partial)
        bad_lse.append(lse)
    err = float((lse_merge(torch.stack(bad), torch.stack(bad_lse)) - want).abs().max())
    assert err > 100 * ATOL * float(want.abs().max()), err


# ---- T = 1 is commit b5b516b3's kernel ------------------------------------------------------

@pytest.fixture(scope="module")
def base_module():
    return load_reference(BASE, BASE_PATH, f"mla_decode_{BASE}")


@needs_reference(BASE, BASE_PATH)
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("cp", CPS)
def test_one_row_is_commit_b5b516b3s_partial_bit_for_bit(cp, batch, selected, base_module,
                                                          monkeypatch):
    pages = 32 if selected else 16
    starts = [_crossing(25 if selected else 9, 1), 1500][:batch]
    step = global_step(batch, 1, cp, pages=pages, starts=starts, seed=900 + cp + batch,
                       selected=selected)
    partials, lses = run_ranks(step, cp, monkeypatch)
    before, before_lse = run_ranks(step, cp, monkeypatch, module=base_module)
    assert torch.equal(partials, before)
    assert torch.equal(lses, before_lse)


# ---- refusals -------------------------------------------------------------------------------

def _refusal_step(selected=False):
    step = global_step(2, 2, 2, pages=4, starts=[_crossing(2, 2), 40], seed=1001,
                       selected=selected, width=128)
    return step, shard(step, 2, 0)


def _call(step, part, **change):
    args = dict(q_lift=step.q, bank=step.bank, block_table=part.table,
                position=part.position, written=step.written, owner=part.owner,
                softmax_scale=SCALE, page_size=PAGE, topk_indices=part.topk)
    args.update(change)
    MD.reset_mla_decode_dispatch_counters()
    try:
        return MD.mla_decode_partial_attention(**args)
    finally:
        if change:
            assert MD.mla_decode_dispatch_counters() == (0, 0)


def test_per_row_operands_must_hold_every_row():
    step, part = _refusal_step()
    with pytest.raises(MD.MlaDecodeAttentionError, match=r"position must be \[batch \* rows\]"):
        _call(step, part, position=part.position[::2].contiguous())
    with pytest.raises(MD.MlaDecodeAttentionError, match=r"owner must be \[batch \* rows\]"):
        _call(step, part, owner=part.owner[::2].contiguous())
    with pytest.raises(MD.MlaDecodeAttentionError, match="owner must be 0 or 1"):
        _call(step, part, owner=part.owner + 2)


def test_positions_must_advance_by_owner_along_a_request():
    step, part = _refusal_step()
    # Request 0's rows 255, 256 on rank 0: 128 (not held), then 128 (held).
    assert part.position[:2].tolist() == [128, 128] and part.owner[:2].tolist() == [0, 1]
    stalled = part.position.clone()
    stalled[2:] = stalled[2]
    with pytest.raises(MD.MlaDecodeAttentionError, match="advance by owner"):
        _call(step, part, position=stalled)
    with pytest.raises(MD.MlaDecodeAttentionError, match="non-negative"):
        _call(step, part, position=part.position - 200)
    window = int(part.table.shape[1]) * PAGE
    # Request 1's first row held at the window's end: no local row there.
    held_late = torch.tensor([window - 1, window - 1, window, window + 1], dtype=torch.int32)
    with pytest.raises(MD.MlaDecodeAttentionError, match="inside its local window"):
        _call(step, part, position=held_late,
              owner=torch.tensor([0, 1, 1, 0], dtype=torch.int32))


def test_a_selected_row_this_rank_does_not_hold_must_not_name_its_position():
    step, part = _refusal_step(selected=True)
    names = part.topk.clone()
    names[0, 7] = int(part.position[0])  # row 0 (position 255, rank 1's) names 128
    with pytest.raises(MD.MlaDecodeAttentionError, match="non-owner"):
        _call(step, part, topk_indices=names)


def test_too_many_row_units_for_one_score_tile_are_refused(monkeypatch):
    # 8 heads x 9 rows = 72 (row, head) units: the value pass's hi/lo rows (144) do not fit
    # 128 partitions.
    _grid(monkeypatch, "2")
    step = global_step(1, 9, MD.GROUP, pages=4, starts=[200], seed=1003)
    part = shard(step, 2, 0)
    with pytest.raises(MD.MlaDecodeAttentionError, match="key-split kernel.*72 .row, head. units"):
        _call(step, part)


def test_the_t1_messages_are_todays():
    step = global_step(2, 1, 2, pages=4, starts=[300, 40], seed=1005)
    part = shard(step, 2, 0)
    with pytest.raises(MD.MlaDecodeAttentionError, match=r"owner must be \[batch\]"):
        _call(step, part, owner=part.owner[:1].contiguous())
    with pytest.raises(MD.MlaDecodeAttentionError, match=r"position must be \[batch\]"):
        _call(step, part, position=part.position[:1].contiguous())


# ---- the NKI front end ----------------------------------------------------------------------

ROW = "dcp_trow_frontend"
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "PYTHONDONTWRITEBYTECODE": "1"}
#: (gathered heads, rows): DCP 2 at k = 1 and 3, DCP 4 at k = 3, DCP 8 at k = 1.
FRONTEND_SHAPES = ((2, 2), (2, 4), (4, 4), (8, 2))
FRONTEND_BATCHES = (1, 2)


def _open_device_nodes() -> int:
    held = [os.path.realpath(f"/proc/self/fd/{h}") for h in os.listdir("/proc/self/fd")]
    return len([one for one in held if "/dev/neuron" in one])


def _compile_each_entry() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    import nki
    import nki.language as nl
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    @nki.jit
    def body_that_reads_an_undefined_name(x_hbm):
        out = nl.ndarray(x_hbm.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        held = nl.ndarray((1, x_hbm.shape[1]), dtype=nl.float32, buffer=nl.sbuf)
        nl.store(out[0:1, :], value=held * no_such_name_anywhere)  # noqa: F821
        return out

    bf, i32 = torch.bfloat16, torch.int32

    def fake(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    def launch(entry, programs, b, h, t, pages, width):
        call = wrap_nki(entry)
        call = call[2] if programs == 2 else call
        args = [fake((b * t, h, LATENT), bf), fake((64 * PAGE, LATENT), bf),
                fake((b, pages), i32), fake((b * t,), i32), fake((b * t, LATENT), bf),
                fake((b * t,), i32)]
        if width is not None:
            args.append(fake((b * t, width), i32))
        return call(*args, SCALE, PAGE, MD.SOURCE_DIGEST)

    built = []
    # The served per-rank windows at DCP 2: the dense 2048-row bucket's half (8 pages), a
    # 2304-column selection in a 64-page window. Only geometries the entry admits: one
    # program cannot hold 4 rows of a 2304-column selection (the row budget).
    kernels = (("dense_rows", MD.mla_decode_dense_split_rows_partial_kernel, 8, None),
               ("selected_rows", MD.mla_decode_selected_split_rows_partial_kernel, 64, 2304))
    for name, entry, pages, width in kernels:
        for b in FRONTEND_BATCHES:
            for h, t in FRONTEND_SHAPES:
                for g in (1, 2):
                    label = f"{name}_b{b}_h{h}_t{t}_g{g}"
                    columns = pages * PAGE if width is None else width
                    if not MD._split_partial_serves(h, LATENT, PAGE, columns, g, t,
                                                    width is None):
                        print(f"{ROW}|skipped={label}", flush=True)
                        continue
                    built.append((label, lambda e=entry, g=g, b=b, h=h, t=t, p=pages,
                                  w=width: launch(e, g, b, h, t, p, w)))
    built.append(("undefined_name_body", lambda: wrap_nki(body_that_reads_an_undefined_name)(
        fake((1, 128), torch.float32))))
    print(ROW + "|tree|module=" + MD.__file__, flush=True)
    for name, call in built:
        message = ""
        try:
            with FakeTensorMode():
                call()
        except BaseException as refusal:  # a refusal is a result here, not a test error
            message = " ".join(str(refusal).split())[:2000]
        print(f"{ROW}|entry={name}|refused={bool(message)}|neuron_fds={_open_device_nodes()}"
              f"|diagnostic={message or 'none'}", flush=True)


def _rows() -> list[dict[str, str]]:
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    environment.update(_PIN, PYTHONPATH=str(_ROOT))
    with tempfile.TemporaryDirectory(prefix="dcp_trow_frontend_") as scratch:
        done = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "compile-each-entry"],
            cwd=scratch, env=environment, capture_output=True, text=True, timeout=1800,
            check=False)
    assert done.returncode == 0, done.stderr[-3000:]
    printed = [line for line in done.stdout.splitlines() if line.startswith(ROW + "|")]
    tree = [line for line in printed if line.startswith(ROW + "|tree|")]
    assert tree and f"module={_ROOT}/" in tree[0], tree
    rows = []
    for line in printed:
        if not line.startswith(ROW + "|entry="):
            continue
        head, _, diagnostic = line.partition("|diagnostic=")
        fields = dict(one.split("=", 1) for one in head.split("|")[1:])
        rows.append({**fields, "diagnostic": diagnostic})
    return rows


def test_the_front_end_accepts_both_t_row_kernels() -> None:
    rows = _rows()
    assert {row["neuron_fds"] for row in rows} == {"0"}, rows
    control = [row for row in rows if row["entry"] == "undefined_name_body"]
    assert control and control[0]["refused"] == "True", (
        f"the child accepted a body that reads an undefined name, so it parsed none: {control}")
    entries = [row for row in rows if row["entry"] != "undefined_name_body"]
    # Every served (two-program) geometry is built; the one-program ones the entry admits.
    served = {row["entry"] for row in entries if row["entry"].endswith("_g2")}
    assert len(served) == 2 * len(FRONTEND_BATCHES) * len(FRONTEND_SHAPES), sorted(served)
    refused = [f"{row['entry']}: {row['diagnostic']}" for row in entries
               if row["refused"] != "False"]
    assert not refused, refused


if __name__ == "__main__" and sys.argv[1:] == ["compile-each-entry"]:
    _compile_each_entry()
