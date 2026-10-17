# SPDX-License-Identifier: Apache-2.0
"""DCP mode of ``mla_decode``: per-rank (partial, lse) over a 128-token-block KV shard.

Under decode context parallelism (DCP) each of ``CP`` ranks holds every ``CP``-th
128-token block of a sequence's latent KV, the queries of all ``Hq = CP * H`` heads
(the q all-gather), and attends its own blocks only.
:func:`~vllm_neuron.functional.attention.mla_decode.mla_decode_partial_attention`
returns that rank's softmax-normalised ``partial [Hq, B, L]`` and ``lse [Hq, B]``
(``softmax_scale * max + ln(sum)``); the llama3 LSE merge
(``attention_decode.py:1044-1060``) of the ``CP`` partials is the CP=1 answer.

What is checked, on the CPU simulator, against two CP=1 references -- the fp32 torch
oracle and the kernel's own CP=1 output (``mla_decode_attention`` at ``Hq`` heads):

* CP in {1, 2, 4, 8} (``Hq`` = 1, 2, 4, 8), dense and selected, B in {1, 3}, one program
  and the LNC2 grid, every call on the key-split kernel (both cores under LNC2);
* the owner gate on the step's own row (dense): with every rank counting it, the merge
  is off by orders of magnitude more than the tolerance (the RED half);
* a rank that owns nothing returns the empty partial (0, ``EMPTY_LSE``) exactly;
* selected-mode ownership: ids remapped to local rows or -1, a non-owner's position at
  its owned length (no kernel input), up to the window's end;
* at CP = 1 (``Hq`` = 1, owner all-true) the partial is today's output bit for bit; the
  refusals name the operand (a geometry the key-split kernel does not serve included);
  the NKI front end accepts every new entry.

The merge reference is a local torch copy of the llama3 formula (:func:`lse_merge`),
head-major and in fp32 as the production merge runs; ``dcp_merge.py`` (item 4) is not
imported, so this file stands alone.

Tolerance (derived in ``reports/dcp_item5a.md`` section 5). The merge identity is exact:
``exp(lse_r) * partial_r`` is rank ``r``'s unnormalised numerator, so the merged result is
the CP=1 ratio of numerator and denominator sums, summed by subset. Its error against the
fp32 oracle is (1) the kernel arithmetic a CP=1 call has (exact MM1 products, fp32 sums,
the 16-bit hi/lo split of MM2), bounded by :data:`REL`, the bound the repo holds the CP=1
kernel to (``test_mla_decode_attention.py``); and (2) the roundings that do not cancel
between a rank's partial and its lse, plus the merge's own, at most ``U32 * (5 * max|lse| +
8 * ln K + 3 * ln CP + 2 * CP + 6)`` for ``K`` columns (:func:`merge_eta`). Both are
taken relative to the weighted partials, so the bound is scaled by the cancellation factor
``kappa = sum_r |w_r partial_r| / |merged|`` (>= 1, computed from the outputs). The CP=1
kernel is itself inside that bound of the oracle, so against it the bound doubles.
"""

from __future__ import annotations

import math
import os
import pathlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import pytest
import torch

from vllm_neuron.functional.attention import mla_decode as MD

#: The CP=1 kernel's relative-L2 bound against the fp32 oracle (test_mla_decode_attention.py).
REL = 2e-5
#: fp32 unit roundoff.
U32 = 2.0 ** -24
#: The DCP interleave: rank ``k % CP`` owns the sequence's 128-token block ``k``.
INTERLEAVE = 128
LATENT, PAGE = 512, 128
SCALE = 0.0625


@dataclass
class GlobalCase:
    q: torch.Tensor          # [B, Hq, L] the gathered queries
    bank: torch.Tensor       # [slots, L]
    table: torch.Tensor      # [B, pages] the CP=1 block table
    position: torch.Tensor   # [B] this step's global window row
    written: torch.Tensor    # [B, L]
    topk: torch.Tensor | None  # [B, width] global window rows, -1 = no token
    page: int


@dataclass
class Shard:
    table: torch.Tensor      # [B, local_pages]
    position: torch.Tensor   # [B] the token's local row (owner) or the owned length
    owner: torch.Tensor      # [B] int32, 1 where this rank holds the step's token
    topk: torch.Tensor | None


def global_case(batch, heads, *, pages, positions, seed, selected=False, width=2048,
                page=PAGE, latent=LATENT, bank_pages=None) -> GlobalCase:
    """A CP=1 decode step: ``batch`` requests over ``pages``-page windows of one bank."""
    gen = torch.Generator().manual_seed(seed)
    bank_pages = bank_pages or pages + 8
    bank = torch.randn(bank_pages * page, latent, generator=gen).to(torch.bfloat16)
    q = torch.randn(batch, heads, latent, generator=gen).to(torch.bfloat16)
    table = torch.stack([torch.randperm(bank_pages, generator=gen)[:pages]
                         for _ in range(batch)]).to(torch.int32)
    pos = torch.as_tensor(positions, dtype=torch.int32)
    for b in range(batch):
        table[b, int(pos[b]) // page + 1:] = -1
    written = torch.randn(batch, latent, generator=gen).to(torch.bfloat16)
    topk = None
    if selected:
        topk = torch.full((batch, width), -1, dtype=torch.int32)
        for b in range(batch):
            pick = torch.randint(0, int(pos[b]) + 1, (width,), generator=gen)
            keep = torch.rand(width, generator=gen) < 0.85
            topk[b] = torch.where(keep, pick, torch.full_like(pick, -1)).to(torch.int32)
            topk[b, 5] = int(pos[b])  # this step's own row, selected twice
            topk[b, width - 7] = int(pos[b])
    return GlobalCase(q, bank, table, pos, written, topk, page)


def shard(case: GlobalCase, cp: int, rank: int, local_pages: int | None = None) -> Shard:
    """Rank ``rank``'s view of ``case`` under a ``CP``-way 128-token-block interleave.

    Block ``k`` of a window (rows ``k*128 .. k*128+127``) belongs to rank ``k % CP`` and
    is that rank's local block ``k // CP``. The step's token at global row ``P`` sits in
    block ``P // 128``: its owner attends local rows before the token's local row and
    counts the token's own row; every other rank attends all its blocks before the
    token's block (``position`` = their row count, a multiple of 128) and owns nothing of
    this step. Selected ids map to local rows, or to -1 where another rank holds them.
    """
    page = case.page
    per_block = INTERLEAVE // page
    batch, pages = (int(d) for d in case.table.shape)
    blocks = -(-pages // per_block)
    if local_pages is None:
        local_pages = -(-blocks // cp) * per_block
    table = torch.full((batch, local_pages), -1, dtype=torch.int32)
    for j in range(pages):
        block = j // per_block
        if block % cp == rank:
            table[:, (block // cp) * per_block + j % per_block] = case.table[:, j]
    position = torch.empty(batch, dtype=torch.int32)
    owner = torch.empty(batch, dtype=torch.int32)
    for b in range(batch):
        row = int(case.position[b])
        block = row // INTERLEAVE
        mine = block % cp == rank
        owner[b] = int(mine)
        if mine:
            position[b] = (block // cp) * INTERLEAVE + row % INTERLEAVE
        else:
            position[b] = INTERLEAVE * sum(1 for k in range(block) if k % cp == rank)
    topk = None
    if case.topk is not None:
        ids = case.topk.to(torch.int64)
        block = ids.div(INTERLEAVE, rounding_mode="floor")
        local = (block // cp) * INTERLEAVE + ids % INTERLEAVE
        topk = torch.where((ids >= 0) & (block % cp == rank), local,
                           torch.full_like(ids, -1)).to(torch.int32)
    return Shard(table, position, owner, topk)


def lse_merge(partials: torch.Tensor, lses: torch.Tensor) -> torch.Tensor:
    """The llama3 DCP LSE merge (``attention_decode.py:1044-1060``), head-major, fp32.

    ``partials [CP, Hq, R, L]`` and ``lses [CP, Hq, R]`` -> ``[Hq, R, L]``: each rank's
    partial rescaled by ``exp(lse_r - logsumexp_r(lse))`` and summed (the reduce-scatter's
    sum). A local copy, not ``dcp_merge.py``.
    """
    merged_lse = torch.logsumexp(lses, dim=0)
    return (torch.exp(lses - merged_lse).unsqueeze(-1) * partials).sum(0)


def merge_eta(lses: torch.Tensor, cp: int, columns: int) -> float:
    """The merge's own relative error bound (section 5 of the report): ``max|lse|`` over
    the ranks that attend a column, ``columns`` the CP=1 column count ``K``."""
    live = lses[lses > MD.EMPTY_LSE / 2]
    big = float(live.abs().max()) if live.numel() else 0.0
    return U32 * (5.0 * big + 8.0 * math.log(columns) + 3.0 * math.log(cp) + 2.0 * cp + 6.0)


def kappa(partials: torch.Tensor, lses: torch.Tensor, merged: torch.Tensor) -> float:
    """``sum_r |w_r * partial_r| / |merged|``: how much the merge's sum cancels (>= 1)."""
    weights = torch.exp(lses.double() - torch.logsumexp(lses.double(), dim=0))
    weighted = weights.unsqueeze(-1) * partials.double()
    return float(sum(weighted[r].norm() for r in range(int(partials.shape[0])))
                 / merged.double().norm())


def relative(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got.double() - want.double()).norm() / want.double().norm())


def _grid(monkeypatch, lnc):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)


def run_ranks(case: GlobalCase, cp: int, monkeypatch, lnc=None, owner_override=None):
    """Every rank's (partial, lse), stacked; asserts one NKI launch per rank."""
    _grid(monkeypatch, lnc)
    partials, lses, routes = [], [], []
    for rank in range(cp):
        part = shard(case, cp, rank)
        owner = part.owner if owner_override is None else owner_override
        MD.reset_mla_decode_dispatch_counters()
        partial, lse = MD.mla_decode_partial_attention(
            case.q, case.bank, part.table, part.position, case.written, owner, SCALE,
            case.page, topk_indices=part.topk)
        assert MD.mla_decode_dispatch_counters() == (1, 0)
        assert MD.mla_decode_partial_counts()[0] == 1
        routes.append(MD.mla_decode_split_counts())
        heads, batch = int(case.q.shape[1]), int(case.q.shape[0])
        assert partial.dtype == torch.float32 and tuple(partial.shape) == (heads, batch, LATENT)
        assert lse.dtype == torch.float32 and tuple(lse.shape) == (heads, batch)
        partials.append(partial)
        lses.append(lse)
    return torch.stack(partials), torch.stack(lses), routes


def cp1_kernel(case: GlobalCase, monkeypatch, lnc=None) -> torch.Tensor:
    """The CP=1 kernel at ``Hq`` heads, head-major."""
    _grid(monkeypatch, lnc)
    out = MD.mla_decode_attention(case.q, case.bank, case.table, case.position,
                                  case.written, SCALE, case.page, case.topk)
    return out.permute(1, 0, 2)


def cp1_oracle(case: GlobalCase) -> torch.Tensor:
    return MD.mla_decode_attention_torch_oracle(
        case.q, case.bank, case.table, case.position, case.written, SCALE, case.page,
        case.topk).permute(1, 0, 2)


def check_merged(case, partials, lses, cp, monkeypatch, lnc, label):
    merged = lse_merge(partials, lses)
    width = int(case.topk.shape[1]) if case.topk is not None else int(case.table.shape[1]) * case.page
    eta = merge_eta(lses, cp, width + 1)
    k = kappa(partials, lses, cp1_oracle(case))
    bound_oracle = (REL + eta) * k
    got = relative(merged, cp1_oracle(case))
    assert got <= bound_oracle, f"{label}: vs oracle {got:.3e} > {bound_oracle:.3e} (kappa {k:.2f})"
    want = cp1_kernel(case, monkeypatch, lnc)
    bound_kernel = 2.0 * bound_oracle
    got_k = relative(merged, want)
    assert got_k <= bound_kernel, f"{label}: vs CP=1 kernel {got_k:.3e} > {bound_kernel:.3e}"
    torch.testing.assert_close(merged, want, rtol=10 * bound_kernel,
                               atol=bound_kernel * float(want.abs().max()))
    return got, bound_oracle


# ---- CP partials merge to the CP=1 answer ----------------------------------------------

#: ctx 2047 / 1500 / 300 in a 2048-row dense window (16 pages), and ctx 8191 / 5000 /
#: 2100 in an 8192-row window with 2048 selected rows (64 pages).
DENSE_POS, SELECTED_POS = [2047, 1500, 300], [8191, 5000, 2100]
MERGE_CASES = [(cp, sel, b, lnc) for cp in (1, 2, 4, 8) for sel in (False, True)
               for b in (1, 3) for lnc in (None, "2")]


@pytest.mark.parametrize("cp,selected,batch,lnc", MERGE_CASES)
def test_cp_partials_merge_to_the_cp1_answer(cp, selected, batch, lnc, monkeypatch):
    positions = (SELECTED_POS if selected else DENSE_POS)[:batch]
    case = global_case(batch, cp, pages=64 if selected else 16, positions=positions,
                       seed=1000 * cp + 10 * batch + selected, selected=selected)
    partials, lses, routes = run_ranks(case, cp, monkeypatch, lnc)
    check_merged(case, partials, lses, cp, monkeypatch, lnc,
                 f"CP={cp} {'selected' if selected else 'dense'} B={batch} lnc={lnc}")
    # The key-split kernel serves every Hq, B = 1 included, on both cores under LNC2.
    assert routes == [(1, 1 if lnc == "2" else 0)] * cp, routes


def test_a_large_score_gap_keeps_the_merge_inside_the_derived_bound(monkeypatch):
    # Block 1 (rank 1 at CP=2) scores 4096 against N(0, 22) elsewhere: scale * max = 256,
    # so |lse| ~ 256 and the merge's term U32 * 3 * |lse| is the bound's largest part.
    gen = torch.Generator().manual_seed(29)
    case = global_case(1, 2, pages=16, positions=[2000], seed=29)
    case.table = torch.arange(16, dtype=torch.int32).reshape(1, 16)
    case.bank = torch.randn(24 * PAGE, LATENT, generator=gen).to(torch.bfloat16)
    case.bank[1 * PAGE:2 * PAGE] = 4.0
    case.q = torch.full((1, 2, LATENT), 2.0).to(torch.bfloat16)
    partials, lses, _ = run_ranks(case, 2, monkeypatch, "2")
    assert float(lses.max()) > 200.0
    check_merged(case, partials, lses, 2, monkeypatch, "2", "large gap")


# ---- the owner gate (dense) -------------------------------------------------------------

@pytest.mark.parametrize("lnc", [None, "2"])
def test_only_the_owner_counts_the_own_row(lnc, monkeypatch):
    # GREEN: the shard's owner input. RED: every rank counting the step's own row (the
    # dense kernel without an owner input) double-counts it in the merge.
    case = global_case(1, 2, pages=16, positions=[1500], seed=41)
    partials, lses, _ = run_ranks(case, 2, monkeypatch, lnc)
    good, bound = check_merged(case, partials, lses, 2, monkeypatch, lnc, "owner gate")
    everyone = torch.ones(1, dtype=torch.int32)
    partials_bad, lses_bad, _ = run_ranks(case, 2, monkeypatch, lnc, owner_override=everyone)
    bad = relative(lse_merge(partials_bad, lses_bad), cp1_oracle(case))
    assert bad > 100 * bound, f"own row counted twice: {bad:.3e} is not >> {bound:.3e}"


@pytest.mark.parametrize("lnc", [None, "2"])
@pytest.mark.parametrize("selected", [False, True])
def test_a_rank_that_owns_nothing_returns_the_empty_partial(selected, lnc, monkeypatch):
    # The token sits in block 0, so rank 1 of 2 holds no row of this request yet.
    case = global_case(1, 2, pages=64 if selected else 16, positions=[37], seed=43,
                       selected=selected, width=256)
    partials, lses, _ = run_ranks(case, 2, monkeypatch, lnc)
    rank1 = shard(case, 2, 1)
    assert int(rank1.owner[0]) == 0 and int(rank1.position[0]) == 0
    assert torch.count_nonzero(partials[1]) == 0
    assert torch.equal(lses[1], torch.full_like(lses[1], MD.EMPTY_LSE))
    assert MD.EMPTY_LSE == -1.0e30
    check_merged(case, partials, lses, 2, monkeypatch, lnc, "empty rank")


def test_every_rank_empty_merges_to_zeros(monkeypatch):
    # A padded request: no rank holds a row and none owns the token. lse = -1e30, not -inf,
    # so the merge gives 0, not NaN.
    case = global_case(1, 2, pages=16, positions=[0], seed=47)
    _grid(monkeypatch, "2")
    outs = []
    for rank in range(2):
        part = shard(case, 2, rank)
        outs.append(MD.mla_decode_partial_attention(
            case.q, case.bank, torch.full_like(part.table, -1), torch.zeros(1, dtype=torch.int32),
            case.written, torch.zeros(1, dtype=torch.int32), SCALE, PAGE))
    merged = lse_merge(torch.stack([o[0] for o in outs]), torch.stack([o[1] for o in outs]))
    assert torch.isfinite(merged).all() and torch.count_nonzero(merged) == 0


# ---- selected-mode ownership -------------------------------------------------------------

@pytest.mark.parametrize("lnc", [None, "2"])
def test_selected_ids_remap_to_local_rows_and_a_full_window_is_accepted(lnc, monkeypatch):
    # 3 pages at CP=2: rank 0 holds blocks 0 and 2 (the token's), rank 1 holds block 1 in
    # a one-page window, so its position (owned length 128) equals its window.
    case = global_case(1, 2, pages=3, positions=[300], seed=53, selected=True, width=512)
    _grid(monkeypatch, lnc)
    partials, lses = [], []
    for rank in range(2):
        pages = 2 if rank == 0 else 1
        part = shard(case, 2, rank, local_pages=pages)
        if rank == 1:
            assert int(part.position[0]) == pages * PAGE and int(part.owner[0]) == 0
            assert int(part.topk.max()) < pages * PAGE
        partial, lse = MD.mla_decode_partial_attention(
            case.q, case.bank, part.table, part.position, case.written, part.owner, SCALE,
            PAGE, topk_indices=part.topk)
        partials.append(partial)
        lses.append(lse)
    check_merged(case, torch.stack(partials), torch.stack(lses), 2, monkeypatch, lnc,
                 "selected remap")


@pytest.mark.parametrize("lnc", [None, "2"])
def test_a_dense_non_owner_at_its_window_end_attends_every_row(lnc, monkeypatch):
    # 5 blocks at CP=2, the token in block 4: rank 0 holds blocks 0, 2, 4 (a 4-page window,
    # one pad), rank 1 holds blocks 1 and 3 in a 2-page window, so its position (owned
    # length 256) equals its window.
    case = global_case(1, 2, pages=5, positions=[552], seed=59)
    _grid(monkeypatch, lnc)
    partials, lses = [], []
    for rank in range(2):
        pages = 4 if rank == 0 else 2
        part = shard(case, 2, rank, local_pages=pages)
        if rank == 1:
            assert int(part.position[0]) == pages * PAGE and int(part.owner[0]) == 0
        partial, lse = MD.mla_decode_partial_attention(
            case.q, case.bank, part.table, part.position, case.written, part.owner, SCALE,
            PAGE)
        partials.append(partial)
        lses.append(lse)
    check_merged(case, torch.stack(partials), torch.stack(lses), 2, monkeypatch, lnc,
                 "dense full window")


# ---- CP = 1 is today's kernel --------------------------------------------------------------

@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("lnc", [None, "2"])
@pytest.mark.parametrize("selected", [False, True])
def test_cp1_partial_is_todays_output_bitwise(selected, lnc, batch, monkeypatch):
    # Hq = 1, every owner 1: the partial kernel computes today's values in today's order
    # (under LNC2 each core normalises its latent half by its own sum, as today, and the
    # lse is core 0's).
    case = global_case(batch, 1, pages=64 if selected else 16,
                       positions=(SELECTED_POS if selected else DENSE_POS)[:batch], seed=67,
                       selected=selected)
    partials, lses, routes = run_ranks(case, 1, monkeypatch, lnc)
    assert routes == [(1, 1 if lnc == "2" else 0)]
    today = cp1_kernel(case, monkeypatch, lnc)
    assert torch.equal(partials[0], today)
    assert torch.equal(lse_merge(partials, lses), partials[0])


# ---- refusals -------------------------------------------------------------------------------

def _refusal_case():
    case = global_case(2, 2, pages=4, positions=[300, 40], seed=71, selected=True, width=128)
    part = shard(case, 2, 0)
    return case, part


def test_refusals_name_the_operand():
    case, part = _refusal_case()
    args = (case.q, case.bank, part.table, part.position, case.written)
    call = MD.mla_decode_partial_attention
    with pytest.raises(MD.MlaDecodeAttentionError, match=r"owner must be \[batch\]"):
        call(*args, part.owner[:1], SCALE, PAGE)
    with pytest.raises(MD.MlaDecodeAttentionError, match="owner"):
        call(*args, part.owner + 2, SCALE, PAGE)
    window = int(part.table.shape[1]) * PAGE
    owner_full = torch.tensor([1, 1], dtype=torch.int32)
    with pytest.raises(MD.MlaDecodeAttentionError, match="owner's position"):
        call(case.q, case.bank, part.table, torch.tensor([window, 0], dtype=torch.int32),
             case.written, owner_full, SCALE, PAGE)
    with pytest.raises(MD.MlaDecodeAttentionError, match="owned length"):
        call(case.q, case.bank, part.table, torch.tensor([window + 1, 0], dtype=torch.int32),
             case.written, torch.zeros(2, dtype=torch.int32), SCALE, PAGE)
    selects_position = torch.full((2, 128), -1, dtype=torch.int32)
    selects_position[1, 3] = 17
    with pytest.raises(MD.MlaDecodeAttentionError, match="non-owner"):
        call(case.q, case.bank, part.table, torch.tensor([0, 17], dtype=torch.int32),
             case.written, torch.tensor([1, 0], dtype=torch.int32), SCALE, PAGE,
             topk_indices=selects_position)


def test_the_cp1_call_keeps_todays_one_past_the_window_refusal():
    case, part = _refusal_case()
    window = int(part.table.shape[1]) * PAGE
    with pytest.raises(MD.MlaDecodeAttentionError, match="position"):
        MD.mla_decode_attention(case.q, case.bank, part.table,
                                torch.tensor([window, 0], dtype=torch.int32), case.written,
                                SCALE, PAGE)


@pytest.mark.parametrize("lnc", [None, "2"])
def test_a_geometry_the_key_split_kernel_does_not_serve_is_refused(lnc, monkeypatch):
    _grid(monkeypatch, lnc)
    small = global_case(1, 2, pages=8, positions=[300], seed=73, page=64)
    part = shard(small, 2, 0)
    with pytest.raises(MD.MlaDecodeAttentionError, match="key-split kernel.*page 64"):
        MD.mla_decode_partial_attention(small.q, small.bank, part.table, part.position,
                                        small.written, part.owner, SCALE, 64)
    wide = global_case(1, MD.GROUP + 1, pages=4, positions=[300], seed=79)
    part = shard(wide, 2, 0)
    with pytest.raises(MD.MlaDecodeAttentionError, match=f"{MD.GROUP + 1} heads"):
        MD.mla_decode_partial_attention(wide.q, wide.bank, part.table, part.position,
                                        wide.written, part.owner, SCALE, PAGE)


# ---- the NKI front end ----------------------------------------------------------------------

ROW = "dcp_frontend"
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "PYTHONDONTWRITEBYTECODE": "1"}
#: Front-end entries: (kernel, local window pages, selected width or None, batch, heads).
#: Windows: the dense 2048-row bucket's CP=2 half (8 pages) and an 8192-row window's
#: CP=2 half (32 pages) with 2048 selected rows, the served per-rank shapes at CP=2.
FRONTEND_HEADS = (1, 2, 4, 8)
FRONTEND_BATCHES = (1, 4)


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

    def launch(entry, programs, b, h, pages, width):
        call = wrap_nki(entry)
        call = call[2] if programs == 2 else call
        args = [fake((b, h, LATENT), bf), fake((64 * PAGE, LATENT), bf),
                fake((b, pages), i32), fake((b,), i32), fake((b, LATENT), bf)]
        if width is None:
            args.append(fake((b,), i32))
        else:
            args.append(fake((b, width), i32))
        return call(*args, SCALE, PAGE, MD.SOURCE_DIGEST)

    built = []
    kernels = (("dense_split", MD.mla_decode_dense_split_partial_kernel, 8, None),
               ("selected_split", MD.mla_decode_selected_split_partial_kernel, 32, 2048))
    for name, entry, pages, width in kernels:
        for b in FRONTEND_BATCHES:
            for h in FRONTEND_HEADS:
                for g in (1, 2):
                    built.append((f"{name}_b{b}_h{h}_g{g}",
                                  lambda e=entry, g=g, b=b, h=h, p=pages, w=width:
                                  launch(e, g, b, h, p, w)))
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
    with tempfile.TemporaryDirectory(prefix="dcp_frontend_") as scratch:
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


def test_the_front_end_accepts_every_dcp_entry() -> None:
    rows = _rows()
    assert {row["neuron_fds"] for row in rows} == {"0"}, rows
    control = [row for row in rows if row["entry"] == "undefined_name_body"]
    assert control and control[0]["refused"] == "True", (
        f"the child accepted a body that reads an undefined name, so it parsed none: {control}")
    entries = [row for row in rows if row["entry"] != "undefined_name_body"]
    assert len(entries) == 2 * len(FRONTEND_BATCHES) * len(FRONTEND_HEADS) * 2, len(entries)
    refused = [f"{row['entry']}: {row['diagnostic']}" for row in entries
               if row["refused"] != "False"]
    assert not refused, refused


if __name__ == "__main__" and sys.argv[1:] == ["compile-each-entry"]:
    _compile_each_entry()
