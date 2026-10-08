# SPDX-License-Identifier: Apache-2.0
"""``mla_decode_attention`` with ``T`` query rows per request: the verify step's form.

A speculative verify step hands the decode attention ``T = 1 + k`` rows per request at
positions ``start .. start + T - 1``, in one call. The reference is the path it must
equal: ``T`` sequential single-row steps, each writing its latent row into the bank
before the next one attends. The T-row call sees none of those writes -- it reads the
step's own rows off ``written`` -- so the comparison is bit for bit, and it proves the
per-row causal limit, the overlay of earlier rows of the same step, and the own-row
stand-in at once. Every case reads the dispatch counters, so a torch answer cannot
pass for the kernel.

Two one-row kernels serve ``T = 1``: at the served per-rank shape (one head, latent 512,
128-row pages) the key-split kernel of ``test_mla_decode_split.py``, elsewhere the general
kernel, which is commit ea04c81's bit for bit. A T-row step always runs the general
kernel, so it equals ``T`` sequential general-kernel steps bit for bit and ``T``
sequential key-split steps within that kernel's own bound (:data:`REL`, the bound
``test_mla_decode_split.py`` holds it to against the general kernel).
"""

from __future__ import annotations

import pytest
import torch

from test.hardware.baselines.mla_decode_ea04c81 import load as load_ea04c81
from vllm_neuron.functional.attention import mla_decode as MD

LATENT, PAGE = 512, 128
ROWS = (2, 4, 6)
BATCHES = (1, 4)
#: The key-split kernel's relative bound against the general kernel (test_mla_decode_split.py).
REL = 2e-5


def _case(batch, rows, heads, latent, pages, page, *, seed, selected, width=None,
          starts=None):
    """One step of ``batch`` requests x ``rows`` rows. ``starts[b]`` is row 0's position.

    Each request owns its pages (no two tables share one), as a served batch does, so a
    request's rows can only reach the bank through its own table.
    """
    gen = torch.Generator().manual_seed(seed)
    bank_pages = batch * pages + 2
    bank = (torch.randn(bank_pages * page, latent, generator=gen)).to(torch.bfloat16)
    q = (torch.randn(batch * rows, heads, latent, generator=gen)).to(torch.bfloat16)
    table = torch.randperm(bank_pages, generator=gen)[:batch * pages].reshape(
        batch, pages).to(torch.int32)
    if starts is None:
        starts = torch.randint(0, pages * page - rows, (batch,), generator=gen)
    start = torch.as_tensor(starts, dtype=torch.int32)
    for b in range(batch):
        table[b, (int(start[b]) + rows - 1) // page + 1:] = -1
    written = torch.randn(batch * rows, latent, generator=gen).to(torch.bfloat16)
    topk = None
    if selected:
        topk = torch.full((batch * rows, width), -1, dtype=torch.int32)
        for b in range(batch):
            for t in range(rows):
                live = int(start[b]) + t + 1
                pick = torch.randint(0, live, (width,), generator=gen)
                keep = torch.rand(width, generator=gen) < 0.85
                row = torch.where(keep, pick, torch.full_like(pick, -1)).to(torch.int32)
                # This step's own rows, selected on purpose: the row's own position and
                # every earlier row of the step, one of them twice.
                for j in range(t + 1):
                    row[j] = int(start[b]) + j
                row[t + 1] = int(start[b])
                topk[b * rows + t] = row
    scale = latent ** -0.5
    return q, bank, table, start, written, scale, page, topk


def _slot(table, page, position):
    return int(table[position // page]) * page + position % page


def _sequential(module, q, bank, table, start, written, scale, page, topk):
    """``rows`` single-row steps per request, each writing its row before the next."""
    batch = int(table.shape[0])
    rows = int(q.shape[0]) // batch
    out = torch.empty(batch * rows, int(q.shape[1]), int(q.shape[2]), dtype=torch.float32)
    for b in range(batch):
        live = bank.clone()
        for t in range(rows):
            row = b * rows + t
            pos = start[b:b + 1] + t
            out[row] = module.mla_decode_attention(
                q[row:row + 1], live, table[b:b + 1], pos, written[row:row + 1], scale,
                page, None if topk is None else topk[row:row + 1])[0]
            live[_slot(table[b], page, int(pos))] = written[row]
    return out


def _sequential_general(monkeypatch, *args):
    """The sequential reference through the general kernel only, however the shape serves."""
    with monkeypatch.context() as m:
        m.setattr(MD, "_split_serves", lambda *_: False)
        MD.reset_mla_decode_dispatch_counters()
        out = _sequential(MD, *args)
    assert MD.mla_decode_split_counts() == (0, 0)
    return out


def _within_split_bound(got, want):
    torch.testing.assert_close(got, want, rtol=10 * REL, atol=REL * float(want.abs().max()))


def _run(args, monkeypatch, lnc=None):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    MD.reset_mla_decode_dispatch_counters()
    out = MD.mla_decode_attention(*args[:5], args[5], args[6], args[7])
    assert MD.mla_decode_dispatch_counters() == (1, 0)
    return out


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("rows", ROWS)
def test_dense_rows_equal_sequential_single_row_steps_bit_for_bit(rows, batch, monkeypatch):
    # The bypass regime's window (2048 rows): row 0 at the window's start, mid-window,
    # and against its last page.
    starts = [1017, 0, 2048 - rows, 511][:batch]
    args = _case(batch, rows, 1, LATENT, 16, PAGE, seed=11 + rows + batch,
                 selected=False, starts=starts)
    out = _run(args, monkeypatch)
    assert MD.mla_decode_route_counts() == (1, 0, 0)
    assert MD.mla_decode_split_counts() == (0, 0)
    assert out.shape == (batch * rows, 1, LATENT)
    want = _sequential_general(monkeypatch, *args)
    assert torch.equal(out, want)
    # As dispatched, the one-row steps at this served shape are the key-split kernel's.
    MD.reset_mla_decode_dispatch_counters()
    served = _sequential(MD, *args)
    assert MD.mla_decode_split_counts() == (batch * rows, 0)
    _within_split_bound(out, served)
    torch.testing.assert_close(out, MD.mla_decode_attention_torch_oracle(*args), rtol=2e-4,
                               atol=2e-5 * float(want.abs().max()))


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("rows", ROWS)
def test_selected_rows_equal_sequential_single_row_steps_bit_for_bit(rows, batch,
                                                                     monkeypatch):
    # The selecting regime: a 4096-row window, 2048 selected rows (index_topk), with
    # -1s, duplicates, and this step's rows among them.
    starts = [4096 - rows, 3000, 2052, 4000][:batch]
    args = _case(batch, rows, 1, LATENT, 32, PAGE, seed=21 + rows + batch,
                 selected=True, width=2048, starts=starts)
    out = _run(args, monkeypatch)
    assert MD.mla_decode_route_counts() == (0, 1, 0)
    assert MD.mla_decode_split_counts() == (0, 0)
    want = _sequential_general(monkeypatch, *args)
    assert torch.equal(out, want)
    MD.reset_mla_decode_dispatch_counters()
    served = _sequential(MD, *args)
    assert MD.mla_decode_split_counts() == (batch * rows, 0)
    _within_split_bound(out, served)
    torch.testing.assert_close(out, MD.mla_decode_attention_torch_oracle(*args), rtol=2e-4,
                               atol=2e-5 * float(want.abs().max()))


@pytest.mark.parametrize("selected", [False, True])
def test_heads_and_small_pages_with_rows(selected, monkeypatch):
    # Four heads, latent 256, 16-row pages: several pieces per 128-row chunk, T = 3.
    args = _case(3, 3, 4, 256, 16, 16, seed=41, selected=selected, width=256)
    out = _run(args, monkeypatch)
    MD.reset_mla_decode_dispatch_counters()
    want = _sequential(MD, *args)
    assert MD.mla_decode_split_counts() == (0, 0)
    assert torch.equal(out, want)


@pytest.mark.parametrize("selected", [False, True])
def test_two_programs_agree_with_one_at_rows(selected, monkeypatch):
    args = _case(4, 4, 1, LATENT, 8, PAGE, seed=31, selected=selected, width=512)
    one = _run(args, monkeypatch)
    two = _run(args, monkeypatch, lnc="2")
    assert MD.mla_decode_route_counts()[2] == 1
    assert torch.equal(one, two)


@pytest.mark.parametrize("selected", [False, True])
def test_the_steps_own_rows_are_read_from_written_not_the_bank(selected, monkeypatch):
    args = list(_case(2, 4, 1, LATENT, 4, PAGE, seed=51, selected=selected, width=128))
    out = _run(tuple(args), monkeypatch)
    q, bank, table, start, written = args[:5]
    poisoned = bank.clone()
    for b in range(2):
        for t in range(4):
            poisoned[_slot(table[b], PAGE, int(start[b]) + t)] = 1000.0
    args[1] = poisoned
    assert torch.equal(out, _run(tuple(args), monkeypatch))


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("batch", BATCHES)
def test_one_row_is_commit_ea04c81s_kernel_bit_for_bit(batch, selected, monkeypatch):
    # Two heads: a shape the key-split kernel does not serve, so the one-row call is the
    # general kernel -- ea04c81's, bit for bit.
    before = load_ea04c81().mla_decode
    pages, width = (32, 2048) if selected else (16, None)
    starts = ([4095, 3000, 2052, 4000] if selected else [2047, 1000, 2000, 517])[:batch]
    args = _case(batch, 1, 2, LATENT, pages, PAGE, seed=61 + batch,
                 selected=selected, width=width, starts=starts)
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    want = before.mla_decode_attention(*args[:5], args[5], args[6], args[7])
    out = _run(args, monkeypatch)
    assert MD.mla_decode_split_counts() == (0, 0)
    assert torch.equal(out, want)


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("batch", BATCHES)
def test_one_row_at_the_served_shape_is_the_key_split_kernel_within_its_bound(
        batch, selected, monkeypatch):
    # One head at the served shape: the one-row call is the key-split kernel
    # (test_mla_decode_split.py pins it), within REL of ea04c81's general kernel.
    before = load_ea04c81().mla_decode
    pages, width = (64, 2048) if selected else (16, None)
    starts = ([8191, 3000, 2052, 4000] if selected else [2047, 1000, 2000, 517])[:batch]
    args = _case(batch, 1, 1, LATENT, pages, PAGE, seed=61 + batch,
                 selected=selected, width=width, starts=starts)
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    want = before.mla_decode_attention(*args[:5], args[5], args[6], args[7])
    out = _run(args, monkeypatch)
    assert MD.mla_decode_split_counts() == (1, 0)
    _within_split_bound(out, want)


@pytest.mark.parametrize("selected", [False, True])
def test_rows_keep_the_general_kernel_where_the_split_kernel_serves_one_row(
        selected, monkeypatch):
    # The served shape on two programs: one row per request is the key-split kernel on
    # both programs; T rows per request is the general kernel, the requests split.
    pages, width = (64, 2048) if selected else (16, None)
    one_row = _case(4, 1, 1, LATENT, pages, PAGE, seed=71, selected=selected, width=width)
    _run(one_row, monkeypatch, lnc="2")
    assert MD.mla_decode_split_counts() == (1, 1)
    args = _case(4, 4, 1, LATENT, pages, PAGE, seed=71, selected=selected, width=width)
    out = _run(args, monkeypatch, lnc="2")
    assert MD.mla_decode_split_counts() == (0, 0)
    assert MD.mla_decode_route_counts()[2] == 1
    assert torch.equal(out, _sequential_general(monkeypatch, *args))


def test_refusals_name_the_rows():
    q, bank, table, start, written, scale, page, _ = _case(
        2, 3, 1, LATENT, 4, PAGE, seed=71, selected=False)
    with pytest.raises(MD.MlaDecodeAttentionError, match="whole number of rows"):
        MD.mla_decode_attention(q[:5], bank, table, start, written[:5], scale, page)
    with pytest.raises(MD.MlaDecodeAttentionError, match="written"):
        MD.mla_decode_attention(q, bank, table, start, written[:2], scale, page)
    with pytest.raises(MD.MlaDecodeAttentionError, match="rows must index"):
        MD.mla_decode_attention(q, bank, table, start + 4 * PAGE - 2, written, scale, page)
    with pytest.raises(MD.MlaDecodeAttentionError, match="topk_indices"):
        MD.mla_decode_attention(q, bank, table, start, written, scale, page,
                                torch.zeros((2, 128), dtype=torch.int32))
