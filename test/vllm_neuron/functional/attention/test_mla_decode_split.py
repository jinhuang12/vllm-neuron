# SPDX-License-Identifier: Apache-2.0
"""The key-split decode kernel against the 0a08ff4 decode kernel at the served shapes.

The served per-rank shape is TP=64 of GLM-5.3-Flash: one head, latent 512, page 128,
bf16 query and bank. ctx 1024 is the bypass regime (dense causal prefix in the
2048-row decode bucket, 16 pages); ctx 8192 is the selecting regime (2048 selected
window rows of an 8192-row window, 64 pages). Each case runs at B in {1, 4, 64} on two
programs (LNC2, the served grid) and, at B in {1, 4}, on one program, and compares the
result with the 0a08ff4 kernel loaded from ``test/hardware/baselines/mla_0a08ff4`` and
with the fp32 torch oracle. Every case reads the dispatch counters, so neither a torch
answer nor the general kernel can pass for the split kernel.

Tolerance. Both kernels form the same bf16 products with fp32 sums for the scores and
carry the probabilities to ~16 significand bits (a bf16 hi/lo split) into the value
sum. The split kernel sums each program's half of the keys under its own max and merges
the two halves (and the step's own row) with one fp32 rescale each, so it differs from
0a08ff4 by fp32 summation order and that rescale: relative L2 at most :data:`REL`
(2e-5, the bound ``test_mla_decode_attention.py`` already holds 0a08ff4 to against the
oracle), elementwise ``rtol = 10 * REL`` and ``atol = REL * max|want|``.
"""

from __future__ import annotations

import pytest
import torch

from test.hardware.baselines.mla_0a08ff4 import load as load_0a08ff4
from vllm_neuron.functional.attention import mla_decode as MD

REL = 2e-5
HEADS, LATENT, PAGE = 1, 512, 128
SCALE = 0.0625
#: ctx -> (window pages, selected width or None for the dense bypass).
REGIMES = {1024: (16, None), 8192: (64, 2048)}


def served_case(batch: int, ctx: int, *, seed: int, disjoint: bool = False):
    """Operands for ``batch`` requests at ``ctx``, the served per-rank shape.

    Positions sit just below ``ctx`` and differ per request; request 0 of a batch of
    four or more takes the window's first row (position 0) in the dense regime, so the
    one-row prefix is covered. Selected rows are drawn from each request's causal prefix,
    15% are the ``-1`` sentinel, duplicates are allowed, and request 0 selects its own row.
    Requests draw their pages from one shared pool (they overlap) unless ``disjoint``.
    """
    pages, width = REGIMES[ctx]
    gen = torch.Generator().manual_seed(seed)
    bank_pages = (batch if disjoint else 1) * pages + 24
    bank = torch.randn(bank_pages * PAGE, LATENT, generator=gen).to(torch.bfloat16)
    q = torch.randn(batch, HEADS, LATENT, generator=gen).to(torch.bfloat16)
    written = torch.randn(batch, LATENT, generator=gen).to(torch.bfloat16)
    if disjoint:
        table = torch.randperm(bank_pages, generator=gen)[:batch * pages]
        table = table.reshape(batch, pages).to(torch.int32)
    else:
        table = torch.stack([torch.randperm(bank_pages, generator=gen)[:pages]
                             for _ in range(batch)]).to(torch.int32)
    pos = (ctx - 1 - torch.randint(0, 97, (batch,), generator=gen)).to(torch.int32)
    if width is None and batch >= 4:
        pos[0] = 0
    for b in range(batch):
        table[b, int(pos[b]) // PAGE + 1:] = -1
    topk = None
    if width is not None:
        topk = torch.full((batch, width), -1, dtype=torch.int32)
        for b in range(batch):
            pick = torch.randint(0, int(pos[b]) + 1, (width,), generator=gen)
            keep = torch.rand(width, generator=gen) < 0.85
            topk[b] = torch.where(keep, pick, torch.full_like(pick, -1)).to(torch.int32)
        topk[0, 5] = int(pos[0])
        topk[0, 1500] = int(pos[0])
    return q, bank, table, pos, written, SCALE, PAGE, topk


def run(module, args, monkeypatch, lnc):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    module.reset_mla_decode_dispatch_counters()
    out = module.mla_decode_attention(*args[:7], topk_indices=args[7])
    assert module.mla_decode_dispatch_counters() == (1, 0)
    return out


def check(got, want, label):
    rel = float((got.double() - want.double()).norm() / want.double().norm())
    assert rel <= REL, f"{label}: relative L2 {rel:.3e} > {REL}"
    torch.testing.assert_close(got, want, rtol=REL * 10, atol=REL * float(want.abs().max()))


CASES = [(b, ctx, "2") for b in (1, 4, 64) for ctx in (1024, 8192)]
CASES += [(b, ctx, None) for b in (1, 4) for ctx in (1024, 8192)]


@pytest.mark.parametrize("batch,ctx,lnc", CASES)
def test_split_kernel_matches_0a08ff4_at_the_served_shape(batch, ctx, lnc, monkeypatch):
    args = served_case(batch, ctx, seed=100 * batch + ctx // 1024)
    out = run(MD, args, monkeypatch, lnc)
    programs = 2 if lnc == "2" else 1
    assert MD.mla_decode_split_counts() == (1, 1 if programs == 2 else 0)
    dense = args[7] is None
    assert MD.mla_decode_route_counts()[:2] == ((1, 0) if dense else (0, 1))
    old = run(load_0a08ff4(), args, monkeypatch, lnc)
    check(out, old, f"B={batch} ctx={ctx} lnc={lnc} vs 0a08ff4")
    check(out, MD.mla_decode_attention_torch_oracle(*args), "vs oracle")


@pytest.mark.parametrize("ctx", [1024, 8192])
def test_one_request_runs_on_both_programs(ctx, monkeypatch):
    # The served bs=1 step: two programs, each with half of the request's keys.
    args = served_case(1, ctx, seed=7)
    run(MD, args, monkeypatch, "2")
    assert MD.mla_decode_split_counts() == (1, 1)
    assert MD.mla_decode_route_counts()[2] == 1


@pytest.mark.parametrize("ctx", [1024, 8192])
def test_the_own_row_is_read_from_written_not_the_bank(ctx, monkeypatch):
    # Disjoint pages: a poisoned row belongs to one request's window only.
    args = list(served_case(4, ctx, seed=9, disjoint=True))
    out = run(MD, tuple(args), monkeypatch, "2")
    q, bank, table, pos = args[:4]
    poisoned = bank.clone()
    for b in range(4):
        row = int(table[b, int(pos[b]) // PAGE]) * PAGE + int(pos[b]) % PAGE
        poisoned[row] = 1000.0
    args[1] = poisoned
    again = run(MD, tuple(args), monkeypatch, "2")
    assert torch.equal(out, again)


def test_a_request_that_selects_nothing_returns_zeros(monkeypatch):
    args = list(served_case(2, 8192, seed=13))
    args[7] = args[7].clone()
    args[7][1] = -1
    out = run(MD, tuple(args), monkeypatch, "2")
    assert torch.count_nonzero(out[1]) == 0
    old = run(load_0a08ff4(), tuple(args), monkeypatch, "2")
    assert torch.count_nonzero(old[1]) == 0
    check(out[:1], old[:1], "kept row vs 0a08ff4")


def test_one_and_two_programs_agree(monkeypatch):
    args = served_case(4, 8192, seed=17)
    one = run(MD, args, monkeypatch, None)
    two = run(MD, args, monkeypatch, "2")
    check(two, one, "two programs vs one")


def test_shapes_outside_the_split_kernel_keep_the_general_kernel(monkeypatch):
    # Four heads: the general kernel serves it, and the split counter stays at zero.
    gen = torch.Generator().manual_seed(23)
    q = torch.randn(2, 4, 256, generator=gen).to(torch.bfloat16)
    bank = torch.randn(8 * 128, 256, generator=gen).to(torch.bfloat16)
    table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    pos = torch.tensor([200, 17], dtype=torch.int32)
    written = torch.randn(2, 256, generator=gen).to(torch.bfloat16)
    args = (q, bank, table, pos, written, 256 ** -0.5, 128, None)
    out = run(MD, args, monkeypatch, "2")
    assert MD.mla_decode_split_counts() == (0, 0)
    check(out, MD.mla_decode_attention_torch_oracle(*args), "general kernel vs oracle")


def test_a_large_score_gap_between_the_halves_stays_finite(monkeypatch):
    # Program 1's keys score 4096 against program 0's ~N(0, 22): scale * gap = 256, so a
    # merge that rescaled by exp(scale * (M_other - M_own)) without the shared max would
    # overflow fp32. The window is dense, its own row sits at 2000, so both halves hold keys.
    gen = torch.Generator().manual_seed(29)
    pages = 16
    bank = torch.randn((pages + 8) * PAGE, LATENT, generator=gen).to(torch.bfloat16)
    table = torch.arange(pages, dtype=torch.int32).reshape(1, pages)
    bank[8 * PAGE:pages * PAGE] = 4.0
    q = torch.full((1, HEADS, LATENT), 2.0).to(torch.bfloat16)
    written = torch.randn(1, LATENT, generator=gen).to(torch.bfloat16)
    pos = torch.tensor([2000], dtype=torch.int32)
    args = (q, bank, table, pos, written, SCALE, PAGE, None)
    out = run(MD, args, monkeypatch, "2")
    assert MD.mla_decode_split_counts() == (1, 1)
    assert torch.isfinite(out).all()
    check(out, MD.mla_decode_attention_torch_oracle(*args), "large gap vs oracle")
