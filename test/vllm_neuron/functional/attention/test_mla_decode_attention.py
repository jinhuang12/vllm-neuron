# SPDX-License-Identifier: Apache-2.0
"""``mla_decode_attention``: batched decode attention over a paged latent bank.

Checked against the fp32 torch oracle and against the 5938748 ``mla_sparse_attention``
kernel run one request at a time (its only form: one block table per call), at
B in {1, 4}, dense and selected, on the served per-rank shape (one head, latent 512,
page 128) and on a small shape that reaches the multi-head and sub-128-page branches.
Every case reads the dispatch counters, so a torch answer cannot pass for the kernel.
"""

from __future__ import annotations

import pytest
import torch

from test.hardware.baselines.dsa_5938748 import load as load_5938748
from vllm_neuron.functional.attention import mla_decode as MD

#: MM1 is exact (bf16 products, fp32 sums) and MM2 carries ~16 significand bits of the
#: probabilities, so the kernel matches an fp32 oracle and the fp32 sparse kernel to
#: float32 summation-order noise plus that split.
REL = 2e-5


def _case(batch, heads, latent, pages, page, bank_pages, *, seed, selected, width=None,
          positions=None):
    gen = torch.Generator().manual_seed(seed)
    bank = (torch.randn(bank_pages * page, latent, generator=gen)).to(torch.bfloat16)
    q = (torch.randn(batch, heads, latent, generator=gen)).to(torch.bfloat16)
    table = torch.stack([torch.randperm(bank_pages, generator=gen)[:pages]
                         for _ in range(batch)]).to(torch.int32)
    if positions is None:
        positions = torch.randint(0, pages * page, (batch,), generator=gen)
    pos = torch.as_tensor(positions, dtype=torch.int32)
    for b in range(batch):
        table[b, int(pos[b]) // page + 1:] = -1
    written = torch.randn(batch, latent, generator=gen).to(torch.bfloat16)
    topk = None
    if selected:
        topk = torch.full((batch, width), -1, dtype=torch.int32)
        for b in range(batch):
            live = int(pos[b]) + 1
            pick = torch.randint(0, live, (width,), generator=gen)
            keep = torch.rand(width, generator=gen) < 0.85
            topk[b] = torch.where(keep, pick, torch.full_like(pick, -1)).to(torch.int32)
        topk[0, 0] = int(pos[0])  # this step's own row, selected (and maybe again)
    scale = latent ** -0.5
    return q, bank, table, pos, written, scale, page, topk


def _sparse_5938748(q, bank, table, pos, written, scale, page, topk):
    """The snapshot kernel, one request per call, as the model calls it at batch 1."""
    sparse = load_5938748().mla_sparse
    rows = []
    for b in range(int(q.shape[0])):
        rows.append(sparse.mla_sparse_attention(
            q[b:b + 1], bank, topk[b:b + 1], scale,
            block_table_row=table[b].reshape(-1, 1), written=written[b:b + 1],
            write_offset=pos[b].reshape(1, 1), page_size=page))
    return torch.cat(rows, 0)


def _causal_rows(pos, width):
    cols = torch.arange(width, dtype=torch.int32).expand(int(pos.shape[0]), width)
    return torch.where(cols <= pos.reshape(-1, 1), cols, torch.full_like(cols, -1))


def _check(got, want, label):
    rel = float((got - want).norm() / want.norm())
    assert rel <= REL, f"{label}: relative L2 {rel:.3e} > {REL}"
    torch.testing.assert_close(got, want, rtol=REL * 10, atol=REL * float(want.abs().max()))


def _run(args, monkeypatch, lnc=None):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    MD.reset_mla_decode_dispatch_counters()
    out = MD.mla_decode_attention(*args[:5], args[5], args[6], args[7])
    assert MD.mla_decode_dispatch_counters() == (1, 0)
    return out, MD.mla_decode_route_counts()


@pytest.mark.parametrize("batch", [1, 4])
def test_dense_at_the_served_shape_matches_the_oracle_and_5938748(batch, monkeypatch):
    # ctx 1024 in a 2048-row window (the bucket the bypass serves), plus the window's
    # first and last rows on the extra requests.
    positions = [1023, 0, 2047, 517][:batch]
    args = _case(batch, 1, 512, 16, 128, 40, seed=11 + batch, selected=False,
                 positions=positions)
    out, counts = _run(args, monkeypatch)
    assert counts == (1, 0, 0)
    q, bank, table, pos, written, scale, page, _ = args
    _check(out, MD.mla_decode_attention_torch_oracle(*args), "dense vs oracle")
    _check(out, _sparse_5938748(q, bank, table, pos, written, scale, page,
                                _causal_rows(pos, 2176)), "dense vs 5938748")


@pytest.mark.parametrize("batch", [1, 4])
def test_selected_at_the_served_shape_matches_the_oracle_and_5938748(batch, monkeypatch):
    # ctx up to 4096 in the 4096-row window, 2048 selected rows (index_topk), with -1s,
    # duplicates and this step's own row among them.
    args = _case(batch, 1, 512, 32, 128, 48, seed=21 + batch, selected=True, width=2048,
                 positions=[4095, 3000, 2052, 4000][:batch])
    out, counts = _run(args, monkeypatch)
    assert counts == (0, 1, 0)
    _check(out, MD.mla_decode_attention_torch_oracle(*args), "selected vs oracle")
    _check(out, _sparse_5938748(*args), "selected vs 5938748")


@pytest.mark.parametrize("selected", [False, True])
def test_two_programs_split_the_requests_and_agree_with_one(selected, monkeypatch):
    args = _case(4, 1, 512, 8, 128, 12, seed=31, selected=selected, width=512)
    one, counts_one = _run(args, monkeypatch)
    two, counts_two = _run(args, monkeypatch, lnc="2")
    assert counts_one[2] == 0 and counts_two[2] == 1
    assert torch.equal(one, two)
    _check(two, MD.mla_decode_attention_torch_oracle(*args), "grid [2] vs oracle")


@pytest.mark.parametrize("selected", [False, True])
def test_heads_and_small_pages(selected, monkeypatch):
    # Four heads, latent 256, 16-row pages: several pieces per 128-row chunk.
    args = _case(3, 4, 256, 16, 16, 40, seed=41, selected=selected, width=256)
    out, counts = _run(args, monkeypatch)
    assert counts[0 if not selected else 1] == 1
    _check(out, MD.mla_decode_attention_torch_oracle(*args), "small pages vs oracle")
    if selected:
        _check(out, _sparse_5938748(*args), "small pages vs 5938748")


def test_the_own_row_is_read_from_written_not_the_bank(monkeypatch):
    args = list(_case(2, 1, 512, 4, 128, 6, seed=51, selected=False))
    out, _ = _run(tuple(args), monkeypatch)
    q, bank, table, pos, written = args[:5]
    poisoned = bank.clone()
    for b in range(2):
        row = int(table[b, int(pos[b]) // 128]) * 128 + int(pos[b]) % 128
        poisoned[row] = 1000.0
    args[1] = poisoned
    again, _ = _run(tuple(args), monkeypatch)
    assert torch.equal(out, again)


def test_refusals_name_the_operand():
    q, bank, table, pos, written, scale, page, _ = _case(1, 1, 512, 4, 128, 6, seed=61,
                                                         selected=False)
    with pytest.raises(MD.MlaDecodeAttentionError, match="2-byte float"):
        MD.mla_decode_attention(q.float(), bank.float(), table, pos, written.float(),
                                scale, page)
    with pytest.raises(MD.MlaDecodeAttentionError, match="position"):
        MD.mla_decode_attention(q, bank, table, pos + 4 * 128, written, scale, page)
    with pytest.raises(MD.MlaDecodeAttentionError, match="window"):
        MD.mla_decode_attention(q, bank, table, pos, written, scale, page,
                                torch.full((1, 128), 4 * 128, dtype=torch.int32))
    with pytest.raises(MD.MlaDecodeAttentionError, match="power of two"):
        MD.mla_decode_attention(q, bank[:384 * 2], table[:, :1], pos.clamp(max=383),
                                written, scale, 384,
                                torch.zeros((1, 128), dtype=torch.int32))


def test_a_request_that_selects_nothing_returns_zeros_as_5938748_does(monkeypatch):
    # Request 1 selects only sentinels; request 0 keeps its draw. Neither leaks into the
    # other, and the empty one matches the snapshot kernel's fully masked answer.
    args = list(_case(2, 1, 512, 4, 128, 6, seed=71, selected=True, width=128))
    args[7] = args[7].clone()
    args[7][1] = -1
    out, counts = _run(tuple(args), monkeypatch)
    assert counts == (0, 1, 0)
    assert torch.count_nonzero(out[1]) == 0
    _check(out[:1], MD.mla_decode_attention_torch_oracle(*args)[:1], "kept row vs oracle")
    torch.testing.assert_close(out, _sparse_5938748(*args), rtol=REL * 10,
                               atol=REL * float(out.abs().max()))
