# SPDX-License-Identifier: Apache-2.0
"""Two-PNC sparse prefill writes every query once and preserves its arithmetic."""

import pytest
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention import mla_sparse as MS


def _inputs(seq, dtype, paged):
    generator = torch.Generator().manual_seed(887 + seq)
    window, latent, topk, page = 2048, 512, 2176, 128
    q = (torch.randn(seq, 1, latent, generator=generator) * 0.1).to(dtype)
    bank_rows = 20608 if paged else window
    bank = (torch.randn(bank_rows, latent, generator=generator) * 0.1).to(dtype)
    table = written = offset = None
    cache = bank
    if paged:
        table = torch.randperm(bank_rows // page, generator=generator)[:16]
        table = table.int().reshape(16, 1)
        written = (
            torch.randn(seq, latent, generator=generator) * 0.1
        ).to(dtype)
        offset = torch.tensor([[window - seq]], dtype=torch.int32)
        cache = bank.reshape(-1, page, latent)[table[:, 0].long()].reshape(
            window, latent
        ).clone()
        cache[window - seq:] = written
    indices = torch.randint(
        window, (seq, topk), generator=generator, dtype=torch.int32
    )
    prefix = torch.arange(window - seq + 1, window + 1).reshape(seq, 1)
    indices = torch.where(indices < prefix, indices, -1)
    indices[0] = -1
    indices[0, -1] = window - seq
    indices[1, :128] = window - seq
    indices[-1] = -1
    args = (
        q, bank, indices, latent**-0.5, table, written, offset,
        page if paged else 0, 512, True,
    )
    return args, cache


@pytest.mark.parametrize("seq", [33, 48])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_dual_pnc_preserves_uneven_query_blocks(seq, dtype):
    """33 splits scalar query blocks 17/16; 48 splits 16-query blocks 2/1."""
    args, cache = _inputs(seq, dtype, False)
    call = wrap_nki(MS.mla_sparse_attention_nope_row_tiled_kernel)
    got = call[2](*args)
    single = call(*args)
    expected = MS.mla_sparse_attention_torch_oracle(
        args[0], cache, args[2], args[3]
    )
    torch.testing.assert_close(got, single, rtol=0.0, atol=0.0)
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-5)
    torch.testing.assert_close(
        got[0, 0], cache[2048 - seq].float(), rtol=0.0, atol=0.0
    )
    assert torch.count_nonzero(got[-1]) == 0
    assert torch.isfinite(got).all()


@pytest.mark.parametrize("seq", [33, 48])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dual_pnc_paged_programs_keep_private_overlay_windows(seq, dtype):
    """Independent window staging and tail-key loads produce one shared output."""
    args, cache = _inputs(seq, dtype, True)
    call = wrap_nki(MS.mla_sparse_attention_nope_row_tiled_kernel)
    got = call[2](*args)
    single = call(*args)
    expected = MS.mla_sparse_attention_torch_oracle(
        args[0], cache, args[2], args[3]
    )
    torch.testing.assert_close(got, single, rtol=0.0, atol=0.0)
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-5)
    torch.testing.assert_close(
        got[0, 0], args[5][0].float(), rtol=0.0, atol=0.0
    )
    assert torch.count_nonzero(got[-1]) == 0


@pytest.mark.parametrize(
    "lnc,seq,heads,latent,rope,topk,paged,grid",
    [
        (None, 32, 1, 512, 0, 2176, False, 1),
        ("1", 32, 1, 512, 0, 2176, False, 1),
        ("invalid", 32, 1, 512, 0, 2176, False, 1),
        ("2", 1, 1, 512, 0, 2176, False, 1),
        ("2", 16, 1, 512, 0, 2176, False, 1),
        ("2", 32, 1, 512, 0, 2176, False, 2),
        ("2", 33, 1, 512, 0, 2176, False, 2),
        ("2", 48, 1, 512, 0, 2176, True, 2),
        ("1", 48, 1, 512, 0, 2176, True, 1),
        ("2", 32, 2, 512, 0, 2176, False, 1),
        ("2", 32, 1, 256, 0, 2176, False, 1),
        ("2", 32, 1, 512, 7, 2176, False, 1),
        ("2", 32, 1, 512, 0, 512, False, 1),
    ],
)
def test_public_dispatch_obeys_explicit_lnc_and_geometry(
    monkeypatch, lnc, seq, heads, latent, rope, topk, paged, grid
):
    """The public seam cannot request a second PNC outside the verified contract."""
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    calls = []

    class RecordingCall:
        def __init__(self, entry):
            self.entry, self.grid = entry, 1

        def __getitem__(self, requested):
            self.grid = requested
            return self

        def __call__(self, *args):
            calls.append((self.entry, self.grid))
            return torch.zeros(args[0].shape, dtype=torch.float32)

    monkeypatch.setattr(MS, "wrap_nki", RecordingCall)
    q = torch.zeros(seq, heads, latent)
    bank = torch.zeros(2048, latent)
    selected = torch.zeros(seq, topk, dtype=torch.int32)
    kwargs = {}
    if paged:
        kwargs = {
            "block_table_row": torch.arange(16).int().reshape(16, 1),
            "written": torch.zeros(seq, latent),
            "write_offset": torch.tensor([[2048 - seq]], dtype=torch.int32),
            "page_size": 128,
        }
    if rope:
        kwargs.update(
            q_pe=torch.zeros(seq, heads, rope),
            k_pe=torch.zeros(2048, rope),
        )
    MS.reset_mla_sparse_dispatch_counters()
    MS.reset_mla_sparse_row_tiled_dispatch_counters()
    output = MS.mla_sparse_attention(
        q, bank, selected, latent**-0.5, **kwargs
    )
    assert output.shape == q.shape
    assert len(calls) == 1 and calls[0][1] == grid
    assert MS.mla_sparse_dispatch_counters() == (1, 0)
    assert MS.mla_sparse_row_tiled_dispatch_counters() == (
        int(topk > MS.MOVING_MAX), 0
    )
