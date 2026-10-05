# SPDX-License-Identifier: Apache-2.0
"""Native cache transposes retain the FP32 attention and sentinel semantics."""

import pytest
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention import mla_sparse as MS


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "topk,latent,heads,rope,block_n",
    [
        (640, 128, 1, 0, 512),
        (1152, 256, 7, 7, 256),
        (2048, 512, 2, 64, 512),
        (640, 512, 16, 0, 128),
    ],
)
def test_native_streamed_cache_preserves_fp32_attention(
    dtype, topk, latent, heads, rope, block_n
):
    """Exercise chunk boundaries, repeated keys, masked tiles and a zero query."""
    gen = torch.Generator().manual_seed(731)
    seq, cache_rows = 3, topk + 129
    q = (torch.randn(seq, heads, latent, generator=gen) * 0.1).to(dtype)
    cache = (torch.randn(cache_rows, latent, generator=gen) * 0.1).to(dtype)
    indices = torch.randint(cache_rows, (seq, topk), generator=gen, dtype=torch.int32)
    indices[0, :512] = -1
    indices[0, -128:] = -1
    indices[1, :128] = cache_rows - 1
    indices[1, 128:256] = -1
    indices[2, :] = -1
    scale = latent ** -0.5

    if rope:
        q_pe = (torch.randn(seq, heads, rope, generator=gen) * 0.1).to(dtype)
        k_pe = (torch.randn(cache_rows, rope, generator=gen) * 0.1).to(dtype)
        entry = wrap_nki(MS.mla_sparse_attention_rope_row_tiled_kernel)
        got = entry(q, q_pe, cache, k_pe, indices, scale, block_n, True)
        before = entry(q.float(), q_pe.float(), cache.float(), k_pe.float(),
                       indices, scale, block_n, True)
    else:
        q_pe = k_pe = None
        entry = wrap_nki(MS.mla_sparse_attention_nope_row_tiled_kernel)
        got = entry(q, cache, indices, scale, None, None, None, 0, block_n, True)
        before = entry(q.float(), cache.float(), indices, scale,
                       None, None, None, 0, block_n, True)

    expected = MS.mla_sparse_attention_torch_oracle(
        q, cache, indices, scale, q_pe=q_pe, k_pe=k_pe)
    torch.testing.assert_close(got, before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-5)
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got[2]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_cache_preserves_single_key_and_extreme_online_rescaling(dtype):
    """An empty first tile and changing maxima retain exact sentinel behavior."""
    latent, topk = 128, 640
    q = torch.ones(3, 2, latent, dtype=dtype)
    q[1, 1] = -1
    cache = torch.zeros(topk, latent, dtype=dtype)
    cache[:128] = -10
    cache[-128:] = 10
    cache[333] = torch.linspace(-1, 1, latent).to(dtype)
    indices = torch.arange(topk, dtype=torch.int32).expand(3, topk).clone()
    indices[0] = -1
    indices[0, 512] = 333
    indices[2] = -1
    entry = wrap_nki(MS.mla_sparse_attention_nope_row_tiled_kernel)
    got = entry(q, cache, indices, 1.0, None, None, None, 0, 512, True)
    before = entry(q.float(), cache.float(), indices, 1.0,
                   None, None, None, 0, 512, True)
    expected = MS.mla_sparse_attention_torch_oracle(q, cache, indices, 1.0)
    torch.testing.assert_close(got, before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-5)
    torch.testing.assert_close(
        got[0], cache[333].float().expand(2, latent), rtol=0.0, atol=0.0
    )
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got[2]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_attention_preserves_causal_selection_sentinels(dtype):
    """Future rows masked by the selector cannot affect preceding queries."""
    generator = torch.Generator().manual_seed(797)
    latent, topk = 128, 640
    q = (torch.randn(3, 1, latent, generator=generator) * 0.1).to(dtype)
    cache = (torch.randn(topk, latent, generator=generator) * 0.1).to(dtype)
    columns = torch.arange(topk, dtype=torch.int32).expand(3, topk)
    prefix_lengths = torch.tensor([1, 129, topk]).reshape(3, 1)
    indices = torch.where(columns < prefix_lengths, columns, -1)
    scale = latent**-0.5
    entry = wrap_nki(MS.mla_sparse_attention_nope_row_tiled_kernel)
    got = entry(q, cache, indices, scale, None, None, None, 0, 512, True)
    before = entry(q.float(), cache.float(), indices, scale,
                   None, None, None, 0, 512, True)
    expected = MS.mla_sparse_attention_torch_oracle(q, cache, indices, scale)
    torch.testing.assert_close(got, before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-5)
    torch.testing.assert_close(
        got[0, 0], cache[0].float(), rtol=0.0, atol=0.0
    )

    future_changed = cache.clone()
    future_changed[129:] += 1.0
    changed = entry(q, future_changed, indices, scale,
                    None, None, None, 0, 512, True)
    torch.testing.assert_close(changed[:2], got[:2], rtol=0.0, atol=0.0)
    assert not torch.equal(changed[2], got[2])


@pytest.mark.parametrize("seq", [1, 3])
def test_paged_native_cache_reads_written_overlay_and_2176_column_tail(seq):
    """The captured bank/page geometry includes a tail tile and fresh KV rows."""
    generator = torch.Generator().manual_seed(781)
    page, window_rows, bank_rows, latent, topk = 128, 2048, 20608, 512, 2176
    q = (torch.randn(seq, 1, latent, generator=generator) * 0.1).bfloat16()
    bank = (torch.randn(bank_rows, latent, generator=generator) * 0.1).bfloat16()
    table = torch.randperm(bank_rows // page, generator=generator)[:16].to(
        torch.int32
    ).reshape(16, 1)
    written = (torch.randn(seq, latent, generator=generator) * 0.1).bfloat16()
    offset = torch.tensor([[window_rows - seq]], dtype=torch.int32)
    cache = bank.reshape(-1, page, latent)[table[:, 0].long()].reshape(
        window_rows, latent
    ).clone()
    cache[window_rows - seq:] = written
    indices = torch.randint(
        window_rows, (seq, topk), generator=generator, dtype=torch.int32
    )
    indices[0] = -1
    indices[0, -1] = window_rows - 1
    if seq > 1:
        indices[-1] = -1
    scale = latent**-0.5
    entry = wrap_nki(MS.mla_sparse_attention_nope_row_tiled_kernel)
    got = entry(q, bank, indices, scale, table, written, offset, page, 512, True)
    before = entry(q.float(), bank.float(), indices, scale, table,
                   written.float(), offset, page, 512, True)
    expected = MS.mla_sparse_attention_torch_oracle(q, cache, indices, scale)
    torch.testing.assert_close(got, before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-5)
    torch.testing.assert_close(
        got[0, 0], written[-1].float(), rtol=0.0, atol=0.0
    )
    if seq > 1:
        assert torch.count_nonzero(got[-1]) == 0
