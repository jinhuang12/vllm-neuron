# SPDX-License-Identifier: Apache-2.0
"""``mla_projection_lowp``: the DSA layer's projections on weights read as stored.

The fp8 sites read fp8-e4m3 bytes with their 128 x 128 fp32 scale grid; the indexer's
sites read bf16. Each case runs the kernel in the simulator at the per-rank TP=64 decode
widths, at M in {1, 4} rows (one row per request), and compares it with

* the 5938748 fp32 kernel (``mla_projection`` on the dequantised fp32 weight) -- the
  path this replaces -- and
* an fp64 product of the same operands.

Tolerance, stated once: the kernel rounds the activation to bf16 and folds each fp8
block's multiplier into that bf16 operand, so each output carries about one bf16
rounding (2^-9 relative) per product term. Measured relative L2 against the fp32 kernel
is 1.5e-3 to 1.8e-3 for the fp8 sites and 0 for the bf16 sites (a bf16 activation times a
bf16 weight is exact in fp32); the bound asserted is 4e-3.
"""

from __future__ import annotations

import pytest
import torch

from test.hardware.baselines.dsa_5938748 import load as load_5938748
from vllm_neuron.functional.attention import mla_projections as MP

REL_L2 = 4e-3

#: (site, in_features, out_features, fp8) at TP=64: one MLA head per rank.
SITES = (
    ("q_a_proj", 4096, 1536, True),
    ("q_b_proj", 1536, 256, True),
    ("kv_a_proj_with_mqa", 4096, 512, True),
    ("o_proj", 256, 4096, True),
    ("wq_b", 1536, 4096, False),
    ("wk", 4096, 128, False),
    ("weights_proj", 4096, 32, False),
    ("index_kpool_compress_gate", 4096, 128, False),
)


def _operands(rows, idim, odim, fp8, seed):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, idim, generator=gen).to(torch.bfloat16)
    if fp8:
        w_out_in = (torch.randn(odim, idim, generator=gen) * 48).clamp(-224, 224).to(
            torch.float8_e4m3fn)
        s_out_in = (torch.rand(odim // 128, idim // 128, generator=gen) * 0.5 + 0.75) \
            * idim ** -0.5 / 48
    else:
        w_out_in = (torch.randn(odim, idim, generator=gen) * idim ** -0.5).to(torch.bfloat16)
        s_out_in = None
    return x, w_out_in, s_out_in


def _dequantised_fp32(w_out_in, s_out_in):
    """The 5938748 prep: dequantise in the checkpoint orientation, upcast, transpose."""
    dense = w_out_in.to(torch.float32)
    if s_out_in is not None:
        dense = dense * s_out_in.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return dense.t().contiguous()


def _rel(got, want):
    return float((got.double() - want.double()).norm() / want.double().norm())


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("site", SITES, ids=[s[0] for s in SITES])
def test_matches_the_5938748_fp32_kernel_and_an_fp64_product(site, rows, monkeypatch):
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    name, idim, odim, fp8 = site
    x, w_out_in, s_out_in = _operands(rows, idim, odim, fp8, seed=idim + odim + rows)
    weight, scale = MP.prepare_lowp_projection(w_out_in, s_out_in)
    assert weight.dtype == (torch.float8_e4m3fn if fp8 else torch.bfloat16)

    MP.reset_mla_projection_lowp_counts()
    MP.reset_mla_projection_dispatch_counters()
    got = MP.mla_projection_prepared(x, None, (weight, scale))
    assert MP.mla_projection_lowp_counts() == (1, int(fp8), 0)
    assert MP.mla_projection_dispatch_counters() == (1, 0)
    assert got.dtype == torch.float32 and tuple(got.shape) == (rows, odim)

    fp32_weight = _dequantised_fp32(w_out_in, s_out_in)
    before = load_5938748().mla_projections.mla_projection(x.to(torch.float32), fp32_weight)
    exact = x.double() @ fp32_weight.double()
    rel_before, rel_exact = _rel(got, before), _rel(got, exact)
    print(f"{name} M={rows}: rel L2 vs 5938748 fp32 {rel_before:.2e}, vs fp64 {rel_exact:.2e}")
    assert rel_before <= REL_L2, (name, rel_before)
    assert rel_exact <= REL_L2, (name, rel_exact)


@pytest.mark.parametrize("fp8", [True, False])
def test_two_programs_split_the_columns_and_agree_with_one(fp8, monkeypatch):
    x, w_out_in, s_out_in = _operands(4, 1536, 512, fp8, seed=7)
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    one = MP.mla_projection_lowp(x, *MP.prepare_lowp_projection(w_out_in, s_out_in))
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    weight, scale = MP.prepare_lowp_projection(w_out_in, s_out_in)
    if fp8:
        assert tuple(scale.shape) == (2, 12, 2)  # program-major for the [2] grid
    MP.reset_mla_projection_lowp_counts()
    two = MP.mla_projection_lowp(x, weight, scale)
    assert MP.mla_projection_lowp_counts()[2] == 1
    assert torch.equal(one, two)


def test_a_plain_scale_grid_is_laid_out_per_call(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    x, w_out_in, s_out_in = _operands(1, 512, 256, True, seed=9)
    weight, laid = MP.prepare_lowp_projection(w_out_in, s_out_in)
    plain = MP.lowp_scale_grid(laid)
    assert torch.equal(plain, s_out_in.t().float())
    assert torch.equal(MP.mla_projection_lowp(x, weight, plain),
                       MP.mla_projection_lowp(x, weight, laid))


def test_column_chunks_beyond_the_sbuf_budget_agree_with_one_chunk(monkeypatch):
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    x, w_out_in, s_out_in = _operands(3, 1024, 1024, True, seed=13)
    weight, scale = MP.prepare_lowp_projection(w_out_in, s_out_in)
    whole = MP.mla_projection_lowp(x, weight, scale)
    monkeypatch.setattr(MP, "LOWP_WEIGHT_SBUF_BYTES", 8 * 256)  # 256 fp8 columns
    assert MP._lowp_col_chunk(1024, 1024, 1, True) == 256
    chunked = MP.mla_projection_lowp(x, weight, scale)
    assert torch.equal(whole, chunked)


def test_rows_past_one_tile_and_an_fp32_activation(monkeypatch):
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    x, w_out_in, s_out_in = _operands(130, 256, 256, True, seed=17)
    weight, scale = MP.prepare_lowp_projection(w_out_in, s_out_in)
    got = MP.mla_projection_lowp(x.float(), weight, scale)
    want = MP.mla_projection_lowp_torch_oracle(x, weight, scale)
    assert _rel(got, want) <= REL_L2


@pytest.mark.parametrize("rows", [1, 3])
def test_an_odd_count_of_contraction_blocks_transposes_in_fp32(rows, monkeypatch):
    # 384 = 3 blocks: each row's transpose is an odd column count, the fp32 branch.
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    x, w_out_in, s_out_in = _operands(rows, 384, 256, True, seed=29)
    weight, scale = MP.prepare_lowp_projection(w_out_in, s_out_in)
    got = MP.mla_projection_lowp(x, weight, scale)
    want = MP.mla_projection_lowp_torch_oracle(x, weight, scale)
    assert _rel(got, want) <= REL_L2


def test_refusals_name_the_operand():
    x, w_out_in, s_out_in = _operands(1, 256, 256, True, seed=19)
    weight, scale = MP.prepare_lowp_projection(w_out_in, s_out_in)
    with pytest.raises(MP.MlaProjectionError, match="block dequant"):
        MP.mla_projection_lowp(x, weight)
    with pytest.raises(MP.MlaProjectionError, match="scale grid"):
        MP.mla_projection_lowp(x, weight, scale[..., :1])
    with pytest.raises(MP.MlaProjectionError, match="fp8-e4m3 or bf16"):
        MP.mla_projection_lowp(x, weight.float())
    with pytest.raises(MP.MlaProjectionError, match="contraction-major"):
        MP.mla_projection_lowp(x[:, :128], weight, scale)


def test_the_fp32_route_is_unchanged_without_a_low_precision_operand():
    x, w_out_in, _ = _operands(2, 256, 128, False, seed=23)
    fp32_weight = w_out_in.float().t().contiguous()
    MP.reset_mla_projection_lowp_counts()
    got = MP.mla_projection_prepared(x, fp32_weight, None)
    assert MP.mla_projection_lowp_counts()[0] == 0
    assert torch.equal(got, MP.mla_projection(x.float(), fp32_weight))
