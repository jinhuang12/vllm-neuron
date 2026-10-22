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


# ---- the bf16 store: a partial its bf16 wire rounds next ------------------------------


@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "bf16"])
@pytest.mark.parametrize("rows, lnc", [(4, "1"), (4, "2"), (256, "1"), (256, "2")],
                         ids=["cols-1", "cols-2", "rows-1", "rows-2"])
def test_a_bf16_store_holds_the_bf16_cast_of_the_fp32_result(rows, lnc, fp8, monkeypatch):
    """``bf16_out`` rounds each sum once, to nearest even, in its copy out of PSUM, on
    the column split (a decode batch) and the row split (a prefill chunk) alike."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    x, w_out_in, s_out_in = _operands(rows, 256, 1024, fp8, seed=31 + rows)
    weight, scale = MP.prepare_lowp_projection(w_out_in, s_out_in)
    assert MP.lowp_splits_rows(rows, 1, 256 * 2) == (rows == 256)
    fp32 = MP.mla_projection_lowp(x, weight, scale)
    bf16 = MP.mla_projection_lowp(x, weight, scale, out_dtype=torch.bfloat16)
    assert fp32.dtype == torch.float32 and bf16.dtype == torch.bfloat16
    assert torch.equal(bf16, fp32.to(torch.bfloat16))


def test_the_fp32_route_casts_its_result_for_a_bf16_store():
    x, w_out_in, _ = _operands(2, 256, 128, False, seed=23)
    fp32_weight = w_out_in.float().t().contiguous()
    got = MP.mla_projection_prepared(x, fp32_weight, None, out_dtype=torch.bfloat16)
    want = MP.mla_projection(x.float(), fp32_weight)
    assert got.dtype == torch.bfloat16 and torch.equal(got, want.to(torch.bfloat16))


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16], ids=str)
def test_an_out_dtype_other_than_none_or_bf16_is_refused(out_dtype):
    x, w_out_in, s_out_in = _operands(1, 256, 256, True, seed=19)
    weight, scale = MP.prepare_lowp_projection(w_out_in, s_out_in)
    with pytest.raises(MP.MlaProjectionError, match="None .* or torch.bfloat16"):
        MP.mla_projection_lowp(x, weight, scale, out_dtype=out_dtype)
    with pytest.raises(MP.MlaProjectionError, match="None .* or torch.bfloat16"):
        MP.mla_projection_prepared(x, weight, (weight, scale), out_dtype=out_dtype)
    with pytest.raises(MP.MlaProjectionError, match="None .* or torch.bfloat16"):
        MP.mla_projection_prepared(x, weight.float(), None, out_dtype=out_dtype)


# ---- traced: an fp32 call is the route's own graph ------------------------------------


def _traced(fn, *args):
    """``fn``'s graph as dynamo records it on ``args`` (no compile behind it)."""
    seen = []

    def record(gm, example_inputs):
        seen.append(gm)
        return gm.forward

    torch._dynamo.reset()
    try:
        torch.compile(fn, backend=record, fullgraph=True, dynamic=False)(*args)
    finally:
        torch._dynamo.reset()
    (gm,) = seen
    return gm.graph


def _computed(graph):
    """Every node but the placeholders, whose order follows the route's first reads."""
    return [(node.op, str(node.target), str(node.args), str(node.kwargs))
            for node in graph.nodes if node.op != "placeholder"]


def _returned(graph):
    (output,) = [node for node in graph.nodes if node.op == "output"]
    return output.args[0][0]


@pytest.mark.parametrize("bf16_out", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "bf16"])
def test_the_kernel_call_is_the_result_with_bf16_out_as_asked(fp8, bf16_out, monkeypatch):
    """No cast after the call, and no local of the route renames its node (the compile
    cache key holds node names); ``bf16_out`` True only for a bf16 store."""
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    meta = torch.device("meta")
    x = torch.empty(4, 256, dtype=torch.bfloat16, device=meta)
    weight = torch.empty(256, 1024, dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
                         device=meta)
    scale = torch.empty(2, 8, dtype=torch.float32, device=meta) if fp8 else None
    out_dtype = torch.bfloat16 if bf16_out else None
    graph = _traced(lambda x, w, s: MP.mla_projection_lowp(x, w, s, out_dtype=out_dtype),
                    x, weight, scale)
    (call,) = [node for node in graph.nodes if "nki_kernel_wrapper" in str(node.target)]
    assert _returned(graph) is call
    assert call.name not in MP.mla_projection_lowp.__code__.co_varnames
    assert list(call.kwargs["arg_names"]) == ["x_hbm", "w_hbm", "scale_hbm", "col_chunk",
                                              "bf16_out"]
    assert call.kwargs["args"][4] is bf16_out


def test_without_an_out_dtype_the_fp32_route_traces_the_fp32_kernel():
    meta = torch.device("meta")
    x = torch.empty(4, 256, dtype=torch.bfloat16, device=meta)
    weight = torch.empty(256, 1024, dtype=torch.float32, device=meta)
    route = _traced(lambda x, w: MP.mla_projection_prepared(x, w, None), x, weight)
    kernel = _traced(lambda x, w: MP.mla_projection(x.to(torch.float32), w), x, weight)
    assert _computed(route) == _computed(kernel)
