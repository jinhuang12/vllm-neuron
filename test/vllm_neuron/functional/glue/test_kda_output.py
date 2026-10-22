# SPDX-License-Identifier: Apache-2.0
"""KDA gated output norm + ``o_proj`` through the fused kernel against 0a08ff4.

0a08ff4's ``_gated_output`` normalises ``core`` over the head extent, applies the gain
and ``sigmoid(out_gate)``, and multiplies by ``o_proj.float().t()`` in fp32. The kernel
does the same fp32 elementwise steps in the same order and the projection on the
tensor engine as two bf16 passes, ``g = hi + lo`` (``hi = bf16(g)``,
``lo = bf16(g - hi)``): each product is exact in fp32, and ``g - hi - lo`` is at most
2^-18 of ``|g|``. So the fp32 result is asserted to ``ATTN_RTOL * max|ref|`` against
the 0a08ff4 expression (an absolute bound at the output's scale: an output that
cancels to ~1e-5 carries the error of its ~1e-2 terms), and the layer's bf16 output
(the cast after the reduction) to that bound plus one bf16 step of each element, with
at most ``OUTPUT_MAX_FLIPS`` of the elements changed (observed 0.2-0.3%).

An fp32 ``o_proj`` (the CPU fixtures' dtype; the served checkpoint is bf16) takes one
fp32 pass instead of the two bf16 ones, asserted to the same ``ATTN_RTOL``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from test.hardware.baselines.glue_0a08ff4 import load as load_0a08ff4
from test.vllm_neuron.functional.glue import glue_case
from vllm_neuron.functional.glue import kda_output as fused
from vllm_neuron.model.glm5_next import model_fp8

BATCHES = (1, 4, 64)
ATTN_RTOL = 1e-5
OUTPUT_MAX_FLIPS = 0.01


def _attention(model):
    cfg = glue_case.text_config()
    attn = model.Glm5NextKDAAttention(cfg, glue_case.TP_WORLD)
    prefix = f"model.language_model.layers.{glue_case.KDA_LAYER}."
    glue_case._kda_weights(attn, glue_case._Source(seed=4), prefix, int(cfg.hidden_size))
    return attn, cfg


@pytest.fixture(scope="module")
def pair():
    live, cfg = _attention(model_fp8)
    old, _ = _attention(load_0a08ff4().model_fp8)
    return live, old, cfg


def _operands(batch):
    gen = torch.Generator().manual_seed(70 + batch)
    core = torch.randn(batch, 128, generator=gen) * 0.2
    gate = torch.randn(batch, 128, generator=gen) * 2.0
    hidden = torch.randn(batch, 4096, generator=gen).to(torch.bfloat16)
    return core, gate, hidden


def _bf16_step(*values: torch.Tensor) -> torch.Tensor:
    """One bf16 step at the larger magnitude of each element pair (2^-7 of its binade).

    The larger of the two, because a one-step rounding flip can cross a binade
    boundary, where the lower binade's step is half the flip.
    """
    mag = torch.stack([v.float().abs() for v in values]).amax(0).clamp_min(2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7.0)


@pytest.mark.parametrize("dma", ("1", "0"))
@pytest.mark.parametrize("lnc", ("1", "2"))
@pytest.mark.parametrize("batch", BATCHES)
def test_kernel_matches_the_0a08ff4_expression(pair, batch, lnc, dma, monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    monkeypatch.setenv(fused.DMA_TRANSPOSE_ENV, dma)
    live, _, _ = pair
    core, gate, _ = _operands(batch)
    fused.reset_dispatch_counters()
    got = fused.kda_gated_projection(core, gate, live)
    assert fused.dispatch_counters() == (1, 0)
    want = fused.kda_gated_projection_torch(core, gate, live)
    assert got.shape == want.shape == (batch, 4096) and got.dtype == torch.float32
    bound = ATTN_RTOL * float(want.abs().max())
    assert float((got - want).abs().max()) <= bound


@pytest.mark.parametrize("dma", ("1", "0"))
@pytest.mark.parametrize("lnc", ("1", "2"))
@pytest.mark.parametrize("batch", (1, 64))
@pytest.mark.parametrize("gain_dtype", (torch.bfloat16, torch.float32))
def test_fp32_o_proj_matches_the_0a08ff4_expression(pair, batch, lnc, dma, gain_dtype,
                                                    monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    monkeypatch.setenv(fused.DMA_TRANSPOSE_ENV, dma)
    live, _, _ = pair
    gen = torch.Generator().manual_seed(80 + batch)
    o_proj = live.o_proj_weight.float()
    # fp32 values bf16 cannot hold: a 2^-8 relative jitter on each element.
    o_proj = o_proj + torch.randn(o_proj.shape, generator=gen) * float(
        o_proj.abs().mean()) * 2.0 ** -8
    attn = SimpleNamespace(
        o_proj_weight=o_proj, o_norm_weight=live.o_norm_weight.to(gain_dtype),
        num_kv_heads_per_rank=live.num_kv_heads_per_rank, head_dim=live.head_dim,
        rms_norm_eps=live.rms_norm_eps)
    core, gate, _ = _operands(batch)
    fused.reset_dispatch_counters()
    got = fused.kda_gated_projection(core, gate, attn)
    assert fused.dispatch_counters() == (1, 0)
    want = fused.kda_gated_projection_torch(core, gate, attn)
    assert got.shape == want.shape == (batch, 4096) and got.dtype == torch.float32
    bound = ATTN_RTOL * float(want.abs().max())
    assert float((got - want).abs().max()) <= bound


@pytest.mark.parametrize("batch", BATCHES)
def test_gated_output_matches_0a08ff4(pair, batch):
    live, old, _ = pair
    core, gate, hidden = _operands(batch)
    fused.reset_dispatch_counters()
    got = live._gated_output(core, gate, hidden, phase="decode")
    assert fused.dispatch_counters() == (1, 0)
    want = old._gated_output(core, gate, hidden)
    assert got.dtype == want.dtype == torch.bfloat16
    diff = (got.float() - want.float()).abs()
    bound = _bf16_step(want, got) + ATTN_RTOL * float(want.float().abs().max())
    assert bool((diff <= bound).all()), float(diff.max())
    assert float((diff > 0).float().mean()) <= OUTPUT_MAX_FLIPS


def test_the_kill_switch_restores_0a08ff4_bit_for_bit(pair, monkeypatch):
    live, old, _ = pair
    monkeypatch.setenv("VLLM_NEURON_GLUE_FUSED", "0")
    core, gate, hidden = _operands(4)
    fused.reset_dispatch_counters()
    got = live._gated_output(core, gate, hidden, phase="decode")
    assert fused.dispatch_counters() == (0, 1)
    assert torch.equal(got, old._gated_output(core, gate, hidden))


def test_the_layer_hands_its_phase_to_the_output_kernel(pair, monkeypatch):
    """The phase the layer passes decides the route, whatever the row count."""
    live, _, _ = pair
    core, gate, hidden = _operands(4)
    monkeypatch.setenv("VLLM_NEURON_GLUE_FUSED", "kda_output:prefill")
    for phase, counters in (("prefill", (1, 0)), ("decode", (0, 1))):
        fused.reset_dispatch_counters()
        live._gated_output(core, gate, hidden, phase=phase)
        assert fused.dispatch_counters() == counters, phase


# ── the bf16 store: a partial its bf16 wire rounds next ─────────────────────────────


def _prefill_operands(rows):
    gen = torch.Generator().manual_seed(90 + rows)
    return torch.randn(rows, 128, generator=gen) * 0.2, torch.randn(rows, 128, generator=gen) * 2.0


@pytest.mark.parametrize("lnc", ("1", "2"))
def test_the_prefill_kernel_stores_the_bf16_cast_of_its_fp32_sums(pair, lnc, monkeypatch):
    """``BF16_OUT`` rounds each sum once, to nearest even, as it leaves PSUM: the bits
    of a bf16 cast of the fp32 store, at half the bytes."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    live, _, _ = pair
    core, gate = _prefill_operands(256)
    fused.reset_dispatch_counters()
    fp32 = fused.kda_gated_projection(core, gate, live, phase="prefill")
    bf16 = fused.kda_gated_projection(core, gate, live, phase="prefill", out_dtype=torch.bfloat16)
    assert fused.dispatch_counters() == (2, 0)
    assert fp32.dtype == torch.float32 and bf16.dtype == torch.bfloat16
    assert torch.equal(bf16, fp32.to(torch.bfloat16))


@pytest.mark.parametrize("spec, counters", (("all", (1, 0)), ("0", (0, 1))),
                         ids=("decode-kernel", "torch"))
def test_the_decode_kernel_and_the_torch_route_cast_their_fp32_result(pair, spec, counters,
                                                                      monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_GLUE_FUSED", spec)
    live, _, _ = pair
    core, gate, _ = _operands(4)
    fp32 = fused.kda_gated_projection(core, gate, live, phase="decode")
    fused.reset_dispatch_counters()
    bf16 = fused.kda_gated_projection(core, gate, live, phase="decode", out_dtype=torch.bfloat16)
    assert fused.dispatch_counters() == counters
    assert bf16.dtype == torch.bfloat16 and torch.equal(bf16, fp32.to(torch.bfloat16))


@pytest.mark.parametrize("out_dtype", (torch.float32, torch.float16), ids=str)
def test_an_out_dtype_other_than_none_or_bf16_is_refused(pair, out_dtype):
    live, _, _ = pair
    core, gate, _ = _operands(4)
    with pytest.raises(ValueError, match="None .* or torch.bfloat16"):
        fused.kda_gated_projection(core, gate, live, phase="decode", out_dtype=out_dtype)


# ── traced: an fp32 call is the route's own graph ───────────────────────────────────


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


def _meta_case(rows):
    """Meta operands at the served per-rank KDA shape: one head of 128, ``H = 4096``."""
    meta = torch.device("meta")
    attn = SimpleNamespace(
        o_proj_weight=torch.empty(4096, 128, dtype=torch.bfloat16, device=meta),
        o_norm_weight=torch.empty(128, dtype=torch.bfloat16, device=meta),
        num_kv_heads_per_rank=1, head_dim=128, rms_norm_eps=1e-6)
    core = torch.empty(rows, 128, dtype=torch.float32, device=meta)
    return core, torch.empty_like(core), attn


def _computed(graph):
    """Every node but the placeholders, whose order follows the route's first reads."""
    return [(node.op, str(node.target), str(node.args), str(node.kwargs))
            for node in graph.nodes if node.op != "placeholder"]


def _returned(graph):
    (output,) = [node for node in graph.nodes if node.op == "output"]
    return output.args[0][0]


def test_without_an_out_dtype_the_torch_route_traces_the_torch_expression(monkeypatch):
    """No cast after the fp32 matmul: the graph a decode step traced before the bf16
    store existed (the served default fuses ``kda_output`` at prefill only)."""
    monkeypatch.setenv("VLLM_NEURON_GLUE_FUSED", "0")
    core, gate, attn = _meta_case(4)
    route = _traced(lambda c, g: fused.kda_gated_projection(c, g, attn, phase="decode"),
                    core, gate)
    expression = _traced(lambda c, g: fused.kda_gated_projection_torch(c, g, attn), core, gate)
    assert _computed(route) == _computed(expression)


@pytest.mark.parametrize("rows, phase, flag", ((4, "decode", None), (256, "prefill", False),
                                               (256, "prefill", True)),
                         ids=("decode", "prefill", "prefill-bf16"))
def test_a_kernel_call_is_the_graphs_result_with_bf16_out_as_asked(rows, phase, flag,
                                                                   monkeypatch):
    """The decode kernel has no ``BF16_OUT``; the prefill call records it, True only for
    a bf16 store (the wrapper records every parameter, defaults included). No local of
    the route renames the call's node: the compile cache key holds node names."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    core, gate, attn = _meta_case(rows)
    out_dtype = torch.bfloat16 if flag else None
    graph = _traced(lambda c, g: fused.kda_gated_projection(c, g, attn, phase=phase,
                                                            out_dtype=out_dtype), core, gate)
    (call,) = [node for node in graph.nodes if "nki_kernel_wrapper" in str(node.target)]
    assert _returned(graph) is call
    assert call.name not in fused.kda_gated_projection.__code__.co_varnames
    names = list(call.kwargs["arg_names"])
    if flag is None:
        assert "BF16_OUT" not in names
    else:
        assert call.kwargs["args"][names.index("BF16_OUT")] is flag
