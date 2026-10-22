# SPDX-License-Identifier: Apache-2.0
"""``blockwise_fp8_mlp``'s bf16 store: a partial its bf16 wire rounds next.

``out_dtype=torch.bfloat16`` asks for the bits of a bf16 cast of the fp32 result. The
prefill kernel rounds in its last contraction block's write (``bf16_out``) and stores
half the bytes; the small-M kernel and the three-call route cast their fp32 result.
Each route is run on the simulator at a small whole-tile shape and compared bit for bit
with its own fp32 result cast to bf16. ``out_dtype=None`` keeps the route's own dtype,
and any other dtype is refused.
"""

from __future__ import annotations

import importlib

import pytest
import torch

#: The module, not the function of the same name that ``vllm_neuron.functional`` exports.
bw = importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")

#: Two scale blocks of contraction and two of intermediate: the smallest prefill-kernel
#: geometry with more than one block on each axis.
HIDDEN, INTERMEDIATE = 256, 256
SWIGLU_LIMIT = 10.0


@pytest.fixture(autouse=True)
def _simulator(monkeypatch, tmp_path):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    # The NKI driver writes compile artifacts into the working directory.
    monkeypatch.chdir(tmp_path)


def _operands(tokens: int, seed: int):
    """fp8 weights at magnitudes that make both SwiGLU clamps bind, with distinct grids."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((tokens, HIDDEN), generator=g).to(torch.bfloat16)

    def weight(rows, cols):
        return (torch.randn((rows, cols), generator=g) * 64).clamp(-240, 240).to(
            torch.float8_e4m3fn)

    def scale(rows, cols, magnitude):
        grid = torch.rand((rows // 128, cols // 128), generator=g) + 0.37
        return (grid * magnitude).to(torch.float32)

    return (x, weight(HIDDEN, INTERMEDIATE), weight(HIDDEN, INTERMEDIATE),
            weight(INTERMEDIATE, HIDDEN), scale(HIDDEN, INTERMEDIATE, 3e-3),
            scale(HIDDEN, INTERMEDIATE, 3e-3), scale(INTERMEDIATE, HIDDEN, 1e-3))


def _mlp(ops, out_dtype=None):
    return bw.blockwise_fp8_mlp(*ops, swiglu_limit=SWIGLU_LIMIT, out_dtype=out_dtype)


@pytest.mark.parametrize("lnc", [None, "2"], ids=["one-program", "lnc2"])
def test_the_prefill_kernel_stores_the_bf16_cast_of_its_fp32_result(lnc, monkeypatch):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    ops = _operands(256, seed=41)
    assert bw.prefill_mlp_takes(ops[0], ops[1], ops[2], ops[3])
    fp32, bf16 = _mlp(ops), _mlp(ops, torch.bfloat16)
    assert fp32.dtype == torch.float32 and bf16.dtype == torch.bfloat16
    assert torch.equal(bf16, fp32.to(torch.bfloat16))


def test_the_small_m_kernel_casts_its_fp32_result(monkeypatch):
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    ops = _operands(4, seed=43)
    assert bw.fused_mlp_admissible(4, HIDDEN, INTERMEDIATE)
    fp32, bf16 = _mlp(ops), _mlp(ops, torch.bfloat16)
    assert fp32.dtype == torch.float32 and bf16.dtype == torch.bfloat16
    assert torch.equal(bf16, fp32.to(torch.bfloat16))


def test_the_three_call_route_casts_its_fp32_result(monkeypatch):
    """No NKI route (the simulator off in CPU mode): three ``blockwise_fp8_mm`` calls."""
    monkeypatch.setenv("NKI_SIMULATOR", "0")
    ops = _operands(128, seed=47)
    assert not bw.prefill_mlp_takes(ops[0], ops[1], ops[2], ops[3])
    fp32, bf16 = _mlp(ops), _mlp(ops, torch.bfloat16)
    assert bf16.dtype == torch.bfloat16 and torch.equal(bf16, fp32.to(torch.bfloat16))


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16], ids=str)
def test_an_out_dtype_other_than_none_or_bf16_is_refused(out_dtype):
    with pytest.raises(ValueError, match="None .* or torch.bfloat16"):
        _mlp(_operands(4, seed=53), out_dtype)


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


def _meta_operands(tokens: int, intermediate: int):
    meta = torch.device("meta")

    def weight(rows, cols):
        return torch.empty(rows, cols, dtype=torch.float8_e4m3fn, device=meta)

    def scale(rows, cols):
        return torch.empty(rows // 128, cols // 128, dtype=torch.float32, device=meta)

    return (torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=meta),
            weight(HIDDEN, intermediate), weight(HIDDEN, intermediate),
            weight(intermediate, HIDDEN), scale(HIDDEN, intermediate),
            scale(HIDDEN, intermediate), scale(intermediate, HIDDEN))


@pytest.mark.parametrize("tokens, intermediate", [(4, 256), (128, 512), (256, 256)],
                         ids=["small-m", "three-call", "prefill"])
def test_an_fp32_result_keeps_the_name_its_producer_gives_it(tokens, intermediate):
    """The graph's result is the last call's own node: no cast after it, and no local of
    ``blockwise_fp8_mlp`` renames it. The compile cache key hashes the graph text, node
    names included, so a binding there would re-key every graph that runs the MLP.
    ``I = 512`` is past the prefill kernel's limit, so 128 rows take the three calls."""
    graph = _traced(lambda *ops: bw.blockwise_fp8_mlp(*ops, swiglu_limit=SWIGLU_LIMIT),
                    *_meta_operands(tokens, intermediate))
    (output,) = [node for node in graph.nodes if node.op == "output"]
    result = output.args[0][0]
    assert str(result.target) != "to"
    assert result.name not in bw.blockwise_fp8_mlp.__code__.co_varnames
