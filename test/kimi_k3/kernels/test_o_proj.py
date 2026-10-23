# SPDX-License-Identifier: Apache-2.0
"""fp64 oracle for ``o_proj_tkg`` over the five numerics regimes, class B.

The reference is the checkpoint's own projection modules, built at one TP32
rank's slice from ``modeling_kimi_linear.py`` under ``KIMI_K3_MODEL_PATH``:
KDA ``o_proj`` (3 heads, K 384, bf16), MLA ``q_a_proj`` with
``kv_a_proj_with_mqa`` (K 224, fp32), and the latent MoE
``routed_expert_down_proj`` with the router logits (K 224, 8 rows, fp32).

The kernel runs on a Neuron device through the native compile backend, or on
the host under ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import contextlib
import importlib
import json
import math
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_neuron.functional.attention.o_proj import o_proj_tkg

TP = 32
SEED = 20261023

# name: (reference projection, K, D, rows, live rows, dtype)
SHAPES = {
    "k384_n7168_b4_bf16": ("kda_o_proj", 384, 128, 4, 4, torch.bfloat16),
    "k224_n4480_b8_fp32": ("latent_down_router", 224, 112, 8, 4, torch.float32),
    "k224_n2112_b4_fp32": ("mla_q_a_kv_a", 224, 112, 4, 4, torch.float32),
}


class _Unavailable(nn.Module):
    """Stands in for a KDA-only layer that the projections never call."""

    def __init__(self, *args, **kwargs):
        super().__init__()


def _not_called(*args, **kwargs):
    raise RuntimeError("KDA kernel stand-in called")


@contextlib.contextmanager
def _reference_import_shims():
    """Import the reference file without its KDA kernel package.

    The projections need none of ``fla``, which the serving venv does not
    ship, and transformers 5 moved ``OutputRecorder`` out of ``utils.generic``.
    """
    import transformers.utils.generic as generic
    from transformers.utils.output_capturing import OutputRecorder

    shims = {
        "fla": {},
        "fla.modules": {"FusedRMSNormGated": _Unavailable,
                        "ShortConvolution": _Unavailable},
        "fla.ops": {},
        "fla.ops.kda": {"chunk_kda": _not_called,
                        "fused_recurrent_kda": _not_called},
        "fla.ops.utils": {},
        "fla.ops.utils.index": {"prepare_cu_seqlens_from_mask": _not_called,
                                "prepare_lens_from_mask": _not_called},
        "fla.utils": {"tensor_cache": lambda fn: fn},
    }
    added = [name for name in shims if name not in sys.modules]
    for name in added:
        module = types.ModuleType(name)
        module.__dict__.update(shims[name])
        sys.modules[name] = module
    had_recorder = hasattr(generic, "OutputRecorder")
    if not had_recorder:
        generic.OutputRecorder = OutputRecorder
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = write_bytecode
        if not had_recorder:
            del generic.OutputRecorder
        for name in added:
            sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def reference():
    """The reference projections at one rank's slice, fp64, seeded."""
    path = os.environ.get("KIMI_K3_MODEL_PATH")
    if not path or not os.path.isfile(os.path.join(path, "modeling_kimi_linear.py")):
        pytest.skip("KIMI_K3_MODEL_PATH must name the Kimi K3 checkpoint directory")
    package = types.ModuleType("kimi_k3_reference")
    package.__path__ = [path]
    sys.modules["kimi_k3_reference"] = package
    with _reference_import_shims():
        modeling = importlib.import_module("kimi_k3_reference.modeling_kimi_linear")
        config_mod = importlib.import_module("kimi_k3_reference.configuration_kimi_k3")
    with open(os.path.join(path, "config.json")) as fh:
        text = json.load(fh)["text_config"]
    rank_heads = text["linear_attn_config"]["num_heads"] // TP
    rank_hidden = text["hidden_size"] // TP

    def config(**overrides):
        return config_mod.KimiLinearConfig(**{**text, **overrides})

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(SEED)
    try:
        kda = modeling.KimiDeltaAttention(
            config(linear_attn_config={**text["linear_attn_config"], "num_heads": rank_heads}),
            layer_idx=1,
        )
        mla = modeling.KimiMLAAttention(
            config(hidden_size=rank_hidden, num_attention_heads=rank_heads,
                   num_key_value_heads=rank_heads),
            layer_idx=4,
        )
        moe = modeling.KimiSparseMoeBlock(
            config(hidden_size=rank_hidden, moe_intermediate_size=1, num_shared_experts=None)
        )
    finally:
        torch.set_default_dtype(default_dtype)

    def router_logits(x):
        # KimiMoEGate.forward: F.linear of the hidden rows and the gate weight.
        return F.linear(x, moe.gate.weight)

    projections = {
        "kda_o_proj": ([kda.o_proj.weight], kda.o_proj),
        "mla_q_a_kv_a": (
            [mla.q_a_proj.weight, mla.kv_a_proj_with_mqa.weight],
            lambda x: torch.cat([mla.q_a_proj(x), mla.kv_a_proj_with_mqa(x)], dim=-1),
        ),
        "latent_down_router": (
            [moe.routed_expert_down_proj.weight, moe.gate.weight],
            lambda x: torch.cat([moe.routed_expert_down_proj(x), router_logits(x)], dim=-1),
        ),
    }
    # Round each weight once to the dtype its call runs in, so the oracle reads
    # the same values the kernel does.
    with torch.no_grad():
        for name, _, _, _, _, dtype in SHAPES.values():
            for weight in projections[name][0]:
                weight.copy_(weight.to(dtype).double())
    return projections


def _kernel(attention, weight):
    """Run ``o_proj_tkg`` where its kernel runs and return the output on the host."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        if os.environ.get("NKI_SIMULATOR") != "1":
            pytest.skip("the kernel runs on the host only under NKI_SIMULATOR=1")
        return o_proj_tkg(attention, weight)
    from libtorch_neuronx_lite.compile.native_backend import register

    device = torch.device("neuron", 0)
    compiled = torch.compile(o_proj_tkg, backend=register(), fullgraph=True, dynamic=False)
    return compiled(attention.to(device), weight.to(device)).cpu()


def _project(reference, shape, x):
    """Kernel output and fp64 oracle for token-major rows ``x`` [rows, K]."""
    name, K, D, rows, _, dtype = SHAPES[shape]
    weights, module = reference[name]
    weight = torch.cat([w.detach() for w in weights], dim=0).t().to(dtype).contiguous()
    xq = x.to(dtype)
    attention = xq.view(rows, K // D, D).permute(2, 0, 1).unsqueeze(-1).contiguous()
    out = _kernel(attention, weight)
    with torch.no_grad():
        ref = module(xq.double())
    return out, ref


def _assert_class_b(out, ref):
    """Class B: |err| <= 2^-7 |ref| + 2^-7 row RMS; row cosine >= 0.9999."""
    assert out.shape == ref.shape
    out = out.double()
    rms = ref.pow(2).mean(-1, keepdim=True).sqrt()
    over = ~((out - ref).abs() <= 2**-7 * ref.abs() + 2**-7 * rms)
    assert int(over.sum()) == 0, f"{int(over.sum())} elements outside class B"
    live = rms.squeeze(-1) > 0
    cosine = F.cosine_similarity(out[live], ref[live], dim=-1)
    assert float(cosine.min()) >= 0.9999, f"row cosine {float(cosine.min())}"


def _rows(shape, live_rows):
    """Pad live rows with the other replica's zero rows."""
    _, K, _, rows, live, _ = SHAPES[shape]
    x = torch.zeros(rows, K, dtype=torch.float64)
    x[:live] = live_rows
    return x


def _generator(shape, regime):
    return torch.Generator().manual_seed(SEED + 97 * list(SHAPES).index(shape) + regime)


@pytest.mark.parametrize("shape", SHAPES)
def test_correlated_rows(reference, shape):
    """Rows 0.9 u + 0.1 noise, one input channel 100x the rest."""
    _, K, _, _, live, _ = SHAPES[shape]
    g = _generator(shape, 1)
    u = torch.randn(K, generator=g, dtype=torch.float64)
    x = 0.9 * u + 0.1 * torch.randn(live, K, generator=g, dtype=torch.float64)
    x[:, K // 3] *= 100
    _assert_class_b(*_project(reference, shape, _rows(shape, x)))


@pytest.mark.parametrize("shape", SHAPES)
def test_repeated_tokens(reference, shape):
    """Every live row equal."""
    _, K, _, _, live, _ = SHAPES[shape]
    g = _generator(shape, 2)
    x = torch.randn(1, K, generator=g, dtype=torch.float64).expand(live, K)
    _assert_class_b(*_project(reference, shape, _rows(shape, x)))


@pytest.mark.parametrize("shape", SHAPES)
def test_near_one_gates(reference, shape):
    """Rows gated by sigmoid at +12 and -12 on alternate channels."""
    _, K, _, _, live, _ = SHAPES[shape]
    g = _generator(shape, 3)
    gate = torch.where(torch.arange(K) % 2 == 0, 12.0, -12.0).double()
    x = torch.randn(live, K, generator=g, dtype=torch.float64) * torch.sigmoid(gate)
    _assert_class_b(*_project(reference, shape, _rows(shape, x)))


@pytest.mark.parametrize("shape", SHAPES)
def test_long_carried_state(reference, shape):
    """Rows a 131,072-token attention average hands over: small, near-equal."""
    _, K, _, _, live, _ = SHAPES[shape]
    g = _generator(shape, 4)
    scale = 1 / math.sqrt(131072)
    common = torch.randn(1, K, generator=g, dtype=torch.float64) * scale
    x = common + 0.01 * scale * torch.randn(live, K, generator=g, dtype=torch.float64)
    _assert_class_b(*_project(reference, shape, _rows(shape, x)))


@pytest.mark.parametrize("shape", SHAPES)
def test_one_nan_row(reference, shape):
    """A NaN row gives a NaN output row and leaves every other row's bits."""
    _, K, _, rows, live, _ = SHAPES[shape]
    g = _generator(shape, 5)
    clean = _rows(shape, torch.randn(live, K, generator=g, dtype=torch.float64))
    poisoned = clean.clone()
    poisoned[1] = float("nan")
    out_clean, _ = _project(reference, shape, clean)
    out_nan, _ = _project(reference, shape, poisoned)
    others = [r for r in range(rows) if r != 1]
    assert bool(torch.isnan(out_nan[1]).all()), "the NaN row's output is not all NaN"
    assert torch.equal(out_nan[others], out_clean[others]), "a NaN row changed another row"
    assert int((out_nan[live:] != 0).sum()) == 0, "a zero row's output is not zero"
