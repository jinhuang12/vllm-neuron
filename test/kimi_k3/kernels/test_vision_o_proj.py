# SPDX-License-Identifier: Apache-2.0
"""fp64 oracle for ``o_proj`` on the vision tower's attention output, class B.

The reference is ``MoonViTEncoderLayer.wo`` from the checkpoint's
``modeling_kimi_k3.py`` under ``KIMI_K3_MODEL_PATH``, built at one tower TP4
rank's slice: 3 heads of 128 into the hidden size 1024, with the checkpoint's
bias setting. The rows reach ``o_proj`` token-major, ``[1, T, 3, 128]`` in the
BSNd layout, at the 768-row image slice, the 5504-patch bucket and the
33536-patch bucket. Every item fails if ``o_proj`` takes its torch path.

The kernel runs on a Neuron device through the native compile backend, or on
the host under ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import importlib
import json
import os
import sys
import types
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from nkilib.core.utils.common_types import OProjAttentionLayout

import vllm_neuron.functional.attention.o_proj as o_proj_module
from vllm_neuron.functional.attention.o_proj import o_proj

TP = 4
SEED = 20261024
SLICE_ROWS = 768
BUCKET_ROWS = 5504
LARGEST_BUCKET_ROWS = 33536
IMAGE_PATCHES = 5476  # one 1024 x 1024 image: a 74 x 74 grid
NAN_ROW = 200


def _import_vision_modeling(path):
    """Import the checkpoint's tower code without its text model.

    The tower never calls ``modeling_kimi_linear``, which does not import under
    transformers 5, so a stand-in module takes its name.
    """
    package = types.ModuleType("kimi_k3_vision_reference")
    package.__path__ = [path]
    text_model = types.ModuleType("kimi_k3_vision_reference.modeling_kimi_linear")
    text_model.KimiLinearForCausalLM = object
    sys.modules[package.__name__] = package
    sys.modules[text_model.__name__] = text_model
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module("kimi_k3_vision_reference.modeling_kimi_k3")
    finally:
        sys.dont_write_bytecode = write_bytecode


def _build_reference(path):
    """The checkpoint's ``wo`` at one rank's slice in fp64, its weight rounded once to bf16."""
    with open(os.path.join(path, "config.json")) as fh:
        vision = json.load(fh)["vision_config"]
    modeling = _import_vision_modeling(path)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(SEED)
    try:
        layer = modeling.MoonViTEncoderLayer(
            num_heads=vision["vt_num_attention_heads"] // TP,
            hidden_dim=vision["vt_hidden_size"],
            mlp_dim=vision["vt_intermediate_size"] // TP,
            qkv_hidden_size=vision["qkv_hidden_size"] // TP,
            norm_type=vision["norm_type"],
            mlp_type=vision["mlp_type"],
            attn_bias=vision["attn_bias"],
            linear_bias=vision["linear_bias"],
        )
    finally:
        torch.set_default_dtype(default_dtype)
    with torch.no_grad():
        layer.wo.weight.copy_(layer.wo.weight.to(torch.bfloat16).double())
    return layer


@pytest.fixture(scope="module")
def reference():
    """The checkpoint's encoder layer at one rank's slice; the oracle is its ``wo``."""
    path = os.environ.get("KIMI_K3_MODEL_PATH")
    if not path or not os.path.isfile(os.path.join(path, "modeling_kimi_k3.py")):
        pytest.skip("KIMI_K3_MODEL_PATH must name the Kimi K3 checkpoint directory")
    return _build_reference(path)


def _vision_o_proj(attention, weight):
    """The tower's call: token-major attention rows, no bias."""
    return o_proj(attention, weight, attention_layout=OProjAttentionLayout.BSNd)


def _kernel(attention, weight):
    """Run the tower's call where its kernel runs and return the output on the host."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        if os.environ.get("NKI_SIMULATOR") != "1":
            pytest.skip("the kernel runs on the host only under NKI_SIMULATOR=1")
        return _vision_o_proj(attention, weight)
    from libtorch_neuronx_lite.compile.native_backend import register

    device = torch.device("neuron", 0)
    compiled = torch.compile(_vision_o_proj, backend=register(), fullgraph=True, dynamic=False)
    return compiled(attention.to(device), weight.to(device)).cpu()


def _no_torch_path(*args, **kwargs):
    raise AssertionError("o_proj took its torch path instead of the NKI kernel")


def _project(layer, x):
    """Kernel output [T, H] and fp64 oracle for attention rows ``x`` [T, heads * 128]."""
    xq = x.to(torch.bfloat16)
    attention = xq.view(1, x.shape[0], layer.num_heads, layer.hidden_size_per_attention_head)
    weight = layer.wo.weight.detach().t().to(torch.bfloat16).contiguous()
    with mock.patch.object(o_proj_module, "_torch_o_proj_impl", _no_torch_path):
        out = _kernel(attention, weight)
    with torch.no_grad():
        ref = layer.wo(xq.double())
    return out.squeeze(0), ref


def _assert_class_b(out, ref):
    """Class B: bf16 out; |err| <= 2^-7 |ref| + 2^-7 row RMS; row cosine >= 0.9999."""
    assert out.dtype == torch.bfloat16, f"output dtype {out.dtype}"
    assert out.shape == ref.shape, f"output shape {tuple(out.shape)}"
    out = out.double()
    rms = ref.pow(2).mean(-1, keepdim=True).sqrt()
    over = ~((out - ref).abs() <= 2**-7 * ref.abs() + 2**-7 * rms)
    assert int(over.sum()) == 0, f"{int(over.sum())} elements outside class B"
    cosine = F.cosine_similarity(out, ref, dim=-1)
    assert float(cosine.min()) >= 0.9999, f"row cosine {float(cosine.min())}"


def _generator(rows, regime):
    return torch.Generator().manual_seed(SEED + 97 * rows + regime)


def _correlated_rows(rows, width):
    """Rows 0.9 u + 0.1 noise, one input channel 100x the rest."""
    g = _generator(rows, 1)
    u = torch.randn(width, generator=g, dtype=torch.float64)
    x = 0.9 * u + 0.1 * torch.randn(rows, width, generator=g, dtype=torch.float64)
    x[:, width // 3] *= 100
    return x


def _repeated_tokens(rows, width):
    """Every row equal."""
    g = _generator(rows, 2)
    return torch.randn(1, width, generator=g, dtype=torch.float64).expand(rows, width)


def _equal_scores(rows, width, heads):
    """Attention output under equal scores: every row the image's mean value row per head."""
    g = _generator(rows, 3)
    v = torch.randn(heads, IMAGE_PATCHES, width // heads, generator=g, dtype=torch.float64)
    p = torch.softmax(torch.zeros(IMAGE_PATCHES, dtype=torch.float64), dim=0)
    return torch.einsum("k,nkd->nd", p, v).reshape(1, width).expand(rows, width)


def _random_rows(rows, width, regime):
    """Standard normal rows."""
    return torch.randn(rows, width, generator=_generator(rows, regime), dtype=torch.float64)


ROWS = pytest.mark.parametrize("rows", (SLICE_ROWS, BUCKET_ROWS))


@ROWS
def test_correlated_rows(reference, rows):
    x = _correlated_rows(rows, reference.wo.in_features)
    _assert_class_b(*_project(reference, x))


@ROWS
def test_repeated_tokens(reference, rows):
    x = _repeated_tokens(rows, reference.wo.in_features)
    _assert_class_b(*_project(reference, x))


@ROWS
def test_equal_scores(reference, rows):
    x = _equal_scores(rows, reference.wo.in_features, reference.num_heads)
    _assert_class_b(*_project(reference, x))


def test_largest_bucket(reference):
    x = _random_rows(LARGEST_BUCKET_ROWS, reference.wo.in_features, 4)
    _assert_class_b(*_project(reference, x))


@ROWS
def test_one_nan_row(reference, rows):
    """A NaN row gives a NaN output row and leaves every other row's bits."""
    clean = _random_rows(rows, reference.wo.in_features, 5)
    poisoned = clean.clone()
    poisoned[NAN_ROW] = float("nan")
    out_clean, _ = _project(reference, clean)
    out_nan, _ = _project(reference, poisoned)
    others = [r for r in range(rows) if r != NAN_ROW]
    assert bool(torch.isnan(out_nan[NAN_ROW]).all()), "the NaN row's output is not all NaN"
    assert torch.equal(out_nan[others], out_clean[others]), "a NaN row changed another row"
