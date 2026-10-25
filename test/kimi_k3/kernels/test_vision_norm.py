# SPDX-License-Identifier: Apache-2.0
"""fp64 oracle for ``rmsnorm_residual_prefill`` at the vision tower's two norms.

The tower's last residual add and ``final_layernorm`` run with the residual on, at
width 1024, on the 688 real rows of an image slice, the 5,504-row bucket and the
33,536-row bucket. The projector's ``post_norm`` runs with no residual, at width
7168, on the 172 gathered rows of a slice, one TP4 rank's 344 rows of the served
bucket and its 2,096 rows of the largest. The weights are the checkpoint's, read
from ``KIMI_K3_MODEL_PATH``. The oracle is the checkpoint's add and ``nn.RMSNorm``
in fp64 on the CPU. Every item fails if the wrapper takes its torch path.

The kernel runs on a Neuron device through the native compile backend, or on the
host under ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import importlib
import json
import os
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm_neuron.functional.rmsnorm_residual_prefill import rmsnorm_residual_prefill

# The package exports a function of the same name, so take the module itself.
norm_module = importlib.import_module("vllm_neuron.functional.rmsnorm_residual_prefill")

SEED = 20261024
#: ``nn.RMSNorm(eps=None)`` on bf16 activations takes the float32 epsilon.
FINAL_EPS = float(torch.finfo(torch.float32).eps)
POST_EPS = 1e-5
#: Largest error allowed per element, as a multiple of its class-B bound.
F_BOUND = 1.0
#: name: (rows, width, residual on, eps, weight key)
CALLS = {
    "final_688": (688, 1024, True, FINAL_EPS, "final"),
    "final_5504": (5504, 1024, True, FINAL_EPS, "final"),
    "post_172": (172, 7168, False, POST_EPS, "post"),
    "post_344": (344, 7168, False, POST_EPS, "post"),
}
LARGEST = {
    "final_33536": (33536, 1024, True, FINAL_EPS, "final"),
    "post_2096": (2096, 7168, False, POST_EPS, "post"),
}
WEIGHTS = {
    "final": "vision_tower.encoder.final_layernorm.weight",
    "post": "mm_projector.post_norm.weight",
}
#: Rows whose pre-norm mean square is this many epsilons, so the epsilon sets their scale.
NEAR_ZERO = {5: 0.25, 6: 1.0, 7: 4.0}
NAN_ROW = 37


@pytest.fixture(scope="module")
def weights():
    """The two norm weights, bf16, as the checkpoint stores them."""
    path = os.environ.get("KIMI_K3_MODEL_PATH")
    if not path or not os.path.isfile(os.path.join(path, "model.safetensors.index.json")):
        pytest.skip("KIMI_K3_MODEL_PATH must name the Kimi K3 checkpoint directory")
    with open(os.path.join(path, "model.safetensors.index.json")) as fh:
        shards = json.load(fh)["weight_map"]
    out = {}
    for key, name in WEIGHTS.items():
        with safe_open(os.path.join(path, shards[name]), framework="pt") as f:
            out[key] = f.get_tensor(name)
    return out


def _norm(hidden, gamma, residual, eps):
    return rmsnorm_residual_prefill(hidden, gamma, residual, eps=eps)


def _kernel(hidden, gamma, residual, eps):
    """Run the call where its kernel runs and return the normed rows on the host."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        if os.environ.get("NKI_SIMULATOR") != "1":
            pytest.skip("the kernel runs on the host only under NKI_SIMULATOR=1")
        return _norm(hidden, gamma, residual, eps)[0]
    from libtorch_neuronx_lite.compile.native_backend import register

    device = torch.device("neuron", 0)
    compiled = torch.compile(_norm, backend=register(), fullgraph=True, dynamic=False)
    moved = [t if t is None else t.to(device) for t in (hidden, gamma, residual)]
    return compiled(*moved, eps)[0].cpu()


def _no_torch_path(*args, **kwargs):
    raise AssertionError("rmsnorm_residual_prefill took its torch path instead of the NKI kernel")


def _oracle(hidden, gamma, residual, eps):
    """The checkpoint's ``residual + hidden`` and ``nn.RMSNorm`` in fp64."""
    x = hidden.double() if residual is None else residual.double() + hidden.double()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * gamma.double()


def _run(gamma, hidden, residual, eps):
    """Kernel output and fp64 oracle for bf16 rows."""
    hidden = hidden.to(torch.bfloat16)
    residual = None if residual is None else residual.to(torch.bfloat16)
    with mock.patch.object(norm_module, "_torch_impl", _no_torch_path):
        out = _kernel(hidden, gamma, residual, eps)
    return out, _oracle(hidden, gamma, residual, eps)


def _assert_in_class(out, ref):
    """bf16 out; each |err| within F_BOUND x (2^-7 |ref| + 2^-7 row RMS); row cosine >= 0.9999."""
    assert out.dtype == torch.bfloat16, f"output dtype {out.dtype}"
    assert out.shape == ref.shape, f"output shape {tuple(out.shape)}"
    out = out.double()
    rms = ref.square().mean(-1, keepdim=True).sqrt()
    over = ~((out - ref).abs() <= F_BOUND * 2**-7 * (ref.abs() + rms))
    assert int(over.sum()) == 0, f"{int(over.sum())} elements outside the bound"
    cosine = F.cosine_similarity(out, ref, dim=-1)
    assert float(cosine.min()) >= 0.9999, f"row cosine {float(cosine.min())}"


def _generator(rows, regime):
    return torch.Generator().manual_seed(SEED + 97 * rows + regime)


def _correlated(rows, width, g):
    """Rows 0.9 u + 0.1 noise, one channel 100x the rest."""
    u = torch.randn(width, generator=g, dtype=torch.float64)
    x = 0.9 * u + 0.1 * torch.randn(rows, width, generator=g, dtype=torch.float64)
    x[:, width // 3] *= 100
    return x


def _correlated_inputs(rows, width, residual):
    g = _generator(rows, 1)
    hidden = _correlated(rows, width, g)
    return hidden, _correlated(rows, width, g) if residual else None


def _near_zero_inputs(rows, width, residual, eps):
    """Correlated rows, with rows whose pre-norm mean square is 0.25, 1 and 4 epsilons."""
    g = _generator(rows, 2)
    hidden = _correlated(rows, width, g)
    res = _correlated(rows, width, g) if residual else None
    for row, scale in NEAR_ZERO.items():
        a = torch.randn(width, generator=g, dtype=torch.float64)
        b = torch.randn(width, generator=g, dtype=torch.float64) if residual else 0 * a
        size = (scale * eps / (a + b).square().mean()).sqrt()
        hidden[row] = a * size
        if residual:
            res[row] = b * size
    return hidden, res


def _repeated_inputs(rows, width, residual):
    """Every row the same token."""
    g = _generator(rows, 3)
    hidden = torch.randn(1, width, generator=g, dtype=torch.float64).expand(rows, width)
    if not residual:
        return hidden, None
    return hidden, torch.randn(1, width, generator=g, dtype=torch.float64).expand(rows, width)


def _random_inputs(rows, width, residual, regime):
    """Standard normal rows."""
    g = _generator(rows, regime)
    hidden = torch.randn(rows, width, generator=g, dtype=torch.float64)
    return hidden, torch.randn(rows, width, generator=g, dtype=torch.float64) if residual else None


SERVED = pytest.mark.parametrize("call", list(CALLS))


@SERVED
def test_correlated_rows(weights, call):
    rows, width, residual, eps, key = CALLS[call]
    _assert_in_class(*_run(weights[key], *_correlated_inputs(rows, width, residual), eps))


@SERVED
def test_near_zero_rows(weights, call):
    rows, width, residual, eps, key = CALLS[call]
    _assert_in_class(*_run(weights[key], *_near_zero_inputs(rows, width, residual, eps), eps))


@SERVED
def test_repeated_tokens(weights, call):
    rows, width, residual, eps, key = CALLS[call]
    _assert_in_class(*_run(weights[key], *_repeated_inputs(rows, width, residual), eps))


@pytest.mark.parametrize("call", list(LARGEST))
def test_largest_bucket(weights, call):
    rows, width, residual, eps, key = LARGEST[call]
    _assert_in_class(*_run(weights[key], *_random_inputs(rows, width, residual, 4), eps))


@SERVED
def test_one_nan_row(weights, call):
    """A NaN row gives a NaN output row and leaves every other row's bits."""
    rows, width, residual, eps, key = CALLS[call]
    hidden, res = _random_inputs(rows, width, residual, 5)
    poisoned = hidden.clone()
    poisoned[NAN_ROW] = float("nan")
    out_clean, ref = _run(weights[key], hidden, res, eps)
    out_nan, _ = _run(weights[key], poisoned, res, eps)
    others = [r for r in range(rows) if r != NAN_ROW]
    _assert_in_class(out_clean, ref)
    assert bool(torch.isnan(out_nan[NAN_ROW]).all()), "the NaN row's output is not all NaN"
    assert torch.equal(out_nan[others].view(torch.int16), out_clean[others].view(torch.int16)), (
        "a NaN row changed another row")
