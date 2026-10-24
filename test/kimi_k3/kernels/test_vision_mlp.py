# SPDX-License-Identifier: Apache-2.0
"""The vision tower's MLP block in one fused call, against a float64 oracle.

``mlp`` runs at the tower's per-rank shape at TP 4: the attention output added to
the residual and the sum stored, RMSNorm, fc0, tanh GELU and fc1, with no gate and
1,024 MLP features per rank. The weights are the checkpoint's ``norm1``, ``fc0`` and
``fc1`` for rank 3, cut by the vision weight loader. The oracle is the checkpoint's
``RMSNorm`` and ``MLP2`` in float64 on the CPU, at that rank's features. The
checkpoint directory comes from ``KIMI_K3_CHECKPOINT_DIR``. With
``VLLM_NEURON_CPU_MODE=1`` and ``NKI_SIMULATOR=1`` the kernel runs on the NKI
simulator; otherwise on ``neuron:0``.

The kernel stores the sum, the normed rows (twice), the GELU output and its own
output in bf16. ``BOUND_FACTOR`` is the kernel's own worst reading on the NKI
simulator over these items, in units of the bf16 element bound against float64,
rounded up to the next 0.5. Each element is held to that multiple of the bound, and
the plain bound's reading is printed beside it. The tanh and erf GELU forms differ
by less than any such bound, so one item reads the activation the kernel receives.
"""

from __future__ import annotations

import json
import math
import operator
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from nkilib.core.utils.common_types import ActFnType, NormType
from safetensors import safe_open

from vllm_neuron import envs
from vllm_neuron.functional.mlp import mlp
from vllm_neuron.model.kimi_k3.weight_loaders_vision import vision_weight_loader

CHECKPOINT_DIR = os.environ.get("KIMI_K3_CHECKPOINT_DIR")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT_DIR, reason="KIMI_K3_CHECKPOINT_DIR is not set, so the reference is absent")

ON_DEVICE = not envs.VLLM_NEURON_CPU_MODE

RANK, TP = 3, 4
HIDDEN = 1024
#: Layer 1 for most items; layer 26, whose gamma-scaled fc0 rows are the widest,
#: is the one where unit-RMS rows reach GELU inputs of 20.
LAYER, WIDE_LAYER = 1, 26
#: ``nn.RMSNorm(eps=None)`` on float32 activations.
EPS = float(torch.finfo(torch.float32).eps)
SERVED = (768, 5504)
LARGEST = 33536
LARGE_CHANNEL = 341
#: Rows whose sum has a mean square of this many epsilons, so the epsilon sets their scale.
NEAR_ZERO = {5: 0.25, 6: 1.0, 7: 4.0}
#: Largest magnitude of the wide item's sums and of its GELU inputs.
WIDE = 20.0
NAN_ROW = 37
#: The kernel's worst reading on the NKI simulator over these items, rounded up to 0.5.
BOUND_FACTOR = 5.5
MATMULS = {operator.matmul, torch.matmul, torch.mm, torch.bmm, torch.einsum, F.linear}


def _layer_weights(layer):
    """norm1 gamma, the kernel's fc0 [1024, 1024] and fc1 [1024, 1024], and the stored rank slices."""
    root = Path(CHECKPOINT_DIR)
    shards = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"vision_tower.encoder.blocks.{layer}."
    out = {}
    for name in ("norm1.weight", "mlp.fc0.weight", "mlp.fc1.weight"):
        key = prefix + name
        with safe_open(root / shards[key], framework="pt") as f:
            out[name] = (f.get_tensor(key), vision_weight_loader(key, RANK, TP).load([f.get_slice(key)], RANK))
    features = slice(RANK * HIDDEN, (RANK + 1) * HIDDEN)
    return {"gamma": out["norm1.weight"][0], "fc0": out["mlp.fc0.weight"][1].contiguous(),
            "fc1": out["mlp.fc1.weight"][1].contiguous(), "fc0_rows": out["mlp.fc0.weight"][0][features],
            "fc1_cols": out["mlp.fc1.weight"][0][:, features]}


@pytest.fixture(scope="module")
def weights():
    """Rank 3's MLP weights at the two layers the items read."""
    return {layer: _layer_weights(layer) for layer in (LAYER, WIDE_LAYER)}


def oracle(hidden, add, w):
    """The checkpoint's residual add, norm1 and MLP2 in float64 at the rank's features: (out, sum)."""
    total = hidden.double() + add.double()
    normed = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + EPS) * w["gamma"].double()
    gelu = F.gelu(normed @ w["fc0_rows"].double().t(), approximate="tanh")
    return gelu @ w["fc1_cols"].double().t(), total


_KERNEL_CALLS = {}
_ACTIVATIONS = {}


def _record(gm, example_inputs):
    """Count NKI kernel calls and torch matmuls per shape, note the kernel's activation, then run the graph."""
    tokens = next(t.shape[0] for t in example_inputs
                  if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] not in (1, HIDDEN))
    nodes = [n for n in gm.graph.nodes if n.op in ("call_function", "call_method")]
    kernels = [n for n in nodes if getattr(n.target, "_name", None) == "nki_kernel_wrapper"]
    matmuls = sum(n.target in MATMULS if n.op == "call_function" else n.target in ("matmul", "mm")
                  for n in nodes)
    _KERNEL_CALLS[tokens] = (len(kernels), matmuls)
    _ACTIVATIONS[tokens] = [dict(zip(n.kwargs["arg_names"], n.kwargs["args"])).get("activation_fn")
                            for n in kernels]
    if not ON_DEVICE:
        return gm.forward
    return torch._dynamo.lookup_backend(envs.get_compile_backend_name())(gm, example_inputs)


def _call(hidden, add, fc0, fc1, gamma):
    return mlp(hidden, fc0, fc0, fc1, eps=EPS, ln_w=gamma.reshape(1, -1), skip_gate=True,
               act_fn=ActFnType.GELU_Tanh_Approx, norm_type=NormType.RMS_NORM,
               fused_add_tensor=add, store_fused_add_result=True)


_COMPILED = torch.compile(_call, backend=_record, fullgraph=True, dynamic=False)


def run_mlp(hidden, add, w):
    """Run the call; fail unless it ran one NKI kernel and no torch matmul. Returns (out, sum)."""
    args = (hidden, add, w["fc0"], w["fc1"], w["gamma"])
    if ON_DEVICE:
        out, total = (t.cpu() for t in _COMPILED(*(a.to("neuron:0") for a in args)))
    else:
        out, total = _COMPILED(*args)
    nki, matmuls = _KERNEL_CALLS[hidden.shape[0]]
    assert (nki, matmuls) == (1, 0), f"T {hidden.shape[0]}: {nki} NKI kernel calls, {matmuls} torch matmuls"
    return out, total


def bound_reading(out, ref):
    """Worst error over the bf16 element bound with its row and column, elements over it, lowest row cosine."""
    out, ref = out.double().reshape(-1, ref.shape[-1]), ref.reshape(-1, ref.shape[-1])
    rms = ref.square().mean(-1, keepdim=True).sqrt()
    excess = ((out - ref).abs() / (2 ** -7 * (ref.abs() + rms))).flatten()
    worst = int(excess.argmax())
    cosine = F.cosine_similarity(out, ref, dim=-1).amin().item()
    return excess[worst].item(), divmod(worst, ref.shape[-1]), int((excess > 1.0).sum()), cosine


def _check(cases):
    """Run each (label, hidden, add, weights) case; report every output outside the bound times the factor."""
    misses = []
    for label, hidden, add, w in cases:
        hidden, add = hidden.bfloat16(), add.bfloat16()
        for name, got, ref in zip(("out", "sum"), run_mlp(hidden, add, w), oracle(hidden, add, w)):
            excess, place, over, cosine = bound_reading(got, ref)
            print(f"{label} {name}: worst error / bound {excess:.3f} at row {place[0]}, column {place[1]}; "
                  f"{over} elements over the plain bound; lowest row cosine {cosine:.7f}")
            if not (excess <= BOUND_FACTOR and cosine >= 0.9999):
                misses.append(f"{label} {name}")
    assert not misses, f"outside {BOUND_FACTOR} times the bound: {misses}"


def correlated(tokens, gen):
    """Rows near one direction, one channel at 100 times the rest, and three near-zero sums."""
    hidden = 0.9 * torch.randn(HIDDEN, generator=gen) + 0.1 * torch.randn(tokens, HIDDEN, generator=gen)
    hidden[:, LARGE_CHANNEL] *= 100
    add = 0.25 * torch.randn(tokens, HIDDEN, generator=gen)
    for row, scale in NEAR_ZERO.items():
        noise = torch.randn(HIDDEN, generator=gen)
        hidden[row] = noise * (scale * EPS / noise.square().mean()).sqrt()
        add[row] = 0.0
    return hidden, add


def wide(tokens, gen, w):
    """Sums up to 20 in magnitude whose GELU input at the widest fc0 row sweeps -20 to 20."""
    rows = w["gamma"].double() * w["fc0_rows"].double()
    lengths = rows.norm(dim=1)
    direction = rows[int(lengths.argmax())] / lengths.max()
    cos = torch.linspace(-WIDE, WIDE, tokens, dtype=torch.float64) / (math.sqrt(HIDDEN) * lengths.max())
    noise = torch.randn(tokens, HIDDEN, generator=gen, dtype=torch.float64)
    noise -= (noise @ direction)[:, None] * direction
    noise /= noise.norm(dim=1, keepdim=True)
    unit = math.sqrt(HIDDEN) * (cos[:, None] * direction + (1 - cos.square()).sqrt()[:, None] * noise)
    total = unit * (WIDE / unit.abs().amax(dim=1, keepdim=True))
    add = (0.25 * torch.randn(tokens, HIDDEN, generator=gen)).bfloat16()
    return (total - add.double()).float(), add.float()


def repeated(tokens, gen):
    """Every row the same token and the same attention output."""
    return (torch.randn(1, HIDDEN, generator=gen).expand(tokens, HIDDEN).contiguous(),
            0.25 * torch.randn(1, HIDDEN, generator=gen).expand(tokens, HIDDEN).contiguous())


def random_rows(tokens, gen):
    """Independent standard-normal rows and a quarter-scale attention output."""
    return torch.randn(tokens, HIDDEN, generator=gen), 0.25 * torch.randn(tokens, HIDDEN, generator=gen)


def correlated_cases(weights):
    gen = torch.Generator().manual_seed(1)
    return [(f"T {t}", *correlated(t, gen), weights[LAYER]) for t in SERVED]


def wide_cases(weights):
    gen = torch.Generator().manual_seed(2)
    return [(f"T {t} layer {WIDE_LAYER}", *wide(t, gen, weights[WIDE_LAYER]), weights[WIDE_LAYER]) for t in SERVED]


def repeated_cases(weights):
    gen = torch.Generator().manual_seed(3)
    return [(f"T {t}", *repeated(t, gen), weights[LAYER]) for t in SERVED]


def largest_cases(weights):
    gen = torch.Generator().manual_seed(4)
    return [(f"T {LARGEST}", *random_rows(LARGEST, gen), weights[LAYER])]


def nan_cases(weights):
    gen = torch.Generator().manual_seed(5)
    return [(f"T {t}", *random_rows(t, gen), weights[LAYER]) for t in SERVED]


def test_kernel_runs_tanh_gelu(weights):
    """The kernel receives the tanh form of GELU at both served shapes."""
    gen = torch.Generator().manual_seed(6)
    for tokens in SERVED:
        hidden, add = random_rows(tokens, gen)
        run_mlp(hidden.bfloat16(), add.bfloat16(), weights[LAYER])
        assert _ACTIVATIONS[tokens] == [ActFnType.GELU_Tanh_Approx], f"T {tokens}: {_ACTIVATIONS[tokens]}"


def test_correlated_rows_with_one_large_channel(weights):
    """Rows near one direction with one channel at 100x, and sums near zero where eps sets the scale."""
    _check(correlated_cases(weights))


def test_inputs_up_to_twenty(weights):
    """Sums and GELU inputs up to 20, across the range where tanh GELU leaves erf GELU."""
    _check(wide_cases(weights))


def test_repeated_tokens(weights):
    """Every row the same token."""
    _check(repeated_cases(weights))


def test_largest_bucket(weights):
    """The 33,536-row block of the largest image the tower accepts."""
    _check(largest_cases(weights))


def test_nan_row_stays_in_its_row(weights):
    """One all-NaN row: its outputs are NaN, and every other row equals the clean run."""
    misses = []
    for label, hidden, add, w in nan_cases(weights):
        hidden, add = hidden.bfloat16(), add.bfloat16()
        clean = run_mlp(hidden, add, w)
        for name, got, ref in zip(("out", "sum"), clean, oracle(hidden, add, w)):
            excess, place, over, cosine = bound_reading(got, ref)
            print(f"{label} {name}: clean worst error / bound {excess:.3f} at row {place[0]}, column {place[1]}; "
                  f"{over} elements over the plain bound; lowest row cosine {cosine:.7f}")
            if not (excess <= BOUND_FACTOR and cosine >= 0.9999):
                misses.append(f"{label} {name}: clean run outside {BOUND_FACTOR} times the bound")
        poisoned = hidden.clone()
        poisoned[NAN_ROW] = float("nan")
        others = torch.arange(hidden.shape[0]) != NAN_ROW
        for name, got, base in zip(("out", "sum"), run_mlp(poisoned, add, w), clean):
            if not torch.isnan(got[NAN_ROW]).all():
                misses.append(f"{label} {name}: the NaN row has a finite output")
            if not torch.equal(got[others].view(torch.int16), base[others].view(torch.int16)):
                misses.append(f"{label} {name}: other rows changed")
    assert not misses, misses
