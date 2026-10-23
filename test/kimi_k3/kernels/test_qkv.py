# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm and projection at the Kimi K3 call shapes, against a float64 oracle.

``qkv_proj_fp32acc`` runs at each per-rank width K3 calls it with at TP 32. The 4-row
regimes run twice: through the wrapper, which picks the decode kernel, and on the
prefill kernel directly. The oracle is the checkpoint's own ``KimiRMSNorm`` followed by
``nn.Linear``, in float64 on the CPU (the RMS statistic stays in float32, as the
reference computes it). The checkpoint directory comes from ``KIMI_K3_CHECKPOINT_DIR``.
With ``VLLM_NEURON_CPU_MODE=1`` and ``NKI_SIMULATOR=1`` the kernels run on the NKI
simulator; otherwise on ``neuron:0``.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import nki
import pytest
import torch
import torch.nn.functional as F
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron import envs
from vllm_neuron.functional.attention.qkv_fp32acc import qkv_proj_fp32acc
from vllm_neuron.functional.attention.qkv_fp32acc_cte import qkv_fp32acc_cte
from vllm_neuron.utils.hardware_config import get_logical_nc_config

CHECKPOINT_DIR = os.environ.get("KIMI_K3_CHECKPOINT_DIR")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT_DIR, reason="KIMI_K3_CHECKPOINT_DIR is not set, so the reference is absent")

ON_DEVICE = not envs.VLLM_NEURON_CPU_MODE

#: Output columns per rank: KDA q, k, v, g, f_a and b; the MLA output gate and the
#: shared-expert gate/up; one of the two LM-head calls.
DECODE_WIDTHS = (1667, 384, 2560)
#: The prefill chunk runs the KDA and 384-column projections; the LM head reads one token.
PREFILL_WIDTHS = (1667, 384)
DECODE_TOKENS = 4
PREFILL_TOKENS = 8192
GATE_INPUT = 12.0
NAN_ROW = 2


@pytest.fixture(scope="module")
def reference():
    """The checkpoint's hidden size, epsilon and ``KimiRMSNorm`` class."""
    root = Path(CHECKPOINT_DIR)
    text = json.loads((root / "config.json").read_text())
    text = text.get("text_config", text)
    source = (root / "modeling_kimi_linear.py").read_text()
    norm_class = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.ClassDef) and node.name == "KimiRMSNorm")
    scope = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module([norm_class], []), "modeling_kimi_linear.py", "exec"), scope)
    return text["hidden_size"], text["rms_norm_eps"], scope["KimiRMSNorm"]


@nki.jit
def _prefill_kernel(hidden, qkv_weights, norm_weights, eps: float):
    return qkv_fp32acc_cte(hidden, qkv_weights, norm_weights, eps)


_PREFILL = wrap_nki(_prefill_kernel)


def _project_prefill(hidden, weight, gamma, eps):
    return _PREFILL[get_logical_nc_config()](hidden=hidden, qkv_weights=weight,
                                             norm_weights=gamma.reshape(1, -1), eps=eps)


#: The wrapper's own choice, and the prefill kernel for inputs the wrapper sends to decode.
PATHS = {"wrapper": qkv_proj_fp32acc, "prefill": _project_prefill}
_compiled = {name: torch.compile(fn, backend=envs.get_compile_backend_name(), fullgraph=True,
                                 dynamic=False) for name, fn in PATHS.items()} if ON_DEVICE else None


def _run(hidden, weight, gamma, eps, path="wrapper"):
    if not ON_DEVICE:
        return PATHS[path](hidden, weight, gamma, eps)
    on = [t.to("neuron:0") for t in (hidden, weight, gamma)]
    return _compiled[path](*on, eps).cpu()


def _normed(reference, hidden, gamma):
    size, eps, norm_class = reference
    norm = norm_class(size, eps=eps).double()
    with torch.no_grad():
        norm.weight.copy_(gamma.reshape(-1))
        return norm(hidden.double())


def _oracle(reference, hidden, weight, gamma):
    proj = torch.nn.Linear(weight.shape[0], weight.shape[1], bias=False).double()
    with torch.no_grad():
        proj.weight.copy_(weight.t())
        return proj(_normed(reference, hidden, gamma))


def _weights(size, width, gen):
    weight = (0.02 * torch.randn(size, width, generator=gen)).bfloat16()
    gamma = (1 + 0.25 * torch.randn(1, size, generator=gen)).bfloat16()
    return weight, gamma


def _class_b(out, ref):
    """Worst error over its class-B bound, and the lowest row cosine."""
    out = out.double().reshape(-1, ref.shape[-1])
    ref = ref.reshape(-1, ref.shape[-1])
    rms = ref.square().mean(-1, keepdim=True).sqrt()
    excess = ((out - ref).abs() / (2 ** -7 * (ref.abs() + rms))).amax().item()
    return excess, F.cosine_similarity(out, ref, dim=-1).amin().item()


def _check(reference, cases, path="wrapper"):
    """Run each (label, hidden, weight, gamma) case; report every miss of class B."""
    misses = []
    for label, hidden, weight, gamma in cases:
        out = _run(hidden, weight, gamma, reference[1], path)
        excess, cosine = _class_b(out, _oracle(reference, hidden, weight, gamma))
        print(f"{label}: worst error / bound {excess:.3f}, lowest row cosine {cosine:.6f}")
        if not (excess <= 1.0 and cosine >= 0.9999):
            misses.append(label)
    assert not misses, f"outside class B: {misses}"


@pytest.mark.parametrize("path", PATHS)
def test_correlated_rows_with_one_large_channel(reference, path):
    size, gen = reference[0], torch.Generator().manual_seed(1)
    cases = []
    for width in DECODE_WIDTHS:
        hidden = 0.9 * torch.randn(size, generator=gen) + 0.1 * torch.randn(
            DECODE_TOKENS, 1, size, generator=gen)
        hidden[..., size // 3] *= 100
        cases.append((f"I {width}", hidden.bfloat16(), *_weights(size, width, gen)))
    _check(reference, cases, path)


@pytest.mark.parametrize("path", PATHS)
def test_repeated_tokens(reference, path):
    size, gen = reference[0], torch.Generator().manual_seed(2)
    cases = []
    for width in DECODE_WIDTHS:
        hidden = torch.randn(1, 1, size, generator=gen).expand(DECODE_TOKENS, 1, size)
        cases.append((f"I {width}", hidden.bfloat16().contiguous(), *_weights(size, width, gen)))
    _check(reference, cases, path)


@pytest.mark.parametrize("path", PATHS)
def test_gate_inputs_at_twelve(reference, path):
    """Weights solved so every output, a sigmoid input downstream, is plus or minus 12."""
    size, gen = reference[0], torch.Generator().manual_seed(3)
    cases = []
    for width in DECODE_WIDTHS:
        hidden = torch.randn(DECODE_TOKENS, 1, size, generator=gen).bfloat16()
        _, gamma = _weights(size, width, gen)
        rows = _normed(reference, hidden, gamma).reshape(DECODE_TOKENS, size)
        gram = rows @ rows.T
        noise = 0.02 * torch.randn(size, width, generator=gen, dtype=torch.float64)
        noise -= rows.T @ torch.linalg.solve(gram, rows @ noise)
        signs = torch.randint(0, 2, (DECODE_TOKENS, width), generator=gen).double() * 2 - 1
        weight = noise + rows.T @ torch.linalg.solve(gram, GATE_INPUT * signs)
        cases.append((f"I {width}", hidden, weight.bfloat16(), gamma))
    _check(reference, cases, path)


def test_prefill_chunk(reference):
    size, gen = reference[0], torch.Generator().manual_seed(4)
    cases = []
    for width in PREFILL_WIDTHS:
        hidden = torch.randn(1, PREFILL_TOKENS, size, generator=gen).bfloat16()
        cases.append((f"T {PREFILL_TOKENS} I {width}", hidden, *_weights(size, width, gen)))
    _check(reference, cases)


@pytest.mark.parametrize("path", PATHS)
def test_nan_row_stays_in_its_row(reference, path):
    size, gen = reference[0], torch.Generator().manual_seed(5)
    misses = []
    for width in DECODE_WIDTHS:
        hidden = torch.randn(DECODE_TOKENS, 1, size, generator=gen).bfloat16()
        weight, gamma = _weights(size, width, gen)
        clean = _run(hidden, weight, gamma, reference[1], path)
        poisoned = hidden.clone()
        poisoned[NAN_ROW] = float("nan")
        out = _run(poisoned, weight, gamma, reference[1], path)
        others = [row for row in range(DECODE_TOKENS) if row != NAN_ROW]
        excess, cosine = _class_b(clean, _oracle(reference, hidden, weight, gamma))
        print(f"I {width}: clean worst error / bound {excess:.3f}, lowest row cosine {cosine:.6f}")
        if not torch.isnan(out[NAN_ROW]).all():
            misses.append(f"I {width}: NaN row has a finite output")
        if not torch.equal(out[others].view(torch.int16), clean[others].view(torch.int16)):
            misses.append(f"I {width}: other rows changed")
        if not (excess <= 1.0 and cosine >= 0.9999):
            misses.append(f"I {width}: clean run outside class B")
    assert not misses, misses
