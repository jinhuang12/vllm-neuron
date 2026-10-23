# SPDX-License-Identifier: Apache-2.0
"""MoE tail kernel at the Kimi K3 decode shapes, against a float64 oracle.

``moe_fused_tail`` runs its NKI kernel on one rank's shards at TP 32 and writes the
fp32 pack ``[up | shared | sum of squares]``. The oracle is the checkpoint's own
``SituAndMul`` and an ``nn.Linear`` down projection for the shared part, the latent
RMSNorm weight and an ``nn.Linear`` up projection for the up part, and the sum of
squares of the combined rows, in float64 on the CPU (the SiTU stays in float32, as
the reference computes it). The per-token norm scale is not in the pack: the
consumer applies it after the tail all-reduce. The kernel takes the up weights with the
norm weight folded in, as the loader stores them. Each pack section is graded as
its own class B row. The checkpoint directory comes from ``KIMI_K3_CHECKPOINT_DIR``.
With ``VLLM_NEURON_CPU_MODE=1`` and ``NKI_SIMULATOR=1`` the kernel runs on the NKI
simulator; otherwise on ``neuron:0``, where the runtime needs
``NEURON_RT_ROOT_COMM_ID`` set to load an LNC 2 program.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_neuron import envs
from vllm_neuron.functional.moe import fused_tail

CHECKPOINT_DIR = os.environ.get("KIMI_K3_CHECKPOINT_DIR")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT_DIR, reason="KIMI_K3_CHECKPOINT_DIR is not set, so the reference is absent")

ON_DEVICE = not envs.VLLM_NEURON_CPU_MODE

TP_RANKS = 32
DECODE_TOKENS = 4
#: One user, and 32 users over the two attention replicas.
DECODE_TOKEN_RANGE = (1, 16)
GATE_INPUT = 12.0
NAN_ROW = 2


@pytest.fixture(scope="module")
def reference():
    """Per-rank widths, the SiTU parameters and the checkpoint's ``SituAndMul`` class."""
    root = Path(CHECKPOINT_DIR)
    text = json.loads((root / "config.json").read_text())
    text = text.get("text_config", text)
    source = (root / "modeling_kimi_linear.py").read_text()
    wanted = {"SituAndMul", "_get_situ_activation_params"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    scope = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(nodes, []), "modeling_kimi_linear.py", "exec"), scope)
    beta, linear_beta = scope["_get_situ_activation_params"](SimpleNamespace(**text))
    return SimpleNamespace(
        hidden=text["hidden_size"],
        latent=text["routed_expert_hidden_size"] // TP_RANKS,
        shared=text["moe_intermediate_size"] * text["num_shared_experts"] // TP_RANKS,
        beta=beta,
        linear_beta=linear_beta,
        situ=scope["SituAndMul"](beta=beta, linear_beta=linear_beta),
    )


def _tail(combined, gate_up, up_weight, down_weight, beta, linear_beta):
    return fused_tail.moe_fused_tail(combined, gate_up, up_weight, down_weight,
                                     beta=beta, linear_beta=linear_beta)


_compiled = torch.compile(_tail, backend=envs.get_compile_backend_name(),
                          fullgraph=True, dynamic=False) if ON_DEVICE else None


def _weights(ref, gen):
    """The rank's latent norm weight, up weight [H, K] and down weight [H, I], bf16."""
    norm_weight = (1 + 0.25 * torch.randn(ref.latent, generator=gen)).bfloat16()
    up_weight = (0.02 * torch.randn(ref.hidden, ref.latent, generator=gen)).bfloat16()
    down_weight = (0.02 * torch.randn(ref.hidden, ref.shared, generator=gen)).bfloat16()
    return norm_weight, up_weight, down_weight


def _run(ref, combined, gate_up, weights):
    """The kernel's pack, with the norm weight folded into the up weights in fp32."""
    norm_weight, up_weight, down_weight = weights
    folded = (norm_weight.float()[:, None] * up_weight.float().t()).bfloat16().contiguous()
    down = down_weight.t().contiguous()
    if not ON_DEVICE:
        return _tail(combined, gate_up, folded, down, ref.beta, ref.linear_beta)
    on = [t.to("neuron:0") for t in (combined, gate_up, folded, down)]
    return _compiled(*on, ref.beta, ref.linear_beta).cpu()


def _oracle(ref, combined, gate_up, weights):
    norm_weight, up_weight, down_weight = weights
    up = torch.nn.Linear(ref.latent, ref.hidden, bias=False).double()
    down = torch.nn.Linear(ref.shared, ref.hidden, bias=False).double()
    with torch.no_grad():
        up.weight.copy_(up_weight)
        down.weight.copy_(down_weight)
        rows = combined.double()
        return torch.cat([up(norm_weight.double() * rows), down(ref.situ(gate_up.double())),
                          rows.square().sum(-1, keepdim=True)], dim=-1)


def _class_b(out, ref, hidden):
    """Worst error over its class-B bound, and the lowest row cosine, per pack section."""
    excess, cosine = 0.0, 1.0
    for lo, hi in ((0, hidden), (hidden, 2 * hidden), (2 * hidden, 2 * hidden + 1)):
        o, r = out[:, lo:hi].double(), ref[:, lo:hi]
        rms = r.square().mean(-1, keepdim=True).sqrt()
        excess = max(excess, ((o - r).abs() / (2 ** -7 * (r.abs() + rms))).amax().item())
        cosine = min(cosine, F.cosine_similarity(o, r, dim=-1).amin().item())
    return excess, cosine


def _check(ref, cases):
    """Run each (label, combined, gate_up, weights) case; report every miss of class B."""
    misses = []
    for label, combined, gate_up, weights in cases:
        out = _run(ref, combined, gate_up, weights)
        excess, cosine = _class_b(out, _oracle(ref, combined, gate_up, weights), ref.hidden)
        print(f"{label}: worst error / bound {excess:.3f}, lowest row cosine {cosine:.6f}")
        if not (excess <= 1.0 and cosine >= 0.9999):
            misses.append(label)
    assert not misses, f"outside class B: {misses}"


def _correlated(tokens, width, gen):
    rows = 0.9 * torch.randn(width, generator=gen) + 0.1 * torch.randn(tokens, width, generator=gen)
    rows[:, width // 3] *= 100
    return rows


def test_correlated_rows_with_one_large_channel(reference):
    gen = torch.Generator().manual_seed(1)
    combined = _correlated(DECODE_TOKENS, reference.latent, gen)
    gate_up = _correlated(DECODE_TOKENS, 2 * reference.shared, gen).bfloat16()
    _check(reference, [(f"T {DECODE_TOKENS}", combined, gate_up, _weights(reference, gen))])


def test_repeated_tokens(reference):
    gen = torch.Generator().manual_seed(2)
    combined = torch.randn(1, reference.latent, generator=gen).expand(DECODE_TOKENS, -1)
    gate_up = torch.randn(1, 2 * reference.shared, generator=gen).expand(DECODE_TOKENS, -1)
    _check(reference, [(f"T {DECODE_TOKENS}", combined.contiguous(),
                        gate_up.bfloat16().contiguous(), _weights(reference, gen))])


def test_gate_inputs_at_twelve(reference):
    gen = torch.Generator().manual_seed(3)
    combined = torch.randn(DECODE_TOKENS, reference.latent, generator=gen)
    gate_up = torch.randn(DECODE_TOKENS, 2 * reference.shared, generator=gen)
    gate_up[:, :reference.shared] = GATE_INPUT * gate_up[:, :reference.shared].sign()
    _check(reference, [(f"T {DECODE_TOKENS}", combined, gate_up.bfloat16(),
                        _weights(reference, gen))])


def test_decode_batch_extremes(reference):
    gen = torch.Generator().manual_seed(4)
    cases = []
    for tokens in DECODE_TOKEN_RANGE:
        combined = torch.randn(tokens, reference.latent, generator=gen)
        gate_up = torch.randn(tokens, 2 * reference.shared, generator=gen).bfloat16()
        cases.append((f"T {tokens}", combined, gate_up, _weights(reference, gen)))
    _check(reference, cases)


def test_nan_row_stays_in_its_row(reference):
    gen = torch.Generator().manual_seed(5)
    combined = torch.randn(DECODE_TOKENS, reference.latent, generator=gen)
    gate_up = torch.randn(DECODE_TOKENS, 2 * reference.shared, generator=gen).bfloat16()
    weights = _weights(reference, gen)
    clean = _run(reference, combined, gate_up, weights)
    poisoned_rows, poisoned_gate_up = combined.clone(), gate_up.clone()
    poisoned_rows[NAN_ROW] = float("nan")
    poisoned_gate_up[NAN_ROW] = float("nan")
    out = _run(reference, poisoned_rows, poisoned_gate_up, weights)
    others = [row for row in range(DECODE_TOKENS) if row != NAN_ROW]
    excess, cosine = _class_b(clean, _oracle(reference, combined, gate_up, weights),
                              reference.hidden)
    print(f"T {DECODE_TOKENS}: clean worst error / bound {excess:.3f}, "
          f"lowest row cosine {cosine:.6f}")
    assert torch.isnan(out[NAN_ROW]).all(), "the NaN row's pack is not NaN throughout"
    assert torch.equal(out[others].view(torch.int32), clean[others].view(torch.int32)), \
        "the NaN row changed another row's pack"
    assert excess <= 1.0 and cosine >= 0.9999, "the clean run is outside class B"
