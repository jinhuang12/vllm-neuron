# SPDX-License-Identifier: Apache-2.0
"""Routed experts and router selection at the Kimi K3 decode shapes, against a float64 oracle.

``routed_experts_mx`` runs selection and the distinct-expert MX loop for one rank's
14 local experts of layer 1, read from the checkpoint and repacked to the bank layout
in the kernel module's docstring. ``router_select`` runs the same selection alone.
The oracle is the checkpoint's own ``KimiMoEGate`` (its float32 casts made float64)
and ``KimiBlockSparseMLP`` with the stored fp4 weights in float64 (the SiTU stays in
float32, as the reference computes it), with the kernel's MXFP8 activation
quantization applied to the expert input and to the SiTU output: bf16 rounding, then
32 consecutive values per block scale ``2^(floor(log2 amax) - 7)``. The partial is
graded as class M (per element error at most 2^-8 of the row RMS), the selection as
class S (top-16 sets equal except ties within 2^-20; weights within 1e-6 relative).
The checkpoint directory comes from ``KIMI_K3_CHECKPOINT_DIR``. With
``VLLM_NEURON_CPU_MODE=1`` and ``NKI_SIMULATOR=1`` the kernels run on the NKI
simulator; otherwise on ``neuron:0``.
"""

from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm_neuron import envs
from vllm_neuron.functional.moe import routed_experts_mx as rx
from vllm_neuron.utils.hardware_config import get_logical_nc_config

CHECKPOINT_DIR = os.environ.get("KIMI_K3_CHECKPOINT_DIR")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT_DIR, reason="KIMI_K3_CHECKPOINT_DIR is not set, so the reference is absent")

ON_DEVICE = not envs.VLLM_NEURON_CPU_MODE

LAYER = 1
LOCAL_EXPERTS = 14
#: The rank whose local experts the bank holds: global experts 28 to 41.
RANK = 2
#: Gathered decode rows: 8 users over two replicas, one user, 32 users.
DECODE_ROWS = ((8, 2), (1, 1), (32, 2))
LOGIT_STD = 2.0
GATE_INPUT = 12.0
NAN_ROW = 2
KEY = "language_model.model.layers.{layer}.block_sparse_moe.{name}"
FP4_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                          dtype=torch.float64)


def _tensor(root, weight_map, name):
    key = KEY.format(layer=LAYER, name=name)
    with safe_open(root / weight_map[key], "pt") as handle:
        return handle.get_tensor(key)


@pytest.fixture(scope="module")
def reference():
    """Config values, the reference classes, the score bias and the local experts' stored tensors."""
    root = Path(CHECKPOINT_DIR)
    text = json.loads((root / "config.json").read_text())
    text = text.get("text_config", text)
    assert text["moe_router_activation_func"] == "sigmoid" and text["moe_renormalize"]
    assert text.get("num_expert_group", 1) == 1
    source = (root / "modeling_kimi_linear.py").read_text()
    wanted = {"SituAndMul", "_get_situ_activation_params", "KimiBlockSparseMLP", "KimiMoEGate"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    for node in ast.walk(ast.Module(nodes, [])):
        if isinstance(node, ast.Attribute) and node.attr == "float32":
            node.attr = "float64"  # the gate's fp32 casts, so the oracle runs in fp64
    situ_class = next(node for node in nodes if node.name == "SituAndMul")
    for node in ast.walk(situ_class):
        if isinstance(node, ast.Attribute) and node.attr == "float64":
            node.attr = "float32"  # the SiTU keeps the reference's fp32
    scope = {"torch": torch, "nn": torch.nn, "F": F, "math": math, "ACT2FN": {}}
    exec(compile(ast.Module(nodes, []), "modeling_kimi_linear.py", "exec"), scope)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    experts = []
    for e in range(RANK * LOCAL_EXPERTS, (RANK + 1) * LOCAL_EXPERTS):
        experts.append({f"{w}.{part}": _tensor(root, weight_map, f"experts.{e}.{w}.weight_{part}")
                        for w in ("w1", "w2", "w3") for part in ("packed", "scale")})
    config = SimpleNamespace(**text)
    return SimpleNamespace(
        config=config,
        latent=text["routed_expert_hidden_size"],
        width=text["moe_intermediate_size"],
        experts=text["num_experts"],
        bias=_tensor(root, weight_map, "gate.e_score_correction_bias").float(),
        stored=experts,
        scope=scope,
        programs=get_logical_nc_config(),
    )


def _dequant(packed, scale):
    """Stored MXFP4 bytes [out, in/2] and E8M0 scales [out, in/32] as float64 [out, in]."""
    nibbles = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(packed.shape[0], -1).long()
    values = FP4_VALUES[nibbles].reshape(packed.shape[0], -1, 32)
    return (values * torch.exp2(scale.double() - 127)[..., None]).reshape(packed.shape[0], -1)


def _bank(stored, programs):
    """The bank layout of the kernel module docstring, built from the stored bytes."""
    def gate_up(e):
        w = torch.stack([e["w1.packed"], e["w3.packed"]]).contiguous().view(torch.uint16)
        two, width, k4 = w.shape
        w = w.reshape(2, programs, width // programs // 512, 512, k4 // 128, 128)
        w = w.permute(1, 0, 2, 5, 4, 3).reshape(programs, -1, 128, k4 // 128 * 512)
        s = torch.stack([e["w1.scale"], e["w3.scale"]])
        s = s.reshape(2, programs, width // programs // 512, 512, s.shape[-1] // 16, 16)
        s = s.permute(1, 5, 0, 2, 4, 3).reshape(programs, 16, -1)
        return w, s

    def down(e):
        w = e["w2.packed"].contiguous().view(torch.uint16)
        rows, i4 = w.shape
        kd = i4 // programs // 128
        w = w.reshape(rows // 512, 512, programs, kd, 128).permute(2, 0, 4, 3, 1)
        s = e["w2.scale"].reshape(rows // 512, 512, programs, kd, 16).permute(2, 4, 0, 3, 1)
        return w.reshape(programs, rows // 512, 128, kd * 512), s.reshape(programs, 16, -1)

    parts = [gate_up(e) + down(e) for e in stored]
    return [torch.stack([p[i] for p in parts]).contiguous() for i in range(4)]


def _mx8(values):
    """bf16 rounding, then MXFP8 e4m3 over 32 consecutive values of the last dimension."""
    v = values.to(torch.bfloat16).double()
    groups = v.reshape(*v.shape[:-1], -1, 32)
    amax = groups.abs().amax(-1, keepdim=True)
    exponent = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp_min(1e-300))) - 7,
                           torch.zeros_like(amax))
    step = torch.exp2(exponent)
    codes = (groups / step).float().to(torch.float8_e4m3fn).double()
    return (codes * step).reshape(v.shape)


def _gate(ref, logits):
    """The reference gate in float64 on given logits: an identity weight over the experts."""
    config = SimpleNamespace(**{**vars(ref.config), "hidden_size": ref.experts})
    gate = ref.scope["KimiMoEGate"](config).double().eval()
    with torch.no_grad():
        gate.weight.copy_(torch.eye(ref.experts, dtype=torch.float64))
        gate.e_score_correction_bias.copy_(ref.bias.double())
        index, weight = gate(logits.double()[None])
    return index, weight


def _expert(ref, stored):
    """``KimiBlockSparseMLP`` in fp64 with the stored weights and the kernel's two MX quantizations."""
    mlp = ref.scope["KimiBlockSparseMLP"](ref.config, hidden_size=ref.latent,
                                         intermediate_size=ref.width).double().eval()
    with torch.no_grad():
        for w in ("w1", "w2", "w3"):
            getattr(mlp, w).weight.copy_(_dequant(stored[f"{w}.packed"], stored[f"{w}.scale"]))
    mlp.register_forward_pre_hook(lambda _, args: (_mx8(args[0]),))
    mlp.act_fn.register_forward_hook(lambda _, __, out: _mx8(out))
    return mlp


def _oracle(ref, routed):
    """The fp64 partial [T, D] of this rank's local experts, and the gate's indices."""
    latent, logits = routed[:, :ref.latent].double(), routed[:, ref.latent:]
    index, weight = _gate(ref, logits)
    partial = torch.zeros(routed.shape[0], ref.latent, dtype=torch.float64)
    first = RANK * LOCAL_EXPERTS
    with torch.no_grad():
        for slot in range(LOCAL_EXPERTS):
            mask = index == first + slot
            rows = mask.any(-1)
            if rows.any():
                y = _expert(ref, ref.stored[slot])(latent[rows])
                partial[rows] += (weight * mask).sum(-1)[rows, None] * y
    return partial, index


def _kernel(ref, routed, bank, replicas):
    base = torch.tensor([[RANK * LOCAL_EXPERTS]], dtype=torch.int32)
    situ = ref.scope["_get_situ_activation_params"](ref.config)
    args = dict(beta=situ[0], linear_beta=situ[1], scaling=ref.config.routed_scaling_factor,
                replicas=replicas)

    def run(routed_, bias_, base_, *bank_):
        return rx.routed_experts_mx(routed_, bias_, base_, *bank_, **args)

    if not ON_DEVICE:
        return run(routed, ref.bias, base, *bank)
    compiled = torch.compile(run, backend=envs.get_compile_backend_name(), fullgraph=True,
                             dynamic=False)
    return compiled(*[t.to("neuron:0") for t in (routed, ref.bias, base, *bank)]).cpu()


def _select(ref, logits):
    def run(logits_, bias_):
        return rx.router_select(logits_, bias_, scaling=ref.config.routed_scaling_factor)

    if not ON_DEVICE:
        return run(logits, ref.bias)
    compiled = torch.compile(run, backend=envs.get_compile_backend_name(), fullgraph=True,
                             dynamic=False)
    index, weight = compiled(logits.to("neuron:0"), ref.bias.to("neuron:0"))
    return index.cpu(), weight.cpu()


def _rows(out):
    """Combine order [R, C, T/R, D/C] back to rows [T, D]."""
    replicas, chunks, per_replica, width = out.shape
    return out.permute(0, 2, 1, 3).reshape(replicas * per_replica, chunks * width)


def _class_m(partial, oracle):
    """Worst per element error over 2^-8 of the oracle row RMS (0 where both rows are 0)."""
    rms = oracle.square().mean(-1, keepdim=True).sqrt()
    err = (partial.double() - oracle).abs()
    return torch.where(err == 0, 0.0, err / (2 ** -8 * rms)).amax().item()


def _class_s(ref, logits, index, weight):
    """Worst set excess over the 2^-20 tie band and worst weight error over 1e-6 relative."""
    scores = torch.sigmoid(logits.double())
    choice = scores + ref.bias.double()
    tenth = choice.topk(rx.TOP_K, dim=-1).values[:, -1:]
    idx = index.long()
    assert ((idx >= 0) & (idx < ref.experts)).all(), "an index left its range"
    assert all(len(set(row.tolist())) == rx.TOP_K for row in idx), "an index repeats"
    set_excess = ((tenth - choice.gather(1, idx)).clamp_min(0) / 2 ** -20).amax().item()
    chosen = scores.gather(1, idx)
    want = chosen / (chosen.sum(-1, keepdim=True) + rx.RENORM_GUARD) * ref.config.routed_scaling_factor
    weight_excess = ((weight.double() - want).abs() / (1e-6 * want.abs())).amax().item()
    return set_excess, weight_excess


def _check(ref, cases):
    """Run each (label, routed, replicas) case; report every miss of class M or class S."""
    bank = _bank(ref.stored, ref.programs)
    misses = []
    for label, routed, replicas in cases:
        oracle, gate_index = _oracle(ref, routed)
        visited = sorted({e - RANK * LOCAL_EXPERTS for e in gate_index.flatten().tolist()
                          if 0 <= e - RANK * LOCAL_EXPERTS < LOCAL_EXPERTS})
        m = _class_m(_rows(_kernel(ref, routed, bank, replicas)), oracle)
        index, weight = _select(ref, routed[:, ref.latent:])
        s_set, s_weight = _class_s(ref, routed[:, ref.latent:], index, weight)
        print(f"{label}: {len(visited)} local experts; class M worst / bound {m:.4f}; "
              f"class S set excess {s_set:.3f}, weight error / bound {s_weight:.4f}")
        if not (m <= 1.0 and s_set <= 1.0 and s_weight <= 1.0):
            misses.append(label)
    assert not misses, f"outside class M or S: {misses}"


def _routed(ref, latent, logits):
    return torch.cat([latent, logits], dim=-1).float().contiguous()


def _favour_local(logits, rows, gen):
    """Lift this rank's experts in ``rows`` by 6 to 10 so the loop visits most of them."""
    local = slice(RANK * LOCAL_EXPERTS, (RANK + 1) * LOCAL_EXPERTS)
    count = len(range(*rows.indices(logits.shape[0])))
    logits[rows, local] += 6.0 + 4.0 * torch.rand(count, LOCAL_EXPERTS, generator=gen)
    return logits


def correlated_cases(ref):
    """Regime R1: rows near one direction with one channel 100x, at each decode shape."""
    gen = torch.Generator().manual_seed(1)
    cases = []
    for tokens, replicas in DECODE_ROWS:
        latent = (0.9 * torch.randn(ref.latent, generator=gen)
                  + 0.1 * torch.randn(tokens, ref.latent, generator=gen))
        latent[:, ref.latent // 3] *= 100
        logits = LOGIT_STD * torch.randn(tokens, ref.experts, generator=gen)
        logits = _favour_local(logits, slice(0, tokens, 2), gen)
        cases.append((f"T {tokens}", _routed(ref, latent, logits), replicas))
    return cases


def repeated_cases(ref):
    """Regime R2: one token repeated over the 8-row shape."""
    gen = torch.Generator().manual_seed(2)
    tokens, replicas = DECODE_ROWS[0]
    latent = torch.randn(1, ref.latent, generator=gen).expand(tokens, -1)
    logits = LOGIT_STD * torch.randn(1, ref.experts, generator=gen)
    logits = _favour_local(logits, slice(0, 1), gen).expand(tokens, -1)
    return [(f"T {tokens}", _routed(ref, latent, logits), replicas)]


def twelve_cases(ref):
    """Regime R3: every row's local experts and two others at +12, the rest at -12."""
    gen = torch.Generator().manual_seed(3)
    tokens, replicas = DECODE_ROWS[0]
    latent = torch.randn(tokens, ref.latent, generator=gen)
    logits = torch.full((tokens, ref.experts), -GATE_INPUT)
    logits[:, RANK * LOCAL_EXPERTS:(RANK + 1) * LOCAL_EXPERTS] = GATE_INPUT
    others = torch.randperm(ref.experts - LOCAL_EXPERTS, generator=gen)[:2]
    logits[:, torch.where(others >= RANK * LOCAL_EXPERTS, others + LOCAL_EXPERTS, others)] = GATE_INPUT
    return [(f"T {tokens}", _routed(ref, latent, logits), replicas)]


def nan_cases(ref):
    """Regime R5: the clean 8-row case; the test poisons row ``NAN_ROW``."""
    gen = torch.Generator().manual_seed(5)
    tokens, replicas = DECODE_ROWS[0]
    latent = torch.randn(tokens, ref.latent, generator=gen)
    logits = _favour_local(LOGIT_STD * torch.randn(tokens, ref.experts, generator=gen),
                           slice(0, tokens), gen)
    return [(f"T {tokens}", _routed(ref, latent, logits), replicas)]


def test_correlated_rows_with_one_large_channel(reference):
    _check(reference, correlated_cases(reference))


def test_repeated_tokens(reference):
    _check(reference, repeated_cases(reference))


def test_sigmoid_inputs_at_twelve(reference):
    _check(reference, twelve_cases(reference))


def test_nan_row_stays_in_its_row(reference):
    (_, routed, replicas), = nan_cases(reference)
    tokens = routed.shape[0]
    poisoned = routed.clone()
    poisoned[NAN_ROW] = float("nan")
    bank = _bank(reference.stored, reference.programs)
    clean = _rows(_kernel(reference, routed, bank, replicas))
    out = _rows(_kernel(reference, poisoned, bank, replicas))
    others = [row for row in range(tokens) if row != NAN_ROW]
    oracle, _ = _oracle(reference, routed)
    m = _class_m(clean, oracle)
    clean_index, clean_weight = _select(reference, routed[:, reference.latent:])
    index, weight = _select(reference, poisoned[:, reference.latent:])
    print(f"T {tokens}: clean class M worst / bound {m:.4f}")
    assert torch.isnan(out[NAN_ROW]).all(), "the NaN row's partial is not NaN throughout"
    assert torch.equal(out[others].view(torch.int32), clean[others].view(torch.int32)), \
        "the NaN row changed another row's partial"
    assert ((index >= 0) & (index < reference.experts)).all(), "an index left its range"
    assert torch.isnan(weight[NAN_ROW]).all(), "the NaN row's weights are not NaN"
    assert torch.equal(index[others], clean_index[others]) and torch.equal(
        weight[others].view(torch.int32), clean_weight[others].view(torch.int32)), \
        "the NaN row changed another row's selection"
    assert m <= 1.0, "the clean run is outside class M"
