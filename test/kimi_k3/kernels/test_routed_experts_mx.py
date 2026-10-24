# SPDX-License-Identifier: Apache-2.0
"""Routed experts and router selection at the Kimi K3 decode and prefill shapes, against a float64 oracle.

``routed_experts_mx`` runs selection and the distinct-expert MX loop for one rank's
14 local experts of layer 1, read from the checkpoint and repacked to the bank layout
in the kernel module's docstring. ``router_select`` runs the same selection alone.
``routed_expert_tiles_mx`` runs 6 of those experts on their gathered 128-row tiles.
The oracle is the checkpoint's own ``KimiMoEGate`` (its float32 casts made float64)
and ``KimiBlockSparseMLP`` with the stored fp4 weights in float64 (the SiTU stays in
float32, as the reference computes it), with the kernel's MXFP8 activation
quantization applied to the expert input and to the SiTU output: bf16 rounding, then
32 consecutive values per block scale ``2^(floor(log2 amax) - 7)``, then e4m3
nearest-even. Class M grades the partial in two parts. The kernel's test mode exposes
its MXFP8 intermediate codes: per case, at most the set limit of them differ from the
oracle's, a block with a different scale counting as 32. The oracle then adopts the
kernel's codes and scales and runs the exact down product, and every partial element
is within 2^-8 of that oracle row's RMS. The test mode's partial equals the served
partial bit for bit. Class S grades the selection: top-16 sets equal except ties
within 2^-20, weights within 1e-6 relative.
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
#: The largest share of a case's intermediate codes that may differ from the oracle's:
#: three times the share an emulated correct kernel flips at the device MX product noise
#: measured on that case's own operands. None is not yet set.
CODE_LIMITS = {
    "correlated rows, T 8": 3 * 0.010444164276123047,
    "correlated rows, T 1": 3 * 0.02034069242931548,
    "correlated rows, T 32": 3 * 0.016833941141764324,
    "repeated tokens, T 8": 3 * 0.007101853688557942,
    "sigmoid inputs at 12, T 8": 3 * 0.007426307314918155,
    "small near-equal rows, T 8": 3 * 0.007470731382016782,
    "clean rows, T 8": 3 * 0.007434279830367476,
    "expert tiles, T 128": 3 * 0.017904162406921387,
}
#: Prefill tiles: 6 experts with 128 gathered rows each, the last 16 of them padding.
TILE_ITEMS, TILE_ROWS, TILE_PADDING = 6, 128, 16
#: Rows averaged into each small row: its scale is 1 / sqrt of this.
AVERAGED_ROWS = 131072
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


def _slotted(scale):
    """Scale rows [..., 16, B*512] in 3 slots per quadrant: [..., 48, B/3*512]."""
    *lead, rows, width = scale.shape
    blocks = scale.reshape(*lead, 4, 4, 3, width // 3 // 512, 512)
    return blocks.movedim(-3, -4).reshape(*lead, 48, width // 3)


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
        return w, _slotted(s)

    def down(e):
        w = e["w2.packed"].contiguous().view(torch.uint16)
        rows, i4 = w.shape
        kd = i4 // programs // 128
        w = w.reshape(rows // 512, 512, programs, kd, 128).permute(2, 0, 4, 3, 1)
        s = e["w2.scale"].reshape(rows // 512, 512, programs, kd, 16).permute(2, 4, 0, 3, 1)
        return w.reshape(programs, rows // 512, 128, kd * 512), _slotted(s.reshape(programs, 16, -1))

    parts = [gate_up(e) + down(e) for e in stored]
    return [torch.stack([p[i] for p in parts]).contiguous() for i in range(4)]


def _mx8_codes(values):
    """bf16 rounding, then e4m3 codes [..., n/32, 32] and block exponents [..., n/32]."""
    v = values.to(torch.bfloat16).double()
    groups = v.reshape(*v.shape[:-1], -1, 32)
    amax = groups.abs().amax(-1, keepdim=True)
    exponent = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp_min(1e-300))) - 7,
                           torch.zeros_like(amax))
    codes = (groups / torch.exp2(exponent)).float().to(torch.float8_e4m3fn)
    return codes, exponent.squeeze(-1)


def _mx8(values):
    """bf16 rounding, then MXFP8 e4m3 over 32 consecutive values of the last dimension."""
    codes, exponent = _mx8_codes(values)
    return (codes.double() * torch.exp2(exponent)[..., None]).reshape(values.shape)


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


def _visited(index):
    """The local experts that at least one token chose, ascending."""
    first = RANK * LOCAL_EXPERTS
    return sorted({e - first for e in index.flatten().tolist() if 0 <= e - first < LOCAL_EXPERTS})


def _intermediates(ref, routed, visited):
    """Each visited expert's SiTU output [V, T, I] for every row, before its quantization."""
    latent = _mx8(routed[:, :ref.latent].double())
    out = []
    with torch.no_grad():
        for slot in visited:
            mlp = _expert(ref, ref.stored[slot])
            gate_up = torch.cat([mlp.w1(latent), mlp.w3(latent)], dim=-1)
            out.append(mlp.act_fn.forward(gate_up))
    return torch.stack(out)


def _adopted(ref, index, weight, visited, h):
    """The fp64 partial [T, D] with the visited experts' intermediates ``h`` [V, T, I]."""
    partial = torch.zeros(h.shape[1], ref.latent, dtype=torch.float64)
    for v, slot in enumerate(visited):
        stored = ref.stored[slot]
        down = _dequant(stored["w2.packed"], stored["w2.scale"])
        mask = index == RANK * LOCAL_EXPERTS + slot
        partial += (weight * mask).sum(-1)[:, None] * (h[v] @ down.T)
    return partial


def _kernel(ref, routed, bank, replicas, expose_codes=False):
    base = torch.tensor([[RANK * LOCAL_EXPERTS]], dtype=torch.int32)
    situ = ref.scope["_get_situ_activation_params"](ref.config)
    args = dict(beta=situ[0], linear_beta=situ[1], scaling=ref.config.routed_scaling_factor,
                replicas=replicas, expose_codes=expose_codes)

    def run(routed_, bias_, base_, *bank_):
        return rx.routed_experts_mx(routed_, bias_, base_, *bank_, **args)

    if not ON_DEVICE:
        return run(routed, ref.bias, base, *bank)
    compiled = torch.compile(run, backend=envs.get_compile_backend_name(), fullgraph=True,
                             dynamic=False)
    out = compiled(*[t.to("neuron:0") for t in (routed, ref.bias, base, *bank)])
    return tuple(t.cpu() for t in out) if expose_codes else out.cpu()


def _kernel_codes(codes, scales, visited, tokens, width):
    """The test mode's e4m3 codes [V, T, I/32, 32] and block exponents [V, T, I/32]."""
    _, programs, _, cols = scales.shape
    tiles = width // programs // 512
    rows = cols // tiles
    pick = torch.tensor(visited)
    # Byte j of code [e, g, p, k*rows + t] is intermediate index g*I/G + k*512 + 4p + j.
    raw = codes[pick].view(torch.uint8).reshape(len(visited), programs, 128, tiles, rows, 4)
    raw = raw.permute(0, 4, 1, 3, 2, 5).reshape(len(visited), rows, width // 32, 32)
    # Scale [e, g, 32q + r, k*rows + t] covers index g*I/G + k*512 + 32(4q + r) onward.
    block = scales[pick].reshape(len(visited), programs, 4, 32, tiles, rows)[:, :, :, :4]
    block = block.permute(0, 5, 1, 4, 2, 3).reshape(len(visited), rows, width // 32)
    return raw[:, :tokens].view(torch.float8_e4m3fn), block[:, :tokens].double() - 127


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


def _ordinal(codes):
    """Each e4m3 code's signed position in value order; both zeros are 0."""
    u = codes.view(torch.uint8).long()
    return torch.where(u >= 128, 128 - u, u)


def _code_rule(got, got_exp, want, want_exp):
    """Codes that differ, a block at a different scale counting as 32 and a block of zeros
    as none, and the largest code step."""
    scale_same = got_exp == want_exp
    differ = torch.where(scale_same, (got.double() != want.double()).sum(-1), 32)
    zeros = (got.double() == 0).all(-1) & (want.double() == 0).all(-1)
    differ = torch.where(zeros, 0, differ)
    steps = (_ordinal(got) - _ordinal(want)).abs() * scale_same[..., None]
    return int(differ.sum()), int(steps.amax())


def _worst(partial, adopted):
    """The partial's worst error over 2^-8 of the adopted oracle's row RMS."""
    rms = adopted.square().mean(-1, keepdim=True).sqrt()
    err = (partial.double() - adopted).abs()
    return torch.where(err == 0, 0.0, err / (2 ** -8 * rms)).amax().item()


def _class_m(ref, routed, bank, replicas):
    """One case's differing codes, all codes, largest code step, and the partial's worst error
    over the bound of the oracle that adopts the kernel's codes."""
    tokens = routed.shape[0]
    index, weight = _gate(ref, routed[:, ref.latent:])
    visited = _visited(index)
    served = _rows(_kernel(ref, routed, bank, replicas))
    out, codes, scales = _kernel(ref, routed, bank, replicas, expose_codes=True)
    partial = _rows(out)
    assert torch.equal(partial.view(torch.int32), served.view(torch.int32)), \
        "the test mode changed the partial"
    got, got_exp = _kernel_codes(codes, scales, visited, tokens, ref.width)
    want, want_exp = _mx8_codes(_intermediates(ref, routed, visited))
    differ, step = _code_rule(got, got_exp, want, want_exp)
    h = (got.double() * torch.exp2(got_exp)[..., None]).reshape(len(visited), tokens, -1)
    adopted = _adopted(ref, index, weight, visited, h)
    return SimpleNamespace(differ=differ, codes=got.numel(), step=step,
                           worst=_worst(partial, adopted), visited=visited)


def _class_m_line(label, m):
    limit = CODE_LIMITS[label]
    passed = limit is not None and m.differ <= limit * m.codes and m.worst <= 1.0
    shown = "None" if limit is None else f"{limit:.4%}"
    return passed, (f"{label}: {len(m.visited)} local experts; class M {m.differ} of {m.codes} "
                    f"codes differ ({m.differ / m.codes:.4%}, limit {shown}), largest code step "
                    f"{m.step}, partial worst / bound {m.worst:.4f}")


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
    """Run each (label, routed, replicas) case; report every miss of class M and of class S."""
    bank = _bank(ref.stored, ref.programs)
    m_misses, s_misses = [], []
    for label, routed, replicas in cases:
        passed, line = _class_m_line(label, _class_m(ref, routed, bank, replicas))
        index, weight = _select(ref, routed[:, ref.latent:])
        s_set, s_weight = _class_s(ref, routed[:, ref.latent:], index, weight)
        print(f"{line}; class S set excess {s_set:.3f}, weight error / bound {s_weight:.4f}")
        if not passed:
            m_misses.append(label)
        if not (s_set <= 1.0 and s_weight <= 1.0):
            s_misses.append(label)
    assert not s_misses and not m_misses, (
        f"outside class S: {s_misses}; outside class M (a limit of None is not set): {m_misses}")


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
        cases.append((f"correlated rows, T {tokens}", _routed(ref, latent, logits), replicas))
    return cases


def repeated_cases(ref):
    """Regime R2: one token repeated over the 8-row shape."""
    gen = torch.Generator().manual_seed(2)
    tokens, replicas = DECODE_ROWS[0]
    latent = torch.randn(1, ref.latent, generator=gen).expand(tokens, -1)
    logits = LOGIT_STD * torch.randn(1, ref.experts, generator=gen)
    logits = _favour_local(logits, slice(0, 1), gen).expand(tokens, -1)
    return [(f"repeated tokens, T {tokens}", _routed(ref, latent, logits), replicas)]


def twelve_cases(ref):
    """Regime R3: every row's local experts and two others at +12, the rest at -12."""
    gen = torch.Generator().manual_seed(3)
    tokens, replicas = DECODE_ROWS[0]
    latent = torch.randn(tokens, ref.latent, generator=gen)
    logits = torch.full((tokens, ref.experts), -GATE_INPUT)
    logits[:, RANK * LOCAL_EXPERTS:(RANK + 1) * LOCAL_EXPERTS] = GATE_INPUT
    others = torch.randperm(ref.experts - LOCAL_EXPERTS, generator=gen)[:2]
    logits[:, torch.where(others >= RANK * LOCAL_EXPERTS, others + LOCAL_EXPERTS, others)] = GATE_INPUT
    return [(f"sigmoid inputs at 12, T {tokens}", _routed(ref, latent, logits), replicas)]


def small_rows_cases(ref):
    """Regime R4: small, near-equal rows, as an average over many unit rows gives."""
    gen = torch.Generator().manual_seed(4)
    tokens, replicas = DECODE_ROWS[0]
    shared = torch.randn(ref.latent, generator=gen)
    latent = (shared + 2 ** -6 * torch.randn(tokens, ref.latent, generator=gen)) / math.sqrt(
        AVERAGED_ROWS)
    logits = _favour_local(LOGIT_STD * torch.randn(tokens, ref.experts, generator=gen),
                           slice(0, tokens, 2), gen)
    return [(f"small near-equal rows, T {tokens}", _routed(ref, latent, logits), replicas)]


def nan_cases(ref):
    """Regime R5: the clean 8-row case; the test poisons row ``NAN_ROW``."""
    gen = torch.Generator().manual_seed(5)
    tokens, replicas = DECODE_ROWS[0]
    latent = torch.randn(tokens, ref.latent, generator=gen)
    logits = _favour_local(LOGIT_STD * torch.randn(tokens, ref.experts, generator=gen),
                           slice(0, tokens), gen)
    return [(f"clean rows, T {tokens}", _routed(ref, latent, logits), replicas)]


def test_correlated_rows_with_one_large_channel(reference):
    _check(reference, correlated_cases(reference))


def test_repeated_tokens(reference):
    _check(reference, repeated_cases(reference))


def test_sigmoid_inputs_at_twelve(reference):
    _check(reference, twelve_cases(reference))


def test_small_near_equal_rows(reference):
    _check(reference, small_rows_cases(reference))


def test_nan_row_stays_in_its_row(reference):
    (label, routed, replicas), = nan_cases(reference)
    tokens = routed.shape[0]
    poisoned = routed.clone()
    poisoned[NAN_ROW] = float("nan")
    bank = _bank(reference.stored, reference.programs)
    clean = _rows(_kernel(reference, routed, bank, replicas))
    out = _rows(_kernel(reference, poisoned, bank, replicas))
    others = [row for row in range(tokens) if row != NAN_ROW]
    passed, line = _class_m_line(label, _class_m(reference, routed, bank, replicas))
    clean_index, clean_weight = _select(reference, routed[:, reference.latent:])
    index, weight = _select(reference, poisoned[:, reference.latent:])
    print(line)
    assert torch.isnan(out[NAN_ROW]).all(), "the NaN row's partial is not NaN throughout"
    assert torch.equal(out[others].view(torch.int32), clean[others].view(torch.int32)), \
        "the NaN row changed another row's partial"
    assert ((index >= 0) & (index < reference.experts)).all(), "an index left its range"
    assert torch.isnan(weight[NAN_ROW]).all(), "the NaN row's weights are not NaN"
    assert torch.equal(index[others], clean_index[others]) and torch.equal(
        weight[others].view(torch.int32), clean_weight[others].view(torch.int32)), \
        "the NaN row changed another row's selection"
    assert passed, "the clean run is outside class M (a limit of None is not set)"


def _tile_kernel(ref, rows, weights, bank, expose_codes=False):
    situ = ref.scope["_get_situ_activation_params"](ref.config)

    def run(rows_, weights_, *bank_):
        return rx.routed_expert_tiles_mx(rows_, weights_, *bank_, beta=situ[0],
                                         linear_beta=situ[1], expose_codes=expose_codes)

    if not ON_DEVICE:
        return run(rows, weights, *bank)
    compiled = torch.compile(run, backend=envs.get_compile_backend_name(), fullgraph=True,
                             dynamic=False)
    out = compiled(*[t.to("neuron:0") for t in (rows, weights, *bank)])
    return tuple(t.cpu() for t in out) if expose_codes else out.cpu()


def tile_cases(ref):
    """Regime R1 on prefill tiles: each expert's rows near one direction with one channel 100x."""
    gen = torch.Generator().manual_seed(6)
    rows = (0.9 * torch.randn(TILE_ITEMS, 1, ref.latent, generator=gen)
            + 0.1 * torch.randn(TILE_ITEMS, TILE_ROWS, ref.latent, generator=gen))
    rows[..., ref.latent // 3] *= 100
    weights = 0.2 * torch.rand(TILE_ITEMS, TILE_ROWS, generator=gen)
    weights[:, TILE_ROWS - TILE_PADDING:] = 0.0
    return rows.float().contiguous(), weights.float().contiguous()


def test_expert_tiles_on_gathered_rows(reference):
    label = f"expert tiles, T {TILE_ROWS}"
    rows, weights = tile_cases(reference)
    bank = _bank(reference.stored, reference.programs)
    items = list(range(TILE_ITEMS))
    served = _tile_kernel(reference, rows, weights, bank)
    out, codes, scales = _tile_kernel(reference, rows, weights, bank, expose_codes=True)
    assert torch.equal(out.view(torch.int32), served.view(torch.int32)), \
        "the test mode changed the output"
    got, got_exp = _kernel_codes(codes, scales, items, TILE_ROWS, reference.width)
    want, want_exp = _mx8_codes(torch.cat([_intermediates(reference, rows[n], [n]) for n in items]))
    differ, step = _code_rule(got, got_exp, want, want_exp)
    h = (got.double() * torch.exp2(got_exp)[..., None]).reshape(TILE_ITEMS, TILE_ROWS, -1)
    adopted = torch.stack([
        weights[n].double()[:, None] * (h[n] @ _dequant(reference.stored[n]["w2.packed"],
                                                        reference.stored[n]["w2.scale"]).T)
        for n in items])
    m = SimpleNamespace(differ=differ, codes=got.numel(), step=step,
                        worst=_worst(out.flatten(0, 1), adopted.flatten(0, 1)), visited=items)
    passed, line = _class_m_line(label, m)
    poisoned = rows.clone()
    poisoned[NAN_ROW, 5] = float("nan")
    nan_out = _tile_kernel(reference, poisoned, weights, bank)
    print(line)
    others = torch.ones(TILE_ITEMS, TILE_ROWS, dtype=torch.bool)
    others[NAN_ROW, 5] = False
    assert torch.isnan(nan_out[NAN_ROW, 5]).all(), "the NaN row's output is not NaN throughout"
    assert torch.equal(nan_out[others].view(torch.int32), served[others].view(torch.int32)), \
        "the NaN row changed another row's output"
    assert passed, f"{label} is outside class M (a limit of None is not set)"
