# SPDX-License-Identifier: Apache-2.0
"""The routed-expert kernel's replica mode at the Kimi K3 decode shapes, against a float64 oracle.

``routed_experts_mx`` with ``r`` replica slots runs one rank's bank of layer 1: rank 2's
14 own experts, then the experts its replica slots hold, read from the checkpoint and
repacked to the bank layout in the kernel module's docstring. Slot ``j`` of rank ``k``
has a second home on rank ``k + 1 + j*(64 // (r + 1))`` mod 64.
The oracle chooses each token's experts with the checkpoint's own ``KimiMoEGate`` (its
float32 casts made float64), then makes the replica choice one expert at a time: the
chosen experts with no replica run on their homes; the others are taken slot by slot,
then by bit ``v`` of the home rank (``2^v`` the largest power of two dividing the
slot's rank offset), then by rank, and each runs on whichever of its two homes has run
fewer chosen experts so far, its own home on a tie. It runs the experts rank 2 takes
with ``KimiBlockSparseMLP`` and the stored fp4 weights in float64 (the SiTU in float32,
as the reference computes it), with the kernel's MXFP8 activation quantization applied
to the expert input and to the SiTU output: bf16 rounding, then 32 consecutive values
per block scale ``2^(floor(log2 amax) - 7)``, then e4m3 nearest-even.
Class M grades the partial in two parts. The kernel's test mode exposes its MXFP8
intermediate codes: per case, at most the set limit of them differ from the oracle's, a
block with a different scale counting as 32. The oracle then adopts the kernel's codes
and scales and runs the exact down product, and every partial element is within 2^-8 of
that oracle row's RMS. The test mode's partial equals the served partial bit for bit.
Class S grades the selection, top-16 sets equal except ties within 2^-20 and weights
within 1e-6 relative, and the choice: each bank entry's visit count, the tokens that
chose its expert where rank 2 runs it and 0 elsewhere, and the test mode's record of
which home runs each chosen expert with a replica, on every rank, equal the oracle's
exactly.
The checkpoint directory comes from ``KIMI_K3_CHECKPOINT_DIR``. With
``VLLM_NEURON_CPU_MODE=1`` and ``NKI_SIMULATOR=1`` the kernel runs on the NKI
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
OWN = 14
RANKS = 64
#: The rank whose bank the kernel runs: own experts 28 to 41.
RANK = 2
#: Decode shapes (tokens, token groups, replica slots): 1 user, 8 and 32 users at 3 slots,
#: 8 and 32 users at 2 slots.
SHAPES = ((1, 1, 3), (8, 2, 3), (32, 2, 3), (8, 2, 2), (32, 2, 2))
LOGIT_STD = 2.0
GATE_INPUT = 12.0
LIFT = 8.0
NAN_ROW = 6
KEY = "language_model.model.layers.{layer}.block_sparse_moe.{name}"
#: The largest share of a case's intermediate codes that may differ from the oracle's:
#: three times the share an emulated correct kernel flips at the device MX product noise
#: measured on that case's own operands. None is not yet set.
CODE_LIMITS = {
    "correlated rows, T 1, 3 slots": None,
    "correlated rows, T 8, 3 slots": None,
    "correlated rows, T 32, 3 slots": None,
    "correlated rows, T 8, 2 slots": None,
    "correlated rows, T 32, 2 slots": None,
    "repeated tokens, T 8, 3 slots": None,
    "sigmoid inputs at 12, T 8, 3 slots": None,
    "small near-equal rows, T 8, 3 slots": None,
    "clean rows, T 8, 3 slots": None,
}
#: Rows averaged into each small row: its scale is 1 / sqrt of this.
AVERAGED_ROWS = 131072
FP4_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                          dtype=torch.float64)


def _shifts(slots):
    return [1 + j * (RANKS // (slots + 1)) for j in range(slots)]


def _entries(slots):
    """Global ids of rank 2's bank entries: its own experts, then its replicas."""
    return [RANK * OWN + s for s in range(OWN)] + [
        OWN * ((RANK - shift) % RANKS) + j for j, shift in enumerate(_shifts(slots))]


def _runner(hit, slots):
    """The rank each chosen expert runs on under the replica choice, one expert at a time."""
    load = [sum(k * OWN + s in hit for s in range(slots, OWN)) for k in range(RANKS)]
    runner = {e: e // OWN for e in hit}
    for j, shift in enumerate(_shifts(slots)):
        low = shift & -shift
        for side in (0, low):
            for k in range(RANKS):
                e = k * OWN + j
                if k & low == side and e in hit:
                    other = (k + shift) % RANKS
                    runner[e] = other if load[other] < load[k] else k
                    load[runner[e]] += 1
    return runner


def _tensor(root, weight_map, name):
    key = KEY.format(layer=LAYER, name=name)
    with safe_open(root / weight_map[key], "pt") as handle:
        return handle.get_tensor(key)


@pytest.fixture(scope="module")
def reference():
    """Config values, the reference classes, the score bias and every bank entry's stored tensors."""
    root = Path(CHECKPOINT_DIR)
    text = json.loads((root / "config.json").read_text())
    text = text.get("text_config", text)
    assert text["moe_router_activation_func"] == "sigmoid" and text["moe_renormalize"]
    assert text.get("num_expert_group", 1) == 1 and text["num_experts"] == OWN * RANKS
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
    stored = {}
    for e in sorted({e for _, _, slots in SHAPES for e in _entries(slots)}):
        stored[e] = {f"{w}.{part}": _tensor(root, weight_map, f"experts.{e}.{w}.weight_{part}")
                     for w in ("w1", "w2", "w3") for part in ("packed", "scale")}
    return SimpleNamespace(
        config=SimpleNamespace(**text),
        latent=text["routed_expert_hidden_size"],
        width=text["moe_intermediate_size"],
        experts=text["num_experts"],
        bias=_tensor(root, weight_map, "gate.e_score_correction_bias").float(),
        stored=stored,
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


def _routing(ref, routed, slots):
    """The oracle's choice: kept rows, indices and weights, and rank 2's visited entries and counts."""
    kept = torch.isfinite(routed).all(-1)
    index, weight = _gate(ref, torch.where(kept[:, None], routed[:, ref.latent:], 0.0))
    index = torch.where(kept[:, None], index, -1)
    runner = _runner(set(index[kept].flatten().tolist()), slots)
    entries = _entries(slots)
    counts = torch.tensor([float((index == e).sum()) if runner.get(e) == RANK else 0.0
                           for e in entries])
    visited = [n for n, c in enumerate(counts.tolist()) if c > 0]
    record = torch.zeros(2, slots, RANKS)
    for e, k in runner.items():
        if e % OWN < slots:
            record[int(k != e // OWN), e % OWN, e // OWN] = 1.0
    return SimpleNamespace(kept=kept, index=index, weight=weight, counts=counts, visited=visited,
                           ids=[entries[n] for n in visited], record=record)


def _intermediates(ref, routed, kept, ids):
    """Each visited expert's SiTU output [V, T, I] for every row, before its quantization."""
    latent = _mx8(torch.where(kept[:, None], routed[:, :ref.latent].double(), 0.0))
    out = []
    with torch.no_grad():
        for e in ids:
            mlp = _expert(ref, ref.stored[e])
            gate_up = torch.cat([mlp.w1(latent), mlp.w3(latent)], dim=-1)
            out.append(mlp.act_fn.forward(gate_up))
    return torch.stack(out)


def _adopted(ref, route, h):
    """The fp64 partial [T, D] with the visited experts' intermediates ``h`` [V, T, I]."""
    partial = torch.zeros(h.shape[1], ref.latent, dtype=torch.float64)
    for v, e in enumerate(route.ids):
        stored = ref.stored[e]
        down = _dequant(stored["w2.packed"], stored["w2.scale"])
        affinity = torch.where(route.index == e, route.weight, 0.0).sum(-1)
        partial += affinity[:, None] * (h[v] @ down.T)
    return partial


def _kernel(ref, routed, bank, replicas, slots, expose_codes=False):
    base = torch.tensor([[RANK * OWN]], dtype=torch.int32)
    situ = ref.scope["_get_situ_activation_params"](ref.config)
    args = dict(beta=situ[0], linear_beta=situ[1], scaling=ref.config.routed_scaling_factor,
                replicas=replicas, expose_codes=expose_codes, replica_slots=slots)

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
    pick = torch.tensor(visited, dtype=torch.long)
    # Byte j of code [e, g, p, k*rows + t] is intermediate index g*I/G + k*512 + 4p + j.
    raw = codes[pick].view(torch.uint8).reshape(len(visited), programs, 128, tiles, rows, 4)
    raw = raw.permute(0, 4, 1, 3, 2, 5).reshape(len(visited), rows, width // 32, 32)
    # Scale [e, g, 32q + r, k*rows + t] covers index g*I/G + k*512 + 32(4q + r) onward.
    block = scales[pick].reshape(len(visited), programs, 4, 32, tiles, rows)[:, :, :, :4]
    block = block.permute(0, 5, 1, 4, 2, 3).reshape(len(visited), rows, width // 32)
    return raw[:, :tokens].view(torch.float8_e4m3fn), block[:, :tokens].double() - 127


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


def _measure(ref, routed, bank, replicas, slots):
    """One case's class M reading and the kernel's visit counts against the oracle's choice."""
    tokens = routed.shape[0]
    route = _routing(ref, routed, slots)
    served = _rows(_kernel(ref, routed, bank, replicas, slots))
    out, codes, scales, visits, record = _kernel(ref, routed, bank, replicas, slots, expose_codes=True)
    partial = _rows(out)
    same = torch.equal(partial.view(torch.int32), served.view(torch.int32))
    kept = route.kept.nonzero().flatten()
    got, got_exp = _kernel_codes(codes, scales, route.visited, tokens, ref.width)
    want, want_exp = _mx8_codes(_intermediates(ref, routed, route.kept, route.ids))
    differ, step = _code_rule(got[:, kept], got_exp[:, kept], want[:, kept], want_exp[:, kept])
    h = (got.double() * torch.exp2(got_exp)[..., None]).reshape(len(route.visited), tokens, -1)
    adopted = _adopted(ref, route, h)
    return SimpleNamespace(route=route, served=served, same=same, differ=differ,
                           codes=got[:, kept].numel(), step=step,
                           worst=_worst(partial[kept], adopted[kept]),
                           choice=_same_choice(visits, record, route), visits=visits.flatten().tolist())


def _same_choice(visits, record, route):
    """The kernel's visit counts and its record of the whole choice equal the oracle's."""
    return torch.equal(visits.flatten().double(), route.counts.double()) and torch.equal(
        record.reshape(route.record.shape).double(), route.record.double())


def _choice_line(route, slots):
    stays = sum(1 for n in route.visited if n < slots)
    taken = sum(1 for n in route.visited if n >= OWN)
    moved = sum(1 for n in range(slots) if route.counts[n] == 0 and
                (route.index == RANK * OWN + n).any())
    return f"choice: {stays} own replicated stay, {moved} move, {taken} replicas taken"


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


def _select(ref, logits):
    def run(logits_, bias_):
        return rx.router_select(logits_, bias_, scaling=ref.config.routed_scaling_factor)

    if not ON_DEVICE:
        return run(logits, ref.bias)
    compiled = torch.compile(run, backend=envs.get_compile_backend_name(), fullgraph=True,
                             dynamic=False)
    index, weight = compiled(logits.to("neuron:0"), ref.bias.to("neuron:0"))
    return index.cpu(), weight.cpu()


def _grade(ref, label, routed, replicas, slots):
    """Run one case; return its failed checks and its measurement."""
    bank = _bank([ref.stored[e] for e in _entries(slots)], ref.programs)
    m = _measure(ref, routed, bank, replicas, slots)
    index, weight = _select(ref, routed[:, ref.latent:])
    s_set, s_weight = _class_s(ref, routed[:, ref.latent:], index, weight)
    limit = CODE_LIMITS[label]
    shown = "None" if limit is None else f"{limit:.4%}"
    print(f"{label}: {len(m.route.visited)} entries visited; class M {m.differ} of {m.codes} codes "
          f"differ ({m.differ / m.codes:.4%}, limit {shown}), largest code step {m.step}, partial "
          f"worst / bound {m.worst:.4f}; class S set excess {s_set:.3f}, weight error / bound "
          f"{s_weight:.4f}; {_choice_line(m.route, slots)}; visits and whole choice as the oracle's: {m.choice}")
    failed = [name for name, ok in (
        ("test mode changed the partial", m.same),
        ("class M codes (a limit of None is not set)",
         limit is not None and m.differ <= limit * m.codes),
        ("class M bound", m.worst <= 1.0),
        ("class S selection", s_set <= 1.0 and s_weight <= 1.0),
        ("class S choice", m.choice)) if not ok]
    return failed, m


def _routed(latent, logits):
    return torch.cat([latent, logits], dim=-1).float().contiguous()


def _lift(logits, slots, gen, take=True):
    """Make rows choose expert groups, so the choice goes both ways: rank 2's first 9 experts
    in even rows; its first replica's home rank's experts in rows 1 mod 8 (with
    ``take``, its first replica among them); every expert of its second slot's other home in
    rows 3 mod 8; its second replica in rows 5 mod 8. A row with a group lowers every other
    logit by ``LIFT``, so the score bias cannot outrank the group."""
    shifts = _shifts(slots)
    groups = {0: [RANK * OWN + s for s in range(9)],
              1: [((RANK - shifts[0]) % RANKS) * OWN + s for s in range(0 if take else slots, OWN)],
              3: [((RANK + shifts[1]) % RANKS) * OWN + s for s in range(OWN)],
              5: [((RANK - shifts[1]) % RANKS) * OWN + 1]}
    for row in range(logits.shape[0]):
        group = groups[0] if row % 2 == 0 else groups.get(row % 8)
        if group:
            _choose(logits, row, group, gen)
    return logits


def _choose(logits, row, group, gen):
    """Make row ``row`` choose the experts ``group`` first: every other logit loses ``LIFT``."""
    logits[row] -= LIFT
    logits[row, group] = LIFT + 2.0 * torch.rand(len(group), generator=gen)


def _ties_and_takes(slots):
    """16 experts under which rank 2's choice turns on ties and on the order. Rank 2's first
    replicated expert ties with its other home and stays; then rank 1's, whose other home is
    rank 2, ties and stays (in the other order both would move). Rank 2 takes its second
    replica, whose home runs 3 experts to rank 2's 1. Its third replica ties and stays home.
    The other 6 are experts of rank 40, which none of these meet."""
    homes = [(RANK - shift) % RANKS for shift in _shifts(slots)]
    return ([RANK * OWN, homes[0] * OWN, homes[0] * OWN + slots]
            + [homes[1] * OWN + 1] + [homes[1] * OWN + slots + s for s in range(3)]
            + [homes[2] * OWN + 2] + [homes[2] * OWN + slots + s for s in range(2)]
            + [40 * OWN + slots + s for s in range(6)])


def correlated_case(ref, tokens, replicas, slots):
    """Regime R1: rows near one direction with one channel 100x."""
    gen = torch.Generator().manual_seed(11 + tokens + slots)
    latent = (0.9 * torch.randn(ref.latent, generator=gen)
              + 0.1 * torch.randn(tokens, ref.latent, generator=gen))
    latent[:, ref.latent // 3] *= 100
    logits = _lift(LOGIT_STD * torch.randn(tokens, ref.experts, generator=gen), slots, gen)
    return f"correlated rows, T {tokens}, {slots} slots", _routed(latent, logits), replicas, slots


def repeated_case(ref):
    """Regime R2: one token repeated over the 8-row shape, choosing ``_ties_and_takes``."""
    gen = torch.Generator().manual_seed(12)
    latent = torch.randn(1, ref.latent, generator=gen).expand(8, -1)
    logits = LOGIT_STD * torch.randn(1, ref.experts, generator=gen)
    _choose(logits, 0, _ties_and_takes(3), gen)
    return "repeated tokens, T 8, 3 slots", _routed(latent, logits.expand(8, -1)), 2, 3


def twelve_case(ref):
    """Regime R3: rank 2's own experts and two others at +12, the rest at -12."""
    gen = torch.Generator().manual_seed(13)
    latent = torch.randn(8, ref.latent, generator=gen)
    logits = torch.full((8, ref.experts), -GATE_INPUT)
    logits[:, RANK * OWN:(RANK + 1) * OWN] = GATE_INPUT
    others = torch.randperm(ref.experts - OWN, generator=gen)[:2]
    logits[:, torch.where(others >= RANK * OWN, others + OWN, others)] = GATE_INPUT
    return "sigmoid inputs at 12, T 8, 3 slots", _routed(latent, logits), 2, 3


def small_rows_case(ref):
    """Regime R4: small, near-equal rows, as an average over many unit rows gives."""
    gen = torch.Generator().manual_seed(14)
    shared = torch.randn(ref.latent, generator=gen)
    latent = (shared + 2 ** -6 * torch.randn(8, ref.latent, generator=gen)) / math.sqrt(
        AVERAGED_ROWS)
    logits = _lift(LOGIT_STD * torch.randn(8, ref.experts, generator=gen), 3, gen)
    return "small near-equal rows, T 8, 3 slots", _routed(latent, logits), 2, 3


def nan_case(ref):
    """Regime R5: the clean 8-row case; the test poisons row ``NAN_ROW``. No row chooses the
    first replica, whose home rank runs more experts than rank 2, so a NaN row's choices
    counted in the choice would move that replica to rank 2."""
    gen = torch.Generator().manual_seed(15)
    latent = torch.randn(8, ref.latent, generator=gen)
    logits = _lift(LOGIT_STD * torch.randn(8, ref.experts, generator=gen), 3, gen, take=False)
    return "clean rows, T 8, 3 slots", _routed(latent, logits), 2, 3


def _assert_passes(ref, case):
    failed, _ = _grade(ref, *case)
    assert not failed, f"{case[0]} fails: {failed}"


@pytest.mark.parametrize("shape", SHAPES, ids=[f"T{t}-{s}slots" for t, _, s in SHAPES])
def test_correlated_rows_with_one_large_channel(reference, shape):
    _assert_passes(reference, correlated_case(reference, *shape))


def test_repeated_tokens(reference):
    case = repeated_case(reference)
    assert _routing(reference, case[1], case[3]).visited == [0, OWN + 1], "the case must tie and take"
    _assert_passes(reference, case)


def test_sigmoid_inputs_at_twelve(reference):
    _assert_passes(reference, twelve_case(reference))


def test_small_near_equal_rows(reference):
    _assert_passes(reference, small_rows_case(reference))


def test_nan_row_stays_in_its_row(reference):
    label, routed, replicas, slots = nan_case(reference)
    failed, clean = _grade(reference, label, routed, replicas, slots)
    poisoned = routed.clone()
    poisoned[NAN_ROW] = float("nan")
    bank = _bank([reference.stored[e] for e in _entries(slots)], reference.programs)
    route = _routing(reference, poisoned, slots)
    assert route.visited == clean.route.visited, "the case must not move an expert"
    out, _, _, visits, record = _kernel(reference, poisoned, bank, replicas, slots, expose_codes=True)
    out = _rows(out)
    others = [row for row in range(routed.shape[0]) if row != NAN_ROW]
    clean_index, clean_weight = _select(reference, routed[:, reference.latent:])
    index, weight = _select(reference, poisoned[:, reference.latent:])
    print(f"NaN row {NAN_ROW}: visits {visits.flatten().tolist()}, oracle {route.counts.tolist()}")
    assert torch.isnan(out[NAN_ROW]).all(), "the NaN row's partial is not NaN throughout"
    assert _same_choice(visits, record, route), "the NaN row's choices entered the replica choice"
    assert torch.equal(out[others].view(torch.int32), clean.served[others].view(torch.int32)), \
        "the NaN row changed another row's partial"
    assert ((index >= 0) & (index < reference.experts)).all(), "an index left its range"
    assert torch.isnan(weight[NAN_ROW]).all(), "the NaN row's weights are not NaN"
    assert torch.equal(index[others], clean_index[others]) and torch.equal(
        weight[others].view(torch.int32), clean_weight[others].view(torch.int32)), \
        "the NaN row changed another row's selection"
    assert not failed, f"{label} fails: {failed}"
