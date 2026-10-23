# SPDX-License-Identifier: Apache-2.0
"""AttnRes decode kernel at the Kimi K3 decode shapes, against a float64 oracle.

``attn_res_decode`` folds the producer's tail into the stream, mixes each token's
bank rows and stream, and emits the rank's slice of the RMS-normed mix. The oracle is
the checkpoint's own ``_apply_attn_res`` and ``KimiRMSNorm``, with their fp32 upcasts
made float64, on the CPU; the MoE tail is ``KimiRMSNorm`` on the latent rows, an
``nn.Linear`` up projection and the shared part, from which the test builds the fp32
pack the kernel reads. The oracle rounds to bf16 only the two values the reference
stores in bf16 and reads back, the MoE block output and the new stream. Each decoder call site is a variant: its source count, its
tail, whether it stores a bank row and how many replica rows its K slice has. Every
output is graded as its own class B row set; the other replica's K-slice rows must be
exactly zero. The checkpoint directory comes from ``KIMI_K3_CHECKPOINT_DIR``. With
``VLLM_NEURON_CPU_MODE=1`` and ``NKI_SIMULATOR=1`` the kernel runs on the NKI
simulator; otherwise on ``neuron:0``.
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
from vllm_neuron.functional.attention import attn_res_decode as ard

CHECKPOINT_DIR = os.environ.get("KIMI_K3_CHECKPOINT_DIR")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT_DIR, reason="KIMI_K3_CHECKPOINT_DIR is not set, so the reference is absent")

ON_DEVICE = not envs.VLLM_NEURON_CPU_MODE

DECODE_TOKENS = 4
BANK_SLOTS = 8
TP_RANKS = 32
REPLICAS = 2
NAN_ROW = 2

#: Decoder call sites: sources, tail ("add" or "moe"), bank slot, K-slice replica rows.
SITES = {
    "KDA attention, block start": dict(sources=8, tail="moe", slot=7),
    "KDA attention": dict(sources=5, tail="moe"),
    "MLA attention": dict(sources=5, tail="moe", replicas=1),
    "layer 1 attention": dict(sources=2, tail="add"),
    "MoE MLP": dict(sources=5, tail="add", replicas=REPLICAS),
    "MoE MLP, block start": dict(sources=5, replicas=REPLICAS),
    "dense MLP": dict(sources=2),
    "output": dict(sources=9, tail="moe"),
}
NINE_SOURCE_SITES = ("KDA attention", "MLA attention", "MoE MLP", "MoE MLP, block start", "output")


class _Float64(ast.NodeTransformer):
    """Turn the reference's ``.float()`` upcasts into ``.double()``."""

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == "float":
            node.attr = "double"
        return node


@pytest.fixture(scope="module")
def reference():
    """Config widths, the float64 reference functions and the fixed weights."""
    root = Path(CHECKPOINT_DIR)
    text = json.loads((root / "config.json").read_text())
    text = text.get("text_config", text)
    source = (root / "modeling_kimi_linear.py").read_text()
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in ("KimiRMSNorm", "_apply_attn_res")]
    tree = ast.fix_missing_locations(_Float64().visit(ast.Module(nodes, [])))
    scope = {"torch": torch, "nn": torch.nn}
    exec(compile(tree, "modeling_kimi_linear.py", "exec"), scope)
    hidden, latent, eps = (text["hidden_size"], text["routed_expert_hidden_size"],
                           text["rms_norm_eps"])
    gen = torch.Generator().manual_seed(0)

    def bf16(x):
        return x.bfloat16().double()

    with torch.no_grad():
        res_norm = scope["KimiRMSNorm"](hidden, eps=eps).double()
        res_norm.weight.copy_(bf16(1 + 0.1 * torch.randn(hidden, generator=gen)))
        res_proj = torch.nn.Linear(hidden, 1, bias=False).double()
        res_proj.weight.copy_(bf16(0.02 * torch.randn(1, hidden, generator=gen)))
        norm = scope["KimiRMSNorm"](hidden, eps=eps).double()
        norm.weight.copy_(bf16(1 + 0.25 * torch.randn(hidden, generator=gen)))
        latent_norm = scope["KimiRMSNorm"](latent, eps=eps).double()
        latent_norm.weight.copy_(bf16(1 + 0.25 * torch.randn(latent, generator=gen)))
        up_proj = torch.nn.Linear(latent, hidden, bias=False).double()
        up_proj.weight.copy_(bf16(0.02 * torch.randn(hidden, latent, generator=gen)))
    return SimpleNamespace(hidden=hidden, latent=latent, eps=eps,
                           apply_attn_res=scope["_apply_attn_res"], res_norm=res_norm,
                           res_proj=res_proj, norm=norm, latent_norm=latent_norm,
                           up_proj=up_proj)


def _draw(regime, shape, gen):
    """Float64 rows of ``shape`` (tokens first) with the regime's structure on the last dim."""
    width = shape[-1]
    if regime == "correlated":
        rows = 0.9 * torch.randn(width, generator=gen, dtype=torch.float64) + 0.1 * torch.randn(
            *shape, generator=gen, dtype=torch.float64)
        rows[..., width // 3] *= 100
        return rows
    if regime == "repeated":
        return torch.randn(1, *shape[1:], generator=gen, dtype=torch.float64).expand(
            *shape).clone()
    return torch.randn(*shape, generator=gen, dtype=torch.float64)


def _inputs(ref, regime, site, gen):
    """Kernel inputs of one site under one regime, as float64 values exact in their dtypes."""
    T, H = DECODE_TOKENS, ref.hidden
    C = site["sources"]
    if regime == "equal scores":
        # Every source of a token is the same bf16 row times a power of two, so the
        # normed sources, and with them the scores, are equal.
        base = torch.randn(T, 1, H, generator=gen, dtype=torch.float64).bfloat16().double()
        power = torch.randint(-1, 3, (T, BANK_SLOTS + 1, 1), generator=gen).double()
        rows = base * torch.exp2(power)
        bank, stream = rows[:, :BANK_SLOTS], rows[:, C - 1]
        residual = latent_rows = shared = None
        if site.get("tail") == "add":
            stream = residual = stream / 2
        elif site.get("tail") == "moe":
            stream = shared = stream / 2
            latent_rows = torch.zeros(T, ref.latent, dtype=torch.float64)
    else:
        bank = _draw(regime, (T, BANK_SLOTS, H), gen).bfloat16().double()
        stream = _draw(regime, (T, H), gen).bfloat16().double()
        residual = _draw(regime, (T, H), gen).bfloat16().double()
        latent_rows = _draw(regime, (T, ref.latent), gen)
        shared = _draw(regime, (T, H), gen).float().double()
    return SimpleNamespace(bank=bank, stream=stream, residual=residual,
                           latent_rows=latent_rows, shared=shared)


def _pack(ref, x):
    """The fp32 MoE tail pack ``[up | shared | sum of squares]`` of the latent rows."""
    with torch.no_grad():
        up = ref.up_proj(ref.latent_norm.weight * x.latent_rows)
    ss = x.latent_rows.square().sum(-1, keepdim=True)
    return torch.cat([up, x.shared, ss], dim=-1).float()


def _call(ref, site, x, ranks, poison=False):
    """Run the kernel on one site's inputs; returns the outputs on the CPU."""
    C, tail = site["sources"], site.get("tail")
    stream, residual = x.stream.clone(), None if x.residual is None else x.residual.clone()
    pack = _pack(ref, x) if tail == "moe" else None
    if poison:
        stream[NAN_ROW] = float("nan")
        if tail == "add":
            residual[NAN_ROW] = float("nan")
        if tail == "moe":
            pack[NAN_ROW] = float("nan")
    args = dict(bank=x.bank.bfloat16(), stream=stream.bfloat16(),
                score_weight=ard.attn_res_score_weight(ref.res_norm.weight.bfloat16(),
                                                       ref.res_proj.weight.bfloat16()))
    kwargs = dict(residual=residual.bfloat16() if tail == "add" else None, moe_tail=pack,
                  latent_size=ref.latent if tail == "moe" else None, bank_slot=site.get("slot"),
                  eps=ref.eps)
    if site.get("replicas"):
        kwargs.update(norm_weight=ref.norm.weight.bfloat16(), tp_size=TP_RANKS,
                      tp_rank=torch.tensor([ranks[0]], dtype=torch.int32),
                      dp_size=site["replicas"], dp_rank=torch.tensor([ranks[1]], dtype=torch.int32))
    if ON_DEVICE:
        args = {k: v.to("neuron:0") for k, v in args.items()}
        kwargs = {k: v.to("neuron:0") if torch.is_tensor(v) else v for k, v in kwargs.items()}
        out = _compiled(C, tuple(sorted(site.items())))(args, kwargs)
    else:
        out = ard.attn_res_decode(**args, num_sources=C, **kwargs)
    return [None if o is None else o.cpu() for o in out]


_COMPILED = {}


def _compiled(num_sources, key):
    """The compiled call of one site variant; a new variant starts a fresh compile cache.

    The variants share one code object, so they would otherwise share its recompile limit.
    """
    if _COMPILED.get("key") != (num_sources, key):
        torch._dynamo.reset()

        def run(args, kwargs):
            return ard.attn_res_decode(**args, num_sources=num_sources, **kwargs)
        _COMPILED.update(key=(num_sources, key), call=torch.compile(
            run, backend=envs.get_compile_backend_name(), fullgraph=True, dynamic=False))
    return _COMPILED["call"]


def _bf16(x):
    return x.bfloat16().double()


def _oracle(ref, site, x, ranks):
    """The reference outputs in float64: mix, new stream, K slice, bank slot row.

    The reference stores the MoE block output and the new stream in bf16, and its
    mix reads that bf16 stream, so the oracle rounds those two values and no other.
    """
    C, tail = site["sources"], site.get("tail")
    stream = x.stream
    if tail == "add":
        stream = _bf16(stream + x.residual)
    elif tail == "moe":
        with torch.no_grad():
            moe_out = _bf16(ref.up_proj(ref.latent_norm(x.latent_rows)) + x.shared)
        stream = _bf16(stream + moe_out)
    with torch.no_grad():
        mix = ref.apply_attn_res(stream, x.bank[:, :C - 1], ref.res_proj, ref.res_norm)
        k_slice = None
        if site.get("replicas"):
            width = ref.hidden // TP_RANKS
            k_slice = torch.zeros(site["replicas"] * DECODE_TOKENS, width, dtype=torch.float64)
            rows = slice(ranks[1] * DECODE_TOKENS, (ranks[1] + 1) * DECODE_TOKENS)
            if site["replicas"] == 1:
                rows = slice(0, DECODE_TOKENS)
            k_slice[rows] = ref.norm(mix)[:, ranks[0] * width:(ranks[0] + 1) * width]
    return [mix, stream, k_slice, stream]


def _class_b(out, ref_rows):
    """Worst error over its class-B bound and the lowest row cosine of one output."""
    o, r = out.double(), ref_rows
    rms = r.square().mean(-1, keepdim=True).sqrt()
    excess = ((o - r).abs() / (2 ** -7 * (r.abs() + rms))).amax().item()
    return excess, F.cosine_similarity(o, r, dim=-1).amin().item()


def _grade(site, out, ref_outs, slot):
    """Class B misses of every output; the K slice's zero rows must be exact zeros."""
    names = ("mix", "new stream", "K slice", "bank row")
    misses, worst = [], (0.0, 1.0)
    for name, o, r in zip(names, out, ref_outs):
        if o is None or r is None:
            continue
        if name == "bank row":
            o = o[:, slot]
        if name == "K slice":
            live = r.abs().sum(-1) > 0
            if not torch.equal(o[~live], torch.zeros_like(o[~live])):
                misses.append("K slice zero rows")
            o, r = o[live], r[live]
        excess, cosine = _class_b(o, r)
        worst = (max(worst[0], excess), min(worst[1], cosine))
        if not (excess <= 1.0 and cosine >= 0.9999):
            misses.append(f"{name} ({excess:.3f}, {cosine:.6f})")
    return misses, worst


def _check(ref, regime, site_names, ranks, seed):
    gen = torch.Generator().manual_seed(seed)
    misses = []
    for name in site_names:
        site = SITES[name] if regime != "nine sources" else {**SITES[name], "sources": 9}
        x = _inputs(ref, regime, site, gen)
        out = _call(ref, site, x, ranks)
        miss, (excess, cosine) = _grade(site, out, _oracle(ref, site, x, ranks), site.get("slot"))
        print(f"{regime}, {name}, C {site['sources']}: worst error / bound {excess:.3f}, "
              f"lowest row cosine {cosine:.6f}")
        misses += [f"{name}: {m}" for m in miss]
    assert not misses, f"outside class B: {misses}"


def test_correlated_rows_with_one_large_channel(reference):
    _check(reference, "correlated", SITES, ranks=(5, 1), seed=1)


def test_repeated_tokens(reference):
    _check(reference, "repeated", SITES, ranks=(31, 0), seed=2)


def test_equal_scores(reference):
    _check(reference, "equal scores", SITES, ranks=(0, 1), seed=3)


def test_nine_sources(reference):
    _check(reference, "nine sources", NINE_SOURCE_SITES, ranks=(17, 1), seed=4)


def test_nan_row_stays_in_its_row(reference):
    gen = torch.Generator().manual_seed(5)
    ranks, failures = (9, 1), []
    for name, site in SITES.items():
        before = len(failures)
        x = _inputs(reference, "random", site, gen)
        clean = _call(reference, site, x, ranks)
        miss, _ = _grade(site, clean, _oracle(reference, site, x, ranks), site.get("slot"))
        failures += [f"{name}, clean run: {m}" for m in miss]
        out = _call(reference, site, x, ranks, poison=True)
        for label, o, c in zip(("mix", "new stream", "K slice", "bank"), out, clean):
            if o is None:
                continue
            bits = torch.int16 if o.dtype == torch.bfloat16 else torch.int32
            nan_rows = torch.zeros(o.shape[0], dtype=torch.bool)
            if label == "K slice":
                base = ranks[1] * DECODE_TOKENS if site["replicas"] > 1 else 0
                nan_rows[base + NAN_ROW] = True
            elif label != "bank" or site.get("slot") is None:
                nan_rows[NAN_ROW] = True
            if label == "bank":
                kept = o.clone()
                kept[NAN_ROW, site["slot"]] = 0
                ref_kept = c.clone()
                ref_kept[NAN_ROW, site["slot"]] = 0
                if not torch.isnan(o[NAN_ROW, site["slot"]]).all():
                    failures.append(f"{name}: the NaN row's bank row is not NaN throughout")
                if not torch.equal(kept.view(bits), ref_kept.view(bits)):
                    failures.append(f"{name}: the NaN row changed another bank entry")
                continue
            if not torch.isnan(o[nan_rows]).all():
                failures.append(f"{name}: the NaN row's {label} is not NaN throughout")
            if not torch.equal(o[~nan_rows].view(bits), c[~nan_rows].view(bits)):
                failures.append(f"{name}: the NaN row changed another row's {label}")
        print(f"NaN row, {name}: {'contained' if len(failures) == before else failures[-1]}")
    assert not failures, failures
