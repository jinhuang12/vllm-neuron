# SPDX-License-Identifier: Apache-2.0
"""KDA fused decode step against a float64 oracle: the checkpoint's own KDA module.

The oracle runs ``KimiDeltaAttention.forward`` from the checkpoint's
``modeling_kimi_linear.py`` in float64 on the CPU, at decode (one token per sequence,
the recurrent mode) with the three heads of one TP-32 rank. One-hot projections pass
the kernel's input columns through, and an identity ``o_proj`` returns the gated norm.
The module's library calls (short convolution, gated delta rule, gated RMSNorm) are
written out below in float64 from their upstream definitions. The small weights are
layer 1, heads 0 to 2, of the checkpoint.

The test needs ``KIMI_K3_CHECKPOINT_DIR``: a Kimi K3 checkpoint directory with its
``config.json``, ``modeling_kimi_linear.py``, ``model.safetensors.index.json`` and the
one shard that holds layer 1's attention weights. Without it the test skips.

With ``VLLM_NEURON_CPU_MODE=1`` and ``NKI_SIMULATOR=1`` the kernel runs on the NKI
simulator; otherwise on ``neuron:0``.
"""

from __future__ import annotations

import ast
import json
import os
import typing
from pathlib import Path
from types import SimpleNamespace

import einops
import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from vllm_neuron import envs
from vllm_neuron.functional.kda import fused_decode

CHECKPOINT_DIR = os.environ.get("KIMI_K3_CHECKPOINT_DIR")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT_DIR, reason="KIMI_K3_CHECKPOINT_DIR is not set, so the reference is absent")

ON_DEVICE = not envs.VLLM_NEURON_CPU_MODE

LAYER = 1
HEADS = 3
DIM = 128
WIDTH = HEADS * DIM
COLUMNS = 4 * WIDTH + DIM + HEADS
BANK_SLOTS = 9
SLOT_IDS = {4: (7, 2, 5, 0), 1: (6,)}
LONG_STEPS = 8192
NAN_ROW = 2
GATE_INPUT = 12.0
NEAR_ONE_BIAS = -11.0


def _short_convolution_class():
    class ShortConvolution(nn.Module):
        """Depthwise causal convolution and SiLU, one decode step over a [N, D, W] window."""

        def __init__(self, hidden_size, kernel_size, activation=None, **_):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(hidden_size, 1, kernel_size))

        def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
            window = torch.cat([cache[..., 1:], x.transpose(1, 2)], dim=-1)
            return F.silu((window * self.weight[:, 0]).sum(-1)).unsqueeze(1), window

    return ShortConvolution


def _gated_norm_class():
    class FusedRMSNormGated(nn.Module):
        """RMSNorm over the head dimension, times the weight and a sigmoid output gate."""

        def __init__(self, hidden_size, eps=1e-5, activation="swish", **_):
            super().__init__()
            assert activation == "sigmoid"
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size))

        def forward(self, x, g):
            norm = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
            return x * norm * self.weight * torch.sigmoid(g)

    return FusedRMSNormGated


def _fused_recurrent_kda(q, k, v, g, beta, A_log, dt_bias, initial_state, lower_bound,
                         use_qk_l2norm_in_kernel, use_gate_in_kernel,
                         use_beta_sigmoid_in_kernel, transpose_state_layout, **_):
    """One gated delta-rule step per sequence; the state is [N, H, V, K]."""
    assert use_qk_l2norm_in_kernel and use_gate_in_kernel and use_beta_sigmoid_in_kernel
    assert transpose_state_layout and q.shape[1] == 1
    q, k, v, g, beta = q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0].to(v.dtype)
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * q.shape[-1] ** -0.5
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    state = initial_state * _decay(g, A_log, dt_bias, lower_bound)[:, :, None, :]
    delta = (v - (state * k[:, :, None, :]).sum(-1)) * torch.sigmoid(beta)[..., None]
    state = state + delta[..., None] * k[:, :, None, :]
    return (state * q[:, :, None, :]).sum(-1).unsqueeze(1), state


def _decay(g, a_log, dt_bias, lower_bound):
    """Per-channel state decay, exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias)))."""
    return (lower_bound * torch.sigmoid(a_log.exp()[:, None] * (g + dt_bias.view(g.shape[-2:])))).exp()


@pytest.fixture(scope="module")
def reference():
    """The checkpoint's ``KimiDeltaAttention`` at three heads in float64, and the kernel's weights."""
    root = Path(CHECKPOINT_DIR)
    text = json.loads((root / "config.json").read_text())
    text = text.get("text_config", text)
    linear = dict(text["linear_attn_config"], num_heads=HEADS)
    source = (root / "modeling_kimi_linear.py").read_text()
    kda_class = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.ClassDef) and node.name == "KimiDeltaAttention")
    scope = {"torch": torch, "nn": nn, "rearrange": einops.rearrange, "Unpack": typing.Unpack,
             "KimiDynamicCache": object, "Cache": object,
             "ShortConvolution": _short_convolution_class(),
             "FusedRMSNormGated": _gated_norm_class(),
             "fused_recurrent_kda": _fused_recurrent_kda}
    exec(compile(ast.Module([kda_class], []), "modeling_kimi_linear.py", "exec"), scope)
    config = SimpleNamespace(hidden_size=COLUMNS, linear_attn_config=linear,
                             rms_norm_eps=text["rms_norm_eps"])
    kda = scope["KimiDeltaAttention"](config, layer_idx=LAYER).double().eval()

    prefix = f"language_model.model.layers.{LAYER}.self_attn."
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(root / index[prefix + "A_log"], framework="pt") as shard:
        stored = {name: shard.get_tensor(prefix + name) for name in (
            "A_log", "dt_bias", "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
            "f_b_proj.weight", "o_norm.weight")}
    rank = {name: (value[:HEADS] if name == "A_log" else value[:WIDTH] if value.shape[0] > DIM
                   else value).contiguous() for name, value in stored.items()}
    with torch.no_grad():
        for name, start in (("q_proj", 0), ("k_proj", WIDTH), ("v_proj", 2 * WIDTH),
                            ("g_proj", 3 * WIDTH), ("f_a_proj", 4 * WIDTH),
                            ("b_proj", 4 * WIDTH + DIM)):
            weight = getattr(kda, name).weight
            weight.zero_()
            weight[:, start:start + weight.shape[0]] = torch.eye(weight.shape[0])
        kda.o_proj.weight.zero_()
        kda.o_proj.weight[:WIDTH] = torch.eye(WIDTH)
        for name, value in rank.items():
            kda.get_parameter(name).copy_(value)
    weights = {"q_conv_weight": rank["q_conv1d.weight"], "k_conv_weight": rank["k_conv1d.weight"],
               "v_conv_weight": rank["v_conv1d.weight"], "a_log": rank["A_log"],
               "dt_bias": rank["dt_bias"], "f_b_weight": rank["f_b_proj.weight"],
               "o_norm_weight": rank["o_norm.weight"]}
    return SimpleNamespace(kda=kda, weights=weights, lower=linear["gate_lower_bound"],
                           eps=text["rms_norm_eps"])


def _oracle(ref, x, conv_rows, states):
    """One decode step in float64: core [T, 384], conv rows [T, 3, 1152], states [T, 3, K, V]."""
    windows = torch.cat([torch.zeros_like(conv_rows[:, :1]), conv_rows], 1).double()
    cache = SimpleNamespace(conv_states={LAYER: tuple(
        windows[..., s * WIDTH:(s + 1) * WIDTH].transpose(1, 2) for s in range(3))},
        recurrent_states={LAYER: states.double().transpose(-1, -2)})
    with torch.no_grad():
        core = ref.kda(x.double()[:, None], cache_params=cache)[:, 0, :WIDTH]
    new_rows = torch.cat([w.transpose(1, 2)[:, 1:] for w in cache.conv_states[LAYER]], -1)
    return core, new_rows, cache.recurrent_states[LAYER].transpose(-1, -2)


def _step(x, slots, conv_bank, state_bank, weights, lower_bound, norm_eps):
    return fused_decode.kda_fused_decode(x, slots, conv_bank, state_bank, **weights,
                                         lower_bound=lower_bound, norm_eps=norm_eps)


_compiled = torch.compile(_step, backend=envs.get_compile_backend_name(), fullgraph=True,
                          dynamic=False) if ON_DEVICE else None


def _run(ref, x, slots, conv_bank, state_bank):
    """One kernel step; returns the core output as [T, 384] and both banks, all on the CPU."""
    if ON_DEVICE:
        on = [t.to("neuron:0") for t in (x, slots, conv_bank, state_bank)]
        weights = {name: t.to("neuron:0") for name, t in ref.weights.items()}
        core, conv_bank, state_bank = (t.cpu() for t in _compiled(*on, weights, ref.lower, ref.eps))
    else:
        core, conv_bank, state_bank = _step(x, slots, conv_bank, state_bank, ref.weights, ref.lower, ref.eps)
    tokens = x.shape[0]
    assert core.shape == (DIM, tokens, HEADS, 1), f"core {tuple(core.shape)}, expected [128, T, 3, 1]"
    return core[..., 0].permute(1, 2, 0).reshape(tokens, WIDTH), conv_bank, state_bank


def _class_b(out, ref):
    """Worst error over its class-B bound, and the lowest row cosine."""
    out, ref = out.double().reshape(-1, ref.shape[-1]), ref.reshape(-1, ref.shape[-1])
    rms = ref.square().mean(-1, keepdim=True).sqrt()
    excess = ((out - ref).abs() / (2 ** -7 * (ref.abs() + rms))).amax().item()
    return excess, F.cosine_similarity(out, ref, dim=-1).amin().item()


def _class_f(out, ref):
    """Largest relative Frobenius error over the sequences' states."""
    out, ref = out.double().flatten(1), ref.flatten(1)
    return ((out - ref).norm(dim=1) / ref.norm(dim=1)).amax().item()


def _banks(gen):
    conv_bank = torch.randn(BANK_SLOTS, 3, 3 * WIDTH, generator=gen).bfloat16()
    state_bank = 0.3 * torch.randn(BANK_SLOTS, HEADS, DIM, DIM, generator=gen)
    return conv_bank, state_bank


def _inputs(gen, tokens):
    x = torch.randn(tokens, COLUMNS, generator=gen)
    x[:, 3 * WIDTH:4 * WIDTH] *= 2.0
    x[:, 4 * WIDTH + DIM:] *= 2.0
    return x.bfloat16()


def _check_step(ref, x, tokens, conv_bank, state_bank):
    """Run one step against the oracle; True when the output, rows and other slots all pass."""
    slots = torch.tensor(SLOT_IDS[tokens], dtype=torch.int32)
    core, conv_out, state_out = _run(ref, x, slots, conv_bank.clone(), state_bank.clone())
    want, want_rows, want_states = _oracle(ref, x, conv_bank[slots.long()], state_bank[slots.long()])
    excess, cosine = _class_b(core, want)
    others = [s for s in range(BANK_SLOTS) if s not in SLOT_IDS[tokens]]
    rows_exact = torch.equal(conv_out[slots.long()].double(), want_rows)
    untouched = (torch.equal(conv_out[others].view(torch.int16), conv_bank[others].view(torch.int16))
                 and torch.equal(state_out[others].view(torch.int32), state_bank[others].view(torch.int32)))
    state_error = _class_f(state_out[slots.long()], want_states)
    print(f"T {tokens}: worst error / bound {excess:.3f}, lowest row cosine {cosine:.6f}, "
          f"state error {state_error:.2e}, conv rows exact {rows_exact}, other slots untouched {untouched}")
    return excess <= 1.0 and cosine >= 0.9999 and rows_exact and untouched


def test_correlated_rows_with_one_large_channel(reference):
    gen, misses = torch.Generator().manual_seed(1), []
    for tokens in SLOT_IDS:
        x = 0.9 * torch.randn(COLUMNS, generator=gen) + 0.1 * torch.randn(tokens, COLUMNS, generator=gen)
        x[:, 2 * WIDTH + 37] *= 100
        if not _check_step(reference, x.bfloat16(), tokens, *_banks(gen)):
            misses.append(f"T {tokens}")
    assert not misses, f"outside class B, or a bank row wrong: {misses}"


def test_repeated_tokens(reference):
    gen, misses = torch.Generator().manual_seed(2), []
    for tokens in SLOT_IDS:
        x = _inputs(gen, 1).expand(tokens, COLUMNS).contiguous()
        conv_bank, state_bank = _banks(gen)
        slots = list(SLOT_IDS[tokens])
        conv_bank[slots] = conv_bank[slots[0]].clone()
        state_bank[slots] = state_bank[slots[0]].clone()
        if not _check_step(reference, x, tokens, conv_bank, state_bank):
            misses.append(f"T {tokens}")
    assert not misses, f"outside class B, or a bank row wrong: {misses}"


def test_near_one_gates_and_sigmoid_inputs_at_twelve(reference):
    gen, misses = torch.Generator().manual_seed(3), []
    near_one = SimpleNamespace(**vars(reference))
    near_one.weights = dict(reference.weights, dt_bias=torch.full((WIDTH,), NEAR_ONE_BIAS))
    saved = reference.kda.dt_bias.detach().clone()
    try:
        with torch.no_grad():
            reference.kda.dt_bias.fill_(NEAR_ONE_BIAS)
        for tokens in SLOT_IDS:
            x = _inputs(gen, tokens)
            x[:, 4 * WIDTH:4 * WIDTH + DIM] *= 0.1
            x[:, 3 * WIDTH:4 * WIDTH] = GATE_INPUT * torch.tensor([1.0, -1.0]).repeat(WIDTH // 2)
            x[:, 4 * WIDTH + DIM:] = GATE_INPUT * torch.tensor([1.0, -1.0, 1.0])
            kda = reference.kda
            gate = kda.f_b_proj(kda.f_a_proj(x.double())).view(tokens, HEADS, DIM)
            assert _decay(gate, kda.A_log, kda.dt_bias, reference.lower).amin() >= 1 - 2 ** -12
            if not _check_step(near_one, x, tokens, *_banks(gen)):
                misses.append(f"T {tokens}")
    finally:
        with torch.no_grad():
            reference.kda.dt_bias.copy_(saved)
    assert not misses, f"outside class B, or a bank row wrong: {misses}"


def test_long_carried_state(reference):
    gen, misses = torch.Generator().manual_seed(4), []
    for tokens in SLOT_IDS:
        slots = torch.tensor(SLOT_IDS[tokens], dtype=torch.int32)
        conv_bank, state_bank = _banks(gen)
        rows, states = conv_bank[slots.long()].double(), state_bank[slots.long()].double()
        worst, cosine = 0.0, 1.0
        for _ in range(LONG_STEPS):
            x = _inputs(gen, tokens)
            core, conv_bank, state_bank = _run(reference, x, slots, conv_bank, state_bank)
            want, rows, states = _oracle(reference, x, rows, states)
            excess, cos = _class_b(core, want)
            worst, cosine = max(worst, excess), min(cosine, cos)
        state_error = _class_f(state_bank[slots.long()], states)
        rows_exact = torch.equal(conv_bank[slots.long()].double(), rows)
        print(f"T {tokens}, {LONG_STEPS} steps: worst error / bound {worst:.3f}, lowest row "
              f"cosine {cosine:.6f}, state error {state_error:.2e}, conv rows exact {rows_exact}")
        if not (worst <= 1.0 and cosine >= 0.9999 and state_error <= 1e-4 and rows_exact):
            misses.append(f"T {tokens}")
    assert not misses, f"outside class B or F after {LONG_STEPS} steps: {misses}"


def test_nan_row_stays_in_its_row(reference):
    gen, misses = torch.Generator().manual_seed(5), []
    for tokens in SLOT_IDS:
        slots = torch.tensor(SLOT_IDS[tokens], dtype=torch.int32)
        x = _inputs(gen, tokens)
        conv_bank, state_bank = _banks(gen)
        clean = _run(reference, x, slots, conv_bank.clone(), state_bank.clone())
        row = min(NAN_ROW, tokens - 1)
        poisoned = x.clone()
        poisoned[row] = float("nan")
        core, conv_out, state_out = _run(reference, poisoned, slots, conv_bank.clone(), state_bank.clone())
        others = [t for t in range(tokens) if t != row]
        kept = [s for s in range(BANK_SLOTS) if s != SLOT_IDS[tokens][row]]
        want = _oracle(reference, x, conv_bank[slots.long()], state_bank[slots.long()])[0]
        excess, cosine = _class_b(clean[0], want)
        contained = (torch.isnan(core[row]).all() and torch.isnan(state_out[SLOT_IDS[tokens][row]]).all()
                     and torch.equal(core[others].view(torch.int16), clean[0][others].view(torch.int16))
                     and torch.equal(conv_out[kept].view(torch.int16), clean[1][kept].view(torch.int16))
                     and torch.equal(state_out[kept].view(torch.int32), clean[2][kept].view(torch.int32)))
        print(f"T {tokens}: clean worst error / bound {excess:.3f}, lowest row cosine {cosine:.6f}, "
              f"NaN contained {bool(contained)}")
        if not (contained and excess <= 1.0 and cosine >= 0.9999):
            misses.append(f"T {tokens}")
    assert not misses, f"NaN row leaked or clean rows outside class B: {misses}"
