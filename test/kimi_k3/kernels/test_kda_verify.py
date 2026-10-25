# SPDX-License-Identifier: Apache-2.0
"""KDA verify step against a float64 oracle: the checkpoint's KDA module, one token at a time.

The verify step runs a window of 5 tokens per sequence from one source page and writes
one page per token. The oracle runs ``KimiDeltaAttention.forward`` from the checkpoint's
``modeling_kimi_linear.py`` in float64 on the CPU at decode, one token per call, with the
three heads of one TP-32 rank, and keeps the conv rows and the state after every token.
One-hot projections pass the kernel's input columns through, and an identity ``o_proj``
returns the gated norm. The module's library calls are written out below in float64 from
their upstream definitions. The small weights are layer 1, heads 0 to 2, of the checkpoint.

Each call runs at 1, 4 and 16 sequences on a page buffer laid out as the KV cache binds
it: per page the conv rows from byte 0, then the recurrent state, then padding of fixed
random bytes. Even sequences start from their own first row page, as after prefill; odd
ones from a page of their own. Pages the call does not write must keep every byte.

The test needs ``KIMI_K3_CHECKPOINT_DIR``: a Kimi K3 checkpoint directory with its
``config.json``, ``modeling_kimi_linear.py``, ``model.safetensors.index.json`` and the
one shard that holds layer 1's attention weights. Without it the test skips.

On ``neuron:0`` the kernel runs compiled. With ``VLLM_NEURON_CPU_MODE=1`` it runs on the
NKI simulator when ``NKI_SIMULATOR=1``, and as the torch branch otherwise.
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
CONV_WIDTH = 3 * WIDTH
COLUMNS = 4 * WIDTH + DIM + HEADS
WINDOW = 5
BATCHES = (1, 4, 16)
SPARE_PAGES = 3
PAGE_BYTES = 294_912
CONV_BYTES = 3 * CONV_WIDTH * 2
STATE_END = CONV_BYTES + HEADS * DIM * DIM * 4
CARRIED_TOKENS = 8192
CARRY_WINDOWS = 2731
GATE_INPUT = 12.0
NEAR_ONE_BIAS = -11.0
NAN_TOKENS = {1: ((0, 2),), 4: ((1, 0), (2, 4)), 16: ((3, 1), (9, 3), (14, 4))}


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


def _oracle_step(ref, x, conv_rows, states):
    """One decode step in float64: core [N, 384], conv rows [N, 3, 1152], states [N, 3, K, V]."""
    windows = torch.cat([torch.zeros_like(conv_rows[:, :1]), conv_rows], 1).double()
    cache = SimpleNamespace(conv_states={LAYER: tuple(
        windows[..., s * WIDTH:(s + 1) * WIDTH].transpose(1, 2) for s in range(3))},
        recurrent_states={LAYER: states.double().transpose(-1, -2)})
    with torch.no_grad():
        core = ref.kda(x.double()[:, None], cache_params=cache)[:, 0, :WIDTH]
    new_rows = torch.cat([w.transpose(1, 2)[:, 1:] for w in cache.conv_states[LAYER]], -1)
    return core, new_rows, cache.recurrent_states[LAYER].transpose(-1, -2)


def _oracle(ref, x, conv_rows, states):
    """The window token by token: x [B, W, C] from conv rows [B, 3, 1152] and states [B, 3, K, V].

    Returns core [B, W, 384], and the conv rows [B, W, 3, 1152] and states [B, W, 3, K, V]
    after each token, all float64.
    """
    cores, rows, kept = [], [], []
    conv_rows, states = conv_rows.double(), states.double()
    for j in range(x.shape[1]):
        core, conv_rows, states = _oracle_step(ref, x[:, j], conv_rows, states)
        cores.append(core)
        rows.append(conv_rows)
        kept.append(states)
    return torch.stack(cores, 1), torch.stack(rows, 1), torch.stack(kept, 1)


def _step(x, source, rows, pages, weights, lower_bound, norm_eps):
    return fused_decode.kda_fused_verify(x, source, rows, pages, **weights,
                                         lower_bound=lower_bound, norm_eps=norm_eps)


_compiled = torch.compile(_step, backend=envs.get_compile_backend_name(), fullgraph=True,
                          dynamic=False) if ON_DEVICE else None


def _banks_of(pages):
    """The conv rows [N, 3, 1152] bf16 and states [N, 3, K, V] fp32 of every page."""
    conv = pages[:, :CONV_BYTES].contiguous().view(torch.bfloat16)
    state = pages[:, CONV_BYTES:STATE_END].contiguous().view(torch.float32)
    return conv.reshape(-1, 3, CONV_WIDTH), state.reshape(-1, HEADS, DIM, DIM)


def _layout(batch, gen):
    """Pages for one call: count, source [B] and row pages [B, 5], every id distinct but the
    even sequences' sources, which are their own first row page."""
    count = (WINDOW + 1) * batch + SPARE_PAGES
    order = torch.randperm(count, generator=gen)
    rows = order[:batch * WINDOW].view(batch, WINDOW)
    own = order[batch * WINDOW:batch * (WINDOW + 1)]
    source = torch.where(torch.arange(batch) % 2 == 0, rows[:, 0], own)
    return count, source.int(), rows.int()


def _pages(count, source, conv_rows, states, gen):
    """A page buffer of fixed random bytes with each sequence's starting rows in its source page."""
    pages = torch.randint(-128, 128, (count, PAGE_BYTES), dtype=torch.int8, generator=gen)
    at = source.long()
    pages[at, :CONV_BYTES] = conv_rows.reshape(len(at), -1).view(torch.int8)
    pages[at, CONV_BYTES:STATE_END] = states.reshape(len(at), -1).view(torch.int8)
    return pages


def _starts(batch, gen):
    conv_rows = torch.randn(batch, 3, CONV_WIDTH, generator=gen).bfloat16()
    states = 0.3 * torch.randn(batch, HEADS, DIM, DIM, generator=gen)
    return conv_rows, states


def _inputs(gen, tokens):
    x = torch.randn(tokens, COLUMNS, generator=gen)
    x[:, 3 * WIDTH:4 * WIDTH] *= 2.0
    x[:, 4 * WIDTH + DIM:] *= 2.0
    return x.bfloat16()


def _case(batch, gen, x=None, starts=None):
    """One call's inputs: x [B * 5, C], source, rows and the page buffer."""
    count, source, rows = _layout(batch, gen)
    conv_rows, states = _starts(batch, gen) if starts is None else starts
    x = _inputs(gen, batch * WINDOW) if x is None else x
    return SimpleNamespace(x=x, source=source, rows=rows,
                           pages=_pages(count, source, conv_rows, states, gen))


def _run(ref, case, weights=None):
    """One kernel call; returns core [B, 5, 384] and the page buffer, both on the CPU."""
    weights = ref.weights if weights is None else weights
    args = (case.x, case.source, case.rows, case.pages.clone())
    if ON_DEVICE:
        on = [t.to("neuron:0") for t in args]
        weights = {name: t.to("neuron:0") for name, t in weights.items()}
        core, pages = (t.cpu() for t in _compiled(*on, weights, ref.lower, ref.eps))
    else:
        core, pages = _step(*args, weights, ref.lower, ref.eps)
    batch = case.rows.shape[0]
    assert core.shape == (DIM, batch * WINDOW, HEADS, 1), f"core {tuple(core.shape)}, expected [128, 5B, 3, 1]"
    written = case.rows.flatten().long()
    kept = torch.ones(len(pages), dtype=torch.bool)
    kept[written] = False
    assert torch.equal(pages[kept], case.pages[kept]), "the kernel wrote a page outside the row pages"
    assert torch.equal(pages[written, STATE_END:], case.pages[written, STATE_END:]), \
        "the kernel wrote page padding"
    return core[..., 0].permute(1, 2, 0).reshape(batch, WINDOW, WIDTH), pages


def _class_b(out, ref):
    """Worst error over its class-B bound, and the lowest row cosine."""
    out, ref = out.double().reshape(-1, ref.shape[-1]), ref.reshape(-1, ref.shape[-1])
    rms = ref.square().mean(-1, keepdim=True).sqrt()
    excess = ((out - ref).abs() / (2 ** -7 * (ref.abs() + rms))).amax().item()
    return excess, F.cosine_similarity(out, ref, dim=-1).amin().item()


def _class_f(out, ref):
    """Largest relative Frobenius error over every head's state [K, V]."""
    out, ref = out.double().flatten(-2), ref.flatten(-2)
    return ((out - ref).norm(dim=-1) / ref.norm(dim=-1)).amax().item()


def _rows_match(conv, case):
    """True when each row page holds its window's 3 newest conv rows: the source page's rows,
    then the tokens' q, k and v columns, bit for bit, or NaN where those are NaN."""
    batch = case.rows.shape[0]
    start = _banks_of(case.pages)[0][case.source.long()]
    framed = torch.cat([start, case.x[:, :CONV_WIDTH].view(batch, WINDOW, CONV_WIDTH)], 1)
    for j in range(WINDOW):
        got, want = conv[:, j], framed[:, j + 1:j + 4]
        nan = want.isnan()
        if not (torch.equal(got.isnan(), nan)
                and torch.equal(got[~nan].view(torch.int16), want[~nan].view(torch.int16))):
            return False
    return True


def _grade(ref, case, core, pages, label):
    """Grade one call against the oracle; True when every value is inside its class."""
    batch = case.rows.shape[0]
    conv0, state0 = (t[case.source.long()] for t in _banks_of(case.pages))
    want, _, want_states = _oracle(ref, case.x.view(batch, WINDOW, COLUMNS), conv0, state0)
    conv, states = (t[case.rows.long()] for t in _banks_of(pages))
    rows_exact = _rows_match(conv, case)
    excess, cosine = _class_b(core, want)
    state_error = _class_f(states, want_states)
    print(f"{label} B {batch}: worst error / bound {excess:.3f}, lowest row cosine {cosine:.6f}, "
          f"page state error {state_error:.2e}, conv rows exact {rows_exact}")
    return excess <= 1.0 and cosine >= 0.9999 and state_error <= 1e-4 and rows_exact


def test_correlated_rows_with_one_large_channel(reference):
    gen, misses = torch.Generator().manual_seed(1), []
    for batch in BATCHES:
        x = 0.9 * torch.randn(COLUMNS, generator=gen) + 0.1 * torch.randn(batch * WINDOW, COLUMNS, generator=gen)
        x[:, 2 * WIDTH + 37] *= 100
        case = _case(batch, gen, x=x.bfloat16())
        if not _grade(reference, case, *_run(reference, case), "correlated rows"):
            misses.append(f"B {batch}")
    assert not misses, f"outside class B or F, or a page row wrong: {misses}"


def test_repeated_tokens(reference):
    gen, misses = torch.Generator().manual_seed(2), []
    for batch in BATCHES:
        x = _inputs(gen, 1).expand(batch * WINDOW, COLUMNS).contiguous()
        conv_rows, states = _starts(1, gen)
        starts = (conv_rows.expand(batch, -1, -1).contiguous(), states.expand(batch, -1, -1, -1).contiguous())
        case = _case(batch, gen, x=x, starts=starts)
        if not _grade(reference, case, *_run(reference, case), "repeated tokens"):
            misses.append(f"B {batch}")
    assert not misses, f"outside class B or F, or a page row wrong: {misses}"


def test_near_one_gates_and_sigmoid_inputs_at_twelve(reference):
    gen, misses = torch.Generator().manual_seed(3), []
    near_one = SimpleNamespace(**vars(reference))
    near_one.weights = dict(reference.weights, dt_bias=torch.full((WIDTH,), NEAR_ONE_BIAS))
    saved = reference.kda.dt_bias.detach().clone()
    try:
        with torch.no_grad():
            reference.kda.dt_bias.fill_(NEAR_ONE_BIAS)
        for batch in BATCHES:
            x = _inputs(gen, batch * WINDOW)
            x[:, 4 * WIDTH:4 * WIDTH + DIM] *= 0.1
            x[:, 3 * WIDTH:4 * WIDTH] = GATE_INPUT * torch.tensor([1.0, -1.0]).repeat(WIDTH // 2)
            x[:, 4 * WIDTH + DIM:] = GATE_INPUT * torch.tensor([1.0, -1.0, 1.0])
            kda = reference.kda
            gate = kda.f_b_proj(kda.f_a_proj(x.double())).view(-1, HEADS, DIM)
            assert _decay(gate, kda.A_log, kda.dt_bias, reference.lower).amin() >= 1 - 2 ** -12
            case = _case(batch, gen, x=x)
            if not _grade(near_one, case, *_run(near_one, case), "near-one gates"):
                misses.append(f"B {batch}")
    finally:
        with torch.no_grad():
            reference.kda.dt_bias.copy_(saved)
    assert not misses, f"outside class B or F, or a page row wrong: {misses}"


def _accepted_counts(batch, gen):
    """Tokens kept per window [B, windows], each 1 to 5, summing to the carried count per sequence."""
    counts = torch.randint(1, WINDOW + 1, (batch, CARRY_WINDOWS), generator=gen)
    for b in range(batch):
        while (gap := CARRIED_TOKENS - int(counts[b].sum())) != 0:
            at = int(torch.randint(CARRY_WINDOWS, (1,), generator=gen))
            if gap > 0 and counts[b, at] < WINDOW:
                counts[b, at] += 1
            elif gap < 0 and counts[b, at] > 1:
                counts[b, at] -= 1
    return counts


def test_long_carried_state(reference):
    gen, misses = torch.Generator().manual_seed(4), []
    device = "neuron:0" if ON_DEVICE else "cpu"
    step = _compiled if ON_DEVICE else _step
    weights = {name: t.to(device) for name, t in reference.weights.items()}
    for batch in BATCHES:
        case = _case(batch, gen)
        counts = _accepted_counts(batch, gen)
        conv_rows, states = (t[case.source.long()] for t in _banks_of(case.pages))
        pages, source, rows = case.pages.to(device), case.source, case.rows.to(device)
        worst, cosine, picks = 0.0, 1.0, torch.arange(batch)
        for w in range(CARRY_WINDOWS + 1):
            x = _inputs(gen, batch * WINDOW)
            core, pages = step(x.to(device), source.to(device), rows, pages, weights, reference.lower,
                               reference.eps)
            want, kept_rows, kept_states = _oracle(reference, x.view(batch, WINDOW, COLUMNS), conv_rows, states)
            out, ref = core.cpu()[..., 0].permute(1, 2, 0).reshape(-1, WIDTH), want.reshape(-1, WIDTH)
            # Python's max and min below drop a NaN, so a non-finite output fails here.
            assert torch.equal(~torch.isfinite(out), ~torch.isfinite(ref)), \
                f"B {batch}, window {w}: the output is non-finite where the oracle is not, or the reverse"
            finite = (torch.isfinite(out) & torch.isfinite(ref)).all(-1)
            excess, cos = _class_b(out[finite], ref[finite])
            worst, cosine = max(worst, excess), min(cosine, cos)
            if w < CARRY_WINDOWS:
                last = counts[:, w] - 1
                conv_rows, states = kept_rows[picks, last], kept_states[picks, last]
                source = case.rows[picks, last].int()
        conv, page_states = (t[case.rows.long()] for t in _banks_of(pages.cpu()))
        state_error = _class_f(page_states, kept_states)
        rows_exact = torch.equal(conv.double(), kept_rows)
        print(f"B {batch}, {CARRIED_TOKENS} tokens carried then one window: worst error / bound "
              f"{worst:.3f}, lowest row cosine {cosine:.6f}, page state error {state_error:.2e}, "
              f"conv rows exact {rows_exact}")
        if not (worst <= 1.0 and cosine >= 0.9999 and state_error <= 1e-4 and rows_exact):
            misses.append(f"B {batch}")
    assert not misses, f"outside class B or F after {CARRIED_TOKENS} carried tokens: {misses}"


def test_nan_token_stays_in_its_window(reference):
    gen, misses = torch.Generator().manual_seed(5), []
    for batch in BATCHES:
        case = _case(batch, gen)
        clean_core, clean_pages = _run(reference, case)
        poisoned = SimpleNamespace(**vars(case))
        poisoned.x = case.x.clone()
        for b, j in NAN_TOKENS[batch]:
            poisoned.x[b * WINDOW + j] = float("nan")
        core, pages = _run(reference, poisoned)
        conv, states = (t[case.rows.long()] for t in _banks_of(pages))
        clean_conv, clean_states = (t[case.rows.long()] for t in _banks_of(clean_pages))
        dirty = torch.zeros(batch, WINDOW, dtype=torch.bool)
        for b, j in NAN_TOKENS[batch]:
            dirty[b, j:] = True
        contained = (bool(torch.isnan(core[dirty]).all()) and bool(torch.isnan(states[dirty]).all())
                     and torch.equal(core[~dirty].view(torch.int16), clean_core[~dirty].view(torch.int16))
                     and torch.equal(states[~dirty].view(torch.int32), clean_states[~dirty].view(torch.int32))
                     and torch.equal(conv[~dirty].view(torch.int16), clean_conv[~dirty].view(torch.int16))
                     and _rows_match(conv, poisoned))
        print(f"B {batch}: NaN tokens at {NAN_TOKENS[batch]}, contained {contained}")
        if not (contained and _grade(reference, case, clean_core, clean_pages, "clean run")):
            misses.append(f"B {batch}")
    assert not misses, f"a NaN token leaked, or the clean run is outside its class: {misses}"


def test_each_page_read_back_as_the_source_page(reference):
    gen, misses = torch.Generator().manual_seed(7), []
    for batch in BATCHES:
        first, at = _case(batch, gen), torch.arange(batch)
        _, pages = _run(reference, first)
        _, kept_rows, kept_states = _oracle(reference, first.x.view(batch, WINDOW, COLUMNS),
                                            *(t[first.source.long()] for t in _banks_of(first.pages)))
        for k in range(-(-WINDOW // batch)):
            picked = (at + k * batch) % WINDOW
            second = SimpleNamespace(x=_inputs(gen, batch * WINDOW), rows=first.rows, pages=pages.clone(),
                                     source=first.rows[at, picked].int())
            core, out = _run(reference, second)
            want, _, want_states = _oracle(reference, second.x.view(batch, WINDOW, COLUMNS),
                                           kept_rows[at, picked], kept_states[at, picked])
            conv, states = (t[first.rows.long()] for t in _banks_of(out))
            excess, cosine = _class_b(core, want)
            state_error = _class_f(states, want_states)
            rows_exact = _rows_match(conv, second)
            print(f"B {batch}, source pages {picked.tolist()}: worst error / bound {excess:.3f}, lowest "
                  f"row cosine {cosine:.6f}, page state error {state_error:.2e}, conv rows exact {rows_exact}")
            if not (excess <= 1.0 and cosine >= 0.9999 and state_error <= 1e-4 and rows_exact):
                misses.append(f"B {batch}, source pages {picked.tolist()}")
    assert not misses, f"a page read back as a source is not the state after its tokens: {misses}"
