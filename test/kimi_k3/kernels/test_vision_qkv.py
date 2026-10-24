# SPDX-License-Identifier: Apache-2.0
"""The vision tower's fused RMSNorm, QKV projection and 2D RoPE, against a float64 oracle.

``qkv_proj`` runs at the tower's per-rank shape at TP 4: 3 heads of 128, no residual add.
The weight is the checkpoint's layer-1 ``wqkv`` for rank 3 as the vision loader lands it,
each q and k head in split-half order, and ``norm0`` as stored. The rows are an image's
patches in 2 x 2 window order. The RoPE tables are the worker's float32 tables times the
square root of the attention scale 1 / sqrt(128), rounded once to bf16, as the tower passes
them. The oracle is the checkpoint's ``RMSNorm``, ``wqkv`` and ``apply_rope`` in float64 on
the CPU, with q and k times the same root. The checkpoint directory comes from
``KIMI_K3_CHECKPOINT_DIR``. With ``VLLM_NEURON_CPU_MODE=1`` and ``NKI_SIMULATOR=1`` the
kernel runs on the NKI simulator; otherwise on ``neuron:0``.
"""

from __future__ import annotations

import json
import operator
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from nkilib.core.utils.common_types import NormType, QKVOutputLayout
from safetensors import safe_open

from vllm_neuron import envs
from vllm_neuron.functional.attention.qkv import qkv_proj
from vllm_neuron.model.kimi_k3.vision_inputs import rope_tables
from vllm_neuron.model.kimi_k3.weight_loaders_vision import vision_weight_loader

CHECKPOINT_DIR = os.environ.get("KIMI_K3_CHECKPOINT_DIR")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT_DIR, reason="KIMI_K3_CHECKPOINT_DIR is not set, so the reference is absent")

ON_DEVICE = not envs.VLLM_NEURON_CPU_MODE

LAYER, RANK, TP = 1, 3, 4
HEADS, D_HEAD = 3, 128
#: ``nn.RMSNorm(eps=None)`` on float32 activations.
EPS = float(torch.finfo(torch.float32).eps)
#: The square root of the attention scale, folded into the RoPE tables.
ROOT_SCALE = D_HEAD**-0.25
#: Largest error allowed per element, as a multiple of its class-B bound. The library
#: kernel rounds the normed rows to bf16 twice and stores each RoPE product in bf16; this
#: is its own simulator reading over these items, rounded up to the next 0.5.
FACTOR = 4.0
#: Grid of each shape: the served image, and the largest image the tower accepts.
GRIDS = {768: (74, 74), 5504: (74, 74), 33536: (182, 184)}
#: The 768-row shape is the image's last 688-row slice of its 5,504-row block, padded.
SLICE_START, SLICE_ROWS, SLICE_BLOCK = 4816, 688, 5504
SERVED = (768, 5504)
LARGE_CHANNEL = 341
#: Rows whose mean square is this many epsilons, so the epsilon sets their scale.
NEAR_ZERO = {5: 0.25, 6: 1.0, 7: 4.0}
NAN_ROW = 37
MATMULS = {operator.matmul, torch.matmul, torch.mm, torch.bmm, torch.einsum, F.linear}


@pytest.fixture(scope="module")
def checkpoint():
    """The layer's fused QKV weight [4608, 1024] as stored, the rank's [1024, 1152] as loaded, norm0 [1024]."""
    root = Path(CHECKPOINT_DIR)
    shards = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"vision_tower.encoder.blocks.{LAYER}."
    qkv_key, norm_key = prefix + "wqkv.weight", prefix + "norm0.weight"
    with safe_open(root / shards[qkv_key], framework="pt") as f:
        fused = f.get_tensor(qkv_key)
        weight = vision_weight_loader(qkv_key, RANK, TP).load([f.get_slice(qkv_key)], RANK)
    with safe_open(root / shards[norm_key], framework="pt") as f:
        gamma = f.get_tensor(norm_key)
    return fused, weight, gamma


def _rank_rows(fused):
    """The rank's q, k and v head rows of the fused weight, in the checkpoint's order."""
    width = fused.shape[0] // 3
    heads = slice(RANK * HEADS * D_HEAD, (RANK + 1) * HEADS * D_HEAD)
    return torch.cat([fused[part * width:(part + 1) * width][heads] for part in range(3)])


def _slice(table, tokens):
    """The 768-row shape's rows of a 5,504-row block's table, zero rows after; other shapes as given."""
    if tokens != 768:
        return table
    return torch.cat([table[SLICE_START:SLICE_START + SLICE_ROWS],
                      table.new_zeros(tokens - SLICE_ROWS, *table.shape[1:])])


def _positions(tokens):
    """Grid row and column of each row in window order, and whether the row holds a patch."""
    height, width = GRIDS[tokens]
    a, b, i, j = torch.meshgrid(torch.arange(height // 2), torch.arange(width // 2),
                                torch.arange(2), torch.arange(2), indexing="ij")
    real = torch.stack([(2 * a + i).reshape(-1), (2 * b + j).reshape(-1)], dim=-1).double()
    block = SLICE_BLOCK if tokens == 768 else tokens
    pos = torch.zeros(block, 2, dtype=torch.float64)
    pos[:real.shape[0]] = real
    patch = torch.arange(block) < real.shape[0]
    return _slice(pos, tokens), _slice(patch, tokens)


def _angles(tokens):
    """Angle of each of a head's 64 pairs, per row: the column at even pairs, the row at odd ones."""
    pos, _ = _positions(tokens)
    freqs = 1.0 / 10000.0 ** (torch.arange(0, D_HEAD, 4, dtype=torch.float64) / D_HEAD)
    return torch.stack([torch.outer(pos[:, 1], freqs), torch.outer(pos[:, 0], freqs)],
                       dim=-1).reshape(-1, D_HEAD // 2)


def _tables(tokens):
    """cos and sin [1, T, 128] in bf16: the worker's float32 tables times ROOT_SCALE."""
    block = SLICE_BLOCK if tokens == 768 else tokens
    tables = rope_tables([GRIDS[tokens]], block)
    return tuple((_slice(t, tokens) * ROOT_SCALE).to(torch.bfloat16)[None] for t in tables)


def _oracle(hidden, fused, gamma, tokens):
    """norm0, wqkv and apply_rope in float64 with q and k times ROOT_SCALE, in split halves.

    Rows that hold no patch take the zero table the worker builds for them.
    """
    x = hidden[0].double()
    normed = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * gamma.double()
    qkv = (normed @ _rank_rows(fused).double().t()).reshape(-1, 3 * HEADS, D_HEAD // 2, 2)
    angle = _angles(tokens)
    _, patch = _positions(tokens)
    length = (patch.double() * ROOT_SCALE)[:, None].expand_as(angle).contiguous()
    turn = torch.polar(length, angle)[:, None]
    rotated = torch.view_as_real(torch.view_as_complex(qkv[:, :2 * HEADS].contiguous()) * turn)
    q_k = rotated.transpose(-1, -2).reshape(-1, 2 * HEADS * D_HEAD)
    return torch.cat([q_k, qkv[:, 2 * HEADS:].reshape(-1, HEADS * D_HEAD)], dim=-1)[None]


_KERNEL_CALLS = {}


def _record(gm, example_inputs):
    """Count NKI kernel calls and torch matmuls per shape, then run the graph as the plugin does."""
    tokens = next(t.shape[1] for t in example_inputs
                  if isinstance(t, torch.Tensor) and t.dim() == 3 and t.shape[-1] == 1024)
    nodes = [n for n in gm.graph.nodes if n.op in ("call_function", "call_method")]
    nki = sum(getattr(n.target, "_name", None) == "nki_kernel_wrapper" for n in nodes)
    matmuls = sum(n.target in MATMULS if n.op == "call_function" else n.target in ("matmul", "mm")
                  for n in nodes)
    _KERNEL_CALLS[tokens] = (nki, matmuls)
    if not ON_DEVICE:
        return gm.forward
    return torch._dynamo.lookup_backend(envs.get_compile_backend_name())(gm, example_inputs)


def _call(hidden, weight, gamma, cos, sin):
    return qkv_proj(hidden, weight, norm_weights=gamma.reshape(1, -1), eps=EPS,
                    norm_type=NormType.RMS_NORM, d_head=D_HEAD, output_layout=QKVOutputLayout.BSD,
                    cos_cache=cos, sin_cache=sin, num_q_heads=HEADS, num_kv_heads=HEADS)


_COMPILED = torch.compile(_call, backend=_record, fullgraph=True, dynamic=False)


def _project(hidden, weight, gamma):
    """Run the call; fail unless it ran one NKI kernel and no torch matmul."""
    tokens = hidden.shape[1]
    args = (hidden, weight, gamma, *_tables(tokens))
    out = _COMPILED(*(a.to("neuron:0") for a in args)).cpu() if ON_DEVICE else _COMPILED(*args)
    nki, matmuls = _KERNEL_CALLS[tokens]
    assert (nki, matmuls) == (1, 0), f"T {tokens}: {nki} NKI kernel calls, {matmuls} torch matmuls"
    return out


def _class_b(out, ref):
    """Worst error over its class-B bound with its row and column, and the lowest row cosine."""
    out, ref = out.double().reshape(-1, ref.shape[-1]), ref.reshape(-1, ref.shape[-1])
    rms = ref.square().mean(-1, keepdim=True).sqrt()
    excess = ((out - ref).abs() / (2 ** -7 * (ref.abs() + rms))).flatten()
    worst = int(excess.argmax())
    cosine = F.cosine_similarity(out, ref, dim=-1).amin().item()
    return excess[worst].item(), divmod(worst, ref.shape[-1]), cosine


def _grade(label, out, ref):
    """Print the worst error over the class-B bound; return whether the case is inside FACTOR times it."""
    excess, place, cosine = _class_b(out, ref)
    print(f"{label}: worst error / bound {excess:.3f} at row {place[0]}, column {place[1]}; "
          f"lowest row cosine {cosine:.6f}; inside class B: {excess <= 1.0}")
    return excess <= FACTOR and cosine >= 0.9999


def _check(fused, weight, gamma, cases):
    """Run each (label, hidden) case; report every case outside FACTOR times the class-B bound."""
    misses = []
    for label, hidden in cases:
        hidden = hidden.bfloat16()
        ref = _oracle(hidden, fused, gamma, hidden.shape[1])
        if not _grade(label, _project(hidden, weight, gamma), ref):
            misses.append(label)
    assert not misses, f"outside {FACTOR} times the class-B bound: {misses}"


def _correlated(tokens, gen):
    hidden = 0.9 * torch.randn(1024, generator=gen) + 0.1 * torch.randn(1, tokens, 1024, generator=gen)
    hidden[..., LARGE_CHANNEL] *= 100
    return hidden


def test_correlated_rows_with_one_large_channel(checkpoint):
    """Rows near one direction, with one channel 100 times the rest."""
    gen = torch.Generator().manual_seed(1)
    _check(*checkpoint, [(f"T {t}", _correlated(t, gen)) for t in SERVED])


def test_near_zero_rows(checkpoint):
    """Rows whose mean square is a quarter of, equal to, and four times the epsilon."""
    gen = torch.Generator().manual_seed(2)
    cases = []
    for tokens in SERVED:
        hidden = _correlated(tokens, gen)
        for row, scale in NEAR_ZERO.items():
            noise = torch.randn(1024, generator=gen)
            hidden[0, row] = noise * (scale * EPS / noise.square().mean()).sqrt()
        cases.append((f"T {tokens}", hidden))
    _check(*checkpoint, cases)


def test_repeated_tokens(checkpoint):
    """Every row the same token, so only the rotation tells the outputs apart."""
    gen = torch.Generator().manual_seed(3)
    _check(*checkpoint, [(f"T {t}", torch.randn(1, 1, 1024, generator=gen).expand(1, t, 1024).contiguous())
                         for t in SERVED])


def test_every_grid_position(checkpoint):
    """Every position of the grid, the far corner at the largest angle."""
    gen = torch.Generator().manual_seed(4)
    _check(*checkpoint, [(f"T {t}", torch.randn(1, t, 1024, generator=gen)) for t in SERVED])


def test_largest_bucket(checkpoint):
    """The 33,536-row block of the largest image the tower accepts."""
    gen = torch.Generator().manual_seed(5)
    _check(*checkpoint, [("T 33536", torch.randn(1, 33536, 1024, generator=gen))])


def test_nan_row_stays_in_its_row(checkpoint):
    """One all-NaN row: its outputs are NaN, and every other row equals the clean run."""
    fused, weight, gamma = checkpoint
    gen = torch.Generator().manual_seed(6)
    misses = []
    for tokens in SERVED:
        hidden = torch.randn(1, tokens, 1024, generator=gen).bfloat16()
        clean = _project(hidden, weight, gamma)
        if not _grade(f"T {tokens}, clean", clean, _oracle(hidden, fused, gamma, tokens)):
            misses.append(f"T {tokens}: clean run outside {FACTOR} times the class-B bound")
        poisoned = hidden.clone()
        poisoned[0, NAN_ROW] = float("nan")
        out = _project(poisoned, weight, gamma)
        others = torch.arange(tokens) != NAN_ROW
        if not torch.isnan(out[0, NAN_ROW]).all():
            misses.append(f"T {tokens}: the NaN row has a finite output")
        if not torch.equal(out[0, others].view(torch.int16), clean[0, others].view(torch.int16)):
            misses.append(f"T {tokens}: other rows changed")
    assert not misses, misses
