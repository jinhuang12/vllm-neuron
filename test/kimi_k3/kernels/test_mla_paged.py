# SPDX-License-Identifier: Apache-2.0
"""fp64 oracle test for the paged MLA attention kernels, one item per numerics regime.

The oracle transcribes ``KimiMLAAttention.forward`` from the checkpoint's
``modeling_kimi_linear.py`` to fp64, from the ``q_a`` and ``kv_a`` projections on: the two
RMS norms, ``q_b`` and ``kv_b`` unabsorbed, NoPE attention with the additive causal mask
and the sigmoid output gate. The kernels read the earlier positions as bf16 latent rows from
a paged cache built here, its unused rows NaN and its table padded with -1; the oracle reads
those rows as the cache holds them and the new positions unrounded. Pass rule, per output
row against the oracle: every element within 2^-7 of its value plus 2^-7 of the row RMS,
and row cosine at least 0.9999.

The decode output arrives head dimension first, ``[128, T, H, 1]``, the output projection's
input layout; the checks read it back token-major.

On a Neuron device the wrappers run under ``torch.compile`` on core 0 of
``NEURON_RT_VISIBLE_CORES``; the runtime needs ``NEURON_RT_ROOT_COMM_ID`` set to load the
program. With ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1`` they run in the NKI simulator.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from vllm_neuron import envs
from vllm_neuron.functional.attention.mla_paged import (
    mla_paged_decode,
    mla_paged_prefill,
    prepare_mla_weights,
)

HEADS = 3
Q_LORA = 1536
KV_LORA = 512
ROPE = 64
NOPE = 128
V_DIM = 128
QKV = Q_LORA + KV_LORA + ROPE
LATENT = KV_LORA + ROPE
EPS = 1e-6
SCALE = (NOPE + ROPE) ** -0.5
ELEMENT_TOL = 2.0**-7
ROW_RMS_TOL = 2.0**-7
MIN_ROW_COSINE = 0.9999


def _weights(gen):
    """Norm gains and the rank's ``q_b`` and ``kv_b`` slices in bf16, at the checkpoint's scale."""
    q_norm = (0.65 + 0.03 * torch.randn(Q_LORA, generator=gen)).bfloat16()
    kv_norm = (0.17 + 0.012 * torch.randn(KV_LORA, generator=gen)).bfloat16()
    q_b = (0.03 * torch.randn(HEADS * (NOPE + ROPE), Q_LORA, generator=gen)).bfloat16()
    kv_b = (0.027 * torch.randn(HEADS * (NOPE + V_DIM), KV_LORA, generator=gen)).bfloat16()
    return q_norm, kv_norm, q_b, kv_b


def _rms_norm(x, weight):
    """The checkpoint's ``KimiRMSNorm`` in fp64."""
    return weight.double() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))


def _latent(rows, kv_norm):
    """fp64 latent rows ``[norm(kv_a[:512]), kv_a[512:]]`` of qkv_a rows."""
    x = rows.double()
    return torch.cat((_rms_norm(x[:, Q_LORA:Q_LORA + KV_LORA], kv_norm), x[:, Q_LORA + KV_LORA:]),
                     dim=-1)


def _reference(rows, gate, weights, queries):
    """fp64 attention output of the last ``queries`` of ``rows``, the qkv_a rows of one sequence,
    the earlier rows' latent as the bf16 cache holds it."""
    q_norm, kv_norm, q_b, kv_b = weights
    x = rows.double()
    length = x.shape[0]
    q = (_rms_norm(x[-queries:, :Q_LORA], q_norm) @ q_b.double().T).view(queries, HEADS, -1)
    c = _latent(rows, kv_norm)
    c[:length - queries] = c[:length - queries].bfloat16().double()
    kv = (c[:, :KV_LORA] @ kv_b.double().T).view(length, HEADS, NOPE + V_DIM)
    k_rot = c[:, KV_LORA:].unsqueeze(1).expand(length, HEADS, ROPE)
    key = torch.cat((kv[..., :NOPE], k_rot), dim=-1)
    scores = torch.einsum("qhd,khd->hqk", q, key) * SCALE
    position = torch.arange(length - queries, length).unsqueeze(1)
    mask = torch.zeros(queries, length, dtype=torch.float64)
    mask.masked_fill_(torch.arange(length).unsqueeze(0) > position, float("-inf"))
    probs = F.softmax(scores + mask, dim=-1)
    out = torch.einsum("hqk,khd->qhd", probs, kv[..., NOPE:]).reshape(queries, HEADS * V_DIM)
    return out * torch.sigmoid(gate.double()), c[-queries:]


def _paged_cache(prefixes, block_size, pages, gen):
    """Cache ``[blocks, block_size, 576]`` bf16 and table ``[len(prefixes), pages]`` int32.

    Each sequence's latent rows fill its blocks in order; its blocks are a random draw from
    a pool with spare blocks, every row no sequence owns is NaN, and the table's unused
    entries are -1.
    """
    used = [math.ceil(p.shape[0] / block_size) for p in prefixes]
    blocks = sum(used) + 3
    cache = torch.full((blocks, block_size, LATENT), float("nan"), dtype=torch.bfloat16)
    table = torch.full((len(prefixes), pages), -1, dtype=torch.int32)
    order = torch.randperm(blocks, generator=gen)
    first = 0
    for s, rows in enumerate(prefixes):
        for j in range(used[s]):
            block = int(order[first + j])
            part = rows[j * block_size:(j + 1) * block_size]
            cache[block, :part.shape[0]] = part.bfloat16()
            table[s, j] = block
        first += used[s]
    return cache, table


_compiled = {}


def _run(fn, *args):
    """Run a wrapper on this venue and return its outputs on the CPU."""
    if envs.VLLM_NEURON_CPU_MODE:
        return fn(*args)
    if fn not in _compiled:
        torch.classes.neuron.Runtime().initialize()
        _compiled[fn] = torch.compile(fn, backend=envs.get_compile_backend_name(), fullgraph=True)
    device = torch.device("neuron", 0)
    moved = [a.to(device) if isinstance(a, torch.Tensor) else a for a in args]
    return tuple(t.cpu() for t in _compiled[fn](*moved))


def _decode(weights, rows, gate, block_size, pages, gen):
    """Kernel decode outputs for sequences whose qkv_a rows are ``rows``, the last one new."""
    q_norm, kv_norm, q_b, kv_b = weights
    cache, table = _paged_cache([_latent(r[:-1], kv_norm) for r in rows], block_size, pages, gen)
    lens = torch.tensor([r.shape[0] - 1 for r in rows], dtype=torch.int32)
    new = torch.stack([r[-1] for r in rows]).float()
    out, latent = _run(mla_paged_decode, new, gate, q_norm, kv_norm,
                       *prepare_mla_weights(q_b, kv_b), cache, table, lens, EPS)
    assert tuple(out.shape) == (V_DIM, len(rows), HEADS, 1) and out.dtype == torch.bfloat16
    return out[..., 0].permute(1, 2, 0).reshape(len(rows), HEADS * V_DIM), latent


def _prefill(weights, rows, gate, queries, block_size, pages, gen):
    """Kernel prefill outputs for the last ``queries`` of one sequence's qkv_a ``rows``."""
    q_norm, kv_norm, q_b, kv_b = weights
    prefix = rows.shape[0] - queries
    cache, table = _paged_cache([_latent(rows[:prefix], kv_norm)], block_size, pages, gen)
    return _run(mla_paged_prefill, rows[prefix:].float(), gate, q_norm, kv_norm,
                *prepare_mla_weights(q_b, kv_b), cache, table[0],
                torch.tensor([prefix], dtype=torch.int32), EPS)


def _assert_within_bf16_rule(out, ref):
    assert out.dtype == torch.bfloat16 and out.shape == ref.shape
    out64 = out.double()
    bound = ELEMENT_TOL * ref.abs() + ROW_RMS_TOL * ref.pow(2).mean(-1, keepdim=True).sqrt()
    over = (out64 - ref).abs() > bound
    assert not over.any(), (
        f"{int(over.sum())} elements over the bound, first at {over.nonzero()[0].tolist()}, "
        f"max error {float((out64 - ref).abs().max()):.3e}"
    )
    cosine = F.cosine_similarity(out64, ref, dim=-1)
    assert float(cosine.min()) >= MIN_ROW_COSINE, f"min row cosine {float(cosine.min()):.6f}"


def _check_decode(weights, rows, gate, block_size, pages, gen):
    out, latent = _decode(weights, rows, gate, block_size, pages, gen)
    for s, r in enumerate(rows):
        ref_out, ref_latent = _reference(r, gate[s:s + 1], weights, 1)
        _assert_within_bf16_rule(out[s:s + 1], ref_out)
        _assert_within_bf16_rule(latent[s:s + 1], ref_latent)


def _check_prefill(weights, rows, gate, queries, block_size, pages, gen):
    out, latent = _prefill(weights, rows, gate, queries, block_size, pages, gen)
    ref_out, ref_latent = _reference(rows, gate, weights, queries)
    _assert_within_bf16_rule(out, ref_out)
    _assert_within_bf16_rule(latent, ref_latent)


def _correlated(gen, length):
    """Rows 0.9 u + 0.1 noise around one shared u, one latent channel 100x the rest."""
    u = torch.randn(QKV, generator=gen)
    x = 0.9 * u + 0.1 * torch.randn(length, QKV, generator=gen)
    x[:, Q_LORA + 100] *= 100.0
    return x


def test_correlated_inputs():
    gen = torch.Generator().manual_seed(20261023)
    weights = _weights(gen)
    lens = (1536, 1000, 1, 700)
    rows = [_correlated(gen, n + 1) for n in lens]
    gate = torch.randn(len(lens), HEADS * V_DIM, generator=gen).bfloat16()
    _check_decode(weights, rows, gate, 256, 6, gen)
    gate = torch.randn(700, HEADS * V_DIM, generator=gen).bfloat16()
    _check_prefill(weights, _correlated(gen, 700 + 700), gate, 700, 256, 4, gen)


def test_repeated_tokens():
    gen = torch.Generator().manual_seed(20261023)
    weights = _weights(gen)
    token = torch.randn(1, QKV, generator=gen)
    rows = [token.expand(n + 1, -1) for n in (1024, 1024, 1024)]
    gate = torch.randn(1, HEADS * V_DIM, generator=gen).expand(3, -1).bfloat16()
    _check_decode(weights, rows, gate, 128, 8, gen)
    gate = torch.randn(1, HEADS * V_DIM, generator=gen).expand(256, -1).bfloat16()
    _check_prefill(weights, token.expand(512 + 256, -1), gate, 256, 128, 4, gen)


def test_near_one_gates():
    gen = torch.Generator().manual_seed(20261023)
    weights = _weights(gen)
    rows = [torch.randn(n + 1, QKV, generator=gen) for n in (2047, 513)]
    sign = torch.where(torch.rand(2, HEADS * V_DIM, generator=gen) < 0.5, -12.0, 12.0)
    _check_decode(weights, rows, sign.bfloat16(), 256, 8, gen)
    sign = torch.where(torch.rand(130, HEADS * V_DIM, generator=gen) < 0.5, -12.0, 12.0)
    _check_prefill(weights, torch.randn(256 + 130, QKV, generator=gen), sign.bfloat16(), 130,
                   256, 1, gen)


def test_long_context():
    gen = torch.Generator().manual_seed(20261023)
    weights = _weights(gen)
    total = 131072
    rows = [torch.randn(total, QKV, generator=gen)]
    _check_decode(weights, rows, torch.randn(1, HEADS * V_DIM, generator=gen).bfloat16(), 256,
                  total // 256, gen)
    gate = torch.randn(128, HEADS * V_DIM, generator=gen).bfloat16()
    _check_prefill(weights, rows[0], gate, 128, 256, total // 256, gen)


def test_nan_row_stays_in_its_sequence():
    gen = torch.Generator().manual_seed(20261023)
    weights = _weights(gen)
    rows = [torch.randn(n + 1, QKV, generator=gen) for n in (900, 1200, 640, 33)]
    gate = torch.randn(4, HEADS * V_DIM, generator=gen).bfloat16()
    state = gen.get_state()
    clean = _decode(weights, rows, gate, 256, 6, gen)
    gen.set_state(state)
    rows[2][-1] = float("nan")
    nan_gate = gate.clone()
    nan_gate[2] = float("nan")
    out, latent = _decode(weights, rows, nan_gate, 256, 6, gen)
    others = torch.tensor([0, 1, 3])
    assert torch.isnan(out[2]).all() and torch.isnan(latent[2]).all()
    assert torch.equal(out[others], clean[0][others])
    assert torch.equal(latent[others], clean[1][others])

    # One sequence per prefill: rows before the NaN row are unchanged, the NaN row is NaN.
    seq = torch.randn(400 + 256, QKV, generator=gen)
    pgate = torch.randn(256, HEADS * V_DIM, generator=gen).bfloat16()
    state = gen.get_state()
    clean_prefill = _prefill(weights, seq, pgate, 256, 256, 2, gen)
    # Inside a 128-row query tile and at its last row, so earlier queries of the same tile
    # meet the NaN row as a masked key.
    for nan_row in (200, 255):
        gen.set_state(state)
        poisoned = seq.clone()
        poisoned[400 + nan_row] = float("nan")
        pout, platent = _prefill(weights, poisoned, pgate, 256, 256, 2, gen)
        assert torch.isnan(pout[nan_row]).all() and torch.isnan(platent[nan_row]).all()
        assert torch.equal(pout[:nan_row], clean_prefill[0][:nan_row])
        kept = torch.arange(256) != nan_row
        assert torch.equal(platent[kept], clean_prefill[1][kept])

    for s in others.tolist():
        ref_out, ref_latent = _reference(rows[s], gate[s:s + 1], weights, 1)
        _assert_within_bf16_rule(out[s:s + 1], ref_out)
        _assert_within_bf16_rule(latent[s:s + 1], ref_latent)
