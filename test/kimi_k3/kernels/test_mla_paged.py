# SPDX-License-Identifier: Apache-2.0
"""fp64 oracle test for the paged MLA attention kernels, one item per numerics regime.

The oracle transcribes ``KimiMLAAttention.forward`` from the checkpoint's
``modeling_kimi_linear.py`` to fp64, from the ``q_a`` and ``kv_a`` projections on: the two
RMS norms, ``q_b`` and ``kv_b`` unabsorbed, NoPE attention with the additive causal mask
and the sigmoid output gate. The kernels read the earlier positions as bf16 latent rows from
an int8 page buffer built here, its unused rows NaN and its table padded with -1, and write
the new tokens' latent rows into it at their slots; the oracle reads the cached rows as the
buffer holds them and the new positions unrounded. Pass rule, per output row and per new
latent row against the oracle: every element within 2^-7 of its value plus 2^-7 of the row
RMS, and row cosine at least 0.9999.

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
ROW_BYTES = LATENT * 2
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


def _reference(rows, gate, weights, queries, cached=None):
    """fp64 attention output of the last ``queries`` of ``rows``, the qkv_a rows of one sequence,
    the earlier rows' latent as the bf16 cache holds it, or ``cached`` where given."""
    q_norm, kv_norm, q_b, kv_b = weights
    x = rows.double()
    length = x.shape[0]
    q = (_rms_norm(x[-queries:, :Q_LORA], q_norm) @ q_b.double().T).view(queries, HEADS, -1)
    c = _latent(rows, kv_norm)
    held = c[:length - queries].bfloat16() if cached is None else cached
    c[:length - queries] = held.double()
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
    """Cache ``[blocks, block_size, 576]`` bf16, table ``[len(prefixes), pages]`` int32, each
    sequence's count of filled blocks and the spare blocks.

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
    return cache, table, used, order[first:].tolist()


def _as_pages(cache):
    """The int8 page buffer ``[blocks, page bytes]`` of a cache, one latent block per page."""
    return cache.view(torch.int8).reshape(cache.shape[0], -1)


def _new_slots(table, used, spare, positions, block_size):
    """Slots of new tokens at ``positions[s]`` of sequence ``s``: the row of the sequence's block
    that holds the position, or of a spare block where the position starts a new block."""
    spare, slots = list(spare), []
    for s, pos in enumerate(positions):
        fresh = {}
        for p in pos:
            j = p // block_size
            if j >= used[s] and j not in fresh:
                fresh[j] = spare.pop(0)
            block = int(table[s, j]) if j < used[s] else fresh[j]
            slots.append(block * block_size + p % block_size)
    return torch.tensor(slots, dtype=torch.int32)


def _rows_at(pages, slots, block_size, offset=0):
    """Latent rows ``[len(slots), 576]`` bf16 at ``slots`` of a page buffer whose latent block
    starts ``offset`` bytes into each page."""
    rows = pages.view(torch.bfloat16).reshape(pages.shape[0], -1, LATENT)
    s = slots.long()
    return rows[s // block_size, offset // ROW_BYTES + s % block_size]


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
    """Kernel decode output and the new latent rows read back at their slots, for sequences
    whose qkv_a rows are ``rows``, the last one new."""
    q_norm, kv_norm, q_b, kv_b = weights
    cache, table, used, spare = _paged_cache([_latent(r[:-1], kv_norm) for r in rows], block_size,
                                             pages, gen)
    lens = torch.tensor([r.shape[0] - 1 for r in rows], dtype=torch.int32)
    slots = _new_slots(table, used, spare, [[int(n)] for n in lens], block_size)
    new = torch.stack([r[-1] for r in rows]).float()
    out, buf = _run(mla_paged_decode, new, gate, q_norm, kv_norm, *prepare_mla_weights(q_b, kv_b),
                    _as_pages(cache), 0, block_size, table, lens, slots, EPS)
    assert tuple(out.shape) == (V_DIM, len(rows), HEADS, 1) and out.dtype == torch.bfloat16
    out = out[..., 0].permute(1, 2, 0).reshape(len(rows), HEADS * V_DIM)
    return out, _rows_at(buf, slots, block_size)


def _prefill(weights, rows, gate, queries, block_size, pages, gen):
    """Kernel prefill output and the chunk's latent rows read back at their slots, for the last
    ``queries`` of one sequence's qkv_a ``rows``."""
    q_norm, kv_norm, q_b, kv_b = weights
    prefix = rows.shape[0] - queries
    cache, table, used, spare = _paged_cache([_latent(rows[:prefix], kv_norm)], block_size, pages,
                                             gen)
    slots = _new_slots(table, used, spare, [range(prefix, rows.shape[0])], block_size)
    out, buf = _run(mla_paged_prefill, rows[prefix:].float(), gate, q_norm, kv_norm,
                    *prepare_mla_weights(q_b, kv_b), _as_pages(cache), 0, block_size, table[0],
                    torch.tensor([prefix], dtype=torch.int32), slots, EPS)
    return out, _rows_at(buf, slots, block_size)


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


def _nan_pages(blocks, page_bytes, gen):
    """An int8 page buffer whose every bf16 word is a NaN of random sign and payload."""
    words = torch.randint(1, 128, (blocks, page_bytes // 2), generator=gen, dtype=torch.int32)
    words |= 0x7F80 | (torch.randint(0, 2, words.shape, generator=gen, dtype=torch.int32) << 15)
    return (words - (words >= 0x8000).int() * 0x10000).to(torch.int16).view(torch.int8)


def _check_written(before, after, slots, ref_rows, block_size, offset):
    """The rows at ``slots`` meet the oracle's rows, and no other byte of the buffer changed."""
    _assert_within_bf16_rule(_rows_at(after, slots, block_size, offset), ref_rows)
    rows = before.shape[1] // ROW_BYTES
    s = slots.long()
    keep = torch.ones(before.shape[0] * rows, dtype=torch.bool)
    keep[(s // block_size) * rows + offset // ROW_BYTES + s % block_size] = False
    assert torch.equal(after.reshape(-1, ROW_BYTES)[keep], before.reshape(-1, ROW_BYTES)[keep]), \
        "the kernel wrote outside the new rows"


def test_rows_written_in_place():
    """A decode step, then a prefill chunk, on a NaN-filled page buffer whose latent block sits one
    row into a wider page: each new row lands at its slot, a padding token's (-1) at block 0, and
    no other byte changes."""
    gen = torch.Generator().manual_seed(20261024)
    weights = _weights(gen)
    q_norm, kv_norm, q_b, kv_b = weights
    layouts = prepare_mla_weights(q_b, kv_b)
    block_size, offset, blocks = 256, ROW_BYTES, 12
    buf = _nan_pages(blocks, (block_size + 2) * ROW_BYTES, gen)
    page_rows = buf.view(torch.bfloat16).reshape(blocks, -1, LATENT)
    order = (1 + torch.randperm(blocks - 1, generator=gen)).tolist()  # block 0 is the null block
    lens = (300, 37, 511)
    seqs = [torch.randn(n + 1, QKV, generator=gen) for n in lens]
    table = torch.full((len(lens) + 1, 4), -1, dtype=torch.int32)  # the last row is padding
    for s, rows in enumerate(seqs):
        latent = _latent(rows[:-1], kv_norm).bfloat16()
        for j in range(math.ceil(lens[s] / block_size)):
            table[s, j] = order.pop(0)
            part = latent[j * block_size:(j + 1) * block_size]
            page_rows[int(table[s, j]), 1:1 + part.shape[0]] = part

    def slot(s, p):
        return int(table[s, p // block_size]) * block_size + p % block_size

    new = torch.cat([torch.stack([r[-1] for r in seqs]), torch.randn(1, QKV, generator=gen)])
    gate = torch.randn(len(lens) + 1, HEADS * V_DIM, generator=gen).bfloat16()
    slots = torch.tensor([slot(s, n) for s, n in enumerate(lens)] + [-1], dtype=torch.int32)
    before = buf.clone()
    out, buf = _run(mla_paged_decode, new.float(), gate, q_norm, kv_norm, *layouts, buf, offset,
                    block_size, table, torch.tensor(lens + (0,), dtype=torch.int32), slots, EPS)
    out = out[..., 0].permute(1, 2, 0).reshape(len(lens) + 1, HEADS * V_DIM)
    for s, rows in enumerate(seqs):
        _assert_within_bf16_rule(out[s:s + 1], _reference(rows, gate[s:s + 1], weights, 1)[0])
    _check_written(before, buf, slots.clamp(min=0), _latent(new, kv_norm), block_size, offset)

    # The first sequence goes on with a 300-token chunk over its 301 cached rows, decode's included,
    # into a new block, and one padding token.
    prefix, chunk = len(seqs[0]), torch.randn(301, QKV, generator=gen)
    table[0, 2] = order.pop(0)
    slots = torch.tensor([slot(0, p) for p in range(prefix, prefix + 300)] + [-1],
                         dtype=torch.int32)
    cached = _rows_at(buf, torch.tensor([slot(0, p) for p in range(prefix)]), block_size, offset)
    gate = torch.randn(301, HEADS * V_DIM, generator=gen).bfloat16()
    before = buf.clone()
    out, buf = _run(mla_paged_prefill, chunk.float(), gate, q_norm, kv_norm, *layouts, buf, offset,
                    block_size, table[0], torch.tensor([prefix], dtype=torch.int32), slots, EPS)
    ref = _reference(torch.cat([seqs[0], chunk[:300]]), gate[:300], weights, 300, cached)[0]
    _assert_within_bf16_rule(out[:300], ref)
    _check_written(before, buf, slots.clamp(min=0), _latent(chunk, kv_norm), block_size, offset)
