# SPDX-License-Identifier: Apache-2.0
"""fp64 oracle test for the paged MLA window mode in DCP mode, one item per numerics regime.

A window is the last ``Q`` rows of a sequence, after its cached rows: the target's verify
window, ``Q = 5``, gated and with no rotary, or the draft block, ``Q = 4``, with the interleaved
YaRN rotary on the 64 rope columns and no gate. Query ``i`` attends to the cached rows and the
window rows ``j <= i``. A DCP group of ``d`` ranks splits each sequence's rows by block: block
``j`` lies on rank ``j mod d``, so a window row is a key on its block's rank only, and its slot
is -1 on the other ranks. Each rank holds 3 heads, and the group's ``3 d`` heads see the whole
sequence through the merge.

The verify oracle transcribes ``KimiMLAAttention.forward`` from the checkpoint's
``modeling_kimi_linear.py`` to fp64 over the ``3 d`` heads: the two RMS norms, ``q_b`` and
``kv_b`` unabsorbed, NoPE attention under the causal mask and the sigmoid output gate. The draft
oracle transcribes upstream's DSpark draft attention, a DeepSeek MLA layer with the interleaved
YaRN rotary of the draft config on the query's and the key's rope columns, the rotary table in
fp64 as upstream's DeepSeek scaling rotary builds it; the kernel reads that table rounded once
to fp32. Both read the cached rows as bf16 and the window rows unrounded. Each rank's output is
checked on its own heads, and each window latent row on its owner. Pass rule, per row: every
element within 2^-7 of its value plus 2^-7 of the row RMS, and row cosine at least 0.9999.

Each item runs the group as one process per rank. With ``VLLM_NEURON_CPU_MODE=1
NKI_SIMULATOR=1`` the ranks are gloo processes on the CPU and the kernel runs in the NKI
simulator, which carries the kernel's collectives over gloo. On a Neuron device the group runs
through ``MPExecutor`` on the first ``d`` cores of ``NEURON_RT_VISIBLE_CORES``, each rank's
call under ``torch.compile``; a case whose group needs more cores than that is skipped.
"""

import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from vllm_neuron import envs
from vllm_neuron.functional.attention.mla_paged import mla_paged_window_dcp, prepare_mla_weights

HEADS = 3
Q_LORA = 1536
KV_LORA = 512
ROPE = 64
NOPE = 128
V_DIM = 128
QKV = Q_LORA + KV_LORA + ROPE
LATENT = KV_LORA + ROPE
SCALE = (NOPE + ROPE) ** -0.5
ELEMENT_TOL = 2.0**-7
ROW_RMS_TOL = 2.0**-7
MIN_ROW_COSINE = 0.9999
BLOCK = 256
# The draft config's rotary: rope_theta, rope_scaling and the norms' rms_norm_eps.
THETA = 1.0e6
FACTOR = 32.0
ORIGINAL = 32768
BETA_FAST = 32
BETA_SLOW = 1
MSCALE = 1.0
MSCALE_ALL_DIM = 0.0
ATTN_FACTOR = 0.7426255848312643
# Verify: the target model file's norm epsilon. Draft: the draft config's.
FORMS = {"verify": SimpleNamespace(q=5, eps=1e-6, gated=True),
         "draft": SimpleNamespace(q=4, eps=1e-5, gated=False)}


def _yarn_mscale(scale, mscale):
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def _yarn_table(positions):
    """fp64 cos | sin ``[positions, 64]``, as upstream's DeepSeek scaling rotary computes it."""

    def correction(rotations):
        return ROPE * math.log(ORIGINAL / (rotations * 2 * math.pi)) / (2 * math.log(THETA))

    low = max(math.floor(correction(BETA_FAST)), 0)
    high = min(math.ceil(correction(BETA_SLOW)), ROPE - 1)
    high = high + 0.001 if low == high else high
    ramp = ((torch.arange(ROPE // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    pos_freqs = THETA ** (torch.arange(0, ROPE, 2, dtype=torch.float64) / ROPE)
    mask = 1 - ramp
    inv_freq = (1 - mask) / (FACTOR * pos_freqs) + mask / pos_freqs
    mscale = _yarn_mscale(FACTOR, MSCALE) / _yarn_mscale(FACTOR, MSCALE_ALL_DIM) * ATTN_FACTOR
    angles = torch.arange(positions, dtype=torch.float64)[:, None] * inv_freq
    return torch.cat((angles.cos(), angles.sin()), dim=-1) * mscale


def _rotate(x, table):
    """Upstream's interleaved rotary of ``x [..., 64]`` by the matching ``table [..., 64]``."""
    cos, sin = table[..., :ROPE // 2], table[..., ROPE // 2:]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)


def _weights(gen, ranks):
    """Norm gains and the group's ``q_b`` and ``kv_b`` in bf16, at the checkpoint's scale."""
    heads = HEADS * ranks
    q_norm = (0.65 + 0.03 * torch.randn(Q_LORA, generator=gen)).bfloat16()
    kv_norm = (0.17 + 0.012 * torch.randn(KV_LORA, generator=gen)).bfloat16()
    q_b = (0.03 * torch.randn(heads * (NOPE + ROPE), Q_LORA, generator=gen)).bfloat16()
    kv_b = (0.027 * torch.randn(heads * (NOPE + V_DIM), KV_LORA, generator=gen)).bfloat16()
    return q_norm, kv_norm, q_b, kv_b


def _rms_norm(x, weight, eps):
    return weight.double() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def _latent(rows, kv_norm, form, table):
    """fp64 latent rows ``[norm(kv_a[:512]), kv_a[512:]]`` of qkv_a rows at positions 0..; the
    draft rotates the rope columns by each row's position."""
    x = rows.double()
    rope = x[:, Q_LORA + KV_LORA:]
    if not form.gated:
        rope = _rotate(rope, table[:rows.shape[0]])
    return torch.cat((_rms_norm(x[:, Q_LORA:Q_LORA + KV_LORA], kv_norm, form.eps), rope), dim=-1)


def _reference(rows, gate, weights, form, table):
    """fp64 output ``[Q, heads * 128]`` of the window, the last ``Q`` of one sequence's qkv_a
    ``rows``, and its latent rows; the earlier rows' latent as the bf16 cache holds it."""
    q_norm, kv_norm, q_b, kv_b = weights
    x, q = rows.double(), form.q
    n = rows.shape[0] - q
    heads = kv_b.shape[0] // (NOPE + V_DIM)
    query = (_rms_norm(x[n:, :Q_LORA], q_norm, form.eps) @ q_b.double().T).view(q, heads, -1)
    if not form.gated:
        query[..., NOPE:] = _rotate(query[..., NOPE:], table[n:n + q, None])
    c = _latent(rows, kv_norm, form, table)
    c[:n] = c[:n].bfloat16().double()
    causal = torch.arange(n + q)[None, :] <= n + torch.arange(q)[:, None]
    kv_b = kv_b.double().view(heads, NOPE + V_DIM, KV_LORA)
    out = []
    for h in range(heads):  # head by head: the long context's keys of all heads do not fit
        kv = c[:, :KV_LORA] @ kv_b[h].T
        key = torch.cat((kv[:, :NOPE], c[:, KV_LORA:]), dim=-1)
        scores = (query[:, h] @ key.T * SCALE).masked_fill(~causal, float("-inf"))
        out.append(F.softmax(scores, dim=-1) @ kv[:, NOPE:])
    out = torch.stack(out, dim=1).reshape(q, -1)
    return (out * torch.sigmoid(gate.double()) if form.gated else out), c[n:]


def _shard(rows, gate, weights, ranks, form, table, gen):
    """Each rank's call for sequences whose qkv_a rows are ``rows``, the last ``Q`` the window:
    its own blocks in a NaN-filled page buffer, block 0 the null block, and each window row's
    slot on its owner rank."""
    q_norm, kv_norm, q_b, kv_b = weights
    q = form.q
    lens = [r.shape[0] - q for r in rows]
    latents = [_latent(r[:-q], kv_norm, form, table).bfloat16() for r in rows]
    window = torch.cat([r[-q:] for r in rows]).float()
    positions = torch.tensor([n + i for n in lens for i in range(q)], dtype=torch.int32)
    rope = None if form.gated else table[:max(lens) + q].float()
    width = HEADS * (NOPE + ROPE), HEADS * (NOPE + V_DIM), HEADS * V_DIM
    calls = []
    for rank in range(ranks):
        mine = [[j for j in range((n + q - 1) // BLOCK + 1) if j % ranks == rank] for n in lens]
        count = 1 + sum(map(len, mine))
        ids = (1 + torch.randperm(count - 1, generator=gen)).tolist()
        cache = torch.full((count, BLOCK, LATENT), float("nan"), dtype=torch.bfloat16)
        table_ids = torch.full((len(rows), max(1, max(map(len, mine)))), -1, dtype=torch.int32)
        held, slots, owner = [], [], []
        for s, (n, blocks) in enumerate(zip(lens, mine)):
            block_ids = {}
            for k, j in enumerate(blocks):
                table_ids[s, k] = block_ids[j] = ids.pop(0)
                part = latents[s][j * BLOCK:(j + 1) * BLOCK]
                cache[block_ids[j], :part.shape[0]] = part
            held.append(sum(max(0, min(BLOCK, n - j * BLOCK)) for j in blocks))
            for i in range(q):
                j = (n + i) // BLOCK
                owner.append(int(j % ranks == rank))
                slots.append(block_ids[j] * BLOCK + (n + i) % BLOCK if owner[-1] else -1)
        q_rows = slice(rank * width[0], (rank + 1) * width[0])
        kv_rows = slice(rank * width[1], (rank + 1) * width[1])
        cols = slice(rank * width[2], (rank + 1) * width[2])
        calls.append((window, gate[:, cols].contiguous() if form.gated else None, q_norm, kv_norm,
                      *prepare_mla_weights(q_b[q_rows], kv_b[kv_rows]),
                      cache.view(torch.int8).reshape(count, -1), 0, BLOCK, table_ids,
                      torch.tensor(held, dtype=torch.int32), torch.tensor(slots, dtype=torch.int32),
                      torch.tensor(owner, dtype=torch.int32).view(len(rows), q), form.eps, rope,
                      None if form.gated else positions))
    return calls


def _window(call, group):
    """One rank's window call: the wrapper's positional inputs, then the rotary."""
    return mla_paged_window_dcp(*call[:15], dcp_group=group, rope_table=call[15],
                                positions=call[16])


def _read(call, out, pages):
    """A call's output ``[B Q, 384]`` and the latent rows at its slots."""
    assert tuple(out.shape) == (V_DIM, call[0].shape[0], HEADS, 1)
    assert out.dtype == torch.bfloat16
    slots = call[12].long().clamp(min=0)
    rows = pages.view(torch.bfloat16).reshape(pages.shape[0], -1, LATENT)
    return (out[..., 0].permute(1, 2, 0).reshape(out.shape[1], HEADS * V_DIM),
            rows[slots // BLOCK, slots % BLOCK])


def _rank(rank, root, ranks):
    """One gloo rank: each call's output and window rows."""
    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv", rank=rank, world_size=ranks)
    group = SimpleNamespace(ranks=range(ranks))
    results = [_read(call, *_window(call, group)) for call in torch.load(root / f"calls{rank}.pt")]
    torch.save(results, root / f"out{rank}.pt")
    dist.destroy_process_group()


def _visible_cores():
    """The number of cores ``NEURON_RT_VISIBLE_CORES`` names, as a list or a range."""
    cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    if "," in cores:
        return len(cores.split(","))
    if "-" in cores:
        first, last = cores.split("-")
        return int(last) - int(first) + 1
    return 1 if cores else 0


def _by_value(arg):
    """A tensor as a numpy array and its dtype name, so the worker queue pickles it by value."""
    if not isinstance(arg, torch.Tensor):
        return arg
    raw = arg.view(torch.int16) if arg.dtype == torch.bfloat16 else arg
    return raw.numpy(), str(arg.dtype).removeprefix("torch.")


def _on_device(arg):
    """A tensor sent by value, rebuilt on this worker's device."""
    if not isinstance(arg, tuple):
        return arg
    raw, dtype = arg
    return torch.from_numpy(raw).view(getattr(torch, dtype)).to("neuron:0")


def _compiled_window():
    """The window wrapper under ``torch.compile`` over this worker's DCP KV group."""
    from vllm_neuron.parallel.neuron_parallel_state import get_neuron_dcp_kv_group

    group = SimpleNamespace(ranks=tuple(get_neuron_dcp_kv_group().ranks))
    compiled = torch.compile(lambda *call: _window(call, group),
                             backend=envs.get_compile_backend_name(), fullgraph=True)
    return lambda *args: compiled(*map(_on_device, args))


def _run_on_device(ranks, calls):
    """Run each call on the first ``ranks`` visible cores, one worker per rank."""
    if _visible_cores() < ranks:
        pytest.skip(f"a group of {ranks} ranks needs {ranks} cores in NEURON_RT_VISIBLE_CORES")
    from vllm_neuron.utils.executor import MPExecutor

    executor = MPExecutor(world_size=ranks, model_load=_compiled_window, dcp_size=ranks,
                          load_timeout=3600)
    try:
        results = []
        for call in calls:
            executor.dispatch_per_rank([tuple(map(_by_value, c)) for c in call])
            outs = executor.collect(timeout=3600)
            results.append([_read(c, *o) for c, o in zip(call, outs)])
        return results
    finally:
        executor.shutdown()


def _run(root, ranks, calls):
    """Run each rank's calls over the group; per call, every rank's outputs and window rows."""
    if not envs.VLLM_NEURON_CPU_MODE:
        return _run_on_device(ranks, calls)
    for rank in range(ranks):
        torch.save([c[rank] for c in calls], root / f"calls{rank}.pt")
    (root / "rdzv").unlink(missing_ok=True)
    mp.spawn(_rank, args=(root, ranks), nprocs=ranks, join=True)
    results = [torch.load(root / f"out{rank}.pt") for rank in range(ranks)]
    return [[r[i] for r in results] for i in range(len(calls))]


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


def _check(rows, gate, weights, ranks, form, table, result):
    """Every rank's heads and each window row's latent on its owner meet the oracle."""
    q = form.q
    for s, r in enumerate(rows):
        n = r.shape[0] - q
        ref_out, ref_latent = _reference(r, gate[s * q:(s + 1) * q] if form.gated else None,
                                         weights, form, table)
        for rank, (out, latent) in enumerate(result):
            cols = slice(rank * HEADS * V_DIM, (rank + 1) * HEADS * V_DIM)
            _assert_within_bf16_rule(out[s * q:(s + 1) * q], ref_out[:, cols])
            for i in range(q):
                if ((n + i) // BLOCK) % ranks == rank:
                    _assert_within_bf16_rule(latent[s * q + i:s * q + i + 1], ref_latent[i:i + 1])


def _correlated(gen, length):
    """Rows 0.9 u + 0.1 noise around one shared u, one latent channel 100x the rest."""
    u = torch.randn(QKV, generator=gen)
    x = 0.9 * u + 0.1 * torch.randn(length, QKV, generator=gen)
    x[:, Q_LORA + 100] *= 100.0
    return x


def _gate(gen, rows, ranks, form):
    if not form.gated:
        return None
    return torch.randn(rows, HEADS * ranks * V_DIM, generator=gen).bfloat16()


def _table(rows, form):
    return None if form.gated else _yarn_table(max(r.shape[0] for r in rows))


def _graded(tmp_path, gen, ranks, rows, gate, form):
    """Run the windows of ``rows`` over the group and check them against the oracle."""
    table = _table(rows, form)
    weights = _weights(gen, ranks)
    calls = _shard(rows, gate, weights, ranks, form, table, gen)
    (result,) = _run(tmp_path, ranks, [calls])
    _check(rows, gate, weights, ranks, form, table, result)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("ranks, lens", [(4, (8192, 5000, 1, 2700)),
                                         (16, tuple(8192 - 397 * s for s in range(15)) + (300,))],
                         ids=["4", "16"])
def test_correlated_inputs(tmp_path, ranks, lens, form):
    """At 4 ranks, 4 sequences to 8K, merged by all-to-all; at 16, 16 sequences in two row
    groups, merged by an LSE all-gather and a reduce-scatter."""
    form = FORMS[form]
    gen = torch.Generator().manual_seed(20261027)
    rows = [_correlated(gen, n + form.q) for n in lens]
    _graded(tmp_path, gen, ranks, rows, _gate(gen, len(lens) * form.q, ranks, form), form)


@pytest.mark.parametrize("form", FORMS)
def test_repeated_tokens(tmp_path, form):
    """16 sequences of one repeated row at 4 ranks; every window query is the same."""
    form = FORMS[form]
    gen = torch.Generator().manual_seed(20261027)
    ranks, token = 4, torch.randn(1, QKV, generator=gen)
    rows = [token.expand(2049 + form.q, -1) for _ in range(16)]
    gate = _gate(gen, 1, ranks, form)
    _graded(tmp_path, gen, ranks, rows, None if gate is None else gate.expand(16 * form.q, -1),
            form)


def test_near_one_gates(tmp_path):
    """One 32K sequence at 4 ranks, its keys split between the two programs of each rank."""
    form = FORMS["verify"]
    gen = torch.Generator().manual_seed(20261027)
    ranks = 4
    rows = [torch.randn(32768 + form.q, QKV, generator=gen)]
    sign = torch.rand(form.q, HEADS * ranks * V_DIM, generator=gen) < 0.5
    gate = torch.where(sign, -12.0, 12.0).bfloat16()
    _graded(tmp_path, gen, ranks, rows, gate, form)


@pytest.mark.parametrize("form", FORMS)
def test_long_context(tmp_path, form):
    """One sequence of 131,072 cached rows at 16 ranks: its keys split between the programs and
    its query rows in two row groups."""
    form = FORMS[form]
    gen = torch.Generator().manual_seed(20261027)
    ranks = 16
    rows = [torch.randn(131072 + form.q, QKV, generator=gen)]
    _graded(tmp_path, gen, ranks, rows, _gate(gen, form.q, ranks, form), form)


@pytest.mark.parametrize("value", ["nan", "inf"])
@pytest.mark.parametrize("form", FORMS)
def test_nonfinite_window_row(tmp_path, form, value):
    """A non-finite window row, row 2 of sequence 2 of 5: its sequence's queries 2 and later are
    NaN on every rank, its latent row is non-finite on its owner, and every other query and
    latent row equals the clean run's."""
    form = FORMS[form]
    gen = torch.Generator().manual_seed(20261027)
    ranks, lens, bad, j, q = 4, (900, 1200, 640, 33, 2000), 2, 2, form.q
    rows = [torch.randn(n + q, QKV, generator=gen) for n in lens]
    gate = _gate(gen, len(lens) * q, ranks, form)
    table = _table(rows, form)
    weights = _weights(gen, ranks)
    clean = _shard(rows, gate, weights, ranks, form, table, gen)
    poisoned = []
    for call in clean:
        window, call_gate = call[0].clone(), call[1]
        window[bad * q + j] = float(value)
        if form.gated:
            call_gate = call_gate.clone()
            call_gate[bad * q + j] = float(value)
        poisoned.append((window, call_gate) + call[2:7] + (call[7].clone(),) + call[8:])
    clean_result, result = _run(tmp_path, ranks, [clean, poisoned])
    later = torch.arange(bad * q + j, (bad + 1) * q)
    kept = torch.tensor([t for t in range(len(lens) * q) if t not in later.tolist()],
                        dtype=torch.long)
    owner = ((lens[bad] + j) // BLOCK) % ranks
    for rank, ((out, latent), (clean_out, clean_latent)) in enumerate(zip(result, clean_result)):
        assert torch.isnan(out[later]).all()
        assert torch.equal(out[kept], clean_out[kept])
        # A window row's slot names the null block on every rank but its owner.
        owned = torch.tensor([s * q + i for s, n in enumerate(lens) for i in range(q)
                              if ((n + i) // BLOCK) % ranks == rank], dtype=torch.long)
        assert torch.equal(latent[owned[owned != bad * q + j]],
                           clean_latent[owned[owned != bad * q + j]])
        if rank == owner:
            assert not torch.isfinite(latent[bad * q + j]).any()
    _check(rows, gate, weights, ranks, form, table, clean_result)


@pytest.mark.parametrize("form", FORMS)
def test_ranks_with_no_key(tmp_path, form):
    """At 16 ranks, sequences of 37 and 254 cached rows: 15 ranks hold no key of the first, and
    rank 1 holds only the second's window rows from position 256, so its first two queries
    have no key there; then the 37-row sequence alone, in two row groups split over the
    programs."""
    form = FORMS[form]
    gen = torch.Generator().manual_seed(20261027)
    ranks = 16
    rows = [torch.randn(n + form.q, QKV, generator=gen) for n in (37, 254)]
    gate = _gate(gen, 2 * form.q, ranks, form)
    table = _table(rows, form)
    weights = _weights(gen, ranks)
    both = _shard(rows, gate, weights, ranks, form, table, gen)
    alone = _shard(rows[:1], None if gate is None else gate[:form.q], weights, ranks, form, table,
                   gen)
    result_both, result_alone = _run(tmp_path, ranks, [both, alone])
    _check(rows, gate, weights, ranks, form, table, result_both)
    _check(rows[:1], None if gate is None else gate[:form.q], weights, ranks, form, table,
           result_alone)


@pytest.mark.parametrize("form", FORMS)
def test_window_crosses_block(tmp_path, form):
    """At 4 ranks, four windows that each cross a block boundary, so each window's keys lie on
    two ranks; the last one wraps from rank 3 to rank 0."""
    form = FORMS[form]
    gen = torch.Generator().manual_seed(20261027)
    ranks, lens = 4, (254, 510, 1022, 3070)
    rows = [torch.randn(n + form.q, QKV, generator=gen) for n in lens]
    _graded(tmp_path, gen, ranks, rows, _gate(gen, len(lens) * form.q, ranks, form), form)
