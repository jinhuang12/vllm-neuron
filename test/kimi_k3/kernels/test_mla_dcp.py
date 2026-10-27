# SPDX-License-Identifier: Apache-2.0
"""fp64 oracle test for the paged MLA decode kernel in DCP mode, one item per numerics regime.

A DCP group of ``d`` ranks splits each sequence's cache by block: block ``j`` lies on rank
``j mod d``, so each rank's table lists its own blocks and its length counts its own rows.
The new row is a key on its owner rank only, and its slot is -1 on the other ranks. Each
rank holds 3 heads, and the group's ``3 d`` heads see the whole cache through the merge.
The oracle transcribes ``KimiMLAAttention.forward`` from the checkpoint's
``modeling_kimi_linear.py`` to fp64 over the ``3 d`` heads and the whole sequence: the two
RMS norms, ``q_b`` and ``kv_b`` unabsorbed, NoPE attention and the sigmoid output gate; it
reads the cached rows as bf16 and the new row unrounded. Each rank's output is checked on
its own heads, and the owner's new latent row at its slot. Pass rule, per row: every
element within 2^-7 of its value plus 2^-7 of the row RMS, and row cosine at least 0.9999.

Each item runs the group as one process per rank. With ``VLLM_NEURON_CPU_MODE=1
NKI_SIMULATOR=1`` the ranks are gloo processes on the CPU and the kernel runs in the NKI
simulator, which carries the kernel's collectives over gloo. On a Neuron device the group runs
through ``MPExecutor`` on the first ``d`` cores of ``NEURON_RT_VISIBLE_CORES``, each rank's
call under ``torch.compile``; a case whose group needs more cores than that is skipped.
"""

import functools
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from vllm_neuron import envs
from vllm_neuron.functional.attention.mla_paged import mla_paged_decode_dcp, prepare_mla_weights

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


def _weights(gen, ranks):
    """Norm gains and the group's ``q_b`` and ``kv_b`` in bf16, at the checkpoint's scale."""
    heads = HEADS * ranks
    q_norm = (0.65 + 0.03 * torch.randn(Q_LORA, generator=gen)).bfloat16()
    kv_norm = (0.17 + 0.012 * torch.randn(KV_LORA, generator=gen)).bfloat16()
    q_b = (0.03 * torch.randn(heads * (NOPE + ROPE), Q_LORA, generator=gen)).bfloat16()
    kv_b = (0.027 * torch.randn(heads * (NOPE + V_DIM), KV_LORA, generator=gen)).bfloat16()
    return q_norm, kv_norm, q_b, kv_b


def _rms_norm(x, weight):
    """The checkpoint's ``KimiRMSNorm`` in fp64."""
    return weight.double() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))


def _latent(rows, kv_norm):
    """fp64 latent rows ``[norm(kv_a[:512]), kv_a[512:]]`` of qkv_a rows."""
    x = rows.double()
    return torch.cat((_rms_norm(x[:, Q_LORA:Q_LORA + KV_LORA], kv_norm), x[:, Q_LORA + KV_LORA:]),
                     dim=-1)


def _reference(rows, gate, weights):
    """fp64 output ``[1, heads * 128]`` of the last of one sequence's qkv_a ``rows`` and its
    latent row; the earlier rows' latent as the bf16 cache holds it."""
    q_norm, kv_norm, q_b, kv_b = weights
    x = rows.double()
    heads = kv_b.shape[0] // (NOPE + V_DIM)
    q = (_rms_norm(x[-1:, :Q_LORA], q_norm) @ q_b.double().T).view(heads, NOPE + ROPE)
    c = _latent(rows, kv_norm)
    c[:-1] = c[:-1].bfloat16().double()
    kv_b = kv_b.double().view(heads, NOPE + V_DIM, KV_LORA)
    out = []
    for h in range(heads):  # head by head: the long context's keys of all heads do not fit
        kv = c[:, :KV_LORA] @ kv_b[h].T
        key = torch.cat((kv[:, :NOPE], c[:, KV_LORA:]), dim=-1)
        probs = F.softmax(key @ q[h] * SCALE, dim=0)
        out.append(probs @ kv[:, NOPE:])
    return torch.cat(out).unsqueeze(0) * torch.sigmoid(gate.double()), c[-1:]


def _shard(rows, gate, weights, ranks, block_size, gen):
    """Each rank's call for sequences whose qkv_a rows are ``rows``, the last one new: its own
    blocks in a NaN-filled page buffer, block 0 the null block, and the new row's slot on its
    owner rank."""
    q_norm, kv_norm, q_b, kv_b = weights
    lens = [r.shape[0] - 1 for r in rows]
    latents = [_latent(r[:-1], kv_norm).bfloat16() for r in rows]
    new = torch.stack([r[-1] for r in rows]).float()
    width = HEADS * (NOPE + ROPE), HEADS * (NOPE + V_DIM), HEADS * V_DIM
    calls = []
    for rank in range(ranks):
        mine = [[j for j in range(math.ceil(n / block_size)) if j % ranks == rank] for n in lens]
        owner = [(n // block_size) % ranks == rank for n in lens]
        fresh = [o and n % block_size == 0 for o, n in zip(owner, lens)]
        count = 1 + sum(map(len, mine)) + sum(fresh)
        ids = (1 + torch.randperm(count - 1, generator=gen)).tolist()
        cache = torch.full((count, block_size, LATENT), float("nan"), dtype=torch.bfloat16)
        table = torch.full((len(rows), max(1, max(map(len, mine)))), -1, dtype=torch.int32)
        held, slots = [], []
        for s, blocks in enumerate(mine):
            for k, j in enumerate(blocks):
                table[s, k] = ids.pop(0)
                part = latents[s][j * block_size:(j + 1) * block_size]
                cache[int(table[s, k]), :part.shape[0]] = part
            held.append(sum(min(block_size, lens[s] - j * block_size) for j in blocks))
            if not owner[s]:
                slots.append(-1)
                continue
            block = ids.pop(0) if fresh[s] else int(table[s, len(blocks) - 1])
            slots.append(block * block_size + lens[s] % block_size)
        q_rows = slice(rank * width[0], (rank + 1) * width[0])
        kv_rows = slice(rank * width[1], (rank + 1) * width[1])
        calls.append((new, gate[:, rank * width[2]:(rank + 1) * width[2]].contiguous(), q_norm,
                      kv_norm, *prepare_mla_weights(q_b[q_rows], kv_b[kv_rows]),
                      cache.view(torch.int8).reshape(count, -1), 0, block_size, table,
                      torch.tensor(held, dtype=torch.int32), torch.tensor(slots, dtype=torch.int32),
                      torch.tensor(owner, dtype=torch.int32), EPS))
    return calls


def _read(call, out, pages):
    """A call's output ``[T, 384]`` and the latent rows at its slots."""
    assert tuple(out.shape) == (V_DIM, call[0].shape[0], HEADS, 1)
    assert out.dtype == torch.bfloat16
    slots, block_size = call[12].long().clamp(min=0), call[9]
    rows = pages.view(torch.bfloat16).reshape(pages.shape[0], -1, LATENT)
    return (out[..., 0].permute(1, 2, 0).reshape(out.shape[1], HEADS * V_DIM),
            rows[slots // block_size, slots % block_size])


def _rank(rank, root, ranks):
    """One gloo rank: each call's output and new rows."""
    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv", rank=rank, world_size=ranks)
    group = SimpleNamespace(ranks=range(ranks))
    results = [_read(call, *mla_paged_decode_dcp(*call, dcp_group=group))
               for call in torch.load(root / f"calls{rank}.pt")]
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


def _compiled_dcp():
    """The DCP wrapper under ``torch.compile`` over this worker's DCP KV group."""
    from vllm_neuron.parallel.neuron_parallel_state import get_neuron_dcp_kv_group

    group = SimpleNamespace(ranks=tuple(get_neuron_dcp_kv_group().ranks))
    compiled = torch.compile(functools.partial(mla_paged_decode_dcp, dcp_group=group),
                             backend=envs.get_compile_backend_name(), fullgraph=True)
    return lambda *args: compiled(*map(_on_device, args))


def _run_on_device(ranks, calls):
    """Run each call on the first ``ranks`` visible cores, one worker per rank."""
    if _visible_cores() < ranks:
        pytest.skip(f"a group of {ranks} ranks needs {ranks} cores in NEURON_RT_VISIBLE_CORES")
    from vllm_neuron.utils.executor import MPExecutor

    executor = MPExecutor(world_size=ranks, model_load=_compiled_dcp, dcp_size=ranks,
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
    """Run each rank's calls over the group; per call, every rank's outputs and new rows."""
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


def _check(rows, gate, weights, ranks, block_size, result):
    """Every rank's heads and the owner's new latent row meet the oracle, per sequence."""
    lens = [r.shape[0] - 1 for r in rows]
    for s, r in enumerate(rows):
        ref_out, ref_latent = _reference(r, gate[s:s + 1], weights)
        for rank, (out, latent) in enumerate(result):
            cols = slice(rank * HEADS * V_DIM, (rank + 1) * HEADS * V_DIM)
            _assert_within_bf16_rule(out[s:s + 1], ref_out[:, cols])
            if (lens[s] // block_size) % ranks == rank:
                _assert_within_bf16_rule(latent[s:s + 1], ref_latent)


def _correlated(gen, length):
    """Rows 0.9 u + 0.1 noise around one shared u, one latent channel 100x the rest."""
    u = torch.randn(QKV, generator=gen)
    x = 0.9 * u + 0.1 * torch.randn(length, QKV, generator=gen)
    x[:, Q_LORA + 100] *= 100.0
    return x


def _case(gen, ranks, rows, gate, block_size=256):
    weights = _weights(gen, ranks)
    return weights, _shard(rows, gate, weights, ranks, block_size, gen)


@pytest.mark.parametrize("ranks, lens", [(4, (8192, 5000, 1, 2700)),
                                         (16, tuple(8192 - 397 * s for s in range(15)) + (300,))],
                         ids=["4", "16"])
def test_correlated_inputs(tmp_path, ranks, lens):
    """At 4 ranks, 4 sequences to 8K, merged by all-to-all; at 16, 16 sequences, merged by an
    LSE all-gather and a reduce-scatter."""
    gen = torch.Generator().manual_seed(20261025)
    rows = [_correlated(gen, n + 1) for n in lens]
    gate = torch.randn(len(lens), HEADS * ranks * V_DIM, generator=gen).bfloat16()
    weights, calls = _case(gen, ranks, rows, gate)
    (result,) = _run(tmp_path, ranks, [calls])
    _check(rows, gate, weights, ranks, 256, result)


def test_repeated_tokens(tmp_path):
    gen = torch.Generator().manual_seed(20261025)
    ranks, token = 4, torch.randn(1, QKV, generator=gen)
    rows = [token.expand(2049, -1) for _ in range(16)]
    gate = torch.randn(1, HEADS * ranks * V_DIM, generator=gen).expand(16, -1).bfloat16()
    weights, calls = _case(gen, ranks, rows, gate)
    (result,) = _run(tmp_path, ranks, [calls])
    _check(rows, gate, weights, ranks, 256, result)


def test_near_one_gates(tmp_path):
    """One 32K sequence at 4 ranks, its keys split between the two programs of each rank."""
    gen = torch.Generator().manual_seed(20261025)
    ranks = 4
    rows = [torch.randn(32768 + 1, QKV, generator=gen)]
    sign = torch.where(torch.rand(1, HEADS * ranks * V_DIM, generator=gen) < 0.5, -12.0, 12.0)
    weights, calls = _case(gen, ranks, rows, sign.bfloat16())
    (result,) = _run(tmp_path, ranks, [calls])
    _check(rows, sign.bfloat16(), weights, ranks, 256, result)


@pytest.mark.parametrize("ranks", [16, 32])
def test_long_context(tmp_path, ranks):
    gen = torch.Generator().manual_seed(20261025)
    rows = [torch.randn(131072, QKV, generator=gen)]
    gate = torch.randn(1, HEADS * ranks * V_DIM, generator=gen).bfloat16()
    weights, calls = _case(gen, ranks, rows, gate)
    (result,) = _run(tmp_path, ranks, [calls])
    _check(rows, gate, weights, ranks, 256, result)


def test_nan_row_stays_in_its_sequence(tmp_path):
    """A NaN new row and gate in sequence 2 of 5: its heads are NaN on every rank, its latent
    row NaN on its owner, and every other sequence's output and row equal the clean run's."""
    gen = torch.Generator().manual_seed(20261025)
    ranks, lens, bad = 4, (900, 1200, 640, 33, 2000), 2
    rows = [torch.randn(n + 1, QKV, generator=gen) for n in lens]
    gate = torch.randn(len(lens), HEADS * ranks * V_DIM, generator=gen).bfloat16()
    weights, clean = _case(gen, ranks, rows, gate)
    poisoned = []
    for call in clean:
        new, nan_gate = call[0].clone(), call[1].clone()
        new[bad], nan_gate[bad] = float("nan"), float("nan")
        poisoned.append((new, nan_gate) + call[2:7] + (call[7].clone(),) + call[8:])
    clean_result, result = _run(tmp_path, ranks, [clean, poisoned])
    others = torch.tensor([s for s in range(len(lens)) if s != bad])
    for rank, ((out, latent), (clean_out, clean_latent)) in enumerate(zip(result, clean_result)):
        assert torch.isnan(out[bad]).all()
        assert torch.equal(out[others], clean_out[others])
        # A row's slot names the null block on every rank but its owner.
        owned = torch.tensor([s for s in range(len(lens)) if (lens[s] // 256) % ranks == rank],
                             dtype=torch.long)
        kept = owned[owned != bad]
        assert torch.equal(latent[kept], clean_latent[kept])
        if bad in owned:
            assert torch.isnan(latent[bad]).all()
    _check(rows, gate, weights, ranks, 256, clean_result)


def test_ranks_with_no_key(tmp_path):
    """At 16 ranks, sequences of 37 and 256 cached tokens: 15 and 14 ranks hold no key, and
    the 256-token sequence's new row is the only key on rank 1; then the 37-token one alone."""
    gen = torch.Generator().manual_seed(20261025)
    ranks = 16
    rows = [torch.randn(n + 1, QKV, generator=gen) for n in (37, 256)]
    gate = torch.randn(2, HEADS * ranks * V_DIM, generator=gen).bfloat16()
    weights = _weights(gen, ranks)
    both = _shard(rows, gate, weights, ranks, 256, gen)
    alone = _shard(rows[:1], gate[:1], weights, ranks, 256, gen)
    result_both, result_alone = _run(tmp_path, ranks, [both, alone])
    _check(rows, gate, weights, ranks, 256, result_both)
    _check(rows[:1], gate[:1], weights, ranks, 256, result_alone)
