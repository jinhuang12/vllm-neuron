# SPDX-License-Identifier: Apache-2.0
"""Per-chunk device time of the kernels an 8k-line chunk runs and a 1k-line chunk does not.

Run it ONLY through the device lease, which pins the cores and sets LNC2; the script refuses
to run without them and selects no cores itself::

    export NEURON_LIBTORCH_CACHE_ROOT=<an empty directory>
    python3 /home/ubuntu/glm53f-wt/devlease.py slice <slice> -- python3 \\
        test/hardware/benchmark_mla_sparse_8k.py --base-module <base>/mla_sparse.py \\
        --out bench.json

``--base-module`` is the base revision's ``vllm_neuron/functional/attention/mla_sparse.py``;
it is loaded under its own module name, so both sparse kernels run in one process and one
lease (:mod:`benchmark_sparse_mla_prefill` does the same).

The 8k line serves an 8000-token prompt in chunks of :data:`CHUNK` rows (the last one 832
real rows, padded to 1024) through ONE prefill graph, KV segment :data:`SEGMENT_8K`. Its
shapes are fixed by that segment, so every chunk ``c = 1..8`` runs the same calls; only the
data (each row's causal length, the selected rows) moves with ``c``. Against the 1k line's
graph (KV segment 1024, the dense-window path), the 8k graph replaces the dense-window
kernel with the sparse kernel and adds the DSA indexer's query side. Each of those kernels is
measured here at its served per-rank (TP=64) shape, on the data of chunk ``c``:

* ``sparse``: ``mla_sparse_attention`` (base and this tree, interleaved) at the 8k line's
  chunks ``c1 .. c8``, and at the other served shapes the rebased kernel must hold: the std
  line's KV 2048 and 4096 graphs, the 64k line's first and last chunk, and the two one-row
  decode graphs. Every output is checked against a float64 oracle within the low-precision
  body's derived error budget (``benchmark_sparse_mla_prefill.error_budget``), and every
  timed emission against the variant's first output, bit for bit.
* ``dense``: ``mla_dense_window_attention`` at the 1k line's chunk (KV segment 1024), the
  kernel the 8k graph does not run; checked against its float64 oracle.
* ``indexer``: the eleven 8k-only indexer kernels (:data:`INDEXER`), each called through its
  public seam as ``model_fp8.py`` calls it, 1 and :data:`LAYERS` times per graph on distinct
  operands. The five whose operands carry the chunk's causal lengths run on each chunk's data,
  built by the seams' own torch routes on the CPU; the six without such an operand run once.
* ``floor``: a one-element add, the fixed device cost of one execution.

``--part``, ``--case`` and ``--group`` select one piece, so a lease script can run each piece
in its own process and read the device's fault log in between.

A sample is one execution with its outputs copied to CPU; its device time is the runtime
system trace's ``nc_exec_running`` interval with the two physical cores merged. Each repeat
runs every graph of a group ``--iterations`` times, the call order rotated each call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

#: The worktree root, ahead of any installed copy (the lease sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import benchmark_sparse_mla_prefill as sparse_bench
from benchmark_sparse_mla_prefill import (
    HEADS,
    KPOOL,
    LATENT,
    LEASE_MARKER,
    PAGE,
    SCALE,
    SELECT_K,
    Case,
)

#: Rows per prefill chunk, and the 8k line's prompt (gate_stack2.md section 1).
CHUNK, PROMPT_8K = 1024, 8000
#: The 8k line's one KV segment (b64 serve: ``max_model_len`` 8192, ``kv_segment_size_buckets
#: [8192]``) and the std line's ``max_model_len`` (KV segments 1024, 2048, 4096).
SEGMENT_8K, STD_MAX_MODEL_LEN = 8192, 4096
#: Each rank's latent bank, in pages: the served graphs' ``c_kv`` operand rows / PAGE
#: (b64 graph 89aba8de: bf16[557184, 512]; std graphs: bf16[4736, 512]).
BANK_PAGES_B64, BANK_PAGES_STD = 557184 // PAGE, 4736 // PAGE
#: DSA (MLA) layers per chunk: the served graph calls each kernel below this many times.
LAYERS = 11
#: TP degree; the indexer shards a chunk's rows over the ranks, and the latest rank in the
#: gate's per-chunk completion lines is a top rank, so rank ``RANKS - 1``'s rows are used.
RANKS = 64
RANK = RANKS - 1
RANK_ROWS = CHUNK // RANKS
#: Indexer geometry at GLM-5.3-Flash: 32 heads of 128, ``q_lora_rank`` 1536, hidden 4096.
INDEX_HEADS, INDEX_DIM, Q_LORA, HIDDEN = 32, 128, 1536, 4096
#: Candidate pools at the 8k segment, and the rows of the pooled-key cache the candidate
#: gather reads (the served ``_paged_gather_nki`` operand bf16[2049, 128]).
CANDIDATES = SEGMENT_8K // KPOOL
POOL_CACHE_ROWS = CANDIDATES + 1
#: The dense-window kernel's bar against float64, as benchmark_mla_dense_window.py holds it.
DENSE_ROW_REL = 1e-4


def window_pages(segment: int, max_model_len: int) -> int:
    """Pages of a prefill graph's block-table row: the segment plus one chunk, capped at the
    model length (the served tables: kv1024 16, kv2048 24, kv4096 32, b64 kv8192 64)."""
    return min(segment + CHUNK, max_model_len) // PAGE


def chunk_start(c: int) -> int:
    """Tokens before chunk ``c`` (1-based) of the 8k line's prompt."""
    return (c - 1) * CHUNK


def chunk_rows(c: int) -> int:
    """Real rows of chunk ``c``: CHUNK, the last one the rest of the prompt."""
    return min(CHUNK, PROMPT_8K - chunk_start(c))


N_CHUNKS = -(-PROMPT_8K // CHUNK)
WINDOW_8K = window_pages(SEGMENT_8K, SEGMENT_8K)


def line8k_case(c: int) -> Case:
    """Chunk ``c`` of the 8k line: all CHUNK rows, the padding rows included, are queries at
    ``start + t`` (the runner keeps ``seq_lens = start + t + 1`` on padded rows)."""
    start = chunk_start(c)
    return Case(f"line8k_c{c}", CHUNK, WINDOW_8K, BANK_PAGES_B64, start, start + CHUNK,
                f"8k line, chunk {c} of {N_CHUNKS} ({chunk_rows(c)} real rows), graph kv"
                f"{SEGMENT_8K}")


def std_case(segment: int) -> Case:
    """The std line's KV-``segment`` graph at its last chunk: rows at ``segment ..``, or the
    chunk that ends the model length when the window is shorter."""
    pages = window_pages(segment, STD_MAX_MODEL_LEN)
    start = min(segment, pages * PAGE - CHUNK)
    return Case(f"std_kv{segment}", CHUNK, pages, BANK_PAGES_STD, start, start + CHUNK,
                f"std line, graph kv{segment}, rows {start}..{start + CHUNK - 1}")


#: The sparse kernel's served calls: the 8k line's chunks, the std line's two sparse graphs,
#: the 64k line's first and last chunk, and one decode row on each line's decode graph.
SPARSE_CASES = (
    tuple(line8k_case(c) for c in range(1, N_CHUNKS + 1))
    + (std_case(2048), std_case(4096))
    + tuple(c for c in sparse_bench.CASES if c.name in ("p64k_first", "p64k_last"))
    + (Case("decode_b64_8k", 1, WINDOW_8K, BANK_PAGES_B64, SEGMENT_8K - 1, SEGMENT_8K,
            "decode, b64 graph, one request at context 8192"),
       Case("decode_std_4k", 1, window_pages(STD_MAX_MODEL_LEN, STD_MAX_MODEL_LEN),
            BANK_PAGES_STD, STD_MAX_MODEL_LEN - 1, STD_MAX_MODEL_LEN,
            "decode, std graph, one request at context 4096")))
#: The 1k line's chunk: KV segment 1024, rows 0 .. 1023, the dense-window kernel.
DENSE_CASE = Case("line1k_c1", CHUNK, window_pages(1024, STD_MAX_MODEL_LEN), BANK_PAGES_STD,
                  0, CHUNK, "1k line, chunk 1 of 1, graph kv1024 (dense window)")


@dataclass(frozen=True)
class IndexerKernel:
    """One 8k-only indexer kernel: its served name, and whether its operands carry ``c``."""

    name: str
    kernel: str
    per_chunk: bool


INDEXER = (
    IndexerKernel("wq_b", "mla_projection_lowp_kernel (x fp32[1024,1536], w bf16[1536,4096])",
                  False),
    IndexerKernel("weights_proj",
                  "mla_projection_lowp_kernel (x bf16[1024,4096], w bf16[4096,32])", False),
    IndexerKernel("hadamard128", "_hadamard128_nki (bf16[32768,128])", False),
    IndexerKernel("take_rank_rows", "_take_rank_rows_nki (16 rows of 3 sources)", False),
    IndexerKernel("paged_gather", "_paged_gather_nki (2048 pooled keys)", False),
    IndexerKernel("score_gemm", "_score_gemm_nki (q[16,32,128] x k[2048,128])", False),
    IndexerKernel("causal_bound", "_causal_bound_nki ([16,2048])", True),
    IndexerKernel("topk", "rotational_topk ([16,2048], k=512)", True),
    IndexerKernel("causal_sentinel", "_causal_sentinel_nki ([16,512])", True),
    IndexerKernel("sentinel_order", "_sentinel_order_nki ([16,512])", True),
    IndexerKernel("index_expand", "_index_expand_nki ([1024,512] -> [1024,2176])", True),
)
#: The indexer's timing groups: the kernels without a chunk operand, and each chunk's.
INDEXER_GROUPS = ("fixed",) + tuple(f"c{c}" for c in range(1, N_CHUNKS + 1))


def indexer_seams():
    """``{name: callable}``: each kernel's public seam, as ``model_fp8.py`` calls it."""
    from vllm_neuron.functional.attention.mla_projections import mla_projection_lowp
    from vllm_neuron.functional.dsa.causal_bound import (
        dsa_causal_bound,
        dsa_causal_sentinel,
    )
    from vllm_neuron.functional.dsa.index_expand import dsa_index_expand
    from vllm_neuron.functional.dsa.kpool_hadamard import dsa_hadamard128
    from vllm_neuron.functional.dsa.paged_gather import dsa_paged_gather
    from vllm_neuron.functional.dsa.score_gemm import dsa_score_gemm
    from vllm_neuron.functional.dsa.sentinel_order import dsa_sentinel_order
    from vllm_neuron.functional.dsa.shard_rows import dsa_take_rank_rows
    from vllm_neuron.functional.dsa.topk_select import dsa_topk_select

    return {
        "wq_b": lambda x, w: mla_projection_lowp(x, w),
        "weights_proj": lambda x, w: mla_projection_lowp(x, w),
        "hadamard128": dsa_hadamard128,
        "take_rank_rows": lambda q, w, lens, rank: dsa_take_rank_rows((q, w, lens), rank,
                                                                       RANK_ROWS),
        "paged_gather": lambda pages, page, slot: dsa_paged_gather(pages, page, slot, PAGE),
        "score_gemm": dsa_score_gemm,
        "causal_bound": lambda scores, lens: dsa_causal_bound(scores, lens, KPOOL),
        "topk": lambda bounded: dsa_topk_select(bounded, SELECT_K),
        "causal_sentinel": lambda values, ids: dsa_causal_sentinel(values, ids, CANDIDATES),
        "sentinel_order": dsa_sentinel_order,
        "index_expand": lambda ids, lens: dsa_index_expand(ids, lens, KPOOL),
    }


def selection_chain(c: int, gen):
    """One layer's indexer operands at chunk ``c``, built on the CPU by the seams' references.

    Random fp32 scores for every row of the chunk over the :data:`CANDIDATES` pools, bounded
    by each row's causal length, the top :data:`SELECT_K` (``torch.topk``, the selector's
    reference), the causal sentinel, and the sentinel order (real ids first, both groups in
    their order): the chain ``model_fp8.py`` runs, so each kernel sees the operand its
    predecessor would hand it at this chunk. The seams themselves take their NKI routes on
    any tensor, so their torch references are called directly.
    """
    import torch

    from vllm_neuron.functional.dsa.causal_bound import (
        dsa_causal_bound_torch_oracle,
        dsa_causal_sentinel_torch_oracle,
    )

    lens = (torch.arange(CHUNK, dtype=torch.int32) + chunk_start(c) + 1).reshape(CHUNK, 1)
    scores = torch.randn(CHUNK, CANDIDATES, generator=gen)
    bounded = dsa_causal_bound_torch_oracle(scores, lens, KPOOL)
    values, ids = torch.topk(bounded, SELECT_K, dim=-1)
    ids = ids.to(torch.int32)
    sentinel = dsa_causal_sentinel_torch_oracle(values, ids, CANDIDATES)
    order = torch.argsort((sentinel < 0).to(torch.int8), dim=1, stable=True)
    ordered = sentinel.gather(1, order)
    mine = slice(RANK * RANK_ROWS, (RANK + 1) * RANK_ROWS)
    return {
        "causal_bound": (scores[mine].contiguous(), lens[mine].contiguous()),
        "topk": (bounded[mine].contiguous(),),
        "causal_sentinel": (values[mine].contiguous(), ids[mine].contiguous()),
        "sentinel_order": (sentinel[mine].contiguous(),),
        "index_expand": (ordered.contiguous(), lens.reshape(CHUNK).contiguous()),
    }


def fixed_operands(gen):
    """One layer's operands of the six indexer kernels whose shapes and data carry no ``c``."""
    import torch

    bf16 = torch.bfloat16
    flat = torch.arange(CANDIDATES, dtype=torch.int32)
    return {
        "wq_b": (torch.randn(CHUNK, Q_LORA, generator=gen),
                 (torch.randn(Q_LORA, INDEX_HEADS * INDEX_DIM, generator=gen) * 0.03).to(bf16)),
        "weights_proj": (torch.randn(CHUNK, HIDDEN, generator=gen).to(bf16),
                         (torch.randn(HIDDEN, INDEX_HEADS, generator=gen) * 0.02).to(bf16)),
        "hadamard128": (torch.randn(CHUNK * INDEX_HEADS, INDEX_DIM, generator=gen).to(bf16),),
        "take_rank_rows": (torch.randn(CHUNK, INDEX_HEADS, INDEX_DIM, generator=gen).to(bf16),
                           torch.randn(CHUNK, INDEX_HEADS, generator=gen),
                           torch.arange(1, CHUNK + 1, dtype=torch.int32).reshape(CHUNK, 1),
                           torch.tensor([RANK], dtype=torch.int32)),
        "paged_gather": (torch.randn(POOL_CACHE_ROWS, INDEX_DIM, generator=gen).to(bf16),
                         torch.div(flat, PAGE, rounding_mode="floor"),
                         torch.remainder(flat, PAGE)),
        "score_gemm": (torch.randn(RANK_ROWS, INDEX_HEADS, INDEX_DIM, generator=gen).to(bf16),
                       torch.randn(CANDIDATES, INDEX_DIM, generator=gen).to(bf16),
                       torch.rand(RANK_ROWS, INDEX_HEADS, generator=gen) * 0.1),
    }


def leaves(value):
    """Every tensor of a nested tuple of outputs, copied to CPU (the call's synchronisation)."""
    import torch

    if isinstance(value, torch.Tensor):
        return [value.to("cpu")]
    return [leaf for item in value for leaf in leaves(item)]


#: Graphs one Python function is compiled into here (one per seam, call count and shape);
#: past dynamo's default of 8 it would run the function eagerly instead.
RECOMPILE_LIMIT = 64


def compiled(fn):
    """``fn`` as one Neuron graph; a recompile past :data:`RECOMPILE_LIMIT` raises."""
    import torch

    torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False)


def repeated(seam, calls: int, arity: int):
    """A graph of ``calls`` seam calls, each on its own ``arity`` operands of the flat input."""

    def call(*flat):
        return tuple(seam(*flat[i * arity:(i + 1) * arity]) for i in range(calls))

    return compiled(call)


def time_group(graphs: dict, inputs: dict, iterations: int, reference: dict | None = None):
    """One repeat: ``iterations`` calls of every graph, the order rotated each call.

    Returns ``({name: device us per call}, {name: [emissions, identical]})``: each emission
    is compared bit for bit with ``reference[name]`` when one is given.
    """
    import torch
    from nrtpy._nrtpy import SystemTraceSession

    names = list(graphs)
    order, identity = [], {name: [0, 0] for name in names}
    with SystemTraceSession() as trace:
        for it in range(iterations):
            first = it % len(names)
            for name in names[first:] + names[:first]:
                got = leaves(graphs[name](*inputs[name]))
                order.append(name)
                if reference is not None and name in reference:
                    identity[name][0] += 1
                    identity[name][1] += int(all(torch.equal(a, b)
                                                 for a, b in zip(got, reference[name])))
        events = trace.fetch_events_json()
    device = sparse_bench.device_intervals(events)
    if len(device) != len(order):
        raise AssertionError(f"the trace has {len(device)} executions for {len(order)} calls")
    per = {name: [] for name in names}
    for name, value in zip(order, device):
        per[name].append(value)
    return per, identity


def repeat_groups(graphs, inputs, args, reference=None):
    """``--reps`` repeats of :func:`time_group`: rep medians, their median, spread, identity."""
    for _ in range(args.warmup):
        for name, graph in graphs.items():
            leaves(graph(*inputs[name]))
    medians = {name: [] for name in graphs}
    identity = {name: [0, 0] for name in graphs}
    for _ in range(args.reps):
        per, ident = time_group(graphs, inputs, args.iterations, reference)
        for name in graphs:
            medians[name].append(statistics.median(per[name]))
            identity[name][0] += ident[name][0]
            identity[name][1] += ident[name][1]
    out = {}
    for name, values in medians.items():
        median = statistics.median(values)
        out[name] = {"rep_medians_us": values, "median_us": median,
                     "rep_spread": (max(values) - min(values)) / median}
        if reference is not None and name in reference:
            out[name]["emissions"], out[name]["identical"] = identity[name]
    return out


def to_device(tensors):
    return tuple(t.to("neuron:0") for t in tensors)


def sparse_graph(module):
    """The model's seam call of one sparse module, compiled."""
    attend = module.mla_sparse_attention

    def call(q, bank, indices, table, written, offset):
        return attend(q, bank, indices, SCALE, block_table_row=table, written=written,
                      write_offset=offset, page_size=PAGE)

    return compiled(call)


def run_sparse_case(case: Case, modules: dict, floor, args, failures: list) -> dict:
    """Numerics over ``--seeds`` operand sets, then interleaved timing with identity."""
    import torch

    torch._dynamo.reset()
    graphs = {name: sparse_graph(module) for name, module in modules.items()}
    numerics, timed, reference, first_call_s = [], None, None, {}
    for seed_index in range(args.seeds):
        seed = 1000 * seed_index + case.tokens + case.start
        q, bank, indices, table, written, offset, window = sparse_bench.make_operands(case, seed)
        inputs = to_device((q, bank, indices, table, written, offset))
        outputs = {}
        for name, graph in graphs.items():
            started = time.perf_counter()
            outputs[name] = graph(*inputs).to("cpu")
            first_call_s.setdefault(name, time.perf_counter() - started)
        want = sparse_bench.oracle(q, window, indices)
        bar = sparse_bench.operand_budget(q, window, indices)
        row = {"seed": seed, "error_budget": bar,
               "bit_equal_after_vs_base": bool(torch.equal(outputs["base"], outputs["after"])),
               "max_abs_after_vs_base": float((outputs["after"] - outputs["base"]).abs().max()),
               "rel_l2_after_vs_base": sparse_bench.rel_l2(outputs["after"], outputs["base"])}
        for name, got in outputs.items():
            row[f"rel_l2_{name}_vs_float64"] = sparse_bench.rel_l2(got, want)
            if not torch.isfinite(got).all() or row[f"rel_l2_{name}_vs_float64"] > bar["rel_l2"]:
                failures.append(f"{case.name} seed {seed}: {name} outside its error budget")
        if row["rel_l2_after_vs_base"] > bar["rel_l2"]:
            failures.append(f"{case.name} seed {seed}: after vs base outside the error budget")
        if args.save_dir is not None:
            for name, got in outputs.items():
                torch.save(got, args.save_dir / f"{case.name}_seed{seed}_{name}.pt")
        numerics.append(row)
        if timed is None:
            timed, reference = inputs, {name: [out] for name, out in outputs.items()}
    graphs["floor"] = floor["graph"]
    inputs = {name: timed for name in modules}
    inputs["floor"] = floor["inputs"]
    timing = repeat_groups(graphs, inputs, args, reference)
    for name in modules:
        if timing[name]["identical"] != timing[name]["emissions"]:
            failures.append(f"{case.name}: {name} emissions differ from its first output")
    timing["after_over_base"] = timing["after"]["median_us"] / timing["base"]["median_us"]
    timing["noise_floor"] = max(timing[name]["rep_spread"] for name in modules)
    return {"case": asdict(case), "entitlement": sparse_bench.entitlement(case),
            "timing": timing, "numerics": numerics, "first_call_s": first_call_s}


def run_dense(floor, args, failures: list) -> dict:
    """The 1k line's dense-window call: numerics against float64, then timing."""
    import benchmark_mla_dense_window as dense_bench
    import torch

    from vllm_neuron.functional.attention.mla_dense_window import (
        mla_dense_window_attention,
    )

    torch._dynamo.reset()
    case = DENSE_CASE

    def call(q, bank, seq_lens, table, written, offset):
        return mla_dense_window_attention(q, bank, seq_lens, SCALE, block_table_row=table,
                                          written=written, write_offset=offset,
                                          page_size=PAGE)

    graph = compiled(call)
    q, bank, _, table, written, offset, window = sparse_bench.make_operands(case, case.tokens)
    seq_lens = torch.arange(case.tokens, dtype=torch.int32) + case.start + 1
    inputs = to_device((q, bank, seq_lens, table, written, offset))
    started = time.perf_counter()
    out = graph(*inputs).to("cpu")
    first_call_s = time.perf_counter() - started
    error = dense_bench.row_rel(out, dense_bench.oracle(q, window, seq_lens, case.tokens),
                                case.tokens)
    if not torch.isfinite(out).all() or error > DENSE_ROW_REL:
        failures.append(f"{case.name}: dense window row rel {error} > {DENSE_ROW_REL}")
    timing = repeat_groups({"dense": graph, "floor": floor["graph"]},
                           {"dense": inputs, "floor": floor["inputs"]}, args, {"dense": [out]})
    if timing["dense"]["identical"] != timing["dense"]["emissions"]:
        failures.append(f"{case.name}: dense emissions differ from its first output")
    return {"case": asdict(case), "row_rel_vs_float64": error, "bar": DENSE_ROW_REL,
            "timing": timing, "first_call_s": first_call_s}


def indexer_group(label: str):
    """``(kernels, layers)`` of one :data:`INDEXER_GROUPS` group: its kernels and each layer's
    operands, from a generator seeded by the group alone."""
    import torch

    if label == "fixed":
        gen = torch.Generator().manual_seed(8000)
        return ([k for k in INDEXER if not k.per_chunk],
                [fixed_operands(gen) for _ in range(LAYERS)])
    c = int(label[1:])
    gen = torch.Generator().manual_seed(8000 + c)
    return ([k for k in INDEXER if k.per_chunk],
            [selection_chain(c, gen) for _ in range(LAYERS)])


def run_indexer(floor, args) -> dict:
    """Every 8k-only indexer kernel in graphs of 1 and LAYERS calls, per chunk where it moves."""
    import torch

    seams = indexer_seams()
    out = {"kernels": [asdict(k) for k in INDEXER], "chunks": {}}
    for label in args.group or INDEXER_GROUPS:
        kernels, layers = indexer_group(label)
        torch._dynamo.reset()
        graphs, inputs = {"floor": floor["graph"]}, {"floor": floor["inputs"]}
        for kernel in kernels:
            arity = len(layers[0][kernel.name])
            for calls in (1, LAYERS):
                name = f"{kernel.name}_x{calls}"
                graphs[name] = repeated(seams[kernel.name], calls, arity)
                inputs[name] = to_device([t for layer in layers[:calls]
                                          for t in layer[kernel.name]])
        timing = repeat_groups(graphs, inputs, args)
        floor_us = timing["floor"]["median_us"]
        rows = {}
        for kernel in kernels:
            one = timing[f"{kernel.name}_x1"]["median_us"]
            many = timing[f"{kernel.name}_x{LAYERS}"]["median_us"]
            rows[kernel.name] = {
                "x1_us": one, f"x{LAYERS}_us": many, "floor_us": floor_us,
                "per_chunk_us": many - floor_us,
                "per_call_us": (many - one) / (LAYERS - 1),
                "noise_floor": max(timing[f"{kernel.name}_x1"]["rep_spread"],
                                   timing[f"{kernel.name}_x{LAYERS}"]["rep_spread"])}
        out["chunks"][label] = {"rows": rows, "timing": timing}
        print(json.dumps({"indexer": label, "per_chunk_us": {
            k: round(v["per_chunk_us"], 1) for k, v in rows.items()}}), flush=True)
    return out


def build_floor():
    """The fixed cost of one execution: a one-element add."""
    import torch

    return {"graph": compiled(lambda x: x + 1),
            "inputs": to_device((torch.zeros(1, dtype=torch.int32),))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-module", type=Path, required=True,
                        help="the base revision's mla_sparse.py")
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    parser.add_argument("--part", action="append", choices=("sparse", "dense", "indexer"),
                        help="part to run (repeatable; default: all three)")
    parser.add_argument("--case", action="append", choices=[c.name for c in SPARSE_CASES],
                        help="sparse case (repeatable; default: every case)")
    parser.add_argument("--group", action="append", choices=INDEXER_GROUPS,
                        help="indexer group (repeatable; default: every group): 'fixed' = the "
                             "kernels without a chunk operand, 'cN' = chunk N's")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=4, help="calls per graph per repeat")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=2, help="operand sets checked per case")
    parser.add_argument("--save-dir", type=Path, help="keep every checked output as .pt here")
    args = parser.parse_args()
    if min(args.reps, args.iterations, args.seeds) < 1 or args.warmup < 0:
        parser.error("use reps, iterations and seeds >= 1 and warmup >= 0")
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise SystemExit("hardware benchmark: unset VLLM_NEURON_CPU_MODE and NKI_SIMULATOR")
    if not os.environ.get(LEASE_MARKER):
        raise SystemExit(f"run under devlease.py slice <name>, which sets {LEASE_MARKER}")
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        raise SystemExit("the served geometry is LNC2: NEURON_LOGICAL_NC_CONFIG must be 2")
    cache_root = os.environ.get("NEURON_LIBTORCH_CACHE_ROOT")
    if not cache_root:
        raise SystemExit("set NEURON_LIBTORCH_CACHE_ROOT to a compile cache directory")
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    parts = args.part or ["sparse", "dense", "indexer"]

    import vllm_neuron  # registers the Neuron compilation backend
    from vllm_neuron.functional.attention import mla_sparse as live

    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise SystemExit(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    modules = {"base": sparse_bench.load_module(args.base_module.resolve(),
                                                "mla_sparse_benchmark_base"), "after": live}
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_dir is not None:
        args.save_dir = args.save_dir.resolve()
        args.save_dir.mkdir(parents=True, exist_ok=True)
    # The compiler writes its logs to the working directory: keep them with the cache.
    scratch = Path(cache_root) / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = {
        "environment": {key: os.environ.get(key) for key in (
            LEASE_MARKER, "NEURON_LOGICAL_NC_CONFIG", "NEURON_PLATFORM_TARGET_OVERRIDE",
            "NEURON_LIBTORCH_CACHE_ROOT", "NEURON_CC_FLAGS")},
        "tree": str(ROOT),
        "modules": {name: {"path": m.__file__,
                           "sha256": hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
                    for name, m in modules.items()},
        "geometry": {"chunk": CHUNK, "prompt_8k": PROMPT_8K, "segment_8k": SEGMENT_8K,
                     "layers": LAYERS, "ranks": RANKS, "rank": RANK, "rank_rows": RANK_ROWS,
                     "heads": HEADS, "latent": LATENT, "page": PAGE, "softmax_scale": SCALE,
                     "candidates": CANDIDATES, "select_k": SELECT_K, "kpool": KPOOL},
        "timing_method": {"reps": args.reps, "iterations": args.iterations,
                          "warmup": args.warmup, "order": "rotated every call within a group",
                          "device": "system trace nc_exec_running, physical cores merged",
                          "synchronization": "outputs copied to CPU on every call",
                          "noise_floor": "largest (max - min) / median of a graph's rep medians"},
        "sparse": [], "dense": None, "indexer": None, "failures": [],
    }
    floor = build_floor()

    def save():
        out.write_text(json.dumps(report, indent=1) + "\n")

    if "sparse" in parts:
        for case in SPARSE_CASES:
            if args.case and case.name not in args.case:
                continue
            row = run_sparse_case(case, modules, floor, args, report["failures"])
            report["sparse"].append(row)
            save()
            t = row["timing"]
            print(json.dumps({"case": case.name, "base_us": round(t["base"]["median_us"], 1),
                              "after_us": round(t["after"]["median_us"], 1),
                              "after_over_base": round(t["after_over_base"], 4),
                              "noise": round(t["noise_floor"], 4),
                              "identical": [t[n]["identical"] for n in modules],
                              "emissions": [t[n]["emissions"] for n in modules]}), flush=True)
    if "dense" in parts:
        report["dense"] = run_dense(floor, args, report["failures"])
        save()
        print(json.dumps({"dense_us": round(report["dense"]["timing"]["dense"]["median_us"], 1),
                          "row_rel": report["dense"]["row_rel_vs_float64"]}), flush=True)
    if "indexer" in parts:
        report["indexer"] = run_indexer(floor, args)
        save()
    save()
    for failure in report["failures"]:
        print(f"FAIL {failure}", flush=True)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
