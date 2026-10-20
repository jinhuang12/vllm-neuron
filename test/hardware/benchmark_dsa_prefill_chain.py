# SPDX-License-Identifier: Apache-2.0
"""Time the DSA prefill indexer chain on one rank's rows of a chunk, per kernel.

Two subcommands.

``run`` (device; through the lease, which pins the cores and the LNC -- this script
selects none). One process times ONE graph: ``--layers`` (default 11, the model's DSA
layer count) layers of the prefill indexer chain at ``--cands`` candidate pools, as
``Glm5NextDSAIndexer`` runs it on one rank after the row cut, with every operand drawn
per layer:

    keys    = candidate_keys(indexer, pool_cache, cands)                 [cands, D]
    ids     = indexer._select_pool_ids(query, keys, weights, lengths,
                                       shardable=False)                  [R, k]
              (score GEMM with the head sum, causal bound, top-k, sentinel, order)
    indices = indexer.expand_indices(ids_chunk, lengths_chunk)           [T, width]

``R`` (``--rows``, default 16) is the rows one rank selects at a 1024-row chunk over 64
ranks; the expansion runs on the chunk's ``T`` (``--tokens``) rows, as after the
all-gather (the gather itself is a collective and is not in this graph). The rows are the
last rank's rows of the chunk that ends at ``cands * index_kpool`` tokens, so the last row
sees every candidate. The tree is ``--tree`` (default this file's tree), imported from
there, so one script times the base tree and the candidate alike.

``--graph scores`` times the gather and the score GEMM alone (layer 0's whole score
tile is an output), the form that pins the score kernel's bits.

For each seed of ``--seeds`` the graph runs once and its outputs (``chain``: every
layer's ``ids`` and layer 0's expansion; ``scores``: layer 0's scores) are saved under
``--records``;
with ``--reference`` they must equal that run's saved outputs bit for bit before any
timing, else the run stops. Then each seed's operands are timed ``--iterations`` times
after ``--warmup`` calls; device time per execution comes from the runtime system trace
(the LNC2 physical-core intervals of an execution merged). With ``--profile-dir`` one
more execution per seed runs under the device profiler for the per-kernel split.

``split`` (CPU only; any python with ``duckdb``). Reads a profile written by
``neuron-explorer view -d <profile dir> --output-format parquet --output-file <P>`` and
reports, per execution, each kernel's active time (the union of its
instruction intervals) on each physical core, labelled ``file.py:function`` by the
enclosing top-level function of the instruction's source line, and the Vector-engine
busy time of the score kernel (the head sum). Each source file's instructions are also
taken together (``files``: the union over its functions, which overlap). Instructions with
no NKI source location are compiler ops: ``compiler:wait`` for semaphore waits (one core
idling while the other runs a one-program kernel shows up there), ``compiler`` for the rest.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

DEVICE = "neuron:0"
#: Rows of the pooled-key store per page, for a tree whose prefill keys still go through
#: ``Glm5NextDSAIndexer._gather_candidates``: its page table is arithmetic, so any page size
#: names the same rows.
PAGE_SIZE = 128
#: Ranks a chunk's selection rows divide over at the served TP=64
#: (``indexer_shard.shard_degree``); ``--rows`` defaults to ``--tokens`` over this.
SERVED_RANKS = 64
#: Elements of each later layer's expansion kept live by the graph's outputs.
TAIL = 8


# ------------------------------------------------------------------------------------
# run
# ------------------------------------------------------------------------------------


def import_tree(tree: Path):
    """``(torch, model_fp8)`` with ``vllm_neuron`` imported from ``tree``, not the venv."""
    sys.path.insert(0, str(tree))
    sys.dont_write_bytecode = True
    import torch

    import vllm_neuron  # registers the Neuron compilation backend
    if not vllm_neuron.__file__.startswith(str(tree) + "/"):
        raise ValueError(f"imported vllm_neuron from {vllm_neuron.__file__}, not {tree}")
    from vllm_neuron.model.glm5_next import model_fp8
    return torch, model_fp8


def chunk_lengths(end: int, rows: int, tokens: int, torch):
    """``(rank_lengths [rows], chunk_lengths [tokens])`` int32 of the chunk ending at ``end``
    tokens.

    The rank's rows are the chunk's last ``rows``; row ``i`` of the chunk has length
    ``end - tokens + 1 + i``.
    """
    chunk = torch.arange(end - tokens + 1, end + 1, dtype=torch.int32)
    return chunk[tokens - rows:].clone(), chunk


def layer_operands(cfg, cands: int, rows: int, tokens: int, seed: int, layer: int, torch):
    """One layer's CPU operands: query, gate weights of both signs, a pooled-key store."""
    gen = torch.Generator().manual_seed(int(seed) * 1000 + int(layer))
    heads, dim = int(cfg.index_n_heads), int(cfg.index_head_dim)
    return {
        "query": torch.randn(rows, heads, dim, generator=gen).to(torch.bfloat16),
        "weights": torch.randn(rows, heads, generator=gen) * (dim ** -0.5) * (heads ** -0.5),
        # One trash row past the candidates, as the side cache is allocated.
        "pool_cache": (torch.randn(cands + 1, dim, generator=gen) * 0.5).to(torch.bfloat16),
    }


def chunk_pool_ids(k: int, tokens: int, torch):
    """``[tokens, k]`` int32 pool ids for the expansion: pools ``0 .. k - 1`` on every row.

    The expansion's cost does not depend on which pools are named, and every row of a
    chunk here is longer than ``k`` pools, so each id is a legal pool of its row.
    """
    return torch.arange(k, dtype=torch.int32).repeat(tokens, 1).contiguous()


def candidate_keys(indexer, pool_cache, cands: int):
    """The prefill indexer's candidate keys, as the tree under test reads them.

    A tree with ``_candidate_keys`` reads the store's prefix in place; an older one (the
    base of this series) copies it through ``_gather_candidates`` and ``dsa_paged_gather``.
    """
    if hasattr(indexer, "_candidate_keys"):
        return indexer._candidate_keys(pool_cache, cands)
    return indexer._gather_candidates(pool_cache, cands, PAGE_SIZE)


def build_graph(indexer, kind: str, layers: int, cands: int, torch):
    """The timed graph: ``chain`` (gather, selection, expansion) or ``scores`` (gather, scores)."""
    def chain(*flat):
        ids_out, expanded = [], []
        for layer in range(layers):
            query, weights, pool_cache, lengths, chunk_ids, chunk_lens = flat[6 * layer:6 * layer + 6]
            keys = candidate_keys(indexer, pool_cache, cands)
            ids_out.append(indexer._select_pool_ids(query, keys, weights, lengths,
                                                    shardable=False))
            expansion = indexer.expand_indices(chunk_ids, chunk_lens)
            expanded.append(expansion if layer == 0 else expansion[0:1, 0:TAIL])
        return (*ids_out, *expanded)

    def scores(*flat):
        out = []
        for layer in range(layers):
            query, weights, pool_cache = flat[6 * layer:6 * layer + 3]
            keys = candidate_keys(indexer, pool_cache, cands)
            score = indexer.score_pools(query, keys, weights)
            out.append(score if layer == 0 else score[0:1, 0:TAIL])
        return tuple(out)

    fn = {"chain": chain, "scores": scores}[kind]
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


def device_intervals(events_json: str) -> list[float]:
    """Per-execution device time in us, in execution order; LNC2 core intervals merged."""
    starts, intervals = {}, {}
    for event in json.loads(events_json)["events"]:
        if event["event_type"] != "nc_exec_running":
            continue
        key = (event["nc_idx"], event["tracking_id"])
        if event["phase"] == "start":
            starts[key] = event
        elif event["phase"] == "stop":
            start = starts.pop(key)
            intervals.setdefault(start["data"]["exec_id"], []).append(
                (start["data"]["nc_timestamp_ns"], event["data"]["nc_timestamp_ns"]))
    return [(max(end for _, end in intervals[e]) - min(b for b, _ in intervals[e])) / 1000.0
            for e in sorted(intervals)]


def time_calls(graph, inputs, iterations: int) -> list[float]:
    from nrtpy._nrtpy import SystemTraceSession

    with SystemTraceSession() as trace:
        for _ in range(iterations):
            outputs = graph(*inputs)
            outputs[0].to("cpu")
        events = trace.fetch_events_json()
    device = device_intervals(events)
    if len(device) != iterations:
        raise AssertionError(f"system trace has {len(device)} executions for {iterations} calls")
    return device


def profile_call(graph, inputs, directory: Path, torch) -> None:
    import libtorch_neuronx_lite.envs as libtorch_envs

    directory.mkdir(parents=True, exist_ok=True)
    runtime = torch.classes.neuron.Runtime()
    runtime.start_profiling(str(directory), ["device_profile", "system_profile"], None, None,
                            libtorch_envs.get_neuron_compile_cache_dir())
    try:
        graph(*inputs)[0].to("cpu")
    finally:
        runtime.stop_profiling()


def floors(cfg, cands: int, rows: int, tokens: int, width: int) -> dict:
    """Per-layer floors in us: bytes at 716 GB/s, FLOPs at 153 TFLOP/s, Vector elements at
    128 partitions x 0.96 GHz (the kernel bar of the floor-gap wave), one entry per stage."""
    heads, dim = int(cfg.index_n_heads), int(cfg.index_head_dim)
    k = int(cfg.index_topk) // int(cfg.index_kpool)
    hbm, peak, dve = 716e9, 153.0e12, 128 * 0.96e9
    key_bytes = cands * dim * 2
    score_bytes = rows * cands * 4
    stage = {
        "gather": {"bytes": 2 * key_bytes},
        "score_gemm": {"bytes": key_bytes + rows * heads * dim * 2 + rows * heads * 4
                       + score_bytes,
                       "flops": 2 * rows * heads * dim * cands,
                       "vector_elements": rows * heads * cands},
        "causal_bound": {"bytes": 2 * score_bytes + rows * 4,
                         "vector_elements": rows * cands},
        "topk": {"bytes": score_bytes + rows * k * 8, "vector_elements": rows * cands},
        "sentinel_and_order": {"bytes": rows * k * 8 + 2 * rows * k * 4},
        "expand": {"bytes": tokens * k * 4 + tokens * 4 + tokens * width * 4},
    }
    for entry in stage.values():
        entry["floor_us"] = max(entry.get("bytes", 0) / hbm, entry.get("flops", 0) / peak,
                                entry.get("vector_elements", 0) / dve) * 1e6
    return stage


def cmd_run(args) -> None:
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise ValueError("Hardware benchmark cannot run in CPU mode or the NKI simulator")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    if not args.use_compile_cache:
        # The graph cache key ignores kernel bodies, so an edited kernel could otherwise be
        # timed from a stale NEFF.
        os.environ["NEURON_LIBTORCH_DISABLE_COMPILE_CACHE"] = "1"
    tree = args.tree.resolve()
    torch, model_fp8 = import_tree(tree)
    cfg = model_fp8.Glm5NextTextConfig()
    indexer = model_fp8.Glm5NextDSAIndexer(cfg)
    pool = int(cfg.index_kpool)
    rows = args.rows or -(-args.tokens // SERVED_RANKS)
    head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain",
                            "--untracked-files=no", "--", "vllm_neuron"], check=True,
                           capture_output=True, text=True).stdout.strip()
    records = args.records.resolve()
    records.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or records) / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    end = args.chunk_end or args.cands * pool
    store = args.store_cands or args.cands
    if not end // pool <= args.cands <= store:
        raise ValueError(f"need the chunk's complete pools ({end // pool}) <= --cands "
                         f"({args.cands}) <= --store-cands ({store})")
    lengths, chunk_lens = chunk_lengths(end, rows, args.tokens, torch)
    graph = build_graph(indexer, args.graph, args.layers, args.cands, torch)
    tag = f"{args.label}_{args.graph}_C{args.cands}"
    report = {
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT",
            "NEURON_LIBTORCH_DISABLE_COMPILE_CACHE")},
        "tree": {"root": str(tree), "head": head, "vllm_neuron_modified_vs_head": dirty},
        "label": args.label, "graph": args.graph, "cands": args.cands, "rows": rows, "tokens": args.tokens,
        "chunk_end": end, "store_cands": store,
        "layers": args.layers, "select_k": indexer.select_k(),
        "args": {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items()},
        "unit": "device us per execution of the whole graph (all layers = one chunk)",
        "seeds": [],
    }
    for seed in args.seeds:
        cpu = []
        for layer in range(args.layers):
            ops = layer_operands(cfg, store, rows, args.tokens, seed, layer, torch)
            ops["lengths"] = lengths
            ops["chunk_ids"] = chunk_pool_ids(indexer.select_k(), args.tokens, torch)
            ops["chunk_lens"] = chunk_lens
            cpu.append(ops)
        flat = tuple(ops[key].to(DEVICE) for ops in cpu for key in (
            "query", "weights", "pool_cache", "lengths", "chunk_ids", "chunk_lens"))
        outputs = [t.to("cpu") for t in graph(*flat)]
        saved = records / f"{tag}_seed{seed}.pt"
        torch.save(outputs, saved)
        entry = {"seed": seed, "outputs": str(saved)}
        if args.reference is not None:
            reference = torch.load(args.reference / f"{args.reference_label}_{args.graph}"
                                   f"_C{args.reference_cands or args.cands}_seed{seed}.pt")
            equal = [bool(torch.equal(a, b)) for a, b in zip(outputs, reference)]
            entry["equals_reference"] = equal
            if not all(equal):
                report["seeds"].append(entry)
                out.write_text(json.dumps(report, indent=2) + "\n")
                raise SystemExit(f"seed {seed}: outputs differ from the reference; not timed")
        for _ in range(args.warmup):
            graph(*flat)[0].to("cpu")
        device = time_calls(graph, flat, args.iterations)
        entry["device_us"] = {"median": statistics.median(device), "min": min(device),
                              "max": max(device), "samples": device}
        if args.profile_dir is not None:
            where = args.profile_dir.resolve() / f"{tag}_seed{seed}"
            profile_call(graph, flat, where, torch)
            entry["profile"] = str(where)
        report["seeds"].append(entry)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"label": args.label, "graph": args.graph, "cands": args.cands,
                          "seed": seed,
                          "device_us_median": entry["device_us"]["median"],
                          "equals_reference": entry.get("equals_reference")}), flush=True)
    if args.graph == "chain":
        width = int(outputs[args.layers].shape[1])
    else:
        from vllm_neuron.functional.dsa.index_expand import index_expand_width
        width = int(index_expand_width(indexer.select_k(), pool))
    report["floors_per_layer"] = floors(cfg, args.cands, rows, args.tokens, width)
    report["floor_per_chunk_us"] = args.layers * sum(
        stage["floor_us"] for stage in report["floors_per_layer"].values())
    out.write_text(json.dumps(report, indent=2) + "\n")


# ------------------------------------------------------------------------------------
# split
# ------------------------------------------------------------------------------------


def _union_us(spans) -> float:
    """Time inside at least one of these ``(start, end)`` intervals (ns), microseconds."""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(spans):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total / 1000.0


def _function_table(path: str) -> list[tuple[int, int, str]]:
    """``(first line, last line, name)`` of every top-level function in ``path``."""
    import ast
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, SyntaxError):
        return []
    return [(node.lineno, node.end_lineno or node.lineno, node.name) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _labeler():
    """``nki_source_location -> file.py:function`` (``compiler`` for no location)."""
    tables: dict[str, list] = {}
    cache: dict[str, str] = {}

    def label(location) -> str:
        if not location:
            return "compiler"
        if location in cache:
            return cache[location]
        path, _, rest = location.partition(":")
        name = path.rsplit("/", 1)[-1]
        line = int(rest.split(":")[0]) if rest.split(":")[0].isdigit() else -1
        table = tables.setdefault(path, _function_table(path))
        owner = next((f for a, b, f in table if a <= line <= b), "?")
        cache[location] = f"{name}:{owner}"
        return cache[location]
    return label


def cmd_split(args) -> None:
    import duckdb
    found = sorted(glob.glob(f"{args.parquet}/*_session_*"))
    if not found:
        raise ValueError(f"no parquet session directory under {args.parquet}")
    path = found[0]
    # The thread count is fixed at connect: the default pool starts one thread per host CPU,
    # which stalls under a taskset of a few cores.
    con = duckdb.connect(config={"threads": args.threads})
    q = lambda t: f"'{path}/{t}.parquet'"
    execs = con.sql(f"SELECT execution_index, execution_start_ts, execution_end_ts "
                    f"FROM {q('ExecutionInfo')} ORDER BY 2").fetchall()
    label = _labeler()
    per_exec = []
    for idx, s0, e0 in execs:
        rows = con.sql(f"""
            SELECT nki_source_location, pcore_idx, engine, opcode, start_ts, end_ts
            FROM {q('Instruction')}
            WHERE start_ts >= {s0} AND end_ts <= {e0}
            """).fetchall()
        spans: dict[tuple[str, int], list] = {}
        files: dict[tuple[str, int], list] = {}
        head_sum: dict[int, float] = {}
        for location, core, engine, opcode, start, end in rows:
            name = label(location)
            if name == "compiler" and str(opcode) == "EVENT_SEMAPHORE":
                name = "compiler:wait"
            spans.setdefault((name, int(core)), []).append((start, end))
            files.setdefault((name.split(":")[0], int(core)), []).append((start, end))
            if (name.startswith("score_gemm.py:") and engine == "Vector"
                    and not str(opcode).startswith(("DMA", "EVENT", "DRAIN"))):
                head_sum[int(core)] = head_sum.get(int(core), 0.0) + (end - start) / 1000.0
        kernels = {}
        for (name, core), items in sorted(spans.items()):
            kernels.setdefault(name, {})[f"pcore{core}"] = {
                "instructions": len(items), "active_us": _union_us(items)}
        by_file = {}
        for (name, core), items in sorted(files.items()):
            by_file.setdefault(name, {})[f"pcore{core}"] = _union_us(items)
        per_exec.append({"execution": int(idx), "wall_us": (e0 - s0) / 1000.0,
                         "kernels": kernels, "files": by_file,
                         "score_gemm_vector_busy_us": {f"pcore{c}": v
                                                       for c, v in sorted(head_sum.items())}})
    print(json.dumps({"profile": path, "executions": per_exec}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--tree", type=Path, default=Path(__file__).resolve().parents[2])
    run.add_argument("--label", required=True, help="names the saved outputs and profiles")
    run.add_argument("--graph", choices=("chain", "scores"), default="chain")
    run.add_argument("--cands", type=int, required=True, help="candidate pools scanned")
    run.add_argument("--chunk-end", type=int,
                     help="tokens the chunk ends at (default cands * index_kpool)")
    run.add_argument("--store-cands", type=int,
                     help="pools the drawn store holds (default --cands): a scan of fewer "
                          "candidates than a wider window's store reads the same keys")
    run.add_argument("--reference-cands", type=int,
                     help="--cands of the reference run (default --cands)")
    run.add_argument("--rows", type=int, help="rows one rank selects (default tokens / 64)")
    run.add_argument("--tokens", type=int, default=1024, help="the chunk's rows")
    run.add_argument("--layers", type=int, default=11)
    run.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--iterations", type=int, default=20)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--records", type=Path, required=True)
    run.add_argument("--reference", type=Path, help="records directory of the reference run")
    run.add_argument("--reference-label", default="before")
    run.add_argument("--profile-dir", type=Path)
    run.add_argument("--use-compile-cache", action="store_true")
    split = sub.add_parser("split")
    split.add_argument("--parquet", required=True,
                       help="explorer --output-format parquet --output-file directory")
    split.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.cmd == "run":
        if args.layers < 1 or args.iterations < 1 or args.warmup < 0:
            raise ValueError("Use positive layer and iteration counts")
        cmd_run(args)
    else:
        cmd_split(args)


if __name__ == "__main__":
    main()
