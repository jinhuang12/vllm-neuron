# SPDX-License-Identifier: Apache-2.0
"""Device A/B of the DSA prefill selection chain: one rank's sharded rows against all rows.

Run it through the device lease, which pins the cores and the LNC; this script selects no
cores::

    python3 devlease.py slice dsa -- python \\
        test/vllm_neuron/functional/dsa/indexer_shard_device.py \\
        --cands 16384 --label dsa-r1 --output /tmp/indexer_shard_device/c16384_r1.json

For one candidate width ``C`` it compiles and times five graphs on one logical core:

* ``replicated`` -- the as-built chain on all ``T`` rows of a chunk:
  ``Glm5NextDSAIndexer.score_pools`` then ``select_bounded_pools`` (score GEMM, causal
  bound, top-k, sentinel, order) -> ``[T, k]`` int32 pool ids.
* ``sharded`` -- one rank's share at degree ``d``: ``indexer_shard.select_local_rows`` on
  the same ``T``-row operands (one NKI launch cuts the three operands, then the same two
  stages run on ``R = ceil(T / d)`` rows) -> ``[R, k]`` int32, what the all-gather sends.
  The rank is a device operand, as in the model. The all-gather is not timed: one chip has
  no TP group.
* ``precut`` -- the same two stages on the rank's ``R`` rows cut on the CPU beforehand:
  the sharded chain without the row cut. ``sharded - precut`` is what the cut adds to the
  chain on the device.
* ``cut`` -- the row cut alone: ``select_local_rows`` with a selection that returns its
  three operands. After timing it runs for the first, a middle and the last rank, and each
  output must equal ``index_select`` on the CPU reference index, bit for bit.
* ``launch`` -- a graph that only scales a one-element input: the fixed cost of one
  execution, subtracted from the graphs above.

Each graph returns its outputs and a float32 sum of one of them; a timed call copies only
the sum to the CPU, which waits for the whole graph. Each call is timed twice: on the host
clock (``wall``) and on the device clock, from the runtime's system trace
(``nc_exec_running`` start and stop per physical core, merged per execution, the reduction
``test/hardware/benchmark_mhc_decode.py`` uses). The device clock excludes host dispatch
and the output copy, so ``net_ms`` (the reported value) is the device median less the
launch graph's device median; ``wall_net_ms`` is the same on the host clock. Each graph is
timed in its own block of calls: a call that switches to another compiled graph pays a few
hundred microseconds more on the host clock, so the blocks are not interleaved. The launch
graph is timed again after the others (``after_us``), which shows how far the subtracted
baseline drifts.

The chunk sits at the end of a ``index_kpool * C``-token context, so every row sees all
``C`` pools. After timing, the sharded ids must equal the replicated ids on the rank's rows
as per-row sets, and every stage must have taken the NKI route. The run also records every
node of each traced graph (``graph_ops``): the op inventory of the selection path. Each
graph compiles after a ``torch._dynamo.reset()`` (so the stage counters see its own trace),
so a graph called again after a later graph's compile (the launch graph's second timing,
the profiles) is traced again (``traces`` > 1); that trace must give the same nodes
(``traces_identical``), and it finds the same compile-cache entry. One JSON object is
written per run.

``--compile-only GRAPH`` compiles that one graph, runs it once and writes only its record:
for a width where a graph does not compile, so each path's compiler error is in its own log
(the compile cache ends the process on a compiler error).

``--profile-dir DIR`` adds a device profile of ``--profile-iterations`` calls of each graph
after the timing (``DIR/<label>/<graph>``), ingested by ``neuron-explorer view
--ingest-only`` into ``DIR/explorer-data``; ``indexer_shard_profile.py`` turns an ingested
profile into per-op device time.

``--skip-bound`` times the chain without the causal bound: ``_select_bounded`` (the tail
``select_bounded_pools`` shares with ``forward_requests``) on the raw scores. At
``C = 65536`` the bound's whole-width SBUF row does not compile on either path, so that
width is measured this way. The first call there compiles for about 7 minutes; the lease
allows 10, so run one width per lease.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

#: The worktree root, ahead of any installed copy: the lease command sets no PYTHONPATH.
#: Bytecode is not written, so a run leaves no ``__pycache__`` in the tree.
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
from vllm_neuron.functional.dsa import shard_rows  # noqa: E402
from vllm_neuron.functional.dsa.indexer_shard import row_shard, select_local_rows  # noqa: E402
from test.vllm_neuron.functional.dsa import indexer_shard_case as case  # noqa: E402

DEVICE = "neuron:0"
#: The prefill chunk every calibrated number is per (``rules.json`` reference point p1).
CHUNK = 1024
#: The graphs one run compiles and times, in that order (see the module docstring).
GRAPHS = ("launch", "replicated", "sharded", "precut", "cut")


def compiled(fn, traces: list):
    """``fn`` compiled for the device, appending the nodes of each trace to ``traces``."""
    device_backend = torch._dynamo.lookup_backend("neuron_libtorch")

    def recording(gm, example_inputs):
        ops = []
        for node in gm.graph.nodes:
            if node.op not in ("call_function", "call_method", "call_module"):
                continue
            value = node.meta.get("example_value")
            values = value if isinstance(value, (tuple, list)) else [value]
            ops.append({"op": node.op, "target": str(node.target),
                        "out": [[list(v.shape), str(v.dtype).replace("torch.", "")]
                                for v in values if isinstance(v, torch.Tensor)]})
        traces.append(ops)
        return device_backend(gm, example_inputs,
                              options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})

    return torch.compile(fn, backend=recording, fullgraph=True, dynamic=False)


def operands(tokens: int, cands: int, seed: int):
    """``case_operands`` with the chunk moved to the end of an ``index_kpool * cands``
    context: ``(query [T, H, D] bf16, keys [C, D] bf16, weights [T, H] fp32, seq_lens [T]
    int32)``."""
    query, keys, weights, _ = case.case_operands(tokens, cands, seed)
    end = cands * case.make_indexer(cands).index_kpool
    seq_lens = torch.arange(end - tokens + 1, end + 1, dtype=torch.int32)
    return query, keys, weights, seq_lens


def device_intervals(events_json: str) -> list[float]:
    """Device microseconds per execution, in execution order: from the first physical
    core's ``nc_exec_running`` start to the last one's stop."""
    starts, windows = {}, {}
    for event in json.loads(events_json)["events"]:
        if event["event_type"] != "nc_exec_running":
            continue
        key = (event["nc_idx"], event["tracking_id"])
        if event["phase"] == "start":
            starts[key] = event
        elif event["phase"] == "stop":
            start = starts.pop(key)
            windows.setdefault(start["data"]["exec_id"], []).append(
                (start["data"]["nc_timestamp_ns"], event["data"]["nc_timestamp_ns"]))
    return [(max(end for _, end in spans) - min(begin for begin, _ in spans)) / 1000.0
            for _, spans in sorted(windows.items())]


def stats(samples: list[float]) -> dict:
    """Median, p10, p90, min and max of ``samples`` (microseconds)."""
    ordered = sorted(samples)
    return {"median_us": statistics.median(ordered),
            "p10_us": ordered[int(0.1 * (len(ordered) - 1))],
            "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
            "min_us": ordered[0], "max_us": ordered[-1]}


def measure(fn, inputs, warmup: int, iterations: int) -> dict:
    """One block of calls that each wait for the graph: host and device microseconds."""
    from nrtpy._nrtpy import SystemTraceSession

    for _ in range(warmup):
        fn(*inputs)[1].to("cpu")
    wall = []
    with SystemTraceSession() as trace:
        for _ in range(iterations):
            started = time.perf_counter_ns()
            fn(*inputs)[1].to("cpu")
            wall.append((time.perf_counter_ns() - started) / 1000.0)
        events = trace.fetch_events_json()
    device = device_intervals(events)
    if len(device) != iterations:
        raise RuntimeError(f"the system trace holds {len(device)} executions for "
                           f"{iterations} calls")
    return {"iterations": iterations, "wall": stats(wall), "device": stats(device)}


def first_call(name, graph, inputs, out):
    """Compile ``graph`` and run it once, recording into ``out[name]``; return the compiled
    graph and its first output on the CPU."""
    torch._dynamo.reset()
    case.reset_stage_counters()
    shard_rows.reset_shard_rows_dispatch_counters()
    traces = []
    fn = compiled(graph, traces)
    started = time.perf_counter()
    first_out = fn(*inputs)[0]
    first_out = tuple(t.to("cpu") for t in first_out) if isinstance(first_out, tuple) else (
        first_out.to("cpu"))
    out[name] = {"first_call_s": time.perf_counter() - started,
                 "routes": case.stage_counters(),
                 "cut_routes": shard_rows.shard_rows_dispatch_counters(), "graph_traces": traces}
    return fn, first_out


def write_record(record: dict, path: Path) -> None:
    """Write ``record``, each graph's traces as its nodes (``graph_ops``), its trace count
    and whether every trace gave those nodes."""
    for body in record["graphs"].values():
        traces = body.pop("graph_traces")
        body.update({"graph_ops": traces[0], "traces": len(traces),
                     "traces_identical": all(t == traces[0] for t in traces)})
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")


def profile(name: str, fn, inputs, args) -> dict:
    """A device profile of ``fn``, ingested by ``neuron-explorer`` for
    ``indexer_shard_profile.py``."""
    import libtorch_neuronx_lite.envs as libtorch_envs

    directory = args.profile_dir / args.label / name
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    runtime = torch.classes.neuron.Runtime()
    runtime.start_profiling(str(directory), ["device_profile", "system_profile"], None, None,
                            libtorch_envs.get_neuron_compile_cache_dir())
    try:
        for _ in range(args.profile_iterations):
            fn(*inputs)[1].to("cpu")
    finally:
        runtime.stop_profiling()
    data = args.profile_dir / "explorer-data"
    display = f"{args.label}-{name}"
    ingest = subprocess.run(
        ["neuron-explorer", "view", "-d", str(directory), "--display-name", display,
         "--ingest-only", "--data-path", str(data)],
        capture_output=True, text=True, timeout=1800, check=False)
    if ingest.returncode != 0:
        raise RuntimeError(f"ingest of {directory} failed: {ingest.stderr[-2000:]}")
    return {"profile_dir": str(directory), "explorer_data": str(data), "display_name": display,
            "iterations": args.profile_iterations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cands", type=int, required=True)
    parser.add_argument("--degree", type=int, default=64)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skip-bound", action="store_true",
                        help="time the chain without the causal bound (see the docstring)")
    parser.add_argument("--label", required=True, help="lease name and repeat, for the record")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, default=None,
                        help="also profile each graph and ingest it here (see the docstring)")
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument("--compile-only", choices=GRAPHS, default=None,
                        help="compile and run one graph once, then stop (see the docstring)")
    args = parser.parse_args()
    if os.environ.get("NKI_SIMULATOR") not in (None, "", "0") or os.environ.get(
            "VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("a device run needs NKI_SIMULATOR and VLLM_NEURON_CPU_MODE off")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise ValueError(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    if not 0 <= args.rank < args.degree:
        raise ValueError(f"rank {args.rank} is not a rank of degree {args.degree}")

    indexer = case.make_indexer(args.cands)
    shard = row_shard(CHUNK, args.degree)
    host = operands(CHUNK, args.cands, args.seed)
    query, keys, weights, seq_lens = (t.to(DEVICE) for t in host)
    rank = torch.full((1,), args.rank, dtype=torch.int32, device=DEVICE)
    probe = torch.ones(1, dtype=torch.float32, device=DEVICE)
    mine = shard_rows.rank_row_index(CHUNK, shard.rows, args.rank, "cpu")
    rank_rows = tuple(host[j].index_select(0, mine).to(DEVICE) for j in (0, 2, 3))

    def select(rows_query, rows_weights, rows_seq_lens):
        scores = indexer.score_pools(rows_query, keys, rows_weights)
        if args.skip_bound:
            return indexer._select_bounded(scores)
        return indexer.select_bounded_pools(scores, rows_seq_lens)

    def replicated(q, w, s):
        ids = select(q, w, s)
        return ids, ids.to(torch.float32).sum()

    def sharded(q, w, s, r):
        ids = select_local_rows(select, q, w, s, shard, r)
        return ids, ids.to(torch.float32).sum()

    def precut(q, w, s):
        ids = select(q, w, s)
        return ids, ids.to(torch.float32).sum()

    def cut(q, w, s, r):
        taken = select_local_rows(lambda *rows: rows, q, w, s, shard, r)
        return taken, taken[2].to(torch.float32).sum()

    def launch(x):
        y = x * 2.0
        return y, y.sum()

    record = {"label": args.label, "cands": args.cands, "tokens": CHUNK,
              "degree": args.degree, "rows": shard.rows, "rank": args.rank,
              "causal_bound": not args.skip_bound,
              "select_k": int(indexer.select_k()), "heads": int(indexer.index_n_heads),
              "head_dim": int(indexer.index_head_dim),
              "visible_cores": os.environ["NEURON_RT_VISIBLE_CORES"],
              "lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
              "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS", ""), "graphs": {}}
    graphs = record["graphs"]
    chunk = (query, weights, seq_lens)
    plan = {"launch": (launch, (probe,)), "replicated": (replicated, chunk),
            "sharded": (sharded, (*chunk, rank)), "precut": (precut, rank_rows),
            "cut": (cut, (*chunk, rank))}
    assert tuple(plan) == GRAPHS, (tuple(plan), GRAPHS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.compile_only is not None:
        first_call(args.compile_only, *plan[args.compile_only], graphs)
        write_record(record, args.output)
        return
    calls, first = {}, {}
    for name, (graph, inputs) in plan.items():
        fn, first[name] = first_call(name, graph, inputs, graphs)
        calls[name] = (fn, inputs)
        graphs[name].update(measure(fn, inputs, args.warmup, args.iterations))
    graphs["launch"]["after"] = measure(*calls["launch"], args.warmup, args.iterations)
    whole, local, cut_fn = first["replicated"], first["sharded"], calls["cut"][0]
    cut_exact = {}
    for other in sorted({0, args.degree // 2, args.degree - 1}):
        taken = cut_fn(query, weights, seq_lens,
                       torch.full((1,), other, dtype=torch.int32, device=DEVICE))[0]
        index = shard_rows.rank_row_index(CHUNK, shard.rows, other, "cpu")
        cut_exact[str(other)] = all(
            torch.equal(t.to("cpu"), h.index_select(0, index))
            for t, h in zip(taken, (host[0], host[2], host[3])))

    rows = range(args.rank * shard.rows, min((args.rank + 1) * shard.rows, CHUNK))
    same = sum(int(set(local[i].tolist()) == set(whole[row].tolist()))
               for i, row in enumerate(rows))
    same_precut = torch.equal(local, first["precut"])
    record["check"] = {"rows": len(rows), "same_set_rows": same, "cut_exact_by_rank": cut_exact,
                       "sharded_equals_precut": same_precut,
                       "routes_off_nki": {g: {k: v for k, v in body["routes"].items()
                                              if v[1] != 0}
                                          for g, body in graphs.items()},
                       "cut_routes": {g: body["cut_routes"] for g, body in graphs.items()}}
    for clock, key in (("device", "net_ms"), ("wall", "wall_net_ms")):
        floor = graphs["launch"][clock]["median_us"]
        net = {g: (graphs[g][clock]["median_us"] - floor) / 1000.0
               for g in ("replicated", "sharded", "precut", "cut")}
        net["saved"] = net["replicated"] - net["sharded"]
        net["cut_in_chain"] = net["sharded"] - net["precut"]
        record[key] = net
    if args.profile_dir is not None:
        record["profiles"] = {name: profile(name, fn, inputs, args)
                              for name, (fn, inputs) in calls.items()}
    write_record(record, args.output)
    print(json.dumps({"label": args.label, "cands": args.cands, "net_ms": record["net_ms"],
                      "wall_net_ms": record["wall_net_ms"], "check": record["check"],
                      "first_call_s": {g: round(b["first_call_s"], 1)
                                       for g, b in graphs.items()}}), flush=True)
    if same != len(rows):
        raise SystemExit(f"sharded ids differ from replicated on {len(rows) - same} rows")
    off_nki = {g: v for g, v in record["check"]["routes_off_nki"].items() if v}
    if off_nki:
        raise SystemExit(f"a stage left the NKI route: {off_nki}")
    for g in ("sharded", "cut"):
        if graphs[g]["cut_routes"] != (1, 0):
            raise SystemExit(f"the {g} graph's row cut left the NKI route: "
                             f"{graphs[g]['cut_routes']}")
    if not all(cut_exact.values()):
        raise SystemExit(f"the row cut differs from the reference: {cut_exact}")
    if not same_precut:
        raise SystemExit("the sharded ids differ from the chain on the CPU-cut rows")
    retraced = [g for g, body in graphs.items() if not body["traces_identical"]]
    if retraced:
        raise SystemExit(f"a second trace gave other nodes: {retraced}")


if __name__ == "__main__":
    main()
