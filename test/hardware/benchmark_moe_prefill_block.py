# SPDX-License-Identifier: Apache-2.0
"""One MoE block of the prefill graph on the device: this tree against 8aa22fa.

Run it through the device lease (``devlease.py slice mhc``), which pins the cores and the
LNC; this script does not select cores.

What runs. ``test/vllm_neuron/functional/moe/moe_prefill_case.block_step`` from either
model tree with the same weights and operands: the attention site's ``mhc_post`` (the
attention-side ``hyper_connection`` kernel), the feed-forward site's ``mhc_pre``, the
experts' RMSNorm, the router, EP rank 0's 18 routed experts at I=512, the shared expert,
the feed-forward ``mhc_post``. One rank's TP=64 EP=16 shard, checkpoint layer 3's mHC
leaves, gains, router and correction bias; random fp8 expert banks. Compiled through
``neuron_libtorch`` with the served neuronx-cc arguments, one graph per (T, collectives,
variant).

Collectives, two modes (``--collectives``):

* ``one-rank`` (the goal's measurement): the served prefill graph's two row-parallel
  all-reduces at their own sites, the attention output's (``project_output``, before the
  attention-side ``hyper_connection`` combine; served HLO ``all-reduce.7248``) and the
  feed-forward one (``_ffn_half``; ``all-reduce.7863``), over a one-rank gloo group, so
  each is numerically the identity but is an HLO ``all-reduce`` the compiler schedules.
  The feed-forward one is the model's own call: ``_resolve_tp_group`` answers a stand-in
  group whose ``all_reduce`` is the functional collective, as the served group's is.
* ``identity``: no collective in the graph (``_resolve_tp_group`` answers ``None``).
  neuronx-cc then fuses the collapse into the norm at 8aa22fa already, so this mode does
  NOT reproduce the served loop; it is reported to show the change costs nothing there.

What is measured, per (T, variant):

1. Device time per execution from the runtime system trace (``nc_exec_running``, the two
   physical cores' intervals merged), ``--iterations`` timed calls after ``--warmup``,
   before/after interleaved. Median and p90.
2. A device profile of ``--profile-iterations`` calls, ingested with ``neuron-explorer``
   and sampled every 2 us with the serving profile's rules (``benchmark_layer_glue.
   bucket_main``, frozen there from the serving-profile breakdown tool):
   per physical core the highest-priority instruction covering the sample (NKI kernel
   by source file / unnamed compiler op > DMA issue > semaphore > core barrier); a sample
   where both cores compute is split 0.5/0.5. The goal's window is the PREFILL_BREAKDOWN
   item: from the end of the attention-side ``hyper_connection`` kernel to the router
   kernel. Kernels interleave (the router's weight loads start early), so the ends are
   defined by data: the window opens at the last instruction of the first
   ``mhc/hyper_connection.py`` instance (instances split at the largest gap between its
   instructions) and closes at the router kernel's first instruction that reads the
   activations: the fused kernel's RMSNorm stage (``rmsnorm_tkg``) at 8aa22fa, the router
   GEMM (``router_topk`` ``MATMUL``) here. The window's compiler-op ms is the number the
   goal bounds; the block's total compiler-op ms and the window's instruction counts by
   opcode are reported too (the loop is ~4,000 fp32 ``MATMUL`` + ``LDWEIGHTS`` per pcore).

3. Router accuracy (``--accuracy``, default on): a second graph per (T, collectives,
   variant), ``moe_prefill_case.block_step_tapped``, returns the collapsed rows the
   router read and its logits and index; on the host, both are compared with
   ``moe_prefill_case.exact_router`` on those same rows (fp64, the fused kernel's two
   bf16 roundings) and its noaux_tc selection. This graph is not timed.

The compile cache is used (``NEURON_LIBTORCH_CACHE_ROOT``): its key ignores kernel
bodies, so give it a fresh root after a kernel edit. Profiles are analysed by a second
invocation (``--analyze-only``), off the device lease.
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
import tempfile
import time

COMPILER_OPS = "compiler ops (unnamed XLA)"
HC_LABEL = "mhc/hyper_connection.py"
COLLECTIVES = ("one-rank", "identity")


def bucket_main(name: str, out_path: str, global_dir: str) -> None:
    """Bucket one ingested profile; runs under a python that has duckdb (no torch)."""
    import collections
    import glob

    import duckdb

    step = 2_000
    path = sorted(glob.glob(f"{global_dir}/{name}_*_session_*@latest"))[0]
    c = duckdb.connect()
    c.sql("SET threads=16")

    def q(table):
        return f"'{path}/{table}.parquet'"

    def spans(table):
        if Path(f"{path}/{table}.parquet").exists():
            return f"SELECT start_ts s, end_ts e FROM {q(table)}"
        return "SELECT 0::BIGINT s, 0::BIGINT e WHERE false"

    execs = c.sql(f"SELECT execution_index, execution_start_ts, execution_end_ts "
                  f"FROM {q('ExecutionInfo')} ORDER BY 2").fetchall()
    lab = ("CASE WHEN compiler_opcode = 'PSEUDO_CORE_BARRIER' THEN 'wait: core barrier' "
           "WHEN opcode = 'EVENT_SEMAPHORE' THEN 'wait: semaphore' "
           "WHEN opcode LIKE 'DMA%' THEN 'dma issue' "
           "WHEN nki_source_location IS NOT NULL AND nki_source_location <> '' THEN "
           "regexp_replace(split_part(nki_source_location, ':', 1), "
           "'^.*/(vllm_neuron|site-packages)/(functional/)?', '') "
           f"ELSE '{COMPILER_OPS}' END")
    pri = ("CASE WHEN compiler_opcode = 'PSEUDO_CORE_BARRIER' THEN 5 "
           "WHEN opcode = 'EVENT_SEMAPHORE' THEN 4 WHEN opcode LIKE 'DMA%' THEN 3 "
           "WHEN nki_source_location IS NOT NULL AND nki_source_location <> '' THEN 1 "
           "ELSE 2 END")
    c.sql(f"CREATE TABLE x AS SELECT pcore_idx core, start_ts s, end_ts e, {pri} pri, "
          f"{lab} lab, engine, opcode FROM {q('Instruction')}")
    c.sql(f"CREATE TABLE cc AS {spans('CcOp')}")
    c.sql(f"CREATE TABLE dma AS {spans('DmaPacket')}")
    traced = []
    for idx, s0, e0 in execs:
        last = c.sql(f"SELECT core, max(e) FROM x WHERE s >= {s0} AND e <= {e0} "
                     f"GROUP BY 1").fetchall()
        t_end = min((m for _, m in last), default=s0)
        if len(last) >= 2 and t_end >= e0 - 2_000_000:
            traced.append((idx, s0, e0))
    waits = ("CASE WHEN incc THEN 'wait: collective' WHEN indma THEN 'wait: DMA in flight' "
             "WHEN least(p0, p1) = 4 THEN 'wait: semaphore' "
             "WHEN least(p0, p1) = 5 THEN 'wait: core barrier' "
             "WHEN least(p0, p1) = 3 THEN 'dma issue' ELSE 'idle (nothing traced)' END")

    def buckets(s0, e0) -> dict:
        c.sql(f"CREATE OR REPLACE TABLE t AS SELECT range AS ts FROM range({s0}, {e0}, {step})")
        c.sql(f"CREATE OR REPLACE TABLE pc AS SELECT t.ts, x.core, "
              f"arg_min(x.lab, x.pri * 1000000 + (x.e - x.s) // 1000) lab, min(x.pri) pri "
              f"FROM t JOIN (SELECT * FROM x WHERE e > {s0} AND s < {e0}) x "
              f"ON x.s <= t.ts AND x.e > t.ts GROUP BY t.ts, x.core")
        c.sql(f"CREATE OR REPLACE TABLE dq AS SELECT DISTINCT t.ts FROM t JOIN "
              f"(SELECT * FROM dma WHERE e > {s0} AND s < {e0}) d ON d.s <= t.ts AND d.e > t.ts")
        c.sql(f"CREATE OR REPLACE TABLE ccq AS SELECT DISTINCT t.ts FROM t JOIN "
              f"(SELECT * FROM cc WHERE e > {s0} AND s < {e0}) d ON d.s <= t.ts AND d.e > t.ts")
        c.sql("CREATE OR REPLACE TABLE s AS SELECT t.ts, c0.lab l0, coalesce(c0.pri, 9) p0, "
              "c1.lab l1, coalesce(c1.pri, 9) p1, t.ts IN (SELECT ts FROM ccq) incc, "
              "t.ts IN (SELECT ts FROM dq) indma FROM t "
              "LEFT JOIN pc c0 ON c0.ts = t.ts AND c0.core = 0 "
              "LEFT JOIN pc c1 ON c1.ts = t.ts AND c1.core = 1")
        out = {}
        for label, w in c.sql(
                "SELECT lab, sum(w) FROM ("
                "SELECT l0 lab, 0.5 w FROM s WHERE p0 <= 2 AND p1 <= 2 "
                "UNION ALL SELECT l1, 0.5 FROM s WHERE p0 <= 2 AND p1 <= 2 "
                "UNION ALL SELECT l0, 1.0 FROM s WHERE p0 <= 2 AND p1 > 2 "
                "UNION ALL SELECT l1, 1.0 FROM s WHERE p1 <= 2 AND p0 > 2 "
                f"UNION ALL SELECT {waits}, 1.0 FROM s WHERE p0 > 2 AND p1 > 2) "
                "GROUP BY 1").fetchall():
            out[label] = float(w) * step / 1e6
        return out

    whole = collections.defaultdict(dict)
    window = collections.defaultdict(dict)
    rows = []
    for idx, s0, e0 in traced:
        hc = c.sql(f"SELECT s, e FROM x WHERE s >= {s0} AND e <= {e0} "
                   f"AND lab = '{HC_LABEL}' ORDER BY s").fetchall()
        hc_end = None
        if len(hc) >= 2:
            gaps = [hc[i + 1][0] - hc[i][0] for i in range(len(hc) - 1)]
            split = gaps.index(max(gaps)) + 1
            hc_end = max(e for _, e in hc[:split])
        router_start = c.sql(
            f"SELECT min(s) FROM x WHERE s >= {hc_end or s0} AND e <= {e0} AND "
            f"(lab LIKE '%rmsnorm_tkg%' OR (lab LIKE '%router_topk%' AND opcode = 'MATMUL'))"
        ).fetchone()[0]
        kernels = c.sql(
            f"SELECT lab, core, min(s) - {s0}, max(e) - {s0}, count(*) FROM x "
            f"WHERE s >= {s0} AND e <= {e0} AND pri = 1 GROUP BY 1, 2 ORDER BY 3").fetchall()
        row = {"execution": idx, "execution_ms": (e0 - s0) / 1e6,
               "kernels_first_last_us": [
                   {"label": k, "pcore": core, "first_us": a / 1e3, "last_us": b / 1e3,
                    "instructions": n} for k, core, a, b, n in kernels]}
        for label, ms in buckets(s0, e0).items():
            whole[label][idx] = ms
        if router_start and hc_end:
            row["window_us"] = [(hc_end - s0) / 1e3, (router_start - s0) / 1e3]
            row["window_ms"] = (router_start - hc_end) / 1e6
            for label, ms in buckets(hc_end, router_start).items():
                window[label][idx] = ms
            row["window_opcodes_by_pcore"] = [
                {"pcore": core, "opcode": op, "engine": eng, "label": lb, "count": n}
                for core, op, eng, lb, n in c.sql(
                    f"SELECT core, opcode, engine, lab, count(*) FROM x WHERE s >= {hc_end} "
                    f"AND e <= {router_start} GROUP BY 1, 2, 3, 4 ORDER BY 5 DESC "
                    f"LIMIT 24").fetchall()]
        rows.append(row)
    steps = [i for i, *_ in traced]
    n = max(len(steps), 1)

    def mean(d):
        return sum(d.get(i, 0.0) for i in steps) / n

    windows = [r["window_ms"] for r in rows if "window_ms" in r]
    Path(out_path).write_text(json.dumps(dict(
        trace_dir=path, executions=len(execs), fully_traced_executions=steps,
        mean_execution_ms=sum(r["execution_ms"] for r in rows) / n,
        mean_window_ms=sum(windows) / len(windows) if windows else None,
        buckets_ms={k: mean(v) for k, v in whole.items()},
        window_buckets_ms={k: mean(v) for k, v in window.items()},
        executions_detail=rows,
    ), indent=1))


if __name__ == "__main__" and sys.argv[1:2] == ["--bucket"]:
    bucket_main(*sys.argv[2:5])
    sys.exit(0)

#: The worktree root, ahead of any installed copy (the lease command sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

DEVICE = "neuron:0"
TOKENS = (1024, 64)
DUCKDB_PYTHON = Path(os.environ.get("MOE_PREFILL_DUCKDB_PYTHON", "/tmp/pqvenv/bin/python"))
#: The served model's neuronx-cc arguments (``benchmark_moe_decode.MODEL_COMPILER_ARGS``).
MODEL_COMPILER_ARGS = [
    "--auto-cast=none",
    "-O1",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 "
    "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3",
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
]


def stats(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)
    return {
        "iterations": len(ordered),
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def device_intervals(events_json: str) -> list[float]:
    """Per-execution device ms, in execution order (physical-core intervals merged)."""
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
    return [(max(e for _, e in w) - min(s for s, _ in w)) / 1e6
            for _, w in sorted(intervals.items())]


def load_baseline():
    from test.hardware.baselines.moe_prefill_8aa22fa import load

    return load()


class _OneRankGroup:
    """``_resolve_tp_group``'s answer in ``one-rank`` mode: the served group's in-place
    ``all_reduce``, as the functional collective over the one-rank world."""

    def all_reduce(self, tensor):
        import torch.distributed as dist
        from torch.distributed._functional_collectives import all_reduce

        tensor.copy_(all_reduce(tensor, "sum", dist.group.WORLD))


def init_one_rank_world() -> None:
    """A one-rank gloo world, and a separate port for the runtime's communicator."""
    import socket

    import torch.distributed as dist

    if dist.is_initialized():
        return
    ports = []
    for _ in range(2):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(ports[0]),
                      NEURON_RT_ROOT_COMM_ID=f"127.0.0.1:{ports[1]}")
    dist.init_process_group("gloo", rank=0, world_size=1)


def build_variant(model, tokens: int, collectives: str, args):
    """``(graph, inputs, case)`` for one MoE block of ``model`` at ``tokens`` rows."""
    import torch

    from test.vllm_neuron.functional.moe import moe_prefill_case as case_lib

    torch.manual_seed(0)
    case = case_lib.moe_layer(model, device=DEVICE)
    ins = case_lib.block_inputs(case.cfg, tokens, device=DEVICE)
    quant = case_lib.quant_config(model)
    rank = torch.tensor(0, dtype=torch.int64, device=DEVICE)
    resolve = model._resolve_tp_group

    def step(attn_out, streams, post_mix, comb_mix, rank):
        if collectives == "identity":
            return case_lib.block_step(model, case, attn_out, streams, post_mix,
                                       comb_mix, rank, quant)
        import torch.distributed as dist
        from torch.distributed._functional_collectives import all_reduce

        # The attention output's row-parallel reduction: fp32, then the layer's cast.
        attn_out = all_reduce(attn_out.float(), "sum", dist.group.WORLD).to(attn_out.dtype)
        model._resolve_tp_group = _OneRankGroup  # the feed-forward reduction's group
        try:
            return case_lib.block_step(model, case, attn_out, streams, post_mix,
                                       comb_mix, rank, quant)
        finally:
            model._resolve_tp_group = resolve

    graph = torch.compile(step, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                          options={"compiler_args": args.compiler_args})
    inputs = (ins["attn_out"], ins["streams"], ins["post_mix"], ins["comb_mix"], rank)
    return graph, inputs, case


def router_accuracy(model, tokens: int, collectives: str, args) -> dict:
    """Device router logits and selection against the exact router on the same rows."""
    import torch

    from test.vllm_neuron.functional.moe import moe_prefill_case as case_lib
    from test.vllm_neuron.functional.moe.decode_fixtures import index_sets_equal, tie_rows
    from vllm_neuron.functional.moe.router import noaux_tc_correct_torch_oracle

    torch.manual_seed(0)
    case = case_lib.moe_layer(model, device=DEVICE)
    ins = case_lib.block_inputs(case.cfg, tokens, device=DEVICE)
    quant = case_lib.quant_config(model)
    rank = torch.tensor(0, dtype=torch.int64, device=DEVICE)
    resolve = model._resolve_tp_group

    def step(attn_out, streams, post_mix, comb_mix, rank):
        if collectives == "one-rank":
            import torch.distributed as dist
            from torch.distributed._functional_collectives import all_reduce

            attn_out = all_reduce(attn_out.float(), "sum",
                                  dist.group.WORLD).to(attn_out.dtype)
            model._resolve_tp_group = _OneRankGroup
        try:
            return case_lib.block_step_tapped(model, case, attn_out, streams, post_mix,
                                              comb_mix, rank, quant)
        finally:
            model._resolve_tp_group = resolve

    graph = torch.compile(step, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                          options={"compiler_args": args.compiler_args})
    _out, rows, logits, index = (t.cpu() for t in graph(
        ins["attn_out"], ins["streams"], ins["post_mix"], ins["comb_mix"], rank))
    bank = case.layer.mlp.experts
    bias = bank.router_bias.detach().cpu().float()
    exact = case_lib.exact_router(rows, case.layer.post_attention_layernorm_weight.detach().cpu(),
                                  bank.router_weight.detach().cpu(), float(case.cfg.rms_norm_eps))
    exact_index, _ = noaux_tc_correct_torch_oracle(
        exact.float(), bias.unsqueeze(0), bool(case.cfg.norm_topk_prob),
        float(case.cfg.routed_scaling_factor))
    ties = tie_rows(exact.float(), bias, 1e-4)
    other = ~index_sets_equal(index.long(), exact_index.long())
    err = (logits.double() - exact).abs()
    return {
        "logits_rel_l2_vs_exact": float((logits.double() - exact).norm() / exact.norm()),
        "logits_max_abs_err": float(err.max()),
        "logits_mean_abs_err": float(err.mean()),
        "rows_index_set_differs_from_exact": int(other.sum()),
        "of_which_not_tie_rows": int((other & ~ties).sum()),
        "exact_tie_rows_1e-4": int(ties.sum()),
        "rows": int(tokens),
    }


def time_graphs(graphs: dict, inputs: dict, args) -> dict:
    from nrtpy._nrtpy import SystemTraceSession

    names = list(graphs)
    for _ in range(args.warmup):
        for name in names:
            graphs[name](*inputs[name]).to("cpu")
    order, host = [], {n: [] for n in names}
    with SystemTraceSession() as trace:
        for iteration in range(args.iterations):
            rotation = names[iteration % len(names):] + names[:iteration % len(names)]
            for name in rotation:
                t0 = time.perf_counter_ns()
                graphs[name](*inputs[name]).to("cpu")
                host[name].append((time.perf_counter_ns() - t0) / 1e6)
                order.append(name)
        events = trace.fetch_events_json()
    device_all = device_intervals(events)
    if len(device_all) != len(order):
        raise AssertionError(f"system trace has {len(device_all)} executions for "
                             f"{len(order)} calls")
    device = {n: [] for n in names}
    for name, value in zip(order, device_all):
        device[name].append(value)
    return {n: {"device": stats(device[n]), "host": stats(host[n])} for n in names}


def profile_graph(name: str, graph, inputs, args) -> Path:
    import torch
    import libtorch_neuronx_lite.envs as libtorch_envs

    directory = args.profile_dir / name
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    runtime = torch.classes.neuron.Runtime()
    runtime.start_profiling(str(directory), ["device_profile", "system_profile"], None,
                            None, libtorch_envs.get_neuron_compile_cache_dir())
    try:
        for _ in range(args.profile_iterations):
            graph(*inputs).to("cpu")
    finally:
        runtime.stop_profiling()
    return directory


def analyse(name: str, directory: Path, args) -> dict:
    """Ingest one profile and bucket it, whole execution and the goal's window."""
    display = f"moe-prefill-{name}"
    data = args.explorer_data
    global_dir = data / "profiles" / "global"
    for stale in global_dir.glob(f"{display}_*"):
        shutil.rmtree(stale, ignore_errors=True)
    ingest = subprocess.run(
        ["neuron-explorer", "view", "-d", str(directory), "--display-name", display,
         "--ingest-only", "--data-path", str(data)],
        capture_output=True, text=True, timeout=1500)
    if ingest.returncode != 0:
        raise RuntimeError(f"ingest of {directory} failed: {ingest.stderr[-2000:]}")
    out_json = directory / "buckets.json"
    run = subprocess.run(
        [str(DUCKDB_PYTHON), str(Path(__file__).resolve()), "--bucket", display,
         str(out_json), str(global_dir)], capture_output=True, text=True, timeout=1500)
    if run.returncode != 0:
        raise RuntimeError(f"bucketing of {display} failed: {run.stderr[-2000:]}")
    report = json.loads(out_json.read_text())
    window = report["window_buckets_ms"]
    return {
        "profile": display,
        "trace_dir": report["trace_dir"],
        "buckets_json": str(out_json),
        "fully_traced_executions": report["fully_traced_executions"],
        "mean_execution_ms": report["mean_execution_ms"],
        "window_ms": report["mean_window_ms"],
        "window_compiler_ops_ms": float(window.get(COMPILER_OPS, 0.0)),
        "window_buckets_ms": dict(sorted(window.items(), key=lambda kv: -kv[1])),
        "compiler_ops_ms_total": float(report["buckets_ms"].get(COMPILER_OPS, 0.0)),
        "buckets_ms": dict(sorted(report["buckets_ms"].items(), key=lambda kv: -kv[1])),
        "window_opcodes_first_execution": next(
            (r.get("window_opcodes_by_pcore") for r in report["executions_detail"]
             if "window_opcodes_by_pcore" in r), None),
    }


def agreement(after, before) -> dict:
    a, b = after.float(), before.float()
    rows = (a != b).flatten(1).any(-1)
    return {
        "rel_l2_after_vs_before": float((a - b).norm() / b.norm().clamp_min(1e-30)),
        "max_abs_diff": float((a - b).abs().max()),
        "max_abs_before": float(b.abs().max()),
        "rows_differing": int(rows.sum()),
        "rows": int(rows.numel()),
    }


def graph_name(tokens: int, collectives: str, variant: str) -> str:
    return f"t{tokens}_{collectives}_{variant}"


def run_tokens(tokens: int, collectives: str, baseline, live, args) -> dict:
    import torch

    torch._dynamo.reset()
    variants = {"before": baseline.model_fp8, "after": live}
    graphs, inputs, outputs, compile_s = {}, {}, {}, {}
    for variant, model in variants.items():
        t0 = time.time()
        graph, ins, case = build_variant(model, tokens, collectives, args)
        outputs[variant] = graph(*ins).to("cpu")
        compile_s[variant] = time.time() - t0
        graphs[variant] = graph
        inputs[variant] = ins
    # The first-call outputs, for comparisons across modes and against the simulator.
    saved = args.output.with_name(f"{args.output.stem}.t{tokens}_{collectives}.pt")
    torch.save({variant: out.float() for variant, out in outputs.items()}, saved)
    result = {
        "T": tokens,
        "collectives": collectives,
        "outputs_file": str(saved),
        "checkpoint_weights": bool(case.checkpoint),
        "compile_and_first_call_s": compile_s,
        "agreement": agreement(outputs["after"], outputs["before"]),
    }
    result["timing"] = time_graphs(graphs, inputs, args)
    if args.accuracy:
        result["router_accuracy"] = {}
        for variant, model in variants.items():
            torch._dynamo.reset()
            result["router_accuracy"][variant] = router_accuracy(model, tokens, collectives, args)
    if args.profile_dir is not None and collectives in args.profile_collectives:
        result["profiles"] = {}
        for variant, graph in graphs.items():
            directory = profile_graph(graph_name(tokens, collectives, variant), graph,
                                      inputs[variant], args)
            result["profiles"][variant] = {"profile_dir": str(directory)}
    return result


def summarise(cases: list[dict]) -> dict:
    out = {}
    for case in cases:
        t = case["timing"]
        row = {
            "block_median_before_ms": t["before"]["device"]["median_ms"],
            "block_median_after_ms": t["after"]["device"]["median_ms"],
            "block_p90_before_ms": t["before"]["device"]["p90_ms"],
            "block_p90_after_ms": t["after"]["device"]["p90_ms"],
            "block_iterations": t["after"]["device"]["iterations"],
        }
        for variant, acc in case.get("router_accuracy", {}).items():
            row[f"router_logits_rel_l2_vs_exact_{variant}"] = acc["logits_rel_l2_vs_exact"]
            row[f"router_rows_other_set_vs_exact_{variant}"] = (
                acc["rows_index_set_differs_from_exact"])
        profiles = case.get("profiles", {})
        if all("window_compiler_ops_ms" in profiles.get(v, {}) for v in ("before", "after")):
            b = profiles["before"]["window_compiler_ops_ms"]
            a = profiles["after"]["window_compiler_ops_ms"]
            row.update({
                "window_compiler_ops_before_ms": b,
                "window_compiler_ops_after_ms": a,
                "window_after_over_before": a / b if b else None,
                "window_before_ms": profiles["before"]["window_ms"],
                "window_after_ms": profiles["after"]["window_ms"],
                "block_compiler_ops_before_ms": profiles["before"]["compiler_ops_ms_total"],
                "block_compiler_ops_after_ms": profiles["after"]["compiler_ops_ms_total"],
            })
        out[f"T={case['T']} collectives={case['collectives']}"] = row
    return out


def analyse_only(args) -> None:
    report = json.loads(args.output.read_text())
    for case in report["cases"]:
        for variant, entry in case.get("profiles", {}).items():
            name = graph_name(case["T"], case["collectives"], variant)
            entry.update(analyse(name, Path(entry["profile_dir"]), args))
            print(json.dumps({"T": case["T"], "collectives": case["collectives"],
                              "variant": variant,
                              "window_compiler_ops_ms": entry["window_compiler_ops_ms"],
                              "window_ms": entry["window_ms"]}), flush=True)
    report["summary"] = summarise(report["cases"])
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=1), flush=True)


def main() -> None:
    from test.vllm_neuron import artifacts

    profiles = artifacts.campaign_path("glm53f-wt2", "moe-prefill-profiles")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=list(TOKENS))
    parser.add_argument("--collectives", nargs="+", default=list(COLLECTIVES),
                        choices=COLLECTIVES)
    parser.add_argument("--profile-collectives", nargs="*", default=list(COLLECTIVES),
                        choices=COLLECTIVES, help="the modes whose graphs are profiled")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument("--profile-dir", type=Path, default=profiles)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--explorer-data", type=Path, default=profiles / "explorer-data")
    parser.add_argument("--no-accuracy", dest="accuracy", action="store_false",
                        help="skip the router-accuracy graphs")
    parser.add_argument("--analyze-only", action="store_true",
                        help="ingest and bucket the profiles --output names (no device)")
    parser.add_argument("--merge", action="store_true",
                        help="keep the cases already in --output that this run does not redo")
    parser.add_argument("--compiler-args", nargs="*",
                        default=(os.environ.get("NEURON_CC_FLAGS", "").split()
                                 or MODEL_COMPILER_ARGS))
    args = parser.parse_args()
    for name in ("output", "profile_dir", "explorer_data"):
        setattr(args, name, getattr(args, name).resolve())
    if args.analyze_only:
        analyse_only(args)
        return
    if args.no_profile:
        args.profile_dir = None
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under devlease.py, which pins NEURON_RT_VISIBLE_CORES")

    import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
    from vllm_neuron.model.glm5_next import model_fp8 as live

    if not Path(live.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(f"model_fp8 imported from {live.__file__}, not {ROOT}")
    baseline = load_baseline()
    # The compilers write their intermediates into the working directory.
    os.chdir(tempfile.mkdtemp(prefix="moe-prefill-bench-cwd-"))
    report = {
        "environment": {k: os.environ.get(k) for k in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_LIBTORCH_CACHE_ROOT", "VLLM_NEURON_MOE_PREFILL_ROUTER")},
        "device": "neuron:0 (one logical core = 2 physical cores at LNC2)",
        "baseline": {"commit": baseline.commit, "directory": baseline.directory},
        "compiler_args": args.compiler_args,
        "shapes": {"hidden": 4096, "mhc_streams": 4, "tp_world": 64, "ep_degree": 16,
                   "local_experts": 18, "expert_intermediate": 512,
                   "shared_intermediate": 128, "routed_experts": 288, "top_k": 8},
        "collectives": {"one-rank": "the served graph's two all-reduces over a one-rank "
                                    "gloo world (numerically identity)",
                        "identity": "no collective in the graph"},
        "method": __doc__.split("What is measured", 1)[1].split("The compile cache", 1)[0].strip(),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "profile_iterations": args.profile_iterations,
        "cases": [],
    }
    if args.merge and args.output.exists():
        previous = json.loads(args.output.read_text())
        redo = {(t, c) for t in args.tokens for c in args.collectives}
        report["cases"] = [c for c in previous.get("cases", [])
                           if (c["T"], c.get("collectives")) not in redo]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if "one-rank" in args.collectives:
        init_one_rank_world()
    for tokens in args.tokens:
        for collectives in args.collectives:
            t0 = time.time()
            case = run_tokens(tokens, collectives, baseline, live, args)
            case["wall_s"] = time.time() - t0
            report["cases"].append(case)
            report["summary"] = summarise(report["cases"])
            args.output.write_text(json.dumps(report, indent=1) + "\n")
            print(json.dumps({"T": tokens, "collectives": collectives,
                              "agreement": case["agreement"],
                              "router_accuracy": case.get("router_accuracy"),
                              "compile_s": case["compile_and_first_call_s"],
                              "median_ms": {v: case["timing"][v]["device"]["median_ms"]
                                            for v in ("before", "after")}}), flush=True)


if __name__ == "__main__":
    main()
