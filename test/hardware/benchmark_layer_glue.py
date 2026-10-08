# SPDX-License-Identifier: Apache-2.0
"""Compiler-op glue of one decoder layer's decode step: this tree against 0a08ff4.

Run it through the device lease (``devlease.py slice mhc``), which pins the cores and the
LNC; this script does not select cores.

What runs. One decoder layer of ``Glm5NextModel.forward`` at decode, built by
``test/vllm_neuron/functional/glue/glue_case.py`` from either model tree with the same
weights and operands: the layer forward (its attention-half mHC site) and the
feed-forward mHC site around ``Glm5NextModel._ffn_half``. Two layers, each its own graph:

* ``kda``: checkpoint layer 4, linear attention (one head per rank) + MoE.
* ``dsa``: checkpoint layer 3, sparse attention (one head per rank, ctx 1024 in the
  2048-row decode window, so the dense bypass serves it as on the served lines) + MoE.

Shapes are one rank's TP=64 EP=16 shard: hidden 4096, 4 mHC streams, KDA 1 head x 128,
MLA 1 head, 18 local experts at I=512, shared expert I=128, bf16/fp8 as stored. No process
group is initialised, so both row-parallel reductions are the identity. The carriers are
the runner's at 0a08ff4: one bank view per request. Compiled through ``neuron_libtorch``
with the served neuronx-cc arguments.

What is measured, per (layer, B, variant):

1. Device time per execution from the runtime system trace (``nc_exec_running``, the two
   physical cores' intervals merged), ``--iterations`` timed calls after ``--warmup``,
   before/after interleaved. Median and p90.
2. A device profile of ``--profile-iterations`` calls, ingested with ``neuron-explorer``
   and bucketed with the rules of ``/home/ubuntu/glm53f-wt2/profile/breakdown.py`` (the
   serving profile's own tool, frozen in :func:`bucket_main`): every 2 us sample is labelled per physical core, and compiler-op
   (unnamed XLA) time is owned by the last NKI kernel that ran before it on that core.
   Kernel labels are normalised to the live tree's file names, so the 0a08ff4 snapshot's
   kernels carry the same owner names as the served profile.

Owners. The goal's owners are the five kernels the serving profile charges the glue to:
``mhc/hyper_connection.py``, ``mhc/sinkhorn.py``, ``kda/fused_decode.py``,
``attention/mla_projections.py`` and ``moe/expert_decode.py``. A kernel this branch adds
sits inside one of their gaps and takes over the glue that is left there, so its owned
time is charged back to the owner whose gap it sits in (:data:`NEW_OWNER_CHARGED_TO`).
The total compiler-op time over every owner is reported too, so nothing moves out of
sight by changing hands.

The compile cache is used (``NEURON_LIBTORCH_CACHE_ROOT``): its key ignores kernel
bodies, so give it a fresh root after a kernel edit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time


def bucket_main(name: str, out_path: str, global_dir: str) -> None:
    """Bucket one ingested profile; runs under a python that has duckdb (no torch).

    The sample rules of ``/home/ubuntu/glm53f-wt2/profile/breakdown.py`` (worker-13's
    serving-profile tool, as of 2026-10-07 02:00Z), frozen here so a later edit of that
    file cannot move this benchmark's numbers: 2 us samples; per physical core the
    highest-priority instruction covering the sample (NKI kernel by source file / unnamed
    compiler op > DMA issue > semaphore > core barrier); a sample where both cores compute
    is split 0.5/0.5, one core computing gets 1.0; a compiler op is owned by the last NKI
    kernel that ran before it on the same core. Only executions traced to the end on both
    cores are used. One change to the frozen rules: a table the profile does not carry (a
    one-layer graph has no collectives, so no ``CcOp``) is read as empty.
    """
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
           "ELSE 'compiler ops (unnamed XLA)' END")
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
    per = collections.defaultdict(lambda: collections.defaultdict(float))
    owned = collections.defaultdict(lambda: collections.defaultdict(float))
    core_act = collections.defaultdict(lambda: collections.defaultdict(float))
    waits = ("CASE WHEN incc THEN 'wait: collective' WHEN indma THEN 'wait: DMA in flight' "
             "WHEN least(p0, p1) = 4 THEN 'wait: semaphore' "
             "WHEN least(p0, p1) = 5 THEN 'wait: core barrier' "
             "WHEN least(p0, p1) = 3 THEN 'dma issue' ELSE 'idle (nothing traced)' END")
    for idx, s0, e0 in traced:
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
        for label, w in c.sql(
                "SELECT lab, sum(w) FROM ("
                "SELECT l0 lab, 0.5 w FROM s WHERE p0 <= 2 AND p1 <= 2 "
                "UNION ALL SELECT l1, 0.5 FROM s WHERE p0 <= 2 AND p1 <= 2 "
                "UNION ALL SELECT l0, 1.0 FROM s WHERE p0 <= 2 AND p1 > 2 "
                "UNION ALL SELECT l1, 1.0 FROM s WHERE p1 <= 2 AND p0 > 2 "
                f"UNION ALL SELECT {waits}, 1.0 FROM s WHERE p0 > 2 AND p1 > 2) "
                "GROUP BY 1").fetchall():
            per[label][idx] = float(w) * step / 1e6
        per["__step_ms"][idx] = (e0 - s0) / 1e6
        for core, label, n in c.sql(
                "SELECT 0, l0, count(*) FROM s WHERE p0 <= 2 GROUP BY 2 "
                "UNION ALL SELECT 1, l1, count(*) FROM s WHERE p1 <= 2 GROUP BY 2").fetchall():
            core_act[(core, label)][idx] = n * step / 1e6
        c.sql("CREATE OR REPLACE TABLE f AS SELECT core, ts, lab, pri, "
              "last_value(CASE WHEN pri = 1 THEN lab END IGNORE NULLS) OVER "
              "(PARTITION BY core ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) "
              "prev FROM (SELECT 0 core, ts, l0 lab, p0 pri FROM s "
              "UNION ALL SELECT 1, ts, l1, p1 FROM s)")
        c.sql("CREATE OR REPLACE TABLE o AS SELECT ts, core, prev, "
              "last_value(CASE WHEN pri = 1 THEN ts END IGNORE NULLS) OVER "
              "(PARTITION BY core ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) "
              "prev_ts FROM f")
        c.sql("CREATE OR REPLACE TABLE s2 AS SELECT s.*, a.prev pv0, b.prev pv1, "
              "CASE WHEN coalesce(a.prev_ts, -1) >= coalesce(b.prev_ts, -1) THEN a.prev "
              "ELSE b.prev END pvany FROM s JOIN o a ON a.ts = s.ts AND a.core = 0 "
              "JOIN o b ON b.ts = s.ts AND b.core = 1")
        for label, own, w in c.sql(
                "SELECT lab, own, sum(w) FROM ("
                "SELECT l0 lab, pv0 own, 0.5 w FROM s2 WHERE p0 = 2 AND p1 <= 2 "
                "UNION ALL SELECT l1, pv1, 0.5 FROM s2 WHERE p1 = 2 AND p0 <= 2 "
                "UNION ALL SELECT l0, pv0, 1.0 FROM s2 WHERE p0 = 2 AND p1 > 2 "
                "UNION ALL SELECT l1, pv1, 1.0 FROM s2 WHERE p1 = 2 AND p0 > 2 "
                f"UNION ALL SELECT {waits}, pvany, 1.0 FROM s2 WHERE p0 > 2 AND p1 > 2) "
                "GROUP BY 1, 2").fetchall():
            owned[(label, own or "<step start>")][idx] = float(w) * step / 1e6
    steps = [i for i, *_ in traced]
    n = max(len(steps), 1)

    def mean(d):
        return sum(d.get(i, 0.0) for i in steps) / n

    labels = sorted({label for (_, label) in core_act})
    Path(out_path).write_text(json.dumps(dict(
        trace_dir=path, executions=len(execs), fully_traced_steps=steps,
        mean_step_ms=mean(per["__step_ms"]),
        buckets_ms={k: mean(v) for k, v in per.items() if k != "__step_ms"},
        bucket_by_owner_ms={f"{k[0]} | after {k[1]}": mean(v) for k, v in owned.items()},
        per_label_cores=[dict(label=label, active_ms_pcore0=mean(core_act[(0, label)]),
                              active_ms_pcore1=mean(core_act[(1, label)]))
                         for label in labels],
    ), indent=1))


if __name__ == "__main__" and sys.argv[1:2] == ["--bucket"]:
    bucket_main(*sys.argv[2:5])
    sys.exit(0)

#: The worktree root, ahead of any installed copy (the lease command sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
from vllm_neuron.model.glm5_next import model_fp8 as live_model  # noqa: E402
from test.vllm_neuron.functional.glue import glue_case as case_lib  # noqa: E402

DEVICE = "neuron:0"
FAMILIES = ("kda", "dsa")
BATCHES = (1, 64)

#: The served model's neuronx-cc arguments (``benchmark_moe_decode.MODEL_COMPILER_ARGS``).
MODEL_COMPILER_ARGS = [
    "--auto-cast=none",
    "-O1",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 "
    "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3",
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
]

#: The goal's owners, as the serving profile labels them.
TARGET_OWNERS = (
    "mhc/hyper_connection.py",
    "mhc/sinkhorn.py",
    "kda/fused_decode.py",
    "attention/mla_projections.py",
    "moe/expert_decode.py",
)
#: The owners the goal names at each batch size.
TARGET_OWNERS_AT = {
    1: TARGET_OWNERS,
    64: ("mhc/hyper_connection.py", "attention/mla_projections.py", "moe/expert_decode.py"),
}
#: A kernel this branch adds -> the owner whose gap it sits in. Also ``<step start>``:
#: the glue before the graph's first kernel (the attention site's mHC pre), which in the
#: served stack follows the previous layer's last kernel, its FFN-site combine
#: (``mhc/hyper_connection.py``), and is charged there, as the serving profile does.
NEW_OWNER_CHARGED_TO = {
    "<step start>": "mhc/hyper_connection.py",
    "glue/mhc_pre.py": "mhc/hyper_connection.py",
    "glue/kda_projections.py": "mhc/sinkhorn.py",
    "glue/kda_output.py": "kda/fused_decode.py",
}
COMPILER_OPS = "compiler ops (unnamed XLA)"
DUCKDB_PYTHON = Path(os.environ.get("GLUE_DUCKDB_PYTHON", "/tmp/pqvenv/bin/python"))
SNAPSHOT_LABEL = re.compile(r"^.*/glue_0a08ff4_snapshot/")
SNAPSHOT_FILES = {
    "hyper_connection.py": "mhc/hyper_connection.py",
    "sinkhorn.py": "mhc/sinkhorn.py",
    "fused_decode.py": "kda/fused_decode.py",
    "mla_projections.py": "attention/mla_projections.py",
}


def load_baseline(directory: Path):
    """The 0a08ff4 model module from the snapshot package at ``directory``."""
    import importlib.util

    init = directory / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "glue_0a08ff4_loader", init, submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load baseline package {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["glue_0a08ff4_loader"] = module
    spec.loader.exec_module(module)
    return module.load()


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


def compiled(fn, args):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": args.compiler_args})


def _flatten(carriers: dict):
    """``(names, tensors, statics)``: every tensor or tuple of tensors becomes inputs."""
    names, tensors, statics = [], [], {}
    for key, value in carriers.items():
        if key == "banks":
            continue
        if isinstance(value, tuple):
            names.append((key, len(value)))
            tensors.extend(value)
        elif torch.is_tensor(value):
            names.append((key, None))
            tensors.append(value)
        else:
            statics[key] = value
    return names, tensors, statics


def _unflatten(names, tensors, statics) -> dict:
    out, at = dict(statics), 0
    for key, count in names:
        if count is None:
            out[key] = tensors[at]
            at += 1
        else:
            out[key] = tuple(tensors[at:at + count])
            at += count
    return out


def build_variant(model, family: str, batch: int, args):
    """``(graph, inputs, banks)`` for one layer of ``model`` at ``batch`` requests."""
    torch.manual_seed(0)
    build = case_lib.kda_layer if family == "kda" else case_lib.dsa_layer
    case = build(model, device=DEVICE)
    carriers = (case_lib.kda_carriers if family == "kda" else case_lib.dsa_carriers)(
        case, batch, device=DEVICE)
    quant = case_lib.quant_config(model)
    names, tensors, statics = _flatten(carriers)
    rank = torch.tensor(0, dtype=torch.int64, device=DEVICE)

    def step(streams, rank, *flat):
        return case_lib.layer_step(model, case, streams, _unflatten(names, flat, statics),
                                   rank, quant)

    streams = case_lib.streams_input(case.cfg, batch, device=DEVICE)
    return compiled(step, args), (streams, rank, *tensors), carriers["banks"], case


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


def _normalise_owner(label: str) -> str:
    label = label.split(" | after ", 1)[-1]
    if SNAPSHOT_LABEL.match(label):
        return SNAPSHOT_FILES.get(SNAPSHOT_LABEL.sub("", label), label)
    return label


def analyse(name: str, directory: Path, args) -> dict:
    """Ingest one profile and bucket its compiler-op time by owner (:func:`bucket_main`)."""
    display = f"glue-{name}"
    data = args.explorer_data
    global_dir = data / "profiles" / "global"
    for stale in global_dir.glob(f"{display}*"):
        shutil.rmtree(stale, ignore_errors=True)
    # The profile's top directory, which holds the trace and the ``neffs/`` the runtime
    # copied (the serving tools' ``-d rank_NN`` form).
    source = directory
    ingest = subprocess.run(
        ["neuron-explorer", "view", "-d", str(source), "--display-name", display,
         "--ingest-only", "--data-path", str(data)],
        capture_output=True, text=True, timeout=900)
    if ingest.returncode != 0:
        raise RuntimeError(f"ingest of {source} failed: {ingest.stderr[-2000:]}")
    out_json = directory / "buckets.json"
    run = subprocess.run(
        [str(DUCKDB_PYTHON), str(Path(__file__).resolve()), "--bucket", display,
         str(out_json), str(global_dir)], capture_output=True, text=True, timeout=900)
    if run.returncode != 0:
        raise RuntimeError(f"bucketing of {display} failed: {run.stderr[-2000:]}")
    report = json.loads(out_json.read_text())
    owners: dict[str, float] = {}
    for key, ms in report["bucket_by_owner_ms"].items():
        if not key.startswith(COMPILER_OPS):
            continue
        owner = _normalise_owner(key)
        owners[owner] = owners.get(owner, 0.0) + float(ms)
    charged = {}
    for owner, ms in owners.items():
        target = NEW_OWNER_CHARGED_TO.get(owner, owner)
        charged[target] = charged.get(target, 0.0) + ms
    buckets = report["buckets_ms"]
    return {
        "profile": display,
        "trace_dir": report["trace_dir"],
        "fully_traced_executions": report["fully_traced_steps"],
        "mean_execution_ms": report["mean_step_ms"],
        "compiler_ops_ms_total": float(buckets.get(COMPILER_OPS, 0.0)),
        "compiler_ops_ms_by_owner": dict(sorted(owners.items(), key=lambda kv: -kv[1])),
        "compiler_ops_ms_by_owner_charged": dict(
            sorted(charged.items(), key=lambda kv: -kv[1])),
        "buckets_ms": buckets,
        "per_label_cores": report["per_label_cores"],
    }


def rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.float(), want.float()
    return float((got - want).norm() / want.norm().clamp_min(1e-30))


def run_case(family: str, batch: int, baseline, args) -> dict:
    torch._dynamo.reset()
    variants = {"before": baseline.model_fp8, "after": live_model}
    graphs, inputs, outputs = {}, {}, {}
    compile_s = {}
    for variant, model in variants.items():
        t0 = time.time()
        graph, ins, _banks, case = build_variant(model, family, batch, args)
        # First call compiles; the banks are written in place, so the agreement check
        # reads the first call's output from fresh operands in both trees.
        outputs[variant] = graph(*ins).to("cpu")
        compile_s[variant] = time.time() - t0
        graphs[f"{family}_b{batch}_{variant}"] = graph
        inputs[f"{family}_b{batch}_{variant}"] = ins
    # The first-call outputs, for a check against the simulator on the same operands.
    saved = args.output.with_name(f"{args.output.stem}.{family}_b{batch}.pt")
    torch.save({variant: out.float() for variant, out in outputs.items()}, saved)
    result = {
        "family": family,
        "B": batch,
        "outputs_file": str(saved),
        "checkpoint_weights": bool(case.checkpoint),
        "compile_and_first_call_s": compile_s,
        "agreement": {
            "rel_l2_after_vs_before": rel_l2(outputs["after"], outputs["before"]),
            "max_abs_diff": float((outputs["after"].float()
                                   - outputs["before"].float()).abs().max()),
            "max_abs_before": float(outputs["before"].float().abs().max()),
        },
    }
    timing = time_graphs(graphs, inputs, args)
    result["timing"] = {name.rsplit("_", 1)[-1]: value for name, value in timing.items()}
    if args.profile_dir is not None:
        result["profiles"] = {}
        for name, graph in graphs.items():
            directory = profile_graph(name, graph, inputs[name], args)
            entry = {"profile_dir": str(directory)}
            if args.analyze:
                entry.update(analyse(name, directory, args))
            result["profiles"][name.rsplit("_", 1)[-1]] = entry
    return result


def summarise(cases: list[dict]) -> dict:
    """Per B: the targeted owners' compiler-op ms, before and after, summed over layers."""
    out = {}
    for batch in sorted({c["B"] for c in cases}):
        rows = [c for c in cases if c["B"] == batch and "profiles" in c
                and all("compiler_ops_ms_by_owner_charged" in c["profiles"][v]
                        for v in ("before", "after"))]
        if not rows:
            continue
        targets = TARGET_OWNERS_AT.get(batch, TARGET_OWNERS)
        per_owner = {}
        for owner in TARGET_OWNERS:
            before = sum(c["profiles"]["before"]["compiler_ops_ms_by_owner_charged"]
                         .get(owner, 0.0) for c in rows)
            after = sum(c["profiles"]["after"]["compiler_ops_ms_by_owner_charged"]
                        .get(owner, 0.0) for c in rows)
            per_owner[owner] = {"before_ms": before, "after_ms": after}
        before_t = sum(per_owner[o]["before_ms"] for o in targets)
        after_t = sum(per_owner[o]["after_ms"] for o in targets)
        total_b = sum(c["profiles"]["before"]["compiler_ops_ms_total"] for c in rows)
        total_a = sum(c["profiles"]["after"]["compiler_ops_ms_total"] for c in rows)
        sub_b = sum(c["timing"]["before"]["device"]["median_ms"] for c in rows)
        sub_a = sum(c["timing"]["after"]["device"]["median_ms"] for c in rows)
        out[str(batch)] = {
            "layers": [c["family"] for c in rows],
            "targeted_owners": list(targets),
            "per_owner_ms": per_owner,
            "targeted_before_ms": before_t,
            "targeted_after_ms": after_t,
            "targeted_after_over_before": after_t / before_t if before_t else None,
            "all_compiler_ops_before_ms": total_b,
            "all_compiler_ops_after_ms": total_a,
            "sub_block_median_before_ms": sub_b,
            "sub_block_median_after_ms": sub_a,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-module", type=Path, required=True,
                        help="test/hardware/baselines/glue_0a08ff4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument("--profile-dir", type=Path,
                        default=Path("/home/ubuntu/glm53f-wt2/glue-profiles"))
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--explorer-data", type=Path,
                        default=Path("/home/ubuntu/glm53f-wt2/glue-profiles/explorer-data"))
    parser.add_argument("--no-analyze", dest="analyze", action="store_false")
    parser.add_argument("--merge", action="store_true",
                        help="keep the cases already in --output that this run does not redo")
    parser.add_argument("--compiler-args", nargs="*",
                        default=(os.environ.get("NEURON_CC_FLAGS", "").split()
                                 or MODEL_COMPILER_ARGS))
    args = parser.parse_args()
    if args.no_profile:
        args.profile_dir = None
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under devlease.py, which pins NEURON_RT_VISIBLE_CORES")
    if not Path(live_model.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(f"model_fp8 imported from {live_model.__file__}, not {ROOT}")
    baseline = load_baseline(args.baseline_module)
    # The compilers write their intermediates (KLIR binaries, neuronxcc-* folders) into
    # the working directory: run in a scratch one, not the worktree.
    for name in ("output", "profile_dir", "explorer_data"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    os.chdir(tempfile.mkdtemp(prefix="glue-bench-cwd-"))
    report = {
        "environment": {k: os.environ.get(k) for k in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_LIBTORCH_CACHE_ROOT", "NEURON_LIBTORCH_DISABLE_COMPILE_CACHE")},
        "device": "neuron:0 (one logical core = 2 physical cores at LNC2)",
        "baseline": {"commit": baseline.commit, "directory": baseline.directory},
        "compiler_args": args.compiler_args,
        "shapes": {
            "hidden": 4096, "mhc_streams": 4, "tp_world": case_lib.TP_WORLD,
            "ep_degree": case_lib.EP_DEGREE, "kda_heads_per_rank": 1, "kda_head_dim": 128,
            "mla_heads_per_rank": 1, "local_experts": 18, "expert_intermediate": 512,
            "shared_intermediate": 128, "dsa_context": 1024, "dsa_window_rows": 2048,
            "weights": "mHC, norms, KDA, router from the checkpoint (rank 0 rows); "
                       "expert banks and DSA attention random",
        },
        "method": __doc__.split("What is measured", 1)[1].split("Owners.", 1)[0].strip(),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "profile_iterations": args.profile_iterations,
        "cases": [],
    }
    if args.merge and args.output.exists():
        previous = json.loads(args.output.read_text())
        redo = {(f, b) for f in args.families for b in args.batches}
        report["cases"] = [c for c in previous.get("cases", [])
                           if (c["family"], c["B"]) not in redo]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        for family in args.families:
            t0 = time.time()
            case = run_case(family, batch, baseline, args)
            case["wall_s"] = time.time() - t0
            report["cases"].append(case)
            report["summary"] = summarise(report["cases"])
            args.output.write_text(json.dumps(report, indent=1) + "\n")
            line = {"family": family, "B": batch, "agreement": case["agreement"],
                    "median_ms": {v: case["timing"][v]["device"]["median_ms"]
                                  for v in ("before", "after")}}
            if "profiles" in case and args.analyze:
                line["compiler_ops_ms"] = {
                    v: case["profiles"][v]["compiler_ops_ms_by_owner_charged"]
                    for v in ("before", "after")}
            print(json.dumps(line), flush=True)
    print(json.dumps(report.get("summary", {}), indent=1), flush=True)


if __name__ == "__main__":
    main()
