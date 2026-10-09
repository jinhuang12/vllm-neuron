# SPDX-License-Identifier: Apache-2.0
"""Time the DSA indexer score GEMM at the served prefill shapes, against a baseline commit.

Run it through the device lease, which pins the cores and the LNC; this script does not
select cores::

    devlease.py slice NAME -- python3 test/hardware/benchmark_dsa_score_gemm.py --out FILE

Shapes (``--shapes``): the two served prefill calls of ``_score_gemm_nki``, ``[tokens, cands]``
= ``[1024, 2048]`` (a 1024-token chunk at ``max_model_len`` 8192) and ``[2048, 16384]`` (a
2048-token chunk at 65536), with the module's head count and head dimension. ``cands`` is the
pooled key count, ``max_model_len // index_kpool``.

Variants:

* ``before`` -- ``dsa_score_gemm`` of ``--baseline-commit``'s ``score_gemm.py``, read with
  ``git show`` into a file beside ``--out`` and imported from there under its own module name
  (the kernel cache keys a kernel by its qualified name).
* ``after`` -- this tree's ``dsa_score_gemm``.

Both are the public seam, so each takes the launch production takes (under LNC2 this tree's
seam launches two programs; the baseline's always launches one).

Per-call device time is the slope between an ``R``-call graph and a one-call graph
(``R = --chain``; every call of the long graph reads its own query, so no two calls are the
same computation), ``(device(R) - device(1)) / (R - 1)``, paired per iteration, with device
time from the runtime system trace (the LNC2 physical-core intervals of an execution merged).
``--reps`` rounds each time every graph ``--iterations`` times, the four graphs interleaved in
a rotating order. Reported per variant: the median of each round, the median and spread over
rounds, and the noise floor, the larger relative spread ``(max - min) / median`` of the two
variants' round medians. The one-call graph returns the whole output; the long graph returns
one element per call, so the timed loop copies little.

Numerics: for each of ``--seeds`` seeds and each shape the one-call graphs' outputs are saved
under ``--records``/``numerics`` as ``.pt`` files and compared with ``torch.equal``, and the
``after`` output is also compared with the module's CPU reference ``_dsa_score_gemm_torch``.

Compilation and warmup are excluded. The compile cache is off unless ``--use-compile-cache``:
the graph cache key ignores kernel bodies, so an edited kernel could be timed from a stale NEFF.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

#: This file's repository. Run as a script, Python puts ``test/hardware`` on ``sys.path`` and
#: the venv's own ``vllm_neuron`` is another checkout; the "after" side must be this tree.
#: Bytecode is not written, so a run leaves no ``__pycache__`` in the tree.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
from vllm_neuron.functional.dsa import score_gemm as current  # noqa: E402

DEVICE = "neuron:0"
MODULE_PATH = "vllm_neuron/functional/dsa/score_gemm.py"
#: ``[tokens, cands]`` of the served prefill calls (reports/op_inventory.json, family
#: "DSA attention | _score_gemm_nki #0": a 1024-token chunk at max_model_len 8192, and a
#: 2048-token chunk of a 65534-token prompt at max_model_len 65536).
SHAPES = {"1k": (1024, 2048), "64k": (2048, 16384)}
#: Tokens per slab of the chunked CPU reference: one slab's per-head scores stay < 300 MB.
REFERENCE_SLAB = 128
VARIANTS = ("before", "after")


def load_baseline(commit: str, records: Path):
    """The baseline commit's module, written beside the records and imported from there."""
    source = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{MODULE_PATH}"],
                            check=True, capture_output=True, text=True).stdout
    path = records / "baseline_score_gemm.py"
    path.write_text(source)
    name = f"_score_gemm_baseline_{commit}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load the baseline module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path, hashlib.sha256(source.encode()).hexdigest()


def compiled(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


def operands(tokens: int, cands: int, seed: int, queries: int = 1):
    """``queries`` bf16 queries, one bf16 key and fp32 weights; weights of both signs."""
    gen = torch.Generator().manual_seed(seed)
    heads, head_dim = current.INDEX_N_HEADS, current.INDEX_HEAD_DIM
    qs = [torch.randn((tokens, heads, head_dim), generator=gen).to(torch.bfloat16)
          for _ in range(queries)]
    k = torch.randn((cands, head_dim), generator=gen).to(torch.bfloat16)
    weights = torch.randn((tokens, heads), generator=gen, dtype=torch.float32)
    return qs, k, weights


def one_call(seam):
    def graph(q, k, weights):
        return seam(q, k, weights)
    return graph


def chained_calls(seam, links: int):
    def graph(k, weights, *qs):
        return torch.cat([seam(q, k, weights)[0, 0:1] for q in qs[:links]])
    return graph


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


def time_round(graphs: dict, inputs: dict, iterations: int) -> dict:
    """One round: every graph ``iterations`` times, in a rotating order; device us per call."""
    from nrtpy._nrtpy import SystemTraceSession

    names = list(graphs)
    order = []
    with SystemTraceSession() as trace:
        for iteration in range(iterations):
            shift = iteration % len(names)
            for name in names[shift:] + names[:shift]:
                graphs[name](*inputs[name]).to("cpu")
                order.append(name)
        events = trace.fetch_events_json()
    device = device_intervals(events)
    if len(device) != len(order):
        raise AssertionError(f"system trace has {len(device)} executions for {len(order)} calls")
    samples = {name: [] for name in names}
    for name, value in zip(order, device):
        samples[name].append(value)
    return samples


def spread(values: list[float]) -> dict:
    middle = statistics.median(values)
    return {"median_us": middle, "min_us": min(values), "max_us": max(values),
            "relative_spread": (max(values) - min(values)) / middle}


def reference_scores(q, k, weights) -> torch.Tensor:
    """``_dsa_score_gemm_torch`` over token slabs, so the per-head scores stay small."""
    return torch.cat([current._dsa_score_gemm_torch(q[t:t + REFERENCE_SLAB], k,
                                                    weights[t:t + REFERENCE_SLAB])
                      for t in range(0, q.shape[0], REFERENCE_SLAB)])


def numerics(graphs: dict, tokens: int, cands: int, seed: int, name: str, records: Path) -> dict:
    (q,), k, weights = operands(tokens, cands, seed)
    device_inputs = (q.to(DEVICE), k.to(DEVICE), weights.to(DEVICE))
    outputs, files = {}, {}
    for variant in VARIANTS:
        outputs[variant] = graphs[f"{variant}_k1"](*device_inputs).to("cpu")
        files[variant] = records / f"{name}_seed{seed}_{variant}.pt"
        torch.save(outputs[variant], files[variant])
    reference = reference_scores(q, k, weights)
    residual = (outputs["after"].double() - reference.double())
    differing = int(torch.ne(outputs["after"], outputs["before"]).sum())
    return {
        "seed": seed,
        "after_equals_before": bool(torch.equal(outputs["after"], outputs["before"])),
        "elements_differing": differing,
        "after_finite": bool(torch.isfinite(outputs["after"]).all()),
        "after_vs_torch_max_abs": float(residual.abs().max()),
        "after_vs_torch_relative_l2": float(residual.norm() / reference.double().norm()),
        "torch_max_abs": float(reference.abs().max()),
        "files": {variant: str(path) for variant, path in files.items()},
    }


def shape_case(name: str, seams: dict, args) -> dict:
    torch._dynamo.reset()
    tokens, cands = SHAPES[name]
    heads, head_dim = current.INDEX_N_HEADS, current.INDEX_HEAD_DIM
    links = args.chain
    graphs, inputs = {}, {}
    qs, k, weights = operands(tokens, cands, seed=args.timing_seed, queries=links)
    qd = [q.to(DEVICE) for q in qs]
    kd, wd = k.to(DEVICE), weights.to(DEVICE)
    compile_s = {}
    for variant in VARIANTS:
        for key, graph, graph_inputs in (
                (f"{variant}_k1", one_call(seams[variant]), (qd[0], kd, wd)),
                (f"{variant}_k{links}", chained_calls(seams[variant], links), (kd, wd, *qd))):
            graphs[key] = compiled(graph)
            inputs[key] = graph_inputs
            started = time.perf_counter()
            graphs[key](*graph_inputs).to("cpu")
            compile_s[key] = time.perf_counter() - started
    current.reset_score_gemm_dispatch_counters()
    for _ in range(args.warmup):
        for key in graphs:
            graphs[key](*inputs[key]).to("cpu")
    rounds = {variant: [] for variant in VARIANTS}
    for _ in range(args.reps):
        samples = time_round(graphs, inputs, args.iterations)
        for variant in VARIANTS:
            pairs = [(long - one) / (links - 1) for long, one in
                     zip(samples[f"{variant}_k{links}"], samples[f"{variant}_k1"])]
            rounds[variant].append({
                "per_call_median_us": statistics.median(pairs),
                "one_call_graph_median_us": statistics.median(samples[f"{variant}_k1"]),
                f"k{links}_graph_median_us": statistics.median(samples[f"{variant}_k{links}"]),
            })
    records = args.records / "numerics"
    records.mkdir(parents=True, exist_ok=True)
    checks = [numerics(graphs, tokens, cands, seed, name, records) for seed in args.seeds]
    flops = 2 * tokens * heads * cands * head_dim
    nbytes = (tokens * heads * head_dim * 2 + cands * head_dim * 2 + tokens * heads * 4
              + tokens * cands * 4)
    case = {"shape": name, "tokens": tokens, "heads": heads, "cands": cands,
            "head_dim": head_dim, "chain_links": links, "compile_and_first_call_s": compile_s,
            "flops_per_call": flops, "bytes_per_call": nbytes,
            "after_programs": current._programs(tokens), "numerics": checks}
    for variant in VARIANTS:
        per_round = [r["per_call_median_us"] for r in rounds[variant]]
        case[variant] = {"rounds": rounds[variant], "per_call": spread(per_round),
                         "achieved_tflops": flops / (statistics.median(per_round) * 1e6)}
    case["noise_floor_relative"] = max(case[v]["per_call"]["relative_spread"] for v in VARIANTS)
    case["speedup"] = case["before"]["per_call"]["median_us"] / case["after"]["per_call"]["median_us"]
    if args.profile_dir is not None:
        case["profile_dir"] = capture_profile(graphs, inputs, [f"{v}_k1" for v in VARIANTS],
                                              args.profile_dir / name)
    return case


def capture_profile(graphs: dict, inputs: dict, names, directory: Path) -> dict:
    """One device profile per graph, each of one execution, in ``directory / name``.

    ``neuron-explorer view`` reads a profile against the NEFF of its first execution, so
    every graph gets a session of its own.
    """
    import libtorch_neuronx_lite.envs as libtorch_envs

    runtime = torch.classes.neuron.Runtime()
    out = {}
    for name in names:
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)
        runtime.start_profiling(str(target), ["device_profile", "system_profile"], None, None,
                                libtorch_envs.get_neuron_compile_cache_dir())
        try:
            graphs[name](*inputs[name]).to("cpu")
        finally:
            runtime.stop_profiling()
        out[name] = str(target)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="the JSON report")
    parser.add_argument("--records", type=Path,
                        help="directory for the baseline source and the .pt outputs "
                             "(default: the report's directory)")
    parser.add_argument("--baseline-commit", default="f3a833f")
    parser.add_argument("--shapes", nargs="+", choices=sorted(SHAPES), default=list(SHAPES))
    parser.add_argument("--chain", type=int, default=3, help="calls in the long graph")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10, help="timed calls per graph per round")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--timing-seed", type=int, default=7)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--use-compile-cache", action="store_true")
    parser.add_argument("--time-limit", type=int, default=5400, help="seconds before the run aborts")
    args = parser.parse_args()
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise ValueError("Hardware benchmark cannot run in CPU mode or the NKI simulator")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    if not Path(current.__file__).resolve().is_relative_to(ROOT):
        raise ValueError(f"imported {current.__name__} from {current.__file__}, not {ROOT}")
    if args.chain < 2 or args.reps < 1 or args.iterations < 1 or args.warmup < 0:
        raise ValueError("Use --chain >= 2 and positive rounds and iterations")
    signal.alarm(args.time_limit)
    if not args.use_compile_cache:
        os.environ["NEURON_LIBTORCH_DISABLE_COMPILE_CACHE"] = "1"
    args.out = args.out.resolve()
    args.records = (args.records or args.out.parent).resolve()
    args.records.mkdir(parents=True, exist_ok=True)
    if args.profile_dir is not None:
        args.profile_dir = args.profile_dir.resolve()
    baseline, baseline_path, baseline_sha = load_baseline(args.baseline_commit, args.records)
    seams = {"before": baseline.dsa_score_gemm, "after": current.dsa_score_gemm}
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", MODULE_PATH],
                           check=True, capture_output=True, text=True).stdout.strip()
    # The NKI and neuronx-cc drivers write artifacts into the working directory.
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or args.records) / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = {
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT",
            "NEURON_LIBTORCH_DISABLE_COMPILE_CACHE")},
        "tree": {"root": str(ROOT), "head": head, "module_modified_vs_head": bool(dirty),
                 "module": current.__file__},
        "baseline": {"commit": args.baseline_commit, "file": str(baseline_path),
                     "sha256": baseline_sha},
        "args": {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items()},
        "method": "per-call = (R-call graph - 1-call graph) / (R - 1), paired per iteration; "
                  "device time from the runtime system trace, LNC2 core intervals merged; "
                  "graphs interleaved in a rotating order",
        "cases": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name in args.shapes:
        case = shape_case(name, seams, args)
        report["cases"].append(case)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"shape": name,
                          "before_us": case["before"]["per_call"]["median_us"],
                          "after_us": case["after"]["per_call"]["median_us"],
                          "speedup": case["speedup"],
                          "noise_floor": case["noise_floor_relative"],
                          "bit_equal": [c["after_equals_before"] for c in case["numerics"]]}),
              flush=True)


if __name__ == "__main__":
    main()
