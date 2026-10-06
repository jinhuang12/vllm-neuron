# SPDX-License-Identifier: Apache-2.0
"""Per-site device time of the decode-path mHC kernels against their 5938748 versions.

One mHC site is what one layer call issues at decode: one
``sinkhorn_normalise_blocks`` (``mhc_pre``) and one ``hyper_connection_combine``
(``mhc_post``), 90 sites per decode step. Shapes are the target's: hidden 4096,
4 residual streams, 20 Sinkhorn iterations, fp32 at the kernel boundary (the
layer casts the bf16 carrier to fp32 before both calls), ``B`` tokens.

Method. Each kernel is timed in its own dependency chain inside one compiled
graph, ``K`` calls long: Sinkhorn feeds its output back as the next call's input
(a doubly stochastic matrix is a valid positive affinity), and the combine feeds
its output streams back as the next call's residual. A chain cannot overlap one
call with the next, which matches the layer, where each site's combine needs that
site's Sinkhorn and the next site's Sinkhorn needs this site's combine (through the
projection). Two chain lengths are compiled, ``K = 1`` and ``K = --chain``; the
per-call device time is the slope ``(t_K - t_1) / (K - 1)``, which removes the
fixed graph-launch cost that a single call would carry. Device time is read off
the runtime's system trace (``nc_exec_running`` start/stop per physical core,
merged per execution), so host dispatch and the output copy are excluded; host
wall time is reported too. Per-site time = Sinkhorn slope + combine slope.

Select the cores in the environment (``devlease.py slice``); this script does not
pick hardware. Every compiled chain is checked against a CPU reference before it
is timed, and the new kernels against the old ones on device.

The compile cache is off unless ``--use-compile-cache`` is given. Its graph key
holds a kernel's module-qualified name, grid and shapes but not its compiled body,
and its NKI key hashes only the kernel function's own source, not the helpers it
calls or the module constants it reads; with it on, an edited kernel can be timed
from a stale NEFF. The "after" side is always this file's repository: its root
goes first on ``sys.path``, and the import is checked.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

#: This file's repository. Run as a script, Python puts ``test/hardware`` on
#: ``sys.path`` rather than the working directory, and the venv then resolves
#: ``vllm_neuron`` to another checkout; the "after" side must be this tree.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
import libtorch_neuronx_lite.envs as libtorch_envs  # noqa: E402

from vllm_neuron.functional.mhc import hyper_connection as new_combine  # noqa: E402
from vllm_neuron.functional.mhc import sinkhorn as new_sinkhorn  # noqa: E402

HIDDEN = 4096
STREAMS = new_sinkhorn.MHC_STREAMS
ITERS = new_sinkhorn.SINKHORN_ITERS
HC_EPS = 1e-6
SITES_PER_STEP = 90


def load_baseline(directory: Path):
    """Import ``sinkhorn.py`` and ``hyper_connection.py`` from a snapshot directory.

    The module name carries the directory name. The compile cache keys a graph by
    the kernel's module-qualified name, not by its compiled body, so two snapshot
    directories imported under one name would share each other's cached NEFFs.
    """
    tag = re.sub(r"\W", "_", directory.resolve().name)
    modules = {}
    for name in ("sinkhorn", "hyper_connection"):
        path = directory / f"{name}.py"
        module_name = f"_mhc_baseline_{tag}_{name}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load baseline snapshot {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["sinkhorn"], modules["hyper_connection"]


def stats_us(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)
    us = [value * 1000.0 for value in ordered]
    return {
        "iterations": len(us),
        "median_us": statistics.median(us),
        "mean_us": statistics.mean(us),
        "p90_us": us[min(len(us) - 1, int(0.9 * len(us)))],
        "min_us": us[0],
        "max_us": us[-1],
    }


def device_intervals(events_json: str) -> list[float]:
    """Per-execution device time in ms, in execution order.

    Physical-core intervals of one LNC2 execution are merged (first start to last
    stop), the same reduction ``benchmark_sparse_mla_mhc.py`` uses.
    """
    starts = {}
    intervals: dict[int, list[tuple[int, int]]] = {}
    for event in json.loads(events_json)["events"]:
        if event["event_type"] != "nc_exec_running":
            continue
        key = (event["nc_idx"], event["tracking_id"])
        if event["phase"] == "start":
            starts[key] = event
        elif event["phase"] == "stop":
            start = starts.pop(key)
            execution = start["data"]["exec_id"]
            intervals.setdefault(execution, []).append(
                (start["data"]["nc_timestamp_ns"], event["data"]["nc_timestamp_ns"])
            )
    out = []
    for execution in sorted(intervals):
        windows = intervals[execution]
        out.append(
            (max(end for _, end in windows) - min(begin for begin, _ in windows))
            / 1_000_000
        )
    return out


def site_inputs(batch: int):
    g = torch.Generator().manual_seed(90 + batch)
    x = torch.randn((batch, HIDDEN), generator=g, dtype=torch.float32)
    residual = torch.randn((batch, STREAMS, HIDDEN), generator=g, dtype=torch.float32)
    post = 2.0 * torch.rand((batch, STREAMS, 1), generator=g, dtype=torch.float32)
    logits = 2.0 * torch.randn((batch, STREAMS, STREAMS), generator=g)
    comb_start = torch.softmax(logits, dim=-1) + HC_EPS
    return x, residual, post, comb_start


def sinkhorn_chain(entry, length: int):
    def chain(comb):
        for _ in range(length):
            comb = entry(comb, iters=ITERS)
        return comb

    return chain


def combine_chain(entry, length: int):
    def chain(x, residual, post, comb):
        for _ in range(length):
            residual = entry(x, residual, post, comb)
        return residual

    return chain


def cpu_sinkhorn_chain(comb, length: int):
    """Per-block Sinkhorn in float64 torch: row pass then column pass, 20 times."""
    eps = new_sinkhorn.SINKHORN_DENOM_EPS
    work = comb.to(torch.float64)
    for _ in range(length * ITERS):
        work = work / (work.sum(dim=-1, keepdim=True) + eps)
        work = work / (work.sum(dim=-2, keepdim=True) + eps)
    return work.to(torch.float32)


def cpu_combine_chain(x, residual, post, comb, length: int):
    for _ in range(length):
        residual = new_combine.hyper_connection_torch_oracle(x, residual, post, comb)
    return residual


def compile_fn(fn, args):
    return torch.compile(
        fn,
        backend="neuron_libtorch",
        fullgraph=True,
        dynamic=False,
        options={"compiler_args": args.compiler_args},
    )


def time_graphs(graphs: dict, inputs: dict, args) -> dict:
    """Interleave every graph's timed calls; return host and device samples."""
    from nrtpy._nrtpy import SystemTraceSession

    names = list(graphs)
    for _ in range(args.warmup):
        for name in names:
            graphs[name](*inputs[name]).to("cpu")
    host = {name: [] for name in names}
    order = []
    with SystemTraceSession() as trace:
        for iteration in range(args.iterations):
            rotation = names[iteration % len(names):] + names[:iteration % len(names)]
            for name in rotation:
                started = time.perf_counter_ns()
                graphs[name](*inputs[name]).to("cpu")
                host[name].append((time.perf_counter_ns() - started) / 1_000_000)
                order.append(name)
        events_json = trace.fetch_events_json()
    device_all = device_intervals(events_json)
    if len(device_all) != len(order):
        raise AssertionError(
            f"system trace has {len(device_all)} executions for {len(order)} calls"
        )
    device = {name: [] for name in names}
    for name, value in zip(order, device_all):
        device[name].append(value)
    return {"host": host, "device": device}


def slope(samples_long: list[float], samples_one: list[float], length: int) -> dict:
    """Per-call time from paired K-long and 1-long samples, in microseconds."""
    pairs = [
        (long_ms - one_ms) / (length - 1)
        for long_ms, one_ms in zip(samples_long, samples_one)
    ]
    return stats_us(pairs)


def run_batch(batch: int, old_sinkhorn, old_combine, args) -> dict:
    torch._dynamo.reset()
    x, residual, post, comb_start = site_inputs(batch)
    k = args.chain
    entries = {
        "before": (old_sinkhorn.sinkhorn_normalise_blocks, old_combine.hyper_connection_combine),
        "after": (new_sinkhorn.sinkhorn_normalise_blocks, new_combine.hyper_connection_combine),
    }
    device = "neuron:0"
    sk_in = (comb_start.to(device),)
    cb_in = tuple(v.to(device) for v in (x, residual, post, comb_start))

    # CPU references for the K-long chains.
    ref_sk = cpu_sinkhorn_chain(comb_start, k)
    ref_cb = cpu_combine_chain(x, residual, post, comb_start, k)

    graphs, inputs, checks = {}, {}, {}
    for variant, (sk_entry, cb_entry) in entries.items():
        for length in (1, k):
            sk_name = f"{variant}_sinkhorn_k{length}"
            cb_name = f"{variant}_combine_k{length}"
            graphs[sk_name] = compile_fn(sinkhorn_chain(sk_entry, length), args)
            graphs[cb_name] = compile_fn(combine_chain(cb_entry, length), args)
            inputs[sk_name] = sk_in
            inputs[cb_name] = cb_in
        sk_out = graphs[f"{variant}_sinkhorn_k{k}"](*sk_in).to("cpu")
        cb_out = graphs[f"{variant}_combine_k{k}"](*cb_in).to("cpu")
        for value in (sk_out, cb_out):
            if not torch.isfinite(value).all():
                raise AssertionError(f"{variant}: nonfinite device output")
        torch.testing.assert_close(sk_out, ref_sk, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            cb_out, ref_cb, rtol=1e-4, atol=1e-4 * float(ref_cb.abs().max())
        )
        checks[variant] = {
            "sinkhorn_max_abs_vs_cpu": float((sk_out - ref_sk).abs().max()),
            "combine_max_abs_vs_cpu": float((cb_out - ref_cb).abs().max()),
            "_sk": sk_out,
            "_cb": cb_out,
        }
    equivalence = {
        "sinkhorn_max_abs_after_vs_before": float(
            (checks["after"]["_sk"] - checks["before"]["_sk"]).abs().max()
        ),
        "combine_max_abs_after_vs_before": float(
            (checks["after"]["_cb"] - checks["before"]["_cb"]).abs().max()
        ),
    }
    for variant in checks:
        checks[variant].pop("_sk")
        checks[variant].pop("_cb")

    samples = time_graphs(graphs, inputs, args)
    result = {"B": batch, "hidden": HIDDEN, "streams": STREAMS, "iters": ITERS,
              "chain_length": k, "checks": checks, "equivalence": equivalence}
    for variant in entries:
        per_kernel = {}
        for kernel in ("sinkhorn", "combine"):
            long_name = f"{variant}_{kernel}_k{k}"
            one_name = f"{variant}_{kernel}_k1"
            per_kernel[kernel] = {
                "device_per_call": slope(
                    samples["device"][long_name], samples["device"][one_name], k
                ),
                "device_single_call_graph": stats_us(samples["device"][one_name]),
                "host_single_call_graph": stats_us(samples["host"][one_name]),
            }
        site_pairs = [
            (sl - s1 + cl - c1) / (k - 1)
            for sl, s1, cl, c1 in zip(
                samples["device"][f"{variant}_sinkhorn_k{k}"],
                samples["device"][f"{variant}_sinkhorn_k1"],
                samples["device"][f"{variant}_combine_k{k}"],
                samples["device"][f"{variant}_combine_k1"],
            )
        ]
        per_kernel["site"] = stats_us(site_pairs)
        result[variant] = per_kernel
    before = result["before"]["site"]["median_us"]
    after = result["after"]["site"]["median_us"]
    result["site_median_speedup"] = before / after
    result["site_after_over_before"] = after / before
    result["bounded_step_saving_ms"] = (before - after) * SITES_PER_STEP / 1000.0

    if args.profile_dir is not None:
        profile_dir = args.profile_dir / f"b{batch}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        runtime = torch.classes.neuron.Runtime()
        runtime.start_profiling(
            str(profile_dir),
            ["device_profile", "system_profile"],
            None,
            None,
            libtorch_envs.get_neuron_compile_cache_dir(),
        )
        try:
            for _ in range(args.profile_iterations):
                for variant in entries:
                    for kernel in ("sinkhorn", "combine"):
                        name = f"{variant}_{kernel}_k1"
                        graphs[name](*inputs[name]).to("cpu")
        finally:
            runtime.stop_profiling()
        result["profile_dir"] = str(profile_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-module", type=Path, required=True,
                        help="directory holding the 5938748 sinkhorn.py and hyper_connection.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--chain", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument("--compiler-args", default=os.environ.get("NEURON_CC_FLAGS", ""))
    parser.add_argument("--use-compile-cache", action="store_true",
                        help="reuse cached NEFFs (off by default; see the module docstring)")
    args = parser.parse_args()
    if not args.use_compile_cache:
        os.environ["NEURON_LIBTORCH_DISABLE_COMPILE_CACHE"] = "1"
    for module in (new_sinkhorn, new_combine):
        if not Path(module.__file__).resolve().is_relative_to(REPO_ROOT):
            raise RuntimeError(f"{module.__name__} imported from {module.__file__}, not {REPO_ROOT}")
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Pin NEURON_RT_VISIBLE_CORES before running the benchmark")
    if args.chain < 2:
        raise ValueError("--chain must be at least 2 to take a slope")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("Use nonnegative warmup and positive iteration counts")
    if any(b < 1 for b in args.batches):
        raise ValueError("Batch sizes must be positive")
    old_sinkhorn, old_combine = load_baseline(args.baseline_module)
    report = {
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_LOGICAL_NC_CONFIG",
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NEURON_LIBTORCH_CACHE_ROOT",
                "NEURON_LIBTORCH_DISABLE_COMPILE_CACHE",
            )
        },
        "device": "neuron:0 (one logical core = 2 physical cores at LNC2)",
        "after_modules": [new_sinkhorn.__file__, new_combine.__file__],
        "compile_cache": "used" if args.use_compile_cache else "disabled",
        "baseline_module": str(args.baseline_module.resolve()),
        "method": (
            "per-call device time = slope between a K-long and a 1-long dependency "
            "chain of the kernel in one compiled graph, from the runtime system "
            "trace; per site = sinkhorn + combine"
        ),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        case = run_batch(batch, old_sinkhorn, old_combine, args)
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in case.items() if k in (
            "B", "equivalence", "site_median_speedup", "bounded_step_saving_ms")}
        ), flush=True)
        for variant in ("before", "after"):
            print(variant, json.dumps({
                kern: case[variant][kern]["device_per_call"]["median_us"]
                if kern != "site" else case[variant]["site"]["median_us"]
                for kern in ("sinkhorn", "combine", "site")
            }), flush=True)


if __name__ == "__main__":
    main()
