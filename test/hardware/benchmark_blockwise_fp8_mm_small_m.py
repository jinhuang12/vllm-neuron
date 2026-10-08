# SPDX-License-Identifier: Apache-2.0
"""Small-M dense fp8 kernels on Neuron, against an unchanged module snapshot.

Set the Neuron cores before launching (``devlease.py slice`` does). This script
does not select cores or provision hardware. All references are computed on CPU.
Compilation and warmup are excluded from the reported latency.

Two modes:

``--mode mlp`` (default): the fused small-M SwiGLU MLP, ``blockwise_fp8_mlp``,
at the per-rank TP=64 decode shapes: hidden 4096 and intermediate 128 (shared
expert) or 256 (dense MLP), M in ``--tokens`` (default 1 and 64). "before" is the
snapshot module's ``blockwise_fp8_mlp`` (``--baseline-module``, the 0a08ff4 file,
whose kernel runs on one physical core); "after" is this tree's. Per-call time is
the device time slope between an R-long and a 1-long dependency chain of the MLP
in one compiled graph (R = ``--chain``, distinct weights per link, each output
cast to bf16 and fed to the next link as in a decode step), from the runtime
system trace; host times are reported beside it. Each graph runs
``--iterations`` timed executions, interleaved across graphs. The one-link
graphs are checked against a CPU reference and against each other first.
``--profile-dir`` captures a device profile of the one-link graphs after timing.

``--mode mm``: the original comparison of the single-GEMM small-M kernel with the
padded snapshot kernel (``blockwise_fp8_mm_kernel``), one call per graph.

The compile cache is off unless ``--use-compile-cache``: the graph cache key
ignores kernel bodies, so an edited kernel could otherwise be timed from a stale
NEFF. ``--merge`` adds this run's cases to an existing ``--output`` file (a case
with the same name is replaced), so the cases can be spread over device jobs.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import time

#: This file's repository. Run as a script, Python puts ``test/hardware`` on
#: ``sys.path`` and the venv resolves ``vllm_neuron`` to another checkout; the
#: "after" side must be this tree.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as functional  # noqa: E402
from torch.nn.functional import silu  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
import libtorch_neuronx_lite.envs as libtorch_envs  # noqa: E402
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki  # noqa: E402

from vllm_neuron.functional.blockwise_fp8_mm import (  # noqa: E402
    blockwise_fp8_mm_small_m_kernel,
    blockwise_fp8_mm_torch_oracle,
    to_kernel_scale_layout,
)

HIDDEN = 4096
SWIGLU_LIMIT = 10.0
DEVICE = "neuron:0"
#: Round-1 dense tolerance (test_blockwise_fp8_mlp_small_m.py):
#: ``max|got - ref| <= MLP_TOLERANCE * max|ref|``.
MLP_TOLERANCE = 2e-3


def load_baseline_module(path: Path):
    """Import ``blockwise_fp8_mm.py`` from a snapshot file or directory.

    The module name carries the directory name, so two snapshots never share a
    kernel name (the compile cache keys a kernel by its qualified name).
    """
    path = path.resolve()
    if path.is_dir():
        path = path / "blockwise_fp8_mm.py"
    tag = re.sub(r"\W", "_", path.parent.name)
    name = f"_dense_baseline_{tag}_blockwise_fp8_mm"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load baseline snapshot {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.to(torch.float32)
    expected = expected.to(torch.float32)
    residual = actual - expected
    return {
        "max_absolute_difference": residual.abs().max().item(),
        "max_abs_reference": expected.abs().max().item(),
        "difference_norm": residual.norm().item(),
        "relative_l2": (residual.norm() / expected.norm().clamp_min(1e-30)).item(),
        "cosine_similarity": functional.cosine_similarity(
            actual.reshape(1, -1), expected.reshape(1, -1)
        ).item(),
    }


def compile_fn(fn):
    return torch.compile(
        fn,
        backend="neuron_libtorch",
        fullgraph=True,
        dynamic=False,
        options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")},
    )


def stats_us(samples_us: list[float]) -> dict:
    ordered = sorted(samples_us)
    return {
        "iterations": len(ordered),
        "median_us": statistics.median(ordered),
        "mean_us": statistics.mean(ordered),
        "p90_us": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def device_intervals(events_json: str) -> list[float]:
    """Per-execution device time in us, in execution order.

    The physical-core intervals of one LNC2 execution are merged (first start to
    last stop), as in ``benchmark_mhc_decode.py``.
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
    return [
        (max(end for _, end in intervals[e]) - min(b for b, _ in intervals[e])) / 1000.0
        for e in sorted(intervals)
    ]


def time_graphs(graphs: dict, inputs: dict, args) -> dict:
    """Interleave every graph's timed calls; host and device samples in us."""
    from nrtpy._nrtpy import SystemTraceSession

    names = list(graphs)
    for _ in range(args.warmup):
        for name in names:
            graphs[name](*inputs[name]).to("cpu")
    host = {name: [] for name in names}
    order = []
    with SystemTraceSession() as trace:
        for iteration in range(args.iterations):
            shift = iteration % len(names)
            for name in names[shift:] + names[:shift]:
                started = time.perf_counter_ns()
                graphs[name](*inputs[name]).to("cpu")
                host[name].append((time.perf_counter_ns() - started) / 1000.0)
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


def capture_profile(graphs: dict, inputs: dict, names, directory: Path, repeats: int):
    directory.mkdir(parents=True, exist_ok=True)
    runtime = torch.classes.neuron.Runtime()
    runtime.start_profiling(
        str(directory),
        ["device_profile", "system_profile"],
        None,
        None,
        libtorch_envs.get_neuron_compile_cache_dir(),
    )
    try:
        for _ in range(repeats):
            for name in names:
                graphs[name](*inputs[name]).to("cpu")
    finally:
        runtime.stop_profiling()
    return str(directory)


# --------------------------------------------------------------------------- #
# mlp mode
# --------------------------------------------------------------------------- #

def mlp_operands(tokens: int, inter: int, links: int, seed: int):
    """Decode-like operands, as in test_blockwise_fp8_mlp_small_m.py.

    Unit activations, fp8 weights and distinct non-power-of-two block scales
    sized so both clamps are active. ``links`` distinct weight sets.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((tokens, HIDDEN), generator=g).to(torch.bfloat16)

    def weight(rows, cols):
        raw = torch.randn((rows, cols), generator=g) * 64
        return raw.clamp(-240, 240).to(torch.float8_e4m3fn)

    def scale(rows, cols, magnitude):
        grid = torch.rand((rows // 128, cols // 128), generator=g) + 0.37
        return (grid * magnitude).to(torch.float32)

    layers = [
        (weight(HIDDEN, inter), weight(HIDDEN, inter), weight(inter, HIDDEN),
         scale(HIDDEN, inter, 3e-3), scale(HIDDEN, inter, 3e-3),
         scale(inter, HIDDEN, 1e-3))
        for _ in range(links)
    ]
    return x, layers


def mlp_cpu_reference(x, layer):
    """The three-call route in torch fp32: clamps, silu, bf16 cast, down."""
    gw, uw, dw, gs, us, ds = layer
    gate = blockwise_fp8_mm_torch_oracle(x, gw, gs).clamp(max=SWIGLU_LIMIT)
    up = blockwise_fp8_mm_torch_oracle(x, uw, us).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    return blockwise_fp8_mm_torch_oracle((silu(gate) * up).to(torch.bfloat16), dw, ds)


def mlp_chain(entry, links: int):
    def chain(h, *weights):
        for link in range(links):
            out = entry(h, *weights[6 * link:6 * link + 6],
                        swiglu_limit=SWIGLU_LIMIT)
            h = out.to(torch.bfloat16)
        return out

    return chain


def mlp_case(tokens: int, inter: int, baseline, current, args) -> dict:
    torch._dynamo.reset()
    name = f"mlp_m{tokens}_i{inter}"
    links = args.chain
    x, layers = mlp_operands(tokens, inter, links, seed=4100 + tokens + inter)
    reference = mlp_cpu_reference(x, layers[0])
    xd = x.to(DEVICE)
    dev = [t.to(DEVICE) for layer in layers for t in layer]
    entries = {"before": baseline.blockwise_fp8_mlp,
               "after": current.blockwise_fp8_mlp}
    graphs, inputs = {}, {}
    current.reset_mlp_dispatch_counters()
    reset_grid = getattr(current, "reset_mlp_launch_counters", None)
    if reset_grid is not None:
        reset_grid()
    for variant, entry in entries.items():
        for length in (1, links):
            key = f"{variant}_k{length}"
            graphs[key] = compile_fn(mlp_chain(entry, length))
            inputs[key] = (xd, *dev[:6 * length])
    outputs = {key: graphs[key](*inputs[key]).to("cpu") for key in graphs}
    for key, value in outputs.items():
        if not torch.isfinite(value).all():
            raise AssertionError(f"{name}: {key} returned nonfinite values")
    checks = {
        "before_vs_cpu": metrics(outputs["before_k1"], reference),
        "after_vs_cpu": metrics(outputs["after_k1"], reference),
        "after_vs_before": metrics(outputs["after_k1"], outputs["before_k1"]),
        "after_vs_before_chain": metrics(
            outputs[f"after_k{links}"], outputs[f"before_k{links}"]
        ),
    }
    bound = MLP_TOLERANCE * reference.abs().max().item()
    worst = (outputs["after_k1"] - outputs["before_k1"]).abs().max().item()
    if worst > bound:
        raise AssertionError(
            f"{name}: after differs from before by {worst:.3e} > {bound:.3e}"
        )
    launch = {"after_mlp_dispatch_counters": list(current.mlp_dispatch_counters())}
    read_grid = getattr(current, "mlp_launch_counters", None)
    if read_grid is not None:
        launch["after_mlp_launch_counters"] = list(read_grid())
    samples = time_graphs(graphs, inputs, args)
    result = {
        "case": name,
        "M": tokens,
        "H": HIDDEN,
        "I": inter,
        "site": "shared expert" if inter == 128 else "dense MLP (layers 0-2)",
        "chain_links": links,
        "tolerance": f"max|after - before| <= {MLP_TOLERANCE} * max|ref| "
                     f"= {bound:.4e}; observed {worst:.4e}",
        "checks": checks,
        "launch": launch,
    }
    for variant in entries:
        long_key, one_key = f"{variant}_k{links}", f"{variant}_k1"
        per_call = {}
        for clock in ("device", "host"):
            pairs = [
                (long - one) / (links - 1)
                for long, one in zip(samples[clock][long_key], samples[clock][one_key])
            ]
            per_call[clock] = stats_us(pairs)
        result[variant] = {
            "per_call_device": per_call["device"],
            "per_call_host": per_call["host"],
            "graph_k1_device": stats_us(samples["device"][one_key]),
            f"graph_k{links}_device": stats_us(samples["device"][long_key]),
        }
    before = result["before"]["per_call_device"]
    after = result["after"]["per_call_device"]
    result["speedup_median"] = before["median_us"] / after["median_us"]
    result["speedup_p90"] = before["p90_us"] / after["p90_us"]
    result["after_over_before_median"] = after["median_us"] / before["median_us"]
    if args.profile_dir is not None:
        result["profile_dir"] = capture_profile(
            graphs, inputs, ["before_k1", "after_k1"],
            args.profile_dir / name, args.profile_iterations,
        )
    return result


# --------------------------------------------------------------------------- #
# mm mode (the original single-GEMM comparison)
# --------------------------------------------------------------------------- #

def measure(model, inputs, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        model(*inputs).to("cpu")
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        model(*inputs).to("cpu")
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    samples.sort()
    return {
        "iterations": iterations,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p90_ms": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "samples_ms": samples,
    }


def mm_case(tokens: int, rows: int, cols: int, baseline_kernel, args) -> dict:
    # Each shape is an independent comparison. Keep earlier case variants out
    # of the Python guard cache; compiler artifacts remain in the disk cache.
    torch._dynamo.reset()
    generator = torch.Generator().manual_seed(772 + rows + cols)
    x = torch.randn((tokens, rows), generator=generator).to(torch.bfloat16)
    raw = torch.randn((rows, cols), generator=generator).clamp(-240, 240)
    weight = raw.to(torch.float8_e4m3fn)
    scale = torch.rand(
        (rows // 128, cols // 128), generator=generator, dtype=torch.float32
    ) + 0.125
    scale_t = to_kernel_scale_layout(scale, rows, cols)
    expected = blockwise_fp8_mm_torch_oracle(x, weight, scale)
    padded = torch.zeros((128, rows), dtype=x.dtype)
    padded[:tokens] = x

    baseline_call = wrap_nki(baseline_kernel)
    optimized_call = wrap_nki(blockwise_fp8_mm_small_m_kernel)

    def baseline(activation, weights, scales):
        return baseline_call(x=activation, weight=weights, weight_scale_t=scales)[
            :tokens
        ]

    def optimized(activation, weights, scales):
        return optimized_call(x=activation, weight=weights, weight_scale_t=scales)

    compiled_baseline = compile_fn(baseline)
    compiled_optimized = compile_fn(optimized)
    weights_device = weight.to(DEVICE)
    scales_device = scale_t.to(DEVICE)
    baseline_inputs = (padded.to(DEVICE), weights_device, scales_device)
    optimized_inputs = (x.to(DEVICE), weights_device, scales_device)
    old = compiled_baseline(*baseline_inputs).to("cpu")
    new = compiled_optimized(*optimized_inputs).to("cpu")
    old_reference = metrics(old, expected)
    new_reference = metrics(new, expected)
    equivalence = metrics(new, old)
    for value in (old, new):
        if not torch.isfinite(value).all():
            raise AssertionError("Device GEMM returned nonfinite values")
        torch.testing.assert_close(value, expected, rtol=3e-2, atol=1e-3)
    if new_reference["relative_l2"] > 1e-3:
        raise AssertionError(f"Small-M differs from CPU oracle: {new_reference}")
    if equivalence["relative_l2"] > 1e-3:
        raise AssertionError(f"Small-M differs from padded baseline: {equivalence}")
    baseline_timing = measure(
        compiled_baseline, baseline_inputs, args.warmup, args.iterations
    )
    optimized_timing = measure(
        compiled_optimized, optimized_inputs, args.warmup, args.iterations
    )
    profile_dir = None
    compiler_commands = []
    if args.profile_dir is not None:
        # Capture after timing so inspect overhead does not affect the latency
        # comparison. Alternate warmed baseline and optimized executions.
        profile_dir = capture_profile(
            {"baseline": compiled_baseline, "optimized": compiled_optimized},
            {"baseline": baseline_inputs, "optimized": optimized_inputs},
            ["baseline", "optimized"],
            args.profile_dir / f"m{tokens}_k{rows}_n{cols}",
            args.profile_iterations,
        )
        cache_dir = Path(libtorch_envs.get_neuron_compile_cache_dir())
        for neff in sorted((Path(profile_dir) / "neffs").glob("*/graph_*.neff")):
            command_file = cache_dir / neff.parent.name / "command.txt"
            if command_file.exists():
                compiler_commands.append({
                    "neff": neff.name,
                    "command_file": str(command_file),
                    "command": command_file.read_text(),
                })
    return {
        "case": f"mm_m{tokens}_k{rows}_n{cols}",
        "M": tokens,
        "K": rows,
        "N": cols,
        "baseline_vs_cpu": old_reference,
        "optimized_vs_cpu": new_reference,
        "optimized_vs_baseline": equivalence,
        "baseline": baseline_timing,
        "optimized": optimized_timing,
        "median_speedup": baseline_timing["median_ms"]
        / optimized_timing["median_ms"],
        "synchronization": "same-size MxN FP32 output copied to CPU each iteration",
        "profile_dir": profile_dir,
        "profile_iterations_per_variant": args.profile_iterations,
        "compiler_options": {
            "compiler_args": os.environ.get("NEURON_CC_FLAGS", "")
        },
        "compiler_commands": compiler_commands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-module", type=Path, required=True,
                        help="snapshot blockwise_fp8_mm.py, or its directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("mlp", "mm"), default="mlp")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--tokens", type=int, nargs="+")
    parser.add_argument("--intermediates", type=int, nargs="+", default=[128, 256],
                        help="mlp mode: per-rank intermediate widths")
    parser.add_argument("--shape", type=int, nargs=2, action="append",
                        help="mm mode: K N")
    parser.add_argument("--chain", type=int, default=16,
                        help="mlp mode: links in the long chain graph")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--use-compile-cache", action="store_true",
                        help="reuse cached NEFFs (off by default; see the module docstring)")
    parser.add_argument("--merge", action="store_true",
                        help="add the cases to an existing --output file")
    parser.add_argument("--time-limit", type=int, default=570,
                        help="seconds before the run aborts (device jobs stay < 10 min)")
    args = parser.parse_args()
    signal.alarm(args.time_limit)
    if not args.use_compile_cache:
        os.environ["NEURON_LIBTORCH_DISABLE_COMPILE_CACHE"] = "1"
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Pin NEURON_RT_VISIBLE_CORES before running the benchmark")
    tokens = args.tokens or ([1, 64] if args.mode == "mlp" else [1, 7])
    if any(not 0 < value < 128 for value in tokens):
        raise ValueError("Token counts must be between 1 and 127")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("Use nonnegative warmup and positive iteration counts")
    if args.profile_iterations < 1:
        raise ValueError("Use positive profile iteration counts")
    if args.chain < 2:
        raise ValueError("--chain must be at least 2 to take a slope")
    current = importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")
    if not Path(current.__file__).resolve().is_relative_to(REPO_ROOT):
        raise RuntimeError(f"{current.__name__} imported from {current.__file__}")
    baseline = load_baseline_module(args.baseline_module)
    args.output = args.output.resolve()
    if args.profile_dir is not None:
        args.profile_dir = args.profile_dir.resolve()
    # The NKI and neuronx-cc drivers write artifacts into the working directory.
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or "/tmp") / "bench-cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = None
    if args.merge and args.output.exists():
        report = json.loads(args.output.read_text())
    if report is None:
        report = {"cases": []}
    report.update({
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_NUM_CORES",
                "NEURON_LOGICAL_NC_CONFIG",
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NEURON_LIBTORCH_CACHE_ROOT",
                "NEURON_LIBTORCH_DISABLE_COMPILE_CACHE",
            )
        },
        "device": "neuron:0 = one logical core (LNC2: 2 physical cores) of the slice",
        "after_module": current.__file__,
        "baseline_module": baseline.__file__,
        "compile_cache": "used" if args.use_compile_cache else "disabled",
    })
    if args.mode == "mlp":
        report["method"] = (
            "per-call = (R-link chain graph - 1-link graph) / (R - 1), paired per "
            "iteration; device time from the runtime system trace (LNC2 physical-"
            "core intervals merged), host time beside it"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(case):
        report["cases"] = [c for c in report["cases"]
                           if c.get("case") != case["case"]] + [case]
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    if args.mode == "mlp":
        for count in tokens:
            for inter in args.intermediates:
                case = mlp_case(count, inter, baseline, current, args)
                record(case)
                print(json.dumps({
                    "case": case["case"],
                    "before_us": case["before"]["per_call_device"]["median_us"],
                    "after_us": case["after"]["per_call_device"]["median_us"],
                    "speedup": case["speedup_median"],
                    "launch": case["launch"],
                }), flush=True)
    else:
        shapes = args.shape or [(4096, 128), (128, 4096), (4096, 256), (256, 4096)]
        for count in tokens:
            for rows, cols in shapes:
                case = mm_case(count, rows, cols, baseline.blockwise_fp8_mm_kernel, args)
                record(case)
                print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
