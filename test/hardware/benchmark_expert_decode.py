# SPDX-License-Identifier: Apache-2.0
"""Routed-expert decode microbenchmark: the 0a08ff4 kernel vs this tree's, at B rows.

Run under ``devlease.py slice moe-t`` (one trn2 chip, LNC2). This script does not
select cores. All references are computed on CPU.

What is timed: one rank's 18-expert shard at EP=16, TP=64 (I = 512 per rank,
H = 4096) on the router's global ``[T, 288]`` output and a device rank tensor.
Before: the 0a08ff4 ``fused_fp8_decode_experts`` over the 0a08ff4 kernel copy in
``--baseline-module`` (``test/hardware/baselines/moe_0a08ff4``). After: this
tree's ``fused_fp8_decode_experts``. Both return bf16 ``[T, H]``, as the model
calls them, and both run two programs (both LNC2 cores) when LNC2 is set.

Routing. Each layer has a fixed seeded table from
``decode_fixtures.batch_routing_table`` (Zipf-skewed expert popularity, the
served shape: rank 0 of the bs=64 profile visits 6..17 of 18 local experts per
layer, mean 10.8). The tables, their per-layer hit histograms and the seeds are
in the JSON. T=1 also runs fixed 0/1/2-hit layers, T=4 a fixed 4-expert layer.

Method. Host launch and sync cost far more than one layer, so each variant is
compiled twice into one graph: 1 layer and ``L`` layers (distinct inputs, routing
and weight copies per layer). Layer ``l`` reads layer ``l-1``'s output, as in the
model: the expert input is ``x_l + 2^-8 * y_{l-1}``. The per-layer cost is the
slope ``(t_L - median(t_1)) / (L - 1)`` per timed ``t_L`` sample; median and p90
over the samples are reported. The device checks compare each variant's 1-layer
graph with the torch oracle, and after with before.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

# Import this worktree's vllm_neuron, not the venv's editable install.
REPO = Path(__file__).resolve().parents[2]
if sys.path[:1] != [str(REPO)]:
    sys.path.insert(0, str(REPO))

import torch

import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
import libtorch_neuronx_lite.envs as libtorch_envs

from vllm_neuron.functional.moe import expert_decode
from vllm_neuron.functional.moe.expert_decode import expert_decode_torch_oracle
from vllm_neuron.functional.moe.fused_fp8 import fused_fp8_decode_experts
from vllm_neuron.functional.moe.fused_fp8_pack import PackedExperts
from vllm_neuron.functional.moe.moe_blockwise_fp8 import _swiglu_bound_operand
from test.vllm_neuron.functional.moe.decode_fixtures import (
    LOCAL_EXPERTS,
    LOCAL_INTERMEDIATE,
    RANK,
    SWIGLU_LIMIT,
    batch_routing_table,
    hit_histogram,
    packed_expert_bank,
    routed_affinities,
)

HIDDEN = 4096
DEVICE = "neuron:0"
#: Weight of the previous layer's output in the next layer's input.
CHAIN = 2.0 ** -8

#: The served model's neuronx-cc arguments (``neuron_model_runner`` at -O1).
MODEL_COMPILER_ARGS = [
    "--auto-cast=none",
    "-O1",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 "
    "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3",
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
]


def scenarios(tokens: int, layers: int):
    """name -> per-layer local hit tables."""
    realistic = [batch_routing_table(tokens, seed=1000 * tokens + layer)
                 for layer in range(layers)]
    out = {"realistic": realistic}
    if tokens == 1:
        out["0_hits"] = [[[]]] * layers
        out["1_hit"] = [[[7]]] * layers
        out["2_hits"] = [[[3, 11]]] * layers
    if tokens == 4:
        out["4_distinct"] = [[[3, 11], [11], [], [0, 3, 17]]] * layers
    return out


def load_baseline(path: Path):
    name = "moe_0a08ff4"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, path / "__init__.py", submodule_search_locations=[str(path)])
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load baseline package {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{name}.pipeline")


def compiler_args():
    return os.environ.get("NEURON_CC_FLAGS") or MODEL_COMPILER_ARGS


def compile_fn(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": compiler_args()})


def expert_graph(variant: str, layers: int, baseline, bounds):
    def chained(xs, affs, weights, scales, rank):
        y = None
        for layer in range(layers):
            x = xs[layer]
            if y is not None:  # layer l reads layer l-1's output, as in the model
                x = (x + CHAIN * y).to(torch.bfloat16)
            packed = PackedExperts(weights[layer], scales[layer])
            if variant == "before":
                y = baseline.decode_experts(x, affs[layer], packed, bounds, rank,
                                            out_dtype=torch.bfloat16)
            else:
                y = fused_fp8_decode_experts(x, affs[layer], packed, bounds, rank,
                                             out_dtype=torch.bfloat16)
        return y

    return compile_fn(chained)


def samples_ms(fn, inputs, warmup, iterations):
    for _ in range(warmup):
        fn(*inputs).to("cpu")
    out = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        fn(*inputs).to("cpu")
        out.append((time.perf_counter_ns() - started) / 1e6)
    return out


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def per_layer(one, many, layers):
    base = statistics.median(one)
    slopes = [(t - base) / (layers - 1) * 1000.0 for t in many]  # us
    return {
        "median_us": statistics.median(slopes),
        "p90_us": percentile(slopes, 0.9),
        "min_us": min(slopes),
        "one_layer_call_median_ms": base,
        f"{layers}_layer_call_median_ms": statistics.median(many),
    }


def profile(label, graph, inputs, args):
    if args.profile_dir is None:
        return None
    path = args.profile_dir / label
    path.mkdir(parents=True, exist_ok=True)
    runtime = torch.classes.neuron.Runtime()
    runtime.start_profiling(str(path), ["device_profile", "system_profile"], None, None,
                            libtorch_envs.get_neuron_compile_cache_dir())
    try:
        for _ in range(args.profile_iterations):
            graph(*inputs).to("cpu")
    finally:
        runtime.stop_profiling()
    return str(path)


def cases_for(tokens, baseline, args):
    torch._dynamo.reset()
    layers = args.layers
    bank = packed_expert_bank()
    bounds_cpu = _swiglu_bound_operand(SWIGLU_LIMIT, SWIGLU_LIMIT, torch.device("cpu"))
    bounds = bounds_cpu.to(DEVICE)
    weights = [bank.weights.to(DEVICE) for _ in range(layers)]
    scales = [bank.scales.to(DEVICE) for _ in range(layers)]
    rank = torch.tensor([RANK], dtype=torch.int64).to(DEVICE)
    xs = [torch.randn(tokens, HIDDEN, generator=torch.Generator().manual_seed(200 + l)
                      ).to(torch.bfloat16) for l in range(layers)]
    graphs = {v: (expert_graph(v, 1, baseline, bounds), expert_graph(v, layers, baseline, bounds))
              for v in args.variants}
    out = []
    for name, tables in scenarios(tokens, layers).items():
        affs = [routed_affinities(tables[l], seed=300 + l) for l in range(layers)]
        inputs = {
            n: ([x.to(DEVICE) for x in xs[:n]], [a.to(DEVICE) for a in affs[:n]],
                weights[:n], scales[:n], rank)
            for n in (1, layers)
        }
        case = {
            "T": tokens, "scenario": name, "layers": layers,
            "plan": list(expert_decode.decode_plan(tokens)),
            "routing_seeds": ([1000 * tokens + l for l in range(layers)]
                              if name == "realistic" else None),
            "local_hits_per_layer": tables,
            "hit_histogram_per_layer": [hit_histogram(t) for t in tables],
            "distinct_local_experts_per_layer": [
                hit_histogram(t)["distinct_local_experts"] for t in tables],
        }
        ref = expert_decode_torch_oracle(xs[0], affs[0], bank, bounds_cpu, RANK,
                                         out_dtype=torch.float32)
        results = {}
        for variant, (one, many) in graphs.items():
            got = one(*inputs[1]).to("cpu").to(torch.float32)
            results[variant] = got
            scale = float(ref.abs().max()) if bool(ref.abs().max() > 0) else 1.0
            check = {"max_abs_vs_oracle": float((got - ref).abs().max()),
                     "max_abs_ref": float(ref.abs().max())}
            if not torch.allclose(got, ref, rtol=2.0 ** -7, atol=1e-3 * scale + 1e-6):
                raise AssertionError(f"T={tokens} {name} {variant} disagrees with the "
                                     f"oracle: {check}")
            case.setdefault("checks", {})[variant] = check
            case[variant] = timed(one, many, inputs, layers, args)
            if tokens == args.profile_tokens and name == "realistic":
                case[variant]["profile_dir"] = profile(
                    f"experts_T{tokens}_{name}_{variant}", one, inputs[1], args)
        if "before" in results and "after" in results:
            diff = (results["after"] - results["before"]).abs()
            case["checks"]["after_vs_before_max_abs"] = float(diff.max())
            case["speedup_median"] = case["before"]["median_us"] / case["after"]["median_us"]
            case["after_over_before_median"] = (case["after"]["median_us"]
                                                / case["before"]["median_us"])
        out.append(case)
        print(json.dumps({k: v for k, v in case.items()
                          if k not in ("checks", "local_hits_per_layer",
                                       "hit_histogram_per_layer")}), flush=True)
    return out


def timed(one, many, inputs, layers, args):
    t1 = samples_ms(one, inputs[1], args.warmup, args.iterations)
    tl = samples_ms(many, inputs[layers], args.warmup, args.iterations)
    result = per_layer(t1, tl, layers)
    result["iterations"] = args.iterations
    result["warmup"] = args.warmup
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 64])
    parser.add_argument("--variants", nargs="+", default=["before", "after"])
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-tokens", type=int, default=64)
    parser.add_argument("--profile-iterations", type=int, default=1)
    args = parser.parse_args()
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under devlease.py, which pins NEURON_RT_VISIBLE_CORES")
    if args.layers < 2:
        raise ValueError("layers >= 2")
    if not Path(vllm_neuron.__file__).resolve().is_relative_to(REPO):
        raise RuntimeError(f"imported {vllm_neuron.__file__}, not this tree ({REPO})")
    baseline_path = args.baseline_module.resolve()
    baseline = load_baseline(baseline_path)
    args.output = args.output.resolve()
    if args.profile_dir is not None:
        args.profile_dir = args.profile_dir.resolve()
    workdir = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or "/tmp") / "benchmark_workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    lnc2 = os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2"
    report = {
        "environment": {k: os.environ.get(k) for k in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT")},
        "compiler_args": compiler_args(),
        "vllm_neuron": str(Path(vllm_neuron.__file__).resolve().parent),
        "baseline_module": str(baseline_path),
        "method": __doc__.split("Method.")[1].strip(),
        "cores": ("one trn2 chip, logical core 0 of the slice; LNC2: both physical "
                  "cores (2 programs) in both variants" if lnc2 else "LNC1, 1 program"),
        "programs": {"before": 2 if lnc2 else 1, "after": 2 if lnc2 else 1},
        "shapes": {"H": HIDDEN, "E_global": 288, "top_k": 8, "E_local": LOCAL_EXPERTS,
                   "I_local": LOCAL_INTERMEDIATE, "rank": RANK},
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.tokens:
        report["cases"].extend(cases_for(tokens, baseline, args))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{k: c.get(k) for k in ("T", "scenario", "speedup_median")}
                      for c in report["cases"]], indent=2), flush=True)


if __name__ == "__main__":
    main()
