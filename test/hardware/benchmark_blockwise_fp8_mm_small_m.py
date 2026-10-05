# SPDX-License-Identifier: Apache-2.0
"""Compare unpadded dense FP8 GEMM with an unchanged module snapshot on Neuron.

Set the Neuron core and LNC environment before launching. This script deliberately
does not select cores or provision hardware. All references are computed on CPU.
Both device variants return exactly M rows, so their synchronization copies have
the same size. Compilation and warmup are excluded from the reported latency.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as functional

import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
import libtorch_neuronx_lite.envs as libtorch_envs
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.blockwise_fp8_mm import (
    blockwise_fp8_mm_small_m_kernel,
    blockwise_fp8_mm_torch_oracle,
    to_kernel_scale_layout,
)


def load_baseline(path: Path):
    name = "_dense_fp8_baseline_snapshot"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load baseline snapshot {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.blockwise_fp8_mm_kernel


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.to(torch.float32)
    expected = expected.to(torch.float32)
    residual = actual - expected
    return {
        "max_absolute_difference": residual.abs().max().item(),
        "difference_norm": residual.norm().item(),
        "relative_l2": (residual.norm() / expected.norm().clamp_min(1e-30)).item(),
        "cosine_similarity": functional.cosine_similarity(
            actual.reshape(1, -1), expected.reshape(1, -1)
        ).item(),
    }


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


def run_case(tokens: int, rows: int, cols: int, baseline_kernel, args) -> dict:
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

    compiled_baseline = torch.compile(
        baseline,
        backend="neuron_libtorch",
        fullgraph=True,
        dynamic=False,
        options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")},
    )
    compiled_optimized = torch.compile(
        optimized,
        backend="neuron_libtorch",
        fullgraph=True,
        dynamic=False,
        options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")},
    )
    weights_device = weight.to("neuron:0")
    scales_device = scale_t.to("neuron:0")
    baseline_inputs = (padded.to("neuron:0"), weights_device, scales_device)
    optimized_inputs = (x.to("neuron:0"), weights_device, scales_device)
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
        profile_dir = args.profile_dir / f"m{tokens}_k{rows}_n{cols}"
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
                compiled_baseline(*baseline_inputs).to("cpu")
                compiled_optimized(*optimized_inputs).to("cpu")
        finally:
            runtime.stop_profiling()
        cache_dir = Path(libtorch_envs.get_neuron_compile_cache_dir())
        for neff in sorted((profile_dir / "neffs").glob("*/graph_*.neff")):
            command_file = cache_dir / neff.parent.name / "command.txt"
            compiler_commands.append({
                "neff": neff.name,
                "command_file": str(command_file),
                "command": command_file.read_text(),
            })
    return {
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
        "profile_dir": str(profile_dir) if profile_dir is not None else None,
        "profile_iterations_per_variant": args.profile_iterations,
        "compiler_options": {
            "compiler_args": os.environ.get("NEURON_CC_FLAGS", "")
        },
        "compiler_commands": compiler_commands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 7])
    parser.add_argument("--shape", type=int, nargs=2, action="append")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile-iterations", type=int, default=5)
    args = parser.parse_args()
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Pin NEURON_RT_VISIBLE_CORES before running the benchmark")
    if any(not 0 < tokens < 128 for tokens in args.tokens):
        raise ValueError("Token counts must be between 1 and 127")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("Use nonnegative warmup and positive iteration counts")
    if args.profile_iterations < 1:
        raise ValueError("Use positive profile iteration counts")
    shapes = args.shape or [(4096, 128), (128, 4096), (4096, 256), (256, 4096)]
    baseline_kernel = load_baseline(args.baseline_module)
    report = {
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_NUM_CORES",
                "NEURON_LOGICAL_NC_CONFIG",
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
            )
        },
        "baseline_module": str(args.baseline_module.resolve()),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.tokens:
        for rows, cols in shapes:
            case = run_case(tokens, rows, cols, baseline_kernel, args)
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
