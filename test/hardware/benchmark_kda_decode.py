# SPDX-License-Identifier: Apache-2.0
"""One KDA layer decode step on Neuron: the 5938748 kernels vs the fused kernel.

"Before" is the 5938748 decode region (``loader.old_decode_core``: conv carrier
select and update, the ``nkilib`` depthwise conv, silu, the gate clamp, the
recurrent carrier select, the decode-state step and the carrier write-backs),
run on the 5938748 kernel snapshots one request at a time, as the model ran it.
"After" is :func:`kda_fused_decode` over all ``B`` requests, plus the two
carrier copies the model call site makes. "After (loop)" is the per-request model
form: the fused kernel once per request, ``B`` launches.

Each timed graph holds ``L`` independent layers (own carriers, shared weights),
because a layer runs inside a decode graph that holds 34 of them; per-layer
times are graph time / ``L``. ``L = 1`` is reported too, where one graph launch
and its synchronization copy are inside the number.

Shapes are the TP=64 decode shapes: one KDA head per rank, ``K = V = 128``, a
4-tap conv over 384 channels, a bfloat16 ``SD`` conv carrier, a float32
recurrent carrier, ``gate_lower_bound = -5``. All references run on CPU.

Set the Neuron core and LNC environment before launching (``devlease.py``
does); this script refuses to run without ``NEURON_RT_VISIBLE_CORES``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as functional

# This checkout, not the interpreter's editable install: the fused kernel under
# test exists only here, and the baseline loader checks this checkout's helpers.
REPO = Path(__file__).resolve().parents[2]
if sys.path[:1] != [str(REPO)]:
    sys.path.insert(0, str(REPO))

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
import libtorch_neuronx_lite.envs as libtorch_envs  # noqa: E402

from vllm_neuron.functional.kda.fused_decode import (  # noqa: E402
    fused_decode_grid,
    kda_fused_decode,
    kda_fused_decode_torch_reference,
)

HEADS = 1
KDIM = 128
TAPS = 4
LOWER = -5.0


def load_baseline(directory: Path):
    spec = importlib.util.spec_from_file_location(
        "_kda_5938748_bench_loader", directory / "loader.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {directory / 'loader.py'}")
    loader = importlib.util.module_from_spec(spec)
    # Registered, because a compiled graph resolves the loader's globals by module name.
    sys.modules[spec.name] = loader
    spec.loader.exec_module(loader)
    return loader, loader.load_baseline(directory)


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.to(torch.float32)
    expected = expected.to(torch.float32)
    residual = actual - expected
    return {
        "max_absolute_difference": residual.abs().max().item(),
        "relative_l2": (residual.norm() / expected.norm().clamp_min(1e-30)).item(),
        "cosine_similarity": functional.cosine_similarity(
            actual.reshape(1, -1), expected.reshape(1, -1)
        ).item(),
        "mismatched_elements": int((actual != expected).sum().item()),
    }


def make_case(batch: int, layers: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    width = HEADS * KDIM
    channels = 3 * width
    weights = {
        "q_conv1d_weight": (torch.randn(width, 1, TAPS, generator=gen) * 0.5).to(
            torch.bfloat16
        ),
        "k_conv1d_weight": (torch.randn(width, 1, TAPS, generator=gen) * 0.5).to(
            torch.bfloat16
        ),
        "v_conv1d_weight": (torch.randn(width, 1, TAPS, generator=gen) * 0.5).to(
            torch.bfloat16
        ),
        "A_log": torch.log(torch.rand(HEADS, generator=gen) * 15 + 1),
        "dt_bias": torch.randn(width, generator=gen) * 0.5,
    }
    step = {
        "q_in": torch.randn(layers, batch, width, generator=gen),
        "k_in": torch.randn(layers, batch, width, generator=gen),
        "v_in": torch.randn(layers, batch, width, generator=gen),
        "raw_gate": torch.randn(layers, batch, width, generator=gen) * 2,
        "raw_beta": torch.randn(layers, batch, HEADS, generator=gen),
    }
    carriers = {
        "conv": torch.randn(layers, batch, TAPS - 1, channels, generator=gen).to(
            torch.bfloat16
        ),
        "rec": torch.randn(layers, batch, HEADS, KDIM, KDIM, generator=gen) * 0.5,
    }
    position = torch.arange(1, batch + 1, dtype=torch.int32) * 17
    return {"weights": weights, "step": step, "carriers": carriers, "position": position}


STEP_KEYS = ("q_in", "k_in", "v_in", "raw_gate", "raw_beta")
WEIGHT_KEYS = ("q_conv1d_weight", "k_conv1d_weight", "v_conv1d_weight", "A_log", "dt_bias")


def build_variants(loader, kernels, layers: int, batch: int):
    """Graphs that return ``(core, conv, rec)``, the carriers advanced.

    The advanced carriers leave each graph as outputs, and nothing is written
    into a view: the backend turns an in-place write into a new tensor that
    replaces the written node, so a write into a slice of a larger tensor never
    reaches that tensor. The old region writes its carriers in place, so it is
    handed one standalone copy per layer and request, whose later uses do see
    the write. The fused kernel returns its carriers as new tensors already.
    """

    def unpack(args):
        step = dict(zip(STEP_KEYS, args[0:5]))
        conv, rec, position = args[5], args[6], args[7]
        weights = dict(zip(WEIGHT_KEYS, args[8:13]))
        return step, conv, rec, position, weights

    def before(*args):
        step, conv, rec, position, weights = unpack(args)
        cores, convs, recs = [], [], []
        for layer in range(layers):
            per_request = []
            for b in range(batch):
                conv_state = conv[layer, b].clone()
                recurrent_state = rec[layer, b].clone()
                per_request.append(
                    loader.old_decode_core(
                        kernels,
                        *(step[key][layer, b : b + 1] for key in STEP_KEYS),
                        conv_state=conv_state,
                        recurrent_state=recurrent_state,
                        gate_lower_bound=LOWER,
                        conv_state_dim_first=False,
                        start_position=position[b],
                        **weights,
                    )
                )
                convs.append(conv_state)
                recs.append(recurrent_state)
            cores.append(torch.cat(per_request, dim=0))
        shape = (layers, batch)
        return (
            torch.stack(cores),
            torch.stack(convs).reshape(*shape, *conv.shape[2:]),
            torch.stack(recs).reshape(*shape, *rec.shape[2:]),
        )

    def after(*args):
        step, conv, rec, position, weights = unpack(args)
        cores, convs, recs = [], [], []
        for layer in range(layers):
            out = kda_fused_decode(
                *(step[key][layer] for key in STEP_KEYS),
                conv_state=conv[layer],
                recurrent_state=rec[layer],
                gate_lower_bound=LOWER,
                conv_state_dim_first=False,
                start_position=position,
                **weights,
            )
            cores.append(out.core)
            convs.append(out.conv_state)
            recs.append(out.recurrent_state)
        return torch.stack(cores), torch.stack(convs), torch.stack(recs)

    def after_loop(*args):
        step, conv, rec, position, weights = unpack(args)
        cores, convs, recs = [], [], []
        for layer in range(layers):
            per_request = []
            for b in range(batch):
                out = kda_fused_decode(
                    *(step[key][layer, b : b + 1] for key in STEP_KEYS),
                    conv_state=conv[layer, b : b + 1],
                    recurrent_state=rec[layer, b : b + 1],
                    gate_lower_bound=LOWER,
                    conv_state_dim_first=False,
                    start_position=position[b : b + 1],
                    **weights,
                )
                per_request.append(out.core)
                convs.append(out.conv_state[0])
                recs.append(out.recurrent_state[0])
            cores.append(torch.cat(per_request, dim=0))
        shape = (layers, batch)
        return (
            torch.stack(cores),
            torch.stack(convs).reshape(*shape, *conv.shape[2:]),
            torch.stack(recs).reshape(*shape, *rec.shape[2:]),
        )

    def floor(*args):
        # The same inputs and outputs, no layer: the launch, carrier copy and
        # synchronisation cost every variant's graph time contains.
        return args[0] + 0.0, args[5].clone(), args[6].clone()

    variants = {"before": before, "after": after, "floor": floor}
    if batch > 1:
        variants["after_loop"] = after_loop
    return variants


def cpu_reference(case: dict, layers: int) -> dict:
    """The fused step's torch reference on CPU, layer by layer."""
    cores, convs, recs = [], [], []
    position = case["position"].reshape(1, -1)
    for layer in range(layers):
        out = kda_fused_decode_torch_reference(
            *(case["step"][key][layer] for key in STEP_KEYS),
            conv_state=case["carriers"]["conv"][layer],
            recurrent_state=case["carriers"]["rec"][layer],
            gate_lower_bound=LOWER,
            conv_state_dim_first=False,
            start_position=position,
            **case["weights"],
        )
        cores.append(out.core)
        convs.append(out.conv_state)
        recs.append(out.recurrent_state)
    return {"core": torch.stack(cores), "conv": torch.stack(convs), "rec": torch.stack(recs)}


def device_inputs(case: dict, device: str) -> tuple:
    step = tuple(case["step"][key].to(device) for key in STEP_KEYS)
    carriers = (
        case["carriers"]["conv"].clone().to(device),
        case["carriers"]["rec"].clone().to(device),
        case["position"].to(device),
    )
    weights = tuple(case["weights"][key].to(device) for key in WEIGHT_KEYS)
    return step + carriers + weights


def measure(model, inputs, warmup: int, iterations: int, layers: int) -> dict:
    for _ in range(warmup):
        model(*inputs)[0].to("cpu")
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        model(*inputs)[0].to("cpu")
        samples.append((time.perf_counter_ns() - started) / 1_000)
    samples.sort()
    p90 = samples[min(len(samples) - 1, int(0.9 * len(samples)))]
    median = statistics.median(samples)
    return {
        "iterations": iterations,
        "warmup": warmup,
        "graph_median_us": median,
        "graph_p90_us": p90,
        "graph_min_us": samples[0],
        "graph_max_us": samples[-1],
        "median_us": median / layers,
        "p90_us": p90 / layers,
        "samples_us": samples,
    }


def run_case(batch: int, layers: int, loader, kernels, args) -> dict:
    torch._dynamo.reset()
    case = make_case(batch, layers, seed=9100 + 13 * batch + layers)
    reference = cpu_reference(case, layers)
    compiled = {
        name: torch.compile(
            fn,
            backend="neuron_libtorch",
            fullgraph=True,
            dynamic=False,
            options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")},
        )
        for name, fn in build_variants(loader, kernels, layers, batch).items()
    }
    result = {"B": batch, "layers": layers, "variants": {}}
    outputs = {}
    for name, model in compiled.items():
        inputs = device_inputs(case, "neuron:0")
        started = time.perf_counter()
        returned = model(*inputs)
        core = returned[0].to("cpu")
        first_call_s = time.perf_counter() - started
        conv = returned[1].to("cpu")
        rec = returned[2].to("cpu")
        if not torch.isfinite(core).all() or not torch.isfinite(rec).all():
            raise AssertionError(f"{name} returned non-finite values")
        outputs[name] = {"core": core, "conv": conv, "rec": rec}
        if name == "floor":
            result["variants"][name] = {"first_call_s": first_call_s}
            continue
        result["variants"][name] = {
            "first_call_s": first_call_s,
            "vs_cpu_reference": {
                "core": metrics(core, reference["core"]),
                "recurrent_state": metrics(rec, reference["rec"]),
                "conv_state": metrics(conv, reference["conv"]),
            },
        }
        # The reference advanced the carriers, so a variant that returned them
        # unadvanced reads a relative L2 near 1 here, not near 0.
        for key, label in (("core", "core"), ("recurrent_state", "rec"),
                           ("conv_state", "conv")):
            rel = result["variants"][name]["vs_cpu_reference"][key]["relative_l2"]
            if rel > 1e-3:
                raise AssertionError(
                    f"{name} {label} differs from the CPU reference: rel_l2 {rel}"
                )
    for name in compiled:
        if name in ("before", "floor"):
            continue
        result["variants"][name]["vs_before"] = {
            key: metrics(outputs[name][key], outputs["before"][key])
            for key in ("core", "rec", "conv")
        }
        for key in ("core", "rec", "conv"):
            rel = result["variants"][name]["vs_before"][key]["relative_l2"]
            if rel > 1e-3:
                raise AssertionError(f"{name} {key} differs from before: rel_l2 {rel}")
    for name, model in compiled.items():
        inputs = device_inputs(case, "neuron:0")
        result["variants"][name]["timing"] = measure(
            model, inputs, args.warmup, args.iterations, layers
        )
    before = result["variants"]["before"]["timing"]
    for name in compiled:
        timing = result["variants"][name]["timing"]
        timing["speedup_vs_before"] = before["median_us"] / timing["median_us"]
    if args.profile_dir is not None:
        result["profile_dir"] = profile(compiled, case, layers, batch, args)
    result["before"] = {
        "median_us": before["median_us"],
        "p90_us": before["p90_us"],
    }
    after = result["variants"]["after"]["timing"]
    result["after"] = {"median_us": after["median_us"], "p90_us": after["p90_us"]}
    result["after_over_before_median"] = after["median_us"] / before["median_us"]
    # Graph time less the empty graph's, per layer: the launch floor removed.
    floor_us = result["variants"]["floor"]["timing"]["graph_median_us"]
    result["launch_floor_us"] = floor_us
    result["above_floor_per_layer_median_us"] = {
        name: (result["variants"][name]["timing"]["graph_median_us"] - floor_us)
        / layers
        for name in compiled
        if name != "floor"
    }
    return result


def summarize(cases: list) -> dict:
    """``before``/``after`` per layer at each ``B``, read at the deepest graph.

    A decode graph holds 34 KDA layers, so the deepest stack is the layer time a
    decode step pays; the one-layer graph is listed beside it with its launch
    floor, which the fused kernel cannot remove.
    """
    summary = {}
    for batch in sorted({case["B"] for case in cases}):
        rows = sorted(
            (case for case in cases if case["B"] == batch), key=lambda c: c["layers"]
        )
        deepest = rows[-1]
        summary[f"B={batch}"] = {
            "layers_in_graph": deepest["layers"],
            "before": deepest["before"],
            "after": deepest["after"],
            "after_over_before_median": deepest["after_over_before_median"],
            "above_floor_per_layer_median_us": deepest[
                "above_floor_per_layer_median_us"
            ],
            "by_graph_depth": {
                str(case["layers"]): {
                    "before_median_us": case["before"]["median_us"],
                    "after_median_us": case["after"]["median_us"],
                    "after_over_before_median": case["after_over_before_median"],
                    "launch_floor_us": case["launch_floor_us"],
                    "above_floor_per_layer_median_us": case[
                        "above_floor_per_layer_median_us"
                    ],
                }
                for case in rows
            },
        }
    return summary


def profile(compiled, case, layers, batch, args) -> str:
    directory = args.profile_dir / f"b{batch}_l{layers}"
    directory.mkdir(parents=True, exist_ok=True)
    runtime = torch.classes.neuron.Runtime()
    inputs = {name: device_inputs(case, "neuron:0") for name in compiled}
    runtime.start_profiling(
        str(directory),
        ["device_profile", "system_profile"],
        None,
        None,
        libtorch_envs.get_neuron_compile_cache_dir(),
    )
    try:
        for _ in range(args.profile_iterations):
            for name, model in compiled.items():
                model(*inputs[name])[0].to("cpu")
    finally:
        runtime.stop_profiling()
    return str(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 34])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile-iterations", type=int, default=3)
    args = parser.parse_args()
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Pin NEURON_RT_VISIBLE_CORES before running the benchmark")
    if args.warmup < 0 or args.iterations < 1 or args.profile_iterations < 1:
        raise ValueError("Use nonnegative warmup and positive iteration counts")
    if any(b < 1 for b in args.batches) or any(n < 1 for n in args.layers):
        raise ValueError("Batches and layer counts must be positive")
    args.baseline_module = args.baseline_module.resolve()
    args.output = args.output.resolve()
    if args.profile_dir is not None:
        args.profile_dir = args.profile_dir.resolve()
    loader, kernels = load_baseline(args.baseline_module)
    # The kernel compiler writes its artifacts into the working directory; keep
    # them out of the checkout.
    cache_root = os.environ.get("NEURON_LIBTORCH_CACHE_ROOT")
    workdir = (
        Path(cache_root) / "benchmark_kda_decode_workdir"
        if cache_root
        else Path(tempfile.mkdtemp(prefix="benchmark_kda_decode_"))
    )
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    report = {
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_NUM_CORES",
                "NEURON_LOGICAL_NC_CONFIG",
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NEURON_LIBTORCH_CACHE_ROOT",
            )
        },
        "vllm_neuron": str(Path(vllm_neuron.__file__).resolve().parent),
        "fused_grid": list(fused_decode_grid(KDIM)),
        "cores": (
            "every graph runs on neuron:0, one logical core of the leased slice; "
            "under NEURON_LOGICAL_NC_CONFIG=2 that is two physical cores, and a "
            "fused grid of [2] runs one program on each"
        ),
        "shapes": {
            "heads_per_rank": HEADS,
            "head_dim": KDIM,
            "conv_taps": TAPS,
            "conv_channels": 3 * HEADS * KDIM,
            "conv_carrier": "bfloat16 [B, 3, 384] (SD)",
            "recurrent_carrier": "float32 [B, 1, 128, 128]",
        },
        "timing_unit": (
            "microseconds per layer step = host wall time of one graph launch "
            "plus its core copy to CPU, divided by the graph's layer count; the "
            "advanced carriers are graph outputs left on device"
        ),
        "baseline_module": str(args.baseline_module),
        "compiler_workdir": str(workdir),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for layers in args.layers:
        for batch in args.batches:
            case = run_case(batch, layers, loader, kernels, args)
            report["cases"].append(case)
            report["summary"] = summarize(report["cases"])
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            summary = {k: v for k, v in case.items() if k != "variants"}
            summary["timing_us"] = {
                name: {
                    key: value
                    for key, value in variant["timing"].items()
                    if key != "samples_us"
                }
                for name, variant in case["variants"].items()
            }
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
