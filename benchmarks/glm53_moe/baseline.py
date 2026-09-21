"""Run the three real NKI limbs on Trainium, without starting a model server.

Device latency is measured by NKI's runtime benchmark after warmup. The sum
of the three device times excludes launches and host transfers. This is an
isolated kernel baseline, not a serving latency measurement.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from .reference import make_fixture, metrics, routed_down_reference, routed_reference, swiglu


def to_numpy(tensor):
    """Preserve BF16 and FP8 bits when passing CPU tensors to standalone NKI."""
    import ml_dtypes
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    if tensor.dtype == torch.float8_e4m3fn:
        return tensor.view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
    return tensor.numpy()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def packed_inputs(case):
    """Build only the public routed operands, with no production math helpers."""
    experts, hidden, _, intermediate = case.gate_weight.shape
    gate_scales = case.gate_scales.reshape(experts, 1, -1).expand(-1, 128, -1).contiguous()
    down_scales = case.down_scales.reshape(experts, 1, -1).expand(-1, 128, -1).contiguous()
    iota = torch.arange(128, dtype=torch.int32).reshape(128, 1)
    rows, expert_ids = case.row_index.reshape(-1, 1), case.expert_index.reshape(-1, 1)
    return {
        "gate_up": {
            "hidden": case.hidden,
            "weight_bank": case.gate_weight.reshape(experts * hidden, 2 * intermediate),
            "scale_bank": gate_scales.reshape(experts * 128, -1),
            "row_index": rows, "expert_index": expert_ids, "iota": iota,
        },
        "down": {
            "weight_bank": case.down_weight.reshape(experts * intermediate, hidden),
            "scale_bank": down_scales.reshape(experts * 128, -1),
            "affinity_bank": case.affinity.reshape(-1, 1),
            "row_index": rows, "expert_index": expert_ids, "iota": iota,
        },
    }


def bounds_operand(gate_upper, up_upper):
    if gate_upper is None and up_upper is None:
        return torch.zeros(128, 1, dtype=torch.float32)
    gate = float("inf") if gate_upper is None else gate_upper
    up = float("inf") if up_upper is None else up_upper
    return torch.tensor([gate, -up, up], dtype=torch.float32).expand(128, -1).contiguous()


def compile_kernel(kernel, inputs, artifact_dir, lnc=2):
    """NKI 0.6 standalone compiler path; this helper performs no device run.

    These compiler APIs are internal to the installed NKI version. Record that
    version with every result. Keep all compiler artifacts for review.
    """
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    from nki.framework.compiled import compile_kernel_to_nir

    artifact_dir = Path(artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=False)
    arrays = {name: to_numpy(value) if isinstance(value, torch.Tensor) else value
              for name, value in inputs.items()}
    options = CompileOptions(target="trn2", lnc=lnc, artifacts_dir=str(artifact_dir),
                             output_path=str(artifact_dir / "kernel.neff"))
    # Match @nki.jit's default compile path, which disables experimental backend
    # transforms. The shipped kernel's scheduling choices remain in force.
    options = options.disable_backend_optimizations()
    nir = compile_kernel_to_nir(kernel[lnc], inputs=arrays, compile_opts=options,
                                enable_cache=False)
    compiled = compile_bir_to_neff(
        options, nir, input_arrays=[],
        argument_names=[spec.name for spec in nir.descriptor.input_specs],
        output_arg_names=[spec.name for spec in nir.descriptor.output_specs],
    )
    return compiled, compiled.prepare_inputs(arrays)


def summarize_trace(events_json, expected_iterations):
    """Group PNC events by actual launch ID; keep the slowest core per launch.

    Per-core durations use each core's clock. Taking their maximum needs no
    assumption that different PNC clock origins match. Also retain the raw
    timestamp span as a diagnostic, not the primary latency.
    """
    starts, launches = {}, {}
    for event in json.loads(events_json)["events"]:
        if event.get("event_type") != "nc_exec_running":
            continue
        key = event["tracking_id"]
        data = event["data"]
        if event["phase"] == "start":
            starts[key] = data
        elif event["phase"] == "stop":
            start = starts.pop(key)
            if data["nc_timestamp_ns"] < start["nc_timestamp_ns"]:
                raise ValueError("Device timestamp moved backward")
            launches.setdefault(start["exec_id"], []).append({
                "core": start["device_core_idx"], "start_ns": start["nc_timestamp_ns"],
                "stop_ns": data["nc_timestamp_ns"],
                "duration_us": (data["nc_timestamp_ns"] - start["nc_timestamp_ns"]) / 1000,
            })
    if starts or len(launches) != expected_iterations:
        raise ValueError(f"Incomplete timing evidence: {len(starts)} open events, {len(launches)} launches; expected {expected_iterations}")
    durations = [max(event["duration_us"] for event in events) for events in launches.values()]
    spans = [(max(event["stop_ns"] for event in events) - min(event["start_ns"] for event in events)) / 1000 for events in launches.values()]
    return {
        "mean_us": statistics.mean(durations), "min_us": min(durations),
        "max_us": max(durations), "std_us": statistics.pstdev(durations),
        "iterations": len(durations), "launch_max_core_us": durations,
        "launch_timestamp_span_us": spans,
        "core_events_per_launch": [len(events) for events in launches.values()],
        "timing_definition": "maximum PNC duration per exec_id; mean across launches",
    }


def measure(compiled, arrays, warmup=5, iterations=50):
    """Return real output and device-side latency; no CPU timing substitute."""
    import nrtpy.spike_model as spike_model
    original_parser = spike_model._parse_trace_durations
    trace_file = Path(compiled.artifacts_dir) / "benchmark_events.json"
    captured = {}

    def record_trace(events_json):
        trace_file.write_text(events_json)
        durations = original_parser(events_json)
        captured["durations_ms"] = durations
        captured["launch_summary"] = summarize_trace(events_json, iterations)
        return durations

    # Capture the exact timing evidence the installed benchmark parses. It
    # otherwise discards the trace and hides the event count from its caller.
    spike_model._parse_trace_durations = record_trace
    try:
        result = compiled.benchmark(warmup=warmup, iterations=iterations, **arrays)
    finally:
        spike_model._parse_trace_durations = original_parser
    if len(result.outputs) != 1:
        raise ValueError(f"Expected one output, got {list(result.outputs)}")
    array = next(iter(result.outputs.values()))
    if array.dtype.name != "float32":
        raise ValueError(f"The contribution contract requires FP32, got {array.dtype}")
    output = torch.from_numpy(array.copy())
    return output, {
        **captured["launch_summary"],
        "runtime_per_event_mean_us": result.latency * 1e6,
        "warmup": result.benchmark_warmup,
        "timing_event_count": len(captured["durations_ms"]),
        "timing_event_durations_ms": captured["durations_ms"],
        "raw_timing_trace": str(trace_file),
        "neff": compiled.neff_path,
        "compile_seconds": compiled.compilation_time,
        "compile_mac_count": compiled.bir.mac_count,
        "compile_hbm_bytes": compiled.bir.hbm_bytes,
    }


def replay_stages(case, source_dir, output_dir):
    """Replay saved NEFFs with exact fixture inputs and retain every boundary."""
    from nki.runtime import SpikeModel, SpikeTensor
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    operands, actual = packed_inputs(case), []
    for name in ("gate_up", "swiglu", "down"):
        if name == "gate_up":
            values = operands[name]
        elif name == "swiglu":
            values = {"gate_up": actual[0], "bounds": bounds_operand(case.gate_upper, case.up_upper)}
        else:
            values = {"intermediate_t": actual[1], **operands[name]}
        model = SpikeModel.load_from_neff(str(Path(source_dir) / name / "kernel.neff"), core_id=0)
        inputs = {key: SpikeTensor.from_numpy(to_numpy(value), name=key, core_id=0)
                  for key, value in values.items()}
        result = model(inputs)
        if len(result) != 1:
            raise ValueError(f"Expected one output for {name}")
        array = next(iter(result.values())).numpy()
        # Runtime metadata labels these NEFF outputs float32r. Its NumPy shim
        # exports a four-byte void element; the kernel contract is FP32.
        if array.dtype.kind == "V":
            if array.dtype.itemsize != 4:
                raise ValueError(f"Unexpected opaque output dtype {array.dtype}")
            array = array.view("float32")
        actual.append(torch.from_numpy(array.copy()).float())
        torch.save(actual[-1], output_dir / f"{name}.pt")
        del model
    report = diagnose_stages(case, actual)
    (output_dir / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def diagnose_stages(case, actual):
    """Separate approximation/rounding drift from incorrect down arithmetic."""
    expected = routed_reference(case)
    local_activation = swiglu(actual[0], case.gate_upper, case.up_upper).T.contiguous()
    local_down = routed_down_reference(actual[1], case)
    rounded_actual, rounded_expected = actual[1].to(torch.bfloat16), expected[1].to(torch.bfloat16)
    changed = rounded_actual != rounded_expected
    entries = []
    for i, p in changed.nonzero()[:32].tolist():
        lo, hi = sorted([float(rounded_actual[i, p]), float(rounded_expected[i, p])])
        entries.append({"intermediate": i, "position": p,
                        "actual_fp32": float(actual[1][i, p]),
                        "cpu_fp32": float(expected[1][i, p]),
                        "cpu_from_actual_gate_fp32": float(local_activation[i, p]),
                        "actual_bf16": float(rounded_actual[i, p]),
                        "cpu_bf16": float(rounded_expected[i, p]),
                        "bf16_midpoint": (lo + hi) / 2})
    return {
        "full_oracle": {name: metrics(value, oracle) for name, value, oracle in zip(("gate_up", "swiglu", "down"), actual, expected)},
        "activation_from_actual_gate": metrics(actual[1], local_activation),
        "down_from_actual_activation": metrics(actual[2], local_down),
        "bf16_activation_elements_changed": int(changed.sum()),
        "bf16_elements_changed_from_actual_gate_cpu": int((rounded_actual != local_activation.to(torch.bfloat16)).sum()),
        "first_changed_elements": entries,
    }


def run_case(case, directory, warmup=5, iterations=50):
    from vllm_neuron.functional.moe import moe_blockwise_fp8 as kernels

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(inspect.getfile(kernels)).resolve()
    (directory / "source.json").write_text(json.dumps({
        "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "functions": ["moe_gate_up_blockwise_fp8_kernel", "moe_swiglu_transposed_kernel", "moe_down_blockwise_fp8_kernel"],
    }, indent=2) + "\n")
    expected = routed_reference(case)
    operands = packed_inputs(case)
    stage_names = ("gate_up", "swiglu", "down")
    functions = (kernels.moe_gate_up_blockwise_fp8_kernel,
                 kernels.moe_swiglu_transposed_kernel,
                 kernels.moe_down_blockwise_fp8_kernel)
    reports, actual = {}, []
    for index, (name, kernel) in enumerate(zip(stage_names, functions)):
        if name == "gate_up":
            inputs = operands[name]
        elif name == "swiglu":
            inputs = {"gate_up": actual[0],
                      "bounds": bounds_operand(case.gate_upper, case.up_upper)}
        else:
            inputs = {"intermediate_t": actual[1], **operands[name]}
        compiled, arrays = compile_kernel(kernel, inputs, directory / name)
        output, timing = measure(compiled, arrays, warmup, iterations)
        actual.append(output)
        reports[name] = {"timing": timing, "accuracy": metrics(output, expected[index])}
        # Persist partial results in case a later compiler stage fails.
        (directory / "stages.json").write_text(json.dumps(reports, indent=2) + "\n")
    real = case.row_index >= 0
    torch.save(actual[-1][real], directory / "real_contributions.pt")
    torch.save(actual[-1], directory / "padded_contributions.pt")
    report = {
        "stages": reports,
        "output_real_rows": metrics(actual[-1][real], expected[-1][real]),
        "padded_output_is_zero": bool(torch.count_nonzero(actual[-1][~real]) == 0),
        "device_sum_us": sum(reports[name]["timing"]["mean_us"] for name in stage_names),
        "pass": all(reports[name]["accuracy"]["allclose"] for name in stage_names),
        "real_contributions": str(directory / "real_contributions.pt"),
    }
    report["pass"] &= report["padded_output_is_zero"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--block", type=int, choices=(128, 256), default=256)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--kind", choices=("random", "cancellation", "zeros", "clamp", "routing"), default="random")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if "NEURON_RT_VISIBLE_CORES" not in os.environ:
        raise RuntimeError("Set the lead-assigned NEURON_RT_VISIBLE_CORES before running")
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        raise RuntimeError("Set NEURON_LOGICAL_NC_CONFIG=2 for this baseline")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in ("torch", "nki", "neuronx-cc", "libtorch-neuronx-lite"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not found"
    report = {
        "schema": "glm53-moe-kernel-baseline-v1", "host": platform.node(),
        "versions": versions,
        "environment": {name: os.environ.get(name) for name in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS")},
        "hidden": 4096, "intermediate": 512, "block": args.block,
        "comparator": "production block256" if args.block == 256 else "separate block128 seam comparator",
        "timing_scope": "sum of independent device-side stage means; excludes launches and host copies",
        "kind": args.kind, "cases": {},
    }
    for q in args.q:
        case = make_fixture(q=q, experts=args.experts, block=args.block, kind=args.kind)
        started = time.time()
        try:
            case_report = run_case(case, args.output / f"q{q}", args.warmup, args.iterations)
        except Exception as error:
            report["cases"][str(q)] = {"status": "error", "type": type(error).__name__,
                                      "message": str(error), "pass": False}
            (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
            raise
        case_report["elapsed_seconds"] = time.time() - started
        case_report["input_sha256"] = {name: tensor_sha256(value) for name, value in vars(case).items() if isinstance(value, torch.Tensor)}
        report["cases"][str(q)] = case_report
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"q": q, "device_sum_us": case_report["device_sum_us"], "pass": case_report["pass"]}), flush=True)
    return 0 if all(case["pass"] for case in report["cases"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
