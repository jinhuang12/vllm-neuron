"""Standalone NKI compiler and launch timing helpers.

Adapted locally from the prior MoE measurement plumbing. No MoE module imports.
"""


import hashlib
import json
from pathlib import Path
import statistics

import torch


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
        raise ValueError(f"The attention contract requires FP32, got {array.dtype}")
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
