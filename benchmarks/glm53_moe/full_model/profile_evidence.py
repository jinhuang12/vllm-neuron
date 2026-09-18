#!/usr/bin/env python3
"""Capture one live decode profile after timing, then analyze its saved evidence.

Capture uses only the standard library. Analysis needs pyarrow==21.0.0 in a
separate environment. It never replays a NEFF. A rank0 occupied-time result is
not a global critical-path estimate or an end-to-end speedup prediction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
KERNELS = ("moe_gate_up_blockwise_fp8_kernel", "moe_swiglu_transposed_kernel",
           "moe_down_blockwise_fp8_kernel", "moe_fused_fp8_kernel")
SOURCES = ("vllm_neuron/functional/moe/moe_blockwise_fp8.py",
           "vllm_neuron/functional/moe/moe_fused_fp8.py")
COMPUTATION_RE = re.compile(
    r"moe_(?:gate_up_blockwise_fp8|swiglu_transposed|down_blockwise_fp8|fused_fp8)_kernel_AwsNeuronNkiKernelWrapper\.\d+")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write(path: Path, data) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def read(path: Path):
    return json.loads(path.read_text())


def inventory(root: Path) -> list[dict]:
    return [{"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.suffix in (".neff", ".ntff", ".pb")]


def http(output: Path, name: str, url: str, body: dict | None) -> dict:
    """Persist the raw response, including an HTTP error. Never retry."""
    payload = b"" if body is None else json.dumps(body).encode()
    started = time.time_ns()
    request = urllib.request.Request(url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    record = {"url": url, "body": body, "start_unix_ns": started}
    raw = b""
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            raw = response.read()
            record.update(status=response.status, headers=dict(response.headers))
    except urllib.error.HTTPError as error:
        raw = error.read()
        record.update(status=error.code, headers=dict(error.headers), error=str(error))
    except Exception as error:
        record.update(status=None, error=f"{type(error).__name__}: {error}")
    record["end_unix_ns"] = time.time_ns()
    (output / f"{name}.body").write_bytes(raw)
    record["raw_body_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        record["response"] = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        record["response"] = None
    write(output / f"{name}.json", record)
    return record


def capture(args) -> dict:
    output, launch_path = Path(args.output), Path(args.launch)
    launch = read(launch_path)
    workload = Path(args.workload)
    complete = Path(args.timed_results) / "complete.json"
    done = read(complete)
    if (done.get("cohorts"), done.get("measured_requests"), done.get("measured_output_tokens")) != (5, 50, 1600):
        raise ValueError("The five timed cohorts must complete before profiling")
    if done.get("workload_sha256") != digest(workload):
        raise ValueError("The profile workload differs from the timed workload")
    for index in range(5):
        cohort = read(Path(args.timed_results) / f"cohort-{index:02}.json")
        if (cohort.get("completed"), cohort.get("total_output_tokens")) != (10, 320):
            raise ValueError(f"Timed cohort{index} did not complete")
    rows = [json.loads(line) for line in workload.read_text().splitlines()]
    if len(rows) != 10 or any(row.get("output_tokens") != 32 for row in rows):
        raise ValueError("Expected the frozen ten-prompt,32-output workload")
    server = launch["server_arguments"]
    additional = json.loads(server[server.index("--additional-config") + 1])
    profile_config = additional.get("neuron_profiler", {})
    profiler = json.loads(server[server.index("--profiler-config") + 1])
    if profile_config.get("neuron_cores") != [0] or profiler != {
            "profiler": "cuda", "delay_iterations": 3, "max_iterations": 1}:
        raise ValueError("Expected the frozen rank0,delay3,max1 profiler settings")
    profile_root = Path(profile_config["output_dir"])
    if any(p.stat().st_size for p in profile_root.rglob("*.ntff")):
        raise ValueError("This launch already contains a device capture; do not mix attempts")
    output.mkdir(parents=True, exist_ok=False)
    body = {"model": launch["weights"], "prompt": rows[0]["prompt"],
            "max_tokens": 32, "temperature": 0, "seed": 0, "ignore_eos": True,
            "return_token_ids": True, "stream": False}
    manifest = {"schema_version": 1, "arm": launch["arm"], "launch": str(launch_path.resolve()),
        "launch_sha256": digest(launch_path), "workload_sha256": digest(workload),
        "timed_complete_sha256": digest(complete), "profile_root": str(profile_root),
        "cache": launch["cache"], "source": launch["source"],
        "server_log": str(Path(args.server_log).resolve()), "request": body,
        "profile_script_sha256": digest(Path(__file__)), "before": inventory(profile_root)}
    write(output / "capture-inputs.json", manifest)
    started = http(output, "start-profile", args.base_url + "/start_profile", None)
    completion = None
    try:
        if started.get("status") == 200:
            completion = http(output, "completion", args.base_url + "/v1/completions", body)
    finally:
        stopped = http(output, "stop-profile", args.base_url + "/stop_profile", None)
    errors = []
    if started.get("status") != 200:
        errors.append("start_profile did not return200")
    if stopped.get("status") != 200:
        errors.append("stop_profile did not return200")
    try:
        response = completion["response"]
        assert completion["status"] == 200
        assert response["usage"]["completion_tokens"] == 32
        assert len(response["choices"]) == 1
        assert response["choices"][0]["finish_reason"] == "length"
        assert len(response["choices"][0]["token_ids"]) == 32
    except (AssertionError, KeyError, TypeError):
        errors.append("The fixed32-token completion did not validate")
    deadline = time.monotonic() + args.flush_timeout
    previous_sizes = None
    while time.monotonic() < deadline and not errors:
        sizes = {str(p): p.stat().st_size for p in profile_root.rglob("*")
                 if p.is_file() and p.suffix in (".ntff", ".pb")}
        if sizes == previous_sizes and any(k.endswith(".ntff") and v for k, v in sizes.items()) and any(k.endswith("ntrace.pb") and v for k, v in sizes.items()):
            break
        previous_sizes = sizes
        time.sleep(1)
    manifest["after"] = inventory(profile_root)
    if not any(item["path"].endswith(".ntff") and item["bytes"] for item in manifest["after"]):
        errors.append("No nonempty device trace was written")
    if not any(item["path"].endswith("ntrace.pb") and item["bytes"] for item in manifest["after"]):
        errors.append("No nonempty system trace was written")
    try:
        from collect import collect
        cache_report = collect(Path(launch["cache"]), Path(args.server_log))
    except Exception as error:
        cache_report = {"error": f"{type(error).__name__}: {error}"}
        errors.append("Cannot bind captured graphs to the cache and server log")
    write(output / "cache-evidence.json", cache_report)
    manifest.update(status="CAPTURED" if not errors else "UNAVAILABLE", reasons=errors)
    write(output / "capture.json", manifest)
    return manifest


def ingest(output: Path, explorer: str) -> bool:
    capture_record = read(output / "capture.json")
    if capture_record["status"] != "CAPTURED":
        return False
    command = [explorer, "view", "--session-dir", capture_record["profile_root"],
               "--data-path", str(output.resolve() / "explorer"),
               "--display-name", "glm53-" + capture_record["arm"] + "-" + uuid.uuid4().hex[:8],
               "--ingest-only"]
    write(output / "ingest-command.json", command)
    try:
        with (output / "ingest.log").open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        record = {"returncode": result.returncode}
    except OSError as error:
        record = {"returncode": None, "error": f"{type(error).__name__}: {error}"}
    write(output / "ingest-result.json", record)
    return record["returncode"] == 0


def union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def nc_windows(events: list[dict], device: dict) -> list[dict]:
    grouped = defaultdict(list)
    for row in events:
        if row["name"] != "nc_exec_running" or row["trace_event_source"] != "neuron_hw":
            continue
        if (row["instance_id"], row["process_id"], row["nc_idx"]) != (
                device["instance_id"], device["process_id"], device["nc_id"]):
            continue
        attrs = json.loads(row.get("extra_attributes_json") or "{}")
        flow = tuple(row.get("flow_id") or [])
        if not flow or "model_id" not in attrs:
            raise ValueError("Live execution lacks a flow/model identity")
        grouped[(attrs["model_id"], flow)].append(row)
    result = []
    for (model_id, flow), rows in grouped.items():
        if len(rows) != 2 or len({row["pcore_idx"] for row in rows}) != 2:
            raise ValueError("Expected one event from each LNC2 physical core")
        start, end = min(row["start_ts"] for row in rows), max(row["end_ts"] for row in rows)
        if max(row["start_ts"] for row in rows) >= min(row["end_ts"] for row in rows):
            raise ValueError("LNC2 execution events do not overlap")
        result.append({"model_id": model_id, "flow_id": list(flow), "start_ts": start,
                       "end_ts": end, "duration_ns": end - start, "events": rows})
    return sorted(result, key=lambda row: row["start_ts"])


def pair_execution(execution: dict, windows: list[dict]) -> dict:
    duration = execution["execution_end_ts"] - execution["execution_start_ts"]
    if duration <= 0:
        raise ValueError("Device execution duration is not positive")
    tolerance = max(10_000, round(duration * 0.002))
    candidates = [row for row in windows if abs(row["duration_ns"] - duration) <= tolerance]
    if len(candidates) != 1:
        raise ValueError(f"Device execution has {len(candidates)} possible live windows; cannot bind clocks")
    result = dict(candidates[0])
    result["device_duration_ns"] = duration
    result["duration_difference_ns"] = result["duration_ns"] - duration
    result["maximum_duration_difference_ns"] = tolerance
    result["device_to_system_offset_ns"] = result["start_ts"] - execution["execution_start_ts"]
    return result


def analyze(output: Path, parquet_root: Path | None = None) -> dict:
    report = {"status": "UNAVAILABLE", "rank": 0, "logical_core_mode": 2,
              "full_model_dispatch_proven": False,
              "complete_moe_occupied_time_fraction": None,
              "observed_moe_occupied_time_lower_bound_fraction": None,
              "global_critical_path_fraction": None, "reasons": [],
              "script_sha256": digest(Path(__file__))}
    try:
        import pyarrow.parquet as pq
        capture_record = read(output / "capture.json")
        if capture_record["status"] != "CAPTURED":
            raise ValueError("Capture did not complete: " + "; ".join(capture_record["reasons"]))
        if parquet_root is None and read(output / "ingest-result.json").get("returncode") != 0:
            raise ValueError("Neuron Explorer ingestion did not complete")
        cache = read(output / "cache-evidence.json")
        root = parquet_root or output / "explorer"
        schemas = {}
        for path in sorted(root.rglob("*.parquet")):
            if path.stem in {"Instruction", "ExecutionInfo", "Metadata", "NeffHeader",
                             "SystemProfileEvents", "DeviceProfileList", "Warning"}:
                table = pq.ParquetFile(path)
                schemas[str(path)] = {"rows": table.metadata.num_rows, "schema": str(table.schema_arrow)}
        write(output / "parquet-schemas.json", schemas)
        sys_tables = list(root.rglob("SystemProfileEvents.parquet"))
        if len(sys_tables) != 1:
            raise ValueError(f"Expected one system trace, found{len(sys_tables)}")
        system_dir = sys_tables[0].parent
        devices = pq.read_table(system_dir / "DeviceProfileList.parquet").to_pylist()
        if len(devices) != 1:
            raise ValueError(f"Expected one profiled worker, found{len(devices)}")
        device = devices[0]
        dev_dirs = [p for p in root.rglob(device["device_profile_name"] + "@latest") if p.is_dir()]
        if len(dev_dirs) != 1:
            raise ValueError("Cannot bind DeviceProfileList to one device profile directory")
        dev_dir = dev_dirs[0]
        metadata = pq.read_table(dev_dir / "Metadata.parquet").to_pylist()
        if len(metadata) != 1 or metadata[0]["num_physical_cores"] != 2 or metadata[0].get("is_simulation"):
            raise ValueError("Device trace is not a real LNC2 capture")
        log = Path(capture_record["server_log"]).read_text(errors="replace")
        rank0_pids = {int(pid) for pid in re.findall(r"Worker_TP0(?:_EP0)?\s+pid=(\d+)", log)}
        if device["process_id"] not in rank0_pids:
            raise ValueError("Profile worker PID is not identified as TP0 in the server log")
        executions = pq.read_table(dev_dir / "ExecutionInfo.parquet").to_pylist()
        if len(executions) != 1:
            raise ValueError(f"Expected max_iterations1 to yield one device execution, found{len(executions)}")
        execution = executions[0]
        neff_name = Path(execution["neff_name"]).name
        captured_neffs = [item for item in capture_record["after"]
                          if Path(item["path"]).name == neff_name]
        hashes = {item["sha256"] for item in captured_neffs}
        if len(hashes) != 1:
            raise ValueError("Executed NEFF name does not resolve to one captured NEFF hash")
        neff_hash = next(iter(hashes))
        if any(digest(Path(item["path"])) != neff_hash for item in captured_neffs):
            raise ValueError("Captured NEFF bytes changed after the capture manifest")
        graphs = [graph for graph in cache["graph_dispatch"]
                  if any(item["sha256"] == neff_hash for item in graph["neffs"])]
        if len(graphs) != 1:
            raise ValueError("Captured NEFF hash does not bind to one cached full-model FX graph")
        graph = graphs[0]
        expected = "candidate_dispatch_complete" if capture_record["arm"] == "candidate" else "baseline_dispatch_complete"
        if not graph[expected]:
            raise ValueError("Executed graph does not contain the expected42-layer MoE route")
        headers = pq.read_table(dev_dir / "NeffHeader.parquet").to_pylist()
        if not headers or {Path(row["network_name"]).name for row in headers} != {neff_name}:
            raise ValueError("NEFF headers and ExecutionInfo name different executables")
        if any(row.get("pcore_count_per_lnc") != 2 for row in headers):
            raise ValueError("NEFF header does not confirm LNC2")
        events = pq.read_table(sys_tables[0], filters=[("name", "=", "nc_exec_running")]).to_pylist()
        windows = nc_windows(events, device)
        if len({row["model_id"] for row in windows}) != 1:
            raise ValueError("System trace contains more than one loaded model identity")
        paired = pair_execution(execution, windows)
        report.update(device_profile=device, execution=execution, live_execution=paired,
                      neff_sha256=neff_hash, neff_name=neff_name, graph_key=graph["cache_key"],
                      graph_dispatch=graph["counts"], full_model_dispatch_proven=True,
                      rank0_pid=device["process_id"], neff_headers=headers,
                      execution_binding_method="Same captured worker, one device execution and one live model; unique LNC2 duration match anchors relative device timestamps")
        table = pq.ParquetFile(dev_dir / "Instruction.parquet")
        mandatory = {"start_ts", "end_ts", "duration_ns", "pcore_idx", "engine"}
        if not mandatory.issubset(table.schema_arrow.names):
            raise ValueError("Instruction schema lacks timestamps or physical-core identity")
        label_columns = [name for name in ("hlo_name", "hlo_attrs", "nki_source_location",
                         "bir_debug_info_source_location", "kernel_instruction_name")
                         if name in table.schema_arrow.names]
        if not label_columns:
            raise ValueError("Instruction schema has no source or kernel labels")
        columns = sorted(mandatory | set(label_columns))
        starts, ends = execution["execution_start_ts"], execution["execution_end_ts"]
        offset = paired["device_to_system_offset_ns"]
        intervals, cores, tagged_cores, symbols = [], set(), set(), set()
        counts, examples = Counter(), []
        tagged_count, instruction_count = 0, 0
        expected_symbols = set(graph.get("hlo_moe_computation_symbols", []))
        rows_path = output / "moe-instruction-intervals.jsonl"
        with rows_path.open("x") as selected:
            for batch in table.iter_batches(batch_size=8192, columns=columns):
                for row in batch.to_pylist():
                    if row["end_ts"] <= starts or row["start_ts"] >= ends:
                        continue
                    instruction_count += 1
                    cores.add(row["pcore_idx"])
                    label = " ".join(str(row.get(name) or "") for name in label_columns)
                    if not any(name in label for name in (*SOURCES, *KERNELS)):
                        continue
                    start = max(row["start_ts"] + offset, paired["start_ts"])
                    end = min(row["end_ts"] + offset, paired["end_ts"])
                    if end <= start:
                        continue
                    tagged_count += 1
                    tagged_cores.add(row["pcore_idx"])
                    counts[row["engine"]] += 1
                    symbols.update(set(COMPUTATION_RE.findall(label)) & expected_symbols)
                    intervals.append((start, end))
                    saved = {**row, "system_start_ts": start, "system_end_ts": end}
                    selected.write(json.dumps(saved) + "\n")
                    if len(examples) < 5:
                        examples.append(saved)
        if cores != {0, 1}:
            raise ValueError(f"Instruction trace does not cover both local physical cores: {sorted(cores)}")
        if not tagged_count:
            raise ValueError("No instruction has a source or symbol label for the identified MoE kernels")
        merged = union(intervals)
        occupied_ns = sum(end - start for start, end in merged)
        fraction = occupied_ns / paired["duration_ns"]
        if not 0 <= fraction <= 1:
            raise ValueError("Merged MoE intervals exceed the live execution window")
        warnings = pq.read_table(dev_dir / "Warning.parquet").to_pylist() if (dev_dir / "Warning.parquet").exists() else []
        loss_warning = any(re.search(r"overflow|dropped|truncated|lost|missing", json.dumps(row), re.I)
                           for row in warnings)
        every_computation_seen = bool(expected_symbols) and symbols == expected_symbols
        report.update(status="PARTIAL",
            source_attributed_interval_fraction=fraction,
            observed_moe_occupied_time_lower_bound_fraction=None if loss_warning else fraction,
            complete_moe_occupied_time_fraction=None,
            observed_moe_occupied_time_ns=occupied_ns,
            attributed_instruction_count=tagged_count, all_instruction_count=instruction_count,
            attributed_physical_cores=sorted(tagged_cores), physical_cores=sorted(cores),
            instruction_counts_by_engine=dict(counts), matched_hlo_symbols=sorted(symbols),
            expected_hlo_symbol_count=len(expected_symbols), warnings=warnings,
            every_compiled_moe_computation_has_a_label=every_computation_seen,
            trace_loss_warning=loss_warning,
            selected_rows_sha256=digest(rows_path), selected_examples=examples,
            merged_system_intervals=merged,
            metric="Union of source-attributed MoE instruction intervals divided by the same rank0 live decode window",
            limits="Rank0 diagnostic only. Trace loss can invalidate interval durations; a loss-affected ratio is not a certified lower bound. This is not a global critical-path or E2E removable fraction.")
        report["reasons"].append(
            "Trace loss makes interval quality uncertain; retain the observed ratio without a lower-bound claim"
            if loss_warning else
            "Complete MoE attribution is unknown; report only the observed lower bound")
    except Exception as error:
        report["reasons"].append(f"{type(error).__name__}: {error}")
    write(output / "analysis.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("capture", "run"):
        p = sub.add_parser(mode)
        for name in ("launch", "timed-results", "server-log", "output"):
            p.add_argument("--" + name, required=True)
        p.add_argument("--workload", default=str(HERE / "workload.jsonl"))
        p.add_argument("--base-url", default="http://127.0.0.1:18004")
        p.add_argument("--flush-timeout", type=float, default=60)
        p.add_argument("--explorer", default="/opt/aws/neuron/bin/neuron-explorer")
    p = sub.add_parser("analyze")
    p.add_argument("--output", required=True)
    p.add_argument("--parquet-root", type=Path)
    p = sub.add_parser("ingest")
    p.add_argument("--output", required=True)
    p.add_argument("--explorer", default="/opt/aws/neuron/bin/neuron-explorer")
    args = parser.parse_args()
    output = Path(args.output)
    failed = False
    if args.mode in ("capture", "run"):
        record = capture(args)
        print(json.dumps({"capture_status": record["status"], "reasons": record["reasons"]}), flush=True)
        failed = record["status"] == "UNAVAILABLE"
    if args.mode in ("ingest", "run"):
        failed = not ingest(output, args.explorer) or failed
    if args.mode in ("analyze", "run"):
        result = analyze(output, getattr(args, "parquet_root", None))
        print(json.dumps({"analysis_status": result["status"], "reasons": result["reasons"]}), flush=True)
        failed = result["status"] == "UNAVAILABLE" or failed
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
