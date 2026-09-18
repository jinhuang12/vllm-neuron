#!/usr/bin/env python3
"""Replay original A's request history, then call the unchanged timing helper.

--plan reads and hashes inputs only. The launch owner reviews that manifest
and supplies its SHA256 to run mode. No arm is launched by this controller.
Diagnostic comparison failures remain visible and do not change timing gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

if __package__:
    from . import experiment
    from . import ordered_trace as common
else:
    import experiment
    import ordered_trace as common

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/ubuntu/glm53-moe-fullmodel-20260917")
SUPPLEMENT_SHA256 = "df470e4526ee8ca311bdf5dc43dc06efa8106626ee1159d0150fa37eac62c9e8"
TIMING_HELPER_SHA256 = "b8ab8c3b12cb43db46860840d15251092c4dfc34bc8cf38599098fd7a3410f51"
CACHE_HELPER_SHA256 = "67a30f1c4b0403f8fea345c5bacebe8679c9188c54f0b7954b12f4049e3e0cbc"
LIMITATION = "Original A had five extra repeat-diagnostic requests before timing; original B did not. This control replays A's history and does not remove that A/B limitation."

require, digest, read, write = common.require, common.digest, common.read, common.write


def build_manifest(root):
    """Read only: this deterministic object is frozen before the new run."""
    supplement_path = root / "harness/preservation/ordered-trace-contract.json"
    require(digest(supplement_path) == SUPPLEMENT_SHA256, "Supplement contract SHA256 mismatch")
    supplement, contract = read(supplement_path), read(root / "harness/contract.json")
    require(digest(root / "harness/contract.json") == supplement["prior_contract_sha256"], "Original contract changed")
    require(digest(root / "harness/experiment.py") == supplement["client_sha256"], "Original client changed")
    require(digest(root / "harness/config.json") == supplement["config_sha256"], "Original config changed")
    require(digest(root / "harness/workload.jsonl") == contract["workload"]["sha256"], "Original workload changed")
    require(digest(root / "scripts/run_diagnostic_arm.py") == TIMING_HELPER_SHA256, "Timing helper changed")
    require(digest(root / "scripts/cache_file_state.py") == CACHE_HELPER_SHA256, "Cache helper changed")
    baseline = root / "artifacts/baseline-r1"
    launch = read(baseline / "launch.json")
    files = [root / "harness" / name for name in ("experiment.py", "config.json", "workload.jsonl", "contract.json", "collect.py")]
    files += [supplement_path, HERE / "ordered_trace.py", Path(__file__).resolve(),
              root / "scripts/run_diagnostic_arm.py", root / "scripts/cache_file_state.py"]
    files += [baseline / name for name in ("launch.json", "runtime.json", "source-manifest.json", "repeatability.json")]
    require(read(baseline / "repeatability.json")["pass"] is False, "Original A/A failure must remain visible")
    for part in (1, 2):
        directory = baseline / f"correctness-a{part}"
        common.validate_part(directory / "capture.json", root / "harness/workload.jsonl",
                             {"sha256": contract["workload"]["sha256"]}, launch["weights"])
        experiment.validate_capture(read(directory / "capture.json"))
        files.extend([directory / "capture.json", *[directory / f"request-{i:02}.json" for i in range(10)]])
    bodies = []
    for index in range(5):
        path = baseline / "repeat-diagnostic-1" / f"request-{index}.json"
        row = read(path)
        experiment.validate_capture({"requests": [row]})
        require(row["body"]["model"] == launch["weights"] and row["body"]["max_tokens"] == 32,
                "Unexpected original diagnostic model/body")
        bodies.append(row["body"])
        files.append(path)
    manifest = {"schema_version": 1, "scope": "Postbaseline control with the original A prehistory",
                "root": str(root), "run": "postbaseline-r1", "arm": "postbaseline",
                "files": {str(p.resolve()): digest(p) for p in sorted(set(files))},
                "source_capture_order": ["correctness-a1", "correctness-a2"],
                "source_diagnostic_order": list(range(5)), "diagnostic_bodies": bodies,
                "http_completion_counts": [0, 10, 20, 25, 100],
                "prehistory_requests": 25, "prehistory_output_tokens": 800,
                "benchmark_warmups": 20, "benchmark_preliminary_requests": 5,
                "measured_requests": 50, "measured_output_tokens": 1600,
                "original_AB_history_limitation": LIMITATION}
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {"inputs_sha256": hashlib.sha256(encoded).hexdigest(), "manifest": manifest}


def verify_manifest(root, expected):
    bundle = build_manifest(root)
    require(bundle["inputs_sha256"] == expected, "Reviewed input manifest SHA256 mismatch")
    return bundle


def run_process(command, env, output, destination, label):
    """Retain all child output and stop its owned process group on server exit."""
    write(destination / f"{label}-command.json", {"argv": command, "at_utc": common.now(),
          "environment_overrides": {k: env[k] for k in ("VLLM_NEURON_CPU_MODE", "PYTHONPATH",
              "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "PYTHONOPTIMIZE")}})
    common.check_alive(output)
    with (destination / f"{label}.log").open("x") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                common.check_alive(output)
                time.sleep(1)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            write(destination / f"{label}-exit.json", {"returncode": child.returncode, "at_utc": common.now()})
    common.check_alive(output)
    return child.returncode


def verify_launch(args, supplement):
    original = read(args.root / "artifacts/baseline-r1/launch.json")
    launch, fresh = read(args.output / "launch.json"), read(args.output / "fresh-process.json")
    require(launch["arm"] == "postbaseline" and launch["mode"] == "serve", "Wrong postbaseline launch arm/mode")
    require(Path(launch["output"]).resolve() == args.output, "Wrong launch output")
    require(launch["run_token"] != original["run_token"]
            and launch["container_name"] != original["container_name"], "Postbaseline process is not fresh")
    for key in ("image_id", "config_sha256"):
        require(launch[key] == original[key] == supplement[key], f"Postbaseline {key} mismatch")
    require(launch["config"] == original["config"] == read(args.root / "harness/config.json"), "Server config mismatch")
    for key in ("source", "weights", "deps", "cache"):
        require(launch[key] == original[key], f"Postbaseline must use original A {key}")
    require(launch["harness_sha256"]["experiment.py"] == supplement["client_sha256"], "Launched client changed")
    require(launch["contract_sha256"] in (SUPPLEMENT_SHA256, supplement["prior_contract_sha256"]), "Unrelated launch contract")
    for key, expected in {"schema_version": 1, "arm": "postbaseline", "contract_sha256": SUPPLEMENT_SHA256,
                          "run_token": launch["run_token"], "container_name": launch["container_name"],
                          "launch_sha256": digest(args.output / "launch.json"), "fresh_process": True,
                          "model_requests_before_capture": 0,
                          "exclusive_request_owner": "postbaseline_history.py"}.items():
        require(fresh.get(key) == expected, f"Fresh-process record mismatch: {key}")
    require("--read-only" in launch["docker_command"], "Container root must be read-only")
    for key in ("source", "weights", "deps"):
        path = launch[key]
        require(f"type=bind,src={path},dst={path},readonly" in launch["docker_command"], f"Missing read-only {key} mount")
    require(all(launch["environment"].get(k) == v for k, v in launch["config"]["environment"].items()),
            "Launch environment differs from config")
    server = common.server_arguments(launch["config"], Path(launch["weights"]), args.output,
                                     18004, original["profile_routes_enabled"])
    require(launch["server_arguments"] == server, "Postbaseline server arguments mismatch")
    for name in ("benchmark", "measurement-status.json", "performance-diagnostic-scope.json",
                 "cache-ready-file-state.json", "cache-after-timing-file-state.json", "cache-after.json"):
        require(not (args.output / name).exists(), f"Timing output already exists: {name}")
    return launch


def replay_five(args):
    verify_manifest(args.root, args.inputs_sha256)
    destination = args.output / "postbaseline-history/history-five"
    destination.mkdir(exist_ok=False)
    responses = []
    for index in range(5):
        common.check_alive(args.output)
        source = args.root / "artifacts/baseline-r1/repeat-diagnostic-1" / f"request-{index}.json"
        original = read(source)
        entry = {"source_file": str(source), "source_sha256": digest(source), "body": original["body"],
                 "start_unix": time.time()}
        write(destination / f"request-{index}-input.json", entry)
        try:
            entry["response"] = experiment.request("http://127.0.0.1:18004", original["body"])
            entry["end_unix"] = time.time()
            write(destination / f"request-{index}.json", entry)
            experiment.validate_capture({"requests": [entry]})
            responses.append(entry)
        except Exception as error:
            write(destination / f"request-{index}-error.json", {"error": f"{type(error).__name__}: {error}"})
            raise
    write(destination / "complete.json", {"requests": 5, "output_tokens": 160, "inputs_sha256": args.inputs_sha256})
    return 0


def diagnostic_result(path, code, left, right):
    require(code in (0, 1), f"Diagnostic comparator failed with exit {code}")
    result = read(path)
    require(result["pass"] is (code == 0)
            and result["left_sha256"] == digest(left) and result["right_sha256"] == digest(right)
            and result["required_absolute_logprob_delta"] == 0.0
            and math.isfinite(result["max_absolute_logprob_delta"])
            and result["max_absolute_logprob_delta"] >= 0.0
            and isinstance(result["mismatches"], list), "Malformed diagnostic comparison result")
    require(result["pass"] is (not result["mismatches"] and result["max_absolute_logprob_delta"] == 0.0),
            "Diagnostic comparison result is inconsistent")
    return {"returncode": code, "pass": result["pass"], "sha256": digest(path)}


def verify_timing(output, workload_sha256):
    complete = read(output / "benchmark/complete.json")
    require(complete == {"cohorts": 5, "measured_requests": 50, "measured_output_tokens": 1600,
                         "workload_sha256": workload_sha256}, "Timing helper completion scope mismatch")
    warmup = read(output / "benchmark/warmup.json")
    require(len(warmup) == 20 and all(x["usage"]["completion_tokens"] == 32
            and x["choices"][0]["finish_reason"] == "length" for x in warmup), "Incomplete benchmark warmups")
    for index in range(5):
        cohort = read(output / "benchmark" / f"cohort-{index:02}.json")
        require(cohort["completed"] == 10 and cohort["total_output_tokens"] == 320
                and cohort["output_lens"] == [32] * 10 and not any(cohort.get("errors", [])), "Incomplete timed cohort")
    require(read(output / "measurement-status.json")["phase"] == "diagnostic_complete", "Timing helper did not finish")
    require(read(output / "cache-timing-comparison.json")["unchanged"] is True, "Compile/cache state changed during timing")


def run(args):
    args.root, args.output = args.root.resolve(), args.output.resolve()
    require(args.root == ROOT and args.output == ROOT / "artifacts/postbaseline-r1", "Timing helper requires its canonical root/run")
    require(args.output.is_dir(), "Fresh server artifact directory does not exist")
    destination = args.output / "postbaseline-history"
    destination.mkdir(exist_ok=False)
    status = {"schema_version": 1, "status": "RUNNING", "arm": "postbaseline", "started_utc": common.now(),
              "driver_sha256": digest(Path(__file__)), "inputs_sha256": args.inputs_sha256,
              "original_AA_repeatability_gate": "FAIL, retained unchanged", "original_AB_history_limitation": LIMITATION,
              "prehistory_requests_completed": 0, "prehistory_output_tokens_completed": 0, "diagnostic_comparisons": []}
    write(destination / "start.json", status)
    stage = "preflight"
    try:
        common.check_alive(args.output)
        bundle = verify_manifest(args.root, args.inputs_sha256)
        write(destination / "input-manifest.json", bundle)
        supplement = read(args.root / "harness/preservation/ordered-trace-contract.json")
        launch = verify_launch(args, supplement)
        common.wait_ready(args.output, "http://127.0.0.1:18004", args.wait_seconds)
        runtime = common.verify_runtime(args.output, launch, supplement, "baseline")
        common.check_http_log(args.output, 0, destination, "before")
        write(destination / "preflight.json", {"status": "PASS", "input_sha256": {
            name: digest(args.output / name) for name in ("launch.json", "fresh-process.json", "runtime.json", "source-manifest.json")}})
        status.update(run_token=launch["run_token"], container_name=launch["container_name"])
        env = common.client_environment(launch)
        client = [runtime["executable"], str(args.root / "harness/experiment.py")]
        workload = args.root / "harness/workload.jsonl"
        workload_sha256 = digest(workload)
        for index in (1, 2):
            stage = f"capture-a{index}"
            verify_manifest(args.root, args.inputs_sha256)
            part = destination / f"capture-a{index}"
            code = run_process([*client, "capture", "--model", launch["weights"], "--base-url", "http://127.0.0.1:18004",
                                "--workload", str(workload), "--output", str(part)], env, args.output, destination, stage)
            require(code == 0, f"Capture failed with exit {code}")
            common.validate_part(part / "capture.json", workload, {"sha256": workload_sha256}, launch["weights"])
            experiment.validate_capture(read(part / "capture.json"))
            status["prehistory_requests_completed"] += 10
            status["prehistory_output_tokens_completed"] += 320
            common.check_http_log(args.output, index * 10, destination, f"after-a{index}")
            stage = f"compare-a{index}"
            left = args.root / "artifacts/baseline-r1" / f"correctness-a{index}/capture.json"
            right = part / "capture.json"
            result = destination / f"comparison-a{index}.json"
            code = run_process([*client, "compare", "--left", str(left), "--right", str(right), "--output", str(result)],
                               env, args.output, destination, stage)
            status["diagnostic_comparisons"].append(diagnostic_result(result, code, left, right))
        stage = "replay-five"
        code = run_process([runtime["executable"], str(Path(__file__).resolve()), "--replay-five",
                            "--root", str(args.root), "--output", str(args.output), "--inputs-sha256", args.inputs_sha256],
                           env, args.output, destination, stage)
        require(code == 0, f"Five-request replay failed with exit {code}")
        for index in range(5):
            captured = read(destination / "history-five" / f"request-{index}.json")
            original = read(args.root / "artifacts/baseline-r1/repeat-diagnostic-1" / f"request-{index}.json")
            require(captured["body"] == original["body"], "Five-request replay body changed")
            experiment.validate_capture({"requests": [captured]})
        require(read(destination / "history-five/complete.json") ==
                {"requests": 5, "output_tokens": 160, "inputs_sha256": args.inputs_sha256}, "Incomplete five-request replay")
        status["prehistory_requests_completed"] += 5
        status["prehistory_output_tokens_completed"] += 160
        common.check_http_log(args.output, 25, destination, "after-five")
        stage = "timing"
        verify_manifest(args.root, args.inputs_sha256)
        code = run_process([runtime["executable"], str(args.root / "scripts/run_diagnostic_arm.py"),
                            "--arm", "postbaseline", "--run", "postbaseline-r1"], env, args.output, destination, stage)
        require(code == 0, f"Timing helper failed with exit {code}")
        verify_timing(args.output, workload_sha256)
        common.check_http_log(args.output, 100, destination, "after-timing")
        status.update(status="COMPLETE", measured_requests=50, measured_output_tokens=1600,
                      total_http_completions=100, timing_acceptance="Parent applies unchanged performance gates; this receipt does not pass them.")
        return_code = 0
    except (Exception, KeyboardInterrupt) as error:
        status.update(status="FAIL", failed_stage=stage, error=f"{type(error).__name__}: {error}")
        return_code = 1
    status["finished_utc"] = common.now()
    write(destination / "status.json", status)
    print(json.dumps({"status": status["status"], "output": str(destination), "error": status.get("error")}))
    return return_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--replay-five", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inputs-sha256")
    parser.add_argument("--wait-seconds", type=float, default=28800)
    args = parser.parse_args()
    args.output = args.output or args.root / "artifacts/postbaseline-r1"
    if args.plan:
        print(json.dumps(build_manifest(args.root.resolve()), indent=2))
        return 0
    if not args.inputs_sha256:
        parser.error("run requires the reviewed --inputs-sha256")
    return replay_five(args) if args.replay_five else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
