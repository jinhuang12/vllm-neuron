#!/usr/bin/env python3
"""Capture one frozen ordered trace on an already launched fresh server.

The launch owner supplies fresh-process.json. This controller never launches,
resets, profiles, or warms up the server. It invokes unchanged experiment.py.
All controller outputs are exclusive; failed attempts and raw rows remain.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

if __package__:
    from .launch import server_arguments
else:
    from launch import server_arguments

HERE = Path(__file__).resolve().parent
SCOPE = "Exact preservation on this fresh-process ordered trace only; not HF correctness or prompt independence."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def check_alive(output):
    require(not (output / "exit.json").exists(), "Server exit.json exists; aborting this attempt")


def source_tree(source):
    files = {str(p.relative_to(source)): digest(p)
             for p in sorted((source / "vllm_neuron").rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    for name in ("pyproject.toml", "setup.py", "setup.cfg"):
        if (source / name).is_file():
            files[name] = digest(source / name)
    require(files, "Production source tree is empty")
    return files, hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def verify_inputs(args, destination):
    contract_path = args.contract.resolve()
    raw = contract_path.read_bytes()
    with (destination / "contract.json").open("xb") as stream:
        stream.write(raw)
    require(hashlib.sha256(raw).hexdigest() == args.contract_sha256, "Contract SHA256 mismatch")
    contract = json.loads(raw)
    require(contract["schema_version"] == 1, "Unknown contract schema")
    require(contract["capture"]["parts_in_order"] == [1, 2]
            and contract["capture"]["requests"] == 20
            and contract["capture"]["output_tokens_per_request"] == 32
            and contract["comparison"]["logprob_absolute_tolerance"] == 0.0
            and contract["comparison"]["required_passes"] == 2, "Unsupported trace contract")
    require([part["file"] for part in contract["sequence"]] == ["part-1.jsonl", "part-2.jsonl"],
            "Contract part order changed")
    parts = []
    for part in contract["sequence"]:
        path = contract_path.parent / part["file"]
        require(digest(path) == part["sha256"], f"Workload SHA256 mismatch: {path.name}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        require(len(rows) == part["requests"] == 10
                and all(isinstance(row["prompt"], str) and row["output_tokens"] == 32 for row in rows),
                f"Invalid workload: {path.name}")
        parts.append(path)
    require(digest(HERE / "experiment.py") == contract["client_sha256"], "Experiment client SHA256 mismatch")
    require(digest(HERE / "config.json") == contract["config_sha256"], "Server config SHA256 mismatch")
    launch, fresh = read(args.output / "launch.json"), read(args.output / "fresh-process.json")
    require(launch["arm"] == args.arm and launch["mode"] == "serve", "Launch arm/mode mismatch")
    require(Path(launch["output"]).resolve() == args.output, "Launch output mismatch")
    require(launch["image_id"] == contract["image_id"]
            and launch["config_sha256"] == contract["config_sha256"]
            and launch["config"] == read(HERE / "config.json"), "Launch image/config mismatch")
    require(all(launch["environment"].get(k) == v for k, v in launch["config"]["environment"].items()),
            "Launch environment differs from frozen config")
    require(launch["harness_sha256"]["experiment.py"] == contract["client_sha256"], "Launched client hash mismatch")
    require(launch["contract_sha256"] in (args.contract_sha256, contract["prior_contract_sha256"]),
            "Launch contract is unrelated to this supplement")
    for key, expected in {"schema_version": 1, "arm": args.arm, "contract_sha256": args.contract_sha256,
                          "run_token": launch["run_token"], "container_name": launch["container_name"],
                          "launch_sha256": digest(args.output / "launch.json"), "fresh_process": True,
                          "model_requests_before_capture": 0,
                          "exclusive_request_owner": "ordered_trace.py"}.items():
        require(fresh.get(key) == expected, f"Fresh-process record mismatch: {key}")
    require(launch["run_token"] and launch["container_name"], "Missing fresh launch identity")
    command = launch["docker_command"]
    require("--read-only" in command, "Server container root must be read-only")
    for key in ("source", "weights", "deps"):
        path = launch[key]
        require(f"type=bind,src={path},dst={path},readonly" in command, f"Missing read-only {key} mount")
    server = launch["server_arguments"]
    port = int(server[server.index("--port") + 1])
    require(server == server_arguments(launch["config"], Path(launch["weights"]), args.output,
                                       port, launch["profile_routes_enabled"]), "Server arguments changed")
    return contract, parts, launch, f"http://127.0.0.1:{port}"


def wait_ready(output, base_url, timeout):
    deadline = time.monotonic() + timeout
    while True:
        check_alive(output)
        if (output / "runtime.json").is_file() and (output / "source-manifest.json").is_file():
            try:
                with urllib.request.urlopen(base_url + "/health", timeout=5) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                pass
        require(time.monotonic() < deadline, "Timed out waiting for server health/provenance")
        time.sleep(2)


def verify_environment(observed, expected):
    observed = dict(observed)
    # Both recorded arms show this exact SDK cv2 import side effect. Permit
    # this prefix only; every other launcher environment value must match.
    cv2_prefix = expected["VIRTUAL_ENV"] + "/lib/python3.12/site-packages/cv2/../../lib64:"
    if observed.get("LD_LIBRARY_PATH") == cv2_prefix + expected["LD_LIBRARY_PATH"]:
        observed["LD_LIBRARY_PATH"] = expected["LD_LIBRARY_PATH"]
    require(all(observed.get(k) == v for k, v in expected.items()), "Runtime launch environment mismatch")


def verify_runtime(output, launch, contract, arm):
    runtime, manifest = read(output / "runtime.json"), read(output / "source-manifest.json")
    expected = contract[f"{arm}_production_tree_sha256"]
    require(runtime["status"] == "PASS" and runtime["mode"] == "serve"
            and runtime["server_cli_validated"] is True, "Runtime preflight did not pass")
    require(runtime["versions"] == contract["runtime_versions"], "Runtime versions mismatch")
    require(runtime["source_tree_sha256"] == manifest["tree_sha256"] == expected,
            "Recorded production source tree mismatch")
    source = Path(launch["source"])
    files, actual = source_tree(source)
    require(manifest["root"] == str(source) and files == manifest["files"] and actual == expected,
            "Live production source tree mismatch")
    require(runtime["server_arguments"] == launch["server_arguments"], "Runtime server arguments mismatch")
    require(runtime["executable"] == str(Path(launch["environment"]["VIRTUAL_ENV"]) / "bin/python"),
            "Runtime SDK Python mismatch")
    verify_environment(runtime["environment"], launch["environment"])
    for name, root in (("vllm_neuron", source), ("transformers", Path(launch["deps"])),
                       ("tokenizers", Path(launch["deps"]))):
        require(root in Path(runtime["modules"][name]["path"]).parents, f"Wrong runtime module path: {name}")
    return runtime


def check_http_log(output, expected, destination, label):
    """Permit health GETs; reject any extra POST or failed completion request."""
    path = output / "server.log"
    require(path.is_file(), "Missing server.log for request ownership check")
    raw = path.read_bytes()
    rows = re.findall(r'"POST ([^ ]+) HTTP/[^" ]+"\s+(\d{3})', raw.decode(errors="replace"))
    report = {"expected_completions": expected, "post_requests": rows, "log_bytes": len(raw),
              "log_sha256": hashlib.sha256(raw).hexdigest(), "at_utc": now()}
    write(destination / f"http-{label}.json", report)
    require(len(rows) == expected and all(path == "/v1/completions" and status == "200" for path, status in rows),
            f"Unexpected server POST history: expected {expected} successful completions, found {rows}")


def client_environment(launch):
    env = os.environ.copy()
    env.update({"VLLM_NEURON_CPU_MODE": "1", "PYTHONPATH": f"{launch['deps']}:{launch['source']}",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONOPTIMIZE": "0"})
    return env


def run_client(command, env, output, destination, label):
    write(destination / f"{label}-command.json", {"argv": command, "at_utc": now(),
          "environment_overrides": {k: env[k] for k in ("VLLM_NEURON_CPU_MODE", "PYTHONPATH",
              "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "PYTHONOPTIMIZE")}})
    check_alive(output)
    with (destination / f"{label}.log").open("x") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                check_alive(output)
                time.sleep(1)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            write(destination / f"{label}-exit.json", {"returncode": child.returncode, "at_utc": now()})
    require(child.returncode == 0, f"Client failed: {label}, exit={child.returncode}")
    check_alive(output)


def validate_part(path, workload, contract_part, model):
    capture = read(path)
    rows = [json.loads(line) for line in workload.read_text().splitlines()]
    require(capture["workload_sha256"] == contract_part["sha256"], "Capture workload hash mismatch")
    require(len(capture["requests"]) == 10, "Capture does not contain ten requests")
    require(sorted(p.name for p in path.parent.glob("request-*.json")) ==
            [f"request-{i:02}.json" for i in range(10)], "Unexpected raw request file set")
    for index, (entry, row) in enumerate(zip(capture["requests"], rows)):
        expected_body = {"model": model, "prompt": row["prompt"], "max_tokens": 32,
                         "temperature": 0, "seed": 0, "ignore_eos": True, "logprobs": 5,
                         "return_token_ids": True, "return_tokens_as_token_ids": True, "stream": False}
        require(entry["index"] == index and entry["body"] == expected_body, "Capture request order/body mismatch")
        raw = path.parent / f"request-{index:02}.json"
        require(read(raw) == entry, "Capture differs from its raw request file")
        choice = entry["response"]["choices"][0]
        require(choice["finish_reason"] == "length" and len(choice["token_ids"]) == 32
                and entry["response"]["usage"]["completion_tokens"] == 32, "Incomplete capture response")
    return {"capture_sha256": digest(path), "workload_sha256": contract_part["sha256"],
            "requests": 10, "output_tokens": 320,
            "raw_sha256": {p.name: digest(p) for p in sorted(path.parent.glob("request-*.json"))}}


def verify_reference(args, contract, parts, launch):
    require(args.reference is not None, "Candidate requires --reference")
    reference = args.reference.resolve() / "ordered-trace"
    status = read(reference / "status.json")
    require(status["status"] == "CAPTURED" and status["arm"] == "baseline"
            and status["contract_sha256"] == args.contract_sha256
            and status["driver_sha256"] == digest(Path(__file__))
            and status["completed_requests"] == 20 and status["completed_output_tokens"] == 640,
            "Baseline reference is not a complete capture for this contract")
    require(digest(reference / "contract.json") == args.contract_sha256, "Baseline contract copy mismatch")
    require(status["run_token"] != launch["run_token"] and status["container_name"] != launch["container_name"],
            "Baseline and candidate must have different fresh processes")
    require(len(status["parts"]) == 2, "Baseline part count mismatch")
    for index in range(2):
        receipt = validate_part(reference / f"part-{index + 1}" / "capture.json", parts[index],
                                contract["sequence"][index], launch["weights"])
        require(receipt == status["parts"][index], "Baseline part changed after capture")
    return reference


def run(args):
    args.output = args.output.resolve()
    require(args.output.is_dir(), "--output must be the existing fresh server artifact directory")
    destination = args.output / "ordered-trace"
    destination.mkdir(exist_ok=False)
    status = {"schema_version": 1, "status": "RUNNING", "arm": args.arm, "scope": SCOPE,
              "contract_sha256": args.contract_sha256, "driver_sha256": digest(Path(__file__)),
              "original_AA_repeatability_gate": "FAIL, retained unchanged",
              "supplement_does_not_pass_original_correctness_gate": True,
              "started_utc": now(), "completed_requests": 0, "completed_output_tokens": 0, "parts": []}
    write(destination / "start.json", status)
    stage = "preflight"
    try:
        check_alive(args.output)
        contract, parts, launch, base_url = verify_inputs(args, destination)
        status.update(run_token=launch["run_token"], container_name=launch["container_name"])
        reference = verify_reference(args, contract, parts, launch) if args.arm == "candidate" else None
        require(args.arm == "candidate" or args.reference is None, "Baseline cannot take --reference")
        wait_ready(args.output, base_url, args.wait_seconds)
        runtime = verify_runtime(args.output, launch, contract, args.arm)
        check_http_log(args.output, 0, destination, "before")
        write(destination / "preflight.json", {"status": "PASS", "at_utc": now(),
              "input_sha256": {name: digest(args.output / name) for name in
                               ("launch.json", "fresh-process.json", "runtime.json", "source-manifest.json")},
              "client_sha256": digest(HERE / "experiment.py"), "part_sha256": [digest(p) for p in parts]})
        env = client_environment(launch)
        client = [runtime["executable"], str(HERE / "experiment.py")]
        for index, part in enumerate(parts, 1):
            stage = f"capture-part-{index}"
            # Recheck immutable inputs before each child; there are no model retries.
            require(digest(part) == contract["sequence"][index - 1]["sha256"]
                    and digest(HERE / "experiment.py") == contract["client_sha256"], "Capture inputs changed")
            capture_dir = destination / f"part-{index}"
            command = [*client, "capture", "--model", launch["weights"], "--base-url", base_url,
                       "--workload", str(part), "--output", str(capture_dir)]
            run_client(command, env, args.output, destination, stage)
            receipt = validate_part(capture_dir / "capture.json", part, contract["sequence"][index - 1], launch["weights"])
            status["parts"].append(receipt)
            status["completed_requests"] += 10
            status["completed_output_tokens"] += 320
            check_http_log(args.output, index * 10, destination, f"after-part-{index}")
        if reference is not None:
            verify_reference(args, contract, parts, launch)
            status["comparisons"] = []
            for index in (1, 2):
                stage = f"compare-part-{index}"
                require(digest(HERE / "experiment.py") == contract["client_sha256"], "Comparator client changed")
                result_path = destination / f"comparison-part-{index}.json"
                run_client([*client, "compare", "--left", str(reference / f"part-{index}" / "capture.json"),
                            "--right", str(destination / f"part-{index}" / "capture.json"), "--output", str(result_path)],
                           env, args.output, destination, stage)
                result = read(result_path)
                require(result["pass"] is True and result["max_absolute_logprob_delta"] == 0.0
                        and result["required_absolute_logprob_delta"] == 0.0 and result["mismatches"] == [],
                        f"Exact comparison failed: part {index}")
                require(result["left_sha256"] == status_from_reference(args.reference, index)
                        and result["right_sha256"] == status["parts"][index - 1]["capture_sha256"],
                        "Comparison input hashes changed")
                status["comparisons"].append({"part": index, "sha256": digest(result_path), "pass": True})
            status["reference"] = {"path": str(reference), "status_sha256": digest(reference / "status.json")}
        check_alive(args.output)
        check_http_log(args.output, 20, destination, "complete")
        status["status"] = "PASS" if args.arm == "candidate" else "CAPTURED"
        return_code = 0
    except (Exception, KeyboardInterrupt) as error:
        status.update(status="FAIL", failed_stage=stage, error=f"{type(error).__name__}: {error}")
        return_code = 1
    status["finished_utc"] = now()
    write(destination / "status.json", status)
    print(json.dumps({"status": status["status"], "output": str(destination), "error": status.get("error")}))
    return return_code


def status_from_reference(reference, index):
    return read(reference / "ordered-trace/status.json")["parts"][index - 1]["capture_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--wait-seconds", type=float, default=28800)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
