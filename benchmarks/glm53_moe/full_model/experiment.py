#!/usr/bin/env python3
"""Capture raw completion IDs and run fixed vLLM serving cohorts.

All output paths are new directories. Request failures abort the arm; there are
no retries or replacement samples. The upstream benchmark owns timing metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def request(base, body):
    req = urllib.request.Request(base + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def validate_capture(capture):
    for row in capture["requests"]:
        response = row["response"]
        assert len(response["choices"]) == 1
        choice = response["choices"][0]
        assert choice["finish_reason"] == "length", choice
        assert len(choice["token_ids"]) == 32, choice
        assert response["usage"]["completion_tokens"] == 32
        assert len(choice["prompt_token_ids"]) == response["usage"]["prompt_tokens"]
        logs = choice["logprobs"]
        assert len(logs["token_logprobs"]) == len(logs["top_logprobs"]) == 32
        assert all(math.isfinite(v) for v in logs["token_logprobs"])
        assert all(len(top) >= 5 and all(math.isfinite(v) for v in top.values())
                   for top in logs["top_logprobs"])


def capture(args, rows):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = {"workload_sha256": digest(args.workload), "requests": []}
    for i, row in enumerate(rows):
        body = {"model": args.model, "prompt": row["prompt"], "max_tokens": 32,
                "temperature": 0, "seed": 0, "ignore_eos": True, "logprobs": 5,
                "return_token_ids": True, "return_tokens_as_token_ids": True,
                "stream": False}
        start = time.time()
        response = request(args.base_url, body)
        entry = {"index": i, "body": body, "start_unix": start,
                 "end_unix": time.time(), "response": response}
        write(output / f"request-{i:02}.json", entry)
        result["requests"].append(entry)
        validate_capture({"requests": [entry]})
        print(f"Captured prompt {i + 1}/{len(rows)}", flush=True)
    write(output / "capture.json", result)


def compare(args):
    left, right = (json.loads(Path(p).read_text()) for p in (args.left, args.right))
    validate_capture(left)
    validate_capture(right)
    assert left["workload_sha256"] == right["workload_sha256"]
    assert len(left["requests"]) == len(right["requests"]) == 10
    mismatches = []
    max_logprob_delta = 0.0
    for i, (a, b) in enumerate(zip(left["requests"], right["requests"])):
        assert a["body"] == b["body"]
        ca, cb = a["response"]["choices"][0], b["response"]["choices"][0]
        for key in ("prompt_token_ids", "token_ids", "finish_reason", "stop_reason"):
            if ca.get(key) != cb.get(key):
                mismatches.append({"prompt": i, "field": key})
        la, lb = ca["logprobs"], cb["logprobs"]
        for step, (ta, tb) in enumerate(zip(la["top_logprobs"], lb["top_logprobs"])):
            if set(ta) != set(tb):
                mismatches.append({"prompt": i, "step": step, "field": "top5_ids"})
            for token in set(ta) & set(tb):
                max_logprob_delta = max(max_logprob_delta, abs(ta[token] - tb[token]))
        for va, vb in zip(la["token_logprobs"], lb["token_logprobs"]):
            max_logprob_delta = max(max_logprob_delta, abs(va - vb))
    result = {"left_sha256": digest(args.left), "right_sha256": digest(args.right),
              "mismatches": mismatches, "max_absolute_logprob_delta": max_logprob_delta,
              "required_absolute_logprob_delta": 0.0,
              "pass": not mismatches and max_logprob_delta == 0.0}
    write(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


def benchmark(args, rows):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    # Two complete passes cover every prompt and populate prefix-cache state.
    warmup = []
    for i in range(20):
        body = {"model": args.model, "prompt": rows[i % len(rows)]["prompt"],
                "temperature": 0, "seed": 0, "max_tokens": 32, "ignore_eos": True}
        response = request(args.base_url, body)
        assert response["usage"]["completion_tokens"] == 32
        assert response["choices"][0]["finish_reason"] == "length"
        warmup.append(response)
        print(f"Warmup {i + 1}/20", flush=True)
    write(output / "warmup.json", warmup)
    for cohort in range(5):
        name = f"cohort-{cohort:02}.json"
        command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
                   "--backend", "vllm", "--base-url", args.base_url,
                   "--endpoint", "/v1/completions", "--model", args.model,
                   "--dataset-name", "custom", "--dataset-path", str(Path(args.workload).resolve()),
                   "--custom-output-len", "32", "--num-prompts", "10", "--num-warmups", "0",
                   "--disable-shuffle", "--skip-chat-template", "--max-concurrency", "1",
                   "--request-rate", "inf", "--temperature", "0", "--ignore-eos", "--seed", "0",
                   "--save-result", "--save-detailed", "--percentile-metrics", "ttft,tpot,itl,e2el",
                   "--result-dir", str(output.resolve()), "--result-filename", name]
        write(output / f"cohort-{cohort:02}-command.json", command)
        start = time.time()
        with (output / f"cohort-{cohort:02}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        write(output / f"cohort-{cohort:02}-window.json",
              {"start_unix": start, "end_unix": time.time(),
               "scope": "Includes client startup; encloses the timed window. Extra requests depend on the installed client's ready-check policy."})
        result = json.loads((output / name).read_text())
        assert result["completed"] == 10, result
        assert result["total_output_tokens"] == 320, result
        assert result["output_lens"] == [32] * 10, result
        assert not any(result.get("errors", [])), result
        print(f"Cohort {cohort + 1}/5: {result['output_throughput']:.6f} output tokens/s", flush=True)
    write(output / "complete.json", {"cohorts": 5, "measured_requests": 50,
                                     "measured_output_tokens": 1600,
                                     "workload_sha256": digest(args.workload)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("capture", "benchmark"):
        p = sub.add_parser(mode)
        p.add_argument("--model", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--base-url", default="http://127.0.0.1:18004")
        p.add_argument("--workload", default=str(HERE / "workload.jsonl"))
    p = sub.add_parser("compare")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "compare":
        compare(args)
        return
    rows = [json.loads(line) for line in Path(args.workload).read_text().splitlines()]
    assert len(rows) == 10 and all(row["output_tokens"] == 32 for row in rows)
    (capture if args.mode == "capture" else benchmark)(args, rows)


if __name__ == "__main__":
    main()
