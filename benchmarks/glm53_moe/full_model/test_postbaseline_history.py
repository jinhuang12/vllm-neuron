"""Self-contained local mocks for the original-A prehistory controller."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from benchmarks.glm53_moe.full_model import experiment, launch
from benchmarks.glm53_moe.full_model import postbaseline_history as driver


def save(path, value):
    path.write_text(json.dumps(value, indent=2))


def response(token=0):
    return {"choices": [{"finish_reason": "length", "stop_reason": None,
             "token_ids": [token] * 32, "prompt_token_ids": [1, 2],
             "logprobs": {"token_logprobs": [-0.5] * 32,
                          "top_logprobs": [{str(i): -float(i) for i in range(5)} for _ in range(32)]}}],
            "usage": {"completion_tokens": 32, "prompt_tokens": 2}}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    harness = root / "harness"
    harness.mkdir(parents=True)
    (harness / "preservation").mkdir()
    original = driver.HERE
    for name in ("experiment.py", "config.json", "contract.json", "workload.jsonl", "collect.py", "ordered_trace.py"):
        shutil.copyfile(original / name, harness / name)
    (root / "scripts").mkdir()
    timing = root / "scripts/run_diagnostic_arm.py"; timing.write_text("# fixture timing helper\n")
    cache = root / "scripts/cache_file_state.py"; cache.write_text("# fixture cache helper\n")
    monkeypatch.setattr(driver, "HERE", harness)
    monkeypatch.setattr(driver, "ROOT", root)
    monkeypatch.setattr(driver, "TIMING_HELPER_SHA256", driver.digest(timing))
    monkeypatch.setattr(driver, "CACHE_HELPER_SHA256", driver.digest(cache))
    source = root / "baseline-source"
    (source / "vllm_neuron").mkdir(parents=True)
    (source / "vllm_neuron/__init__.py").write_text("# fixture production\n")
    files, tree = driver.common.source_tree(source)
    supplement = driver.read(original / "preservation/ordered-trace-contract.json")
    supplement.update(baseline_production_tree_sha256=tree, candidate_production_tree_sha256=tree)
    supplement_path = harness / "preservation/ordered-trace-contract.json"
    save(supplement_path, supplement)
    monkeypatch.setattr(driver, "SUPPLEMENT_SHA256", driver.digest(supplement_path))
    monkeypatch.setattr(driver.common, "wait_ready", lambda *a: None)
    baseline = root / "artifacts/baseline-r1"
    output = root / "artifacts/postbaseline-r1"
    for path, arm in ((baseline, "baseline"), (output, "postbaseline")):
        path.mkdir(parents=True)
        options = argparse.Namespace(mode="serve", arm=arm, source=str(source), weights=str(root / "weights"),
            output=str(path), cache=str(root / "cache/baseline"), deps=str(root / "deps"),
            venv=str(root / "venv"), sdk=str(root / "sdk"), port=18004, no_profile=False,
            contract=str(harness / "contract.json"), contract_sha256=driver.digest(harness / "contract.json"))
        plan = launch.build_plan(options, supplement["image_id"], "fresh-" + arm)
        save(path / "launch.json", plan)
        save(path / "source-manifest.json", {"root": str(source), "files": files, "tree_sha256": tree})
        save(path / "runtime.json", {"status": "PASS", "mode": "serve", "server_cli_validated": True,
             "versions": supplement["runtime_versions"], "source_tree_sha256": tree,
             "server_arguments": plan["server_arguments"], "executable": str(root / "venv/bin/python"),
             "environment": plan["environment"], "modules": {
                 "vllm_neuron": {"path": str(source / "vllm_neuron/__init__.py")},
                 "transformers": {"path": str(root / "deps/transformers/__init__.py")},
                 "tokenizers": {"path": str(root / "deps/tokenizers/__init__.py")}}})
    save(output / "fresh-process.json", {"schema_version": 1, "arm": "postbaseline",
         "contract_sha256": driver.SUPPLEMENT_SHA256, "run_token": plan["run_token"],
         "container_name": plan["container_name"], "launch_sha256": driver.digest(output / "launch.json"),
         "fresh_process": True, "model_requests_before_capture": 0,
         "exclusive_request_owner": "postbaseline_history.py"})
    (output / "server.log").write_text('INFO: "GET /health HTTP/1.1" 200 OK\n')
    save(baseline / "repeatability.json", {"pass": False})
    monkeypatch.setattr(experiment, "request", lambda *a: response())
    workload = harness / "workload.jsonl"
    rows = [json.loads(line) for line in workload.read_text().splitlines()]
    for index in (1, 2):
        args = argparse.Namespace(output=str(baseline / f"correctness-a{index}"), workload=str(workload),
                                  model=str(root / "weights"), base_url="unused")
        experiment.capture(args, rows)
    repeats = baseline / "repeat-diagnostic-1"
    repeats.mkdir()
    original_row = driver.read(baseline / "correctness-a1/request-00.json")
    for index in range(5):
        row = json.loads(json.dumps(original_row))
        row["body"]["prompt"] = "diagnostic body " + str(index)
        save(repeats / f"request-{index}.json", row)
    bundle = driver.build_manifest(root)
    return argparse.Namespace(root=root, output=output, inputs_sha256=bundle["inputs_sha256"], wait_seconds=0)


class MockChildren:
    def __init__(self, monkeypatch, args, *, mismatch=False, fail_request=None, timing_code=0, compare_code=None):
        self.args = args
        self.commands = []
        self.requests = []
        self.mismatch = mismatch
        self.fail_request = fail_request
        self.timing_code = timing_code
        self.compare_code = compare_code
        monkeypatch.setattr(driver, "run_process", self.run)
        monkeypatch.setattr(experiment, "request", self.request)

    def request(self, base, body):
        self.requests.append(body)
        if len(self.requests) == self.fail_request:
            raise OSError("mock request failure")
        with (self.args.output / "server.log").open("a") as stream:
            stream.write('INFO: "POST /v1/completions HTTP/1.1" 200 OK\n')
        return response(int(self.mismatch))

    def run(self, command, env, output, destination, label):
        self.commands.append((label, command))
        assert env["VLLM_NEURON_CPU_MODE"] == env["OMP_NUM_THREADS"] == "1"
        assert env["PYTHONOPTIMIZE"] == "0"
        if label == "replay-five":
            try:
                return driver.replay_five(self.args)
            except OSError:
                return 1
        if label == "timing":
            assert command[-4:] == ["--arm", "postbaseline", "--run", "postbaseline-r1"]
            if self.timing_code:
                return self.timing_code
            benchmark = output / "benchmark"; benchmark.mkdir()
            save(benchmark / "warmup.json", [response() for _ in range(20)])
            for index in range(5):
                save(benchmark / f"cohort-{index:02}.json", {"completed": 10, "total_output_tokens": 320,
                     "output_lens": [32] * 10, "errors": [""] * 10})
            save(benchmark / "complete.json", {"cohorts": 5, "measured_requests": 50,
                 "measured_output_tokens": 1600, "workload_sha256": driver.digest(self.args.root / "harness/workload.jsonl")})
            save(output / "measurement-status.json", {"phase": "diagnostic_complete"})
            save(output / "cache-timing-comparison.json", {"unchanged": True})
            with (output / "server.log").open("a") as stream:
                stream.write('INFO: "POST /v1/completions HTTP/1.1" 200 OK\n' * 75)
            return 0
        options = dict(zip((s[2:].replace("-", "_") for s in command[3::2]), command[4::2]))
        args = argparse.Namespace(**options)
        if command[2] == "capture":
            rows = [json.loads(line) for line in Path(args.workload).read_text().splitlines()]
            try:
                experiment.capture(args, rows)
            except OSError:
                return 1
            return 0
        try:
            experiment.compare(args)
        except SystemExit as error:
            return error.code if self.compare_code is None else self.compare_code
        return 0 if self.compare_code is None else self.compare_code


def status(args):
    return driver.read(args.output / "postbaseline-history/status.json")


def test_plan_is_stable_and_sends_no_requests(prepared, monkeypatch):
    monkeypatch.setattr(experiment, "request", lambda *a: pytest.fail("plan sent request"))
    one = driver.build_manifest(prepared.root)
    two = driver.build_manifest(prepared.root)
    assert one == two and one["inputs_sha256"] == prepared.inputs_sha256
    assert not (prepared.output / "postbaseline-history").exists()


@pytest.mark.parametrize("mismatch", [False, True])
def test_full_history_order_and_diagnostic_failure_do_not_change_timing(prepared, monkeypatch, mismatch):
    children = MockChildren(monkeypatch, prepared, mismatch=mismatch)
    assert driver.run(prepared) == 0
    got = status(prepared)
    assert got["status"] == "COMPLETE"
    assert [label for label, _ in children.commands] == ["capture-a1", "compare-a1", "capture-a2", "compare-a2", "replay-five", "timing"]
    assert len(children.requests) == 25
    assert got["prehistory_requests_completed"] == 25 and got["prehistory_output_tokens_completed"] == 800
    assert got["total_http_completions"] == 100 and got["measured_requests"] == 50
    assert [r["pass"] for r in got["diagnostic_comparisons"]] == [not mismatch] * 2
    assert "Original A had five extra" in got["original_AB_history_limitation"]
    originals = prepared.root / "artifacts/baseline-r1/repeat-diagnostic-1"
    assert children.requests[20:] == [driver.read(originals / f"request-{i}.json")["body"] for i in range(5)]
    with pytest.raises(FileExistsError):
        driver.run(prepared)


@pytest.mark.parametrize("failure", ["manifest", "body_changed", "helper_changed", "missing_fresh", "prior_post", "wrong_runtime", "existing_timing"])
def test_preflight_failure_sends_no_requests(prepared, monkeypatch, failure):
    children = MockChildren(monkeypatch, prepared)
    if failure == "manifest":
        prepared.inputs_sha256 = "0" * 64
    elif failure == "body_changed":
        p = prepared.root / "artifacts/baseline-r1/repeat-diagnostic-1/request-3.json"
        row = driver.read(p); row["body"]["prompt"] += " changed"; save(p, row)
    elif failure == "helper_changed":
        (prepared.root / "scripts/run_diagnostic_arm.py").write_text("changed")
    elif failure == "missing_fresh":
        (prepared.output / "fresh-process.json").unlink()
    elif failure == "prior_post":
        (prepared.output / "server.log").write_text('INFO: "POST /v1/completions HTTP/1.1" 200 OK\n')
    elif failure == "wrong_runtime":
        p = prepared.output / "runtime.json"; row = driver.read(p); row["versions"]["torch"] = "wrong"; save(p, row)
    else:
        (prepared.output / "benchmark").mkdir()
    assert driver.run(prepared) == 1
    assert status(prepared)["status"] == "FAIL"
    assert children.requests == [] and children.commands == []


@pytest.mark.parametrize("fail_at,stage,raw_count", [(3, "capture-a1", 2), (22, "replay-five", 1)])
def test_request_failure_preserves_partial_output_and_never_runs_timing(prepared, monkeypatch, fail_at, stage, raw_count):
    children = MockChildren(monkeypatch, prepared, fail_request=fail_at)
    assert driver.run(prepared) == 1
    assert status(prepared)["failed_stage"] == stage
    assert not any(label == "timing" for label, _ in children.commands)
    destination = prepared.output / "postbaseline-history"
    if stage == "capture-a1":
        assert len(list((destination / "capture-a1").glob("request-*.json"))) == raw_count
    else:
        assert (destination / "history-five/request-0.json").is_file()
        assert (destination / "history-five/request-1-input.json").is_file()
        assert (destination / "history-five/request-1-error.json").is_file()


def test_comparator_process_failure_cannot_be_treated_as_diagnostic_mismatch(prepared, monkeypatch):
    children = MockChildren(monkeypatch, prepared, compare_code=9)
    assert driver.run(prepared) == 1
    assert status(prepared)["failed_stage"] == "compare-a1"
    assert len(children.requests) == 10
    assert (prepared.output / "postbaseline-history/comparison-a1.json").is_file()


def test_timing_helper_error_is_not_complete(prepared, monkeypatch):
    children = MockChildren(monkeypatch, prepared, timing_code=7)
    assert driver.run(prepared) == 1
    assert len(children.requests) == 25
    assert status(prepared)["failed_stage"] == "timing"


def test_real_local_child_exit_and_logs_are_retained(tmp_path):
    env = driver.common.client_environment({"deps": "/deps", "source": "/source"})
    code = driver.run_process([sys.executable, "-c", "print('child failure'); raise SystemExit(7)"],
                              env, tmp_path, tmp_path, "child")
    assert code == 7 and driver.read(tmp_path / "child-exit.json")["returncode"] == 7
    assert "child failure" in (tmp_path / "child.log").read_text()
