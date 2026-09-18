"""Local controller checks; no server, socket, SDK, or device is started."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from benchmarks.glm53_moe.full_model import experiment, launch, ordered_trace as driver


def save(path, value):
    path.write_text(json.dumps(value, indent=2))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    harness = tmp_path / "harness"
    harness.mkdir()
    original = driver.HERE
    for name in ("config.json", "experiment.py"):
        shutil.copyfile(original / name, harness / name)
    shutil.copytree(original / "preservation", harness / "preservation")
    monkeypatch.setattr(driver, "HERE", harness)
    source = tmp_path / "source"
    (source / "vllm_neuron").mkdir(parents=True)
    (source / "vllm_neuron/__init__.py").write_text("# fixture production source\n")
    files, tree = driver.source_tree(source)
    contract_path = harness / "preservation/ordered-trace-contract.json"
    contract = driver.read(contract_path)
    contract.update(baseline_production_tree_sha256=tree, candidate_production_tree_sha256=tree)
    save(contract_path, contract)
    monkeypatch.setattr(driver, "wait_ready", lambda *a: None)

    def arm(name, reference=None):
        output = tmp_path / name
        output.mkdir()
        options = argparse.Namespace(mode="serve", arm=name, source=str(source),
            weights=str(tmp_path / "weights"), output=str(output), cache=str(tmp_path / (name + "-cache")),
            deps=str(tmp_path / "deps"), venv=str(tmp_path / "venv"), sdk=str(tmp_path / "sdk"),
            port=18004, no_profile=False, contract=str(contract_path), contract_sha256=driver.digest(contract_path))
        plan = launch.build_plan(options, contract["image_id"], "fresh-" + name)
        save(output / "launch.json", plan)
        save(output / "fresh-process.json", {"schema_version": 1, "arm": name,
             "contract_sha256": driver.digest(contract_path), "run_token": plan["run_token"],
             "container_name": plan["container_name"], "launch_sha256": driver.digest(output / "launch.json"),
             "fresh_process": True, "model_requests_before_capture": 0,
             "exclusive_request_owner": "ordered_trace.py"})
        save(output / "source-manifest.json", {"root": str(source), "files": files, "tree_sha256": tree})
        save(output / "runtime.json", {"status": "PASS", "mode": "serve", "server_cli_validated": True,
             "versions": contract["runtime_versions"], "source_tree_sha256": tree,
             "server_arguments": plan["server_arguments"], "executable": str(tmp_path / "venv/bin/python"),
             "environment": plan["environment"], "modules": {
                 "vllm_neuron": {"path": str(source / "vllm_neuron/__init__.py")},
                 "transformers": {"path": str(tmp_path / "deps/transformers/__init__.py")},
                 "tokenizers": {"path": str(tmp_path / "deps/tokenizers/__init__.py")}}})
        (output / "server.log").write_text('INFO: "GET /health HTTP/1.1" 200 OK\n')
        return argparse.Namespace(arm=name, output=output, contract=contract_path,
             contract_sha256=driver.digest(contract_path), reference=reference, wait_seconds=0)
    return arm


class MockClient:
    """Use the unchanged capture/comparator with mocked completion responses."""
    def __init__(self, monkeypatch, fail_request=None, fail_compare=False, mismatch=False):
        self.commands = []
        self.requests = 0
        self.fail_request = fail_request
        self.fail_compare = fail_compare
        self.mismatch = mismatch
        monkeypatch.setattr(driver, "run_client", self.run)
        monkeypatch.setattr(experiment, "request", self.request)

    def request(self, base, body):
        self.requests += 1
        if self.requests == self.fail_request:
            raise OSError("mock request failure")
        with (self.output / "server.log").open("a") as stream:
            stream.write('INFO: "POST /v1/completions HTTP/1.1" 200 OK\n')
        token = 1 if self.mismatch else 0
        return {"choices": [{"finish_reason": "length", "stop_reason": None,
                "token_ids": [token] * 32, "prompt_token_ids": [1, 2],
                "logprobs": {"token_logprobs": [-0.5] * 32,
                             "top_logprobs": [{str(i): -float(i) for i in range(5)} for _ in range(32)]}}],
                "usage": {"completion_tokens": 32, "prompt_tokens": 2}}

    def run(self, command, env, output, destination, label):
        self.commands.append(command)
        self.output = output
        assert env["VLLM_NEURON_CPU_MODE"] == env["OMP_NUM_THREADS"] == "1"
        assert env["PYTHONOPTIMIZE"] == "0"
        mode = command[2]
        options = dict(zip((s[2:].replace("-", "_") for s in command[3::2]), command[4::2]))
        args = argparse.Namespace(**options)
        if mode == "capture":
            rows = [json.loads(line) for line in Path(args.workload).read_text().splitlines()]
            experiment.capture(args, rows)
        else:
            try:
                experiment.compare(args)
            except SystemExit as error:
                raise subprocess.CalledProcessError(error.code, command) from error
            if self.fail_compare:
                raise subprocess.CalledProcessError(7, command)


def result(args):
    return driver.read(args.output / "ordered-trace/status.json")


def test_two_parts_run_in_order_with_exact_20_request_scope(fixture, monkeypatch):
    args = fixture("baseline")
    client = MockClient(monkeypatch)
    assert driver.run(args) == 0
    status = result(args)
    assert status["status"] == "CAPTURED"
    assert status["completed_requests"] == 20 and status["completed_output_tokens"] == 640
    assert status["original_AA_repeatability_gate"] == "FAIL, retained unchanged"
    assert [Path(c[c.index("--workload") + 1]).name for c in client.commands] == ["part-1.jsonl", "part-2.jsonl"]
    assert client.requests == 20
    assert all(len(p["raw_sha256"]) == 10 for p in status["parts"])
    with pytest.raises(FileExistsError):
        driver.run(args)
    assert client.requests == 20


@pytest.mark.parametrize("mutation", ["missing_part", "bad_part", "bad_contract", "bad_client",
                                      "missing_fresh", "wrong_fresh", "wrong_runtime", "wrong_source", "prior_request", "exited"])
def test_invalid_inputs_abort_without_model_requests(fixture, monkeypatch, mutation):
    args = fixture("baseline")
    client = MockClient(monkeypatch)
    part = args.contract.parent / "part-2.jsonl"
    if mutation == "missing_part":
        part.unlink()
    elif mutation == "bad_part":
        part.write_text(part.read_text() + "\n")
    elif mutation == "bad_contract":
        args.contract_sha256 = "0" * 64
    elif mutation == "bad_client":
        (driver.HERE / "experiment.py").write_text("altered")
    elif mutation == "missing_fresh":
        (args.output / "fresh-process.json").unlink()
    elif mutation == "wrong_fresh":
        p = args.output / "fresh-process.json"; x = driver.read(p); x["run_token"] = "wrong"; save(p, x)
    elif mutation == "wrong_runtime":
        p = args.output / "runtime.json"; x = driver.read(p); x["versions"]["torch"] = "wrong"; save(p, x)
    elif mutation == "wrong_source":
        p = Path(driver.read(args.output / "launch.json")["source"]) / "vllm_neuron/__init__.py"; p.write_text("altered")
    elif mutation == "prior_request":
        (args.output / "server.log").write_text('INFO: "POST /v1/completions HTTP/1.1" 200 OK\n')
    else:
        save(args.output / "exit.json", {"returncode": 1})
    assert driver.run(args) == 1
    assert result(args)["status"] == "FAIL"
    assert client.requests == 0 and not client.commands


def test_capture_failure_preserves_partial_rows_and_does_not_start_second_part(fixture, monkeypatch):
    args = fixture("baseline")
    client = MockClient(monkeypatch, fail_request=3)
    assert driver.run(args) == 1
    assert len(client.commands) == 1 and client.requests == 3
    assert result(args)["failed_stage"] == "capture-part-1"
    part = args.output / "ordered-trace/part-1"
    assert sorted(p.name for p in part.iterdir()) == ["request-00.json", "request-01.json"]
    assert not (args.output / "ordered-trace/part-2").exists()


def test_candidate_uses_same_contract_parts_and_unchanged_exact_comparator(fixture, monkeypatch):
    baseline = fixture("baseline")
    MockClient(monkeypatch)
    assert driver.run(baseline) == 0
    candidate = fixture("candidate", baseline.output)
    client = MockClient(monkeypatch)
    assert driver.run(candidate) == 0
    assert [c[2] for c in client.commands] == ["capture", "capture", "compare", "compare"]
    assert result(candidate)["status"] == "PASS"
    assert [x["part"] for x in result(candidate)["comparisons"]] == [1, 2]


@pytest.mark.parametrize("failure", ["nonzero_with_pass_json", "token_mismatch", "changed_reference", "same_process"])
def test_candidate_rejects_comparison_and_reference_failures(fixture, monkeypatch, failure):
    baseline = fixture("baseline")
    MockClient(monkeypatch)
    assert driver.run(baseline) == 0
    candidate = fixture("candidate", baseline.output)
    if failure == "changed_reference":
        p = baseline.output / "ordered-trace/part-2/capture.json"; x = driver.read(p); x["requests"].reverse(); save(p, x)
    elif failure == "same_process":
        p = baseline.output / "ordered-trace/status.json"; x = driver.read(p)
        x["run_token"] = driver.read(candidate.output / "launch.json")["run_token"]; save(p, x)
    client = MockClient(monkeypatch, fail_compare=failure == "nonzero_with_pass_json", mismatch=failure == "token_mismatch")
    assert driver.run(candidate) == 1
    assert result(candidate)["status"] == "FAIL"
    if failure in ("changed_reference", "same_process"):
        assert client.requests == 0
    else:
        assert client.requests == 20 and len(client.commands) == 3
        assert (candidate.output / "ordered-trace/part-2/capture.json").is_file()


def test_extra_post_during_capture_fails_ownership_check(fixture, monkeypatch):
    args = fixture("baseline")
    client = MockClient(monkeypatch)
    original = client.request
    def interleave(base, body):
        response = original(base, body)
        if client.requests == 1:
            with (args.output / "server.log").open("a") as stream:
                stream.write('INFO: "POST /v1/chat/completions HTTP/1.1" 200 OK\n')
        return response
    monkeypatch.setattr(experiment, "request", interleave)
    assert driver.run(args) == 1
    assert client.requests == 10 and len(client.commands) == 1
    assert (args.output / "ordered-trace/part-1/capture.json").is_file()


def test_run_client_rejects_nonzero_exit_and_saves_child_output(tmp_path):
    env = driver.client_environment({"deps": "/deps", "source": "/source"})
    command = [sys.executable, "-c", "print('raw child output'); raise SystemExit(9)"]
    with pytest.raises(ValueError, match="exit=9"):
        driver.run_client(command, env, tmp_path, tmp_path, "failure")
    assert driver.read(tmp_path / "failure-exit.json")["returncode"] == 9
    assert "raw child output" in (tmp_path / "failure.log").read_text()


def test_wait_ready_stops_on_server_exit_before_health_request(tmp_path, monkeypatch):
    save(tmp_path / "exit.json", {"returncode": 1})
    monkeypatch.setattr(driver.urllib.request, "urlopen", lambda *a, **k: pytest.fail("health called after exit"))
    with pytest.raises(ValueError, match="exit.json"):
        driver.wait_ready(tmp_path, "http://example.invalid", 0)


def test_recorded_sdk_cv2_library_prefix_is_the_only_environment_exception():
    # Exact library-path values observed in both original A/B launch/runtime
    # receipts. Keep this minimal fixture independent of archived run files.
    venv = "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0"
    launched = {"VIRTUAL_ENV": venv, "LD_LIBRARY_PATH": "/opt/aws/neuron/lib:" + venv + "/lib",
                "NEURON_EXECUTION_BACKEND": "lite"}
    observed = dict(launched, LD_LIBRARY_PATH=venv +
                    "/lib/python3.12/site-packages/cv2/../../lib64:" + launched["LD_LIBRARY_PATH"])
    driver.verify_environment(observed, launched)
    altered = dict(observed, LD_LIBRARY_PATH="/unrelated:" + observed["LD_LIBRARY_PATH"])
    with pytest.raises(ValueError, match="environment mismatch"):
        driver.verify_environment(altered, launched)
    altered = dict(observed, NEURON_EXECUTION_BACKEND="wrong")
    with pytest.raises(ValueError, match="environment mismatch"):
        driver.verify_environment(altered, launched)
