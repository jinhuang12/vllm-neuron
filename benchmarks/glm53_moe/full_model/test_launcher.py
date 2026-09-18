from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from benchmarks.glm53_moe.full_model.launch import HERE, build_plan, server_arguments, sha256
from benchmarks.glm53_moe.full_model.collect import collect


def args(tmp_path, mode="serve"):
    return argparse.Namespace(mode=mode, arm="baseline", source=str(tmp_path / "source"),
        weights=str(tmp_path / "weights"), output=str(tmp_path / "output"),
        cache=str(tmp_path / "cache"), deps=str(tmp_path / "deps"),
        venv=str(tmp_path / "venv"), sdk=str(tmp_path / "sdk"),
        port=18004, no_profile=False, contract=None, contract_sha256=None)


def test_server_geometry_and_profiler_do_not_depend_on_arm(tmp_path):
    a = args(tmp_path)
    first = build_plan(a, "image-id", "run-one")
    a.arm = "candidate"
    second = build_plan(a, "image-id", "run-two")
    assert first["server_arguments"] == second["server_arguments"]
    assert first["environment"]["NEURON_EXECUTION_BACKEND"] == "lite"
    assert first["environment"]["NEURON_LOGICAL_NC_CONFIG"] == "2"
    command = first["server_arguments"]
    for key, value in (("tensor-parallel-size", "64"), ("num-gpu-blocks-override", "161"),
                       ("max-num-seqs", "1"), ("max-num-batched-tokens", "1024"),
                       ("max-model-len", "4096"), ("port", "18004")):
        assert command[command.index("--" + key) + 1] == value
    for key in ("--no-async-scheduling", "--enable-prefix-caching", "--enable-chunked-prefill"):
        assert key in command
    additional = json.loads(command[command.index("--additional-config") + 1])
    assert additional["neuron_config"]["ep_degree"] == 16
    assert additional["neuron_profiler"]["neuron_cores"] == [0]


def test_hardware_mounts_are_exact_and_smoke_exposes_none(tmp_path):
    serve = build_plan(args(tmp_path), "image-id", "run")
    command = serve["docker_command"]
    devices = [command[i + 1] for i, x in enumerate(command) if x == "--device"]
    assert devices == [f"/dev/neuron{i}" for i in range(16)]
    assert "--privileged" not in command
    assert "--read-only" in command
    assert command[command.index("--network") + 1] == "host"
    assert command[command.index("--ipc") + 1] == "host"
    smoke = build_plan(args(tmp_path, "smoke"), "image-id", "run")
    assert "--device" not in smoke["docker_command"]
    assert smoke["environment"]["VLLM_NEURON_CPU_MODE"] == "1"
    assert "VLLM_NEURON_CPU_MODE" not in serve["environment"]
    assert "NEURON_RT_VISIBLE_CORES" not in serve["environment"]


def test_source_dependencies_and_weights_are_read_only(tmp_path):
    a = args(tmp_path)
    command = build_plan(a, "image-id", "run")["docker_command"]
    for name in ("source", "weights", "deps", "venv", "sdk"):
        path = getattr(a, name)
        assert f"type=bind,src={path},dst={path},readonly" in command
    assert f"type=bind,src={a.cache}/tmp,dst=/tmp" in command


def test_writable_paths_cannot_shadow_source(tmp_path):
    a = args(tmp_path)
    a.output = a.source + "/output"
    with pytest.raises(ValueError, match="overlap"):
        build_plan(a, "image-id", "run")


def test_contract_hash_must_match(tmp_path):
    a = args(tmp_path)
    contract = tmp_path / "contract.json"
    contract.write_text('{"frozen":true}')
    a.contract = str(contract)
    a.contract_sha256 = "incorrect"
    with pytest.raises(ValueError, match="SHA256"):
        build_plan(a, "image-id", "run")
    a.contract_sha256 = sha256(contract)
    assert build_plan(a, "image-id", "run")["contract_sha256"] == sha256(contract)


def test_profiler_can_be_omitted_without_changing_geometry(tmp_path):
    config = json.loads((HERE / "config.json").read_text())
    command = server_arguments(config, tmp_path / "weights", tmp_path / "out", 18004, False)
    assert "--profiler-config" not in command
    additional = json.loads(command[command.index("--additional-config") + 1])
    assert set(additional) == {"neuron_config"}
    assert command[command.index("--num-gpu-blocks-override") + 1] == "161"


def test_cache_evidence_separates_membership_from_named_hits(tmp_path):
    root = tmp_path / "neuron" / "compile_cache" / ("a" * 32)
    root.mkdir(parents=True)
    (root / ("graph_" + "a" * 32 + ".neff")).write_bytes(b"example-neff")
    log = tmp_path / "server.log"
    log.write_text("Local cache hit for key: " + "b" * 32 + "\nnum_gpu_blocks is: 161\n")
    report = collect(tmp_path, log)
    assert report["entries"] == ["a" * 32]
    assert report["cache_hit_graph_keys"] == ["b" * 32]
    assert len(report["files"]) == 1
    assert len(report["runtime_rows"]) == 2
