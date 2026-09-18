#!/usr/bin/env python3
"""Prepare or launch a GLM full-model comparison with a selected runtime config.

The benchmark lead owns device scheduling. Plan and smoke expose no devices.
Serve requires the exact SHA256 of the reviewed measurement contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import uuid

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def server_arguments(config: dict, weights: Path, output: Path, port: int,
                     profile: bool) -> list[str]:
    result = ["-m", "vllm.entrypoints.openai.api_server", "--model", str(weights),
              "--host", "127.0.0.1", "--port", str(port)]
    for key, value in config["server"].items():
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            result.append("--" + ("" if value else "no-") + flag)
        else:
            result.extend(["--" + flag, str(value)])
    additional = json.loads(json.dumps(config["additional_config"]))
    if profile:
        additional["neuron_profiler"] = {
            **config["neuron_profiler"], "output_dir": str(output / "profiles")}
        result.extend(["--profiler-config", json.dumps(config["profiler_config"])])
    result.extend(["--additional-config", json.dumps(additional)])
    return result


def build_plan(args: argparse.Namespace, image_id: str, token: str) -> dict:
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    source, weights, output, cache, deps, venv, sdk = (
        Path(getattr(args, name)).resolve()
        for name in ("source", "weights", "output", "cache", "deps", "venv", "sdk"))
    for writable in (output, cache):
        for readonly in (source, weights, deps, venv, sdk, HERE, config_path):
            if writable == readonly or readonly in writable.parents or writable in readonly.parents:
                raise ValueError(f"Writable and read-only mounts overlap: {writable}, {readonly}")
    env = dict(config["environment"])
    env.update({
        "PATH": f"{venv}/bin:{sdk}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "VIRTUAL_ENV": str(venv),
        "LD_LIBRARY_PATH": f"{sdk}/lib:{venv}/lib",
        "PYTHONPATH": f"{deps}:{source}",
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "HF_HOME": str(cache / "huggingface"),
        "VLLM_CACHE_ROOT": str(cache),
        "NEURON_LIBTORCH_CACHE_ROOT": str(cache),
        "TMPDIR": str(cache / "tmp"),
        "VLLM_NEURON_NEFF_LOAD_RUN": token,
    })
    if args.mode == "smoke":
        env["VLLM_NEURON_CPU_MODE"] = "1"
    name = f"glm53-full-{args.arm}-{token}"
    command = ["docker", "run", "--rm", "--init", "--read-only", "--name", name,
               "--network", "host", "--ipc", "host", "--ulimit", "memlock=-1:-1",
               "--user", f"{os.getuid()}:{os.getgid()}"]
    for path in (source, weights, deps, venv, sdk):
        command.extend(["--mount", f"type=bind,src={path},dst={path},readonly"])
    command.extend(["--mount", f"type=bind,src={HERE},dst=/harness,readonly"])
    command.extend(["--mount", f"type=bind,src={config_path},dst=/harness-config.json,readonly"])
    for path in (output, cache):
        command.extend(["--mount", f"type=bind,src={path},dst={path}"])
    # Keep temporary compiler files beside each arm's cache. Some NKI cache
    # metadata stores absolute temporary paths that must survive a restart.
    command.extend(["--mount", f"type=bind,src={cache / 'tmp'},dst=/tmp"])
    if args.mode in ("serve", "plan"):
        for number in range(16):
            command.extend(["--device", f"/dev/neuron{number}"])
    for key, value in sorted(env.items()):
        command.extend(["--env", f"{key}={value}"])
    command.extend(["--workdir", str(output), image_id, str(venv / "bin/python"),
                    "/harness/runtime.py", "--config", "/harness-config.json",
                    "--mode", "serve" if args.mode == "plan" else args.mode,
                    "--source", str(source), "--deps", str(deps),
                    "--output", str(output), "--cache", str(cache), "--"])
    server = server_arguments(config, weights, output, args.port, not args.no_profile)
    command.extend(server)
    contract = Path(args.contract).resolve() if args.contract else None
    contract_hash = sha256(contract) if contract else None
    if args.contract_sha256 and contract_hash != args.contract_sha256:
        raise ValueError("The measurement contract SHA256 does not match")
    return {"schema_version": 1, "arm": args.arm, "mode": args.mode,
            "image_id": image_id, "container_name": name, "run_token": token,
            "source": str(source), "weights": str(weights), "output": str(output),
            "cache": str(cache), "deps": str(deps),
            "config": config, "config_sha256": sha256(config_path),
            "contract": str(contract) if contract else None,
            "contract_sha256": contract_hash,
            "profile_routes_enabled": not args.no_profile,
            "environment": env, "server_arguments": server,
            "docker_command": command, "docker_command_shell": shlex.join(command),
            "harness_sha256": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "smoke", "serve"])
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--arm", required=True, choices=["baseline", "candidate", "postbaseline"])
    for name in ("source", "weights", "output", "cache", "deps"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--venv", default="/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0")
    parser.add_argument("--sdk", default="/opt/aws/neuron")
    parser.add_argument("--contract")
    parser.add_argument("--contract-sha256")
    parser.add_argument("--port", type=int, default=18004)
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args()
    if args.mode == "serve" and (not args.contract or not args.contract_sha256):
        parser.error("serve requires --contract and --contract-sha256 from the frozen plan")
    config = json.loads(args.config.read_text())
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", config["image"]], text=True).strip()
    if image_id != config["image_id"]:
        parser.error(f"Docker image changed: expected {config['image_id']}, found {image_id}")
    plan = build_plan(args, image_id, uuid.uuid4().hex)
    if args.mode == "plan":
        print(json.dumps(plan, indent=2))
        return
    for name in ("source", "weights", "deps", "venv", "sdk"):
        if not Path(getattr(args, name)).is_dir():
            parser.error(f"Missing directory: {getattr(args, name)}")
    if args.mode == "serve":
        for number in range(16):
            if not Path(f"/dev/neuron{number}").is_char_device():
                parser.error(f"Missing Neuron device {number}")
    output, cache = Path(plan["output"]), Path(plan["cache"])
    output.mkdir(parents=True, exist_ok=True)
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    with (output / "launch.json").open("x") as stream:
        json.dump(plan, stream, indent=2)
    if args.contract:
        (output / "measurement-contract.json").write_bytes(Path(args.contract).read_bytes())
    result = subprocess.run(plan["docker_command"])
    (output / "exit.json").write_text(json.dumps({"returncode": result.returncode}, indent=2))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
