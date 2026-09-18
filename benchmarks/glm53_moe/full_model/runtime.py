#!/usr/bin/env python3
"""Record the resolved runtime before replacing this process with vLLM."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGES = ("torch", "vllm", "transformers", "tokenizers", "safetensors",
            "huggingface-hub", "numpy", "nki", "libtorch-neuronx-lite", "neuronx-cc",
            "pydantic", "msgspec", "triton", "sentencepiece")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def source_manifest(source: Path) -> dict:
    files = {str(p.relative_to(source)): file_hash(p)
             for p in sorted((source / "vllm_neuron").rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    for name in ("pyproject.toml", "setup.py", "setup.cfg"):
        path = source / name
        if path.is_file():
            files[name] = file_hash(path)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"root": str(source), "files": files, "tree_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "serve"], required=True)
    for name in ("source", "deps", "output", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    server_args = args.server_args
    if server_args[:1] == ["--"]:
        server_args = server_args[1:]
    source = args.source.resolve()
    deps = args.deps.resolve()
    modules = {}
    for name in ("vllm_neuron", "vllm", "torch", "transformers", "tokenizers",
                 "libtorch_neuronx_lite", "nki"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve() if module.__file__ else None
        modules[name] = {"path": str(path), "sha256": file_hash(path) if path else None}
        if name == "vllm_neuron" and (path is None or source not in path.parents):
            raise RuntimeError(f"Plugin import escaped the selected arm: {path}")
        if name in ("transformers", "tokenizers") and (path is None or deps not in path.parents):
            raise RuntimeError(f"Dependency overlay was not selected: {path}")
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    config = json.loads(args.config.read_text())
    for name, expected in config["runtime_versions"].items():
        if versions[name] != expected:
            raise RuntimeError(f"Unexpected {name} version: expected {expected}, found {versions[name]}")
    from vllm.config import ProfilerConfig
    from vllm_neuron.vllm.worker.neuron_profiler import NeuronProfilerConfig
    profiler = ProfilerConfig(**config["profiler_config"])
    neuron_profiler = NeuronProfilerConfig(config["neuron_profiler"])
    manifest = source_manifest(source)
    (args.output / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
    report = {
        "status": "PASS", "mode": args.mode, "python": sys.version,
        "executable": sys.executable, "versions": versions, "modules": modules,
        "config_sha256": file_hash(args.config),
        "source_tree_sha256": manifest["tree_sha256"],
        "devices_exposed": sorted(str(p) for p in Path("/dev").glob("neuron*")),
        "environment": {k: v for k, v in sorted(os.environ.items())
                        if k.startswith(("VLLM_", "NEURON_", "PYTHON", "HF_", "TRANSFORMERS_"))
                        or k in ("PATH", "LD_LIBRARY_PATH", "TMPDIR", "VIRTUAL_ENV", "XDG_CACHE_HOME")},
        "profiler_config_validated": repr(profiler),
        "neuron_profiler_config_validated": vars(neuron_profiler),
        "server_arguments": server_args,
    }
    # Parse the real installed CLI, with the full launch arguments, but never
    # build an engine in smoke mode. --help alone would not validate values.
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    cli = make_arg_parser(FlexibleArgumentParser())
    parsed = cli.parse_args(server_args[2:])
    validate_parsed_serve_args(parsed)
    report["server_cli_validated"] = True
    (args.output / "runtime.json").write_text(json.dumps(report, indent=2, default=str))
    print("GLM53_RUNTIME_PROVENANCE " + json.dumps(report, default=str), flush=True)
    subprocess.run([sys.executable, "/harness/collect.py", "--cache", str(args.cache),
                    "--output", str(args.output / "cache-before.json")], check=True)
    if args.mode == "smoke":
        if report["devices_exposed"]:
            raise RuntimeError("Smoke mode must not expose Neuron devices")
        return
    if len(report["devices_exposed"]) != 16:
        raise RuntimeError("Full-model serving requires all 16 Neuron devices")
    os.execv(sys.executable, [sys.executable, *server_args])


if __name__ == "__main__":
    main()
