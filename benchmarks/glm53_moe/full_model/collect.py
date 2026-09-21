#!/usr/bin/env python3
"""Save cache identities and named graph evidence from a server log.

Cache membership alone does not prove that an entry ran. Cache-hit log rows
identify entries selected by the compiler. A live profile binds executed
graphs to NEFFs; keep that proof separate from this cache inventory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


MOE_SIGNATURES = {
    ("hidden", "weight_bank", "scale_bank", "row_index", "expert_index", "iota"):
        "moe_gate_up_blockwise_fp8_kernel",
    ("gate_up", "bounds"): "moe_swiglu_transposed_kernel",
    ("intermediate_t", "weight_bank", "scale_bank", "affinity_bank", "row_index", "expert_index", "iota"):
        "moe_down_blockwise_fp8_kernel",
    ("hidden", "weights", "scales", "row_ids", "expert_ids", "affinity", "bounds",
     "BLOCK_M", "BLOCK_N", "BLOCK_K"): "moe_fused_fp8_kernel",
}


def graph_dispatch(path: Path) -> dict:
    """Count exact NKI argument signatures and trace weight operands to layers.

    Registry integers vary by process. The emitted argument names identify
    the function interface; the weight dependency identifies the model layer.
    This does not execute or rewrite the graph.
    """
    dependencies, direct_layers, known_layers, calls = {}, {}, {}, []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        name_match = re.match(r"\s*%([A-Za-z0-9_]+)\s*:", line)
        if not name_match:
            continue
        name = name_match.group(1)
        dependencies[name] = re.findall(r"%([A-Za-z0-9_]+)", line)[1:]
        if "placeholder[target=" in line:
            direct_layers[name] = sorted({int(n) for n in re.findall(
                r"layers_modules_(\d+)_modules_mlp_modules_experts", line)})

        def layer_roots(node: str, seen: set[str]) -> set[int]:
            if node in seen:
                return set()
            seen.add(node)
            if node in known_layers:
                return set(known_layers[node])
            if node in direct_layers:
                return set(direct_layers[node])
            result = set()
            for parent in dependencies.get(node, []):
                result.update(layer_roots(parent, seen))
            return result

        if "call_function[target=torch.ops.higher_order.nki_kernel_wrapper]" not in line:
            continue
        signature = re.search(r"arg_names: \[([^]]+)\]", line)
        if not signature:
            continue
        arguments = tuple(item.strip() for item in signature.group(1).split(","))
        kernel = MOE_SIGNATURES.get(arguments)
        if kernel is None:
            continue
        # Gate, down, and fused calls take the weight bank as argument1.
        # SwiGLU takes the preceding gate result as argument0.
        argument_nodes = dependencies[name]
        position = 0 if kernel == "moe_swiglu_transposed_kernel" else 1
        layers = sorted(layer_roots(argument_nodes[position], set())) if len(argument_nodes) > position else []
        known_layers[name] = layers
        index = re.search(r"kernel_idx: (\d+)", line)
        calls.append({"line": number, "node": name, "kernel": kernel,
                      "kernel_idx": int(index.group(1)) if index else None,
                      "layers": layers, "arg_names": list(arguments)})
    counts = Counter(call["kernel"] for call in calls)
    expected_layers = list(range(3, 45))
    per_kernel = {kernel: sorted(layer for call in calls if call["kernel"] == kernel
                                for layer in call["layers"]) for kernel in MOE_SIGNATURES.values()}
    legacy = list(MOE_SIGNATURES.values())[:3]
    legacy_complete = all(counts[k] == 42 and per_kernel[k] == expected_layers for k in legacy)
    fused_complete = counts["moe_fused_fp8_kernel"] == 42 and per_kernel["moe_fused_fp8_kernel"] == expected_layers
    return {"file": str(path), "sha256": digest(path), "counts": dict(counts),
            "layers_by_kernel": per_kernel, "calls": calls,
            "baseline_dispatch_complete": legacy_complete and counts["moe_fused_fp8_kernel"] == 0,
            "candidate_dispatch_complete": fused_complete and all(counts[k] == 0 for k in legacy),
            "rule": "Exactly one identified call per kernel per routed layer3..44; no opposite-route calls",
            "limits": "Compiled FX dispatch is not live execution or timing proof."}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def collect(cache: Path, log: Path | None) -> dict:
    root = cache / "neuron" / "compile_cache"
    files = []
    if root.exists():
        for p in sorted(root.rglob("*")):
            if p.is_file() and (p.suffix == ".neff" or p.name in (
                    "fxgraph.txt", "example_inputs.txt", "graph.hlo", "command.txt",
                    ".artifact_metadata_v0.json", ".compilation_complete")):
                files.append({"path": str(p.relative_to(cache)), "size": p.stat().st_size,
                              "sha256": digest(p)})
    rows, hits, compiled, neff_names = [], set(), set(), set()
    if log is not None:
        with log.open(errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if any(key in line for key in ("Local cache hit for key:", "Compilation cache key:", "neff_load_",
                        "num_gpu_blocks", "KV cache need:", "GPU KV cache size:",
                        "Initializing a V1 LLM engine", "Starting vLLM server")):
                    rows.append({"line": line_number, "text": line.rstrip()})
                hits.update(re.findall(r"Local cache hit for key:\s*([0-9a-f]{32,64})", line))
                compiled.update(re.findall(r"Compilation cache key:\s*([0-9a-f]{32,64})", line))
                neff_names.update(re.findall(r"graph_([0-9a-f]{32,64})\.neff", line))
    dispatch = []
    if root.exists():
        for graph in sorted(root.glob("*/fxgraph.txt")):
            item = graph_dispatch(graph)
            item["cache_key"] = graph.parent.name
            item["named_in_compile_log"] = graph.parent.name in compiled
            item["named_in_cache_hit_log"] = graph.parent.name in hits
            item["neffs"] = [f for f in files if f["path"].startswith(
                str(graph.parent.relative_to(cache)) + "/") and f["path"].endswith(".neff")]
            hlo = graph.with_name("graph.hlo")
            # The SDK writes protobuf HLO, not text. These serialized computation
            # names provide a second identity check and profile search terms.
            symbols = []
            if hlo.is_file():
                symbols = [m.decode("ascii") for m in re.findall(
                    rb"moe_(?:gate_up_blockwise_fp8|swiglu_transposed|down_blockwise_fp8|fused_fp8)_kernel_AwsNeuronNkiKernelWrapper\.\d+",
                    hlo.read_bytes())]
            item["hlo_moe_computation_symbols"] = symbols
            item["hlo_moe_symbol_counts"] = dict(Counter(
                name.split("_AwsNeuronNkiKernelWrapper.")[0] for name in symbols))
            dispatch.append(item)
    return {"schema_version": 1, "at_utc": datetime.now(timezone.utc).isoformat(),
            "cache_root": str(cache), "compile_root": str(root),
            "entries": sorted(p.name for p in root.iterdir()) if root.exists() else [],
            "files": files, "server_log": str(log) if log else None,
            "server_log_sha256": digest(log) if log else None,
            "cache_hit_graph_keys": sorted(hits), "compilation_graph_keys": sorted(compiled),
            "log_named_neff_keys": sorted(neff_names), "graph_dispatch": dispatch,
            "runtime_rows": rows,
            "limits": "An inventory is not execution proof. Use named compiler rows and live profiles."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(collect(args.cache, args.log), indent=2))


if __name__ == "__main__":
    main()
