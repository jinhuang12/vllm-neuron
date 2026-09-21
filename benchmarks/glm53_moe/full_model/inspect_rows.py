#!/usr/bin/env python3
"""Read fused MoE row shapes from saved FX and protobuf HLO artifacts.

No runtime or device module is imported. A call is joined across FX and HLO
by its packed-weight input identity, never by its position in a call list.
Full-model mode requires the decode and prefill graphs, each with layers3..44.
This proves compiled operand shapes, not live execution or performance.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import types


SIGNATURE = ("hidden", "weights", "scales", "row_ids", "expert_ids", "affinity",
             "bounds", "BLOCK_M", "BLOCK_N", "BLOCK_K")
WRAPPER = re.compile(r"moe_fused_fp8_kernel_AwsNeuronNkiKernelWrapper\.\d+")
DTYPES = {"float32": "F32", "float64": "F64", "float16": "F16", "bfloat16": "BF16",
          "float8_e4m3fn": "F8E4M3FN", "float8_e5m2": "F8E5M2", "bool": "PRED",
          "int8": "S8", "int16": "S16", "int32": "S32", "int64": "S64",
          "uint8": "U8", "uint16": "U16", "uint32": "U32", "uint64": "U64"}


class UnknownMapping(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise UnknownMapping(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def schemas(schema_root=None):
    """Load generated protobuf files without executing the SDK package initializer."""
    require("libtorch_neuronx_lite" not in sys.modules,
            "Run this inspector in a fresh process without the Neuron runtime imported")
    if schema_root is None:
        spec = importlib.util.find_spec("libtorch_neuronx_lite")
        require(spec is not None and spec.submodule_search_locations,
                "Generated SDK protobuf schema is unavailable; use the SDK Python or --schema-root")
        schema_root = Path(next(iter(spec.submodule_search_locations)))
    schema_root = schema_root.resolve()
    files = [schema_root / "pyhlo/xla_data_pb2.py", schema_root / "pyhlo/service/hlo_pb2.py",
             schema_root / "compile/hlo.py"]
    require(all(p.is_file() for p in files), "Unknown SDK protobuf schema layout")
    stub = types.ModuleType("libtorch_neuronx_lite")
    stub.__path__ = [str(schema_root)]
    sys.modules[stub.__name__] = stub
    from libtorch_neuronx_lite.pyhlo import xla_data_pb2
    from libtorch_neuronx_lite.pyhlo.service import hlo_pb2
    return hlo_pb2, xla_data_pb2, [{"path": str(p), "sha256": digest(p)} for p in files]


def read_fx(path):
    nodes, placeholders, calls = {}, [], []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        match = re.match(r"\s*%([A-Za-z0-9_]+)\s*:", line)
        if not match:
            continue
        name = match.group(1)
        require(name not in nodes, f"Duplicate FX node {name}")
        nodes[name] = {"line": line_number, "text": line.strip(),
                       "parents": re.findall(r"%([A-Za-z0-9_]+)", line)[1:]}
        placeholder = re.search(r"placeholder\[target=([^\]]+)\]", line)
        if placeholder:
            placeholders.append({"node": name, "target": placeholder.group(1), "line": line_number})
        if "call_function[target=torch.ops.higher_order.nki_kernel_wrapper]" not in line:
            continue
        names = re.search(r"arg_names: \[([^]]+)\]", line)
        if names is None or tuple(x.strip() for x in names.group(1).split(",")) != SIGNATURE:
            continue
        arguments = re.search(r"args: \((.*?)\), arg_names:", line)
        require(arguments is not None, f"Unknown fused FX argument syntax at line {line_number}")
        args = [x.strip() for x in arguments.group(1).split(",")]
        require(len(args) == len(SIGNATURE), f"Unknown fused argument count at line {line_number}")
        require(all(re.fullmatch(r"%[A-Za-z0-9_]+", x) for x in args[:7]),
                f"Fused tensor operands are not seven FX nodes at line {line_number}")
        require(all(re.fullmatch(r"[0-9]+", x) for x in args[7:]),
                f"Unknown fused tile constants at line {line_number}")
        calls.append({"node": name, "line": line_number, "weight_node": args[1][1:],
                      "row_node": args[3][1:], "arg_names": list(SIGNATURE),
                      "tile_constants": dict(zip(SIGNATURE[7:], map(int, args[7:])))})
    by_node = {p["node"]: i for i, p in enumerate(placeholders)}
    for call in calls:
        require(call["weight_node"] in by_node,
                f"Packed weight operand {call['weight_node']} is not a direct FX input")
        require(call["weight_node"].endswith("_packed_weights_"),
                f"Unrecognized packed-weight input {call['weight_node']}")
        call["weight_input_index"] = by_node[call["weight_node"]]
        call["weight_input_name"] = placeholders[call["weight_input_index"]]["target"]
        layers = re.findall(r"layers_modules_(\d+)_modules_mlp_modules_experts", call["weight_node"])
        require(len(layers) <= 1, "Ambiguous FX layer identity")
        call["layer"] = int(layers[0]) if layers else None
        frontier, seen = [call["row_node"]], set()
        excerpt = []
        for _ in range(3):
            next_frontier = []
            for node in frontier:
                if node in seen:
                    continue
                require(node in nodes, f"Unknown FX row dependency {node}")
                seen.add(node)
                excerpt.append({"line": nodes[node]["line"], "text": nodes[node]["text"]})
                next_frontier.extend(nodes[node]["parents"])
            frontier = next_frontier
        call["row_definition_excerpt"] = sorted(excerpt, key=lambda row: row["line"])
    return placeholders, calls


def read_examples(path):
    text = path.read_text()
    inputs = {}
    pattern = r"Input (\d+):\n  Shape: ([^\n]+)\n  Dtype: ([^\n]+)\n  Device: ([^\n]+)"
    for match in re.finditer(pattern, text):
        number, dtype = int(match.group(1)), match.group(3)
        require(dtype in DTYPES, f"Unknown example input dtype {dtype}")
        shape = ast.literal_eval(match.group(2))
        require(isinstance(shape, tuple) and all(isinstance(x, int) and x >= 0 for x in shape),
                f"Unknown example input shape {shape!r}")
        require(number not in inputs, f"Duplicate example input {number}")
        inputs[number] = {"shape": list(shape), "dtype": DTYPES[dtype],
                          "line": text[:match.start()].count("\n") + 1}
    require(sorted(inputs) == list(range(len(inputs))), "Example inputs are not a complete ordered tensor list")
    return inputs


def shape_record(shape, data_schema):
    require(not shape.tuple_shapes and not any(shape.is_dynamic_dimension),
            "Tuple or dynamic operand shape is unsupported")
    return {"shape": list(shape.dimensions), "dtype": data_schema.PrimitiveType.Name(shape.element_type)}


def inspect_graph(directory, hlo_schema, data_schema, *, full_model=False, expected_count=None, expected_q=None):
    directory = directory.resolve()
    paths = {name: directory / name for name in
             ("fxgraph.txt", "example_inputs.txt", "graph.hlo", ".artifact_metadata_v0.json")}
    require(all(p.is_file() for p in paths.values()), f"Incomplete artifact inputs in {directory}")
    placeholders, fx_calls = read_fx(paths["fxgraph.txt"])
    examples = read_examples(paths["example_inputs.txt"])
    require(len(placeholders) == len(examples), "FX and example input counts differ")
    metadata = json.loads(paths[".artifact_metadata_v0.json"].read_text())
    require(metadata.get("version") == 0 and metadata.get("has_rng_seed_parameter") is False,
            "Unsupported metadata version or appended RNG parameter")
    unused = metadata.get("unused_input_indices")
    require(isinstance(unused, list) and len(set(unused)) == len(unused)
            and all(isinstance(i, int) and 0 <= i < len(placeholders) for i in unused),
            "Invalid unused-input index mapping")
    kept = [i for i in range(len(placeholders)) if i not in set(unused)]
    module = hlo_schema.HloModuleProto()
    module.ParseFromString(paths["graph.hlo"].read_bytes())
    computations = {c.id: c for c in module.computations}
    require(module.entry_computation_id in computations, "Missing HLO entry computation")
    entry = computations[module.entry_computation_id]
    instructions = {i.id: i for i in entry.instructions}
    parameters = {i.parameter_number: i for i in entry.instructions if i.opcode == "parameter"}
    require(sorted(parameters) == list(range(len(kept))), "HLO parameter order cannot be mapped to saved FX inputs")
    for number, original in enumerate(kept):
        require(shape_record(parameters[number].shape, data_schema) ==
                {key: examples[original][key] for key in ("shape", "dtype")},
                f"HLO parameter {number} does not match saved input {original}")

    def weight_input(node_id):
        visited = []
        while True:
            require(node_id in instructions and node_id not in visited, "Unknown or cyclic HLO weight dependency")
            visited.append(node_id)
            node = instructions[node_id]
            if node.opcode == "parameter":
                return kept[node.parameter_number], visited
            require(node.opcode in ("convert", "reshape", "bitcast", "transpose", "copy")
                    and len(node.operand_ids) == 1, f"Unknown HLO weight transform {node.opcode}")
            node_id = node.operand_ids[0]

    fx_by_weight = {}
    for call in fx_calls:
        require(call["weight_input_index"] not in fx_by_weight, "Several fused FX calls share one weight input")
        fx_by_weight[call["weight_input_index"]] = call
    wrappers = {c.id: c for c in module.computations if WRAPPER.fullmatch(c.name)}
    calls, seen, used_wrappers = [], set(), set()
    for caller in module.computations:
        for call in caller.instructions:
            matched = [i for i in call.called_computation_ids if i in wrappers]
            if not matched:
                continue
            require(caller.id == entry.id and call.opcode == "call" and len(matched) == 1
                    and len(call.called_computation_ids) == 1 and len(call.operand_ids) == 7,
                    "Unknown fused HLO call form or nested call context")
            wrapper = wrappers[matched[0]]
            used_wrappers.add(wrapper.id)
            params = {i.parameter_number: i for i in wrapper.instructions if i.opcode == "parameter"}
            require(sorted(params) == list(range(7)), "Fused wrapper does not have seven ordered tensor inputs")
            root = next((i for i in wrapper.instructions if i.id == wrapper.root_id), None)
            require(root is not None and root.opcode == "call" and len(root.called_computation_ids) == 1
                    and list(root.operand_ids) == [params[i].id for i in range(7)],
                    "Fused wrapper does not forward all seven arguments in order")
            kernel = computations[root.called_computation_ids[0]]
            require(re.fullmatch(r"HloMoeFusedFp8KernelNkiKernelCallImpl\.\d+", kernel.name),
                    "Unknown fused NKI implementation computation")
            kernel_params = {i.parameter_number: i for i in kernel.instructions if i.opcode == "parameter"}
            native = next((i for i in kernel.instructions if i.id == kernel.root_id), None)
            require(sorted(kernel_params) == list(range(7)) and native is not None
                    and native.opcode == "custom-call" and native.custom_call_target == "AwsNeuronCustomNativeKernel"
                    and list(native.operand_ids) == [kernel_params[i].id for i in range(7)],
                    "Fused native call does not consume all seven arguments in order")
            shapes = {}
            for number, argument in enumerate(SIGNATURE[:7]):
                require(call.operand_ids[number] in instructions, "Missing HLO call operand")
                actual = shape_record(instructions[call.operand_ids[number]].shape, data_schema)
                require(actual == shape_record(params[number].shape, data_schema),
                        f"Caller and wrapper disagree for {argument}")
                require(actual == shape_record(kernel_params[number].shape, data_schema),
                        f"Caller and native kernel disagree for {argument}")
                shapes[argument] = actual
            weight, trace = weight_input(call.operand_ids[1])
            require(weight in fx_by_weight and weight not in seen,
                    "HLO call does not join one-to-one to a fused FX weight input")
            seen.add(weight)
            fx = fx_by_weight[weight]
            rows, experts, hidden = (shapes[key]["shape"] for key in ("row_ids", "expert_ids", "hidden"))
            require(shapes["row_ids"]["dtype"] == "S32" and len(rows) == 2 and min(rows) > 0,
                    "row_ids must be positive rank-two int32")
            require(shapes["expert_ids"]["dtype"] == "S32" and experts == [rows[0], 1]
                    and len(hidden) == 2 and hidden[0] > 1, "Unknown fused routing geometry")
            require(shape_record(call.shape, data_schema) ==
                    {"shape": [rows[0], rows[1], hidden[1]], "dtype": "F32"},
                    "Fused output shape does not agree with row_ids")
            if expected_q is not None:
                require(rows[1] == expected_q, f"Expected q{expected_q}, got {rows[1]} at FX line {fx['line']}")
            calls.append({**fx, "row_ids": shapes["row_ids"], "operands": shapes,
                          "hlo_computation": wrapper.name, "hlo_call": call.name,
                          "hlo_row_parameter": params[3].name,
                          "hlo_native_computation": kernel.name, "hlo_native_call": native.name,
                          "hlo_native_row_parameter": kernel_params[3].name,
                          "hlo_row_operand": instructions[call.operand_ids[3]].name,
                          "hlo_weight_trace_ids": trace,
                          "fx_source": {"path": str(paths["fxgraph.txt"]), "line": fx["line"]},
                          "hlo_source": {"path": str(paths["graph.hlo"]), "instruction_id": call.id}})
    require(calls and len(calls) == len(fx_calls) and used_wrappers == set(wrappers)
            and len(seen) == len(fx_by_weight),
            "Fused FX calls and named HLO wrappers do not form a complete one-to-one mapping")
    if expected_count is not None:
        require(len(calls) == expected_count, f"Expected {expected_count} fused calls, got {len(calls)}")
    leg, model_input = "component", None
    if full_model:
        candidates = [i for i, p in enumerate(placeholders) if p["target"] == "L_input_ids_"]
        require(len(candidates) == 1, "Cannot identify the full-model input_ids input")
        idx = candidates[0]
        require(idx in kept and examples[idx]["dtype"] == "S32", "Unknown model input_ids dtype or unused input")
        model_input = {"input_index": idx, **examples[idx], "path": str(paths["example_inputs.txt"])}
        require(model_input["shape"] in ([1], [1024]), "Unknown full-model token bucket")
        leg = "decode" if model_input["shape"] == [1] else "prefill"
        required_q = 1 if leg == "decode" else 256
        require(len(calls) == 42 and sorted(c["layer"] for c in calls if c["layer"] is not None) == list(range(3, 45)),
                "Full-model fused calls do not cover each routed layer3..44 exactly once")
        require(all(c["row_ids"]["shape"][1] == required_q for c in calls),
                f"Not every {leg} call uses q{required_q}")
    sources = [{"path": str(p), "sha256": digest(p)} for p in paths.values()]
    neffs = sorted(directory.glob("*.neff"))
    sources.extend({"path": str(p), "sha256": digest(p)} for p in neffs)
    return {"status": "PASS", "artifact_dir": str(directory), "cache_key": metadata.get("cache_key"),
            "leg": leg, "model_input_ids": model_input, "fused_call_count": len(calls),
            "q_values": sorted({c["row_ids"]["shape"][1] for c in calls}), "calls": calls,
            "compiled_neff_present": bool(neffs), "sources": sources,
            "input_order_rule": "SDK final graph.hlo orders parameters by original FX input order after unused_input_indices are removed; compile/hlo.py is hashed with the schema sources."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cache", type=Path)
    group.add_argument("--graph-dir", type=Path, action="append")
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-q", type=int)
    parser.add_argument("--schema-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "FAIL", "graphs": [], "inspector_sha256": digest(Path(__file__).resolve()),
              "limits": "Saved compiler operand shapes only; not live execution, numerical correctness, or timing proof."}
    try:
        h, d, schema_files = schemas(args.schema_root)
        report["schema_sources"] = schema_files
        directories = args.graph_dir or [p.parent for p in sorted(
            (args.cache / "neuron/compile_cache").glob("*/graph.hlo"))]
        require(directories, "No saved graph artifacts found")
        for directory in directories:
            report["graphs"].append(inspect_graph(directory, h, d, full_model=args.full_model,
                                                  expected_count=args.expected_count, expected_q=args.expected_q))
        if args.full_model:
            require(sorted(g["leg"] for g in report["graphs"]) == ["decode", "prefill"],
                    "Full-model mode requires exactly one decode graph and one prefill graph")
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["runtime_modules_imported"] = [name for name in ("torch", "nrtpy", "libtorch_neuronx_lite._C")
                                           if name in sys.modules]
    if report["runtime_modules_imported"]:
        report["status"], report["error"] = "FAIL", "A runtime module was imported"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "error") if key in report}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
