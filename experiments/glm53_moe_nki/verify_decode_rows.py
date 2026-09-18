"""Native proof for compact routed rows through the real model load hook.

Run only after the parent releases the device. Use the verification Docker
runner with NEURON_EXECUTION_BACKEND=lite. Default cases cover H4096/I512,
18 local experts, top8, T1, and a smaller H256/I256/T3 case. This verifier
reports baseline preservation and the fixed CPU oracle gate separately.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def routing_affinities(tokens, local_experts=18, top_k=8, mode="mixed"):
    """Exact binary scores with zero inactive experts and at most top_k routes."""
    if not 0 < top_k <= local_experts:
        raise ValueError("Require 0 < top_k <= local_experts")
    result = torch.zeros(tokens, 2 * local_experts, dtype=torch.float32)
    if mode == "empty":
        return result
    columns = torch.arange(top_k)[None, :]
    if mode == "mixed":
        selected = (5 * torch.arange(tokens)[:, None] + 5 * columns) % (2 * local_experts)
    elif mode == "concentrated":
        selected = columns.expand(tokens, -1)
    else:
        raise ValueError(f"Unknown routing mode: {mode}")
    weights = (columns.float() + 1) / 64
    return result.scatter(1, selected, weights.expand(tokens, -1)).contiguous()


def prepare_model(case, device, top_k):
    from experiments.glm53_moe_nki.verify_model_api import checkpoint_operands
    from vllm_neuron.model.glm5_next import model_fp8
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    experts = case.gate_weight.shape[0]
    config = Glm5NextTextConfig(
        hidden_size=case.hidden.shape[1],
        moe_intermediate_size=case.down_weight.shape[1],
        n_routed_experts=2 * experts, num_experts_per_tok=top_k,
        swiglu_limit=10.0,
    )
    bank = model_fp8.Glm5NextRoutedExperts(config, world_size=2, ep_degree=2)
    for name, tensor in checkpoint_operands(case).items():
        tensor = tensor.to(device)
        if name.endswith("_weight"):
            setattr(bank, name, torch.nn.Parameter(tensor, requires_grad=False))
        else:
            setattr(bank, name.replace("_scale", "_weight_scale_inv"), tensor)
    root = model_fp8.Glm5NextForConditionalGeneration.__new__(
        model_fp8.Glm5NextForConditionalGeneration
    )
    torch.nn.Module.__init__(root)
    root.bank = bank
    assert root._run_load_time_preps(device) == (0, 1)
    assert all(getattr(bank, name) is None for name in bank.RELEASED_AFTER_PREP)
    assert set(bank._prepared_kernel_operands) == {"packed_weights", "packed_scales"}
    return root, bank


def operand_metadata(value):
    if isinstance(value, torch.fx.Node):
        value = value.meta.get("example_value", value.meta.get("val"))
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {"constant": value if isinstance(value, (int, float, bool, str)) else None}


def verify_case(output, *, hidden_size, intermediate, tokens, experts=18, top_k=8, block=256):
    # Imports stay inside the native entry point. Importing routing fixtures
    # from CPU tests must not initialize a device runtime.
    import libtorch_neuronx_lite  # noqa: F401
    from libtorch_neuronx_lite.compile.backend import compile as neuron_compile
    from libtorch_neuronx_lite.nki.nki_hop import kernel_registry
    from benchmarks.glm53_moe.reference import compact_reference, make_fixture, metrics
    from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
        to_down_kernel_scale_operand, to_gate_up_kernel_scale_operand,
    )
    from vllm_neuron.model.glm5_next import model_fp8
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    case = make_fixture(q=tokens, hidden=hidden_size, intermediate=intermediate,
                        experts=experts, block=block)
    torch.save(case, output / "fixture.pt")
    device = torch.device("neuron:0")
    root, bank = prepare_model(case, device, top_k)
    quant = model_fp8.Glm5NextQuantConfig.from_model_config(Glm5NextConfig())
    legacy = {
        "gate_up_proj_weight": case.gate_weight.to(device),
        "down_proj_weight": case.down_weight.to(device),
        "gate_up_scale_operands": torch.stack([
            to_gate_up_kernel_scale_operand(grid, hidden_size, intermediate)
            for grid in case.gate_scales
        ]).to(device),
        "down_scale_operands": torch.stack([
            to_down_kernel_scale_operand(grid, intermediate, hidden_size)
            for grid in case.down_scales
        ]).to(device),
    }
    graphs = {"compact": [], "legacy": []}

    def backend_for(route):
        def backend(graph, sample_inputs, **kwargs):
            kernels = []
            for node in graph.graph.nodes:
                if node.op == "call_function" and "nki_kernel_wrapper" in str(node.target):
                    function = kernel_registry.get_func(node.kwargs["kernel_idx"])
                    kernels.append({
                        "name": function.__name__,
                        "operands": {name: operand_metadata(value) for name, value in
                                     zip(node.kwargs["arg_names"], node.kwargs["args"])},
                    })
            graphs[route].append({"graph": str(graph.graph), "kernels": kernels})
            (output / "graphs.json").write_text(json.dumps(graphs, indent=2) + "\n")
            return neuron_compile(graph, sample_inputs, **kwargs)
        return backend

    def compact_forward(hidden, affinity, rank):
        return bank(hidden, affinity, quant, block_size=block, expert_parallel_rank=rank)

    def legacy_forward(hidden, affinity, rank):
        return bank.block_quant_expert_mm(
            hidden, affinity, quant_config=quant, block_size=block,
            expert_parallel_rank=rank, **legacy,
        )

    compiled = {name: torch.compile(function, backend=backend_for(name), fullgraph=True,
                                    dynamic=False, options={"compiler_workdir": str(output / f"compile-{name}")})
                for name, function in (("compact", compact_forward), ("legacy", legacy_forward))}
    # Compute each expert once from checkpoint grids. Routing variants only
    # change the post-down affinity and FP32 expert sum in this independent oracle.
    expert_outputs = [compact_reference(
        case.hidden[:-1], case.gate_weight[e], case.gate_scales[e],
        case.down_weight[e], case.down_scales[e], torch.ones(tokens), 10.0, 10.0,
    )[-1] for e in range(experts)]

    def reference(scores, rank):
        local = scores[:, rank * experts:(rank + 1) * experts]
        result = torch.zeros(tokens, hidden_size, dtype=torch.float32)
        for e in range(experts):
            result += expert_outputs[e] * local[:, e, None]
        return result.to(torch.bfloat16)

    mixed = routing_affinities(tokens, experts, top_k)
    concentrated = routing_affinities(tokens, experts, top_k, "concentrated")
    variants = (
        ("mixed_rank0", mixed, 0, None),
        ("changed_affinity_rank0", mixed.roll(7, 1).contiguous(), 0, None),
        ("mixed_rank1", mixed, 1, None),
        ("empty_rank1", torch.zeros_like(mixed), 1, None),
        ("local_top8_rank0", concentrated, 0, None),
        ("repeat_mixed_rank0", mixed, 0, "mixed_rank0"),
        ("repeat_local_top8_rank0", concentrated, 0, "local_top8_rank0"),
        ("repeat_local_top8_rank0_again", concentrated, 0, "local_top8_rank0"),
    )
    results, previous = [], {}
    hidden = case.hidden[:-1].to(device)
    for name, scores, rank, repeat_of in variants:
        rank_tensor = torch.tensor([rank], dtype=torch.int64, device=device)
        scores_device = scores.to(device)
        actual = compiled["compact"](hidden, scores_device, rank_tensor).cpu()
        existing = compiled["legacy"](hidden, scores_device, rank_tensor).cpu()
        expected = reference(scores, rank)
        repeats = {route: bool(torch.equal(value.view(torch.int16), previous[repeat_of][route].view(torch.int16)))
                   for route, value in (("compact", actual), ("legacy", existing))} if repeat_of else None
        record = {
            "case": name, "ep_rank": rank, "shape": list(actual.shape),
            "compact_finite": bool(torch.isfinite(actual).all()),
            "legacy_finite": bool(torch.isfinite(existing).all()),
            "bit_exact_to_legacy": bool(torch.equal(actual.view(torch.int16), existing.view(torch.int16))),
            "compact_cpu_oracle": metrics(actual, expected),
            "legacy_cpu_oracle": metrics(existing, expected),
            "repeat_of": repeat_of, "repeat_bit_exact": repeats,
            "compile_counts": {route: len(items) for route, items in graphs.items()},
        }
        results.append(record)
        previous[name] = {"compact": actual, "legacy": existing}
        torch.save({"compact": actual, "legacy": existing, "cpu": expected,
                    "affinity": scores, "rank": rank}, output / f"{name}.pt")
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(record), flush=True)

    compact_kernels = graphs["compact"][0]["kernels"]
    row_shapes = [k["operands"].get("row_ids", {}).get("shape") for k in compact_kernels
                  if k["name"] == "moe_fused_fp8_kernel"]
    checks = {
        "bit_exact_to_legacy": all(r["bit_exact_to_legacy"] for r in results),
        "both_routes_finite": all(r["compact_finite"] and r["legacy_finite"] for r in results),
        "repeated_inputs_bit_exact": all(all(r["repeat_bit_exact"].values())
                                         for r in results if r["repeat_bit_exact"] is not None),
        "one_compile_per_route": all(len(items) == 1 for items in graphs.values()),
        "one_fused_dispatch": [k["name"] for k in compact_kernels] == ["moe_fused_fp8_kernel"],
        "three_legacy_dispatches": len(graphs["legacy"][0]["kernels"]) == 3,
        "compiled_compact_row_width": len(row_shapes) == 1 and row_shapes[0] is not None
                                      and row_shapes[0][1] == min(tokens, block),
    }
    baseline_pass = all(checks.values())
    cpu_pass = all(r[route + "_cpu_oracle"]["allclose"] for r in results
                   for route in ("compact", "legacy"))
    report = {
        "status": "PASS" if baseline_pass and cpu_pass else
                  "FAIL_CPU_ORACLE" if baseline_pass else "FAIL_BASELINE_PRESERVATION",
        "baseline_preservation": "PASS" if baseline_pass else "FAIL",
        "cpu_oracle_gate": "PASS" if cpu_pass else "FAIL",
        "scope": "Prepared routed-expert model forward with device mapping and FP32 combine; no model server",
        "hidden": hidden_size, "intermediate": intermediate, "tokens": tokens,
        "mapping_block": block, "compact_rows": min(tokens, block),
        "local_experts": experts, "global_experts": 2 * experts, "top_k": top_k,
        "prepared_operand_count": len(bank._prepared_kernel_operands),
        "released_source_count": len(root.released_parameters()),
        "checks": checks, "fused_row_shapes": row_shapes, "results": results,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "vllm_neuron/model/glm5_next/model_fp8.py",
            "vllm_neuron/functional/moe/moe_blockwise.py",
            "vllm_neuron/functional/moe/moe_blockwise_fp8.py",
            "vllm_neuron/functional/moe/moe_fused_fp8.py",
            "vllm_neuron/functional/moe/fused_fp8.py",
            "vllm_neuron/functional/moe/fused_fp8_pack.py",
            "vllm_neuron/functional/moe/fused_fp8_config.py",
            "experiments/glm53_moe_nki/verify_decode_rows.py",
        )},
        "limits": "Fixed CPU gate: atol=1e-5, rtol=3e-2. Baseline agreement does not turn an inherited CPU failure into a pass. No serving or performance claim.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=("production", "small"),
                        default=["production", "small"])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    dimensions = {"production": (4096, 512, 1), "small": (256, 256, 3)}
    reports = {}
    for name in args.cases:
        h, i, t = dimensions[name]
        reports[name] = verify_case(output / name, hidden_size=h, intermediate=i, tokens=t)
    summary = {"status": "PASS" if all(r["status"] == "PASS" for r in reports.values()) else "FAIL",
               "cases": {name: {k: report[k] for k in ("status", "baseline_preservation", "cpu_oracle_gate")}
                         for name, report in reports.items()}}
    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
