"""Verify the real routed-expert model forward through native torch.compile.

The model uses its production preparation and release hook. The verifier
compares its fused route with the unchanged three-kernel direct-call route
and an independent CPU oracle. Routing and EP rank change without recompiling.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import libtorch_neuronx_lite
from libtorch_neuronx_lite.compile.backend import compile as neuron_compile
from libtorch_neuronx_lite.nki.nki_hop import kernel_registry

from benchmarks.glm53_moe.reference import compact_reference, make_fixture, metrics
from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
    to_down_kernel_scale_operand,
    to_gate_up_kernel_scale_operand,
)
from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.glm5_next.config import Glm5NextConfig, Glm5NextTextConfig


def checkpoint_operands(case):
    return {
        "gate_proj_weight": case.gate_weight[:, :, 0].transpose(1, 2).contiguous(),
        "up_proj_weight": case.gate_weight[:, :, 1].transpose(1, 2).contiguous(),
        "down_proj_weight": case.down_weight.transpose(1, 2).contiguous(),
        "gate_proj_scale": case.gate_scales[:, :, 0].transpose(1, 2).contiguous(),
        "up_proj_scale": case.gate_scales[:, :, 1].transpose(1, 2).contiguous(),
        "down_proj_scale": case.down_scales.transpose(1, 2).contiguous(),
    }


def prepare_model(case, device):
    """Drive the production load hook over a real routed-expert module."""
    local_experts = case.gate_weight.shape[0]
    config = Glm5NextTextConfig(
        hidden_size=case.hidden.shape[1],
        moe_intermediate_size=case.down_weight.shape[1],
        n_routed_experts=2 * local_experts,
        num_experts_per_tok=2,
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


def independent_model_reference(case, affinity, rank):
    hidden = case.hidden[:-1]
    experts = case.gate_weight.shape[0]
    local = affinity[:, rank * experts:(rank + 1) * experts]
    out = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(experts):
        out += compact_reference(
            hidden, case.gate_weight[expert], case.gate_scales[expert],
            case.down_weight[expert], case.down_scales[expert],
            local[:, expert], 10.0, 10.0,
        )[-1]
    return out.to(hidden.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--block", type=int, default=256)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    # The compiler creates temporary folders in cwd. Keep them in the output
    # mount so the source tree can stay read-only inside Docker.
    os.chdir(args.output)
    if args.tokens < 2:
        parser.error("At least two tokens are required to exercise routing")
    torch.set_num_threads(1)
    case = make_fixture(q=args.tokens, hidden=args.hidden,
                        intermediate=args.intermediate, experts=2, block=args.block)
    device = torch.device("neuron:0")
    root, bank = prepare_model(case, device)
    quant = model_fp8.Glm5NextQuantConfig.from_model_config(Glm5NextConfig())

    # Separate legacy operands exist only in the verifier. The model retains
    # only its two packed banks after the production hook releases originals.
    legacy = {
        "gate_up_proj_weight": case.gate_weight.to(device),
        "down_proj_weight": case.down_weight.to(device),
        "gate_up_scale_operands": torch.stack([
            to_gate_up_kernel_scale_operand(grid, args.hidden, args.intermediate)
            for grid in case.gate_scales
        ]).to(device),
        "down_scale_operands": torch.stack([
            to_down_kernel_scale_operand(grid, args.intermediate, args.hidden)
            for grid in case.down_scales
        ]).to(device),
    }
    graphs = {"fused": [], "legacy": []}

    def backend_for(route):
        def backend(graph, sample_inputs, **kwargs):
            kernels = []
            for node in graph.graph.nodes:
                if node.op == "call_function" and "nki_kernel_wrapper" in str(node.target):
                    func = kernel_registry.get_func(node.kwargs["kernel_idx"])
                    kernels.append(func.__name__)
            graphs[route].append({"graph": str(graph.graph), "kernels": kernels})
            (args.output / "graphs.json").write_text(json.dumps(graphs, indent=2) + "\n")
            return neuron_compile(graph, sample_inputs, **kwargs)
        return backend

    def fused_forward(hidden, affinity, rank):
        return bank(hidden, affinity, quant, block_size=args.block, expert_parallel_rank=rank)

    def legacy_forward(hidden, affinity, rank):
        return bank.block_quant_expert_mm(
            hidden, affinity, quant_config=quant, block_size=args.block,
            expert_parallel_rank=rank, **legacy,
        )

    compiled = {}
    for name, function in (("fused", fused_forward), ("legacy", legacy_forward)):
        compiled[name] = torch.compile(
            function, backend=backend_for(name), fullgraph=True, dynamic=False,
            options={"compiler_workdir": str(args.output / f"compile-{name}")},
        )
    patterns = torch.tensor([
        [0.21, 0.37, 0.00, 0.00],
        [0.00, 0.19, 0.43, 0.00],
        [0.00, 0.00, 0.31, 0.61],
        [0.73, 0.00, 0.00, 0.17],
    ], dtype=torch.float32)
    affinity = patterns[torch.arange(args.tokens) % patterns.shape[0]].contiguous()
    hidden = case.hidden[:-1].to(device)
    variants = (
        ("original_rank0", affinity, 0),
        ("reordered_rank0", affinity.flip(0).contiguous(), 0),
        ("original_rank1", affinity, 1),
        ("inactive_rank1", torch.zeros_like(affinity), 1),
        ("repeat_rank0", affinity, 0),
    )
    results = []
    for name, scores, rank in variants:
        rank_tensor = torch.tensor([rank], dtype=torch.int64, device=device)
        scores_device = scores.to(device)
        actual = compiled["fused"](hidden, scores_device, rank_tensor).cpu()
        existing = compiled["legacy"](hidden, scores_device, rank_tensor).cpu()
        expected = independent_model_reference(case, scores, rank)
        record = {
            "case": name, "ep_rank": rank, "shape": list(actual.shape),
            "dtype": str(actual.dtype), "finite": bool(torch.isfinite(actual).all()),
            "bit_exact_to_existing_chain": bool(torch.equal(actual.view(torch.int16), existing.view(torch.int16))),
            "fused_cpu_oracle": metrics(actual, expected),
            "legacy_cpu_oracle": metrics(existing, expected),
            "compile_counts": {key: len(value) for key, value in graphs.items()},
        }
        results.append(record)
        torch.save({"fused": actual, "legacy": existing, "cpu": expected,
                    "affinity": scores, "rank": rank}, args.output / f"{name}.pt")
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    assert all(record["finite"] and record["bit_exact_to_existing_chain"] for record in results)
    assert all(len(value) == 1 for value in graphs.values()), "Routing values must not cause recompilation"
    assert graphs["fused"][0]["kernels"] == ["moe_fused_fp8_kernel"]
    assert len(graphs["legacy"][0]["kernels"]) == 3
    prepared = bank._prepared_kernel_operands
    report = {
        "status": "PASS_EXISTING_CHAIN", "scope": "Routed-expert model forward, including real preparation, release, device mapping and FP32 combine; no model server",
        "hidden": args.hidden, "intermediate": args.intermediate,
        "tokens": args.tokens, "block": args.block, "local_experts": 2, "global_experts": 4,
        "prepared_operand_count": len(prepared),
        "prepared_bytes": sum(t.numel() * t.element_size() for t in prepared.values()),
        "released_source_count": len(root.released_parameters()),
        "all_cpu_oracles_pass": all(record["fused_cpu_oracle"]["allclose"] for record in results),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
            "vllm_neuron/model/glm5_next/model_fp8.py", "vllm_neuron/functional/moe/moe_fused_fp8.py",
            "vllm_neuron/functional/moe/fused_fp8_pack.py", "vllm_neuron/functional/moe/fused_fp8.py",
            "vllm_neuron/functional/moe/fused_fp8_config.py", "experiments/glm53_moe_nki/verify_model_api.py",
        )},
        "results": results,
        "limits": "The checkpoint loader keeps its existing 256-wide shard rule. Wider prepared-input support does not change checkpoint sharding eligibility. CPU tolerance remains atol=1e-5, rtol=3e-2; any inherited CPU failures remain failures.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("results", "source_sha256")}), flush=True)


if __name__ == "__main__":
    main()
