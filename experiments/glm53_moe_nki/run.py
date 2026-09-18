"""Compile, validate, and measure one fused MoE candidate on Trainium2."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.glm53_moe.baseline import compile_kernel, measure
from benchmarks.glm53_moe.reference import compact_reference, make_fixture, metrics


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inputs_and_reference(q, experts=2, kind="random", real=None, hidden=4096, intermediate=512):
    real = q if real is None else real
    case = make_fixture(q=real, experts=experts, hidden=hidden, intermediate=intermediate, kind=kind, block=max(256, ((q + 127) // 128) * 128))
    pack = load_file("fused_pack", ROOT / "vllm_neuron/functional/moe/fused_fp8_pack.py")
    prepared = pack.pack_experts(case.gate_weight.reshape(experts, hidden, 2 * intermediate),
                                 case.down_weight, case.gate_scales, case.down_scales)
    rows = torch.full((experts, q), -1, dtype=torch.int32)
    expected = torch.zeros((experts, q, hidden), dtype=torch.float32)
    for e in range(experts):
        indices = torch.roll(torch.arange(real, dtype=torch.int32), e)
        positions = torch.arange(real) if kind != "routing" else torch.linspace(0, q - 1, real).long()
        rows[e, positions] = indices
        expected[e, positions] = compact_reference(
            case.hidden[indices.long()], case.gate_weight[e], case.gate_scales[e],
            case.down_weight[e], case.down_scales[e], case.affinity[indices.long(), e],
            case.gate_upper, case.up_upper,
        )[-1]
    return {
        "hidden": case.hidden, "weights": prepared.weights, "scales": prepared.scales,
        "row_ids": rows, "expert_ids": case.expert_index.reshape(-1, 1),
        "affinity": case.affinity.reshape(-1, 1),
        "bounds": torch.tensor([[case.gate_upper, -case.up_upper, case.up_upper]], dtype=torch.float32).expand(128, -1).contiguous(),
    }, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--real", type=int)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--block-m", type=int)
    parser.add_argument("--block-n", type=int)
    parser.add_argument("--block-k", type=int)
    parser.add_argument("--inactive", type=int, default=0)
    parser.add_argument("--kind", choices=("random", "clamp", "cancellation", "zeros", "routing"), default="random")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, default=ROOT / "vllm_neuron/functional/moe/moe_fused_fp8.py")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise RuntimeError("Set NEURON_RT_VISIBLE_CORES to the assigned core")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, expected = inputs_and_reference(args.q, args.experts, args.kind, args.real, args.hidden, args.intermediate)
    if not 0 <= args.inactive <= args.experts:
        raise ValueError("inactive must be between zero and experts")
    if args.inactive:
        inputs["row_ids"][-args.inactive:] = -1
        expected[-args.inactive:] = 0
    module = load_file("fused", args.kernel)
    config = load_file("fused_config", ROOT / "vllm_neuron/functional/moe/fused_fp8_config.py")
    import inspect
    if "BLOCK_M" in inspect.signature(module.moe_fused_fp8_kernel.func).parameters:
        tiles = config.select_tiles(args.hidden, args.intermediate, args.q, args.block_m, args.block_n, args.block_k)
        inputs.update(dict(zip(("BLOCK_M", "BLOCK_N", "BLOCK_K"), tiles)))
    else:
        tiles = None
    start = time.time()
    compiled, arrays = compile_kernel(module.moe_fused_fp8_kernel, inputs, args.output / "compile")
    output, timing = measure(compiled, arrays, warmup=10, iterations=args.iterations)
    result = {
        "hidden": args.hidden, "intermediate": args.intermediate, "tiles": tiles,
        "q": args.q, "real": args.real or args.q, "experts": args.experts, "inactive": args.inactive, "kind": args.kind,
        "source": str(args.kernel), "source_sha256": hashlib.sha256(args.kernel.read_bytes()).hexdigest(),
        "timing": timing, "accuracy": metrics(output, expected),
        "elapsed_seconds": time.time() - start,
    }
    torch.save({k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}, args.output / "inputs.pt")
    torch.save(expected, args.output / "expected.pt")
    torch.save(output, args.output / "output.pt")
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"q": args.q, "real": result["real"], "experts": args.experts,
                      "kind": args.kind, "mean_us": timing["mean_us"],
                      "accuracy": result["accuracy"]}), flush=True)
    return 0 if result["accuracy"]["allclose"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
