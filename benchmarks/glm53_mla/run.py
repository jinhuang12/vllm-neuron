"""Measure sparse MLA with a CPU oracle and an optional external baseline.

Run with the Neuron SDK on an assigned idle logical core. Outputs include the
inputs, source copies, device outputs, compiler artifacts and launch traces.
These are isolated kernel measurements, not serving measurements.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import torch

from .reference import make_fixture, metrics, sparse_reference
from .runtime import compile_kernel, measure, tensor_sha256


def load_source(path):
    path = Path(path).resolve()
    name = "mla_" + hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def select_kernel(module, case):
    """Apply the public seam's shape dispatch to standalone input tensors."""
    seq, heads, latent = case.q_lift.shape
    rope = 0 if case.q_pe is None else case.q_pe.shape[-1]
    topk = case.indices.shape[-1]
    module._require_admissible(seq, heads, latent, rope, topk, case.cache.shape[0], case.scale)
    suffix = "_row_tiled" if topk > module.MOVING_MAX else (
        "_tiled" if latent % module.LATENT_TILE or latent > module.MOVING_MAX else "")
    name = f"mla_sparse_attention_{'rope' if rope else 'nope'}{suffix}_kernel"
    inputs = dict(q_lift_hbm=case.q_lift, c_kv_hbm=case.cache,
                  topk_hbm=case.indices, softmax_scale=case.scale)
    if rope:
        inputs.update(q_pe_hbm=case.q_pe, k_pe_hbm=case.k_pe)
    return name, getattr(module, name), inputs


def run_case(source, case, output, warmup, iterations, block_n=None, staged=False):
    output.mkdir(parents=True, exist_ok=False)
    source = source.resolve()
    source_bytes = source.read_bytes()
    module = load_source(source)
    name, kernel, inputs = select_kernel(module, case)
    if block_n is not None or staged:
        if "row_tiled" not in name:
            raise ValueError("Tile overrides require the row-tiled entry")
        inputs.update(BLOCK_N=512 if block_n is None else block_n, STREAM_KV=not staged)
    (output / "kernel_source.py").write_bytes(source_bytes)
    torch.save(vars(case), output / "inputs.pt")
    expected = sparse_reference(case)
    torch.save(expected, output / "reference.pt")
    report = {"kernel": name, "source_path": str(source),
              "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "input_sha256": {key: tensor_sha256(value) for key, value in inputs.items()
                               if isinstance(value, torch.Tensor)},
              "softmax_scale": case.scale, "block_n": block_n, "staged": staged}
    (output / "source.json").write_text(json.dumps(report, indent=2) + "\n")
    compiled, arrays = compile_kernel(kernel, inputs, output / "kernel")
    result, timing = measure(compiled, arrays, warmup, iterations)
    torch.save(result, output / "output.pt")
    empty = (case.indices == -1).all(dim=1)
    report.update(timing=timing, accuracy=metrics(result, expected),
                  empty_rows_zero=bool(torch.count_nonzero(result[empty]) == 0),
                  source_unchanged=source.read_bytes() == source_bytes)
    report["pass"] = (report["accuracy"]["allclose"] and report["empty_rows_zero"]
                      and report["source_unchanged"])
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return result, report


def compare_outputs(candidate, baseline, block_n):
    """Default grouping requires exact output; alternate grouping uses tolerance."""
    result = {"bitwise_equal": bool(torch.equal(candidate, baseline)),
              "accuracy": metrics(candidate, baseline),
              "bitwise_required": block_n in (None, 512)}
    result["pass"] = result["accuracy"]["allclose"] and (
        not result["bitwise_required"] or result["bitwise_equal"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2]
                        / "vllm_neuron/functional/attention/mla_sparse.py")
    parser.add_argument("--baseline-source", type=Path,
                        help="Sparse MLA module from an external baseline checkout")
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--cache-rows", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=2176)
    parser.add_argument("--rope", type=int, default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--kind", choices=("random", "duplicates", "sentinel", "zeros"), default="random")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--block-n", type=int, choices=(128, 256, 384, 512),
                        help="Omit to use the production entry's defaults")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if "NEURON_RT_VISIBLE_CORES" not in os.environ or os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        raise RuntimeError("Set the assigned NEURON_RT_VISIBLE_CORES and NEURON_LOGICAL_NC_CONFIG=2")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("Warmup must be nonnegative and iterations must be positive")
    torch.set_num_threads(1)
    case = make_fixture(args.seq, args.heads, args.latent, args.cache_rows, args.topk,
                        args.rope, getattr(torch, args.dtype), args.kind, scale=args.scale)
    baseline = None
    if args.baseline_source:
        args.output.mkdir(parents=True, exist_ok=False)
        baseline, base_report = run_case(args.baseline_source, case, args.output / "baseline",
                                         args.warmup, args.iterations)
        candidate_dir = args.output / "candidate"
    else:
        candidate_dir = args.output
    candidate, report = run_case(args.source, case, candidate_dir, args.warmup, args.iterations,
                                 args.block_n, args.staged)
    report.update(host=platform.node(), visible_cores=os.environ["NEURON_RT_VISIBLE_CORES"],
                  versions={name: importlib.metadata.version(name) for name in
                            ("torch", "nki", "neuronx-cc", "libtorch-neuronx-lite")},
                  arguments={key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()})
    if baseline is not None:
        report["canonical"] = compare_outputs(candidate, baseline, args.block_n)
        report["canonical"].update(source_sha256=base_report["source_sha256"],
                                    baseline_mean_us=base_report["timing"]["mean_us"],
                                    speedup=base_report["timing"]["mean_us"] / report["timing"]["mean_us"])
        report["pass"] = report["pass"] and base_report["pass"] and report["canonical"]["pass"]
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("pass", "accuracy", "canonical") if key in report}
                     | {"mean_us": report["timing"]["mean_us"]}), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
