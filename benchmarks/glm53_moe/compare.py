"""Compare saved fused outputs with the existing three-kernel device chain.

This tool performs no compilation or hardware execution. It verifies shared
input bytes, checks packed tiles without calling the production packer, then
aligns output rows by expert and token ID. CPU-oracle failures remain separate.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .baseline import tensor_sha256
from .reference import make_fixture, metrics


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_packed_inputs(case, inputs):
    """Check each logical 128x128 tile against its independent checkpoint view."""
    experts = case.gate_weight.shape[0]
    weights, scales = inputs["weights"], inputs["scales"]
    if weights.dtype != torch.float8_e4m3fn or tuple(weights.shape) != (experts, 12, 128, 32, 128):
        raise ValueError("Unexpected fused FP8 weight contract")
    if scales.dtype != torch.float32 or tuple(scales.shape) != (experts, 12, 32):
        raise ValueError("Unexpected fused scale contract")
    checked = 0
    for expert in range(experts):
        for panel in range(12):
            for hidden_block in range(32):
                h = slice(hidden_block * 128, (hidden_block + 1) * 128)
                if panel < 8:
                    half, intermediate_block = divmod(panel, 4)
                    i = slice(intermediate_block * 128, (intermediate_block + 1) * 128)
                    expected = case.gate_weight[expert, h, half, i]
                    scale = case.gate_scales[expert, hidden_block, half, intermediate_block]
                else:
                    intermediate_block = panel - 8
                    i = slice(intermediate_block * 128, (intermediate_block + 1) * 128)
                    expected = case.down_weight[expert, i, h]
                    scale = case.down_scales[expert, intermediate_block, hidden_block]
                actual = weights[expert, panel, :, hidden_block, :]
                if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
                    raise ValueError(f"Weight bytes differ: expert={expert}, panel={panel}, hidden_block={hidden_block}")
                if scales[expert, panel, hidden_block] != scale:
                    raise ValueError(f"Scale differs: expert={expert}, panel={panel}, hidden_block={hidden_block}")
                checked += 1
    for name, expected in (("hidden", case.hidden), ("affinity", case.affinity.reshape(-1, 1))):
        actual = inputs[name]
        if actual.dtype != expected.dtype or actual.shape != expected.shape or tensor_sha256(actual) != tensor_sha256(expected):
            raise ValueError(f"Fused {name} differs from baseline input bytes")
    bounds = torch.tensor([case.gate_upper, -case.up_upper, case.up_upper], dtype=torch.float32).expand(128, 3)
    if not torch.equal(inputs["bounds"], bounds):
        raise ValueError("Clamp bounds differ")
    if inputs["bounds"].dtype != torch.float32:
        raise ValueError("Clamp bounds must use FP32")
    return checked


def compare_runs(baseline_dir, fused_dir):
    baseline_dir, fused_dir = Path(baseline_dir).resolve(), Path(fused_dir).resolve()
    baseline_manifest = baseline_dir / "metrics.json"
    corrected_manifest = baseline_dir / "metrics-launch-max.json"
    if corrected_manifest.exists():
        baseline_manifest = corrected_manifest
    base_report = json.loads(baseline_manifest.read_text())
    fused_manifest = fused_dir / "metrics.json"
    fused_report = json.loads(fused_manifest.read_text())
    inputs_file = fused_dir / "inputs.pt"
    inputs = torch.load(inputs_file, map_location="cpu", weights_only=True)
    real = fused_report["real"]
    base_case = base_report["cases"][str(real)]
    block = base_report["block"]
    experts = fused_report.get("experts", int(inputs["weights"].shape[0]))
    if base_report["kind"] != fused_report["kind"]:
        raise ValueError("Fixture kinds differ")
    case = make_fixture(q=real, experts=experts, block=block, kind=base_report["kind"])
    actual_hashes = {name: tensor_sha256(value) for name, value in vars(case).items()
                     if isinstance(value, torch.Tensor)}
    if base_case.get("input_sha256") != actual_hashes:
        raise ValueError("Baseline fixture bytes do not match its recorded hashes")
    checked_tiles = verify_packed_inputs(case, inputs)
    baseline_output_file = baseline_dir / f"q{real}" / "padded_contributions.pt"
    baseline = torch.load(baseline_output_file, map_location="cpu", weights_only=True)
    baseline = baseline.reshape(experts, block, 4096)
    fused_output_file = fused_dir / "output.pt"
    fused = torch.load(fused_output_file, map_location="cpu", weights_only=True)
    if baseline.dtype != torch.float32 or fused.dtype != torch.float32:
        raise ValueError("Output contract requires FP32")
    if tuple(fused.shape) != tuple(inputs["row_ids"].shape) + (4096,):
        raise ValueError("Fused output shape does not match its routing")
    by_expert = {int(expert): slot for slot, expert in enumerate(case.expert_index)}
    rows = case.row_index.reshape(experts, block)
    selected = torch.empty_like(fused)
    for slot, expert in enumerate(inputs["expert_ids"].flatten().tolist()):
        source_slot = by_expert[expert]
        positions = {token: position for position, token in enumerate(rows[source_slot].tolist())}
        wanted = inputs["row_ids"][slot].tolist()
        # A padding output must also have been executed by the baseline. Never
        # invent a zero when a saved run contains no corresponding padding row.
        missing = set(wanted) - positions.keys()
        if missing:
            raise ValueError(f"Baseline has no executed rows for token IDs {missing}")
        selected[slot] = baseline[source_slot, [positions[token] for token in wanted]]
    comparison = metrics(fused, selected)
    comparison["bit_exact"] = bool(torch.equal(fused.view(torch.int32), selected.view(torch.int32)))
    comparison["different_elements"] = int((fused != selected).sum())
    baseline_source_file = baseline_dir / f"q{real}" / "source.json"
    baseline_source = json.loads(baseline_source_file.read_text())
    return {
        "schema": "glm53-moe-existing-vs-fused-v1",
        "q": fused_report["q"], "real": real, "experts": experts,
        "baseline_block": block, "kind": fused_report["kind"],
        "inactive_blocks": fused_report.get("inactive", 0),
        "input_identity": {"baseline_hashes_verified": True, "packed_weight_and_scale_tiles_verified": checked_tiles,
                           "hidden_affinity_and_bounds_verified": True, "routing_aligned_by_expert_and_token": True},
        "comparison": comparison,
        "baseline_cpu_oracle": base_case["output_real_rows"],
        "fused_cpu_oracle": fused_report["accuracy"],
        "baseline_device_sum_us": base_case["device_sum_us"],
        "fused_device_us": fused_report["timing"]["mean_us"],
        "ratio_of_separate_device_measurements": base_case["device_sum_us"] / fused_report["timing"]["mean_us"],
        "timing_scope": "separate runs; baseline sum of three kernels; excludes launch and host-copy costs",
        "fused_source_sha256": fused_report["source_sha256"],
        "baseline_source_sha256": baseline_source["sha256"],
        "evidence_sha256": {str(path): sha256_file(path) for path in
                            (baseline_manifest, baseline_source_file, fused_manifest, inputs_file, baseline_output_file, fused_output_file)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = compare_runs(args.baseline, args.fused)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["comparison"]["allclose"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
