"""Check the model loader and fused kernel with checkpoint weights up to448."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from run import ROOT, inputs_and_reference, load_file
from benchmarks.glm53_moe.baseline import compile_kernel, measure
from benchmarks.glm53_moe.reference import compact_reference, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw-diagnostic', action='store_true', help='Bypass preparation to reproduce the unsupported raw-range failure')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    inputs, _ = inputs_and_reference(2, 2)
    inputs['weights'] = (inputs['weights'].float() * 512).clamp(-448, 448).to(torch.float8_e4m3fn)
    inputs['scales'] = inputs['scales'] / 512
    pack = load_file('range_pack', ROOT / 'vllm_neuron/functional/moe/fused_fp8_pack.py')
    gate, down, gs, ds = pack.unpack_experts(pack.PackedExperts(inputs['weights'], inputs['scales']))
    raw_min, raw_max = float(inputs['weights'].float().min()), float(inputs['weights'].float().max())
    rejected_raw = False
    try:
        pack.pack_experts(gate, down, gs, ds)
    except ValueError:
        rejected_raw = True
    assert rejected_raw, 'The public packer must reject unsupported raw checkpoint values'
    if not args.raw_diagnostic:
        from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
            downscale_fp8_weight_bytes, compensate_block_scales, needs_240_downscale,
        )
        assert needs_240_downscale(), 'This check requires the actual Trn2 loader branch'
        gate, down = downscale_fp8_weight_bytes(gate), downscale_fp8_weight_bytes(down)
        gs = torch.stack([compensate_block_scales(x.reshape(32, 8)).scale_inv.reshape(32, 2, 4) for x in gs])
        ds = torch.stack([compensate_block_scales(x).scale_inv for x in ds])
        prepared = pack.pack_experts(gate, down, gs, ds)
        inputs['weights'], inputs['scales'] = prepared.weights, prepared.scales
    expected = []
    for block in range(2):
        expert = int(inputs['expert_ids'][block, 0])
        rows = inputs['row_ids'][block].long()
        expected.append(compact_reference(inputs['hidden'][rows], gate[expert].reshape(4096, 2, 512),
                        gs[expert], down[expert], ds[expert],
                        inputs['affinity'].reshape(-1, 2)[rows, expert], 10.0, 10.0)[-1])
    expected = torch.stack(expected)
    kernel_path = ROOT / 'vllm_neuron/functional/moe/moe_fused_fp8.py'
    kernel = load_file('range_kernel', kernel_path).moe_fused_fp8_kernel
    compiled, arrays = compile_kernel(kernel, inputs, args.output / 'compile')
    output, timing = measure(compiled, arrays, warmup=10, iterations=100)
    floating = inputs['weights'].float()
    report = {'q': 2, 'experts': 2, 'source_sha256': hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
              'raw_min_fp8': raw_min, 'raw_max_fp8': raw_max,
              'raw_rejected_by_packer': rejected_raw, 'actual_model_loader_used': not args.raw_diagnostic,
              'out_of_contract_diagnostic': args.raw_diagnostic,
              'min_fp8': float(floating.min()), 'max_fp8': float(floating.max()),
              'abs_above_240_count': int((floating.abs() > 240).sum()),
              'accuracy': metrics(output, expected), 'timing': timing}
    assert raw_min == -448 and raw_max == 448
    if not args.raw_diagnostic:
        assert report['min_fp8'] == -224 and report['max_fp8'] == 224
    torch.save(inputs, args.output / 'inputs.pt')
    torch.save(output, args.output / 'output.pt')
    torch.save(expected, args.output / 'expected.pt')
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'timing'}))
    assert report['accuracy']['allclose']


if __name__ == '__main__':
    main()
