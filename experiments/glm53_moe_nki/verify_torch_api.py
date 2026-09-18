"""Run the public API through the installed native Torch compile backend."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import libtorch_neuronx_lite
from libtorch_neuronx_lite.compile.backend import compile as neuron_compile
from vllm_neuron.functional.moe.fused_fp8 import PackedExperts, fused_fp8_experts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--block-m', type=int)
    parser.add_argument('--block-n', type=int)
    parser.add_argument('--block-k', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    inputs = torch.load(args.run / 'inputs.pt', map_location='cpu', weights_only=True)
    expected = torch.load(args.run / 'output.pt', map_location='cpu', weights_only=True)
    dev = {name: value.to('neuron:0') for name, value in inputs.items()}
    packed = PackedExperts(dev['weights'], dev['scales'])
    compile_calls = []

    def backend(graph, sample_inputs, **kwargs):
        compile_calls.append(str(graph.graph))
        return neuron_compile(graph, sample_inputs, **kwargs)

    operation = torch.compile(fused_fp8_experts, backend=backend, fullgraph=True, dynamic=False,
                              options={'compiler_workdir': str(args.output / 'compile')})
    results = []
    for case in ('original', 'reordered', 'inactive'):
        if case == 'original':
            rows, experts, want = inputs['row_ids'], inputs['expert_ids'], expected
        else:
            rows, experts, want = inputs['row_ids'].flip(0), inputs['expert_ids'].flip(0), expected.flip(0)
            if case == 'inactive':
                rows[0] = -1
                want[0] = 0
        got = operation(dev['hidden'], packed, rows.contiguous().to('neuron:0'),
                        experts.contiguous().to('neuron:0'),
                        dev['affinity'].reshape(dev['hidden'].shape[0], -1), dev['bounds'],
                        block_m=args.block_m, block_n=args.block_n, block_k=args.block_k).cpu()
        want = want.reshape_as(got)
        record = {'case': case, 'bitwise_equal': bool(torch.equal(got, want)),
                  'max_abs': float((got - want).abs().max()), 'shape': list(got.shape),
                  'dtype': str(got.dtype), 'compile_calls': len(compile_calls)}
        results.append(record)
        torch.save(got, args.output / f'{case}.pt')
        (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(record), flush=True)
    (args.output / 'graphs.txt').write_text('\n\n'.join(compile_calls))
    assert len(compile_calls) == 1, 'Routing values must not trigger recompilation'
    assert all(row['bitwise_equal'] for row in results)


if __name__ == '__main__':
    main()
