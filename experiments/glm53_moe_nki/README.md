# Generic fused block-FP8 MoE kernel

The kernel derives H, local I, expert count, and routing capacity from tensor
shapes. It has compile-time `BLOCK_M`, `BLOCK_N`, and `BLOCK_K` parameters.
Changing a shape or tile setting creates a compiled specialization. Routing
indices and affinities remain device inputs within that specialization.

The normal `Glm5NextRoutedExperts.forward` uses the fused path. Its public
arguments, routing, FP32 contribution combine, and final BF16 cast stay the
same. Weight preparation builds the packed bank once. It supports meta shape
preparation, then validates real CPU weights before transfer to the device.
The prepared state stores two tensors; it does not keep a second unpacked
weight bank. Legacy direct calls to `block_quant_expert_mm` with the old four
operands retain the existing three-stage compatibility path.

For T tokens, the model passes only the first `min(T, block_size)` rows of
each routing block to the fused kernel and the FP32 combine. An expert can
receive at most T distinct tokens, and the mapping puts them before padding.
This keeps the existing routing map while removing unused rows from small
decode calls. The direct-call compatibility path keeps its full block width.

## Shape and tile parameters

| Parameter | Meaning | Constraint |
| --- | --- | --- |
| H | Hidden width, inferred from tensors | Positive multiple of 128 |
| I | Local expert intermediate width, inferred from packed shape | Positive multiple of 128 |
| E | Local expert count | Positive |
| q | Capacity per routing block | Positive; can exceed 512 |
| `BLOCK_M` | Token rows computed in one on-chip tile | 1..512; partial last tile supported |
| `BLOCK_N` | Output channels grouped in the schedule | Positive multiple of 128 |
| `BLOCK_K` | Contraction channels grouped in the schedule | Positive multiple of 128 |

The 128 values are the checkpoint's scale block and the inner hardware tile.
They are not a fixed model shape. A K group of 1024 still computes eight
separate 128x128 products, applies each scale, and adds in contraction order.
N/K groups can end with partial groups of these inner tiles.

The public wrapper selects defaults from metadata when tile settings are
omitted. This policy is a conservative heuristic, not an autotuner. Explicit
settings must also fit the compiler's SBUF allocation. The kernel does not
promise that every aligned combination fits on hardware.

The existing checkpoint loader still has its own 256-aligned sharding policy.
The standalone kernel and packing API support 128-aligned dimensions. That
extension does not change checkpoint eligibility.

## API

```python
from vllm_neuron.functional.moe.fused_fp8 import (
    PackedExperts, pack_experts, fused_fp8_experts,
)

# Once during load, on prepared CPU tensors (or meta for shape tracing):
packed = pack_experts(gate_up, down, gate_up_scales, down_scales)
packed = PackedExperts(packed.weights.to(device), packed.scales.to(device))

# Inside the native neuron_libtorch torch.compile graph:
contributions = fused_fp8_experts(
    padded_hidden, packed, row_ids, expert_ids, padded_affinity, bounds,
    block_m=128, block_n=1024, block_k=4096,
)
```

The NKI entry point exposes the same settings as uppercase `BLOCK_M/N/K`.
The wrapper uses the repository's `wrap_nki` interface. See
`verify_torch_api.py` for native capture with non-default tile settings.
Direct eager execution is not its supported route.

| Input | Contract |
| --- | --- |
| Gate/up before packing | Prepared FP8 E4M3FN `[E,H,2*I]` |
| Down before packing | Prepared FP8 E4M3FN `[E,I,H]` |
| Scales before packing | FP32 `[E,H/128,2,I/128]` and `[E,I/128,H/128]` |
| Packed weights | FP8 `[E,3*(I/128),128,H/128,128]` |
| Packed scales | FP32 `[E,3*(I/128),H/128]` |
| Hidden | BF16 `[T+1,H]`; final row zero |
| Row IDs | int32 `[blocks,q]`; token IDs in `[0,T)` or `-1` |
| Expert IDs | int32 `[blocks,1]`; IDs in `[0,E)` |
| Affinity | FP32 `[T+1,E]`; final row zero |
| Bounds | FP32 `[128,3]`; repeated gate upper, up lower, up upper |
| Output | FP32 `[blocks*q,H]`; original block and row order |

All kernel operands must be contiguous and on the same device. The caller
must meet index-value and zero-padding contracts. The wrapper validates
metadata without reading live routing values. Empty blocks produce zeros;
they still incur computation and weight loads.

Packing preserves every input byte and scale. It requires model-prepared
finite weights within [-240,240] and their matching compensated scales.
The existing Trn2 loader halves raw checkpoint weights and compensates/floors
scales. That earlier step can lose subnormal values. The packer does not
repeat it. Raw checkpoint values up to 448 are rejected by the real packer.

## Reproduce

From the source worktree on a Trainium2 host, select an unused assigned logical core:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_RT_VISIBLE_CORES=0
python experiments/glm53_moe_nki/run.py \
  --hidden 384 --intermediate 640 --q 7 --real 5 \
  --block-m 4 --block-n 256 --block-k 256 \
  --output /absolute/new-run-directory
NEURON_EXECUTION_BACKEND=lite python experiments/glm53_moe_nki/verify_torch_api.py \
  --run /absolute/new-run-directory --output /absolute/new-api-directory \
  --block-m 4 --block-n 256 --block-k 256
python experiments/glm53_moe_nki/capture_profile.py /absolute/new-run-directory
```

Each output directory must be new. CPU tests:

```bash
python3 -m pytest -q benchmarks/glm53_moe/test_reference.py \
  experiments/glm53_moe_nki/test_pack.py \
  experiments/glm53_moe_nki/test_generic_pack.py

# With the model's SDK dependencies installed:
VLLM_NEURON_CPU_MODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
python3 -m pytest -q experiments/glm53_moe_nki/test_model_integration.py \
  experiments/glm53_moe_nki/test_decode_rows.py
```

Run the compact-row native verifier on an assigned core with the SDK active:

```bash
NEURON_EXECUTION_BACKEND=lite python experiments/glm53_moe_nki/verify_decode_rows.py \
  --output /absolute/new-native-results --cases production small
```

It generates its own fixtures and checks changing routes, EP ranks, empty
local routes, and repeated calls. It reports legacy-output preservation and
CPU-reference agreement separately. Saved result JSON is not an input.

The reusable [Docker runner](../../verification/glm53-moe/run_container.sh)
provides `build`, `smoke`, `cpu-tests`, and `run` modes. Set `GLM53_SOURCE`,
`GLM53_OUTPUT`, `GLM53_VENV`, and `GLM53_SDK` for the local checkout and SDK.
Keep output outside the source tree. Hardware mode also requires the assigned
`GLM53_NEURON_DEVICE` and `NEURON_RT_VISIBLE_CORES`.

## Evidence

The [full-model report](../../verification/glm53-moe-fullmodel/RESULTS.md)
records measurements, limits, and an immutable link to the original evidence.
The [full-model tools](../../benchmarks/glm53_moe/full_model/README.md) support
new baseline/candidate comparisons. Generated captures, graphs, timings, and
logs belong in an external result directory; the tests generate their inputs.

Larger fixtures retain the original CPU-oracle BF16-boundary failures at the
unchanged tolerance. Exact preservation of existing device output is a
separate check. Component results alone do not establish full-model accuracy,
serving speedup, or global optimality.
