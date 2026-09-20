# Ordered-combine CPU test provenance

These helpers make the regression tests runnable from the repository alone.
They retain exact source blocks from the reviewed DMA investigation. Neither
test suite needs the investigation worktree or an untracked diagnostics directory.
The standalone helper suite runs without a device or the Neuron SDK. The model
gate suite uses the normal CPU model test dependencies.

The source worktree was `glm53-dma-fix-20260919`, based on commit
`eda35acc24fb1a6a66e890f7c38dc401de8f6d83`. Both source files were under
`verification/next-optimization/diagnostics/`.

| Original file | SHA-256 |
| --- | --- |
| `baseline_components.py` | `adb7266c09ba2f1b83d3024a22d57d2a30669138b9e106c6cde0622496d8e7ec` |
| `compare_ordered_combine.py` | `07178caf83120f9238f324faa00a6ac6a8e49199e58ad95be3b8b6befe70bd35` |

The tables below identify each copied block. These are SHA-256 hashes of its
original UTF-8 source lines, including the final newline. Imports and module
headers were adapted for the packaged location. Function bodies and case
constants are unchanged. Native launch, profile, and report code is omitted.

| Packaged helper | Source block | Original lines | Block SHA-256 |
| --- | --- | --- | --- |
| `_ordered_combine_reference.py` | `MODEL` | 34–34 | `dd636a2c9effdbe6739f57b4eda77b9ef9b6878e23674445d9187f338ab89905` |
| `_ordered_combine_reference.py` | `MAPPING` | 35–35 | `7fab01bc7214e106853dd55decae86320a725f898ebfb96c381127be67fcce04` |
| `_ordered_combine_reference.py` | `digest` | 41–43 | `e993878d37d0cc8f77cb09a90ffeedff2b86da9887f66fb1976e1f8970dbfa25` |
| `_ordered_combine_reference.py` | `source_tree` | 50–51 | `a07cf18f0f8583d477640565bbb2caa9ac259c0872abb1717da9525810e8c3d1` |
| `_ordered_combine_reference.py` | `cpu_source_functions` | 63–72 | `5c7c21ea5f8a5d4fa4f34b6d2903633b57aaf646d6237eec0cb247a4c789a3df` |
| `_ordered_combine_reference.py` | `scatter_expression` | 75–119 | `850beb5811250af18f18a43c2cce4b829e0190aeb4e2b148cb91c7a931d32f3b` |
| `_ordered_combine_reference.py` | `mapping_fixture` | 122–175 | `028bb962b0bf5d246eb317947fa38f3860493a061faa31018e654802d36fac72` |
| `_ordered_combine_fixtures.py` | `CASES` | 23–30 | `64ec2d05805e00de5322ed722d6d8b416dc5a701c590ce18f8bcae36856da57c` |
| `_ordered_combine_fixtures.py` | `NATIVE_CASES` | 31–31 | `fbed03c5da868521968741e5e34a4d45b100944b6558efaac84759c046bc3bfc` |
| `_ordered_combine_fixtures.py` | `DENSE_CASES` | 33–33 | `1d9092abb26106f45969bcb6616925bc063a47d1e4127c0b76392ce6c7420986` |
| `_ordered_combine_fixtures.py` | `compare` | 36–53 | `f95b40568bba1a5942db550415cceac2aed391d66d983202ced4bc6458e2aa7c` |
| `_ordered_combine_fixtures.py` | `preservation_pass` | 56–60 | `24985fa0050ab12243c34179e13bd32a76a64a41deddb2ab02ea83b57da94955` |
| `_ordered_combine_fixtures.py` | `prove_map` | 63–73 | `b05c8e34faaa575ac4ab08ed6ab0d9022f8db7db754b3a90667bee0d39e80c69` |
| `_ordered_combine_fixtures.py` | `concentrated_fixture` | 76–124 | `8b969b949451389ffe0460e1e0c368212a53fc11aeb9f4036db9ed298c756edf` |
| `_ordered_combine_fixtures.py` | `fixtures` | 127–189 | `e746c5b1aa8b78f0f8aaad91f0faa9057653fe6fb016a0912b0fa3225be8a50d` |

`_ordered_combine_reference.py` extracts the real mapping functions from
`vllm_neuron/functional/moe/moe_blockwise.py` on each run. It also extracts the
legacy `index_add` expression from the model's fallback. It validates the final
cast and returns the FP32 result before that cast. It substitutes only the same
input device. The oracle does not call the candidate combine implementation.

`_ordered_combine_fixtures.py` preserves all shape cases, route variants, padding
controls, cancellation values, strict FP32 bit comparisons, and fixed CPU bounds.
The tests retain their original assertions and parametrization.

Run from the repository root:

```sh
VLLM_NEURON_CPU_MODE=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider experiments/glm53_moe_nki/test_ordered_combine.py
```

The model-gate tests also use the normal CPU model test dependencies. CPU results
verify contracts and test packaging. They do not establish native correctness,
full-model acceptance, or end-to-end speed.
