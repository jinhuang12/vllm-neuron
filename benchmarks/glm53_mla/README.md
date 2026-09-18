# Sparse MLA checks

The row-tiled kernel loads selected KV rows from HBM. Its public Python API
and geometry checks stay unchanged. The NKI entries accept `BLOCK_N` values
128, 256, 384 and 512, plus `STREAM_KV=False` for full-cache staging.
The public API uses streaming with width 512. This preserves the original
softmax grouping. Other widths use a different grouping and can round differently.

These tools generate inputs and compare each kernel with an independent CPU
oracle. An optional baseline module comes from a separate checkout; the tools
do not contain a second copy of the old kernel. The default grouping requires
bitwise equality with that baseline. Other widths require the existing
`rtol=1e-2, atol=1e-5` tolerance. Baseline and candidate must both pass the CPU
comparison. Duplicate indices count as separate softmax terms. Sentinel `-1`
contributes no term, and a query with only sentinels returns exact zeros.

## Portable checks

```bash
python3 -m pytest -q benchmarks/glm53_mla
```

## Native Docker checks

Use an idle, assigned Trainium2 logical core with the Neuron SDK installed.
The [runner](../../verification/glm53-mla/run_container.sh) reuses the SDK
read-only. Its default OS image is the one built by the existing
[MoE Docker runner](../../verification/glm53-moe/run_container.sh). Override
`MLA_IMAGE`, `MLA_VENV` and `MLA_SDK` for another installed environment.
Set `MLA_DEPS` if the source needs a separate dependency overlay.

Use absolute paths and a new output directory. Keep all outputs outside the
source tree. For example, with the baseline module in `/absolute/base`:

```bash
export MLA_SOURCE=/absolute/candidate-source
export MLA_INPUT=/absolute/base
export MLA_OUTPUT=/absolute/results/mla
export MLA_WORKDIR="$MLA_OUTPUT"
export MLA_NEURON_DEVICE=/dev/neuron0
export NEURON_RT_VISIBLE_CORES=0

bash "$MLA_SOURCE/verification/glm53-mla/run_container.sh" smoke
bash "$MLA_SOURCE/verification/glm53-mla/run_container.sh" cpu-tests
bash "$MLA_SOURCE/verification/glm53-mla/run_container.sh" run \
  python -m benchmarks.glm53_mla.suite \
  --baseline-source "$MLA_INPUT/mla_sparse.py" --output "$MLA_OUTPUT/suite"

bash "$MLA_SOURCE/verification/glm53-mla/run_container.sh" run \
  env NEURON_EXECUTION_BACKEND=lite NEURON_LIBTORCH_DISABLE_COMPILE_CACHE=1 \
  python "$MLA_SOURCE/verification/glm53-mla/verify_torch_api.py" \
  --run "$MLA_OUTPUT/suite/bf16-sentinel/candidate" \
  --output "$MLA_OUTPUT/public-api"
```

`smoke`, `cpu-tests` and `check` expose no Neuron devices. `check` accepts a
command, such as the existing SDK regression suite with `NKI_SIMULATOR=1`.
`run` exposes only the assigned device. Do not run these jobs during a serving
benchmark on the same host.

The native matrix covers decode, prefill, 4K/16K caches, BF16/FP16/FP32,
duplicates, all-sentinel and one-valid queries, an odd cache length, RoPE,
score-tile tails, and the tile controls. To run one shape, use
`python -m benchmarks.glm53_mla.run --help` with the same runtime.

The public-API verifier uses `torch.compile` with the Lite backend. It checks
dynamic selection values without recompilation, eager invalid-input guards,
and bitwise agreement with the standalone default kernel. It also updates the
first and last cache rows with `index_copy` inside the compiled function, then
reads the last row through sparse attention. Two distinct updates must match
the CPU result and differ from a stale-cache control. The sentinel fixture
adds a padded query that must remain zero after either update.

Each run saves source and input hashes, tensors, compiler artifacts and raw
launch traces. Kernel timing is the maximum physical-core duration per launch,
averaged after warmup. This is an isolated kernel measurement. Use the existing
[full-model tools](../glm53_moe/full_model/README.md) for serving comparisons.
