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

## Short-prompt prefill

GLM-5.3-Flash can select sparse MLA query buckets independently of the KV
segment size. Set these explicit values in `additional_config.neuron_config`,
with `max_num_batched_tokens=1024`:

```json
{
  "num_batched_tokens_buckets": [128, 1024],
  "kv_segment_size_buckets": [1024]
}
```

For a single-request prefill in the 128-row bucket, the runner keeps other
model operators at 1024 rows and the latent cache carrier at 2048 rows.
It passes only 128 query and index rows to sparse MLA. It then restores the
output to 1024 rows with FP32 zeros before the existing cast and projection.
Real token counts still control cache writes and recurrent state. The row
counts come from the configured buckets. Decode and automatic bucket
selection stay unchanged. Other models retain their existing bucket checks.

The [baseline config](configs/prefill-baseline.json) and
[prefix config](configs/prefill-prefix.json) record the measured environment.
They differ only in query buckets. Pass either to the existing full-model
launcher with `--config`; use `--no-profile` for serving measurements. The
recorded baseline source is commit `4d218326`, which already contains the
streamed sparse MLA kernel. Use separate source, cache and output directories
for baseline, candidate and repeat baseline. Follow the full-model guide for
Docker smoke checks, the 30-prompt exact comparison, warmups and five timing
cohorts. Do not run device checks while timing a serving arm.

To check bucket validation, graph capture, real-token masking, owned cache
state and the next decode in Docker, use the environment setup above:

```bash
MLA_WORKDIR="$MLA_SOURCE" \
bash "$MLA_SOURCE/verification/glm53-mla/run_container.sh" check \
  env VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q -p no:cacheprovider \
  test/vllm_neuron/worker/test_independent_prefill_buckets.py \
  test/vllm_neuron/model/test_neuron_config_glm5next.py \
  test/vllm_neuron/worker/test_warmup_all_rank_execution.py \
  test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_capture_sites.py \
  test/vllm_neuron/model/glm5_next/test_mla_prefix_rows.py \
  test/vllm_neuron/model/glm5_next/tiny/test_independent_prefill_state.py
```

The measured serving scope is concurrency one, 5–89 prompt tokens and 32
output tokens, with TP64/EP16 and the pinned checkpoint/runtime. This option
does not add fresh-request prefix reuse or support for noncontiguous sparse
KV pages. Native boundary checks covered prompts up to 128 tokens with
output lengths chosen to stay within the supported range. A later baseline
boundary request hit the existing noncontiguous-page guard after timing had
closed; only its completed core timing window was used in the comparison.
These results establish preservation of the baseline, not HF accuracy or
GPU parity.
