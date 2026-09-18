# Full-model comparison reproduction

The launcher runs either source tree in one pinned Docker image. It mounts the
source, SDK, venv, dependency overlay, and checkpoint read-only. Each arm gets a
separate writable cache and output directory. The post-baseline control reuses
the baseline cache. Temporary compiler files remain beside that cache because
some NKI cache records contain absolute temporary paths.

The benchmark lead owns the measurement contract, workload, correctness checks,
and device schedule. Do not infer a full-model result from the launcher smoke
test. No server or device work was part of that test.

## Frozen runtime settings

`benchmarks/glm53_moe/full_model/config.json` fixes TP64, EP16, sequence limit 1,
maximum length 4096, batched-token limit 1024, BF16 activations, FP8 weights,
128-token KV blocks, and 161 KV blocks. Prefix caching and chunked prefill are
enabled. Async scheduling is disabled. The API binds to 127.0.0.1:18004.

The original server reports 161 resolved KV blocks. The pinned configuration
and each arm's runtime record retain that value. The baseline is PR #4 revision
`c0a7e5394cbbe4bcb8a691f0a56c908234289dc3`.

The image tag is `glm53-moe-verification:ubuntu24.04`. The launcher requires
image ID `sha256:8b6d8ccd82c10303c7d2c85d76af025d16142912d8bac6642a10dadf8c776d5c`.
Serving exposes exactly `/dev/neuron0` through `/dev/neuron15`, with host
networking and IPC. `NEURON_EXECUTION_BACKEND=lite` and logical core mode2 are
explicit. No host environment variables are forwarded implicitly.

The staged overlay contains Transformers 5.16.1 and Tokenizers 0.23.1. All 2679
non-bytecode files match the source environment on `trn2-1`. The combined tree
SHA256 is `b2ade0d380022daa8f555e7b149d2698310a033ad5d098ef0badb8a239424353`.
The complete distribution-version comparison found only those two runtime
differences and `pytest-timeout`, which is test-only and was not copied.
The installed SDK venv was not changed. The dependency comparison is retained
in `recovered/dependencies-comparison.json`; the full dependency inventories
remain with the remote artifacts.

## Launch commands

Run on `trn2-2`. The lead supplies the frozen contract path and its SHA256.
The launcher records both before Docker starts. An existing `launch.json`
prevents reuse of an output directory.

```bash
GLM53_ROOT=/home/ubuntu/glm53-moe-fullmodel-20260917
python3 "$GLM53_ROOT/harness/launch.py" plan \
  --arm baseline --source "$GLM53_ROOT/baseline-source" \
  --weights "$GLM53_ROOT/models/GLM-5.3-Flash-04c4e9e9" \
  --deps "$GLM53_ROOT/deps" \
  --output "$GLM53_ROOT/artifacts/baseline-a1" \
  --cache "$GLM53_ROOT/cache/baseline"
```

`plan` prints the exact prospective serving command and does not start Docker.
For a CPU-only import and argument-parser check, use `smoke` and a new output
directory. `smoke` exposes no Neuron devices.

After the lead freezes and reviews the measurement contract, change `plan` to
`serve` and add `--contract /absolute/path/contract.json` and
`--contract-sha256 THE_REVIEWED_SHA256`. Redirect the process output to a server
log outside the source tree. Use the candidate source and a separate candidate
cache for B. Use `--arm postbaseline`, the unchanged baseline source, a new
output directory, and the original baseline cache for the drift control.

The recorded campaign stages `benchmarks/glm53_moe/full_model/` at
`$GLM53_ROOT/harness/`. Copy the two unchanged `support/*.py` files to
`$GLM53_ROOT/scripts/`. Those support scripts retain the campaign's absolute
root and pinned interpreter; the post-baseline controller verifies their
hashes. They reproduce this campaign and are not a general launcher.

The fresh-process controllers require a launch-owner receipt in each server's
output directory. It binds the run token, container, launch hash, and reviewed
contract, and attests that no model request preceded the controller. The
controller also checks the HTTP log. `ordered_trace.py` owns the two held-out
parts. `postbaseline_history.py --plan` prints the exact input manifest for
the original A request replay; its run mode requires the reviewed manifest
hash. Keep the original input records when reproducing that replay.

The frozen campaign scripts retain an accounting error: they assume one
preliminary endpoint request per benchmark cohort. The pinned vLLM client
defaults its endpoint ready check to zero. It sends no such request. Thus
`postbaseline_history.py` reports `FAIL` at its final count check after all
measurements complete: it expects 100, while the correct total is 95. Do not
treat that historical script as an unattended acceptance checker. Keep its
original output and use the [request-accounting review](REQUEST_ACCOUNTING_REVIEW.md)
for this capture. A new campaign must record its actual client request policy
before it freezes a new contract. The [result report](RESULTS.md) records this
deviation and the separate original A/A failure.

The host launch record includes the full Docker arguments, image ID, environment,
run token, contract hash, and harness hashes. Inside Docker, `runtime.json`
records the resolved module paths and versions. `source-manifest.json` hashes
the production package. The process refuses a plugin import outside its selected
source tree or a Transformer/Tokenizer import outside the staged overlay.

## Graph and profile evidence

Before and after measurement, collect the cache and server log:

```bash
python3 "$GLM53_ROOT/harness/collect.py" \
  --cache "$GLM53_ROOT/cache/baseline" \
  --log "$GLM53_ROOT/artifacts/baseline-a1-server.log" \
  --output "$GLM53_ROOT/artifacts/baseline-a1/cache-after.json"
```

The collector hashes NEFFs, FX graphs, HLO, example inputs, and compile metadata.
It preserves named cache-hit and compiler rows from the log. Cache membership
alone does not prove execution. Match a selected graph key to its NEFF hash and
the separate live profile before attributing device time to that graph.

Both arms enable the same live profiler configuration: `cuda` profiler slot,
delay 3, maximum 1 iteration, worker rank 0, device and system traces. The profiler
stays off until `/start_profile` is called. Run all timed cohorts first. Then
call `/start_profile`, submit one fixed 32-token request, and call `/stop_profile`.
Each launch writes to its own `profiles` directory. Do not change KV blocks or
memory geometry for profiling. If profiling cannot start, retain the failed
attempt and report the missing profile; do not infer device time from a replay.

## Results and validation

[RESULTS.md](RESULTS.md) states the current verdict and evidence limits. The
evidence package retains raw completion captures, benchmark cohorts, runtime
records, and source hashes. The `recovered` folder retains the original PR4
checkpoint manifest and dependency comparison. The full original capture
scripts and inventories remain with the remote artifacts. They are historical
evidence, not launch scripts for this experiment.

Launcher validation uses focused unit tests. The Docker smoke confirms
the clean baseline import, dependency overlay, Lite runtime selection, full
server CLI arguments, and both profiler configuration schemas. It does not
construct an engine, compile a graph, or measure the model.

The repository's default pytest paths do not include these benchmark and
experiment folders. Run the portable checks explicitly:

```bash
python3 -m pytest -q benchmarks/glm53_moe/test_reference.py \
  experiments/glm53_moe_nki/test_pack.py \
  experiments/glm53_moe_nki/test_generic_pack.py \
  benchmarks/glm53_moe/full_model/test_launcher.py \
  benchmarks/glm53_moe/full_model/test_ordered_trace.py \
  benchmarks/glm53_moe/full_model/test_postbaseline_history.py
```

Model and native checks need the Neuron SDK. See the
[kernel API and reproduction guide](../../experiments/glm53_moe_nki/README.md)
and the [native compact-row check](../glm53-moe/decode-rows-r1/README.md).
