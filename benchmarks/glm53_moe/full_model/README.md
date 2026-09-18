# GLM full-model comparison tools

These tools can compare future changes to the GLM model. They keep test
inputs and measurement code in the repository. Write outputs to a new
directory outside the source tree. Historical measurements and their limits
are in the [results report](../../../verification/glm53-moe-fullmodel/RESULTS.md).

## Launch a comparison arm

`launch.py` accepts `plan`, `smoke`, and `serve`. It mounts the chosen source,
SDK, Python environment, dependency overlay, checkpoint, and runtime config
read-only. Each arm needs separate output and cache directories. `smoke`
validates imports and the actual server CLI inside Docker without devices.
`serve` exposes all 16 Neuron devices; use it only on an assigned idle host.

Copy `config.json` to the result directory's parent and pass `--config`.
The default records the measured TP64/EP16 GLM setup. For a new environment,
set its Docker image tag and exact image ID, runtime version pins, and server
settings in the copy. The runtime checks the same selected file. Build the
OS image with [the Docker runner](../../../verification/glm53-moe/run_container.sh)
and stage Transformers and Tokenizers in the dependency overlay first.

```bash
python benchmarks/glm53_moe/full_model/launch.py smoke \
  --arm baseline --config /absolute/experiment/config.json \
  --source /absolute/baseline-source --weights /absolute/checkpoint \
  --deps /absolute/dependency-overlay --venv /absolute/sdk-venv \
  --sdk /opt/aws/neuron --output /absolute/results/baseline-smoke \
  --cache /absolute/cache/baseline-smoke
```

`plan` prints the prospective command. `serve` also requires `--contract`
and `--contract-sha256` for a separately written measurement plan. Record the
source revisions, workload and request order, runtime, and acceptance limits
before capture. The launcher records the plan bytes and checks its hash;
it does not validate the scientific design or apply the acceptance gates.

## Capture correctness and performance

Use one fresh server per source. Keep model identifiers, request order,
runtime, and checkpoint identical. With the client SDK active:

```bash
python benchmarks/glm53_moe/full_model/experiment.py capture \
  --model /absolute/checkpoint --base-url http://127.0.0.1:18004 \
  --workload benchmarks/glm53_moe/full_model/workload.jsonl \
  --output /absolute/results/baseline/correctness
python benchmarks/glm53_moe/full_model/experiment.py compare \
  --left /absolute/results/baseline/correctness/capture.json \
  --right /absolute/results/candidate/correctness/capture.json \
  --output /absolute/results/comparison.json
python benchmarks/glm53_moe/full_model/experiment.py benchmark \
  --model /absolute/checkpoint --output /absolute/results/baseline/benchmark
```

The capture/comparison tools use ten prompts, 32 output tokens, exact token
IDs, stop reasons, and top-five logprobs with zero tolerance. The two files
under `preservation/` provide another 20-request sequence: capture part 1,
then part 2 without intervening requests, and compare each part separately.
These workloads can check preservation of a baseline; they are not an HF
accuracy oracle. Diagnose baseline repeatability before interpreting changes.

Benchmarking uses 20 warmups and five ten-request cohorts at concurrency 1.
Inspect the installed client's endpoint ready-check policy before counting
requests. It may skip preliminary requests. `analyze.py` takes `--baseline`,
`--candidate`, and `--postbaseline` benchmark directories plus `--output`.
It uses all five cohorts and a fixed whole-cohort bootstrap. Correctness,
runtime identity, cache stability, and request history need separate checks.

`collect.py` records cache files and hashes. `inspect_rows.py` checks routed
row shapes in compiler graphs. `profile_evidence.py` binds graphs to profile
captures; cache membership alone does not prove execution. Each has `--help`.

## Portable checks

```bash
python3 -m pytest -q benchmarks/glm53_moe/test_reference.py \
  experiments/glm53_moe_nki/test_pack.py \
  experiments/glm53_moe_nki/test_generic_pack.py \
  benchmarks/glm53_moe/full_model/test_launcher.py
```

The [kernel guide](../../../experiments/glm53_moe_nki/README.md) also lists
the SDK integration tests and native verifiers. They generate their fixtures
and do not need the archived campaign outputs.
