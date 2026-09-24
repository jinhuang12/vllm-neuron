# GLM-5.3-Flash compact state and concurrent decode: draft

This change is not ready to merge. Single-request serving matches the measured
baseline, but two-request serving fails the unchanged exact correctness check.
There is no measured candidate throughput or accepted speedup.

## Branch and dependencies

The draft targets `release-0.24.0.1.1.0`, the base of PR #10. It includes PR #10's
cleaned port at `a9f30889068935a88bc707fe853894bccfaa0299`. Until that port merges,
the release comparison includes its commits too.

PR #10 does not contain the ordered MoE combine from PR #9. This draft carries
that prerequisite from `d9df12e668b1486e9bbae7e3fc952dd08a256c92`, preserving the
starting point used by the optimization campaign. Its kernel is unchanged.

The original campaign starts at the PR #9 merge,
`54d4b550ff97b2e906fa9446f707ebd4ba5369ff`. The draft preserves the latest
candidate's executable AST in the model, runner and worker, apart from the 12
error-message string changes already in PR #10. Comments and docstrings follow
the cleaned branch. Tests use that branch's renamed helper modules.

## Changes

- Allocate KDA state by admitted request count. Keep the padded slot layout,
  dtypes and scheduler cache specs. The measured configuration needs 8.50 MiB
  per rank for KDA state, down from 684.25 MiB.
- Keep per-request KDA arithmetic and state writes. Join output partials for one
  FP32 tensor-parallel reduction per KDA layer before the model-dtype cast.
- Preserve the tested singleton rounding with a minor-axis request layout for
  tensor-parallel sums and per-request MHC preparation and KDA input norms.
- Run sparse attention once per request with separate page tables and side-cache
  views. Write latent updates through the original three-dimensional bank.
- Give startup warmup separate temporary slots and pages. Refuse synthetic cache
  writes when a live request owns state. Concurrent prefill remains unsupported.

The native serving scope is TP64, EP16, DP1, LNC2, with parallel tracing disabled
and decode batches of one or two. Larger-batch component tests do not extend
that full-model scope.

## Validation and remaining failure

The cleaned draft passed **372 selected tests** in 137.55 seconds on 2026-09-24.
These ran without devices in Docker image
`sha256:8b6d8ccd82c10303c7d2c85d76af025d16142912d8bac6642a10dadf8c776d5c`,
with `VLLM_NEURON_CPU_MODE=1`, `NKI_SIMULATOR=1`, `NKI_PRECISE_FP=1` and
`NEURON_PLATFORM_TARGET_OVERRIDE=trn2`. This is CPU and simulator evidence only.

The selected pytest paths were:

```text
test/vllm_neuron/worker
test/vllm_neuron/model/glm5_next/test_request_keyed_state.py
test/vllm_neuron/model/glm5_next/test_kda_batched_decode.py
test/vllm_neuron/model/glm5_next/test_kda_input_norm_rows.py
test/vllm_neuron/model/glm5_next/test_dsa_concurrent_decode.py
test/vllm_neuron/model/glm5_next/test_mhc_concurrent_decode.py
test/vllm_neuron/model/glm5_next/test_tp_row_reduction.py
test/vllm_neuron/model/glm5_next/test_kda_layer.py
test/vllm_neuron/model/glm5_next/test_kda_request_operands.py
test/vllm_neuron/model/glm5_next/test_kda_reduction.py
test/vllm_neuron/model/glm5_next/test_mhc_composition.py
test/vllm_neuron/functional/moe/test_ordered_combine.py
```

Historical results below belong to the campaign source, not a new serving run
of this draft branch. The runtime used PyTorch 2.11.0, vLLM 0.24.0,
libtorch-neuronx-lite 2.11.0.1.0.1284+f49d8626 and Neuron compiler
2.27.5334.0+f702b353 on Trainium2.

| Check | Result |
| --- | --- |
| Merged baseline, three timed cohorts | 3,840 output tokens / 408.712003 s = 9.395369 tokens/s |
| Latest candidate, selected Docker tests | 76 passed |
| Latest candidate, one-request serving | Exact baseline token IDs and log probabilities |
| Latest candidate, two-request serving | Failed: all 10 first-decode log-probability comparisons differ; four later token sequences diverge |
| Candidate throughput | Not run; correctness gate failed |
| Latest observation replay | 384 cases across 64 ranks, independently checked |

The latest replay preserves prior outputs and state. Its captured attention
result, residual streams, post gate and residual mixing matrix match between
singleton and paired execution. The combined residual streams differ. The
remaining interval contains the FP32 input casts, original MHC post method and
final BF16 cast. This does not yet establish a kernel defect or a production fix.

A small replay of the original MHC post method is written in the campaign
workspace. It has not run on hardware. Its output must reproduce all 384 saved
same-arm results before it can support attribution. Four full-serving candidates
have failed; component passes have not cleared the serving failure.

## Evidence retained outside this PR

The full diagnostic scripts, tensors, compiler artifacts and failed attempts stay
in the campaign workspace. They are not included in the release diff.

- Workspace: `/home/jinhun/vllm-neuron-campaigns/glm53-batched-kda-20260921/verification/batched-kda`
- Serving report: `e2e-report.json`, latest source `candidate-r12-row-norm`.
- Original model SHA256: `c4ae77a20392cc115f02ae42f74847b19cb9d0a76b490430df3a1a3b451d23eb`.
- R5 observation report SHA256: `a5c6c4d05277049b80781cf3f61fff387e43587e769a2a31fccc41049cce1fa3`.
- Independent R5 review SHA256: `e3ffdb419de3a49790a4cc53ea4ba69c48045ee6a8bc9e9e765002e43f5fe32b`.
- Frozen R5 reference SHA256: `8c8c2047365f1bf9fa13f8ba5155666d690ce61789455dcd19b5dcbc1bae2f90`.

Merge requires an exact full-model correctness pass, followed by the registered
throughput comparison. Neither requirement has been met.
