# GLM-5.3-Flash fused MoE: full-model measurements

Status: all three timed runs are complete. All six frozen performance tests
pass. The original and held-out traces pass their cross-arm preservation
checks. The original A/A repeatability failure remains open. A separate
request-accounting audit records and resolves a protocol deviation below.

The baseline is PR #4 commit
`c0a7e5394cbbe4bcb8a691f0a56c908234289dc3`. Its measured run finished before
the final candidate was deployed. The candidate packs expert weights once,
uses one fused NKI call per routed layer, and removes unused rows from small
decode calls. Routing, scale application, FP32 combine, and the final BF16
cast retain their existing semantics.

## Measured serving performance

Both arms ran the full model on the same trn2-2 instance in the same Docker
image, with TP64, EP16, concurrency 1, and fixed cache geometry. Each arm used
20 warmup requests, then five cohorts of ten measured requests. The ten short prompts repeated in
fixed order. Every measured request generated 32 tokens with EOS ignored.
Both arms completed 50 measured requests and 1,600 output tokens without error.

| Metric | Original | Fused candidate | Baseline control |
| --- | ---: | ---: | ---: |
| Pooled output tokens/s | 1.369340 | 4.045770 | 1.368737 |
| Mean end-to-end latency | 23.3686 s | 7.9092 s | 23.3788 s |
| Mean time to first token | 6.4265 s | 4.4034 s | 6.4274 s |
| Mean time per output token | 546.517 ms | 113.088 ms | 546.820 ms |
| Mean of cohort p99 end-to-end latency | 23.4032 s | 7.9457 s | 23.4137 s |

The A/B ratio is 2.9545x, with a 66.15% decrease in mean end-to-end
latency. This is a warm, concurrency-one result for this workload. It is not
a result for longer prompts, other batch sizes, or aggregate serving capacity.
The unchanged analysis includes all five cohorts per arm. Its 10,000 whole-
cohort bootstrap samples give a 95% interval of +195.062% to +195.767% for the
mean OTPS gain. Mean and mean-of-cohort-p99 end-to-end latency improve. Mean
and mean-of-cohort-p99 time to first token improve. The final baseline control
differs by -0.0440% in mean cohort OTPS, within the frozen 5% limit. All six
performance tests pass. See [the complete analysis](performance-comparison.json).

The cache file sets, contents, sizes, and modification times stayed unchanged
during all three timed runs. All arms used the same runtime versions and server
arguments. Separate live profiles followed the timed runs. Profiles do not
contribute to these performance numbers.

The original A run had five extra diagnostic requests between its first two
correctness passes and its 20 warmups. B had the two correctness passes followed
directly by the same 20 warmups. The post-baseline control replays all 25 of A's
initial requests before the same warmups and timing. Its first two correctness
passes also match the original A passes exactly. The A/B history difference
remains a limitation of the timing comparison.

The frozen contract and post-baseline controller expected one preliminary
endpoint request per cohort. The installed vLLM client defaults its endpoint
ready check to zero, so it sent none in any arm. All 15 cohort logs confirm
this. The actual completion counts are 95 for A, 90 for B, and 95 for the
control: each has 20 warmups and 50 measured requests, after its recorded
history. All responses have HTTP status 200. The controller's original
`FAIL` record and the original contract remain unchanged. The independent
[request-accounting review](REQUEST_ACCOUNTING_REVIEW.md) accepts the captured
work under this documented deviation. This correction changes no workload,
measurement, correctness tolerance, or performance threshold. The original
contract is not reported as an overall pass.

## Correctness evidence and limits

The initial A/A exact repeatability check failed before candidate exposure.
The second pass changed 114 generated token IDs across five of ten prompts;
logprob responses also changed. The same pattern appears in B/B. This failure
remains recorded. Its cause is not established by this experiment.

At matching request positions across fresh launches, A1 equals B1 and A2
equals B2 exactly. The unchanged comparator checked prompt and output token
IDs, stop reason, top-five token sets, and their logprobs. All 20 paired
requests and 640 output tokens passed, with maximum logprob difference zero.
This demonstrates preservation on those traces; it does not establish prompt
independence or HF accuracy.

After that finding, a separate protocol was frozen before any new captures.
It uses ten new prompts in a 20-request order with adjacent repeats and later
returns. A and B each start in a fresh process. There are no intervening model
requests, resets, retries, changed prompts, sequence shifts, or selected
subsequences. The same comparator must pass both ten-request parts with zero
logprob tolerance. Both parts passed on September 18 at 03:06 UTC. All 20
paired requests and 640 generated tokens matched, with maximum logprob
difference zero. The controller verified exactly 20 completion requests on
each fresh server. Across the original and held-out traces, all 40 request
pairs and 1,280 output token IDs match. This supplemental result does not turn
the failed original A/A gate into a pass.

Component verification is separate: all 16 compact-row native test cases
match the legacy combined output bit for bit and pass the fixed CPU tolerance.
The CPU tests cover routing completeness, weight coordinates, load-time
packing, source-bank release, and unchanged compatibility dispatch. The clean
PR snapshot passed all 111 portable checks. Its 257 production/build files
match the deployed candidate source hash exactly, including `pyproject.toml`
and `setup.py`.

The same clean snapshot also passed 63 SDK CPU tests inside Docker: 14 loader
and release checks, nine tiny-model checks, and 40 integration/compact-row
checks. This container had no Neuron devices or network and used read-only
source, dependency, SDK, and environment mounts. The separate native checks
and full-model captures provide hardware evidence.

The base port's HF accuracy shortfall and its known DSA prefill alias issue
remain open. The fused MoE change does not claim to fix either issue. The alias
issue has not been established as the cause of the A/A failure.

## Executed graph and profile limits

The candidate's full-model decode and prefill graphs each contain all 42
fused routed-expert layers. Decode passes row IDs of shape `[8,1]`; prefill
keeps the 256-row capacity. A separate live capture binds the executed decode
NEFF to that graph. The baseline live capture binds its executed decode NEFF
to 42 instances of each legacy gate/up, SwiGLU, and down kernel.

Baseline profile events include loss warnings. The candidate profile has
conflicting physical-core metadata. These traces support execution identity,
but do not support a certified removable-time fraction or global critical-path
bound. No Amdahl speedup estimate is used as an acceptance result.

## Reproduction and artifacts

See `README.md` for the pinned runtime and launcher. The frozen inputs and
measurement scripts are under `benchmarks/glm53_moe/full_model/`. The original
contract, supplemental contract, failures, raw captures, cohort records, source
manifests, and cache records remain separate.

Large server logs, compiler graphs, NEFFs, NTFFs, and profile tables remain on
`trn2-2:/home/ubuntu/glm53-moe-fullmodel-20260917/`. The small evidence package
records hashes and the exact source and runtime used for each arm.
