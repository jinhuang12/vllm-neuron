# GLM-5.3-Flash fused MoE results

The full model ran on all 16 devices of `trn2-2` on September 18, 2026.
The baseline was PR #4 commit `c0a7e5394cbbe4bcb8a691f0a56c908234289dc3`.
Its measured run finished before the final candidate was deployed.

All arms used the same checkpoint, image, runtime, TP64/EP16, and cache
geometry. Each used 20 warmups, then five cohorts of ten requests at
concurrency 1. Ten short prompts repeated in fixed order. Every measured
request generated 32 tokens with EOS ignored: 50 requests and 1,600 output
tokens per arm, with no errors.

| Metric | Original | Fused | Baseline control |
| --- | ---: | ---: | ---: |
| Pooled output tokens/s | 1.369340 | 4.045770 | 1.368737 |
| Mean end-to-end latency | 23.3686 s | 7.9092 s | 23.3788 s |
| Mean time to first token | 6.4265 s | 4.4034 s | 6.4274 s |
| Mean time per output token | 546.517 ms | 113.088 ms | 546.820 ms |
| Mean of cohort p99 end-to-end latency | 23.4032 s | 7.9457 s | 23.4137 s |

Throughput improved by 2.9545x; mean latency fell by 66.15%. The unchanged
10,000-sample whole-cohort bootstrap gave a 95% interval of +195.062% to
+195.767% for mean OTPS gain. All six performance checks passed, including
the -0.0440% baseline-control drift. Cache files stayed unchanged during
timing. These results apply to this warm, concurrency-one workload.

## Correctness and limits

All 40 paired requests across the original and separately frozen new-prompt
traces matched exactly: 1,280 output tokens per arm, stop reasons, top-five
token sets, and logprobs at zero tolerance. All 16 native compact-row cases
matched the legacy combined output bit for bit and passed the fixed CPU
tolerance. The measured source passed 111 portable checks and 63 SDK CPU
tests in Docker. The portable total includes historical controller tests
that are now archived; use the retained test commands below for future work.

The original A/A repeatability check failed before candidate exposure.
The base port's HF accuracy shortfall and DSA prefill alias issue remain
open. Exact preservation on the declared traces does not prove prompt
independence or HF accuracy. Larger standalone fixtures also retain their
CPU-oracle BF16-boundary failures at the original tolerance.

A had five extra diagnostic requests before warmup; B did not. The final
baseline control repeats A's history. That difference remains a timing
limitation. The frozen contract also assumed one preliminary request per
cohort, but the installed client sent zero in all arms. An independent
audit reconciled 95/90/95 successful requests for A/B/control. The original
controller failure remains in the archive; no measurements or numerical
thresholds changed.

Full-model graphs and separate live captures confirmed the fused path in
all 42 routed layers. Profile event loss and conflicting core metadata
prevent a global critical-path or removable-time claim.

## Evidence and future checks

The [original evidence and review records](https://github.com/jinhuang12/vllm-neuron/tree/9e8a5bbc8b2765b7838679283c88bd8fd2513b37/verification)
remain at an immutable commit. That snapshot includes raw responses, cohort
results, manifests, original failures, and the request-accounting audit.
They are historical outputs, not regression fixtures.

Use the [kernel tests and native verifiers](../../experiments/glm53_moe_nki/README.md)
and [full-model comparison tools](../../benchmarks/glm53_moe/full_model/README.md)
for new changes. Keep generated results outside the source tree.
