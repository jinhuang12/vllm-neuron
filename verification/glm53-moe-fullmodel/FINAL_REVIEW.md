# Final adversarial review: fused GLM-5.3-Flash MoE

**VERDICT: PASS.**

**Goal:** Publish a bounded, hardware-verified optimization PR into `campaign/glm-5.3-flash-port`, with the pre-existing A/A and HF accuracy issues and the protocol/history limits disclosed. Full-port accuracy repair is outside this goal.

No BLOCKING or HIGH finding remains. This verdict approves the reviewed PR body and report for that scope. It does not report the original frozen campaign contract as an overall pass.

The review root is `/home/jinhun/vllm-neuron-campaigns/glm53-moe-nki-20260917`. Evidence links below resolve within that root.

## Claims assessed

| Claim | Verdict | Primary evidence |
|---|---|---|
| All six frozen numerical performance gates pass. | CONFIRMED, high confidence. | [performance-comparison.json](performance-comparison.json), lines 230–245: gain `195.45454400300497`, CI `[195.0615731995415,195.76700802807432]`, drift `-0.04402201190463195`, and `"performance_pass": true`. I recomputed the complete result from all 15 raw cohorts using the unchanged analyzer; every JSON field agrees exactly. |
| Post-A used the original source, runtime, and baseline cache. | CONFIRMED, high confidence. | [post-A launch](postbaseline-r1/launch.json), lines 5–11: pinned image, `baseline-source`, and `cache/baseline`; [post-A runtime](postbaseline-r1/runtime.json), `source_tree_sha256` field: `5817aa193c3409af726dc699f5fc16c4de8885d1aa6e01b72b1313f00b8f7a08`. Its complete source manifest equals A's. |
| The staged production source is the tested and measured candidate. | CONFIRMED, high confidence. | [clean-source receipt](clean-snapshot-r1/source-tree.json), lines 2–9: 257 files, tree `c366203b7643192407527bec6ff5866cfdf8773ab8bed80a101638aa9abf6c93`, `"pass": true`. I checked every staged production/build file against the deployed candidate manifest. No production, test, benchmark, or experiment file differs from tested tree `1f1e90088a4ec65267bca0e8f7efec1302a9b722`. |
| Correctness claims preserve the original baseline on the declared traces. | CONFIRMED, high confidence. | [preservation summary](preservation-summary.json), lines 3–10: exact scope, 40 pairs, 1,280 output tokens, zero tolerance, and original A/A `"FAIL; unchanged"`. The [held-out review](HELDOUT_REVIEW.md), lines 20–24, records independent checks of all paired raw records. |
| The request-accounting failure prevents use of the completed measurements. | REFUTED, high confidence; LOW residual documentation defect, corrected by the appended audit. | [accounting review](REQUEST_ACCOUNTING_REVIEW.md), “Independent request and cache checks” and “Narrow correction”: all three arms used zero preliminary requests; counts are 95/90/95. The original contract and failed controller remain unchanged. |
| The final report and PR body disclose the limits that matter to the goal. | CONFIRMED, high confidence. | [RESULTS.md](RESULTS.md), lines 46–63, 67–89, 105–120; `/tmp/glm53-fused-moe-pr-body.md:21`: warm concurrency-one scope, A/B history difference, original A/A/HF/DSA issues, accounting deviation, and profile limits. |

## Numerical checks

The unchanged analyzer uses 10,000 independent, seeded resamples of five whole cohorts per arm. Its six gate predicates match the frozen contract: positive lower confidence bound, improved mean and mean-of-cohort-p99 E2E latency, mean and mean-of-cohort-p99 TTFT within 5%, and absolute post-A mean-OTPS drift within 5%. Evidence: `benchmarks/glm53_moe/full_model/analyze.py:46–75` and `contract.json:62`.

| Gate | Observed value | Result |
|---|---|---|
| Positive lower 95% OTPS-gain bound | +195.061573% | PASS |
| Mean E2E improves | 23.368558 s to 7.909164 s | PASS |
| Mean of cohort p99 E2E improves | 23.403170 s to 7.945730 s | PASS |
| Mean TTFT does not regress more than 5% | 6.426544 s to 4.403422 s | PASS |
| Mean of cohort p99 TTFT does not regress more than 5% | 6.433392 s to 4.409053 s | PASS |
| Absolute post-A mean-OTPS drift is at most 5% | 0.044022% | PASS |

These values come from the baseline/candidate/postbaseline `mean_of_cohorts` fields and the gate fields in [performance-comparison.json](performance-comparison.json). The headline pooled rates are 1.369340, 4.045770, and 1.368737 output tokens/s. The cohort-mean ratio is 2.954545; mean E2E falls 66.154679%. The report distinguishes pooled rates from the mean-cohort bootstrap.

## Runtime, source, and validation checks

Across A, B, and post-A, I compared image ID, resolved versions, Python executable, dependency module paths and hashes, weights path, config and contract hashes, all 16 exposed devices, and server arguments after normalizing only the profile output directory. They agree. Post-A's environment differs from A only by the fresh run token. B's additional differences are its declared source/cache paths and run token. Evidence: each arm's `launch.json` and `runtime.json`; [post-A source manifest](postbaseline-r1/source-manifest.json), `files` and `tree_sha256`.

All three ten-file cache maps remain identical before and after timing. All 15 cohort files have ten completed requests, zero failures/errors, and ten output lengths of 32. All three arms have 20 valid saved warmups. The independently counted server-log prefixes support those counts. Evidence and hashes are in [request-accounting.json](request-accounting.json), `arms`.

Post-A's first two captures match A's corresponding original captures at zero tolerance. I verified both comparator input hashes. Evidence: `postbaseline-r1/postbaseline-history/comparison-a1.json` and `comparison-a2.json`, `pass`, `left_sha256`, `right_sha256`, and zero-delta fields.

Prior independent component and graph checks remain applicable to the unchanged candidate: [compact-row review](COMPACT_ROWS_REVIEW.md), “Claims assessed,” records all 16 native cases; [execution summary](execution-summary.json), `candidate.identity.counts`, records `"moe_fused_fp8_kernel": 42`. The clean logs record `111 passed` at [portable log](clean-snapshot-r1/pytest.log):3 and `63 passed` at [SDK log](clean-sdk-r1/sdk-tests.log):14. I rechecked the SDK log hash against its receipt. No new device test was needed for this final document/evidence review.

## Residual risks and scope

The original A/A repeatability gate remains failed. Exact cross-arm preservation does not prove prompt independence or HF accuracy. The known DSA prefill alias issue remains open. The five extra original A history requests remain a timing limitation; post-A's matching history and stable rate do not erase it. Loss and conflicting metadata limit profile attribution, so no global critical-path or removable-time bound is accepted. The final report and PR body state all of these limits.

The accounting supplement changes no captured workload, numerical threshold, cohort selection, or raw evidence. It records the difference between the frozen preliminary-request assumption and the actual client behavior. The reproduction guide explicitly warns that the historical controller still fails its incorrect count check. This is sufficient for the completed acquisition; no rerun is required by the evidence reviewed here.

No sub-agents were used in this final pass. I read local and remote evidence and performed lightweight hash and numerical checks. I made no production, test, harness, contract, or raw-result changes.

## Reviewed document hashes

| Document | SHA256 |
|---|---|
| `verification/glm53-moe-fullmodel/RESULTS.md` | `78c3a9558dc1f915ed507fc5654ecb7d1e787492238f39739e0d08fc148a20ef` |
| `verification/glm53-moe-fullmodel/README.md` | `ae11a8af24b5739bd548d93e7a9b416dfab9330dbf5b4c9341fd3d34a10b21b5` |
| `/tmp/glm53-fused-moe-pr-body.md` | `247f9b58d01741d1aea94a358e36a35612b5820321931a1524608872b7cfab02` |
| `verification/glm53-moe-fullmodel/performance-comparison.json` | `779b9bc21c9e145e48f513187d6f8ef354759260d29071172ea50e83feebdcc4` |
| `verification/glm53-moe-fullmodel/preservation-summary.json` | `65a2f2356f2d49d1595b4d23344f85353e9920c6db4ab93dfa594e2d31737786` |
| `verification/glm53-moe-fullmodel/source-change-summary.json` | `9e22130f64950b691eefad3d752342908848aebf6fbb6b7d04cc7b213318324c` |
| `verification/glm53-moe-fullmodel/REQUEST_ACCOUNTING_REVIEW.md` | `4b278ce54cd435ce0e9c19db0a4312fc15bf9ab5f32fc5dddf622a376e3afdbc` |
| `verification/glm53-moe-fullmodel/request-accounting.json` | `9a6cb7aa4aebf534e93ef0cb0d5a29e6b76214daa5ca7bf61f45acda0027d22f` |
| `benchmarks/glm53_moe/full_model/analyze.py` | `30c9f1889784e16fd5173027c8245b08d9b71d2cfe08ff15ff6aab995d52525f` |
| `benchmarks/glm53_moe/full_model/contract.json` | `2965fe097dba8f8c6b97e12fedbfaf69aa4549d3462a3dba1255eef92053e924` |
| `verification/glm53-moe-fullmodel/postbaseline-r1/postbaseline-history/status.json` | `a88cf86797b64b0c78da2b0f69f67e44b2f5838a5d4b74584e26a7131e530f56` |

The preservation summary's current publication field says review is pending. It may be updated to this scoped PASS while keeping all evidence fields and original-failure statements unchanged. Stage this final report and the current accounting review with the reviewed packet. Any later change to production, thresholds, captures, or the scope of the published claim requires a new review.
