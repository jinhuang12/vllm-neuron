# Request-accounting review

**VERDICT: PASS — acquisition under the recorded accounting deviation.**

**Goal:** Determine whether the completed post-baseline timing control can be accepted from primary evidence after the request-accounting controller failure, without changing performance or correctness thresholds.

This review accepts the collected timing data for the separate frozen performance analysis. It does not mark the original contract or controller as passed. It does not give final publication approval.

## Claims assessed

| Claim | Verdict | Evidence |
|---|---|---|
| Each cohort sent one preliminary request. | REFUTED, high confidence; LOW residual after this correction. | The installed client's `serve.py:852–868` sends that request only when `ready_check_timeout_sec > 0`. Lines 1808–1812 set the default to zero. All 15 saved cohort logs contain `Skipping endpoint ready check.` at line 13. |
| The post-baseline run completed every required measured request. | CONFIRMED, high confidence. | `postbaseline-r1/benchmark/complete.json:2–5` records five cohorts, 50 measured requests, 1,600 output tokens, and the frozen workload hash. I checked every cohort's ten successful 32-token responses and empty errors. The server records all 95 actual completions. |
| All three arms used the same benchmark client work. | CONFIRMED, high confidence. | Every saved cohort command is identical after normalizing only result directory and filename. Every cohort log records `ready_check_timeout_sec=0` and `num_warmups=0`. Each arm has 20 saved warmups and 50 measured requests. |
| The controller failed after successful acquisition. | CONFIRMED, high confidence. | `postbaseline_history.py:270–272` checks helper success and validates all timing records before the erroneous count of 100. `postbaseline-history/timing-exit.json:2` records `"returncode": 0`; `status.json:3,26–28` retains `"status": "FAIL"`, timing stage, and the count error. |

The installed client source is `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/lib/python3.12/site-packages/vllm/benchmarks/serve.py` on `trn2-2`, SHA256 `b09e85eed6f5f7a9075b40a19fc1dd8e94a1c449665c2c6510f349e0409e77b7`. The separate `num_warmups > 0` branch starts at line 870. Thus `--num-warmups 0` does not explain the skipped preliminary request; the zero ready-check timeout does. Exact excerpts and all 15 log/result/command hashes are in [request-accounting.json](request-accounting.json).

## Independent request and cache checks

I counted every HTTP POST in each complete saved server log. All requests before profiling were successful `/v1/completions` calls. A and B each later have exactly three profile-related POSTs: start, one completion, and stop. Post-A has no profile POSTs.

| Arm | History | Warmups | Measured | Preliminary | Total before profiling | Last measured completion line |
|---|---:|---:|---:|---:|---:|---:|
| A, baseline-r1 | 25 | 20 | 50 | 0 | 95 | 505031 |
| B, candidate-r1 | 20 | 20 | 50 | 0 | 90 | 481616 |
| Post-A, postbaseline-r1 | 25 | 20 | 50 | 0 | 95 | 504565 |

The source logs are `/home/ubuntu/glm53-moe-fullmodel-20260917/artifacts/<arm>/server.log` on `trn2-2`. Each cited final line contains `"POST /v1/completions HTTP/1.1" 200 OK`. The hashed prefix ends immediately after that line's newline. Full-log hashes, byte offsets, boundary lines, and later profile calls are retained in the JSON record.

| Arm | Prefix bytes | Prefix SHA256 |
|---|---:|---|
| A | 75203881 | `338791d1bf4cbb6aa3f765f886a7edb222a7a4964806247d591386fa7a1551b7` |
| B | 71803393 | `e56630cbfbb823b84d2dc3e3c6622c938b9a3302c1a6d8c3da601e4097a56670` |
| Post-A | 74977813 | `17ef0acbbe6f934a3732dea4577b74346e189399599a18fbd7c1a5f75df27ae7` |

I independently compared each arm's `cache-ready-file-state.json` and `cache-after-timing-file-state.json` file maps. All ten entries are equal in every arm. Each arm's 20 saved warmups has 32 output tokens and finish reason `length`. All 15 cohorts have ten completions, 320 output tokens, ten output lengths of 32, and no errors. No cohort was removed or replaced.

## Narrow correction and preserved limits

The frozen `benchmarks/glm53_moe/full_model/contract.json:44` declares one preliminary request per cohort. The frozen controller at `postbaseline_history.py:272–274` therefore expects 100 requests. Both statements conflict with the actual installed client behavior across all three arms. This is a protocol bookkeeping deviation, not only a post-A log-count error.

Keep the original contract, input manifest, driver, failed status, commands, and raw results unchanged. This review and its JSON record append the correction: zero preliminary requests in every cohort, and total pre-profile counts of 95/90/95. Correct current result prose and warn reproduction users that the frozen controller's count check will fail with this client default. No controller rewrite or new measurement is required to assess this completed acquisition.

The performance and correctness thresholds stay unchanged. The original A/A failure stays failed. A's five additional diagnostic history requests remain a limitation relative to B; post-A repeats A's 25-request history. Equal benchmark client work does not erase that difference. Final performance acceptance must still use every cohort and the frozen drift, confidence-interval, latency, and completion gates.

No sub-agent, model request, rerun, source edit, contract edit, or raw-artifact edit was used in this review. The only new files are this report and its accounting record.
