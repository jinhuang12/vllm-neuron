VERDICT: PASS

## Adversarial Review: Held-out ordered-trace preservation

**Goal:** Verify exact preservation of PR4 behavior on the frozen twenty-request trace under the recorded fresh-process startup procedure.

**Overall verdict: PASS.** The bounded preservation claim is supported. The original A/A gate remains failed.

Evidence root: `/home/jinhun/vllm-neuron-campaigns/glm53-moe-nki-20260917/verification/glm53-moe-fullmodel/`. Links below resolve under this root. Review date: 2026-09-18.

### Claims assessed

| Claim | Verdict | Confidence | Primary evidence |
| --- | --- | --- | --- |
| Both corresponding parts match at the frozen zero tolerance. | CONFIRMED | High | [Part 1](candidate-heldout-r1/ordered-trace/comparison-part-1.json), lines 4–7, and [part 2](candidate-heldout-r1/ordered-trace/comparison-part-2.json), lines 4–7: `"mismatches": []`, `"max_absolute_logprob_delta": 0.0`, `"pass": true`. |
| Both arms completed the full trace. | CONFIRMED | High | [Baseline status](baseline-heldout-r1/ordered-trace/status.json), lines 11–12, and [candidate status](candidate-heldout-r1/ordered-trace/status.json), lines 11–12: `"completed_requests": 20`, `"completed_output_tokens": 640`. |
| The request history starts with a fresh process and contains no extra API model requests. | CONFIRMED | High | Both [baseline](baseline-heldout-r1/fresh-process.json) and [candidate](candidate-heldout-r1/fresh-process.json) receipts, lines 8–10, record `"fresh_process": true` and `"model_requests_before_capture": 0`. Their HTTP receipts record 0, 10, 20, and 20 completions. I verified each receipt against its saved server-log prefix. |
| The original failed gate remains visible. | CONFIRMED | High | [Original repeatability](baseline-r1/repeatability.json), line 943: `"pass": false`. [Candidate status](candidate-heldout-r1/ordered-trace/status.json), lines 8–9: `"FAIL, retained unchanged"` and `"supplement_does_not_pass_original_correctness_gate": true`. |

### Independent checks

I read all forty individual request records and the four aggregate captures. Every individual record matches its aggregate entry and recorded SHA256. Both comparison-file hashes and their input hashes match the candidate status receipt. The candidate reference hash matches the completed baseline status. Evidence: each arm's `ordered-trace/status.json`, `parts` section, and the two comparison files above.

At all twenty paired positions, request bodies, prompt token IDs, generated token IDs, finish/stop reasons, and complete logprob objects match exactly. This covers 640 generated token positions, 640 generated-token log probabilities, and 3,200 top-five log probabilities. All probabilities are finite; every request finishes with `length` and exactly 32 generated tokens. I recomputed these checks directly from both arms' `ordered-trace/part-{1,2}/capture.json` records, independently of the comparator's reported pass.

The contract copies and recorded client/driver hashes match the frozen files. The contract SHA256 is `df470e4526ee8ca311bdf5dc43dc06efa8106626ee1159d0150fa37eac62c9e8`. Its freeze time, 02:20:01 UTC, precedes all new requests. The two part hashes match the frozen twenty-position order. The ten distinct prompt texts have no overlap with the original workload. Evidence: [contract](candidate-heldout-r1/ordered-trace/contract.json), lines 3 and 37–69; both arms' `ordered-trace/preflight.json`; the four capture files.

Each arm ran part 1 then part 2 in chronological order with successful child exits. The HTTP histories contain only twenty successful `/v1/completions` POSTs. I checked all four prefix hashes per arm against the retained full logs, including [baseline before](baseline-heldout-r1/ordered-trace/http-before.json), [candidate before](candidate-heldout-r1/ordered-trace/http-before.json), and both `http-complete.json` receipts. No additional POST appears after capture. The full server logs remain archive evidence; their bound prefix hashes are retained in these small receipts.

### Runtime and freshness

The baseline and candidate have distinct launch tokens and container names. Their prelaunch device records show all sixteen devices without Neuron processes. The freshness receipts bind those records and the launch files by hash. Evidence: both arms' `fresh-process.json` and `prelaunch-neuron-ls.json`.

Both launches use the pinned image, configuration, checkpoint path, SDK versions, and dependency overlay. Their source, checkpoint, and dependency mounts are read-only. Runtime module paths select the intended source and overlay. I verified the preflight input hashes and recomputed each source-manifest tree hash. Evidence: both arms' `launch.json`, `runtime.json`, `source-manifest.json`, and `ordered-trace/preflight.json`.

The runtime source trees match the contract: baseline `5817aa193c3409af726dc699f5fc16c4de8885d1aa6e01b72b1313f00b8f7a08`; candidate `c366203b7643192407527bec6ff5866cfdf8773ab8bed80a101638aa9abf6c93`. These values appear at line 52 of each arm's `runtime.json`.

### Residual limits and assessment

The supported claim is exact output preservation on this recorded ordered trace after the specified fresh startup. Byte-identical hidden state, prompt independence, broader input coverage, and HF correctness were not established. The frozen [contract](candidate-heldout-r1/ordered-trace/contract.json), lines 4 and 106, sets this scope and retains the original failure.

No material defect remains in this supplemental correctness gate. Serving-performance acceptance and the final publication review remain separate gates. No sub-agent, model request, device execution, or remote mutation was used for this review.
