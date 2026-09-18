# Compact-row integration review

**Goal:** Check that the final packed-route change removes only padded rows, keeps the same token mapping for the fused kernel and FP32 combine, and preserves the tested routed-expert model outputs.

**VERDICT: PASS.** No material defect remains in this scoped change.

Reviewed model SHA256: `6c8948978030845c83cc11b2acba91f6eca5ae8fa7f5cb119d558687d7494f2d`.

## Claims assessed

| Claim | Verdict | Primary evidence |
|---|---|---|
| Compaction affects only the packed route and uses the same IDs for kernel and combine. | CONFIRMED | `/home/jinhun/vllm-neuron-campaigns/glm53-moe-nki-20260917/vllm_neuron/model/glm5_next/model_fp8.py:2258,2269–2275`: `rows = min(tokens, block)`, contiguous prefix selection, and replacement of `token_position_to_id`. Lines 2318–2325 use those IDs in FP32 `index_add`. The legacy branch at 2282–2303 retains the full mapping. |
| Removed rows contain no real token routes. | CONFIRMED | `/home/jinhun/vllm-neuron-campaigns/glm53-moe-nki-20260917/vllm_neuron/functional/moe/moe_blockwise.py:103,389–417,454–458`: binary expert mask, per-expert count, cumulative prefix placement, and `-1` initialization. For T below the block width, each expert has at most T rows; for larger T, the selected width is unchanged. |
| Native execution covers production decode and a non-aligned multirow case. | CONFIRMED | `/home/jinhun/vllm-neuron-campaigns/glm53-moe-nki-20260917/verification/glm53-moe/decode-rows-r1/native/production/report.json:6–30`: H4096/I512/T1, E18/top8, fused rows `[8,1]`. The sibling `native/small/report.json:6–30` records H256/I256/T3 and rows `[18,3]`. Both report one compile per route, one fused dispatch, and three legacy dispatches. |
| All recorded model outputs preserve legacy behavior and pass the fixed CPU gate. | CONFIRMED | Both native reports, lines 2–4, state `PASS` for baseline preservation and the CPU oracle. Their `results` contain eight routing/rank/empty/repeat variants each. I independently loaded all 16 saved `.pt` outputs, checked bitwise compact/legacy equality and repeats, and reproduced every CPU metric with one CPU thread. |
| CPU regression tests detect the intended change. | CONFIRMED | `/home/jinhun/vllm-neuron-campaigns/glm53-moe-nki-20260917/verification/glm53-moe/decode-rows-r1/logs/decode-rows-red-r2.log:343–346`: failures at T1/T3/T127, `3 failed, 27 passed`. The sibling `decode-rows-green-r1.log:20` records `40 passed`. Current `experiments/glm53_moe_nki/test_decode_rows.py:110–118` asserts compact width, contiguity, exact IDs, and intercepts the imported production mapping correctly. |

## Independent checks and limits

All 28 entries in the native evidence `sha256.json` match their local files. Every source hash recorded by both native reports matches the current source. I also inspected both saved graphs: each fused kernel receives the compact IDs, and each final combine uses an FP32 zero tensor and `index_add`.

The fixed CPU limits remain `atol=1e-5`, `rtol=0.03`; no threshold changed. Exact comparison concerns the returned BF16 model output after FP32 combine. It is not a claim that every intermediate FP32 value is bitwise identical.

The current CPU test SHA256 is `0f0bb9f818be72cedb77a2c5e9fb2b1897d5c06428fedd6463300d44945fc8e5`. The earlier preparation report described an earlier test revision; this report covers the corrected mapping interception and contiguous fixture.

No sub-agent or new device execution was used for this review. Full-model correctness, A/B timing, and the unresolved baseline repeatability gate remain outside this PASS.
