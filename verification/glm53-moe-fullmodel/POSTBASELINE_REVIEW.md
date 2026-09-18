# Post-baseline replay review

Independent reviewer: `moe_plan_review`. Verdict: PASS before post-baseline
execution on September 18, 2026.

The reviewed controller SHA256 is
`03971f1a1a8018626d4130485d4b1ba5c43e45e8816ba2d3dab8e1421e4d9554`.
Independent validation passed 15 tests and nine additional diagnostic-result
fault probes. The controller keeps real numerical diagnostic failures, stops
on request or tool failures, and invokes the unchanged pinned timing helper.

The reviewer independently recomputed the actual input manifest SHA256:
`f27f6b2602aabbb608d09e7d6aba7e27bb00eac6cd96aa04f6956767adce9d2a`.
All 41 input hashes matched both local records and the live host. Raw bodies
and timestamps establish the original order: A1, A2, then diagnostics 0–4.
That is 25 requests and 800 output tokens. The full expected count is 100:
25 history requests, 20 warmups, five preliminary requests, and 50 measured
requests. Both included support scripts match their pinned hashes.

The original A/A result stays failed. A had five diagnostic requests before
warmup that B did not have. This post-baseline replay does not remove that
limitation from the original A/B comparison. The performance gates remain
unchanged.
