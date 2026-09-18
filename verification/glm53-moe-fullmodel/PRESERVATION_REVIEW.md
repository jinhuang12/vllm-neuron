# Supplemental ordered-trace review

Independent reviewer: `moe_plan_review`. Verdict: PASS before any held-out
model request on September 18, 2026.

The reviewer verified contract SHA256
`df470e4526ee8ca311bdf5dc43dc06efa8106626ee1159d0150fa37eac62c9e8`
and controller SHA256
`4c78a0159b19e02bc602db3aca07fd61284faff7d64f807da91c4ee8871dbc0b`.
The new prompts, request order, client, runtime, and source hashes were frozen
before capture. The controller requires a fresh process, zero earlier model
requests, the exact two-part order, and both unchanged comparator passes.
It keeps partial or failed results and does not retry model requests.

Independent validation passed 21 tests. Four extra fault probes changed only
part 2: a logprob by `1e-12`, a top-five key, a prompt token ID, and the stop
reason. Each failed the second comparison after the first passed and kept all
20 raw requests. The known SDK library-path prefix is allowed narrowly.

This contract tests exact preservation on one fresh-process ordered trace.
It does not pass the original failed A/A gate or establish prompt independence
or HF accuracy. Publication still requires this capture's results and the
unchanged performance gates.
