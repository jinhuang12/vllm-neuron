# Dynamic-\`q_len\` ragged paged attention (experimental)

This standalone NKI kernel is a **first correctness baseline** for reproducing the
variable-length query path from
[vLLM TPU RPA v3](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py).
It is **not wired into the Neuron vLLM scheduler, model graph, or CTE/TKG
selection**. A merged prefill/decode serving step is not enabled by this change.

Implementation: [\`dynamic_ragged_paged_nki.py\`](../../../vllm_neuron/functional/attention/dynamic_ragged_paged_nki.py)

## Design

Each program handles one request and one Q head. It:
1. Reads \`q_start\`, \`q_end\` and post-append \`kv_len\` from HBM metadata.
2. Uses \`nl.fori_loop(start, end, body)\` to process **only active query
   tokens**; no graph recompilation for each exact \`q_len\`.
3. Calculates the causal prefix of each query, then iterates **only its
   visible KV pages** with a second runtime \`nl.fori_loop\`.
4. Looks up physical KV pages in a request-specific page table.
5. Evaluates QK and PV via TensorE, with a causal mask on the last page,
   a stable online softmax, and an FP32 accumulator.
6. Writes its own query-head output range into a shared HBM output buffer.

Mathematically, for request \`r\` and its local query index \`i\`:

\`\`\`text
q_len[r]       = cu_q_lens[r+1] - cu_q_lens[r]
old_len[r]     = kv_lens[r] - q_len[r]
visible_kv_len = old_len[r] + i + 1
kv_head        = q_head // (num_q_heads // num_kv_heads)
\`\`\`

There is no cross-request attention.

### Shapes

| Input | Shape | Notes |
| --- | --- | --- |
| \`q\` | \`[Hq, 128, max_tokens]\` | BF16; active tokens packed across requests |
| \`key_pages\` | \`[pages, Hkv, 128, 128]\` | BF16; K stored \`[dim, token]\` |
| \`value_pages\` | \`[pages, Hkv, 128, 128]\` | BF16; V stored \`[token, dim]\` |
| \`page_table\` | \`[reqs, max_pages_per_req]\` | INT32 physical page IDs |
| \`cu_q_lens\` | \`[reqs + 1]\` | INT32 cumulative active query lengths |
| \`kv_lens\` | \`[reqs]\` | INT32 **post-append** total lengths |
| Output | \`[Hq, 128, max_tokens]\` | BF16; padded slots unspecified |

Fixed geometry: head dimension 128, cache page size 128. Head grouping supports
standard MHA, MQA and GQA, provided \`Hq % Hkv == 0\`.

**Caller preconditions:** \`0 <= cu_q_lens[0] <= ... <=
cu_q_lens[reqs] <= max_tokens\`; \`kv_lens[r] >= q_len[r]\`;
\`kv_lens[r] <= 128 * max_pages_per_req\`; every consumed \`page_table\`
entry in \`[0, pages)\`. All current-step K/V must **already be committed**
into the cache, including the appended positions read by the current step.
Updates to shared cache pages require a correct copy-on-write policy; the
kernel does not implement or enforce that policy. Invalid metadata may
address invalid memory. A host wrapper validates static shapes, not runtime
metadata values.

### Why this is not yet a performance-optimized TPU-equivalent kernel

- **Query tile = 1**, so prefill work repeats KV reads for each query.
  TPU RPA v3 normally uses query tiles and asynchronous double buffering.
- The prototype uses noncontiguous \`[Hq, D, T]\` Q storage to avoid
  a Q transpose. Packing and DMA efficiency need profiling.
- It does not append KV entries or implement KV-cache writes.
- The kernel cannot by itself run prefill and decode in one model graph.
  That requires a scheduler/metadata/model graph integration and compiled
  mixed buckets.
- It only implements **causal** attention at BF16 D=128, without sliding
  windows, speculative tokens, quantized cache, multi-query tiles or
  asynchronous memory-pipeline optimization.
- \`nl.fori_loop\` is supported by recent Neuron SDKs, but the Neuron
  compiler must still validate that these specific nested dynamic loops
  and indirect page DMAs lower on the selected Trainium2 target.

## Validation and hardware gate

CPU-only reference tests use shuffled physical KV pages and an independent
dense attention oracle across mixed, decode-only, zero-query, and
page-boundary cases.

\`\`\`bash
pytest -q test/vllm_neuron/functional/attention/test_dynamic_ragged_paged_nki.py
\`\`\`

With Neuron SDK/NKI installed, explicitly run the optional simulator test:

\`\`\`bash
NKI_RPA_SIMULATE=1 pytest -q \
  test/vllm_neuron/functional/attention/test_dynamic_ragged_paged_nki.py
\`\`\`

**No compiler/hardware pass is claimed.** A correctness-complete next gate
needs a Neuron SDK installation on a Trn2 image to run the NKI simulator,
compile a NEFF for Trn2, and compare actual NKI outputs with the independent
CPU oracle. Run profiler measurements **only after device correctness**.
Suggested scenarios: \`q_len=[1,1,1]\`, \`[1,3,2]\`, \`[2,129]\`,
\`[0,1,0]\`, long KV (>2048), GQA/MQA and unaligned final pages.

Use the official
[AWS Neuron agentic skills](https://github.com/aws-neuron/neuron-agentic-development):
\`neuron-nki-writing\`, \`neuron-nki-docs\`, \`neuron-nki-debugging\`,
and \`neuron-nki-profiling\`. NKI runtime loops are described in
[the \`nl.fori_loop\` API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.language.fori_loop.html);
[dynamic address patterns](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki-aps.html)
document indirect DMA addressing.

## Follow-on performance plan

1. **Compiler gate**: resolve any NKI verifier errors; verify true dynamic
   query and KV bounds on Trn2, rather than padded static loops.
2. **Fusion**: integrate packed metadata into a mixed graph with shared
   QKV/dense operations. Keep existing CTE/TKG fallback.
3. **Tile multiple queries** (e.g. 16/32/64 rows), preserve per-row
   causal masks and online softmax; reduce redundant KV reads.
4. **Memory pipeline**: prefetch page metadata and paged K/V with
   double-buffered SBUF and compare TensorE/VectorE overlap.
5. **Cache updates**: add a separate safe cache-append stage or fused writes
   with explicit cache ownership semantics, including page tail writes.
6. **Tune**: compare mixed vs separated P/D under matched request traces,
   reporting TTFT, ITL p50/p99, output tokens/s, compilation latency and
   SBUF/PSUM allocation. Reject a mixed path that only improves throughput
   by introducing unacceptable decode tail latency.
