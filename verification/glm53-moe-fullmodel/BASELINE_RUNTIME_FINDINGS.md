# Baseline repeatability: runtime findings

**Goal:** Find evidence for the unchanged baseline's repeatability failure before candidate B. Keep the exact preservation gate intact.

**Assessment:** The difference is present before sampling. Its cause remains unknown. The current full-model preservation gate cannot pass. The frozen contract says, “If A1/A2 differ, halt acceptance” (`/home/jinhun/vllm-neuron-campaigns/glm53-moe-nki-20260917/benchmarks/glm53_moe/full_model/contract.json:58`). No threshold change follows from this investigation.

All remote paths below are on `trn2-2`. This review read files, process environments, and installed code. It sent no model requests and changed no server settings.

## Confirmed exclusions

**Sampling randomness does not explain the first-step raw-logprob drift.** The installed `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/lib/python3.12/site-packages/vllm/sampling_params.py:680–682` returns `SamplingType.GREEDY` for temperature zero. In the same installation, `vllm/v1/sample/sampler.py:80–88` computes `raw_logprobs = self.compute_logprobs(logits)` before sampling; lines 239–271 use `logits.argmax(dim=-1)` and return directly for an all-greedy batch. Line 306 uses `logits.log_softmax(dim=-1, dtype=torch.float32)`.

The baseline uses this sampler: `/home/ubuntu/glm53-moe-fullmodel-20260917/baseline-source/vllm_neuron/vllm/worker/neuron_model_runner.py:49` imports `Sampler` from that module; lines 9118–9124 move logits to CPU; lines 9995–10001 invoke it when on-device sampling is off. Lines 2034–2041 create a seeded generator only for `SamplingType.RANDOM_SEED`. The active server log contains “Using vLLM Sampler” at `/home/ubuntu/glm53-moe-fullmodel-20260917/artifacts/baseline-r1/server.log:228656`.

**The difference is not only a common softmax normalization offset or later autoregressive divergence.** Saved requests 0 and 1 have equal request bodies and equal prompt IDs `[785, 6722, 315, 9621, 374]`. Their first-step values are:

| Token | Request 0 | Request 1 | Difference, 1 minus 0 |
|---|---:|---:|---:|
| 12089 | -0.6169655919075012 | -0.5276957750320435 | +0.0892698169 |
| 264 | -2.7419655323028564 | -2.965195655822754 | -0.2232301235 |
| 825 | -2.9919655323028564 | -2.840195655822754 | +0.1517698765 |

Evidence: `/home/ubuntu/glm53-moe-fullmodel-20260917/artifacts/baseline-r1/repeat-diagnostic-1/request-0.json:131–134` and `request-1.json:131–134`; prompt IDs are at lines 393–399 in both files. The difference spread is 0.375. Subtracting two token log probabilities cancels the common log-normalizer. Thus relative logits differ at the first output position. This localizes the failure; it does not identify which model operation or state caused it.

## Remaining unknowns

- **Stochastic rounding is unverified as a cause.** A read-only `/proc/<pid>/environ` check across all 68 processes in the launch's container found no `NEURON_RT_STOCHASTIC_ROUNDING_EN`, `NEURON_RT_STOCHASTIC_ROUNDING_SEED`, `NEURON_RT_DBG_CC_STOCHASTIC_ROUNDING_EN`, `XLA_USE_BF16`, or `XLA_DOWNCAST_BF16`. Command output was `process_environment_groups [({}, 68)] read_failures []`. This is an environment observation, not a hardware-mode measurement. The [official NRT developer guide, model-load environment section](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/guides/nrt-developer-guide.html) states `NEURON_RT_STOCHASTIC_ROUNDING_EN=<true/false>, default=false`. No affirmative evidence supports blaming stochastic rounding.
- **Changing collective reduction order is unverified.** The [official intra-node collective guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/explore/intranode-collective-comm.html), “Mesh ReduceScatter” and “KangaRing ReduceScatter” sections, describes reduction algorithms. It does not establish which live reduction order this execution used or show that the order changed. General floating-point non-associativity is not causal evidence.
- **Mutable state and actual first-forward inputs are not captured in these HTTP receipts.** The receipts contain request bodies and responses; they do not establish identical internal tensors or incoming cache state. Equal prompt IDs alone do not resolve that gap.

## Independent check of the state and scatter findings

**Confirmed: the compiled prefill loses the DSA tail writes.** I independently enumerated the remote FX placeholders and metadata. In `/home/ubuntu/glm53-moe-fullmodel-20260917/cache/baseline/neuron/compile_cache/c3f74100dfeed892706f9b3488c90c2b/.artifact_metadata_v0.json:6–18`, all 11 prefill-tail inputs are unused: `115,277,439,601,763,925,1087,1249,1411,1573,1735`. None occurs in `io_map`. Its `example_inputs.txt:14–16` gives input 2 shape `(1024,)`; `fxgraph.txt:117` identifies input 115 as `L_layer_carriers_3_prefill_tail_`. That FX graph still contains `getitem(tail, 0)` followed by `copy_` at lines 4217–4222, and the corresponding row-1 write at lines 4223–4228. The decode graph `301f479170329f3b9bffbc4f9c5736be` retains and aliases all 11 tail inputs; its `fxgraph.txt:3887` writes directly with `copy_(l_layer_carriers_3_tail_, new_tail)`.

The installed SDK explains this loss. In `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/lib/python3.12/site-packages/libtorch_neuronx_lite/fx_passes/aliasing_pass.py:543–544`, `operator.getitem` is recognized as a view only when an argument contains a slice. Integer indexing by 0 or 1 does not meet that rule or the alias operation sets at lines 18–60. Root resolution returns `None` for an unrecognized operation at lines 1017–1020. In the same installation, `compile/backend.py:509–513` removes unused inputs from execution; lines 524–526 reuse an input as an output only through `io_map`. Thus these are lost state writes, not merely missing debug labels.

**Limit:** This defect concerns prefill-to-decode state handoff. It does not explain the observed first-token drift: those tail inputs are absent from the compiled prefill execution. Nor does missing seeding alone prove request-to-request contamination; the source has a separate slot-clearing path. The local evidence copies are in `verification/glm53-moe-fullmodel/baseline-r1/state-investigation/`. Their prefill/decode metadata hashes match the remote files (`4a28543f…` and `c09e5ea9…`). No runtime state snapshot or repair was performed in this review.

**Confirmed: MoE combine is ADD, but execution order remains unknown.** Independent protobuf reads of both remote `graph.hlo` files found 42 MoE-shaped scatters per graph, each with an `add` combiner. Decode `scatter.8308` uses `ScatterCombiner.8304`, whose root is `add.8307`. The decode `log-neuron-cc.txt:3698` maps `_scatter.8308` to `indirect_rmw float32` into `non_local float32 (2, 4096)`. This refutes a simple-overwrite interpretation. It does not prove repeated-address reduction order or deterministic execution. The same protobuf reads found no explicit RNG opcode, and both metadata files state `"has_rng_seed_parameter": false`; this does not establish the hardware rounding mode.

## Bounded next diagnostic

No further read-only check of the current receipts can distinguish changing internal inputs/state from execution variability. After the active diagnostic measurement, the next useful test is to compare exact first-forward inputs, incoming mutable state, and raw output logits across repeated baseline executions. Existing SDK hooks can observe the actual post-filter execution inputs and raw outputs: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/lib/python3.12/site-packages/libtorch_neuronx_lite/compile/backend.py:547–557`, `pre_execute_hook(execute_inputs, ...)` and `execute_hook(execute_inputs, outputs, ...)`. If inputs/state differ, locate that difference first. If they match and outputs differ, restore one frozen input/state snapshot for repeat execution before isolating rounding or a collective. This is a proposed diagnostic, not an established cause or an instruction to change the current server.

A paired comparison of old and fused MoE operations on the same captured operands can support a scoped component-preservation claim. It cannot replace the failed full-model gate. Do not derive a full-model acceptance tolerance from the observed baseline variation or from candidate results. Resolve the cause and freeze a defensible comparison before treating B as acceptance evidence.
