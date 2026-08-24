# GLM-5.2 (`GlmMoeDsaForCausalLM`)

This package contains the direct vLLM-Neuron implementation for the pinned
GLM-5.2 checkpoint.

## Support contract

- Architecture: `GlmMoeDsaForCausalLM`
- Tensor parallel size: 64
- Expert parallel size: 64
- Main decoder layers: 78
- MTP layers: excluded from main execution
- Checkpoint format: block FP8 weights converted to row-scaled FP8 for routed
  experts, with BF16 activations
- On-device sampling: greedy only

`GlmMoeDsaConfig` validates the complete pinned architecture and quantization
contract. It rejects unsupported checkpoint variants.

## Validated runtime envelope

Revision `ae56198` was validated on one Trainium2 `trn2.48xlarge` instance on
2026-08-20.

- Fresh TP64 compile: 384 HLO graphs, 384 NEFF files, and 384 completion
  markers, with no compiler errors.
- Cached activation: prefill buckets 16, 128, and 512 and decode buckets 1, 8,
  and 32, with no cache misses or compiler processes.
- Offline concurrency 32: 288 of 288 requests passed for prompt lengths 16,
  128, and 480 and output lengths 1, 16, and 32.
- OpenAI-compatible concurrency 32: 576 of 576 chat-completion and completion
  requests passed for the same prompt and output matrix.
- Length boundary: one offline request with 2,016 prompt tokens and 32 output
  tokens completed at the 2,048-token limit.

The CUDA reference run on eight H200 GPUs captured 384 requests with the same
model, tokenizer, and greedy sampling. Its exact prompts and outputs were used
for a 320-request Neuron diagnostic. Token sequences diverged across backends,
while representative longer responses retained the same meaning. FP8 execution
and backend reduction order are not bit-exact. This is a semantic smoke result,
not a promise of token-for-token equality.

## Current limits

- Concurrency 32 is validated through 512 total prompt and output tokens.
- The 2,048-token result is a single offline boundary request. It does not
  establish concurrency-32 or HTTP service at that length.
- Context lengths above 2,048 tokens are not validated.
- The selective block-FP8 MoE and selected-latent MLA paths remain
  experimental and disabled by default.
- This validation does not include performance targets.

## Package structure

| Module | Responsibility |
| --- | --- |
| `factory.py` | Registry entry and TP contract |
| `config.py` | Hugging Face to Neuron config mapping |
| `model.py` | Decoder, LM head, cache binding, and weight loading |
| `attention.py` | MLA projection and sparse attention |
| `indexer.py` | DSA index selection and packed indexer keys |
| `mlp.py` | Dense and shared SwiGLU MLP |
| `moe.py` | Router, routed experts, and expert parallel execution |
| `cache.py` | MLA and indexer cache specifications and paged access |
| `block_fp8.py` | Block FP8 linear primitives |
| `weight_loaders.py` | Checkpoint grammar, accounting, and shard plans |

## vLLM-Neuron integration

The implementation uses the shared model interfaces for:

- registry construction through `GlmMoeDsaForCausalLM` in `factory.py`;
- tensor-parallel embeddings and linear layers;
- `Sampler` for on-device sampling;
- `KVSpec` and runner cache binding;
- safetensors checkpoint loading and parameter weight loaders.

The package exports only the registry factory, config, and reusable model
components. The implementation class stays in `model.py`, as it does for the
closest sibling models.

## Tests

Focused tests are in `test/vllm_neuron/model/glm_moe_dsa/`.

- `test_stage1.py`: architecture and registration
- `test_stage2.py`: checkpoint contract and shard planning
- `test_stage3.py`: attention, indexer, cache, and FP8 primitives
- `test_stage4.py`: dense MLP, router, and experts
- `test_stage5.py`: model, runner, sampling, and cache integration
- `test_stage6.py`: FP8 execution and memory gates

Runtime changes require a new validation run from the earliest affected stage.
Do not reuse hardware evidence from a candidate with different runtime hashes.
