# GLM-5.2 (`GlmMoeDsaForCausalLM`)

This package contains the direct vLLM-Neuron implementation for the pinned
GLM-5.2 checkpoint.

## Support contract

- Architecture: `GlmMoeDsaForCausalLM`
- Tensor parallel size: 64
- Expert parallel size: 64
- Main decoder layers: 78
- MTP layers: excluded from main execution
- Checkpoint format: block FP8 weights with BF16 activations
- On-device sampling: greedy only

`GlmMoeDsaConfig` validates the complete pinned architecture and quantization
contract. It rejects unsupported checkpoint variants.

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
