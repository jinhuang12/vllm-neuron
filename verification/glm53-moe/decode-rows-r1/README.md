# Compact routed rows

The packed model path keeps the original 256-row mapping. It sends the first `min(T,256)` columns of each mapping block to the fused kernel and uses those same token IDs in the FP32 combine. Each block stores real tokens first. Its remaining rows carry padding. The legacy path is unchanged.

The production source SHA256 is `6c8948978030845c83cc11b2acba91f6eca5ae8fa7f5cb119d558687d7494f2d` for `vllm_neuron/model/glm5_next/model_fp8.py`.

CPU RED: 3 expected failures at T1, T3, and T127; 27 passes. The first run also found a test interception error. Both logs are retained. CPU GREEN: all 30 new tests and all 10 prior model integration tests passed. The new tests cover mapping prefixes, empty blocks, full blocks, tails at T257/T511/T512, device-rank routing semantics, and the unchanged legacy block width.

The native Docker verifier passed on trn2-2, `/dev/neuron1`, logical core 4, LNC2, with `NEURON_EXECUTION_BACKEND=lite`.

| Case | H | I | Local experts | Top-k | T | Fused row shape | Legacy preservation | Fixed CPU gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Production | 4096 | 512 | 18 | 8 | 1 | `[8,1]` | PASS | PASS |
| Small multirow | 256 | 256 | 18 | 8 | 3 | `[18,3]` | PASS | PASS |

Each case ran eight affinity, EP-rank, empty-route, and repeat variants. All 16 outputs were bit-exact to the legacy route. Both routes repeated their outputs exactly. Each route compiled once per shape. The compact graph contains one fused dispatch; the legacy graph contains three dispatches. The real model preparation hook released all six source banks and retained the two packed operands.

The CPU gate remains `atol=1e-5, rtol=0.03`. The largest production CPU absolute difference was `0.00006103515625`; all values passed the gate. The small case matched the CPU oracle exactly. This proves the scoped routed-expert change. It makes no full-model correctness or serving performance claim.

Reproduction, from the prior proof source root on trn2-2, after device assignment:

```sh
GLM53_NEURON_DEVICE=/dev/neuron1 NEURON_RT_VISIBLE_CORES=4 \
  bash verification/glm53-moe/run_container.sh run -- \
  env NEURON_EXECUTION_BACKEND=lite \
  python experiments/glm53_moe_nki/verify_decode_rows.py \
  --output /home/ubuntu/glm53-moe-nki-20260917/verification/decode-rows-native-new
```

The verifier returns failure if either baseline preservation or the CPU gate fails. The native result reports, graph text, and passing CPU log are included here. `sha256.json` records the full artifact hashes. Per-call tensors, earlier logs, full fixtures, and compiler artifacts remain on trn2-2 at `/home/ubuntu/glm53-moe-nki-20260917/verification/decode-rows-native-r1` and the campaign's verification logs.
