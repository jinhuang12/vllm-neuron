# Request admission

The Neuron platform refuses a request that the server cannot serve. The refusal
occurs on the request path, before the request goes to the engine. The client gets
HTTP 400 with a message that names the problem. The engine does not see the request,
so it continues to serve other requests.

Code: `vllm_neuron/vllm/admission.py`. Tests: `test/vllm_neuron/vllm/test_admission.py`
and `test/vllm_neuron/vllm/test_admission_http.py`.

## How it works

```
[1. HTTP request] --> [2. render + SamplingParams] --> [3. AsyncLLM.generate call]
                                                              |
                                          [4. NeuronPlatform.validate_request]
                                             |                        |
                                       refused: ValueError      admitted
                                             |                        |
                                     [5. HTTP 400]        [6. EngineCore]
```

1. `NeuronPlatform.check_and_update_config` makes an `AdmissionPolicy` from the
   served config. This occurs in the API server process.
2. vLLM calls `NeuronPlatform.validate_request` for each request, from
   `InputProcessor.process_inputs`. The policy checks the request there.
3. `AsyncLLM.generate` is an async generator. Its body starts only when the response
   is read. For a streamed request, that is after vLLM sends HTTP 200. Thus the
   platform also runs `validate_request` when `generate` is called
   (`install_eager_validation`). Streamed and plain requests both get HTTP 400.
4. vLLM's exception handler changes a `ValueError` into HTTP 400.

## What is refused

| Request | Server | Message names |
|---|---|---|
| Prompt longer than the prefill window | GLM-5.3-Flash with segmented prefill | the prompt length and the window |
| `temperature` > 0 (not the default), `top_k` (not 0, -1, 1), `top_p` < 1, `min_p` > 0, `seed`, `n` > 1 | `on_device_sampling_config.all_greedy: true` | each knob and its value |
| `logprobs`, `logprob_token_ids` | on-device sampling with async scheduling, or the GLM-5.3-Flash sampler | `logprobs=N` |
| `prompt_logprobs` | on-device sampling | `prompt_logprobs=N` |

A server with `on_device_sampling_config: null` (host sampling) does not refuse the
sampling or logprobs requests.

## The prefill window

The GLM-5.3-Flash runner reads the KV of a prefill chunk through a block table. The
table width is fixed for the compiled graph:

```
window_blocks = min(table_width, ceil((kv_segment + query_bucket) / page))
window_tokens = page * window_blocks
```

- `kv_segment`: the largest value in `kv_segment_size_buckets`.
- `query_bucket`: the largest value in `num_batched_tokens_buckets`. The runner widens
  a one-request prefill to this value.
- `page`: the KV block size (`hybrid_kv_block_size`, 128).
- `table_width`: `ceil(max_model_len / page)`. This is never less than `max_model_len`.

The last prefill chunk of a prompt of P tokens holds `ceil(P / page)` pages. If P is
more than `window_tokens`, the runner stops the engine. Thus the policy refuses P more
than `window_tokens`. If `window_tokens` is not less than `max_model_len`, vLLM's own
`max_model_len` check is sufficient, and the policy adds no limit.

| Serve line | Segment | Query bucket | max_model_len | Window |
|---|---|---|---|---|
| Standard (bs=1) | 1024 | 1024 | 4096 | 2048 tokens |
| bs=64 @ 8k | 8192 | 1024 | 8192 | none (max_model_len) |

To serve longer prompts on the standard line, increase `kv_segment_size_buckets`.

## Default sampling parameters

vLLM fills each sampling knob that the client does not send. It uses the model's
`generation_config.json`, then vLLM's defaults. For GLM-5.3-Flash, a request without
`temperature` and `top_p` arrives with `temperature=1.0` and `top_p=0.95`. The policy
cannot tell this request from a request that sends these values. Thus, under
`all_greedy`, a request with all knobs at the server defaults is admitted and served
greedy. The server logs one warning. To get greedy output explicitly, send
`temperature=0`.

## Limits

- The policy does not check `presence_penalty`, `frequency_penalty`,
  `repetition_penalty`, `logit_bias`, `bad_words` or `allowed_token_ids`. The on-device
  sampler reads one `[top_k, top_p, temperature]` row per request and the
  structured-output mask (`functional/full_vocab_sampling.py`), and nothing else.
- The full on-device sampler (`all_greedy: false`) applies `temperature`, `top_k` and
  `top_p`. It does not apply `min_p` or `seed`. The policy does not refuse these.
- A request that the scheduler preempts and recomputes prefills its prompt and its
  output again. The Neuron scheduler caps concurrency so that it does not preempt.
