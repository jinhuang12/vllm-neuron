# SPDX-License-Identifier: Apache-2.0
"""Neuron scheduler admission: each request reserves the KV blocks of its worst case.

The engine configs are Kimi K3's served decode forms, built in CPU mode with no
workers; the KV cache config holds the form's four groups, one latent and three
state. Set ``KIMI_K3_CHECKPOINT`` to the checkpoint directory; the default is
``~/models/moonshotai/Kimi-K3``. Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import dataclasses
import logging
import os
from pathlib import Path

import pytest
import torch
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    get_max_concurrency_for_kv_cache_config,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_neuron.model.kv_cache import KeyOnlyAttentionSpec
from vllm_neuron.vllm.core.scheduler import NeuronScheduler
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

CHECKPOINT = Path(
    os.environ.get(
        "KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3"
    )
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

TP, DP = 32, 2
LATENT = 576
KDA_HEADS, KDA_HEAD_DIM, CONV_TAPS = 96, 128, 4
USERS_PER_REPLICA, OUTPUT_TOKENS = 16, 128
# (DCP, maximum length, prefill chunk, prompt tokens) of the 2K and 8K 32-user cells.
SHAPE_2K = (4, 32768, 8192, 2048 - OUTPUT_TOKENS)
SHAPE_8K = (16, 131072, 8192, 8192 - OUTPUT_TOKENS)
MAX_STEPS = 4000


def _engine_config(dcp: int, max_model_len: int, chunk: int) -> VllmConfig:
    """The form's engine config after the platform's block-size step."""
    config = EngineArgs(
        model=str(CHECKPOINT),
        tensor_parallel_size=TP,
        data_parallel_size=DP,
        enable_expert_parallel=True,
        decode_context_parallel_size=dcp,
        max_model_len=max_model_len,
        max_num_seqs=32,
        max_num_batched_tokens=chunk,
        long_prefill_token_threshold=chunk,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.90,
        skip_tokenizer_init=True,
    ).create_engine_config()
    current_platform.update_block_size_for_backend(config)
    assert config.cache_config.block_size == 256
    return config


def _kv_cache_config(config: VllmConfig, num_blocks: int) -> KVCacheConfig:
    """vLLM's KV cache config of the form's layers at a block count."""
    cache = config.cache_config
    shapes = MambaStateShapeCalculator.kda_state_shape(
        TP, KDA_HEADS, KDA_HEAD_DIM, conv_kernel_size=CONV_TAPS
    )
    dtypes = MambaStateDtypeCalculator.kda_state_dtype(
        torch.bfloat16, cache.mamba_cache_dtype
    )
    latent = KeyOnlyAttentionSpec(
        block_size=cache.block_size,
        num_kv_heads=1,
        head_size=LATENT,
        dtype=torch.bfloat16,
    )
    state = MambaSpec(
        block_size=cache.mamba_block_size,
        shapes=tuple(shapes),
        dtypes=tuple(dtypes),
        page_size_padded=cache.mamba_page_size_padded,
        mamba_cache_mode=cache.mamba_cache_mode,
        num_speculative_blocks=config.num_speculative_tokens,
    )
    text = config.model_config.hf_text_config
    kda_layers = set(text.linear_attn_config["kda_layers"])
    specs = {
        f"layers.{i}.self_attn": state if i + 1 in kda_layers else latent
        for i in range(text.num_hidden_layers)
    }
    layouts = NeuronWorker.__new__(NeuronWorker).get_supported_kv_cache_layouts()
    resolve_kv_cache_layout(config, [layouts], list(specs.values()))
    groups = get_kv_cache_groups(config, specs)
    slots = max(len(g.layer_names) for g in groups)
    kv_config = get_kv_cache_config_from_groups(
        config, groups, num_blocks * slots * latent.page_size_bytes
    )
    state_groups = [g for g in kv_config.kv_cache_groups if g.kv_cache_spec == state]
    assert len(kv_config.kv_cache_groups) == 4 and len(state_groups) == 3
    assert kv_config.num_blocks == num_blocks
    return kv_config


def _scheduler(config: VllmConfig, kv_config: KVCacheConfig) -> NeuronScheduler:
    """The scheduler as the engine core builds it, with no workers."""
    config.cache_config.num_gpu_blocks = kv_config.num_blocks
    block_size, hash_block_size = resolve_kv_cache_block_sizes(kv_config, config)
    return NeuronScheduler(
        vllm_config=config,
        kv_cache_config=kv_config,
        structured_output_manager=StructuredOutputManager(config),
        block_size=block_size,
        hash_block_size=hash_block_size,
    )


def _requests(prompt_tokens: int, max_tokens: int, count: int = USERS_PER_REPLICA):
    """Requests of one prompt length and one output limit."""
    return [
        Request(
            request_id=f"r{i}",
            prompt_token_ids=[1] * prompt_tokens,
            sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
            pooling_params=None,
            arrival_time=float(i),
        )
        for i in range(count)
    ]


def _serve(scheduler: NeuronScheduler, requests: list[Request]) -> dict:
    """Drive schedule and update_from_output, one sampled token per request per step."""
    for request in requests:
        scheduler.add_request(request)
    most_running, preemptions, steps = 0, 0, 0
    while scheduler.has_unfinished_requests() and steps < MAX_STEPS:
        output = scheduler.schedule()
        steps += 1
        most_running = max(most_running, len(scheduler.running))
        preemptions += len(output.preempted_req_ids or ())
        req_ids = list(output.num_scheduled_tokens)
        sampled = [
            [7]
            if scheduler.requests[r].num_computed_tokens
            >= scheduler.requests[r].num_prompt_tokens
            else []
            for r in req_ids
        ]
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={r: i for i, r in enumerate(req_ids)},
                sampled_token_ids=sampled,
            ),
        )
    return {
        "most_running": most_running,
        "preemptions": preemptions + sum(r.num_preemptions for r in requests),
        "completed": sum(r.num_output_tokens == r.max_tokens for r in requests),
    }


@pytest.mark.parametrize("num_blocks", [251, 151])
@pytest.mark.parametrize("shape", [SHAPE_2K, SHAPE_8K], ids=["2K-32", "8K-32"])
def test_sixteen_short_requests_run_at_once(shape, num_blocks):
    """16 requests of 5 blocks each, 80 in all, run at once at 251 and at 151 blocks."""
    dcp, max_model_len, chunk, prompt_tokens = shape
    config = _engine_config(dcp, max_model_len, chunk)
    scheduler = _scheduler(config, _kv_cache_config(config, num_blocks))
    requests = _requests(prompt_tokens, OUTPUT_TOKENS)
    assert [scheduler._kv_blocks_reserved(r) for r in requests] == [5] * 16
    assert _serve(scheduler, requests) == {
        "most_running": 16,
        "preemptions": 0,
        "completed": 16,
    }


def test_requests_over_the_pool_wait_and_none_is_preempted(caplog):
    """Seven 35-block requests fit in 251 blocks; the rest wait, none is preempted."""
    config = _engine_config(4, 32768, 8192)
    scheduler = _scheduler(config, _kv_cache_config(config, 251))
    requests = _requests(32768 - OUTPUT_TOKENS, OUTPUT_TOKENS)
    assert [scheduler._kv_blocks_reserved(r) for r in requests] == [35] * 16
    with caplog.at_level(logging.DEBUG, logger="vllm_neuron.vllm.core.scheduler"):
        served = _serve(scheduler, requests)
    assert served == {"most_running": 7, "preemptions": 0, "completed": 16}
    assert any("Prefill deferred" in r.getMessage() for r in caplog.records)


def test_default_output_limit_reserves_the_maximum_length():
    """At vLLM's default output limit a request reserves the maximum length's blocks."""
    config = _engine_config(4, 32768, 8192)
    kv_config = _kv_cache_config(config, 251)
    scheduler = _scheduler(config, kv_config)
    prompt_tokens = 2048 - OUTPUT_TOKENS
    # vLLM's input processor sets an unset output limit to the rest of the length.
    (request,) = _requests(prompt_tokens, 32768 - prompt_tokens, count=1)
    per_request = kv_config.num_blocks / get_max_concurrency_for_kv_cache_config(
        config, kv_config
    )
    assert scheduler._kv_blocks_reserved(request) == round(per_request) == 35


def test_speculative_state_blocks_and_lookahead_are_read_from_the_config(monkeypatch):
    """With 4 speculative state blocks and 4 lookahead tokens, a request takes 3 + 7."""
    config = _engine_config(4, 32768, 8192)
    kv_config = _kv_cache_config(config, 251)
    speculative_blocks = iter([2, 1, 1])
    kv_config.kv_cache_groups = [
        dataclasses.replace(
            g,
            kv_cache_spec=dataclasses.replace(
                g.kv_cache_spec, num_speculative_blocks=next(speculative_blocks)
            ),
        )
        if isinstance(g.kv_cache_spec, MambaSpec)
        else g
        for g in kv_config.kv_cache_groups
    ]
    # The platform refuses a speculative config under DCP, so the lookahead comes
    # through the config property the scheduler reads.
    monkeypatch.setattr(VllmConfig, "num_lookahead_tokens", property(lambda self: 4))
    scheduler = _scheduler(config, kv_config)
    assert scheduler.num_lookahead_tokens == 4
    requests = _requests(2048 - OUTPUT_TOKENS, OUTPUT_TOKENS)
    assert [scheduler._kv_blocks_reserved(r) for r in requests] == [3 + 7] * 16
