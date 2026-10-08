# SPDX-License-Identifier: Apache-2.0
"""The served prefill window is max_model_len: any prompt up to max_model_len - 1 fits.

The GLM-5.3-Flash runner reads each prefill chunk's KV through a block table of
``min(table_width, ceil((kv_segment_size + query_bucket) / page))`` pages
(``_glm5next_model_kwargs``). Until now ``kv_segment_size_buckets`` held one value and every
chunk read ``buckets[0]``, so the standard line (segment 1024, query bucket 1024) served at
most 2048 tokens of a 4096-token model: a 3000-token prompt stopped the engine at
"a request longer than its bucket cannot be served by this window", and admission then
answered HTTP 400 instead. These tests pin the replacement:

1. the runner accepts several segments, compiles one prefill graph per (query, segment)
   pair and refuses at startup a list whose largest segment does not cover
   ``max_model_len`` (so no config that starts has a window below ``max_model_len``);
2. a list the user did not set is completed with the covering segment;
3. every chunk of a request carries the smallest segment at least as long as the
   request, and a window that holds the chunk; a decode step keeps the first segment;
4. warmup builds each (query, segment) graph with that segment's window.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 PYTHONPATH=$PWD python -m pytest \\
        test/vllm_neuron/worker/test_prompt_window.py
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MLAAttentionSpec
from vllm.v1.worker.block_table import MultiGroupBlockTable

from vllm_neuron.model.glm5_next.factory import Glm5NextForConditionalGeneration
from vllm_neuron.utils.bucket_utils import prefill_window_tokens
from vllm_neuron.vllm.worker import neuron_model_runner, neuron_worker
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

from test.vllm_neuron.worker import test_independent_prefill_buckets as buckets

pytestmark = [pytest.mark.fast]

PAGE = 128
QUERY = 1024
#: The standard line after this change, and the line it replaces.
STANDARD = {"kv_segment_size_buckets": [1024, 2048, 4096], "num_batched_tokens_buckets": [QUERY]}
OLD_STANDARD = {"kv_segment_size_buckets": [1024], "num_batched_tokens_buckets": [QUERY]}
#: The bs=64 @ 8k line as served, and with a 1k segment for 1k prompts.
BS64 = {"kv_segment_size_buckets": [8192], "num_batched_tokens_buckets": [QUERY]}
BS64_TWO = {"kv_segment_size_buckets": [1024, 8192], "num_batched_tokens_buckets": [QUERY]}
REQUEST = "prompt-window-request"


def _started(monkeypatch, knobs, *, max_model_len=4096):
    config = buckets._config(
        monkeypatch, Glm5NextForConditionalGeneration, knobs, max_model_len=max_model_len
    )
    return buckets._runner(monkeypatch, config)


def _targets(monkeypatch, runner):
    worker = NeuronWorker.__new__(NeuronWorker)
    worker.model_runner = runner
    warmed = []
    monkeypatch.setattr(
        runner, "warmup_prefill", lambda query, segment: warmed.append((query, segment))
    )
    monkeypatch.setattr(
        neuron_worker, "run_warmup_on_all_ranks", lambda phase, bucket, work: work()
    )
    worker._warmup_prefill()
    return worker._prefill_compile_targets(), warmed


# ── startup: the list, the graphs, the coverage rule ────────────────────────────────


def test_the_standard_line_resolves_three_segments_and_compiles_three_prefill_graphs(
    monkeypatch,
):
    runner = _started(monkeypatch, STANDARD)
    assert runner.neuron_config.kv_segment_size_buckets == [1024, 2048, 4096]
    assert runner.neuron_config.num_batched_tokens_buckets == [QUERY]
    assert runner.max_num_batched_tokens == QUERY
    targets, warmed = _targets(monkeypatch, runner)
    assert targets == [(QUERY, 1024), (QUERY, 2048), (QUERY, 4096)]
    assert warmed == targets


def test_the_old_standard_line_is_refused_at_startup_naming_the_window(monkeypatch):
    """[1024] + [1024] at max_model_len 4096 would serve 2048 tokens: no such server starts."""
    with pytest.raises(ValueError) as refused:
        _started(monkeypatch, OLD_STANDARD)
    message = str(refused.value)
    assert "prefill window" in message and "2048" in message and "4096" in message, message


def test_the_old_standard_line_covers_a_2048_token_model(monkeypatch):
    runner = _started(monkeypatch, OLD_STANDARD, max_model_len=2048)
    assert runner.neuron_config.kv_segment_size_buckets == [1024]


@pytest.mark.parametrize(
    "knobs, segments",
    [
        ({}, [1024, 4096]),  # max_num_batched_tokens 1024 < 4096 auto-enables [1024]
        ({"num_batched_tokens_buckets": [128, 1024]}, [1024, 4096]),
    ],
)
def test_the_auto_enabled_list_is_completed_to_cover_max_model_len(
    monkeypatch, knobs, segments
):
    runner = _started(monkeypatch, knobs)
    assert runner.neuron_config.kv_segment_size_buckets == segments
    assert runner.max_num_batched_tokens == QUERY
    targets, _ = _targets(monkeypatch, runner)
    queries = runner.neuron_config.num_batched_tokens_buckets
    assert targets == [(query, segment) for segment in segments for query in queries]


def test_the_auto_enabled_list_is_left_alone_when_it_covers(monkeypatch):
    runner = _started(monkeypatch, {}, max_model_len=2048)
    assert runner.neuron_config.kv_segment_size_buckets == [1024]


@pytest.mark.parametrize(
    "knobs, targets",
    [
        (BS64, [(QUERY, 8192)]),
        (BS64_TWO, [(QUERY, 1024), (QUERY, 8192)]),
    ],
)
def test_the_bs64_line_starts_with_one_or_two_segments(monkeypatch, knobs, targets):
    runner = _started(monkeypatch, knobs, max_model_len=8192)
    assert _targets(monkeypatch, runner)[0] == targets


@pytest.mark.parametrize(
    "knobs, max_model_len",
    [(STANDARD, 4096), ({}, 4096), (BS64, 8192), (BS64_TWO, 8192), (OLD_STANDARD, 2048)],
)
def test_every_line_that_starts_has_a_window_of_at_least_max_model_len(
    monkeypatch, knobs, max_model_len
):
    """What admission relies on: a started runner serves any prompt vLLM admits."""
    runner = _started(monkeypatch, knobs, max_model_len=max_model_len)
    window = prefill_window_tokens(
        runner.neuron_config.kv_segment_size_buckets,
        runner.neuron_config.num_batched_tokens_buckets,
        PAGE,
    )
    assert window >= max_model_len, (window, max_model_len)


# ── the choice per request, through the runner's own metadata builder ───────────────


def _group(names):
    return SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(
                list(names),
                MLAAttentionSpec(
                    block_size=PAGE,
                    num_kv_heads=1,
                    head_size=128,
                    dtype=torch.bfloat16,
                    sliding_window=None,
                    attention_chunk_size=None,
                ),
            )
        ]
    )


def _metadata_runner(names, segments, *, max_model_len=4096) -> NeuronModelRunner:
    """A runner shell carrying what ``_build_attention_metadata`` reads, with a real block table."""
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.max_model_len = max_model_len
    runner.max_num_reqs = 1
    runner.device = torch.device("cpu")
    runner.cp_world_size = 1
    runner._dcp_size = 1
    runner._is_synthetic_model = False
    runner.neuron_config = SimpleNamespace(
        kv_segment_size_buckets=list(segments),
        num_batched_tokens_buckets=[QUERY],
        decode_context_length_buckets=None,
        enable_structured_outputs=False,
    )
    runner.kv_cache_config = _group(names)
    runner.input_batch = SimpleNamespace(
        req_ids=[REQUEST],
        num_reqs=1,
        block_table=MultiGroupBlockTable(
            max_num_reqs=1,
            max_model_len=max_model_len,
            max_num_batched_tokens=QUERY,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[PAGE],
            kernel_block_sizes=[PAGE],
        ),
        num_computed_tokens_cpu_tensor=torch.zeros(1, dtype=torch.int32),
        num_tokens_no_spec=np.zeros(1, dtype=np.int32),
    )
    return runner


def _chunk_metadata(runner, *, cached: int, real: int, total: int) -> dict:
    """Allocate the chunk's pages the scheduler's way, then build the runner's metadata."""
    table = runner.input_batch.block_table[0]
    held = int(table.num_blocks_per_row[0])
    pages = -(-(cached + real) // PAGE)
    # Page 0 is vLLM's null block, so a request's pages start at 1.
    table.append_row(list(range(1 + held, 1 + pages)), 0)
    runner.input_batch.num_computed_tokens_cpu_tensor[0] = cached
    runner.input_batch.num_tokens_no_spec[0] = total
    return runner._build_attention_metadata(1, QUERY, QUERY, 0, cached, host_only=True)


def _chunks(prompt: int):
    return [(cached, min(QUERY, prompt - cached)) for cached in range(0, prompt, QUERY)]


def _window_pages(segment: int, max_model_len: int = 4096) -> int:
    """The converter's window for a one-request prefill chunk on ``segment``."""
    return min(max_model_len // PAGE, -(-(segment + QUERY) // PAGE))


@pytest.mark.parametrize(
    "prompt, segment", [(700, 1024), (1500, 2048), (3000, 4096), (4095, 4096)]
)
def test_every_chunk_of_a_request_carries_its_segment_and_a_window_that_holds_it(
    prompt, segment, caplog
):
    names = ["layers.0.self_attn", "layers.1.self_attn"]
    runner = _metadata_runner(names, STANDARD["kv_segment_size_buckets"])
    chosen = []
    with caplog.at_level(logging.INFO, logger=neuron_model_runner.__name__):
        for cached, real in _chunks(prompt):
            metadata = _chunk_metadata(runner, cached=cached, real=real, total=prompt)
            assert sorted(metadata) == names
            chosen.append({entry["kv_segment_size"] for entry in metadata.values()})
            # The chunk's pages fit the window the converter will size from the segment.
            assert -(-(cached + real) // PAGE) <= _window_pages(segment)
    assert chosen == [{segment}] * len(_chunks(prompt))
    logged = [record.getMessage() for record in caplog.records if "KV segment" in record.getMessage()]
    assert logged, caplog.text
    assert all(f"KV segment {segment}" in line and REQUEST in line for line in logged), logged


def test_a_request_longer_than_every_segment_takes_the_largest():
    """[1024, 2048] at max_model_len 3072: a 3000-token request takes 2048, whose window
    (2048 + 1024 = 3072) the validator proved covers the model."""
    runner = _metadata_runner(["layers.0.self_attn"], [1024, 2048], max_model_len=3072)
    for cached, real in _chunks(3000):
        metadata = _chunk_metadata(runner, cached=cached, real=real, total=3000)
        assert {entry["kv_segment_size"] for entry in metadata.values()} == {2048}
        assert -(-(cached + real) // PAGE) <= _window_pages(2048, 3072)


def test_a_decode_step_keeps_the_first_segment():
    """Its warmup graph was built with it; the decode converter does not read it."""
    runner = _metadata_runner(["layers.0.self_attn"], STANDARD["kv_segment_size_buckets"])
    _chunk_metadata(runner, cached=0, real=QUERY, total=3000)
    runner.input_batch.num_computed_tokens_cpu_tensor[0] = 3000
    decode = runner._build_attention_metadata(1, 1, 1, 0, 3000, host_only=True)
    assert {entry["kv_segment_size"] for entry in decode.values()} == {1024}


def test_a_batch_without_a_request_length_takes_the_chunks_padded_end():
    """No ``num_tokens_no_spec``: the segment holds cached + the padded chunk, so the
    window still holds the chunk (seg + query >= cached + query)."""
    runner = _metadata_runner(["layers.0.self_attn"], STANDARD["kv_segment_size_buckets"])
    del runner.input_batch.num_tokens_no_spec
    table = runner.input_batch.block_table[0]
    table.append_row(list(range(1, 1 + 24)), 0)
    runner.input_batch.num_computed_tokens_cpu_tensor[0] = 2048
    metadata = runner._build_attention_metadata(1, QUERY, QUERY, 0, 2048, host_only=True)
    assert {entry["kv_segment_size"] for entry in metadata.values()} == {4096}


def test_a_runner_without_segments_carries_zero():
    runner = _metadata_runner(["layers.0.self_attn"], [])
    runner.neuron_config.kv_segment_size_buckets = None
    metadata = _chunk_metadata(runner, cached=0, real=QUERY, total=QUERY)
    assert {entry["kv_segment_size"] for entry in metadata.values()} == {0}


# ── warmup: one graph per segment, built with that segment ──────────────────────────


def _warmup_runner(segments) -> NeuronModelRunner:
    runner = _metadata_runner(["layers.0.self_attn"], segments)
    runner.drafter = None
    return runner


@pytest.mark.parametrize("segment", [1024, 2048, 4096])
def test_warmup_metadata_carries_the_segment_it_is_built_for(segment):
    runner = _warmup_runner(STANDARD["kv_segment_size_buckets"])
    metadata = runner._build_warmup_attention_metadata(
        QUERY, 1, cached_seq_len=0, decode_token_threshold=1, kv_segment_size=segment
    )
    assert {entry["kv_segment_size"] for entry in metadata.values()} == {segment}


def test_warmup_metadata_defaults_to_the_first_segment():
    runner = _warmup_runner(STANDARD["kv_segment_size_buckets"])
    metadata = runner._build_warmup_attention_metadata(
        QUERY, 1, cached_seq_len=0, decode_token_threshold=1
    )
    assert {entry["kv_segment_size"] for entry in metadata.values()} == {1024}


def test_prefill_synthetic_inputs_carry_the_segment(monkeypatch):
    runner = _warmup_runner(STANDARD["kv_segment_size_buckets"])
    runner.uses_mrope = False
    runner.supports_mm_inputs = False
    runner.enable_prompt_embeds = False
    runner.rank_tensor = torch.zeros(1, dtype=torch.int32)
    runner.speculative_config = None
    runner.vision_neuron_config = None
    for segment in (1024, 2048, 4096):
        kwargs = runner._build_prefill_synthetic_inputs(QUERY, segment)
        assert {entry["kv_segment_size"] for entry in kwargs["attn_metadata"].values()} == {
            segment
        }
