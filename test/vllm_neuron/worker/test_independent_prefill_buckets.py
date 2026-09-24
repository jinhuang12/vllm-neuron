# SPDX-License-Identifier: Apache-2.0
"""Resolve independent GLM query buckets through the platform and CPU runner."""

import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.registry import ModelRegistry

from vllm_neuron.model.glm5_next.factory import Glm5NextForConditionalGeneration
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3.factory import Qwen3ForCausalLM
from vllm_neuron.utils.bucket_utils import validate_kv_segment_size_buckets
from vllm_neuron.vllm.core.scheduler import NeuronScheduler
from vllm_neuron.vllm.platform import NeuronPlatform
from vllm_neuron.vllm.worker import neuron_model_runner, neuron_worker
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker


CAPABILITY_FIELD = "_model_supports_independent_prefill_buckets"


def _config(monkeypatch, model_cls, neuron_config):
    """Keep model resolution real at the class boundary, without loading weights."""
    monkeypatch.setattr(
        ModelRegistry,
        "resolve_model_cls",
        lambda architectures, model_config: (model_cls, model_cls.__name__),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=[model_cls.__name__],
            runner_type="generate",
            uses_mrope=False,
            uses_xdrope_dim=0,
            hf_config=SimpleNamespace(vocab_size=256),
            max_model_len=4096,
            enable_prompt_embeds=False,
            get_inputs_embeds_size=lambda: 0,
        ),
        additional_config={"neuron_config": dict(neuron_config)},
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=1024,
            async_scheduling=False,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(block_size=128, enable_prefix_caching=True),
        speculative_config=None,
        kv_transfer_config=None,
    )
    NeuronPlatform._resolve_sampling_from_the_model_class(config)
    return config


def _runner(monkeypatch, config):
    # This test covers configuration and dispatch, before model/cache loading.
    monkeypatch.setattr(
        neuron_model_runner,
        "MULTIMODAL_REGISTRY",
        SimpleNamespace(supports_multimodal_inputs=lambda model_config: False),
    )
    monkeypatch.setattr(neuron_model_runner, "get_tp_group", lambda: None)
    return NeuronModelRunner(config, device=torch.device("cpu"))


def test_generic_segmented_kernel_still_requires_equal_query_buckets():
    with pytest.raises(ValueError, match="must match"):
        validate_kv_segment_size_buckets([1024], [128, 1024])
    assert validate_kv_segment_size_buckets([1024], [1024]) == [1024]


@pytest.mark.parametrize(
    "segments,match",
    [
        (None, "non-empty list"),
        ([], "non-empty list"),
        ((1024,), "non-empty list"),
        (["1024"], "must be an integer"),
        ([128], "not a supported segment size"),
        ([1024, 512], "strictly ascending"),
        ([512, 1024], "Only one segment size"),
    ],
)
def test_independent_queries_keep_all_segment_guards(segments, match):
    with pytest.raises(ValueError, match=match):
        validate_kv_segment_size_buckets(
            segments, [128, 1024], allow_independent_query_buckets=True
        )


def test_model_capability_survives_config_serialization():
    assert not NeuronConfig()._model_supports_independent_prefill_buckets
    config = NeuronConfig.from_dict(
        {CAPABILITY_FIELD: True, "on_device_sampling_config": None}
    )
    restored = NeuronConfig.from_dict(json.loads(json.dumps(dataclasses.asdict(config))))
    assert restored._model_supports_independent_prefill_buckets is True


@pytest.mark.parametrize("explicit_segments", [False, True])
def test_qwen_cannot_enable_independent_queries_from_user_config(
    monkeypatch, explicit_segments
):
    knobs = {CAPABILITY_FIELD: True, "num_batched_tokens_buckets": [128, 1024]}
    if explicit_segments:
        knobs["kv_segment_size_buckets"] = [1024]
    config = _config(monkeypatch, Qwen3ForCausalLM, knobs)
    # Qwen takes the sampler early return. It must still clear this field.
    assert config.additional_config["neuron_config"][CAPABILITY_FIELD] is False
    with pytest.raises(ValueError, match="must match"):
        _runner(monkeypatch, config)


@pytest.mark.parametrize("explicit_segments", [False, True])
def test_glm_resolves_independent_queries_and_warms_each_pair(
    monkeypatch, explicit_segments
):
    knobs = {"num_batched_tokens_buckets": [128, 1024]}
    if explicit_segments:
        knobs["kv_segment_size_buckets"] = [1024]
    config = _config(monkeypatch, Glm5NextForConditionalGeneration, knobs)
    runner = _runner(monkeypatch, config)
    assert runner.neuron_config._model_supports_independent_prefill_buckets is True
    assert runner.neuron_config.num_batched_tokens_buckets == [128, 1024]
    assert runner.neuron_config.kv_segment_size_buckets == [1024]
    assert runner.max_num_batched_tokens == 1024
    assert config.scheduler_config.max_num_batched_tokens == 1024
    assert runner.max_model_len == 4096

    worker = NeuronWorker.__new__(NeuronWorker)
    worker.model_runner = runner
    assert worker._prefill_compile_targets() == [(128, 1024), (1024, 1024)]
    warmed = []
    monkeypatch.setattr(
        runner, "warmup_prefill", lambda query, segment: warmed.append((query, segment))
    )
    monkeypatch.setattr(
        neuron_worker, "run_warmup_on_all_ranks", lambda phase, bucket, work: work()
    )
    worker._warmup_prefill()
    assert warmed == [(128, 1024), (1024, 1024)]

    scheduler = NeuronScheduler.__new__(NeuronScheduler)
    scheduler.num_batched_tokens_buckets = runner.neuron_config.num_batched_tokens_buckets
    for real_tokens, expected in [(1, 128), (5, 128), (128, 128), (129, 1024), (1024, 1024)]:
        assert scheduler._calculate_padded_count(real_tokens) == expected


@pytest.mark.parametrize("explicit_segments", [False, True])
def test_glm_automatic_query_buckets_are_unchanged(monkeypatch, explicit_segments):
    knobs = {"kv_segment_size_buckets": [1024]} if explicit_segments else {}
    config = _config(monkeypatch, Glm5NextForConditionalGeneration, knobs)
    runner = _runner(monkeypatch, config)
    assert runner.neuron_config.num_batched_tokens_buckets == [1024]
    assert runner.neuron_config.kv_segment_size_buckets == [1024]


@pytest.mark.parametrize("queries", [[1024, 128], [128, 512], [0, 1024], ["128", 1024]])
def test_glm_keeps_query_bucket_validation(monkeypatch, queries):
    config = _config(
        monkeypatch,
        Glm5NextForConditionalGeneration,
        {"num_batched_tokens_buckets": queries, "kv_segment_size_buckets": [1024]},
    )
    with pytest.raises(ValueError, match="num_batched_tokens_buckets"):
        _runner(monkeypatch, config)


def test_independent_queries_do_not_allow_an_unmaterialized_cached_prefix(monkeypatch):
    config = _config(
        monkeypatch,
        Glm5NextForConditionalGeneration,
        {"num_batched_tokens_buckets": [128, 1024], "kv_segment_size_buckets": [1024]},
    )
    runner = _runner(monkeypatch, config)
    runner._glm5next_side_cache_positions = {}
    with pytest.raises(ValueError, match="holds no sequence cursor"):
        runner._glm5next_position_arm(0, 1024, side_caches=[], is_prefill=True)


def _carrier_step(
    *, width=128, real_tokens=5, cached=0, requests=1, device="cpu", sparse=True
):
    """Build real runner carriers from small hybrid banks and host metadata."""
    banks = []
    if sparse:
        banks.append({
            "name": "model.layers.0.self_attn",
            "family": "self_attn",
            "block_size": 128,
            "latent_cache": torch.zeros((4096, 1, 8), device=device),
        })
    banks.append({
        "name": "model.layers.1.attention",
        "family": "linear_attn",
        "state_slots": 2,
        "conv_state": torch.zeros((2, 4, 6), device=device),
        "recurrent_state": torch.zeros((2, 2, 6, 6), device=device),
    })
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = SimpleNamespace(
        glm5next_layer_banks=banks,
        text_config=SimpleNamespace(
            index_kpool=4, index_head_dim=8,
            qk_nope_head_dim=8, qk_rope_head_dim=8,
        ),
    )
    runner.neuron_config = SimpleNamespace(num_batched_tokens_buckets=[128, 1024])
    runner.max_model_len = 4096
    runner.max_num_reqs = 2
    req_ids = [f"request-{i}" for i in range(requests)] if real_tokens is not None else []
    runner.input_batch = SimpleNamespace(req_ids=req_ids)
    runner._glm5next_request_tokens = (
        [real_tokens] * requests if real_tokens is not None else None
    )
    runner._glm5next_live_side_caches(banks)
    if cached:
        runner._glm5next_request_slot_table = dict(zip(req_ids, range(requests)))
        runner._glm5next_side_cache_positions = {i: cached for i in range(requests)}
    real = width if real_tokens is None else real_tokens
    used = (cached + real + 127) // 128
    rows = [list(range(i * 16, i * 16 + used)) + [0] * (32 - used)
            for i in range(requests)]
    entry = {
        "max_query_len": width,
        "decode_token_threshold": 1,
        "block_size": 128,
        "max_blocks_per_seq": 32,
        "kv_segment_size": 1024,
        "host_block_table": rows,
        "host_num_computed_tokens": [cached] * requests,
        # These tensors may be on meta. The converter must use only host values.
        "block_table_tensor": torch.tensor(rows, dtype=torch.int32, device=device),
        "cached_seq_len": torch.full((requests, 1), cached, device=device),
        "slot_mapping": torch.arange(width, device=device),
    }
    return runner, {
        "input_ids": torch.arange(1, width + 1, dtype=torch.int32, device=device),
        "positions": torch.arange(width, device=device),
        "rotary_position_ids": torch.arange(width, device=device),
        "attn_metadata": {bank["name"]: entry for bank in banks},
        "sampling_positions": torch.tensor([real - 1], device=device),
    }


@pytest.mark.parametrize("real_tokens,cached", [(5, 0), (128, 0), (5, 1024), (None, 0)])
def test_sparse_prefix_keeps_model_width_and_real_request_extent(real_tokens, cached):
    runner, kwargs = _carrier_step(real_tokens=real_tokens, cached=cached)
    metadata_map = kwargs["attn_metadata"]
    entries = {name: dict(entry) for name, entry in metadata_map.items()}
    input_ids = kwargs["input_ids"]
    original_ids = input_ids.clone()
    real = 128 if real_tokens is None else real_tokens
    if real_tokens is None:
        runner._glm5next_request_slot_table = {"other-request": 1}
        runner._glm5next_side_cache_positions = {1: 17}
        side = runner._glm5next_side_cache_set
        side[0]["tail"].fill_(7)
        before = side[0]["tail"].clone()
        pointer = side[0]["tail"].data_ptr()
        with pytest.raises(ValueError, match="startup without live requests"):
            runner._glm5next_model_kwargs(kwargs)
        assert runner._glm5next_request_slot_table == {"other-request": 1}
        assert runner._glm5next_side_cache_positions == {1: 17}
        assert runner._glm5next_side_cache_set is side
        assert side[0]["tail"].data_ptr() == pointer
        assert torch.equal(side[0]["tail"], before)
        assert kwargs["input_ids"] is input_ids
        assert torch.equal(input_ids, original_ids)
        return

    converted = runner._glm5next_model_kwargs(kwargs)

    assert kwargs["input_ids"] is input_ids
    assert torch.equal(input_ids, original_ids)
    assert converted["input_ids"].shape == (1024,)
    assert torch.equal(converted["input_ids"][:128], original_ids)
    assert torch.count_nonzero(converted["input_ids"][128:]) == 0
    assert converted["sampling_positions"] is kwargs["sampling_positions"]
    assert kwargs["attn_metadata"] is metadata_map
    for name, entry in metadata_map.items():
        assert entry.keys() == entries[name].keys()
        for key, value in entry.items():
            assert value is entries[name][key]
        assert entry["max_query_len"] == 128

    sparse, linear = converted["layer_carriers"]
    assert sparse["active_mla_query_rows"] == 128
    # The bank travels whole, and the window is what the block table names, so the
    # length this leg is sized by comes off the table's width.
    assert sparse["latent_cache"].shape[0] == 4096
    assert sparse["block_table_row"].shape == (16, 1)
    assert sparse["latent_slots"].shape == (1024,)
    assert sparse["seq_lens"].shape == (1024,)
    assert sparse["slot_mapping"].shape == (1024,)
    assert torch.all(sparse["slot_mapping"][real:] == -1)
    assert int(sparse["prefill_end_position"]) == cached + real
    assert "active_mla_query_rows" not in linear
    assert int(linear["real_tokens"].item()) == real
    assert linear["row_mask"].shape == (1, 1024, 1)
    assert torch.all(linear["row_mask"][:, :real])
    assert not torch.any(linear["row_mask"][:, real:])
    assert runner._glm5next_side_cache_positions == {0: cached + real}
    assert runner._glm5next_side_cache_cursor == cached + real


def test_sparse_startup_prefix_keeps_model_width_without_claiming_state():
    runner, kwargs = _carrier_step(real_tokens=None)
    converted = runner._glm5next_model_kwargs(kwargs)
    sparse, linear = converted["layer_carriers"]
    assert converted["input_ids"].shape == (1024,)
    assert torch.equal(converted["input_ids"][:128], kwargs["input_ids"])
    assert torch.count_nonzero(converted["input_ids"][128:]) == 0
    assert sparse["active_mla_query_rows"] == 128
    assert sparse["latent_slots"].shape == (1024,)
    assert int(sparse["prefill_end_position"]) == 128
    assert linear["row_mask"].shape == (1, 1024, 1)
    assert torch.all(linear["row_mask"][:, :128])
    assert not torch.any(linear["row_mask"][:, 128:])
    assert runner._glm5next_request_slot_table == {}
    assert runner._glm5next_side_cache_positions == {}
    assert not hasattr(runner, "_glm5next_side_cache_cursor")


def test_sparse_prefix_conversion_uses_host_metadata_during_capture():
    runner, kwargs = _carrier_step(real_tokens=None, device="meta")
    converted = runner._glm5next_model_kwargs(kwargs)
    sparse, linear = converted["layer_carriers"]
    assert converted["input_ids"].shape == (1024,)
    assert converted["input_ids"].device.type == "meta"
    assert sparse["active_mla_query_rows"] == 128
    assert sparse["latent_cache"].shape[0] == 4096
    assert sparse["block_table_row"].shape == (16, 1)
    assert linear["row_mask"].shape == (1, 1024, 1)
    assert runner._glm5next_side_cache_positions == {}


def test_sparse_prefix_ignores_metadata_for_unbound_layers():
    runner, kwargs = _carrier_step()
    unrelated = object()
    kwargs["attn_metadata"]["unbound-layer"] = unrelated
    converted = runner._glm5next_model_kwargs(kwargs)
    assert converted["input_ids"].shape == (1024,)
    assert converted["layer_carriers"][0]["active_mla_query_rows"] == 128
    assert kwargs["attn_metadata"]["unbound-layer"] is unrelated


@pytest.mark.parametrize("width", [1, 1024])
def test_decode_and_largest_query_keep_original_operands(width):
    runner, kwargs = _carrier_step(width=width, real_tokens=width)
    converted = runner._glm5next_model_kwargs(kwargs)
    assert converted["input_ids"] is kwargs["input_ids"]
    assert all("active_mla_query_rows" not in carrier
               for carrier in converted["layer_carriers"])
    sparse, linear = converted["layer_carriers"]
    assert linear["row_mask"].shape == (1, width, 1)
    assert runner._glm5next_side_cache_positions == {0: width}
    if width == 1:
        assert "prefill_end_position" not in sparse
        assert int(sparse["position"]) == 0
    else:
        assert sparse["latent_cache"].shape[0] == 4096
        assert sparse["block_table_row"].shape == (16, 1)
        assert int(sparse["prefill_end_position"]) == width


def test_direct_runner_without_query_config_keeps_original_width():
    runner, kwargs = _carrier_step()
    del runner.neuron_config
    converted = runner._glm5next_model_kwargs(kwargs)
    assert converted["input_ids"] is kwargs["input_ids"]
    sparse, linear = converted["layer_carriers"]
    assert "active_mla_query_rows" not in sparse
    assert sparse["latent_cache"].shape[0] == 4096
    # With no bucket configured the window is the nine blocks the request needs, so
    # the block-table row is sized by the table's own width.
    assert sparse["block_table_row"].shape == (9, 1)
    assert linear["row_mask"].shape == (1, 128, 1)


def test_recurrent_only_stack_keeps_original_query_width():
    runner, kwargs = _carrier_step(sparse=False)
    converted = runner._glm5next_model_kwargs(kwargs)
    assert converted["input_ids"] is kwargs["input_ids"]
    (linear,) = converted["layer_carriers"]
    assert "active_mla_query_rows" not in linear
    assert linear["row_mask"].shape == (1, 128, 1)


def test_generic_model_kwargs_stay_unchanged():
    runner, kwargs = _carrier_step()
    del runner.model.glm5next_layer_banks
    assert runner._glm5next_model_kwargs(kwargs) is kwargs


def test_sparse_prefix_rejects_real_count_beyond_selected_width():
    runner, kwargs = _carrier_step(real_tokens=129)
    with pytest.raises(ValueError, match="129 real token.*active MLA prefix of 128"):
        runner._glm5next_model_kwargs(kwargs)
    assert runner._glm5next_side_cache_positions == {}


def test_sparse_prefill_still_refuses_multiple_requests():
    runner, kwargs = _carrier_step(real_tokens=64, requests=2)
    with pytest.raises(ValueError, match="concurrent sparse prefill"):
        runner._glm5next_model_kwargs(kwargs)
    assert kwargs["input_ids"].shape == (128,)
    assert runner._glm5next_side_cache_positions == {}
