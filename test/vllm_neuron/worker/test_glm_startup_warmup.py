# SPDX-License-Identifier: Apache-2.0
"""Actual synthetic builders feed bounded, unowned GLM request-state views."""

from types import SimpleNamespace

import pytest
import torch

from test.vllm_neuron.model.glm5_next.test_request_keyed_state import (
    DECLARED_PAGE_SIZE,
    _banks,
    _runner,
)


def _startup(capacity):
    banks = _banks()
    recurrent = banks[1]
    recurrent["state_slots"] = capacity
    for key in ("conv_state", "recurrent_state"):
        held = recurrent[key]
        recurrent[key] = held.new_zeros((capacity, *held.shape[1:]))
    runner = _runner(banks, request_ids=[])
    runner.max_num_reqs = capacity
    runner.device = torch.device("cpu")
    runner.neuron_config = SimpleNamespace(
        num_batched_tokens_buckets=None,
        decode_context_length_buckets=None,
        kv_segment_size_buckets=None,
        enable_structured_outputs=False,
    )
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1)
    )
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[bank["name"]],
                kv_cache_spec=SimpleNamespace(block_size=DECLARED_PAGE_SIZE),
            )
            for bank in banks
        ]
    )
    runner.requests = {}
    runner.drafter = None
    runner.speculative_config = None
    runner.uses_mrope = False
    runner.enable_prompt_embeds = False
    runner.supports_mm_inputs = False
    runner.rank_tensor = torch.tensor([0], dtype=torch.int64)
    runner._dcp_size = 1
    runner.cp_world_size = 1
    return runner, banks


@pytest.mark.parametrize("capacity", [1, 2, 8])
def test_actual_decode_builder_produces_distinct_unowned_views(capacity):
    runner, banks = _startup(capacity)
    generic = runner._build_decode_synthetic_inputs(
        capacity, context_len=1, decode_token_threshold=1
    )
    converted = runner._glm5next_model_kwargs(generic)
    assert "_glm5next_synthetic" not in converted
    sparse, recurrent = converted["layer_carriers"]
    assert converted["input_ids"].shape == (capacity,)
    assert len(recurrent["conv_state"]) == capacity
    assert len({part.data_ptr() for part in recurrent["conv_state"]}) == capacity
    assert (
        len({part.untyped_storage().data_ptr() for part in recurrent["conv_state"]})
        == 1
    )
    assert recurrent["start_position"].tolist() == [0] * capacity
    for index in range(capacity):
        table = (
            sparse["block_table_row"]
            if capacity == 1
            else sparse["block_table_row"][index]
        )
        slots = (
            sparse["latent_slots"] if capacity == 1 else sparse["latent_slots"][index]
        )
        assert (
            table.shape
            == generic["attn_metadata"][banks[0]["name"]]["host_block_table"][index]
            .reshape(-1, 1)
            .shape
        )
        assert table[0, 0].item() == index
        assert torch.all(table[1:] == -1)
        assert slots.tolist() == [index * DECLARED_PAGE_SIZE]
        assert slots.item() < banks[0]["latent_cache"].shape[0]
        assert (
            recurrent["conv_state"][index].data_ptr()
            == banks[1]["conv_state"][index].data_ptr()
        )
    assert runner._glm5next_request_slot_table == {}
    assert runner._glm5next_side_cache_positions == {}


def test_actual_prefill_builder_uses_enough_distinct_pages_for_one_request():
    runner, banks = _startup(2)
    tokens = DECLARED_PAGE_SIZE + 3
    generic = runner._build_prefill_synthetic_inputs(tokens, 0)
    converted = runner._glm5next_model_kwargs(generic)
    sparse, recurrent = converted["layer_carriers"]
    assert sparse["block_table_row"][:2, 0].tolist() == [0, 1]
    assert torch.all(sparse["block_table_row"][2:] == -1)
    assert sparse["latent_slots"].tolist() == list(range(tokens))
    assert (
        sparse["prefill_tail"].data_ptr()
        == runner._glm5next_side_cache_set[0]["tail"][0].data_ptr()
    )
    assert len(recurrent["conv_state"]) == 1
    assert runner._glm5next_request_slot_table == {}
    assert runner._glm5next_side_cache_positions == {}


@pytest.mark.parametrize(
    "owner_source", ["slot_table", "retained_requests", "input_batch"]
)
@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_live_request_refuses_before_storage_or_ownership_changes(owner_source, phase):
    runner, banks = _startup(2)
    side = runner._glm5next_live_side_caches(banks)
    tensors = [
        banks[0]["latent_cache"],
        banks[1]["conv_state"],
        banks[1]["recurrent_state"],
        side[0]["pool_cache"],
        side[0]["tail"],
    ]
    for index, tensor in enumerate(tensors):
        tensor.fill_(index + 1)
    values = [tensor.clone() for tensor in tensors]
    pointers = [tensor.data_ptr() for tensor in tensors]
    if owner_source == "slot_table":
        runner._glm5next_request_slot_table = {"held": 1}
    elif owner_source == "retained_requests":
        runner.requests = {"held": object()}
    else:
        runner.input_batch.req_ids = ["held"]
    runner._glm5next_side_cache_positions = {1: 9}
    table = dict(runner._glm5next_request_slot_table)
    positions = dict(runner._glm5next_side_cache_positions)
    generic = (
        runner._build_decode_synthetic_inputs(2, context_len=1)
        if phase == "decode"
        else runner._build_prefill_synthetic_inputs(DECLARED_PAGE_SIZE + 3, 0)
    )
    assert generic["_glm5next_synthetic"] is True
    with pytest.raises(ValueError, match="startup without live requests"):
        runner._glm5next_model_kwargs(generic)
    assert runner._glm5next_side_cache_set is side
    assert runner._glm5next_request_slot_table == table
    assert runner._glm5next_side_cache_positions == positions
    for tensor, before, pointer in zip(tensors, values, pointers):
        assert tensor.data_ptr() == pointer
        assert torch.equal(tensor, before)


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_generic_models_do_not_gain_a_synthetic_marker(phase):
    runner, _ = _startup(2)
    del runner.model.glm5next_layer_banks
    runner.input_batch.req_ids = ["held"]
    generic = (
        runner._build_decode_synthetic_inputs(2, context_len=1)
        if phase == "decode"
        else runner._build_prefill_synthetic_inputs(DECLARED_PAGE_SIZE + 3, 0)
    )
    assert "_glm5next_synthetic" not in generic
    assert runner._glm5next_model_kwargs(generic) is generic


def test_excess_synthetic_requests_refuse_before_cache_allocation():
    runner, _ = _startup(2)
    generic = runner._build_decode_synthetic_inputs(3, context_len=1)
    with pytest.raises(ValueError, match="3 requests against a capacity of 2"):
        runner._glm5next_model_kwargs(generic)
    assert not hasattr(runner, "_glm5next_side_cache_set")


def test_dp1_dummy_never_builds_or_dispatches_a_model_call():
    runner, _ = _startup(2)
    runner._build_decode_synthetic_inputs = lambda *args, **kwargs: pytest.fail(
        "DP1 dummy built inputs"
    )
    runner._glm5next_model_kwargs = lambda *args, **kwargs: pytest.fail(
        "DP1 dummy converted inputs"
    )
    runner.execute_dummy_batch()
