# SPDX-License-Identifier: Apache-2.0
"""Step operands share uploads while each layer keeps its own mutable state."""

from unittest.mock import patch

import pytest
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

pytestmark = [pytest.mark.fast, pytest.mark.forked]


def _stack(linear=34, sparse=11):
    banks, side, geometries = [], [], []
    for index in range(linear):
        banks.append({
            "name": f"linear.{index}",
            "family": "linear_attn",
            "state_slots": 2,
            "conv_state": torch.full((2, 2, 3), float(index)),
            "recurrent_state": torch.full((2, 2, 3), float(index)),
        })
        side.append({})
        geometries.append({"state_slot": 1})
    for index in range(sparse):
        banks.append({
            "name": f"sparse.{index}",
            "family": "self_attn",
            "block_size": 4,
            "latent_cache": torch.full((4, 4, 4), float(index)),
        })
        side.append({
            "pool_cache": torch.full((2, 8, 2), float(index)),
            "tail": torch.full((2, 2, 4, 2), float(index)),
        })
        # Different layers can address different physical pages. Only operands
        # independent of the page allocation may be shared.
        geometries.append({
            "state_slot": 1,
            "block_ids": [index % 4],
            "page_size": 4,
            "window_blocks": 4,
        })
    return banks, side, geometries


def _build(banks, side, geometries, **overrides):
    options = {
        "geometries": geometries,
        "is_prefill": True,
        "tokens": 8,
        "start_position": 0,
        "softmax_scale": 0.25,
        "max_seq_len": 16,
        "index_kpool": 4,
        "real_tokens": 3,
    }
    options.update(overrides)
    return NeuronModelRunner._glm5next_layer_carriers(banks, side, **options)


def test_full_stack_uploads_step_operands_once_without_device_stack():
    banks, side, geometries = _stack()
    moves = []
    original_to = torch.Tensor.to

    def record_move(tensor, *args, **kwargs):
        moves.append((tensor.shape, tensor.dtype))
        return original_to(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "to", record_move), patch.object(
        torch, "stack", side_effect=AssertionError("request axis must be built on host")
    ):
        carriers = _build(banks, side, geometries)

    # 3 KDA operands, 4 DSA step operands, and 2 page-dependent operands
    # per DSA layer. This count does not grow with the 34 recurrent layers.
    assert len(moves) == 3 + 4 + 2 * 11
    assert torch.equal(carriers[0]["real_tokens"], torch.tensor([[3]], dtype=torch.int32))
    assert torch.equal(
        carriers[0]["row_mask"].flatten(),
        torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32),
    )
    assert torch.equal(
        carriers[34]["seq_lens"], torch.arange(1, 9, dtype=torch.int32)
    )
    assert carriers[34]["prefill_end_position"].item() == 3
    assert (carriers[34]["slot_mapping"] == -1).all()
    for index, carrier in enumerate(carriers[:34]):
        for key in ("start_position", "real_tokens", "row_mask"):
            assert carrier[key] is carriers[0][key]
        assert carrier["recurrent_state"][0].data_ptr() == banks[index]["recurrent_state"][1].data_ptr()
    for index, carrier in enumerate(carriers[34:]):
        for key in ("seq_lens", "start_position", "slot_mapping", "prefill_end_position"):
            assert carrier[key] is carriers[34][key]
        assert carrier["latent_cache"] is banks[34 + index]["latent_cache"]
        assert carrier["prefill_tail"].data_ptr() == side[34 + index]["tail"][1].data_ptr()
        expected_slot = (index % 4) * 4
        assert carrier["latent_slots"].tolist() == [
            expected_slot, expected_slot + 1, *([expected_slot + 2] * 6)
        ]
        assert carrier["block_table_row"].flatten().tolist() == [index % 4, -1, -1, -1]


def test_next_step_builds_fresh_values_and_keeps_the_previous_step_intact():
    banks, side, geometries = _stack(linear=2, sparse=2)
    first = _build(banks, side, geometries)
    second = _build(
        banks, side, geometries, is_prefill=False, tokens=1, start_position=3,
        real_tokens=1,
    )
    assert first[0]["start_position"].tolist() == [0]
    assert second[0]["start_position"].tolist() == [3]
    assert second[0]["row_mask"].shape == (1, 1, 1)
    assert second[2]["seq_lens"].tolist() == [4]
    assert second[2]["position"].item() == 3
    assert "prefill_tail" not in second[2]
    for key in ("start_position", "real_tokens", "row_mask"):
        assert first[0][key] is not second[0][key]


def test_multiple_request_axis_preserves_positions_and_state_slots():
    banks, side, geometries = _stack(linear=3, sparse=0)
    for geometry in geometries:
        geometry["state_slots"] = [1, 0]
    carriers = _build(
        banks, side, geometries, is_prefill=False, tokens=2, real_tokens=2,
        requests=2, request_starts=[7, 12], request_real_tokens=[1, 1],
    )
    assert carriers[0]["start_position"].tolist() == [7, 12]
    assert carriers[0]["real_tokens"].tolist() == [[1], [1]]
    assert carriers[0]["row_mask"].tolist() == [[[1.0]], [[1.0]]]
    carriers[1]["recurrent_state"][0].fill_(19)
    assert (banks[1]["recurrent_state"][1] == 19).all()
    assert (banks[1]["recurrent_state"][0] == 1).all()
    assert (banks[0]["recurrent_state"] == 0).all()


def test_expert_rank_operand_reuses_only_the_same_rank_and_device(monkeypatch):
    from vllm_neuron.parallel import neuron_parallel_state

    monkeypatch.setattr(neuron_parallel_state, "get_neuron_ep_degree", lambda: 1)
    runner = object.__new__(NeuronModelRunner)
    cpu = torch.device("cpu")
    first = runner._glm5next_parallel_kwargs(device=cpu)
    second = runner._glm5next_parallel_kwargs(device=cpu)
    assert first["expert_parallel_rank"] is second["expert_parallel_rank"]
    assert first["expert_parallel_rank"].tolist() == [0]
    meta = runner._glm5next_parallel_kwargs(device=torch.device("meta"))
    assert meta["expert_parallel_rank"].device.type == "meta"
    assert meta["expert_parallel_rank"] is not first["expert_parallel_rank"]
