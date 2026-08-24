# SPDX-License-Identifier: Apache-2.0
"""Product tests for row-FP8 weights on the supported Trn2 MoE kernel."""

from __future__ import annotations

import importlib
import os

import pytest
import torch

import vllm_neuron.model.glm_moe_dsa.moe as moe_module
from vllm_neuron.model.glm_moe_dsa.block_fp8 import (
    RowFP8Linear,
    dequantize_row_fp8,
    quantize_block_fp8_to_row,
)
from vllm_neuron.model.glm_moe_dsa.moe import GlmMoeDsaRoutedExperts


def _small_fp8_experts() -> GlmMoeDsaRoutedExperts:
    experts = GlmMoeDsaRoutedExperts(
        hidden_size=256,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        expert_parallel_size=1,
        fp8_weights=True,
        device="cpu",
    )
    for expert_id, expert in enumerate(experts.experts):
        value = 0.125 * (expert_id + 1)
        for projection in (expert.gate_proj, expert.up_proj, expert.down_proj):
            assert isinstance(projection, RowFP8Linear)
            projection.weight.data.fill_(value)
            projection.weight_scale_inv.data.fill_(2.0)
    return experts


def test_block_fp8_to_row_fp8_preserves_zero_and_bounded_error() -> None:
    torch.manual_seed(17)
    weight = (torch.randn(256, 384) * 8).to(torch.float8_e4m3fn)
    weight[0].zero_()
    block_scale = torch.rand(2, 3) * 0.02 + 0.001

    row_weight, row_scale = quantize_block_fp8_to_row(weight, block_scale)
    actual = dequantize_row_fp8(row_weight, row_scale)
    rows = torch.arange(256) // 128
    columns = torch.arange(384) // 128
    expected = weight.float() * block_scale[rows[:, None], columns[None, :]]

    assert row_weight.dtype is torch.float8_e4m3fn
    assert row_scale.shape == (256,)
    assert row_scale.dtype is torch.float32
    assert torch.equal(actual[0], torch.zeros_like(actual[0]))
    torch.testing.assert_close(actual, expected, rtol=0.13, atol=0.002)


def test_builds_exact_fp8_kernel_and_scale_layouts() -> None:
    experts = _small_fp8_experts()

    gate_up, down = experts._kernel_weights()
    gate_up_scale, down_scale = experts._row_fp8_kernel_scales()

    assert gate_up.shape == (4, 256, 2, 128)
    assert down.shape == (4, 128, 256)
    assert gate_up.dtype is torch.float8_e4m3fn
    assert down.dtype is torch.float8_e4m3fn
    assert gate_up_scale.shape == (4, 2, 128)
    assert down_scale.shape == (4, 256)
    assert torch.equal(gate_up_scale, torch.full_like(gate_up_scale, 2.0))
    assert torch.equal(down_scale, torch.full_like(down_scale, 2.0))
    assert experts.expert_parallel_rank_tensor.tolist() == [[0]]
    assert "expert_parallel_rank_tensor" not in experts.state_dict()


@pytest.mark.parametrize("is_decode", [False, True], ids=["prefill", "decode"])
def test_row_fp8_uses_supported_moe_tkg_for_both_phases(
    monkeypatch, is_decode: bool
) -> None:
    experts = _small_fp8_experts()
    token_count = 1 if is_decode else 3
    hidden = torch.ones((token_count, 256), dtype=torch.bfloat16)
    affinities = torch.full((token_count, 4), 0.25, dtype=torch.float32)
    selected = torch.zeros((token_count, 2), dtype=torch.int32)
    calls = []

    monkeypatch.delenv("GLM_ENABLE_EXPERIMENTAL_SELECTIVE_FP8_MOE", raising=False)
    monkeypatch.setattr(moe_module, "can_run_kernel", lambda _: True)
    monkeypatch.setattr(
        moe_module,
        "selective_block_fp8_moe_nki",
        lambda *args, **kwargs: pytest.fail("experimental kernel was selected"),
    )

    def fake_moe_tkg(**kwargs):
        calls.append(kwargs)
        return torch.zeros_like(kwargs["hidden_input"])

    monkeypatch.setattr(moe_module.NF, "moe_tkg", fake_moe_tkg)

    result = experts(hidden, affinities, selected, is_decode=is_decode)

    assert torch.equal(result, torch.zeros_like(hidden))
    assert len(calls) == 1
    assert calls[0]["expert_gate_up_weights"].shape == (4, 256, 2, 128)
    assert calls[0]["expert_down_weights"].shape == (4, 128, 256)
    assert calls[0]["expert_gate_up_weights"].dtype is torch.float8_e4m3fn
    assert calls[0]["expert_down_weights"].dtype is torch.float8_e4m3fn
    assert calls[0]["expert_gate_up_weights_scale"].shape == (4, 2, 128)
    assert calls[0]["expert_down_weights_scale"].shape == (4, 256)
    assert torch.equal(calls[0]["expert_index"], selected)
    assert calls[0]["rank_id"] is experts.expert_parallel_rank_tensor
    assert calls[0]["mask_unselected_experts"] is True


def test_row_fp8_prefill_is_chunked_to_kernel_token_limit(monkeypatch) -> None:
    experts = _small_fp8_experts()
    hidden = torch.ones((257, 256), dtype=torch.bfloat16)
    affinities = torch.full((257, 4), 0.25, dtype=torch.float32)
    selected = torch.arange(514, dtype=torch.int32).reshape(257, 2) % 4
    calls = []

    monkeypatch.setattr(moe_module, "can_run_kernel", lambda _: True)

    def fake_moe_tkg(**kwargs):
        calls.append(kwargs)
        return torch.zeros_like(kwargs["hidden_input"])

    monkeypatch.setattr(moe_module.NF, "moe_tkg", fake_moe_tkg)

    result = experts(hidden, affinities, selected, is_decode=False)

    assert result.shape == hidden.shape
    assert [call["hidden_input"].shape[0] for call in calls] == [128, 128, 1]
    assert torch.equal(calls[0]["expert_index"], selected[:128])
    assert torch.equal(calls[1]["expert_index"], selected[128:256])
    assert torch.equal(calls[2]["expert_index"], selected[256:])


@pytest.mark.parametrize(
    ("gate_scale", "down_scale", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_moe_tkg_accepts_fp8_only_with_both_scales(
    monkeypatch, gate_scale: bool, down_scale: bool, expected: bool
) -> None:
    functional = importlib.import_module("vllm_neuron.functional.moe.moe_tkg")
    monkeypatch.setattr(functional, "can_run_kernel", lambda _: True)
    gate_up = torch.empty((4, 256, 2, 128), dtype=torch.float8_e4m3fn)
    down = torch.empty((4, 128, 256), dtype=torch.float8_e4m3fn)

    assert (
        functional._can_use_kernel(
            hidden_input=torch.empty((1, 256), dtype=torch.bfloat16),
            expert_gate_up_weights=gate_up,
            expert_down_weights=down,
            expert_gate_up_weights_scale=(
                torch.ones((4, 2, 128)) if gate_scale else None
            ),
            expert_down_weights_scale=(torch.ones((4, 256)) if down_scale else None),
        )
        is expected
    )


@pytest.mark.skipif(
    os.getenv("GLM_ROW_FP8_MOE_HARDWARE") != "1",
    reason="explicit production-shape row-FP8 MoE hardware gate",
)
def test_neuron_row_fp8_moe_production_shape() -> None:
    expert_parallel_rank = int(os.getenv("GLM_ROW_FP8_EP_RANK", "0"))
    if not 0 <= expert_parallel_rank < 64:
        pytest.fail("GLM_ROW_FP8_EP_RANK must be in 0..63")
    experts = GlmMoeDsaRoutedExperts(
        hidden_size=6144,
        intermediate_size=2048,
        num_experts=256,
        top_k=8,
        expert_parallel_size=64,
        expert_parallel_rank=expert_parallel_rank,
        fp8_weights=True,
        device="cpu",
    ).eval()
    for expert in experts.experts:
        expert.gate_proj.weight.data.fill_(1.0)
        expert.up_proj.weight.data.fill_(1.0)
        expert.down_proj.weight.data.fill_(1.0)
        expert.gate_proj.weight_scale_inv.data.fill_(1.0 / 6144)
        expert.up_proj.weight_scale_inv.data.fill_(1.0 / 6144)
        expert.down_proj.weight_scale_inv.data.fill_(1.0 / 2048)

    hidden_cpu = torch.ones((1, 6144), dtype=torch.bfloat16)
    affinities_cpu = torch.zeros((1, 256), dtype=torch.float32)
    global_expert = expert_parallel_rank * 4
    affinities_cpu[0, global_expert] = 1.0
    selected_cpu = torch.tensor(
        [[global_expert] + [(global_expert + 4 + offset) % 256 for offset in range(7)]],
        dtype=torch.int32,
    )
    expected = experts.experts[0](hidden_cpu).float()

    experts = experts.to("neuron:0")
    compiled = torch.compile(
        experts,
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={"compiler_workdir": os.environ["GLM_ROW_FP8_MOE_COMPILE_DIR"]},
    )
    actual = (
        compiled(
            hidden_cpu.to("neuron:0"),
            affinities_cpu.to("neuron:0"),
            selected_cpu.to("neuron:0"),
            is_decode=True,
        )
        .cpu()
        .float()
    )

    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) == actual.numel()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
