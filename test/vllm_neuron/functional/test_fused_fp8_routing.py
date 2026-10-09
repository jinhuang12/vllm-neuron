# SPDX-License-Identifier: Apache-2.0
"""Fused expert routing with no routed row."""

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.moe.fused_fp8_pack import pack_experts
from vllm_neuron.functional.moe.moe_fused_fp8 import (
    moe_fused_fp8_kernel,
)


def _case(row_ids):
    torch.manual_seed(20261003)
    experts, hidden, intermediate = 3, 256, 128
    x = torch.randn(8, hidden).to(torch.bfloat16)
    x[-1].zero_()
    gate_up = (torch.randn(experts, hidden, 2 * intermediate) / 16).to(
        torch.float8_e4m3fn
    )
    down = (torch.randn(experts, intermediate, hidden) / 16).to(
        torch.float8_e4m3fn
    )
    gu_scale = torch.rand(experts, hidden // 128, 2, intermediate // 128) + 0.5
    down_scale = torch.rand(experts, intermediate // 128, hidden // 128) + 0.5
    packed = pack_experts(gate_up, down, gu_scale, down_scale)
    affinity = torch.rand(8, experts)
    affinity[-1].zero_()
    bounds = torch.tensor([0.6, -0.4, 0.5]).repeat(128, 1)
    ids = torch.tensor(row_ids, dtype=torch.int32)
    expert_ids = torch.tensor([[2], [0], [1]], dtype=torch.int32)
    return (x, packed, ids, expert_ids, affinity, bounds), (
        gate_up.float(), down.float(), gu_scale, down_scale
    )


def test_all_empty_routing_returns_initialized_zeros(monkeypatch):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "1")
    inputs, _ = _case([[-1] * 33 for _ in range(3)])
    x, packed, row_ids, expert_ids, affinity, bounds = inputs
    output = wrap_nki(moe_fused_fp8_kernel)[2](
        x, packed.weights, packed.scales, row_ids, expert_ids,
        affinity.reshape(-1, 1), bounds, BLOCK_M=33
    )
    assert output.shape == (3, 33, 256)
    assert torch.equal(output, torch.zeros_like(output))
