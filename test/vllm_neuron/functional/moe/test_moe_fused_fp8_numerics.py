# SPDX-License-Identifier: Apache-2.0
"""Tensor Engine operand forms of the fused FP8 expert kernel.

The simulator accepts PE
operand access patterns that the hardware does not: a stationary with two free
dimensions passes here and gives wrong products on Trn2 (records of the
5debd95 device run). A test therefore checks that every stationary
the kernel passes to ``nc_matmul`` has one free dimension.
"""

import pytest
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.moe import moe_fused_fp8 as _module
from vllm_neuron.functional.moe.fused_fp8_pack import pack_experts
from vllm_neuron.functional.moe.moe_fused_fp8 import moe_fused_fp8_kernel

def _stationary_shapes(monkeypatch):
    """Record the shape of every ``nc_matmul`` stationary the kernel issues."""
    shapes = []
    real = _module.nisa.nc_matmul

    def recording(*args, **kwargs):
        shapes.append(tuple(kwargs["stationary"].shape))
        return real(*args, **kwargs)

    monkeypatch.setattr(_module.nisa, "nc_matmul", recording)
    return shapes


@pytest.mark.parametrize("skip_padding", [True, False])
def test_every_stationary_has_one_free_dimension(skip_padding, monkeypatch):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "1")
    shapes = _stationary_shapes(monkeypatch)
    args, _ = _case(blocks=4, q=64, hidden=512, intermediate=512)
    wrap_nki(moe_fused_fp8_kernel)[2](*args, BLOCK_M=64, BLOCK_N=256, BLOCK_K=256,
                                      SKIP_PADDING=skip_padding)
    # Per routed slice: 2I/128 * H/128 gate/up and I/128 * H/128 down products.
    assert len(shapes) >= 3 * (512 // 128) * (512 // 128)
    assert {len(shape) for shape in shapes} == {2}
    assert all(shape[0] == 128 and shape[1] <= 128 for shape in shapes)


def _case(blocks, q, hidden, intermediate, experts=3, tokens=40, seed=20261008):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens + 1, hidden, generator=generator).to(torch.bfloat16)
    x[-1].zero_()
    gate_up = (torch.randn(experts, hidden, 2 * intermediate, generator=generator) / 16
               ).to(torch.float8_e4m3fn)
    down = (torch.randn(experts, intermediate, hidden, generator=generator) / 16
            ).to(torch.float8_e4m3fn)
    nh, ni = hidden // 128, intermediate // 128
    gu_scale = torch.rand(experts, nh, 2, ni, generator=generator) + 0.5
    down_scale = torch.rand(experts, ni, nh, generator=generator) + 0.5
    packed = pack_experts(gate_up, down, gu_scale, down_scale)
    affinity = torch.rand(tokens + 1, experts, generator=generator)
    affinity[-1].zero_()
    bounds = torch.tensor([0.6, -0.4, 0.5]).repeat(128, 1)
    # Block b holds a b-dependent number of rows with holes; block 1 is empty.
    ids = torch.full((blocks, q), -1, dtype=torch.int32)
    for block in range(blocks):
        count = 0 if block == 1 else (7 * block + 5) % q + 1
        rows = torch.randperm(q, generator=generator)[:count]
        ids[block, rows] = torch.randint(0, tokens, (count,), generator=generator,
                                         dtype=torch.int32)
    expert_ids = torch.tensor([[b % experts] for b in range(blocks)], dtype=torch.int32)
    args = (x, packed.weights, packed.scales, ids, expert_ids, affinity.reshape(-1, 1), bounds)
    return args, (gate_up.float(), down.float(), gu_scale, down_scale)
