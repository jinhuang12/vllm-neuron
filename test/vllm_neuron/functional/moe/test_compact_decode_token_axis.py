# SPDX-License-Identifier: Apache-2.0
"""``compact_decode_kernel`` with a real token axis: ``1 <= q <= 64`` rows per block.

At 5938748 the kernel asserted ``q == 1``. The token axis now feeds one m-block
of ``q`` rows to every product, as the general kernel does with ``BLOCK_M = q``.
"""

from __future__ import annotations

import pytest
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.moe.fused_fp8_pack import pack_experts
from vllm_neuron.functional.moe.moe_fused_fp8_decode import compact_decode_kernel


def _case(q, active, holes, seed=20261005):
    """8 blocks of ``q`` rows over 8 experts; ``active`` blocks hold tokens."""
    gen = torch.Generator().manual_seed(seed + q)
    experts, hidden, intermediate = 8, 128, 128
    x = torch.randn((q + 1, hidden), generator=gen).to(torch.bfloat16)
    x[-1].zero_()
    gate_up = (torch.randn((experts, hidden, 2 * intermediate), generator=gen) / 16
               ).to(torch.float8_e4m3fn)
    down = (torch.randn((experts, intermediate, hidden), generator=gen) / 16
            ).to(torch.float8_e4m3fn)
    gu_scale = torch.rand((experts, 1, 2, 1), generator=gen) + 0.5
    down_scale = torch.rand((experts, 1, 1), generator=gen) + 0.5
    packed = pack_experts(gate_up, down, gu_scale, down_scale)
    rows = torch.full((8, q), -1, dtype=torch.int32)
    expert_ids = torch.arange(experts, dtype=torch.int32).reshape(-1, 1)
    affinity = torch.zeros((q + 1, experts), dtype=torch.float32)
    for block in active:
        # Real tokens first, then ``holes`` padding ids at the end of the block.
        real = q - holes if q > holes else 1
        rows[block, :real] = torch.arange(real, dtype=torch.int32)
        affinity[:real, block] = torch.rand(real, generator=gen) + 0.25
    bounds = torch.tensor([0.6, -0.4, 0.5], dtype=torch.float32).repeat(128, 1)
    return (x, packed.weights, packed.scales, rows, expert_ids,
            affinity.reshape(-1, 1), bounds)


def test_compact_token_axis_refuses_more_than_64_rows():
    args = _case(65, [0], 0)
    with pytest.raises(Exception, match="64"):
        wrap_nki(compact_decode_kernel)[1](*args, BLOCK_N=256, BLOCK_K=256)
