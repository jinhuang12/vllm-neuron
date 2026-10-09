# SPDX-License-Identifier: Apache-2.0
"""Prefill schedule of the fused FP8 expert kernel: zero fill.

The expected outputs below are derived from the routing ids
alone, so a schedule change that drops, duplicates or misclassifies a block
fails here.
"""

import pytest
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.moe.fused_fp8_pack import pack_experts
from vllm_neuron.functional.moe.moe_fused_fp8 import moe_fused_fp8_kernel


def _case(row_ids, experts=3, hidden=256, intermediate=128, tokens=8, seed=20261008):
    torch.manual_seed(seed)
    x = torch.randn(tokens, hidden).to(torch.bfloat16)
    x[-1].zero_()
    gate_up = (torch.randn(experts, hidden, 2 * intermediate) / 16).to(torch.float8_e4m3fn)
    down = (torch.randn(experts, intermediate, hidden) / 16).to(torch.float8_e4m3fn)
    gu_scale = torch.rand(experts, hidden // 128, 2, intermediate // 128) + 0.5
    down_scale = torch.rand(experts, intermediate // 128, hidden // 128) + 0.5
    packed = pack_experts(gate_up, down, gu_scale, down_scale)
    affinity = torch.rand(tokens, experts)
    affinity[-1].zero_()
    bounds = torch.tensor([0.6, -0.4, 0.5]).repeat(128, 1)
    ids = torch.tensor(row_ids, dtype=torch.int32)
    expert_ids = torch.tensor([[b % experts] for b in range(ids.shape[0])], dtype=torch.int32)
    args = (x, packed.weights, packed.scales, ids, expert_ids, affinity.reshape(-1, 1), bounds)
    return args, ids


def _run(args, programs=2, **tiles):
    return wrap_nki(moe_fused_fp8_kernel)[programs](*args, **tiles)


@pytest.mark.parametrize("block_m", [256, 64])
@pytest.mark.parametrize("programs", [1, 2])
def test_class_boundaries_match_the_dense_schedule(programs, block_m):
    # One block per span: empty, 1 row, the 64- and 128-row class edges and
    # one past them, a lone row in the last tile, and a full block. With
    # 64-row slices the wider classes pipeline several slices per block.
    lasts = [-1, 0, 63, 64, 127, 128, 255, 200]
    ids = [[-1] * 256 for _ in lasts]
    for block, last in enumerate(lasts):
        for row in range(0, last + 1, 3):
            ids[block][row] = (row + block) % 7
        if last >= 0:
            ids[block][last] = block % 7
    ids[7] = [-1] * 255 + [5]
    args, row_ids = _case(ids)
    dense = _run(args, programs, BLOCK_M=256, SKIP_PADDING=False)
    actual = _run(args, programs, BLOCK_M=block_m)
    # BLAS may round a column of a narrower simulator product differently.
    torch.testing.assert_close(actual, dense, rtol=1e-5, atol=1e-7)
    assert torch.count_nonzero(actual[row_ids < 0]) == 0


@pytest.mark.parametrize("programs", [1, 2])
def test_empty_blocks_past_the_overlapped_slots_are_zeroed(programs):
    # 20 blocks for 3 experts give ceil(20/3)-1 = 6 zero slots per item; two
    # routed blocks leave 18 empty ones, more than the item loop's slots hold.
    ids = [[-1] * 256 for _ in range(20)]
    ids[3][:5] = [0, 1, 2, 3, 4]
    ids[11][:40] = [i % 7 for i in range(40)]
    args, row_ids = _case(ids)
    output = _run(args, programs, BLOCK_M=256)
    dense = _run(args, programs, BLOCK_M=256, SKIP_PADDING=False)
    torch.testing.assert_close(output, dense, rtol=1e-5, atol=1e-7)
    routed = torch.zeros(20, dtype=torch.bool)
    routed[[3, 11]] = True
    assert torch.equal(output[~routed], torch.zeros_like(output[~routed]))
    assert torch.count_nonzero(output[3, :5]) > 0 and torch.count_nonzero(output[11, :40]) > 0


@pytest.mark.parametrize("programs", [1, 2])
def test_every_block_routed_writes_no_zero_block(programs):
    # No empty block: every zero-fill slot reads the out-of-range id.
    ids = [[(row + block) % 7 if row <= 10 + 70 * block else -1 for row in range(256)]
           for block in range(4)]
    args, row_ids = _case(ids)
    output = _run(args, programs, BLOCK_M=256)
    dense = _run(args, programs, BLOCK_M=256, SKIP_PADDING=False)
    torch.testing.assert_close(output, dense, rtol=1e-5, atol=1e-7)
    assert torch.count_nonzero(output[row_ids < 0]) == 0
