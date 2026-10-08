# SPDX-License-Identifier: Apache-2.0
"""Prefill schedule of the fused FP8 expert kernel: row classes and zero fill.

Every routed block is computed once, over the row prefix that its last routed
row tile needs; rows past that prefix and every empty block are written with
zeros. The expected lists and outputs below are derived from the routing ids
alone, so a schedule change that drops, duplicates or misclassifies a block
fails here.
"""

import pytest
import torch
import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.allocator import create_auto_alloc_manager

from vllm_neuron.functional.moe import moe_fused_fp8 as _module
from vllm_neuron.functional.moe.fused_fp8_pack import pack_experts
from vllm_neuron.functional.moe.moe_fused_fp8 import moe_fused_fp8_kernel


@nki.jit
def _worklist_probe(row_ids, expert_ids):
    """Copy every compacted block list, and its count, to HBM."""
    program, programs = nl.program_id(0), nl.num_programs(0)
    blocks, q = row_ids.shape
    manager = create_auto_alloc_manager()
    manager.open_scope("worklist_probe")
    step = _module._row_step(q, q)
    span = _module._routing_spans(row_ids, step, manager)
    paired, classes, empty = _module._routing_worklists(
        span, _module._experts_row(expert_ids), _module._row_classes(q, step),
        manager, merge_pairs=(q == 256), extra=1)
    result = []
    for worklist in ([paired] if paired else []) + classes + [empty]:
        listed = _module._list_entries(worklist, blocks)
        entries = nl.ndarray((programs, listed.shape[1]), dtype=nl.int32,
                             buffer=nl.shared_hbm)
        count = nl.ndarray((programs, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=entries[program:program + 1, :], src=listed)
        nisa.dma_copy(dst=count[program:program + 1, :], src=worklist[1])
        result += [entries, count]
    manager.close_scope()
    return tuple(result)


def _row_step(q):
    step = min(q, 32)
    while q % step:
        step -= 1
    return step


def _class_bounds(q):
    """Row tiles that bound each class: doubling widths below a block, then the block."""
    tiles = q // _row_step(q)
    bounds, width = [], 2
    while width < tiles:
        bounds.append(width)
        width *= 2
    return bounds + [tiles]


def _expected_lists(ids, experts, merge):
    blocks, q = ids.shape
    step = _row_step(q)
    span = []
    for block in range(blocks):
        routed = torch.nonzero(ids[block] >= 0).flatten()
        span.append(0 if routed.numel() == 0 else int(routed[-1]) // step + 1)
    bounds = _class_bounds(q)
    classes, low = [], 0
    for high in bounds:
        classes.append([b for b in range(blocks) if low < span[b] <= high])
        low = high
    pairs = []
    if merge:
        dense, index = classes[-1], 0
        while index < blocks - 1:
            if index in dense and index + 1 in dense and experts[index] == experts[index + 1]:
                pairs.append(index)
                index += 2
            else:
                index += 1
        members = {b for first in pairs for b in (first, first + 1)}
        classes[-1] = [b for b in classes[-1] if b not in members]
    empty = [b for b in range(blocks) if span[b] == 0]
    return ([pairs] if merge else []) + classes + [empty]


@pytest.mark.parametrize("programs", [1, 2])
@pytest.mark.parametrize("blocks,q", [(1, 256), (3, 256), (9, 256), (49, 256), (5, 33), (4, 64)])
def test_worklists_split_blocks_by_last_routed_row_tile(blocks, q, programs):
    generator = torch.Generator().manual_seed(20261008 + blocks + q)
    ids = torch.full((blocks, q), -1, dtype=torch.int32)
    for block in range(blocks):
        # Last routed row from empty through every class boundary to a full block.
        last = (block * 37) % (q + 1) - 1
        if last >= 0:
            ids[block, last] = block
            fill = torch.randperm(last + 1, generator=generator)[: (last + 1) // 2]
            ids[block, fill] = block
    experts = torch.tensor([b // 2 for b in range(blocks)], dtype=torch.int32).reshape(-1, 1)
    outputs = wrap_nki(_worklist_probe)[programs](ids, experts)
    expected = _expected_lists(ids, experts.flatten().tolist(), merge=(q == 256))
    assert len(outputs) == 2 * len(expected)
    covered = []
    for index, wanted in enumerate(expected):
        entries, count = outputs[2 * index], outputs[2 * index + 1]
        for program in range(programs):
            assert int(count[program, 0]) == len(wanted)
            assert entries[program, :len(wanted)].tolist() == wanted
            # Every unused entry is the out-of-range block id: a zero-fill slot
            # that reads it is skipped by the DMA engine.
            assert torch.all(entries[program, len(wanted):] == blocks)
        covered += wanted
    if q == 256:
        covered += [first + 1 for first in expected[0]]
    assert sorted(covered) == list(range(blocks))


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

