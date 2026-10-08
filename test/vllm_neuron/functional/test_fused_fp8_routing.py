# SPDX-License-Identifier: Apache-2.0
"""Fused expert routing, including empty blocks and holes inside row tiles."""

import pytest
import torch
import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.allocator import create_auto_alloc_manager

from vllm_neuron.functional.moe.fused_fp8_pack import pack_experts
from vllm_neuron.functional.moe import moe_fused_fp8 as _module
from vllm_neuron.functional.moe.moe_fused_fp8 import (
    _phase_bounds,
    moe_fused_fp8_kernel,
)


@nki.jit
def _routing_phase_boundary_probe(counts):
    program, programs = nl.program_id(0), nl.num_programs(0)
    output = nl.ndarray((programs, counts.shape[0]), dtype=nl.int32,
                        buffer=nl.shared_hbm)
    manager = create_auto_alloc_manager()
    manager.open_scope("phase_boundary_probe")
    count = manager.alloc((1, 8), nl.float32)[:, :1]
    result = manager.alloc((1, 8), nl.int32)[:, :1]
    for index in range(counts.shape[0]):
        nisa.dma_copy(dst=count, src=counts[index:index + 1, :])
        boundary = _phase_bounds(count, program, programs, manager)
        nisa.register_store(dst=result, src=boundary)
        nisa.dma_copy(dst=output[program:program + 1, index:index + 1],
                      src=result)
    manager.close_scope()
    return output


@pytest.mark.parametrize("programs", [1, 2])
def test_dense_phase_pair_count_covers_every_integer_boundary(programs, monkeypatch):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "1")
    # Includes all 392 row-tile slots in the production 49-by-256 bucket.
    counts = torch.arange(513, dtype=torch.float32).reshape(-1, 1)
    actual = wrap_nki(_routing_phase_boundary_probe)[programs](counts)
    dense_pairs = (torch.arange(513) + programs - 1) // programs
    expected = dense_pairs * programs + torch.arange(programs).reshape(-1, 1)
    assert torch.equal(actual, expected.to(torch.int32))


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


def _reference(inputs, weights):
    x, packed, row_ids, expert_ids, affinity, bounds = inputs
    gate_up, down, gu_scale, down_scale = weights
    intermediate = down.shape[1]
    output = torch.zeros(*row_ids.shape, x.shape[1])
    for block in range(row_ids.shape[0]):
        expert = int(expert_ids[block, 0])
        valid = row_ids[block] >= 0
        rows = row_ids[block, valid].long()
        if rows.numel() == 0:
            continue
        selected = x[rows].float()
        projection = torch.zeros(rows.numel(), 2 * intermediate)
        for k in range(x.shape[1] // 128):
            product = selected[:, k * 128:(k + 1) * 128] @ gate_up[
                expert, k * 128:(k + 1) * 128
            ]
            block_scales = gu_scale[expert, k].reshape(-1).repeat_interleave(128)
            projection += product * block_scales
        gate, up = projection.split(intermediate, dim=1)
        gate = gate.clamp(max=float(bounds[0, 0]))
        up = up.clamp(min=float(bounds[0, 1]), max=float(bounds[0, 2]))
        activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
        result = torch.zeros(rows.numel(), x.shape[1])
        for k in range(intermediate // 128):
            product = activated[:, k * 128:(k + 1) * 128] @ down[
                expert, k * 128:(k + 1) * 128
            ]
            result += product * down_scale[expert, k].repeat_interleave(128)
        output[block, valid] = result * affinity[rows, expert, None]
    return output


@pytest.mark.parametrize("width", [1, 33, 64])
def test_empty_blocks_partial_tiles_and_noncontiguous_valid_rows(width, monkeypatch):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "1")
    ids = [[-1] * width for _ in range(3)]
    ids[0][0] = 2
    ids[0][-1] = 0
    ids[2][width // 2] = 6
    inputs, weights = _case(ids)
    x, packed, row_ids, expert_ids, affinity, bounds = inputs
    kernel = wrap_nki(moe_fused_fp8_kernel)[2]
    args = (x, packed.weights, packed.scales, row_ids, expert_ids,
            affinity.reshape(-1, 1), bounds)
    baseline = kernel(*args, BLOCK_M=width, BLOCK_N=256, BLOCK_K=256,
                      SKIP_PADDING=False)
    actual = kernel(*args, BLOCK_M=width, BLOCK_N=256, BLOCK_K=256,
                    SKIP_PADDING=True)
    expected = _reference(inputs, weights)
    # BLAS may round a one-row tail differently from its wider matrix call.
    torch.testing.assert_close(actual, baseline, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=1e-5)
    assert torch.count_nonzero(actual[row_ids < 0]) == 0


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


def test_dense_block_uses_full_width_with_clamps_and_scales(monkeypatch):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "1")
    inputs, weights = _case([
        [i % 7 for i in range(64)],
        [-1] * 64,
        [i % 7 for i in range(47)] + [-1] * 17,
    ])
    x, packed, row_ids, expert_ids, affinity, bounds = inputs
    kernel = wrap_nki(moe_fused_fp8_kernel)[2]
    args = (x, packed.weights, packed.scales, row_ids, expert_ids,
            affinity.reshape(-1, 1), bounds)
    baseline = kernel(*args, BLOCK_M=64, SKIP_PADDING=False)
    actual = kernel(*args, BLOCK_M=64, SKIP_PADDING=True)
    torch.testing.assert_close(actual, baseline, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(actual, _reference(inputs, weights),
                               rtol=3e-2, atol=1e-5)
    assert torch.count_nonzero(actual[row_ids < 0]) == 0


def test_sparse_holes_cross_all_row_tile_boundaries(monkeypatch):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "1")
    positions = [0, 31, 32, 63, 64, 95, 96, 127, 128, 159, 160,
                 191, 192, 223, 224, 255]
    ids = [[-1] * 256 for _ in range(3)]
    for index, position in enumerate(positions):
        ids[0][position] = index % 7
    for index, position in enumerate(positions[::2]):
        ids[2][position] = (index + 2) % 7
    inputs, weights = _case(ids)
    x, packed, row_ids, expert_ids, affinity, bounds = inputs
    args = (x, packed.weights, packed.scales, row_ids, expert_ids,
            affinity.reshape(-1, 1), bounds)
    actual = wrap_nki(moe_fused_fp8_kernel)[2](
        *args, BLOCK_M=256, BLOCK_N=256, BLOCK_K=256
    )
    torch.testing.assert_close(actual, _reference(inputs, weights),
                               rtol=3e-2, atol=1e-5)
    assert torch.count_nonzero(actual[row_ids < 0]) == 0


@nki.jit
def _ownership_probe(row_ids, expert_ids):
    program, programs = nl.program_id(0), nl.num_programs(0)
    manager = create_auto_alloc_manager()
    manager.open_scope("ownership_probe")
    q = row_ids.shape[1]
    step = _module._row_step(q, q)
    span = _module._routing_spans(row_ids, step, manager)
    paired, classes, empty = _module._routing_worklists(
        span, _module._experts_row(expert_ids), _module._row_classes(q, step), manager,
        merge_pairs=(q == 256),
    )
    result = []
    blocks = row_ids.shape[0]
    for source in ([paired] if paired else []) + classes + [empty]:
        listed = _module._list_entries(source, blocks)
        destination = nl.ndarray((programs, listed.shape[1]), dtype=nl.int32,
                                  buffer=nl.shared_hbm)
        count = nl.ndarray((programs, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=destination[program:program + 1, :], src=listed)
        nisa.dma_copy(dst=count[program:program + 1, :], src=source[1])
        result += [destination, count]
    manager.close_scope()
    return tuple(result)


@pytest.mark.parametrize("programs", [1, 2])
@pytest.mark.parametrize("blocks,q", [(1, 256), (2, 256), (3, 256), (7, 256), (49, 256), (3, 33)])
def test_worklists_cover_original_slots_once(blocks, q, programs, monkeypatch):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "1")
    generator = torch.Generator().manual_seed(20261003 + blocks + q)
    counts = [q, q, q // 2 + 1, q // 2, 0, q // 2 + 7, q]
    counts = [counts[i % len(counts)] for i in range(blocks)]
    ids = torch.full((blocks, q), -1, dtype=torch.int32)
    for block, count in enumerate(counts):
        ids[block, torch.randperm(q, generator=generator)[:count]] = block
    experts = torch.tensor([i // 3 for i in range(blocks)], dtype=torch.int32).reshape(-1, 1)
    actual = wrap_nki(_ownership_probe)[programs](ids, experts)
    lists = []
    for entries, count in zip(actual[0::2], actual[1::2]):
        assert torch.equal(entries, entries[:1].expand_as(entries))
        assert torch.equal(count, count[:1].expand_as(count))
        length = int(count[0, 0])
        assert torch.all(entries[0, length:] == blocks)
        lists.append(entries[0, :length].tolist())
    pairs = lists[0] if q == 256 else []
    owners = [slot for first in pairs for slot in (first, first + 1)]
    owners += [block for owned in lists[1 if q == 256 else 0:] for block in owned]
    # Every original block has exactly one owner: a pair, a row class, or the
    # zero list, which holds exactly the blocks without a routed row.
    assert sorted(owners) == list(range(blocks))
    assert lists[-1] == [block for block, count in enumerate(counts) if count == 0]
    for first in pairs:
        assert experts[first] == experts[first + 1]
        assert counts[first] > 0 and counts[first + 1] > 0


@nki.jit
def _small_pair_body_probe(hidden, weights, scales, row_ids, expert_ids,
                            affinity, bounds, FIRST=0):
    """Exercise the same 512-row body with small independent reference weights."""
    blocks, q = row_ids.shape
    h, nh = hidden.shape[1], weights.shape[3]
    program, programs = nl.program_id(0), nl.num_programs(0)
    output = nl.ndarray((blocks, q, h), dtype=nl.float32, buffer=nl.shared_hbm)
    zeros = nl.ndarray((128, h), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zeros, value=0)
    for block in range(program, blocks, programs):
        for start in range(0, q, 128):
            nisa.dma_copy(dst=output[block, start:start + 128, :], src=zeros)
    if programs == 2:
        nisa.core_barrier(data=output, cores=(0, 1))
    clamp = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=clamp, src=bounds)
    block = nisa.register_alloc(FIRST)
    loaded = _module._expert_operands(weights, scales, _module._experts_row(expert_ids),
                                      block, 2, 2)
    compute = (hidden, weights, row_ids, expert_ids, affinity, clamp, output)
    _module._compute_rows(compute, loaded, block, 0, 2 * q)
    return output


@pytest.mark.parametrize("expert_ids_values,first", [([0, 0, 0], 0), ([0, 0, 1], 0), ([0, 1, 1], 1)])
def test_512_row_body_retains_original_256_row_block_stride(expert_ids_values, first, monkeypatch):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "1")
    ids = [[-1] * 256 for _ in range(3)]
    for block in range(3):
        for row in range(133 + 9 * block):
            ids[block][(row * 31) % 256] = (row + block) % 7
    inputs, weights = _case(ids)
    x, packed, row_ids, expert_ids, affinity, bounds = inputs
    expert_ids = torch.tensor(expert_ids_values, dtype=torch.int32).reshape(3, 1)
    inputs = (x, packed, row_ids, expert_ids, affinity, bounds)
    actual = wrap_nki(_small_pair_body_probe)[2](
        x, packed.weights, packed.scales, row_ids, expert_ids,
        affinity.reshape(-1, 1), bounds, FIRST=first)
    expected = _reference(inputs, weights)
    expected[:first].zero_()
    expected[first + 2:].zero_()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=1e-5)
    assert torch.count_nonzero(actual[row_ids < 0]) == 0
