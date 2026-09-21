"""The MoE weight-bank gather addresses stay exact through the engine's float32 index math.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
        python -m pytest -s -rA test/vllm_neuron/functional/moe/test_moe_bank_addresses.py

``nisa.tensor_scalar`` casts its int32 data to float32 and back, so a bank row address at or
above 2**24 loses its low bits. The kernels therefore address a weight bank in contraction rows
(``expert * H + h``), never in 128-wide column blocks. Four readings: the kernel's own address
for the last tile of the last expert of the served checkpoint, read back through the simulator;
the rounding itself, read back at the column-block view's address as the control that arms the
float32 model; a census of every gather address at the single-rank, served and unit geometries;
and an eighteen-expert routed gate/up whose last block reaches the last expert.
"""

from __future__ import annotations

import os
import struct

import nki
import nki.isa as nisa
import nki.language as nl
import pytest
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.moe import moe_blockwise_fp8 as moe
from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
    GATE_UP_FUSION,
    GATE_UP_SCALE_BLOCK,
    TILE_SIZE,
    down_bank_address,
    gate_up_bank_address,
    gate_up_dispatch_counters,
    moe_gate_up_blockwise_fp8,
    reset_gate_up_dispatch_counters,
    to_gate_up_kernel_scale_operand,
)

EXACT_LIMIT = 2**24
GEOMETRIES = {  # name: (experts per rank, H, I per rank)
    "single_rank": (288, 4096, 2048),
    "served_tp64_ep16": (18, 4096, 512),
    "unit": (2, 512, 512),
}
E18, H18, I18, BLOCK18 = 18, 512, 512, 128
_FP8 = torch.float8_e4m3fn
_SCALE_VALUES = (1.25, 1.75, 2.5, 3.5)


def _require_simulator() -> None:
    """These readings drive the NKI simulator in CPU mode and nothing else."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1" or os.environ.get("NKI_SIMULATOR") != "1":
        pytest.skip("needs VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1")


def _engine_int32(value: int) -> int:
    """``value`` after a round trip through float32, the engine's arithmetic type."""
    return int(struct.unpack("f", struct.pack("f", float(value)))[0])


def _read_back(expert: int, stride: int, offset: int, step: int) -> list[int]:
    """The 128-row index tile ``_bank_rows`` builds for one expert, read back from the simulator."""

    @nki.jit
    def index_tile(expert_index, iota):
        out = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        ramp = moe._row_iota(iota, TILE_SIZE)

        def one_block(at_block):
            broadcast = moe._broadcast_row(expert_index, 0, TILE_SIZE, at_block=at_block)
            index = moe._bank_rows(TILE_SIZE, broadcast, ramp, stride, offset, step)
            nisa.dma_copy(
                dst=out.ap(pattern=[[1, TILE_SIZE], [1, 1]], offset=0, scalar_offset=at_block, indirect_dim=0),
                src=index,
            )

        nl.fori_loop(0, 1, one_block)
        return out

    expert_t = torch.tensor([[expert]], dtype=torch.int32)
    return wrap_nki(index_tile)(expert_t, moe._partition_iota(expert_t.device)).reshape(-1).tolist()


def _exact(expert: int, stride: int, offset: int, step: int) -> list[int]:
    """The addresses ``_bank_rows`` is asked for, in exact integer arithmetic."""
    return [expert * stride + offset + step * p for p in range(TILE_SIZE)]


def test_the_last_tile_of_the_last_expert_addresses_its_own_row():
    """The kernel's address for expert 287's last h tile, up block 15, reads back exact and in the bank."""
    _require_simulator()
    experts, h_extent, i_extent = GEOMETRIES["single_rank"]
    h0, column_block = h_extent - TILE_SIZE, 2 * (i_extent // GATE_UP_SCALE_BLOCK) - 1
    stride, offset, step, column = gate_up_bank_address(h_extent, h0, column_block)
    got = _read_back(experts - 1, stride, offset, step)
    want = _exact(experts - 1, stride, offset, step)
    print(
        f"ADDRESS|last_tile|expert={experts - 1} h0={h0} column_block={column_block}|first={got[0]} "
        f"last={got[-1]}|exact_first={want[0]} exact_last={want[-1]}|column_elements={column}|"
        f"bank_rows={experts * h_extent}|below_2_24={max(got) < EXACT_LIMIT}"
    )
    assert got == want, [(p, got[p], want[p]) for p in range(TILE_SIZE) if got[p] != want[p]][:4]
    assert max(got) < EXACT_LIMIT and max(got) < experts * h_extent
    assert column == column_block * GATE_UP_SCALE_BLOCK


def test_the_engine_rounds_an_int32_address_above_two_to_the_24():
    """Control: at the column-block view's address the read-back rounds exactly as float32 predicts."""
    _require_simulator()
    stride, offset, step = 4096 * 32, 3968 * 32 + 31, 32
    got = _read_back(287, stride, offset, step)
    want = _exact(287, stride, offset, step)
    predicted = [_engine_int32(_engine_int32(287 * stride) + offset + step * p) for p in range(TILE_SIZE)]
    print(f"ADDRESS|control|first={got[0]} exact={want[0]} predicted={predicted[0]}|last={got[-1]} exact={want[-1]}")
    assert got != want, "the column-block view's address read back exact; the control is not armed"
    assert got[0] == 37744672 and want[0] == 37744671
    assert got == predicted


def _census(name: str) -> dict:
    """Every gate/up and down gather address of one geometry, exact against the float32 round trip."""
    experts, h_extent, i_extent = GEOMETRIES[name]
    n_i_blocks, n_h_blocks = i_extent // GATE_UP_SCALE_BLOCK, h_extent // GATE_UP_SCALE_BLOCK
    wrong, total, largest = 0, 0, 0
    for expert in range(experts):
        for h0 in range(0, h_extent, TILE_SIZE):
            for column_block in range(GATE_UP_FUSION * n_i_blocks):
                stride, offset, step, _ = gate_up_bank_address(h_extent, h0, column_block)
                exact = expert * stride + offset + step * (TILE_SIZE - 1)
                got = _engine_int32(_engine_int32(expert * stride) + offset + step * (TILE_SIZE - 1))
                total += 1
                wrong += got != exact
                largest = max(largest, exact)
        for i0 in range(0, i_extent, TILE_SIZE):
            for h_block in range(n_h_blocks):
                stride, offset, step, _ = down_bank_address(i_extent, i0, h_block)
                exact = expert * stride + offset + step * (TILE_SIZE - 1)
                got = _engine_int32(_engine_int32(expert * stride) + offset + step * (TILE_SIZE - 1))
                total += 1
                wrong += got != exact
                largest = max(largest, exact)
    return {"geometry": name, "gathers": total, "wrong": wrong, "largest_row": largest}


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_every_gather_address_is_exact(name: str):
    """No gather address of the geometry reaches 2**24, so none rounds."""
    reading = _census(name)
    print(f"ADDRESS|census|{reading}|limit={EXACT_LIMIT}")
    assert reading["wrong"] == 0, reading
    assert reading["largest_row"] < EXACT_LIMIT, reading


def _fp8_values(seed: int, *shape: int) -> torch.Tensor:
    """``k/8`` for ``k`` in ``1..7``: on the fp8-e4m3 grid, so every cast is exact."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32) / 8.0


def _scale_grid(experts: int, n_h_blocks: int, n_i_blocks: int) -> torch.Tensor:
    """``[E, H//128, 2, I//128]`` fp32 scales, distinct per block, on two mantissa families."""
    grid = torch.empty((experts, n_h_blocks, GATE_UP_FUSION, n_i_blocks), dtype=torch.float32)
    for expert in range(experts):
        for h_block in range(n_h_blocks):
            for half in range(GATE_UP_FUSION):
                for i_block in range(n_i_blocks):
                    value = _SCALE_VALUES[(h_block % 2) * 2 + (i_block % 2)]
                    quad = (h_block // 2) * (n_i_blocks // 2) + (i_block // 2)
                    grid[expert, h_block, half, i_block] = value * 2.0 ** (((quad + half + expert) % 4) - 2)
    return grid


def _expanded(grid: torch.Tensor, expert: int, h_extent: int, i_extent: int) -> torch.Tensor:
    """``[H, 2*I]`` -- one scale per element, expanded from the block grid."""
    per_block = grid[expert].repeat_interleave(GATE_UP_SCALE_BLOCK, dim=0)
    return per_block.repeat_interleave(GATE_UP_SCALE_BLOCK, dim=2).reshape(h_extent, GATE_UP_FUSION * i_extent)


def test_eighteen_experts_route_the_last_block_to_the_last_expert():
    """Eighteen experts, eighteen blocks, block k to expert k: every block equals its model reference."""
    _require_simulator()
    n_h_blocks, n_i_blocks = H18 // GATE_UP_SCALE_BLOCK, I18 // GATE_UP_SCALE_BLOCK
    tokens = E18 * BLOCK18
    hidden = _fp8_values(201, tokens, H18).to(torch.bfloat16)
    weights = _fp8_values(202, E18, H18, GATE_UP_FUSION * I18).to(_FP8)
    grid = _scale_grid(E18, n_h_blocks, n_i_blocks)
    operands = torch.stack([to_gate_up_kernel_scale_operand(grid[e], H18, I18) for e in range(E18)])
    padded = torch.cat([hidden, torch.zeros((1, H18), dtype=hidden.dtype)])
    row_index = torch.arange(tokens, dtype=torch.int32).reshape(-1, 1)
    block_to_expert = torch.arange(E18, dtype=torch.int32).reshape(-1, 1)
    reset_gate_up_dispatch_counters()
    out = moe_gate_up_blockwise_fp8(padded, weights, operands, row_index, block_to_expert, BLOCK18).to(torch.float32)
    dispatches = gate_up_dispatch_counters()
    assert tuple(out.shape) == (tokens, GATE_UP_FUSION * I18), tuple(out.shape)
    equal_blocks, worst = 0, 0.0
    for expert in range(E18):
        rows = slice(expert * BLOCK18, (expert + 1) * BLOCK18)
        weight = weights[expert].to(torch.float32) * _expanded(grid, expert, H18, I18)
        want = hidden[rows].to(torch.float32) @ weight
        scales64 = _expanded(grid, expert, H18, I18).to(torch.float64)
        want64 = hidden[rows].to(torch.float64) @ (weights[expert].to(torch.float64) * scales64)
        assert torch.equal(want64, want.to(torch.float64)), f"block {expert}: the fp64 and fp32 references differ"
        assert float(want.abs().max()) > 0.0, f"block {expert}: an all-zero reference measures nothing"
        diff = float((out[rows] - want).abs().max())
        worst = max(worst, diff)
        equal_blocks += torch.equal(out[rows], want)
        print(
            f"ADDRESS|routed|block={expert} expert={expert} "
            f"bit_equal={int(torch.equal(out[rows], want))} max_abs_diff={diff}"
        )
    print(
        f"ADDRESS|routed_verdict|blocks_equal={equal_blocks}/{E18} "
        f"worst_max_abs_diff={worst} dispatches={dispatches}"
    )
    assert dispatches[0] == 1 and dispatches[1] == 0, dispatches
    assert equal_blocks == E18
