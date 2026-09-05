# SPDX-License-Identifier: Apache-2.0
"""Selective block-scaled FP8 MoE kernel for GLM-5.2 on Trn2."""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import torch
import torch.nn.functional as F
from nki.isa.constants import oob_mode

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from .block_fp8 import BLOCK_COLS, BLOCK_ROWS, dequantize_block_fp8


def _kernel_assert(condition: bool, message: str) -> None:
    assert condition, "[INTERNAL_ERROR] [NCC_INKI016] " + message


def _div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


@torch.no_grad()
def selective_block_fp8_moe_reference(
    hidden_states: torch.Tensor,
    local_affinities: torch.Tensor,
    gate_weights: torch.Tensor,
    gate_scales: torch.Tensor,
    up_weights: torch.Tensor,
    up_scales: torch.Tensor,
    down_weights: torch.Tensor,
    down_scales: torch.Tensor,
) -> torch.Tensor:
    """Execute only locally routed experts using checkpoint-native FP8 blocks.

    Weight shapes are ``gate/up=[E_local,I,H]`` and
    ``down=[E_local,H,I]``.  Every scale is FP32 and covers one 128x128
    checkpoint block.  The affinity is applied after the expert down
    projection.
    """

    if hidden_states.ndim != 2 or local_affinities.ndim != 2:
        raise ValueError("hidden_states and local_affinities must be rank 2")
    num_experts, intermediate_size, hidden_size = gate_weights.shape
    expected_gate_shape = (num_experts, intermediate_size, hidden_size)
    expected_down_shape = (num_experts, hidden_size, intermediate_size)
    if tuple(up_weights.shape) != expected_gate_shape:
        raise ValueError("gate and up weights must have identical shapes")
    if tuple(down_weights.shape) != expected_down_shape:
        raise ValueError("down weight shape does not match gate/up dimensions")
    if tuple(hidden_states.shape[1:]) != (hidden_size,):
        raise ValueError("hidden size does not match expert weights")
    if tuple(local_affinities.shape) != (hidden_states.shape[0], num_experts):
        raise ValueError("local affinities must be [T,E_local]")

    output = torch.zeros_like(hidden_states)
    for expert_id in range(num_experts):
        token_ids = torch.nonzero(
            local_affinities[:, expert_id] != 0, as_tuple=False
        ).flatten()
        if token_ids.numel() == 0:
            continue
        expert_input = hidden_states.index_select(0, token_ids)
        gate = F.linear(
            expert_input,
            dequantize_block_fp8(gate_weights[expert_id], gate_scales[expert_id]).to(
                expert_input.dtype
            ),
        )
        up = F.linear(
            expert_input,
            dequantize_block_fp8(up_weights[expert_id], up_scales[expert_id]).to(
                expert_input.dtype
            ),
        )
        down = F.linear(
            F.silu(gate) * up,
            dequantize_block_fp8(down_weights[expert_id], down_scales[expert_id]).to(
                expert_input.dtype
            ),
        )
        scaled = down * local_affinities[token_ids, expert_id : expert_id + 1].to(
            down.dtype
        )
        output.index_add_(0, token_ids, scaled)
    return output


@nki.jit
def _selective_block_fp8_moe_nki(
    hidden_states,
    expert_affinities_masked,
    token_position_to_id,
    block_to_expert,
    conditions,
    gate_weight_0,
    gate_scale_0,
    up_weight_0,
    up_scale_0,
    down_weight_0,
    down_scale_0,
    gate_weight_1,
    gate_scale_1,
    up_weight_1,
    up_scale_1,
    down_weight_1,
    down_scale_1,
    gate_weight_2,
    gate_scale_2,
    up_weight_2,
    up_scale_2,
    down_weight_2,
    down_scale_2,
    gate_weight_3,
    gate_scale_3,
    up_weight_3,
    up_scale_3,
    down_weight_3,
    down_scale_3,
    block_size=128,
):
    """Process only active expert-contiguous blocks.

    The block mapping is the existing ``build_blockwise_mapping`` contract.
    Active blocks must precede padded blocks. The actual campaign pin is NKI
    0.5.0+28631259367.ga768afa6, whose installed source supports
    register-backed dynamic loops. Each loop loads exactly one expert's raw
    FP8 tensors. Every 128x128 Tensor Engine partial is multiplied by its FP32
    checkpoint scale before FP32 accumulation.
    """

    _kernel_assert(len(hidden_states.shape) == 2, "hidden_states must be [T,H]")
    _kernel_assert(len(gate_weight_0.shape) == 2, "gate weight must be [I,H]")
    _kernel_assert(len(up_weight_0.shape) == 2, "up weight must be [I,H]")
    _kernel_assert(len(down_weight_0.shape) == 2, "down weight must be [H,I]")
    _kernel_assert(gate_weight_0.shape == up_weight_0.shape, "gate/up shape mismatch")
    _kernel_assert(gate_scale_0.shape == up_scale_0.shape, "gate/up scale mismatch")
    _kernel_assert(block_size <= 128, "block_size must fit the partition dimension")
    _kernel_assert(block_size > 0, "block_size must be positive")
    _kernel_assert(nl.num_programs(axes=0) == 2, "selective MoE requires LNC2")

    token_count, hidden_size = hidden_states.shape
    num_experts = 4
    intermediate_size, weight_hidden_size = gate_weight_0.shape
    _kernel_assert(
        expert_affinities_masked.shape == (token_count * num_experts, 1),
        "flattened affinity shape mismatch",
    )
    _kernel_assert(
        token_position_to_id.shape[0] % block_size == 0,
        "token mapping must contain whole blocks",
    )
    num_mapping_blocks = token_position_to_id.shape[0] // block_size
    _kernel_assert(
        block_to_expert.shape[0] == num_mapping_blocks,
        "expert mapping length mismatch",
    )
    _kernel_assert(
        conditions.shape[0] == num_mapping_blocks,
        "condition length mismatch",
    )
    _kernel_assert(hidden_size == weight_hidden_size, "hidden size mismatch")
    _kernel_assert(
        down_weight_0.shape == (hidden_size, intermediate_size),
        "down weight shape mismatch",
    )
    _kernel_assert(hidden_size % BLOCK_COLS == 0, "H must be divisible by 128")
    _kernel_assert(intermediate_size % BLOCK_ROWS == 0, "I must be divisible by 128")
    _kernel_assert(
        gate_scale_0.shape
        == (intermediate_size // BLOCK_ROWS, hidden_size // BLOCK_COLS),
        "gate scale grid mismatch",
    )
    _kernel_assert(
        down_scale_0.shape
        == (hidden_size // BLOCK_ROWS, intermediate_size // BLOCK_COLS),
        "down scale grid mismatch",
    )
    shard_id = nl.program_id(axis=0)
    output_rows_per_shard = BLOCK_ROWS // 2

    # Both programs execute identical control flow. Each initializes and later
    # accumulates only its disjoint 64-column slice of every output block.
    output = nl.ndarray(
        hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.shared_hbm
    )
    for token_tile_idx in nl.affine_range(_div_ceil(token_count, BLOCK_ROWS)):
        token_start = token_tile_idx * BLOCK_ROWS
        token_size = min(BLOCK_ROWS, token_count - token_start)
        zeros = nl.ndarray(
            (token_size, output_rows_per_shard),
            dtype=hidden_states.dtype,
            buffer=nl.sbuf,
        )
        nisa.memset(dst=zeros, value=0.0)
        for output_block in nl.sequential_range(hidden_size // BLOCK_ROWS):
            output_start = output_block * BLOCK_ROWS + shard_id * output_rows_per_shard
            nisa.dma_copy(
                dst=output[
                    token_start : token_start + token_size,
                    output_start : output_start + output_rows_per_shard,
                ],
                src=zeros,
            )
    nisa.core_barrier(output, (0, 1))
    condition_tile = nl.ndarray(
        (1, conditions.shape[0]), dtype=nl.int32, buffer=nl.sbuf
    )
    nisa.dma_copy(dst=condition_tile, src=conditions.reshape((1, conditions.shape[0])))
    block_expert_tile = nl.ndarray(
        (1, block_to_expert.shape[0]), dtype=nl.int32, buffer=nl.sbuf
    )
    nisa.dma_copy(
        dst=block_expert_tile,
        src=block_to_expert.reshape((1, block_to_expert.shape[0])),
    )
    block_index = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=block_index, value=0)

    # Each phase is compile-time specialized to one direct expert. Only the
    # number of active blocks remains runtime dynamic.
    for expert_phase in range(num_experts):
        if expert_phase == 0:
            expert_gate_weight = gate_weight_0
            expert_gate_scale = gate_scale_0
            expert_up_weight = up_weight_0
            expert_up_scale = up_scale_0
            expert_down_weight = down_weight_0
            expert_down_scale = down_scale_0
        elif expert_phase == 1:
            expert_gate_weight = gate_weight_1
            expert_gate_scale = gate_scale_1
            expert_up_weight = up_weight_1
            expert_up_scale = up_scale_1
            expert_down_weight = down_weight_1
            expert_down_scale = down_scale_1
        elif expert_phase == 2:
            expert_gate_weight = gate_weight_2
            expert_gate_scale = gate_scale_2
            expert_up_weight = up_weight_2
            expert_up_scale = up_scale_2
            expert_down_weight = down_weight_2
            expert_down_scale = down_scale_2
        else:
            expert_gate_weight = gate_weight_3
            expert_gate_scale = gate_scale_3
            expert_up_weight = up_weight_3
            expert_up_scale = up_scale_3
            expert_down_weight = down_weight_3
            expert_down_scale = down_scale_3

        expert_blocks = nl.ndarray(
            (1, num_mapping_blocks), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=expert_blocks,
            data=block_expert_tile,
            op0=nl.equal,
            operand0=expert_phase,
        )
        active_expert_blocks = nl.ndarray(
            (1, num_mapping_blocks), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=active_expert_blocks,
            data1=expert_blocks,
            data2=condition_tile,
            op=nl.multiply,
        )
        expert_block_count = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=expert_block_count,
            data=active_expert_blocks,
            op=nl.add,
            axis=1,
        )
        expert_block_count_register = nisa.register_alloc()
        nisa.register_load(
            dst=expert_block_count_register,
            src=expert_block_count,
        )
        expert_id = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=expert_id, value=expert_phase)
        for _ in nl.dynamic_range(0, expert_block_count_register):
            token_ids = nl.ndarray((block_size, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=token_ids,
                src=token_position_to_id.reshape((num_mapping_blocks, block_size)).ap(
                    pattern=[[1, block_size]],
                    offset=0,
                    scalar_offset=block_index,
                    indirect_dim=0,
                ),
            )
            hidden_block = nl.ndarray(
                (block_size, hidden_size), dtype=hidden_states.dtype, buffer=nl.sbuf
            )
            nisa.memset(dst=hidden_block, value=0.0)
            nisa.dma_copy(
                dst=hidden_block,
                src=hidden_states.ap(
                    pattern=[[hidden_size, block_size], [1, hidden_size]],
                    offset=0,
                    vector_offset=token_ids,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )

            intermediate_hbm = nl.ndarray(
                (2, intermediate_size, block_size),
                dtype=hidden_states.dtype,
                buffer=nl.shared_hbm,
            )
            for intermediate_block in nl.sequential_range(
                intermediate_size // BLOCK_ROWS
            ):
                gate_accumulated = nl.ndarray(
                    (BLOCK_ROWS, block_size), dtype=nl.float32, buffer=nl.sbuf
                )
                up_accumulated = nl.ndarray(
                    (BLOCK_ROWS, block_size), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.memset(dst=gate_accumulated, value=0.0)
                nisa.memset(dst=up_accumulated, value=0.0)
                for hidden_block_index in nl.sequential_range(
                    hidden_size // BLOCK_COLS
                ):
                    hidden_transposed = nl.ndarray(
                        (BLOCK_COLS, block_size),
                        dtype=hidden_states.dtype,
                        buffer=nl.sbuf,
                    )
                    nisa.dma_transpose(
                        dst=hidden_transposed,
                        src=hidden_block[
                            0:block_size,
                            hidden_block_index
                            * BLOCK_COLS : (hidden_block_index + 1)
                            * BLOCK_COLS,
                        ],
                        axes=(1, 0),
                    )
                    # The compile-time expert phase binds direct checkpoint tensors.
                    gate_weight_row_major = nl.ndarray(
                        (BLOCK_ROWS, BLOCK_COLS),
                        dtype=expert_gate_weight.dtype,
                        buffer=nl.sbuf,
                    )
                    up_weight_row_major = nl.ndarray(
                        (BLOCK_ROWS, BLOCK_COLS),
                        dtype=expert_up_weight.dtype,
                        buffer=nl.sbuf,
                    )
                    weight_offset = (
                        intermediate_block * BLOCK_ROWS * hidden_size
                        + hidden_block_index * BLOCK_COLS
                    )
                    weight_pattern = [
                        [hidden_size, BLOCK_ROWS],
                        [1, BLOCK_COLS],
                    ]
                    nisa.dma_copy(
                        dst=gate_weight_row_major,
                        src=expert_gate_weight.ap(
                            pattern=weight_pattern, offset=weight_offset
                        ),
                    )
                    nisa.dma_copy(
                        dst=up_weight_row_major,
                        src=expert_up_weight.ap(
                            pattern=weight_pattern, offset=weight_offset
                        ),
                    )
                    gate_weight_bf16 = nl.ndarray(
                        (BLOCK_ROWS, BLOCK_COLS), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    up_weight_bf16 = nl.ndarray(
                        (BLOCK_ROWS, BLOCK_COLS), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=gate_weight_bf16, src=gate_weight_row_major)
                    nisa.tensor_copy(dst=up_weight_bf16, src=up_weight_row_major)
                    gate_weight_transposed = nl.ndarray(
                        (BLOCK_COLS, BLOCK_ROWS),
                        dtype=nl.bfloat16,
                        buffer=nl.psum,
                    )
                    up_weight_transposed = nl.ndarray(
                        (BLOCK_COLS, BLOCK_ROWS),
                        dtype=nl.bfloat16,
                        buffer=nl.psum,
                    )
                    nisa.nc_transpose(
                        dst=gate_weight_transposed,
                        data=gate_weight_bf16,
                    )
                    nisa.nc_transpose(
                        dst=up_weight_transposed,
                        data=up_weight_bf16,
                    )
                    gate_weight = nl.ndarray(
                        (BLOCK_COLS, BLOCK_ROWS),
                        dtype=expert_gate_weight.dtype,
                        buffer=nl.sbuf,
                    )
                    up_weight = nl.ndarray(
                        (BLOCK_COLS, BLOCK_ROWS),
                        dtype=expert_up_weight.dtype,
                        buffer=nl.sbuf,
                    )
                    nisa.tensor_copy(dst=gate_weight, src=gate_weight_transposed)
                    nisa.tensor_copy(dst=up_weight, src=up_weight_transposed)
                    gate_partial_psum = nl.ndarray(
                        (BLOCK_ROWS, block_size), dtype=nl.float32, buffer=nl.psum
                    )
                    up_partial_psum = nl.ndarray(
                        (BLOCK_ROWS, block_size), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(
                        dst=gate_partial_psum,
                        stationary=gate_weight,
                        moving=hidden_transposed,
                        accumulate=False,
                    )
                    nisa.nc_matmul(
                        dst=up_partial_psum,
                        stationary=up_weight,
                        moving=hidden_transposed,
                        accumulate=False,
                    )
                    gate_partial = nl.ndarray(
                        (BLOCK_ROWS, block_size), dtype=nl.float32, buffer=nl.sbuf
                    )
                    up_partial = nl.ndarray(
                        (BLOCK_ROWS, block_size), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=gate_partial, src=gate_partial_psum)
                    nisa.tensor_copy(dst=up_partial, src=up_partial_psum)
                    gate_scale_scalar = nl.ndarray(
                        (1, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    up_scale_scalar = nl.ndarray(
                        (1, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    scale_offset = (
                        intermediate_block * (hidden_size // BLOCK_COLS)
                        + hidden_block_index
                    )
                    nisa.dma_copy(
                        dst=gate_scale_scalar,
                        src=expert_gate_scale.ap(pattern=[[1, 1]], offset=scale_offset),
                    )
                    nisa.dma_copy(
                        dst=up_scale_scalar,
                        src=expert_up_scale.ap(pattern=[[1, 1]], offset=scale_offset),
                    )
                    gate_scale = nl.ndarray(
                        (BLOCK_ROWS, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    up_scale = nl.ndarray(
                        (BLOCK_ROWS, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    scale_shuffle_mask = [0] * 32
                    for scale_channel in range(4):
                        scale_channel_start = scale_channel * 32
                        nisa.nc_stream_shuffle(
                            dst=gate_scale[
                                scale_channel_start : scale_channel_start + 32, 0:1
                            ],
                            src=gate_scale_scalar,
                            shuffle_mask=scale_shuffle_mask,
                        )
                        nisa.nc_stream_shuffle(
                            dst=up_scale[
                                scale_channel_start : scale_channel_start + 32, 0:1
                            ],
                            src=up_scale_scalar,
                            shuffle_mask=scale_shuffle_mask,
                        )
                    gate_scaled = nl.ndarray(
                        (BLOCK_ROWS, block_size), dtype=nl.float32, buffer=nl.sbuf
                    )
                    up_scaled = nl.ndarray(
                        (BLOCK_ROWS, block_size), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_scalar(
                        dst=gate_scaled,
                        data=gate_partial,
                        op0=nl.multiply,
                        operand0=gate_scale,
                    )
                    nisa.tensor_scalar(
                        dst=up_scaled,
                        data=up_partial,
                        op0=nl.multiply,
                        operand0=up_scale,
                    )
                    nisa.tensor_tensor(
                        dst=gate_accumulated,
                        data1=gate_accumulated,
                        data2=gate_scaled,
                        op=nl.add,
                    )
                    nisa.tensor_tensor(
                        dst=up_accumulated,
                        data1=up_accumulated,
                        data2=up_scaled,
                        op=nl.add,
                    )

                gate_rounded = nl.ndarray(
                    (BLOCK_ROWS, block_size),
                    dtype=hidden_states.dtype,
                    buffer=nl.sbuf,
                )
                up_rounded = nl.ndarray(
                    (BLOCK_ROWS, block_size), dtype=hidden_states.dtype, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=gate_rounded, src=gate_accumulated)
                nisa.tensor_copy(dst=up_rounded, src=up_accumulated)
                gate_activated = nl.ndarray(
                    (BLOCK_ROWS, block_size),
                    dtype=hidden_states.dtype,
                    buffer=nl.sbuf,
                )
                intermediate = nl.ndarray(
                    (BLOCK_ROWS, block_size), dtype=hidden_states.dtype, buffer=nl.sbuf
                )
                nisa.activation(dst=gate_activated, data=gate_rounded, op=nl.silu)
                nisa.tensor_tensor(
                    dst=intermediate,
                    data1=gate_activated,
                    data2=up_rounded,
                    op=nl.multiply,
                )
                nisa.dma_copy(
                    dst=intermediate_hbm[
                        shard_id,
                        intermediate_block
                        * BLOCK_ROWS : (intermediate_block + 1)
                        * BLOCK_ROWS,
                        0:block_size,
                    ],
                    src=intermediate,
                )

            affinity = nl.ndarray((block_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=affinity, value=0.0)
            affinity_address = nl.ndarray(
                (block_size, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=affinity_address,
                data=token_ids,
                op0=nl.multiply,
                operand0=num_experts,
            )
            expert_id_broadcast = nl.ndarray(
                (BLOCK_ROWS, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            shuffle_mask = [0] * 32
            for channel_index in range(4):
                channel_start = channel_index * 32
                nisa.nc_stream_shuffle(
                    dst=expert_id_broadcast[channel_start : channel_start + 32, 0:1],
                    src=expert_id.ap(pattern=[[1, 1], [1, 1]], offset=0),
                    shuffle_mask=shuffle_mask,
                )
            nisa.tensor_tensor(
                dst=affinity_address,
                data1=affinity_address,
                data2=expert_id_broadcast[0:block_size, 0:1],
                op=nl.add,
            )
            nisa.dma_copy(
                dst=affinity,
                src=expert_affinities_masked.ap(
                    pattern=[[1, block_size], [1, 1]],
                    offset=0,
                    vector_offset=affinity_address,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )

            for output_block in nl.sequential_range(hidden_size // BLOCK_ROWS):
                output_accumulated = nl.ndarray(
                    (output_rows_per_shard, block_size),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                nisa.memset(dst=output_accumulated, value=0.0)
                for intermediate_block in nl.sequential_range(
                    intermediate_size // BLOCK_COLS
                ):
                    intermediate = nl.ndarray(
                        (BLOCK_COLS, block_size),
                        dtype=hidden_states.dtype,
                        buffer=nl.sbuf,
                    )
                    nisa.dma_copy(
                        dst=intermediate,
                        src=intermediate_hbm[
                            shard_id,
                            intermediate_block
                            * BLOCK_COLS : (intermediate_block + 1)
                            * BLOCK_COLS,
                            0:block_size,
                        ],
                    )
                    down_weight_row_major = nl.ndarray(
                        (output_rows_per_shard, BLOCK_COLS),
                        dtype=expert_down_weight.dtype,
                        buffer=nl.sbuf,
                    )
                    down_offset = (
                        output_block * BLOCK_ROWS * intermediate_size
                        + shard_id * output_rows_per_shard * intermediate_size
                        + intermediate_block * BLOCK_COLS
                    )
                    down_weight_pattern = [
                        [intermediate_size, output_rows_per_shard],
                        [1, BLOCK_COLS],
                    ]
                    nisa.dma_copy(
                        dst=down_weight_row_major,
                        src=expert_down_weight.ap(
                            pattern=down_weight_pattern, offset=down_offset
                        ),
                    )
                    down_weight_bf16 = nl.ndarray(
                        (output_rows_per_shard, BLOCK_COLS),
                        dtype=nl.bfloat16,
                        buffer=nl.sbuf,
                    )
                    nisa.tensor_copy(dst=down_weight_bf16, src=down_weight_row_major)
                    down_weight_transposed = nl.ndarray(
                        (BLOCK_COLS, output_rows_per_shard),
                        dtype=nl.bfloat16,
                        buffer=nl.psum,
                    )
                    nisa.nc_transpose(
                        dst=down_weight_transposed,
                        data=down_weight_bf16,
                    )
                    down_weight = nl.ndarray(
                        (BLOCK_COLS, output_rows_per_shard),
                        dtype=expert_down_weight.dtype,
                        buffer=nl.sbuf,
                    )
                    nisa.tensor_copy(dst=down_weight, src=down_weight_transposed)
                    down_partial_psum = nl.ndarray(
                        (output_rows_per_shard, block_size),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    nisa.nc_matmul(
                        dst=down_partial_psum,
                        stationary=down_weight,
                        moving=intermediate,
                        accumulate=False,
                    )
                    down_partial = nl.ndarray(
                        (output_rows_per_shard, block_size),
                        dtype=nl.float32,
                        buffer=nl.sbuf,
                    )
                    nisa.tensor_copy(dst=down_partial, src=down_partial_psum)
                    down_scale_scalar = nl.ndarray(
                        (1, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    down_scale_offset = (
                        output_block * (intermediate_size // BLOCK_COLS)
                        + intermediate_block
                    )
                    nisa.dma_copy(
                        dst=down_scale_scalar,
                        src=expert_down_scale.ap(
                            pattern=[[1, 1]], offset=down_scale_offset
                        ),
                    )
                    down_scale = nl.ndarray(
                        (output_rows_per_shard, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    down_scale_shuffle_mask = [0] * 32
                    for down_scale_channel in range(2):
                        down_scale_start = down_scale_channel * 32
                        nisa.nc_stream_shuffle(
                            dst=down_scale[
                                down_scale_start : down_scale_start + 32, 0:1
                            ],
                            src=down_scale_scalar,
                            shuffle_mask=down_scale_shuffle_mask,
                        )
                    down_scaled = nl.ndarray(
                        (output_rows_per_shard, block_size),
                        dtype=nl.float32,
                        buffer=nl.sbuf,
                    )
                    nisa.tensor_scalar(
                        dst=down_scaled,
                        data=down_partial,
                        op0=nl.multiply,
                        operand0=down_scale,
                    )
                    nisa.tensor_tensor(
                        dst=output_accumulated,
                        data1=output_accumulated,
                        data2=down_scaled,
                        op=nl.add,
                    )

                output_transpose_psum = nl.ndarray(
                    (block_size, output_rows_per_shard),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                nisa.nc_transpose(dst=output_transpose_psum, data=output_accumulated)
                output_block_sbuf = nl.ndarray(
                    (block_size, output_rows_per_shard),
                    dtype=hidden_states.dtype,
                    buffer=nl.sbuf,
                )
                nisa.tensor_copy(dst=output_block_sbuf, src=output_transpose_psum)
                scaled_output = nl.ndarray(
                    (block_size, output_rows_per_shard),
                    dtype=hidden_states.dtype,
                    buffer=nl.sbuf,
                )
                nisa.tensor_scalar(
                    dst=scaled_output,
                    data=output_block_sbuf,
                    op0=nl.multiply,
                    operand0=affinity,
                )
                output_access = output.ap(
                    pattern=[[hidden_size, block_size], [1, output_rows_per_shard]],
                    offset=(
                        output_block * BLOCK_ROWS + shard_id * output_rows_per_shard
                    ),
                    vector_offset=token_ids,
                    indirect_dim=0,
                )
                nisa.dma_compute(
                    dst=output_access,
                    srcs=[output_access, scaled_output],
                    reduce_op=nl.add,
                    unique_indices=True,
                    oob_mode=oob_mode.skip,
                )

            nisa.tensor_scalar(
                dst=block_index, data=block_index, op0=nl.add, operand0=1
            )
    return output


_wrapped_selective_block_fp8_moe = wrap_nki(_selective_block_fp8_moe_nki)


def selective_block_fp8_moe_nki(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    conditions: torch.Tensor,
    gate_weight_0: torch.Tensor,
    gate_scale_0: torch.Tensor,
    up_weight_0: torch.Tensor,
    up_scale_0: torch.Tensor,
    down_weight_0: torch.Tensor,
    down_scale_0: torch.Tensor,
    gate_weight_1: torch.Tensor,
    gate_scale_1: torch.Tensor,
    up_weight_1: torch.Tensor,
    up_scale_1: torch.Tensor,
    down_weight_1: torch.Tensor,
    down_scale_1: torch.Tensor,
    gate_weight_2: torch.Tensor,
    gate_scale_2: torch.Tensor,
    up_weight_2: torch.Tensor,
    up_scale_2: torch.Tensor,
    down_weight_2: torch.Tensor,
    down_scale_2: torch.Tensor,
    gate_weight_3: torch.Tensor,
    gate_scale_3: torch.Tensor,
    up_weight_3: torch.Tensor,
    up_scale_3: torch.Tensor,
    down_weight_3: torch.Tensor,
    down_scale_3: torch.Tensor,
    *,
    block_size: int = 128,
) -> torch.Tensor:
    """Launch the selective checkpoint-native FP8 MoE kernel."""

    return _wrapped_selective_block_fp8_moe[2](
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        conditions=conditions,
        gate_weight_0=gate_weight_0,
        gate_scale_0=gate_scale_0,
        up_weight_0=up_weight_0,
        up_scale_0=up_scale_0,
        down_weight_0=down_weight_0,
        down_scale_0=down_scale_0,
        gate_weight_1=gate_weight_1,
        gate_scale_1=gate_scale_1,
        up_weight_1=up_weight_1,
        up_scale_1=up_scale_1,
        down_weight_1=down_weight_1,
        down_scale_1=down_scale_1,
        gate_weight_2=gate_weight_2,
        gate_scale_2=gate_scale_2,
        up_weight_2=up_weight_2,
        up_scale_2=up_scale_2,
        down_weight_2=down_weight_2,
        down_scale_2=down_scale_2,
        gate_weight_3=gate_weight_3,
        gate_scale_3=gate_scale_3,
        up_weight_3=up_weight_3,
        up_scale_3=up_scale_3,
        down_weight_3=down_weight_3,
        down_scale_3=down_scale_3,
        block_size=block_size,
    )
