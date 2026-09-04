# SPDX-License-Identifier: Apache-2.0
"""128x128 block-scaled FP8 linear execution for GLM-5.2 on Trn2."""

from __future__ import annotations

import math
import warnings

import nki
import nki.isa as nisa
import nki.language as nl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from nkilib.core.utils.tensor_view import TensorView
from torch import nn
from torch.distributed._functional_collectives import all_gather_tensor, all_reduce

from vllm_neuron.envs import is_native_backend
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear
from vllm_neuron.utils.neuron_utils import can_run_kernel

BLOCK_ROWS = 128
BLOCK_COLS = 128
TRN2_FP8_MAX = 240.0
OCP_FP8_MAX = 448.0
FP8_STORAGE_SCALE = TRN2_FP8_MAX / OCP_FP8_MAX
FP8_INVERSE_SCALE_ADJUSTMENT = OCP_FP8_MAX / TRN2_FP8_MAX


def _kernel_assert(condition: bool, message: str) -> None:
    assert condition, "[INTERNAL_ERROR] [NCC_INKI016] " + message


@nki.jit
def _block_fp8_linear_nki(
    hidden,
    weight,
    weight_scale_inv,
    row_offset=0,
    col_offset=0,
):
    """Compute BF16 hidden @ dequant(FP8 weight).T using FP32 PSUM."""

    _kernel_assert(len(hidden.shape) == 2, "hidden must be rank 2")
    _kernel_assert(len(weight.shape) == 2, "weight must be rank 2")
    _kernel_assert(len(weight_scale_inv.shape) == 2, "scale must be rank 2")
    _kernel_assert(hidden.shape[1] == weight.shape[1], "linear K mismatch")
    _kernel_assert(0 <= row_offset < BLOCK_ROWS, "invalid row block offset")
    _kernel_assert(0 <= col_offset < BLOCK_COLS, "invalid column block offset")

    token_count = hidden.shape[0]
    output_width = weight.shape[0]
    contraction_width = weight.shape[1]
    output_blocks = (row_offset + output_width + BLOCK_ROWS - 1) // BLOCK_ROWS
    contraction_blocks = (col_offset + contraction_width + BLOCK_COLS - 1) // BLOCK_COLS
    _kernel_assert(
        weight_scale_inv.shape[0] == output_blocks,
        "scale rows do not cover the local output shard",
    )
    _kernel_assert(
        weight_scale_inv.shape[1] == contraction_blocks,
        "scale columns do not cover the local input shard",
    )

    output = nl.ndarray(
        (token_count, output_width), dtype=hidden.dtype, buffer=nl.shared_hbm
    )
    program_count = nl.num_programs(axes=0)
    program_id = nl.program_id(axis=0)
    blocks_per_program = (output_blocks + program_count - 1) // program_count
    first_output_block = program_id * blocks_per_program

    for local_output_block in nl.affine_range(blocks_per_program):
        output_block = first_output_block + local_output_block
        output_start = max(0, output_block * BLOCK_ROWS - row_offset)
        output_end = min(output_width, (output_block + 1) * BLOCK_ROWS - row_offset)
        output_size = max(0, output_end - output_start)

        if output_size > 0:
            for token_block in nl.affine_range(
                (token_count + BLOCK_ROWS - 1) // BLOCK_ROWS
            ):
                token_start = token_block * BLOCK_ROWS
                token_size = min(BLOCK_ROWS, token_count - token_start)
                accumulated = nl.ndarray(
                    (token_size, output_size), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.memset(dst=accumulated, value=0.0)

                for contraction_block in nl.affine_range(contraction_blocks):
                    contraction_start = max(
                        0, contraction_block * BLOCK_COLS - col_offset
                    )
                    contraction_end = min(
                        contraction_width,
                        (contraction_block + 1) * BLOCK_COLS - col_offset,
                    )
                    contraction_size = max(0, contraction_end - contraction_start)

                    if contraction_size > 0:
                        hidden_transposed = nl.ndarray(
                            (contraction_size, token_size),
                            dtype=hidden.dtype,
                            buffer=nl.sbuf,
                        )
                        nisa.dma_transpose(
                            dst=hidden_transposed,
                            src=hidden[
                                token_start : token_start + token_size,
                                contraction_start : contraction_start
                                + contraction_size,
                            ],
                            axes=(1, 0),
                        )
                        weight_tile = nl.ndarray(
                            (output_size, contraction_size),
                            dtype=weight.dtype,
                            buffer=nl.sbuf,
                        )
                        nisa.dma_copy(
                            dst=weight_tile,
                            src=weight[
                                output_start : output_start + output_size,
                                contraction_start : contraction_start
                                + contraction_size,
                            ],
                        )
                        weight_bf16 = nl.ndarray(
                            (output_size, contraction_size),
                            dtype=nl.bfloat16,
                            buffer=nl.sbuf,
                        )
                        nisa.tensor_copy(dst=weight_bf16, src=weight_tile)
                        weight_transpose_psum = nl.ndarray(
                            (contraction_size, output_size),
                            dtype=nl.bfloat16,
                            buffer=nl.psum,
                        )
                        nisa.nc_transpose(dst=weight_transpose_psum, data=weight_bf16)
                        weight_transposed = nl.ndarray(
                            (contraction_size, output_size),
                            dtype=weight.dtype,
                            buffer=nl.sbuf,
                        )
                        nisa.tensor_copy(
                            dst=weight_transposed, src=weight_transpose_psum
                        )
                        partial_psum = nl.ndarray(
                            (token_size, output_size),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        nisa.nc_matmul(
                            dst=partial_psum,
                            stationary=hidden_transposed,
                            moving=weight_transposed,
                            accumulate=False,
                        )
                        partial = nl.ndarray(
                            (token_size, output_size),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        nisa.tensor_copy(dst=partial, src=partial_psum)
                        scale = nl.ndarray(
                            (token_size, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        nisa.dma_copy(
                            dst=scale,
                            src=TensorView(
                                weight_scale_inv[
                                    output_block : output_block + 1,
                                    contraction_block : contraction_block + 1,
                                ]
                            )
                            .broadcast(dim=0, size=token_size)
                            .get_view(),
                        )
                        scaled = nl.ndarray(
                            (token_size, output_size),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        nisa.tensor_scalar(
                            dst=scaled,
                            data=partial,
                            op0=nl.multiply,
                            operand0=scale,
                        )
                        nisa.tensor_tensor(
                            dst=accumulated,
                            data1=accumulated,
                            data2=scaled,
                            op=nl.add,
                        )

                result = nl.ndarray(
                    (token_size, output_size),
                    dtype=hidden.dtype,
                    buffer=nl.sbuf,
                )
                nisa.tensor_copy(dst=result, src=accumulated)
                nisa.dma_copy(
                    dst=output[
                        token_start : token_start + token_size,
                        output_start : output_start + output_size,
                    ],
                    src=result,
                )
    return output


_wrapped_block_fp8_linear = wrap_nki(_block_fp8_linear_nki)


def scale_shape_for_local_weight(
    weight_shape: tuple[int, int], row_offset: int, col_offset: int
) -> tuple[int, int]:
    return (
        math.ceil((row_offset + weight_shape[0]) / BLOCK_ROWS),
        math.ceil((col_offset + weight_shape[1]) / BLOCK_COLS),
    )


def dequantize_block_fp8(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    row_offset: int = 0,
    col_offset: int = 0,
) -> torch.Tensor:
    """Reference dequantization for tests and non-Neuron execution."""

    expected = scale_shape_for_local_weight(tuple(weight.shape), row_offset, col_offset)
    if tuple(weight_scale_inv.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(weight_scale_inv.shape)} does not cover "
            f"weight {tuple(weight.shape)} at offsets {(row_offset, col_offset)}; "
            f"expected {expected}"
        )
    rows = (torch.arange(weight.shape[0], device=weight.device) + row_offset) // 128
    cols = (torch.arange(weight.shape[1], device=weight.device) + col_offset) // 128
    expanded = weight_scale_inv[rows[:, None], cols[None, :]].float()
    return weight.float() * expanded


def quantize_block_fp8_to_row(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    row_offset: int = 0,
    col_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert checkpoint block-FP8 storage to Trn2 row-scaled FP8.

    The supported Trn2 MoE kernels consume one inverse scale per output row.
    Conversion happens in the CPU checkpoint loader, before device transfer,
    so the compiled graph retains FP8 weights and contains no full-weight
    dequantization operations.
    """

    dequantized = dequantize_block_fp8(
        weight,
        weight_scale_inv,
        row_offset=row_offset,
        col_offset=col_offset,
    )
    row_max = dequantized.abs().amax(dim=1)
    row_scale = row_max / TRN2_FP8_MAX
    row_scale = torch.where(row_max == 0, torch.ones_like(row_scale), row_scale)
    quantized = (dequantized / row_scale[:, None]).clamp(-TRN2_FP8_MAX, TRN2_FP8_MAX)
    return quantized.to(torch.float8_e4m3fn), row_scale.to(torch.float32)


def dequantize_row_fp8(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
) -> torch.Tensor:
    """Reference dequantization for row-scaled FP8 MoE weights."""

    if weight.ndim != 2:
        raise ValueError("row-FP8 weight must be rank 2")
    if tuple(weight_scale_inv.shape) != (weight.shape[0],):
        raise ValueError(
            f"row scale shape {tuple(weight_scale_inv.shape)} does not match "
            f"weight output rows {weight.shape[0]}"
        )
    return weight.float() * weight_scale_inv[:, None]


def block_fp8_linear(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    row_offset: int = 0,
    col_offset: int = 0,
) -> torch.Tensor:
    """Execute one block-scaled FP8 linear while retaining FP8 HBM storage."""

    original_shape = hidden.shape[:-1]
    hidden_2d = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    if can_run_kernel(hidden_2d):
        output = _wrapped_block_fp8_linear[2](
            hidden_2d,
            weight,
            weight_scale_inv,
            row_offset=row_offset,
            col_offset=col_offset,
        )
    else:
        output = F.linear(
            hidden_2d,
            dequantize_block_fp8(
                weight,
                weight_scale_inv,
                row_offset=row_offset,
                col_offset=col_offset,
            ).to(hidden_2d.dtype),
        )
    return output.reshape(*original_shape, weight.shape[0])


class _BlockFP8Mixin:
    row_offset: int
    col_offset: int

    def _allocate_block_fp8(
        self,
        shape: tuple[int, int],
        *,
        row_offset: int,
        col_offset: int,
        device: torch.device | str | None,
    ) -> None:
        self.row_offset = row_offset
        self.col_offset = col_offset
        self.weight = nn.Parameter(
            torch.empty(shape, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.empty(
                scale_shape_for_local_weight(shape, row_offset, col_offset),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )

    def _block_linear(self, hidden: torch.Tensor) -> torch.Tensor:
        return block_fp8_linear(
            hidden,
            self.weight,
            self.weight_scale_inv,
            row_offset=self.row_offset,
            col_offset=self.col_offset,
        )


class BlockFP8Linear(_BlockFP8Mixin, nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        row_offset: int = 0,
        col_offset: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter("bias", None)
        self._allocate_block_fp8(
            (out_features, in_features),
            row_offset=row_offset,
            col_offset=col_offset,
            device=device,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._block_linear(hidden)


class RowFP8Linear(BlockFP8Linear):
    """FP8 linear storage formatted for the supported Trn2 MoE kernels."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.row_offset = 0
        self.col_offset = 0
        self.register_parameter("bias", None)
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.empty((out_features,), dtype=torch.float32, device=device),
            requires_grad=False,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if can_run_kernel(hidden):
            raise RuntimeError(
                "RowFP8Linear must be dispatched through the fused MoE kernel"
            )
        return F.linear(
            hidden,
            dequantize_row_fp8(self.weight, self.weight_scale_inv).to(hidden.dtype),
        )


class BlockFP8ColumnParallelLinear(_BlockFP8Mixin, ColumnParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        gather_output: bool = False,
        device: torch.device | str | None = None,
        tp_group=None,
        tp_size_override: int | None = None,
        tp_rank_override: int = 0,
    ) -> None:
        nn.Module.__init__(self)
        if dist.is_initialized():
            self.tp_group = tp_group if tp_group is not None else dist.group.WORLD
            self.tp_size = dist.get_world_size(self.tp_group)
            self.tp_rank = dist.get_rank(self.tp_group)
        else:
            self.tp_group = None
            self.tp_size = tp_size_override or 1
            self.tp_rank = tp_rank_override
        if out_features % self.tp_size:
            raise ValueError("column-parallel output is not divisible by TP")
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_rank = out_features // self.tp_size
        self.gather_output = gather_output
        self.register_parameter("bias", None)
        local_start = self.tp_rank * self.out_features_per_rank
        self._allocate_block_fp8(
            (self.out_features_per_rank, in_features),
            row_offset=local_start % BLOCK_ROWS,
            col_offset=0,
            device=device,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        local_output = self._block_linear(hidden)
        if self.tp_size == 1 or not self.gather_output:
            return local_output
        return all_gather_tensor(local_output, 1, self.tp_group)


class BlockFP8RowParallelLinear(_BlockFP8Mixin, RowParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        input_is_parallel: bool = True,
        device: torch.device | str | None = None,
        tp_group=None,
        tp_size_override: int | None = None,
        tp_rank_override: int = 0,
    ) -> None:
        nn.Module.__init__(self)
        if dist.is_initialized():
            self.tp_group = tp_group if tp_group is not None else dist.group.WORLD
            self.tp_size = dist.get_world_size(self.tp_group)
            self.tp_rank = dist.get_rank(self.tp_group)
        else:
            self.tp_group = None
            self.tp_size = tp_size_override or 1
            self.tp_rank = tp_rank_override
        if in_features % self.tp_size:
            raise ValueError("row-parallel input is not divisible by TP")
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_rank = in_features // self.tp_size
        self.input_is_parallel = input_is_parallel
        if not input_is_parallel:
            warnings.warn(
                "input_is_parallel=False produces non-SPMD code", stacklevel=2
            )
        self.register_parameter("bias", None)
        local_start = self.tp_rank * self.in_features_per_rank
        self._allocate_block_fp8(
            (out_features, self.in_features_per_rank),
            row_offset=0,
            col_offset=local_start % BLOCK_COLS,
            device=device,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel and self.tp_size > 1:
            hidden = torch.chunk(hidden, self.tp_size, dim=-1)[self.tp_rank]
        local_output = self._block_linear(hidden)
        if self.tp_size == 1:
            return local_output
        if is_native_backend():
            return all_reduce(local_output, reduceOp="sum", group=self.tp_group)
        dist.all_reduce(local_output, op=dist.ReduceOp.SUM, group=self.tp_group)
        return local_output


def fp8_local_linear(
    in_features: int,
    out_features: int,
    *,
    row_offset: int = 0,
    col_offset: int = 0,
    device: torch.device | str | None = None,
) -> BlockFP8Linear:
    return BlockFP8Linear(
        in_features,
        out_features,
        row_offset=row_offset,
        col_offset=col_offset,
        device=device,
    )
