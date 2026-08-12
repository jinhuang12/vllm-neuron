# SPDX-License-Identifier: Apache-2.0
"""Dense and shared SwiGLU MLPs for the pinned GLM-5.2 topology."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .block_fp8 import BLOCK_COLS, BLOCK_ROWS, BlockFP8Linear
from .config import GlmMoeDsaConfig


class GlmMoeDsaSwiGLUMLP(nn.Module):
    """One TP-sharded SwiGLU MLP contribution.

    Gate and up projections are column-sharded. The down projection is
    row-sharded. The caller sums the returned local contributions across TP.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        fp8_weights: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("hidden and intermediate sizes must be positive")
        if tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if intermediate_size % tensor_parallel_size:
            raise ValueError(
                f"intermediate_size={intermediate_size} is not divisible by "
                f"TP={tensor_parallel_size}"
            )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_rank = tensor_parallel_rank
        self.local_intermediate_size = intermediate_size // tensor_parallel_size
        shard_start = tensor_parallel_rank * self.local_intermediate_size
        if fp8_weights:
            self.gate_proj = BlockFP8Linear(
                hidden_size,
                self.local_intermediate_size,
                row_offset=shard_start % BLOCK_ROWS,
                device=device,
            )
            self.up_proj = BlockFP8Linear(
                hidden_size,
                self.local_intermediate_size,
                row_offset=shard_start % BLOCK_ROWS,
                device=device,
            )
            self.down_proj = BlockFP8Linear(
                self.local_intermediate_size,
                hidden_size,
                col_offset=shard_start % BLOCK_COLS,
                device=device,
            )
        else:
            self.gate_proj = nn.Linear(
                hidden_size,
                self.local_intermediate_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.up_proj = nn.Linear(
                hidden_size,
                self.local_intermediate_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.down_proj = nn.Linear(
                self.local_intermediate_size,
                hidden_size,
                bias=False,
                dtype=dtype,
                device=device,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)

    @classmethod
    def dense_from_config(
        cls,
        config: GlmMoeDsaConfig,
        *,
        tensor_parallel_size: int = 64,
        tensor_parallel_rank: int = 0,
        device: torch.device | str | None = None,
    ) -> GlmMoeDsaSwiGLUMLP:
        return cls(
            config.hidden_size,
            config.intermediate_size,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
            fp8_weights=bool(config.quantization_config),
            dtype=config.torch_dtype,
            device=device,
        )

    @classmethod
    def shared_from_config(
        cls,
        config: GlmMoeDsaConfig,
        *,
        tensor_parallel_size: int = 64,
        tensor_parallel_rank: int = 0,
        device: torch.device | str | None = None,
    ) -> GlmMoeDsaSwiGLUMLP:
        return cls(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
            fp8_weights=bool(config.quantization_config),
            dtype=config.torch_dtype,
            device=device,
        )
