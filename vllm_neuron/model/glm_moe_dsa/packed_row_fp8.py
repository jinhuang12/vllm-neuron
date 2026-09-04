# SPDX-License-Identifier: Apache-2.0
"""Load-time packing for GLM-5.2 row-scaled FP8 routed experts."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .block_fp8 import quantize_block_fp8_to_row

BlockFP8Pair = tuple[torch.Tensor, torch.Tensor]
GateUpBlockFP8Pair = tuple[BlockFP8Pair, BlockFP8Pair]


def pack_gate_up_row_fp8_bank(
    experts: Sequence[GateUpBlockFP8Pair],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert and pack gate/up checkpoint pairs as ``[E,H,2,I]``."""

    packed_weights = []
    packed_scales = []
    for gate_pair, up_pair in experts:
        gate_weight, gate_scale = quantize_block_fp8_to_row(*gate_pair)
        up_weight, up_scale = quantize_block_fp8_to_row(*up_pair)
        packed_weights.append(torch.stack((gate_weight.T, up_weight.T), dim=1))
        packed_scales.append(torch.stack((gate_scale, up_scale), dim=0))
    return torch.stack(packed_weights, dim=0), torch.stack(packed_scales, dim=0)


def pack_down_row_fp8_bank(
    experts: Sequence[BlockFP8Pair],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert and pack down checkpoint pairs as ``[E,I,H]``."""

    converted = [quantize_block_fp8_to_row(*pair) for pair in experts]
    return (
        torch.stack([weight.T for weight, _ in converted], dim=0),
        torch.stack([scale for _, scale in converted], dim=0),
    )


class PackedRowFP8Banks(nn.Module):
    """Final parameter layouts consumed directly by ``NF.moe_tkg``."""

    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_weights = nn.Parameter(
            torch.empty(
                (num_local_experts, hidden_size, 2, intermediate_size),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.down_weights = nn.Parameter(
            torch.empty(
                (num_local_experts, intermediate_size, hidden_size),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.gate_up_scales = nn.Parameter(
            torch.empty(
                (num_local_experts, 2, intermediate_size),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )
        self.down_scales = nn.Parameter(
            torch.empty(
                (num_local_experts, hidden_size),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )
