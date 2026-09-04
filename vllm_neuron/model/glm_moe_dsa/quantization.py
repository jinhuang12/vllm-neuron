# SPDX-License-Identifier: Apache-2.0
"""FP8 metadata contract for the pinned GLM-5.2 checkpoint.

This module describes checkpoint storage. It does not select an execution
kernel and it does not claim that model integration is complete.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ScaleCoverage:
    """Scale blocks needed by one tensor-parallel weight shard."""

    scale_slices: tuple[slice, ...]
    local_scale_shape: tuple[int, ...]
    weight_offset_in_first_block: int


@dataclass(frozen=True)
class Fp8BlockQuantization:
    """Pinned FP8 storage and activation contract.

    Weights use e4m3 storage. Activations are quantized dynamically, so the
    checkpoint has no static activation-scale tensors. Each inverse weight
    scale covers a 128 by 128 block, including partial edge blocks.
    """

    format: str = "e4m3"
    activation_scheme: str = "dynamic"
    weight_block_size: tuple[int, int] = (128, 128)
    scale_suffix: str = "weight_scale_inv"
    checkpoint_weight_dtype: str = "F8_E4M3"
    checkpoint_scale_dtype: str = "F32"
    torch_weight_dtype: torch.dtype = torch.float8_e4m3fn
    modules_to_not_convert: tuple[str, ...] = ()

    @classmethod
    def from_hf_config(cls, config: dict[str, Any]) -> Fp8BlockQuantization:
        """Validate and parse the checkpoint ``quantization_config``."""
        if not isinstance(config, dict):
            raise TypeError("quantization_config must be a mapping")

        expected = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
        mismatches = [
            f"{name}={config.get(name)!r} (expected {value!r})"
            for name, value in expected.items()
            if config.get(name) != value
        ]
        if mismatches:
            raise ValueError("Unsupported GLM-5.2 FP8 config: " + ", ".join(mismatches))

        exclusions = config.get("modules_to_not_convert", ())
        if not isinstance(exclusions, (list, tuple)) or not all(
            isinstance(item, str) for item in exclusions
        ):
            raise ValueError("modules_to_not_convert must be a sequence of names")
        return cls(modules_to_not_convert=tuple(exclusions))

    def expected_scale_shape(self, weight_shape: Sequence[int]) -> tuple[int, int]:
        """Return the inverse-scale grid for one two-dimensional weight."""
        if len(weight_shape) != 2 or any(int(size) <= 0 for size in weight_shape):
            raise ValueError(
                f"FP8 block weights must be positive 2-D tensors: {weight_shape}"
            )
        return (
            math.ceil(int(weight_shape[0]) / self.weight_block_size[0]),
            math.ceil(int(weight_shape[1]) / self.weight_block_size[1]),
        )

    def validate_header_pair(
        self,
        *,
        weight_dtype: str,
        weight_shape: Sequence[int],
        scale_dtype: str,
        scale_shape: Sequence[int],
    ) -> None:
        """Validate one FP8 weight and its inverse-scale header entry."""
        if weight_dtype != self.checkpoint_weight_dtype:
            raise ValueError(
                f"Expected {self.checkpoint_weight_dtype} weight, got {weight_dtype}"
            )
        if scale_dtype != self.checkpoint_scale_dtype:
            raise ValueError(
                f"Expected {self.checkpoint_scale_dtype} inverse scale, got {scale_dtype}"
            )
        expected = self.expected_scale_shape(weight_shape)
        if tuple(scale_shape) != expected:
            raise ValueError(
                f"Inverse-scale shape {tuple(scale_shape)} does not match "
                f"weight shape {tuple(weight_shape)} and block size "
                f"{self.weight_block_size}; expected {expected}"
            )

    def scale_coverage_for_weight_shard(
        self,
        weight_shape: Sequence[int],
        *,
        shard_dim: int | None,
        rank: int,
        world_size: int,
    ) -> ScaleCoverage:
        """Return scale-grid slices that cover a raw TP weight shard.

        A TP boundary can split a 128-element FP8 block. The returned offset
        records that case for later kernel/layout integration.
        """
        scale_shape = self.expected_scale_shape(weight_shape)
        if shard_dim is None:
            return ScaleCoverage(
                scale_slices=(slice(None), slice(None)),
                local_scale_shape=scale_shape,
                weight_offset_in_first_block=0,
            )
        if shard_dim not in (0, 1):
            raise ValueError(f"shard_dim must be 0, 1, or None; got {shard_dim}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside world_size {world_size}")
        dimension = int(weight_shape[shard_dim])
        if dimension % world_size:
            raise ValueError(
                f"weight dimension {dimension} is not divisible by world_size {world_size}"
            )

        local = dimension // world_size
        start = rank * local
        end = start + local
        block = self.weight_block_size[shard_dim]
        block_start = start // block
        block_end = math.ceil(end / block)
        slices = [slice(None), slice(None)]
        slices[shard_dim] = slice(block_start, block_end)
        local_scale_shape = list(scale_shape)
        local_scale_shape[shard_dim] = block_end - block_start
        return ScaleCoverage(
            scale_slices=tuple(slices),
            local_scale_shape=tuple(local_scale_shape),
            weight_offset_in_first_block=start % block,
        )


PINNED_FP8 = Fp8BlockQuantization()
