# SPDX-License-Identifier: Apache-2.0
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader,
    SafetensorsWeightLoader,
)
from torch.distributed._functional_collectives import all_gather_tensor

# torch 2.11+ marks _maybe_view_chunk_cat as skip in dynamo, breaking
# all_gather_tensor with gather_dim != 0 under torch.compile(fullgraph=True).
try:
    from torch._utils import _maybe_view_chunk_cat

    torch._dynamo.allow_in_graph(_maybe_view_chunk_cat)
except (ImportError, AttributeError):
    pass


class ColumnParallelLinear(nn.Module):
    """
    TODO: Add docstring
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        device=None,
        dtype=None,
        tp_group=None,
    ):
        super().__init__()

        # Use TP group if provided, otherwise fall back to world
        if dist.is_initialized():
            self.tp_group = tp_group if tp_group is not None else dist.group.WORLD
            self.tp_size = dist.get_world_size(self.tp_group)
            self.tp_rank = dist.get_rank(self.tp_group)
        else:
            self.tp_group = None
            self.tp_size = 1
            self.tp_rank = 0

        self.in_features = in_features
        self.out_features = out_features
        # TODO: Add flag to enable padding
        assert out_features % self.tp_size == 0, (
            f"CPL layer out_features {out_features} is not evenly divisible by tp_size {self.tp_size}"
        )
        self.out_features_per_rank = out_features // self.tp_size
        self.weight = nn.Parameter(
            torch.empty(
                (self.out_features_per_rank, in_features), device=device, dtype=dtype
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_rank, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.gather_output = gather_output

        # Set weight loaders for weight and bias parameters
        weight_loader: SafetensorsWeightLoader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.out_features_per_rank,
            num_shards=self.tp_size,
        )
        set_weight_loader(self.weight, weight_loader)
        if bias:
            set_weight_loader(self.bias, weight_loader)

    def reset_parameters(self) -> None:
        # Match expectation to Linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"

        # PyTorch handles missing keys by adding them to missing_keys list
        if weight_key in state_dict:
            full_weight = state_dict[weight_key]
            # We only shard if the weight is not pre-sharded
            if not full_weight.shape == self.weight.data.shape:
                start_idx = self.tp_rank * self.out_features_per_rank
                end_idx = start_idx + self.out_features_per_rank
                state_dict[weight_key] = full_weight[start_idx:end_idx]

        # PyTorch handles missing keys by adding them to missing_keys list
        if bias_key in state_dict and self.bias is not None:
            full_bias = state_dict[bias_key]
            # We only shard if the weight is not pre-sharded
            if not full_weight.shape == self.bias.data.shape:
                start_idx = self.tp_rank * self.out_features_per_rank
                end_idx = start_idx + self.out_features_per_rank
                state_dict[bias_key] = full_bias[start_idx:end_idx]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        local_output = F.linear(x, self.weight, self.bias)

        if self.tp_size == 1 or not self.gather_output:
            return local_output

        output = all_gather_tensor(local_output, 1, self.tp_group)

        return output
