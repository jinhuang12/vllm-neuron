# SPDX-License-Identifier: Apache-2.0
"""Vision patch-merge adapter for GLM-5.3-Flash (``glm5_next``).

The three learned stages that close the vision tower: an RMSNorm, a spatially strided projection that widens
the feature axis from the tower's ``hidden_size`` to the decoder's ``out_hidden_size``, and a gated MLP.
Together they turn ``spatial_merge_size**2`` adjacent patch vectors into one language-model token. The patch
grid and the patch order arrive already decided by the tower.

The reference performs the strided projection with ``nn.Conv2d(hidden_size, out_hidden_size,
kernel_size=spatial_merge_size, stride=spatial_merge_size)``. Kernel equals stride with no padding, dilation
or grouping, so the convolution visits every input element once and is therefore one matrix multiply:

    out[n, o] = sum over (c, i, j) of  W[o, c, i, j] * feat[n, i, j, c]  +  b[o]

This module composes the identical map as a regroup followed by one linear, which drops the NCHW transpose
that only ``nn.Conv2d``'s argument order required. The regroup reads ``(i, j, c)`` -- patch row, patch
column, then channel -- the order the reference's own ``view(-1, sms, sms, hidden)`` already has in memory,
so a plain ``reshape`` is correct. Any other flattening order pairs each weight with the wrong feature and
still yields a tensor of the right shape and dtype, so :meth:`Glm5NextVisionAdapter.downsample_weight_from_conv`
repacks the reference kernel to match.

This adapter has no tensor-parallel sharding: the tower composition owns the vision TP group. It also loads
no checkpoint weights; that belongs to the tower.
"""
from __future__ import annotations

import torch
from torch import nn

from vllm_neuron.model.glm5_next.config import Glm5NextVisionConfig

# The reference resolves this activation through transformers' ``ACT2FN`` table.
# Only the activations the checkpoint uses are accepted here, because a silent
# fallback to another activation changes what the merger computes and no shape
# check would notice.
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


class Glm5NextVisionRMSNorm(nn.Module):
    """The tower's final RMSNorm, matching the reference ``Glm5NextRMSNorm``.

    Normalises in float32, casts back to the input dtype, and only then multiplies by the learned weight.
    That order matters: multiplying before the cast rounds a different product.

    Args:
        hidden_size: width of the axis normalised over.
        eps: variance epsilon, from ``Glm5NextVisionConfig.rms_norm_eps``. The checkpoint carries a separate
            value for the tower and the decoder.
        dtype: weight dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Glm5NextVisionMergerMLP(nn.Module):
    """The merger MLP, matching the reference ``Glm5NextVisionPatchMerger``.

    Five bias-free linears and two activations: a square projection, a LayerNorm, a GELU, then a clamped
    SwiGLU. The gate and up projections are separate tensors in this checkpoint, not one fused
    ``gate_up_proj``.

    The clamp is deliberately asymmetric: ``gate`` is bounded from above only and ``up`` on both sides, both
    at ``swiglu_limit``. Clamping ``gate`` from below too would change its response to negative
    pre-activations.

    Args:
        dim: the merged token width, i.e. ``out_hidden_size``.
        context_dim: the MLP's inner width, i.e. ``projection_intermediate_size``.
        hidden_act: name of the gate activation; must be a key of ``_ACTIVATIONS``.
        swiglu_limit: the clamp bound from the checkpoint.
        dtype: weight dtype.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_act: str,
        swiglu_limit: float,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if hidden_act not in _ACTIVATIONS:
            raise ValueError(
                f"hidden_act {hidden_act!r} is not one this adapter implements "
                f"(known: {sorted(_ACTIVATIONS)}). The reference resolves it through "
                f"transformers' ACT2FN table; add it here deliberately rather than "
                f"falling back to another activation."
            )
        self.proj = nn.Linear(dim, dim, bias=False, dtype=dtype)
        self.post_projection_norm = nn.LayerNorm(dim, dtype=dtype)
        self.gate_proj = nn.Linear(dim, context_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(dim, context_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(context_dim, dim, bias=False, dtype=dtype)
        self.act1 = nn.GELU()
        self.act_fn = _ACTIVATIONS[hidden_act]()
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = self.act1(self.post_projection_norm(hidden_states))
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        # <-- MODEL-SPECIFIC: gate is clamped from above only, up on both sides.
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextVisionAdapter(nn.Module):
    """Norm, strided projection and merger MLP, composed as the reference composes them.

    :meth:`forward` returns the reference's ``pooler_output``. To get the reference's
    ``last_hidden_state``, the tensor before the merger, call :meth:`regroup_and_downsample` and
    :attr:`merger` in turn instead of ``forward``.

    Args:
        config: the vision config; supplies ``hidden_size``, ``out_hidden_size``, ``spatial_merge_size``,
            ``projection_intermediate_size``, ``hidden_act``, ``swiglu_limit`` and ``rms_norm_eps``.
        dtype: weight dtype.
    """

    def __init__(
        self,
        config: Glm5NextVisionConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.out_hidden_size = config.out_hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size**2
        self.merged_hidden_size = config.hidden_size * self.spatial_merge_unit

        self.post_layernorm = Glm5NextVisionRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        # The strided convolution as one linear. Weight is stored already
        # transposed for a right-multiply, so the forward is a plain ``@``.
        self.downsample_weight = nn.Parameter(
            torch.empty(self.merged_hidden_size, config.out_hidden_size, dtype=dtype)
        )
        self.downsample_bias = nn.Parameter(
            torch.empty(config.out_hidden_size, dtype=dtype)
        )
        self.merger = Glm5NextVisionMergerMLP(
            dim=config.out_hidden_size,
            context_dim=config.projection_intermediate_size,
            hidden_act=config.hidden_act,
            swiglu_limit=config.swiglu_limit,
            dtype=dtype,
        )

    def downsample_weight_from_conv(self, conv_weight: torch.Tensor) -> torch.Tensor:
        """Repack the reference's conv kernel into this module's linear weight.

        The kernel arrives in the ``[out_channels, in_channels, kh, kw]`` layout ``nn.Conv2d`` and the
        checkpoint use. :meth:`regroup_and_downsample` reads the block in ``(i, j, c)`` order, so the weight
        is read the same way: move ``in_channels`` last, flatten, and transpose for the right-multiply.

        Args:
            conv_weight: ``[out_hidden_size, hidden_size, spatial_merge_size, spatial_merge_size]``.

        Returns:
            ``[merged_hidden_size, out_hidden_size]``, ready to assign to :attr:`downsample_weight`.
        """
        out_channels = conv_weight.shape[0]
        expected = (
            self.out_hidden_size,
            self.hidden_size,
            self.spatial_merge_size,
            self.spatial_merge_size,
        )
        if tuple(conv_weight.shape) != expected:
            raise ValueError(
                f"conv kernel shape {tuple(conv_weight.shape)} does not match this "
                f"adapter's {expected}"
            )
        return conv_weight.permute(0, 2, 3, 1).reshape(out_channels, -1).t().contiguous()

    def regroup_and_downsample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Group ``spatial_merge_unit`` adjacent patches and project them in one matmul.

        Args:
            hidden_states: ``[..., hidden_size]``, already normalised, whose flattened token count is a
                multiple of ``spatial_merge_unit``.

        Returns:
            ``[num_merged_tokens, out_hidden_size]``.
        """
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"last dim {hidden_states.shape[-1]} is not hidden_size {self.hidden_size}"
            )
        tokens = hidden_states.numel() // self.hidden_size
        if tokens % self.spatial_merge_unit != 0:
            raise ValueError(
                f"token count {tokens} is not a multiple of spatial_merge_unit "
                f"{self.spatial_merge_unit}; merging would cross token-group bounds"
            )
        # The (i, j, c) order this reads is already the memory order, so on a
        # contiguous input the reshape is metadata only.
        regrouped = hidden_states.reshape(tokens // self.spatial_merge_unit, self.merged_hidden_size)
        return regrouped @ self.downsample_weight + self.downsample_bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the adapter.

        Args:
            hidden_states: ``[..., hidden_size]`` tower output, patch order already decided upstream.

        Returns:
            ``[num_merged_tokens, out_hidden_size]``, the reference's ``pooler_output``.
        """
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.regroup_and_downsample(hidden_states)
        return self.merger(hidden_states)
