# SPDX-License-Identifier: Apache-2.0
"""Vision patch-merge adapter for GLM-5.3-Flash (``glm5_next``).

WHAT THIS FILE IS FOR. The vision tower ends with three learned stages that turn ``spatial_merge_size²``
adjacent patch vectors into one language-model token: an RMSNorm, a spatially strided projection that also
widens the feature axis from the tower's ``hidden_size`` to the decoder's ``out_hidden_size``, and a gated
MLP. The reference spends five lines on them (``modeling_glm5_next.py:1812-1820``) and this file is their
Neuron-side twin. It owns no vision arithmetic: the patch grid and the patch ORDER arrive already decided
from the tower, exactly as they do in the reference.

THE ONE THING WORTH READING SLOWLY. The reference performs the strided projection with
``nn.Conv2d(in_channels=hidden_size, out_channels=out_hidden_size, kernel_size=spatial_merge_size,
stride=spatial_merge_size)`` (``modeling_glm5_next.py:1757-1762``), fed an NCHW tensor it builds with a
``view`` and a ``permute``. Because the kernel equals the stride and there is no padding, no dilation and no
grouping, that convolution sees each ``spatial_merge_size × spatial_merge_size`` block exactly once and
produces exactly one output vector per block. A convolution that visits each input element once is a single
matrix multiply wearing a different name:

    out[n, o] = sum over (c, i, j) of  W[o, c, i, j] * feat[n, i, j, c]  +  b[o]

So this file composes the same map as a metadata-only regroup followed by ONE linear, and the transpose the
reference needs only to satisfy ``nn.Conv2d``'s argument order disappears. Nothing is approximated: the two
forms are the same arithmetic on the same numbers.

THE INDEX ORDER IS THE WHOLE RISK. Flattening the block in the wrong order pairs each weight with the wrong
feature and still produces a tensor of the right shape, right dtype and plausible magnitude. The regroup
here reads ``(i, j, c)`` -- patch row, patch column, then channel -- which is the order the reference's own
``view(-1, sms, sms, hidden)`` already lays out in memory, so a plain ``reshape`` is exactly right. On a
contiguous input it is metadata only; on a non-contiguous one torch copies, and the arithmetic is the same
either way. :meth:`Glm5NextVisionAdapter.downsample_weight_from_conv` is the matching repack of the
reference's kernel, and it lives here rather than in the test so that the production path and the measured
path are the same lines. ``test_vision_adapter.py`` pins the pairing from both sides: the composed form is
compared against ``torch.nn.functional.conv2d`` on the same kernel, and a deliberately PERMUTED regroup is
required to FAIL that comparison.

WHAT THIS FILE DOES NOT DO. It declares no tensor-parallel sharding. The template it follows,
``Qwen3VLVisionPatchMerger`` (``qwen3_vl/vision_encoder_bf16.py:495-629``), shards its MLP across the vision
TP group and all-reduces; this adapter does not, because the tower composition owns the vision TP group and
sharding this stage would add a code path that nothing here measures. It also does not load checkpoint
weights: the repack above is the piece a loader needs, and the loader itself belongs to the tower.
"""
from __future__ import annotations

import torch
from torch import nn

from vllm_neuron.model.glm5_next.config import Glm5NextVisionConfig

# The reference resolves its second activation through ``ACT2FN[hidden_act]``
# (``modeling_glm5_next.py:1526``), a table of every activation transformers
# knows. Only the ones this checkpoint actually declares are accepted here: a
# silent fallback to some other activation would change what the merger computes
# and no shape check would notice. ``hidden_act`` reads ``"silu"`` in the
# checkpoint's ``vision_config``.
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


class Glm5NextVisionRMSNorm(nn.Module):
    """The tower's final RMSNorm, matching ``Glm5NextRMSNorm`` step for step.

    The reference (``modeling_glm5_next.py:1541-1555``) casts to float32, takes the mean square over the
    last axis, scales by its reciprocal square root, casts BACK to the input dtype, and only then multiplies
    by the learned weight. The order of those last two steps is not cosmetic: multiplying before the cast
    would round a different product, so it is reproduced here rather than tidied.

    Args:
        hidden_size: width of the axis normalised over.
        eps: variance epsilon. The tower's own, from ``Glm5NextVisionConfig.rms_norm_eps``, which the
            checkpoint declares separately from the decoder's.
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
    """The merger MLP, matching ``Glm5NextVisionPatchMerger`` stage for stage.

    Five learned maps and two activations (``modeling_glm5_next.py:1517-1537``): a square projection, a
    LayerNorm, a GELU, then a clamped SwiGLU whose gate and up projections are SEPARATE tensors in this
    checkpoint -- not one fused ``gate_up_proj``. Every linear is bias-free, as the reference's
    ``bias: bool = False`` default makes them.

    The clamp is asymmetric and that asymmetry is the checkpoint's, not a typo: ``gate`` is bounded from
    ABOVE only, ``up`` on BOTH sides, and the bound is ``swiglu_limit``. Clamping ``gate`` from below as
    well would quietly change the gate's response to negative pre-activations.

    Args:
        dim: the merged token width, i.e. ``out_hidden_size``.
        context_dim: the MLP's inner width, i.e. ``projection_intermediate_size``.
        hidden_act: name of the gate activation; must be a key of ``_ACTIVATIONS``.
        swiglu_limit: the clamp bound the checkpoint declares.
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
        # <-- MODEL-SPECIFIC: the checkpoint clamps gate from above only and up
        # on both sides, then multiplies (modeling_glm5_next.py:1535-1537).
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextVisionAdapter(nn.Module):
    """Norm, strided projection and merger MLP, composed as the reference composes them.

    ``forward`` is the whole adapter and returns what the reference returns as ``pooler_output``. The
    reference also exposes the tensor BEFORE the merger as ``last_hidden_state``
    (``modeling_glm5_next.py:1821-1824``); a caller that needs it calls :meth:`regroup_and_downsample` and
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

        The kernel arrives as ``[out_channels, in_channels, kh, kw]`` -- the layout ``nn.Conv2d`` holds and
        the layout a checkpoint stores. The regroup in :meth:`regroup_and_downsample` reads the block in
        ``(i, j, c)`` order, so the weight has to be read the same way: move ``in_channels`` last, flatten,
        and transpose for the right-multiply.

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
        # Metadata-only: the (i, j, c) order this reads is the order the
        # reference's own view(-1, sms, sms, hidden) already has in memory.
        regrouped = hidden_states.reshape(tokens // self.spatial_merge_unit, self.merged_hidden_size)
        return regrouped @ self.downsample_weight + self.downsample_bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the adapter.

        Args:
            hidden_states: ``[..., hidden_size]`` tower output, patch order already decided upstream.

        Returns:
            ``[num_merged_tokens, out_hidden_size]`` -- the reference's ``pooler_output``.
        """
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.regroup_and_downsample(hidden_states)
        return self.merger(hidden_states)
