# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash vision tower.

Covers the transformer-block half of the reference's ``Glm5NextVisionModel.forward``. The
surrounding stages live elsewhere:

=====================================  ==================================
reference stage                        where it lives here
=====================================  ==================================
``patch_embed``                        the NKI patch-embed seam
``rotary_pos_emb`` + ``cos``/``sin``   ``utils/vision_rope.py``
``blocks`` loop                        this module
``post_layernorm``                     ``vision_adapter.py``
``view``/``permute``/``downsample``    ``vision_adapter.py``
``merger``                             ``vision_adapter.py``
=====================================  ==================================

:meth:`Glm5NextVisionEncoder.forward` therefore returns the reference's ``pooler_output``, the
merged ``out_hidden_size``-wide embeddings the encoder cache stores, not its
``last_hidden_state``.

Three facts make that delegation exact:

1. **The temporal patch axis is a broadcast copy, so the convolution is 2-D.** The image processor
   expands every patch row over ``temporal_patch_size``, so all its temporal slices are the same
   pixels. The reference's ``nn.Conv3d`` over kernel depth ``temporal_patch_size`` then computes
   ``sum_t W[:, :, t] * x_0``, which equals a depth-1 convolution whose weight is ``W`` summed over
   ``t``. :func:`patch_embed_filters_from_conv3d` performs that sum, and the tower feeds the seam a
   single temporal slice.
2. **Attention is bidirectional within one frame.** The reference groups ``cu_seqlens`` with
   ``merge_temporal=False``, so each frame is its own attention segment of ``h * w`` tokens.
   :func:`compute_attention_bounds` expresses that as the per-row key ranges
   ``NF.flash_attention`` takes with ``causal_mask=False``.
3. **Tokens arrive block-major over spatial-merge blocks.** The processor's permute orders the
   flattened token axis as merge-block row, merge-block column, then row and column within the
   block. That lets the vision adapter treat every ``spatial_merge_size ** 2`` consecutive tokens as
   one merge block, and it is the order the rope's position ids use.

The tower is single-rank: neither the adapter nor the rope is sharded, and this module exposes no
tensor-parallel surface. The reference's split points, for when it gains one, are ``qkv`` and
``gate``/``up`` by column, ``proj`` and ``down`` by row.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm_neuron.functional as NF
from vllm_neuron.functional.vision.patch_embed import BATCH_RANGE, patch_embed

from .config import Glm5NextVisionConfig
from .utils.vision_rope import compute_vision_rotary_pos_emb
from .vision_adapter import Glm5NextVisionAdapter, Glm5NextVisionRMSNorm

# A kernel constraint, not a model choice: the attention kernel reads its bounds
# in 128-row tiles, so the sequence handed to it is padded up to a multiple of
# this and sliced back after. ``qwen3_vl/vision_encoder_bf16.py`` pads the same
# kernel the same way.
_ATTN_SEQ_ALIGN = 128

# The most patch rows one seam call may carry. Read off the seam's own batch range
# so it cannot drift away from what the seam admits.
_PATCH_ROWS_PER_CALL = BATCH_RANGE[1]

# Mirrors the map of the same name in ``vision_adapter.py``, which is private to
# that module.
_ACTIVATIONS: dict[str, type[nn.Module]] = {"silu": nn.SiLU, "gelu": nn.GELU}


class Glm5NextVisionEncoderError(ValueError):
    """Raised when tower inputs cannot describe a GLM-5.3-Flash vision batch."""


def patch_embed_filters_from_conv3d(conv_weight: torch.Tensor) -> torch.Tensor:
    """Convert the reference's patch-embed convolution to the seam's layout.

    The reference holds ``[C_out, C_in, temporal_patch_size, P, P]``; the patch-embed seam takes
    ``[K_d, K_h, K_w, C_in, C_out]`` with ``K_d == 1``. Summing over the temporal axis is exact
    rather than an approximation, because the input's temporal slices are identical copies -- see
    fact 1 in this module's docstring.

    Args:
        conv_weight: ``[C_out, C_in, temporal_patch_size, P, P]``.

    Returns:
        ``[1, P, P, C_in, C_out]``.

    Raises:
        Glm5NextVisionEncoderError: the weight is not 5-D.
    """
    if conv_weight.dim() != 5:
        raise Glm5NextVisionEncoderError(
            f"patch-embed convolution weight must be 5-D "
            f"[C_out, C_in, temporal_patch_size, P, P], got "
            f"{conv_weight.dim()}-D {tuple(conv_weight.shape)}"
        )
    summed = conv_weight.sum(dim=2)
    return summed.permute(2, 3, 1, 0).unsqueeze(0).contiguous()


def compute_attention_bounds(
    grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-frame bidirectional attention bounds from the vision grid.

    The reference expresses the same grouping as ``cu_seqlens`` with ``merge_temporal=False``: one
    segment per frame of ``h * w`` tokens. ``NF.flash_attention`` takes the equivalent as a half-open
    key range per query row, so a token attends to its own frame and to nothing else.

    Args:
        grid_thw: ``[num_items, 3]`` of (temporal, height, width) patch counts.

    Returns:
        ``(bound_min, bound_max)``, each ``[total_patches, 1]`` int32 on
        ``grid_thw``'s device.

    Raises:
        Glm5NextVisionEncoderError: the grid is not ``[num_items, 3]`` or is
            empty.
    """
    if grid_thw.dim() != 2 or grid_thw.shape[-1] != 3:
        raise Glm5NextVisionEncoderError(
            f"grid_thw must be [num_items, 3], got {tuple(grid_thw.shape)}"
        )
    if grid_thw.shape[0] == 0:
        raise Glm5NextVisionEncoderError(
            "grid_thw is empty; a tower call needs at least one image or video"
        )

    frame_lengths = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    )
    total_patches = int(frame_lengths.sum())
    bound_min = torch.zeros(
        total_patches, 1, dtype=torch.int32, device=grid_thw.device
    )
    bound_max = torch.zeros(
        total_patches, 1, dtype=torch.int32, device=grid_thw.device
    )
    start = 0
    for length in frame_lengths.tolist():
        end = start + int(length)
        bound_min[start:end, 0] = start
        bound_max[start:end, 0] = end
        start = end
    return bound_min, bound_max


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2-D vision rotary embeddings to query and key.

    Matches the reference operation for operation, including the float32 interior and the restore to
    the input dtype. The arithmetic order is part of the contract: it decides how each product
    rounds.

    Args:
        q: ``[seq, num_heads, head_dim]``.
        k: ``[seq, num_heads, head_dim]``.
        cos: ``[seq, head_dim]``.
        sin: ``[seq, head_dim]``.

    Returns:
        ``(q, k)`` rotated, each in its input dtype.
    """
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims, as the reference does."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Glm5NextVisionMLP(nn.Module):
    """Clamped SwiGLU feed-forward, matching the reference.

    The clamp is asymmetric on purpose: the gate is bounded above only, the up-projection on both
    sides. ``vision_adapter.py`` carries the same asymmetry for the merger's own SwiGLU.
    """

    def __init__(
        self, config: Glm5NextVisionConfig, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        if config.hidden_act not in _ACTIVATIONS:
            raise Glm5NextVisionEncoderError(
                f"hidden_act {config.hidden_act!r} is not one of "
                f"{sorted(_ACTIVATIONS)}"
            )
        # The reference passes ``bias=config.attention_bias`` when it builds this
        # MLP, so the feed-forward bias follows the attention bias flag rather
        # than a flag of its own.
        bias = config.attention_bias
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=bias, dtype=dtype
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=bias, dtype=dtype
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=bias, dtype=dtype
        )
        self.act_fn = _ACTIVATIONS[config.hidden_act]()
        self.swiglu_limit = float(config.swiglu_limit)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``[..., hidden_size]`` in, ``[..., hidden_size]`` out."""
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextVisionAttention(nn.Module):
    """Bidirectional per-frame attention, matching the reference.

    Two things separate this tower's attention from the qwen3_vl template the package otherwise
    follows: the query and key are RMS-normalised per head before the rotary embedding, and the fused
    ``qkv`` carries ``config.attention_bias``.
    """

    def __init__(
        self, config: Glm5NextVisionConfig, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise Glm5NextVisionEncoderError(
                f"hidden_size {config.hidden_size} is not divisible by "
                f"num_heads {config.num_heads}"
            )
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.q_norm = Glm5NextVisionRMSNorm(
            self.head_dim, eps=config.rms_norm_eps, dtype=dtype
        )
        self.k_norm = Glm5NextVisionRMSNorm(
            self.head_dim, eps=config.rms_norm_eps, dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> torch.Tensor:
        """``[seq, hidden_size]`` in and out; bounds ``[seq, 1]``."""
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        # The kernel batches over heads: [seq, heads, head_dim] -> [heads, seq,
        # head_dim], the [B, S, D] layout tp_q/tp_k select.
        query_states = query_states.permute(1, 0, 2)
        key_states = key_states.permute(1, 0, 2)
        value_states = value_states.permute(1, 0, 2)

        padded_min, padded_max = _pad_bounds(bound_min, bound_max, seq_length)
        query_states = _pad_sequence(query_states, seq_length)
        key_states = _pad_sequence(key_states, seq_length)
        value_states = _pad_sequence(value_states, seq_length)

        attn_output = NF.flash_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scaling,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            bound_min=padded_min.expand(self.num_heads, -1, -1).contiguous(),
            bound_max=padded_max.expand(self.num_heads, -1, -1).contiguous(),
        )

        attn_output = attn_output[:, :seq_length]
        attn_output = attn_output.permute(1, 0, 2).reshape(
            seq_length, self.num_heads * self.head_dim
        )
        return self.proj(attn_output)


def _aligned_length(seq_length: int) -> int:
    """Round a sequence length up to the kernel's tile."""
    tiles = (seq_length + _ATTN_SEQ_ALIGN - 1) // _ATTN_SEQ_ALIGN
    return tiles * _ATTN_SEQ_ALIGN


def _pad_sequence(tensor: torch.Tensor, seq_length: int) -> torch.Tensor:
    """Zero-pad ``[heads, seq, head_dim]`` up to the kernel's tile."""
    pad = _aligned_length(seq_length) - seq_length
    if pad == 0:
        return tensor
    return F.pad(tensor, (0, 0, 0, pad))


def _pad_bounds(
    bound_min: torch.Tensor, bound_max: torch.Tensor, seq_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extend ``[seq, 1]`` bounds over the padded tile, then add the head axis.

    A padded query row gets the range ``[0, 1)`` rather than the empty ``[0, 0)``. The caller slices
    both away, but an empty range masks every key and drives that row's softmax to NaN. Padded key
    rows need no handling: every real row's ``bound_max`` stops at its own frame, which ends at or
    before ``seq_length``.
    """
    padded_length = _aligned_length(seq_length)
    pad = padded_length - seq_length
    if pad:
        bound_min = F.pad(bound_min, (0, 0, 0, pad), value=0)
        bound_max = F.pad(bound_max, (0, 0, 0, pad), value=1)
    return bound_min.unsqueeze(0), bound_max.unsqueeze(0)


class Glm5NextVisionBlock(nn.Module):
    """Pre-norm residual block, matching the reference.

    Both norms are RMSNorm on the vision config's own epsilon. The qwen3_vl template uses
    ``nn.LayerNorm`` here; this reference does not.
    """

    def __init__(
        self, config: Glm5NextVisionConfig, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.norm1 = Glm5NextVisionRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.norm2 = Glm5NextVisionRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.attn = Glm5NextVisionAttention(config, dtype=dtype)
        self.mlp = Glm5NextVisionMLP(config, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> torch.Tensor:
        """``[seq, hidden_size]`` in and out."""
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cos, sin, bound_min, bound_max
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Glm5NextVisionEncoder(nn.Module):
    """The GLM-5.3-Flash vision tower, patch pixels to merged embeddings."""

    def __init__(
        self, config: Glm5NextVisionConfig, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads

        # Held in the seam's own filter layout rather than as an ``nn.Conv3d``, so
        # no weight is reshaped on the forward path.
        # :func:`patch_embed_filters_from_conv3d` converts a reference checkpoint
        # into it once.
        self.patch_embed_filters = nn.Parameter(
            torch.empty(
                1,
                config.patch_size,
                config.patch_size,
                config.in_channels,
                config.hidden_size,
                dtype=dtype,
            )
        )
        self.patch_embed_bias = nn.Parameter(
            torch.empty(config.hidden_size, dtype=dtype)
        )
        self.blocks = nn.ModuleList(
            [Glm5NextVisionBlock(config, dtype=dtype) for _ in range(config.depth)]
        )
        self.adapter = Glm5NextVisionAdapter(config, dtype=dtype)

    def patch_embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed flat patch rows through the NKI patch-embed seam.

        Args:
            pixel_values: ``[total_patches, patch_dim]`` where ``patch_dim`` is
                ``in_channels * temporal_patch_size * patch_size ** 2``.

        Returns:
            ``[total_patches, hidden_size]``.

        Raises:
            Glm5NextVisionEncoderError: the row width is not ``patch_dim``.
        """
        if pixel_values.dim() != 2:
            raise Glm5NextVisionEncoderError(
                f"pixel_values must be 2-D [total_patches, patch_dim], got "
                f"{pixel_values.dim()}-D {tuple(pixel_values.shape)}"
            )
        patch_dim = (
            self.in_channels * self.temporal_patch_size * self.patch_size**2
        )
        if pixel_values.shape[-1] != patch_dim:
            raise Glm5NextVisionEncoderError(
                f"pixel_values row width {pixel_values.shape[-1]} does not equal "
                f"in_channels * temporal_patch_size * patch_size ** 2 = "
                f"{patch_dim}"
            )

        total_patches = pixel_values.shape[0]
        rows = pixel_values.reshape(
            total_patches,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        # One temporal slice: they are copies of each other (fact 1 in this
        # module's docstring) and the filters already carry the sum over that
        # axis. This is the ``D == 1`` input the seam requires.
        rows = rows[:, :, :1]

        # The seam refuses a batch above its own range, and one patch row is one
        # batch element, so a real image is embedded in chunks.
        embedded = []
        for start in range(0, total_patches, _PATCH_ROWS_PER_CALL):
            chunk = rows[start : start + _PATCH_ROWS_PER_CALL]
            out = patch_embed(
                chunk,
                self.patch_embed_filters,
                self.patch_size,
                bias=self.patch_embed_bias,
            )
            embedded.append(out.reshape(chunk.shape[0], self.hidden_size))
        return torch.cat(embedded, dim=0)

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Run the tower.

        Args:
            pixel_values: ``[total_patches, patch_dim]`` flat patch rows, in the
                processor's block-major order.
            grid_thw: ``[num_items, 3]`` of (temporal, height, width) patch
                counts, one row per image or video.

        Returns:
            ``[num_merged_tokens, out_hidden_size]``, the reference's ``pooler_output``.
        """
        hidden_states = self.patch_embed(pixel_values)
        cos, sin = compute_vision_rotary_pos_emb(
            grid_thw, self.head_dim, self.spatial_merge_size
        )
        cos = cos.to(device=hidden_states.device, dtype=hidden_states.dtype)
        sin = sin.to(device=hidden_states.device, dtype=hidden_states.dtype)
        bound_min, bound_max = compute_attention_bounds(grid_thw)
        bound_min = bound_min.to(hidden_states.device)
        bound_max = bound_max.to(hidden_states.device)

        if cos.shape[0] != hidden_states.shape[0]:
            raise Glm5NextVisionEncoderError(
                f"grid_thw describes {cos.shape[0]} patches but pixel_values "
                f"carries {hidden_states.shape[0]}"
            )

        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin, bound_min, bound_max)
        return self.adapter(hidden_states)
