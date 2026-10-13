# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash vision tower.

Covers the transformer-block half of the reference's ``Glm5NextVisionModel.forward``. The
surrounding stages live elsewhere:

=====================================  ==================================
reference stage                        where it lives here
=====================================  ==================================
``patch_embed``                        :meth:`Glm5NextVisionEncoder.patch_embed`
``rotary_pos_emb`` + ``cos``/``sin``   ``utils/vision_rope.py``
``blocks`` loop                        this module
``post_layernorm``                     ``vision_adapter.py``
``view``/``permute``/``downsample``    ``vision_adapter.py``
``merger``                             ``vision_adapter.py``
=====================================  ==================================

:meth:`Glm5NextVisionEncoder.encode_block` therefore returns the reference's ``pooler_output``,
the merged ``out_hidden_size``-wide embeddings the encoder cache stores, not its
``last_hidden_state``. :meth:`Glm5NextVisionEncoder.forward`, the compiled graph, encodes a bucket
of fixed-size attention blocks, one rank's run of them under vision data parallelism, and writes
those rows into the encoder cache; :func:`pack_rank_blocks` lays the images out for it, one image
per block.

Three facts make that delegation exact:

1. **The temporal patch axis is a broadcast copy, so the convolution is 2-D.** The image processor
   expands every patch row over ``temporal_patch_size``, so all its temporal slices are the same
   pixels. The reference's ``nn.Conv3d`` over kernel depth ``temporal_patch_size`` then computes
   ``sum_t W[:, :, t] * x_0``, which equals a depth-1 convolution whose weight is ``W`` summed over
   ``t``. :func:`patch_embed_filters_from_conv3d` performs that sum, and
   :meth:`Glm5NextVisionEncoder.patch_embed` multiplies a single temporal slice by the result.
2. **Attention is bidirectional within one frame.** The reference groups ``cu_seqlens`` with
   ``merge_temporal=False``, so each frame is its own attention segment of ``h * w`` tokens.
   :func:`compute_attention_bounds` expresses that as the per-row key ranges
   ``NF.flash_attention`` takes with ``causal_mask=False``.
3. **Tokens arrive block-major over spatial-merge blocks.** The processor's permute orders the
   flattened token axis as merge-block row, merge-block column, then row and column within the
   block. That lets the vision adapter treat every ``spatial_merge_size ** 2`` consecutive tokens as
   one merge block, and it is the order the rope's position ids use.

The tower has no tensor-parallel surface: neither the adapter nor the rope is sharded, and every
rank holds the whole tower. Ranks split a bucket's blocks by vision data parallelism instead. The
reference's tensor-parallel split points, for when the tower gains them, are ``qkv`` and
``gate``/``up`` by column, ``proj`` and ``down`` by row.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm_neuron.functional as NF
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

from .config import Glm5NextVisionConfig
from .utils.vision_rope import compute_vision_rotary_pos_emb
from .vision_adapter import Glm5NextVisionAdapter, Glm5NextVisionRMSNorm

# A kernel constraint, not a model choice: the attention kernel reads its bounds
# in 128-row tiles, so the sequence handed to it is padded up to a multiple of
# this and sliced back after. ``qwen3_vl/vision_encoder_bf16.py`` pads the same
# kernel the same way.
_ATTN_SEQ_ALIGN = 128

# Mirrors the map of the same name in ``vision_adapter.py``, which is private to
# that module.
_ACTIVATIONS: dict[str, type[nn.Module]] = {"silu": nn.SiLU, "gelu": nn.GELU}

#: The checkpoint prefix of every tower tensor.
VISION_CHECKPOINT_PREFIX = "model.visual."

# The tower parameters whose checkpoint name is not their own name. Every other
# parameter is named as in the checkpoint, except that the adapter's stages sit
# at the checkpoint's top level (``post_layernorm.*``, ``merger.*``) rather than
# under ``adapter.``.
_CHECKPOINT_RENAMES: dict[str, str] = {
    "patch_embed_filters": "patch_embed.proj.weight",
    "patch_embed_bias": "patch_embed.proj.bias",
    "adapter.downsample_weight": "downsample.weight",
    "adapter.downsample_bias": "downsample.bias",
}
_ADAPTER_PREFIX = "adapter."


class Glm5NextVisionEncoderError(ValueError):
    """Raised when tower inputs cannot describe a GLM-5.3-Flash vision batch."""


def patch_embed_filters_from_conv3d(conv_weight: torch.Tensor) -> torch.Tensor:
    """Convert the reference's patch-embed convolution to the tower's filter layout.

    The reference holds ``[C_out, C_in, temporal_patch_size, P, P]``; the tower holds
    ``[K_d, K_h, K_w, C_in, C_out]`` with ``K_d == 1``, the layout of the NKI patch-embed seam.
    Summing over the temporal axis is exact rather than an approximation, because the input's
    temporal slices are identical copies -- see fact 1 in this module's docstring.

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


class Glm5NextVisionHeadRMSNorm(Glm5NextVisionRMSNorm):
    """The attention's per-head RMSNorm of the query or key, ``[seq, heads, head_dim]`` in and out.

    It computes :class:`Glm5NextVisionRMSNorm` over ``head_dim``, with one change: each row's sum of
    squares per head is one float32 matmul of the squares, flattened to ``[seq, heads * head_dim]``,
    with a 0/1 head indicator ``[heads * head_dim, heads]``. Neuron lowers the reduce over the
    ``head_dim`` axis to float32 Tensor Engine transposes, which took most of the tower's device time;
    the matmul form needs none of those transposes.

    The products are the same float32 squares and only their sum order changes, so the class against the
    reference is SUM-ORDER, not bit-equal. The tower output, measured against the reduce form, stays within
    the device bound of ``test/hardware/benchmark_glue_block.py``: ``agreement(layers=depth)``, rel-L2
    ``DEVICE_REL_L2_TOL`` per layer.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        seq, heads, head_dim = h.shape
        # ind[lane, head] is 1 where lane // head_dim == head: the 0/1 head indicator.
        lanes = torch.arange(heads * head_dim, device=h.device) // head_dim
        ind = (lanes[:, None] == torch.arange(heads, device=h.device)[None, :]).to(
            torch.float32
        )
        variance = ((h * h).reshape(seq, heads * head_dim) @ ind) / head_dim
        h = h * torch.rsqrt(variance + self.variance_epsilon).unsqueeze(-1)
        return self.weight * h.to(input_dtype)


class Glm5NextVisionAttention(nn.Module):
    """Bidirectional per-frame attention, matching the reference.

    Two things separate this tower's attention from the qwen3_vl template the package otherwise
    follows: the query and key are RMS-normalised per head before the rotary embedding
    (:class:`Glm5NextVisionHeadRMSNorm`), and the fused ``qkv`` carries ``config.attention_bias``.
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
        self.q_norm = Glm5NextVisionHeadRMSNorm(
            self.head_dim, eps=config.rms_norm_eps, dtype=dtype
        )
        self.k_norm = Glm5NextVisionHeadRMSNorm(
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

    Both norms are RMSNorm on the vision config's ``block_norm_eps``, which is not the epsilon
    of the attention's q_norm and k_norm. The qwen3_vl template uses ``nn.LayerNorm`` here; this
    reference does not.
    """

    def __init__(
        self, config: Glm5NextVisionConfig, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.norm1 = Glm5NextVisionRMSNorm(
            config.hidden_size, eps=config.block_norm_eps, dtype=dtype
        )
        self.norm2 = Glm5NextVisionRMSNorm(
            config.hidden_size, eps=config.block_norm_eps, dtype=dtype
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
        self.patch_dim = config.patch_dim

        # Held in the NKI patch-embed seam's filter layout rather than as an
        # ``nn.Conv3d``; :meth:`patch_embed` flattens it to its matmul's
        # ``[in_channels * patch_size ** 2, hidden_size]``.
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

        # The two parameters held in another layout than the checkpoint's convert
        # as they load. Each converter casts to the tower dtype first, so the
        # patch-embed kernel is summed over its temporal axis in that dtype.
        set_weight_loader(
            self.patch_embed_filters,
            SafetensorsWeightLoader(
                transform=lambda slices, rank: patch_embed_filters_from_conv3d(
                    slices[0][:].to(dtype)
                )
            ),
        )
        set_weight_loader(
            self.adapter.downsample_weight,
            SafetensorsWeightLoader(
                transform=lambda slices, rank: self.adapter.downsample_weight_from_conv(
                    slices[0][:].to(dtype)
                )
            ),
        )

    def checkpoint_mappings(self, prefix: str = "") -> dict[str, str]:
        """Map every tower parameter to its checkpoint key.

        The map has the form the checkpoint reader takes: parameter name, under ``prefix``, to
        checkpoint key. The tower is replicated, so every key loads whole on every rank. A
        parameter whose key the checkpoint lacks is refused by the reader, by name.

        Args:
            prefix: this module's path in the model that loads it, with the trailing dot.

        Returns:
            ``{prefix + parameter name: checkpoint key}`` for every parameter of the tower.
        """
        return {
            prefix + name: VISION_CHECKPOINT_PREFIX
            + _CHECKPOINT_RENAMES.get(name, name.removeprefix(_ADAPTER_PREFIX))
            for name, _ in self.named_parameters()
        }

    def refuse_unmapped_checkpoint_keys(self, checkpoint_names: set[str]) -> None:
        """Refuse a checkpoint tower tensor that no parameter of this tower loads.

        The reader refuses the other direction, a parameter with no tensor. Together the two
        checks mean the loaded tower is the checkpoint's whole tower.

        Args:
            checkpoint_names: every tensor name in the checkpoint.

        Raises:
            Glm5NextVisionEncoderError: a checkpoint tensor under
                :data:`VISION_CHECKPOINT_PREFIX` has no parameter here.
        """
        mapped = set(self.checkpoint_mappings().values())
        unused = sorted(
            key
            for key in checkpoint_names
            if key.startswith(VISION_CHECKPOINT_PREFIX) and key not in mapped
        )
        if unused:
            raise Glm5NextVisionEncoderError(
                f"{len(unused)} checkpoint tensors under {VISION_CHECKPOINT_PREFIX!r} have no "
                f"vision tower parameter, for example {unused[:4]}; the tower would load "
                f"without them"
            )

    def patch_embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed flat patch rows as one matmul plus the bias.

        The reference's convolution has a window equal to its stride, so each patch row gives one
        output pixel: ``bias + x @ W``, where ``x`` is the row's single temporal slice flattened in
        ``(in_channels, patch_size, patch_size)`` order and ``W`` holds the filters in that order.
        All rows go through one ``addmm``.

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
        if pixel_values.shape[-1] != self.patch_dim:
            raise Glm5NextVisionEncoderError(
                f"pixel_values row width {pixel_values.shape[-1]} does not equal "
                f"in_channels * temporal_patch_size * patch_size ** 2 = "
                f"{self.patch_dim}"
            )

        total_patches = pixel_values.shape[0]
        # One temporal slice: they are copies of each other (fact 1 in this
        # module's docstring) and the filters already carry the sum over that
        # axis.
        rows = pixel_values.reshape(
            total_patches,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )[:, :, 0]
        rows = rows.reshape(
            total_patches, self.in_channels * self.patch_size * self.patch_size
        )
        # [1, P, P, C_in, C_out] -> [C_in, P, P, C_out] -> [C_in * P * P, C_out],
        # the order of ``rows``.
        weight = (
            self.patch_embed_filters[0]
            .permute(2, 0, 1, 3)
            .reshape(rows.shape[1], self.hidden_size)
        )
        return torch.addmm(self.patch_embed_bias, rows, weight)

    def encode_block(
        self,
        pixel_values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> torch.Tensor:
        """Run the tower over one attention block.

        Args:
            pixel_values: ``[block_size, patch_dim]`` flat patch rows, in the processor's
                block-major order.
            cos: ``[block_size, head_dim]``, in the tower dtype.
            sin: ``[block_size, head_dim]``, in the tower dtype.
            bound_min: ``[block_size, 1]`` int32, each row's first key.
            bound_max: ``[block_size, 1]`` int32, one past each row's last key.

        Returns:
            ``[block_size // spatial_merge_size ** 2, out_hidden_size]``, the reference's
            ``pooler_output`` for the rows of the block.
        """
        hidden_states = self.patch_embed(pixel_values)
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin, bound_min, bound_max)
        return self.adapter(hidden_states)

    def forward(
        self,
        pixel_values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
        encoder_cache_buffer: torch.Tensor,
        write_block_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode this rank's blocks of a bucket and write every block into the encoder cache.

        Every shape is fixed by the bucket, so one compiled graph serves every call of that
        bucket; :func:`pack_rank_blocks` lays the images out. Under vision data parallelism
        each rank takes only its own run of blocks, so every rank's graph has the same shapes
        and one compiled graph serves every rank. The ranks gather the encoded runs, in rank
        order, so every rank's cache holds every block.

        Args:
            pixel_values: ``[blocks_per_rank, block_size, patch_dim]`` in the tower dtype,
                where ``blocks_per_rank`` is the bucket's block count over the vision DP size.
            cos: ``[blocks_per_rank, block_size, head_dim]`` in the tower dtype.
            sin: ``[blocks_per_rank, block_size, head_dim]`` in the tower dtype.
            bound_min: ``[blocks_per_rank, block_size, 1]`` int32, block-local.
            bound_max: ``[blocks_per_rank, block_size, 1]`` int32, block-local.
            encoder_cache_buffer: ``[num_cache_blocks, block_size // spatial_merge_size ** 2,
                out_hidden_size]``, written in place.
            write_block_ids: ``[num_blocks]`` int64, the cache block each block of the bucket
                goes to, every rank's run in rank order; padding blocks go to the cache's
                scratch block.

        Returns:
            ``encoder_cache_buffer``, the same tensor.

        Raises:
            Glm5NextVisionEncoderError: ``write_block_ids`` that does not name one cache block
                for each block of every rank's run.
        """
        # Vision data parallelism: the ranks of one vision DP group split the
        # bucket's blocks (``VisionNeuronConfig.resolve_tp_dp`` pads the block count
        # to a multiple of the group). The group is None with one vision DP rank,
        # which encodes every block. Resolved here, at the call, as this package
        # resolves every group.
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_vision_dp_group,
        )

        dp_group = get_neuron_vision_dp_group()
        dp_size = dp_group.world_size if dp_group is not None else 1
        if write_block_ids.shape[0] != pixel_values.shape[0] * dp_size:
            raise Glm5NextVisionEncoderError(
                f"{write_block_ids.shape[0]} cache block ids for {dp_size} vision DP "
                f"ranks of {pixel_values.shape[0]} attention blocks each"
            )
        merged = torch.stack(
            [
                self.encode_block(
                    pixel_values[b], cos[b], sin[b], bound_min[b], bound_max[b]
                )
                for b in range(pixel_values.shape[0])
            ]
        )
        if dp_size > 1:
            merged = dp_group.all_gather(merged, dim=0)
        encoder_cache_buffer.index_put_((write_block_ids,), merged)
        return encoder_cache_buffer


def pack_images_into_blocks(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    block_size: int,
    num_blocks: int,
    config: Glm5NextVisionConfig,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Lay images out one per attention block, as :meth:`Glm5NextVisionEncoder.forward` takes them.

    Image ``i`` fills the first ``t * h * w`` rows of block ``i``, so its attention bounds,
    its rope rows and its patch-embed calls start at row 0 of the block, as they start at row 0
    of the image. The rest of each block, and every block past the last image, is padding: zero
    pixels and zero rope, and each padding row attends to itself alone. An empty key range would
    drive that row's softmax to NaN; its merged rows land in the cache block's tail or in the
    scratch block, where no position reads them.

    Args:
        pixel_values: ``[total_patches, patch_dim]``, the processor's rows of every image in turn.
        grid_thw: ``[num_images, 3]`` of (temporal, height, width) patch counts.
        block_size: rows per attention block, ``vision_attention_block_size``.
        num_blocks: blocks of the bucket; at least the number of images.
        config: the tower config, for the patch width, head width and merge size.
        dtype: the tower dtype. The processor emits float32 rows, and the patch embed takes its
            input in the filters' dtype.

    Returns:
        ``pixel_values``, ``cos``, ``sin``, ``bound_min`` and ``bound_max``, each with a leading
        ``[num_blocks, block_size]``.

    Raises:
        Glm5NextVisionEncoderError: more images than blocks, an image with more rows than a block,
            or rows that disagree with the grid. An image is never truncated.
    """
    head_dim = config.hidden_size // config.num_heads
    rows_per_image = _rows_per_image(
        pixel_values, grid_thw, block_size=block_size, num_blocks=num_blocks
    )
    rows = torch.arange(block_size, dtype=torch.int32).reshape(1, block_size, 1)
    packed = {
        "pixel_values": torch.zeros(
            num_blocks, block_size, config.patch_dim, dtype=dtype
        ),
        "cos": torch.zeros(num_blocks, block_size, head_dim, dtype=dtype),
        "sin": torch.zeros(num_blocks, block_size, head_dim, dtype=dtype),
        "bound_min": rows.repeat(num_blocks, 1, 1),
        "bound_max": (rows + 1).repeat(num_blocks, 1, 1),
    }
    start = 0
    for block, image_rows in enumerate(rows_per_image):
        grid = grid_thw[block : block + 1]
        cos, sin = compute_vision_rotary_pos_emb(
            grid, head_dim, config.spatial_merge_size
        )
        bound_min, bound_max = compute_attention_bounds(grid)
        packed["pixel_values"][block, :image_rows] = pixel_values[
            start : start + image_rows
        ].to(dtype)
        packed["cos"][block, :image_rows] = cos.to(dtype)
        packed["sin"][block, :image_rows] = sin.to(dtype)
        packed["bound_min"][block, :image_rows] = bound_min
        packed["bound_max"][block, :image_rows] = bound_max
        start += image_rows
    return packed


def pack_rank_blocks(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    block_size: int,
    num_blocks: int,
    config: Glm5NextVisionConfig,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Lay out this vision DP rank's run of a bucket's blocks, the tower's input on this rank.

    The ranks of a vision DP group split the bucket's blocks in rank order, and
    :meth:`Glm5NextVisionEncoder.forward` takes the rank's run: each rank packs and sends to its
    device only its own run, and every rank's graph has the same shapes. Image ``i`` goes to
    block ``i`` (:func:`pack_images_into_blocks`), so a rank's run holds a run of the images.
    The layout is checked over every image first: every rank then refuses the same batch, and no
    rank waits in the tower's gather for a rank that refused. With one vision DP rank, the run is
    the whole bucket.

    Args:
        pixel_values: ``[total_patches, patch_dim]``, the processor's rows of every image in turn.
        grid_thw: ``[num_images, 3]`` of (temporal, height, width) patch counts.
        block_size: rows per attention block, ``vision_attention_block_size``.
        num_blocks: blocks of the bucket over every rank; at least the number of images.
        config: the tower config, for the patch width, head width and merge size.
        dtype: the tower dtype.

    Returns:
        :func:`pack_images_into_blocks`'s tensors for this rank's run, each with a leading
        ``[num_blocks // vision_dp_size, block_size]``.

    Raises:
        Glm5NextVisionEncoderError: a block count the vision DP ranks cannot split evenly, or a
            batch :func:`pack_images_into_blocks` refuses.
    """
    from vllm_neuron.parallel.neuron_parallel_state import get_neuron_vision_dp_group

    dp_group = get_neuron_vision_dp_group()
    if dp_group is None:
        return pack_images_into_blocks(
            pixel_values,
            grid_thw,
            block_size=block_size,
            num_blocks=num_blocks,
            config=config,
            dtype=dtype,
        )
    if num_blocks % dp_group.world_size:
        raise Glm5NextVisionEncoderError(
            f"{num_blocks} attention blocks do not split over {dp_group.world_size} "
            f"vision DP ranks; a rank would drop blocks"
        )
    rows_per_image = _rows_per_image(
        pixel_values, grid_thw, block_size=block_size, num_blocks=num_blocks
    )
    blocks_per_rank = num_blocks // dp_group.world_size
    first = dp_group.rank_in_group * blocks_per_rank
    images = slice(first, first + blocks_per_rank)
    start = sum(rows_per_image[:first])
    stop = start + sum(rows_per_image[images])
    return pack_images_into_blocks(
        pixel_values[start:stop],
        grid_thw[images],
        block_size=block_size,
        num_blocks=blocks_per_rank,
        config=config,
        dtype=dtype,
    )


def _rows_per_image(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    block_size: int,
    num_blocks: int,
) -> list[int]:
    """Each image's patch rows, once the batch is known to fit one image per block.

    Raises:
        Glm5NextVisionEncoderError: as :func:`pack_images_into_blocks`.
    """
    rows_per_image = [int(t * h * w) for t, h, w in grid_thw.tolist()]
    if len(rows_per_image) > num_blocks:
        raise Glm5NextVisionEncoderError(
            f"{len(rows_per_image)} images do not fit a bucket of {num_blocks} attention "
            f"blocks; the tower places one image per block"
        )
    too_large = [rows for rows in rows_per_image if rows > block_size]
    if too_large:
        raise Glm5NextVisionEncoderError(
            f"an image of {too_large[0]} patch rows is larger than one attention block of "
            f"{block_size} rows (vision_attention_block_size); it would be cut, so it is "
            f"refused"
        )
    if sum(rows_per_image) != pixel_values.shape[0]:
        raise Glm5NextVisionEncoderError(
            f"grid_thw describes {sum(rows_per_image)} patch rows but pixel_values carries "
            f"{pixel_values.shape[0]}"
        )
    return rows_per_image
