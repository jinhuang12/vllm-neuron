# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS MXFP4 Implementation
=============================================

Annotated implementation of GPT-OSS with MXFP4 quantized expert weights.
Adapted from BF16 model with MXFP4-specific weight formats and shuffling.

Key differences from BF16:
- Expert weights stored in MXFP4 tiled format (uint16 blocks + uint8 scales)
- Hidden dimension shuffled for hardware memory layout optimization
- MXFP4-specific weight loaders with EP support
- MoE CTE kernel uses shard_on_block_mx implementation

Supported parallelism: TP, SP, DP, EP, and DP+EP combinations.
Starting model: GPT-OSS-20B (32 experts, top-4, GQA 64Q/8KV).

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    GPT-OSS specific. Change when porting.
  # <-- MXFP4: ...             MXFP4 format specific. Change for different quant formats.
"""

import logging
import math

import nki.language as nl
import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.models.interfaces import SupportsEagle3

import vllm_neuron.functional as NF
from vllm_neuron.functional.attention.attention_decode import (
    _swizzle_packed_k,
    _unswizzle_packed_k,
    packed_fp8_viable_for_bucket as _packed_fp8_viable_for_bucket,
)

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import (
    FP8_CLAMP_MAX,
    validate_fp8_segmented_supported,
)
from vllm_neuron.utils.weight_loader import (
    ShardSpec,
    set_weight_loader,
    with_rank_override,
)

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler

import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.vllm.spec_decode.decorator import async_speculative_decoding

from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    MoEAllToAllVStrategy,
    RouterActFnType,
    QuantizationType,
)
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation

# <-- MXFP4: Weight loaders for MXFP4 tiled format with shuffling
from vllm_neuron.model.gpt_oss.weight_loaders_mxfp4 import (
    Q_HEIGHT,
    Q_WIDTH,
    _get_i_tiling_shard_i,
    expert_parallel_mxfp4_down_bias_loader,
    expert_parallel_mxfp4_down_scale_loader,
    expert_parallel_mxfp4_down_weight_loader,
    expert_parallel_mxfp4_gate_up_bias_loader,
    expert_parallel_mxfp4_gate_up_blocks_loader,
    expert_parallel_mxfp4_gate_up_scale_loader,
    fused_qkv_bias_loader as mxfp4_fused_qkv_bias_loader,
    fused_qkv_shuffling_weight_loader,
    o_proj_weight_loader as mxfp4_o_proj_weight_loader,
    shuffling_weight_loader,
)

from .config import GptOssConfig
from .model_bf16 import _precompute_decode_attn_masks

# Token-count (B*S) threshold at/above which the cross-DP-EP decode combine uses
# ReduceScatter[World] -> AllGather[TP] instead of AllReduce[World] + slice.
# Matches the MoE convention in functional/moe/router.py (shard_on_tokens = T >= 128).
DECODE_RSAG_TOKEN_THRESHOLD = 128

DEFAULT_SELECTIVE_LOADING_THRESHOLD = 1.0

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: GPT-OSS uses RMSNorm with variance on unpadded portion
# <-- MXFP4: Hidden dim is shuffled, so norm weights are also shuffled
# =============================================================================


class GptOssRMSNorm(nn.Module):
    """RMS Normalization with support for padded hidden dimensions.

    <-- MODEL-SPECIFIC: GPT-OSS pads hidden_size to 3072 for hardware alignment.
    The variance is computed only on the unpadded portion, and the padded positions
    are zeroed after normalization.
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.torch_dtype)
        )
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.hidden_size = config.hidden_size
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # <-- MXFP4: Hidden dim is shuffled, so padded zeros are interleaved.
        # Adjust mean by (H_padded / H_actual) to compensate for zero-padded positions.
        scaling_factor = self.hidden_size / self.unpadded_hidden_size
        variance = hidden_states.pow(2).mean(-1, keepdim=True) * scaling_factor
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * hidden_states
        # <-- MXFP4: Zero shuffled padded positions via reshape to [4, H//4] layout
        output_shape = output.shape
        output = output.view(*output_shape[:-1], 4, self.hidden_size // 4)
        output[..., self.unpadded_hidden_size // 4 :] = 0.0
        output = output.view(output_shape)
        return output.to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (YaRN)
# <-- MODEL-SPECIFIC: GPT-OSS uses YaRN scaling for RoPE
# =============================================================================


class GptOssRotaryEmbedding(nn.Module):
    """Rotary Position Embedding with YaRN scaling.

    <-- MODEL-SPECIFIC: YaRN interpolation/extrapolation is GPT-OSS specific.
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_parameters["rope_theta"]
        rope_parameters = config.rope_parameters or {}
        self.scaling_factor = rope_parameters.get("factor", 1.0)
        self.beta_slow = rope_parameters.get("beta_slow", 1.0)
        self.beta_fast = rope_parameters.get("beta_fast", 32.0)
        self.initial_context_length = rope_parameters.get(
            "original_max_position_embeddings", 8192
        )

        inv_freq, concentration = self._compute_inv_freq_and_concentration("cpu")
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("concentration", concentration, persistent=False)

    def _compute_inv_freq_and_concentration(self, device):
        """<-- MODEL-SPECIFIC: YaRN frequency computation."""
        freq = self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
            / self.head_dim
        )
        concentration = 0.1 * math.log(self.scaling_factor) + 1.0
        d_half = self.head_dim / 2
        low = (
            d_half
            * math.log(self.initial_context_length / (self.beta_fast * 2 * math.pi))
            / math.log(self.rope_theta)
        )
        high = (
            d_half
            * math.log(self.initial_context_length / (self.beta_slow * 2 * math.pi))
            / math.log(self.rope_theta)
        )
        interpolation = 1.0 / (self.scaling_factor * freq)
        extrapolation = 1.0 / freq
        ramp = (torch.arange(d_half, dtype=torch.float32, device=device) - low) / (
            high - low
        )
        mask = 1 - ramp.clamp(0, 1)
        inv_freq = interpolation * (1 - mask) + extrapolation * mask
        concentration = torch.tensor(concentration, device=device)
        return inv_freq, concentration

    def forward(self, position_ids, device, dtype):
        inv_freq_expanded = self.inv_freq[None, :].float()
        position_ids_expanded = position_ids[:, None].float()
        freqs = position_ids_expanded @ inv_freq_expanded
        cos = freqs.cos() * self.concentration
        sin = freqs.sin() * self.concentration
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (sinks, sliding window, GQA, RoPE application)
# <-- MXFP4: Attention weights use shuffling weight loaders
# =============================================================================


# NOTE: RoPE is fused into NF.qkv_proj for prefill; decode handles RoPE in
# its own dedicated kernel call. The standalone `_apply_rotary_emb` /
# `apply_rotary_pos_emb` helpers are no longer used by either path.


def _resolve_use_all2all(config: GptOssConfig) -> bool:
    """True iff decode should route through the NeuronAll2AllManager path.

    Equivalent to `enable_expert_parallel AND all2all_manager is not None`,
    but derived from config only (no parallel-state lookup) — the manager
    attaches iff both conditions hold (see neuron_communicator.py).
    """
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    return (
        vllm_config.parallel_config.enable_expert_parallel
        and config.neuron_config is not None
        and config.neuron_config.all2all_backend == "neuron"
    )


def _replicated_sinks_loader(
    shard_size: int,
    num_shards: int,
    attention_dp_size: int,
    tp_rank: int,
):
    """Load sinks replicated across attention DP ranks.

    Each DP rank within the same TP position has a different shard of sinks.
    This loader concatenates all DP shards for the given TP position so the
    decode path doesn't need a runtime collective.

    Returns a tensor of size [shard_size * attention_dp_size].
    """
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    ddp = attention_dp_size

    def transform(slices, rank):
        assert len(slices) == 1
        slice_obj = slices[0]
        shards = []
        for dp_rank in range(ddp):
            effective_rank = dp_rank + tp_rank * ddp
            start = (effective_rank % num_shards) * shard_size
            end = start + shard_size
            shards.append(slice_obj[start:end])
        return torch.cat(shards, dim=0).to(torch.float32)

    return SafetensorsWeightLoader(transform=transform)


class GptOssAttention(nn.Module):
    """Multi-head attention with TP head sharding.

    >>> PARALLELISM: TP <<<
    <-- MODEL-SPECIFIC: GQA, sinks, sliding window, YaRN RoPE
    <-- MXFP4: Weight loaders use shuffling for hidden dim alignment
    """

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.rms_norm_eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5
        self.max_seq_len = config.max_position_embeddings

        # <-- MODEL-SPECIFIC: Sliding window on alternating (even) layers
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else None

        # Fused-SWA prefill is on by default for every SWA layer (non-SWA layers
        # have sliding_window=None, so fuse_swa is False for them). fuse_swa alone
        # is NOT the final gate: the fused swa_fused_cte kernel is only validated
        # on the packed-FP8 KV cache, so forward_prefill additionally requires
        # self.fp8_packed (known at bind_kv_cache time, after __init__) — bf16-KV
        # serving always takes the qkv+segmented path.
        self.fuse_swa = self.sliding_window is not None

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Attention DP setup <<<
        self.attention_dp_size = (
            config.neuron_config.attention_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_dp_group,
            get_neuron_attention_dp_rank,
        )

        self.attention_dp_group = get_neuron_attention_dp_group()
        self.attention_dp_rank = get_neuron_attention_dp_rank()

        # >>> PARALLELISM: Attention TP group (TP * attn_dp) <<<
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_tp_group,
        )

        self.attn_tp_group = get_neuron_attention_tp_group()

        # >>> PARALLELISM: All2All decode path detection <<<
        # SP-sharded hidden_states across tp_group when EP is on with an attached all2all manager.
        self.use_all2all = _resolve_use_all2all(config)

        effective_q_shards = self.world_size * self.attention_dp_size

        # >>> PARALLELISM: Head sharding calculation <<<
        self.num_attention_heads_per_rank = (
            self.num_attention_heads // effective_q_shards
        )

        self.kv_needs_a2a = (
            self.attention_dp_size > 1
            and self.num_key_value_heads > self.world_size
            and self.num_key_value_heads % effective_q_shards == 0
        )

        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1

        # KV heads-per-rank used in the QKV projection output. When
        # kv_needs_a2a, attention DP shards KV further than world_size, so
        # the projection emits a smaller K/V block than `num_key_value_heads_per_rank`.
        # Stored on self so the prefill path can pass the correct value to
        # NF.qkv_proj's fused-RoPE (which uses num_kv_heads to delimit the
        # K block within the projection output).
        self.num_kv_heads_for_weight = (
            self.num_key_value_heads // effective_q_shards
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )
        num_kv_heads_for_weight = self.num_kv_heads_for_weight  # local alias

        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // num_kv_heads_for_weight
        )

        self.num_q_heads_after_a2a = (
            self.num_attention_heads_per_rank * self.attention_dp_size
        )
        self.num_kv_heads_after_a2a = (
            num_kv_heads_for_weight * self.attention_dp_size
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )

        # >>> PARALLELISM: QKV weight shapes for TP * attention DP <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = num_kv_heads_for_weight * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in_features = (
            self.num_attention_heads * self.head_dim
        ) // effective_q_shards

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.qkv_proj_bias = nn.Parameter(torch.empty(qkv_size, dtype=self.dtype))
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=self.dtype)
        )
        self.o_proj_bias = nn.Parameter(torch.empty(self.hidden_size, dtype=self.dtype))

        # <-- MODEL-SPECIFIC: Learnable attention sinks (per-head)
        # Pre-gathered across attention DP so no runtime all_gather is needed.
        self.sinks = nn.Parameter(
            torch.empty(self.num_q_heads_after_a2a, dtype=torch.float32)
        )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        self.k_cache = None
        self.v_cache = None

        # When True, the K cache uses the swizzled packed FP8 layout
        # ([num_blocks, (kv_heads,) block_len // 2, d_head, 2]); see model_bf16.
        self.fp8_packed = False

        # KV cache quantization scales are set during weight loading.
        # We also need floats since tensor.item() causes a graph break,
        # and kernels currently use floats for perf reasons.
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """<-- MXFP4: Shuffling weight loaders for attention weights.

        With attention DP: Q/O are sharded across TP*attention DP using
        an interleaved effective rank.
        """
        ddp = self.attention_dp_size
        effective_q_shards = self.world_size * ddp
        effective_q_rank = self.attention_dp_rank + self.tp_group.rank_in_group * ddp

        qkv_loader = fused_qkv_shuffling_weight_loader(
            self.q_size,
            self.kv_size,
            (self.hidden_size, self.q_size + 2 * self.kv_size),
            self.num_kv_replicas,
            attention_dp_size=ddp,
            kv_needs_a2a=self.kv_needs_a2a,
        )
        qkv_loader = with_rank_override(qkv_loader, rank=effective_q_rank)
        set_weight_loader(self.qkv_proj_weight, qkv_loader)

        qkv_bias_loader = mxfp4_fused_qkv_bias_loader(
            self.q_size,
            self.kv_size,
            self.num_kv_replicas,
            attention_dp_size=ddp,
            kv_needs_a2a=self.kv_needs_a2a,
        )
        qkv_bias_loader = with_rank_override(qkv_bias_loader, rank=effective_q_rank)
        set_weight_loader(self.qkv_proj_bias, qkv_bias_loader)

        o_loader = mxfp4_o_proj_weight_loader(
            (self.num_attention_heads * self.head_dim) // effective_q_shards,
            (
                (self.num_attention_heads * self.head_dim) // effective_q_shards,
                self.hidden_size,
            ),
            effective_q_shards,
        )
        o_loader = with_rank_override(o_loader, rank=effective_q_rank)
        set_weight_loader(self.o_proj_weight, o_loader)

        o_bias_loader = shuffling_weight_loader(
            0, (self.hidden_size,), None, effective_q_shards
        )
        o_bias_loader = with_rank_override(o_bias_loader, rank=effective_q_rank)
        set_weight_loader(self.o_proj_bias, o_bias_loader)

        # Sinks are per-head biases — pre-gathered across attention DP at load
        # time so no runtime all_gather is needed during decode.
        sinks_loader = _replicated_sinks_loader(
            shard_size=self.num_attention_heads_per_rank,
            num_shards=effective_q_shards,
            attention_dp_size=self.attention_dp_size,
            tp_rank=self.tp_group.rank_in_group,
        )
        set_weight_loader(self.sinks, sinks_loader)

    # ── Forward dispatch ─────────────────────────────────────────────────

    def forward(
        self,
        hidden_states,
        positions,
        position_embeddings,
        attn_metadata=None,
        attn_mask=None,
        rsag_dispatch: bool = False,
    ):
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                attn_mask,
                attn_metadata,
                rsag_dispatch=rsag_dispatch,
            )
        else:
            # >>> PARALLELISM: All-gather from SP before attention <<<
            if self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            return self.forward_prefill(
                hidden_states, positions, position_embeddings, attn_metadata
            )

    def _write_paged_kv_cache(self, k, v, slot_mapping, block_size):
        """Scatter post-RoPE K/V into the paged cache at slot_mapping positions.

        Used on the prefill paths where the cache write is NOT folded into the
        qkv kernel (full prefill needs raw K/V for flash attention; packed-FP8
        segmented prefill cannot use the in-kernel write). FP8 caches store
        ``fp8(clamp(tensor * scale))``; bf16 caches store directly. V is never
        packed; packed K is un-swizzled, scattered, then re-swizzled in place.
        """
        if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            k_flat = (
                (k.reshape(-1, self.head_dim) * self.k_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.k_cache.dtype)
            )
            v_flat = (
                (v.reshape(-1, self.head_dim) * self.v_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.k_cache.dtype)
            )
        else:
            k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
            v_flat = v.reshape(-1, self.head_dim).to(self.k_cache.dtype)

        nkh = self.num_key_value_heads_per_rank
        block_indices = (slot_mapping // block_size).repeat(nkh)
        position_indices = (slot_mapping % block_size).repeat(nkh)
        head_indices = torch.arange(
            nkh, dtype=torch.long, device=k.device
        ).repeat_interleave(slot_mapping.shape[0])
        index = (block_indices, head_indices, position_indices)

        # V is never packed → scatters directly.
        self.v_cache.index_put_(index, v_flat)
        if self.fp8_packed:
            # Packed K cache: un-swizzle, scatter, re-swizzle in place (see
            # model_bf16.forward_prefill for the rationale).
            k_unpacked = _unswizzle_packed_k(self.k_cache)
            k_unpacked.index_put_(index, k_flat)
            self.k_cache.copy_(_swizzle_packed_k(k_unpacked))
        else:
            self.k_cache.index_put_(index, k_flat)

    # ── Prefill path ─────────────────────────────────────────────────────

    def forward_prefill(
        self, hidden_states, positions, position_embeddings, attn_metadata=None
    ):
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # <-- MODEL-SPECIFIC: YaRN RoPE — fuse into qkv_proj kernel.
        # GptOssRotaryEmbedding emits cos/sin of shape [T, d_head/2]; the
        # kernel's split-in-half RoPE expects [B, T, d_head] with the
        # second-half cos/sin matching the first half (broadcast). cat
        # along last dim duplicates the values to satisfy this layout.
        cos, sin = position_embeddings
        cos_cache = torch.cat([cos, cos], dim=-1).unsqueeze(0)
        sin_cache = torch.cat([sin, sin], dim=-1).unsqueeze(0)

        # >>> PARALLELISM: KV cache metadata <<<
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        # >>> PARALLELISM: TP sharded QKV (with fused RoPE for Q and K) <<<
        # `num_kv_heads_for_weight` (not `num_key_value_heads_per_rank`) is the
        # K-block width in the projection output when `kv_needs_a2a=True`; the
        # kernel uses this to delimit which heads to rotate as K vs read as V.
        kv_is_fp8 = self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

        # ── KV cache write + Attention ────────────────────────────────────
        # <-- MODEL-SPECIFIC: sinks, sliding window
        #
        # Single combined segmented-prefill path (bf16 and packed-FP8 KV alike,
        # in-kernel post-RoPE write on both). Split by layer type:
        #   - SWA layers (self.sliding_window is not None): one fused
        #     swa_fused_cte call = qkv proj + RoPE + sliding-window attn
        #     (per-head sink + block-KV prior) + o-proj + in-kernel K/V write.
        #   - Non-SWA layers: qkv_proj folds the in-kernel K/V write, then
        #     segmented_attention attends over the cache.
        # Packed-FP8 no longer needs the external un-swizzle/scatter/re-swizzle
        # (qkv_cte + swa_fused_cte + attention_segmented_cte all speak the same
        # 5D per-head packed layout that bind_kv_cache allocates).
        used_swa_fused = False
        if kv_segment_size:
            # attention_segmented_cte cannot read a non-packed FP8 K cache;
            # fail fast on the unsupported combo (see helper for rationale).
            validate_fp8_segmented_supported(kv_is_fp8, self.fp8_packed)

            # FP8 write scale-arg direction: qkv_cte and swa_fused_cte quantize
            # by DIVIDING by the scale arg (internal reciprocal), whereas decode
            # and the external scatter store K*self.k_scale. self.k_scale is the
            # pre-inverted 1/s_ckpt, so to store the identical K/s_ckpt bytes the
            # in-kernel write must divide by s_ckpt = 1/self.k_scale. (No-op at
            # k_scale=v_scale=1, today's direct_cast checkpoints.)
            k_scale_write = (1.0 / self.k_scale) if kv_is_fp8 else None
            v_scale_write = (1.0 / self.v_scale) if kv_is_fp8 else None

            # Fused-SWA is only validated on the packed-FP8 KV cache (fp8_packed
            # is bound from the cache shape in bind_kv_cache); bf16-KV serving
            # always takes the qkv+segmented fallback below. Both conditions are
            # trace-time constants, so this folds away in the compiled graph.
            if self.fuse_swa and self.fp8_packed:
                # ── SWA layer (fused-SWA enabled): fully fused kernel
                #    (write + attn + o-proj). ──
                # prior_tokens is [1, 1] int32; sink is [B=1, num_q_heads] fp32.
                prior_tok = cached_seq_len.to(torch.int32).reshape(1, 1)
                sink_b = self.sinks.to(torch.float32).reshape(
                    1, self.num_attention_heads_per_rank
                )
                fused_out, _, _ = NF.swa_fused_attention(
                    hidden_states=hidden_states.unsqueeze(0),
                    qkv_weight=self.qkv_proj_weight,
                    op_weight=self.o_proj_weight,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    block_tables=block_table.to(torch.int32),
                    cos_cache=cos_cache,
                    sin_cache=sin_cache,
                    sink=sink_b,
                    prior_tokens=prior_tok,
                    qkv_bias=self.qkv_proj_bias.unsqueeze(0),
                    op_bias=self.o_proj_bias.unsqueeze(0),
                    # Kernel dequants the FP8 cache internally via k_scale/v_scale,
                    # so scale stays the pure softmax scale (no folded dequant).
                    scale=self.scaling,
                    sliding_window=self.sliding_window,
                    block_size=block_size,
                    num_q_heads=self.num_attention_heads_per_rank,
                    num_kv_heads=self.num_kv_heads_for_weight,
                    d_head=self.head_dim,
                    k_scale=k_scale_write,
                    v_scale=v_scale_write,
                )
                # Kernel emits the post-o-proj [B, S, H] output; skip the shared
                # external o_proj below.
                attn_output = fused_out.squeeze(0)
                used_swa_fused = True
            else:
                # ── Non-SWA layer or a non-packed (bf16) KV cache: in-kernel
                #    post-RoPE (packed-)FP8 K/V write folded into qkv_proj, then
                #    segmented attention.
                #    segmented_attention receives sliding_window below, so the
                #    window is still applied correctly on the SWA-fallback path. ──
                # Sanitize slot_mapping for the qkv_cte in-kernel scatter: it uses
                # slot_mapping values as direct DMA offsets into the cache, with no
                # oob_mode.skip — out-of-range slots cause a hardware OOB. Remap
                # sentinels (< 0) and stale values (>= num_blocks * block_size) to
                # slot 0; the K/V they would have written are unused (padding tokens
                # never read). Mirrors PR #2306's decode path. (The fused-SWA path
                # above scatters via block_tables and does not need this.)
                num_blocks_total = self.k_cache.shape[0]
                max_slot = num_blocks_total * block_size
                slot_mapping_clamped = torch.where(
                    (slot_mapping < 0) | (slot_mapping >= max_slot),
                    torch.zeros_like(slot_mapping),
                    slot_mapping,
                ).to(torch.int32)
                # qkv_cte reads slot_mapping as (batch, seqlen) and does
                # flatten_dims(0, 1) internally, so it must be 2D. Prefill is a
                # single sequence → reshape the flat [tokens] mapping to [1, tokens].
                slot_mapping_clamped = slot_mapping_clamped.reshape(1, tokens)
                q, _, _ = NF.qkv_proj(
                    hidden=hidden_states.unsqueeze(0),
                    qkv_weights=self.qkv_proj_weight,
                    bias=self.qkv_proj_bias.unsqueeze(0),
                    d_head=self.head_dim,
                    cos_cache=cos_cache,
                    sin_cache=sin_cache,
                    num_q_heads=self.num_attention_heads_per_rank,
                    num_kv_heads=self.num_kv_heads_for_weight,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    k_scale=k_scale_write,
                    v_scale=v_scale_write,
                    fp8_max=FP8_CLAMP_MAX if kv_is_fp8 else None,
                    fp8_min=-FP8_CLAMP_MAX if kv_is_fp8 else None,
                    kv_dtype=self.k_cache.dtype if kv_is_fp8 else None,
                    use_block_kv=True,
                    fp8_packed=self.fp8_packed,
                    block_size=block_size,
                    # slot_mapping_clamped is already [B, S] (reshaped to
                    # (1, tokens) above), which is the qkv_cte [B, S] contract.
                    # Do NOT .unsqueeze(0) — that was added by #2773 for the
                    # 1-D slot_mapping in model_bf16; here it would make the
                    # tensor 3-D (1, 1, tokens) and the kernel flattens dim(0,1)
                    # to a size-1 batch, tripping an out-of-bound slot_mapping
                    # access on the packed-FP8 prefill path.
                    slot_mapping=slot_mapping_clamped,
                )
                q = (
                    q.squeeze(0)
                    .view(tokens, self.num_attention_heads_per_rank, self.head_dim)
                    .transpose(0, 1)
                )

                # FP8 KV dequant for the segmented read: the in-kernel write
                # stored K/s_ckpt (= K*self.k_scale, identical bytes to the old
                # external scatter and to decode), so fold the K-dequant into the
                # softmax scale (scale = scaling / k_scale_float) and divide the
                # V-dequant out of the output. Do NOT pass k_scale/v_scale to the
                # segmented wrapper: its CPU fallback ignores them and expects a
                # caller-folded scale, and the two dequant conventions differ from
                # decode's. Both folds are exact scalar no-ops at k_scale=v_scale=1.
                attn_output = NF.segmented_attention(
                    q,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    block_tables=block_table,
                    prior_tokens=cached_seq_len,
                    block_size=block_size,
                    kv_segment_size=kv_segment_size,
                    scale=self.scaling / self.k_scale_float,
                    sliding_window=self.sliding_window,
                    sink=self.sinks.unsqueeze(1),
                    tp_q=True,
                    tp_out=True,
                    fp8_packed=self.fp8_packed,
                )  # [Nh, Dh, T]
                if self.v_scale_float != 1.0:
                    attn_output = attn_output / self.v_scale_float
        else:
            # Full prefill: kernel returns concatenated QKV; cache is written
            # via index_put_ since flash_attention needs raw K/V tensors.
            qkv = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=self.qkv_proj_bias.unsqueeze(0),
                d_head=self.head_dim,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                num_q_heads=self.num_attention_heads_per_rank,
                num_kv_heads=self.num_kv_heads_for_weight,
            ).squeeze(0)

            q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
            q = q.view(
                tokens, self.num_attention_heads_per_rank, self.head_dim
            ).transpose(0, 1)
            k = k.view(
                tokens, self.num_key_value_heads_per_rank, self.head_dim
            ).transpose(0, 1)
            v = v.view(
                tokens, self.num_key_value_heads_per_rank, self.head_dim
            ).transpose(0, 1)

            # KV cache update via index_put_ (flash_attention needs raw K/V).
            self._write_paged_kv_cache(k, v, slot_mapping, block_size)

            # <-- MODEL-SPECIFIC: GQA repeat for non-segmented path
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)

            attn_output = NF.flash_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v,
                scale=self.scaling,
                sliding_window=self.sliding_window,
                sink=self.sinks.unsqueeze(1),
                tp_q=False,
                tp_out=True,
            )  # [Nh, Dh, T]

        # The fused-SWA kernel already applied the output projection internally;
        # only the qkv+segmented and full-prefill paths need the external o_proj.
        if not used_swa_fused:
            attn_output = NF.o_proj(
                attn_output.unsqueeze(0),
                self.o_proj_weight,
                self.o_proj_bias.unsqueeze(0),
            ).squeeze(0)

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # ── Decode path ──────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states,
        positions,
        position_embeddings,
        attn_mask,
        attn_metadata,
        rsag_dispatch: bool = False,
    ):
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        # >>> PARALLELISM: All2All decode SP boundary <<<
        # All-gather the SP-sharded hidden_states across attn_tp_group before the kernel; post-O-proj RS re-shards.
        if self.use_all2all and self.world_size > 1:
            _B_local = block_table.shape[0]
            _T_pre = hidden_states.shape[0]
            _B_full = _B_local * self.attention_dp_size
            assert (_T_pre * self.world_size) % _B_full == 0, (
                "all2all decode shape mismatch: pre-AG hidden_states.shape[0]="
                f"{_T_pre}, B_local={_B_local}, attention_dp_size="
                f"{self.attention_dp_size}, world_size={self.world_size}. "
                "Expected token-SP shard "
                "(hidden_states tokens = B_local * attention_dp_size * S_decode "
                "/ world_size)."
            )
            hidden_states = self.attn_tp_group.all_gather(hidden_states, dim=0)

        B_local = block_table.shape[0]
        B = B_local * self.attention_dp_size
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)
        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_kernel = (
            cos[:, :half_d]
            .view(B_local, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(B_local, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )

        pos_ids = positions.view(1, B_local * S_decode)
        start_pos = None

        pos_ids_kernel = None
        swa_start_pos_ids_kernel = None

        if attn_mask is None:
            if self.sliding_window is not None:
                # The model runner has already trimmed block_table to the
                # window-relevant blocks for SWA layers and provided per-seq
                # swa_kv_pos_offset = start_block * block_size. Shift pos_ids into
                # the trimmed window frame before computing the causal mask.
                swa_kv_pos_offset = attn_metadata[layer_name].get("swa_kv_pos_offset")
                if swa_kv_pos_offset is not None:
                    pos_ids = pos_ids - swa_kv_pos_offset.view(B, 1).expand(
                        B, S_decode
                    ).reshape(1, B * S_decode)
                    # Clamp the trimmed-frame position to >= 0 (see
                    # _precompute_decode_attn_masks for the full rationale).
                    # No-op for a real row; for a no-token padding/freed row
                    # (positions zero-padded to 0, while swa_kv_pos_offset is
                    # positive) it prevents pos_ids = 0 - offset < 0 from
                    # driving the kernel's on-chip mask wrap branch to mark the
                    # whole window valid over an all-(-1) block_table ->
                    # stale-K NaN -> 1006.
                    pos_ids = torch.clamp(pos_ids, min=0)
                start_pos = torch.clamp(pos_ids - self.sliding_window + 1, min=0)

            pos_ids_kernel = pos_ids.view(B_local, S_decode).to(torch.float32)
            swa_start_pos_ids_kernel = (
                start_pos.view(B_local, S_decode).to(torch.float32)
                if start_pos is not None
                else None
            )

        # Packed FP8 decided per decode bucket (see model_bf16 for rationale):
        # viable buckets read the packed cache directly; non-viable buckets
        # un-swizzle to the standard layout, run the unpacked kernel, and
        # re-swizzle the in-kernel-updated buffer back into the packed cache.
        use_packed_kernel = self.fp8_packed and _packed_fp8_viable_for_bucket(
            block_len=block_size,
            bs=B_local,
            q_head=self.num_q_heads_after_a2a,
            s_active=S_decode,
            s_prior=S_ctx,
        )
        k_cache_arg = (
            self.k_cache
            if (use_packed_kernel or not self.fp8_packed)
            else _unswizzle_packed_k(self.k_cache)
        )

        # >>> PARALLELISM: Fused megakernel with TP-sharded weights <<<
        # In-kernel KV cache update (update_cache=True): the kernel writes K/V
        # in place and the FX aliasing pass threads the write back to the
        # K_cache/V_cache tensors passed in. With update_cache=True the API
        # returns only the attention output.
        output = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.unpadded_hidden_size,
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=self.qkv_proj_bias.unsqueeze(0),
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=block_table,
            K_cache=k_cache_arg,
            V_cache=self.v_cache,
            attention_mask=attn_mask,
            pos_ids=pos_ids_kernel,
            swa_start_pos_ids=swa_start_pos_ids_kernel,
            softmax_scale=self.scaling / self.k_scale_float,
            # <-- MODEL-SPECIFIC: Sink tokens
            sink=self.sinks.unsqueeze(1),
            update_cache=True,
            kv_cache_update_idx=slot_mapping.view(B_local, S_decode).to(torch.uint32),
            fp8_packed=use_packed_kernel,
            # Kernel currently requires fusing the V scale dequantization into W_out for KV quantization
            W_out=self.o_proj_weight / self.v_scale_float,
            bias_out=self.o_proj_bias.unsqueeze(0),
            transposed_out=False,
            out_in_sb=False,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            attention_dp=self.attention_dp_size,
            attention_dp_group=self.attention_dp_group.device_group
            if self.attention_dp_group
            else None,
            attention_dp_rank=self.attention_dp_rank,
            kv_needs_a2a=self.kv_needs_a2a,
        )

        # Non-viable packed bucket: re-swizzle the updated standard-layout
        # buffer back into the packed cache.
        if self.fp8_packed and not use_packed_kernel:
            self.k_cache.copy_(_swizzle_packed_k(k_cache_arg))

        # >>> PARALLELISM: Sum O-proj partials across TP * attn_dp <<<
        # ReduceScatter[TP] (token-shard to [T/tp, H]) on the all2all path and on
        # rsag_dispatch; otherwise AllReduce[TP].
        if (self.use_all2all or rsag_dispatch) and self.world_size > 1:
            output = self.attn_tp_group.reduce_scatter(output, dim=0)
        else:
            self.attn_tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: MoE Experts (MXFP4)
# Mixed: PARALLELISM (TP sharding, EP, cross-DP EP) +
#        MODEL-SPECIFIC (SwiGLU, routing) +
#        MXFP4 (tiled weight format, scales, shuffling)
# =============================================================================


class GptOssExperts(nn.Module):
    """Expert feed-forward layers with MXFP4 quantized weights.

    >>> PARALLELISM: TP + EP <<<
    <-- MODEL-SPECIFIC: 32 experts, top-4, SwiGLU with clamping
    <-- MXFP4: Weights in tiled uint16 blocks + uint8 scales, passed directly to kernel
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()

        # >>> PARALLELISM: TP + EP configuration <<<
        # - Without EP: experts are replicated, intermediate dim is TP-sharded.
        # - With EP (variable degree): experts are partitioned across ep_degree
        #   ranks, intermediate dim is sharded across tp_degree = world_size / ep_degree.
        #   Pure EP (ep_degree = world_size) gives tp_degree = 1 (no intermediate sharding).
        # moe_group = tp_group for outer collectives (all-reduce, reduce-scatter).
        # ep_tp_group = TP sub-group within EP partition for blockwise mapping.
        self.tp_group = get_tp_group()
        self.rank = self.tp_group.rank_in_group

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        self.ep_enabled = vllm_config.parallel_config.enable_expert_parallel
        self.dp_size = vllm_config.parallel_config.data_parallel_size

        # >>> PARALLELISM: MLP DP setup <<<
        self.mlp_dp_size = (
            config.neuron_config.mlp_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_mlp_tp_group,
        )

        self.mlp_tp_group = get_neuron_mlp_tp_group()
        self.mlp_tp_rank = self.mlp_tp_group.rank_in_group

        if self.ep_enabled:
            # EP enabled: read degree from neuron parallel state (always
            # initialized as GroupCoordinator when EP is active).
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_degree,
                get_neuron_ep_group,
                get_neuron_ep_rank,
                get_neuron_ep_tp_group,
            )

            self.ep_degree = get_neuron_ep_degree()
            self.ep_rank = get_neuron_ep_rank()
            # EP-TP sub-group: used for blockwise mapping TP coordination.
            # For pure EP (tp_degree=1) this is a single-rank group (no-op collectives).
            # For variable EP+TP this is the TP sub-group within the EP partition.
            self.ep_tp_group = get_neuron_ep_tp_group()
            self.tp_degree = self.ep_tp_group.world_size
            # all2all manager: used to orchestrate all2all dispatch/combine EP communication
            self.all2all_manager = (
                get_neuron_ep_group().device_communicator.all2all_manager
            )
            self.use_all2all = self.all2all_manager is not None
        else:
            # No EP: all experts on all ranks, intermediate dim sharded across
            # TP * mlp_dp_size via the MLP TP group.
            self.ep_degree = 1
            self.ep_rank = 0
            self.tp_degree = self.mlp_tp_group.world_size
            self.ep_tp_group = self.tp_group
            self.all2all_manager = None
            self.use_all2all = False
        self.moe_group = self.mlp_tp_group if not self.ep_enabled else self.tp_group

        # >>> PARALLELISM: Cross-DP EP <<<
        # When EP degree exceeds the TP group size, experts span across DP
        # replicas. MoE needs cross-DP collectives (all-gather/reduce-scatter)
        # to exchange tokens between DP replicas before/after expert computation.
        self.cross_dp_ep = self.dp_size > 1 and self.ep_enabled
        if self.cross_dp_ep:
            from vllm.distributed.parallel_state import (
                get_dp_group,
                get_wide_ep_group,
            )

            self.dp_group = get_dp_group()
            # World-spanning group with device communicator for cross-DP all-reduce
            self.wide_ep_group = get_wide_ep_group()

        self.total_num_experts = config.num_local_experts
        self.num_local_experts = config.num_local_experts // self.ep_degree
        self.num_experts_per_token = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # >>> PARALLELISM: Intermediate dim sharded across TP degree <<<
        self.intermediate_size_per_rank = self.intermediate_size // self.tp_degree
        self.block_size = 256

        # <-- MODEL-SPECIFIC: SwiGLU activation parameters
        self.alpha = config.swiglu_alpha
        self.limit = config.swiglu_limit

        # <-- MXFP4: Derived sizes for tiled format
        self.num_mx_groups = self.hidden_size // 32

        # <-- MODEL-SPECIFIC: Pre-MLP RMSNorm
        self.post_attention_layernorm = GptOssRMSNorm(config)

        # >>> PARALLELISM: Router weights replicated on all ranks (NOT EP-sharded) <<<
        self.router_weight = nn.Parameter(
            torch.empty(self.total_num_experts, self.hidden_size, dtype=torch.float16)
        )
        self.router_bias = nn.Parameter(
            torch.zeros(self.total_num_experts, dtype=torch.float16)
        )

        # <-- MXFP4: Expert weights in tiled format (uint16 blocks + uint8 scales)
        # gate_up_proj weight: [E_local, 128, 2, H//512, I_per_rank]
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                128,
                2,
                self.hidden_size // 512,
                self.intermediate_size_per_rank,
                dtype=torch.uint16,
            ),
            requires_grad=False,
        )
        # gate_up_proj scale: [E_local, 16, 2, H//512, I_per_rank]
        self.gate_up_proj_scale = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                16,
                2,
                self.hidden_size // 512,
                self.intermediate_size_per_rank,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(
                self.num_local_experts,
                self.intermediate_size_per_rank * 2,
                dtype=config.torch_dtype,
            )
        )

        # <-- MXFP4: down_proj weight tiled format
        num_I_tiles, q_blocks_per_I_tile = _get_i_tiling_shard_i(
            self.intermediate_size_per_rank
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                q_blocks_per_I_tile * Q_HEIGHT,
                num_I_tiles,
                self.hidden_size,
                dtype=torch.uint16,
            ),
            requires_grad=False,
        )
        self.down_proj_scale = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                q_blocks_per_I_tile,
                num_I_tiles,
                self.hidden_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.down_proj_bias = nn.Parameter(
            torch.zeros(
                self.num_local_experts, self.hidden_size, dtype=config.torch_dtype
            )
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """<-- MXFP4: EP-aware weight loaders for MXFP4 tiled format."""
        # Linear EP placement: rank k owns experts [k*L, (k+1)*L)
        local_expert_indices = list(
            range(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts,
            )
        )
        # The EP loaders pre-slice each input to this rank's experts, so the
        # expert dim of each expected_shape is the local expert count (equals
        # total_num_experts when ep_degree == 1).
        num_local_experts = len(local_expert_indices)

        # Router and norm: shuffled hidden dim, no shard
        set_weight_loader(
            self.router_weight,
            shuffling_weight_loader(
                shuffle_dim=1,
                expected_shard_shape=(self.total_num_experts, self.hidden_size),
                shard_spec=None,
            ),
        )
        set_weight_loader(
            self.post_attention_layernorm.weight,
            shuffling_weight_loader(
                shuffle_dim=0,
                expected_shard_shape=(self.hidden_size,),
                shard_spec=None,
            ),
        )

        # <-- MXFP4: Expert weights with EP filtering + TP sharding
        # with_rank_override ensures the MLP TP group rank is used for weight loading
        _ro = self.mlp_tp_rank

        gate_up_blocks = expert_parallel_mxfp4_gate_up_blocks_loader(
            local_expert_indices=local_expert_indices,
            shard_spec=ShardSpec(
                dim=1,
                size=self.intermediate_size_per_rank * 2,
                num_shards=self.tp_degree,
            ),
            expected_shape=(
                num_local_experts,
                self.intermediate_size_per_rank * 2,
                self.num_mx_groups,
                16,
            ),
        )
        set_weight_loader(
            self.gate_up_proj_weight, with_rank_override(gate_up_blocks, rank=_ro)
        )

        gate_up_scale = expert_parallel_mxfp4_gate_up_scale_loader(
            local_expert_indices=local_expert_indices,
            shard_spec=ShardSpec(
                dim=1,
                size=self.intermediate_size_per_rank * 2,
                num_shards=self.tp_degree,
            ),
            expected_shape=(
                num_local_experts,
                self.intermediate_size_per_rank * 2,
                self.num_mx_groups,
            ),
        )
        set_weight_loader(
            self.gate_up_proj_scale, with_rank_override(gate_up_scale, rank=_ro)
        )

        gate_up_bias = expert_parallel_mxfp4_gate_up_bias_loader(
            local_expert_indices=local_expert_indices,
            shard_spec=ShardSpec(
                dim=1,
                size=self.intermediate_size_per_rank * 2,
                num_shards=self.tp_degree,
            ),
            expected_shape=(
                num_local_experts,
                self.intermediate_size_per_rank * 2,
            ),
            hidden_act_bias=1.0,
        )
        set_weight_loader(
            self.gate_up_proj_bias, with_rank_override(gate_up_bias, rank=_ro)
        )

        down_weight = expert_parallel_mxfp4_down_weight_loader(
            local_expert_indices=local_expert_indices,
            shard_spec=ShardSpec(
                dim=2,
                size=self.intermediate_size_per_rank // 32,
                num_shards=self.tp_degree,
            ),
            expected_shape=(
                num_local_experts,
                self.hidden_size,
                self.intermediate_size_per_rank // 32,
                16,
            ),
        )
        set_weight_loader(
            self.down_proj_weight, with_rank_override(down_weight, rank=_ro)
        )

        down_scale = expert_parallel_mxfp4_down_scale_loader(
            local_expert_indices=local_expert_indices,
            shard_spec=ShardSpec(
                dim=2,
                size=self.intermediate_size_per_rank // 32,
                num_shards=self.tp_degree,
            ),
            expected_shape=(
                num_local_experts,
                self.hidden_size,
                self.intermediate_size_per_rank // 32,
            ),
        )
        set_weight_loader(
            self.down_proj_scale, with_rank_override(down_scale, rank=_ro)
        )

        down_bias = expert_parallel_mxfp4_down_bias_loader(
            local_expert_indices=local_expert_indices,
            expected_shape=(num_local_experts, self.hidden_size),
            scale=self.tp_degree,
        )
        set_weight_loader(self.down_proj_bias, with_rank_override(down_bias, rank=_ro))

    def forward(self, hidden_states, positions, is_decode, rank, rsag_dispatch=False):
        if is_decode:
            return self.forward_decode(hidden_states, rank, rsag_dispatch=rsag_dispatch)
        else:
            return self.forward_prefill(hidden_states, positions, rank)

    def _run_moe_block_tkg(self, hidden_states):
        """Run the fused MoE decode kernel.

        >>> PARALLELISM: use_all_experts and rank_id for EP <<<
        <-- MODEL-SPECIFIC: Activation, routing, clamping params
        <-- MXFP4: Passes scale params to kernel
        """
        total_tokens = hidden_states.shape[0]
        perc_experts_loaded = (
            total_tokens * self.num_experts_per_token / self.num_local_experts
        )
        use_all_experts = (
            perc_experts_loaded >= DEFAULT_SELECTIVE_LOADING_THRESHOLD
            or self.ep_degree > 1
        )
        rank_id = None
        if use_all_experts:
            rank_id = torch.tensor(
                [[self.ep_rank]], dtype=torch.int32, device=hidden_states.device
            )

        intermediate_size_per_partition = self.gate_up_proj_weight.shape[-1]
        num_I_TP_blocks = math.ceil(intermediate_size_per_partition / 512.0)
        I_TP_block_size = intermediate_size_per_partition // num_I_TP_blocks

        return NF.moe_block_tkg(
            inp=hidden_states.unsqueeze(0),
            gamma=self.post_attention_layernorm.weight.unsqueeze(0).to(torch.float32),
            router_weights=self.router_weight.T,
            expert_gate_up_weights=self.gate_up_proj_weight,
            expert_down_weights=self.down_proj_weight,
            rank_id=rank_id,
            top_k=self.num_experts_per_token,
            # <-- MXFP4: Scale parameters
            expert_gate_up_weights_scale=self.gate_up_proj_scale,
            expert_down_weights_scale=self.down_proj_scale,
            router_bias=self.router_bias.unsqueeze(0),
            expert_gate_up_bias=self.gate_up_proj_bias.view(
                self.num_local_experts,
                I_TP_block_size // Q_WIDTH,
                2,
                num_I_TP_blocks,
                Q_WIDTH,
            ),
            expert_down_bias=self.down_proj_bias,
            eps=self.rms_norm_eps,
            router_act_fn=RouterActFnType.SOFTMAX,
            router_pre_norm=False,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            hidden_act_fn=ActFnType.Swish,
            gate_clamp_upper_limit=self.limit,
            gate_clamp_lower_limit=None,
            up_clamp_upper_limit=self.limit + 1,
            up_clamp_lower_limit=-self.limit + 1,
            router_mm_dtype=nl.float16,
            hidden_actual=self.unpadded_hidden_size,
            is_all_expert=use_all_experts,
            skip_router_logits=True,
        )

    def _run_moe_tkg(self, hidden_states):
        """Run the MoE expert MLP decode kernel.

        >>> PARALLELISM: is_all_expert, is_all_expert_dynamic for EP
        >>> PARALLELISM: all_to_all_v_strategy for EP with all2all communication
        <-- MODEL-SPECIFIC: Activation, clamping params
        <-- MXFP4: Passes scale params to kernel
        """
        total_tokens = hidden_states.shape[0]
        rank_id = torch.tensor(
            [[self.ep_rank]], dtype=torch.int32, device=hidden_states.device
        )

        intermediate_size_per_partition = self.gate_up_proj_weight.shape[-1]
        num_I_TP_blocks = math.ceil(intermediate_size_per_partition / 512.0)
        I_TP_block_size = intermediate_size_per_partition // num_I_TP_blocks

        # FIXME: better algorithm to decide block size
        decode_block_size = max(4, total_tokens // 4)

        return NF.moe_tkg(
            hidden_input=hidden_states,
            expert_gate_up_weights=self.gate_up_proj_weight,
            expert_down_weights=self.down_proj_weight,
            expert_affinities=None,
            expert_index=None,
            is_all_expert=True,
            rank_id=rank_id,
            # <-- MXFP4: Scale parameters
            expert_gate_up_weights_scale=self.gate_up_proj_scale,
            expert_down_weights_scale=self.down_proj_scale,
            expert_gate_up_bias=self.gate_up_proj_bias.view(
                self.num_local_experts,
                I_TP_block_size // Q_WIDTH,
                2,
                num_I_TP_blocks,
                Q_WIDTH,
            ),
            expert_down_bias=self.down_proj_bias,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.Swish,
            gate_clamp_upper_limit=self.limit,
            gate_clamp_lower_limit=None,
            up_clamp_upper_limit=self.limit + 1,
            up_clamp_lower_limit=-self.limit + 1,
            is_all_expert_dynamic=True,
            block_size=decode_block_size,
            # >>> PARALLELISM: PACK_OUTPUT_ROWS is used to prepare output layout for all2all combine
            all_to_all_v_strategy=MoEAllToAllVStrategy.PACK_OUTPUT_ROWS,
        )

    def forward_decode(self, hidden_states, rank, rsag_dispatch=False):
        """Decode MoE forward.

        Two paths selected by `self.use_all2all`:
        - all2allv: EP with all2allv collective.
            >>> PARALLELISM: compute RMSNorm, router in SP, then use all2allv to dispatch tokens to EP ranks
        - fused: fused MoE kernel.
            >>> PARALLELISM: Cross-DP EP dispatch/combine. rsag_dispatch
            >>>   (B*S>=128) swaps AllGather[DP]/AllReduce[World] for
            >>>   AllGather[World]/ReduceScatter[World]; the layer's
            >>>   AllGather[TP] restores the layout.
        """

        # >>> PARALLELISM: Compute RMSNorm, Router TopK, and MXFP8 quantization in SP
        # >>> PARALLELISM: EP A2A-v dispatch
        if self.use_all2all:
            norm_quant_out, expert_index, expert_affinities_masked = (
                NF.rmsnorm_router_topk_tkg(
                    hidden_states=hidden_states.unsqueeze(0),
                    gamma=self.post_attention_layernorm.weight.unsqueeze(0).to(
                        torch.float32
                    ),
                    router_weights=self.router_weight.T,
                    router_bias=self.router_bias.unsqueeze(0),
                    eps=self.rms_norm_eps,
                    top_k=self.num_experts_per_token,
                    hidden_actual=self.unpadded_hidden_size,
                    # <-- MXFP4: Quantize to MXFP8 before all2all-v collective
                    quantization_type=QuantizationType.MX,
                    router_mm_dtype=torch.float16,
                    router_act_fn=RouterActFnType.SOFTMAX,
                )
            )

            hidden_states = self.all2all_manager.dispatch(
                hidden_states=norm_quant_out,
                topk_weights=expert_affinities_masked,
                topk_ids=expert_index,
                is_sequence_parallel=True,
            )
        # >>> PARALLELISM: Cross-DP EP dispatch <<<
        # rsag_dispatch: AllGather[World] the SP input; else AllGather[DP].
        elif self.cross_dp_ep:
            if rsag_dispatch:
                hidden_states = self.wide_ep_group.all_gather(hidden_states, dim=0)
            else:
                hidden_states = self.dp_group.all_gather(hidden_states, dim=0)

        # Dispatch to MoE kernel based on collective strategy
        moe_func = self._run_moe_tkg if self.use_all2all else self._run_moe_block_tkg
        output = moe_func(hidden_states)

        # >>> PARALLELISM: EP A2A-v combine + TopK reduction
        if self.use_all2all:
            output = self.all2all_manager.combine(output, is_sequence_parallel=True)
        # >>> PARALLELISM: Cross-DP EP combine <<<
        # rsag_dispatch: ReduceScatter[World] (stays token-sharded; the layer's
        #   AllGather[TP] restores it). Else AllReduce[World] + DP slice.
        elif self.cross_dp_ep:
            if rsag_dispatch:
                output = self.wide_ep_group.reduce_scatter(output, dim=0)
            else:
                output = self.wide_ep_group.all_reduce(output)
                dp_rank = self.dp_group.rank_in_group
                tokens_per_dp = output.shape[0] // self.dp_size
                output = output[dp_rank * tokens_per_dp : (dp_rank + 1) * tokens_per_dp]
        elif self.moe_group.world_size > 1:
            output = self.moe_group.all_reduce(output)

        return output

    def forward_prefill(self, hidden_states, positions, rank):
        hidden_states = self.post_attention_layernorm(hidden_states)

        expert_affinities = NF.router(
            hidden_states=hidden_states,
            router_weights=self.router_weight.T,
            top_k=self.num_experts_per_token,
            router_bias=self.router_bias,
            activation="softmax",
            computation_dtype=torch.float16,
        )

        # >>> PARALLELISM: All-gather from SP <<<
        if self.tp_group.world_size > 1:
            expert_affinities = self.tp_group.all_gather(expert_affinities, dim=0)
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # Compute padding mask from positions (True = real token, False = padding).
        # Must be computed before cross-DP gather since positions from different
        # DP replicas are not monotonically increasing when concatenated.
        padding_mask = None
        if positions is not None:
            last_real_idx = torch.argmax(positions)
            token_indices = torch.arange(positions.shape[0], device=positions.device)
            padding_mask = token_indices <= last_real_idx

        # >>> PARALLELISM: Cross-DP EP gather — collect tokens from all DP replicas <<<
        if self.cross_dp_ep:
            expert_affinities = self.dp_group.all_gather(expert_affinities, dim=0)
            hidden_states = self.dp_group.all_gather(hidden_states, dim=0)
            if padding_mask is not None:
                padding_mask = self.dp_group.all_gather(padding_mask, dim=0)

        # >>> PARALLELISM: Map global affinities to local experts <<<
        if self.ep_degree > 1:
            local_expert_indices = torch.arange(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts,
                device=hidden_states.device,
                dtype=torch.int32,
            )
            expert_affinities = NF.get_local_expert_affinities(
                expert_affinities, local_expert_indices
            )

        # Build blockwise mapping
        # >>> PARALLELISM: ep_tp_group provides TP rank coordination within
        # the EP partition. For pure TP this is the full tp_group; for pure EP
        # it is a single-rank group (no sharding); for variable EP+TP it is
        # the TP sub-group within the EP partition.
        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=expert_affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.num_experts_per_token,
            block_size=self.block_size,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
            padding_mask=padding_mask,
        )

        # <-- MXFP4: Conditions padded for dynamic-while
        conditions = torch.cat([conditions, torch.zeros(2, device=conditions.device)])

        I_TP = self.gate_up_proj_weight.shape[-1]
        num_I_TP_blocks = math.ceil(I_TP / 512)
        I_TP_block_size = I_TP // num_I_TP_blocks

        # <-- MXFP4: shard_on_block_mx implementation with scale params
        output = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block_mx,
            conditions=conditions,
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=self.gate_up_proj_weight,
            down_proj_weight=self.down_proj_weight,
            gate_and_up_proj_bias=self.gate_up_proj_bias.reshape(
                self.num_local_experts,
                I_TP_block_size // Q_WIDTH,
                2,
                num_I_TP_blocks,
                Q_WIDTH,
            ),
            down_proj_bias=self.down_proj_bias,
            activation_function=ActFnType.Swish,
            block_size=self.block_size,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            gate_up_proj_scale=self.gate_up_proj_scale,
            down_proj_scale=self.down_proj_scale,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            skip_weight=True,
            gate_clamp_upper_limit=self.limit,
            gate_clamp_lower_limit=None,
            up_clamp_upper_limit=self.limit + 1,
            up_clamp_lower_limit=-self.limit + 1,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
            # Routing shape for the kernel's auto-computed static/dynamic block
            # split (n_dynamic_blocks=-1). Best-case static estimate is
            # max(1, ceil(ceil(T*top_k / ep_degree) / block_size)).
            top_k=self.num_experts_per_token,
            ep_degree=self.ep_degree,
        )

        # >>> PARALLELISM: Cross-DP EP all-reduce and slice <<<
        if self.cross_dp_ep:
            # All-reduce across entire world (all TP and DP ranks)
            output = self.wide_ep_group.all_reduce(output)
            # Slice to keep only this DP replica's tokens
            dp_rank = self.dp_group.rank_in_group
            tokens_per_dp = output.shape[0] // self.dp_size
            start_idx = dp_rank * tokens_per_dp
            end_idx = start_idx + tokens_per_dp
            output = output[start_idx:end_idx]
            # Slice to keep only this TP rank's SP chunk
            tp_rank = self.moe_group.rank_in_group
            tokens_per_tp = output.shape[0] // self.moe_group.world_size
            start_idx = tp_rank * tokens_per_tp
            end_idx = start_idx + tokens_per_tp
            output = output[start_idx:end_idx]
        # >>> PARALLELISM: Combine expert results and return to SP layout <<<
        elif self.moe_group.world_size > 1:
            output = self.moe_group.reduce_scatter(output, dim=0)

        return output


# =============================================================================
# Section 5: MLP Wrapper
# =============================================================================


class GptOssMLP(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.experts = GptOssExperts(config)
        self.dtype = config.torch_dtype

    def forward(self, hidden_states, positions, is_decode, rank, rsag_dispatch=False):
        hidden_states = hidden_states.to(self.dtype)
        return self.experts(
            hidden_states,
            positions=positions,
            is_decode=is_decode,
            rank=rank,
            rsag_dispatch=rsag_dispatch,
        )


# =============================================================================
# Section 6: Decoder Layer
# =============================================================================


def _dp_transition(
    x: torch.Tensor,
    current_group,
    target_group,
    dim: int = 0,
) -> torch.Tensor:
    """Transition tensor between DP gathered states."""
    current_dp = current_group.world_size
    target_dp = target_group.world_size
    if current_dp == target_dp:
        return x
    if current_dp > target_dp:
        per_dp = x.shape[dim] // current_dp
        start = (current_group.rank_in_group // target_dp) * target_dp * per_dp
        return x.narrow(dim, start, target_dp * per_dp)
    per_dp = x.shape[dim] // current_dp
    x = x.narrow(dim, current_group.rank_in_group * per_dp, per_dp)
    return target_group.all_gather(x, dim=dim)


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = GptOssRMSNorm(config)
        self.self_attn = GptOssAttention(config, layer_idx=layer_idx)
        self.mlp = GptOssMLP(config)
        self.layer_idx = layer_idx

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        # >>> PARALLELISM: DP sizes for batch state transitions <<<
        nc = config.neuron_config
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_dp_group,
            get_neuron_mlp_dp_group,
        )

        self.attn_dp_group = get_neuron_attention_dp_group()
        self.mlp_dp_group = get_neuron_mlp_dp_group()

        # <-- MXFP4: Shuffled hidden dim for layernorm
        set_weight_loader(
            self.input_layernorm.weight,
            shuffling_weight_loader(
                shuffle_dim=0,
                expected_shard_shape=(config.hidden_size,),
                shard_spec=None,
            ),
        )

        # >>> PARALLELISM: All2All decode path detection (mirrors GptOssAttention/Experts) <<<
        self.use_all2all = _resolve_use_all2all(config)

    def _is_decode(self, attn_metadata):
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def _use_rsag_dispatch(self, hidden_states) -> bool:
        """Whether the decode cross-DP EP transition uses ReduceScatter[TP] ->
        AllGather[World] (token-sharded across the region) instead of
        AllReduce[TP] -> AllGather[DP].

        Mutually exclusive with the sparse all2all transport (its own transition).
        Gated: cross-DP EP, global B*S >= 128, divisible by the world size. The
        attn_dp == mlp_dp guard keeps the attention<->MoE _dp_transition a no-op
        so the residual slice is a plain narrow. ``hidden_states`` is the
        TP-replicated [B_local*S, H] decode tensor, so global B*S = shape[0] *
        dp_size (== the combine's ``output.shape[0]``).
        """
        experts = self.mlp.experts
        if (
            not experts.cross_dp_ep
            or self.use_all2all
            or self.attn_dp_group.world_size != self.mlp_dp_group.world_size
        ):
            return False
        dp_size = experts.dp_size
        global_tokens = hidden_states.shape[0] * dp_size
        world_size = self.world_size * dp_size
        return (
            global_tokens >= DECODE_RSAG_TOKEN_THRESHOLD
            and global_tokens % world_size == 0
        )

    def forward(
        self,
        hidden_states,
        positions,
        position_embeddings,
        attn_metadata=None,
        attn_mask=None,
        rank=None,
    ):
        is_decode = self._is_decode(attn_metadata)

        if not is_decode:
            return self._forward_prefill(
                hidden_states, positions, position_embeddings, attn_metadata, rank
            )

        return self._forward_decode(
            hidden_states,
            positions,
            position_embeddings,
            attn_metadata,
            rank,
            attn_mask,
        )

    def _forward_decode(
        self,
        hidden_states,
        positions,
        position_embeddings,
        attn_metadata,
        rank,
        attn_mask,
    ):
        # ── Decode: batch state transitions between modules ──
        # SP-sharded on the all2all path — skip the DP transition slice.
        if not getattr(self, "use_all2all", False):
            hidden_states = _dp_transition(
                hidden_states, self.mlp_dp_group, self.attn_dp_group
            )

        # Decide the DP->EP transition once and thread it into attention + MoE
        # so the paired collectives agree (excluded on the all2all path).
        rsag_dispatch = self._use_rsag_dispatch(hidden_states)

        residual = hidden_states
        # <-- MODEL-SPECIFIC: Pre-attention RMSNorm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
            attn_mask=attn_mask,
            rsag_dispatch=rsag_dispatch,
        )

        # rsag_dispatch: attention exits token-sharded [B_local*S/tp, H], so
        # narrow the TP-replicated residual to this rank's matching token range
        # (a plain slice, not reduce_scatter, which would sum the replicas).
        if rsag_dispatch:
            tp_rank = self.tp_group.rank_in_group
            per_rank = residual.shape[0] // self.world_size
            residual = residual.narrow(0, tp_rank * per_rank, per_rank)

        hidden_states = residual + hidden_states

        if not getattr(self, "use_all2all", False):
            hidden_states = _dp_transition(
                hidden_states, self.attn_dp_group, self.mlp_dp_group
            )

        residual = hidden_states
        hidden_states = self.mlp(
            hidden_states,
            positions=positions,
            is_decode=True,
            rank=rank,
            rsag_dispatch=rsag_dispatch,
        )
        # rsag_dispatch: MLP output and residual are token-sharded (combine's
        # ReduceScatter[World]); AllGather[TP] restores [B_local*S, H].
        hidden_states = residual + hidden_states
        if rsag_dispatch:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states

    def _forward_prefill(
        self, hidden_states, positions, position_embeddings, attn_metadata, rank
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(
            hidden_states, positions=positions, is_decode=False, rank=rank
        )
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 7: Model Backbone
# =============================================================================


class GptOssModel(nn.Module):
    def __init__(self, config: GptOssConfig, batch_size: int):
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Embedding TP group (TP * embedding_dp_size) + DP column <<<
        self.embedding_dp_size = (
            config.neuron_config.embedding_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_embedding_tp_group,
            get_neuron_embedding_dp_group,
            get_neuron_mlp_dp_group,
        )

        emb_tp_group = get_neuron_embedding_tp_group()
        emb_device_group = emb_tp_group.device_group
        self.embedding_dp_group = get_neuron_embedding_dp_group()
        self.embedding_tp_rank = emb_tp_group.rank_in_group
        self.mlp_dp_group = get_neuron_mlp_dp_group()

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=emb_device_group,
        )

        self.layers = nn.ModuleList(
            [
                GptOssDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = GptOssRMSNorm(config)
        self.rotary_emb = GptOssRotaryEmbedding(config)

        # <-- MXFP4: Shuffled weight loaders for embedding and final norm
        emb_loader = shuffling_weight_loader(
            shuffle_dim=1,
            expected_shard_shape=(
                self.embed_tokens.vocab_size_per_rank,
                config.hidden_size,
            ),
            shard_spec=ShardSpec(
                dim=0,
                size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
            ),
        )
        emb_loader = with_rank_override(emb_loader, rank=self.embedding_tp_rank)
        set_weight_loader(self.embed_tokens.weight, emb_loader)
        set_weight_loader(
            self.norm.weight,
            shuffling_weight_loader(
                shuffle_dim=0,
                expected_shard_shape=(config.hidden_size,),
                shard_spec=None,
            ),
        )

        # Eagle3 speculative decoding: layer indices whose hidden states
        # are collected for the draft model.  Empty until the drafter sets them.
        self.aux_hidden_state_layers = []

        # >>> PARALLELISM: All2All decode SP exit boundary <<<
        # SP-sharded through the stack when set — post-norm all-gather in forward() recovers the full sequence.
        self.use_all2all = _resolve_use_all2all(config)

    def forward(
        self,
        input_ids,
        positions,
        attn_metadata=None,
        rank=None,
        inputs_embeds=None,
        is_token_ids=None,
    ):
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        # >>> PARALLELISM: Embedding DP — gather input_ids (tiny, just token IDs) <<<
        if not is_prefill and self.embedding_dp_size > 1:
            input_ids = self.embedding_dp_group.all_gather(input_ids, dim=0)

        emb_rank = rank
        if rank is not None and self.embedding_dp_size > 1:
            emb_rank = rank + (self.embedding_tp_rank - self.rank)
        # SP-shard across tp_group on the all2all decode path (mirrors prefill); the matching AG is at end-of-forward.
        sp_sharded = is_prefill or self.use_all2all
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=sp_sharded, rank=emb_rank
        )

        # >>> PARALLELISM: Transition emb_dp → mlp_dp for decoder layers <<<
        # All2all decode keeps SP-sharded across tp_group; skip the DP slice which would over-shard.
        if not is_prefill and not self.use_all2all:
            hidden_states = _dp_transition(
                hidden_states, self.embedding_dp_group, self.mlp_dp_group
            )

        # <-- MODEL-SPECIFIC: GPT-OSS accepts unpadded prompt embeds and pads
        # them to the runtime hidden dim used by this model variant.
        if inputs_embeds is not None:
            target_dim = hidden_states.shape[-1]
            current_dim = inputs_embeds.shape[-1]
            if current_dim > target_dim:
                raise ValueError(
                    f"inputs_embeds dim ({current_dim}) exceeds hidden dim "
                    f"({target_dim}) for GPT-OSS prompt embeddings."
                )
            inputs_embeds = torch.nn.functional.pad(
                inputs_embeds, (0, target_dim - current_dim)
            )

        # >>> PARALLELISM: SP prompt-embed path <<<
        # Shard inputs_embeds/is_token_ids to match SP layout before merging.
        if (
            is_prefill
            and self.world_size > 1
            and inputs_embeds is not None
            and is_token_ids is not None
        ):
            local_len = hidden_states.shape[0]
            start = self.rank * local_len
            inputs_embeds = inputs_embeds[start : start + local_len]
            is_token_ids = is_token_ids[start : start + local_len]

        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        # <-- MODEL-SPECIFIC: Pre-compute decode attention masks ──────────────
        # SWA layers (even idx) get a sliding-window mask; non-SWA layers
        # (odd idx) get a full causal mask. Both are pre-computed so the
        # kernel skips in-layer mask generation.
        swa_mask = None
        full_mask = None
        if not is_prefill:
            swa_mask, full_mask = _precompute_decode_attn_masks(
                sliding_window=self.config.sliding_window,
                attn_metadata=attn_metadata,
                positions=positions,
                num_q_heads_after_a2a=self.layers[0].self_attn.num_q_heads_after_a2a,
            )

        aux_hidden_states = []
        for idx, decoder_layer in enumerate(self.layers):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states)
            # <-- MODEL-SPECIFIC: Sliding window on alternating (even) layers
            attn_mask = swa_mask if idx % 2 == 0 else full_mask
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_mask=attn_mask,
                attn_metadata=attn_metadata,
                rank=rank,
            )

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP all-gather <<<
        # All-gather the SP-sharded hidden_states so downstream (lm_head/sampler/Eagle3) sees the full sequence.
        sp_sharded = is_prefill or self.use_all2all
        if sp_sharded and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            # TODO: make Eagle3 drafter accept SP-partitioned aux states to avoid this all-gather
            aux_hidden_states = [
                self.tp_group.all_gather(aux, dim=0) for aux in aux_hidden_states
            ]

        return hidden_states, aux_hidden_states


# =============================================================================
# Section 8: Language Model Head
# =============================================================================


@async_speculative_decoding
class GptOssForCausalLM(nn.Module, SupportsEagle3):
    def __init__(self, config: GptOssConfig, batch_size: int):
        super().__init__()
        self.config = config
        self.model = GptOssModel(config, batch_size)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: LM Head TP group + DP column <<<
        self.lm_head_dp_size = (
            config.neuron_config.lm_head_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_lm_head_tp_group,
            get_neuron_lm_head_dp_group,
            get_neuron_mlp_dp_group,
        )

        lm_head_tp_group = get_neuron_lm_head_tp_group()
        self.lm_head_tp_group = lm_head_tp_group
        lm_head_device_group = lm_head_tp_group.device_group
        self.lm_head_dp_group = get_neuron_lm_head_dp_group()
        self.mlp_dp_group = get_neuron_mlp_dp_group()
        lm_head_tp_rank = lm_head_tp_group.rank_in_group
        self.lm_head_tp_rank = lm_head_tp_rank

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        debug_logits_enabled = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        ) or debug_logits_enabled

        # >>> PARALLELISM: Column-parallel LM head <<<
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=lm_head_device_group,
        )

        # <-- MXFP4: Shuffled weight loader for LM head
        lm_head_loader = shuffling_weight_loader(
            shuffle_dim=1,
            expected_shard_shape=(
                self.lm_head.out_features_per_rank,
                config.hidden_size,
            ),
            shard_spec=ShardSpec(
                dim=0,
                size=self.lm_head.out_features_per_rank,
                num_shards=self.lm_head.tp_size,
            ),
        )
        lm_head_loader = with_rank_override(lm_head_loader, rank=lm_head_tp_rank)
        set_weight_loader(self.lm_head.weight, lm_head_loader)

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=lm_head_device_group,
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        inputs_embeds=None,
        is_token_ids=None,
        attn_metadata=None,
        sampling_positions=None,
        sampling_params=None,
        spec_decode_metadata=None,
        logit_mask=None,
        rank=None,
        **kwargs,  # @async_speculative_decoding injects async-spec args
    ):
        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states, aux_hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

        # >>> PARALLELISM: Slice to local before sampling position selection <<<
        mlp_dp = self.mlp_dp_group.world_size
        if mlp_dp > 1:
            local_size = hidden_states.shape[0] // mlp_dp
            dp_rank = self.mlp_dp_group.rank_in_group
            hidden_states = hidden_states[
                dp_rank * local_size : (dp_rank + 1) * local_size
            ]

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )

        # >>> PARALLELISM: Transition local → lm_head_dp <<<
        if self.lm_head_dp_size > 1:
            hidden_states_for_logits = self.lm_head_dp_group.all_gather(
                hidden_states_for_logits, dim=0
            )
            if self.on_device_sampling_config is not None:
                # ODS samples on the gathered lm_head_dp row layout. Keep
                # per-row inputs aligned with logits until sampling finishes.
                if logit_mask is not None:
                    if logit_mask.shape[0] == sampling_positions.shape[0]:
                        logit_mask = self.lm_head_dp_group.all_gather(logit_mask, dim=0)
                    elif logit_mask.shape[0] != hidden_states_for_logits.shape[0]:
                        raise ValueError(
                            "logit_mask rows must match either local or "
                            "lm_head_dp-gathered sampling rows. "
                            f"Got logit_mask={logit_mask.shape}, "
                            f"local_rows={sampling_positions.shape[0]}, "
                            f"gathered_rows={hidden_states_for_logits.shape[0]}."
                        )
                if sampling_params is not None:
                    sampling_params = self.lm_head_dp_group.all_gather(
                        sampling_params, dim=0
                    )

        hidden_states_for_logits = hidden_states_for_logits.to(self.config.torch_dtype)
        logits = self.lm_head(hidden_states_for_logits)

        # >>> PARALLELISM: Gather sharded logits for logprobs computation <<<
        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.lm_head_tp_group.all_gather(logits, dim=1)

        # >>> PARALLELISM: Slice back to local batch <<<
        if self.lm_head_dp_size > 1:
            B_local = sampling_positions.shape[0]
            dp_rank = self.lm_head_dp_group.rank_in_group
            if self.on_device_sampling_config is None:
                logits = logits[dp_rank * B_local : (dp_rank + 1) * B_local]
            if gathered_logits is not None:
                gathered_logits = gathered_logits[
                    dp_rank * B_local : (dp_rank + 1) * B_local
                ]

        # EAGLE3 draft warmup traces auxiliary hidden states using the target
        # model dtype, so the draft NEFF's input dtype is fixed at trace time.
        # The dtype contract is enforced at the source (the MoE fallback in
        # functional/moe/moe_cte.py); no cast here, so any future upstream
        # fp32 leak fails loudly at the drafter boundary instead of being
        # silently masked (mirrors the bf16 model; the cat keeps the target
        # NEFF emitting a single tensor instead of a list).
        aux_hidden_states_concat = None
        if len(aux_hidden_states) > 0:
            aux_hidden_states_concat = torch.cat(aux_hidden_states, dim=-1)

        if self.on_device_sampling_config is None:
            if aux_hidden_states_concat is not None:
                return logits, aux_hidden_states_concat
            return logits

        lm_head_rank = rank
        if rank is not None and self.lm_head_dp_size > 1:
            lm_head_rank = rank + (self.lm_head_tp_rank - self.rank)
        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=lm_head_rank
        )
        if self.lm_head_dp_size > 1:
            # Slice sampled tokens after sampler; slicing logits earlier would
            # desynchronize SO masks from lm_head_dp-gathered logits.
            # spec_decode_metadata is local to this worker's input batch, so
            # rejection_sampler below also expects the local lm_head_dp row slice.
            B_local = sampling_positions.shape[0]
            dp_rank = self.lm_head_dp_group.rank_in_group
            sampled_tokens = sampled_tokens[dp_rank * B_local : (dp_rank + 1) * B_local]

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            rejection_sampled_tokens = rejection_sampler(
                spec_decode_metadata,
                sampled_tokens,
            )
            if aux_hidden_states_concat is not None:
                return (
                    rejection_sampled_tokens,
                    aux_hidden_states_concat,
                    gathered_logits,
                )
            return rejection_sampled_tokens

        if aux_hidden_states_concat is not None:
            return sampled_tokens, aux_hidden_states_concat, gathered_logits
        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = GptOssConfig.from_configs(hf_config, neuron_config)
        return cls(config, batch_size=1)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if layers is not None:
            self.model.aux_hidden_state_layers = list(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        if self.model.aux_hidden_state_layers:
            return tuple(self.model.aux_hidden_state_layers)
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    # >>> PARALLELISM: KV spec uses per-rank head counts (TP-sharded) <<<
    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=layer.self_attn.sliding_window,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches):
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            k_cache = kv_caches[layer_name][0]
            v_cache = kv_caches[layer_name][1]
            layer.self_attn.k_cache = k_cache
            layer.self_attn.v_cache = v_cache
            # Packed FP8 K cache is one rank higher than V (never packed); see
            # model_bf16.bind_kv_cache. The hasattr guard keeps this safe for any
            # attention class that predates the flag.
            if hasattr(layer.self_attn, "fp8_packed"):
                layer.self_attn.fp8_packed = k_cache.dim() == v_cache.dim() + 1

    def load_weights(self, checkpoint_path, device, cache_dir):
        """Load MXFP4 weights from checkpoint.

        >>> PARALLELISM: Weight loaders handle TP sharding <<<
        <-- MODEL-SPECIFIC: HF checkpoint key → model parameter mappings
        <-- MXFP4: Expert weights map to separate blocks/scales checkpoint keys
        """
        tp_rank = self.rank
        tp_size = self.world_size

        mappings = dict()
        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"

            # Attention
            mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{prefix}.self_attn.qkv_proj_bias"] = [
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.bias",
            ]
            mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                f"{prefix}.self_attn.o_proj.weight"
            )
            mappings[f"{prefix}.self_attn.o_proj_bias"] = (
                f"{prefix}.self_attn.o_proj.bias"
            )
            mappings[f"{prefix}.self_attn.sinks"] = f"{prefix}.self_attn.sinks"
            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )

            # MoE
            mappings[f"{prefix}.mlp.experts.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )
            mappings[f"{prefix}.mlp.experts.router_weight"] = (
                f"{prefix}.mlp.router.weight"
            )
            mappings[f"{prefix}.mlp.experts.router_bias"] = f"{prefix}.mlp.router.bias"

            # <-- MXFP4: Separate blocks and scales checkpoint keys
            mappings[f"{prefix}.mlp.experts.gate_up_proj_weight"] = (
                f"{prefix}.mlp.experts.gate_up_proj_blocks"
            )
            mappings[f"{prefix}.mlp.experts.gate_up_proj_scale"] = (
                f"{prefix}.mlp.experts.gate_up_proj_scales"
            )
            mappings[f"{prefix}.mlp.experts.gate_up_proj_bias"] = (
                f"{prefix}.mlp.experts.gate_up_proj_bias"
            )
            mappings[f"{prefix}.mlp.experts.down_proj_weight"] = (
                f"{prefix}.mlp.experts.down_proj_blocks"
            )
            mappings[f"{prefix}.mlp.experts.down_proj_scale"] = (
                f"{prefix}.mlp.experts.down_proj_scales"
            )
            mappings[f"{prefix}.mlp.experts.down_proj_bias"] = (
                f"{prefix}.mlp.experts.down_proj_bias"
            )

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        ).state_dict

        self._load_kv_cache_scales(checkpoint, device)

        self.load_state_dict(rank_sharded, strict=False, assign=True)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Lightweight weight loading used during CPU compile.

        Loads ONLY the tiny KV-cache quantization scales (not the full MXFP4
        weights) so the traced graph bakes the correct ``softmax_scale`` /
        ``W_out`` scale constants instead of the 1.0 defaults. Expert weights
        and their scales enter the graph as parameters (not Python-float
        constants), so they are not needed at trace time.
        """
        from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES
        from vllm.config import get_current_vllm_config

        # Scales only exist for quantized KV caches; nothing to load otherwise.
        # Skipping the checkpoint I/O here lets CPU compile run with
        # skip_weights (no safetensors on disk) when the KV cache is unquantized.
        if (
            get_current_vllm_config().cache_config.cache_dtype
            not in QUANTIZED_KV_CACHE_DTYPES
        ):
            return
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        # load_weights() populates the tensor index via load_sharded_pipelined;
        # the lite path skips that, so index explicitly or every key lookup
        # below misses and silently leaves the scales at 1.0.
        checkpoint._ensure_indexed()
        self._load_kv_cache_scales(checkpoint, device)

    def _load_kv_cache_scales(
        self, checkpoint: SafetensorsCheckpoint, device: torch.device
    ):
        """Load KV cache quantization scales from checkpoint if provided."""
        from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()

        for layer_id in range(len(self.model.layers)):
            attn = self.model.layers[layer_id].self_attn

            if vllm_config.cache_config.cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
                continue

            for scale_name in ("k_scale", "v_scale"):
                key = f"model.layers.{layer_id}.self_attn.{scale_name}"
                if key in checkpoint._tensor_name_to_file:
                    # Invert scales: checkpoint stores scales for (tensor / scale),
                    # but the kernel quantizes via (tensor * scale), so invert once here.
                    val = 1.0 / checkpoint._get_slice(key)[:].to(
                        dtype=torch.bfloat16, device=device
                    )
                else:
                    val = torch.ones(1, dtype=torch.bfloat16, device=device)
                setattr(attn, scale_name, val.reshape(1, 1))

            attn.k_scale_float = attn.k_scale.item()
            attn.v_scale_float = attn.v_scale.item()
