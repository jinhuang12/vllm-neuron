# SPDX-License-Identifier: Apache-2.0
import logging

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.nn as neuron_nn
from vllm_neuron import functional as NF
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.llama3.model import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    last_dim_padding_weight_loader,
    set_weight_loader,
    sharding_weight_loader_with_padding,
)

logger = logging.getLogger(__name__)


def extract_accepted_tokens(
    input_ids: torch.Tensor,
    sampling_positions: torch.Tensor,
    raw_sampled_token_ids: torch.Tensor,
    vocab_size: int,
    num_speculative_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """On-device extract_next_token_ids + compute_token_indices_to_sample.

    1. Finds the last valid token per row in raw_sampled_token_ids (valid = not -1
       and < vocab_size). Scatters it into input_ids at sampling_positions.
    2. Adjusts sampling_positions by subtracting rejected token count per request.

    Args:
        input_ids: Shifted target token IDs [T].
        sampling_positions: Indices into input_ids for each request [bs].
        raw_sampled_token_ids: [bs, max_spec_len+1] with -1 for rejected,
            or [bs, 1] for prefill.
        vocab_size: Model vocabulary size.
        num_speculative_tokens: Number of speculative tokens per step.

    Returns:
        (patched_input_ids, adjusted_sampling_positions, next_token_ids)
        where next_token_ids is the bonus (last valid) token per request
        [bs] — exposed so the draft NEFF output can include it as the
        first column, allowing the next target NEFF to consume a single
        device future without any intermediate Python-side ops.
    """
    valid_mask = (raw_sampled_token_ids != -1) & (raw_sampled_token_ids < vocab_size)
    valid_count = valid_mask.sum(dim=1)  # [bs]
    last_valid_idx = torch.clamp(valid_count - 1, min=0)  # [bs]
    next_token_ids = (
        raw_sampled_token_ids.gather(1, last_valid_idx.unsqueeze(1).to(torch.long))
        .squeeze(1)
        .to(torch.int32)
    )  # [bs]
    # Guard: if all entries in a row are invalid, use 0 instead of garbage
    next_token_ids = torch.where(
        valid_count > 0, next_token_ids, torch.zeros_like(next_token_ids)
    )

    # Adjust sampling_positions for rejected tokens BEFORE scatter.
    # Prefill: raw_sampled has 1 column → no adjustment needed.
    # Steady-state: raw_sampled has num_spec+1 columns → adjust for rejections.
    if raw_sampled_token_ids.shape[1] > 1:
        num_rejected = torch.clamp(num_speculative_tokens + 1 - valid_count, min=0)
        sampling_positions = torch.clamp(
            sampling_positions - num_rejected.to(sampling_positions.dtype), min=0
        )

    # Patch input_ids at adjusted sampling_positions with next_token_ids
    input_ids = input_ids.scatter(0, sampling_positions.to(torch.long), next_token_ids)

    return input_ids, sampling_positions, next_token_ids


def compute_slot_mapping(
    positions: torch.Tensor,
    block_table_tensor: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Compute slot_mapping on-device from positions and block table.

    Maps each position to its KV cache slot index:
        slot = block_id * block_size + (position % block_size)

    where block_id is looked up from block_table_tensor using the block
    number derived from the position.

    Args:
        positions: Token positions [bs].
        block_table_tensor: Block table [bs, max_blocks_per_seq].
        block_size: Number of slots per block.

    Returns:
        Slot indices [bs].
    """
    block_numbers = positions // block_size
    block_ids = block_table_tensor.gather(dim=1, index=block_numbers.view(-1, 1)).view(
        -1
    )
    slot = block_ids * block_size + (positions % block_size)
    # If the gathered block id is the inactive-block sentinel (-1), propagate
    # PAD_SLOT_ID (-1) to slot mapping
    return torch.where(block_ids < 0, torch.full_like(slot, -1), slot)


def _make_rmsnorm(config: LlamaConfig) -> LlamaRMSNorm:
    """Create RMSNorm, padding-aware if hidden_size != unpadded_hidden_size.

    When the draft model uses padded hidden dimensions (e.g. GPT-OSS 2880→3072),
    variance must be computed only on the unpadded portion to match GPU behavior.
    """
    if config.hidden_size != config.unpadded_hidden_size:
        return _PaddedRMSNorm(config)
    return LlamaRMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)


class _PaddedRMSNorm(nn.Module):
    """RMSNorm that computes variance only on the unpadded hidden dimensions."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.torch_dtype)
        )
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = (
            hidden_states[..., : self.unpadded_hidden_size]
            .pow(2)
            .mean(-1, keepdim=True)
        )
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * hidden_states
        output[..., self.unpadded_hidden_size :] = 0.0
        return output.to(input_dtype)


def fc_interleaved_padding_weight_loader(
    unpadded_hidden: int, padded_hidden: int
) -> SafetensorsWeightLoader:
    """Pad fc weight with interleaved zeros for 3 aux hidden states.

    The fc layer in Eagle3 combines 3 auxiliary hidden states from the target model.
    When the target model uses padded hidden dimensions, the fc weight needs to be
    padded to match.

    Checkpoint weight: [unpadded_hidden, unpadded_hidden * 3]
    Padded weight:     [padded_hidden, padded_hidden * 3]

    The input dimension has 3 concatenated aux hidden states, each needs
    padding inserted after its real data portion.

    Args:
        unpadded_hidden: Original hidden size (e.g., 2880)
        padded_hidden: Target hidden size after padding (e.g., 3072)

    Returns:
        SafetensorsWeightLoader that pads the fc weight appropriately.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, f"Expected 1 slice for fc weight, got {len(slices)}"
        assert padded_hidden >= unpadded_hidden, (
            f"padded_hidden ({padded_hidden}) must be >= unpadded_hidden ({unpadded_hidden})"
        )

        weight = slices[0][:]  # [out=unpadded_hidden, in=unpadded_hidden * 3]

        if padded_hidden == unpadded_hidden:
            return weight  # No padding needed

        # Split input dim into 3 aux parts, pad each
        aux_weights = torch.split(weight, unpadded_hidden, dim=1)
        in_pad = padded_hidden - unpadded_hidden
        padded_aux = [torch.nn.functional.pad(w, (0, in_pad)) for w in aux_weights]
        weight = torch.cat(padded_aux, dim=1)  # [unpadded_hidden, padded_hidden * 3]

        # Pad output dim
        out_pad = padded_hidden - weight.shape[0]
        if out_pad > 0:
            weight = torch.nn.functional.pad(weight, (0, 0, 0, out_pad))

        return weight  # [padded_hidden, padded_hidden * 3]

    return SafetensorsWeightLoader(transform=transform)


def embedding_sharding_padding_weight_loader(
    vocab_size_per_rank: int,
    num_shards: int,
    padded_hidden_size: int,
) -> SafetensorsWeightLoader:
    """Shard embedding along vocab dim (0) and pad along hidden dim (1).

    This loader handles both:
    1. Vocabulary dimension sharding for tensor parallelism
    2. Hidden dimension padding for alignment requirements

    Args:
        vocab_size_per_rank: Size of each vocabulary shard per rank.
        num_shards: Total number of shards (TP size).
        padded_hidden_size: Target hidden dimension after padding.

    Returns:
        SafetensorsWeightLoader that shards and pads the embedding.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, (
            f"Expected 1 slice for embedding weight, got {len(slices)}"
        )
        # Get the full embedding weight
        full_weight = slices[0][:]  # [vocab_size, hidden_size]

        # Shard along vocab dimension
        start_idx = rank * vocab_size_per_rank
        end_idx = start_idx + vocab_size_per_rank
        # Handle case where last shard might be smaller
        actual_end = min(end_idx, full_weight.shape[0])
        sharded_weight = full_weight[start_idx:actual_end]

        # Pad to full shard size if needed (for last rank)
        if sharded_weight.shape[0] < vocab_size_per_rank:
            pad_rows = vocab_size_per_rank - sharded_weight.shape[0]
            sharded_weight = torch.nn.functional.pad(
                sharded_weight, (0, 0, 0, pad_rows)
            )

        # Pad hidden dimension
        pad_amount = padded_hidden_size - sharded_weight.shape[-1]
        if pad_amount > 0:
            sharded_weight = torch.nn.functional.pad(sharded_weight, (0, pad_amount))

        return sharded_weight  # [shard_size, padded_hidden_size]

    return SafetensorsWeightLoader(transform=transform)


def _eagle3_qkv_weight_loader(
    q_size,
    kv_size,
    shard_dim,
    num_shards,
    is_storage_transposed,
    num_kv_replicas,
    padded_hidden_size,
    unpadded_hidden_size,
):
    """QKV weight loader with interleaved padding for eagle3's concat input.

    The eagle3 QKV input is concat(embeds, hidden_states), each independently
    padded. The weight's input dimension must match this layout:
        [embeds_unpadded, pad, hidden_unpadded, pad]
    NOT:
        [embeds_unpadded, hidden_unpadded, pad_at_end]
    """
    base_loader = fused_qkv_weight_loader(
        q_size=q_size,
        kv_size=kv_size,
        shard_dim=shard_dim,
        num_shards=num_shards,
        is_storage_transposed=is_storage_transposed,
        num_kv_replicas=num_kv_replicas,
        # No padding in base loader — we handle it here
        padded_hidden_size=None,
        unpadded_hidden_size=None,
    )

    pad = padded_hidden_size - unpadded_hidden_size
    if pad <= 0:
        return base_loader

    def transform(slices, rank):
        # Get unpadded fused QKV: [2*unpadded, qkv_size]
        result = base_loader.transform(slices, rank)
        # Split into embeds half and hidden half
        embeds_w = result[:unpadded_hidden_size, :]
        hidden_w = result[unpadded_hidden_size:, :]
        # Pad each half independently
        embeds_w = torch.nn.functional.pad(embeds_w, (0, 0, 0, pad))
        hidden_w = torch.nn.functional.pad(hidden_w, (0, 0, 0, pad))
        return torch.cat([embeds_w, hidden_w], dim=0)

    return SafetensorsWeightLoader(transform=transform)


class Eagle3LlamaDecoderLayer(LlamaDecoderLayer):
    """Eagle3 decoder layer that extends LlamaDecoderLayer.

    Dual-mode, mirroring upstream ``llama_eagle3.py`` ``LlamaDecoderLayer``
    (:51, :95-108). The mode is selected by ``is_fusion_layer``:

    Fusion mode (logical layer 0 — the eagle3 ``midlayer``):
    1. Takes both `embeds` (token embeddings) and `hidden_states` (from target model)
    2. Concatenates them before attention: [embeds, hidden_states] -> 2x hidden_size
    3. Additional `hidden_norm` for normalizing target hidden states
    4. Overrides QKV projection to accept 2x hidden_size input

    Plain mode (logical layers 1..N-1 — ordinary Llama decoder layers):
    1. Ignores `embeds`; consumes `hidden_states` only (1x hidden_size QKV)
    2. No `hidden_norm`, no embeds concat — a standard residual-chained
       Llama decoder layer (``input_layernorm(hidden) -> attn -> residual ->
       post_attention_layernorm -> mlp -> residual``).

    ``is_fusion_layer`` defaults to True so the single-layer construction path
    (RedHat-style 1-layer drafters, and every prior caller) is unchanged.
    """

    def __init__(
        self, config: LlamaConfig, layer_idx: int, is_fusion_layer: bool = True
    ):
        super().__init__(config, layer_idx)
        self.config = config
        self.is_fusion_layer = is_fusion_layer

        padded = config.hidden_size != config.unpadded_hidden_size

        # Override norms with padding-aware variants when hidden_size is padded
        if padded:
            self.input_layernorm = _make_rmsnorm(config)
            self.post_attention_layernorm = _make_rmsnorm(config)

        attn = self.self_attn
        if is_fusion_layer:
            # Fusion layer: override QKV projection weight to accept 2x
            # hidden_size input (concat of embeds + target hidden states).
            qkv_size = attn.q_size + 2 * attn.kv_size
            qkv_input_size = 2 * config.hidden_size

            attn.qkv_proj_weight = nn.Parameter(
                torch.empty(qkv_input_size, qkv_size, dtype=config.torch_dtype)
            )

            # Re-setup weight loader for the wider QKV.
            # Use interleaved padding: the input is concat(embeds, hidden_states),
            # each padded independently, so zeros appear at positions
            # [unpadded:hidden] and [hidden+unpadded:2*hidden], not at the end.
            set_weight_loader(
                attn.qkv_proj_weight,
                _eagle3_qkv_weight_loader(
                    q_size=attn.q_size,
                    kv_size=attn.kv_size,
                    shard_dim=1,
                    num_shards=attn.world_size,
                    is_storage_transposed=True,
                    num_kv_replicas=attn.num_kv_replicas,
                    padded_hidden_size=config.hidden_size,
                    unpadded_hidden_size=config.unpadded_hidden_size,
                ),
            )
        elif padded:
            # Plain layer: base LlamaAttention already built a 1x-hidden QKV,
            # but its loader has no hidden-dim padding. Re-set with end-padding
            # on the single hidden input segment (no interleave — there is only
            # one hidden segment, not concat(embeds, hidden)).
            set_weight_loader(
                attn.qkv_proj_weight,
                fused_qkv_weight_loader(
                    q_size=attn.q_size,
                    kv_size=attn.kv_size,
                    shard_dim=1,
                    num_shards=attn.world_size,
                    is_storage_transposed=True,
                    num_kv_replicas=attn.num_kv_replicas,
                    padded_hidden_size=config.hidden_size,
                    unpadded_hidden_size=config.unpadded_hidden_size,
                ),
            )
        # else (plain + unpadded): base loader is already correct.

        # Override weight loaders for o_proj and MLP when hidden_size is padded
        if padded:
            # o_proj: [num_heads*head_dim // tp, hidden_size] — pad hidden_size dim (dim 1)
            set_weight_loader(
                attn.o_proj_weight,
                sharding_weight_loader_with_padding(
                    shard_dim=0,
                    shard_size=(attn.num_attention_heads * attn.head_dim)
                    // attn.world_size,
                    num_shards=attn.world_size,
                    is_storage_transposed=True,
                    pad_dim=1,
                    padded_size=config.hidden_size,
                    unpadded_size=config.unpadded_hidden_size,
                ),
            )

            mlp = self.mlp
            intermediate_per_rank = config.intermediate_size // mlp.world_size
            # gate_proj/up_proj: [hidden_size, intermediate // tp] — pad hidden_size dim (dim 0)
            gate_up_loader = sharding_weight_loader_with_padding(
                shard_dim=1,
                shard_size=intermediate_per_rank,
                num_shards=mlp.world_size,
                is_storage_transposed=True,
                pad_dim=0,
                padded_size=config.hidden_size,
                unpadded_size=config.unpadded_hidden_size,
            )
            set_weight_loader(mlp.gate_proj_weight, gate_up_loader)
            set_weight_loader(mlp.up_proj_weight, gate_up_loader)
            # down_proj: [intermediate // tp, hidden_size] — pad hidden_size dim (dim 1)
            set_weight_loader(
                mlp.down_proj_weight,
                sharding_weight_loader_with_padding(
                    shard_dim=0,
                    shard_size=intermediate_per_rank,
                    num_shards=mlp.world_size,
                    is_storage_transposed=True,
                    pad_dim=1,
                    padded_size=config.hidden_size,
                    unpadded_size=config.unpadded_hidden_size,
                ),
            )

        # Additional norm for target hidden states — fusion layer only.
        # Plain layers (1..N-1) do NOT fuse target hidden states, so they
        # carry no ``hidden_norm`` param (the checkpoint has none, and
        # ``load_state_dict(strict=True)`` must not see an extra param).
        if is_fusion_layer:
            self.hidden_norm = _make_rmsnorm(config)

    def forward(
        self,
        embeds: torch.Tensor,  # [T, hidden_size] (fusion layer only)
        hidden_states: torch.Tensor,  # [T, hidden_size]
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Fusion layer: eagle3 concat(embeds, target-hidden) attention.
        Plain layer: ordinary Llama decoder layer over ``hidden_states``.

        Returns:
            hidden_states: Output tensor of shape [T, hidden_size].
        """
        is_decode = self._is_decode(attn_metadata)

        if self.is_fusion_layer:
            # Normalize token embeddings
            embeds = self.input_layernorm(embeds)

            # Normalize target hidden states and save residual
            if self.config.norm_before_residual:
                hidden_states = self.hidden_norm(hidden_states)
                residual = hidden_states
            else:
                residual = hidden_states
                hidden_states = self.hidden_norm(hidden_states)

            # Eagle3: concatenate embeddings with hidden states
            attn_input = torch.cat(
                [embeds, hidden_states], dim=-1
            )  # [T, 2*hidden_size]
        else:
            # Plain Llama decoder layer: pre-attention norm on hidden only,
            # no embeds concat, no hidden_norm (upstream :106-108).
            residual = hidden_states
            attn_input = self.input_layernorm(hidden_states)

        # Self Attention (QKV input width matches the layer mode)
        hidden_states = self.self_attn(
            hidden_states=attn_input,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )

        # Post-attention residual + norm + MLP
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        hidden_states = residual + hidden_states

        return hidden_states


class Eagle3LlamaModel(nn.Module):
    """Eagle3 model backbone without the language model head.

    Contains embedding, decoder layers, hidden state combination, normalization,
    and rotary embeddings. Used as the base model inside Eagle3LlamaForCausalLM.

    Args:
        config: LlamaConfig with model hyperparameters.
        start_layer_idx: Layer index for the first decoder layer.
    """

    def __init__(
        self,
        config: LlamaConfig,
        start_layer_idx: int,
    ):
        super().__init__()
        self.config = config

        hidden_size = config.hidden_size
        unpadded_hidden_size = config.unpadded_hidden_size

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=hidden_size,
            dtype=config.torch_dtype,
            tp_group=get_tp_group().device_group,
        )

        # N-layer eagle3 backbone (upstream llama_eagle3.py :163-171). Logical
        # layer 0 is the eagle3 fusion layer (``midlayer`` in the checkpoint:
        # 2x-width qkv, hidden_norm, embeds concat); logical layers 1..N-1 are
        # plain Llama decoder layers (1x-width qkv, no hidden_norm). Absolute
        # ``layer_idx`` starts at ``start_layer_idx`` (the drafter's KV/graph
        # layer names are offset past the target's layers), so fusion selection
        # keys on the LOGICAL index, not the absolute one.
        assert config.num_hidden_layers >= 1
        self.layers = nn.ModuleList(
            [
                Eagle3LlamaDecoderLayer(
                    config,
                    layer_idx=start_layer_idx + logical_idx,
                    is_fusion_layer=(logical_idx == 0),
                )
                for logical_idx in range(config.num_hidden_layers)
            ]
        )

        # Combine 3 auxiliary hidden states from target model
        self.fc = nn.Linear(
            hidden_size * 3,
            hidden_size,
            bias=False,
            dtype=config.torch_dtype,
        )

        self.norm = _make_rmsnorm(config)
        self.rotary_emb = LlamaRotaryEmbedding(config)

        # Weight loaders
        set_weight_loader(
            self.embed_tokens.weight,
            embedding_sharding_padding_weight_loader(
                vocab_size_per_rank=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
                padded_hidden_size=hidden_size,
            ),
        )
        set_weight_loader(
            self.fc.weight,
            fc_interleaved_padding_weight_loader(unpadded_hidden_size, hidden_size),
        )
        set_weight_loader(self.norm.weight, last_dim_padding_weight_loader(hidden_size))
        for layer in self.layers:
            set_weight_loader(
                layer.input_layernorm.weight,
                last_dim_padding_weight_loader(hidden_size),
            )
            # hidden_norm exists only on the fusion layer (logical layer 0).
            if layer.is_fusion_layer:
                set_weight_loader(
                    layer.hidden_norm.weight,
                    last_dim_padding_weight_loader(hidden_size),
                )
            set_weight_loader(
                layer.post_attention_layernorm.weight,
                last_dim_padding_weight_loader(hidden_size),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.LongTensor,
        target_hidden_states: torch.Tensor,  # [T, hidden_size] (already combined)
        attn_metadata: object | None = None,
        rank: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Eagle3 model backbone.

        Args:
            input_ids: Input token IDs of shape [T].
            positions: Position indices of shape [T].
            target_hidden_states: Hidden states from target model of shape [T, hidden_size].
            attn_metadata: Optional attention metadata for the attention layer.
            rank: Rank tensor for lookup tables.

        Returns:
            tuple: (hidden_states, hidden_prenorm) where hidden_states is the normalized output
                   and hidden_prenorm is the pre-norm state (used as recurrent state).
        """
        first_layer_name = f"layers.{self.layers[0].layer_idx}.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        embeds = self.embed_tokens(input_ids, scatter_tokens=is_prefill, rank=rank)

        # Scatter target_hidden_states to match SP layout
        if is_prefill and self.embed_tokens.tp_size > 1:
            chunk_size = target_hidden_states.shape[0] // self.embed_tokens.tp_size
            tp_rank = get_tensor_model_parallel_rank()
            target_hidden_states = target_hidden_states[
                tp_rank * chunk_size : (tp_rank + 1) * chunk_size
            ]

        # Position embeddings
        position_embeddings = self.rotary_emb(
            positions, device=embeds.device, dtype=embeds.dtype
        )

        # Residual-chained forward over all N layers (upstream
        # llama_eagle3.py :210-218). Logical layer 0 (fusion) consumes both
        # ``embeds`` and the target hidden states; logical layers 1..N-1
        # (plain) ignore ``embeds`` and chain on the running hidden state.
        # Each plugin decoder layer owns its residual internally, so chaining
        # is a plain hidden-state hand-off.
        hidden_states = target_hidden_states
        for layer in self.layers:
            hidden_states = layer(
                embeds=embeds,
                hidden_states=hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )

        # Save pre-norm state for recurrent (matches vLLM's hidden_prenorm)
        hidden_prenorm = hidden_states

        # Final norm
        hidden_states = self.norm(hidden_states)

        # All-gather back from SP layout
        if is_prefill and self.embed_tokens.tp_size > 1:
            tp_group = get_tp_group()
            hidden_states = tp_group.all_gather(hidden_states, dim=0)
            hidden_prenorm = tp_group.all_gather(hidden_prenorm, dim=0)

        return hidden_states, hidden_prenorm

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Combine 3 auxiliary hidden states into one via fc layer."""
        return self.fc(hidden_states)


class Eagle3LlamaForCausalLM(nn.Module):
    """Eagle3 draft model with hidden state combination."""

    def __init__(
        self,
        config: LlamaConfig,
        start_layer_idx: int,
    ):
        super().__init__()
        self.config = config

        on_device_sampling_config = None
        if config.neuron_config is not None:
            on_device_sampling_config = config.neuron_config.on_device_sampling_config
        self.on_device_sampling = on_device_sampling_config is not None

        # Return per-pass logits from forward() only when debug_logits is
        # enabled (tests validate per-pass logits against GPU goldens). In
        # production, the logits stack is unused — skipping it avoids a
        # torch.stack op and reduces the NEFF output schema size.
        self._return_logits = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )

        self.model = Eagle3LlamaModel(
            config=config,
            start_layer_idx=start_layer_idx,
        )

        lm_head_vocab = config.draft_vocab_size or config.vocab_size
        self.has_vocab_mapping = (
            config.draft_vocab_size is not None
            and config.draft_vocab_size != config.vocab_size
        )
        if self.has_vocab_mapping:
            self.register_buffer(
                "_draft_to_target",
                torch.zeros(config.draft_vocab_size, dtype=torch.int32),
                persistent=False,
            )

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            lm_head_vocab,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling,
        )

        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=self.lm_head.out_features_per_rank,
                num_shards=self.lm_head.tp_size,
                pad_dim=1,
                padded_size=config.hidden_size,
                unpadded_size=config.unpadded_hidden_size,
            ),
        )

        if self.on_device_sampling:
            self.sampler = Sampler(
                on_device_sampling_config, process_group=get_tp_group().device_group
            )

        # P-EAGLE (parallel drafting). Defaults keep the sequential path
        # byte-identical; EagleProposer sets these before load_weights when
        # parallel_drafting is enabled, and load_weights loads mask_hidden.
        self.parallel_drafting = False
        self.ptd_token_id = None

    def _sample_draft_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a draft token from logits (greedy or on-device)."""
        if self.on_device_sampling:
            return self.sampler(logits).to(torch.int32)
        return torch.argmax(logits, dim=-1).to(torch.int32)

    def _map_output_token_ids_to_target_vocab(
        self,
        stacked_tokens: torch.Tensor,
        drafts_only_stacked: torch.Tensor,
        bonus_token_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map draft outputs to target vocab while preserving target bonus IDs."""
        if not self.has_vocab_mapping:
            return stacked_tokens, drafts_only_stacked

        mapped_drafts_only = self._draft_to_target[drafts_only_stacked.long()]
        if bonus_token_ids is None:
            return mapped_drafts_only, mapped_drafts_only

        mapped_stacked = torch.cat(
            [bonus_token_ids.unsqueeze(1), mapped_drafts_only], dim=1
        )
        return mapped_stacked, mapped_drafts_only

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.LongTensor,
        initial_target_hidden_states: torch.Tensor,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        raw_sampled_token_ids: torch.Tensor | None = None,
        prev_sampled_token_ids: torch.Tensor | None = None,
        prev_num_draft_tokens: torch.Tensor | None = None,
        req_indices_per_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full draft propose: KV cache update + recurrent decode loop.

        Processes the initial KV cache update pass, then runs num_speculative_tokens - 1
        recurrent decode steps. The recurrent loop is unrolled by torch.compile into a
        single static graph since num_speculative_tokens is a compile-time constant.

        Args:
            input_ids: Shifted target token IDs [T].
            positions: Position indices [T].
            initial_target_hidden_states: Concatenated 3x auxiliary hidden
                states from the target model [T, hidden_size*3].
            attn_metadata: Attention metadata for the draft model layer.
            sampling_positions: Indices into the token sequence for each request [bs].
            rank: TP rank tensor for lookup tables.
            raw_sampled_token_ids: Raw rejection sampler output [bs, max_spec_len+1]
                with -1 for rejected, or [bs, 1] for prefill. Used by
                _extract_accepted_tokens to patch input_ids and adjust
                sampling_positions on-device.

        Returns:
            ``(stacked_tokens, drafts_only_stacked, stacked_logits)``.

            ``stacked_tokens`` is ``[bs, 1+num_spec]`` (column 0 = bonus from
            this step's rejection sampler, columns 1.. = drafts for the next
            step). Flatten with ``view(-1)`` for the next target NEFF's
            ``input_ids``.

            ``drafts_only_stacked`` is ``[bs, num_spec]`` (drafts only,
            contiguous). Flatten with ``view(-1)`` for the next target
            NEFF's ``spec_decode_metadata.draft_token_ids``. Separate output
            because slicing ``stacked_tokens[:, 1:]`` produces a non-contiguous
            view that ``.contiguous()`` cannot resolve on Neuron device.

            ``stacked_logits`` is ``[bs, num_spec, vocab]`` or ``None`` —
            populated only when ``debug_logits_dir`` is set
            (used by ``test_eagle3_fused_propose.py`` for per-pass logit
            validation). In production it's ``None`` to skip the
            ``torch.stack`` and shrink the NEFF output schema.
        """
        if (
            prev_sampled_token_ids is not None
            and prev_num_draft_tokens is not None
            and req_indices_per_token is not None
        ):
            positions, attn_metadata = (
                NF.correct_spec_decode_positions_and_slot_mapping(
                    positions,
                    attn_metadata,
                    prev_sampled_token_ids,
                    prev_num_draft_tokens,
                    req_indices_per_token,
                    self.config.vocab_size,
                )
            )

        # On-device extract_next_token_ids + compute_token_indices_to_sample
        if raw_sampled_token_ids is not None:
            input_ids, sampling_positions, bonus_token_ids = (
                self._extract_accepted_tokens(
                    input_ids,
                    sampling_positions,
                    raw_sampled_token_ids,
                )
            )
        else:
            bonus_token_ids = None

        # Initial pass: combine aux hidden states and update KV cache.
        target_hidden_states = self.model.combine_hidden_states(
            initial_target_hidden_states
        )

        # P-EAGLE single-pass parallel drafting. Opt-in via parallel_drafting;
        # the sequential recurrent path below is unchanged when the flag is off.
        if self.parallel_drafting:
            return self._forward_parallel(
                input_ids=input_ids,
                positions=positions,
                target_hidden_states=target_hidden_states,
                attn_metadata=attn_metadata,
                sampling_positions=sampling_positions,
                rank=rank,
                bonus_token_ids=bonus_token_ids,
            )

        hidden_states, recurrent_state = self.model(
            input_ids=input_ids,
            positions=positions,
            target_hidden_states=target_hidden_states,
            attn_metadata=attn_metadata,
            rank=rank,
        )

        # Gather per-request states at sampling positions
        hidden_states = hidden_states[sampling_positions]
        recurrent_state = recurrent_state[sampling_positions]
        cur_positions = positions[sampling_positions]

        # Sample first draft token
        logits = self.lm_head(hidden_states)
        draft_token_ids = self._sample_draft_token(logits)

        # Collect draft tokens. When bonus_token_ids is provided
        # (steady-state spec), prepend it so the output shape is
        # [bs, 1 + num_spec], directly consumable as the next target NEFF's
        # flattened input_ids (via view(-1)). This keeps the
        # bonus-extraction-and-cat inside this compiled graph — no
        # Python-side ops between draft NEFF and next target NEFF.
        all_draft_tokens = (
            [bonus_token_ids, draft_token_ids]
            if bonus_token_ids is not None
            else [draft_token_ids]
        )
        # Per-pass logits only collected in debug mode (saves a torch.stack
        # and shrinks the NEFF output schema in production).
        all_logits = [logits] if self._return_logits else None

        # Extract block_table and block_size from attn_metadata for slot_mapping
        first_layer_name = f"layers.{self.model.layers[0].layer_idx}.self_attn"
        base_meta = attn_metadata[first_layer_name]
        block_table_tensor = base_meta["block_table_tensor"]
        block_size = base_meta["block_size"]
        max_blocks_per_seq = base_meta["max_blocks_per_seq"]

        # Recurrent decode loop (torch.compile unrolls this Python for-loop
        # into a single static graph since num_speculative_tokens is constant).
        # When num_speculative_tokens == 1, loop runs 0 times.
        for _ in range(self.num_speculative_tokens - 1):
            cur_input_ids = draft_token_ids
            # Map draft token IDs to target vocab for embedding lookup
            if self.has_vocab_mapping:
                cur_input_ids = self._draft_to_target[cur_input_ids.long()]
            cur_positions = cur_positions + 1

            # On-device slot_mapping computation
            slot_mapping = compute_slot_mapping(
                cur_positions, block_table_tensor, block_size
            )

            # Build step attn_metadata (only slot_mapping changes)
            step_attn_metadata = {}
            for layer in self.model.layers:
                ln = f"layers.{layer.layer_idx}.self_attn"
                step_attn_metadata[ln] = {
                    "block_table_tensor": block_table_tensor,
                    "slot_mapping": slot_mapping,
                    "max_query_len": 1,
                    "block_size": block_size,
                    "max_blocks_per_seq": max_blocks_per_seq,
                    "decode_token_threshold": 1,
                }

            # Recurrent decode step
            hidden_states, recurrent_state = self.model(
                input_ids=cur_input_ids,
                positions=cur_positions,
                target_hidden_states=recurrent_state,
                attn_metadata=step_attn_metadata,
                rank=rank,
            )

            # Sample next draft token
            logits = self.lm_head(hidden_states)
            draft_token_ids = self._sample_draft_token(logits)

            all_draft_tokens.append(draft_token_ids)
            if all_logits is not None:
                all_logits.append(logits)

        stacked_tokens = torch.stack(all_draft_tokens, dim=1)
        # Drafts-only stack: stride is (1,) so the runner can flatten with
        # ``.view(-1)`` and feed it as the next step's
        # ``spec_decode_metadata.draft_token_ids`` without an on-device
        # ``[:, 1:].contiguous()`` (which is unsupported on Neuron). When
        # bonus_token_ids is None (prefill / first decode), this is a copy
        # of stacked_tokens; under torch.compile, CSE elides the
        # duplication.
        drafts_only_stacked = torch.stack(
            all_draft_tokens[1:] if bonus_token_ids is not None else all_draft_tokens,
            dim=1,
        )

        stacked_tokens, drafts_only_stacked = (
            self._map_output_token_ids_to_target_vocab(
                stacked_tokens, drafts_only_stacked, bonus_token_ids
            )
        )

        stacked_logits = (
            torch.stack(all_logits, dim=1) if all_logits is not None else None
        )
        return stacked_tokens, drafts_only_stacked, stacked_logits

    def _forward_parallel(
        self,
        input_ids: torch.Tensor,
        positions: torch.LongTensor,
        target_hidden_states: torch.Tensor,
        attn_metadata: object,
        sampling_positions: torch.Tensor,
        rank: torch.Tensor | None,
        bonus_token_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """P-EAGLE single-pass parallel-drafting forward.

        Replaces the sequential recurrent loop with ONE backbone pass over
        ``num_speculative_tokens`` (K) slots per request. Slot 0 carries the
        real seed token and its fc-combined target hidden state; slots
        ``1..K-1`` carry ``ptd_token_id`` and the learned ``mask_hidden``
        vector (also fc-combined). All K draft tokens are sampled from the
        single pass.

        Mirrors upstream ``SpecDecodeBaseProposer.propose`` parallel branch
        (vllm/v1/spec_decode/llm_base_proposer.py:471-473, one forward then
        greedy-sample all K) and mask-hidden substitution (:743-753 with the
        fc projection of mask_hidden at :1208-1218). Static shapes as functions
        of ``(batch, K)``; no data-dependent control flow.
        """
        assert self.ptd_token_id is not None, (
            "parallel_drafting forward requires ptd_token_id; EagleProposer "
            "resolves it from the draft config."
        )
        assert hasattr(self, "mask_hidden"), (
            "parallel_drafting forward requires the mask_hidden buffer; it is "
            "loaded from the drafter checkpoint in load_weights."
        )
        K = self.num_speculative_tokens
        bs = sampling_positions.shape[0]
        hidden_dim = target_hidden_states.shape[-1]

        # Per-request seed: the real token, its position, and combined hidden.
        seed_tokens = input_ids[sampling_positions].to(torch.int32)
        if self.has_vocab_mapping:
            seed_tokens = self._draft_to_target[seed_tokens.long()].to(torch.int32)
        seed_positions = positions[sampling_positions]
        seed_hidden = target_hidden_states[sampling_positions]  # [bs, H]

        # Expand into the K-slot parallel-draft layout (increment-2 ops):
        # slot 0 = real seed token, slots 1..K-1 = ptd_token_id.
        expanded_ids = NF.build_parallel_draft_input_ids(
            seed_tokens, K, self.ptd_token_id
        )  # [bs, K]
        expanded_positions = NF.build_parallel_draft_positions(seed_positions, K)
        hidden_mask = NF.build_parallel_draft_hidden_mask(
            bs, K, device=seed_hidden.device
        )  # [bs, K]

        # fc-project the learned mask_hidden the same way real aux hidden
        # states are combined, then substitute it into the masked slots.
        mask_hidden_combined = self.model.combine_hidden_states(self.mask_hidden)
        expanded_hidden = seed_hidden.unsqueeze(1).expand(bs, K, hidden_dim)
        expanded_hidden = NF.substitute_mask_hidden(
            expanded_hidden, hidden_mask, mask_hidden_combined.view(-1)
        )  # [bs, K, H]

        flat_input_ids = expanded_ids.reshape(-1)  # [bs*K]
        flat_positions = expanded_positions.reshape(-1)  # [bs*K]
        flat_hidden = expanded_hidden.reshape(-1, hidden_dim)  # [bs*K, H]

        # Decode-style attn_metadata over K query tokens per request. Slot
        # mapping is computed per expanded slot; the block table is repeated
        # per slot for the gather, matching the sequential loop convention.
        first_layer_name = f"layers.{self.model.layers[0].layer_idx}.self_attn"
        base_meta = attn_metadata[first_layer_name]
        block_table_tensor = base_meta["block_table_tensor"]
        block_size = base_meta["block_size"]
        max_blocks_per_seq = base_meta["max_blocks_per_seq"]

        block_table_per_slot = block_table_tensor.repeat_interleave(K, dim=0)
        slot_mapping = compute_slot_mapping(
            flat_positions, block_table_per_slot, block_size
        )

        parallel_attn_metadata = {}
        for layer in self.model.layers:
            ln = f"layers.{layer.layer_idx}.self_attn"
            parallel_attn_metadata[ln] = {
                "block_table_tensor": block_table_tensor,
                "slot_mapping": slot_mapping,
                "max_query_len": K,
                "block_size": block_size,
                "max_blocks_per_seq": max_blocks_per_seq,
                "decode_token_threshold": K,
            }

        hidden_states, _ = self.model(
            input_ids=flat_input_ids,
            positions=flat_positions,
            target_hidden_states=flat_hidden,
            attn_metadata=parallel_attn_metadata,
            rank=rank,
        )

        logits = self.lm_head(hidden_states)  # [bs*K, vocab]
        draft_token_ids = self._sample_draft_token(logits)  # [bs*K]
        drafts_only_stacked = draft_token_ids.view(bs, K)  # [bs, K]

        # Match the sequential output contract: prepend the bonus token when
        # available so stacked_tokens is [bs, 1+K], drafts_only is [bs, K].
        if bonus_token_ids is not None:
            stacked_tokens = torch.cat(
                [bonus_token_ids.view(bs, 1), drafts_only_stacked], dim=1
            )
        else:
            stacked_tokens = drafts_only_stacked

        stacked_tokens, drafts_only_stacked = (
            self._map_output_token_ids_to_target_vocab(
                stacked_tokens, drafts_only_stacked, bonus_token_ids
            )
        )
        # Parallel path returns no per-pass logits stack (debug-logits are a
        # sequential-only feature); keep the 3-tuple output contract.
        return stacked_tokens, drafts_only_stacked, None

    def _load_mask_hidden(self, checkpoint_path: str) -> None:
        """Load the learned ``mask_hidden`` vector for parallel drafting.

        Mirrors upstream ``Eagle3LlamaForCausalLM.load_weights``
        (vllm/model_executor/models/llama_eagle3.py:404-409): when
        ``parallel_drafting`` is enabled the drafter checkpoint MUST contain a
        ``mask_hidden`` tensor, else raise a clear ValueError. When
        ``parallel_drafting`` is off this is a no-op, so existing sequential
        EAGLE3 checkpoints (which need not ship mask_hidden) load unchanged.
        """
        if not self.parallel_drafting:
            return
        import os

        from safetensors import safe_open

        files = sorted(
            f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")
        )
        mask_hidden = None
        for f in files:
            with safe_open(os.path.join(checkpoint_path, f), framework="pt") as sf:
                if "mask_hidden" in sf.keys():
                    mask_hidden = sf.get_tensor("mask_hidden")
                    break
        if mask_hidden is None:
            raise ValueError(
                "mask_hidden not found in the drafter checkpoint but the model "
                "is configured for parallel drafting (P-EAGLE). A "
                "parallel-drafting EAGLE3 checkpoint must ship a mask_hidden "
                "tensor (e.g. amazon/GPT-OSS-20B-P-EAGLE)."
            )
        padded = self._pad_mask_hidden(mask_hidden)
        self.register_buffer(
            "mask_hidden", padded.to(self.config.torch_dtype), persistent=False
        )

    def _pad_mask_hidden(self, mask_hidden: torch.Tensor) -> torch.Tensor:
        """Pad checkpoint mask_hidden to the (padded) fc input layout.

        The checkpoint stores mask_hidden as 3 concatenated aux hidden states
        at the unpadded hidden size. When the drafter runs with a padded hidden
        size (e.g. GPT-OSS 2880->3072), each aux segment is padded independently
        to match fc's interleaved-padded input layout (see
        ``fc_interleaved_padding_weight_loader``). Returns ``[1, 3*padded]``
        (or ``[1, N]`` unchanged when no padding applies).
        """
        unpadded = self.config.unpadded_hidden_size
        padded = self.config.hidden_size
        flat = mask_hidden.reshape(-1)
        if flat.numel() == 3 * unpadded and padded != unpadded:
            aux = torch.split(flat, unpadded)
            pad = padded - unpadded
            flat = torch.cat([torch.nn.functional.pad(a, (0, pad)) for a in aux])
        return flat.view(1, -1)

    def _extract_accepted_tokens(
        self,
        input_ids: torch.Tensor,
        sampling_positions: torch.Tensor,
        raw_sampled_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """On-device extract_next_token_ids + compute_token_indices_to_sample.

        Thin wrapper around module-level extract_accepted_tokens().
        See that function for full documentation.
        """
        return extract_accepted_tokens(
            input_ids,
            sampling_positions,
            raw_sampled_token_ids,
            self.config.vocab_size,
            self.num_speculative_tokens,
        )

    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ):
        llama_config = LlamaConfig.from_configs(
            hf_config=config, neuron_config=neuron_config
        )
        return cls(
            llama_config,
            start_layer_idx=start_layer_idx,
        )

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        """Load rank-sharded checkpoint to device with pipelined data movement.

        Supports two checkpoint naming conventions:
        - "midlayer.*" (used by some EAGLE3 checkpoints for layer 0)
        - "layers.{layer_id}.*" (standard naming)

        Args:
            checkpoint_path: Directory or HuggingFace model ID containing the
                checkpoint files.
            device: Target device for weight tensors.
            cache_dir: Cache directory for HuggingFace downloads.
        """
        import os

        from safetensors import safe_open

        if not os.path.isdir(checkpoint_path):
            from huggingface_hub import snapshot_download

            checkpoint_path = snapshot_download(checkpoint_path, cache_dir=cache_dir)

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        safetensor_files = sorted(
            f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")
        )
        checkpoint_keys = set()
        for f in safetensor_files:
            with safe_open(os.path.join(checkpoint_path, f), framework="pt") as sf:
                checkpoint_keys.update(sf.keys())

        use_midlayer_naming = any(k.startswith("midlayer.") for k in checkpoint_keys)

        mappings = {}
        mappings["model.embed_tokens.weight"] = "embed_tokens.weight"
        mappings["model.norm.weight"] = "norm.weight"
        mappings["model.fc.weight"] = "fc.weight"
        mappings["lm_head.weight"] = "lm_head.weight"

        for layer_id in range(len(self.model.layers)):
            if use_midlayer_naming and layer_id == 0:
                ckpt_prefix = "midlayer"
            else:
                ckpt_prefix = f"layers.{layer_id}"

            mappings[f"model.layers.{layer_id}.self_attn.qkv_proj_weight"] = [
                f"{ckpt_prefix}.self_attn.q_proj.weight",
                f"{ckpt_prefix}.self_attn.k_proj.weight",
                f"{ckpt_prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"model.layers.{layer_id}.self_attn.o_proj_weight"] = (
                f"{ckpt_prefix}.self_attn.o_proj.weight"
            )
            mappings[f"model.layers.{layer_id}.mlp.up_proj_weight"] = (
                f"{ckpt_prefix}.mlp.up_proj.weight"
            )
            mappings[f"model.layers.{layer_id}.mlp.gate_proj_weight"] = (
                f"{ckpt_prefix}.mlp.gate_proj.weight"
            )
            mappings[f"model.layers.{layer_id}.mlp.down_proj_weight"] = (
                f"{ckpt_prefix}.mlp.down_proj.weight"
            )
            mappings[f"model.layers.{layer_id}.input_layernorm.weight"] = (
                f"{ckpt_prefix}.input_layernorm.weight"
            )
            mappings[f"model.layers.{layer_id}.post_attention_layernorm.weight"] = (
                f"{ckpt_prefix}.post_attention_layernorm.weight"
            )
            # hidden_norm exists only on the fusion layer (logical layer 0).
            # Plain layers (1..N-1) carry no hidden_norm — neither the param
            # nor a checkpoint tensor — so mapping it would break the strict
            # load on both the param side (missing) and checkpoint side.
            if self.model.layers[layer_id].is_fusion_layer:
                mappings[f"model.layers.{layer_id}.hidden_norm.weight"] = (
                    f"{ckpt_prefix}.hidden_norm.weight"
                )

        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        rank_sharded_checkpoint = checkpoint.load_sharded_pipelined(
            tp_rank,
            tp_size,
            self,
            mappings,
            device,
        ).state_dict
        self.load_state_dict(rank_sharded_checkpoint, strict=True, assign=True)

        # Load d2t and pre-compute direct mapping buffer
        if self.has_vocab_mapping:
            for f in safetensor_files:
                with safe_open(os.path.join(checkpoint_path, f), framework="pt") as sf:
                    if "d2t" in sf.keys():
                        d2t = sf.get_tensor("d2t")
                        # Pre-compute direct mapping as int32 buffer (moves with .to(device))
                        draft_ids = torch.arange(len(d2t))
                        setattr(
                            self,
                            "_draft_to_target",
                            (draft_ids + d2t).to(torch.int32),
                        )

        # P-EAGLE: load the learned mask_hidden vector when parallel drafting
        # is enabled (no-op otherwise; raises clearly if the checkpoint lacks
        # it). Mirrors upstream llama_eagle3.py:404-409.
        self._load_mask_hidden(checkpoint_path)

    def get_kv_spec(self):
        """Returns KV cache specification for vLLM integration.

        Returns:
            KVSpec: Specification containing layer-wise KV cache requirements.
        """
        layers = []
        for _, layer in enumerate(self.model.layers):
            layer_name = f"layers.{layer.layer_idx}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        """Binds pre-allocated KV cache tensors to the model layers.

        Args:
            kv_caches: Dictionary mapping layer names to [key_cache, value_cache] tensor pairs.

        Raises:
            RuntimeError: If KV cache for any required layer is not initialized.
        """
        for _, layer in enumerate(self.model.layers):
            layer_name = f"layers.{layer.layer_idx}.self_attn"
            if layer_name not in kv_caches:
                raise RuntimeError(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]
