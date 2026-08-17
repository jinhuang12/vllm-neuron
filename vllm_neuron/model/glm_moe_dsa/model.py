# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 main decoder and causal-LM integration for vLLM-Neuron."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from vllm_neuron.model.kv_cache import KVSpec
from vllm_neuron.nn import ColumnParallelLinear, Sampler
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .attention import GlmMoeDsaAttention, GlmMoeDsaRMSNorm
from .block_fp8 import (
    FP8_INVERSE_SCALE_ADJUSTMENT,
    FP8_STORAGE_SCALE,
)
from .cache import (
    MAIN_INDEXER_LAYER_INDICES,
    build_glm_mla_cache_spec,
    gather_paged_cache_pair,
    write_paged_cache_pair,
)
from .config import GlmMoeDsaConfig
from .indexer import GlmMoeDsaIndexer, pack_indexer_keys, unpack_indexer_keys
from .mlp import GlmMoeDsaSwiGLUMLP
from .moe import GlmMoeDsaMoE
from .quantization import PINNED_FP8
from .weight_loaders import (
    PINNED_EP,
    Disposition,
    TPShardSpec,
    classify_checkpoint_key,
    load_checkpoint_index,
    load_checkpoint_manifest,
    local_load_plan,
    tp_shard_spec_for_key,
)


@dataclass(frozen=True)
class WeightLoadReport:
    """Exact checkpoint accounting retained after a full or lightweight load."""

    indexed_keys: int
    local_keys: int
    load_targets: int
    fp8_scales: int
    skipped_mtp: int
    mapped_sources: int


def _causal_selection(batch: int, sequence: int, device: torch.device) -> torch.Tensor:
    key = torch.arange(sequence, dtype=torch.int64, device=device)
    query = key.view(1, sequence, 1)
    selected = key.view(1, 1, sequence).expand(batch, sequence, sequence)
    return torch.where(selected <= query, selected, -torch.ones_like(selected))


def _is_decode_from_metadata(
    attn_metadata: dict[str, dict[str, Any]] | None,
) -> bool:
    if not attn_metadata:
        return False
    first = next(iter(attn_metadata.values()))
    return int(first["max_query_len"]) <= int(first["decode_token_threshold"])


def _resolve_tp_groups(tp_group: Any, tensor_parallel_size: int) -> tuple[Any, Any]:
    """Return the vLLM collective wrapper and its torch process group."""

    if tensor_parallel_size == 1:
        return None, None
    if not dist.is_initialized():
        raise RuntimeError("GLM-5.2 TP64 requires initialized distributed state")
    if tp_group is not None:
        return tp_group, getattr(tp_group, "device_group", tp_group)

    from vllm.distributed.parallel_state import get_tp_group

    collective_group = get_tp_group()
    return collective_group, collective_group.device_group


class GlmMoeDsaDecoderLayer(nn.Module):
    """One pre-norm GLM-5.2 decoder layer."""

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        layer_idx: int,
        *,
        tensor_parallel_size: int,
        expert_parallel_size: int,
        expert_parallel_rank: int,
        tp_group: Any = None,
        collective_group: Any = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if layer_idx not in config.main_layer_indices:
            raise ValueError(f"layer {layer_idx} is outside main decoder execution")
        if config.num_attention_heads % tensor_parallel_size:
            raise ValueError(
                f"num_attention_heads={config.num_attention_heads} is not divisible "
                f"by TP={tensor_parallel_size}"
            )

        self.layer_idx = layer_idx
        self.is_dense = layer_idx < config.first_k_dense_replace
        self.is_moe = not self.is_dense
        self.tp_group = collective_group
        self.world_size = tensor_parallel_size
        self.input_layernorm = GlmMoeDsaRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=config.torch_dtype,
            device=device,
        )
        self.post_attention_layernorm = GlmMoeDsaRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=config.torch_dtype,
            device=device,
        )
        self.self_attn = GlmMoeDsaAttention(
            hidden_size=config.hidden_size,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            local_heads=config.num_attention_heads // tensor_parallel_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=float(config.rope_parameters.get("rope_theta", 8_000_000.0)),
            tp_group=tp_group,
            fp8_weights=bool(config.quantization_config),
            dtype=config.torch_dtype,
            device=device,
        )
        self.self_attn.indexer = None
        if layer_idx in MAIN_INDEXER_LAYER_INDICES:
            self.self_attn.indexer = GlmMoeDsaIndexer(
                hidden_size=config.hidden_size,
                q_lora_rank=config.q_lora_rank,
                topk=config.index_topk,
                rope_theta=float(config.rope_parameters.get("rope_theta", 8_000_000.0)),
                fp8_weights=bool(config.quantization_config),
                dtype=config.torch_dtype,
                device=device,
            )

        if self.is_dense:
            self.mlp = GlmMoeDsaSwiGLUMLP.dense_from_config(
                config,
                tensor_parallel_size=tensor_parallel_size,
                tensor_parallel_rank=expert_parallel_rank,
                device=device,
            )
        else:
            self.mlp = GlmMoeDsaMoE(
                config,
                tensor_parallel_size=tensor_parallel_size,
                expert_parallel_size=expert_parallel_size,
                expert_parallel_rank=expert_parallel_rank,
                device=device,
            )

        self.mla_k_cache: torch.Tensor | None = None
        self.mla_v_cache: torch.Tensor | None = None
        self.indexer_k_cache: torch.Tensor | None = None
        self.indexer_v_cache: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        *,
        selected_indices: torch.Tensor | None = None,
        is_decode: bool = False,
        attn_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("decoder hidden_states must be [batch, sequence, hidden]")

        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        projection = self.self_attn.project(normalized, positions)

        indexer = self.self_attn.indexer
        mla_name = f"model.layers.{self.layer_idx}.self_attn.mla_cache"
        mla_meta = attn_metadata.get(mla_name) if attn_metadata else None
        if mla_meta is not None:
            write_paged_cache_pair(
                self.mla_k_cache,
                self.mla_v_cache,
                projection.latent_cache,
                mla_meta["slot_mapping"],
                int(mla_meta["block_size"]),
            )
            attention_latents = None
        else:
            attention_latents = projection.latent_cache

        if indexer is not None:
            index_projection = indexer.project(normalized, projection.q_lora, positions)
            if mla_meta is not None:
                assert attn_metadata is not None
                indexer_name = (
                    f"model.layers.{self.layer_idx}.self_attn.indexer.k_cache"
                )
                indexer_meta = attn_metadata[indexer_name]
                packed_indexer_keys = pack_indexer_keys(index_projection.keys)
                write_paged_cache_pair(
                    self.indexer_k_cache,
                    self.indexer_v_cache,
                    packed_indexer_keys,
                    indexer_meta["slot_mapping"],
                    int(indexer_meta["block_size"]),
                )
                cached_indexer_keys = unpack_indexer_keys(
                    gather_paged_cache_pair(
                        self.indexer_k_cache,
                        self.indexer_v_cache,
                        indexer_meta["block_table_tensor"],
                    ),
                    dtype=index_projection.keys.dtype,
                )
                selected_indices = indexer.select_paged(
                    index_projection,
                    cached_indexer_keys,
                    positions,
                    indexer_meta["block_table_tensor"],
                    block_size=int(indexer_meta["block_size"]),
                    physical_block_count=self.indexer_k_cache.shape[0],
                )
            else:
                cached_indexer_keys = index_projection.keys
                selected_indices = indexer.select(
                    index_projection,
                    cached_indexer_keys,
                    positions,
                    positions,
                )
        elif selected_indices is None:
            selected_indices = _causal_selection(
                hidden_states.shape[0], hidden_states.shape[1], hidden_states.device
            )

        if mla_meta is not None:
            use_selected_latent_mla = self.self_attn.should_use_selected_latent_mla(
                projection.queries,
                selected_indices,
                mla_k_cache=self.mla_k_cache,
                mla_v_cache=self.mla_v_cache,
                block_table=mla_meta["block_table_tensor"],
                block_size=int(mla_meta["block_size"]),
                is_decode=is_decode,
            )
            if not use_selected_latent_mla:
                attention_latents = gather_paged_cache_pair(
                    self.mla_k_cache,
                    self.mla_v_cache,
                    mla_meta["block_table_tensor"],
                )
        else:
            use_selected_latent_mla = False
        if use_selected_latent_mla:
            assert mla_meta is not None
            assert self.mla_k_cache is not None
            assert self.mla_v_cache is not None
            attention_output = self.self_attn.attend_selected_latents(
                projection.queries,
                selected_indices,
                self.mla_k_cache,
                self.mla_v_cache,
                mla_meta["block_table_tensor"],
                int(mla_meta["block_size"]),
            )
        else:
            assert attention_latents is not None
            attention_output = self.self_attn.attend(
                projection.queries, attention_latents, selected_indices
            )
        hidden_states = residual + attention_output
        residual = hidden_states
        mlp_input = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            flat = mlp_input.reshape(-1, mlp_input.shape[-1])
            mlp_output = self.mlp(flat, is_decode=is_decode).view_as(mlp_input)
        else:
            mlp_output = self.mlp(mlp_input)
        if self.world_size > 1:
            if self.tp_group is None:
                raise RuntimeError("GLM-5.2 TP/EP collective group is not initialized")
            if hasattr(self.tp_group, "all_reduce"):
                mlp_output = self.tp_group.all_reduce(mlp_output)
            else:
                dist.all_reduce(mlp_output, group=self.tp_group)
        return residual + mlp_output, selected_indices


class GlmMoeDsaModel(nn.Module):
    """The 78 main decoder layers. Layer 78 MTP tensors are excluded."""

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        *,
        tensor_parallel_size: int,
        expert_parallel_size: int,
        expert_parallel_rank: int,
        tp_group: Any = None,
        collective_group: Any = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tensor_parallel_size = tensor_parallel_size
        self.embed_tokens = VocabDimShardedEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            device=device,
            tp_group=tp_group,
        )
        self.layers = nn.ModuleList(
            GlmMoeDsaDecoderLayer(
                config,
                layer_idx,
                tensor_parallel_size=tensor_parallel_size,
                expert_parallel_size=expert_parallel_size,
                expert_parallel_rank=expert_parallel_rank,
                tp_group=tp_group,
                collective_group=collective_group,
                device=device,
            )
            for layer_idx in config.main_layer_indices
        )
        self.norm = GlmMoeDsaRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=config.torch_dtype,
            device=device,
        )
        self.main_layer_indices = config.main_layer_indices
        self.excluded_mtp_layer_indices = config.mtp_layer_indices

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: dict[str, dict[str, Any]] | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError("GLM-5.2 eager forward accepts one packed sequence")
        hidden_states = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        if inputs_embeds is not None:
            if inputs_embeds.shape != hidden_states.shape:
                raise ValueError("inputs_embeds must match embedded token shape")
            if is_token_ids is None:
                hidden_states = inputs_embeds
            else:
                hidden_states = torch.where(
                    is_token_ids.to(torch.bool).unsqueeze(-1),
                    hidden_states,
                    inputs_embeds,
                )

        if attn_metadata:
            first_meta = next(iter(attn_metadata.values()))
            batch = first_meta["block_table_tensor"].shape[0]
            if hidden_states.shape[0] % batch:
                raise ValueError("packed token count must be divisible by cache batch")
            batched_hidden = hidden_states.view(batch, -1, hidden_states.shape[-1])
        else:
            batched_hidden = hidden_states.unsqueeze(0)
        batched_positions = positions.to(torch.int64)
        if batched_positions.ndim == 1:
            batched_positions = batched_positions.unsqueeze(0)
        selected_indices = None
        if (
            batched_positions.numel()
            == batched_hidden.shape[0] * batched_hidden.shape[1]
        ):
            batched_positions = batched_positions.reshape(
                batched_hidden.shape[0], batched_hidden.shape[1]
            )
        is_decode = _is_decode_from_metadata(attn_metadata)
        for layer in self.layers:
            batched_hidden, selected_indices = layer(
                batched_hidden,
                batched_positions,
                selected_indices=selected_indices,
                is_decode=is_decode,
                attn_metadata=attn_metadata,
            )
        normalized = self.norm(batched_hidden)
        return normalized.reshape(-1, normalized.shape[-1])


class GlmMoeDsaForCausalLM(nn.Module):
    """Direct vLLM-Neuron GLM-5.2 main-workload model."""

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        tensor_parallel_size: int,
        *,
        expert_parallel_rank: int = 0,
        tp_group: Any = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tensor_parallel_size = tensor_parallel_size
        self.world_size = tensor_parallel_size
        self.rank = expert_parallel_rank
        self.tp_group, linear_tp_group = _resolve_tp_groups(
            tp_group, tensor_parallel_size
        )
        self.model = GlmMoeDsaModel(
            config,
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=tensor_parallel_size,
            expert_parallel_rank=expert_parallel_rank,
            tp_group=linear_tp_group,
            collective_group=self.tp_group,
            device=device,
        )
        sampling_config = getattr(
            config.neuron_config, "on_device_sampling_config", None
        )
        if sampling_config is not None and not sampling_config.all_greedy:
            raise ValueError("GLM-5.2 on-device sampling supports greedy only")
        self.on_device_sampling_config = sampling_config
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=not self.on_device_sampling_config,
            dtype=config.torch_dtype,
            device=device,
            tp_group=linear_tp_group,
        )
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=linear_tp_group,
            )
        self.main_layer_indices = self.model.main_layer_indices
        self.excluded_mtp_layer_indices = self.model.excluded_mtp_layer_indices
        self.last_load_report: WeightLoadReport | None = None

    @classmethod
    def from_configs(
        cls,
        hf_config,
        neuron_config,
        tensor_parallel_size: int,
        *,
        expert_parallel_rank: int = 0,
        tp_group: Any = None,
    ) -> GlmMoeDsaForCausalLM:
        config = GlmMoeDsaConfig.from_configs(hf_config, neuron_config)
        return cls(
            config,
            tensor_parallel_size,
            expert_parallel_rank=expert_parallel_rank,
            tp_group=tp_group,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: dict[str, dict[str, Any]] | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        del spec_decode_metadata, kwargs
        hidden_states = self.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
            attn_metadata=attn_metadata,
            rank=rank,
        )
        logits = self.compute_logits(hidden_states, sampling_positions)
        if self.on_device_sampling_config is None and logit_mask is not None:
            if logit_mask.shape != logits.shape:
                raise ValueError("logit_mask must match full gathered logits")
            logits = logits.masked_fill(~logit_mask, float("-inf"))
        if self.on_device_sampling_config is not None:
            sampled_tokens = self.sampler(
                logits,
                sampling_params,
                logit_mask=logit_mask,
                tp_rank=rank,
            )
            return sampled_tokens, None
        return logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sampling_positions is not None:
            hidden_states = torch.index_select(
                hidden_states, 0, sampling_positions.to(torch.int64)
            )
        return self.lm_head(hidden_states)

    def get_kv_spec(self) -> KVSpec:
        return build_glm_mla_cache_spec(dtype=self.config.torch_dtype)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        for layer_idx, layer in enumerate(self.model.layers):
            mla_name = f"model.layers.{layer_idx}.self_attn.mla_cache"
            if mla_name not in kv_caches or len(kv_caches[mla_name]) != 2:
                raise ValueError(f"KV cache for {mla_name} is not initialized")
            layer.mla_k_cache, layer.mla_v_cache = kv_caches[mla_name]
            if layer_idx in MAIN_INDEXER_LAYER_INDICES:
                indexer_name = f"model.layers.{layer_idx}.self_attn.indexer.k_cache"
                if indexer_name not in kv_caches or len(kv_caches[indexer_name]) != 2:
                    raise ValueError(f"KV cache for {indexer_name} is not initialized")
                layer.indexer_k_cache, layer.indexer_v_cache = kv_caches[indexer_name]

    def _checkpoint_key_for_parameter(self, parameter_name: str) -> str:
        router_marker = ".mlp.router.gate."
        if router_marker in parameter_name:
            return parameter_name.replace(router_marker, ".mlp.gate.")

        match = re.match(
            r"^model\.layers\.(\d+)\.mlp\.experts\.experts\.(\d+)\.(.+)$",
            parameter_name,
        )
        if match is not None:
            layer_idx = int(match.group(1))
            local_expert = int(match.group(2))
            global_expert = self.model.layers[layer_idx].mlp.experts.global_expert_ids[
                local_expert
            ]
            return (
                f"model.layers.{layer_idx}.mlp.experts.{global_expert}.{match.group(3)}"
            )
        return parameter_name

    def checkpoint_mappings(self, manifest) -> dict[str, str | list[str]]:
        by_key = manifest.by_key
        mappings: dict[str, str | list[str]] = {}
        for name, _ in self.named_parameters():
            checkpoint_key = self._checkpoint_key_for_parameter(name)
            if checkpoint_key not in by_key:
                raise ValueError(
                    f"No pinned checkpoint tensor for model parameter {name!r}: "
                    f"expected {checkpoint_key!r}"
                )
            entry = by_key[checkpoint_key]
            if entry.info.disposition not in (
                Disposition.LOAD_TARGET,
                Disposition.FP8_SCALE,
            ):
                raise ValueError(f"Parameter maps to non-load target {checkpoint_key}")
            mappings[name] = checkpoint_key
        return mappings

    def _shard_dim(self, checkpoint_key: str, shape: tuple[int, ...]) -> int | None:
        if ".self_attn.indexer." in checkpoint_key:
            return None
        try:
            return tp_shard_spec_for_key(
                checkpoint_key,
                shape,
                world_size=self.tensor_parallel_size,
            ).shard_dim
        except ValueError:
            if checkpoint_key in ("model.embed_tokens.weight", "lm_head.weight"):
                return 0
            return None

    def _fp8_weight_loader(
        self,
        checkpoint_key: str,
        global_shape: tuple[int, int],
        shard_dim: int | None,
    ) -> SafetensorsWeightLoader:
        world_size = self.tensor_parallel_size

        def transform(sources, rank: int) -> torch.Tensor:
            if len(sources) != 1:
                raise ValueError(f"FP8 tensor {checkpoint_key} requires one weight")
            local_rank = self.rank if world_size > 1 else rank
            spec = TPShardSpec(global_shape, shard_dim, world_size)
            weight_slices = spec.slices_for_rank(local_rank)
            weight = sources[0][weight_slices].float()
            return (weight * FP8_STORAGE_SCALE).to(torch.float8_e4m3fn)

        return SafetensorsWeightLoader(transform=transform)

    def _fp8_scale_loader(
        self,
        checkpoint_key: str,
        global_shape: tuple[int, int],
        shard_dim: int | None,
    ) -> SafetensorsWeightLoader:
        world_size = self.tensor_parallel_size

        def transform(sources, rank: int) -> torch.Tensor:
            if len(sources) != 1:
                raise ValueError(f"FP8 scale {checkpoint_key} requires one grid")
            local_rank = self.rank if world_size > 1 else rank
            coverage = PINNED_FP8.scale_coverage_for_weight_shard(
                global_shape,
                shard_dim=shard_dim,
                rank=local_rank,
                world_size=world_size,
            )
            inverse_scale = sources[0][coverage.scale_slices].float()
            return inverse_scale * FP8_INVERSE_SCALE_ADJUSTMENT

        return SafetensorsWeightLoader(transform=transform)

    def _install_weight_loaders(self, manifest, mappings) -> None:
        by_key = manifest.by_key
        for name, parameter in self.named_parameters():
            checkpoint_key = mappings[name]
            entry = by_key[checkpoint_key]
            shape = entry.header.shape
            if entry.info.disposition is Disposition.FP8_SCALE:
                weight_key = checkpoint_key.removesuffix("_scale_inv")
                weight_shape = by_key[weight_key].header.shape
                shard_dim = self._shard_dim(weight_key, weight_shape)
                set_weight_loader(
                    parameter,
                    self._fp8_scale_loader(
                        checkpoint_key,
                        weight_shape,
                        shard_dim,
                    ),
                )
                continue

            shard_dim = self._shard_dim(checkpoint_key, shape)
            if entry.header.dtype == "F8_E4M3":
                if parameter.dtype is not torch.float8_e4m3fn:
                    raise ValueError(
                        f"FP8 checkpoint tensor {checkpoint_key} maps to "
                        f"non-FP8 parameter {parameter.dtype}"
                    )
                set_weight_loader(
                    parameter,
                    self._fp8_weight_loader(checkpoint_key, shape, shard_dim),
                )
            elif shard_dim is not None:
                set_weight_loader(
                    parameter,
                    sharding_weight_loader(
                        shard_dim=shard_dim,
                        shard_size=parameter.shape[shard_dim],
                        num_shards=self.tensor_parallel_size,
                    ),
                )

    def _account_manifest(self, manifest, mappings) -> WeightLoadReport:
        plan = local_load_plan(manifest, ep_rank=self.rank)
        planned_keys = {entry.key for entry in plan}
        mapped_keys = {
            key
            for source in mappings.values()
            for key in (source if isinstance(source, list) else [source])
        }
        if mapped_keys != planned_keys:
            missing = sorted(planned_keys - mapped_keys)[:8]
            extra = sorted(mapped_keys - planned_keys)[:8]
            raise ValueError(
                "Checkpoint mapping does not exactly account for the rank-local plan; "
                f"missing={missing}, extra={extra}"
            )
        counts = Counter(entry.info.disposition for entry in plan)
        skipped_mtp = sum(
            entry.info.disposition is Disposition.INTENTIONAL_SKIP
            for entry in manifest.entries
        )
        return WeightLoadReport(
            indexed_keys=len(manifest.entries),
            local_keys=len(plan),
            load_targets=counts[Disposition.LOAD_TARGET],
            fp8_scales=counts[Disposition.FP8_SCALE],
            skipped_mtp=skipped_mtp,
            mapped_sources=len(mapped_keys),
        )

    def _materialize_router_selection_identities(self, device: torch.device) -> None:
        """Create generated router state omitted from the checkpoint.

        vLLM constructs the model under a meta-device context. The selection
        identities are nonpersistent buffers, so checkpoint loading cannot
        replace those meta tensors before the runner calls ``model.to(device)``.
        """
        for layer in self.model.layers:
            if not layer.is_moe:
                continue
            router = layer.mlp.router
            router.selection_identity = torch.eye(
                router.num_experts,
                dtype=torch.float32,
                device=device,
            )

    def load_weights(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None = None,
    ) -> None:
        index_path = Path(checkpoint_path) / "model.safetensors.index.json"
        manifest = load_checkpoint_manifest(index_path)
        mappings = self.checkpoint_mappings(manifest)
        report = self._account_manifest(manifest, mappings)
        self._install_weight_loaders(manifest, mappings)

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        if dist.is_initialized():
            result = checkpoint.load_sharded_pipelined(
                self.rank,
                self.world_size,
                self,
                mappings,
                device,
                strict=True,
            )
        else:
            result = checkpoint.load_sharded(
                self.rank,
                self.world_size,
                self,
                mappings,
                device,
                strict=True,
            )
        self.load_state_dict(result.state_dict, strict=True, assign=True)
        self._materialize_router_selection_identities(device)
        self.last_load_report = report

    def load_weights_lite(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None = None,
    ) -> None:
        del device, cache_dir
        index_path = Path(checkpoint_path) / "model.safetensors.index.json"
        index = load_checkpoint_index(index_path)
        infos = [classify_checkpoint_key(key) for key in index.key_to_shard]
        local_infos = [
            info
            for info in infos
            if info.disposition is not Disposition.INTENTIONAL_SKIP
            and PINNED_EP.rank_loads(info, self.rank)
        ]
        counts = Counter(info.disposition for info in local_infos)
        self.last_load_report = WeightLoadReport(
            indexed_keys=len(infos),
            local_keys=len(local_infos),
            load_targets=counts[Disposition.LOAD_TARGET],
            fp8_scales=counts[Disposition.FP8_SCALE],
            skipped_mtp=sum(
                info.disposition is Disposition.INTENTIONAL_SKIP for info in infos
            ),
            mapped_sources=0,
        )
