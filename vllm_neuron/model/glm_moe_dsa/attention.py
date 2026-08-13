# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 sparse MLA built from Neuron-native PyTorch operations."""

from __future__ import annotations

import os
from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl
import torch
import torch.nn.functional as F
from torch import nn

from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki
from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

from .block_fp8 import (
    BlockFP8ColumnParallelLinear,
    BlockFP8Linear,
    BlockFP8RowParallelLinear,
)
from .cache import MLA_CACHE_HEAD_SIZE
from .indexer import apply_interleaved_rope, rotary_cos_sin
from .sparse_mla import (
    selected_latent_mla_decode,
    validate_selected_latent_mla_decode_contract,
)


SELECTED_LATENT_MLA_ENV = "GLM_ENABLE_EXPERIMENTAL_SELECTED_LATENT_MLA"
SELECTED_LATENT_MLA_CONTEXT_BUCKETS = (4096, 8192)


class GlmMoeDsaRMSNorm(nn.Module):
    def __init__(
        self,
        width: int,
        eps: float = 1.0e-5,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        variance = values.float().square().mean(dim=-1, keepdim=True)
        normalized = values.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(values.dtype)


@dataclass(frozen=True)
class MlaProjection:
    q_lora: torch.Tensor
    queries: torch.Tensor
    latent_cache: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor


def _kernel_assert(condition: bool, error_text: str) -> None:
    assert condition, (
        "[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: " + error_text
    )


@nki.jit
def _cached_kv_projection_nki(cache2d, weight):
    """Project the 512 latent columns of a contiguous 576-wide MLA cache."""

    _kernel_assert(len(cache2d.shape) == 2, "cache must be rank 2")
    _kernel_assert(cache2d.shape[1] == MLA_CACHE_HEAD_SIZE, "cache width must be 576")
    _kernel_assert(len(weight.shape) == 2, "weight must be rank 2")
    _kernel_assert(weight.shape[1] == 512, "projection K must be 512")

    row_count = cache2d.shape[0]
    output_width = weight.shape[0]
    cache_is_fp8 = cache2d.dtype == nl.float8_e4m3
    projection_dtype = weight.dtype if cache_is_fp8 else cache2d.dtype
    output = nl.ndarray(
        (row_count, output_width), dtype=projection_dtype, buffer=nl.shared_hbm
    )

    for row_tile_index in nl.affine_range((row_count + 127) // 128):
        row_start = row_tile_index * 128
        row_size = min(128, row_count - row_start)
        for output_tile_index in nl.affine_range((output_width + 127) // 128):
            output_start = output_tile_index * 128
            output_size = min(128, output_width - output_start)
            result_psum = nl.ndarray(
                (row_size, output_size), dtype=nl.float32, buffer=nl.psum
            )

            for k_tile_index in nl.affine_range(4):
                k_start = k_tile_index * 128

                cache_tile = cache2d[
                    row_start : row_start + row_size,
                    k_start : k_start + 128,
                ]
                if cache_is_fp8:
                    cache_fp8 = nl.ndarray(
                        (row_size, 128), dtype=cache2d.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_copy(dst=cache_fp8, src=cache_tile)
                    cache_bf16 = nl.ndarray(
                        (row_size, 128), dtype=weight.dtype, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=cache_bf16, src=cache_fp8)
                    cache_transpose = nl.ndarray(
                        (128, row_size), dtype=weight.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_transpose(
                        dst=cache_transpose,
                        src=cache_bf16,
                        axes=(1, 0),
                    )
                else:
                    cache_transpose = nl.ndarray(
                        (128, row_size), dtype=cache2d.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_transpose(
                        dst=cache_transpose,
                        src=cache_tile,
                        axes=(1, 0),
                    )

                weight_transpose = nl.ndarray(
                    (128, output_size), dtype=weight.dtype, buffer=nl.sbuf
                )
                nisa.dma_transpose(
                    dst=weight_transpose,
                    src=weight[
                        output_start : output_start + output_size,
                        k_start : k_start + 128,
                    ],
                    axes=(1, 0),
                )

                nisa.nc_matmul(
                    dst=result_psum,
                    stationary=cache_transpose,
                    moving=weight_transpose,
                    accumulate=(k_tile_index > 0),
                )

            result = nl.ndarray(
                (row_size, output_size), dtype=projection_dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=result, src=result_psum)
            nisa.dma_copy(
                dst=output[
                    row_start : row_start + row_size,
                    output_start : output_start + output_size,
                ],
                src=result,
            )
    return output


def _sparse_score_matmul(
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Tile non-contracting query and key axes to at most 256 entries."""

    query_rows = queries.float().permute(0, 2, 1, 3)
    key_columns = keys.float().permute(0, 2, 3, 1)
    query_count = query_rows.shape[-2]
    key_count = key_columns.shape[-1]
    tile_size = 256
    if query_count <= tile_size and key_count <= tile_size:
        return torch.matmul(query_rows, key_columns)

    score_rows = []
    for query_start in range(0, query_count, tile_size):
        query_tile = query_rows[..., query_start : query_start + tile_size, :]
        score_columns = []
        for key_start in range(0, key_count, tile_size):
            key_tile = key_columns[..., :, key_start : key_start + tile_size]
            score_columns.append(torch.matmul(query_tile, key_tile))
        score_rows.append(torch.cat(score_columns, dim=-1))
    return torch.cat(score_rows, dim=-2)


def sparse_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    selected_indices: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Attend only to selected cache positions.

    The implementation uses traceable gather, matmul, mask, and softmax ops.
    These are native torch/XLA operations on Neuron.  ``-1`` selection entries
    are padding and contribute exactly zero.
    """

    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("Q, K, and V must use [batch, sequence, heads, dim]")
    if queries.shape[:1] != keys.shape[:1] or keys.shape[:3] != values.shape[:3]:
        raise ValueError("Q, K, and V batch/head shapes do not match")
    if selected_indices.shape[:2] != queries.shape[:2]:
        raise ValueError("selected indices must match query batch and sequence")
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("query and key head dimensions must match")

    batch, query_count, _head_count, head_dim = queries.shape
    if scale is None:
        scale = head_dim**-0.5
    scores = _sparse_score_matmul(queries, keys) * scale
    key_count = keys.shape[1]
    valid = (selected_indices >= 0) & (selected_indices < key_count)
    selected = selected_indices.clamp(0, key_count - 1)
    selected_mask = torch.zeros(
        batch,
        query_count,
        keys.shape[1],
        dtype=torch.int16,
        device=queries.device,
    ).scatter_add(2, selected, valid.to(torch.int16))
    valid_scores = (selected_mask > 0).unsqueeze(1)
    score_floor = torch.full(
        (),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
        device=scores.device,
    )
    scores = torch.where(valid_scores, scores, score_floor)
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(valid_scores, weights, torch.zeros_like(weights))
    normalizer = weights.sum(dim=-1, keepdim=True)
    safe_normalizer = torch.where(
        normalizer > 0,
        normalizer,
        torch.ones_like(normalizer),
    )
    weights = weights / safe_normalizer
    output = torch.matmul(
        weights,
        values.float().permute(0, 2, 1, 3),
    ).permute(0, 2, 1, 3)
    return output.to(queries.dtype)


class GlmMoeDsaAttention(nn.Module):
    """q-LoRA/kv-LoRA projections and sparse MLA for TP-local heads."""

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        local_heads: int,
        num_heads: int | None = None,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rms_norm_eps: float = 1.0e-5,
        rope_theta: float = 8_000_000.0,
        tp_group=None,
        fp8_weights: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if kv_lora_rank + qk_rope_head_dim != MLA_CACHE_HEAD_SIZE:
            raise ValueError(
                "GLM-5.2 MLA cache must contain 512 latent and 64 rotary values"
            )
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.local_heads = local_heads
        self.num_heads = num_heads if num_heads is not None else local_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.enable_selected_latent_mla = os.environ.get(SELECTED_LATENT_MLA_ENV) == "1"

        if fp8_weights:
            tp_size = self.num_heads // local_heads
            self.q_a_proj = BlockFP8Linear(hidden_size, q_lora_rank, device=device)
            self.q_b_proj = BlockFP8ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                tp_group=tp_group,
                tp_size_override=tp_size,
                device=device,
            )
            self.kv_a_proj_with_mqa = BlockFP8Linear(
                hidden_size, kv_lora_rank + qk_rope_head_dim, device=device
            )
            self.kv_b_proj = BlockFP8ColumnParallelLinear(
                kv_lora_rank,
                self.num_heads * (qk_nope_head_dim + v_head_dim),
                tp_group=tp_group,
                tp_size_override=tp_size,
                device=device,
            )
            self.o_proj = BlockFP8RowParallelLinear(
                self.num_heads * v_head_dim,
                hidden_size,
                input_is_parallel=True,
                tp_group=tp_group,
                tp_size_override=tp_size,
                device=device,
            )
        else:
            self.q_a_proj = nn.Linear(
                hidden_size, q_lora_rank, bias=False, dtype=dtype, device=device
            )
            set_weight_loader(self.q_a_proj.weight, SafetensorsWeightLoader())
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                tp_group=tp_group,
                dtype=dtype,
                device=device,
            )
            self.kv_a_proj_with_mqa = nn.Linear(
                hidden_size,
                kv_lora_rank + qk_rope_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
            )
            set_weight_loader(self.kv_a_proj_with_mqa.weight, SafetensorsWeightLoader())
            self.kv_b_proj = ColumnParallelLinear(
                kv_lora_rank,
                self.num_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
                tp_group=tp_group,
                dtype=dtype,
                device=device,
            )
            self.o_proj = RowParallelLinear(
                self.num_heads * v_head_dim,
                hidden_size,
                bias=False,
                input_is_parallel=True,
                tp_group=tp_group,
                dtype=dtype,
                device=device,
            )
        self.q_a_layernorm = GlmMoeDsaRMSNorm(
            q_lora_rank, eps=rms_norm_eps, dtype=dtype, device=device
        )
        self.kv_a_layernorm = GlmMoeDsaRMSNorm(
            kv_lora_rank, eps=rms_norm_eps, dtype=dtype, device=device
        )

        if self.q_b_proj.out_features_per_rank != local_heads * self.qk_head_dim:
            raise ValueError("local_heads does not match q_b TP shard width")
        if self.kv_b_proj.out_features_per_rank != local_heads * (
            qk_nope_head_dim + v_head_dim
        ):
            raise ValueError("local_heads does not match kv_b TP shard width")
        if self.o_proj.in_features_per_rank != local_heads * v_head_dim:
            raise ValueError("local_heads does not match o_proj TP shard width")

    def project(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> MlaProjection:
        q_lora = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_lora).view(
            *hidden_states.shape[:-1], self.local_heads, self.qk_head_dim
        )
        q_nope = q[..., : self.qk_nope_head_dim].contiguous()
        q_pe = q[..., self.qk_nope_head_dim :].contiguous()

        latent_raw = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent = latent_raw[..., : self.kv_lora_rank].contiguous()
        k_pe = latent_raw[..., self.kv_lora_rank :].contiguous()
        kv_latent = self.kv_a_layernorm(kv_latent)
        kv = self.kv_b_proj(kv_latent).view(
            *hidden_states.shape[:-1],
            self.local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, values = kv.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)

        cos, sin = rotary_cos_sin(
            positions,
            self.qk_rope_head_dim,
            theta=self.rope_theta,
            dtype=q.dtype,
        )
        q_pe = apply_interleaved_rope(q_pe, cos, sin)
        k_pe = apply_interleaved_rope(k_pe, cos, sin)
        k_pe_heads = k_pe.unsqueeze(-2).expand(
            *k_pe.shape[:-1], self.local_heads, self.qk_rope_head_dim
        )
        queries = torch.cat((q_nope, q_pe), dim=-1)
        keys = torch.cat((k_nope, k_pe_heads), dim=-1)
        latent_cache = torch.cat((kv_latent, k_pe), dim=-1)
        return MlaProjection(q_lora, queries, latent_cache, keys, values)

    def expand_cached_latents(
        self, latent_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct TP-local K/V from the 576-wide MLA cache."""

        if latent_cache.shape[-1] != MLA_CACHE_HEAD_SIZE:
            raise ValueError(f"latent cache width must be {MLA_CACHE_HEAD_SIZE}")
        kv_latent = latent_cache[..., : self.kv_lora_rank]
        k_pe = latent_cache[..., self.kv_lora_rank :]
        if isinstance(self.kv_b_proj, BlockFP8ColumnParallelLinear):
            kv = self.kv_b_proj(kv_latent.to(torch.bfloat16)).view(
                *latent_cache.shape[:-1],
                self.local_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
        elif can_run_kernel(latent_cache):
            cache2d = latent_cache.contiguous().view(-1, MLA_CACHE_HEAD_SIZE)
            kv2d = wrap_nki(_cached_kv_projection_nki)[1](
                cache2d, self.kv_b_proj.weight
            )
            if self.local_heads != 1:
                raise ValueError("Neuron cached MLA expansion requires one local head")
            kv = kv2d.view(
                *latent_cache.shape[:-1],
                self.local_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
        else:
            kv = F.linear(
                kv_latent.to(self.kv_b_proj.weight.dtype),
                self.kv_b_proj.weight,
                bias=None,
            ).view(
                *latent_cache.shape[:-1],
                self.local_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
        k_nope = kv[..., : self.qk_nope_head_dim]
        values = kv[..., self.qk_nope_head_dim :]
        k_pe = (
            k_pe.to(k_nope.dtype)
            .unsqueeze(-2)
            .expand(*k_pe.shape[:-1], self.local_heads, self.qk_rope_head_dim)
        )
        return torch.cat((k_nope, k_pe), dim=-1), values

    def attend(
        self,
        queries: torch.Tensor,
        latent_cache: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        keys, values = self.expand_cached_latents(latent_cache)
        output = sparse_attention(
            queries,
            keys,
            values,
            selected_indices,
            scale=self.qk_head_dim**-0.5,
        )
        return self.o_proj(output.flatten(-2))

    def attend_selected_latents(
        self,
        queries: torch.Tensor,
        selected_indices: torch.Tensor,
        mla_k_cache: torch.Tensor,
        mla_v_cache: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Attend directly from selected physical cache rows."""

        output = selected_latent_mla_decode(
            queries,
            mla_k_cache,
            mla_v_cache,
            block_table,
            selected_indices.to(torch.int32),
            self.kv_b_proj.weight,
            self.kv_b_proj.weight_scale_inv,
            block_size=block_size,
            row_offset=self.kv_b_proj.row_offset,
        )
        return self.o_proj(output.flatten(-2))

    def should_use_selected_latent_mla(
        self,
        queries: torch.Tensor,
        selected_indices: torch.Tensor,
        *,
        mla_k_cache: torch.Tensor | None,
        mla_v_cache: torch.Tensor | None,
        block_table: torch.Tensor,
        block_size: int,
        is_decode: bool,
    ) -> bool:
        """Select an evidenced decode bucket, or preserve the fallback."""

        if not self.enable_selected_latent_mla:
            return False
        if (
            not is_decode
            or queries.ndim != 4
            or queries.shape[1] != 1
            or not can_run_kernel(queries)
        ):
            return False
        logical_key_count = block_table.shape[1] * block_size
        if logical_key_count <= 2048:
            return False
        if logical_key_count not in SELECTED_LATENT_MLA_CONTEXT_BUCKETS:
            raise ValueError(
                "selected-latent MLA production contract violation: "
                f"unsupported context bucket {logical_key_count}; expected one of "
                f"{SELECTED_LATENT_MLA_CONTEXT_BUCKETS}"
            )

        errors: list[str] = []
        if mla_k_cache is None or mla_v_cache is None:
            errors.append("paired physical MLA caches must be allocated")
        if (
            self.num_heads != 64
            or self.local_heads != 1
            or self.kv_lora_rank != 512
            or self.qk_nope_head_dim != 192
            or self.qk_rope_head_dim != 64
            or self.qk_head_dim != 256
            or self.v_head_dim != 256
        ):
            errors.append("attention dimensions must match pinned GLM TP64")
        if not isinstance(self.kv_b_proj, BlockFP8ColumnParallelLinear):
            errors.append("kv_b projection must use block FP8")
        else:
            if self.kv_b_proj.tp_size != 64:
                errors.append("kv_b projection must use TP64")
            if self.kv_b_proj.col_offset != 0:
                errors.append("kv_b column offset must be zero")
        if errors:
            raise ValueError(
                "selected-latent MLA production contract violation: "
                + "; ".join(errors)
            )

        assert mla_k_cache is not None
        assert mla_v_cache is not None

        selected_for_kernel = selected_indices.to(torch.int32)
        validate_selected_latent_mla_decode_contract(
            queries,
            mla_k_cache,
            mla_v_cache,
            block_table,
            selected_for_kernel,
            self.kv_b_proj.weight,
            self.kv_b_proj.weight_scale_inv,
            block_size=block_size,
            row_offset=self.kv_b_proj.row_offset,
        )
        return True
