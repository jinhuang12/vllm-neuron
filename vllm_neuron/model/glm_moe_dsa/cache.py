# SPDX-License-Identifier: Apache-2.0
"""Neuron-native cache contracts for GLM-5.2 MLA and DSA.

The main attention cache stores one normalized 512-wide latent vector and one
64-wide rotary key per token.  Indexer layers store one packed FP8 key plus a
four-byte FP32 inverse scale.  Each logical payload is split evenly across the
physical K and V tensors that vLLM allocates for one LayerSpec.
"""

from __future__ import annotations

from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl
import torch
import torch.nn.functional as F

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

MAIN_LAYER_COUNT = 78
MLA_CACHE_HEAD_SIZE = 576
MLA_CACHE_PART_SIZE = MLA_CACHE_HEAD_SIZE // 2
INDEXER_KEY_DIM = 128
INDEXER_CACHE_BYTES = 132
INDEXER_CACHE_PART_BYTES = INDEXER_CACHE_BYTES // 2
MAIN_INDEXER_LAYER_INDICES = (
    0,
    1,
    2,
    6,
    10,
    14,
    18,
    22,
    26,
    30,
    34,
    38,
    42,
    46,
    50,
    54,
    58,
    62,
    66,
    70,
    74,
)


def _kernel_assert(condition: bool, error_text: str) -> None:
    assert condition, (
        "[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: " + error_text
    )


def _div_ceil(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@nki.jit
def _paged_cache_write_nki(cache, values, slot_mapping):
    """Scatter cache rows and skip negative/out-of-range scheduler slots."""

    _kernel_assert(len(cache.shape) == 4, "cache must be rank four")
    _kernel_assert(cache.shape[1] == 1, "cache must have one authoritative head")
    width = cache.shape[-1]
    slot_count = cache.shape[0] * cache.shape[2]
    token_count = slot_mapping.shape[0]
    _kernel_assert(values.shape == (token_count, width), "value shape mismatch")

    flat_cache = cache.reshape((slot_count, width))
    flat_values = values.reshape((token_count, width))
    flat_slots = slot_mapping.reshape((token_count, 1))
    tile_size = 128
    for tile_index in nl.affine_range(_div_ceil(token_count, tile_size)):
        start = tile_index * tile_size
        size = min(tile_size, token_count - start)
        value_tile = nl.ndarray((size, width), dtype=cache.dtype, buffer=nl.sbuf)
        index_tile = nl.ndarray((size, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=value_tile,
            src=flat_values[start : start + size, 0:width],
        )
        nisa.dma_copy(
            dst=index_tile,
            src=flat_slots[start : start + size, 0:1],
        )
        destination = flat_cache.ap(
            pattern=[[width, size], [1, width]],
            vector_offset=index_tile,
            indirect_dim=0,
        )
        nisa.dma_copy(
            dst=destination,
            src=value_tile,
            dge_mode=nisa.dge_mode.swdge,
            oob_mode=nisa.oob_mode.skip,
        )
    return cache


@nki.jit
def _paged_cache_gather_nki(cache, block_table):
    """Gather physical cache pages in each request's logical table order."""

    _kernel_assert(len(cache.shape) == 4, "cache must be rank four")
    _kernel_assert(cache.shape[1] == 1, "cache must have one authoritative head")
    page_width = cache.shape[2] * cache.shape[3]
    entry_count = block_table.shape[0] * block_table.shape[1]
    flat_cache = cache.reshape((cache.shape[0], page_width))
    flat_table = block_table.reshape((entry_count, 1))
    output = nl.ndarray(
        (entry_count, page_width), dtype=cache.dtype, buffer=nl.shared_hbm
    )
    tile_size = 128
    for tile_index in nl.affine_range(_div_ceil(entry_count, tile_size)):
        start = tile_index * tile_size
        size = min(tile_size, entry_count - start)
        index_tile = nl.ndarray((size, 1), dtype=nl.uint32, buffer=nl.sbuf)
        page_tile = nl.ndarray((size, page_width), dtype=cache.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=index_tile,
            src=flat_table[start : start + size, 0:1],
        )
        nisa.memset(dst=page_tile, value=0)
        source = flat_cache.ap(
            pattern=[[page_width, size], [1, page_width]],
            vector_offset=index_tile,
            indirect_dim=0,
        )
        nisa.dma_copy(
            dst=page_tile,
            src=source,
            dge_mode=nisa.dge_mode.swdge,
            oob_mode=nisa.oob_mode.skip,
        )
        nisa.dma_copy(
            dst=output[start : start + size, 0:page_width],
            src=page_tile,
        )
    return output


def write_paged_cache(
    cache: torch.Tensor,
    values: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Write valid scheduler slots; every sentinel is a strict no-op."""

    if block_size != cache.shape[2]:
        raise ValueError("cache block size does not match attention metadata")
    flat_slots = slot_mapping.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1])
    if can_run_kernel(cache):
        updated = wrap_nki(_paged_cache_write_nki)[1](
            cache,
            flat_values,
            flat_slots.to(torch.uint32),
        )
        cache.copy_(updated)
        return

    valid = (flat_slots >= 0) & (flat_slots < cache.shape[0] * block_size)
    slots = flat_slots[valid].to(torch.int64)
    cache.index_put_(
        (slots // block_size, torch.zeros_like(slots), slots % block_size),
        flat_values[valid].to(cache.dtype),
    )


def gather_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """Gather request-local pages and zero invalid physical block IDs."""

    if can_run_kernel(cache):
        valid = (block_table >= 0) & (block_table < cache.shape[0])
        oob_sentinel = torch.full_like(block_table, cache.shape[0])
        safe_table = torch.where(valid, block_table, oob_sentinel)
        pages = wrap_nki(_paged_cache_gather_nki)[1](
            cache,
            safe_table.to(torch.uint32),
        )
        return pages.reshape(
            block_table.shape[0],
            block_table.shape[1] * cache.shape[2],
            cache.shape[3],
        )

    table = block_table.to(torch.int64)
    valid = (table >= 0) & (table < cache.shape[0])
    safe = torch.where(valid, table, torch.zeros_like(table))
    pages = cache[safe, 0]
    pages = torch.where(valid[..., None, None], pages, torch.zeros_like(pages))
    return pages.flatten(1, 2)


def write_paged_cache_pair(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    values: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Split one logical payload evenly across paired paged-cache tensors."""

    if k_cache.shape != v_cache.shape:
        raise ValueError("paired cache tensors must have identical shapes")
    part_size = k_cache.shape[-1]
    if values.shape[-1] != part_size * 2:
        raise ValueError(
            f"logical cache width must be twice physical width {part_size}"
        )
    write_paged_cache(
        k_cache,
        values[..., :part_size].contiguous(),
        slot_mapping,
        block_size,
    )
    write_paged_cache(
        v_cache,
        values[..., part_size:].contiguous(),
        slot_mapping,
        block_size,
    )


def gather_paged_cache_pair(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """Reassemble one logical payload from paired paged-cache tensors."""

    if k_cache.shape != v_cache.shape:
        raise ValueError("paired cache tensors must have identical shapes")
    return torch.cat(
        (
            gather_paged_cache(k_cache, block_table),
            gather_paged_cache(v_cache, block_table),
        ),
        dim=-1,
    )


def build_glm_mla_cache_spec(
    *,
    dtype: torch.dtype = torch.bfloat16,
    prefix: str = "model.layers",
) -> KVSpec:
    """Return independent vLLM-Neuron LayerSpec entries for both caches.

    Layer 78 belongs to MTP and is deliberately absent from this main-model
    cache contract.
    """

    layers = [
        LayerSpec(
            name=f"{prefix}.{layer}.self_attn.mla_cache",
            num_kv_heads=1,
            head_size=MLA_CACHE_PART_SIZE,
            dtype=dtype,
        )
        for layer in range(MAIN_LAYER_COUNT)
    ]
    layers.extend(
        LayerSpec(
            name=f"{prefix}.{layer}.self_attn.indexer.k_cache",
            num_kv_heads=1,
            head_size=INDEXER_CACHE_PART_BYTES,
            dtype=torch.uint8,
        )
        for layer in MAIN_INDEXER_LAYER_INDICES
    )
    return KVSpec(layers=layers)


@dataclass
class DualCacheState:
    """Mutable cache storage for one attention layer.

    ``slots`` always uses ``[request_index, sequence_position]`` pairs.  This
    avoids an ambient global slot mapping and makes request isolation explicit.
    All tensor operations are ordinary PyTorch operations that lower through
    torch/XLA on Neuron.
    """

    mla: torch.Tensor
    indexer: torch.Tensor | None
    lengths: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        max_batch: int,
        max_sequence_length: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        with_indexer: bool = True,
    ) -> DualCacheState:
        if max_batch <= 0 or max_sequence_length <= 0:
            raise ValueError("cache dimensions must be positive")
        mla = torch.zeros(
            max_batch,
            max_sequence_length,
            MLA_CACHE_HEAD_SIZE,
            dtype=dtype,
            device=device,
        )
        indexer = None
        if with_indexer:
            indexer = torch.zeros(
                max_batch,
                max_sequence_length,
                INDEXER_CACHE_BYTES,
                dtype=torch.uint8,
                device=device,
            )
        lengths = torch.zeros(max_batch, dtype=torch.int64, device=device)
        return cls(mla=mla, indexer=indexer, lengths=lengths)

    @property
    def max_batch(self) -> int:
        return int(self.mla.shape[0])

    @property
    def max_sequence_length(self) -> int:
        return int(self.mla.shape[1])

    def write(
        self,
        slots: torch.Tensor,
        mla_values: torch.Tensor,
        indexer_values: torch.Tensor | None = None,
    ) -> None:
        """Write exact cache slots and update per-request used lengths."""

        if slots.ndim != 2 or slots.shape[-1] != 2:
            raise ValueError("slots must have shape [tokens, 2]")
        if mla_values.shape != (slots.shape[0], MLA_CACHE_HEAD_SIZE):
            raise ValueError(
                f"MLA values must have shape [{slots.shape[0]}, {MLA_CACHE_HEAD_SIZE}]"
            )
        if indexer_values is not None:
            if self.indexer is None:
                raise ValueError("indexer values supplied to an MLA-only cache")
            if indexer_values.shape != (slots.shape[0], INDEXER_CACHE_BYTES):
                raise ValueError(
                    "indexer values must have shape "
                    f"[{slots.shape[0]}, {INDEXER_CACHE_BYTES}]"
                )

        request = slots[:, 0].to(torch.int64)
        position = slots[:, 1].to(torch.int64)
        # The scheduler validates dynamic slot values before compiled execution.
        # Keep the same fail-fast checks in eager tests without introducing a
        # tensor-to-Python synchronization into the torch/XLA graph.
        if not torch.compiler.is_compiling():
            if bool(torch.any(request < 0)) or bool(
                torch.any(request >= self.max_batch)
            ):
                raise IndexError("request index is outside the cache")
            if bool(torch.any(position < 0)) or bool(
                torch.any(position >= self.max_sequence_length)
            ):
                raise IndexError("sequence position is outside the cache")

            linear = request * self.max_sequence_length + position
            if torch.unique(linear).numel() != linear.numel():
                raise ValueError("duplicate cache slots in one update")

        self.mla[request, position] = mla_values.to(self.mla.dtype)
        if indexer_values is not None:
            assert self.indexer is not None
            self.indexer[request, position] = indexer_values

        per_request = F.one_hot(request, num_classes=self.max_batch).to(torch.int64)
        used = per_request * (position + 1).unsqueeze(-1)
        self.lengths.copy_(torch.maximum(self.lengths, used.amax(dim=0)))

    def read_mla(
        self, request_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read request-local MLA cache rows and their valid lengths."""

        request_indices = request_indices.to(torch.int64)
        return self.mla[request_indices], self.lengths[request_indices]

    def read_indexer(
        self, request_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read request-local packed indexer rows and valid lengths."""

        if self.indexer is None:
            raise ValueError("this layer has no indexer cache")
        request_indices = request_indices.to(torch.int64)
        return self.indexer[request_indices], self.lengths[request_indices]
