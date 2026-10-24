# SPDX-License-Identifier: Apache-2.0
"""Image rows from encoder cache blocks written into the prefill stream in place.

The kernel gathers each block's rows from the cache in tiles of 128 and
scatters them to their stream rows by position, dropping positions outside
the stream. It writes into the stream it receives and returns that stream,
so no copy of the stream is made. The cache rows hold the hidden width only:
a model with deepstack levels uses ``merge_multimodal_embeddings``.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.functional.merge_multimodal_embeddings import (
    global_to_local_positions,
    scatter_with_dummy_row,
)
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel


def merge_image_rows(
    hidden_states: torch.Tensor,
    encoder_embedding_buffer: torch.Tensor,
    encoder_block_ids: torch.Tensor,
    encoder_positions: torch.Tensor,
    rank: torch.Tensor,
) -> torch.Tensor:
    """Write the cache blocks' rows into ``hidden_states`` at their positions; return it.

    Args:
        hidden_states: ``[local_len, hidden]`` bf16 stream, written in place.
        encoder_embedding_buffer: ``[num_blocks, block_size, hidden]`` cache blocks.
        encoder_block_ids: ``[slots]`` int32 block ids, the scratch block in unused slots.
        encoder_positions: ``[slots, block_size]`` int32 global positions; a position
            outside this rank's rows writes nothing.
        rank: ``[1]`` int32 sequence-shard rank; the stream holds rows
            ``rank * local_len`` to ``(rank + 1) * local_len``.

    Returns:
        ``hidden_states`` itself, its image rows written.
    """
    if not _can_use_kernel(hidden_states, encoder_embedding_buffer):
        return _merge_image_rows_torch(
            hidden_states, encoder_embedding_buffer, encoder_block_ids, encoder_positions, rank
        )
    if not isinstance(rank, torch.Tensor):
        raise TypeError("The NKI image-row merge requires a runtime rank tensor.")
    local_offset = rank.reshape(1).to(torch.int32) * hidden_states.shape[0]
    wrap_nki(_merge_image_rows_kernel)[get_logical_nc_config()](
        stream=hidden_states,
        cache=encoder_embedding_buffer,
        block_ids=encoder_block_ids,
        positions=encoder_positions,
        local_offset=local_offset,
    )
    return hidden_states


def _can_use_kernel(hidden_states: torch.Tensor, encoder_embedding_buffer: torch.Tensor) -> bool:
    """True on a kernel venue for a bf16 stream and cache rows of the stream's width."""
    if not can_run_kernel(hidden_states):
        return False
    if hidden_states.dtype != torch.bfloat16 or encoder_embedding_buffer.dtype != torch.bfloat16:
        return False
    return encoder_embedding_buffer.shape[-1] == hidden_states.shape[-1]


def _merge_image_rows_torch(hidden_states, encoder_embedding_buffer, encoder_block_ids,
                            encoder_positions, rank):
    """Reference for the CPU venue: the same rows written into ``hidden_states``."""
    rows = encoder_embedding_buffer.index_select(0, encoder_block_ids.long())
    rows = rows.reshape(-1, rows.shape[-1])
    local = global_to_local_positions(
        encoder_positions.reshape(-1).long(), hidden_states.shape[0], rank)
    return hidden_states.copy_(scatter_with_dummy_row(hidden_states, local, rows))


@nki.jit
def _merge_image_rows_kernel(stream, cache, block_ids, positions, local_offset):
    """Each core gathers and scatters its share of the 128-row tiles; returns ``stream``."""
    num_blocks, block_size, width = cache.shape
    local_len, hidden = stream.shape
    slots = block_ids.shape[0]
    P = nl.tile_size.pmax
    kernel_assert(width == hidden, f"Expected cache rows of width {hidden}, got {width}")
    kernel_assert(tuple(positions.shape) == (slots, block_size),
                  f"Expected positions [{slots}, {block_size}], got {tuple(positions.shape)}")

    tiles_per_block = div_ceil(block_size, P)
    n_tiles = slots * tiles_per_block
    n_prog = nl.num_programs(axes=0)
    prog = nl.program_id(axis=0)
    per_core = div_ceil(n_tiles, n_prog)
    first = prog * per_core
    last = min(first + per_core, n_tiles)
    if first >= last:
        return stream

    cache_rows = cache.reshape((num_blocks * block_size, width))
    ids = nl.ndarray((1, slots), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ids[0, :], src=block_ids.reshape((1, slots)))
    offset_one = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=offset_one[0, :], src=local_offset)
    offset = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    stream_shuffle_broadcast(src=offset_one, dst=offset)

    base = None
    for tile in range(first, last):
        b = tile // tiles_per_block
        row0 = (tile % tiles_per_block) * P
        rows = min(P, block_size - row0)
        if base is None or row0 == 0:
            block = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
            stream_shuffle_broadcast(src=ids[0:1, b:b + 1], dst=block)
            base = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=base, data=block, op0=nl.multiply, operand0=block_size)

        # Cache row of each partition: block_ids[b] * block_size + row0 + p.
        lane = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=lane, pattern=[[0, 1]], offset=row0, channel_multiplier=1)
        src = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=src, data1=base[0:rows, :], data2=lane, op=nl.add)
        tile_rows = nl.ndarray((rows, width), dtype=cache.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=tile_rows,
            src=cache_rows.ap(pattern=[[width, rows], [1, width]], offset=0,
                              vector_offset=src, indirect_dim=0),
            name=f"merge_image_rows_gather_{tile}",
        )

        # Stream row of each partition; a row outside [0, local_len) is skipped.
        dst = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=dst, src=positions[b, row0:row0 + rows].reshape((1, rows)).permute((1, 0)))
        nisa.tensor_tensor(dst=dst, data1=dst, data2=offset[0:rows, :], op=nl.subtract)
        nisa.dma_copy(
            dst=stream.ap(pattern=[[hidden, rows], [1, hidden]], offset=0,
                          vector_offset=dst, indirect_dim=0),
            src=tile_rows,
            oob_mode=oob_mode.skip,
            name=f"merge_image_rows_scatter_{tile}",
        )
    return stream
