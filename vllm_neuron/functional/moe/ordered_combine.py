# SPDX-License-Identifier: Apache-2.0
"""Private FP32 combine for the captured blockwise MoE mapping.

Inputs are the fused expert output [blocks*rows,H] and its original int32
[blocks,rows] map. Valid IDs are unique within a block and lie in [0,tokens).
Negative IDs are padding. The caller proves this value contract; this helper
does not accept arbitrary scatter maps or promise positive-OOB behavior.
The result is FP32 [tokens,H]. The model owns its existing final dtype cast.
"""

import torch

# Trainium2 hardware partition extent, checked against the pinned SDK below.
_PARTITION_MAX = 128


def _row_waves(rows, limit):
    """Split a static row extent without a one-partition final wave."""
    waves = []
    start = 0
    while start < rows:
        size = min(limit, rows - start)
        if rows - start == limit + 1:
            size -= 1
        waves.append((start, size))
        start += size
    return tuple(waves)


def _supported_geometry(contribution, row_ids, tokens, BLOCK_H=512):
    # Two FP32 working tiles use at most 32 KiB per partition, plus indices.
    # This bound is independent of the model's H and keeps the first schedule
    # within the pinned target's SBUF budget. Wider tuning tiles retain Torch.
    return (tokens > 1 and row_ids.shape[1] > 1
            and contribution.shape[1] >= 2 and contribution.shape[1] % 2 == 0
            and row_ids.shape[1] % _PARTITION_MAX == 0
            and contribution.shape[0] % (2 * _PARTITION_MAX) == 0
            and contribution.shape[1] % (2 * BLOCK_H) == 0
            and BLOCK_H <= 4096 and contribution.is_contiguous() and row_ids.is_contiguous())


def _torch_combine(contribution, row_ids, tokens):
    flat_ids = row_ids.reshape(-1)
    wanted = torch.where(flat_ids < 0, torch.full_like(flat_ids, tokens), flat_ids).long()
    return torch.zeros(tokens + 1, contribution.shape[1], dtype=torch.float32,
                       device=contribution.device).index_add(0, wanted, contribution)[:tokens]


def _ordered_combine_kernel(contribution, row_ids, tokens, BLOCK_H=512):
    """Add block contributions in order; two cores own disjoint hidden columns."""
    blocks, rows = row_ids.shape
    hidden = contribution.shape[1]
    partitions = nl.tile_size.pmax
    programs = nl.num_programs(axes=0)
    program = nl.program_id(axis=0)
    kernel_assert(programs == 2, "ordered combine requires LNC2")
    kernel_assert(hidden % programs == 0, "hidden columns must divide across cores")
    kernel_assert(tokens > 1 and rows > 1, "single-row geometry keeps the Torch route")
    local_hidden = hidden // programs
    kernel_assert(partitions == _PARTITION_MAX, "unsupported hardware partition extent")
    kernel_assert(rows % partitions == 0, "partial mapping rows keep the Torch route")
    kernel_assert(blocks * rows % (2 * partitions) == 0, "partial flattened halves keep the Torch route")
    kernel_assert(local_hidden % BLOCK_H == 0, "partial hidden tiles keep the Torch route")
    output = nl.ndarray((tokens, hidden), dtype=nl.float32, buffer=nl.shared_hbm)
    flat_ids = row_ids.reshape((blocks * rows, 1))
    waves = _row_waves(rows, partitions)

    for hidden_tile in range(div_ceil(local_hidden, BLOCK_H)):
        width = min(BLOCK_H, local_hidden - hidden_tile * BLOCK_H)
        column = program * local_hidden + hidden_tile * BLOCK_H
        zeros = nl.ndarray((partitions, width), dtype=nl.float32, buffer=nl.sbuf)
        values = nl.ndarray((partitions, width), dtype=nl.float32, buffer=nl.sbuf)
        indices = nl.ndarray((partitions, 16), dtype=nl.int32, buffer=nl.sbuf)
        # The pinned parser emits memset but does not expose its instruction
        # handle. Its SBUF write is a data dependency of this first DMA read.
        # Start the explicit completion chain with that DMA's valid handle.
        nisa.memset(dst=zeros, value=0.0)
        initial_height = min(partitions, tokens)
        previous = nisa.dma_copy(
            dst=output.ap(pattern=[[hidden, initial_height], [1, width]], offset=column),
            src=zeros[:initial_height, :width],
        )
        for token_tile in range(1, div_ceil(tokens, partitions)):
            start = token_tile * partitions
            height = min(partitions, tokens - start)
            current = nisa.dma_copy(
                dst=output.ap(pattern=[[hidden, height], [1, width]],
                              offset=start * hidden + column),
                src=zeros[:height, :width],
            )
            current.depends_on(previous)
            previous = current

        for block in range(blocks):
            for wave_index in range(len(waves)):
                wave = waves[wave_index]
                start, height = wave[0], wave[1]
                flat_start = block * rows + start
                index_load = nisa.dma_copy(
                    dst=indices[:height, :1],
                    src=flat_ids.ap(pattern=[[1, height], [1, 1]], offset=flat_start),
                )
                value_load = nisa.dma_copy(
                    dst=values[:height, :width],
                    src=contribution.ap(pattern=[[hidden, height], [1, width]],
                                        offset=flat_start * hidden + column),
                )
                # Both buffers are reused. Complete the preceding RMW before
                # either load can overwrite its input or its indirect indices.
                index_load.depends_on(previous)
                value_load.depends_on(previous)
                destination = output.ap(
                    pattern=[[hidden, height], [1, width]], offset=column,
                    vector_offset=indices[:height, :1], indirect_dim=0,
                )
                # Gather RMW keeps the FP32 accumulation in the values tile.
                # The dependent scatter completes before the next wave can
                # reuse either the values tile or its indirect indices.
                current = nisa.dma_compute(
                    dst=values[:height, :width],
                    srcs=[destination, values[:height, :width]],
                    reduce_op=nl.add, unique_indices=True, oob_mode=nisa.oob_mode.skip,
                )
                current.depends_on(previous)
                current.depends_on(index_load)
                current.depends_on(value_load)
                store = nisa.dma_copy(
                    dst=destination, src=values[:height, :width],
                    oob_mode=nisa.oob_mode.skip,
                )
                store.depends_on(current)
                previous = store
    return output


# CPU reference tests can import this private module without the Neuron SDK.
# A broken installed SDK still raises: only an absent top-level nki is optional.
try:
    import nki
except ModuleNotFoundError as error:
    if error.name != "nki":
        raise
    _NATIVE = None
else:
    import nki.isa as nisa
    import nki.language as nl
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.core.utils.kernel_assert import kernel_assert
    from nkilib.core.utils.kernel_helpers import div_ceil
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    _NATIVE = wrap_nki(nki.jit(_ordered_combine_kernel))[2]


def ordered_combine(contribution, row_ids, tokens, *, BLOCK_H=512):
    """Combine a proven captured mapping, preserving the pre-cast FP32 result.

    CPU, decode, partial mapping rows, flattened halves or hidden tiles, strided
    operands and unsupported tile sizes keep the original Torch expression. Value checks
    stay with the mapping producer; no tensor values are read on the host.
    """
    if type(tokens) is not int or tokens < 1:
        raise ValueError("tokens must be a positive static integer")
    if type(BLOCK_H) is not int or BLOCK_H < 1:
        raise ValueError("BLOCK_H must be a positive static integer")
    if contribution.ndim != 2 or contribution.dtype != torch.float32 or contribution.shape[1] < 1:
        raise ValueError("contribution must be FP32 [blocks*rows,H] with positive H")
    if row_ids.ndim != 2 or row_ids.dtype != torch.int32 or min(row_ids.shape) < 1:
        raise ValueError("row_ids must be int32 [blocks,rows] with positive extents")
    if contribution.shape[0] != row_ids.numel() or contribution.device != row_ids.device:
        raise ValueError("contributions and row IDs must have matching rows and device")
    if (_NATIVE is not None and contribution.device.type not in ("cpu", "meta")
            and _supported_geometry(contribution, row_ids, tokens, BLOCK_H)
            and can_run_kernel(contribution)):
        return _NATIVE(contribution=contribution, row_ids=row_ids,
                       tokens=tokens, BLOCK_H=BLOCK_H)
    return _torch_combine(contribution, row_ids, tokens)
