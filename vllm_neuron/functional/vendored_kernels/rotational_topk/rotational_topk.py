# SPDX-License-Identifier: Apache-2.0
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rotational top-k kernel finding the k largest elements along a dimension.

Uses multi-stage rotation and reduction optimized for NeuronCore architecture.

Vendored from nkilib ``core/topk`` (KaenaNeuronKernelLibrary 88ffd98c, see this package's
``__init__``) and patched here: ``_topk_rotated_core`` gathers each stage's global indices into
its own columns of the ``indices`` tile it reads rather than a fresh tile, and ``rotational_topk``
builds ``indices`` per row tile so that each of its columns is written once. The gather then
never writes SBUF its sources occupy (a gather dst/src alias, which the device's pieced gather
turns into wrong indices). Carry the patch upstream, or apply it again, before nkilib's copy
replaces this one.

It also diverges from upstream in how the sorted output is built: consecutive row tiles share
one ``sort``, so that its passes hold a row on every partition rather than on ``tile_size`` of
them. Upstream sorts each row tile alone. The passes cost the same on any number of partitions
and work on each partition alone, so the output is the same, bit for bit, in fewer passes.
"""

from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from .cascaded_max_utils import predicated_folded_load, unfolded_store
from .rotational_topk_utils import (
    HW_PARAMS,
    RotationalTopkConfig,
    TopkConfig,
    _get_dtype_min,
    build_rotation_matrix,
    build_stage_offsets,
    insert,
    naive_scanning_topk,
    reshape_with_dma,
    rotate,
    sort,
    topk_core,
    validate_config,
    validate_topk_input,
)


@nki.jit
def rotational_topk(
    inp: nl.NkiTensor, config: RotationalTopkConfig
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Find the k largest elements along the last dimension using rotational algorithm.

    This kernel implements a multi-stage rotational reduction algorithm that efficiently
    finds top-k elements by rotating local maxima across stages and accumulating results.
    The algorithm is optimized for NeuronCore architecture with support for LNC sharding.

    Dimensions:
        B: Batch size
        S: Sequence length
        V: Vocabulary size (dimension to reduce over)
        k: Number of top elements to retrieve

    Args:
        inp (nl.NkiTensor): [B, S, V] or [BxS, V], Input tensor in HBM
        config (RotationalTopkConfig): Configuration object containing algorithm parameters

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor]: A tuple containing:
            - topk_values: [B, S, k], Top-k values with original shape preserved
            - topk_indices: [B, S, k], Global indices of top-k elements

    Notes:
        - Falls back to scanning approach if only 1 stage fits in memory
        - Supports optional sorting of output via config.sorted flag
        - Uses LNC sharding for parallel execution across multiple cores
        - Handles padding when k is not divisible by 8
        - Optimizes tile size based on vocab_size, k, and sort requirements
        - HW constraints:
            * vocab_size/n_stages must be <= 2^14 (DVE instruction limit)
        - Supports tiling over BxS dimension when BxS > 128
        - Tested range: vocab_size up to 151,936, k up to 2,048, batch up to 1,024

    Pseudocode:
        # Validate inputs
        validate_topk_input(inp)
        validate_config(config.topk_config)

        # Handle single-stage case
        if n_stages == 1:
            return naive_scanning_topk(inp, config.topk_config)

        # Row tiles in groups of sort_group (pmax // tile_size when sorted, else 1)
        for group in row_tile_groups:
            for tile in group:
                # Multi-stage rotational algorithm per tile
                value, global_index = _topk_rotated_core(inp, config, tile)
                if sorted:
                    flat_value[tile rows] = reshape_with_dma(value, n_stages)
                    flat_index[tile rows] = reshape_with_dma(global_index, n_stages)
                else:
                    unfolded_store(value, global_index to HBM)

            # Optional sorting (per group: a row per partition)
            if sorted:
                sorted_val, sorted_idx = sort(flat_value, flat_index)
                dma_copy(sorted_val[:true_k], sorted_idx[:true_k] to HBM)

        return topk_values, topk_indices
    """
    validate_topk_input(
        inp, n_fold=config.n_stages, local_top_k_per_stage=config.local_top_k_per_stage
    )
    validate_config(config.topk_config)

    # Query runtime shard info (replaces the old config.update_shard_info() call).
    # prg_id and n_prgs must come from the NKI runtime context inside the kernel,
    # not from config construction time outside the kernel.
    shard_info = get_verified_program_sharding_info("topk", (0, 1), 2)
    kernel_assert(shard_info[1] == config.n_prgs or config.BxS == 1, "n_prgs mismatch")
    if config.BxS > 1:
        n_prgs = shard_info[1]
        prg_id = shard_info[2]
    else:
        kernel_assert(
            config.n_prgs == 1,
            f"n_prgs mismatch, BxS {config.BxS}, n_programs {config.n_prgs}",
        )
        n_prgs = config.n_prgs
        prg_id = config.prg_id

    BxS = config.BxS
    true_k = config.orig_k
    sorted_flag = config.sorted
    index_dtype = config.topk_config.index_dtype
    output_shape = (BxS, true_k)

    # Trivial case: k == vocab_size, return input as-is with sequential indices.
    if true_k == config.vocab_size:
        kernel_assert(
            not sorted_flag,
            f"sorted=True is not supported when k == vocab_size ({true_k}). Use k < vocab_size for sorted output.",
        )
        P_MAX = nl.tile_size.pmax
        topk_indices = nl.ndarray(output_shape, dtype=index_dtype, buffer=nl.shared_hbm)

        tile_rows = min(BxS, P_MAX)
        idx_sb = nl.ndarray((tile_rows, true_k), dtype=index_dtype, buffer=nl.sbuf)
        nisa.iota(idx_sb, [[1, true_k]], offset=0)

        n_full_tiles = BxS // tile_rows
        remainder = BxS % tile_rows
        for tile_idx in nl.affine_range(n_full_tiles):
            nisa.dma_copy(
                dst=topk_indices[nl.ds(tile_idx * tile_rows, tile_rows), :], src=idx_sb
            )
        if remainder > 0:
            nisa.dma_copy(
                dst=topk_indices[nl.ds(n_full_tiles * tile_rows, remainder), :],
                src=idx_sb[nl.ds(0, remainder), :],
            )

        return inp, topk_indices

    # Handle single-stage case (falls back to scanning)
    if config.n_stages == 1:
        # Create a runtime-corrected TopkConfig with the actual prg_id/n_prgs
        # (the original update_shard_info() did this by reconstructing topk_config)
        runtime_topk_config = TopkConfig(
            inp_shape=config.topk_config.inp_shape,
            k=config.orig_k,
            sorted=config.sorted,
            inp_dtype=config.inp_dtype,
            index_dtype=config.index_dtype,
            BxS=config.topk_config.BxS,
            vocab_size=config.topk_config.vocab_size,
            out_shape=config.topk_config.out_shape,
            n_prgs=n_prgs,
            prg_id=prg_id,
            per_lnc_BxS=config.topk_config.per_lnc_BxS,
            _pmax=config.topk_config._pmax,
        )
        topk_values, topk_indices = naive_scanning_topk(
            inp=inp, topk_config=runtime_topk_config
        )
        return topk_values, topk_indices

    topk_values = nl.ndarray(output_shape, dtype=inp.dtype, buffer=nl.shared_hbm)
    topk_indices = nl.ndarray(output_shape, dtype=index_dtype, buffer=nl.shared_hbm)

    tile_size = config.tile_size
    n_bxs_tiles = config.n_bxs_tiles
    lnc_batch_start = prg_id * config.per_lnc_BxS
    lnc_batch_end = min(lnc_batch_start + config.per_lnc_BxS, BxS)
    # Row tiles whose rows one `sort` takes together. `sort` holds a row per partition, so one
    # row tile fills only `tile_size` of them, and its passes cost the same on any number of
    # partitions; consecutive tiles therefore share one `sort`, as many as fill the
    # partitions. Every instruction of `sort` reads and writes each partition alone, so a
    # row's output does not depend on the rows beside it.
    sort_group = min(n_bxs_tiles, nl.tile_size.pmax // tile_size) if sorted_flag else 1

    # Hoist tile-invariant constants out of the loop
    n_stages = config.n_stages
    stage_free_size = config.stage_free_size
    total_partition_dim = n_stages * tile_size
    local_top_k_per_stage = config.local_top_k_per_stage
    # The columns of one row tile's `indices` (laid out in `_topk_rotated_core`): the widest
    # `data` a stage gathers from (the chunk's global indices and the rotated indices every stage
    # but the last inserts), then one block of gathered global indices per stage.
    index_free_dim = (
        stage_free_size
        + (n_stages - 1) * local_top_k_per_stage
        + n_stages * local_top_k_per_stage
    )

    chunk_index = nl.ndarray((total_partition_dim, stage_free_size), dtype=nl.float32)
    nisa.iota(dst=chunk_index, pattern=[[1, stage_free_size]], offset=0)

    stage_offsets = build_stage_offsets(n_stages, tile_size, stage_free_size)

    rotation = build_rotation_matrix(n_stages, tile_size, inp.dtype)
    rotation_f32 = nl.ndarray(
        (total_partition_dim, total_partition_dim), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.tensor_copy(dst=rotation_f32, src=rotation, engine=nisa.vector_engine)

    for group_start in nl.sequential_range(0, n_bxs_tiles, sort_group):
        group_tiles = min(sort_group, n_bxs_tiles - group_start)
        group_batch_start = lnc_batch_start + group_start * tile_size
        group_batch_end = min(group_batch_start + group_tiles * tile_size, lnc_batch_end)

        if sorted_flag:
            # The group's rows, a row per partition: each tile's rows reach their own
            # partitions by the DMA that takes them out of the stages layout.
            flat_shape = (group_tiles * tile_size, n_stages * local_top_k_per_stage)
            flat_value = nl.ndarray(flat_shape, dtype=inp.dtype, buffer=nl.sbuf)
            flat_index = nl.ndarray(flat_shape, dtype=index_dtype, buffer=nl.sbuf)

        for member in nl.static_range(group_tiles):
            tile_batch_start = group_batch_start + member * tile_size
            tile_batch_end = min(tile_batch_start + tile_size, lnc_batch_end)

            # One `indices` per row tile, so that each of its columns is written once (see
            # `_topk_rotated_core`). The scalar engine fills it: the top-k keeps the vector
            # engine busy, so the fill of one tile can run under the top-k of the tile before.
            indices = nl.ndarray((total_partition_dim, index_free_dim), dtype=nl.float32)
            nisa.tensor_scalar(
                dst=indices[:, nl.ds(0, stage_free_size)],
                data=chunk_index,
                op0=nl.add,
                operand0=stage_offsets,
                engine=nisa.scalar_engine,
            )

            value, global_index = _topk_rotated_core(
                inp=inp,
                config=config,
                batch_start=tile_batch_start,
                batch_end=tile_batch_end,
                rotation=rotation,
                rotation_f32=rotation_f32,
                indices=indices,
            )

            if sorted_flag:
                member_rows = nl.ds(member * tile_size, tile_size)
                reshape_with_dma(value, n_stages, dst=flat_value[member_rows, :])
                reshape_with_dma(global_index, n_stages, dst=flat_index[member_rows, :])
            else:
                global_index_int = nl.ndarray(
                    global_index.shape, dtype=index_dtype, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=global_index_int, src=global_index)

                unfolded_store(
                    global_index_int[:, :],
                    topk_indices,
                    fold_factor=config.n_stages,
                    batch_start=tile_batch_start,
                    batch_end=tile_batch_end,
                )
                unfolded_store(
                    value[:, :],
                    topk_values,
                    fold_factor=config.n_stages,
                    batch_start=tile_batch_start,
                    batch_end=tile_batch_end,
                )

        if sorted_flag:
            trimmed_val, trimmed_idx = sort(flat_value, flat_index, true_k)
            # The group's rows are consecutive, in HBM as on the partitions: only its last
            # tile can be short.
            group_bxs = group_batch_end - group_batch_start
            hbm_slice = nl.ds(group_batch_start, group_bxs)
            sbuf_slice = nl.ds(0, group_bxs)
            nisa.dma_copy(
                dst=topk_indices[hbm_slice, :true_k],
                src=trimmed_idx[sbuf_slice, :true_k],
            )
            nisa.dma_copy(
                dst=topk_values[hbm_slice, :true_k],
                src=trimmed_val[sbuf_slice, :true_k],
            )

    return topk_values, topk_indices


def _topk_rotated_core(
    inp: nl.NkiTensor,
    config: RotationalTopkConfig,
    batch_start: int,
    batch_end: int,
    rotation: nl.NkiTensor,
    rotation_f32: nl.NkiTensor,
    indices: nl.NkiTensor,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Core rotational top-k algorithm implementation.

    Performs multi-stage rotation and reduction to find top-k elements efficiently
    by rotating local maxima across stages and accumulating results.
    Uses on-chip index generation via nisa.iota + nisa.tensor_scalar,
    fast_folded_load with targeted padding, and skips rotation on the last stage.

    Args:
        inp (nl.NkiTensor): [BxS, V], Input tensor in HBM
        config (RotationalTopkConfig): Configuration with algorithm parameters
        batch_start (int): Start index for batch tile
        batch_end (int): End index for batch tile
        rotation (nl.NkiTensor): [total_partition_dim, total_partition_dim], Rotation matrix
        rotation_f32 (nl.NkiTensor): The rotation matrix in float32, for the indices
        indices (nl.NkiTensor): [total_partition_dim, stage_free_size + (2 * n_stages - 1) *
            local_top_k_per_stage], This tile's global indices, written by the caller in the
            first stage_free_size columns and in the rest only here

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor]: A tuple containing:
            - value: [total_partition_dim, local_top_k_per_stage], Top-k values
            - global_index: [total_partition_dim, local_top_k_per_stage], Global indices,
              a column slice of indices

    Pseudocode:
        # Initialize buffers with on-chip index generation
        values = folded_load(inp, n_stages)
        indices[:, :stage_free_size] = iota(0..stage_free_size) + stage_offsets  # by the caller
        rotation_matrix = load_circulant_permutation(n_stages, BxS)
        gathered_start = stage_free_size + (local_top_k * (n_stages - 1))

        # Iterative rotation and top-k
        for stage_idx in range(n_stages):
            offset = stage_free_size + (local_top_k * stage_idx)
            local_vals, local_idx = topk_core(values[:, :offset], k=local_top_k)
            global_idx = indices[:, gathered_start + (local_top_k * stage_idx):][:, :local_top_k]
            global_idx[...] = gather(indices[:, :offset], local_idx)
            if stage_idx < n_stages - 1:
                rotated_vals = matmul(rotation_matrix, local_vals)
                rotated_idx = matmul(rotation_matrix, global_idx)
                values[:, offset:offset+local_top_k] = rotated_vals
                indices[:, offset:offset+local_top_k] = rotated_idx

        return local_vals, global_idx
    """
    n_stages = config.n_stages
    local_top_k_per_stage = config.local_top_k_per_stage
    stage_free_size = config.stage_free_size
    BxS_size = config.tile_size

    total_partition_dim = n_stages * BxS_size
    concatenated_stage_free_dim = stage_free_size + (n_stages * local_top_k_per_stage)
    # The last stage reads the widest `data`, `indices[:, :offset]` at its `offset`; each
    # stage's gathered global indices go to its own block of columns past it.
    gathered_start = stage_free_size + (local_top_k_per_stage * (n_stages - 1))

    values = nl.ndarray(
        (total_partition_dim, concatenated_stage_free_dim), dtype=inp.dtype
    )

    # `predicated_folded_load` writes only `batch_bound` of the `n_stages * tile_size`
    # partitions, so a ragged tile leaves whole partitions uninitialised. `rotate`'s
    # `nisa.nc_matmul` then contracts the partition axis with 0/1 weights, so one
    # non-finite word in an unwritten partition turns every output partition into NaN
    # (`0.0 * inf`). The scanning path guards the same case in `rotational_topk_utils`.
    if (batch_end - batch_start) < BxS_size:
        nisa.memset(values, value=_get_dtype_min(inp.dtype))

    predicated_folded_load(
        data_hbm=inp,
        fold_factor=n_stages,
        data_sb=values,
        batch_start=batch_start,
        batch_end=batch_end,
    )

    for stage_idx in nl.static_range(n_stages):
        offset = stage_free_size + (local_top_k_per_stage * stage_idx)

        value, local_index = topk_core(data=values[:, :offset], k=local_top_k_per_stage)

        # Gather into this stage's own block of `indices`, which is disjoint by its columns from
        # `data` and is not `local_index`. The device runs `nc_n_gather` in pieces that each read
        # their part of `data` and of the indices from SBUF, so a destination over a source lets
        # a piece read what an earlier piece wrote (the gather dst/src alias); the simulator
        # gathers in one step and never shows it. A fresh destination tile is born at the
        # gather, where on the last stage both sources die, and the backend's SBUF placement
        # (`address_rotation_sb`) put it on the first bytes of `indices`. A column slice keeps
        # the destination in the memory location of `data` only while each column of `indices`
        # is written once: the backend may give a tile whose columns are written again a new
        # memory location there (an SSA clone), which it places like a fresh tile. So the
        # caller allocates `indices` per row tile, and every insert and gather here writes
        # columns nothing else writes.
        global_index = indices[
            :,
            nl.ds(
                gathered_start + (local_top_k_per_stage * stage_idx),
                local_index.shape[1],
            ),
        ]
        # Tile the gather into chunks no wider than the nc_n_gather ISA group size. A
        # single gather wider than this splits into multiple internal ISA groups, and
        # that multi-group form corrupts the tail elements of the last BxS tile on
        # hardware while the simulator (which executes the gather atomically) stays
        # correct (NKILIB-1592).
        gather_group_size = HW_PARAMS.gather_group_size
        gather_width = local_index.shape[1]
        n_gather_tiles = (gather_width + gather_group_size - 1) // gather_group_size
        for gather_tile in nl.static_range(n_gather_tiles):
            chunk = min(
                gather_group_size, gather_width - gather_tile * gather_group_size
            )
            chunk_slice = nl.ds(gather_tile * gather_group_size, chunk)
            nisa.nc_n_gather(
                dst=global_index[:, chunk_slice],
                data=indices[:, :offset],
                indices=local_index[:, chunk_slice],
            )

        if stage_idx < n_stages - 1:
            rotated_index = nl.ndarray(
                global_index.shape, dtype=nl.float32, buffer=nl.psum
            )
            rotated = nl.ndarray(value.shape, dtype=nl.float32, buffer=nl.psum)

            rotate(dst=rotated_index, tensor=global_index, rotation_matrix=rotation_f32)
            rotate(dst=rotated, tensor=value, rotation_matrix=rotation)

            insert(tensor=values, values=rotated, offset=offset)
            insert(tensor=indices, values=rotated_index, offset=offset)

    return value, global_index


# NOTE: the upstream nkilib host-side ``topk()`` dispatcher,
# ``SUPPORTED_TOPK_METHOD_MAPPING``, and the ``_kernel`` grid wrapper are
# intentionally NOT vendored here. vLLM-Neuron owns its own dispatch + grid
# launch (functional/topk.py: ``_select_topk_method`` + ``wrap_nki``), so only
