# SPDX-License-Identifier: Apache-2.0
"""Selected-latent MLA decode for pinned GLM-5.2 context buckets.

The kernel absorbs the non-rotary query into the MLA projection, gathers only
the DSA-selected latent rows, and performs online softmax.  It never exports a
``[batch, query, selected, 576]`` tensor to Torch/HBM.

This kernel is deliberately narrow: one query, one TP-local head, BF16 or raw
E4M3 cache, checkpoint-native block-FP8 projection weights, and an exact static
selection width for each supported context bucket. Production dispatch is
guarded by an exact default-off contract.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from nkilib.core.utils.tensor_view import TensorView

from vllm_neuron.nki.nki_hop import wrap_nki


_CACHE_WIDTH = 576
_CACHE_HALF_WIDTH = _CACHE_WIDTH // 2
_LATENT_WIDTH = 512
_ROPE_WIDTH = 64
_QK_NOPE_WIDTH = 192
_QK_WIDTH = 256
_VALUE_WIDTH = 256
_WEIGHT_WIDTH = _QK_NOPE_WIDTH + _VALUE_WIDTH
_SELECTED_TILE = 128
_MASK_FLOOR = -9984.0
SELECTED_LATENT_MLA_BLOCK_SIZES = (16, 32)
SELECTED_LATENT_MLA_SHORT_CONTEXT_BUCKETS = (128, 512)
SELECTED_LATENT_MLA_LONG_CONTEXT_BUCKETS = (4096, 8192)
SELECTED_LATENT_MLA_SHORT_WIDTH = 512
SELECTED_LATENT_MLA_LONG_WIDTH = 2048


def _kernel_assert(condition: bool, error_text: str) -> None:
    assert condition, (
        "[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: " + error_text
    )


@nki.jit
def _selected_latent_mla_decode_nki(
    queries,
    mla_k_cache,
    mla_v_cache,
    block_table,
    selected_indices,
    weight,
    weight_scale_inv,
    block_size,
    row_offset=0,
):
    """Compute one-head MLA decode directly from selected paged-cache rows.

    Dimensions:
        B: Static batch size.
        P: Number of physical cache blocks.
        L: Maximum logical blocks per request.
        K: Static selected width: 512 for short buckets or 2048 for long
            buckets.

    Args:
        queries: ``[B, 1, 1, 256]`` BF16 query tensor in HBM.
        mla_k_cache: ``[P, 1, block_size, 288]`` BF16 or raw E4M3 first cache
            half.
        mla_v_cache: ``[P, 1, block_size, 288]`` BF16 or raw E4M3 second cache
            half.
        block_table: ``[B, L]`` int32 logical-to-physical block mapping.
        selected_indices: ``[B, 1, K]`` int32 logical row indices in HBM.
        weight: ``[448, 512]`` raw E4M3 rank-local kv_b projection weight in
            HBM. The first 192 rows project keys and the final 256 rows project
            values.
        weight_scale_inv: ``[4, 4]`` FP32 128x128 inverse-scale grid in HBM.
        block_size: Static number of rows per physical cache block.
        row_offset: Rank-local first-row offset in its global 128-row FP8
            block. TP64 uses zero on even ranks and 64 on odd ranks.

    Returns:
        ``[B, 1, 1, 256]`` BF16 attention output in shared HBM.

    Notes:
        - Negative/out-of-range indices and invalid physical blocks are
          padding and contribute zero.
        - The largest selected-latent allocation is one ``[128, 576]`` SBUF
          tile.  No selected-latent tensor is materialized in HBM.
        - Block-FP8 scales are applied to FP32 matmul partials before
          accumulation, matching ``block_fp8.py``.
        - Query/key score and online-softmax accumulation use FP32.
    """

    _kernel_assert(len(queries.shape) == 4, "queries must be rank four")
    _kernel_assert(queries.shape[1:] == (1, 1, _QK_WIDTH), "query shape mismatch")
    _kernel_assert(len(mla_k_cache.shape) == 4, "K cache must be rank four")
    _kernel_assert(mla_v_cache.shape == mla_k_cache.shape, "cache halves mismatch")
    _kernel_assert(
        mla_v_cache.dtype == mla_k_cache.dtype,
        "cache half dtypes mismatch",
    )
    _kernel_assert(
        mla_k_cache.dtype == queries.dtype or mla_k_cache.dtype == nl.float8_e4m3,
        "cache dtype must be BF16 or raw E4M3",
    )
    _kernel_assert(mla_k_cache.shape[1] == 1, "cache must have one head")
    _kernel_assert(
        mla_k_cache.shape[3] == _CACHE_HALF_WIDTH,
        "cache half width mismatch",
    )
    _kernel_assert(
        block_size == 16 or block_size == 32,
        "pinned cache block size must be 16 or 32",
    )
    _kernel_assert(block_size == mla_k_cache.shape[2], "cache block size mismatch")
    _kernel_assert(len(block_table.shape) == 2, "block table must be rank two")
    _kernel_assert(block_table.shape[0] == queries.shape[0], "batch mismatch")
    _kernel_assert(len(selected_indices.shape) == 3, "selection must be rank three")
    _kernel_assert(
        selected_indices.shape[:2] == queries.shape[:2],
        "selection/query shape mismatch",
    )
    _kernel_assert(
        weight.shape == (_WEIGHT_WIDTH, _LATENT_WIDTH),
        "weight shape mismatch",
    )
    _kernel_assert(
        weight_scale_inv.shape == (4, 4),
        "weight scale shape mismatch",
    )
    _kernel_assert(row_offset == 0 or row_offset == 64, "row offset mismatch")
    batch_size = queries.shape[0]
    physical_block_count = mla_k_cache.shape[0]
    logical_block_count = block_table.shape[1]
    logical_key_count = logical_block_count * block_size
    physical_row_count = physical_block_count * block_size
    short_context = logical_key_count == 128 or logical_key_count == 512
    long_context = logical_key_count == 4096 or logical_key_count == 8192
    _kernel_assert(
        short_context or long_context,
        "unsupported logical key bucket",
    )
    selected_count = selected_indices.shape[2]
    expected_selected_count = (
        SELECTED_LATENT_MLA_SHORT_WIDTH
        if short_context
        else SELECTED_LATENT_MLA_LONG_WIDTH
    )
    _kernel_assert(
        selected_count == expected_selected_count,
        "selected width does not match logical key bucket",
    )
    if block_size == 16:
        block_shift = 4
        block_row_mask = 15
    else:
        block_shift = 5
        block_row_mask = 31
    scale = _QK_WIDTH**-0.5
    output = nl.ndarray(
        (batch_size, 1, 1, _VALUE_WIDTH),
        dtype=queries.dtype,
        buffer=nl.shared_hbm,
    )

    flat_queries = queries.reshape((batch_size, _QK_WIDTH))
    flat_selected = selected_indices.reshape((batch_size, selected_count))
    flat_k_cache = mla_k_cache.reshape((physical_row_count, _CACHE_HALF_WIDTH))
    flat_v_cache = mla_v_cache.reshape((physical_row_count, _CACHE_HALF_WIDTH))
    flat_block_table = block_table.reshape((batch_size * logical_block_count, 1))

    for batch_index in nl.affine_range(batch_size):
        # q_nope @ W_k -> absorbed 512-wide latent query.  Accumulate each
        # aligned output-row FP8 block only after applying its own scale.
        absorbed_query_fp32 = nl.ndarray(
            (1, _LATENT_WIDTH), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=absorbed_query_fp32, value=0.0)
        for latent_tile_index in nl.affine_range(4):
            latent_start = latent_tile_index * _SELECTED_TILE
            absorbed_tile = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.memset(dst=absorbed_tile, value=0.0)
            for output_block in nl.affine_range(4):
                key_start = max(0, output_block * _SELECTED_TILE - row_offset)
                key_end = min(
                    _QK_NOPE_WIDTH,
                    (output_block + 1) * _SELECTED_TILE - row_offset,
                )
                key_size = max(0, key_end - key_start)
                if key_size > 0:
                    q_nope = nl.ndarray(
                        (key_size, 1), dtype=queries.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_transpose(
                        dst=q_nope,
                        src=flat_queries[
                            batch_index : batch_index + 1,
                            key_start:key_end,
                        ],
                        axes=(1, 0),
                    )
                    key_weight_fp8 = nl.ndarray(
                        (key_size, _SELECTED_TILE),
                        dtype=weight.dtype,
                        buffer=nl.sbuf,
                    )
                    nisa.dma_copy(
                        dst=key_weight_fp8,
                        src=weight[
                            key_start:key_end,
                            latent_start : latent_start + _SELECTED_TILE,
                        ],
                    )
                    key_weight = nl.ndarray(
                        (key_size, _SELECTED_TILE),
                        dtype=nl.bfloat16,
                        buffer=nl.sbuf,
                    )
                    nisa.tensor_copy(dst=key_weight, src=key_weight_fp8)
                    absorbed_partial_psum = nl.ndarray(
                        (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(
                        dst=absorbed_partial_psum,
                        stationary=q_nope,
                        moving=key_weight,
                        accumulate=False,
                    )
                    absorbed_partial = nl.ndarray(
                        (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(
                        dst=absorbed_partial,
                        src=absorbed_partial_psum,
                    )
                    key_scale = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=key_scale,
                        src=weight_scale_inv[
                            output_block : output_block + 1,
                            latent_tile_index : latent_tile_index + 1,
                        ],
                    )
                    scaled_absorbed_partial = nl.ndarray(
                        (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_scalar(
                        dst=scaled_absorbed_partial,
                        data=absorbed_partial,
                        op0=nl.multiply,
                        operand0=key_scale,
                    )
                    nisa.tensor_tensor(
                        dst=absorbed_tile,
                        data1=absorbed_tile,
                        data2=scaled_absorbed_partial,
                        op=nl.add,
                    )
            nisa.tensor_copy(
                dst=absorbed_query_fp32[
                    0:1, latent_start : latent_start + _SELECTED_TILE
                ],
                src=absorbed_tile,
            )
        absorbed_query = nl.ndarray(
            (1, _LATENT_WIDTH), dtype=queries.dtype, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=absorbed_query, src=absorbed_query_fp32)

        q_rope = nl.ndarray((_ROPE_WIDTH, 1), dtype=queries.dtype, buffer=nl.sbuf)
        nisa.dma_transpose(
            dst=q_rope,
            src=flat_queries[batch_index : batch_index + 1, _QK_NOPE_WIDTH:_QK_WIDTH],
            axes=(1, 0),
        )

        running_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        running_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        latent_accumulator = nl.ndarray(
            (1, _LATENT_WIDTH), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=running_max, value=_MASK_FLOOR)
        nisa.memset(dst=running_sum, value=0.0)
        nisa.memset(dst=latent_accumulator, value=0.0)

        for selected_tile_index in nl.sequential_range(
            selected_count // _SELECTED_TILE
        ):
            selected_start = selected_tile_index * _SELECTED_TILE
            index_rows = nl.ndarray((_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_transpose(
                dst=index_rows,
                src=flat_selected[
                    batch_index : batch_index + 1,
                    selected_start : selected_start + _SELECTED_TILE,
                ],
                axes=(1, 0),
            )

            logical_blocks = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=logical_blocks,
                data=index_rows,
                op0=nl.right_shift,
                operand0=block_shift,
                engine=nisa.engine.vector,
            )
            logical_rows_in_block = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=logical_rows_in_block,
                data=index_rows,
                op0=nl.bitwise_and,
                operand0=block_row_mask,
                engine=nisa.engine.vector,
            )
            block_table_rows = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            logical_block_valid_low = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            logical_block_valid_high = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            logical_block_valid_float = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            logical_block_valid = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=logical_block_valid_low,
                data=logical_blocks,
                op0=nl.greater,
                operand0=-1,
            )
            nisa.tensor_scalar(
                dst=logical_block_valid_high,
                data=logical_blocks,
                op0=nl.less,
                operand0=logical_block_count,
            )
            nisa.tensor_tensor(
                dst=logical_block_valid_float,
                data1=logical_block_valid_low,
                data2=logical_block_valid_high,
                op=nl.multiply,
            )
            nisa.tensor_copy(
                dst=logical_block_valid,
                src=logical_block_valid_float,
            )
            block_table_rows_plus_one = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=block_table_rows_plus_one,
                data=logical_blocks,
                op0=nl.add,
                operand0=batch_index * logical_block_count + 1,
            )
            nisa.tensor_tensor(
                dst=block_table_rows,
                data1=block_table_rows_plus_one,
                data2=logical_block_valid,
                op=nl.multiply,
            )
            nisa.tensor_scalar(
                dst=block_table_rows,
                data=block_table_rows,
                op0=nl.add,
                operand0=-1,
            )
            physical_blocks = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.memset(dst=physical_blocks, value=-1)
            nisa.dma_copy(
                dst=physical_blocks,
                src=flat_block_table.ap(
                    pattern=[[1, _SELECTED_TILE], [1, 1]],
                    vector_offset=block_table_rows,
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=nisa.oob_mode.skip,
            )
            physical_block_offsets = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=physical_block_offsets,
                data=physical_blocks,
                op0=nl.multiply,
                operand0=block_size,
            )
            physical_rows = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=physical_rows,
                data1=physical_block_offsets,
                data2=logical_rows_in_block,
                op=nl.add,
            )

            selected_cache_storage = nl.ndarray(
                (_SELECTED_TILE, _CACHE_WIDTH),
                dtype=mla_k_cache.dtype,
                buffer=nl.sbuf,
            )
            nisa.memset(dst=selected_cache_storage, value=0.0)
            nisa.dma_copy(
                dst=selected_cache_storage[0:_SELECTED_TILE, 0:_CACHE_HALF_WIDTH],
                src=flat_k_cache.ap(
                    pattern=[
                        [_CACHE_HALF_WIDTH, _SELECTED_TILE],
                        [1, _CACHE_HALF_WIDTH],
                    ],
                    vector_offset=physical_rows,
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=nisa.oob_mode.skip,
            )
            nisa.dma_copy(
                dst=selected_cache_storage[
                    0:_SELECTED_TILE, _CACHE_HALF_WIDTH:_CACHE_WIDTH
                ],
                src=flat_v_cache.ap(
                    pattern=[
                        [_CACHE_HALF_WIDTH, _SELECTED_TILE],
                        [1, _CACHE_HALF_WIDTH],
                    ],
                    vector_offset=physical_rows,
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=nisa.oob_mode.skip,
            )
            selected_cache = nl.ndarray(
                (_SELECTED_TILE, _CACHE_WIDTH),
                dtype=queries.dtype,
                buffer=nl.sbuf,
            )
            nisa.tensor_copy(dst=selected_cache, src=selected_cache_storage)

            # Scores = absorbed_q @ latent.T + q_rope @ cached_rope.T.
            score_psum = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.psum
            )
            for latent_tile_index in nl.affine_range(4):
                latent_start = latent_tile_index * _SELECTED_TILE
                latent_transpose = nl.ndarray(
                    (_SELECTED_TILE, _SELECTED_TILE),
                    dtype=queries.dtype,
                    buffer=nl.sbuf,
                )
                nisa.dma_transpose(
                    dst=latent_transpose,
                    src=selected_cache[
                        0:_SELECTED_TILE,
                        latent_start : latent_start + _SELECTED_TILE,
                    ],
                    axes=(1, 0),
                )
                absorbed_tile = nl.ndarray(
                    (_SELECTED_TILE, 1), dtype=queries.dtype, buffer=nl.sbuf
                )
                nisa.dma_transpose(
                    dst=absorbed_tile,
                    src=absorbed_query[
                        0:1, latent_start : latent_start + _SELECTED_TILE
                    ],
                    axes=(1, 0),
                )
                nisa.nc_matmul(
                    dst=score_psum,
                    stationary=absorbed_tile,
                    moving=latent_transpose,
                    accumulate=(latent_tile_index > 0),
                )

            rope_transpose = nl.ndarray(
                (_ROPE_WIDTH, _SELECTED_TILE),
                dtype=queries.dtype,
                buffer=nl.sbuf,
            )
            nisa.dma_transpose(
                dst=rope_transpose,
                src=selected_cache[0:_SELECTED_TILE, _LATENT_WIDTH:_CACHE_WIDTH],
                axes=(1, 0),
            )
            nisa.nc_matmul(
                dst=score_psum,
                stationary=q_rope,
                moving=rope_transpose,
                accumulate=True,
            )

            scores = nl.ndarray((1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=scores, src=score_psum)
            nisa.tensor_scalar(dst=scores, data=scores, op0=nl.multiply, operand0=scale)

            index_columns = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.dma_transpose(dst=index_columns, src=index_rows, axes=(1, 0))
            valid_low = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            valid_high = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            physical_block_columns = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.dma_transpose(
                dst=physical_block_columns,
                src=physical_blocks,
                axes=(1, 0),
            )
            valid_physical_low = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            valid_physical_high = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            valid = nl.ndarray((1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=valid_low,
                data=index_columns,
                op0=nl.greater,
                operand0=-1,
            )
            nisa.tensor_scalar(
                dst=valid_high,
                data=index_columns,
                op0=nl.less,
                operand0=logical_key_count,
            )
            nisa.tensor_scalar(
                dst=valid_physical_low,
                data=physical_block_columns,
                op0=nl.greater,
                operand0=-1,
            )
            nisa.tensor_scalar(
                dst=valid_physical_high,
                data=physical_block_columns,
                op0=nl.less,
                operand0=physical_block_count,
            )
            nisa.tensor_tensor(
                dst=valid, data1=valid_low, data2=valid_high, op=nl.multiply
            )
            nisa.tensor_tensor(
                dst=valid,
                data1=valid,
                data2=valid_physical_low,
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=valid,
                data1=valid,
                data2=valid_physical_high,
                op=nl.multiply,
            )
            invalid_bias = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=invalid_bias,
                data=valid,
                op0=nl.multiply,
                operand0=-_MASK_FLOOR,
                op1=nl.add,
                operand1=_MASK_FLOOR,
            )
            nisa.tensor_tensor(
                dst=scores,
                data1=scores,
                data2=valid,
                op=nl.multiply,
            )
            nisa.tensor_tensor(dst=scores, data1=scores, data2=invalid_bias, op=nl.add)

            tile_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_max, data=scores, op=nl.maximum, axis=1)
            new_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=new_max, data1=running_max, data2=tile_max, op=nl.maximum
            )

            old_shift = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            old_scale = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=old_shift, data1=running_max, data2=new_max, op=nl.subtract
            )
            nisa.activation(dst=old_scale, data=old_shift, op=nl.exp)

            score_shift = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            probabilities = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=score_shift,
                data=scores,
                op0=nl.subtract,
                operand0=new_max,
            )
            nisa.activation(dst=probabilities, data=score_shift, op=nl.exp)
            nisa.tensor_tensor(
                dst=probabilities,
                data1=probabilities,
                data2=valid,
                op=nl.multiply,
            )
            tile_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_sum, data=probabilities, op=nl.add, axis=1)

            scaled_running_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=scaled_running_sum,
                data1=running_sum,
                data2=old_scale,
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=running_sum,
                data1=scaled_running_sum,
                data2=tile_sum,
                op=nl.add,
            )

            probability_column = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=queries.dtype, buffer=nl.sbuf
            )
            probabilities_compute = nl.ndarray(
                (1, _SELECTED_TILE), dtype=queries.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=probabilities_compute, src=probabilities)
            nisa.dma_transpose(
                dst=probability_column, src=probabilities_compute, axes=(1, 0)
            )
            weighted_latent_psum = nl.ndarray(
                (1, _LATENT_WIDTH), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(
                dst=weighted_latent_psum,
                stationary=probability_column,
                moving=selected_cache[0:_SELECTED_TILE, 0:_LATENT_WIDTH],
                accumulate=False,
            )
            weighted_latent = nl.ndarray(
                (1, _LATENT_WIDTH), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=weighted_latent, src=weighted_latent_psum)
            scaled_latent = nl.ndarray(
                (1, _LATENT_WIDTH), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=scaled_latent,
                data=latent_accumulator,
                op0=nl.multiply,
                operand0=old_scale,
            )
            nisa.tensor_tensor(
                dst=latent_accumulator,
                data1=scaled_latent,
                data2=weighted_latent,
                op=nl.add,
            )
            nisa.tensor_copy(dst=running_max, src=new_max)

        inverse_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        safe_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        normalized_latent = nl.ndarray(
            (1, _LATENT_WIDTH), dtype=queries.dtype, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=safe_sum,
            data=running_sum,
            op0=nl.maximum,
            operand0=1.0e-20,
        )
        nisa.reciprocal(dst=inverse_sum, data=safe_sum)
        nisa.tensor_scalar(
            dst=normalized_latent,
            data=latent_accumulator,
            op0=nl.multiply,
            operand0=inverse_sum,
        )

        accumulated_value = nl.ndarray(
            (1, _VALUE_WIDTH), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=accumulated_value, value=0.0)
        for output_block in nl.affine_range(4):
            value_row_start = max(
                _QK_NOPE_WIDTH,
                output_block * _SELECTED_TILE - row_offset,
            )
            value_row_end = min(
                _WEIGHT_WIDTH,
                (output_block + 1) * _SELECTED_TILE - row_offset,
            )
            value_size = max(0, value_row_end - value_row_start)
            if value_size > 0:
                value_output_start = value_row_start - _QK_NOPE_WIDTH
                value_block = nl.ndarray(
                    (1, value_size), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.memset(dst=value_block, value=0.0)
                for latent_tile_index in nl.affine_range(4):
                    latent_start = latent_tile_index * _SELECTED_TILE
                    latent_column = nl.ndarray(
                        (_SELECTED_TILE, 1),
                        dtype=queries.dtype,
                        buffer=nl.sbuf,
                    )
                    nisa.dma_transpose(
                        dst=latent_column,
                        src=normalized_latent[
                            0:1, latent_start : latent_start + _SELECTED_TILE
                        ],
                        axes=(1, 0),
                    )
                    value_weight_fp8 = nl.ndarray(
                        (value_size, _SELECTED_TILE),
                        dtype=weight.dtype,
                        buffer=nl.sbuf,
                    )
                    nisa.dma_copy(
                        dst=value_weight_fp8,
                        src=weight[
                            value_row_start:value_row_end,
                            latent_start : latent_start + _SELECTED_TILE,
                        ],
                    )
                    value_weight_bf16 = nl.ndarray(
                        (value_size, _SELECTED_TILE),
                        dtype=nl.bfloat16,
                        buffer=nl.sbuf,
                    )
                    nisa.tensor_copy(
                        dst=value_weight_bf16,
                        src=value_weight_fp8,
                    )
                    value_weight_transpose_psum = nl.ndarray(
                        (_SELECTED_TILE, value_size),
                        dtype=nl.bfloat16,
                        buffer=nl.psum,
                    )
                    nisa.nc_transpose(
                        dst=value_weight_transpose_psum,
                        data=value_weight_bf16,
                    )
                    value_weight = nl.ndarray(
                        (_SELECTED_TILE, value_size),
                        dtype=nl.bfloat16,
                        buffer=nl.sbuf,
                    )
                    nisa.tensor_copy(
                        dst=value_weight,
                        src=value_weight_transpose_psum,
                    )
                    value_partial_psum = nl.ndarray(
                        (1, value_size), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(
                        dst=value_partial_psum,
                        stationary=latent_column,
                        moving=value_weight,
                        accumulate=False,
                    )
                    value_partial = nl.ndarray(
                        (1, value_size), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=value_partial, src=value_partial_psum)
                    value_scale = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=value_scale,
                        src=weight_scale_inv[
                            output_block : output_block + 1,
                            latent_tile_index : latent_tile_index + 1,
                        ],
                    )
                    scaled_value_partial = nl.ndarray(
                        (1, value_size), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_scalar(
                        dst=scaled_value_partial,
                        data=value_partial,
                        op0=nl.multiply,
                        operand0=value_scale,
                    )
                    nisa.tensor_tensor(
                        dst=value_block,
                        data1=value_block,
                        data2=scaled_value_partial,
                        op=nl.add,
                    )
                nisa.tensor_copy(
                    dst=accumulated_value[
                        0:1,
                        value_output_start : value_output_start + value_size,
                    ],
                    src=value_block,
                )
        result = nl.ndarray((1, _VALUE_WIDTH), dtype=queries.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result, src=accumulated_value)
        nisa.dma_copy(
            dst=output[batch_index : batch_index + 1, 0:1, 0:1, 0:_VALUE_WIDTH],
            src=result.reshape((1, 1, 1, _VALUE_WIDTH)),
        )

    return output


@nki.jit
def _selected_latent_mla_decode_b32_k512_block32_weight_reuse_nki(
    queries,
    mla_k_cache,
    mla_v_cache,
    block_table,
    selected_indices,
    weight,
    weight_scale_inv,
    block_size,
    row_offset=0,
):
    """Reuse projection-weight preparation for the exact B32/K512 graph.

    Query and value projections run across the 32 request rows before and
    after the request-local attention phase. Each projection weight tile is
    therefore loaded and converted once instead of once per request. Attention
    addressing, masking, online softmax, and normalization remain request
    local. The only cross-request persistent SBUF tensor is the BF16
    ``[128, 4, 32]`` latent batch. Latent features stay on the 128-wide
    partition axis while tile and request indices stay on free dimensions; no
    scratch tensor is materialized in HBM.
    """

    _kernel_assert(
        queries.shape == (32, 1, 1, _QK_WIDTH),
        "weight-reuse query shape must be [32, 1, 1, 256]",
    )
    _kernel_assert(queries.dtype == nl.bfloat16, "queries must be BF16")
    _kernel_assert(len(mla_k_cache.shape) == 4, "K cache must be rank four")
    _kernel_assert(mla_v_cache.shape == mla_k_cache.shape, "cache halves mismatch")
    _kernel_assert(
        mla_v_cache.dtype == mla_k_cache.dtype,
        "cache half dtypes mismatch",
    )
    _kernel_assert(
        mla_k_cache.dtype == queries.dtype or mla_k_cache.dtype == nl.float8_e4m3,
        "cache dtype must be BF16 or raw E4M3",
    )
    _kernel_assert(
        mla_k_cache.shape[1:] == (1, 32, _CACHE_HALF_WIDTH),
        "weight-reuse cache shape mismatch",
    )
    _kernel_assert(block_size == 32, "weight-reuse block size must be 32")
    _kernel_assert(
        block_table.shape == (32, 16),
        "weight-reuse block table must be [32, 16]",
    )
    _kernel_assert(
        selected_indices.shape == (32, 1, SELECTED_LATENT_MLA_SHORT_WIDTH),
        "weight-reuse selection must be [32, 1, 512]",
    )
    _kernel_assert(
        weight.shape == (_WEIGHT_WIDTH, _LATENT_WIDTH),
        "weight shape mismatch",
    )
    _kernel_assert(
        weight_scale_inv.shape == (4, 4),
        "weight scale shape mismatch",
    )
    _kernel_assert(row_offset == 0 or row_offset == 64, "row offset mismatch")

    batch_size = 32
    logical_block_count = 16
    logical_key_count = 512
    selected_count = SELECTED_LATENT_MLA_SHORT_WIDTH
    physical_block_count = mla_k_cache.shape[0]
    physical_row_count = physical_block_count * block_size
    block_shift = 5
    block_row_mask = 31
    scale = _QK_WIDTH**-0.5
    output = nl.ndarray(
        (32, 1, 1, _VALUE_WIDTH),
        dtype=queries.dtype,
        buffer=nl.shared_hbm,
    )

    flat_queries = queries.reshape((batch_size, _QK_WIDTH))
    flat_selected = selected_indices.reshape((batch_size, selected_count))
    flat_k_cache = mla_k_cache.reshape((physical_row_count, _CACHE_HALF_WIDTH))
    flat_v_cache = mla_v_cache.reshape((physical_row_count, _CACHE_HALF_WIDTH))
    flat_block_table = block_table.reshape((batch_size * logical_block_count, 1))

    # The single persistent cross-request tensor first holds absorbed queries.
    # Each request column is overwritten with its normalized latent after
    # attention. Only latent features occupy the partition dimension.
    latent_batch = nl.ndarray(
        (_SELECTED_TILE, 4, 32), dtype=queries.dtype, buffer=nl.sbuf
    )

    # Batched q_nope @ W_k. Weight DMA and E4M3->BF16 preparation are outside
    # the request loop, reducing identical weight work by exactly 32x.
    for latent_tile_index in nl.affine_range(4):
        latent_start = latent_tile_index * _SELECTED_TILE
        absorbed_tile = nl.ndarray(
            (32, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=absorbed_tile, value=0.0)
        for output_block in nl.affine_range(4):
            key_start = max(0, output_block * _SELECTED_TILE - row_offset)
            key_end = min(
                _QK_NOPE_WIDTH,
                (output_block + 1) * _SELECTED_TILE - row_offset,
            )
            key_size = max(0, key_end - key_start)
            if key_size > 0:
                q_nope = nl.ndarray(
                    (key_size, 32), dtype=queries.dtype, buffer=nl.sbuf
                )
                nisa.dma_transpose(
                    dst=q_nope,
                    src=flat_queries[0:32, key_start:key_end],
                    axes=(1, 0),
                )
                key_weight_fp8 = nl.ndarray(
                    (key_size, _SELECTED_TILE),
                    dtype=weight.dtype,
                    buffer=nl.sbuf,
                )
                nisa.dma_copy(
                    dst=key_weight_fp8,
                    src=weight[
                        key_start:key_end,
                        latent_start : latent_start + _SELECTED_TILE,
                    ],
                )
                key_weight = nl.ndarray(
                    (key_size, _SELECTED_TILE),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                nisa.tensor_copy(dst=key_weight, src=key_weight_fp8)
                absorbed_partial_psum = nl.ndarray(
                    (32, _SELECTED_TILE), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_matmul(
                    dst=absorbed_partial_psum,
                    stationary=q_nope,
                    moving=key_weight,
                    accumulate=False,
                )
                absorbed_partial = nl.ndarray(
                    (32, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=absorbed_partial, src=absorbed_partial_psum)
                key_scale = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=key_scale,
                    src=TensorView(
                        weight_scale_inv[
                            output_block : output_block + 1,
                            latent_tile_index : latent_tile_index + 1,
                        ]
                    )
                    .broadcast(dim=0, size=32)
                    .get_view(),
                )
                scaled_absorbed_partial = nl.ndarray(
                    (32, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=scaled_absorbed_partial,
                    data=absorbed_partial,
                    op0=nl.multiply,
                    operand0=key_scale,
                )
                nisa.tensor_tensor(
                    dst=absorbed_tile,
                    data1=absorbed_tile,
                    data2=scaled_absorbed_partial,
                    op=nl.add,
                )
        absorbed_tile_bf16 = nl.ndarray(
            (32, _SELECTED_TILE), dtype=queries.dtype, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=absorbed_tile_bf16, src=absorbed_tile)
        absorbed_tile_transpose_psum = nl.ndarray(
            (_SELECTED_TILE, 32), dtype=queries.dtype, buffer=nl.psum
        )
        nisa.nc_transpose(
            dst=absorbed_tile_transpose_psum,
            data=absorbed_tile_bf16,
        )
        nisa.tensor_copy(
            dst=latent_batch[0:_SELECTED_TILE, latent_tile_index, 0:32],
            src=absorbed_tile_transpose_psum,
        )

    # Keep paged addressing and online softmax request-local and identical to
    # the published selected-MLA path.
    for batch_index in nl.affine_range(32):
        q_rope = nl.ndarray((_ROPE_WIDTH, 1), dtype=queries.dtype, buffer=nl.sbuf)
        nisa.dma_transpose(
            dst=q_rope,
            src=flat_queries[
                batch_index : batch_index + 1, _QK_NOPE_WIDTH:_QK_WIDTH
            ],
            axes=(1, 0),
        )

        running_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        running_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        latent_accumulator = nl.ndarray(
            (1, _LATENT_WIDTH), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=running_max, value=_MASK_FLOOR)
        nisa.memset(dst=running_sum, value=0.0)
        nisa.memset(dst=latent_accumulator, value=0.0)

        for selected_tile_index in nl.sequential_range(
            selected_count // _SELECTED_TILE
        ):
            selected_start = selected_tile_index * _SELECTED_TILE
            index_rows = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.dma_transpose(
                dst=index_rows,
                src=flat_selected[
                    batch_index : batch_index + 1,
                    selected_start : selected_start + _SELECTED_TILE,
                ],
                axes=(1, 0),
            )

            logical_blocks = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=logical_blocks,
                data=index_rows,
                op0=nl.right_shift,
                operand0=block_shift,
                engine=nisa.engine.vector,
            )
            logical_rows_in_block = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=logical_rows_in_block,
                data=index_rows,
                op0=nl.bitwise_and,
                operand0=block_row_mask,
                engine=nisa.engine.vector,
            )
            block_table_rows = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            logical_block_valid_low = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            logical_block_valid_high = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            logical_block_valid_float = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            logical_block_valid = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=logical_block_valid_low,
                data=logical_blocks,
                op0=nl.greater,
                operand0=-1,
            )
            nisa.tensor_scalar(
                dst=logical_block_valid_high,
                data=logical_blocks,
                op0=nl.less,
                operand0=logical_block_count,
            )
            nisa.tensor_tensor(
                dst=logical_block_valid_float,
                data1=logical_block_valid_low,
                data2=logical_block_valid_high,
                op=nl.multiply,
            )
            nisa.tensor_copy(dst=logical_block_valid, src=logical_block_valid_float)
            block_table_rows_plus_one = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=block_table_rows_plus_one,
                data=logical_blocks,
                op0=nl.add,
                operand0=batch_index * logical_block_count + 1,
            )
            nisa.tensor_tensor(
                dst=block_table_rows,
                data1=block_table_rows_plus_one,
                data2=logical_block_valid,
                op=nl.multiply,
            )
            nisa.tensor_scalar(
                dst=block_table_rows,
                data=block_table_rows,
                op0=nl.add,
                operand0=-1,
            )
            physical_blocks = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.memset(dst=physical_blocks, value=-1)
            nisa.dma_copy(
                dst=physical_blocks,
                src=flat_block_table.ap(
                    pattern=[[1, _SELECTED_TILE], [1, 1]],
                    vector_offset=block_table_rows,
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=nisa.oob_mode.skip,
            )
            physical_block_offsets = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=physical_block_offsets,
                data=physical_blocks,
                op0=nl.multiply,
                operand0=block_size,
            )
            physical_rows = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=physical_rows,
                data1=physical_block_offsets,
                data2=logical_rows_in_block,
                op=nl.add,
            )

            selected_cache_storage = nl.ndarray(
                (_SELECTED_TILE, _CACHE_WIDTH),
                dtype=mla_k_cache.dtype,
                buffer=nl.sbuf,
            )
            nisa.memset(dst=selected_cache_storage, value=0.0)
            nisa.dma_copy(
                dst=selected_cache_storage[0:_SELECTED_TILE, 0:_CACHE_HALF_WIDTH],
                src=flat_k_cache.ap(
                    pattern=[
                        [_CACHE_HALF_WIDTH, _SELECTED_TILE],
                        [1, _CACHE_HALF_WIDTH],
                    ],
                    vector_offset=physical_rows,
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=nisa.oob_mode.skip,
            )
            nisa.dma_copy(
                dst=selected_cache_storage[
                    0:_SELECTED_TILE, _CACHE_HALF_WIDTH:_CACHE_WIDTH
                ],
                src=flat_v_cache.ap(
                    pattern=[
                        [_CACHE_HALF_WIDTH, _SELECTED_TILE],
                        [1, _CACHE_HALF_WIDTH],
                    ],
                    vector_offset=physical_rows,
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=nisa.oob_mode.skip,
            )
            selected_cache = nl.ndarray(
                (_SELECTED_TILE, _CACHE_WIDTH),
                dtype=queries.dtype,
                buffer=nl.sbuf,
            )
            nisa.tensor_copy(dst=selected_cache, src=selected_cache_storage)

            score_psum = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.psum
            )
            for latent_tile_index in nl.affine_range(4):
                latent_start = latent_tile_index * _SELECTED_TILE
                latent_transpose = nl.ndarray(
                    (_SELECTED_TILE, _SELECTED_TILE),
                    dtype=queries.dtype,
                    buffer=nl.sbuf,
                )
                nisa.dma_transpose(
                    dst=latent_transpose,
                    src=selected_cache[
                        0:_SELECTED_TILE,
                        latent_start : latent_start + _SELECTED_TILE,
                    ],
                    axes=(1, 0),
                )
                nisa.nc_matmul(
                    dst=score_psum,
                    stationary=latent_batch[
                        0:_SELECTED_TILE,
                        latent_tile_index,
                        batch_index : batch_index + 1,
                    ],
                    moving=latent_transpose,
                    accumulate=(latent_tile_index > 0),
                )

            rope_transpose = nl.ndarray(
                (_ROPE_WIDTH, _SELECTED_TILE),
                dtype=queries.dtype,
                buffer=nl.sbuf,
            )
            nisa.dma_transpose(
                dst=rope_transpose,
                src=selected_cache[0:_SELECTED_TILE, _LATENT_WIDTH:_CACHE_WIDTH],
                axes=(1, 0),
            )
            nisa.nc_matmul(
                dst=score_psum,
                stationary=q_rope,
                moving=rope_transpose,
                accumulate=True,
            )

            scores = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=scores, src=score_psum)
            nisa.tensor_scalar(
                dst=scores, data=scores, op0=nl.multiply, operand0=scale
            )

            index_columns = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.dma_transpose(dst=index_columns, src=index_rows, axes=(1, 0))
            valid_low = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            valid_high = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            physical_block_columns = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.dma_transpose(
                dst=physical_block_columns,
                src=physical_blocks,
                axes=(1, 0),
            )
            valid_physical_low = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            valid_physical_high = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            valid = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=valid_low,
                data=index_columns,
                op0=nl.greater,
                operand0=-1,
            )
            nisa.tensor_scalar(
                dst=valid_high,
                data=index_columns,
                op0=nl.less,
                operand0=logical_key_count,
            )
            nisa.tensor_scalar(
                dst=valid_physical_low,
                data=physical_block_columns,
                op0=nl.greater,
                operand0=-1,
            )
            nisa.tensor_scalar(
                dst=valid_physical_high,
                data=physical_block_columns,
                op0=nl.less,
                operand0=physical_block_count,
            )
            nisa.tensor_tensor(
                dst=valid, data1=valid_low, data2=valid_high, op=nl.multiply
            )
            nisa.tensor_tensor(
                dst=valid,
                data1=valid,
                data2=valid_physical_low,
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=valid,
                data1=valid,
                data2=valid_physical_high,
                op=nl.multiply,
            )
            invalid_bias = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=invalid_bias,
                data=valid,
                op0=nl.multiply,
                operand0=-_MASK_FLOOR,
                op1=nl.add,
                operand1=_MASK_FLOOR,
            )
            nisa.tensor_tensor(
                dst=scores,
                data1=scores,
                data2=valid,
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=scores, data1=scores, data2=invalid_bias, op=nl.add
            )

            tile_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_max, data=scores, op=nl.maximum, axis=1)
            new_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=new_max, data1=running_max, data2=tile_max, op=nl.maximum
            )

            old_shift = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            old_scale = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=old_shift, data1=running_max, data2=new_max, op=nl.subtract
            )
            nisa.activation(dst=old_scale, data=old_shift, op=nl.exp)

            score_shift = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            probabilities = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=score_shift,
                data=scores,
                op0=nl.subtract,
                operand0=new_max,
            )
            nisa.activation(dst=probabilities, data=score_shift, op=nl.exp)
            nisa.tensor_tensor(
                dst=probabilities,
                data1=probabilities,
                data2=valid,
                op=nl.multiply,
            )
            tile_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_sum, data=probabilities, op=nl.add, axis=1)

            scaled_running_sum = nl.ndarray(
                (1, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=scaled_running_sum,
                data1=running_sum,
                data2=old_scale,
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=running_sum,
                data1=scaled_running_sum,
                data2=tile_sum,
                op=nl.add,
            )

            probability_column = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=queries.dtype, buffer=nl.sbuf
            )
            probabilities_compute = nl.ndarray(
                (1, _SELECTED_TILE), dtype=queries.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=probabilities_compute, src=probabilities)
            nisa.dma_transpose(
                dst=probability_column, src=probabilities_compute, axes=(1, 0)
            )
            weighted_latent_psum = nl.ndarray(
                (1, _LATENT_WIDTH), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(
                dst=weighted_latent_psum,
                stationary=probability_column,
                moving=selected_cache[0:_SELECTED_TILE, 0:_LATENT_WIDTH],
                accumulate=False,
            )
            weighted_latent = nl.ndarray(
                (1, _LATENT_WIDTH), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=weighted_latent, src=weighted_latent_psum)
            scaled_latent = nl.ndarray(
                (1, _LATENT_WIDTH), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=scaled_latent,
                data=latent_accumulator,
                op0=nl.multiply,
                operand0=old_scale,
            )
            nisa.tensor_tensor(
                dst=latent_accumulator,
                data1=scaled_latent,
                data2=weighted_latent,
                op=nl.add,
            )
            nisa.tensor_copy(dst=running_max, src=new_max)

        inverse_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        safe_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        normalized_latent = nl.ndarray(
            (1, _LATENT_WIDTH), dtype=queries.dtype, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=safe_sum,
            data=running_sum,
            op0=nl.maximum,
            operand0=1.0e-20,
        )
        nisa.reciprocal(dst=inverse_sum, data=safe_sum)
        nisa.tensor_scalar(
            dst=normalized_latent,
            data=latent_accumulator,
            op0=nl.multiply,
            operand0=inverse_sum,
        )
        for latent_tile_index in nl.affine_range(4):
            latent_start = latent_tile_index * _SELECTED_TILE
            normalized_tile = nl.ndarray(
                (1, _SELECTED_TILE), dtype=queries.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(
                dst=normalized_tile,
                src=normalized_latent[
                    0:1, latent_start : latent_start + _SELECTED_TILE
                ],
            )
            normalized_tile_transpose_psum = nl.ndarray(
                (_SELECTED_TILE, 1), dtype=queries.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(
                dst=normalized_tile_transpose_psum,
                data=normalized_tile,
            )
            nisa.tensor_copy(
                dst=latent_batch[
                    0:_SELECTED_TILE,
                    latent_tile_index,
                    batch_index : batch_index + 1,
                ],
                src=normalized_tile_transpose_psum,
            )

    # Batched normalized_latent @ W_v. As above, each FP8 weight tile and
    # scale are prepared once for all 32 requests. Output blocks are written
    # directly so no second batch-wide accumulation tensor is required.
    for output_block in nl.affine_range(4):
        value_row_start = max(
            _QK_NOPE_WIDTH,
            output_block * _SELECTED_TILE - row_offset,
        )
        value_row_end = min(
            _WEIGHT_WIDTH,
            (output_block + 1) * _SELECTED_TILE - row_offset,
        )
        value_size = max(0, value_row_end - value_row_start)
        if value_size > 0:
            value_output_start = value_row_start - _QK_NOPE_WIDTH
            value_block = nl.ndarray(
                (32, value_size), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.memset(dst=value_block, value=0.0)
            for latent_tile_index in nl.affine_range(4):
                latent_start = latent_tile_index * _SELECTED_TILE
                value_weight_fp8 = nl.ndarray(
                    (value_size, _SELECTED_TILE),
                    dtype=weight.dtype,
                    buffer=nl.sbuf,
                )
                nisa.dma_copy(
                    dst=value_weight_fp8,
                    src=weight[
                        value_row_start:value_row_end,
                        latent_start : latent_start + _SELECTED_TILE,
                    ],
                )
                value_weight_bf16 = nl.ndarray(
                    (value_size, _SELECTED_TILE),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                nisa.tensor_copy(dst=value_weight_bf16, src=value_weight_fp8)
                value_weight_transpose_psum = nl.ndarray(
                    (_SELECTED_TILE, value_size),
                    dtype=nl.bfloat16,
                    buffer=nl.psum,
                )
                nisa.nc_transpose(
                    dst=value_weight_transpose_psum,
                    data=value_weight_bf16,
                )
                value_weight = nl.ndarray(
                    (_SELECTED_TILE, value_size),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                nisa.tensor_copy(dst=value_weight, src=value_weight_transpose_psum)
                value_partial_psum = nl.ndarray(
                    (32, value_size), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_matmul(
                    dst=value_partial_psum,
                    stationary=latent_batch[
                        0:_SELECTED_TILE, latent_tile_index, 0:32
                    ],
                    moving=value_weight,
                    accumulate=False,
                )
                value_partial = nl.ndarray(
                    (32, value_size), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=value_partial, src=value_partial_psum)
                value_scale = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=value_scale,
                    src=TensorView(
                        weight_scale_inv[
                            output_block : output_block + 1,
                            latent_tile_index : latent_tile_index + 1,
                        ]
                    )
                    .broadcast(dim=0, size=32)
                    .get_view(),
                )
                scaled_value_partial = nl.ndarray(
                    (32, value_size), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=scaled_value_partial,
                    data=value_partial,
                    op0=nl.multiply,
                    operand0=value_scale,
                )
                nisa.tensor_tensor(
                    dst=value_block,
                    data1=value_block,
                    data2=scaled_value_partial,
                    op=nl.add,
                )
            value_result = nl.ndarray(
                (32, value_size), dtype=queries.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=value_result, src=value_block)
            nisa.dma_copy(
                dst=output[
                    0:32,
                    0:1,
                    0:1,
                    value_output_start : value_output_start + value_size,
                ],
                src=value_result.reshape((32, 1, 1, value_size)),
            )

    return output


def _should_use_b32_k512_block32_weight_reuse(
    queries: torch.Tensor,
    mla_k_cache: torch.Tensor,
    mla_v_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected_indices: torch.Tensor,
    *,
    block_size: int,
    requested: bool,
) -> bool:
    """Return true only for the one reviewed static specialization."""

    return bool(
        requested
        and queries.shape == (32, 1, 1, _QK_WIDTH)
        and mla_k_cache.ndim == 4
        and mla_k_cache.shape[1:] == (1, 32, _CACHE_HALF_WIDTH)
        and mla_v_cache.shape == mla_k_cache.shape
        and block_table.shape == (32, 16)
        and selected_indices.shape
        == (32, 1, SELECTED_LATENT_MLA_SHORT_WIDTH)
        and block_size == 32
    )


def selected_latent_mla_decode(
    queries: torch.Tensor,
    mla_k_cache: torch.Tensor,
    mla_v_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected_indices: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    block_size: int,
    row_offset: int = 0,
    use_b32_weight_reuse: bool = False,
) -> torch.Tensor:
    """Launch selected-latent MLA directly against physical paged caches."""

    validate_selected_latent_mla_decode_contract(
        queries,
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected_indices,
        weight,
        weight_scale_inv,
        block_size=block_size,
        row_offset=row_offset,
    )

    kernel = _selected_latent_mla_decode_nki
    if _should_use_b32_k512_block32_weight_reuse(
        queries,
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected_indices,
        block_size=block_size,
        requested=use_b32_weight_reuse,
    ):
        kernel = _selected_latent_mla_decode_b32_k512_block32_weight_reuse_nki

    return wrap_nki(kernel)[1](
        queries,
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected_indices,
        weight,
        weight_scale_inv,
        block_size,
        row_offset=row_offset,
    )


def validate_selected_latent_mla_decode_contract(
    queries: torch.Tensor,
    mla_k_cache: torch.Tensor,
    mla_v_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected_indices: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    block_size: int,
    row_offset: int = 0,
) -> None:
    """Fail before NKI lowering when the proven decode contract is not exact."""

    errors: list[str] = []
    if queries.ndim != 4 or queries.shape[1:] != (1, 1, _QK_WIDTH):
        errors.append("queries must have shape [batch, 1, 1, 256]")
    if queries.dtype is not torch.bfloat16:
        errors.append("queries must be BF16")
    if mla_k_cache.ndim != 4:
        errors.append("MLA K cache must be rank four")
    elif mla_k_cache.shape[1:] != (1, block_size, _CACHE_HALF_WIDTH):
        errors.append("MLA K cache must have shape [blocks, 1, block_size, 288]")
    if mla_v_cache.shape != mla_k_cache.shape:
        errors.append("MLA cache halves must have identical shapes")
    if mla_v_cache.dtype is not mla_k_cache.dtype:
        errors.append("MLA cache halves must have identical dtypes")
    if mla_k_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        errors.append("MLA cache halves must be BF16 or raw E4M3")
    if block_size not in SELECTED_LATENT_MLA_BLOCK_SIZES:
        errors.append("block_size must be 16 or 32")
    if block_table.ndim != 2 or (
        queries.ndim == 4 and block_table.shape[0] != queries.shape[0]
    ):
        errors.append("block table must have shape [batch, logical_blocks]")
    if block_table.dtype is not torch.int32:
        errors.append("block table must be int32")
    if selected_indices.ndim != 3 or (
        queries.ndim == 4 and selected_indices.shape[:2] != queries.shape[:2]
    ):
        errors.append("selected indices must have shape [batch, 1, selected]")
    elif block_table.ndim == 2:
        logical_key_count = block_table.shape[1] * block_size
        if logical_key_count in SELECTED_LATENT_MLA_SHORT_CONTEXT_BUCKETS:
            expected_selected_width = SELECTED_LATENT_MLA_SHORT_WIDTH
        elif logical_key_count in SELECTED_LATENT_MLA_LONG_CONTEXT_BUCKETS:
            expected_selected_width = SELECTED_LATENT_MLA_LONG_WIDTH
        else:
            expected_selected_width = None
            errors.append(
                f"unsupported logical key bucket {logical_key_count}"
            )
        if (
            expected_selected_width is not None
            and selected_indices.shape[2] != expected_selected_width
        ):
            errors.append(
                "selected width must be "
                f"{expected_selected_width} for logical key bucket "
                f"{logical_key_count}"
            )
    if selected_indices.dtype is not torch.int32:
        errors.append("selected indices must be int32")
    if weight.shape != (_WEIGHT_WIDTH, _LATENT_WIDTH):
        errors.append("kv_b weight must have shape [448, 512]")
    if weight.dtype is not torch.float8_e4m3fn:
        errors.append("kv_b weight must be raw E4M3")
    if weight_scale_inv.shape != (4, 4):
        errors.append("kv_b inverse scales must have shape [4, 4]")
    if weight_scale_inv.dtype is not torch.float32:
        errors.append("kv_b inverse scales must be FP32")
    if row_offset not in (0, 64):
        errors.append("kv_b row offset must be 0 or 64")

    tensors = (
        mla_k_cache,
        mla_v_cache,
        block_table,
        selected_indices,
        weight,
        weight_scale_inv,
    )
    if any(tensor.device != queries.device for tensor in tensors):
        errors.append("all selected-MLA tensors must use the query device")
    if errors:
        raise ValueError("selected-latent MLA contract violation: " + "; ".join(errors))


__all__ = [
    "SELECTED_LATENT_MLA_BLOCK_SIZES",
    "SELECTED_LATENT_MLA_LONG_CONTEXT_BUCKETS",
    "SELECTED_LATENT_MLA_LONG_WIDTH",
    "SELECTED_LATENT_MLA_SHORT_CONTEXT_BUCKETS",
    "SELECTED_LATENT_MLA_SHORT_WIDTH",
    "selected_latent_mla_decode",
    "validate_selected_latent_mla_decode_contract",
]
