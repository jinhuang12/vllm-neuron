# SPDX-License-Identifier: Apache-2.0
"""Selected-latent MLA decode spike for GLM-5.2 long contexts.

The kernel absorbs the non-rotary query into the MLA projection, gathers only
the DSA-selected latent rows, and performs online softmax.  It never exports a
``[batch, query, selected, 576]`` tensor to Torch/HBM.

This spike is deliberately narrow: one query, one TP-local head, BF16 cache,
checkpoint-native block-FP8 projection weights, and at most 2048 selected
positions. Production wiring must wait for a hardware numeric gate.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from vllm_neuron.nki.nki_hop import wrap_nki


_CACHE_WIDTH = 576
_LATENT_WIDTH = 512
_ROPE_WIDTH = 64
_QK_NOPE_WIDTH = 192
_QK_WIDTH = 256
_VALUE_WIDTH = 256
_WEIGHT_WIDTH = _QK_NOPE_WIDTH + _VALUE_WIDTH
_SELECTED_TILE = 128
_MASK_FLOOR = -9984.0


def _kernel_assert(condition: bool, error_text: str) -> None:
    assert condition, (
        "[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: " + error_text
    )


@nki.jit
def _selected_latent_mla_decode_nki(
    queries,
    latent_cache,
    selected_indices,
    weight,
    weight_scale_inv,
    row_offset=0,
):
    """Compute one-head MLA decode from selected logical latent rows.

    Dimensions:
        B: Static batch size.
        T: Logical cache length, greater than 2048 for this spike.
        K: Selected width, a multiple of 128 and at most 2048.

    Args:
        queries: ``[B, 1, 1, 256]`` BF16 query tensor in HBM.
        latent_cache: ``[B, T, 576]`` BF16 logical MLA cache in HBM.
        selected_indices: ``[B, 1, K]`` int32 logical row indices in HBM.
        weight: ``[448, 512]`` raw E4M3 rank-local kv_b projection weight in
            HBM. The first 192 rows project keys and the final 256 rows project
            values.
        weight_scale_inv: ``[4, 4]`` FP32 128x128 inverse-scale grid in HBM.
        row_offset: Rank-local first-row offset in its global 128-row FP8
            block. TP64 uses zero on even ranks and 64 on odd ranks.

    Returns:
        ``[B, 1, 1, 256]`` BF16 attention output in shared HBM.

    Notes:
        - Negative and out-of-range indices are padding and contribute zero.
        - The largest selected-latent allocation is one ``[128, 576]`` SBUF
          tile.  No selected-latent tensor is materialized in HBM.
        - Block-FP8 scales are applied to FP32 matmul partials before
          accumulation, matching ``block_fp8.py``.
        - Query/key score and online-softmax accumulation use FP32.
    """

    _kernel_assert(len(queries.shape) == 4, "queries must be rank four")
    _kernel_assert(queries.shape[1:] == (1, 1, _QK_WIDTH), "query shape mismatch")
    _kernel_assert(len(latent_cache.shape) == 3, "latent cache must be rank three")
    _kernel_assert(latent_cache.shape[0] == queries.shape[0], "batch mismatch")
    _kernel_assert(latent_cache.shape[2] == _CACHE_WIDTH, "cache width mismatch")
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
    selected_count = selected_indices.shape[2]
    _kernel_assert(selected_count <= 2048, "selected width exceeds 2048")
    _kernel_assert(selected_count % _SELECTED_TILE == 0, "selected width alignment")

    batch_size = queries.shape[0]
    key_count = latent_cache.shape[1]
    scale = _QK_WIDTH**-0.5
    output = nl.ndarray(
        (batch_size, 1, 1, _VALUE_WIDTH),
        dtype=queries.dtype,
        buffer=nl.shared_hbm,
    )

    flat_queries = queries.reshape((batch_size, _QK_WIDTH))
    flat_selected = selected_indices.reshape((batch_size, selected_count))

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

        cache_rows = latent_cache[batch_index].reshape((key_count, _CACHE_WIDTH))
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

            selected_cache = nl.ndarray(
                (_SELECTED_TILE, _CACHE_WIDTH),
                dtype=latent_cache.dtype,
                buffer=nl.sbuf,
            )
            nisa.memset(dst=selected_cache, value=0.0)
            selected_source = cache_rows.ap(
                pattern=[[_CACHE_WIDTH, _SELECTED_TILE], [1, _CACHE_WIDTH]],
                vector_offset=index_rows,
                indirect_dim=0,
            )
            nisa.dma_copy(
                dst=selected_cache,
                src=selected_source,
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=nisa.oob_mode.skip,
            )

            # Scores = absorbed_q @ latent.T + q_rope @ cached_rope.T.
            score_psum = nl.ndarray(
                (1, _SELECTED_TILE), dtype=nl.float32, buffer=nl.psum
            )
            for latent_tile_index in nl.affine_range(4):
                latent_start = latent_tile_index * _SELECTED_TILE
                latent_transpose = nl.ndarray(
                    (_SELECTED_TILE, _SELECTED_TILE),
                    dtype=latent_cache.dtype,
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
                dtype=latent_cache.dtype,
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
                operand0=key_count,
            )
            nisa.tensor_tensor(
                dst=valid, data1=valid_low, data2=valid_high, op=nl.multiply
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


def selected_latent_mla_decode(
    queries: torch.Tensor,
    latent_cache: torch.Tensor,
    selected_indices: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    row_offset: int = 0,
) -> torch.Tensor:
    """Launch the bounded selected-latent MLA decode spike."""

    return wrap_nki(_selected_latent_mla_decode_nki)[1](
        queries,
        latent_cache,
        selected_indices,
        weight,
        weight_scale_inv,
        row_offset=row_offset,
    )


__all__ = ["selected_latent_mla_decode"]
