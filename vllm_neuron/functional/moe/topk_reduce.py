# SPDX-License-Identifier: Apache-2.0
"""topk_reduce functional API."""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

import torch
import torch.distributed as dist

from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

_P_MAX = 128
_K_MAX = 8
_TK_PADDED_MIN = 1
_TK_PADDED_MAX = 16 * 1024
_N_16BIT_ELEM_PER_INT32 = 2
_SUPPORTED_INPUT_DTYPES = [nl.bfloat16, nl.float16]

_INT32_MAX = 2**31 - 1


def topk_reduce(
    input: torch.Tensor, T: int, K: int, is_sequence_parallel: bool = False
) -> torch.Tensor:
    """Compute sparse MoE Top-K reduction across all2all collective output buffer.

    Gathers scattered rows by packed global token index and reduces along the K dimension.
    Each token has between 1 and K rows scattered at arbitrary positions in the input, with remaining
    rows being padding.

    Token indices are 1-indexed. When is_sequence_parallel=False, indices are 1..T.
    When is_sequence_parallel=True, indices are global: rank r owns (r*T+1)..(r*T+T),
    and the offset rank_id*T is subtracted before indexing into the output.

    Padded rows must have index -1.

    Shapes: input [TK_padded, H+2] → output [T, H].
        TK_padded >= 8, H divisible by 2, T <= 128, K <= 8.

    Args:
        input (torch.Tensor): [TK_padded, H + 2] bf16/fp16. Sparse buffer with
            up to T*K routed rows. Final 2 bf16 columns encode a packed int32
            global token index (1-indexed, -1 for padding).
        T (int): Number of output tokens per rank (1 to 128). When
            is_sequence_parallel=True, T represents the global number of tokens / world_size.
        K (int): Max routed experts per token (1 to 8).
        is_sequence_parallel (bool): If True, token indices are expected to represent
            global token indices. Using local token indices with is_sequence_parallel=True,
            or using global token indices with is_sequence_parallel=False, will result in
            out of bounds at runtime. Defaults to False.

    Returns:
        torch.Tensor: [T, H] bf16/fp16. out[t] = sum of all rows with index
            (rank_id*T + t + 1) in SP mode, or (t + 1) in non-SP mode.

    Example:
        >>> # Non-SP, dense: T=2, K=2, H=4, input shape [4, 6]
        >>> # indices [1, 1, 2, 2] → out[0] = row0 + row1, out[1] = row2 + row3
        >>> out = topk_reduce(input, T=2, K=2)  # shape [2, 4]

        >>> # Non-SP, sparse: T=2, K=2, H=4, input shape [8, 6]
        >>> # indices [1, -1, 1, 2, -1, -1, 2, -1] → out[0] = row0 + row2, out[1] = row3 + row6
        >>> out = topk_reduce(input, T=2, K=2)  # shape [2, 4]

        >>> # SP: rank 1, T=8, K=2, H=4, input shape [16, 6], world_size=4
        >>> # global indices [9, 9, 10, 10, ..., 16, 16] (rank 1 owns 9..16)
        >>> # offset = rank * T = 1 * 8 = 8, local indices after subtract: [1, 1, 2, 2, ..., 8, 8]
        >>> out = topk_reduce(input, T=8, K=2, is_sequence_parallel=True)  # shape [8, 4]
    """
    _validate_inputs(input, T, K)
    token_base_index = dist.get_rank() * T + 1 if is_sequence_parallel else 1

    if _can_use_kernel(input, T, K):
        wrapped = wrap_nki(_topk_reduce_nki)
        return wrapped[2](input, T, K, token_base_index)
    else:
        return _cpu_topk_reduce(input, T, K, token_base_index)


def _validate_inputs(input: torch.Tensor, T: int, K: int) -> None:
    """Validate topk_reduce arguments."""
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T=}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K=}")

    # TODO: take away this check when torch impl works on HW
    H = input.shape[1] - 2
    if not _can_use_kernel(input, T, K) and str(input.device) != "cpu":
        raise ValueError(
            f"Expected 1<=T<{_P_MAX}, K<={_K_MAX} and H divisible by 2 for topk_reduce execution on hardware, got {T=} {K=} {H=}"
        )


def _can_use_kernel(input: torch.Tensor, T: int, K: int) -> bool:
    """Check if the NKI topk_reduce kernel can be used.

    Kernel constraints:
        - Device must support NKI kernels (can_run_kernel)
        - 1 < T <= 128 (pmax)
        - K <= 8
        - H (input.shape[1] - 2) must be even (divisible by LNC=2)
    """
    if not can_run_kernel(input):
        return False

    H = input.shape[1] - 2

    if T > _P_MAX or T < 1:
        return False
    if K > _K_MAX:
        return False
    if H % 2 != 0:
        return False

    return True


def _cpu_topk_reduce(
    input: torch.Tensor, T: int, K: int, token_base_index: int = 1
) -> torch.Tensor:
    """CPU-only reference implementation using scatter_add."""

    H = input.shape[1] - 2
    indices = input.view(torch.int32)[:, -1]  # (N,) packed token indices

    # Map global 1-indexed token indices to local 1-indexed
    indices = indices - (token_base_index - 1)
    # Shift indices: valid tokens are 1..T → map to rows 1..T; <=0 → maps to 0
    # We accumulate into row 0 as a garbage bin, then discard it.
    out = torch.zeros(T + 1, H, dtype=input.dtype)
    bucket = indices.clamp(0, _INT32_MAX).long()
    out.scatter_add_(0, bucket.unsqueeze(1).expand(-1, H), input[:, :H])
    return out[1:]  # discard row 0 (garbage from -1 and 0 indices)


# TODO: upstream into nkilib when kernel is finalized/stable
@nki.jit
def _topk_reduce_nki(
    input: nl.ndarray,
    T: int,
    K: int,
    token_base_index: int = 1,
):
    """
    Compute MoE Top-K reduction across sparse all_to_all_v() collective output buffer.

    Gathers scattered rows by packed global token index and reduces along
    the K dimension. Supports LNC sharding on the H dimension.

    Token indices are 1-indexed and padded rows must have index -1. The kernel
    searches for indices [token_base_index .. token_base_index+T-1]; with the
    default token_base_index=1 this is the local range [1..T], and rank_id*T+1
    selects the global range [rank_id*T+1 .. rank_id*T+T] for sequence parallel.

    Dimensions:
        TK_padded: n_src_ranks * T, padded input row count
        H: Hidden dimension size (must be divisible by LNC)
        T: Total number of input tokens (up to 128)
        K: Number of routed experts per token (up to 8)

    Args:
        input (nl.ndarray): [TK_padded, H + 2]@HBM, bf16/fp16. Sparse input buffer containing T*K
            scattered outputs. Global token index is packed as int32 in the final 2x
            columns of each row (1-indexed, -1 for padding).
        T (int): Total number of input tokens.
        K (int): Number of routed experts per token.
        token_base_index (int): First token index to search for (default: 1).
            For sequence parallel mode, use rank_id * T + 1 so the kernel
            searches for global indices [rank_id*T+1 .. rank_id*T+T].

    Returns:
        output_hbm (nl.ndarray): [T, H]@HBM, bf16/fp16. Ordered and reduced output.
            out[t] = sum of all rows with index (token_base_index + t).

    Pseudocode:
        global_token_indices = extract_int32_index(input[:, H:])
        for token_idx in range(T):
            matching_rows = find_rows_where(global_token_indices == token_base_index + token_idx)
            output[token_idx] = sum(input[matching_rows, :H])
    """

    # Shapes, LNC sharding strategy
    _P_MAX = nl.tile_size.pmax
    TK_padded, H_padded = input.shape
    H = H_padded - _N_16BIT_ELEM_PER_INT32
    _, n_prgs, prg_id = get_verified_program_sharding_info("topk_reduce", (0, 1))
    H_local = H // n_prgs
    H_local_slice = nl.ds(H_local * prg_id, H_local)

    # Validation
    kernel_assert(
        input.dtype in _SUPPORTED_INPUT_DTYPES,
        f"Expected input.dtype in {_SUPPORTED_INPUT_DTYPES}, got {input.dtype=}",
    )
    kernel_assert(1 <= T <= _P_MAX, f"T must be in [1, {_P_MAX}], got {T=}")
    kernel_assert(K <= _K_MAX, f"K must be <= {_K_MAX}")
    kernel_assert(H % n_prgs == 0, f"Expected H divisible by LNC, got {H=} {n_prgs=}")
    kernel_assert(
        _TK_PADDED_MIN <= TK_padded <= _TK_PADDED_MAX,
        f"Expected input.shape[0] between {_TK_PADDED_MIN} and {_TK_PADDED_MAX}, got {input.shape=}",
    )

    # CCE DMA with vector offset requires >1 partitions, pad to 2 when P=1
    T_compute = max(T, 2)

    # Allocations
    reduced_sb = nl.ndarray((T_compute, H_local), dtype=input.dtype, buffer=nl.sbuf)
    global_token_indices_sb = nl.ndarray(
        (T_compute, TK_padded), dtype=nl.int32, buffer=nl.sbuf
    )
    output_hbm = nl.ndarray((T, H), dtype=input.dtype, buffer=nl.shared_hbm)

    # DMA transpose indices [TK_padded, 1] -> [1, TK_padded]
    nisa.dma_transpose(
        src=input.ap(
            pattern=[
                [H_padded // _N_16BIT_ELEM_PER_INT32, TK_padded],
                [1, 1],
                [1, 1],
                [1, 1],
            ],
            offset=H // _N_16BIT_ELEM_PER_INT32,
            dtype=nl.int32,
        ),
        dst=global_token_indices_sb.ap(
            pattern=[[TK_padded, 1], [1, 1], [1, 1], [1, TK_padded]],
            offset=0,
        ),
    )

    # Broadcast [1, TK_padded] -> [T, TK_padded]
    # FIXME: (1) Move broadcast to DMA engines or PE (2) LNC shard on tokens when T>32
    stream_shuffle_broadcast(global_token_indices_sb, global_token_indices_sb)

    # Find indices [T_compute, K]
    # For each token, there may be between 1 and K corresponding rows in the input.
    # nc_find_index8 requires free dim >= 8, so pad if TK_padded < 8
    _NC_FIND_INDEX_MIN_FREE = 8
    if TK_padded < _NC_FIND_INDEX_MIN_FREE:
        find_index_data_sb = nl.ndarray(
            (T_compute, _NC_FIND_INDEX_MIN_FREE), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.memset(find_index_data_sb, 0)  # 0 won't match any 1-indexed token
        nisa.tensor_copy(find_index_data_sb[:, :TK_padded], global_token_indices_sb)
    else:
        find_index_data_sb = global_token_indices_sb

    arange_token_indices_T = nl.ndarray(
        (T_compute, _K_MAX), dtype=nl.uint32, buffer=nl.sbuf
    )
    gather_token_indices = nl.ndarray(
        (T_compute, _K_MAX), dtype=nl.uint32, buffer=nl.sbuf
    )
    nisa.memset(gather_token_indices, -1)

    # Generate search values [token_base_index .. token_base_index+T-1]
    nisa.iota(
        pattern=[[0, _K_MAX]],
        offset=token_base_index,
        channel_multiplier=1,
        dst=arange_token_indices_T,
    )
    nisa.nc_find_index8(
        data=find_index_data_sb,
        vals=arange_token_indices_T,
        dst=gather_token_indices,
    )

    # Use DMA + rmw add to reduce over topK
    # NOTE: first DMA uses skipping when using padding, since padded partitions aren't reduced
    k0_oob_mode = oob_mode.skip if T_compute > T else oob_mode.error
    for k_idx in range(K):
        src_access = input.ap(
            pattern=[[H, T_compute], [1, H_local]],
            offset=H_local * prg_id,
            vector_offset=gather_token_indices.ap(
                pattern=[[_K_MAX, T_compute], [1, 1]],
                offset=k_idx,
            ),
            indirect_dim=0,
        )

        # If src does not contain at least one row for each token, we throw an OOB error.
        if k_idx == 0:
            nisa.dma_copy(
                dst=reduced_sb[:, :],
                src=src_access,
                oob_mode=k0_oob_mode,
            )

        # Tokens routed to experts on the same EP rank will have fewer than K rows, so we DMA skip for k_idx=1...K-1.
        # Ex: K=2, E=8, EP=4, token 0 is routed to experts {0, 1} -> 1 row for token 0 in input.
        else:
            nisa.dma_compute(
                dst=reduced_sb[:, :],
                srcs=[src_access, reduced_sb[:, :]],
                reduce_op=nl.add,
                unique_indices=True,
                oob_mode=oob_mode.skip,
            )

    # Save reduced output — each core writes its H shard. Only the first T partitions
    # contain real tokens; padded partitions (when T<T_compute) are discarded.
    nisa.dma_copy(output_hbm[:, H_local_slice], reduced_sb[:T, :])

    return output_hbm
