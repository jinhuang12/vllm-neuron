# SPDX-License-Identifier: Apache-2.0
import nki
import nki.isa as nisa
import nki.language as nl

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import GroupCoordinator
from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki
from ..argsort_unstable import argsort_unstable

_TORCH_NKI_DTYPE_MAP = {
    torch.float8_e4m3fn: nl.float8_e4m3fn,
    torch.bfloat16: nl.bfloat16,
    torch.int32: nl.int32,
}

_INT32_MAX = 2**31 - 1


def permute_routed_tokens(
    hidden_input: torch.Tensor,
    expert_index: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    group: GroupCoordinator,
    is_sequence_parallel: bool = False,
) -> torch.Tensor:
    """
    Prepare tokens for all2all dispatch by permuting tokens by destination rank and
    concatenating hidden_input, per-rank affinities, and token indices. This enables hidden
    states and metadata to be dispatched in a single all2all call.

    When a token is routed to multiple experts that are on the same destination rank,
    the token appears only once in the contiguous group of rows that will be dispatched to that rank.
    This means that the final rows of the output buffer may be padded with 0s.

    Args:
        hidden_input (torch.Tensor): [T, n_input_cols] bf16 or fp8 tensor of hidden states.
        expert_index (torch.Tensor): [T, K] int32 tensor of top-K expert indices per token.
        expert_affinities_masked (torch.Tensor): [T, E] bf16 tensor of expert affinities,
            with zeros for non-routed token/expert pairs.
        group (GroupCoordinator): The distributed group coordinator. Its world_size determines
            the number of destination ranks (n_dst_ranks).
        is_sequence_parallel (bool): If True, token indices are global. The rank offset
            (rank_id * T) is added to local token indices so that each rank's tokens have
            globally unique IDs. Defaults to False.

    Returns:
        torch.Tensor: [T*K, n_output_cols] tensor where each row is
            [hidden_state, local_affinities, token_index] sorted by destination rank.
            token_index is 1-indexed (1..T) because MoE will determine routed tokens by checking for nonzero indices.
            Rows beyond the actual token count are zero-padded.

    Example:
        >>> # T=4 tokens, K=2, E=8 experts, 4 ranks (group.world_size=4), fp8 hidden
        >>> hidden_input = torch.randn(4, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        >>> expert_index = torch.tensor([[0, 1], [2, 5], [7, 4], [1, 6]], dtype=torch.int32)
        >>> affinities = torch.zeros(4, 8, dtype=torch.bfloat16)
        >>> out = permute_routed_tokens(hidden_input, expert_index, affinities, group=group)
        >>> # n_local_experts = E // world_size = 2
        >>> # Token 0 → experts 0,1 → both rank 0 (DEDUP: counts as 1 row)
        >>> # Token 1 → experts 2,5 → ranks 1,2 (2 rows)
        >>> # Token 2 → experts 7,4 → ranks 3,2 (2 rows)
        >>> # Token 3 → experts 1,6 → ranks 0,3 (2 rows)
        >>> # Total valid rows = 7, padding rows = 1 (T*K=8 total)
        >>> #
        >>> # Output rows grouped by dst rank:
        >>> #   Row 0: token 0 → rank 0
        >>> #   Row 1: token 3 → rank 0
        >>> #   Row 2: token 1 → rank 1
        >>> #   Row 3: token 1 → rank 2
        >>> #   Row 4: token 2 → rank 2
        >>> #   Row 5: token 2 → rank 3
        >>> #   Row 6: token 3 → rank 3
        >>> #   Row 7: [padding — zero-filled]
        >>> # out.shape = (T*K=8, H*2 + n_local_experts + 2) = (8, 260)
    """

    # Convert from GroupCoordinator -> size
    replica_group_size = group.world_size

    _validate_inputs(
        hidden_input, expert_index, expert_affinities_masked, replica_group_size
    )

    if _can_use_kernel(hidden_input):
        # TODO: add kernel integration
        pass

    sequence_parallel_rank_id = (
        torch.tensor([dist.get_rank()], dtype=torch.int32, device=hidden_input.device)
        if is_sequence_parallel
        else None
    )

    return _torch_impl(
        hidden_input,
        expert_index,
        expert_affinities_masked,
        replica_group_size,
        sequence_parallel_rank_id,
    )


def _validate_inputs(
    hidden_input: torch.Tensor,
    expert_index: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    replica_group_size: int,
) -> None:
    """Validate inputs for permute_routed_tokens."""
    T, K = expert_index.shape
    _, E = expert_affinities_masked.shape

    assert hidden_input.shape[0] == T, (
        f"hidden_input rows ({hidden_input.shape[0]}) must match expert_index rows ({T})"
    )
    assert expert_affinities_masked.shape[0] == T, (
        f"expert_affinities rows ({expert_affinities_masked.shape[0]}) must match T ({T})"
    )
    assert E % replica_group_size == 0, (
        f"E must be divisible by replica_group_size, got {E=}, {replica_group_size=}"
    )
    assert not (hidden_input.element_size() < 2 and hidden_input.shape[-1] % 2 != 0), (
        f"Expected dim1 of hidden_input divisible by 2 when hidden_input is fp8, got {hidden_input.shape=} {hidden_input.dtype=}"
    )


def _can_use_kernel(hidden_input: torch.Tensor) -> bool:
    """Check if the NKI kernel can be used for permute_routed_tokens."""
    if not can_run_kernel(hidden_input):
        return False

    # TODO: add kernel integration
    return False


def _torch_impl(
    hidden_input: torch.Tensor,
    expert_index: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    replica_group_size: int,
    sequence_parallel_rank_id: torch.Tensor = None,
):
    """Torch implementation of permute_routed_tokens."""

    # Step 1: Extract constants
    T, K = expert_index.shape
    _, E = expert_affinities_masked.shape
    num_experts_per_rank = E // replica_group_size

    # Step 2: Build inverse argsort array, with adjustment for de-duped token/rank pairs
    # Step 2.1: Build de-duped expert rank mapping
    expert_ranks = (expert_index // num_experts_per_rank).to(torch.int32)
    expert_ranks_deduped = expert_ranks.clone()
    for k in range(1, K):
        # Compare column k against all prior columns [0..k-1] at once
        matches = (
            expert_ranks_deduped[:, :k] == expert_ranks_deduped[:, k : k + 1]
        ).any(dim=1)
        expert_ranks_deduped[:, k] = torch.where(
            matches, -1, expert_ranks_deduped[:, k]
        )
    dedupe_count = (expert_ranks_deduped == -1).sum()

    # Step 2.2: Compute inverse argsort
    token_argsort = argsort_unstable(expert_ranks_deduped.flatten().to(torch.int32))
    token_inv_argsort = torch.zeros_like(token_argsort)
    token_inv_argsort[token_argsort] = torch.arange(
        T * K, dtype=torch.int32, device=expert_index.device
    )

    # Step 2.3: Adjust inverse argsort so that all de-duped tokens have idx 0
    token_inv_argsort_adjusted = (
        (token_inv_argsort - dedupe_count + 1).clamp(0, _INT32_MAX).to(torch.int32)
    )

    # Step 3: Concatenate [hidden | affinities | token idx], with bitcast to hidden.dtype
    # Step 3.1: Broadcast [T, H] -> [T*K, H]
    hidden_input_bc_K = hidden_input.unsqueeze(1).expand(-1, K, -1).reshape(T * K, -1)

    # Step 3.2: Gather E/EP affinities per T, K pair
    # Build gather indices: [T, K, n_local]
    offsets = expert_ranks * num_experts_per_rank  # [T, K]
    local_idx = torch.arange(
        num_experts_per_rank, dtype=torch.int32, device=expert_ranks.device
    )
    gather_idx = offsets.unsqueeze(-1) + local_idx  # [T, K, n_local]

    # Expand affinities to [T, K, E] then gather along last dim
    expert_affinities_masked_sliced = (
        expert_affinities_masked.unsqueeze(1)
        .expand(-1, K, -1)
        .gather(2, gather_idx.to(torch.int32))
        .reshape(T * K, num_experts_per_rank)
    )

    # Step 3.3: Build 1-indexed token indices [T*K, 1], with optional adjustment for SP rank.
    token_indices = (
        torch.arange(1, T + 1, dtype=torch.int32, device=expert_index.device)
        .repeat_interleave(K)
        .reshape(T * K, 1)
    )
    if sequence_parallel_rank_id is not None:
        token_indices.add_(sequence_parallel_rank_id * T)

    # Step 3.4: Concatenate with bitcast to hidden_input.dtype
    data_concat = torch.concat(
        [
            hidden_input_bc_K,
            _bitcast(expert_affinities_masked_sliced, hidden_input_bc_K.dtype),
            _bitcast(token_indices, hidden_input_bc_K.dtype),
        ],
        dim=1,
    )

    # Step 4: Group tokens by destination rank, with de-dupe
    # Bitcast to a same-width integer dtype, which is supported on CPU and doesn't canonicalize NaNs.
    if str(expert_index.device) == "cpu":
        int_scatter_dtype = (
            torch.int8 if data_concat.element_size() == 1 else torch.int16
        )
        data_concat = _bitcast(data_concat, int_scatter_dtype)

    # Scatter tokens into output buffer using adjusted inv argsort array. De-dupes are scattered into row 0
    # NOTE: padded rows have index -2 for better debuggability. -2 index post dispatch = metadata was incorrect.
    n_idx_cols = 4 // data_concat.element_size()
    n_data_cols = data_concat.shape[-1] - n_idx_cols
    n_rows = T * K + 1
    zeros_part = torch.zeros(
        (n_rows, n_data_cols),
        dtype=data_concat.dtype,
        device=expert_index.device,
    )
    neg_two_int32 = torch.full(
        (n_rows, 1), -2, dtype=torch.int32, device=expert_index.device
    )
    neg_two_native = _bitcast(neg_two_int32, data_concat.dtype)
    output_permuted = torch.concat([zeros_part, neg_two_native], dim=1)
    output_permuted.scatter_(
        0, token_inv_argsort_adjusted.unsqueeze(1).expand_as(data_concat), data_concat
    )

    # CPU mode does not support fp8 scatter_; convert back to hidden.dtype
    if str(expert_index.device) == "cpu":
        output_permuted = _bitcast(output_permuted, hidden_input_bc_K.dtype)

    # Discard row 0, which contains garbage/de-duped tokens
    return output_permuted[1:, :]


# FIXME: everything below this line is a hack, remove when FX->HLO natively supports bitcasting
def _bitcast(data, dtype):
    # Same-dtype bitcast is a no-op; skip it
    if data.dtype == dtype:
        return data
    elif str(data.device) != "cpu":
        wrapped = wrap_nki(_bitcast_nki)
        nki_dtype = _TORCH_NKI_DTYPE_MAP[dtype]
        return wrapped[2](data, nki_dtype)
    else:
        return data.view(dtype)


@nki.jit
def _bitcast_nki(data, nki_dtype):
    data = data.view(nki_dtype)
    data_new_dtype = nl.ndarray(data.shape, nki_dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(data_new_dtype, data)

    return data_new_dtype
