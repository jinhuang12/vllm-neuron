# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

PLACEHOLDER_TOKEN_ID = -1
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 128


@torch.no_grad()
def rejection_sampler(
    spec_decode_metadata: SpecDecodeMetadata,
    target_token_ids: torch.Tensor,  # [num_target_tokens] - sampled token IDs for all positions including bonus
) -> torch.Tensor:
    """
    Perform greedy rejection sampling by comparing draft and target token IDs.

    On-device rejection sampler for speculative decoding. Performs deterministic
    rejection sampling by comparing draft tokens with target tokens. Accepts tokens
    sequentially until the first mismatch.

    The sampling step is performed externally by the Sampler module.
    This function only handles the rejection logic.

    Args:
        spec_decode_metadata: Object containing:
            - draft_token_ids: Flattened draft tokens [num_draft_tokens]
            - num_draft_tokens: Actual number of draft tokens per request (list of ints)
            - max_spec_len: Maximum number of speculative draft tokens per request
            - cu_num_draft_tokens: Cumulative number of draft tokens [batch_size]
        target_token_ids: Sampled token IDs from all positions [num_target_tokens]
            where num_target_tokens = sum(num_draft_tokens) + batch_size
            Positions are interleaved: [drafts_0, bonus_0, drafts_1, bonus_1, ...]

    Returns:
        ``[batch_size, max_spec_len+1]`` int32 with ``-1`` for rejected
        positions. The last accepted token per request (used by the
        async-spec-decode pipeline) is extracted by the
        ``@async_speculative_decoding`` decorator's epilogue, not here,
        to keep this kernel free of pipeline-specific concerns.

    Example:
        >>> # spec_decode_metadata contains draft_token_ids, cu_num_draft_tokens, etc.
        >>> # target_token_ids: [num_target_tokens] sampled tokens including bonus positions
        >>> result = rejection_sampler(spec_decode_metadata, target_token_ids)
        >>> # result[0] might be [5, 10, 15, 99] if all drafts accepted + bonus
        >>> # or [5, 11, -1, -1] if second position rejected
    """

    assert spec_decode_metadata.max_spec_len <= MAX_SPEC_LEN

    device = target_token_ids.device
    max_spec_len = spec_decode_metadata.max_spec_len
    draft_ids = spec_decode_metadata.draft_token_ids
    cu = spec_decode_metadata.cu_num_draft_tokens
    batch_size = cu.shape[0]

    # Extract bonus tokens from interleaved structure: [drafts_0, bonus_0, drafts_1, bonus_1, ...]
    batch_idx = torch.arange(batch_size, device=device)
    bonus_tokens = target_token_ids[cu + batch_idx]  # [batch_size]

    # Setup 2D indexing: [batch_size, max_spec_len]
    pos = torch.arange(max_spec_len, device=device).unsqueeze(0)  # [1, max_spec_len]
    num_drafts = torch.diff(
        cu, prepend=torch.zeros(1, device=device, dtype=cu.dtype)
    )  # [batch_size]
    valid = pos < num_drafts.unsqueeze(1)  # [batch_size, max_spec_len]

    # Convert flattened draft tokens to 2D
    cu_start = torch.cat(
        [torch.zeros(1, device=device, dtype=cu.dtype), cu[:-1]]
    )  # [batch_size]
    draft_idx = (cu_start.unsqueeze(1) + pos).clamp(max=draft_ids.shape[0] - 1)
    drafts_2d = torch.where(valid, draft_ids[draft_idx], PLACEHOLDER_TOKEN_ID)

    # Convert flattened target tokens to 2D
    target_idx = (cu_start.unsqueeze(1) + batch_idx.unsqueeze(1) + pos).clamp(
        max=target_token_ids.shape[0] - 1
    )
    targets_2d = torch.where(valid, target_token_ids[target_idx], PLACEHOLDER_TOKEN_ID)

    # Sequential acceptance: accept until first mismatch
    matches = (drafts_2d == targets_2d) & valid
    # Find first mismatch position using argmax on inverted matches
    # For rows with all matches, argmax will return 0, so we need special handling
    mismatches = ~matches  # [batch_size, max_spec_len]
    # Use argmax to find first True (first mismatch)
    # If no mismatch, argmax returns 0, so we use a large sentinel at the end
    mismatches_with_sentinel = torch.cat(
        [mismatches, torch.ones(cu.shape[0], 1, device=device, dtype=torch.bool)], dim=1
    )  # [batch_size, max_spec_len + 1]
    first_mismatch_pos = mismatches_with_sentinel.float().argmax(dim=1)  # [batch_size]
    # Create accept mask: position <= first_mismatch_pos
    # We accept up to and INCLUDING the first mismatch (corrected target token)
    pos_indices = torch.arange(max_spec_len, device=device).unsqueeze(
        0
    )  # [1, max_spec_len]
    accept = (pos_indices <= first_mismatch_pos.unsqueeze(1)) & valid
    output = torch.where(accept, targets_2d, PLACEHOLDER_TOKEN_ID)

    # Add bonus token if all drafts accepted
    all_accepted = (matches & valid).all(dim=1)
    result = torch.cat(
        [
            output,
            torch.full(
                (batch_size, 1), PLACEHOLDER_TOKEN_ID, device=device, dtype=torch.int32
            ),
        ],
        dim=1,
    )
    result[batch_idx, num_drafts.long()] = torch.where(
        all_accepted, bonus_tokens, PLACEHOLDER_TOKEN_ID
    )

    return result
