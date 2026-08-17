# SPDX-License-Identifier: Apache-2.0
"""Scatter vision embeddings into hidden_states for Qwen3-VL prefill.

Handles sequence-parallel (SP) coordinate remapping and deepstack
construction. Used by the prefill graph to merge on-device encoder
cache block views into the token embedding sequence.
"""

from __future__ import annotations

import torch


def global_to_local_positions(
    positions: torch.Tensor, local_len: int, rank: int
) -> torch.Tensor:
    """Remap global batch positions to an SP rank's local coordinates.

    Positions outside this rank's shard [rank*local_len, (rank+1)*local_len)
    become the sentinel value (local_len), which writes to a dummy row
    during index_put_.
    """
    local_start = rank * local_len
    local_positions = positions - local_start
    out_of_range = (local_positions < 0) | (local_positions >= local_len)
    return torch.where(out_of_range, local_len, local_positions)


def scatter_with_dummy_row(
    target: torch.Tensor,
    positions: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Scatter values into target at positions using a dummy row for sentinels.

    Positions equal to target.shape[0] write to an appended dummy row
    that is discarded after the scatter. This avoids branching on
    sentinel values inside torch.compile-traced graphs.
    """
    dummy = torch.zeros(1, target.shape[-1], dtype=target.dtype, device=target.device)
    with_dummy = torch.cat([target, dummy], dim=0)
    with_dummy.index_put_((positions,), values.to(target.dtype))
    return with_dummy[: target.shape[0]]


def merge_vision_embeddings(
    hidden_states: torch.Tensor,
    vision_embedding_blocks: tuple[torch.Tensor, ...],
    vision_positions: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scatter vision embeddings from cache blocks into hidden_states.

    Handles SP coordinate remapping and deepstack construction.
    Both main and deepstack embeddings use local positions.

    Args:
        hidden_states: [local_len, visual_dim] from embed_tokens (SP-sharded).
        vision_embedding_blocks: tuple of [block_size, fat_dim] cache views.
        vision_positions: [max_num_vision_blocks, block_size] global positions.
        rank: SP rank for coordinate remapping.

    Returns:
        (hidden_states, deepstack_vision_embeds) where:
        - hidden_states: [local_len, visual_dim] with main embeds scattered in
        - deepstack_vision_embeds: [N, local_len, visual_dim] or None
    """
    gathered = torch.stack(vision_embedding_blocks)
    flat_embeds = gathered.reshape(-1, gathered.shape[-1])
    positions_flat = vision_positions.reshape(-1)

    visual_dim = hidden_states.shape[-1]
    local_len = hidden_states.shape[0]
    local_positions = global_to_local_positions(positions_flat, local_len, rank)

    # Scatter main vision embeddings
    main_embeds = flat_embeds[:, :visual_dim]
    hidden_states = scatter_with_dummy_row(hidden_states, local_positions, main_embeds)

    # Build deepstack in local coordinates
    deepstack_vision_embeds = None
    fat_dim = flat_embeds.shape[-1]
    if fat_dim > visual_dim:
        N_ds = (fat_dim - visual_dim) // visual_dim
        ds_embeds = flat_embeds[:, visual_dim:].reshape(-1, N_ds, visual_dim)
        deepstack_vision_embeds = torch.zeros(
            N_ds,
            local_len,
            visual_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for n in range(N_ds):
            deepstack_vision_embeds[n] = scatter_with_dummy_row(
                deepstack_vision_embeds[n], local_positions, ds_embeds[:, n, :]
            )

    return hidden_states, deepstack_vision_embeds
