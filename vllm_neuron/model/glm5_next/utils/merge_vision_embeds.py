# SPDX-License-Identifier: Apache-2.0
"""Scatter GLM-5.3-Flash vision embeddings into hidden_states for prefill.

Templated on ``vllm_neuron/model/qwen3_vl/utils/merge_vision_embeds.py``, which
is this fork's landed merge for a block-packed vision tower. Two differences,
both forced by what GLM-5.3-Flash's tower actually emits:

* **No deepstack.** Qwen3-VL's tower concatenates one merged tensor per
  ``deepstack_visual_indexes`` entry onto the main embedding, so its cache rows
  are ``fat_dim = out_hidden_size * (1 + num_deepstack)`` wide and its merge
  splits them back apart. ``Glm5NextVisionConfig`` declares no such field and
  ``Glm5NextVisionModel.forward`` (``modeling_glm5_next.py:1820-1824``) returns
  one ``pooler_output`` of width ``out_hidden_size``, so a GLM cache row is
  exactly as wide as the decoder's hidden state.
* **A width mismatch raises instead of slicing.** The template writes
  ``flat_embeds[:, :visual_dim]``, which is correct when a fat row is expected
  and silent when it is not. Here a row that is not exactly ``hidden_size`` wide
  means the tower and the decoder disagree, so this module refuses rather than
  truncating a disagreement into plausible-looking numbers.

The sequence-parallel coordinate remap and the dummy-row scatter are kept
verbatim in intent from the template: both stay branch-free so the prefill graph
traces under ``torch.compile``.
"""

from __future__ import annotations

import torch


class Glm5NextVisionMergeError(ValueError):
    """Raised when the merge inputs cannot describe a GLM-5.3-Flash prefill."""


def global_to_local_positions(
    positions: torch.Tensor, local_len: int, rank: int
) -> torch.Tensor:
    """Remap global batch positions to a sequence-parallel rank's local view.

    A position outside this rank's shard ``[rank*local_len, (rank+1)*local_len)``
    becomes the sentinel ``local_len``, which :func:`scatter_with_dummy_row`
    sends to a row it then discards.

    Args:
        positions: any shape, integer batch positions in global coordinates.
        local_len: this rank's shard length in tokens.
        rank: this rank's sequence-parallel index.

    Returns:
        Same shape as ``positions``, in local coordinates or the sentinel.
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
    """Scatter ``values`` into ``target`` at ``positions``, sentinels discarded.

    A position equal to ``target.shape[0]`` addresses an appended zero row that
    is sliced off on return. That is what keeps the scatter branch-free: no
    ``torch.compile`` graph break on a data-dependent mask.

    Args:
        target: ``[local_len, width]``.
        positions: ``[num_values]``, local coordinates or the sentinel.
        values: ``[num_values, width]``.

    Returns:
        ``[local_len, width]`` with ``values`` written in.
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
) -> torch.Tensor:
    """Scatter vision embeddings from encoder-cache blocks into hidden_states.

    Args:
        hidden_states: ``[local_len, hidden_size]`` from ``embed_tokens``,
            already sequence-parallel sharded.
        vision_embedding_blocks: cache block views, each
            ``[block_size, hidden_size]``. Zero-copy views on device; any
            trailing padding block is addressed only by sentinel positions.
        vision_positions: ``[num_blocks, block_size]`` global batch positions;
            the sentinel for a don't-care slot is any value outside this rank's
            shard, which the remap turns into the dummy row.
        rank: sequence-parallel rank of ``hidden_states``.

    Returns:
        ``[local_len, hidden_size]`` with vision embeddings in place.

    Raises:
        Glm5NextVisionMergeError: no blocks were given, or a block's width is
            not ``hidden_states``' width.
    """
    if not vision_embedding_blocks:
        raise Glm5NextVisionMergeError(
            "merge_vision_embeddings needs at least one cache block view; got "
            "an empty tuple. A request with no vision items must not reach this "
            "merge at all -- the prefill graph skips it."
        )

    gathered = torch.stack(vision_embedding_blocks)
    flat_embeds = gathered.reshape(-1, gathered.shape[-1])
    positions_flat = vision_positions.reshape(-1)

    hidden_size = hidden_states.shape[-1]
    embed_width = flat_embeds.shape[-1]
    if embed_width != hidden_size:
        raise Glm5NextVisionMergeError(
            f"cache row width {embed_width} does not equal hidden_states width "
            f"{hidden_size}. GLM-5.3-Flash's tower emits one pooler_output of "
            f"width out_hidden_size and declares no deepstack, so these two "
            f"widths are the same number by construction; a difference means "
            f"the tower and the decoder disagree, and slicing it away here "
            f"would hide that."
        )
    if positions_flat.shape[0] != flat_embeds.shape[0]:
        raise Glm5NextVisionMergeError(
            f"{positions_flat.shape[0]} positions for {flat_embeds.shape[0]} "
            f"embedding rows; every cached row needs exactly one position slot."
        )

    local_positions = global_to_local_positions(
        positions_flat, hidden_states.shape[0], rank
    )
    return scatter_with_dummy_row(hidden_states, local_positions, flat_embeds)
