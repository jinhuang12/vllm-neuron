# SPDX-License-Identifier: Apache-2.0
"""CPU-side 2-D rotary position embeddings for the GLM-5.3-Flash vision tower.

Every patch in the tower carries a position on two axes, its row and its column inside its own
image, and the rope rotates a head against both. This module turns a patch grid into the
``(cos, sin)`` pair the tower's attention applies, on CPU, before the compiled model runs.

WHAT THE TWO AXES DO TO THE WIDTHS. The table holds ``head_dim // 4`` frequencies. Each patch
looks them up twice, once for its row and once for its column, which gives ``head_dim // 2``
columns; the pair is then repeated to ``head_dim`` so that the two halves of a head see the
same angle. The whole head therefore rotates -- there is no pass-through slice. This is why the
references ask for half as many frequencies as a 1-D rope would
(``partial_rotary_factor: 0.5`` on the vLLM side, ``head_dim // 2`` on the HF side): the second
axis, not a shorter rotation, is what consumes the other half.

REFERENCES, both read at a pin. HF transformers 5.16.1
``Glm5NextVisionModel.forward`` -> ``get_vision_position_ids`` +
``Glm5NextVisionRotaryEmbedding`` + ``emb = cat(rotary, rotary)`` + ``(emb.cos(), emb.sin())``;
vLLM ``Glm5NextVisionModel.rot_pos_emb`` at commit ``878631b6``, which packs the same numbers
half as wide and lets its apply function broadcast them. This module returns the HF packing,
because the fork's own vision attention applies ``q * cos + rotate_half(q) * sin``.

The grid itself is never computed here. It arrives as ``grid_thw`` from the multimodal bridge,
which reads it off the transformers processor.
"""

from __future__ import annotations

import torch

#: The rope base both references use for this checkpoint. Its vision config declares no other.
DEFAULT_VISION_ROPE_THETA = 10000.0


class Glm5NextVisionRopeError(ValueError):
    """A grid or a head width this rope cannot lay out.

    Raised instead of letting a reshape fail deep inside, because the two ways to get here --
    a head width that does not divide into two axes, and a grid whose sides are not whole
    merge blocks -- both name a config mistake the caller can fix.
    """


def compute_vision_rotary_pos_emb(
    grid_thw: torch.Tensor,
    head_dim: int,
    spatial_merge_size: int,
    theta: float = DEFAULT_VISION_ROPE_THETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(cos, sin)`` pair for every patch of every item in ``grid_thw``.

    Args:
        grid_thw: ``[num_items, 3]`` -- (temporal, height, width) patch counts per item, as the
            multimodal bridge reports them.
        head_dim: Attention head width. Must be a multiple of 4, since two axes each take half
            of half of it.
        spatial_merge_size: Merge block side. Positions are laid out block-major over
            ``merge_size x merge_size`` blocks, so that the merger later sees each block's
            patches next to each other.
        theta: Rope base.

    Returns:
        ``(cos, sin)``, each ``[total_patches, head_dim]`` float32 on CPU, in the order the
        patch rows arrive.
    """
    if head_dim % 4:
        raise Glm5NextVisionRopeError(
            f"head_dim {head_dim} is not a multiple of 4. A 2-D rope splits the head in two and "
            "gives each axis half of that, so a head that is not a multiple of 4 cannot be "
            "filled exactly."
        )

    position_ids = vision_position_ids(grid_thw, spatial_merge_size)

    half_dim = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim)
    )
    # One row per position value either axis can take, so an index can never fall off the end.
    num_positions = int(position_ids.max().item()) + 1
    freq_table = torch.outer(
        torch.arange(num_positions, dtype=torch.float32), inv_freq
    )  # [num_positions, head_dim // 4]

    rotary = freq_table[position_ids].flatten(1)  # [total_patches, head_dim // 2]
    emb = torch.cat((rotary, rotary), dim=-1)  # [total_patches, head_dim]
    return emb.cos(), emb.sin()


def vision_position_ids(
    grid_thw: torch.Tensor, spatial_merge_size: int
) -> torch.Tensor:
    """Return the ``(row, column)`` index of every patch, block-major over merge blocks.

    Block-major means the four patches of one 2x2 merge block are adjacent in the sequence,
    which is what lets the merger regroup by a reshape later. A video repeats its spatial
    layout once per temporal element.

    Args:
        grid_thw: ``[num_items, 3]`` -- (temporal, height, width) patch counts per item.
        spatial_merge_size: Merge block side.

    Returns:
        ``[total_patches, 2]`` int64 on CPU: column 0 is the row index, column 1 the column
        index, both counted in patches within their own item.
    """
    if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
        raise Glm5NextVisionRopeError(
            f"grid_thw has shape {tuple(grid_thw.shape)}; it must be [num_items, 3] carrying "
            "(temporal, height, width) patch counts per item."
        )
    if grid_thw.shape[0] == 0:
        raise Glm5NextVisionRopeError(
            "grid_thw is empty. The tower is not called without an item, so an empty grid is a "
            "caller mistake rather than a case to return nothing for."
        )

    merge = spatial_merge_size
    per_item = []
    for temporal, height, width in grid_thw.tolist():
        temporal, height, width = int(temporal), int(height), int(width)
        if temporal < 1:
            raise Glm5NextVisionRopeError(
                f"temporal element {temporal} is not positive; every item has at least one."
            )
        if height % merge or width % merge:
            raise Glm5NextVisionRopeError(
                f"grid {height}x{width} is not whole merge blocks of {merge}. The bridge sizes "
                "the canvas so that it is; a grid that is not says the grid and the merge size "
                "disagree."
            )
        rows, columns = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij"
        )
        # Reading order becomes (block row, block column, row in block, column in block).
        block_shape = (height // merge, merge, width // merge, merge)
        rows = rows.reshape(block_shape).transpose(1, 2).flatten()
        columns = columns.reshape(block_shape).transpose(1, 2).flatten()
        per_item.append(torch.stack([rows, columns], dim=-1).repeat(temporal, 1))

    return torch.cat(per_item, dim=0)
