# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 vision inputs: window order, key bounds, RoPE tables, pos-embed weights.

Patches enter the tower in 2 x 2 window order, so the patch merger after the
tower is a view. The RoPE tables follow the same order. A block packs one or
more images in turn and pads with zero rows up to its bucket. Each worker
builds its RoPE tables and window-order index from the grids and keeps them
in a bounded cache of device tensors.
"""

from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F

MERGE = 2
HEAD_DIM = 128
ROPE_THETA = 10000
POS_EMB_SIDE = 64
TABLE_CACHE_BYTES = 64_000_000

Grids = tuple[tuple[int, int], ...]


class GridTables(NamedTuple):
    """A block's worker tables: RoPE cos and sin [bucket, 128], window-order index."""

    cos: torch.Tensor
    sin: torch.Tensor
    index: torch.Tensor


def _as_grids(grids: Sequence[Sequence[int]], bucket: int) -> Grids:
    """The block's (height, width) grids, checked to tile into windows and fit."""
    out = tuple((int(h), int(w)) for h, w in grids)
    for h, w in out:
        if h % MERGE or w % MERGE:
            raise ValueError(f"a {h} x {w} grid does not tile into 2 x 2 windows")
    rows = sum(h * w for h, w in out)
    if rows > bucket:
        raise ValueError(f"{rows} patches do not fit the {bucket}-row bucket")
    return out


def window_order_index(h: int, w: int) -> torch.Tensor:
    """Raster index of each patch of an h x w grid in window order, int64 [h * w]."""
    index = torch.arange(h * w).view(h // MERGE, MERGE, w // MERGE, MERGE)
    return index.permute(0, 2, 1, 3).reshape(-1)


def _block_index(grids: Grids) -> torch.Tensor:
    """Window-order index of a block of images whose raster patches are concatenated."""
    parts, offset = [], 0
    for h, w in grids:
        parts.append(window_order_index(h, w) + offset)
        offset += h * w
    return torch.cat(parts)


def order_patches(
    patches: torch.Tensor, grids: Sequence[Sequence[int]], bucket: int
) -> torch.Tensor:
    """The block's patches in window order, images in turn, zero rows up to the bucket.

    ``patches`` holds each image's patches in raster order, the images concatenated.
    """
    grids = _as_grids(grids, bucket)
    index = _block_index(grids)
    if patches.shape[0] != index.numel():
        raise ValueError(
            f"{patches.shape[0]} patches for grids holding {index.numel()}"
        )
    out = patches.new_zeros((bucket, *patches.shape[1:]))
    out[: index.numel()] = patches[index]
    return out


def key_bounds(
    grids: Sequence[Sequence[int]], bucket: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's key range [bound_min, bound_max), [bucket, 1] int32; pads (0, 0)."""
    grids = _as_grids(grids, bucket)
    bound_min = torch.zeros((bucket, 1), dtype=torch.int32)
    bound_max = torch.zeros((bucket, 1), dtype=torch.int32)
    start = 0
    for h, w in grids:
        end = start + h * w
        bound_min[start:end] = start
        bound_max[start:end] = end
        start = end
    return bound_min, bound_max


def rope_tables(
    grids: Sequence[Sequence[int]], bucket: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE cos and sin, each [bucket, 128] fp32, in window order, split-half layout.

    Pair j of a head, the reference's interleaved columns 2j and 2j + 1, sits at
    columns j and j + 64, and both columns take angle j: the width frequency
    j // 2 for even j, the height frequency j // 2 for odd j. RoPE(x) is then
    [x1, x2] * cos + [-x2, x1] * sin over the halves x1 and x2. Each image's
    positions start at its own (0, 0). Pad rows are zero.
    """
    grids = _as_grids(grids, bucket)
    dim_range = torch.arange(0, HEAD_DIM, 4)[: HEAD_DIM // 4].float()
    freqs = 1.0 / (ROPE_THETA ** (dim_range / HEAD_DIM))
    angles = []
    for h, w in grids:
        index = window_order_index(h, w)
        col = (index % w).float()
        row = (index // w).float()
        pairs = torch.stack(
            [torch.outer(col, freqs), torch.outer(row, freqs)], dim=-1
        )
        angles.append(pairs.reshape(h * w, HEAD_DIM // 2))
    angle = torch.cat(angles)
    # torch.polar rounds each element as the reference's table does.
    cis = torch.polar(torch.ones_like(angle), angle)
    rows = angle.shape[0]
    cos = torch.zeros((bucket, HEAD_DIM), dtype=torch.float32)
    sin = torch.zeros((bucket, HEAD_DIM), dtype=torch.float32)
    half = HEAD_DIM // 2
    for table, part in ((cos, cis.real), (sin, cis.imag)):
        table[:rows, :half] = part
        table[:rows, half:] = part
    return cos, sin


def bilinear_weights(out_size: int, in_size: int = POS_EMB_SIDE) -> torch.Tensor:
    """Bilinear resize matrix [out_size, in_size] fp32, no corner alignment."""
    # The kernel's own fp32 source index and lambda: one linear resize of the identity.
    eye = torch.eye(in_size, dtype=torch.float32).unsqueeze(0)
    resized = F.interpolate(eye, size=out_size, mode="linear", align_corners=False)
    return resized[0].T.contiguous()


def pos_embed_weights(h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Wh [h, 64] and Ww [w, 64]: Wh @ table @ Ww.T resizes the pos-embed table."""
    return bilinear_weights(h), bilinear_weights(w)


def build_grid_tables(grids: Sequence[Sequence[int]], bucket: int) -> GridTables:
    """The block's RoPE tables and window-order index, from its grids alone."""
    grids = _as_grids(grids, bucket)
    cos, sin = rope_tables(grids, bucket)
    return GridTables(cos, sin, _block_index(grids))


class TableCache:
    """A worker's least-recently-used cache of device tables, bounded in bytes."""

    def __init__(
        self, device: torch.device | str = "cpu", max_bytes: int = TABLE_CACHE_BYTES
    ):
        self.device = torch.device(device)
        self.max_bytes = max_bytes
        self.nbytes = 0
        self._entries: OrderedDict[Hashable, GridTables] = OrderedDict()

    def get(self, key: Hashable, build: Callable[[], GridTables]) -> GridTables:
        """The tables under ``key``, else ``build()``'s on the device, kept if small."""
        tables = self._entries.get(key)
        if tables is not None:
            self._entries.move_to_end(key)
            return tables
        tables = GridTables(*(t.to(self.device) for t in build()))
        size = sum(t.nbytes for t in tables)
        if size > self.max_bytes:
            return tables
        while self.nbytes + size > self.max_bytes:
            _, old = self._entries.popitem(last=False)
            self.nbytes -= sum(t.nbytes for t in old)
        self._entries[key] = tables
        self.nbytes += size
        return tables


def grid_tables(
    cache: TableCache, grids: Sequence[Sequence[int]], bucket: int
) -> GridTables:
    """The block's worker tables from its grids, through the worker's cache."""
    grids = _as_grids(grids, bucket)
    # The form leads the key, so another layout of the same grids keeps its own entry.
    return cache.get(
        ("group", grids, bucket), lambda: build_grid_tables(grids, bucket)
    )
