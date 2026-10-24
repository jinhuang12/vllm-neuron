# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 vision tower and projector loading from the checkpoint.

A parameter maps to the checkpoint key its dotted name ends with, from the
``vision_tower`` or ``mm_projector`` component on; any parent module path is
dropped, so ``visual.vision_tower.encoder.blocks.0.wqkv.weight`` reads
``vision_tower.encoder.blocks.0.wqkv.weight``. Matrices land as ``[in, out]``,
the stored ``[out, in]`` transposed. Rank ``r`` of an encoder tensor-parallel
group of ``T`` holds ``h = heads / T`` attention heads of width ``d`` and a
``1 / T`` slice of the MLP width ``I``:

- ``wqkv`` ``[hidden, 3 * h * d]``: heads ``r * h`` to ``(r + 1) * h - 1`` of
  q, then of k, then of v; the features of each q and k head in
  ``half_split_order``.
- ``wo`` ``[h * d, hidden]``: the input rows of those heads.
- ``mlp.fc0`` ``[hidden, I / T]`` and ``mlp.fc1`` ``[I / T, hidden]``: MLP
  features ``r * I / T`` to ``(r + 1) * I / T - 1``.
- ``patch_embed.proj`` ``[C * p * p, hidden]``: one row per patch value, in the
  (channel, row, column) order the processor flattens a patch in.
- ``mm_projector.proj.0`` and ``mm_projector.proj.2``: whole on every rank.
- Norm weights and the position table: as stored.
"""

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import model_parallel_is_initialized

from vllm_neuron.parallel.neuron_parallel_state import get_neuron_encoder_tp_group
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

# Width of one tower attention head: qkv_hidden_size 1536 over 12 heads.
HEAD_DIM = 128

_ROOTS = ("vision_tower", "mm_projector")


def half_split_order(head_dim: int = HEAD_DIM) -> torch.Tensor:
    """Return, for each feature of a head in split-half order, its stored feature.

    The checkpoint stores rotary pair ``i`` interleaved, at features ``2i`` and
    ``2i + 1``. Split-half order puts it at features ``i`` and
    ``i + head_dim / 2``, the layout a half-split rotary embedding reads.
    """
    return torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])


def checkpoint_key(name: str) -> str | None:
    """Return the checkpoint key a vision parameter reads, or None for any other name."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part in _ROOTS:
            return ".".join(parts[i:])
    return None


def encoder_tp() -> tuple[int, int]:
    """Return this process's rank and size in the encoder tensor-parallel group.

    Outside a distributed run the tower is whole on one rank.
    """
    if not model_parallel_is_initialized():
        return 0, 1
    group = get_neuron_encoder_tp_group()
    return group.rank_in_group, group.world_size


def _span(total: int, rank: int, size: int) -> slice:
    """Return rank ``rank``'s slice of ``total`` items split evenly over ``size`` ranks."""
    if total % size:
        raise ValueError(f"{total} does not split over encoder tensor-parallel size {size}")
    width = total // size
    return slice(rank * width, (rank + 1) * width)


def _qkv_loader(rank: int, size: int, head_dim: int) -> SafetensorsWeightLoader:
    """Return the loader of the fused ``[3 * heads * d, hidden]`` QKV weight."""
    order = half_split_order(head_dim)

    def transform(slices: list, _rank: int) -> torch.Tensor:
        stored = slices[0]
        width = stored.get_shape()[0] // 3
        if width % (size * head_dim):
            raise ValueError(
                f"{width // head_dim} heads do not split over encoder tensor-parallel size {size}"
            )
        mine = _span(width, rank, size)
        parts = []
        for i in range(3):
            rows = stored[i * width + mine.start : i * width + mine.stop]
            if i < 2:
                rows = rows.reshape(-1, head_dim, rows.shape[-1])[:, order].reshape(rows.shape)
            parts.append(rows)
        return torch.cat(parts).T

    return SafetensorsWeightLoader(transform=transform)


def _sharded_loader(stored_dim: int, rank: int, size: int) -> SafetensorsWeightLoader:
    """Return the loader that takes rank ``rank``'s slice of a matrix's stored dim, transposed."""

    def transform(slices: list, _rank: int) -> torch.Tensor:
        stored = slices[0]
        index = [slice(None), slice(None)]
        index[stored_dim] = _span(stored.get_shape()[stored_dim], rank, size)
        return stored[tuple(index)].T

    return SafetensorsWeightLoader(transform=transform)


def _patch_loader() -> SafetensorsWeightLoader:
    """Return the loader of the patch conv weight as a ``[C * p * p, hidden]`` matrix."""

    def transform(slices: list, _rank: int) -> torch.Tensor:
        weight = slices[0][:]
        return weight.reshape(weight.shape[0], -1).T

    return SafetensorsWeightLoader(transform=transform)


def vision_weight_loader(
    key: str, rank: int, size: int, head_dim: int = HEAD_DIM
) -> SafetensorsWeightLoader:
    """Return the loader that lands checkpoint tensor ``key`` in rank ``rank``'s layout."""
    if key.endswith(".wqkv.weight"):
        return _qkv_loader(rank, size, head_dim)
    if key.endswith((".wo.weight", ".mlp.fc1.weight")):
        return _sharded_loader(1, rank, size)
    if key.endswith(".mlp.fc0.weight"):
        return _sharded_loader(0, rank, size)
    if key == "vision_tower.patch_embed.proj.weight":
        return _patch_loader()
    if key.startswith("mm_projector.proj."):
        return SafetensorsWeightLoader(transform=lambda slices, _rank: slices[0][:].T)
    return SafetensorsWeightLoader()


def vision_checkpoint_mappings(
    model: nn.Module,
    rank: int | None = None,
    size: int | None = None,
    head_dim: int = HEAD_DIM,
) -> dict[str, str]:
    """Map the model's vision parameters to checkpoint keys and attach their loaders.

    ``rank`` and ``size`` default to the encoder tensor-parallel group's. A
    model with no vision parameters gives an empty map.
    """
    mappings: dict[str, str] = {}
    for name, param in model.named_parameters():
        key = checkpoint_key(name)
        if key is None:
            continue
        if rank is None or size is None:
            rank, size = encoder_tp()
        mappings[name] = key
        set_weight_loader(param, vision_weight_loader(key, rank, size, head_dim))
    return mappings
