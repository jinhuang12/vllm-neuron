# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 checkpoint loading: MXFP4 expert repack and the latent norm fold.

Parameter names follow the checkpoint keys without the ``language_model.``
prefix, with two exceptions:

- ``model.layers.{i}.block_sparse_moe.experts.{part}`` holds the repacked
  expert bank, one leading entry per loaded expert; ``part`` is one of
  ``EXPERT_PARTS``.
- ``model.layers.{i}.block_sparse_moe.routed_expert_up_proj.weight`` loads
  with ``routed_expert_norm.weight`` folded into its input columns, so the
  norm weight has no parameter.

Repacked expert layout, per expert, for ``G`` programs (LNC), latent width
``D`` and expert width ``I``. Program ``g`` takes expert-width columns
``g * I/G`` to ``(g + 1) * I/G``. A K tile is 512 contraction values. Each
uint16 weight element is two stored bytes in their stored order, four fp4
values; partition ``p`` of a K tile holds K values ``4p`` to ``4p + 3``. Scale
row ``r`` of a K tile covers K values ``32r`` to ``32r + 31``.

- ``gate_up_weight`` ``[G, 2 * NI, 128, (D/512) * 512]``, ``NI = I/(512G)``:
  chunk ``n < NI``, column ``c`` is gate (w1) row ``g * I/G + n * 512 + c``;
  chunk ``n >= NI`` is up (w3) row ``g * I/G + (n - NI) * 512 + c``. Free index
  ``k * 512 + c`` is K tile ``k`` of the latent width.
- ``gate_up_scale`` ``[G, 16, 2 * NI * (D/512) * 512]``: free index
  ``(n * D/512 + k) * 512 + c``.
- ``down_weight`` ``[G, D/512, 128, NI * 512]``: chunk ``d``, column ``c`` is
  w2 row ``d * 512 + c``. Free index ``k * 512 + c`` is K tile ``k`` of the
  program's expert-width columns.
- ``down_scale`` ``[G, 16, (D/512) * NI * 512]``: free index
  ``(d * NI + k) * 512 + c``.
"""

import logging
import os

import torch
import torch.nn as nn

from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    SliceView,
    get_weight_loader,
    set_weight_loader,
)

from .config import CHECKPOINT_PREFIX, KimiK3TextConfig

logger = logging.getLogger(__name__)

EXPERT_PARTS = ("gate_up_weight", "gate_up_scale", "down_weight", "down_scale")

# Checkpoint tensors of one expert that each bank part reads, in order.
_PART_SOURCES = {
    "gate_up_weight": ("w1.weight_packed", "w3.weight_packed"),
    "gate_up_scale": ("w1.weight_scale", "w3.weight_scale"),
    "down_weight": ("w2.weight_packed",),
    "down_scale": ("w2.weight_scale",),
}

# Tensor kinds the checkpoint stores in F32; they load as F32 whatever the
# parameter's declared dtype.
F32_SUFFIXES = (
    "self_attn.q_conv1d.weight",
    "self_attn.k_conv1d.weight",
    "self_attn.v_conv1d.weight",
    "self_attn.A_log",
    "self_attn.dt_bias",
    "self_attn.o_norm.weight",
    "block_sparse_moe.gate.e_score_correction_bias",
)

_UP_SUFFIX = "block_sparse_moe.routed_expert_up_proj.weight"
_NORM_SUFFIX = "block_sparse_moe.routed_expert_norm.weight"
_A_LOG_SUFFIX = "self_attn.A_log"
_EXPERTS = "block_sparse_moe.experts"


def _repack_gate_up(gate: torch.Tensor, up: torch.Tensor, g: int) -> torch.Tensor:
    """Repack stored w1 and w3 bytes [I, D/2] into the gate/up weight layout."""
    i, half_d = gate.shape
    nk = half_d * 2 // 512

    def one(w):
        w = w.contiguous().view(torch.uint16)  # [I, D/4]
        w = w.reshape(g, i // g // 512, 512, nk, 128).permute(0, 1, 4, 3, 2)
        return w.reshape(g, i // g // 512, 128, nk * 512)

    return torch.cat([one(gate), one(up)], dim=1).contiguous()


def _repack_gate_up_scale(gate: torch.Tensor, up: torch.Tensor, g: int) -> torch.Tensor:
    """Repack stored w1 and w3 scales [I, D/32] into the gate/up scale layout."""
    i, groups = gate.shape
    nk = groups // 16

    def one(s):
        s = s.reshape(g, i // g // 512, 512, nk, 16).permute(0, 4, 1, 3, 2)
        return s  # [G, 16, I/(G*512), NK, 512]

    out = torch.cat([one(gate), one(up)], dim=2)
    return out.reshape(g, 16, -1).contiguous()


def _repack_down(down: torch.Tensor, g: int) -> torch.Tensor:
    """Repack stored w2 bytes [D, I/2] into the down weight layout."""
    d, half_i = down.shape
    kd = half_i * 2 // g // 512
    w = down.contiguous().view(torch.uint16)  # [D, I/4]
    w = w.reshape(d // 512, 512, g, kd, 128).permute(2, 0, 4, 3, 1)
    return w.reshape(g, d // 512, 128, kd * 512).contiguous()


def _repack_down_scale(scale: torch.Tensor, g: int) -> torch.Tensor:
    """Repack stored w2 scales [D, I/32] into the down scale layout."""
    d, groups = scale.shape
    kd = groups // g // 16
    s = scale.reshape(d // 512, 512, g, kd, 16).permute(2, 4, 0, 3, 1)
    return s.reshape(g, 16, -1).contiguous()


def _repack_part(part: str, tensors: list[torch.Tensor], g: int) -> torch.Tensor:
    """Repack one expert's source tensors for one bank part."""
    if part == "gate_up_weight":
        return _repack_gate_up(tensors[0], tensors[1], g)
    if part == "gate_up_scale":
        return _repack_gate_up_scale(tensors[0], tensors[1], g)
    if part == "down_weight":
        return _repack_down(tensors[0], g)
    return _repack_down_scale(tensors[0], g)


def repack_expert(
    w1_packed: torch.Tensor,
    w1_scale: torch.Tensor,
    w3_packed: torch.Tensor,
    w3_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    num_programs: int = 2,
) -> dict[str, torch.Tensor]:
    """Repack one expert's stored MXFP4 tensors into the four bank parts."""
    sources = {
        "w1.weight_packed": w1_packed,
        "w1.weight_scale": w1_scale,
        "w3.weight_packed": w3_packed,
        "w3.weight_scale": w3_scale,
        "w2.weight_packed": w2_packed,
        "w2.weight_scale": w2_scale,
    }
    return {
        part: _repack_part(part, [sources[s] for s in _PART_SOURCES[part]], num_programs)
        for part in EXPERT_PARTS
    }


def expert_bank_loader(part: str, num_programs: int = 2) -> SafetensorsWeightLoader:
    """Return a loader that stacks the repacked part of every mapped expert."""
    width = len(_PART_SOURCES[part])

    def transform(slices: list, rank: int) -> torch.Tensor:
        experts = [
            _repack_part(part, [s[:] for s in slices[i : i + width]], num_programs)
            for i in range(0, len(slices), width)
        ]
        return torch.stack(experts)

    return SafetensorsWeightLoader(transform=transform)


def fold_latent_norm(up_weight: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
    """Multiply the norm weight into the up weight's input columns in fp32, then cast to bf16."""
    return (up_weight.float() * norm_weight.float()).to(torch.bfloat16)


def folded_up_loader(inner: SafetensorsWeightLoader) -> SafetensorsWeightLoader:
    """Return a loader that folds the norm weight, then applies ``inner``."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        folded = fold_latent_norm(slices[0][:], slices[1][:])
        return inner.load([SliceView(folded)], rank)

    return SafetensorsWeightLoader(transform=transform)


def a_log_loader(num_heads: int, inner: SafetensorsWeightLoader) -> SafetensorsWeightLoader:
    """Return a loader that keeps the first num_heads stored entries, then applies ``inner``."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        stored = slices[0][:].reshape(-1)
        return inner.load([SliceView(stored[:num_heads])], rank)

    return SafetensorsWeightLoader(transform=transform)


def checkpoint_mappings(
    model: nn.Module,
    text_config: KimiK3TextConfig,
    expert_ids: list[int] | None = None,
    num_programs: int = 2,
) -> tuple[dict[str, str | list[str]], dict[str, torch.dtype]]:
    """Map parameters to checkpoint keys and attach the checkpoint-layout loaders.

    Returns the mappings and the F32 dtype overrides. ``expert_ids`` names the
    experts each bank holds, in order; the default is the first ``E`` experts,
    where ``E`` is the bank's leading size.
    """
    mappings: dict[str, str | list[str]] = {}
    for name, param in model.named_parameters():
        prefix, _, leaf = name.rpartition(".")
        if prefix.endswith(_EXPERTS) and leaf in EXPERT_PARTS:
            ids = expert_ids if expert_ids is not None else range(param.shape[0])
            mappings[name] = [
                f"{CHECKPOINT_PREFIX}{prefix}.{e}.{src}"
                for e in ids
                for src in _PART_SOURCES[leaf]
            ]
            set_weight_loader(param, expert_bank_loader(leaf, num_programs))
        elif name.endswith(_UP_SUFFIX):
            norm = name[: -len(_UP_SUFFIX)] + _NORM_SUFFIX
            mappings[name] = [CHECKPOINT_PREFIX + name, CHECKPOINT_PREFIX + norm]
            set_weight_loader(param, folded_up_loader(get_weight_loader(param)))
        elif name.endswith(_NORM_SUFFIX):
            raise ValueError(
                f"{name} is folded into the latent up weight and must not be a parameter"
            )
        else:
            mappings[name] = CHECKPOINT_PREFIX + name
            if name.endswith(_A_LOG_SUFFIX):
                inner = get_weight_loader(param)
                set_weight_loader(param, a_log_loader(text_config.kda_num_heads, inner))
    dtype_override = {
        name: torch.float32 for name in mappings if name.endswith(F32_SUFFIXES)
    }
    return mappings, dtype_override


def load_weights(
    model: nn.Module,
    text_config: KimiK3TextConfig,
    checkpoint_path: str | os.PathLike,
    device: torch.device,
    cache_dir: str | None = None,
    *,
    rank: int = 0,
    world_size: int = 1,
    expert_ids: list[int] | None = None,
    num_programs: int = 2,
    pipelined: bool = True,
) -> None:
    """Load the K3 checkpoint into ``model``'s parameters.

    Every tensor lands in its parameter's dtype, except the F32 kinds, which
    land in F32. ``pipelined=False`` reads without the distributed page-cache
    pipeline, for CPU runs.
    """
    mappings, dtype_override = checkpoint_mappings(
        model, text_config, expert_ids, num_programs
    )
    checkpoint = SafetensorsCheckpoint(str(checkpoint_path), cache_dir)
    if pipelined:
        result = checkpoint.load_sharded_pipelined(
            rank, world_size, model, mappings, device, dtype_override=dtype_override
        )
        state = result.state_dict
    else:
        result = checkpoint.load_sharded(rank, world_size, model, mappings, device)
        params = dict(model.named_parameters())
        state = {
            name: t.to(dtype_override.get(name, params[name].dtype))
            for name, t in result.state_dict.items()
        }
    if any(name.endswith(_A_LOG_SUFFIX) for name in state):
        logger.info(
            "Kimi K3 A_log: loading the first %d stored entries; the rest are unused",
            text_config.kda_num_heads,
        )
    model.load_state_dict(state, strict=False, assign=True)
