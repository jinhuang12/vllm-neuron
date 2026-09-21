# SPDX-License-Identifier: Apache-2.0
"""Blockwise-fp8 scale publishing for MoE expert weights.

A host-side pass, run once at weight load; it holds no device code. A
blockwise-fp8 checkpoint carries one fp32 scale per ``[128, 128]`` weight block,
and the block-quant matmuls that consume these weights index that same grid, so
both tensors pass through unchanged: every published scale is its input scale bit
for bit, and no weight byte is rescaled.

Passing the grid through is what keeps the publish exact. A matmul that carries
one scale per ``256 x 256`` block applies it after a 128-term accumulation, so
folding four ``[128, 128]`` scales into one and rescaling the other three blocks'
bytes by the ratio is bit-exact only where the four scales are mutually
power-of-two related and the retained one is itself a power of two. Real
checkpoints do not satisfy that, and the rescaled bytes go past the largest
magnitude fp8-e4m3 holds; the cast does not saturate, so there is no value to
emit. On the ``[128, 128]`` grid one scale block is exactly one contraction tile,
so scaling before and after accumulation coincide and nothing has to be remapped.

The ``256``-block geometry is still described here, by
:func:`consumer_scale_shape` and :func:`flat_scale_index`, because that is how a
consumer addresses a scale. One consumer constraint is deliberately not checked
here: the consumer asserts ``I_TP % 256 == 0`` but indexes
``I_TP // NUM_SHARDS``, so with two shards only the even multiples of ``256`` are
admissible. That follows from the caller's TP degree, not from this per-expert
publish.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import torch

#: Edge of one consumer scale block, along both H and I_TP.
BLOCK_QUANT_SIZE = 256
#: Edge of one contraction tile, and of one checkpoint scale block.
TILE_SIZE = 128
I_TILES_PER_BLOCK = BLOCK_QUANT_SIZE // TILE_SIZE

_FP32 = torch.float32
_FP8 = torch.float8_e4m3fn

DOWN = "down"
GATE_UP = "gate_up"


class BlockwiseFp8RetileError(ValueError):
    """A shape or dtype this producer refuses rather than silently coerce.

    An extent that is not a whole number of ``256 x 256`` blocks has no total
    publish, and emitting scales for the blocks that happen to fit would drop the
    remainder without a signal.
    """


def _fp32(value: float) -> float:
    """Round a Python float to the nearest fp32 and return it as a Python float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", _fp32(value)))[0]


def is_pow2_exact(value: float) -> bool:
    """Return True when ``value`` is an exact power of two, tested on its bits.

    Requires ``value > 0`` and an all-zero IEEE-754 significand field. Zero,
    subnormals, infinities and NaNs are rejected explicitly, because an
    infinity's significand field is all zeros too and the mask alone would accept
    it. A float ``log2`` is avoided: it reintroduces the rounding this predicate
    exists to exclude.
    """
    value = _fp32(float(value))
    if not value > 0.0:
        return False
    pattern = _fp32_bits(value)
    exponent = (pattern >> 23) & 0xFF
    if exponent in (0x00, 0xFF):  # subnormal-or-zero, or inf/NaN
        return False
    return (pattern & 0x007FFFFF) == 0


def consumer_scale_shape(
    num_experts: int, rows: int, cols: int, projection: str = DOWN
) -> tuple[int, int]:
    """Return the flat scale-tensor shape the consumer allocates, as ``(E, N)``.

    ``rows`` is the H axis and ``cols`` the I axis of one expert's weight.

    * ``down`` scales are logically ``[E, I_TP//256, H//256, TILE_SIZE]``,
      allocated as ``(E, I_blocks * (H // 256) * TILE_SIZE)``.
    * ``gate_up`` scales are logically ``[E, H//256, 2, I_TP//256, TILE_SIZE]``,
      allocated as ``(E, H_blocks * 2 * I_blocks * TILE_SIZE)``; the factor 2 is
      the gate/up fusion.

    The trailing ``TILE_SIZE`` axis is a partition broadcast: the consumer reads a
    width-1 slice of it, that is 128 copies of one scalar.
    """
    _require_blocked(rows, cols)
    h_256 = rows // BLOCK_QUANT_SIZE
    i_256 = cols // BLOCK_QUANT_SIZE
    if num_experts < 1:
        raise BlockwiseFp8RetileError(f"num_experts must be >= 1, got {num_experts}")
    if projection == DOWN:
        return (num_experts, i_256 * h_256 * TILE_SIZE)
    if projection == GATE_UP:
        return (num_experts, h_256 * 2 * i_256 * TILE_SIZE)
    raise BlockwiseFp8RetileError(
        f"projection must be {DOWN!r} or {GATE_UP!r}, got {projection!r}"
    )


def flat_scale_index(
    h_tile: int,
    i_tile: int,
    h_256: int,
    i_256: int,
    projection: str = DOWN,
    gate_or_up: int = 0,
) -> int:
    """Return the flat ``256``-block scale index for a ``[128, 128]`` tile.

    ``h_256`` and ``i_256`` are the ``256``-block counts of the axis, the I count
    taken over the sharded I extent. Both projections are addressed, because a
    consumer reads scales for the down bank and for the fused gate/up bank::

        down:    i_block * h_256 + h_block
        gate_up: (h_block * 2 + gate_or_up) * i_256 + i_block

    where each block index is the tile index divided by ``I_TILES_PER_BLOCK``.
    """
    h_block = h_tile // I_TILES_PER_BLOCK
    i_block = i_tile // I_TILES_PER_BLOCK
    if projection == DOWN:
        return i_block * h_256 + h_block
    if projection == GATE_UP:
        return (h_block * 2 + gate_or_up) * i_256 + i_block
    raise BlockwiseFp8RetileError(
        f"projection must be {DOWN!r} or {GATE_UP!r}, got {projection!r}"
    )


def _require_blocked(rows: int, cols: int) -> None:
    bad = [
        (name, extent)
        for name, extent in (("H", rows), ("I", cols))
        if extent <= 0 or extent % BLOCK_QUANT_SIZE
    ]
    if bad:
        detail = ", ".join(f"{name}={extent}" for name, extent in bad)
        raise BlockwiseFp8RetileError(
            f"weight extent [{rows},{cols}] is not a whole number of "
            f"{BLOCK_QUANT_SIZE}x{BLOCK_QUANT_SIZE} blocks ({detail} is not a "
            f"positive multiple of {BLOCK_QUANT_SIZE}). Refusing rather than "
            f"truncating: the remainder has no scale in the consumer's layout."
        )


#: Names of passes that ran on meta tensors, where there was no value to inspect.
SKIPPED_VALUE_CENSUSES: list[str] = []


@dataclass(frozen=True)
class RetiledBlockScales:
    """One expert bank's published scales and weights, plus their counts."""

    #: The checkpoint's own grid, ``(E, H//128, I//128)`` fp32, values unchanged.
    published_scales: torch.Tensor
    #: The weight as received, ``(E, H, I)``. No byte is rescaled.
    published_weights: torch.Tensor
    projection: str
    gate_or_up: int
    #: Always 0: no scale is remapped, so no slot is emitted unsupplied, no input
    #: scale is dropped, and no rescale can be inexact.
    emitted_unsupplied: int = 0
    input_scales_dropped: int = 0
    inexact_rescales: int = 0


def retile_block_scales(
    weights: torch.Tensor,
    scales: torch.Tensor,
    projection: str = DOWN,
    gate_or_up: int = 0,
) -> RetiledBlockScales:
    """Publish one expert bank's ``[128, 128]`` scales as the checkpoint holds them.

    Args:
        weights: ``(E, H, I)``, fp8-e4m3 or fp32 holding fp8-grid values.
        scales: ``(E, H // 128, I // 128)`` fp32, one scale per checkpoint block.
        projection: :data:`DOWN` or :data:`GATE_UP`.
        gate_or_up: the fusion selector, :data:`GATE_UP` only.

    Raises:
        BlockwiseFp8RetileError: on a weight that is not ``(E, H, I)``, an extent
            that is not ``256``-blocked, a scale grid that does not match the
            weight, or an unusable dtype. Never on a value: nothing here divides,
            multiplies or casts.
    """
    if weights.dim() != 3:
        raise BlockwiseFp8RetileError(
            f"weights must be (E, H, I), got shape {tuple(weights.shape)}"
        )
    if scales.dim() != 3:
        raise BlockwiseFp8RetileError(
            f"scales must be (E, H//{TILE_SIZE}, I//{TILE_SIZE}), got shape "
            f"{tuple(scales.shape)}"
        )
    experts, rows, cols = (int(extent) for extent in weights.shape)
    _require_blocked(rows, cols)
    if scales.dtype != _FP32:
        raise BlockwiseFp8RetileError(f"scales must be fp32, got {scales.dtype}")
    if weights.dtype not in (_FP8, _FP32):
        raise BlockwiseFp8RetileError(
            f"weights must be {_FP8} or {_FP32}, got {weights.dtype}"
        )
    want_scales = (experts, rows // TILE_SIZE, cols // TILE_SIZE)
    if tuple(int(extent) for extent in scales.shape) != want_scales:
        raise BlockwiseFp8RetileError(
            f"scale grid {tuple(scales.shape)} does not match weight "
            f"[{rows},{cols}] over {experts} experts; expected {want_scales}"
        )

    if weights.device.type == "meta":
        # A meta pass carries no values and this producer reads none, so record
        # the skip rather than inspect.
        SKIPPED_VALUE_CENSUSES.append("retile_block_scales")

    return RetiledBlockScales(
        published_scales=scales,
        published_weights=weights,
        projection=projection,
        gate_or_up=gate_or_up,
    )
