# SPDX-License-Identifier: Apache-2.0
"""Blockwise-fp8 scale publish: the checkpoint's ``[128, 128]`` grid, unaltered.

A **host-side** producer, run once at weight-load time. A blockwise-fp8
checkpoint carries one fp32 scale per ``[128, 128]`` weight block, and the
block-quant matmuls this fork calls index that same grid, so the pair the
checkpoint delivered is the pair the kernels consume. This module publishes that
pair: every emitted scale is its input scale bit for bit, and no weight byte is
rescaled. It executes no device code and contains no kernel.

WHY NO SCALE IS REMAPPED HERE ANY MORE
-------------------------------------
The vendor block-quant matmul carries one scale per ``256 x 256`` block, so a
producer feeding it had to retain one scale per four checkpoint blocks and
rescale the other three blocks' bytes by ``s / S``. That mapping is bit-exact
only where the four constituent scales are mutually power-of-two related AND the
retained one is itself a power of two, because the vendor applies its scale
after a 128-term accumulation rather than per term
(``bwmm_shard_on_I.py:2114``-``:2138``). Read on the real checkpoint, no expert
``256``-block satisfies that precondition and a majority push a rescaled byte
past the largest magnitude fp8-e4m3 holds -- which has no representation, and no
remedy but a refusal, since the cast does not saturate and a clamp would ship
numbers the checkpoint does not contain.

The matmuls this fork now calls index the ``[128, 128]`` grid instead. There one
scale block is exactly one contraction tile, so a result is scaled before any
accumulation and *scale-then-accumulate* and *accumulate-then-scale* coincide.
Nothing has to be remapped, so nothing is: the retained scale, the three ratio
constraints, the power-of-two predicate, the per-tile rescale and the refusal
that guarded it are all gone, and with them every arithmetic operation this
module used to perform on a scale.

WHAT THIS MODULE STILL REFUSES
------------------------------
Shapes and dtypes it cannot publish: a weight that is not ``(E, H, I)``, a grid
that is not the weight's own ``(E, H // 128, I // 128)``, a grid that is not
fp32, a weight dtype that is neither fp8-e4m3 nor fp32, and an extent that is
not a whole number of ``256 x 256`` blocks -- the last because the layout
:func:`consumer_scale_shape` describes has no slot for a partial block, so a
weight with a remainder has no total publish. It refuses on no VALUE at all:
with no rescale there is no byte to push out of range, and with no retained
scale there is nothing to divide by.

PROVENANCE -- transcribed, never re-derived
-------------------------------------------
The geometry helpers below stay because other callers transcribe the same vendor
allocations: the installed ``nkilib`` block-quant matmul
``core/moe/moe_cte/bwmm_shard_on_I.py`` (2,791 L, sha256
``b2b5f7530f7bb46aad0f0e871343b7fdae6b4509712f163a9b3df2d8769c935d``). Bare
``file:line`` cites refer to that vendor file. Constants come from their
**defining** assignments, never from an inline comment: sole constant
``BLOCK_QUANT_SIZE = 256`` at ``:50``, ``TILE_SIZE = 128`` at
``moe_cte_utils.py:58``.

WHAT THIS MODULE DOES NOT DO -- declared, so the omissions read as decisions
---------------------------------------------------------------------------
* It does not enforce the consumer's *sharded* I-extent constraint. The vendor
  asserts ``I_TP % 256 == 0`` (``:681``) while the scale index divides the
  un-asserted ``I_TP_sharded = I_TP // NUM_SHARDS`` (``:91``) by 256, so at
  ``NUM_SHARDS == 2`` the practically admissible set is the **even** multiples
  of 256. That is a sharding-time property of the caller's TP degree, not of
  this per-expert publish.
* It selects no quantisation enum member and reads none.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import torch

# --- Constants, from their DEFINING assignments ------------------------------ #
# :50            BLOCK_QUANT_SIZE = 256   (framing comment at :49 -- "scales are
#                organized as 256x256 blocks along (H, I_TP)")
# moe_cte_utils.py:58   TILE_SIZE = 128   (imported at :31)
BLOCK_QUANT_SIZE = 256
TILE_SIZE = 128
# :1345 / :2110  i_tiles_per_block = BLOCK_QUANT_SIZE // TILE_SIZE
I_TILES_PER_BLOCK = BLOCK_QUANT_SIZE // TILE_SIZE

_FP32 = torch.float32
_FP8 = torch.float8_e4m3fn

DOWN = "down"
GATE_UP = "gate_up"


class BlockwiseFp8RetileError(ValueError):
    """A shape or dtype this producer refuses, named rather than silently coerced.

    Raised in preference to truncating: a weight whose extent is not a whole
    number of ``256 x 256`` blocks has no total publish, and emitting scales for
    the blocks that happen to fit would drop the remainder without a signal.
    """


# --------------------------------------------------------------------------- #
# An fp32 bit-level predicate. No producer here reads it: nothing in this module #
# needs a power of two any more. It stays because it is the exact test a landed  #
# fixture precondition measures its own scales with, and an inexact rewrite of   #
# it there would reintroduce the rounding it exists to exclude.                  #
# --------------------------------------------------------------------------- #
def _fp32(value: float) -> float:
    """Round a Python float to the nearest fp32 and return it as a Python float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", _fp32(value)))[0]


def is_pow2_exact(value: float) -> bool:
    """Is ``value`` an exact power of two? A **bit-pattern** test.

    ``value > 0`` and the IEEE-754 significand field all zeros. Subnormals,
    zero, infinities and NaNs are rejected explicitly: an infinity's significand
    field *is* all zeros, so the mask alone would call it a power of two.

    Deliberately not ``math.log2(value).is_integer()``. A float log reintroduces
    the rounding this predicate exists to exclude.
    """
    value = _fp32(float(value))
    if not value > 0.0:
        return False
    pattern = _fp32_bits(value)
    exponent = (pattern >> 23) & 0xFF
    if exponent in (0x00, 0xFF):  # subnormal-or-zero, or inf/NaN
        return False
    return (pattern & 0x007FFFFF) == 0


# --------------------------------------------------------------------------- #
# The consumer's own geometry -- transcribed.                                   #
# --------------------------------------------------------------------------- #
def consumer_scale_shape(
    num_experts: int, rows: int, cols: int, projection: str = DOWN
) -> tuple[int, int]:
    """The scale-tensor shape the vendor consumer allocates, as a 2-tuple.

    ``rows`` is the H axis and ``cols`` the I axis of one expert's weight.

    Transcribed from the consumer's own allocations:

      * ``down_proj_scale``  -- logical ``[E, I_TP//256, H//256, TILE_SIZE]``
        (:1987, "pre-broadcasted"), allocated
        ``(dims.E, I_blocks_total * (dims.H // BLOCK_QUANT_SIZE) * TILE_SIZE)``
        at :1995.
      * ``gate_up_proj_scale`` -- logical ``[E, H//256, 2, I_TP//256, TILE_SIZE]``,
        allocated ``(dims.E, H_blocks * 2 * I_blocks_total * TILE_SIZE)`` at
        :1127. The factor 2 is the gate/up fusion.

    The trailing ``TILE_SIZE`` axis is the partition broadcast: the consumer reads
    ``dp_block_scale[0:TILE_SIZE, flat_idx : flat_idx + 1]`` -- width 1, i.e. 128
    copies of one scalar (:2005, :2138).
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
    """Flat ``256``-block index for a ``[128, 128]`` tile -- transcribed.

    DOWN (:2110, :2112, :2116, :2131, :2132)::

        i_tiles_per_block = BLOCK_QUANT_SIZE // TILE_SIZE
        i_block           = i_tile_idx // i_tiles_per_block
        flat_dp_idx       = i_block * num_h_256_blocks + h_256_within_H1024

    GATE_UP (:1345, :1348, :1349, :1380, :1381, :1382)::

        h_block          = h_lin // i_tiles_per_block
        i_block          = i_tile_idx // i_tiles_per_block
        I_blocks_sharded = dims.I_TP_sharded // BLOCK_QUANT_SIZE
        flat_scale_idx   = (h_block * 2 + gate_or_up) * I_blocks_sharded + i_block

    Both forms are kept because the enumeration found exactly four scale
    consumption sites (:1387, :1396, :2138, :2145) and no fifth; narrowing to one
    would silently drop half the measured geometry.
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


# --------------------------------------------------------------------------- #
# Results.                                                                     #
# --------------------------------------------------------------------------- #
#: Value censuses skipped because the tensor they would measure carried no values.
#: A shape-only pass appends one name; a pass with values never appends. It is a
#: record and not a control: nothing in this module reads it, and a test clears it,
#: runs a pass and reads which censuses that pass reached.
SKIPPED_VALUE_CENSUSES: list[str] = []


@dataclass(frozen=True)
class RetiledBlockScales:
    """What the producer publishes, plus the counts that make it checkable."""

    #: The checkpoint's own grid, ``(E, H//128, I//128)`` fp32. Every value is the
    #: input value, so a consumer reads the numbers the checkpoint ships.
    published_scales: torch.Tensor
    #: The weight as received, ``(E, H, I)``. No byte is rescaled.
    published_weights: torch.Tensor
    projection: str
    gate_or_up: int
    #: ZERO BY ABSENCE, NOT BY MEASUREMENT. No mapping runs, so no slot is emitted
    #: unsupplied, no input scale is dropped and no rescale is inexact. The three
    #: keys stay because landed readings assert them, and the docstring above is
    #: what says why they cannot be anything else.
    emitted_unsupplied: int = 0
    input_scales_dropped: int = 0
    inexact_rescales: int = 0


# --------------------------------------------------------------------------- #
# The producer.                                                                #
# --------------------------------------------------------------------------- #
def retile_block_scales(
    weights: torch.Tensor,
    scales: torch.Tensor,
    projection: str = DOWN,
    gate_or_up: int = 0,
) -> RetiledBlockScales:
    """Publish one expert bank's ``[128, 128]`` block scales as the checkpoint holds them.

    Args:
        weights: ``(E, H, I)``, fp8-e4m3 or fp32 holding fp8-grid values.
        scales: ``(E, H // 128, I // 128)`` fp32, one scale per checkpoint block.
        projection: :data:`DOWN` or :data:`GATE_UP`.
        gate_or_up: the fusion selector, :data:`GATE_UP` only.

    Raises:
        BlockwiseFp8RetileError: on a weight that is not ``(E, H, I)``, an extent
            that is not ``256``-blocked, a scale grid that does not match the
            weight, or an unusable dtype. On no value: nothing here divides,
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
        # A meta pass carries no values, and this producer reads none: the grid it
        # publishes is the grid it received. The name is still appended, because a
        # landed reading asks which censuses a shape-only pass reached and an
        # answer that changed silently would be worse than one that stays.
        SKIPPED_VALUE_CENSUSES.append("retile_block_scales")

    return RetiledBlockScales(
        published_scales=scales,
        published_weights=weights,
        projection=projection,
        gate_or_up=gate_or_up,
    )
