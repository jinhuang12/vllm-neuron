# SPDX-License-Identifier: Apache-2.0
"""Blockwise-fp8 scale retile: checkpoint ``[128, 128]`` -> consumer ``256 x 256``.

WHAT THIS IS
------------
A **host-side** producer, run once at weight-load time. A blockwise-fp8
checkpoint carries one fp32 scale per ``[128, 128]`` weight block; the block-quant
matmul this fork calls carries one fp32 scale per ``256 x 256`` block, so one
consumer scale stands in for **four** checkpoint blocks. This module emits the
consumer's scale tensor and the correspondingly rescaled fp8 weights.

It executes no device code and contains no kernel. The matmul it feeds is the
kernel-class part and is authored separately.

THE MECHANISM
-------------
For each ``256 x 256`` block the producer retains one fp32 scale ``S`` -- the
block's ``(h_tile 0, i_tile 0)`` checkpoint scale -- and rescales that block's fp8
weights by ``s / S`` per ``[128, 128]`` sub-block::

    w' = fp8(w * (s / S))          and the consumer computes   w' * S

When ``s / S`` is an exact power of two the rescale only shifts an fp8 exponent,
so it is bit-exact and ``w' * S`` reproduces ``w * s`` bit-for-bit. When it is
not, fp8's three mantissa bits round and the retile loses information.

THE COMPLETE LOSSLESSNESS CONDITION -- BOTH CONJUNCTS
----------------------------------------------------
This producer *measures* two conjuncts per ``256 x 256`` block and enforces
neither:

1. the four constituent ``[128, 128]`` scales are **mutually power-of-two
   related** -- three independent ratio constraints against the retained scale,
   evaluated in the order ``((0,1), (1,0), (1,1))``;
2. the retained ``256``-block scale is **itself a power of two**.

Conjunct 2 is owed because the consumer accumulates two ``i_tiles`` in PSUM and
applies the scale **after** the accumulation, not elementwise
(``bwmm_shard_on_I.py:2114``-``:2138``). Applying one scale after a 128-term
accumulation is not the same rounding as applying it per term unless that scale
is itself a power of two. Measured, both directions, at
``increments/evidence-071.md`` F1: **720 fp32 ulp** with a non-pow2 block scale,
**0 ulp** with a pow2 one.

Both conjuncts are tested by a **bit-pattern** predicate (:func:`is_pow2_exact`),
never by ``log2`` and never against a tolerance -- a float-log test would
reintroduce exactly the inexactness the predicate exists to exclude.

PROVENANCE -- transcribed, never re-derived
-------------------------------------------
The consumer is the installed ``nkilib`` block-quant matmul
``core/moe/moe_cte/bwmm_shard_on_I.py`` (2,791 L, sha256
``b2b5f7530f7bb46aad0f0e871343b7fdae6b4509712f163a9b3df2d8769c935d``), enumerated
at ``increments/wp6-scale-consumer-geometry.md`` sections 2.1-2.3 and transcribed
into ``test/vllm_neuron/functional/moe/test_blockwise_fp8_retile_losslessness.py``.
Bare ``file:line`` cites below refer to that vendor file. Constants come from
their **defining** assignments, never from an inline comment.

The granularity is a fixed property of the substrate, not a choice: sole constant
``BLOCK_QUANT_SIZE = 256`` at ``:50``, in 1 of 10 ``bwmm_*`` files
(``increments/evidence-007.md``).

WHAT THIS MODULE DOES NOT DO -- declared, so the omissions read as decisions
---------------------------------------------------------------------------
* It does not **enforce** losslessness. Real checkpoint scales are not generally
  powers of two, so the satisfying fraction over real weights is expected to be
  materially below 100%. That reading, and any tolerance question it raises,
  belongs to the first attempt that loads a real checkpoint.
* It does not enforce the consumer's *sharded* I-extent constraint. The vendor
  asserts ``I_TP % 256 == 0`` (``:681``) while the scale index divides the
  un-asserted ``I_TP_sharded = I_TP // NUM_SHARDS`` (``:91``) by 256, so at
  ``NUM_SHARDS == 2`` the practically admissible set is the **even** multiples of
  256 (``increments/evidence-070.md`` F6). That is a sharding-time property of
  the caller's TP degree, not of this per-expert re-layout, and is recorded here
  rather than silently enforced or silently ignored.
* It selects no quantisation enum member and reads none.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable

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

#: The three ratio constraints, in the order the design declares them, taken
#: against the block's ``(h_tile 0, i_tile 0)`` scale.
RATIO_ORDER: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1))

DOWN = "down"
GATE_UP = "gate_up"


class BlockwiseFp8RetileError(ValueError):
    """A shape or dtype this producer refuses, named rather than silently coerced.

    Raised in preference to truncating: a weight whose extent is not a whole
    number of ``256 x 256`` blocks has no total retile, and emitting scales for
    the blocks that happen to fit would drop the remainder without a signal.
    """


# --------------------------------------------------------------------------- #
# fp32 bit-level helpers. Every predicate here is exact-equality, so all of it  #
# works on the stored bit pattern.                                             #
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


def pow2_exponent(value: float) -> int | None:
    """The integer ``k`` with ``value == 2 ** k``, or ``None`` if there is none."""
    if not is_pow2_exact(value):
        return None
    return ((_fp32_bits(_fp32(float(value))) >> 23) & 0xFF) - 127


# --------------------------------------------------------------------------- #
# The consumer's own geometry -- transcribed.                                   #
# --------------------------------------------------------------------------- #
def consumer_scale_shape(
    num_experts: int, rows: int, cols: int, projection: str = DOWN
) -> tuple[int, int]:
    """The scale-tensor shape the consumer allocates, as a 2-tuple.

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
@dataclass(frozen=True)
class BlockLosslessnessRecord:
    """Both losslessness conjuncts for one ``256 x 256`` block, as measured."""

    expert: int
    h_block: int
    i_block: int
    block_scale: float
    ratios: tuple[float, ...]
    ratios_pow2: tuple[bool, ...]
    quad_mutually_pow2: bool  # conjunct 1
    block_scale_pow2: bool  # conjunct 2

    @property
    def lossless(self) -> bool:
        """The COMPLETE condition -- both conjuncts, never conjunct 1 alone."""
        return self.quad_mutually_pow2 and self.block_scale_pow2

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.expert, self.h_block, self.i_block)


@dataclass(frozen=True)
class RetiledBlockScales:
    """What the producer emits, plus the counts that make it checkable."""

    #: Ships to the consumer. Shape :func:`consumer_scale_shape`, fp32.
    consumer_scales: torch.Tensor
    #: The mapping itself: one retained scale per block, ``(E, i_256, h_256)``.
    block_scales: torch.Tensor
    #: Ships to the consumer, fp8, ``(E, H, I)``.
    retiled_weights: torch.Tensor
    #: ``s = S * 2 ** shift`` per ``[128, 128]`` tile, ``(E, h_tiles, i_tiles)``
    #: int32, with ``_NO_SHIFT`` where no integer shift exists. This is what makes
    #: the layout invertible; the consumer never reads it.
    tile_exponent_shifts: torch.Tensor
    projection: str
    gate_or_up: int
    #: Emitted slots whose value does not decode back to an input scale.
    emitted_unsupplied: int
    #: Input scales the layout cannot reproduce bit-exactly.
    input_scales_dropped: int
    #: fp8 sub-block rescales that were not bit-exact.
    inexact_rescales: int
    records: tuple[BlockLosslessnessRecord, ...]

    def losslessness_fraction(self) -> tuple[int, int]:
        """``(k, N)`` -- blocks satisfying BOTH conjuncts, over all blocks."""
        return sum(1 for record in self.records if record.lossless), len(self.records)


_NO_SHIFT = -(2**31)


# --------------------------------------------------------------------------- #
# The producer.                                                                #
# --------------------------------------------------------------------------- #
def retile_block_scales(
    weights: torch.Tensor,
    scales: torch.Tensor,
    projection: str = DOWN,
    gate_or_up: int = 0,
    *,
    index_fn: Callable[..., int] | None = None,
) -> RetiledBlockScales:
    """Retile one expert bank's ``[128, 128]`` block scales onto ``256`` granularity.

    Args:
        weights: ``(E, H, I)``, fp8-e4m3 or fp32 holding fp8-grid values.
        scales: ``(E, H // 128, I // 128)`` fp32, one scale per checkpoint block.
        projection: :data:`DOWN` or :data:`GATE_UP`.
        gate_or_up: the fusion selector, :data:`GATE_UP` only.
        index_fn: **liveness-control injection point only.** Production callers
            never pass it. It exists so a test can drive the emitted-slot counter
            with a deliberately wrong flattening and show the counted zero moves;
            a counter that cannot return non-zero is not a counter (D1.5).

    Raises:
        BlockwiseFp8RetileError: on an extent that is not ``256``-blocked, a scale
            grid that does not match the weight, or an unusable dtype.
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

    h_tiles, i_tiles = rows // TILE_SIZE, cols // TILE_SIZE
    h_256, i_256 = rows // BLOCK_QUANT_SIZE, cols // BLOCK_QUANT_SIZE
    index = index_fn or (
        lambda h_tile, i_tile: flat_scale_index(
            h_tile, i_tile, h_256, i_256, projection, gate_or_up
        )
    )

    weights_fp32 = weights.to(_FP32)
    shape = consumer_scale_shape(experts, rows, cols, projection)
    # NaN marks a slot the producer never wrote, so a read of one is DETECTABLE
    # rather than merely wrong (the evidence-071.md section 9.2 lesson: a range
    # check cannot see a transposed flattening).
    consumer_scales = torch.full(shape, float("nan"), dtype=_FP32)
    block_scales = torch.full((experts, i_256, h_256), float("nan"), dtype=_FP32)
    retiled = torch.empty_like(weights_fp32)
    shifts = torch.full((experts, h_tiles, i_tiles), _NO_SHIFT, dtype=torch.int32)
    records: list[BlockLosslessnessRecord] = []
    inexact = 0

    for expert in range(experts):
        for h_block in range(h_256):
            for i_block in range(i_256):
                retained = scales[expert, h_block * I_TILES_PER_BLOCK,
                                  i_block * I_TILES_PER_BLOCK]
                block_scales[expert, i_block, h_block] = retained
                ratios: list[float] = []
                ratios_pow2: list[bool] = []
                for h_off, i_off in RATIO_ORDER:
                    ratio = _fp32(
                        float(
                            scales[
                                expert,
                                h_block * I_TILES_PER_BLOCK + h_off,
                                i_block * I_TILES_PER_BLOCK + i_off,
                            ]
                            / retained
                        )
                    )
                    ratios.append(ratio)
                    ratios_pow2.append(is_pow2_exact(ratio))
                records.append(
                    BlockLosslessnessRecord(
                        expert=expert,
                        h_block=h_block,
                        i_block=i_block,
                        block_scale=float(retained),
                        ratios=tuple(ratios),
                        ratios_pow2=tuple(ratios_pow2),
                        quad_mutually_pow2=all(ratios_pow2),
                        block_scale_pow2=is_pow2_exact(float(retained)),
                    )
                )
                for h_off in range(I_TILES_PER_BLOCK):
                    for i_off in range(I_TILES_PER_BLOCK):
                        h_tile = h_block * I_TILES_PER_BLOCK + h_off
                        i_tile = i_block * I_TILES_PER_BLOCK + i_off
                        ratio = scales[expert, h_tile, i_tile] / retained
                        window = (
                            expert,
                            slice(h_tile * TILE_SIZE, (h_tile + 1) * TILE_SIZE),
                            slice(i_tile * TILE_SIZE, (i_tile + 1) * TILE_SIZE),
                        )
                        wanted = weights_fp32[window] * ratio
                        stored = wanted.to(_FP8).to(_FP32)
                        if not torch.equal(stored, wanted):
                            inexact += 1
                        retiled[window] = stored
                        exponent = pow2_exponent(float(ratio))
                        if exponent is not None:
                            shifts[expert, h_tile, i_tile] = exponent
                # The consumer's scale axis is pre-broadcast to TILE_SIZE copies
                # of one scalar (:2005, :2138 read width 1 over 128 partitions).
                flat = index(h_block * I_TILES_PER_BLOCK, i_block * I_TILES_PER_BLOCK)
                if 0 <= flat < shape[1] // TILE_SIZE:
                    consumer_scales[
                        expert, flat * TILE_SIZE : (flat + 1) * TILE_SIZE
                    ] = retained

    result = RetiledBlockScales(
        consumer_scales=consumer_scales,
        block_scales=block_scales,
        retiled_weights=retiled.to(_FP8),
        tile_exponent_shifts=shifts,
        projection=projection,
        gate_or_up=gate_or_up,
        emitted_unsupplied=0,
        input_scales_dropped=0,
        inexact_rescales=inexact,
        records=tuple(records),
    )
    # Both counted zeros are MEASURED off the emitted artifact, never assumed.
    return RetiledBlockScales(
        consumer_scales=result.consumer_scales,
        block_scales=result.block_scales,
        retiled_weights=result.retiled_weights,
        tile_exponent_shifts=result.tile_exponent_shifts,
        projection=result.projection,
        gate_or_up=result.gate_or_up,
        emitted_unsupplied=count_emitted_unsupplied(result, scales),
        input_scales_dropped=count_input_scales_dropped(result, scales),
        inexact_rescales=result.inexact_rescales,
        records=result.records,
    )


def count_emitted_unsupplied(
    result: RetiledBlockScales, scales: torch.Tensor
) -> int:
    """Emitted slots carrying a value the input did not supply.

    Walks the EMITTED tensor slot by slot and decodes each slot's flat index back
    to ``(i_block, h_block)`` under the declared layout's own flattening -- so it
    never consults the index function that wrote the tensor. Three ways a slot can
    carry an unsupplied value, all counted:

    1. the slot was never written, so the NaN sentinel survives;
    2. it decodes to a block outside the input's block grid;
    3. its value is not the input scale at that block's ``(h_tile 0, i_tile 0)``
       position -- which is what a collision under a wrong flattening produces.

    Case 3 is the one that matters. A mis-transcribed flattening stays inside the
    tensor's extent on a non-square block grid, so a range check alone calls it
    clean; only decoding back to coordinates catches it (the
    ``increments/evidence-071.md`` section 9.2 lesson).

    For :data:`GATE_UP` the population is the slots belonging to THIS call's
    ``gate_or_up`` half; the other half is legitimately unwritten by a single
    un-fused call and is excluded rather than counted as a hole.
    """
    experts, i_256, h_256 = (int(extent) for extent in result.block_scales.shape)
    slots = result.consumer_scales.shape[1] // TILE_SIZE
    count = 0
    for expert in range(experts):
        for flat in range(slots):
            if result.projection == DOWN:
                i_block, h_block = flat // h_256, flat % h_256
            else:
                quotient, i_block = flat // i_256, flat % i_256
                h_block, half = quotient // 2, quotient % 2
                if half != result.gate_or_up:
                    continue
            column = result.consumer_scales[
                expert, flat * TILE_SIZE : (flat + 1) * TILE_SIZE
            ]
            if bool(torch.isnan(column).any()):
                count += 1
                continue
            if not (0 <= h_block < h_256 and 0 <= i_block < i_256):
                count += 1
                continue
            want = scales[
                expert, h_block * I_TILES_PER_BLOCK, i_block * I_TILES_PER_BLOCK
            ]
            if not bool(torch.all(column == want)):
                count += 1
    return count


def count_input_scales_dropped(
    result: RetiledBlockScales, scales: torch.Tensor
) -> int:
    """Input ``[128, 128]`` scales the emitted layout cannot reproduce exactly.

    Measured by actually re-expanding and comparing, never inferred: a scale is
    dropped exactly when :func:`expand_to_checkpoint_scales` does not return it
    bit-for-bit.
    """
    rebuilt = expand_to_checkpoint_scales(result)
    return int((rebuilt != scales).sum().item())


def expand_to_checkpoint_scales(result: RetiledBlockScales) -> torch.Tensor:
    """The inverse: rebuild the ``(E, H//128, I//128)`` checkpoint scale grid.

    ``s = S * 2 ** shift``. Multiplying an fp32 value by an exact power of two
    only changes its exponent field, so the reconstruction is bit-exact wherever
    a shift exists. Where none does, the slot is filled with NaN -- an
    unrepresentable reconstruction is reported, not rounded.

    This round trip is deliberately **not** the losslessness proof: any
    invertible re-layout satisfies it. Losslessness is
    :meth:`RetiledBlockScales.losslessness_fraction`.
    """
    experts, h_tiles, i_tiles = (
        int(extent) for extent in result.tile_exponent_shifts.shape
    )
    out = torch.full((experts, h_tiles, i_tiles), float("nan"), dtype=_FP32)
    for expert in range(experts):
        for h_tile in range(h_tiles):
            for i_tile in range(i_tiles):
                shift = int(result.tile_exponent_shifts[expert, h_tile, i_tile])
                if shift == _NO_SHIFT:
                    continue
                retained = result.block_scales[
                    expert, i_tile // I_TILES_PER_BLOCK, h_tile // I_TILES_PER_BLOCK
                ]
                out[expert, h_tile, i_tile] = retained * _fp32(2.0**shift)
    return out


def count_unrecorded_conjunct2_failures(
    scales: torch.Tensor, records: tuple[BlockLosslessnessRecord, ...]
) -> int:
    """Blocks where conjunct 1 holds and conjunct 2 fails WITHOUT being recorded.

    The population is recomputed here straight from ``scales``, independently of
    ``records``, and the recorded set is subtracted from it. Passing a truncated
    ``records`` drives this non-zero, which is what makes the declared zero a
    measurement rather than a restatement of how the record list was built.
    """
    experts, h_tiles, i_tiles = (int(extent) for extent in scales.shape)
    population: set[tuple[int, int, int]] = set()
    for expert in range(experts):
        for h_block in range(h_tiles // I_TILES_PER_BLOCK):
            for i_block in range(i_tiles // I_TILES_PER_BLOCK):
                retained = float(
                    scales[
                        expert,
                        h_block * I_TILES_PER_BLOCK,
                        i_block * I_TILES_PER_BLOCK,
                    ]
                )
                ratios_pow2 = [
                    is_pow2_exact(
                        _fp32(
                            float(
                                scales[
                                    expert,
                                    h_block * I_TILES_PER_BLOCK + h_off,
                                    i_block * I_TILES_PER_BLOCK + i_off,
                                ]
                            )
                            / retained
                        )
                    )
                    for h_off, i_off in RATIO_ORDER
                ]
                if all(ratios_pow2) and not is_pow2_exact(retained):
                    population.add((expert, h_block, i_block))
    recorded = {
        record.key
        for record in records
        if record.quad_mutually_pow2 and not record.block_scale_pow2
    }
    return len(population - recorded)
