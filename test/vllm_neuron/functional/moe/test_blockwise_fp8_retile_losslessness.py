"""Is the blockwise-fp8 scale RETILE lossless? -- a host-side measurement.

This module measures a mapping. It ships none: no source file changes for it,
and nothing here is imported by the plugin.

WHAT IT COMPARES
----------------
A blockwise-fp8 checkpoint carries one fp32 scale per ``[128, 128]`` weight
block. The consumer kernel carries one fp32 scale per ``256 x 256`` block --
so one kernel scale stands in for **four** checkpoint blocks. Retiling the
checkpoint's scales into the consumer's layout is therefore lossy in general,
and the question this file answers is *under exactly which condition it is not*.

Two dequantisations are built independently and compared:

* **path O -- original semantics.** Each ``[128, 128]`` sub-block is multiplied
  by its own checkpoint scale. Written directly from the checkpoint's
  definition, and deliberately independent of anything the consumer does.
* **path K -- consumer semantics.** The retiled scale tensor is indexed by the
  consumer's *own* block-index arithmetic, transcribed below with its
  ``file:line`` provenance, and applied to the retiled fp8 weights.

Comparing a round trip through the producer's own layout would be invariant to
the claim: any invertible re-layout passes it. Comparing against the original
``[128, 128]`` semantics is the comparison that can fail.

THE RETILE UNDER TEST
---------------------
For each ``256 x 256`` block the producer emits one fp32 scale ``S`` -- the
block's ``(h_tile 0, i_tile 0)`` checkpoint scale -- and rescaled fp8 weights
``w' = fp8(w * (s / S))``. When ``s / S`` is an exact power of two that rescale
only shifts an fp8 exponent, so it is bit-exact, and ``w' * S`` reproduces
``w * s`` bit-for-bit. When it is not, fp8's three mantissa bits round and the
retile loses information. That is the whole mechanism, and every arm below is a
measurement of it.

PROVENANCE
----------
The consumer is the installed ``nkilib`` block-quant matmul
``core/moe/moe_cte/bwmm_shard_on_I.py`` (2,791 L, sha256
``b2b5f7530f7bb46aad0f0e871343b7fdae6b4509712f163a9b3df2d8769c935d``), read and
enumerated at ``increments/wp6-scale-consumer-geometry.md`` sections 2.1-2.3.
Its arithmetic is **transcribed** here, never re-derived: bare ``file:line``
cites below all refer to that file, and constants come from their defining
assignments rather than from inline comments.

DECLARED CONSTRUCTION -- stated before any reading, so no arm is tuned to pass
-----------------------------------------------------------------------------
* fp8 weights are drawn as multiples of ``1/16`` in ``[-7.5, 7.5]`` and pushed
  through ``float8_e4m3fn``, so every weight sits exactly on the fp8 grid.
* Within a ``256 x 256`` block the four checkpoint scales are
  ``base * 2 ** shift`` over the fixed shift table ``[[0, 1], [2, -1]]``, and
  ``base`` differs per block and is **deliberately not a power of two**. So the
  four scales are mutually power-of-two-related without any of them being a
  power of two -- the general case, not the easy one.
* Every generator is seeded, so every number below is reproducible.
* Reductions are accumulated by explicit ascending-index loops rather than by a
  BLAS call, so no reading depends on a library's reduction order.

NON-SCOPE, DECLARED
-------------------
The *satisfying fraction over real checkpoint weights* is not measured here and
cannot be: this file builds **synthetic** scales, so counting how many of them
satisfy the predicate would be counting a property this file chose. That reading
belongs to the first attempt that loads the real checkpoint, at the bar it
already carries.
"""

from __future__ import annotations

import ast
import math
import struct

import pytest
import torch

# --- Constants, from their DEFINING assignments (never from an inline comment) -
# bwmm_shard_on_I.py:50   BLOCK_QUANT_SIZE = 256
#   with the module's own framing at :49 -- "scales are organized as 256x256
#   blocks along (H, I_TP)"
# moe_cte_utils.py:58     TILE_SIZE = 128   (imported at bwmm_shard_on_I.py:31)
BLOCK_QUANT_SIZE = 256
TILE_SIZE = 128
# :1345 / :2110  i_tiles_per_block = BLOCK_QUANT_SIZE // TILE_SIZE
I_TILES_PER_BLOCK = BLOCK_QUANT_SIZE // TILE_SIZE

FP8 = torch.float8_e4m3fn
FP32 = torch.float32

# The three ratio constraints, in the order the design declares them, taken
# against the block's (h_tile 0, i_tile 0) scale: r_k = s_k / s_(0,0) for
# k in {(0,1), (1,0), (1,1)}.
RATIO_ORDER: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1))

# Fixed per-block exponent shift table, indexed [h_offset][i_offset].
SHIFT_TABLE: tuple[tuple[int, int], ...] = ((0, 1), (2, -1))

SEED = 20260830


# --------------------------------------------------------------------------- #
# Path K's index arithmetic -- TRANSCRIBED, not invented.                      #
# --------------------------------------------------------------------------- #
def scale_flat_index_down(
    h_tile_idx: int, i_tile_idx: int, num_h_256_blocks: int
) -> int:
    """Flat scale index for the DOWN projection (reduction axis = I).

    Transcribed verbatim from the consumer:

      :2110  i_tiles_per_block = BLOCK_QUANT_SIZE // TILE_SIZE
      :2112  for i_block in range(GUP_N_TILES // i_tiles_per_block):
      :2116  i_i = i_block * i_tiles_per_block + i_sub
      :2131  h_256_within_H1024 = h_j * (PSUM_SIZE // BLOCK_QUANT_SIZE) + h_256_idx
      :2132  flat_dp_idx = i_block * num_h_256_blocks + h_256_within_H1024
      :2138  operand0=dp_block_scale[0:TILE_SIZE, flat_dp_idx : flat_dp_idx + 1]

    The declared scale shape is ``[E, I_TP//256, H//256, TILE_SIZE]`` (:1987),
    which is why ``i_block`` is the outer term and ``num_h_256_blocks`` the
    stride. ``i_tile_idx`` indexes 128-wide tiles of I -- ``GUP_N_TILES =
    div_ceil(I_TP_sharded, TILE_SIZE)`` (:92) -- i.e. the checkpoint's own
    ``[128, .]`` granularity, so the floor division by ``i_tiles_per_block`` is
    the tile-to-256-block map and nothing else.
    """
    i_tiles_per_block = BLOCK_QUANT_SIZE // TILE_SIZE
    i_block = i_tile_idx // i_tiles_per_block
    h_256_within_H1024 = h_tile_idx // i_tiles_per_block
    return i_block * num_h_256_blocks + h_256_within_H1024


def scale_flat_index_gate_up(
    h_tile_idx: int, i_tile_idx: int, i_blocks_sharded: int, gate_or_up: int = 0
) -> int:
    """Flat scale index for the GATE/UP projection (scales indexed on H and I).

    Transcribed verbatim from the consumer:

      :1345  i_tiles_per_block = BLOCK_QUANT_SIZE // TILE_SIZE
      :1347  h_lin = h_outer_idx * h_inner_tripcount + h_inner_idx
      :1348  h_block = h_lin // i_tiles_per_block
      :1349  i_block = i_tile_idx // i_tiles_per_block
      :1380  i_block = i_tile_idx // (BLOCK_QUANT_SIZE // TILE_SIZE)
      :1381  I_blocks_sharded = dims.I_TP_sharded // BLOCK_QUANT_SIZE
      :1382  flat_scale_idx = (h_block * 2 + gate_or_up) * I_blocks_sharded + i_block
      :1387  operand0=gup_block_scale[0:TILE_SIZE, flat_scale_idx : flat_scale_idx + 1]

    Kept beside the down-projection form because the two are the only sites that
    index a block scale -- the enumeration found four consumption sites (:1387,
    :1396, :2138, :2145) and no fifth. The ``* 2 + gate_or_up`` term is the
    gate/up fusion, so a single un-fused weight uses ``gate_or_up = 0``.
    """
    i_tiles_per_block = BLOCK_QUANT_SIZE // TILE_SIZE
    h_block = h_tile_idx // i_tiles_per_block
    i_block = i_tile_idx // i_tiles_per_block
    return (h_block * 2 + gate_or_up) * i_blocks_sharded + i_block


def scale_flat_index_TRANSPOSED(
    h_tile_idx: int, i_tile_idx: int, num_i_256_blocks: int
) -> int:
    """A DELIBERATELY WRONG flattening: h-major instead of i-major.

    Never used to measure anything. It exists only to prove the transcription
    guard can return a non-zero count, because a guard that cannot fail is not a
    guard. On a non-square block grid it stays inside the tensor's extent while
    reading the wrong scale, which is exactly the failure a range check alone
    cannot see.
    """
    i_tiles_per_block = BLOCK_QUANT_SIZE // TILE_SIZE
    return (h_tile_idx // i_tiles_per_block) * num_i_256_blocks + (
        i_tile_idx // i_tiles_per_block
    )


# --------------------------------------------------------------------------- #
# fp32 bit-level helpers. Everything here works on the stored bit pattern,     #
# because every bar in this file is an exact-equality bar.                     #
# --------------------------------------------------------------------------- #
def fp32(value: float) -> float:
    """Round a Python float (fp64) to the nearest fp32 and return it."""
    return torch.tensor([value], dtype=FP32).item()


def fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", fp32(value)))[0]


def fp32_from_bits(pattern: int) -> float:
    return struct.unpack("<f", struct.pack("<I", pattern & 0xFFFFFFFF))[0]


def fp32_next_up(value: float) -> float:
    """The next fp32 above ``value``: one **fp32** ulp, by bit increment.

    ``math.nextafter`` steps in fp64. An fp64 step off an fp32 value rounds
    straight back to the same fp32 value, so using it here would silently
    perturb nothing and turn the deliberate-mismatch arm into a test that
    always passes. ``test_fp32_step_is_not_the_fp64_step`` pins that.
    """
    value = fp32(value)
    if not value > 0.0:
        raise AssertionError(f"fp32_next_up wants a positive value, got {value!r}")
    return fp32_from_bits(fp32_bits(value) + 1)


def is_pow2_exact(ratio: float) -> bool:
    """Is ``ratio`` an exact power of two? A BIT-PATTERN test.

    ``ratio > 0`` and the IEEE-754 significand field all zeros. Never ``log2``
    and never against a tolerance: a float-log test would reintroduce exactly
    the inexactness this predicate exists to exclude.

    Infinities and NaNs are rejected explicitly. An infinity's significand field
    *is* all zeros, so the bit test alone would call it a power of two.
    """
    ratio = fp32(ratio)
    if not ratio > 0.0 or not math.isfinite(ratio):
        return False
    pattern = fp32_bits(ratio)
    exponent = (pattern >> 23) & 0xFF
    if exponent in (0x00, 0xFF):  # subnormal-or-zero, or inf/NaN
        return False
    return (pattern & 0x007FFFFF) == 0


def is_pow2_frexp(ratio: float) -> bool:
    """The equivalent form the design names, kept as an independent cross-check."""
    ratio = fp32(ratio)
    if not ratio > 0.0 or not math.isfinite(ratio):
        return False
    return math.frexp(ratio)[0] == 0.5


def ulp_distance(left: float, right: float) -> int:
    """Count of representable fp32 values between ``left`` and ``right``."""

    def ordered(value: float) -> int:
        signed = struct.unpack("<i", struct.pack("<f", fp32(value)))[0]
        return signed if signed >= 0 else -2147483648 - signed

    return abs(ordered(left) - ordered(right))


def require_nonzero(count: int, what: str) -> int:
    """A count of zero from an instrument that must see something is a BROKEN
    instrument, not an absence. Fail loudly rather than pass quietly."""
    if count == 0:
        raise AssertionError(
            f"FATAL: broken instrument -- {what} counted 0, which cannot be "
            "right. No verdict is drawn from this reading."
        )
    return count


# --------------------------------------------------------------------------- #
# The synthetic checkpoint, the retile under test, and the two paths.          #
# --------------------------------------------------------------------------- #
def build_checkpoint(
    rows: int, cols: int, base: float = 0.017772, perturb: tuple | None = None
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Synthetic blockwise-fp8 checkpoint: fp8 weights + known per-block scales.

    ``rows`` is the H axis and ``cols`` the I axis, both multiples of 256.
    ``perturb`` is ``((h_tile, i_tile), how)`` with ``how`` in ``{"ulp", "x1.5"}``
    and drives the two negative arms.
    """
    if rows % BLOCK_QUANT_SIZE or cols % BLOCK_QUANT_SIZE:
        raise AssertionError(f"shape [{rows},{cols}] is not 256-blocked")
    dims = {
        "h_tiles": rows // TILE_SIZE,
        "i_tiles": cols // TILE_SIZE,
        "h_256": rows // BLOCK_QUANT_SIZE,
        "i_256": cols // BLOCK_QUANT_SIZE,
    }
    scales = torch.empty((dims["h_tiles"], dims["i_tiles"]), dtype=FP32)
    for h_block in range(dims["h_256"]):
        for i_block in range(dims["i_256"]):
            # Per-block base, deliberately NOT a power of two.
            block_base = fp32(base * (1.0 + 0.25 * (h_block * dims["i_256"] + i_block)))
            for h_off in range(I_TILES_PER_BLOCK):
                for i_off in range(I_TILES_PER_BLOCK):
                    scales[h_block * 2 + h_off, i_block * 2 + i_off] = fp32(
                        block_base * (2.0 ** SHIFT_TABLE[h_off][i_off])
                    )
    if perturb is not None:
        (h_tile, i_tile), how = perturb
        before = float(scales[h_tile, i_tile])
        if how == "ulp":
            scales[h_tile, i_tile] = fp32_next_up(before)
        elif how == "x1.5":
            scales[h_tile, i_tile] = fp32(before * 1.5)
        else:
            raise AssertionError(f"unknown perturbation {how!r}")
        if float(scales[h_tile, i_tile]) == before:
            raise AssertionError(
                f"FATAL: perturbation {how!r} changed nothing at "
                f"({h_tile},{i_tile}); the negative arm would pass vacuously."
            )
    generator = torch.Generator().manual_seed(SEED)
    raw = torch.randint(-120, 121, (rows, cols), generator=generator).to(FP32) / 16.0
    weights = raw.to(FP8).to(FP32)  # exactly on the fp8 grid
    return weights, scales, dims


def retile(
    weights: torch.Tensor, scales: torch.Tensor, dims: dict[str, int]
) -> tuple[torch.Tensor, dict, torch.Tensor, int]:
    """The producer under test: checkpoint layout -> consumer layout.

    Emits one fp32 scale per ``256 x 256`` block -- the block's
    ``(h_tile 0, i_tile 0)`` scale, which is the reference the ratio constraints
    are taken against -- and fp8 weights rescaled by ``s / S``.

    Returns the flat scale tensor, the producer's own supplied map keyed by its
    2-D ``(i_block, h_block)`` coordinates, the retiled weights, and a count of
    blocks whose fp8 rescale was **not** exact.
    """
    supplied: dict[tuple[int, int], float] = {}
    retiled = torch.empty_like(weights)
    inexact = 0
    for h_block in range(dims["h_256"]):
        for i_block in range(dims["i_256"]):
            block_scale = scales[h_block * 2, i_block * 2]
            supplied[(i_block, h_block)] = float(block_scale)
            for h_off in range(I_TILES_PER_BLOCK):
                for i_off in range(I_TILES_PER_BLOCK):
                    h_tile, i_tile = h_block * 2 + h_off, i_block * 2 + i_off
                    ratio = scales[h_tile, i_tile] / block_scale
                    window = (
                        slice(h_tile * TILE_SIZE, (h_tile + 1) * TILE_SIZE),
                        slice(i_tile * TILE_SIZE, (i_tile + 1) * TILE_SIZE),
                    )
                    wanted = weights[window] * ratio
                    stored = wanted.to(FP8).to(FP32)
                    if not torch.equal(stored, wanted):
                        inexact += 1
                    retiled[window] = stored
    # Flat layout follows the declared shape [E, I_TP//256, H//256, TILE_SIZE]
    # (:1987): i_block outer, h_block inner. NaN marks a slot the producer never
    # supplied, so a read of one is detectable rather than merely wrong.
    flat = torch.full((dims["i_256"] * dims["h_256"],), float("nan"), dtype=FP32)
    for (i_block, h_block), value in supplied.items():
        flat[i_block * dims["h_256"] + h_block] = value
    return flat, supplied, retiled, inexact


def dequant_path_o(
    weights: torch.Tensor, scales: torch.Tensor, dims: dict[str, int]
) -> torch.Tensor:
    """Path O: each ``[128, 128]`` sub-block times its own checkpoint scale.

    Written straight from the checkpoint's definition. It never consults the
    consumer's arithmetic, which is what makes the comparison able to fail.
    """
    out = torch.empty_like(weights)
    for h_tile in range(dims["h_tiles"]):
        for i_tile in range(dims["i_tiles"]):
            window = (
                slice(h_tile * TILE_SIZE, (h_tile + 1) * TILE_SIZE),
                slice(i_tile * TILE_SIZE, (i_tile + 1) * TILE_SIZE),
            )
            out[window] = weights[window] * scales[h_tile, i_tile]
    return out


def dequant_path_k(
    retiled: torch.Tensor, flat: torch.Tensor, dims: dict[str, int]
) -> torch.Tensor:
    """Path K: retiled weights times the scale the consumer's index arithmetic
    picks out of the flat scale tensor."""
    out = torch.empty_like(retiled)
    for h_tile in range(dims["h_tiles"]):
        for i_tile in range(dims["i_tiles"]):
            index = scale_flat_index_down(h_tile, i_tile, dims["h_256"])
            window = (
                slice(h_tile * TILE_SIZE, (h_tile + 1) * TILE_SIZE),
                slice(i_tile * TILE_SIZE, (i_tile + 1) * TILE_SIZE),
            )
            out[window] = retiled[window] * flat[index]
    return out


def count_unsupplied_reads(
    dims: dict[str, int],
    flat: torch.Tensor,
    supplied: dict,
    index_fn,
) -> int:
    """Blocks where path K reads a scale the checkpoint never supplied.

    Three ways that can happen, all counted:
      1. the flat index falls outside the tensor's extent;
      2. it lands on a slot the producer never wrote (the NaN sentinel);
      3. it decodes -- under the declared shape's own flattening -- to a
         ``(i_block, h_block)`` pair that is not the block this ``[128, 128]``
         sub-block belongs to.

    Case 3 is the one that matters. A mis-transcribed flattening stays inside the
    extent on a non-square block grid, so a range check alone would call it
    clean; only decoding the index back to coordinates catches it.
    """
    count = 0
    for h_tile in range(dims["h_tiles"]):
        for i_tile in range(dims["i_tiles"]):
            index = index_fn(h_tile, i_tile)
            if not 0 <= index < flat.numel():
                count += 1
                continue
            if math.isnan(float(flat[index])):
                count += 1
                continue
            decoded = (index // dims["h_256"], index % dims["h_256"])
            if decoded not in supplied:
                count += 1
                continue
            if decoded != (i_tile // I_TILES_PER_BLOCK, h_tile // I_TILES_PER_BLOCK):
                count += 1
    return count


def bit_exact_block_count(
    left: torch.Tensor, right: torch.Tensor, block: int
) -> tuple[int, int]:
    """``(k, N)`` -- blocks of side ``block`` where max abs diff is exactly 0."""
    rows, cols = left.shape
    n_h, n_i = rows // block, cols // block
    k = 0
    for h in range(n_h):
        for i in range(n_i):
            window = (
                slice(h * block, (h + 1) * block),
                slice(i * block, (i + 1) * block),
            )
            if (left[window] - right[window]).abs().max().item() == 0.0:
                k += 1
    return k, n_h * n_i


def quad_predicate(
    scales: torch.Tensor, h_block: int, i_block: int
) -> tuple[bool, list[tuple[tuple[int, int], float, bool]]]:
    """The three ratio constraints for one ``256 x 256`` block, in declared order.

    ``r_k = s_k / s_(0,0)`` for ``k`` in ``((0,1), (1,0), (1,1))``, each tested
    by ``is_pow2_exact``. All three must hold: one kernel scale covers four
    checkpoint blocks, so the condition is that all four are mutually
    power-of-two-related, which is three independent constraints and not one.
    """
    reference = scales[h_block * 2, i_block * 2]
    rows: list[tuple[tuple[int, int], float, bool]] = []
    for h_off, i_off in RATIO_ORDER:
        ratio = fp32(
            float(scales[h_block * 2 + h_off, i_block * 2 + i_off] / reference)
        )
        rows.append(((h_off, i_off), ratio, is_pow2_exact(ratio)))
    return all(row[2] for row in rows), rows


def accumulate(
    activations: torch.Tensor, dequantised: torch.Tensor, low: int, high: int
) -> torch.Tensor:
    """Matmul over reduction indices ``[low, high)``, accumulated in ascending
    order by explicit steps. No BLAS call, so the reduction order is this file's
    and no reading depends on a library's choice."""
    out = torch.zeros(
        (activations.shape[0], dequantised.shape[1]), dtype=FP32
    )
    for k in range(low, high):
        out = out + activations[:, k : k + 1] * dequantised[k : k + 1, :]
    return out


# --------------------------------------------------------------------------- #
# Instrument guards. These do not measure the retile; they pin the             #
# instruments that measure it, so a silent instrument cannot read as a pass.   #
# --------------------------------------------------------------------------- #
def test_is_pow2_exact_is_a_bit_pattern_test() -> None:
    """The predicate is a bit-pattern test, and the frexp form agrees with it."""
    table = {
        1.0: True,
        2.0: True,
        0.5: True,
        2.0**-30: True,
        2.0**20: True,
        1.5: False,
        3.0: False,
        0.75: False,
        1.0 + 2.0**-23: False,  # one fp32 ulp above a power of two
        0.0: False,
        -2.0: False,
        float("inf"): False,  # significand IS all zeros -- must still be False
        float("nan"): False,
    }
    wrong = {r: is_pow2_exact(r) for r, want in table.items() if is_pow2_exact(r) is not want}
    assert not wrong, f"is_pow2_exact disagrees with the declared table at {wrong}"
    disagree = {r: (is_pow2_exact(r), is_pow2_frexp(r)) for r in table if is_pow2_exact(r) != is_pow2_frexp(r)}
    assert not disagree, f"bit-pattern and frexp forms disagree at {disagree}"
    # No log-based power-of-two test anywhere in this module, pinned over the
    # parse tree rather than over the text -- a textual screen would trip on the
    # word "log2" in the prose above, and would miss an aliased import.
    with open(__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    banned = {"log2", "log", "log10", "frexp"}
    allowed = [
        (node.lineno, node.end_lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "is_pow2_frexp"
    ]
    assert len(allowed) == 1, "is_pow2_frexp is the single permitted frexp site"
    low, high = allowed[0]
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name in banned:
            calls.append((name, node.lineno))
    # frexp is permitted in exactly one place: the declared cross-check form.
    stray = [
        (n, ln) for n, ln in calls if not (n == "frexp" and low <= ln <= high)
    ]
    assert not stray, f"a log-based power-of-two test crept in at {stray}"
    require_nonzero(len(calls), "log-family calls located by the ast screen")


def test_fp32_step_is_not_the_fp64_step() -> None:
    """An fp64 ``nextafter`` off an fp32 value perturbs nothing after rounding.

    This is the trap that would make the deliberate-mismatch arm pass vacuously,
    so it is pinned here rather than trusted.
    """
    value = fp32(0.017772 * 0.5)
    assert fp32(math.nextafter(value, math.inf)) == value, (
        "fp64 nextafter unexpectedly survived fp32 rounding; the reasoning "
        "behind fp32_next_up needs re-checking"
    )
    stepped = fp32_next_up(value)
    assert stepped != value
    assert ulp_distance(value, stepped) == 1


def test_retile_rescale_is_exact_only_for_pow2_ratios() -> None:
    """The mechanism itself: an exact power-of-two ratio shifts an fp8 exponent
    and loses nothing; a 1.5x ratio rounds in fp8's three mantissa bits."""
    generator = torch.Generator().manual_seed(SEED)
    grid = (
        torch.randint(-120, 121, (4096,), generator=generator).to(FP32) / 16.0
    ).to(FP8).to(FP32)
    for shift in (-2, -1, 0, 1, 2, 3):
        ratio = fp32(2.0**shift)
        wanted = grid * ratio
        assert torch.equal(wanted.to(FP8).to(FP32), wanted), (
            f"fp8 rescale by 2**{shift} was not exact"
        )
    wanted = grid * fp32(1.5)
    rounded = int((wanted.to(FP8).to(FP32) != wanted).sum().item())
    require_nonzero(rounded, "fp8 elements rounded by a 1.5x rescale")
    print(f"[mechanism] 1.5x rescale rounded {rounded}/{grid.numel()} fp8 elements")


# --------------------------------------------------------------------------- #
# M-1 -- bit-exact over the [128,128] blocks of two shapes.                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("side", "expected_blocks"), [(512, 16), (1024, 64)])
def test_m1_bit_exact_over_128_blocks(side: int, expected_blocks: int) -> None:
    weights, scales, dims = build_checkpoint(side, side)
    flat, supplied, retiled, inexact = retile(weights, scales, dims)
    assert inexact == 0, f"{inexact} blocks retiled inexactly; the retile precondition fails"
    path_o = dequant_path_o(weights, scales, dims)
    path_k = dequant_path_k(retiled, flat, dims)
    max_abs_diff = (path_o - path_k).abs().max().item()
    k, n = bit_exact_block_count(path_o, path_k, TILE_SIZE)
    print(f"[M-1] [{side},{side}] max abs diff = {max_abs_diff!r}  bit-exact blocks = {k}/{n}")
    assert n == expected_blocks, f"shape [{side},{side}] gave {n} blocks, expected {expected_blocks}"
    assert max_abs_diff == 0.0, f"max abs diff(O, K) = {max_abs_diff!r}, required exactly 0.0"
    assert k == n, f"bit-exact over {k}/{n} blocks, required {n}/{n}"


# --------------------------------------------------------------------------- #
# M-2 -- split halves of the reduction axis, and the summed ulp distance.      #
# --------------------------------------------------------------------------- #
def test_m2_split_halves_and_summed_ulp_distance() -> None:
    """Two partials over the reduction axis, each bit-exact, and the summed
    result within 1 fp32 ulp of path O -- with the measured distance reported."""
    reduction = BLOCK_QUANT_SIZE  # I, split 2/2 into the consumer's 128-tiles
    weights, scales, dims = build_checkpoint(reduction, BLOCK_QUANT_SIZE)
    flat, supplied, retiled, inexact = retile(weights, scales, dims)
    assert inexact == 0
    path_o = dequant_path_o(weights, scales, dims)
    path_k = dequant_path_k(retiled, flat, dims)
    generator = torch.Generator().manual_seed(SEED + 1)
    activations = torch.randint(-32, 33, (4, reduction), generator=generator).to(FP32) / 8.0

    halves = ((0, TILE_SIZE), (TILE_SIZE, 2 * TILE_SIZE))
    partials_o = [accumulate(activations, path_o, lo, hi) for lo, hi in halves]
    partials_k = [accumulate(activations, path_k, lo, hi) for lo, hi in halves]
    exact = sum(
        1 for a, b in zip(partials_o, partials_k) if (a - b).abs().max().item() == 0.0
    )
    total_o = partials_o[0] + partials_o[1]
    total_k = partials_k[0] + partials_k[1]
    measured_ulp = max(
        ulp_distance(a, b)
        for a, b in zip(total_o.flatten().tolist(), total_k.flatten().tolist())
    )
    print(f"[M-2] partials bit-exact = {exact}/2  summed max ulp distance = {measured_ulp}")
    assert exact == 2, f"partials bit-exact over {exact}/2 split halves, required 2/2"
    assert measured_ulp <= 1, f"summed result is {measured_ulp} fp32 ulp from path O, bound is 1"


# --------------------------------------------------------------------------- #
# M-3 part 1 -- the quad predicate, three ratios per block in declared order.  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("side", "expected_blocks"), [(512, 4), (1024, 16)])
def test_m3_part1_quad_predicate(side: int, expected_blocks: int) -> None:
    _, scales, dims = build_checkpoint(side, side)
    n = dims["h_256"] * dims["i_256"]
    assert n == expected_blocks, f"shape [{side},{side}] gave {n} 256-blocks, expected {expected_blocks}"
    satisfied = 0
    checked = 0
    for h_block in range(dims["h_256"]):
        for i_block in range(dims["i_256"]):
            ok, rows = quad_predicate(scales, h_block, i_block)
            assert [row[0] for row in rows] == list(RATIO_ORDER), (
                f"ratios evaluated in {[row[0] for row in rows]}, "
                f"declared order is {list(RATIO_ORDER)}"
            )
            checked += len(rows)
            satisfied += int(ok)
            if not ok:
                failing = [(k, r) for k, r, good in rows if not good]
                raise AssertionError(
                    f"block (h={h_block}, i={i_block}) is not mutually "
                    f"power-of-two-related; failing ratios {failing}"
                )
    require_nonzero(checked, "ratio constraints evaluated")
    print(f"[M-3.1] quad predicate satisfied = {satisfied}/{n}  ratio constraints checked = {checked}")
    assert checked == 3 * n, f"{checked} constraints over {n} blocks, expected {3 * n}"
    assert satisfied == n, f"quad predicate holds on {satisfied}/{n} blocks, required {n}/{n}"


# --------------------------------------------------------------------------- #
# M-3 part 2 -- the mechanism, bit-exact over the 256-blocks of two shapes.    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("side", "expected_blocks"), [(512, 4), (1024, 16)])
def test_m3_part2_mechanism_bit_exact_over_256_blocks(side: int, expected_blocks: int) -> None:
    weights, scales, dims = build_checkpoint(side, side)
    flat, supplied, retiled, inexact = retile(weights, scales, dims)
    assert inexact == 0
    path_o = dequant_path_o(weights, scales, dims)
    path_k = dequant_path_k(retiled, flat, dims)
    max_abs_diff = (path_o - path_k).abs().max().item()
    k, n = bit_exact_block_count(path_o, path_k, BLOCK_QUANT_SIZE)
    percent = 100.0 * k / n
    print(f"[M-3.2] [{side},{side}] max abs diff = {max_abs_diff!r}  k = {k} (int)  N = {n}  k/N = {percent:.1f}%")
    assert isinstance(k, int)
    assert n == expected_blocks, f"shape [{side},{side}] gave {n} 256-blocks, expected {expected_blocks}"
    assert max_abs_diff == 0.0, f"max abs diff(O, K) = {max_abs_diff!r}, required exactly 0.0"
    assert k == n, f"k/N = {k}/{n} = {percent:.1f}%, required 100%"


# --------------------------------------------------------------------------- #
# M-3 part 3 -- the DETECTOR. Without it part 2 is a tautology over scales     #
# this file chose.                                                            #
# --------------------------------------------------------------------------- #
def test_m3_part3_detector_non_pow2_ratio_breaks_path_k() -> None:
    """One of the four scales moved to a 1.5x (non-pow2) ratio: path K must
    differ from path O, and the predicate must reject that block."""
    perturbed = (1, 1)  # inside 256-block (0, 0), so its ratio is a declared one
    weights, scales, dims = build_checkpoint(512, 512, perturb=(perturbed, "x1.5"))
    flat, supplied, retiled, inexact = retile(weights, scales, dims)
    path_o = dequant_path_o(weights, scales, dims)
    path_k = dequant_path_k(retiled, flat, dims)

    window = (
        slice(perturbed[0] * TILE_SIZE, (perturbed[0] + 1) * TILE_SIZE),
        slice(perturbed[1] * TILE_SIZE, (perturbed[1] + 1) * TILE_SIZE),
    )
    block_diff = (path_o[window] - path_k[window]).abs().max().item()
    ok, rows = quad_predicate(scales, 0, 0)
    ratio = dict(((k, r) for k, r, _ in rows))[perturbed]
    rejected = sum(1 for k, _, good in rows if k == perturbed and not good)

    print(
        f"[M-3.3] detector: perturbed ratio r{perturbed} = {ratio!r}  "
        f"is_pow2_exact = {is_pow2_exact(ratio)}  rejected = {rejected}/1  "
        f"max abs diff = {block_diff!r} in 1/1  retile-inexact blocks = {inexact}"
    )
    assert block_diff > 0.0, (
        "a non-pow2 ratio left path K bit-identical to path O in 1/1 -- the "
        "measurement cannot detect the loss it exists to find"
    )
    assert rejected == 1, f"is_pow2_exact rejected the perturbed ratio {rejected}/1 times, required 1/1"
    assert not ok, "the quad predicate accepted a block with a 1.5x ratio"
    assert inexact == 1, f"{inexact} blocks retiled inexactly, expected exactly 1"


# --------------------------------------------------------------------------- #
# The deliberate-mismatch negative case -- a separate arm from the detector.   #
# --------------------------------------------------------------------------- #
def test_negative_case_one_ulp_scale_perturbation_is_detected() -> None:
    """One ``[128,128]`` scale moved by a single fp32 ulp must break the
    comparison. Without this, a bit-exact pass could be an artefact of comparing
    a tensor with itself."""
    perturbed = (1, 1)
    weights, scales, dims = build_checkpoint(512, 512, perturb=(perturbed, "ulp"))
    flat, supplied, retiled, inexact = retile(weights, scales, dims)
    path_o = dequant_path_o(weights, scales, dims)
    path_k = dequant_path_k(retiled, flat, dims)
    max_abs_diff = (path_o - path_k).abs().max().item()
    differing = int((path_o != path_k).sum().item())
    _, rows = quad_predicate(scales, 0, 0)
    ratio = dict(((k, r) for k, r, _ in rows))[perturbed]
    k, n = bit_exact_block_count(path_o, path_k, TILE_SIZE)
    print(
        f"[negative] 1-ulp perturbation: ratio = {ratio!r}  "
        f"is_pow2_exact = {is_pow2_exact(ratio)}  max abs diff = {max_abs_diff!r}  "
        f"differing elements = {differing}  bit-exact blocks = {k}/{n}"
    )
    assert max_abs_diff > 0.0, "a 1-ulp scale perturbation went undetected"
    require_nonzero(differing, "elements differing under a 1-ulp perturbation")
    assert not is_pow2_exact(ratio), f"ratio {ratio!r} was still called an exact power of two"
    assert k == n - 1, f"{k}/{n} blocks bit-exact, expected exactly one block to break"


# --------------------------------------------------------------------------- #
# The counted zero, and the proof that its guard can fail.                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("rows", "cols", "perturb"),
    [
        (512, 512, None),
        (1024, 1024, None),
        (512, 1024, None),  # non-square block grid: 2 h-blocks x 4 i-blocks
        (512, 512, ((1, 1), "x1.5")),
        (512, 512, ((1, 1), "ulp")),
    ],
)
def test_counted_zero_path_k_never_reads_an_unsupplied_scale(rows, cols, perturb) -> None:
    """Exactly 0, on every branch -- the transcription guard on the index
    function."""
    weights, scales, dims = build_checkpoint(rows, cols, perturb=perturb)
    flat, supplied, retiled, _ = retile(weights, scales, dims)
    count = count_unsupplied_reads(
        dims, flat, supplied, lambda h, i: scale_flat_index_down(h, i, dims["h_256"])
    )
    blocks = dims["h_tiles"] * dims["i_tiles"]
    print(f"[counted-zero] [{rows},{cols}] perturb={perturb} unsupplied reads = {count} over {blocks} blocks")
    assert count == 0, f"path K read a scale the checkpoint never supplied in {count} blocks"


def test_counted_zero_guard_is_live() -> None:
    """The same counter over a deliberately transposed flattening must return a
    NON-zero count, on a non-square block grid where every wrong index still
    lands inside the tensor. A guard that cannot fail proves nothing."""
    weights, scales, dims = build_checkpoint(512, 1024)
    flat, supplied, retiled, _ = retile(weights, scales, dims)
    assert dims["h_256"] != dims["i_256"], "grid must be non-square for this probe to bite"

    correct = count_unsupplied_reads(
        dims, flat, supplied, lambda h, i: scale_flat_index_down(h, i, dims["h_256"])
    )
    transposed = count_unsupplied_reads(
        dims, flat, supplied, lambda h, i: scale_flat_index_TRANSPOSED(h, i, dims["i_256"])
    )
    in_range = all(
        0 <= scale_flat_index_TRANSPOSED(h, i, dims["i_256"]) < flat.numel()
        for h in range(dims["h_tiles"])
        for i in range(dims["i_tiles"])
    )
    print(
        f"[liveness] grid h_256={dims['h_256']} i_256={dims['i_256']} flat={flat.numel()}  "
        f"correct = {correct}  transposed = {transposed}  transposed-stays-in-range = {in_range}"
    )
    assert correct == 0
    assert in_range, "the transposed index left the extent, so a range check would have caught it"
    require_nonzero(transposed, "unsupplied reads under a transposed flattening")


def test_gate_up_index_agrees_with_down_index_on_the_2d_block_map() -> None:
    """Both transcribed consumption paths reduce tiles to the same
    ``(h_block, i_block)`` pair; only the flattening differs. Kept so the
    enumeration is not silently narrowed to one of the two."""
    checked = 0
    for h_tile in range(8):
        for i_tile in range(8):
            down = scale_flat_index_down(h_tile, i_tile, 4)
            gate_up = scale_flat_index_gate_up(h_tile, i_tile, 4, gate_or_up=0)
            assert down % 4 == h_tile // I_TILES_PER_BLOCK
            assert down // 4 == i_tile // I_TILES_PER_BLOCK
            assert gate_up % 4 == i_tile // I_TILES_PER_BLOCK
            assert gate_up // 4 == 2 * (h_tile // I_TILES_PER_BLOCK)
            checked += 1
    require_nonzero(checked, "tile pairs checked across both index forms")
    print(f"[index] both transcribed forms agree on the 2-D block map over {checked} tile pairs")
