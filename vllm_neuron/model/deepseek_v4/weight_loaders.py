# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4 weight loaders
==========================

Checkpoint -> parameter transforms for ``deepseek-ai/DeepSeek-V4-Flash-0731``
(pinned revision ``7872f01b1d1fe23eabc4c98b48bffcef5a386062``, 72317 keys over
48 shards).

Why this module exists as its own file, and why it is this long: the
checkpoint's key convention, its two coexisting quantized weight formats and
its per-layer optional tensors are all *facts about the checkpoint*, not
facts about the math. Keeping them here means ``attention.py`` / ``moe.py`` /
``model.py`` never branch on checkpoint shape, and a checkpoint revision bump
is a one-file diff.

Primary evidence used (this outranks any derived spec):

* ``model.safetensors.index.json`` of the pinned revision -- the authoritative
  key inventory (which layers have which optional tensors).
* DeepSeek's own reference implementation shipped in the checkpoint repo:
  ``convert.py`` (the key-renaming convention, the FP4 value table, the
  nibble order) and ``model.py`` (every parameter's declared shape) and
  ``kernel.py`` (how the block scale is applied, hence its orientation).

Six facts that a "port it like llama3" reflex gets wrong, each enforced
below:

1. **Keys are NOT HF-standard.** There is no ``model.`` prefix and no
   ``self_attn``. DeepSeek's ``convert.py`` (lines 89-96) is the convention:
   strip ``model.``, ``self_attn`` -> ``attn``, ``mlp`` -> ``ffn``,
   ``weight_scale_inv`` -> ``scale``, ``e_score_correction_bias`` -> ``bias``.
   So the real keys are ``layers.N.attn.wq_a.weight``,
   ``layers.N.ffn.experts.E.w1.weight``, ``embed.weight``, ``head.weight``,
   ``norm.weight``, ``layers.N.attn_norm.weight``, ``layers.N.ffn_norm.weight``.
2. **Block scales live at ``.scale``, not ``.weight_scale_inv``**, and despite
   the ``_inv`` in the pre-rename name they are *multipliers*: the reference
   dequantizes with ``weight * scale`` (``convert.py:126``) and the GEMM
   multiplies the accumulator by them (``kernel.py:242-249``). They are stored
   ``float8_e8m0fnu`` and are converted here to an fp32 power of two.
3. **Weights are stored ``[out, in]``** (``model.py:145`` ``Linear`` declares
   ``empty(out_features, in_features)``), which is already the orientation
   ``NF.block_fp8_linear(x, weight_fp8[N_local, K_local], ...)`` wants. Do NOT
   pass ``is_storage_transposed=True`` for them. llama3 does pass it, because
   ITS parameters are ``[in, out]``; ours are not. Copying that flag over
   would transpose every projection and produce silent garbage.
4. **Routed experts are MXFP4-packed and are upcast to MXFP8 at load**, because
   Trainium2 has no FP4 datapath. The upcast happens once, on the host, not per
   forward.
5. **Optional tensors are genuinely optional.** ``ffn.gate.bias`` is absent on
   layers 0, 1, 2 and ``ffn.gate.tid2eid`` exists ONLY there; the KV compressor
   is absent on layers 0, 1; the DSA indexer exists only on ratio-4 layers; and
   the checkpoint carries three MTP blocks while the config declares one.
   Neither absence is an error, so nothing here asks the checkpoint for a key
   the index says is missing.
6. **The stored FP8 bytes are OCP-448 and this venue's are legacy-240**, and the
   two disagree ONLY at exponent field 15, which is finite in the checkpoint's
   grid and inf/NaN in trn2's. BOTH quantized slices re-encode, neither is
   exempt: the routed experts by bitwise exponent folding (LD-23), the
   dense/attention slice by halving through
   :data:`_OCP_TO_LEGACY_HALVED_BYTES` with its block scale doubled (LD-24).
   See "THE 1-BYTE CARRIER DOCTRINE" below. Believing the dense slice was
   exempt is what produced this campaign's third replan.

Public entry points (fixed by the family interface contract; everything else
in this module is private):

* :func:`attach_attention_loaders`
* :func:`attach_moe_loaders`
* :func:`attach_hash_context_loaders`
* :func:`build_checkpoint_mappings`
* :func:`load_block_scale_buffers`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch import nn

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    expert_parallel_grouped_loader,
    set_weight_loader,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    from .config import DeepseekV4Config

logger = logging.getLogger(__name__)

__all__ = [
    "attach_attention_loaders",
    "attach_moe_loaders",
    "attach_hash_context_loaders",
    "build_checkpoint_mappings",
    "load_block_scale_buffers",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Block extent of the 128x128 FP8 weight-scale grid. The checkpoint's
#: ``quantization_config.weight_block_size`` is ``[128, 128]``; the reference
#: ``Linear`` derives the scale shape as
#: ``(ceil(out/128), ceil(in/128))`` (dsv4_ref/model.py:146-148).
_BLOCK = 128

#: MX group size for the routed experts: one E8M0 scale per 32 FP4 elements
#: along K (dsv4_ref/model.py:139-143, and ``fp4_block_size = 32`` in
#: dsv4_ref/convert.py:26).
_MX_GROUP = 32

_FP8_DTYPE = torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# THE 1-BYTE CARRIER DOCTRINE — SHARED BY BOTH QUANTIZED SLICES (LD-24)
#
# Promoted to module level at iteration 5. It used to live only on the LD-23
# expert path, and that placement is exactly how this campaign shipped a defect:
# a reader (and a planner) concluded the dense/attention slice was deliberately
# exempt. IT IS NOT. Both slices obey every clause below.
#
# THE DTYPE IS A CARRIER, NOT AN ENCODING CLAIM. Every 1-byte parameter in this
# family is declared ``torch.float8_e4m3fn`` because that is the only 1-byte
# float torch has, and because it is the KEY that makes
# ``torch_to_nki_dtype`` yield ``nl.float8_e4m3`` on trn2
# (``nki/nki_dtype.py:43,51-53``). The BYTES inside it are LEGACY
# ``nl.float8_e4m3``: bias 7, exponent field 1..14 for normals, field 15
# RESERVED for inf/NaN, amax ``1.875 * 2**7 = 240``. OCP ``float8_e4m3fn``
# instead makes field 15 FINITE (256..448, NaN only at mantissa 7), amax 448.
# The two grids are bit-for-bit IDENTICAL for field 0..14 and disagree ONLY at
# field 15 (port-assessment.md section 2.1, measured leg A).
#
# THEREFORE: never route a weight value through torch's own fp8 cast unless it
# is already known to sit at field <= 14. torch's cast is the OCP encoding and
# would place a large value at field 15, which decodes as inf/NaN on this venue.
# Move exponent fields inside bytes instead.
#
# AND: "it stays torch-side, so the NKI dtype mapper never reaches it" is NOT a
# safety argument on trn2. ``compile/backend.py:690-714`` injects
# ``--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`` into the compiler
# unconditionally on this target, so a plain torch fp8 convert INSIDE the
# compiled graph is legacy-reinterpreted as well (port-assessment.md section
# 2.8). Field 15 is fatal on both paths.
#
# WHAT EACH SLICE DOES ABOUT IT:
#   * routed experts (LD-23) -- never touch torch's fp8 cast at all; the bytes
#     are assembled from :data:`_FP4_TO_FP8_BYTES` and their exponent fields are
#     shifted bitwise into ``[_E4M3_FIELD_MIN, _E4M3_FIELD_MAX]``.
#   * dense / attention (LD-24) -- the checkpoint stores OCP-448 bytes by
#     design (its own quantizer clamps at 448, ``dsv4_ref/kernel.py:47-48``;
#     82.6% of 128x128 blocks hold at least one field-15 byte, measured leg B),
#     so every byte is re-encoded AT LOAD into the legacy grid by
#     :data:`_OCP_TO_LEGACY_HALVED_BYTES`, with the paired block scale doubled
#     to absorb the halving.
# ---------------------------------------------------------------------------

#: Legacy ``nl.float8_e4m3`` biased-exponent field window for NORMAL values.
#: Field 0 is zero/subnormal and field 15 is inf/NaN, so a normal occupies
#: ``1..14``. Shared by both quantized slices; see the carrier doctrine above.
_E4M3_FIELD_MIN = 1
_E4M3_FIELD_MAX = 14

#: The reserved field, named once so no site re-derives ``15``.
_E4M3_FIELD_NAN = 15

# RETIRED (LD-23): the MX tile geometry constants (``_MX_PMAX``,
# ``_MX_Q_WIDTH``, ``_MX_Q_HEIGHT``, ``_MX_TILE_K``, ``_MX_BYTES_PER_WORD``)
# lived here to drive the four ``_tile_mx_*`` transforms. Those transforms
# existed only to feed the MX expert kernels, which lower to NeuronCore-v4
# instructions and cannot execute on this campaign's trn2 (= NeuronCore-v3)
# venue at all -- see the LD-23 section header and R-13. The gen3-legal path
# takes PLAIN layouts, so there is no tiling left to parameterize and the
# constants are removed rather than left as dead code that reads like a live
# contract.

#: FP4 (E2M1) code -> value table, verbatim from DeepSeek's own
#: ``convert.py:11-14``. Note index 8 (the "negative zero" code) maps to
#: ``+0.0`` there, NOT ``-0.0``; we reproduce that exactly rather than
#: re-deriving the table from the OCP spec, so the upcast is bit-identical to
#: the reference conversion.
_FP4_TABLE: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

#: The same table as raw ``float8_e4m3fn`` byte patterns. Every FP4 magnitude
#: (0.5 .. 6.0) is exactly representable in E4M3, so the upcast is lossless --
#: which is why the reference calls its own variant
#: ``cast_e2m1fn_to_e4m3fn`` "losslessly" (convert.py:17-19).
#:
#: We keep the *bytes* (not the fp8 values) because several torch CPU builds
#: lack float8 kernels for ``index_select`` / ``stack``; gathering and stacking
#: in ``uint8`` and reinterpreting once at the end is bit-equivalent and always
#: available. Same reasoning as
#: ``llama3/weight_pack_mx_fp8.py:121-124`` ("route the gather through the byte
#: view, which is bit-equivalent").
_EXPECTED_FP4_FP8_BYTES: tuple[int, ...] = (
    0x00,
    0x30,
    0x38,
    0x3C,
    0x40,
    0x44,
    0x48,
    0x4C,
    0x00,
    0xB0,
    0xB8,
    0xBC,
    0xC0,
    0xC4,
    0xC8,
    0xCC,
)


def _build_fp4_to_fp8_bytes() -> torch.Tensor:
    """Derive the FP4-code -> FP8-byte table and cross-check the constant.

    Derived from :data:`_FP4_TABLE` through torch's own fp8 conversion, then
    asserted against :data:`_EXPECTED_FP4_FP8_BYTES`. Two independent
    derivations of the same 16 bytes is cheap insurance: a single transposed
    nibble in a hand-written table is invisible in review and shows up only as
    degraded output quality after a multi-thousand-second compile.
    """
    table = (
        torch.tensor(_FP4_TABLE, dtype=torch.float32).to(_FP8_DTYPE).view(torch.uint8)
    )
    expected = torch.tensor(_EXPECTED_FP4_FP8_BYTES, dtype=torch.uint8)
    if not torch.equal(table, expected):
        raise RuntimeError(
            "FP4->FP8 byte table mismatch: torch produced "
            f"{[hex(b) for b in table.tolist()]} but this module documents "
            f"{[hex(b) for b in expected.tolist()]}. One of the two is wrong; "
            "do not load experts until it is resolved."
        )
    return table


_FP4_TO_FP8_BYTES = _build_fp4_to_fp8_bytes()


# ---------------------------------------------------------------------------
# LD-24: OCP-448 checkpoint bytes -> legacy nl.float8_e4m3 bytes, halved
#
# WHY (port-assessment.md sections 2.1-2.4). The checkpoint's dense/attention
# block-FP8 bytes are OCP ``float8_e4m3fn``, amax 448, and it emits exponent
# field 15 BY DESIGN -- its own quantizer clamps at +-448
# (``dsv4_ref/kernel.py:47-48``) and 423 of 512 sampled 128x128 blocks (82.6%)
# hold at least one field-15 byte (measured leg B). On trn2 field 15 is
# inf/NaN, so those bytes decode to NaN in compiled code and one NaN weight
# element poisons its whole output column (32 of 1024 outputs non-finite,
# measured leg E1).
#
# THE REMEDY IS AT LOAD, NOT IN THE KERNEL, AND IT IS UNCONDITIONAL. Halve every
# element of every block into the legacy grid and DOUBLE that block's scale to
# absorb it. It cannot be conditional: the weight transform and the scale
# transform are two separate ``SafetensorsWeightLoader`` objects with no shared
# state, and "does this block hold a field-15 byte?" is derivable only from the
# weight bytes, which the scale loader never sees. A disagreement between them
# is silent -- no raise, plausible logits, and only a parity miss to show it.
# Unconditionality also buys commutation: a uniform per-element transform
# commutes with every downstream slice, ``cat`` and shard, so ``attention.py``'s
# fused slice, ``dspark_model.py``'s complementary slice, the fused N-concat and
# ``_grid_shard``'s shared sub-block row all stay untouched.
#
# WHY THE FACTOR IS 2 AND NOT ``240/448``. ``llama3/weight_loaders_static_fp8.py
# :47-70`` solves the same problem for per-TENSOR fp32 scales and uses
# ``240/448``. That factor is wrong here: ``240/448 = 0.5357`` is not a power of
# two, so the compensated block scale would stop being exactly representable as
# ue8m0 and ``NF.block_fp8_linear``'s "weight dequantization introduces no
# rounding" clause would fail. Halving is the only in-range factor that keeps
# the scale ue8m0-exact, and it always fits because ``|ocp| / 2 <= 224 < 240``.
#
# DO NOT IMPLEMENT THIS AS ARITHMETIC ON THE BYTE. ``byte - 8`` is the obvious
# form -- decrement the exponent field -- and it is MEASURED WRONG (leg E2a,
# ``artifacts/repairs/author_kernel_triads-iter1/triads1-ld11-remedy.txt``):
# still 16 of 1024 non-finite outputs, because a field-0 byte underflows the
# exponent field into the SIGN BIT and manufactures a fresh field-15 (NaN)
# pattern. It turned 2 previously-good bytes into NaN while fixing 4. In the
# sampled block 6 of 16384 live bytes sat at field <= 1, so the trap is
# reachable, not exotic. The table below is GENERATED from a float64 reference
# and then asserted exhaustively; the "sign bit preserved for all 256 inputs"
# clause is what kills that shortcut mechanically (R-20).
#
# NOT BIT-EXACT, AND EXACTLY HOW INEXACT. A one-binade shift is exact for every
# byte at field >= 2. Loss occurs only at field <= 1 (``|w| < 2**-5`` in grid
# units) and is at most half a subnormal step, ``2**-10`` in grid units, i.e.
# <= 8.1e-06 of a block whose largest element may be ``240 * scale``. Measured
# on a real block: 16381 of 16384 elements halve EXACTLY (99.9817%), the 3
# inexact ones being the smallest in the block, ``max_rel`` 1.32114e-06 and 0 of
# 1024 non-finite outputs (leg E2b). Recorded as production-config dimension 4
# deviation D-2, planner authority (port-assessment.md section 4).
# ---------------------------------------------------------------------------

#: The two bytes that are NaN in OCP ``float8_e4m3fn`` (exponent field 15,
#: mantissa 7). Legacy ``nl.float8_e4m3`` spells NaN with the same two patterns,
#: so under BOTH grids these mean NaN and there is no value to halve. They are
#: REFUSED at load rather than mapped to anything finite: a NaN weight that
#: loads silently poisons every activation that touches its block and has no
#: other symptom -- the same reason ``_e8m0_to_fp32`` refuses E8M0 code 255.
_OCP_NAN_BYTES: tuple[int, ...] = (0x7F, 0xFF)


def _legacy_e4m3_magnitudes() -> tuple[float, ...]:
    """Every non-negative legacy ``nl.float8_e4m3`` value, indexed by its byte.

    Bytes ``0x00..0x77``, which is ascending VALUE order too: the exponent field
    sits above the mantissa, so the byte is a monotone encoding of the
    magnitude. Field 0 is zero/subnormal (``m/8 * 2**-6``); fields
    ``_E4M3_FIELD_MIN.._E4M3_FIELD_MAX`` are normals
    (``(1 + m/8) * 2**(f-7)``). Field 15 is deliberately ABSENT -- in this grid
    it is inf/NaN, not a value, which is the whole point of LD-24.

    Every entry is a dyadic rational with a small exponent, so float64 holds all
    of them exactly and the rounding search below has no floating-point error to
    account for.
    """
    values = []
    for f in range(_E4M3_FIELD_NAN):  # 0 .. 14
        for m in range(8):
            if f == 0:
                values.append(m / 8.0 * 2.0**-6)
            else:
                values.append((1.0 + m / 8.0) * 2.0 ** (f - 7))
    return tuple(values)


def _ocp_e4m3fn_magnitude(f: int, m: int) -> float:
    """The OCP ``float8_e4m3fn`` magnitude of exponent field ``f``, mantissa ``m``.

    Identical to :func:`_legacy_e4m3_magnitudes` for ``f`` in ``0..14`` -- the
    two grids agree bit-for-bit there -- and defined for ``f == 15`` as well,
    where OCP is FINITE (``256..448``) and legacy is not. ``(15, 7)`` is OCP's
    NaN and has no magnitude; callers must exclude it first.
    """
    if f == _E4M3_FIELD_NAN and m == 7:
        raise ValueError("OCP field 15 mantissa 7 is NaN, not a magnitude")
    if f == 0:
        return m / 8.0 * 2.0**-6
    return (1.0 + m / 8.0) * 2.0 ** (f - 7)


def _encode_legacy_rne(value: float, magnitudes: Sequence[float]) -> int:
    """Round a non-negative ``value`` to its nearest legacy byte, ties to even.

    A linear search over 120 candidates, run 256 times at import: microseconds,
    and it needs no bit reasoning at all, which is the point. ``magnitudes`` is
    ascending, so "ties to even" is exactly "prefer the even byte", the standard
    round-to-nearest-even rule stated on the encoding rather than on the value.
    """
    best, best_err = 0, abs(magnitudes[0] - value)
    for byte in range(1, len(magnitudes)):
        err = abs(magnitudes[byte] - value)
        if err < best_err or (err == best_err and byte % 2 == 0):
            best, best_err = byte, err
    return best


def _rne_half(n: int) -> int:
    """``n / 2`` rounded to the nearest integer, ties to even. Integers only."""
    half, odd = divmod(n, 2)
    return half + 1 if odd and half % 2 == 1 else half


def _build_ocp_to_legacy_halved_bytes() -> torch.Tensor:
    """Derive the 256-entry OCP-byte -> halved-legacy-byte table, and assert it.

    Same idiom as :func:`_build_fp4_to_fp8_bytes`: derive, then cross-check
    against an independent derivation. Here the primary derivation is the
    float64 reference -- "the legacy encoding of ``ocp_decode(b) / 2``" -- and
    the cross-check is the closed bitwise form:

    ==========================  ==================================  ==========
    case                        output                              exact?
    ==========================  ==================================  ==========
    ``f == 15 and m == 7``      REFUSED at load, slot kept as NaN   --
    ``f >= 2``                  ``s | ((f-1) << 3) | m``            exact
    ``f == 1``                  ``s | rne_half(8 + m)``             m even
    ``f == 0``                  ``s | rne_half(m)``                 m even
    ==========================  ==================================  ==========

    Two derivations of the same 256 bytes is cheap insurance for the same
    reason it is on the FP4 table, and here it is load-bearing rather than
    decorative: the obvious hand-derived form (``byte - 8``) is measured wrong
    (see the section header, leg E2a).

    Asserted exhaustively over all 256 bytes:

    1. the bitwise form equals the float64 form (the 254 loadable bytes);
    2. an output carries exponent field 15 **if and only if** its input is an
       OCP NaN byte -- stated as a biconditional, which is strictly stronger
       than "no output at field 15" and also pins the two NaN slots;
    3. the SIGN BIT is preserved for all 256 inputs (R-20: this is the clause
       ``byte - 8`` fails by construction);
    4. the two OCP NaN bytes keep their NaN patterns, so a slot that somehow
       escaped :func:`_reencode_ocp_to_legacy_halved`'s guard stays NaN instead
       of silently becoming a finite weight.
    """
    magnitudes = _legacy_e4m3_magnitudes()
    table: list[int] = [0] * 256
    for b in range(256):
        sign = b & 0x80
        field = (b >> 3) & 0x0F
        mant = b & 0x07
        if field == _E4M3_FIELD_NAN and mant == 7:
            # Kept as its own NaN pattern; refused before any gather.
            table[b] = b
            continue
        table[b] = sign | _encode_legacy_rne(
            _ocp_e4m3fn_magnitude(field, mant) / 2.0, magnitudes
        )

    for b in range(256):
        sign = b & 0x80
        field = (b >> 3) & 0x0F
        mant = b & 0x07
        out = table[b]

        # (3) sign bit preserved -- checked for every byte, NaN slots included.
        if (out & 0x80) != sign:
            raise RuntimeError(
                f"OCP->legacy halving table flipped the sign bit on input "
                f"{b:#04x} -> {out:#04x}. This is the exact failure mode of the "
                "rejected 'byte - 8' shortcut (underflow of exponent field 0 "
                "into the sign bit); refusing to load weights with it."
            )

        # (2) field 15 out IFF NaN in, and (4) the NaN slots keep their pattern.
        out_is_nan_field = ((out >> 3) & 0x0F) == _E4M3_FIELD_NAN
        in_is_ocp_nan = b in _OCP_NAN_BYTES
        if out_is_nan_field != in_is_ocp_nan:
            raise RuntimeError(
                f"OCP->legacy halving table put input {b:#04x} at output "
                f"{out:#04x} (exponent field "
                f"{(out >> 3) & 0x0F}); field {_E4M3_FIELD_NAN} is inf/NaN in "
                "the legacy grid and must appear for the OCP NaN bytes "
                f"{[hex(x) for x in _OCP_NAN_BYTES]} and for nothing else."
            )
        if in_is_ocp_nan:
            if out != b:
                raise RuntimeError(
                    f"OCP NaN byte {b:#04x} must stay {b:#04x} in the table, "
                    f"got {out:#04x}."
                )
            continue

        # (1) the closed bitwise form, independently.
        if field >= 2:
            expected = sign | ((field - 1) << 3) | mant
        elif field == 1:
            expected = sign | _rne_half(8 + mant)
        else:
            expected = sign | _rne_half(mant)
        if out != expected:
            raise RuntimeError(
                f"OCP->legacy halving table disagrees with its own bitwise "
                f"form on input {b:#04x}: float64 reference says {out:#04x}, "
                f"closed form says {expected:#04x}. One of the two is wrong; do "
                "not load dense/attention FP8 weights until it is resolved."
            )

    return torch.tensor(table, dtype=torch.uint8)


_OCP_TO_LEGACY_HALVED_BYTES = _build_ocp_to_legacy_halved_bytes()


def _reencode_ocp_to_legacy_halved(
    tensor: torch.Tensor, param_name: str, ckpt_key: str
) -> torch.Tensor:
    """LD-24: re-encode a 1-byte OCP tensor into halved legacy bytes.

    Applied UNCONDITIONALLY to every element the dense/attention block-FP8
    weight loaders emit. Returns the same shape and the same
    :data:`_FP8_DTYPE` carrier dtype -- only the byte VALUES change, so no
    downstream shape, dtype, signature or call site moves.

    The paired block scale MUST be doubled by
    :func:`_e8m0_to_fp32_doubled`. A weight halved without its scale doubled is
    a silent factor of two: no raise, plausible logits, and only a parity miss
    to show it (R-21).

    The index runs through ``index_select`` on an ``int32`` index rather than
    ``table[raw.long()]`` so a large parameter does not pay an int64 index
    tensor; the two are equivalent and ``index_select`` accepts ``int32``.
    """
    raw = _as_bytes(tensor, param_name, ckpt_key)
    # Both OCP NaN bytes at once: 0x7F and 0xFF differ only in the sign bit.
    if bool(((raw & 0x7F) == 0x7F).any()):
        count = int(((raw & 0x7F) == 0x7F).sum())
        raise ValueError(
            f"Block-FP8 weight {param_name!r} (key {ckpt_key!r}) contains "
            f"{count} OCP float8_e4m3fn NaN byte(s) "
            f"({[hex(x) for x in _OCP_NAN_BYTES]}). Refusing to load: a NaN "
            "weight produces NaN activations with no other symptom, and it is "
            "not something a re-encoding may quietly turn into a finite value."
        )
    flat = torch.index_select(
        _OCP_TO_LEGACY_HALVED_BYTES, 0, raw.reshape(-1).to(torch.int32)
    )
    return flat.reshape(raw.shape).view(_FP8_DTYPE)


def _e8m0_to_fp32_doubled(
    scale: torch.Tensor, param_name: str, ckpt_key: str
) -> torch.Tensor:
    """LD-24's scale leg: ``2 * _e8m0_to_fp32(scale)``, exactly.

    The paired half of :func:`_reencode_ocp_to_legacy_halved`. Every weight
    element was halved, so every block scale doubles and the product is
    unchanged to within the halving's own rounding.

    THE DOUBLING HAPPENS IN THE FP32 DOMAIN, after :func:`_e8m0_to_fp32`, not
    in the E8M0 code domain. Multiplying an fp32 power of two by ``2.0`` is
    exact, and it preserves ``_e8m0_to_fp32``'s deliberately recorded code-0
    semantics (code 0 -> ``0.0``, matching the reference's own ``fast_pow2``).
    A code-domain ``+1`` would turn ``0.0`` into ``2**-126`` and silently
    diverge from a documented decision. An all-zero block (code 0) is
    consistent either way: its weight bytes are all zero, so both ``0.0`` and
    ``2 * 0.0`` give a zero product.

    The doubled scale is STILL AN EXACT POWER OF TWO, so it is still exactly
    representable as ue8m0 and ``NF.block_fp8_linear``'s "weight dequantization
    introduces no rounding" clause survives -- the reason the factor is 2 and
    not ``240/448`` (see the LD-24 section header).

    Code 254 is REJECTED here. ``_e8m0_to_fp32`` already refuses code 255
    (E8M0 NaN); code 254 is ``2**127``, whose double overflows fp32 to ``+inf``,
    and an ``inf`` scale poisons its block exactly as a NaN one would.
    """
    raw = _as_bytes(scale, param_name, ckpt_key)
    if bool((raw >= 254).any()):
        worst = int(raw[raw >= 254][0])
        raise ValueError(
            f"Block scale for {param_name!r} (key {ckpt_key!r}) contains E8M0 "
            f"code {worst}. LD-24 doubles every dense/attention block scale to "
            "absorb the weight halving, and code 254 is 2**127 whose double "
            "overflows fp32 to +inf (code 255 is already NaN). Refusing to "
            "load: an inf or NaN weight scale produces non-finite activations "
            "with no other symptom."
        )
    return _e8m0_to_fp32(scale, param_name, ckpt_key) * 2.0


#: Attribute name under which each attach function records
#: ``(attribute, checkpoint_keys, loader)`` triples on the module it decorates,
#: for :func:`load_block_scale_buffers` to replay. Recorded on the module
#: itself -- not in a module-level registry -- so nothing here is global state
#: and two model instances in one process cannot collide.
_SCALE_SOURCES_ATTR = "_dsv4_loader_sources"


# ---------------------------------------------------------------------------
# Sharding description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Shard:
    """The 2-D region of a ``[out, in]`` checkpoint weight this core owns.

    One rectangle covers every sharding case in the family contract:

    * replicated  -> the whole tensor,
    * column (N)  -> a row band, full K,
    * row (K)     -> a column band, full N,
    * ``wo_a``    -> a row band (its o-group) AND a column band (its head-slice
      of that group's K), which is why a single ``shard_dim`` is not enough and
      ``sharding_weight_loader`` cannot express it.

    The indices are resolved at attach time from the caller's explicit rank
    arguments, so every transform below is a closure over plain ints and
    ignores the ``rank`` the load pipeline passes. That is deliberate: contract
    parallelism shards these weights by ranks that are NOT the global TP rank
    (the o-proj group rank, the 16-way shared-expert subgroup rank, the EP
    rank). Baking the resolved index in is the explicit form of
    :func:`vllm_neuron.utils.weight_loader.with_rank_override`, and it keeps
    each transform a pure function of its inputs.
    """

    row_start: int
    row_size: int
    col_start: int
    col_size: int

    @classmethod
    def replicated(cls, out_dim: int, in_dim: int) -> "_Shard":
        return cls(0, out_dim, 0, in_dim)

    @classmethod
    def rows(cls, start: int, size: int, in_dim: int) -> "_Shard":
        return cls(start, size, 0, in_dim)

    @classmethod
    def cols(cls, out_dim: int, start: int, size: int) -> "_Shard":
        return cls(0, out_dim, start, size)

    @property
    def local_shape(self) -> tuple[int, int]:
        return (self.row_size, self.col_size)

    def key(self) -> tuple[slice, slice]:
        return (
            slice(self.row_start, self.row_start + self.row_size),
            slice(self.col_start, self.col_start + self.col_size),
        )


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------


def _shape_of(slice_obj: Any) -> tuple[int, ...]:
    """Shape of a ``PySafeSlice`` / ``SliceView`` / tensor, as a tuple."""
    get_shape = getattr(slice_obj, "get_shape", None)
    if get_shape is not None:
        return tuple(get_shape())
    return tuple(slice_obj.shape)


def _require_shape(
    param_name: str, ckpt_key: str, slice_obj: Any, expected: tuple[int, ...]
) -> None:
    """Fail loudly, naming parameter AND checkpoint key, on a shape mismatch.

    A wrong shape here is not recoverable downstream: it either raises far away
    from the cause (inside ``load_state_dict``) or, worse, silently loads a
    plausible-looking tensor. Both cost a full recompile to discover.
    """
    actual = _shape_of(slice_obj)
    if actual != tuple(expected):
        raise ValueError(
            f"Checkpoint shape mismatch for parameter {param_name!r} loading "
            f"from key {ckpt_key!r}: expected {tuple(expected)}, checkpoint has "
            f"{actual}."
        )


def _require_slice_count(param_name: str, slices: Sequence[Any], expected: int) -> None:
    if len(slices) != expected:
        raise ValueError(
            f"Loader for {param_name!r} expects {expected} checkpoint slice(s), "
            f"got {len(slices)}. Check this parameter's entry in "
            "build_checkpoint_mappings()."
        )


def _as_bytes(tensor: torch.Tensor, param_name: str, ckpt_key: str) -> torch.Tensor:
    """Reinterpret a 1-byte-per-element tensor as ``uint8`` (never a copy).

    Used for both the ``float8_e8m0fnu`` scales and the ``int8`` MXFP4 weights.
    ``view`` (a bitcast) is mandatory here: ``copy_`` / ``to`` on an
    ``float8_e8m0fnu`` source would *convert* the value instead of moving the
    exponent byte, which is the hazard upstream vLLM records at
    ``deepseek_v4.py:1476-1484``.
    """
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.element_size() != 1:
        raise ValueError(
            f"Cannot byte-view parameter {param_name!r} from key {ckpt_key!r}: "
            f"dtype {tensor.dtype} is {tensor.element_size()} bytes per element, "
            "expected 1."
        )
    return tensor.view(torch.uint8)


def _e8m0_to_fp32(
    scale: torch.Tensor, param_name: str, ckpt_key: str
) -> torch.Tensor:
    """Convert a stored ``float8_e8m0fnu`` scale to its fp32 multiplier.

    E8M0 stores *only* a biased exponent, so its value is ``2**(byte - 127)``.
    An fp32 with that same exponent field and a zero mantissa has exactly the
    same value, so moving the byte into the fp32 exponent field reproduces the
    multiplier bit-exactly and with no floating-point work::

        (s.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)

    This is the same identity the reference kernel uses in the other direction
    (``kernel.py:30-33``: ``fast_pow2(x) = reinterpret_float32((x + 127) << 23)``,
    i.e. the unbiased exponent plus the 127 bias, shifted into place), and it is
    byte-for-byte what upstream vLLM does
    (``deepseek_v4.py:518-519``, ``_ue8m0_uint8_to_float``).

    Verified against ``torch``'s own ``float8_e8m0fnu`` -> fp32 conversion for
    all 256 byte values: identical on codes 1..254, and different only on the
    two edge codes -- code 0 (torch: ``2**-127``; trick: ``0.0``) and code 255
    (torch: ``NaN``; trick: ``+inf``).

    Code 0 is ALLOWED and deliberately yields ``0.0``, because that is what the
    reference's own ``fast_pow2`` produces for it (``kernel.py:30-33``:
    ``(0) << 23`` is ``0.0``), and matching the reference's arithmetic exactly
    matters more here than matching torch's textbook decoding of a value that
    only ever labels an all-zero weight block.

    Code 255 is REJECTED: E8M0's all-ones code is NaN, and a NaN scale would
    silently poison every activation that touches the block instead of failing
    at load. That is precisely the failure this loader is written to avoid.

    The shift is done in ``int32`` (not ``uint32``) because some torch CPU
    builds lack a ``uint32`` left-shift kernel -- the same constraint noted in
    ``llama3/weight_pack_mx_fp8.py:59-72``. Values reach at most
    ``254 << 23``, well inside int32.
    """
    raw = _as_bytes(scale, param_name, ckpt_key)
    if bool((raw == 255).any()):
        raise ValueError(
            f"Block scale for {param_name!r} (key {ckpt_key!r}) contains the "
            "E8M0 code 255, which is NaN. Refusing to load: a NaN weight scale "
            "produces NaN activations with no other symptom."
        )
    return (raw.to(torch.int32) << 23).view(torch.float32)


def _grid_extent(size: int) -> int:
    """``ceil(size / 128)`` -- the scale-grid extent of a weight axis."""
    return (size + _BLOCK - 1) // _BLOCK


def _grid_shard(
    shard: _Shard, param_name: str, ckpt_key: str
) -> tuple[slice, slice]:
    """Map a weight shard onto the block-scale grid.

    A weight sharded along an axis takes the matching slice of the scale grid,
    divided by 128; a replicated weight takes the whole grid.

    Two admissible cases per axis, and one refusal:

    * **Block-aligned** (``start`` and ``size`` both multiples of 128): the
      exact matching grid slice. Every main-stack weight is this case.
    * **Contained sub-block** (the shard is finer than 128 but lies entirely
      inside ONE scale block): the containing block's single row/column, taken
      whole. The scale is then SHARED between the cores that split that block,
      not divided -- and sharing is exactly right, because
      ``NF.block_fp8_linear`` dequantizes every element of a block by that one
      scalar, so each core's dequantized rows are bit-identical to the
      corresponding rows of the full-tensor dequant. The DSpark stage-0
      ``main_proj`` needs this: LD-18 records it column-parallel at
      ``N_local = hidden_size / tp_size`` = 64 at TP=64, which is half a scale
      block, and cores ``2k``/``2k+1`` legitimately share scale row ``k``.
    * **Straddling** (finer than 128 AND crossing a block boundary): refused.
      Here the local grid genuinely cannot be expressed -- the core would need
      a partial of each of two blocks -- so we refuse rather than guess.

    The old form of this helper refused BOTH sub-block cases on the grounds
    that a sub-block shard "no longer matches what ``NF.block_fp8_linear``
    dequantizes". That reasoning holds only for the straddling case; a
    contained shard matches exactly. The guard is narrowed here, not removed.
    """

    def axis(start: int, size: int, label: str) -> slice:
        if start % _BLOCK == 0 and size % _BLOCK == 0:
            return slice(start // _BLOCK, (start + size) // _BLOCK)
        first = start // _BLOCK
        last = (start + size - 1) // _BLOCK
        if first != last:
            raise ValueError(
                f"Block-scale misalignment for parameter {param_name!r} "
                f"(checkpoint key {ckpt_key!r}): the {label} shard "
                f"[{start}, {start + size}) is finer than the {_BLOCK}-wide "
                f"scale block AND straddles blocks {first}..{last}, so this "
                "core would need a partial of each. A sub-block shard is only "
                "admissible when it lies entirely inside one block. Re-check "
                "this weight's row in the family contract's sharding table "
                "(and note the K_local % 128 == 0 invariant in "
                "DeepseekV4Config.block_fp8_linear_plan)."
            )
        return slice(first, first + 1)

    return (
        axis(shard.row_start, shard.row_size, "row (N)"),
        axis(shard.col_start, shard.col_size, "column (K)"),
    )


# ---------------------------------------------------------------------------
# Loader factories: unquantized (bf16 / fp32 / int64) tensors
# ---------------------------------------------------------------------------


def _replicated_loader(
    param_name: str, ckpt_key: str, expected_shape: tuple[int, ...]
) -> SafetensorsWeightLoader:
    """Load a tensor whole, validating its checkpoint shape.

    Used for the norms, the KV-compressor tensors, the indexer's
    ``weights_proj`` / ``wq_b``, the router gate, ``tid2eid`` and the
    hash-context parameters -- every tensor the contract marks replicated. The
    validating identity transform (rather than no loader at all) exists purely
    so a checkpoint-revision shape change is reported here, with the key name,
    instead of surfacing as a ``load_state_dict`` size error later.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank  # replicated: every core loads the same bytes
        _require_slice_count(param_name, slices, 1)
        _require_shape(param_name, ckpt_key, slices[0], expected_shape)
        return slices[0][:]

    return SafetensorsWeightLoader(transform=transform)


def _dim0_slice_loader(
    param_name: str,
    ckpt_key: str,
    expected_shape: tuple[int, ...],
    start: int,
    size: int,
) -> SafetensorsWeightLoader:
    """Load ``[start:start+size]`` along dim 0 of an unquantized tensor.

    Only user today is ``attn_sink`` ``[64]``, sliced to this core's query
    head(s): the reference declares it per *local* head
    (``dsv4_ref/model.py:462``, ``empty(self.n_local_heads)``), so a replicated
    load would apply another core's sink bias to this core's head.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank
        _require_slice_count(param_name, slices, 1)
        _require_shape(param_name, ckpt_key, slices[0], expected_shape)
        return slices[0][start : start + size]

    return SafetensorsWeightLoader(transform=transform)


# ---------------------------------------------------------------------------
# Loader factories: block-128x128 FP8 weights and their scale grids
# ---------------------------------------------------------------------------


def _block_fp8_weight_loader(
    param_name: str,
    ckpt_key: str,
    full_shape: tuple[int, int],
    shard: _Shard,
) -> SafetensorsWeightLoader:
    """Load this core's rectangle of a ``[out, in]`` block-FP8 weight.

    No transpose: the storage orientation already matches
    ``NF.block_fp8_linear(x, weight_fp8[N_local, K_local], ...)``. See the
    module docstring, point 3.

    LD-24: every byte is re-encoded from the checkpoint's OCP-448 grid into the
    venue's legacy ``nl.float8_e4m3`` grid, halved, and this parameter's block
    scale is doubled to match by :func:`_block_fp8_scale_loader`. Both, or the
    port is silently wrong on this tensor (R-21).
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank  # shard resolved at attach time; see _Shard's docstring
        _require_slice_count(param_name, slices, 1)
        _require_shape(param_name, ckpt_key, slices[0], full_shape)
        return _reencode_ocp_to_legacy_halved(
            slices[0][shard.key()], param_name, ckpt_key
        )

    return SafetensorsWeightLoader(transform=transform)


def _block_fp8_scale_loader(
    param_name: str,
    ckpt_key: str,
    full_shape: tuple[int, int],
    shard: _Shard,
) -> SafetensorsWeightLoader:
    """Load this core's fp32 ``[N_local/128, K_local/128]`` block-scale grid.

    Emits fp32 (not the stored ``float8_e8m0fnu``) because that is exactly what
    ``NF.block_fp8_linear``'s ``weight_scale`` argument takes.

    LD-24: the grid is DOUBLED, absorbing the halving
    :func:`_block_fp8_weight_loader` applied to this parameter's bytes. Still an
    exact power of two, so it is still exactly ue8m0. Both, or the port is
    silently wrong on this tensor (R-21).
    """
    grid_shape = (_grid_extent(full_shape[0]), _grid_extent(full_shape[1]))
    grid_key = _grid_shard(shard, param_name, ckpt_key)
    # Derived FROM the resolved slices rather than recomputed from the shard
    # sizes: a contained sub-block shard yields one grid row/column that the
    # sharing cores take whole, so ``size // 128`` would say 0 (see
    # _grid_shard). Checked explicitly because a mis-sharded scale grid is not
    # a crash -- it is a wrong-numbers result found only after a
    # multi-thousand-second compile. At shared_expert_tp = 16 this pins w1/w3
    # to [1, 32] and w2 to [32, 1]; DSpark's main_proj at TP=64 to [1, 96].
    local_grid = (
        grid_key[0].stop - grid_key[0].start,
        grid_key[1].stop - grid_key[1].start,
    )

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank
        _require_slice_count(param_name, slices, 1)
        _require_shape(param_name, ckpt_key, slices[0], grid_shape)
        grid = _e8m0_to_fp32_doubled(slices[0][grid_key], param_name, ckpt_key)
        if tuple(grid.shape) != local_grid:
            raise ValueError(
                f"Local block-scale grid for {param_name!r} (key {ckpt_key!r}) "
                f"came out {tuple(grid.shape)}, expected {local_grid} from shard "
                f"rows [{shard.row_start}, {shard.row_start + shard.row_size}) x "
                f"cols [{shard.col_start}, {shard.col_start + shard.col_size})."
            )
        return grid

    return SafetensorsWeightLoader(transform=transform)


def _fused_block_fp8_weight_loader(
    param_name: str,
    ckpt_keys: Sequence[str],
    full_shapes: Sequence[tuple[int, int]],
    shards: Sequence[_Shard],
) -> SafetensorsWeightLoader:
    """Stack several block-FP8 weights along dim 0 (the output dim).

    ``fused_wqa_wkv`` is the only user: the checkpoint has no fused key, it has
    ``attn.wq_a`` ``[1024, 4096]`` and ``attn.wkv`` ``[512, 4096]``
    (``dsv4_ref/model.py:463-466`` declares them as two separate ``Linear``s),
    and this port fuses them into one ``[1536, 4096]`` parameter so the q and
    kv down-projections share a single GEMM.

    LD-24: every byte of every part is re-encoded and halved, and the fused
    scale grid is doubled by :func:`_fused_block_fp8_scale_loader`. Because the
    transform is uniform it commutes with the N-concat, so re-encoding before
    the ``cat`` and re-encoding after it are the same bytes -- which is also why
    ``attention.py``'s ``[:rows]`` slice and ``dspark_model.py``'s complementary
    slice of this parameter need no edit.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank
        _require_slice_count(param_name, slices, len(ckpt_keys))
        parts = []
        for slice_obj, key, shape, shard in zip(
            slices, ckpt_keys, full_shapes, shards
        ):
            _require_shape(param_name, key, slice_obj, shape)
            parts.append(
                _reencode_ocp_to_legacy_halved(
                    slice_obj[shard.key()], param_name, key
                )
            )
        return torch.cat(parts, dim=0)

    return SafetensorsWeightLoader(transform=transform)


def _fused_block_fp8_scale_loader(
    param_name: str,
    ckpt_keys: Sequence[str],
    full_shapes: Sequence[tuple[int, int]],
    shards: Sequence[_Shard],
) -> SafetensorsWeightLoader:
    """Stack the fused weights' scale grids along dim 0.

    The grids concatenate exactly like the weights, because the fusion is along
    the N axis and each source's N is a whole number of 128-blocks:
    ``[8, 32]`` and ``[4, 32]`` become ``[12, 32]``.

    LD-24: every part's grid is DOUBLED, absorbing the halving
    :func:`_fused_block_fp8_weight_loader` applied to the fused bytes. Both, or
    the port is silently wrong on this tensor (R-21).
    """
    grid_shapes = [(_grid_extent(n), _grid_extent(k)) for n, k in full_shapes]
    grid_keys = [
        _grid_shard(shard, param_name, key) for shard, key in zip(shards, ckpt_keys)
    ]

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank
        _require_slice_count(param_name, slices, len(ckpt_keys))
        parts = []
        for slice_obj, key, grid_shape, grid_key in zip(
            slices, ckpt_keys, grid_shapes, grid_keys
        ):
            _require_shape(param_name, key, slice_obj, grid_shape)
            parts.append(
                _e8m0_to_fp32_doubled(slice_obj[grid_key], param_name, key)
            )
        return torch.cat(parts, dim=0)

    return SafetensorsWeightLoader(transform=transform)


# ---------------------------------------------------------------------------
# Loader factories: routed experts (MXFP4 in the checkpoint -> MXFP8 here)
# ---------------------------------------------------------------------------


def _unpack_mxfp4_to_fp8_bytes(
    packed: torch.Tensor, param_name: str, ckpt_key: str, logical_k: int
) -> torch.Tensor:
    """Unpack ``[N, K/2]`` int8 MXFP4 into ``[N, K]`` fp8 *bytes*.

    Nibble order and value table are taken verbatim from DeepSeek's
    ``convert.py:30-33``::

        x = x.view(torch.uint8)
        low  = x & 0x0F
        high = (x >> 4) & 0x0F
        x = torch.stack([FP4_TABLE[low], FP4_TABLE[high]], dim=-1)...

    i.e. the LOW nibble is the even (first) element along K and the HIGH nibble
    the odd (second) one, and FP4 is packed along the K (last) axis -- also
    stated by the reference GEMM: "B is stored as [N, K//2] in
    float4_e2m1fn_x2 ... packed along the K (last) dimension"
    (``dsv4_ref/kernel.py:450-451``).

    The result is returned as ``uint8`` bytes, not as fp8, so the caller can
    stack experts without needing a float8 CPU kernel.
    """
    if packed.shape[1] * 2 != logical_k:
        raise ValueError(
            f"MXFP4 unpack for {param_name!r} (key {ckpt_key!r}): stored "
            f"{tuple(packed.shape)} implies logical K={packed.shape[1] * 2}, "
            f"expected {logical_k}."
        )
    raw = _as_bytes(packed, param_name, ckpt_key)
    low = _FP4_TO_FP8_BYTES[(raw & 0x0F).long()]
    high = _FP4_TO_FP8_BYTES[(raw >> 4).long()]
    # Interleave back to [low0, high0, low1, high1, ...] along K.
    return torch.stack([low, high], dim=-1).reshape(raw.shape[0], logical_k)


# ---------------------------------------------------------------------------
# LD-23: MXFP4 + E8M0-group-32  ->  legacy-E4M3 + per-output-channel power-of-two
#
# WHY THIS REPLACED THE MX TILING PATH (R-13, recorded in
# ``artifacts/repairs/author_model_family-iter2/r13-moe-mxfp8-gen4-blocker.txt``):
# the MX expert kernels lower to ``nisa.nc_matmul_mx`` / ``nisa.quantize_mx``,
# which are NeuronCore-v4 instructions. This campaign's venue is trn2 =
# NeuronCore-v3, so EVERY MX weight path is unexecutable here, tiled correctly
# or not. The whole MX tiling apparatus (the four ``_tile_mx_*`` transforms and
# their ``_MX_TILERS`` table) is therefore retired, not fixed.
#
# WHAT REPLACES IT: the gen3-legal quantized expert path is 1-byte FP8 with
# *plain* layouts and per-output-channel dequant scales -- ``NF.moe_cte``
# non-MX ``shard_on_i`` at prefill and ``NF.moe_tkg`` ``QuantizationType.ROW``
# at decode, both proven on the installed wheel by
# ``artifacts/repairs/author_model_family-iter3/iter3-moe-gen3-probe.txt``.
# Those kernels take ONE fp32 multiplier per output channel, so the
# checkpoint's per-group-of-32 E8M0 exponents have to be folded into a single
# per-channel exponent here, at load time.
#
# WHY THAT FOLD IS EXACT, AND WHY IT MUST BE ASSERTED RATHER THAN ASSUMED
# (port-assessment.md section 2.5). Every source element is
# ``value = m * 2**e_ck`` where ``m`` is one of the 16 FP4 magnitudes and
# ``e_ck = code - 127`` is its group's E8M0 exponent. Writing the FP8 byte's
# biased exponent field as ``f`` (legacy ``nl.float8_e4m3``: bias 7, ``f`` in
# ``1..14`` for normals, ``15`` reserved for inf/NaN), the element's exponent
# budget is ``q = f_base + e_ck``, where ``f_base`` is the field the FP4
# magnitude alone occupies. Choosing ONE per-channel shift ``p`` and writing
# ``f_tgt = q - p`` reproduces the value EXACTLY -- no rounding whatsoever,
# because only the exponent field moves and the 3 mantissa bits are copied
# untouched -- provided every live element lands in ``1 <= f_tgt <= 14``.
# Placing the channel's maximum at the top of the window (``p = q_max - 14``)
# makes the fold exact iff the channel's ``q`` span is at most 13 binades.
#
# 13 binades is a real, checkpoint-dependent limit, not a formality. When a
# channel exceeds it there is no exact representation, and the two silent
# alternatives are both wrong: rounding loses bits the parity standard is
# measuring, and falling back to another format on this loader's own authority
# would substitute a path nobody planned. So this loader RAISES, naming the
# offending tensor, the output channel, and the measured span. That raise is
# operator finding F-4 part 2 (port-assessment.md section 4): it is the
# planner's decision to make, not this node's.
#
# The arithmetic below was proven bit-exact before it was authored --
# ``scratch/iter3/requant_math.py``, 7 cases including the 13-binade boundary,
# a 14-binade over-window raise, E8M0 code-0 groups, an all-zero channel and
# the real family shape ``[2048, 4096]``.
# ---------------------------------------------------------------------------

# The exponent-field window this fold targets, :data:`_E4M3_FIELD_MIN` ..
# :data:`_E4M3_FIELD_MAX`, is now MODULE-LEVEL doctrine shared with the
# dense/attention slice -- see "THE 1-BYTE CARRIER DOCTRINE" in the Constants
# section. It is not an expert-path-only rule and never was; the promotion
# happened at LD-24, when the dense slice turned out to need the same
# discipline. This path satisfies it by construction: it never routes a value
# through torch's fp8 conversion, it moves exponent fields inside bytes taken
# from :data:`_FP4_TO_FP8_BYTES`.

#: Where the per-channel maximum is placed. Top of the window, so the fold
#: never overflows to inf and the mantissa keeps every bit it had.
_E4M3_FIELD_TOP = _E4M3_FIELD_MAX

#: The exactly-representable ``q`` span, in binades.
_E4M3_EXACT_SPAN = _E4M3_FIELD_MAX - _E4M3_FIELD_MIN


def _requantize_expert_to_pow2_per_channel(
    packed: torch.Tensor,
    scale_codes: torch.Tensor,
    param_name: str,
    ckpt_key: str,
    logical_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[N, K/2]`` MXFP4 + ``[N, K/32]`` E8M0  ->  ``[N, K]`` legacy-E4M3 bytes
    and ``[N]`` fp32 power-of-two per-output-channel scales.

    Rows are OUTPUT channels: the checkpoint stores every expert leaf as
    ``[out, in]`` (family interface contract section 1) and the E8M0 groups run
    along ``in``, so one scale per row is exactly "per output channel".

    Returns ``(bytes, scale)`` such that
    ``legacy_e4m3_decode(bytes) * scale[:, None]`` equals the checkpoint's own
    value ``fp4_value * 2**(code - 127)`` BIT-EXACTLY for every element, or
    raises. There is no tolerance parameter and no rounding: see the section
    header for why exactness is checked rather than hoped for.
    """
    if packed.shape[1] * 2 != logical_k:
        raise ValueError(
            f"MXFP4 requantization for {param_name!r} (key {ckpt_key!r}): "
            f"stored {tuple(packed.shape)} implies logical K="
            f"{packed.shape[1] * 2}, expected {logical_k}."
        )
    n_groups = scale_codes.shape[1]
    if n_groups * _MX_GROUP != logical_k:
        raise ValueError(
            f"MXFP4 requantization for {param_name!r} (key {ckpt_key!r}): "
            f"{n_groups} E8M0 groups x {_MX_GROUP} = {n_groups * _MX_GROUP} "
            f"elements, but the weight has K={logical_k}."
        )
    if packed.shape[0] != scale_codes.shape[0]:
        raise ValueError(
            f"MXFP4 requantization for {param_name!r} (key {ckpt_key!r}): "
            f"weight has {packed.shape[0]} output channels but its scale grid "
            f"has {scale_codes.shape[0]}."
        )

    codes = _as_bytes(scale_codes, param_name, ckpt_key)
    # Code 255 is E8M0's NaN. Rejected for the same reason
    # ``_e8m0_to_fp32`` rejects it: a NaN scale poisons every activation that
    # touches the block and has no other symptom.
    if bool((codes == 255).any()):
        raise ValueError(
            f"Expert scale for {param_name!r} (key {ckpt_key!r}) contains the "
            "E8M0 code 255, which is NaN. Refusing to load."
        )

    # --- the FP4 -> FP8 byte upcast, unchanged (bit-identical to the
    # reference's own "lossless" direction, dsv4_ref/convert.py:17-19) --------
    base = _unpack_mxfp4_to_fp8_bytes(packed, param_name, ckpt_key, logical_k)

    sign = (base & 0x80).to(torch.int32)
    mant = (base & 0x07).to(torch.int32)
    f_base = ((base >> 3) & 0x0F).to(torch.int32)

    # An element contributes NOTHING to the channel's exponent budget when it
    # is a zero. Two independent ways to be zero, both of which must be
    # excluded or a single zero would drag ``q_min`` to nonsense:
    #   * the FP4 code is +-0, i.e. byte 0x00 or 0x80 (note _FP4_TABLE maps
    #     code 8 to +0.0, so in practice only 0x00 occurs);
    #   * the whole group carries E8M0 code 0, whose multiplier is 0.0 -- the
    #     reference's own ``fast_pow2`` semantics, see ``_e8m0_to_fp32``.
    zero_code = (base & 0x7F) == 0
    group_zero = torch.repeat_interleave(codes == 0, _MX_GROUP, dim=1)
    is_zero = zero_code | group_zero

    e_ck = torch.repeat_interleave(codes.to(torch.int32), _MX_GROUP, dim=1) - 127
    q = f_base + e_ck

    # Per-channel shift from the max live ``q``. ``masked_fill`` with a value
    # below any reachable ``q`` (``f_base >= 0``, ``e_ck >= -127``) keeps the
    # max over live elements only.
    live = ~is_zero
    has_live = live.any(dim=1)
    q_live_max = q.masked_fill(is_zero, -(2**20)).amax(dim=1)
    p = torch.where(has_live, q_live_max - _E4M3_FIELD_TOP, torch.zeros_like(q_live_max))

    f_tgt = q - p[:, None]

    # --- the HARD exactness assertion (operator finding F-4 part 2) ---------
    bad = live & ((f_tgt < _E4M3_FIELD_MIN) | (f_tgt > _E4M3_FIELD_MAX))
    if bool(bad.any()):
        rows = torch.nonzero(bad.any(dim=1), as_tuple=False).flatten()
        row = int(rows[0])
        q_row = q[row][live[row]]
        span = int(q_row.max() - q_row.min())
        raise ValueError(
            f"Cannot requantize {param_name!r} (key {ckpt_key!r}) to one "
            f"power-of-two scale per output channel without losing bits: "
            f"{int(bad.sum())} element(s) in {int(rows.numel())} of "
            f"{packed.shape[0]} output channel(s) fall outside legacy "
            f"nl.float8_e4m3's normal exponent window "
            f"[{_E4M3_FIELD_MIN}, {_E4M3_FIELD_MAX}]. First offending output "
            f"channel is {row}: its live elements span "
            f"{int(q_row.min())}..{int(q_row.max())} = {span} binades, and only "
            f"{_E4M3_EXACT_SPAN} are exactly representable. "
            "This loader will NOT round and will NOT substitute another format "
            "on its own authority -- both would silently change the numerics "
            "the parity standard measures. This is operator finding F-4 part 2 "
            "(port-assessment.md section 4): the ladder rung is the planner's "
            "decision."
        )

    # Zeros keep their sign bit and nothing else; live elements keep sign and
    # mantissa and take the shifted exponent field. Purely bitwise -- no
    # floating-point operation touches a weight value anywhere in this
    # function, which is what makes "bit-exact" a fact and not a hope.
    out = torch.where(
        is_zero,
        sign,
        sign | (f_tgt << 3) | mant,
    ).to(torch.uint8)

    # ``2**p`` built by moving ``p`` into the fp32 exponent field, the same
    # bit trick ``_e8m0_to_fp32`` documents (and, transitively, the
    # reference's ``fast_pow2``). ``ldexp`` would also work; this keeps the
    # module's single idiom for "exact power of two" and needs no fp math.
    #
    # Range note: p = q_max - 14 with f_base <= 9 and code <= 254 gives
    # p <= 118, and the smallest live q is >= 1 - 127 = -126 so p >= -140 is
    # possible in principle. An fp32 exponent field only holds 1..254
    # (unbiased -126..127), so a p outside that is not representable; it is
    # rejected here rather than wrapping into a wrong multiplier.
    p_biased = p + 127
    if bool(((p_biased < 1) | (p_biased > 254)).any()):
        worst = int(p[(p_biased < 1) | (p_biased > 254)][0])
        raise ValueError(
            f"Requantized scale for {param_name!r} (key {ckpt_key!r}) needs "
            f"2**{worst}, which is outside fp32's normal exponent range "
            "[-126, 127]. Refusing to load a scale that cannot be represented."
        )
    scale = (p_biased.to(torch.int32) << 23).view(torch.float32)

    return out.reshape(packed.shape[0], logical_k), scale


def _split_weight_and_scale_slices(
    param_name: str, slices: Sequence[Any], n_local: int
) -> tuple[list, list]:
    """Split ``[w_e0..w_e{L-1}, s_e0..s_e{L-1}]`` into its two groups.

    ``expert_parallel_grouped_loader`` is documented to hand a flat list
    grouped BY ITEM across experts (``utils/weight_loader.py:697-704``), and
    both LD-23 loaders bind the weight keys followed by the scale keys, so the
    list is exactly two equal groups. Asserted rather than assumed: getting
    this wrong would silently requantize scale bytes as weights.
    """
    if len(slices) != 2 * n_local:
        raise ValueError(
            f"Loader for {param_name!r} expected {2 * n_local} slices "
            f"({n_local} weights then {n_local} scale grids), got "
            f"{len(slices)}."
        )
    return list(slices[:n_local]), list(slices[n_local:])


def _fp8_pow2_expert_weight_loader(
    param_name: str,
    local_weight_keys: Sequence[str],
    local_scale_keys: Sequence[str],
    out_dim: int,
    logical_k: int,
) -> SafetensorsWeightLoader:
    """Requantize this core's local experts to ``[E_local, K, N]`` FP8 bytes.

    TWO things happen here that the retired MX loader did not do.

    1. THE SCALES ARE FOLDED IN, so this loader needs BOTH checkpoint tensors
       per expert -- the FP4 weight and its E8M0 grid. It gets them because
       :func:`attach_moe_loaders` binds the weight keys followed by the scale
       keys and ``expert_parallel_grouped_loader`` trims each group to the
       local expert range independently.

    2. THE RESULT IS TRANSPOSED to ``[in, out]``. The gen3-legal MoE kernels
       take ``gate_up`` as ``[E, H, 2, I]`` and ``down`` as ``[E, I, H]``
       (contraction axis first, output channel last), while the checkpoint
       stores ``[out, in]``. ``DeepseekV4RoutedExperts`` reinterprets these
       parameters with pure ``view()`` calls, and a ``view`` cannot transpose,
       so the transpose has to happen at load time -- exactly the constraint
       that forced the retired path to pre-tile.

    Every byte is emitted as ``uint8`` and reinterpreted once at the end:
    several torch CPU builds lack float8 ``stack``/``index_select`` kernels,
    and the byte route is bit-equivalent (same reasoning as
    ``llama3/weight_pack_mx_fp8.py:121-124``).
    """
    n_local = len(local_weight_keys)
    weight_shape = (out_dim, logical_k // 2)
    scale_shape = (out_dim, logical_k // _MX_GROUP)

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank  # experts are selected by EP rank, baked in at attach time
        w_slices, s_slices = _split_weight_and_scale_slices(
            param_name, slices, n_local
        )
        per_expert = []
        for w_slice, s_slice, w_key, s_key in zip(
            w_slices, s_slices, local_weight_keys, local_scale_keys
        ):
            _require_shape(param_name, w_key, w_slice, weight_shape)
            _require_shape(param_name, s_key, s_slice, scale_shape)
            byts, _ = _requantize_expert_to_pow2_per_channel(
                _as_bytes(w_slice[:], param_name, w_key),
                s_slice[:],
                param_name,
                w_key,
                logical_k,
            )
            # [out, in] -> [in, out]; ``contiguous`` because the parameter is
            # consumed by ``view``, which refuses a non-contiguous source.
            per_expert.append(byts.t().contiguous())
        stacked = torch.stack(per_expert, dim=0)
        return stacked.contiguous().view(_FP8_DTYPE)

    return SafetensorsWeightLoader(transform=transform)


def _fp8_pow2_expert_scale_loader(
    param_name: str,
    local_weight_keys: Sequence[str],
    local_scale_keys: Sequence[str],
    out_dim: int,
    logical_k: int,
) -> SafetensorsWeightLoader:
    """Emit this core's local experts' ``[E_local, N]`` fp32 per-channel scales.

    Computed by the SAME function as the weights
    (:func:`_requantize_expert_to_pow2_per_channel`), from the same two
    checkpoint tensors, so the scale can never disagree with the bytes it
    scales. Duplicating the derivation instead of caching it costs one extra
    pass over the FP4 bytes at load and buys the guarantee that a future edit
    cannot desynchronize the pair -- the failure mode here is silent numeric
    corruption, not a crash.

    No transpose: the scale is already one value per output channel, and the
    kernels take it as ``[E, N]`` (``moe_tkg`` ROW) or ``[E, 1, N]``
    (``moe_cte`` PER_CHANNEL, produced by an ``unsqueeze`` in ``moe.py``).
    """
    n_local = len(local_weight_keys)
    weight_shape = (out_dim, logical_k // 2)
    scale_shape = (out_dim, logical_k // _MX_GROUP)

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank
        w_slices, s_slices = _split_weight_and_scale_slices(
            param_name, slices, n_local
        )
        per_expert = []
        for w_slice, s_slice, w_key, s_key in zip(
            w_slices, s_slices, local_weight_keys, local_scale_keys
        ):
            _require_shape(param_name, w_key, w_slice, weight_shape)
            _require_shape(param_name, s_key, s_slice, scale_shape)
            _, scale = _requantize_expert_to_pow2_per_channel(
                _as_bytes(w_slice[:], param_name, w_key),
                s_slice[:],
                param_name,
                w_key,
                logical_k,
            )
            per_expert.append(scale)
        return torch.stack(per_expert, dim=0).contiguous()

    return SafetensorsWeightLoader(transform=transform)


# ---------------------------------------------------------------------------
# Binding helper
# ---------------------------------------------------------------------------


def _bind(
    module: nn.Module,
    attr: str,
    ckpt_keys: Sequence[str],
    loader: SafetensorsWeightLoader,
    *,
    required: bool = True,
) -> bool:
    """Attach ``loader`` to ``module.<attr>`` and record its checkpoint keys.

    Two-part on purpose:

    * ``set_weight_loader`` on the tensor is what the pipelined
      ``named_parameters()`` load path reads.
    * the recorded ``(attr, keys, loader)`` triple is what
      :func:`load_block_scale_buffers` replays for anything registered as a
      *buffer* instead of a parameter -- that path cannot go through
      ``set_weight_loader``, because a buffer declared ``None`` has no tensor to
      hang the loader on yet (same problem llama3 solves with its
      ``_FP8_*_SCALE_SOURCES`` tables).

    Returns whether the attribute existed. ``required=False`` is for the
    genuinely optional tensors (compressor, indexer, ``gate_bias``,
    ``tid2eid``).

    ``attr`` may be dotted (``"attn_norm.weight"``): the loader is then hung on
    the leaf tensor and recorded against its *owning* submodule, so
    :func:`load_block_scale_buffers`'s ``named_modules()`` walk still yields the
    correct qualified name. Needed because the decoder layer owns its norms as
    ``DeepseekV4RMSNorm`` submodules, not as flat ``*_norm_weight`` tensors.
    """
    if "." in attr:
        owner_path, _, leaf = attr.rpartition(".")
        owner: nn.Module | None = module
        for part in owner_path.split("."):
            owner = getattr(owner, part, None)
            if owner is None:
                break
        if not isinstance(owner, nn.Module):
            if required:
                raise AttributeError(
                    f"{type(module).__name__} has no submodule {owner_path!r} to "
                    f"bind {attr!r} on; the family interface contract fixes this "
                    "name and the weight loaders bind to it."
                )
            return False
        return _bind(owner, leaf, ckpt_keys, loader, required=required)

    if not hasattr(module, attr):
        if required:
            raise AttributeError(
                f"{type(module).__name__} has no attribute {attr!r}; the family "
                "interface contract fixes this name and the weight loaders bind "
                "to it. Either the module was renamed or the loader call is on "
                "the wrong module."
            )
        return False

    current = getattr(module, attr)
    if isinstance(current, torch.Tensor):
        set_weight_loader(current, loader)

    sources = list(getattr(module, _SCALE_SOURCES_ATTR, ()))
    sources.append((attr, tuple(ckpt_keys), loader))
    setattr(module, _SCALE_SOURCES_ATTR, tuple(sources))
    return True


# ---------------------------------------------------------------------------
# Checkpoint key helpers
# ---------------------------------------------------------------------------


def _layer_key_prefix(layer_idx: int) -> str:
    """Checkpoint prefix for one main-stack transformer block.

    This is the MAIN-STACK spelling only. The DSpark draft stages live under
    ``mtp.{s}`` and are reached by passing ``key_prefix="mtp.{s}"`` to the
    attach functions, not by index arithmetic here: the drafter's stage index
    and the ``layer_idx`` it hands the MoE (``num_hidden_layers + stage``, so
    ``is_hash_moe_layer`` is False and the ``gate.bias`` branch is taken, which
    is what the ``mtp.*`` key census shows) are deliberately different numbers,
    and folding them together here would make one of the two wrong.
    """
    return f"layers.{layer_idx}"


def _resolve_layer_idx(module: nn.Module, kind: str) -> int:
    """Read ``module.layer_idx``, which every family module carries.

    Fixed by the contract's constructors --
    ``DeepseekV4Attention(config, layer_idx, ...)`` and
    ``DeepseekV4MoE(config, layer_idx, ...)`` -- and needed by the attach
    functions whose signature has no ``layer_idx`` of its own, because the
    checkpoint key for every tensor is layer-qualified.
    """
    layer_idx = getattr(module, "layer_idx", None)
    if not isinstance(layer_idx, int):
        raise AttributeError(
            f"{kind} loader needs an integer {type(module).__name__}.layer_idx to "
            "build layer-qualified checkpoint keys; found "
            f"{layer_idx!r}. Store the constructor's layer_idx on the module."
        )
    return layer_idx


# ---------------------------------------------------------------------------
# Public: attention
# ---------------------------------------------------------------------------


def attach_attention_loaders(
    module: nn.Module,
    config: "DeepseekV4Config",
    *,
    tp_size: int,
    tp_rank: int,
    group_rank: int,
    group_size: int,
    shared_tp_size: int,
    shared_tp_rank: int,
    key_prefix: str | None = None,
) -> None:
    """Attach every checkpoint loader a :class:`DeepseekV4Attention` needs.

    >>> PARALLELISM (contract sharding table, verified against the reference's
    own parallel Linears) <<<

    ===================== ============================== =================
    parameter             sharding                       local at TP=64
    ===================== ============================== =================
    ``fused_wqa_wkv``     replicated                     ``[1536, 4096]``
    ``wq_b``              column over query heads        ``[512, 1024]``
    ``wo_a``              one o-group per ``tp/o_groups`` ``[1024, 512]``
                          cores, each owning its
                          head-slice of that group's K
    ``wo_b``              row over K                     ``[4096, 128]``
    ``attn_sink``         head slice                     ``[1]``
    ``q_norm``/``kv_norm`` replicated                    as checkpoint
    compressor (all)      replicated                     as checkpoint
    indexer ``wq_b``      replicated                     ``[8192, 1024]``
    indexer ``weights_proj`` replicated                  ``[64, 4096]``
    ===================== ============================== =================

    ``wo_a`` is the only two-axis shard. The reference makes the layout
    explicit: it declares ``wo_a`` as
    ``ColumnParallelLinear(n_heads * head_dim // n_groups, n_groups *
    o_lora_rank)`` and then consumes it as
    ``wo_a.weight.view(n_local_groups, o_lora_rank, -1)``
    (``dsv4_ref/model.py:468`` and ``:543``). So dim 0 indexes
    ``(o_group, o_lora_rank)`` and dim 1 indexes
    ``(head_within_group, kv_lora_rank)``. With one query head per core at
    TP=64, core ``r`` owns head ``r``, hence o-group ``r // heads_per_group``
    and K-slice ``r % heads_per_group``.

    Args:
        module: The :class:`DeepseekV4Attention` instance.
        config: The family config (dims and per-layer structure).
        tp_size / tp_rank: Global attention tensor-parallel degree and rank.
        group_rank / group_size: This core's rank inside, and the size of, the
            o-projection process group (the group that all-reduces the grouped
            o-proj). ``group_size`` must be ``tp_size // config.o_groups``.
        shared_tp_size / shared_tp_rank: Accepted for signature symmetry with
            :func:`attach_moe_loaders` (the contract fixes one attach
            signature shape for both); attention has no shared-expert weights,
            so they are only validated.
    """
    if shared_tp_size <= 0 or not (0 <= shared_tp_rank < shared_tp_size):
        raise ValueError(
            f"attach_attention_loaders: invalid shared_tp_rank={shared_tp_rank} "
            f"for shared_tp_size={shared_tp_size}."
        )

    layer_idx = _resolve_layer_idx(module, "attention")
    # ``key_prefix`` overrides the main-stack spelling so the DSpark stages can
    # bind the same parameter set under ``mtp.{s}`` (the checkpoint's own
    # namespace) while still reporting a layer_idx to the sharding math.
    prefix = _layer_key_prefix(layer_idx) if key_prefix is None else key_prefix

    hidden = config.hidden_size
    q_lora = config.q_lora_rank
    kv_lora = config.kv_lora_rank
    o_lora = config.o_lora_rank
    o_groups = config.o_groups
    num_heads = config.num_attention_heads

    if num_heads % tp_size != 0:
        raise ValueError(
            f"attach_attention_loaders: num_attention_heads={num_heads} must be "
            f"divisible by tp_size={tp_size} (one contiguous head band per core)."
        )
    heads_per_rank = num_heads // tp_size
    heads_per_group = num_heads // o_groups
    if heads_per_group % heads_per_rank != 0:
        raise ValueError(
            f"attach_attention_loaders: each core's {heads_per_rank} head(s) must "
            f"sit inside one o-group of {heads_per_group} heads; "
            f"tp_size={tp_size} with o_groups={o_groups} splits a core across "
            "groups, which the grouped o-projection cannot express."
        )
    expected_group_size = tp_size // o_groups
    if group_size != expected_group_size:
        raise ValueError(
            f"attach_attention_loaders: group_size={group_size} but "
            f"tp_size // o_groups = {expected_group_size}. The o-projection "
            "group must hold exactly the cores that share one o-group."
        )
    if group_rank != tp_rank % group_size:
        # The head->group->K mapping above is derived from the *global* head
        # index, because wq_b is a plain contiguous column shard by tp_rank. If
        # the o-proj process group was built from a non-contiguous rank mesh
        # (e.g. functional/process_groups.py's TRN2_8x8_MESH rows), its
        # rank_in_group no longer matches the head ordering, and wq_b's head
        # assignment would have to be re-derived from the same ordering.
        # Refuse rather than shard one of the two the wrong way.
        raise ValueError(
            f"attach_attention_loaders: group_rank={group_rank} != "
            f"tp_rank % group_size = {tp_rank % group_size}. The o-projection "
            "group's rank order must match the contiguous query-head order that "
            "wq_b is sharded by. If the group comes from a non-contiguous mesh, "
            "wq_b's head slice must be re-derived from that same ordering."
        )

    global_head = tp_rank * heads_per_rank
    o_group_index = global_head // heads_per_group
    head_in_group = global_head % heads_per_group

    # --- fused q/kv down-projection (two checkpoint tensors, one parameter) --
    wqa_shape = (q_lora, hidden)
    wkv_shape = (kv_lora, hidden)
    fused_keys = (f"{prefix}.attn.wq_a.weight", f"{prefix}.attn.wkv.weight")
    fused_scale_keys = (f"{prefix}.attn.wq_a.scale", f"{prefix}.attn.wkv.scale")
    fused_shards = (
        _Shard.replicated(*wqa_shape),
        _Shard.replicated(*wkv_shape),
    )
    _bind(
        module,
        "fused_wqa_wkv_weight",
        fused_keys,
        _fused_block_fp8_weight_loader(
            "fused_wqa_wkv_weight", fused_keys, (wqa_shape, wkv_shape), fused_shards
        ),
    )
    _bind(
        module,
        "fused_wqa_wkv_scale",
        fused_scale_keys,
        _fused_block_fp8_scale_loader(
            "fused_wqa_wkv_scale",
            fused_scale_keys,
            (wqa_shape, wkv_shape),
            fused_shards,
        ),
    )

    # --- q up-projection: column-parallel over query heads -------------------
    wq_b_shape = (num_heads * kv_lora, q_lora)
    wq_b_shard = _Shard.rows(
        global_head * kv_lora, heads_per_rank * kv_lora, q_lora
    )
    _bind(
        module,
        "wq_b_weight",
        (f"{prefix}.attn.wq_b.weight",),
        _block_fp8_weight_loader(
            "wq_b_weight", f"{prefix}.attn.wq_b.weight", wq_b_shape, wq_b_shard
        ),
    )
    _bind(
        module,
        "wq_b_scale",
        (f"{prefix}.attn.wq_b.scale",),
        _block_fp8_scale_loader(
            "wq_b_scale", f"{prefix}.attn.wq_b.scale", wq_b_shape, wq_b_shard
        ),
    )

    # --- grouped o-projection stage A: row band (o-group) x column band (head)
    wo_a_shape = (o_groups * o_lora, heads_per_group * kv_lora)
    wo_a_shard = _Shard(
        row_start=o_group_index * o_lora,
        row_size=o_lora,
        col_start=head_in_group * kv_lora,
        col_size=heads_per_rank * kv_lora,
    )
    _bind(
        module,
        "wo_a_weight",
        (f"{prefix}.attn.wo_a.weight",),
        _block_fp8_weight_loader(
            "wo_a_weight", f"{prefix}.attn.wo_a.weight", wo_a_shape, wo_a_shard
        ),
    )
    _bind(
        module,
        "wo_a_scale",
        (f"{prefix}.attn.wo_a.scale",),
        _block_fp8_scale_loader(
            "wo_a_scale", f"{prefix}.attn.wo_a.scale", wo_a_shape, wo_a_shard
        ),
    )

    # --- grouped o-projection stage B: row-parallel over K -------------------
    wo_b_shape = (hidden, o_groups * o_lora)
    if wo_b_shape[1] % tp_size != 0:
        raise ValueError(
            f"attach_attention_loaders: wo_b K={wo_b_shape[1]} is not divisible "
            f"by tp_size={tp_size}."
        )
    wo_b_k_local = wo_b_shape[1] // tp_size
    wo_b_shard = _Shard.cols(hidden, tp_rank * wo_b_k_local, wo_b_k_local)
    _bind(
        module,
        "wo_b_weight",
        (f"{prefix}.attn.wo_b.weight",),
        _block_fp8_weight_loader(
            "wo_b_weight", f"{prefix}.attn.wo_b.weight", wo_b_shape, wo_b_shard
        ),
    )
    _bind(
        module,
        "wo_b_scale",
        (f"{prefix}.attn.wo_b.scale",),
        _block_fp8_scale_loader(
            "wo_b_scale", f"{prefix}.attn.wo_b.scale", wo_b_shape, wo_b_shard
        ),
    )

    # --- latent norms (replicated) and the per-head attention sink ----------
    _bind(
        module,
        "q_norm_weight",
        (f"{prefix}.attn.q_norm.weight",),
        _replicated_loader(
            "q_norm_weight", f"{prefix}.attn.q_norm.weight", (q_lora,)
        ),
    )
    _bind(
        module,
        "kv_norm_weight",
        (f"{prefix}.attn.kv_norm.weight",),
        _replicated_loader(
            "kv_norm_weight", f"{prefix}.attn.kv_norm.weight", (kv_lora,)
        ),
    )
    _bind(
        module,
        "attn_sink",
        (f"{prefix}.attn.attn_sink",),
        _dim0_slice_loader(
            "attn_sink",
            f"{prefix}.attn.attn_sink",
            (num_heads,),
            global_head,
            heads_per_rank,
        ),
    )

    # --- KV compressor: unquantized, replicated, ABSENT on layers 0 and 1 ---
    # The checkpoint index confirms the absence: layers 0 and 1 carry no
    # attn.compressor.* keys at all (they are the SWA-only layers,
    # compress_ratios[0:2] == [0, 0]). Asking for those keys would make the
    # loader raise on a checkpoint that is perfectly well-formed.
    if config.has_compressed_cache(layer_idx):
        ratio = config.compress_ratio(layer_idx)
        _attach_compressor_loaders(
            getattr(module, "compressor", None),
            key_prefix=f"{prefix}.attn.compressor",
            name_prefix="compressor",
            hidden=hidden,
            head_dim=kv_lora,
            compress_ratio=ratio,
            norm_dim=kv_lora,
        )

    # --- DSA lightning indexer: ratio-4 layers only -------------------------
    if config.has_indexer(layer_idx):
        indexer = getattr(module, "indexer", None)
        if indexer is None:
            raise AttributeError(
                f"layer {layer_idx} is a ratio-4 layer and its checkpoint carries "
                f"{prefix}.attn.indexer.*, but the attention module has no "
                "'indexer' submodule."
            )
        idx_heads = config.index_n_heads
        idx_dim = config.index_head_dim
        idx_wq_b_shape = (idx_heads * idx_dim, q_lora)
        # >>> PARALLELISM: replicated -- upstream builds this as a
        # ReplicatedLinear, so every core computes the full indexer logits. <<<
        idx_wq_b_shard = _Shard.replicated(*idx_wq_b_shape)
        _bind(
            indexer,
            "wq_b_weight",
            (f"{prefix}.attn.indexer.wq_b.weight",),
            _block_fp8_weight_loader(
                "indexer.wq_b_weight",
                f"{prefix}.attn.indexer.wq_b.weight",
                idx_wq_b_shape,
                idx_wq_b_shard,
            ),
        )
        _bind(
            indexer,
            "wq_b_scale",
            (f"{prefix}.attn.indexer.wq_b.scale",),
            _block_fp8_scale_loader(
                "indexer.wq_b_scale",
                f"{prefix}.attn.indexer.wq_b.scale",
                idx_wq_b_shape,
                idx_wq_b_shard,
            ),
        )
        _bind(
            indexer,
            "weights_proj_weight",
            (f"{prefix}.attn.indexer.weights_proj.weight",),
            _replicated_loader(
                "indexer.weights_proj_weight",
                f"{prefix}.attn.indexer.weights_proj.weight",
                (idx_heads, hidden),
            ),
        )
        _attach_compressor_loaders(
            getattr(indexer, "compressor", None),
            key_prefix=f"{prefix}.attn.indexer.compressor",
            name_prefix="indexer.compressor",
            hidden=hidden,
            head_dim=idx_dim,
            compress_ratio=4,
            norm_dim=idx_dim,
        )


def _attach_compressor_loaders(
    compressor: nn.Module | None,
    *,
    key_prefix: str,
    name_prefix: str,
    hidden: int,
    head_dim: int,
    compress_ratio: int,
    norm_dim: int,
) -> None:
    """Attach the four replicated KV-compressor loaders.

    Shapes follow the reference ``Compressor`` (``dsv4_ref/model.py:289-305``):
    ``coff = 1 + (compress_ratio == 4)`` (the ratio-4 compressor keeps an extra
    overlapping window, hence double width), so

    * ``ape``   -> ``[compress_ratio, coff * head_dim]`` fp32,
    * ``wkv`` / ``wgate`` -> ``[coff * head_dim, hidden]`` bf16, unquantized
      (the reference builds them with ``dtype=float32``, i.e. the no-scale
      branch of its ``Linear``),
    * ``norm``  -> ``[head_dim]``.

    All replicated: the compressor runs identically on every core.
    """
    if compressor is None:
        raise AttributeError(
            f"checkpoint carries {key_prefix}.* but the module has no "
            f"'{name_prefix}' submodule."
        )
    coff = 2 if compress_ratio == 4 else 1
    proj_shape = (coff * head_dim, hidden)
    for attr, suffix, shape in (
        ("wkv_weight", "wkv.weight", proj_shape),
        ("wgate_weight", "wgate.weight", proj_shape),
        ("ape", "ape", (compress_ratio, coff * head_dim)),
        ("norm_weight", "norm.weight", (norm_dim,)),
    ):
        key = f"{key_prefix}.{suffix}"
        _bind(
            compressor,
            attr,
            (key,),
            _replicated_loader(f"{name_prefix}.{attr}", key, shape),
        )


# ---------------------------------------------------------------------------
# Public: MoE
# ---------------------------------------------------------------------------


def attach_moe_loaders(
    module: nn.Module,
    config: "DeepseekV4Config",
    *,
    layer_idx: int,
    tp_size: int,
    tp_rank: int,
    ep_degree: int,
    ep_rank: int,
    shared_tp_size: int,
    shared_tp_rank: int,
    key_prefix: str | None = None,
) -> None:
    """Attach every checkpoint loader a :class:`DeepseekV4MoE` needs.

    >>> PARALLELISM <<<

    * **Router** (``gate_weight``, ``gate_bias``, ``tid2eid``): replicated.
      Every core must score every expert before dispatch.
    * **Routed experts**: expert-parallel, ``n_routed_experts // ep_degree``
      local experts (4 of 256 at EP=64). LD-23: the checkpoint's MXFP4 weights
      and group-32 E8M0 scales are REQUANTIZED here to legacy-E4M3 bytes plus
      one fp32 power-of-two scale per output channel, and the bytes are written
      transposed to ``[in, out]``. Both parameters read both checkpoint
      tensors; see the LD-23 section header for why the fold is bit-exact and
      why it raises instead of rounding.
    * **Shared expert**: sharded over a ``shared_tp_size``-way subgroup and
      replicated across the remaining subgroups (16-way with 4-fold replication
      at TP=64). A 64-way split would give ``K_local = 32`` on ``w2``, which
      breaks the 128-element activation group the block-FP8 path quantizes over
      -- see the note on ``DeepseekV4Config.shared_expert_tp``.

    ``gate_bias`` is bound only on the 40 non-hash layers and ``tid2eid`` only
    on the 3 hash layers (0, 1, 2). The checkpoint index is unambiguous: those
    three layers carry ``ffn.gate.tid2eid`` and no ``ffn.gate.bias``, the other
    40 carry ``ffn.gate.bias`` and no ``tid2eid``, and the reference declares
    exactly that either/or (``dsv4_ref/model.py:562-567``). Neither absence is
    an error.

    Args:
        module: The :class:`DeepseekV4MoE` instance.
        config: The family config.
        layer_idx: Index of the owning decoder layer. Required, not read off the
            module, because every checkpoint key here is layer-qualified and
            because which router tensor exists (``tid2eid`` vs ``bias``) depends
            on it.
        tp_size / tp_rank: Global tensor-parallel degree and rank. Only used for
            validation here: no MoE tensor is sharded on the global TP axis
            (the routed experts use the EP axis, the shared expert its own
            subgroup).
        ep_degree / ep_rank: Expert-parallel degree and this core's EP rank.
        shared_tp_size / shared_tp_rank: The shared expert's subgroup size and
            this core's rank inside it.
    """
    if tp_size <= 0 or not (0 <= tp_rank < tp_size):
        raise ValueError(
            f"attach_moe_loaders: invalid tp_rank={tp_rank} for tp_size={tp_size}."
        )

    if not isinstance(layer_idx, int):
        raise TypeError(
            f"attach_moe_loaders: layer_idx must be an int, got {layer_idx!r}."
        )
    # See ``_layer_key_prefix``: the DSpark stages pass ``mtp.{s}`` while still
    # handing the hash/no-hash decision the out-of-range ``layer_idx``.
    prefix = _layer_key_prefix(layer_idx) if key_prefix is None else key_prefix

    hidden = config.hidden_size
    inter = config.moe_intermediate_size
    n_experts = config.n_routed_experts

    # --- router -------------------------------------------------------------
    _bind(
        module,
        "gate_weight",
        (f"{prefix}.ffn.gate.weight",),
        _replicated_loader(
            "gate_weight", f"{prefix}.ffn.gate.weight", (n_experts, hidden)
        ),
    )

    is_hash = config.is_hash_moe_layer(layer_idx)
    if is_hash:
        _bind(
            module,
            "tid2eid",
            (f"{prefix}.ffn.gate.tid2eid",),
            _replicated_loader(
                "tid2eid",
                f"{prefix}.ffn.gate.tid2eid",
                (config.vocab_size, config.num_experts_per_tok),
            ),
            required=False,
        )
    else:
        _bind(
            module,
            "gate_bias",
            (f"{prefix}.ffn.gate.bias",),
            _replicated_loader(
                "gate_bias", f"{prefix}.ffn.gate.bias", (n_experts,)
            ),
            required=False,
        )

    # --- routed experts (expert-parallel, MXFP4 -> MXFP8) -------------------
    if ep_degree <= 0 or not (0 <= ep_rank < ep_degree):
        raise ValueError(
            f"attach_moe_loaders: invalid ep_rank={ep_rank} for "
            f"ep_degree={ep_degree}."
        )
    if n_experts % ep_degree != 0:
        raise ValueError(
            f"attach_moe_loaders: n_routed_experts={n_experts} is not divisible "
            f"by ep_degree={ep_degree}."
        )
    num_local = n_experts // ep_degree
    local_indices = list(range(ep_rank * num_local, (ep_rank + 1) * num_local))

    experts = getattr(module, "experts", None)
    if experts is None:
        raise AttributeError(
            "DeepseekV4MoE has no 'experts' submodule; the contract fixes "
            "experts.w1_weight / w1_scale / w3_* / w2_* as the routed-expert "
            "parameter names."
        )

    # ``w1``/``w3`` are the gate/up projections (hidden -> intermediate) and
    # ``w2`` the down projection (intermediate -> hidden), matching the
    # reference Expert (dsv4_ref/model.py:596-598). ``w1``/``w3`` contract over
    # ``H``, ``w2`` over ``I``; that is the only per-leaf difference now that
    # the MX tiling is retired (see the LD-23 section header).
    #
    # BOTH loaders take BOTH checkpoint tensors. The requantization folds the
    # E8M0 group scales into one power-of-two per output channel, so neither
    # the bytes nor the scale can be computed from its own tensor alone. The
    # key list is therefore ``weights then scales``: two groups of
    # ``n_experts``, which is exactly the layout
    # ``expert_parallel_grouped_loader`` documents and trims group-wise
    # (``utils/weight_loader.py:697-714``).
    for leaf, out_dim, logical_k in (
        ("w1", inter, hidden),
        ("w3", inter, hidden),
        ("w2", hidden, inter),
    ):
        all_weight_keys = [
            f"{prefix}.ffn.experts.{e}.{leaf}.weight" for e in range(n_experts)
        ]
        all_scale_keys = [
            f"{prefix}.ffn.experts.{e}.{leaf}.scale" for e in range(n_experts)
        ]
        local_weight_keys = [all_weight_keys[e] for e in local_indices]
        local_scale_keys = [all_scale_keys[e] for e in local_indices]
        paired_keys = [*all_weight_keys, *all_scale_keys]

        # The mapping enumerates all 256 per-expert keys (build_checkpoint_mappings
        # is a pure function of the config and cannot know this core's EP rank),
        # so the EP wrapper trims ``slices`` to the local contiguous range before
        # the requantization runs -- non-local experts are never read from disk.
        _bind(
            experts,
            f"{leaf}_weight",
            paired_keys,
            expert_parallel_grouped_loader(
                local_indices,
                _fp8_pow2_expert_weight_loader(
                    f"experts.{leaf}_weight",
                    local_weight_keys,
                    local_scale_keys,
                    out_dim,
                    logical_k,
                ),
                n_experts,
            ),
        )
        _bind(
            experts,
            f"{leaf}_scale",
            paired_keys,
            expert_parallel_grouped_loader(
                local_indices,
                _fp8_pow2_expert_scale_loader(
                    f"experts.{leaf}_scale",
                    local_weight_keys,
                    local_scale_keys,
                    out_dim,
                    logical_k,
                ),
                n_experts,
            ),
        )

    # --- shared expert (block-FP8 on a shared_tp_size-way subgroup) ---------
    shared = getattr(module, "shared_expert", None)
    if shared is None:
        raise AttributeError(
            "DeepseekV4MoE has no 'shared_expert' submodule; the contract fixes "
            "shared_expert.w1_weight / w1_scale / w3_* / w2_* as its parameter "
            "names."
        )
    if shared_tp_size <= 0 or not (0 <= shared_tp_rank < shared_tp_size):
        raise ValueError(
            f"attach_moe_loaders: invalid shared_tp_rank={shared_tp_rank} for "
            f"shared_tp_size={shared_tp_size}."
        )
    if inter % shared_tp_size != 0:
        raise ValueError(
            f"attach_moe_loaders: moe_intermediate_size={inter} is not divisible "
            f"by shared_tp_size={shared_tp_size}."
        )
    inter_local = inter // shared_tp_size

    shared_specs = (
        # (attr leaf, checkpoint leaf, full [out, in], shard)
        (
            "w1",
            "w1",
            (inter, hidden),
            _Shard.rows(shared_tp_rank * inter_local, inter_local, hidden),
        ),
        (
            "w3",
            "w3",
            (inter, hidden),
            _Shard.rows(shared_tp_rank * inter_local, inter_local, hidden),
        ),
        (
            "w2",
            "w2",
            (hidden, inter),
            _Shard.cols(hidden, shared_tp_rank * inter_local, inter_local),
        ),
    )
    for attr_leaf, ckpt_leaf, full_shape, shard in shared_specs:
        w_key = f"{prefix}.ffn.shared_experts.{ckpt_leaf}.weight"
        s_key = f"{prefix}.ffn.shared_experts.{ckpt_leaf}.scale"
        _bind(
            shared,
            f"{attr_leaf}_weight",
            (w_key,),
            _block_fp8_weight_loader(
                f"shared_expert.{attr_leaf}_weight", w_key, full_shape, shard
            ),
        )
        _bind(
            shared,
            f"{attr_leaf}_scale",
            (s_key,),
            _block_fp8_scale_loader(
                f"shared_expert.{attr_leaf}_scale", s_key, full_shape, shard
            ),
        )


# ---------------------------------------------------------------------------
# Public: DSpark draft stages (the stage-conditional tensors only)
# ---------------------------------------------------------------------------


def attach_dspark_stage_loaders(
    module: nn.Module,
    config: "DeepseekV4Config",
    *,
    stage_idx: int,
    tp_size: int,
    tp_rank: int,
) -> None:
    """Attach loaders for the tensors only SOME DSpark stages own.

    The shared per-stage set -- fused ``wq_a``+``wkv``, ``wq_b``, ``wo_a``,
    ``wo_b``, the norms, the hc mixes, the 256 routed experts and the shared
    expert -- is bound by :func:`attach_attention_loaders`,
    :func:`attach_hash_context_loaders` and :func:`attach_moe_loaders` with
    ``key_prefix="mtp.{s}"``, unchanged. Only these are extra:

    * stage 0: ``main_proj`` ``[hidden_size, 12288]`` block-FP8 + its scale
      grid (``dsv4_ref/model.py:832``).
    * last stage: ``confidence_head.proj`` ``[1, hidden_size + markov_rank]``
      (``:810``).

    Two tensors that deliberately get NO loader here: ``main_norm.weight`` and
    the last stage's ``norm.weight``. Both are replicated, unquantized and
    shape-identical to their checkpoint tensors, so the pipelined load's
    identity default is already correct -- the same treatment the main stack's
    final ``norm.weight`` gets. The Markov head's two vocab-sharded tensors and
    the drafter's ``embed_tokens``/``lm_head`` copies get theirs from
    ``sharding_weight_loader`` at construction, because they shard by the
    embedding / lm-head group's rank rather than the attention TP rank.

    >>> PARALLELISM: ``main_proj`` is COLUMN-PARALLEL in N and REPLICATED in K,
    as LD-18 records: ``N_local = hidden_size / tp_size`` = 64 at TP=64.

    K replication is forced rather than chosen -- 12288 / 64 = 192 and
    192 % 128 != 0, so a row shard would break the block-FP8 alignment
    invariant this family maintains everywhere.

    The N shard is half a 128-row scale block, so cores ``2k`` and ``2k+1``
    SHARE scale row ``k``. That is the contained-sub-block case
    :func:`_grid_shard` admits, and sharing is exact: every element of a block
    dequantizes by that one scalar, so each core's rows are bit-identical to
    the matching rows of a full-tensor dequant. The alternative -- replicating
    both axes -- would cost 48 MiB/core of fp8 weight AND make all 64 cores
    redundantly evaluate a ``[T, 12288] x [12288, 4096]`` GEMM that is the
    drafter's single largest, which on a long prefill is the dominant term. The
    shard's price is one all-gather of the ``[T, hidden_size]`` result, which
    :meth:`DeepseekV4DSparkStage.project_main_hidden` owns, because the
    downstream ``NF.mla_qkv`` needs the hidden stream replicated. <<<

    Args:
        module: The :class:`DeepseekV4DSparkStage` instance.
        config: The family config.
        stage_idx: Which stage this is; decides which tensors exist.
        tp_size / tp_rank: Attention tensor-parallel degree and rank.
            ``main_proj``'s N shard is taken by them; nothing else here is
            sharded.
    """
    if tp_size <= 0 or not (0 <= tp_rank < tp_size):
        raise ValueError(
            f"attach_dspark_stage_loaders: invalid tp_rank={tp_rank} for "
            f"tp_size={tp_size}."
        )
    if not (0 <= stage_idx < config.num_dspark_stages):
        raise ValueError(
            f"attach_dspark_stage_loaders: stage_idx={stage_idx} outside the "
            f"{config.num_dspark_stages} stages the weights record."
        )

    prefix = f"mtp.{stage_idx}"

    if stage_idx == 0:
        main_shape = (config.hidden_size, config.dspark_main_hidden_size)
        if main_shape[1] % _BLOCK != 0:
            raise ValueError(
                "attach_dspark_stage_loaders: main_proj K "
                f"({main_shape[1]}) must be a multiple of {_BLOCK} for the "
                "block-FP8 scale grid."
            )
        if main_shape[0] % tp_size != 0:
            raise ValueError(
                "attach_dspark_stage_loaders: main_proj N "
                f"({main_shape[0]}) must be divisible by tp_size={tp_size} for "
                "the column shard LD-18 records."
            )
        n_local = main_shape[0] // tp_size
        main_shard = _Shard.rows(tp_rank * n_local, n_local, main_shape[1])
        weight_key = f"{prefix}.main_proj.weight"
        scale_key = f"{prefix}.main_proj.scale"
        _bind(
            module,
            "main_proj_weight",
            (weight_key,),
            _block_fp8_weight_loader(
                "main_proj_weight", weight_key, main_shape, main_shard
            ),
        )
        _bind(
            module,
            "main_proj_scale",
            (scale_key,),
            _block_fp8_scale_loader(
                "main_proj_scale", scale_key, main_shape, main_shard
            ),
        )

    if stage_idx == config.num_dspark_stages - 1:
        # fp32 destination on purpose: the reference promotes this projection to
        # fp32 because that is what the confidence score needs
        # (``dsv4_ref/model.py:810``). The checkpoint ships it bf16, so the cast
        # happens in the model's ``_cast_to_model_dtype`` -- widening, so it is
        # lossless.
        conf_key = f"{prefix}.confidence_head.proj.weight"
        conf_shape = (1, config.hidden_size + config.dspark_markov_rank)
        _bind(
            module,
            "confidence_head.proj_weight",
            (conf_key,),
            _replicated_loader("confidence_head.proj_weight", conf_key, conf_shape),
        )


# ---------------------------------------------------------------------------
# Public: hash-context (mhc) residual stream
# ---------------------------------------------------------------------------


def attach_hash_context_loaders(
    module: nn.Module,
    config: "DeepseekV4Config",
    *,
    layer_idx: int | None = None,
    scope: str | None = None,
    key_prefix: str | None = None,
) -> None:
    """Attach loaders for the hash-context parameters and the block norms.

    All of these are fp32 (or bf16, for the norms) and replicated: the mhc
    machinery mixes the ``hc_mult``-wide residual stream identically on every
    core, so there is nothing to shard. The loaders exist to validate shapes
    against the config-derived dims, which are non-obvious:

    * ``hc_{attn,ffn}_base``  -> ``[(2 + hc_mult) * hc_mult]`` = ``[24]``
    * ``hc_{attn,ffn}_fn``    -> ``[(2 + hc_mult) * hc_mult, hc_mult * hidden]``
      = ``[24, 16384]``
    * ``hc_{attn,ffn}_scale`` -> ``[3]``
    * ``hc_head_base``        -> ``[hc_mult]`` = ``[4]``
    * ``hc_head_fn``          -> ``[hc_mult, hc_mult * hidden]`` = ``[4, 16384]``
    * ``hc_head_scale``       -> ``[1]``

    Those widths are the reference's (``dsv4_ref/model.py:669-678`` for the
    per-block set, ``:908-910`` for the head set; ``mix_hc = (2 + hc) * hc`` at
    ``kernel.py:374``): ``pre`` and ``post`` take ``hc_mult`` mixes each and the
    Sinkhorn combination matrix takes ``hc_mult**2``, hence ``2*hc + hc**2``,
    and the three ``hc_*_scale`` entries scale those three groups.

    Args:
        module: The module that owns the parameters -- a
            :class:`DeepseekV4HashContext` / decoder layer for the per-block
            set, or the backbone for the head set.
        config: The family config.
        layer_idx: Layer index for the per-block set. Defaults to
            ``module.layer_idx`` when present; unused for the head set, whose
            keys are top-level.
        scope: ``"layer"``, ``"head"`` or ``None``. ``None`` (default)
            auto-detects from which attributes the module actually declares, so
            a module that owns both sets gets both.

    Called positionally with exactly two arguments from ``model.py``; both
    keyword arguments exist only so the head set can be attached explicitly.
    """
    if scope not in (None, "layer", "head"):
        raise ValueError(
            f"attach_hash_context_loaders: scope must be 'layer', 'head' or None, "
            f"got {scope!r}."
        )

    hc_mult = config.hc_mult
    hidden = config.hidden_size
    mix_hc = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * hidden

    do_layer = scope in (None, "layer")
    do_head = scope in (None, "head")

    if do_layer:
        if layer_idx is None:
            layer_idx = getattr(module, "layer_idx", None)
        block_specs: list[tuple[str, str, tuple[int, ...]]] = []
        for which in ("attn", "ffn"):
            block_specs.extend(
                [
                    (f"hc_{which}_base", f"hc_{which}_base", (mix_hc,)),
                    (f"hc_{which}_fn", f"hc_{which}_fn", (mix_hc, hc_dim)),
                    (f"hc_{which}_scale", f"hc_{which}_scale", (3,)),
                ]
            )
        # The block norms live next to the hc parameters because the mhc reduce /
        # expand brackets them. The decoder layer owns them as RMSNorm
        # SUBMODULES (``attn_norm.weight``), so these are dotted binds; the flat
        # ``*_norm_weight`` spelling is also attempted for a module that declares
        # the tensors directly, and a spelling that does not exist is skipped.
        norm_specs = [
            ("attn_norm.weight", "attn_norm.weight", (hidden,)),
            ("ffn_norm.weight", "ffn_norm.weight", (hidden,)),
            ("attn_norm_weight", "attn_norm.weight", (hidden,)),
            ("ffn_norm_weight", "ffn_norm.weight", (hidden,)),
        ]
        if any(hasattr(module, attr) for attr, _, _ in block_specs + norm_specs):
            if not isinstance(layer_idx, int):
                raise ValueError(
                    "attach_hash_context_loaders: the per-block hash-context keys "
                    "are layer-qualified, so layer_idx (or module.layer_idx) is "
                    f"required; got {layer_idx!r}."
                )
            prefix = (
                _layer_key_prefix(layer_idx) if key_prefix is None else key_prefix
            )
            for attr, suffix, shape in block_specs + norm_specs:
                key = f"{prefix}.{suffix}"
                _bind(
                    module,
                    attr,
                    (key,),
                    _replicated_loader(attr, key, shape),
                    required=False,
                )

    if do_head:
        # Top-level keys on the main stack (no prefix). On the DSpark side the
        # SAME three tensors belong to the LAST draft stage
        # (``mtp.2.hc_head_*``, ``dsv4_ref/model.py:834-841``), which is
        # reached by passing ``key_prefix="mtp.2"``.
        head_prefix = "" if key_prefix is None else f"{key_prefix}."
        for attr, key, shape in (
            ("hc_head_base", f"{head_prefix}hc_head_base", (hc_mult,)),
            ("hc_head_fn", f"{head_prefix}hc_head_fn", (hc_mult, hc_dim)),
            ("hc_head_scale", f"{head_prefix}hc_head_scale", (1,)),
        ):
            _bind(
                module,
                attr,
                (key,),
                _replicated_loader(attr, key, shape),
                required=False,
            )


# ---------------------------------------------------------------------------
# Public: checkpoint key mapping
# ---------------------------------------------------------------------------


def build_checkpoint_mappings(
    config: "DeepseekV4Config",
    num_layers: int,
    *,
    mtp: bool = False,
    prefix: str = "model",
) -> dict[str, str | list[str]]:
    """Map each model parameter's qualified name to its checkpoint key(s).

    Every parameter needs an explicit entry, even the 1:1 ones: the load
    pipeline falls back to using the parameter's own dotted name as the
    checkpoint key, and this checkpoint has no ``model.`` prefix, so the
    fallback never matches. Parameters that come from several checkpoint
    tensors (the fused q/kv stack, the per-expert stacks) map to a list, in the
    order their loader's transform expects.

    Entries are emitted only for tensors the pinned checkpoint actually has, as
    the index confirms: ``ffn.gate.bias`` on the 40 non-hash layers,
    ``ffn.gate.tid2eid`` on layers 0-2, the KV compressor from layer 2 on, the
    DSA indexer on the ratio-4 layers. Scale entries are emitted too; they are
    inert for scales that the modules register as buffers (the pipelined load
    iterates ``named_parameters()`` only, so a mapped name with no matching
    parameter is simply never looked up) and are consumed by
    :func:`load_block_scale_buffers` instead.

    Args:
        config: The family config.
        num_layers: Number of main-stack decoder layers. Ignored when
            ``mtp=True`` (the stage count then comes from
            ``config.num_dspark_stages``, which is ruled by the weights --
            see :data:`~.config.CONTRADICTED_CHECKPOINT_FIELDS`).
        mtp: Emit the DSPARK DRAFTER's mapping instead of the main stack's.
            The drafter is a separate top-level module with its own parameter
            tree (``stages.{s}.*``, plus its own ``embed_tokens``/``lm_head``),
            so its mapping is disjoint from the main stack's rather than an
            addition to it; call it once per module. Each stage reuses
            :func:`add_block` verbatim against ``key_prefix = "mtp.{s}"`` and
            ``layer_idx = config.num_hidden_layers + s`` -- which is what makes
            every stage emit ``ffn.gate.bias`` and no ``ffn.gate.tid2eid``
            (``is_hash_moe_layer`` is False past layer 2) and no compressor or
            indexer keys (``compress_ratios[43:46] == [0, 0, 0]``), exactly as
            the ``mtp.*`` key census reports.
        prefix: Dotted attribute path of the backbone inside the top-level
            module whose ``named_parameters()`` will be matched (llama3's
            equivalent is the literal ``"model"``). Keyword-only with a default
            so the recorded call form stays valid. The drafter passes ``""``:
            its stages hang off the top-level module directly.

    Returns:
        ``{parameter name: checkpoint key | [checkpoint keys]}``.
    """
    mappings: dict[str, str | list[str]] = {}
    p = prefix.rstrip(".")

    def put(param: str, keys: str | list[str]) -> None:
        mappings[f"{p}.{param}" if p else param] = keys

    n_experts = config.n_routed_experts

    def add_block(param_layer: str, key_prefix: str, *, layer_idx: int) -> None:
        """Emit one main-stack transformer block's entries."""
        has_compressor = config.has_compressed_cache(layer_idx)
        has_indexer = config.has_indexer(layer_idx)
        is_hash = config.is_hash_moe_layer(layer_idx)

        # norms + hash-context mixes
        for suffix in (
            "attn_norm.weight",
            "ffn_norm.weight",
            "hc_attn_base",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
        ):
            attr = suffix if suffix.endswith(".weight") else suffix
            put(f"{param_layer}.{attr}", f"{key_prefix}.{suffix}")
        # ``*_norm.weight`` may instead be flat ``*_norm_weight`` attributes;
        # emit both spellings. An entry with no matching parameter is inert.
        for flat, suffix in (
            ("attn_norm_weight", "attn_norm.weight"),
            ("ffn_norm_weight", "ffn_norm.weight"),
        ):
            put(f"{param_layer}.{flat}", f"{key_prefix}.{suffix}")

        # attention
        attn = f"{param_layer}.self_attn"
        put(
            f"{attn}.fused_wqa_wkv_weight",
            [f"{key_prefix}.attn.wq_a.weight", f"{key_prefix}.attn.wkv.weight"],
        )
        put(
            f"{attn}.fused_wqa_wkv_scale",
            [f"{key_prefix}.attn.wq_a.scale", f"{key_prefix}.attn.wkv.scale"],
        )
        for leaf in ("wq_b", "wo_a", "wo_b"):
            put(f"{attn}.{leaf}_weight", f"{key_prefix}.attn.{leaf}.weight")
            put(f"{attn}.{leaf}_scale", f"{key_prefix}.attn.{leaf}.scale")
        put(f"{attn}.q_norm_weight", f"{key_prefix}.attn.q_norm.weight")
        put(f"{attn}.kv_norm_weight", f"{key_prefix}.attn.kv_norm.weight")
        put(f"{attn}.attn_sink", f"{key_prefix}.attn.attn_sink")

        if has_compressor:
            for attr, suffix in (
                ("wkv_weight", "wkv.weight"),
                ("wgate_weight", "wgate.weight"),
                ("ape", "ape"),
                ("norm_weight", "norm.weight"),
            ):
                put(
                    f"{attn}.compressor.{attr}",
                    f"{key_prefix}.attn.compressor.{suffix}",
                )

        if has_indexer:
            idx = f"{attn}.indexer"
            ikey = f"{key_prefix}.attn.indexer"
            put(f"{idx}.wq_b_weight", f"{ikey}.wq_b.weight")
            put(f"{idx}.wq_b_scale", f"{ikey}.wq_b.scale")
            put(f"{idx}.weights_proj_weight", f"{ikey}.weights_proj.weight")
            for attr, suffix in (
                ("wkv_weight", "wkv.weight"),
                ("wgate_weight", "wgate.weight"),
                ("ape", "ape"),
                ("norm_weight", "norm.weight"),
            ):
                put(f"{idx}.compressor.{attr}", f"{ikey}.compressor.{suffix}")

        # MoE
        moe = f"{param_layer}.mlp"
        put(f"{moe}.gate_weight", f"{key_prefix}.ffn.gate.weight")
        if is_hash:
            put(f"{moe}.tid2eid", f"{key_prefix}.ffn.gate.tid2eid")
        else:
            put(f"{moe}.gate_bias", f"{key_prefix}.ffn.gate.bias")

        for leaf in ("w1", "w3", "w2"):
            # LD-23: BOTH routed-expert parameters read BOTH checkpoint
            # tensors, because the requantization folds the E8M0 group scales
            # into one power-of-two per output channel and neither result is
            # computable from its own tensor alone. The order is
            # ``all weights then all scales`` -- the two-group layout
            # ``expert_parallel_grouped_loader`` trims group-wise
            # (``utils/weight_loader.py:697-714``) and the layout
            # ``_split_weight_and_scale_slices`` asserts. It must stay in step
            # with the ``paired_keys`` list in ``attach_moe_loaders``.
            expert_weight_keys = [
                f"{key_prefix}.ffn.experts.{e}.{leaf}.weight"
                for e in range(n_experts)
            ]
            expert_scale_keys = [
                f"{key_prefix}.ffn.experts.{e}.{leaf}.scale"
                for e in range(n_experts)
            ]
            paired = [*expert_weight_keys, *expert_scale_keys]
            put(f"{moe}.experts.{leaf}_weight", paired)
            put(f"{moe}.experts.{leaf}_scale", paired)
            put(
                f"{moe}.shared_expert.{leaf}_weight",
                f"{key_prefix}.ffn.shared_experts.{leaf}.weight",
            )
            put(
                f"{moe}.shared_expert.{leaf}_scale",
                f"{key_prefix}.ffn.shared_experts.{leaf}.scale",
            )

    if mtp:
        # ── DSpark drafter ───────────────────────────────────────────────
        # Every stage's shared parameter set comes from ``add_block`` above,
        # unchanged: same fused wq_a+wkv pair, same wq_b/wo_a/wo_b, same 256
        # routed experts + shared expert + gate.bias, same hc mixes and block
        # norms. Only the three stage-specific groups below are extra.
        last = config.num_dspark_stages - 1
        for stage in range(config.num_dspark_stages):
            add_block(
                f"stages.{stage}",
                f"mtp.{stage}",
                layer_idx=config.num_hidden_layers + stage,
            )

        # Stage 0: the target-hidden down-projection (``dsv4_ref/model.py:832``).
        put("stages.0.main_proj_weight", "mtp.0.main_proj.weight")
        put("stages.0.main_proj_scale", "mtp.0.main_proj.scale")
        put("stages.0.main_norm.weight", "mtp.0.main_norm.weight")
        put("stages.0.main_norm_weight", "mtp.0.main_norm.weight")

        # Last stage: final norm, its OWN hc_head set (top-level on the main
        # stack, ``mtp.2``-qualified here -- ``dsv4_ref/model.py:834-841``),
        # the Markov head and the confidence head.
        put(f"stages.{last}.norm.weight", f"mtp.{last}.norm.weight")
        put(f"stages.{last}.norm_weight", f"mtp.{last}.norm.weight")
        for attr in ("hc_head_base", "hc_head_fn", "hc_head_scale"):
            put(f"stages.{last}.{attr}", f"mtp.{last}.{attr}")
        put(
            f"stages.{last}.markov_head.w1.weight",
            f"mtp.{last}.markov_head.markov_w1.weight",
        )
        put(
            f"stages.{last}.markov_head.w2.weight",
            f"mtp.{last}.markov_head.markov_w2.weight",
        )
        put(
            f"stages.{last}.confidence_head.proj_weight",
            f"mtp.{last}.confidence_head.proj.weight",
        )

        # The drafter is a separately compiled module, so it cannot share the
        # target's parameter tensors the way the reference does
        # (``dsv4_ref/model.py:903-904`` assigns the same objects). It loads
        # its own copy of the SAME two checkpoint tensors.
        put("embed_tokens.weight", "embed.weight")
        put("lm_head.weight", "head.weight")
        return mappings

    for layer_idx in range(num_layers):
        add_block(
            f"layers.{layer_idx}",
            _layer_key_prefix(layer_idx),
            layer_idx=layer_idx,
        )
    put("norm.weight", "norm.weight")
    put("hc_head_base", "hc_head_base")
    put("hc_head_fn", "hc_head_fn")
    put("hc_head_scale", "hc_head_scale")

    # Embedding and LM head. Untied (``tie_word_embeddings=false``), so ``head``
    # is its own checkpoint tensor.
    put("embed_tokens.weight", "embed.weight")
    mappings["lm_head.weight"] = "head.weight"

    return mappings


# ---------------------------------------------------------------------------
# Public: post-hoc scale-buffer load
# ---------------------------------------------------------------------------


def load_block_scale_buffers(
    module_tree: nn.Module,
    checkpoint: "SafetensorsCheckpoint",
    rank: int,
    device: torch.device,
) -> None:
    """Load every loader-bound tensor that is a *buffer*, not a parameter.

    Why this exists: the pipelined checkpoint reader walks
    ``named_parameters()`` only, so anything a module registers with
    ``register_buffer(..., persistent=False)`` -- the natural home for a
    dequant scale that must not be touched by ``load_state_dict`` -- is skipped
    entirely, and would silently stay at its ``torch.empty`` contents. llama3
    solves the same problem with its ``_FP8_*_SCALE_SOURCES`` tables; here the
    sources are recorded by the ``attach_*`` functions themselves, so the
    checkpoint-key convention stays in one file.

    Parameters are deliberately skipped: they are already covered by
    :func:`build_checkpoint_mappings` plus the pipelined load, and loading them
    twice would waste the disk read.

    Args:
        module_tree: Any module; its whole subtree is walked.
        checkpoint: An open :class:`SafetensorsCheckpoint`.
        rank: Rank passed to each loader. The shard is already baked into every
            transform in this module, so this only matters for loaders that
            read it.
        device: Device the loaded tensors are moved to.
    """
    available = checkpoint.get_tensor_names()
    loaded = 0

    for module_name, module in module_tree.named_modules():
        for attr, ckpt_keys, loader in getattr(module, _SCALE_SOURCES_ATTR, ()):
            qualified = f"{module_name}.{attr}" if module_name else attr
            current = getattr(module, attr, None)
            if isinstance(current, nn.Parameter):
                continue  # owned by the pipelined named_parameters() load

            missing = [k for k in ckpt_keys if k not in available]
            if missing:
                raise KeyError(
                    f"Missing checkpoint key(s) for buffer {qualified!r}: "
                    f"{missing}. This buffer is bound to those keys by the "
                    "deepseek_v4 weight loaders, so either the checkpoint "
                    "revision changed or the wrong attach function ran."
                )

            slices = [checkpoint._get_slice(k) for k in ckpt_keys]
            tensor = loader.load(slices, rank).to(device)
            setattr(module, attr, tensor)
            loaded += 1

    logger.info(
        "deepseek_v4: loaded %d non-parameter (buffer) tensors from the checkpoint.",
        loaded,
    )
