# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for `inc-glm53f-045` -- the DSA ragged pack, an AUTHORED bf16 NKI kernel pair.

Declared command (D1 Tier N, the block's shorthand expanded to the tier's byte form)::

    PYTHONDONTWRITEBYTECODE=1 VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    python -m pytest test/vllm_neuron/functional/dsa/test_ragged_pack.py \\
        --timeout 60 -v -s -p no:cacheprovider

ONE ITEM PER COUNTED CONJUNCT AND NO ``parametrize`` (§6 rule 4b / rule 6), so the item count is
derivable before the run: NINE. Four round-trip items, one per declared raggedness pattern; the
route-predicate item, which resets and reads the counters once per case as §4b requires; the two
D1.5 controls the increment plan names; the kernel-identity item; and a ninth item that is the
non-vacuity control for the padding-zero reading. NO FIXTURE SETS ``NEURON_PLATFORM_TARGET_OVERRIDE``
OR ``NKI_SIMULATOR`` (§6 rule 3, D2): both resolve at import time and belong in the process
invocation, so a fixture that set either would have measured the wrong thing.

THE FOUR DECLARED PATTERNS, and why these four. Recorded in the increment plan at revision 151.
Row lengths, then the packed length the pack must produce:

    uniform         (128, 128, 128, 128)  -> 512   a tile multiple
    aligned_ragged  (256, 128,  64,  64)  -> 512   a tile multiple
    tile_crossing   (100,  37, 200,  48)  -> 385   NOT a tile multiple (3 * 128 + 1)
    skew            (  1, 511,   3,   5)  -> 520   NOT a tile multiple (4 * 128 + 8)

``tile_crossing`` IS THE DISCRIMINATOR, and the reason is specific. The kernel tiles the partition
axis at ``nl.tile_size.pmax == 128``. An implementation that rounded each sequence up to the tile,
or rounded the total up to the next tile, would produce 512 here instead of 385 -- so the
closed-form length assertion can tell the tile size from a pattern parameter, which is what the
B64-N1 form asks of a case set. ``skew`` adds the other extreme: a length-1 sequence beside a
511-row sequence that straddles four tiles. ``uniform`` and ``aligned_ragged`` share a packed length
of 512 under DIFFERENT raggedness, so the length assertion alone cannot separate them and the
bit-identity assertion has to do that work.

WHY THE PACK IS CHECKED AGAINST AN ORACLE AND NOT ONLY THROUGH THE ROUND TRIP. A round trip can
pass on two mutually cancelling bugs -- a pack that lays rows down wrongly and an unpack that reads
them back by the same wrong rule agree with each other and disagree with reality. So every
round-trip item ALSO compares the packed buffer itself against a torch reference built by slicing
and concatenating, which pins the intermediate rather than only the composition.

WHY THE PADDING IS ZEROED IN THE INPUT. The criterion is that packing then unpacking reproduces the
input bit for bit, and the pack DROPS padding rows by design, so the round trip can only reproduce
padding the caller can predict. Zero is that value, it is what the fork's own convention writes, and
the unpack regenerates it from a memset sentinel row rather than from anything the pack carried --
so the zeros in the result are produced by the unpack's own path and are not an echo of the input.

COMPARISON IS ON RAW BIT PATTERNS, NOT ONLY ON VALUES. ``max abs diff == 0.0`` is the criterion the
increment plan declares and it is asserted; but a value comparison cannot distinguish two different
bit patterns that read as the same float, and this kernel pair has one specific way to produce
exactly that -- ``-0.0`` compares equal to ``+0.0`` while differing in the sign bit. The mechanism
probe measured 32,652 of 65,536 elements landing on ``-0.0`` under the design this file's module
deliberately does NOT use. So the bit patterns are compared as int16 as well, and both readings are
printed for every pattern.

WHY ``bfloat16``. It is the dtype the increment's substrate declaration names and the only one the
campaign has measured on this pin; the module admits it and serves every other dtype through the
torch path, which is what the second control below drives.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional.dsa.ragged_pack import (
    can_run_dsa_ragged_pack,
    dsa_ragged_pack,
    dsa_ragged_unpack,
    ragged_pack_dispatch_counters,
    ragged_pack_kernel_identity,
    reset_ragged_pack_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

WIDTH = 512
PARTITION_TILE = 128  # nl.tile_size.pmax, named here so the tile-crossing claim is legible

# The four declared raggedness patterns, as ``(label, row lengths)``.
PATTERN_UNIFORM = ("uniform", (128, 128, 128, 128))
PATTERN_ALIGNED_RAGGED = ("aligned_ragged", (256, 128, 64, 64))
PATTERN_TILE_CROSSING = ("tile_crossing", (100, 37, 200, 48))
PATTERN_SKEW = ("skew", (1, 511, 3, 5))
PATTERNS = (PATTERN_UNIFORM, PATTERN_ALIGNED_RAGGED, PATTERN_TILE_CROSSING, PATTERN_SKEW)

# Value tolerance, ORDER NAMED INLINE (D3): (rtol, atol) == (0.0, 0.0). BOTH are exactly zero
# because a pack moves rows and computes nothing: the expected difference is not "small", it is
# none. A nonzero tolerance here would hide the only failure mode these items have.
VALUE_RTOL = 0.0
VALUE_ATOL = 0.0


def _closed_form_packed_length(lengths: tuple[int, ...]) -> int:
    """The packed length, computed HERE and not read from the module under test.

    The module derives the same number from the same lengths; this is the second, independent
    derivation, which is what makes the comparison a measurement instead of an echo.
    """
    total = 0
    for length in lengths:
        total += length
    return total


def _fixture(label: str, lengths: tuple[int, ...]) -> dict:
    """Build one declared pattern, plus the packed reference the items compare against."""
    batch = len(lengths)
    max_len = max(lengths)
    packed_len = _closed_form_packed_length(lengths)

    gen = torch.Generator().manual_seed(45_000 + packed_len + max_len)
    padded = torch.randn(batch, max_len, WIDTH, generator=gen).to(torch.bfloat16)
    # Zero every padding position, for the reason the module docstring of this file gives.
    for b, length in enumerate(lengths):
        padded[b, length:, :] = 0
    padded = padded.contiguous()

    # The packed reference: each sequence's valid rows, sliced and concatenated in order.
    want_packed = torch.cat([padded[b, : lengths[b], :] for b in range(batch)], dim=0).contiguous()

    # The length-BLIND reference: what a pack that ignored the lengths and took the first
    # ``packed_len`` rows of the flattened batch would produce. Used only by the control, which
    # asserts it DIFFERS.
    blind_packed = padded.reshape(batch * max_len, WIDTH)[:packed_len, :].contiguous()

    print(
        f"[fixture] {label} lengths={lengths} batch={batch} max_len={max_len} "
        f"packed_len={packed_len} packed_len_is_tile_multiple={packed_len % PARTITION_TILE == 0} "
        f"padded_rows={batch * max_len} width={WIDTH} dtype={padded.dtype}"
    )
    return dict(
        label=label,
        lengths=lengths,
        batch=batch,
        max_len=max_len,
        packed_len=packed_len,
        padded=padded,
        want_packed=want_packed,
        blind_packed=blind_packed,
    )


def _differing_elements(a: torch.Tensor, b: torch.Tensor) -> int:
    """Elements differing in their INT16 VIEW, not numerically.

    Bitwise, because the criterion is bit-identity and because ``-0.0`` compares equal to ``+0.0``
    while differing in the sign bit -- the one hazard this kernel pair has to avoid.
    """
    assert a.shape == b.shape, f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
    ai = a.contiguous().view(torch.int16)
    bi = b.contiguous().view(torch.int16)
    return int((ai != bi).sum().item())


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """The criterion's own reading: ``max abs diff``, computed in float32 so it cannot round."""
    return float((a.float() - b.float()).abs().max().item())


def _padding_nonzero_elements(unpacked: torch.Tensor, lengths: tuple[int, ...]) -> int:
    """Elements at PADDING positions whose int16 view is not exactly ``0x0000``.

    Negative zero counts as nonzero here, deliberately: ``0x8000`` is not the value the unpack
    claims to write. Item nine is the control that proves this reading fires.
    """
    bad = 0
    for b, length in enumerate(lengths):
        tail = unpacked[b, length:, :]
        if tail.numel() == 0:
            continue
        bad += int((tail.contiguous().view(torch.int16) != 0).sum().item())
    return bad


def _round_trip(fx: dict) -> None:
    """The acceptance body for one declared pattern. Called by exactly one item per pattern."""
    label = fx["label"]
    padded, lengths = fx["padded"], fx["lengths"]

    packed = dsa_ragged_pack(padded, lengths)

    # THE CLOSED-FORM LENGTH. The module was never told this number; it derived it from the same
    # lengths, and this compares it against the derivation in this file.
    print(
        f"[acceptance] {label} packed_shape={tuple(packed.shape)} "
        f"closed_form_packed_len={fx['packed_len']} width={WIDTH}"
    )
    assert int(packed.shape[0]) == fx["packed_len"], (
        f"{label}: packed length {int(packed.shape[0])} is not the closed-form "
        f"{fx['packed_len']}"
    )
    assert int(packed.shape[1]) == WIDTH

    # THE INTERMEDIATE, pinned against a torch reference so two cancelling bugs cannot pass.
    packed_diff = _differing_elements(packed.to(torch.bfloat16), fx["want_packed"])
    print(f"[acceptance] {label} packed_vs_oracle_differing_elements={packed_diff}")
    assert packed_diff == 0, f"{label}: the packed buffer differs from the sliced reference"

    unpacked = dsa_ragged_unpack(packed, lengths, fx["max_len"])
    assert tuple(unpacked.shape) == tuple(padded.shape)

    max_abs = _max_abs_diff(unpacked.to(torch.bfloat16), padded)
    differing = _differing_elements(unpacked.to(torch.bfloat16), padded)
    padding_bad = _padding_nonzero_elements(unpacked.to(torch.bfloat16), lengths)
    print(
        f"[acceptance] {label} round_trip_max_abs_diff={max_abs} "
        f"round_trip_differing_elements={differing} "
        f"padding_nonzero_elements={padding_bad} rtol={VALUE_RTOL} atol={VALUE_ATOL}"
    )
    assert max_abs == 0.0, f"{label}: max abs diff {max_abs} is not exactly zero"
    assert differing == 0, f"{label}: {differing} elements differ bitwise after the round trip"
    assert padding_bad == 0, f"{label}: {padding_bad} padding elements are not positive zero"


def test_round_trip_is_bit_identical_on_uniform() -> None:
    """`uniform` -- four equal sequences, packed length 512, a tile multiple."""
    _round_trip(_fixture(*PATTERN_UNIFORM))


def test_round_trip_is_bit_identical_on_aligned_ragged() -> None:
    """`aligned_ragged` -- unequal sequences whose packed length is still 512."""
    _round_trip(_fixture(*PATTERN_ALIGNED_RAGGED))


def test_round_trip_is_bit_identical_on_tile_crossing() -> None:
    """`tile_crossing` -- packed length 385, which is 3 * 128 + 1 and so NOT a tile multiple.

    This is the item that can tell the partition tile size from a pattern parameter: an
    implementation that rounded to the tile would report 512 here.
    """
    fx = _fixture(*PATTERN_TILE_CROSSING)
    assert fx["packed_len"] % PARTITION_TILE != 0, "this pattern must not be tile aligned"
    _round_trip(fx)


def test_round_trip_is_bit_identical_on_skew() -> None:
    """`skew` -- a length-1 sequence beside a 511-row sequence, packed length 520."""
    fx = _fixture(*PATTERN_SKEW)
    assert fx["packed_len"] % PARTITION_TILE != 0, "this pattern must not be tile aligned"
    _round_trip(fx)


def test_route_predicate_two_nki_dispatches_and_zero_torch_fallback_per_pattern() -> None:
    """D13 form R-1: the seams this increment authors count their own dispatches.

    TWO per declared pattern -- one for the pack and one for the unpack -- because the round trip
    is what the acceptance asserts and a single-direction reading would leave the inverse
    unmeasured. The counters are reset at the start of each case and read at its end, which is
    §4b's per-case convention. ``can_run_kernel`` is read on the same payload the seam was handed,
    and the module's torch-fallback counter must be exactly zero: a pure-torch implementation of
    this module would read ``0`` NKI dispatches and therefore cannot pass this item.
    """
    for label, lengths in PATTERNS:
        fx = _fixture(label, lengths)
        reset_ragged_pack_dispatch_counters()
        gate = bool(can_run_kernel(fx["padded"]))
        envelope = bool(can_run_dsa_ragged_pack(fx["padded"], lengths))
        packed = dsa_ragged_pack(fx["padded"], lengths)
        dsa_ragged_unpack(packed, lengths, fx["max_len"])
        nki_dispatch, torch_fallback = ragged_pack_dispatch_counters()
        print(
            f"[route-predicate] {label} can_run_kernel={gate} "
            f"can_run_dsa_ragged_pack={envelope} nki_dispatch={nki_dispatch} "
            f"torch_fallback={torch_fallback}"
        )
        assert gate is True, f"{label}: the runtime NKI gate is not open"
        assert envelope is True, f"{label}: this module does not admit its own declared geometry"
        assert nki_dispatch == 2, f"{label}: expected 2 NKI dispatches, read {nki_dispatch}"
        assert torch_fallback == 0, f"{label}: expected 0 torch fallbacks, read {torch_fallback}"


def test_control_a_length_blind_pack_differs_from_the_real_pack() -> None:
    """D1.5 control one: the bit-identity readings above are not blind to the lengths.

    A pack that ignored the lengths and took the first ``packed_len`` rows of the flattened batch
    would produce a DIFFERENT buffer, and this item measures that difference rather than asserting
    it. Run on ``aligned_ragged``, whose packed length is the same 512 as ``uniform``'s -- so this
    control is exactly what separates the two patterns that the length assertion cannot.
    """
    label, lengths = PATTERN_ALIGNED_RAGGED
    fx = _fixture(label, lengths)
    reset_ragged_pack_dispatch_counters()
    packed = dsa_ragged_pack(fx["padded"], lengths)
    blind_diff = _differing_elements(packed.to(torch.bfloat16), fx["blind_packed"])
    real_diff = _differing_elements(packed.to(torch.bfloat16), fx["want_packed"])
    print(
        f"[control] {label} blind_pack_differing_elements={blind_diff} "
        f"real_pack_differing_elements={real_diff} "
        f"blind_shape={tuple(fx['blind_packed'].shape)}"
    )
    assert real_diff == 0, "the real pack must match the sliced reference"
    assert blind_diff > 0, (
        "a length-blind pack produced the same bytes as the real one, so the bit-identity "
        "readings in this file would pass without the lengths being used at all"
    )


def test_control_an_unadmitted_dtype_takes_the_torch_route() -> None:
    """D1.5 control two: the counted zero above is a measurement, because this drives it to one.

    ``float32`` is outside the module's admitted dtypes, so the gate closes and the torch path
    serves the call. The fallback counter must read exactly ``1`` and the NKI counter exactly
    ``0`` -- which is what proves the ``0`` the route-predicate item reads is a reading and not a
    counter that never moves. The result is checked too, so the fallback is shown to be correct
    rather than merely taken.
    """
    label, lengths = PATTERN_TILE_CROSSING
    fx = _fixture(label, lengths)
    padded_f32 = fx["padded"].to(torch.float32).contiguous()
    reset_ragged_pack_dispatch_counters()
    gate = bool(can_run_dsa_ragged_pack(padded_f32, lengths))
    packed = dsa_ragged_pack(padded_f32, lengths)
    nki_dispatch, torch_fallback = ragged_pack_dispatch_counters()
    want = torch.cat(
        [padded_f32[b, : lengths[b], :] for b in range(len(lengths))], dim=0
    ).contiguous()
    differing = int((packed != want).sum().item())
    print(
        f"[control] unadmitted_dtype={padded_f32.dtype} can_run_dsa_ragged_pack={gate} "
        f"nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
        f"fallback_differing_elements={differing}"
    )
    assert gate is False, "float32 must not be admitted to the kernel"
    assert nki_dispatch == 0, f"expected 0 NKI dispatches on the torch route, read {nki_dispatch}"
    assert torch_fallback == 1, f"expected exactly 1 torch fallback, read {torch_fallback}"
    assert differing == 0, "the torch route returned the wrong rows"


def test_kernel_identity_is_this_modules_own_kernels_through_the_seam() -> None:
    """D13.1: the identity is derived by TAKING the dispatch branch, not read off an import.

    ``None`` before any dispatch is the reading that distinguishes "no kernel ran" from "some
    kernel ran". After a pack it must name this module's own pack kernel, and after an unpack its
    own unpack kernel -- so the identity tracks the direction and cannot be a constant.
    """
    label, lengths = PATTERN_UNIFORM
    fx = _fixture(label, lengths)
    reset_ragged_pack_dispatch_counters()
    before = ragged_pack_kernel_identity()
    print(f"[identity] before_any_dispatch={before}")
    assert before is None, "an identity was reported before any kernel ran"

    packed = dsa_ragged_pack(fx["padded"], lengths)
    after_pack = ragged_pack_kernel_identity()
    print(f"[identity] after_pack_dispatch={after_pack}")
    assert after_pack == (
        "vllm_neuron.functional.dsa.ragged_pack",
        "_ragged_pack_nki",
    ), f"the pack seam dispatched {after_pack}"

    dsa_ragged_unpack(packed, lengths, fx["max_len"])
    after_unpack = ragged_pack_kernel_identity()
    print(f"[identity] after_unpack_dispatch={after_unpack}")
    assert after_unpack == (
        "vllm_neuron.functional.dsa.ragged_pack",
        "_ragged_unpack_nki",
    ), f"the unpack seam dispatched {after_unpack}"


def test_control_the_padding_zero_reading_fires_on_a_nonzero_padding() -> None:
    """The non-vacuity control for the padding-zero reading in every acceptance item above.

    A counted zero needs a control that FIRES (D1.5). This one takes the unpacked result, plants a
    single nonzero value at one padding position, and measures that the very same predicate the
    acceptance items use reports it. It also plants a NEGATIVE ZERO, because that is the value a
    mask-by-multiply design would have written and the one a numeric comparison cannot see.
    """
    label, lengths = PATTERN_SKEW
    fx = _fixture(label, lengths)
    reset_ragged_pack_dispatch_counters()
    packed = dsa_ragged_pack(fx["padded"], lengths)
    unpacked = dsa_ragged_unpack(packed, lengths, fx["max_len"]).to(torch.bfloat16)

    clean = _padding_nonzero_elements(unpacked, lengths)
    # Sequence 0 has length 1 in this pattern, so row 1 of sequence 0 is a padding position.
    planted = unpacked.clone()
    planted[0, 1, 0] = torch.tensor(1.5, dtype=torch.bfloat16)
    fires_on_value = _padding_nonzero_elements(planted, lengths)

    negative_zero = unpacked.clone()
    negative_zero[0, 1, 0] = torch.tensor(-0.0, dtype=torch.bfloat16)
    fires_on_negative_zero = _padding_nonzero_elements(negative_zero, lengths)
    numeric_blind = _max_abs_diff(negative_zero, unpacked)

    print(
        f"[control] {label} padding_nonzero_on_clean={clean} "
        f"padding_nonzero_on_planted_value={fires_on_value} "
        f"padding_nonzero_on_planted_negative_zero={fires_on_negative_zero} "
        f"max_abs_diff_a_numeric_check_would_have_seen={numeric_blind}"
    )
    assert clean == 0, "the unpacked padding was not already zero"
    assert fires_on_value == 1, "the padding-zero reading did not fire on a planted value"
    assert fires_on_negative_zero == 1, (
        "the padding-zero reading did not fire on a planted negative zero, so it could not "
        "distinguish the sentinel design from a mask-by-multiply design"
    )
    assert numeric_blind == 0.0, (
        "a numeric check must be blind to the planted negative zero, which is the whole reason "
        "these readings are taken on bit patterns"
    )
