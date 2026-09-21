# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for the ragged pack/unpack kernel pair.

Four row-length patterns are driven: two that pack to a tile multiple (512)
under different raggedness, and two that do not (385 and 520), so the length
assertion and the bit-identity assertion each catch a different bug on their
own. Padding rows are zeroed on input; the unpack regenerates zero padding
from a sentinel rather than echoing the input, and every round trip is also
checked against a torch reference built by slicing and concatenating, so two
cancelling pack/unpack bugs cannot pass by agreeing with each other.

Comparisons are on raw bit patterns as well as values, because ``-0.0`` reads
equal to ``+0.0`` in a numeric comparison but differs in the sign bit, and
this kernel pair can produce that on the padding rows.
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
PARTITION_TILE = 128  # nl.tile_size.pmax, named here so the tile-crossing checks are legible

# (label, row lengths) for the four patterns.
PATTERN_UNIFORM = ("uniform", (128, 128, 128, 128))
PATTERN_ALIGNED_RAGGED = ("aligned_ragged", (256, 128, 64, 64))
PATTERN_TILE_CROSSING = ("tile_crossing", (100, 37, 200, 48))
PATTERN_SKEW = ("skew", (1, 511, 3, 5))
PATTERNS = (PATTERN_UNIFORM, PATTERN_ALIGNED_RAGGED, PATTERN_TILE_CROSSING, PATTERN_SKEW)

# Both exactly zero: a pack/unpack moves rows and computes nothing, so the
# expected difference is none rather than small.
VALUE_RTOL = 0.0
VALUE_ATOL = 0.0


def _closed_form_packed_length(lengths: tuple[int, ...]) -> int:
    """The packed length, derived independently of the module under test."""
    total = 0
    for length in lengths:
        total += length
    return total


def _fixture(label: str, lengths: tuple[int, ...]) -> dict:
    """Build one row-length pattern and the packed reference for it."""
    batch = len(lengths)
    max_len = max(lengths)
    packed_len = _closed_form_packed_length(lengths)

    gen = torch.Generator().manual_seed(45_000 + packed_len + max_len)
    padded = torch.randn(batch, max_len, WIDTH, generator=gen).to(torch.bfloat16)
    # Zero every padding position: the round trip can only reproduce padding it
    # can predict, and zero is the value this module's unpack writes back.
    for b, length in enumerate(lengths):
        padded[b, length:, :] = 0
    padded = padded.contiguous()

    expected_packed = torch.cat(
        [padded[b, : lengths[b], :] for b in range(batch)], dim=0
    ).contiguous()

    return dict(
        label=label,
        lengths=lengths,
        batch=batch,
        max_len=max_len,
        packed_len=packed_len,
        padded=padded,
        expected_packed=expected_packed,
    )


def _differing_elements(a: torch.Tensor, b: torch.Tensor) -> int:
    """Count elements differing in their int16 view, not their numeric value.

    Bitwise, because ``-0.0`` compares equal to ``+0.0`` while differing in the sign bit.
    """
    assert a.shape == b.shape, f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
    ai = a.contiguous().view(torch.int16)
    bi = b.contiguous().view(torch.int16)
    return int((ai != bi).sum().item())


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max absolute difference, computed in float32 so it cannot round."""
    return float((a.float() - b.float()).abs().max().item())


def _padding_nonzero_elements(unpacked: torch.Tensor, lengths: tuple[int, ...]) -> int:
    """Elements at padding positions whose int16 view is not exactly ``0x0000``.

    Negative zero counts as nonzero here, since it is not the value the unpack claims to write.
    """
    bad = 0
    for b, length in enumerate(lengths):
        tail = unpacked[b, length:, :]
        if tail.numel() == 0:
            continue
        bad += int((tail.contiguous().view(torch.int16) != 0).sum().item())
    return bad


def _round_trip(fx: dict) -> None:
    """Pack then unpack one pattern and check both against a torch reference."""
    label = fx["label"]
    padded, lengths = fx["padded"], fx["lengths"]

    packed = dsa_ragged_pack(padded, lengths)

    assert int(packed.shape[0]) == fx["packed_len"], (
        f"{label}: packed length {int(packed.shape[0])} is not the closed-form "
        f"{fx['packed_len']}"
    )
    assert int(packed.shape[1]) == WIDTH

    packed_diff = _differing_elements(packed.to(torch.bfloat16), fx["expected_packed"])
    assert packed_diff == 0, f"{label}: the packed buffer differs from the sliced reference"

    unpacked = dsa_ragged_unpack(packed, lengths, fx["max_len"])
    assert tuple(unpacked.shape) == tuple(padded.shape)

    max_abs = _max_abs_diff(unpacked.to(torch.bfloat16), padded)
    differing = _differing_elements(unpacked.to(torch.bfloat16), padded)
    padding_bad = _padding_nonzero_elements(unpacked.to(torch.bfloat16), lengths)
    assert max_abs == 0.0, f"{label}: max abs diff {max_abs} is not exactly zero"
    assert differing == 0, f"{label}: {differing} elements differ bitwise after the round trip"
    assert padding_bad == 0, f"{label}: {padding_bad} padding elements are not positive zero"


def test_round_trip_is_bit_identical_on_uniform() -> None:
    """Four equal-length sequences, packed length 512, a tile multiple."""
    _round_trip(_fixture(*PATTERN_UNIFORM))


def test_round_trip_is_bit_identical_on_aligned_ragged() -> None:
    """Unequal sequences whose packed length is still 512."""
    _round_trip(_fixture(*PATTERN_ALIGNED_RAGGED))


def test_round_trip_is_bit_identical_on_tile_crossing() -> None:
    """Packed length 385 (3 * 128 + 1), not a tile multiple."""
    fx = _fixture(*PATTERN_TILE_CROSSING)
    assert fx["packed_len"] % PARTITION_TILE != 0, "this pattern must not be tile aligned"
    _round_trip(fx)


def test_round_trip_is_bit_identical_on_skew() -> None:
    """A length-1 sequence beside a 511-row sequence, packed length 520."""
    fx = _fixture(*PATTERN_SKEW)
    assert fx["packed_len"] % PARTITION_TILE != 0, "this pattern must not be tile aligned"
    _round_trip(fx)


def test_every_pattern_takes_the_kernel_and_not_the_torch_path() -> None:
    """Each pattern dispatches the pack and unpack kernels once each, never the torch path."""
    for label, lengths in PATTERNS:
        fx = _fixture(label, lengths)
        reset_ragged_pack_dispatch_counters()
        gate = bool(can_run_kernel(fx["padded"]))
        envelope = bool(can_run_dsa_ragged_pack(fx["padded"], lengths))
        packed = dsa_ragged_pack(fx["padded"], lengths)
        dsa_ragged_unpack(packed, lengths, fx["max_len"])
        nki_dispatch, torch_fallback = ragged_pack_dispatch_counters()
        assert gate is True, f"{label}: the runtime NKI gate is not open"
        assert envelope is True, f"{label}: this module does not admit this pattern"
        assert nki_dispatch == 2, f"{label}: expected 2 NKI dispatches, read {nki_dispatch}"
        assert torch_fallback == 0, f"{label}: expected 0 torch fallbacks, read {torch_fallback}"


def test_unadmitted_dtype_is_refused_by_the_gate_and_served_by_torch() -> None:
    """float32 storage takes the torch path, which must still pack correctly."""
    label, lengths = PATTERN_TILE_CROSSING
    fx = _fixture(label, lengths)
    padded_f32 = fx["padded"].to(torch.float32).contiguous()
    reset_ragged_pack_dispatch_counters()
    gate = bool(can_run_dsa_ragged_pack(padded_f32, lengths))
    packed = dsa_ragged_pack(padded_f32, lengths)
    nki_dispatch, torch_fallback = ragged_pack_dispatch_counters()
    expected = torch.cat(
        [padded_f32[b, : lengths[b], :] for b in range(len(lengths))], dim=0
    ).contiguous()
    differing = int((packed != expected).sum().item())

    assert gate is False, "float32 must not be admitted to the kernel"
    assert nki_dispatch == 0, f"expected 0 NKI dispatches on the torch route, read {nki_dispatch}"
    assert torch_fallback == 1, f"expected exactly 1 torch fallback, read {torch_fallback}"
    assert differing == 0, "the torch route returned the wrong rows"


def test_kernel_identity_reports_the_dispatched_kernel() -> None:
    """The identity is None before any dispatch, then names this module's own kernel per direction."""
    label, lengths = PATTERN_UNIFORM
    fx = _fixture(label, lengths)
    reset_ragged_pack_dispatch_counters()
    assert ragged_pack_kernel_identity() is None

    packed = dsa_ragged_pack(fx["padded"], lengths)
    after_pack = ragged_pack_kernel_identity()
    assert after_pack == (
        "vllm_neuron.functional.dsa.ragged_pack",
        "_ragged_pack_nki",
    ), f"the pack seam dispatched {after_pack}"

    dsa_ragged_unpack(packed, lengths, fx["max_len"])
    after_unpack = ragged_pack_kernel_identity()
    assert after_unpack == (
        "vllm_neuron.functional.dsa.ragged_pack",
        "_ragged_unpack_nki",
    ), f"the unpack seam dispatched {after_unpack}"
