# SPDX-License-Identifier: Apache-2.0
"""Tests for the paged gather kernel against ``torch.index_select``.

Three page layouts are driven: one aligned length (512 = 4 full pages) and two
raggednesses at the extremes (520 leaves 8 valid slots in the final page, 385
leaves 1 and also makes the final tile short by 127 rows). Raggedness is where a
tiled gather goes wrong, so each layout is its own test.

The page table is a roll by one, which has no fixed points, so a gather that
ignored the page table would read a different physical page for every token. The
positions are ``(i * 37) % seq_len``, a permutation for all three lengths because
37 is coprime to each. A gather moves bytes and computes nothing, so the
comparison is exact: raw bit patterns as well as values, with zero tolerance.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional.dsa.paged_gather import (
    can_run_dsa_paged_gather,
    dsa_paged_gather,
    paged_gather_dispatch_counters,
    paged_gather_kernel_identity,
    reset_paged_gather_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

PAGE_SIZE = 128
WIDTH = 512

# (label, seq_len), where seq_len is the number of valid slots the sequence
# occupies. The final page is ragged when seq_len is not a multiple of PAGE_SIZE.
LAYOUT_ALIGNED = ("aligned", 512)
LAYOUT_RAGGED_8 = ("ragged_8", 520)
LAYOUT_RAGGED_1 = ("ragged_1", 385)
LAYOUTS = (LAYOUT_ALIGNED, LAYOUT_RAGGED_8, LAYOUT_RAGGED_1)

# Both exactly zero: a gather is a permutation of existing rows, so the expected
# difference is none rather than small.
VALUE_RTOL = 0.0
VALUE_ATOL = 0.0


def _page_table(num_pages: int) -> torch.Tensor:
    """Logical page ``i`` -> physical page ``(i + 1) % n``, which has no fixed points."""
    return torch.roll(torch.arange(num_pages, dtype=torch.int32), 1)


def _fixture(seq_len: int) -> dict:
    """Build one page layout and the ``index_select`` reference for it."""
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = _page_table(num_pages)
    positions = (torch.arange(seq_len, dtype=torch.int64) * 37) % seq_len

    gen = torch.Generator().manual_seed(44_000 + seq_len)
    pages = torch.randn(num_pages, PAGE_SIZE, WIDTH, generator=gen).to(torch.bfloat16)
    pages_flat = pages.reshape(num_pages * PAGE_SIZE, WIDTH)

    logical = positions // PAGE_SIZE
    slot = positions % PAGE_SIZE
    page_idx = page_table[logical].to(torch.int32)
    slot_idx = slot.to(torch.int32)

    flat_idx = page_idx.to(torch.int64) * PAGE_SIZE + slot_idx.to(torch.int64)
    expected = torch.index_select(pages_flat, 0, flat_idx)

    return dict(
        seq_len=seq_len,
        num_pages=num_pages,
        page_table=page_table,
        pages_flat=pages_flat,
        page_idx=page_idx,
        slot_idx=slot_idx,
        positions=positions,
        expected=expected,
    )


def _bitwise_differing(a: torch.Tensor, b: torch.Tensor) -> int:
    """Count elements differing in raw bit pattern.

    bfloat16 is viewed as int16, a reinterpretation rather than a conversion, so
    two encodings that read as the same float still count as different.
    """
    view = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    return int((a.contiguous().view(view) != b.contiguous().view(view)).sum())


def _assert_bit_identical(label: str, seq_len: int) -> None:
    """Gather one layout and require it to match ``index_select`` bit for bit."""
    f = _fixture(seq_len)
    got = dsa_paged_gather(f["pages_flat"], f["page_idx"], f["slot_idx"], PAGE_SIZE)

    max_abs_diff = (
        (got.to(torch.float32) - f["expected"].to(torch.float32)).abs().max().item()
    )
    differing = _bitwise_differing(got, f["expected"])

    assert got.shape == (seq_len, WIDTH), (
        f"[{label}] expected one gathered row per token; got shape {tuple(got.shape)}"
    )
    assert got.dtype == torch.bfloat16
    assert max_abs_diff == 0.0, (
        f"[{label}] a gather is a permutation, so the difference against index_select must "
        f"be exactly 0.0; got {max_abs_diff:.3e}"
    )
    assert differing == 0, (
        f"[{label}] {differing} of {f['expected'].numel()} elements differ in raw bit pattern"
    )
    torch.testing.assert_close(
        got.to(torch.float32),
        f["expected"].to(torch.float32),
        rtol=VALUE_RTOL,
        atol=VALUE_ATOL,
    )


def test_gather_is_bit_identical_on_the_aligned_layout() -> None:
    """512 tokens, 4 full pages, no ragged final page."""
    _assert_bit_identical(*LAYOUT_ALIGNED)


def test_gather_is_bit_identical_on_the_ragged_8_layout() -> None:
    """520 tokens: the final page is ragged while the token count still fills whole tiles."""
    _assert_bit_identical(*LAYOUT_RAGGED_8)


def test_gather_is_bit_identical_on_the_ragged_1_layout() -> None:
    """385 tokens: the final page holds one valid slot and the final tile is short by 127 rows."""
    _assert_bit_identical(*LAYOUT_RAGGED_1)


def test_every_layout_takes_the_kernel_and_not_the_torch_path() -> None:
    """Each layout dispatches the kernel exactly once and never enters the fallback."""
    for label, seq_len in LAYOUTS:
        f = _fixture(seq_len)
        reset_paged_gather_dispatch_counters()
        gate = can_run_dsa_paged_gather(
            f["pages_flat"], f["page_idx"], f["slot_idx"], PAGE_SIZE
        )
        dsa_paged_gather(f["pages_flat"], f["page_idx"], f["slot_idx"], PAGE_SIZE)
        nki_dispatch, torch_fallback = paged_gather_dispatch_counters()
        assert nki_dispatch == 1, (
            f"[{label}] expected exactly 1 NKI dispatch; got {nki_dispatch}"
        )
        assert torch_fallback == 0, (
            f"[{label}] the torch fallback must not be entered; it ran {torch_fallback} time(s)"
        )
        assert can_run_kernel(f["pages_flat"]) is True
        assert gate is True


def test_unadmitted_dtype_is_refused_by_the_gate_and_served_by_torch() -> None:
    """float32 storage takes the torch path, which must still gather correctly."""
    f = _fixture(LAYOUT_ALIGNED[1])
    violating = f["pages_flat"].to(torch.float32)
    reset_paged_gather_dispatch_counters()
    gate = can_run_dsa_paged_gather(violating, f["page_idx"], f["slot_idx"], PAGE_SIZE)
    got = dsa_paged_gather(violating, f["page_idx"], f["slot_idx"], PAGE_SIZE)
    nki_dispatch, torch_fallback = paged_gather_dispatch_counters()

    assert gate is False, "an unadmitted dtype must be refused by the gate"
    assert torch_fallback == 1, (
        f"the torch path must run exactly once; got {torch_fallback}"
    )
    assert nki_dispatch == 0, (
        f"the kernel must not be reached; got {nki_dispatch} dispatch(es)"
    )
    expected = f["expected"].to(torch.float32)
    assert _bitwise_differing(got, expected) == 0, (
        "the torch path must gather correctly, not merely count itself"
    )


def test_kernel_identity_reports_the_dispatched_kernel() -> None:
    """The identity is ``None`` before any dispatch and names this module's kernel after one."""
    reset_paged_gather_dispatch_counters()
    assert paged_gather_kernel_identity() is None

    f = _fixture(LAYOUT_ALIGNED[1])
    dsa_paged_gather(f["pages_flat"], f["page_idx"], f["slot_idx"], PAGE_SIZE)
    after = paged_gather_kernel_identity()
    assert after is not None
    module, qualname = after
    assert module == "vllm_neuron.functional.dsa.paged_gather", (
        f"the seam must dispatch this module's own kernel; got module {module}"
    )
    assert qualname == "_paged_gather_nki", (
        f"the seam must dispatch _paged_gather_nki; got qualname {qualname}"
    )
