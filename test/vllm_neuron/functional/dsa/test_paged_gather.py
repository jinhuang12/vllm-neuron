# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for `inc-glm53f-044` -- the DSA paged gather, an ADAPTED NKI kernel.

Declared command (D1 Tier N, the block's shorthand expanded to the tier's byte form)::

    PYTHONDONTWRITEBYTECODE=1 VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    python -m pytest test/vllm_neuron/functional/dsa/test_paged_gather.py \\
        --timeout 60 -v -s -p no:cacheprovider

ONE ITEM PER COUNTED CONJUNCT AND NO ``parametrize`` (§6 rule 4b / rule 6), so the item count
is derivable before the run: SEVEN. Three bit-identity items, one per declared page layout;
the route-predicate item; the two D1.5 controls that make this file's two counted zeros
measurements rather than decoration; and the kernel-identity item. NO FIXTURE SETS
``NEURON_PLATFORM_TARGET_OVERRIDE`` OR ``NKI_SIMULATOR`` (§6 rule 3, D2): both resolve at
import time and belong in the process invocation, so a fixture that set either would have
measured the wrong thing.

THE GEOMETRY, AND WHY THESE THREE LAYOUTS. Pages hold 128 rows of width 512. A sequence's
final page is RAGGED when its length is not a multiple of the page size, and raggedness is
where a tiled gather goes wrong, so the three declared layouts are one aligned length and two
raggednesses chosen at the extremes rather than at random: 512 (4 full pages), 520 (8 valid
slots in the final page) and 385 (1 valid slot in the final page). 385 also crosses the
kernel's own tiling boundary of ``nl.tile_size.pmax == 128`` unevenly, so the short final
TILE and the short final PAGE are exercised together.

WHY THE PAGE TABLE IS A ROLL BY ONE AND NOT A SHUFFLE. A gather that ignored the page table
entirely would still pass a bit-identity check whenever the page table happened to be the
identity. At 3 to 5 pages a random shuffle returns the identity often enough for that to be a
real risk, so the fixture uses a DERANGEMENT -- logical page ``i`` maps to physical page
``(i + 1) % num_pages``, which has zero fixed points by construction. Every gathered row then
lands on a different physical page than a page-table-blind kernel would read, and
``test_bit_identity_discriminates_a_page_table_blind_gather`` MEASURES that rather than
asserting it.

WHY THE POSITIONS ARE A PERMUTATION. ``(i * 37) % seq_len`` visits every valid slot exactly
once for each of the three lengths -- 37 is coprime to all of them -- so the gather is a
permutation of the sequence's rows. That makes the comparison exact: a gather moves bytes and
computes nothing, so any nonzero difference is an indexing bug and never a rounding one, which
is why the acceptance is bit-identity and carries no tolerance at all.

WHY ``bfloat16``. It is the dtype the increment's substrate declaration names and the only one
the campaign has measured on this pin; the module admits it and serves every other dtype
through the torch path, which is what the second control below drives.

COMPARISON IS ON RAW BIT PATTERNS, NOT ON VALUES. ``max abs diff == 0.0`` is the criterion the
increment plan declares, and it is asserted; but a value comparison cannot distinguish two
different bit patterns that both read as the same float, so the bit patterns are compared as
integers as well. Both readings are printed for every layout.
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

# The three declared page layouts, as ``(label, seq_len)``. ``seq_len`` is the number of valid
# slots the sequence occupies; the final page is ragged when it is not a multiple of PAGE_SIZE.
LAYOUT_ALIGNED = ("aligned", 512)
LAYOUT_RAGGED_8 = ("ragged_8", 520)
LAYOUT_RAGGED_1 = ("ragged_1", 385)
LAYOUTS = (LAYOUT_ALIGNED, LAYOUT_RAGGED_8, LAYOUT_RAGGED_1)

# Value tolerance, ORDER NAMED INLINE (D3): (rtol, atol) == (0.0, 0.0). BOTH are exactly zero
# because a gather is a permutation of existing rows: the expected difference is not "small",
# it is none. A nonzero tolerance here would hide the only failure mode this item has.
VALUE_RTOL = 0.0
VALUE_ATOL = 0.0


def _page_table(num_pages: int) -> torch.Tensor:
    """A deterministic DERANGEMENT: logical page ``i`` -> physical page ``(i + 1) % n``.

    Zero fixed points by construction. The module docstring gives the reason a shuffle is
    not used.
    """
    return torch.roll(torch.arange(num_pages, dtype=torch.int32), 1)


def _fixture(seq_len: int) -> dict:
    """Build one declared layout, plus the two references the items compare against."""
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
    want = torch.index_select(pages_flat, 0, flat_idx)
    # The page-table-BLIND reference: the same read with the page table not applied. Used only
    # by the discrimination control, which asserts it DIFFERS.
    blind_idx = logical * PAGE_SIZE + slot
    blind = torch.index_select(pages_flat, 0, blind_idx)

    return dict(
        seq_len=seq_len,
        num_pages=num_pages,
        page_table=page_table,
        pages_flat=pages_flat,
        page_idx=page_idx,
        slot_idx=slot_idx,
        positions=positions,
        want=want,
        blind=blind,
        ragged=(seq_len % PAGE_SIZE) != 0,
        valid_in_last_page=(seq_len % PAGE_SIZE) or PAGE_SIZE,
        fixed_points=int((page_table == torch.arange(num_pages, dtype=torch.int32)).sum()),
    )


def _bitwise_differing(a: torch.Tensor, b: torch.Tensor) -> int:
    """Count of elements differing in RAW BIT PATTERN, not in value.

    bfloat16 is viewed as int16, which is a reinterpretation and not a conversion, so two
    encodings that read as the same float still count as different here.
    """
    view = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    return int((a.contiguous().view(view) != b.contiguous().view(view)).sum())


def _assert_fixture_preconditions(label: str, f: dict) -> None:
    """The fixture's own claims, asserted rather than trusted.

    A fixture that quietly lost its derangement or its permutation would weaken every item
    that reads it, and would do so without failing, so each claim is checked where it is used.
    """
    assert sorted(f["positions"].tolist()) == list(range(f["seq_len"])), (
        f"[{label}] positions must be a permutation of the valid slots"
    )
    assert f["fixed_points"] == 0, (
        f"[{label}] the page table must be a derangement so a page-table-blind gather is "
        f"detectable; it has {f['fixed_points']} fixed point(s)"
    )
    assert f["ragged"] == (f["seq_len"] % PAGE_SIZE != 0)


def _bit_identity_case(label: str, seq_len: int) -> None:
    """Drive one declared layout and assert bit-identity, printing every reading."""
    f = _fixture(seq_len)
    _assert_fixture_preconditions(label, f)
    print(
        f"[fixture] layout={label} seq_len={seq_len} pages={f['num_pages']} "
        f"page_size={PAGE_SIZE} width={WIDTH} ragged={f['ragged']} "
        f"valid_in_last_page={f['valid_in_last_page']} fixed_points={f['fixed_points']} "
        f"dtype={f['pages_flat'].dtype}"
    )

    reset_paged_gather_dispatch_counters()
    got = dsa_paged_gather(f["pages_flat"], f["page_idx"], f["slot_idx"], PAGE_SIZE)
    nki_dispatch, torch_fallback = paged_gather_dispatch_counters()

    max_abs_diff = (
        (got.to(torch.float32) - f["want"].to(torch.float32)).abs().max().item()
    )
    differing = _bitwise_differing(got, f["want"])
    print(
        f"[acceptance] layout={label} seq_len={seq_len} max_abs_diff={max_abs_diff:.3e} "
        f"bitwise_differing={differing}/{f['want'].numel()} out_shape={tuple(got.shape)} "
        f"out_dtype={got.dtype}"
    )
    print(
        f"[route-predicate] layout={label} nki_dispatch={nki_dispatch} "
        f"torch_fallback={torch_fallback} can_run_kernel={can_run_kernel(f['pages_flat'])} "
        f"certifies=vllm_neuron.functional.dsa.paged_gather.dsa_paged_gather"
    )

    assert got.shape == (seq_len, WIDTH), (
        f"[{label}] expected one gathered row per token; got shape {tuple(got.shape)}"
    )
    assert got.dtype == torch.bfloat16
    assert max_abs_diff == 0.0, (
        f"[{label}] a gather is a permutation, so the difference against index_select must "
        f"be exactly 0.0; got {max_abs_diff:.3e}"
    )
    assert differing == 0, (
        f"[{label}] {differing} of {f['want'].numel()} elements differ in raw bit pattern"
    )
    torch.testing.assert_close(
        got.to(torch.float32),
        f["want"].to(torch.float32),
        rtol=VALUE_RTOL,
        atol=VALUE_ATOL,
    )


def test_gather_is_bit_identical_on_the_aligned_layout() -> None:
    """Conjunct 1 -- bit-identity at 512 tokens, 4 full pages, no ragged final page.

    Certifies: ``vllm_neuron/functional/dsa/paged_gather.py::dsa_paged_gather`` -- the seam
    this increment authors, at the layout where every page and every tile is full.
    """
    _bit_identity_case(*LAYOUT_ALIGNED)


def test_gather_is_bit_identical_on_the_ragged_8_layout() -> None:
    """Conjunct 2 -- bit-identity at 520 tokens, 8 valid slots in the final page.

    Certifies: the same seam where the final PAGE is ragged while the token count still
    exceeds a whole number of tiles, so the raggedness is in the paging and not in the
    tiling.
    """
    _bit_identity_case(*LAYOUT_RAGGED_8)


def test_gather_is_bit_identical_on_the_ragged_1_layout() -> None:
    """Conjunct 3 -- bit-identity at 385 tokens, 1 valid slot in the final page.

    Certifies: the same seam at the hardest declared layout, where the final page holds a
    single valid slot AND the final tile is short by 127 rows. Passing the aligned layout
    establishes nothing here, which is why each layout is its own item.
    """
    _bit_identity_case(*LAYOUT_RAGGED_1)


def test_route_predicate_one_dispatch_per_layout_and_no_torch_fallback() -> None:
    """Route predicate (D13 form R-1) -- 1 NKI dispatch per declared layout, fallback 0.

    Certifies: the ``wrap_nki`` dispatch site inside
    ``vllm_neuron/functional/dsa/paged_gather.py::dsa_paged_gather``. A pure-torch
    implementation of this module reads ``nki_dispatch == 0`` on every layout and therefore
    cannot pass this item, which is the whole point of the form.

    THE COUNTED ZERO NAMES ITS POPULATION: ``calls_made`` is printed beside
    ``torch_fallback``, so the zero is read over a stated number of opportunities to be
    non-zero rather than over an unstated one. Its non-vacuity control is the separate item
    ``test_torch_fallback_counter_is_not_vacuous``, which drives the SAME counter non-zero.
    """
    gate_readings: list[bool] = []
    for label, seq_len in LAYOUTS:
        f = _fixture(seq_len)
        # Per-case reset, per §4b's convention: the counter is read at the end of the case it
        # was zeroed at the start of.
        reset_paged_gather_dispatch_counters()
        gate = can_run_dsa_paged_gather(
            f["pages_flat"], f["page_idx"], f["slot_idx"], PAGE_SIZE
        )
        gate_readings.append(gate)
        dsa_paged_gather(f["pages_flat"], f["page_idx"], f["slot_idx"], PAGE_SIZE)
        calls_made = 1
        nki_dispatch, torch_fallback = paged_gather_dispatch_counters()
        print(
            f"[route-predicate] layout={label} seq_len={seq_len} calls_made={calls_made} "
            f"nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
            f"can_run_kernel={can_run_kernel(f['pages_flat'])} "
            f"can_run_dsa_paged_gather={gate} "
            f"certifies=vllm_neuron.functional.dsa.paged_gather.dsa_paged_gather"
        )
        assert nki_dispatch == calls_made, (
            f"[{label}] expected exactly {calls_made} NKI dispatch; got {nki_dispatch}"
        )
        assert torch_fallback == 0, (
            f"[{label}] the torch fallback must not be entered; it ran {torch_fallback} "
            f"time(s) out of {calls_made} call(s)"
        )
        assert can_run_kernel(f["pages_flat"]) is True
        assert gate is True

    print(
        f"[route-predicate] gate_true_on_all_declared_layouts="
        f"{sum(gate_readings)}/{len(gate_readings)}"
    )
    assert all(gate_readings)


def test_bit_identity_discriminates_a_page_table_blind_gather() -> None:
    """D1.5 control -- the three zeros above are zeros the instrument CAN move.

    Certifies: that this file's fixture can tell a correct gather from a gather that ignores
    the page table. ``max abs diff == 0.0`` is a COUNTED ZERO, so on its own it is satisfied
    by any instrument that cannot see a difference at all. This item builds the page-table-
    BLIND reference -- the same read with the table not applied -- and asserts it differs in a
    NON-zero number of raw bit patterns on every declared layout.

    The count is printed as a fraction of the population, so the non-zero reading is read over
    a stated number of rows rather than an unstated one. The derangement in the fixture is what
    makes the expected count the FULL population rather than merely positive.
    """
    for label, seq_len in LAYOUTS:
        f = _fixture(seq_len)
        _assert_fixture_preconditions(label, f)
        differing = _bitwise_differing(f["blind"], f["want"])
        rows_differing = int(
            (
                f["blind"].to(torch.float32) != f["want"].to(torch.float32)
            ).any(dim=1).sum()
        )
        print(
            f"[control] layout={label} seq_len={seq_len} "
            f"page_table_blind_bitwise_differing={differing}/{f['want'].numel()} "
            f"rows_differing={rows_differing}/{seq_len} "
            f"certifies=test_paged_gather._fixture"
        )
        assert differing > 0, (
            f"[{label}] a page-table-blind gather must be DETECTABLE, or the bit-identity "
            f"items above are vacuous; it differed in 0 bit patterns"
        )
        assert rows_differing == seq_len, (
            f"[{label}] the derangement should make every row differ; "
            f"{rows_differing} of {seq_len} differ"
        )


def test_torch_fallback_counter_is_not_vacuous() -> None:
    """D1.5 control -- the route predicate's ``torch_fallback == 0`` is a movable zero.

    Certifies: the fallback limb of
    ``vllm_neuron/functional/dsa/paged_gather.py::dsa_paged_gather``, reached through
    ``can_run_dsa_paged_gather`` returning False. The violating input is ``float32`` paged
    storage, a dtype the module does not admit because it is not measured on this pin; no
    environment variable is touched to produce it, so this control cannot itself perturb the
    tier's pinned invocation (D2).

    A zero that reads zero either way is decoration. This item is what makes the previous
    item's ``torch_fallback == 0`` a measurement.
    """
    f = _fixture(LAYOUT_ALIGNED[1])
    violating = f["pages_flat"].to(torch.float32)
    reset_paged_gather_dispatch_counters()
    gate = can_run_dsa_paged_gather(violating, f["page_idx"], f["slot_idx"], PAGE_SIZE)
    got = dsa_paged_gather(violating, f["page_idx"], f["slot_idx"], PAGE_SIZE)
    nki_dispatch, torch_fallback = paged_gather_dispatch_counters()
    print(
        f"[control] case=unadmitted_dtype dtype={violating.dtype} calls_made=1 "
        f"nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
        f"can_run_dsa_paged_gather={gate} "
        f"certifies=vllm_neuron.functional.dsa.paged_gather._dsa_paged_gather_torch"
    )
    assert gate is False, "an unadmitted dtype must be refused by the gate"
    assert torch_fallback == 1, (
        f"the control must drive the torch fallback exactly once; got {torch_fallback}"
    )
    assert nki_dispatch == 0, (
        f"the control must not reach the NKI seam; got {nki_dispatch} dispatch(es)"
    )
    # The fallback is still the oracle, so it must return the right answer, not merely run.
    want = f["want"].to(torch.float32)
    assert _bitwise_differing(got, want) == 0, (
        "the fallback is the oracle and must gather correctly, not merely count itself"
    )


def test_kernel_identity_is_derived_through_the_seam() -> None:
    """The kernel the seam DISPATCHED is this increment's OWN NKI kernel, read through the seam.

    Certifies: the identity recorded at the dispatch site in
    ``vllm_neuron/functional/dsa/paged_gather.py::dsa_paged_gather``. D13.1 admits a
    ``kernel_identity`` reading as route evidence only when it is derived through the seam the
    test actually drove; a reading taken from a module-level import certifies what was
    imported and nothing about what ran. So this item reads ``None`` BEFORE any dispatch --
    which is what distinguishes "no kernel ran" from "some kernel ran" -- and the authored
    kernel after one.

    This item is also the increment's SUBSTRATE evidence, and it is where ADAPT differs from
    the WRAP of ``inc-glm53f-043``: the kernel the seam dispatches is defined in this
    increment's own module, not in a vendored one, which is what the ADAPT declaration claims.
    """
    reset_paged_gather_dispatch_counters()
    before = paged_gather_kernel_identity()
    print(f"[identity] before_any_dispatch={before}")
    assert before is None

    f = _fixture(LAYOUT_ALIGNED[1])
    dsa_paged_gather(f["pages_flat"], f["page_idx"], f["slot_idx"], PAGE_SIZE)
    after = paged_gather_kernel_identity()
    print(
        f"[identity] after_one_dispatch={after} "
        f"certifies=vllm_neuron.functional.dsa.paged_gather.dsa_paged_gather"
    )
    assert after is not None
    module, qualname = after
    assert module == "vllm_neuron.functional.dsa.paged_gather", (
        f"the seam must dispatch the kernel this increment AUTHORS; got module {module}"
    )
    assert qualname == "_paged_gather_nki", (
        f"the seam must dispatch _paged_gather_nki; got qualname {qualname}"
    )
