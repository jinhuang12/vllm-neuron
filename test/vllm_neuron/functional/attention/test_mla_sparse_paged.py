# SPDX-License-Identifier: Apache-2.0
"""Acceptance for the paged latent window: the sparse attention kernel assembles its window from a
block table instead of slicing one ascending run of blocks out of the latent bank.

EIGHT tests and NO `parametrize` decorator. Four carry `table` in their name and read the four block
layouts the design declares -- two scattered rows in either order, a row with a padding tail, and a
full-length row. One carries `identical` and is the only exact claim in this file. One carries
`sentinel`. Two carry `overlay` and read this step's own rows, at one row and at a whole prefill
chunk.

WHY THE ORACLE CLAIM IS A TOLERANCE AND NOT BIT-IDENTITY. The kernel contracts on the tensor engine
in tiles and normalises with a running maximum; the oracle contracts in torch in one shot. The landed
acceptance of this module reads the oracle at the section 3 module-comparison tolerance for exactly
that reason, and no paged window changes it. What IS exact here is the `identical` item: a table whose
pages are consecutive must produce the SAME BYTES as the unpaged call on the window those pages are,
because both feed one identical staged window into one identical arithmetic. That item is what makes
the four tolerance items mean something -- a page assembled at the wrong offset cannot pass it.

THE WINDOW IS WHAT THE SELECTED ROWS INDEX. A table entry of -1 is padding: the load clamps it to
page 0 so the DMA stays inside the bank, and those window columns carry no token because the producer
never selects them. The `table_pad` item reads both halves of that sentence.

The declared command, and the only one whose result the plan block quotes::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/attention/test_mla_sparse_paged.py -v -s \
        -p no:cacheprovider

Nothing here reads or sets an environment variable: both switches above resolve at import.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional.attention import mla_sparse as MS

#: The served KV block, and the width every table entry addresses. A page is a whole number of the
#: kernel's 16-row DMA transposes, which is why a paged load issues no more DMAs than an unpaged one.
PAGE = 128

#: This checkpoint's own latent rank, and the head count the small items run at. The oracle's cost
#: grows with both, so only the full-table and chunk-overlay items need more than two heads' worth.
LATENT = 512
HEADS = 2

#: The smallest admissible selected-row count, and the scale the landed acceptance uses.
TOPK = 128
SCALE = 0.1

#: Section 3's bf16 module-comparison thresholds. READ FROM THE PLAN, NOT AUTHORED HERE.
RTOL = 1e-2
ATOL = 1e-5

#: The bank holds thirty-two pages: the served model length over the served block, so a full-length
#: table is the widest window this configuration reaches, and page 10 addresses a real page.
BANK_PAGES = 32

#: The prefill chunk this campaign serves, and the row the chunk overlay starts at -- not a multiple
#: of the page, so the overlay is read at an offset no page boundary hides.
CHUNK = 1024
CHUNK_AT = 3000


def _bank(pages: int = BANK_PAGES, latent: int = LATENT) -> torch.Tensor:
    """The latent bank, every row distinguishable: row r column c holds r * latent + c."""
    rows = pages * PAGE
    down = torch.arange(rows, dtype=torch.float32).reshape(rows, 1) * latent
    return down + torch.arange(latent, dtype=torch.float32).reshape(1, latent)


def _window(bank: torch.Tensor, table: list[int]) -> torch.Tensor:
    """The window the table names, a -1 entry clamped to page 0 exactly as the load clamps it."""
    rows = [max(page, 0) * PAGE + inside for page in table for inside in range(PAGE)]
    return bank[rows]


def _queries(seq: int, heads: int, latent: int) -> torch.Tensor:
    """Query latents that depend on every axis, so a transposed or reused tile shows."""
    torch.manual_seed(seq * 1000 + heads * 10 + latent // 128)
    return torch.randn(seq, heads, latent, dtype=torch.float32) * 0.05


def _selected(seq: int, window_rows: int, topk: int = TOPK) -> torch.Tensor:
    """One row of selected window positions per query, spread across the whole window."""
    step = max(window_rows // topk, 1)
    columns = [(index * step) % window_rows for index in range(topk)]
    return torch.tensor([columns for _ in range(seq)], dtype=torch.int32)


def _paged(bank, table, selected, queries, written=None, write_offset=None):
    """The seam under test, called the way the runner will call it.

    The table travels as a COLUMN, `[pages, 1]` int32: the shape the page number is read out of on
    device, and the shape every other index operand in this tree takes.
    """
    offset = torch.zeros(1, 1, dtype=torch.int32) if write_offset is None else write_offset
    rows = torch.zeros(0, bank.shape[1], dtype=bank.dtype) if written is None else written
    return MS.mla_sparse_attention(
        queries, bank, selected, SCALE,
        block_table_row=torch.tensor([[page] for page in table], dtype=torch.int32),
        written=rows, write_offset=offset, page_size=PAGE,
    )


def _oracle(window, selected, queries) -> torch.Tensor:
    """The same attention over the window the table names, in torch."""
    return MS.mla_sparse_attention_torch_oracle(queries, window, selected, SCALE)


def _say(label: str, value: object) -> None:
    """Print one counted value, so the transcript carries the reading and not just a pass."""
    print(f"PAGED_{label}={value}", flush=True)


def _read_one_table(table: list[int], heads: int = HEADS) -> None:
    """Run one table layout and read the kernel against the oracle on the window it names."""
    bank = _bank()
    window = _window(bank, table)
    queries = _queries(1, heads, LATENT)
    selected = _selected(1, window.shape[0])
    got = _paged(bank, table, selected, queries)
    want = _oracle(window, selected, queries)
    worst = float((got - want).abs().max())
    _say("TABLE", ".".join(str(page) for page in table))
    _say("WORST_ABS", f"{worst:.3e}")
    assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
        f"the paged window disagrees with the oracle on table {table}: worst absolute difference "
        f"{worst:.3e} against rtol={RTOL} atol={ATOL}. A window assembled at the wrong page offset "
        f"reads another sequence's latents, which is the defect this increment exists to remove"
    )


def test_table_five_then_ten_agrees_with_the_oracle() -> None:
    """TABLE 1 of 4 -- the layout the serving allocator handed the first multi-block request.

    CERTIFYING COMPONENT: the page loop's source offset, read off the block table.
    """
    _read_one_table([5, 10])


def test_table_ten_then_five_agrees_with_the_oracle() -> None:
    """TABLE 2 of 4 -- the same pages in the other order, which no ascending run can express.

    CERTIFYING COMPONENT: that the window follows the table's ORDER and not the page numbers'.
    """
    _read_one_table([10, 5])


def test_table_with_a_padding_tail_agrees_with_the_oracle() -> None:
    """TABLE 3 of 4 -- a row whose last entry is the -1 padding the runner hands for an unfilled page.

    CERTIFYING COMPONENT: the clamp on the table entry, and the producer's contract that no selected
    column falls in a padded page. Both halves are read: the values against the clamped-window
    oracle, and the selected rows against the padded region's own column range.
    """
    table = [5, 10, -1]
    bank = _bank()
    window = _window(bank, table)
    queries = _queries(1, HEADS, LATENT)
    live = (len(table) - 1) * PAGE
    selected = _selected(1, live)
    got = _paged(bank, table, selected, queries)
    want = _oracle(window, selected, queries)
    worst = float((got - want).abs().max())
    _say("PAD_TABLE_WORST_ABS", f"{worst:.3e}")
    _say("PAD_SELECTED_MAX", int(selected.max()))
    assert int(selected.max()) < live, (
        f"this item's own selected rows reach into the padded page (max {int(selected.max())} "
        f"against {live} live rows), so it would not read the contract it claims to read"
    )
    assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
        f"a padded table entry moved the window: worst absolute difference {worst:.3e}"
    )


def test_table_of_every_page_in_the_bank_agrees_with_the_oracle() -> None:
    """TABLE 4 of 4 -- a full-length row: every page of the bank, at the served window width.

    CERTIFYING COMPONENT: the page loop at the width the registered measurement reaches, where the
    window is the whole model length rather than two pages of it.
    """
    _read_one_table(list(range(BANK_PAGES)), heads=1)


def test_a_consecutive_table_is_bit_identical_to_the_unpaged_call() -> None:
    """The exact item, and the control for the four tolerance items above.

    CERTIFYING COMPONENT: the staged window itself. Pages 0 and 1 in order ARE the first 256 rows of
    the bank, so the paged call and the unpaged call on those rows feed one identical window into one
    identical arithmetic: equal bytes, not a tolerance. A page loaded at a wrong offset, a page loaded
    twice, or a transpose landing at a wrong column cannot survive this.
    """
    bank = _bank()
    table = [0, 1]
    queries = _queries(1, HEADS, LATENT)
    selected = _selected(1, len(table) * PAGE)
    paged = _paged(bank, table, selected, queries)
    unpaged = MS.mla_sparse_attention(queries, _window(bank, table), selected, SCALE)
    _say("IDENTICAL_MAX_ABS_DIFF", float((paged - unpaged).abs().max()))
    assert torch.equal(paged, unpaged), (
        "the paged call and the unpaged call on the same window returned different bytes, so the "
        "page assembly is not a re-addressing of the same load"
    )


def test_sentinel_columns_still_move_nothing_when_the_window_is_paged() -> None:
    """The producer's -1 selected-row sentinel keeps its meaning over a paged window.

    CERTIFYING COMPONENT: the selected-row mask, which the page loop must leave untouched. Half the
    columns of the row are the sentinel; the reading is against the oracle, which carries the same
    semantics, and against a row with no sentinel at all, which must NOT agree.
    """
    bank = _bank()
    table = [5, 10]
    queries = _queries(1, HEADS, LATENT)
    live = _selected(1, len(table) * PAGE)
    masked = live.clone()
    masked[:, TOPK // 2:] = MS.SENTINEL_INDEX
    window = _window(bank, table)
    got = _paged(bank, table, masked, queries)
    want = _oracle(window, masked, queries)
    worst = float((got - want).abs().max())
    apart = float((got - _oracle(window, live, queries)).abs().max())
    _say("SENTINEL_WORST_ABS", f"{worst:.3e}")
    _say("SENTINEL_APART_FROM_UNMASKED", f"{apart:.3e}")
    assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
        f"the sentinel columns of a paged window are not masked as the oracle masks them: worst "
        f"absolute difference {worst:.3e}"
    )
    assert apart > ATOL, (
        "masking half the selected columns changed nothing, so this item would pass on a kernel that "
        "ignores the sentinel entirely"
    )


def test_overlay_of_one_written_row_is_what_the_gather_reads() -> None:
    """OVERLAY 1 of 2 -- one row, the decode step's own latent, at a runtime row offset.

    CERTIFYING COMPONENT: the overlay operand. The bank row under the offset holds one value and the
    overlay another, so a kernel that read the bank instead of the overlay reads the wrong one. This
    is the hazard the landed seam repair removed for the unpaged window: a write the traced graph has
    not ordered before this read is not visible, so the rows travel as an operand.
    """
    bank = _bank()
    table = [5, 10]
    at = 200
    written = torch.full((1, LATENT), -7.5, dtype=torch.float32)
    window = _window(bank, table).index_copy(0, torch.tensor([at]), written)
    queries = _queries(1, HEADS, LATENT)
    selected = _selected(1, len(table) * PAGE)
    got = _paged(bank, table, selected, queries, written,
                 torch.tensor([[at]], dtype=torch.int32))
    want = _oracle(window, selected, queries)
    stale = _oracle(_window(bank, table), selected, queries)
    worst = float((got - want).abs().max())
    _say("OVERLAY_ONE_WORST_ABS", f"{worst:.3e}")
    _say("OVERLAY_ONE_APART_FROM_STALE", f"{float((got - stale).abs().max()):.3e}")
    assert int(selected.min()) <= at <= int(selected.max()), (
        f"row {at} is outside this item's selected range, so the overlay would not be read at all"
    )
    assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
        f"the overlaid row is not what the gather read: worst absolute difference {worst:.3e}"
    )
    assert float((got - stale).abs().max()) > ATOL, (
        "the overlay changed nothing against the bank as it stood, so this item would pass on a "
        "kernel that ignores the overlay operand"
    )


def test_overlay_of_a_whole_prefill_chunk_is_what_the_gather_reads() -> None:
    """OVERLAY 2 of 2 -- the widest overlay the served configuration produces, one prefill chunk.

    CERTIFYING COMPONENT: the overlay at its declared maximum width, at an offset that is not a page
    multiple, so the written rows cross page boundaries inside the staged window. ONE query, because
    the width of the overlay is this item's conjunct and the query count is not.
    """
    bank = _bank()
    table = list(range(BANK_PAGES))
    written = torch.arange(CHUNK, dtype=torch.float32).reshape(CHUNK, 1).expand(CHUNK, LATENT)
    written = (written * -1.0 - 1.0).contiguous()
    rows = torch.arange(CHUNK_AT, CHUNK_AT + CHUNK)
    window = _window(bank, table).index_copy(0, rows, written)
    queries = _queries(1, 1, LATENT)
    selected = _selected(1, len(table) * PAGE)
    got = _paged(bank, table, selected, queries, written,
                 torch.tensor([[CHUNK_AT]], dtype=torch.int32))
    want = _oracle(window, selected, queries)
    worst = float((got - want).abs().max())
    inside = int(((selected >= CHUNK_AT) & (selected < CHUNK_AT + CHUNK)).sum())
    _say("OVERLAY_CHUNK_WORST_ABS", f"{worst:.3e}")
    _say("OVERLAY_CHUNK_SELECTED_INSIDE", inside)
    assert inside > 0, (
        f"none of this item's {TOPK} selected rows falls inside the overlaid chunk, so the width it "
        f"claims to read is not read"
    )
    assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
        f"the overlaid chunk is not what the gather read: worst absolute difference {worst:.3e}"
    )
