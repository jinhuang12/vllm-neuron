# SPDX-License-Identifier: Apache-2.0
"""Tests for the paged latent window of the sparse attention kernel.

The kernel assembles its window from a block table instead of slicing one ascending run
of blocks out of the latent bank. Every window is read twice: exactly against the
unpaged call on the window the table names, because a paged load is a re-addressing of
one identical load and nothing else; and within a tolerance against the torch oracle,
which contracts in one shot while the kernel contracts in tiles and normalises with a
running maximum.

Softmax attention is permutation-invariant in the gathered rows, so a selection whose
in-page offsets repeat from page to page is gathered onto itself when two pages are
swapped. ``_selected`` therefore uses a different offset set in every page, and it also
selects the first and last row of every page, because a page copy that moves 127 of 128
rows leaves stale content only in the rows a selection without page edges never reads.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional.attention import mla_sparse as MS

#: The served KV block, and the width every table entry addresses. A page is a whole
#: number of the kernel's 16-row DMA transposes.
PAGE = 128

#: A second admissible block size, so a body that hardcodes the served one fails. 64 is
#: also a whole number of 16-row transposes, and no offset in this file is aligned to it.
PAGE_ALT = 64

#: A page wider than one staging piece. The window is staged in pieces of 128 rows, the
#: SBUF partition bound, so at this size every page is two pieces and the piece index is
#: arithmetic rather than the page number itself.
PAGE_WIDE = 256

#: The served block size, and a bank of two of them counted in PAGE rows as _bank counts.
#: Staging writes a block this wide in 32 transfers of 128 rows.
PAGE_SERVED = 4096
SERVED_BANK_PAGES = 2 * PAGE_SERVED // PAGE

#: A selected-row count past the moving tile width, which is what routes a call to the
#: row-tiled body. The served indexer emits 2,048, so the row-tiled body is the one
#: production takes; every other test here runs at TOPK and reaches the untiled body.
TOPK_ROWS = 1024
ROW_TILED_TABLE = [7, 2]
ROW_TILED_OVERLAY_AT = 200

#: The bank numbers its own rows, and an unscaled row index reaches 2 ** 21, which drives
#: the scores so far apart that the softmax becomes one weight of 1 and 127 of 0. Under
#: that weighting the output is a copy of one gathered row, and masking a column or
#: overlaying a row shows no change. Dividing by this power of two keeps every value
#: exact and puts the scaled scores in single digits, where every selected row carries
#: weight.
SPREAD = 2.0**18

#: This checkpoint's latent rank, and the head count the small tests run at. The oracle's
#: cost grows with both, so only the full-table and chunk-overlay tests need more.
LATENT = 512
HEADS = 2

#: The smallest admissible selected-row count, and one fixed scale, applied identically
#: to the kernel and to the oracle it is read against.
TOPK = 128
SCALE = 0.1

#: The bf16 module-comparison thresholds this module is read at elsewhere.
RTOL = 1e-2
ATOL = 1e-5

#: Thirty-two pages: the served model length over the served block, so a full-length
#: table is the widest window this configuration reaches.
BANK_PAGES = 32

#: Twice the bank's widest value. The one-row overlay has to move the weighted output
#: outside the band this file's tolerance allows; a row whose value sits inside the
#: bank's own [0, 8] range carries about the weight its neighbours carry and is invisible
#: at that band. The sign comes from the head's own query sum, so the row is built from
#: the queries rather than written down here.
OVERLAY_MAGNITUDE = 16.0

#: The served prefill chunk, and the row the chunk overlay starts at -- not a multiple of
#: the page, so the overlay is read at an offset no page boundary hides.
CHUNK = 1024
CHUNK_AT = 3000

#: The first and last row the chunk writes. An overlay short by its first 16-row block or
#: by its last row leaves those rows holding the bank, and a selection that reads neither
#: edge returns the reference bytes on a kernel that never wrote them.
CHUNK_EDGES = (CHUNK_AT, CHUNK_AT + CHUNK - 1)


def _bank(pages: int = BANK_PAGES, latent: int = LATENT) -> torch.Tensor:
    """The latent bank, every row distinguishable: row r column c holds (r * latent + c) / SPREAD."""
    rows = pages * PAGE
    down = torch.arange(rows, dtype=torch.float32).reshape(rows, 1) * latent
    return (down + torch.arange(latent, dtype=torch.float32).reshape(1, latent)) / SPREAD


def _window(bank: torch.Tensor, table: list[int], page: int = PAGE) -> torch.Tensor:
    """The window the table names, a -1 entry clamped to page 0 exactly as the load clamps it."""
    rows = [max(entry, 0) * page + inside for entry in table for inside in range(page)]
    return bank[rows]


def _queries(seq: int, heads: int, latent: int) -> torch.Tensor:
    """Query latents that depend on every axis, so a transposed or reused tile shows."""
    torch.manual_seed(seq * 1000 + heads * 10 + latent // 128)
    return torch.randn(seq, heads, latent, dtype=torch.float32) * 0.05


def _offsets(index: int, count: int, page: int) -> list[int]:
    """The in-page offsets read in page `index`: both edges, then a stride only this page uses."""
    held = {0, page - 1}
    stride = 2 * index + 3
    pick = 1
    while len(held) < count:
        held.add((pick * stride + index) % page)
        pick += 1
    return sorted(held)


def _selected(seq: int, window_rows: int, topk: int = TOPK, page: int = PAGE) -> torch.Tensor:
    """Selected window positions: both edges of every page, and an offset set no other page repeats."""
    pages = max(window_rows // page, 1)
    columns: list[int] = []
    for index in range(pages):
        count = max(topk // pages + (1 if index < topk % pages else 0), 2)
        columns.extend(index * page + inside for inside in _offsets(index, count, page))
    return torch.tensor([columns[:topk] for _ in range(seq)], dtype=torch.int32)


def _interior(selected: torch.Tensor) -> int:
    """The last selected row that is not its page's first or last, so the overlay lands deep."""
    inside = [int(one) for one in selected[0].tolist() if int(one) % PAGE not in (0, PAGE - 1)]
    assert inside, "every selected row is a page edge, so no overlay row could be read"
    return inside[-1]


def _overlay_row(queries: torch.Tensor) -> torch.Tensor:
    """This step's one written row, in the sign the first head's query sum puts weight on."""
    lean = float(queries[0, 0].sum())
    return torch.full((1, LATENT), OVERLAY_MAGNITUDE if lean > 0 else -OVERLAY_MAGNITUDE,
                      dtype=torch.float32)


def _with_rows(selected: torch.Tensor, wanted: tuple[int, ...]) -> torch.Tensor:
    """The same selection with `wanted` rows put in place of interior columns, keeping every page edge."""
    held = selected.clone()
    spots = [index for index, one in enumerate(held[0].tolist())
             if int(one) % PAGE not in (0, PAGE - 1)]
    assert len(spots) >= len(wanted), "there are no interior columns to give the wanted rows"
    for spot, row in zip(spots, wanted):
        held[:, spot] = row
    return held


def _chunk_rows() -> torch.Tensor:
    """The prefill chunk this step writes, one distinguishable row each, below the bank's range."""
    held = torch.arange(CHUNK, dtype=torch.float32).reshape(CHUNK, 1).expand(CHUNK, LATENT)
    return (held * (-LATENT / SPREAD) - 1.0).contiguous()


def _paged(bank, table, selected, queries, written=None, write_offset=None, page: int = PAGE):
    """The seam under test, called the way the runner will call it.

    The table travels as a column, `[pages, 1]` int32: the shape the page number is read
    out of on device, and the shape every other index operand in this tree takes.
    """
    offset = torch.zeros(1, 1, dtype=torch.int32) if write_offset is None else write_offset
    rows = torch.zeros(0, bank.shape[1], dtype=bank.dtype) if written is None else written
    return MS.mla_sparse_attention(
        queries, bank, selected, SCALE,
        block_table_row=torch.tensor([[entry] for entry in table], dtype=torch.int32),
        written=rows, write_offset=offset, page_size=page,
    )


def _oracle(window, selected, queries) -> torch.Tensor:
    """The same attention over the window the table names, in torch."""
    return MS.mla_sparse_attention_torch_oracle(queries, window, selected, SCALE)


def _unpaged(window, selected, queries) -> torch.Tensor:
    """The same kernel on the same window, reached without a block table: the exact reference."""
    return MS.mla_sparse_attention(queries, window, selected, SCALE)


def _read_one_table(table: list[int], heads: int = HEADS) -> None:
    """Run one table layout and read it twice: against the oracle, and against the unpaged call."""
    bank = _bank()
    window = _window(bank, table)
    queries = _queries(1, heads, LATENT)
    selected = _selected(1, window.shape[0])
    got = _paged(bank, table, selected, queries)
    expected = _oracle(window, selected, queries)
    worst = float((got - expected).abs().max())
    assert torch.allclose(got, expected, rtol=RTOL, atol=ATOL), (
        f"the paged window disagrees with the oracle on table {table}: worst absolute difference "
        f"{worst:.3e} against rtol={RTOL} atol={ATOL}. A window assembled at the wrong page offset "
        f"reads another sequence's latents"
    )
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        f"the paged call on table {table} and the unpaged call on the window it names returned "
        f"different bytes, so the page assembly is not a re-addressing of the same load"
    )


def test_table_five_then_ten_agrees_with_the_oracle() -> None:
    """Two scattered pages in ascending order: the page loop reads its source offset off the table."""
    _read_one_table([5, 10])


def test_table_ten_then_five_agrees_with_the_oracle() -> None:
    """The same two pages in the other order, which no ascending run can express.

    The two pages are read at different in-page offsets, so swapping them changes the
    gathered rows rather than permuting them. The byte comparison against the unpaged
    call on the ordered window is what a swapped load cannot pass; the oracle's band is
    wider than the tilt a swap leaves in the output.
    """
    _read_one_table([10, 5])


def test_table_with_a_padding_tail_agrees_with_the_oracle() -> None:
    """A table row whose last entry is the -1 padding the runner hands for an unfilled page.

    Both halves of the contract are read. With only the producer's half -- no selected
    column inside the padded page -- a kernel that never clamps at all returns the same
    bytes, so the second half selects inside the padded page on purpose and reads the
    clamp's specified behaviour, page 0.
    """
    table = [5, 10, -1]
    bank = _bank()
    window = _window(bank, table)
    queries = _queries(1, HEADS, LATENT)
    live = (len(table) - 1) * PAGE
    selected = _selected(1, live)
    got = _paged(bank, table, selected, queries)
    expected = _oracle(window, selected, queries)
    worst = float((got - expected).abs().max())
    assert torch.allclose(got, expected, rtol=RTOL, atol=ATOL), (
        f"a padded table entry moved the window: worst absolute difference {worst:.3e}"
    )
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        "the padded table's paged call and the unpaged call on the clamped window it names returned "
        "different bytes, so the clamp does not load the page the window is read against"
    )
    inside = torch.tensor([[live + (step * 7) % PAGE for step in range(TOPK)]], dtype=torch.int32)
    padded = _paged(bank, table, inside, queries)
    clamped = _oracle(window, inside, queries)
    assert torch.equal(padded, _unpaged(window, inside, queries)), (
        "rows gathered from the padded page are not what page 0 holds, so the -1 entry is not clamped "
        "to page 0: an unclamped entry reads a wrapped page, an arbitrary page, or uninitialised HBM"
    )
    assert torch.allclose(padded, clamped, rtol=RTOL, atol=ATOL), (
        "the padded page's own columns disagree with the oracle on the clamped window"
    )


def test_table_of_every_page_in_the_bank_agrees_with_the_oracle() -> None:
    """A full-length table row: the page loop at the whole model length rather than two pages of it."""
    _read_one_table(list(range(BANK_PAGES)), heads=1)


def test_a_consecutive_table_is_bit_identical_to_the_unpaged_call() -> None:
    """Pages 0 and 1 in order are the bank's first 256 rows, so paged and unpaged must be equal bytes."""
    bank = _bank()
    table = [0, 1]
    queries = _queries(1, HEADS, LATENT)
    selected = _selected(1, len(table) * PAGE)
    paged = _paged(bank, table, selected, queries)
    unpaged = _unpaged(_window(bank, table), selected, queries)
    assert torch.equal(paged, unpaged), (
        "the paged call and the unpaged call on the same window returned different bytes, so the "
        "page assembly is not a re-addressing of the same load"
    )


def test_sentinel_columns_still_move_nothing_when_the_window_is_paged() -> None:
    """The -1 selected-row sentinel keeps its meaning over a paged window.

    Half the columns of the row are the sentinel. The reading is against the oracle,
    which carries the same semantics, and against a row with no sentinel at all, which
    must not agree at this test's own band.
    """
    bank = _bank()
    table = [5, 10]
    queries = _queries(1, HEADS, LATENT)
    live = _selected(1, len(table) * PAGE)
    masked = live.clone()
    masked[:, TOPK // 2:] = MS.SENTINEL_INDEX
    window = _window(bank, table)
    got = _paged(bank, table, masked, queries)
    expected = _oracle(window, masked, queries)
    unmasked = _oracle(window, live, queries)
    worst = float((got - expected).abs().max())
    assert torch.allclose(got, expected, rtol=RTOL, atol=ATOL), (
        f"the sentinel columns of a paged window are not masked as the oracle masks them: worst "
        f"absolute difference {worst:.3e}"
    )
    assert not torch.allclose(got, unmasked, rtol=RTOL, atol=ATOL), (
        "masking half the selected columns moved the result by less than the band this test's own "
        "tolerance allows, so it would pass on a kernel that ignores the sentinel entirely"
    )


def test_overlay_of_one_written_row_is_what_the_gather_reads() -> None:
    """One overlaid row at a runtime row offset is read instead of the bank row beneath it.

    A write the traced graph has not ordered before this read is not visible, so this
    step's rows travel as an operand. The exact claim is what catches an overlay written
    one row off; the tolerance claim alone does not.
    """
    bank = _bank()
    table = [5, 10]
    queries = _queries(1, HEADS, LATENT)
    selected = _selected(1, len(table) * PAGE)
    at = _interior(selected)
    written = _overlay_row(queries)
    window = _window(bank, table).index_copy(0, torch.tensor([at]), written)
    stale = _oracle(_window(bank, table), selected, queries)
    got = _paged(bank, table, selected, queries, written, torch.tensor([[at]], dtype=torch.int32))
    expected = _oracle(window, selected, queries)
    worst = float((got - expected).abs().max())
    assert torch.allclose(got, expected, rtol=RTOL, atol=ATOL), (
        f"the overlaid row is not what the gather read: worst absolute difference {worst:.3e}"
    )
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        "the overlaid window through the paged call and through the unpaged call returned different "
        "bytes, so the overlay does not land where this step's rows belong"
    )
    assert not torch.allclose(got, stale, rtol=RTOL, atol=ATOL), (
        "the overlay moved the result by less than this test's own band, so it would pass on a kernel "
        "that ignores the overlay operand"
    )


def test_overlay_of_a_whole_prefill_chunk_is_what_the_gather_reads() -> None:
    """The widest overlay the served configuration produces, at an offset that is not a page multiple.

    The written rows therefore cross page boundaries inside the staged window. One query,
    because the width of the overlay is what this test reads and the query count is not.
    """
    bank = _bank()
    table = list(range(BANK_PAGES))
    written = _chunk_rows()
    rows = torch.arange(CHUNK_AT, CHUNK_AT + CHUNK)
    window = _window(bank, table).index_copy(0, rows, written)
    queries = _queries(1, 1, LATENT)
    selected = _with_rows(_selected(1, len(table) * PAGE), CHUNK_EDGES)
    got = _paged(bank, table, selected, queries, written,
                 torch.tensor([[CHUNK_AT]], dtype=torch.int32))
    expected = _oracle(window, selected, queries)
    worst = float((got - expected).abs().max())
    assert torch.allclose(got, expected, rtol=RTOL, atol=ATOL), (
        f"the overlaid chunk is not what the gather read: worst absolute difference {worst:.3e}"
    )
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        "the overlaid window through the paged call and through the unpaged call returned different "
        "bytes, so a chunk written one row off or one row short would pass unseen"
    )


def test_an_overlay_that_fills_the_window_is_read_row_for_row() -> None:
    """An overlay whose rows fill the staged window, at three widths, read against one-step torch.

    A step whose rows reach the window's last row puts the overlay's destination end on
    the tile's end, which the traced venue reads as one element past it, so the tile
    carries pad rows no body reads. All three widths sit on that boundary: one 128-row
    page, 64 rows (narrower than a staging transfer), and the served block of 4,096 rows,
    which staging fills in 32 transfers. The window's last row is selected in each.
    """
    for page, entry, pages in ((PAGE, 7, BANK_PAGES), (PAGE_ALT, 5, BANK_PAGES),
                               (PAGE_SERVED, 1, SERVED_BANK_PAGES)):
        bank = _bank(pages)
        table = [entry]
        width = page
        down = torch.arange(width, dtype=torch.float32).reshape(width, 1).expand(width, LATENT)
        written = (down * (-LATENT / SPREAD) - 1.0).contiguous()
        window = _window(bank, table, page=page).index_copy(0, torch.arange(width), written)
        queries = _queries(1, 1, LATENT)
        columns = [(pick * max(1, width // TOPK)) % width for pick in range(TOPK)]
        columns[-1] = width - 1
        selected = torch.tensor([columns], dtype=torch.int32)
        got = _paged(bank, table, selected, queries, written,
                     torch.tensor([[0]], dtype=torch.int32), page=page)
        expected = _oracle(window, selected, queries)
        worst = float((got - expected).abs().max())
        assert torch.allclose(got, expected, rtol=RTOL, atol=ATOL), (
            f"the overlay filling a {width}-row window is not what the gather read: worst absolute "
            f"difference {worst:.3e}"
        )
        assert torch.equal(got, _unpaged(window, selected, queries)), (
            f"the {width}-row overlay read through the paged call and through the unpaged call on the "
            f"window written in one step returned different bytes, so the split moved a row"
        )


def test_a_step_that_fills_a_two_chunk_window_is_read_row_for_row() -> None:
    """A step filling a 256-row window is written as 128 rows plus 128 more at a runtime offset.

    The test above fills a window the size of one transfer, so its overlay is a single
    pattern. Both edges of both chunks are selected here, so a chunk written at the wrong
    row, dropped or repeated cannot return the reference bytes.
    """
    bank = _bank()
    table = [7, 2]
    width = 2 * PAGE
    down = torch.arange(width, dtype=torch.float32).reshape(width, 1).expand(width, LATENT)
    written = (down * (-LATENT / SPREAD) - 1.0).contiguous()
    window = _window(bank, table).index_copy(0, torch.arange(width), written)
    queries = _queries(1, 1, LATENT)
    edges = (0, PAGE - 1, PAGE, width - 1)
    columns = [edges[pick % len(edges)] if pick < len(edges) else pick % width for pick in range(TOPK)]
    selected = torch.tensor([sorted(set(columns)) + [width - 1] * (TOPK - len(set(columns)))],
                            dtype=torch.int32)
    got = _paged(bank, table, selected, queries, written, torch.tensor([[0]], dtype=torch.int32))
    expected = _oracle(window, selected, queries)
    worst = float((got - expected).abs().max())
    assert torch.allclose(got, expected, rtol=RTOL, atol=ATOL), (
        f"the step filling a {width}-row window in two chunks is not what the gather read: worst "
        f"absolute difference {worst:.3e}"
    )
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        f"the two-chunk step read through the paged call and through the unpaged call on the window "
        f"written in one step returned different bytes, so a chunk landed at the wrong row"
    )


def test_a_second_block_size_is_read_from_the_operand_and_not_assumed() -> None:
    """The page size travels as an operand, so a body that hardcodes the served one fails here.

    The table, the window arithmetic and the overlay offset are all in units of it, and
    64 is a whole number of the kernel's 16-row transposes exactly as 128 is.
    """
    bank = _bank()
    table = [11, 20, 3]
    queries = _queries(1, HEADS, LATENT)
    window = _window(bank, table, page=PAGE_ALT)
    selected = _selected(1, window.shape[0], page=PAGE_ALT)
    got = _paged(bank, table, selected, queries, page=PAGE_ALT)
    expected = _oracle(window, selected, queries)
    worst = float((got - expected).abs().max())
    assert torch.allclose(got, expected, rtol=RTOL, atol=ATOL), (
        f"at a page size of {PAGE_ALT} the paged window disagrees with the oracle: {worst:.3e}"
    )
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        f"at a page size of {PAGE_ALT} the paged call and the unpaged call on the window the table "
        f"names returned different bytes, so the page size is not read from the operand"
    )


def test_a_page_wider_than_one_staging_piece_is_read_whole() -> None:
    """A page of 256 rows is two staging pieces, and both halves of every page carry selected rows.

    At every other page size in this file a page is one piece and the piece index is the
    page number, so a body that stages only a page's first 128 rows, or that computes the
    piece index wrongly, passes all of them.
    """
    bank = _bank()
    table = [7, 2]
    queries = _queries(1, HEADS, LATENT)
    window = _window(bank, table, page=PAGE_WIDE)
    selected = _selected(1, window.shape[0], page=PAGE_WIDE)
    got = _paged(bank, table, selected, queries, page=PAGE_WIDE)
    expected = _oracle(window, selected, queries)
    worst = float((got - expected).abs().max())
    assert torch.allclose(got, expected, rtol=RTOL, atol=ATOL), (
        f"at a page size of {PAGE_WIDE} the paged window disagrees with the oracle: {worst:.3e}"
    )
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        f"at a page size of {PAGE_WIDE} the paged call and the unpaged call on the window the table "
        f"names returned different bytes, so a page wider than one staging piece is not assembled whole"
    )


def test_a_selected_row_past_the_staged_window_is_refused() -> None:
    """The range check is against the window the table names, never the bank the pages come from.

    The bank holds 4,096 rows and this table stages 256 of them, so a selected row of 300
    indexes the bank but not the window, and gathering it would read a page this request
    was never given.
    """
    bank = _bank()
    table = [5, 10]
    queries = _queries(1, HEADS, LATENT)
    selected = _with_rows(_selected(1, len(table) * PAGE), (len(table) * PAGE + 44,))
    message = ""
    try:
        _paged(bank, table, selected, queries)
    except MS.MlaSparseAttentionError as refusal:
        message = " ".join(str(refusal).split())
    assert str(len(table) * PAGE) in message, (
        f"a selected row past the staged window was not refused against the window's own length "
        f"{len(table) * PAGE}: {message or 'it was not refused at all'}"
    )


def test_a_scattered_table_a_padding_tail_and_an_overlay_in_one_call() -> None:
    """The page loop, the clamp and the overlay compose in a single call.

    Each is read alone above; a kernel can pass all three and still order the overlay
    before the page that covers it, or clamp using an offset the overlay moved.
    """
    bank = _bank()
    table = [10, 5, -1]
    queries = _queries(1, HEADS, LATENT)
    live = (len(table) - 1) * PAGE
    selected = _selected(1, live)
    at = _interior(selected)
    written = _overlay_row(queries)
    window = _window(bank, table).index_copy(0, torch.tensor([at]), written)
    got = _paged(bank, table, selected, queries, written, torch.tensor([[at]], dtype=torch.int32))
    expected = _oracle(window, selected, queries)
    stale = _oracle(_window(bank, table), selected, queries)
    worst = float((got - expected).abs().max())
    assert torch.allclose(got, expected, rtol=RTOL, atol=ATOL), (
        f"the scattered table, the padded tail and the overlay do not compose: worst absolute "
        f"difference {worst:.3e}"
    )
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        "the combined call and the unpaged call on the window it names returned different bytes"
    )
    assert not torch.allclose(got, stale, rtol=RTOL, atol=ATOL), (
        "the overlay moved nothing in the combined call, so this test would pass on a kernel that "
        "stages the pages over this step's own rows"
    )


def _rows_selected(window_rows: int, topk: int = TOPK_ROWS) -> torch.Tensor:
    """`topk` columns over a shorter window: every row read, most of them more than once."""
    stride = 7
    return torch.tensor([[(pick * stride) % window_rows for pick in range(topk)]], dtype=torch.int32)


def _staged_path(bank, table, selected, queries, written=None, at=None):
    """The same paged call on the staged load path instead of the streamed one.

    The tile width keeps its default, so the load path is the only difference from the
    seam's own call.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(MS.mla_sparse_attention_nope_row_tiled_kernel)(
        queries, bank, selected, float(SCALE),
        torch.tensor([[entry] for entry in table], dtype=torch.int32),
        written, at, PAGE, MS.MOVING_MAX, False)


def test_the_row_tiled_body_reads_the_staged_window_the_same_way() -> None:
    """The row-tiled body gathers out of a window staged in private memory, the same as every path.

    It reads selected rows out of the cache operand with an indirect DMA instead of
    transposing the cache. Every test above runs at a selected-row count inside one
    moving tile and reaches the untiled body, which never does that. Three exact
    readings: against the unpaged call on the window the table names, against this body's
    other load path, and with one row overlaid against the overlaid window.
    """
    bank = _bank()
    table = ROW_TILED_TABLE
    queries = _queries(1, HEADS, LATENT)
    window = _window(bank, table)
    selected = _rows_selected(window.shape[0])
    overlay = _overlay_row(queries)
    overlaid = window.clone()
    overlaid[ROW_TILED_OVERLAY_AT] = overlay[0]
    at = torch.tensor([[ROW_TILED_OVERLAY_AT]], dtype=torch.int32)
    got = _paged(bank, table, selected, queries)
    unpaged = _unpaged(window, selected, queries)
    staged = _staged_path(bank, table, selected, queries)
    written = _paged(bank, table, selected, queries, written=overlay, write_offset=at)
    assert TOPK_ROWS > MS.MOVING_MAX, (
        f"topk={TOPK_ROWS} does not reach the row-tiled body, so this test reads the same body as "
        f"every test above it"
    )
    assert torch.equal(got, unpaged), (
        "the row-tiled body returned different bytes for the paged window and for the unpaged call on "
        "the window the table names, so streaming out of the staged window is not the same load"
    )
    assert torch.equal(got, staged), (
        "the row-tiled body's two load paths disagree on one window: the streamed gather and the "
        "staged transpose returned different bytes"
    )
    assert torch.equal(written, _unpaged(overlaid, selected, queries)), (
        "with one row overlaid, the row-tiled body and the unpaged call on the overlaid window "
        "returned different bytes, so the overlay is not what the streamed gather reads"
    )
