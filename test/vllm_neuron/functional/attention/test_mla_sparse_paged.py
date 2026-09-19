# SPDX-License-Identifier: Apache-2.0
"""Acceptance for the paged latent window: the sparse attention kernel assembles its window from a
block table instead of slicing one ascending run of blocks out of the latent bank.

TWELVE tests and NO `parametrize` decorator. Four carry `table` in their name and read the four block
layouts the design declares -- two scattered rows in either order, a row with a padding tail, and a
full-length row. One carries `identical` and reads the consecutive table. One carries `sentinel`. Two
carry `overlay` and read this step's own rows, at one row and at a whole prefill chunk. One reads a
second block size, so a body that hardcodes the served one fails. One reads that a selected row past the
staged window is refused against the WINDOW's length and not the bank's. One puts a scattered table, a
padding tail and an overlay in a single call, which no other item combines. The last calls no kernel: it reads
that this file's own numbers can see what the items above claim to see.

EVERY WINDOW CLAIM IS MADE TWICE, ONE EXACT AND ONE A TOLERANCE. The exact claim is against the
UNPAGED call on the window the table names: the same window through the same kernel must return the
same bytes, because a paged load is a re-addressing of one identical load and nothing else. The
tolerance claim is against the torch oracle, and it is a tolerance rather than bit-identity because the
kernel contracts on the tensor engine in tiles and normalises with a running maximum while the oracle
contracts in torch in one shot; the landed acceptance of this module reads the oracle at the same
module-comparison tolerance. The two answer different questions: the exact one says the assembly
re-addresses the same load, the tolerance one says the assembled window holds the latents the table
names at all.

WHAT THE SELECTED ROWS HAVE TO DO, AND WHY IT IS NOT AN ARBITRARY PATTERN. Softmax attention is
permutation-invariant in the gathered rows, so a selection whose in-page offsets repeat from page to
page is gathered onto itself when the pages are SWAPPED, and no reading of any strength can then see a
kernel that loads pages in the wrong order. `_selected` therefore uses a different offset set in every
page. It also selects the FIRST and LAST row of every page, because a page copy that moves 127 of 128
rows leaves stale content only in rows a selection without page edges never reads.

The declared command for this file::

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
#: kernel's 16-row DMA transposes, so the transposes over the staged window are the unpaged ones.
PAGE = 128

#: A second admissible block size, read by one item so that a body which hardcodes the served one
#: fails. 64 is also a whole number of 16-row transposes, and it is not a size this file's offsets or
#: the chunk offset are aligned to.
PAGE_ALT = 64

#: A page WIDER than one staging piece. The window is staged in pieces of 128 rows, the SBUF partition
#: bound, so at this size every page is two pieces and the piece index is arithmetic rather than the page
#: number itself. The served interim block is 4,096, which is 32 pieces; every other item here is one.
PAGE_WIDE = 256

#: A SELECTED-ROW COUNT PAST THE MOVING TILE WIDTH, which is what routes a call to the row-tiled body.
#: Every item above runs at `TOPK` and reaches the untiled body; the served indexer emits 2,048, so the
#: row-tiled body is the one production takes, and it loads the window in its own way.
TOPK_ROWS = 1024
STREAMED_TABLE = [7, 2]
STREAMED_OVERLAY_AT = 200

#: THE SPREAD OF THE BANK'S VALUES IS LOAD-BEARING. The bank numbers its own rows, and an unscaled row
#: index reaches 2 ** 21, which drives every score so far apart that the softmax becomes one weight of
#: 1 and 127 of 0. Under that weighting the output is a copy of ONE gathered row, and an item that
#: masks columns or overlays a row reads no change unless it happens to hit the chosen one. Dividing by
#: this power of two keeps every value exact and puts the scaled scores in single digits, where every
#: selected row carries weight. The last item of this file reads that it is still so.
SPREAD = 2.0**18

#: This checkpoint's own latent rank, and the head count the small items run at. The oracle's cost
#: grows with both, so only the full-table and chunk-overlay items need more than two heads' worth.
LATENT = 512
HEADS = 2

#: The smallest admissible selected-row count, and one fixed scale, applied identically to the kernel
#: and to the oracle it is read against.
TOPK = 128
SCALE = 0.1

#: The landed bf16 module-comparison thresholds this module is already read at.
RTOL = 1e-2
ATOL = 1e-5

#: The bank holds thirty-two pages: the served model length over the served block, so a full-length
#: table is the widest window this configuration reaches, and page 10 addresses a real page.
BANK_PAGES = 32

#: THE ONE-ROW OVERLAY'S MAGNITUDE, and why it is not a small number. The reading it has to make is
#: "this row was read", compared at the band the item's own tolerance allows, roughly rtol times the
#: output. A row whose value sits inside the bank's own range of [0, 8] carries about the weight its
#: neighbours carry, moves the weighted output by under one part in a hundred, and is invisible at that
#: band. Twice the bank's widest value carries enough weight to move the output far outside it -- in the
#: SIGN the head's own query sum rewards, which is why the row is built from the queries rather than
#: written down here.
OVERLAY_MAGNITUDE = 16.0

#: The prefill chunk this campaign serves, and the row the chunk overlay starts at -- not a multiple
#: of the page, so the overlay is read at an offset no page boundary hides.
CHUNK = 1024
CHUNK_AT = 3000

#: The first and last row the chunk writes. THEY ARE SELECTED ON PURPOSE: an overlay short by its first
#: 16-row block or by its last row leaves those rows holding the bank, and a selection that reads
#: neither edge returns the reference bytes on a kernel that never wrote them.
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
    """The LAST selected row that is not its page's first or last, so the overlay lands deep."""
    inside = [int(one) for one in selected[0].tolist() if int(one) % PAGE not in (0, PAGE - 1)]
    assert inside, "every selected row is a page edge, so no overlay row could be read"
    return inside[-1]


def _overlay_row(queries: torch.Tensor) -> torch.Tensor:
    """This step's one written row, in the sign the first head's query sum puts weight on."""
    lean = float(queries[0, 0].sum())
    return torch.full((1, LATENT), OVERLAY_MAGNITUDE if lean > 0 else -OVERLAY_MAGNITUDE,
                      dtype=torch.float32)


def _with_rows(selected: torch.Tensor, wanted: tuple[int, ...]) -> torch.Tensor:
    """The same selection with `wanted` rows put in place of INTERIOR columns, keeping every page edge.

    An item that needs a particular row read cannot rely on the stride having picked it. The rows given
    up are interior ones, because the page edges are a property the last item of this file asserts.
    """
    held = selected.clone()
    spots = [index for index, one in enumerate(held[0].tolist())
             if int(one) % PAGE not in (0, PAGE - 1)]
    assert len(spots) >= len(wanted), "there are no interior columns to give the wanted rows"
    for spot, row in zip(spots, wanted):
        held[:, spot] = row
    return held


def _gathered_rows(table: list[int], selected: torch.Tensor, page: int = PAGE) -> set[int]:
    """Which BANK rows a table gathers under this selection, as integers.

    The bank row index is the reading, never a bank value: the values are the row index divided by a
    power of two, so reading one and truncating it to an integer collapses whole pages onto one number
    and makes two different gathers look identical.
    """
    window = [max(entry, 0) * page + inside for entry in table for inside in range(page)]
    return {window[int(column)] for column in selected[0].tolist()}


def _chunk_rows() -> torch.Tensor:
    """The prefill chunk this step writes, one distinguishable row each, below the bank's range."""
    held = torch.arange(CHUNK, dtype=torch.float32).reshape(CHUNK, 1).expand(CHUNK, LATENT)
    return (held * (-LATENT / SPREAD) - 1.0).contiguous()


def _paged(bank, table, selected, queries, written=None, write_offset=None, page: int = PAGE):
    """The seam under test, called the way the runner will call it.

    The table travels as a COLUMN, `[pages, 1]` int32: the shape the page number is read out of on
    device, and the shape every other index operand in this tree takes.
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


def _say(label: str, value: object) -> None:
    """Print one counted value, so the transcript carries the reading and not just a pass."""
    print(f"PAGED_{label}={value}", flush=True)


def _read_one_table(table: list[int], heads: int = HEADS) -> None:
    """Run one table layout and read it twice: against the oracle, and against the unpaged call."""
    bank = _bank()
    window = _window(bank, table)
    queries = _queries(1, heads, LATENT)
    selected = _selected(1, window.shape[0])
    got = _paged(bank, table, selected, queries)
    want = _oracle(window, selected, queries)
    unpaged = _unpaged(window, selected, queries)
    worst = float((got - want).abs().max())
    _say("TABLE", ".".join(str(entry) for entry in table))
    _say("WORST_ABS", f"{worst:.3e}")
    _say("UNPAGED_MAX_ABS_DIFF", float((got - unpaged).abs().max()))
    assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
        f"the paged window disagrees with the oracle on table {table}: worst absolute difference "
        f"{worst:.3e} against rtol={RTOL} atol={ATOL}. A window assembled at the wrong page offset "
        f"reads another sequence's latents, which is the defect this increment exists to remove"
    )
    assert torch.equal(got, unpaged), (
        f"the paged call on table {table} and the unpaged call on the window it names returned "
        f"different bytes, so the page assembly is not a re-addressing of the same load"
    )


def test_table_five_then_ten_agrees_with_the_oracle() -> None:
    """TABLE 1 of 4 -- the layout the serving allocator handed the first multi-block request.

    CERTIFYING COMPONENT: the page loop's source offset, read off the block table.
    """
    _read_one_table([5, 10])


def test_table_ten_then_five_agrees_with_the_oracle() -> None:
    """TABLE 2 of 4 -- the same pages in the other order, which no ascending run can express.

    CERTIFYING COMPONENT: that the window follows the table's ORDER and not the page numbers'. The
    order is observable because the two pages are read at different in-page offsets, so swapping them
    changes the gathered rows rather than permuting them; the last item of this file reads that.

    THE EXACT CLAIM IS WHAT BINDS THE ORDER HERE. Both orders gather 64 rows from each page, so the
    oracle claim's band is far wider than the tilt a swap leaves in the output; the byte comparison
    against the unpaged call on the ordered window is what a swapped load cannot pass.
    """
    _read_one_table([10, 5])


def test_table_with_a_padding_tail_agrees_with_the_oracle() -> None:
    """TABLE 3 of 4 -- a row whose last entry is the -1 padding the runner hands for an unfilled page.

    CERTIFYING COMPONENT: the clamp on the table entry, and the producer's contract that no selected
    column falls in a padded page. Both halves are read, and the clamp needs the second one: with only
    the contract half, no gathered row lies in the padded columns and a kernel that never clamps at all
    returns the same bytes. The second half therefore selects INSIDE the padded page on purpose. The
    producer never does that -- the reading is of the clamp's specified behaviour, page 0, and of
    nothing about the producer.
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
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        "the padded table's paged call and the unpaged call on the clamped window it names returned "
        "different bytes, so the clamp does not load the page the window is read against"
    )
    inside = torch.tensor([[live + (step * 7) % PAGE for step in range(TOPK)]], dtype=torch.int32)
    padded = _paged(bank, table, inside, queries)
    clamped = _oracle(window, inside, queries)
    _say("PAD_INSIDE_COLUMNS", f"{int(inside.min())}.{int(inside.max())}")
    _say("PAD_INSIDE_WORST_ABS", f"{float((padded - clamped).abs().max()):.3e}")
    assert torch.equal(padded, _unpaged(window, inside, queries)), (
        "rows gathered from the padded page are not what page 0 holds, so the -1 entry is not clamped "
        "to page 0: an unclamped entry reads a wrapped page, an arbitrary page, or uninitialised HBM"
    )
    assert torch.allclose(padded, clamped, rtol=RTOL, atol=ATOL), (
        "the padded page's own columns disagree with the oracle on the clamped window"
    )


def test_table_of_every_page_in_the_bank_agrees_with_the_oracle() -> None:
    """TABLE 4 of 4 -- a full-length row: every page of the bank, at the served window width.

    CERTIFYING COMPONENT: the page loop at the width the registered measurement reaches, where the
    window is the whole model length rather than two pages of it.
    """
    _read_one_table(list(range(BANK_PAGES)), heads=1)


def test_a_consecutive_table_is_bit_identical_to_the_unpaged_call() -> None:
    """The exact item on the consecutive case.

    CERTIFYING COMPONENT: the staged window itself. Pages 0 and 1 in order ARE the first 256 rows of
    the bank, so the paged call and the unpaged call on those rows feed one identical window into one
    identical arithmetic: equal bytes, not a tolerance.
    """
    bank = _bank()
    table = [0, 1]
    queries = _queries(1, HEADS, LATENT)
    selected = _selected(1, len(table) * PAGE)
    paged = _paged(bank, table, selected, queries)
    unpaged = _unpaged(_window(bank, table), selected, queries)
    _say("IDENTICAL_MAX_ABS_DIFF", float((paged - unpaged).abs().max()))
    assert torch.equal(paged, unpaged), (
        "the paged call and the unpaged call on the same window returned different bytes, so the "
        "page assembly is not a re-addressing of the same load"
    )


def test_sentinel_columns_still_move_nothing_when_the_window_is_paged() -> None:
    """The producer's -1 selected-row sentinel keeps its meaning over a paged window.

    CERTIFYING COMPONENT: the selected-row mask, which the page loop must leave untouched. Half the
    columns of the row are the sentinel; the reading is against the oracle, which carries the same
    semantics, and against a row with no sentinel at all, which must NOT agree at this item's own band.
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
    unmasked = _oracle(window, live, queries)
    worst = float((got - want).abs().max())
    _say("SENTINEL_WORST_ABS", f"{worst:.3e}")
    _say("SENTINEL_APART_FROM_UNMASKED", f"{float((got - unmasked).abs().max()):.3e}")
    assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
        f"the sentinel columns of a paged window are not masked as the oracle masks them: worst "
        f"absolute difference {worst:.3e}"
    )
    assert not torch.allclose(got, unmasked, rtol=RTOL, atol=ATOL), (
        "masking half the selected columns moved the result by less than the band this item's own "
        "tolerance allows, so it would pass on a kernel that ignores the sentinel entirely"
    )


def test_overlay_of_one_written_row_is_what_the_gather_reads() -> None:
    """OVERLAY 1 of 2 -- one row, the decode step's own latent, at a runtime row offset.

    CERTIFYING COMPONENT: the overlay operand. The bank row under the offset holds one value and the
    overlay another, so a kernel that read the bank instead of the overlay reads the wrong one. This is
    the hazard the landed seam repair removed for the unpaged window: a write the traced graph has not
    ordered before this read is not visible, so the rows travel as an operand. The exact claim is what
    catches an overlay written one row off; the tolerance claim alone does not.
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
    want = _oracle(window, selected, queries)
    worst = float((got - want).abs().max())
    _say("OVERLAY_ONE_AT", at)
    _say("OVERLAY_ONE_WORST_ABS", f"{worst:.3e}")
    _say("OVERLAY_ONE_APART_FROM_STALE", f"{float((got - stale).abs().max()):.3e}")
    assert at in selected[0].tolist(), f"row {at} is not selected, so the overlay would not be read"
    assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
        f"the overlaid row is not what the gather read: worst absolute difference {worst:.3e}"
    )
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        "the overlaid window through the paged call and through the unpaged call returned different "
        "bytes, so the overlay does not land where this step's rows belong"
    )
    assert not torch.allclose(got, stale, rtol=RTOL, atol=ATOL), (
        "the overlay moved the result by less than this item's own band, so it would pass on a kernel "
        "that ignores the overlay operand"
    )


def test_overlay_of_a_whole_prefill_chunk_is_what_the_gather_reads() -> None:
    """OVERLAY 2 of 2 -- the widest overlay the served configuration produces, one prefill chunk.

    CERTIFYING COMPONENT: the overlay at its declared maximum width, at an offset that is not a page
    multiple, so the written rows cross page boundaries inside the staged window. ONE query, because
    the width of the overlay is this item's conjunct and the query count is not.
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
    want = _oracle(window, selected, queries)
    worst = float((got - want).abs().max())
    inside = int(((selected >= CHUNK_AT) & (selected < CHUNK_AT + CHUNK)).sum())
    _say("OVERLAY_CHUNK_WORST_ABS", f"{worst:.3e}")
    _say("OVERLAY_CHUNK_SELECTED_INSIDE", inside)
    _say("OVERLAY_CHUNK_EDGES", ".".join(str(one) for one in CHUNK_EDGES))
    assert inside > 0, (
        f"none of this item's {TOPK} selected rows falls inside the overlaid chunk, so the width it "
        f"claims to read is not read"
    )
    assert all(one in selected[0].tolist() for one in CHUNK_EDGES), (
        f"this item does not read both edges of the written chunk {CHUNK_EDGES}, so a chunk whose "
        f"first or last block was never written returns the reference bytes and passes"
    )
    assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
        f"the overlaid chunk is not what the gather read: worst absolute difference {worst:.3e}"
    )
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        "the overlaid window through the paged call and through the unpaged call returned different "
        "bytes, so a chunk written one row off or one row short would pass unseen"
    )


def test_a_second_block_size_is_read_from_the_operand_and_not_assumed() -> None:
    """The page size travels as an operand, so a body that hardcodes the served one fails here.

    CERTIFYING COMPONENT: `page_size`. The table, the window arithmetic and the overlay offset are all
    in units of it, and 64 is a whole number of the kernel's 16-row transposes exactly as 128 is.
    """
    bank = _bank()
    table = [11, 20, 3]
    queries = _queries(1, HEADS, LATENT)
    window = _window(bank, table, page=PAGE_ALT)
    selected = _selected(1, window.shape[0], page=PAGE_ALT)
    got = _paged(bank, table, selected, queries, page=PAGE_ALT)
    want = _oracle(window, selected, queries)
    worst = float((got - want).abs().max())
    _say("ALT_PAGE", PAGE_ALT)
    _say("ALT_PAGE_WORST_ABS", f"{worst:.3e}")
    assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
        f"at a page size of {PAGE_ALT} the paged window disagrees with the oracle: {worst:.3e}"
    )
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        f"at a page size of {PAGE_ALT} the paged call and the unpaged call on the window the table "
        f"names returned different bytes, so the page size is not read from the operand"
    )


def test_a_page_wider_than_one_staging_piece_is_read_whole() -> None:
    """A page of 256 rows is two staging pieces, and both halves of every page are read.

    CERTIFYING COMPONENT: the piece loop. At every other page size in this file a page is one piece and
    the piece index IS the page number, so a body that stages only a page's first 128 rows, or that
    computes the piece index wrongly, passes every one of them. Here the second half of each page carries
    selected rows, so a short or misindexed piece returns other bank values.
    """
    bank = _bank()
    table = [7, 2]
    queries = _queries(1, HEADS, LATENT)
    window = _window(bank, table, page=PAGE_WIDE)
    selected = _selected(1, window.shape[0], page=PAGE_WIDE)
    halves = [int(one) % PAGE_WIDE // PAGE for one in selected[0].tolist()]
    got = _paged(bank, table, selected, queries, page=PAGE_WIDE)
    want = _oracle(window, selected, queries)
    worst = float((got - want).abs().max())
    _say("WIDE_PAGE", PAGE_WIDE)
    _say("WIDE_PAGE_SECOND_PIECE_ROWS", halves.count(1))
    _say("WIDE_PAGE_WORST_ABS", f"{worst:.3e}")
    assert halves.count(0) and halves.count(1), (
        f"this selection reads only piece {set(halves)} of each {PAGE_WIDE}-row page, so a body that "
        f"staged one piece and skipped the other would pass"
    )
    assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
        f"at a page size of {PAGE_WIDE} the paged window disagrees with the oracle: {worst:.3e}"
    )
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        f"at a page size of {PAGE_WIDE} the paged call and the unpaged call on the window the table "
        f"names returned different bytes, so a page wider than one staging piece is not assembled whole"
    )


def test_a_selected_row_past_the_staged_window_is_refused() -> None:
    """The bound is the WINDOW the table names, never the bank the pages come from.

    CERTIFYING COMPONENT: the seam's range check under a paged call. The bank holds 4,096 rows and this
    table stages 256 of them, so a selected row of 300 indexes the bank but not the window; gathering it
    would read a page this request was never given. The unpaged call is bounded by the cache it is handed
    and this call must be bounded by the window, not by the bank behind it.
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
    _say("PAST_WINDOW_REFUSAL", message or "none")
    assert str(len(table) * PAGE) in message, (
        f"a selected row past the staged window was not refused against the window's own length "
        f"{len(table) * PAGE}: {message or 'it was not refused at all'}"
    )


def test_a_scattered_table_a_padding_tail_and_an_overlay_in_one_call() -> None:
    """The three parts of the change in ONE call, which no item above combines.

    CERTIFYING COMPONENT: that the page loop, the clamp and the overlay compose. Each is read alone
    above; a kernel can pass all three and still order the overlay before the page that covers it, or
    clamp using an offset the overlay moved.
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
    want = _oracle(window, selected, queries)
    stale = _oracle(_window(bank, table), selected, queries)
    worst = float((got - want).abs().max())
    _say("COMBINED_AT", at)
    _say("COMBINED_WORST_ABS", f"{worst:.3e}")
    assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
        f"the scattered table, the padded tail and the overlay do not compose: worst absolute "
        f"difference {worst:.3e}"
    )
    assert torch.equal(got, _unpaged(window, selected, queries)), (
        "the combined call and the unpaged call on the window it names returned different bytes"
    )
    assert not torch.allclose(got, stale, rtol=RTOL, atol=ATOL), (
        "the overlay moved nothing in the combined call, so this item would pass on a kernel that "
        "stages the pages over this step's own rows"
    )


def _rows_selected(window_rows: int, topk: int = TOPK_ROWS) -> torch.Tensor:
    """`topk` columns over a shorter window: every row read, most of them more than once."""
    stride = 7
    return torch.tensor([[(pick * stride) % window_rows for pick in range(topk)]], dtype=torch.int32)


def _staged_path(bank, table, selected, queries, written=None, at=None):
    """The same paged call through the entry point, on the STAGED load path instead of the streamed one.

    Ten positional operands: the four the entry always took, the four paged ones, then the two tile
    parameters this base declares after them. The tile width keeps its default, so the load path is the
    only difference from the seam's own call.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(MS.mla_sparse_attention_nope_row_tiled_kernel)(
        queries, bank, selected, float(SCALE),
        torch.tensor([[entry] for entry in table], dtype=torch.int32),
        written, at, PAGE, MS.MOVING_MAX, False)


def test_the_row_tiled_body_reads_the_staged_window_the_same_way() -> None:
    """The body PRODUCTION takes reads the staged window, and reads it the same as every other path.

    CERTIFYING COMPONENT: the row-tiled body's own load. It gathers selected rows out of the cache
    operand with an indirect DMA instead of transposing the cache, so with a paged call that gather
    reads a window staged in private memory. Every item above runs at a selected-row count inside one
    moving tile and reaches the untiled body, which never does that.

    THREE READINGS, each exact. Against the unpaged call on the window the table names; against the same
    paged call on this body's other load path, which stages the window it is handed; and with one row
    overlaid, against the overlaid window. A refusal here is a question about where the window may be
    staged, and it is reported rather than worked around.
    """
    bank = _bank()
    table = STREAMED_TABLE
    queries = _queries(1, HEADS, LATENT)
    window = _window(bank, table)
    selected = _rows_selected(window.shape[0])
    overlay = _overlay_row(queries)
    overlaid = window.clone()
    overlaid[STREAMED_OVERLAY_AT] = overlay[0]
    at = torch.tensor([[STREAMED_OVERLAY_AT]], dtype=torch.int32)
    got = _paged(bank, table, selected, queries)
    unpaged = _unpaged(window, selected, queries)
    staged = _staged_path(bank, table, selected, queries)
    written = _paged(bank, table, selected, queries, written=overlay, write_offset=at)
    _say("ROW_TILED_TOPK", TOPK_ROWS)
    _say("ROW_TILED_UNPAGED_MAX_ABS_DIFF", float((got - unpaged).abs().max()))
    _say("ROW_TILED_STAGED_MAX_ABS_DIFF", float((got - staged).abs().max()))
    assert TOPK_ROWS > MS.MOVING_MAX, (
        f"topk={TOPK_ROWS} does not reach the row-tiled body, so this item reads the same body as "
        f"every item above it"
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


def test_this_files_numbers_make_the_items_above_able_to_fail() -> None:
    """The data control, and the only item here that calls no kernel.

    CERTIFYING COMPONENT: this file's own numbers, against three ways they could hide a defect.
    (a) A selection whose offsets repeat per page is gathered onto itself when pages are swapped, so
    table 2 could not see order; the reading is that the two orders gather different bank rows.
    (b) A selection without page edges cannot see a page copy short by one row, and one without the
    written chunk's own edges cannot see an overlay short at either end; the readings are that the first
    and last row of every page are selected, and that both chunk edges are.
    (c) A bank whose values saturate the softmax makes the output a copy of one gathered row, and the
    sentinel and overlay items then read no change on correct arithmetic; the readings are the three
    separations, each against the band the item it protects compares at, and the distance from the
    output to its nearest gathered row.
    """
    bank = _bank()
    queries = _queries(1, HEADS, LATENT)
    forward = _selected(1, 2 * PAGE)
    ascending = _gathered_rows([5, 10], forward)
    swapped = _gathered_rows([10, 5], forward)
    edges = [index * PAGE + inside for index in range(BANK_PAGES) for inside in (0, PAGE - 1)]
    wide_selected = _with_rows(_selected(1, BANK_PAGES * PAGE), CHUNK_EDGES)
    uncovered = [one for one in edges if one not in wide_selected[0].tolist()]
    window = _window(bank, [5, 10])
    plain = _oracle(window, forward, queries)
    masked = forward.clone()
    masked[:, TOPK // 2:] = MS.SENTINEL_INDEX
    sentinel = _oracle(window, masked, queries)
    at = _interior(forward)
    one_row = window.index_copy(0, torch.tensor([at]), _overlay_row(queries))
    overlay = _oracle(one_row, forward, queries)
    wide_window = _window(bank, list(range(BANK_PAGES)))
    chunk = _chunk_rows()
    wide_queries = _queries(1, 1, LATENT)
    wide_plain = _oracle(wide_window, wide_selected, wide_queries)
    wide_overlaid = _oracle(
        wide_window.index_copy(0, torch.arange(CHUNK_AT, CHUNK_AT + CHUNK), chunk),
        wide_selected, wide_queries)
    gathered = window[forward[0].to(torch.int64)]
    nearest = float((gathered - plain[0, 0]).abs().max(dim=1).values.min())
    _say("DATA_SWAP_ROWS_DIFFER", len(ascending ^ swapped))
    _say("DATA_OVERLAY_ONE_AT", at)
    _say("DATA_PAGE_EDGES_UNCOVERED", len(uncovered))
    _say("DATA_CHUNK_EDGES_SELECTED", sum(one in wide_selected[0].tolist() for one in CHUNK_EDGES))
    _say("DATA_SENTINEL_APART", f"{float((sentinel - plain).abs().max()):.3e}")
    _say("DATA_OVERLAY_ONE_APART", f"{float((overlay - plain).abs().max()):.3e}")
    _say("DATA_OVERLAY_CHUNK_APART", f"{float((wide_overlaid - wide_plain).abs().max()):.3e}")
    _say("DATA_NEAREST_GATHERED_ROW", f"{nearest:.3e}")
    assert ascending != swapped, (
        "the two page orders gather the same bank rows, so no reading of any strength can see a "
        "kernel that loads the pages in the wrong order"
    )
    assert uncovered == [], (
        f"{len(uncovered)} page edges are never selected, so a page copy short by one row leaves "
        f"stale content only where nothing reads: {uncovered[:6]}"
    )
    for label, changed, against in (("sentinel", sentinel, plain), ("overlay one", overlay, plain),
                                    ("overlay chunk", wide_overlaid, wide_plain)):
        assert not torch.allclose(changed, against, rtol=RTOL, atol=ATOL), (
            f"the {label} change is inside the band its own item compares at, so that item would "
            f"pass on a kernel which ignored the change entirely"
        )
    assert all(one in wide_selected[0].tolist() for one in CHUNK_EDGES), (
        f"the chunk overlay's own selection does not read both written edges {CHUNK_EDGES}, so an "
        f"overlay short at either end is invisible to the item that certifies its width"
    )
    assert nearest > ATOL, (
        f"the output sits {nearest:.3e} from one of its own gathered rows, so the softmax is putting "
        f"a single weight of 1 on that row and no item above can see a change to any other"
    )
